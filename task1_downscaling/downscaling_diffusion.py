"""
气象数据降尺度 - Diffusion Model (DDPM)
==============================================
条件扩散模型用于气象场降尺度
特点: 生成概率性高分辨率预报, 量化不确定性
参考: Ho et al., 2020 "Denoising Diffusion Probabilistic Models"
依赖: pip install torch numpy scikit-image scipy tqdm
==============================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.act(self.time_proj(t_emb))[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.res_conv(x)


class ConditionedUNet(nn.Module):
    """
    条件UNet - 扩散模型骨干网络
    
    输入: 带噪声HR场 + 时间步 + LR条件场
    输出: 预测的噪声
    """
    def __init__(self, in_channels=2, base_channels=32, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c, time_dim)
        self.enc2 = ConvBlock(c, c*2, time_dim)
        self.enc3 = ConvBlock(c*2, c*4, time_dim)
        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c*4, c*8, time_dim)
        self.up1 = nn.ConvTranspose2d(c*8, c*4, 2, stride=2)
        self.dec1 = ConvBlock(c*8, c*4, time_dim)
        self.up2 = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.dec2 = ConvBlock(c*4, c*2, time_dim)
        self.dec3 = ConvBlock(c*3, c, time_dim)
        self.out_conv = nn.Conv2d(c, 1, 1)
    
    def forward(self, x, t, condition):
        cond_up = nn.functional.interpolate(condition, size=x.shape[-2:], mode='bilinear', align_corners=True)
        x_cond = torch.cat([x, cond_up], dim=1)
        t_emb = self.time_mlp(t)
        e1 = self.enc1(x_cond, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)
        b = self.bottleneck(e3, t_emb)
        d1 = self.dec1(torch.cat([self.up1(b), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d1), e2], dim=1), t_emb)
        e1_r = nn.functional.interpolate(e1, size=d2.shape[-2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d2, e1_r], dim=1), t_emb)
        return self.out_conv(d3)


class DDPM:
    """去噪扩散概率模型"""
    def __init__(self, model, num_timesteps=1000):
        self.model = model
        self.T = num_timesteps
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        if noise is None: noise = torch.randn_like(x_0)
        device = x_0.device
        sa = self.sqrt_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
        sm = self.sqrt_one_minus_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
        return sa * x_0 + sm * noise
    
    def p_losses(self, x_0, condition, t):
        noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(x_0, t, noise)
        pred_noise = self.model(x_noisy, t, condition)
        min_h = min(pred_noise.shape[2], noise.shape[2])
        min_w = min(pred_noise.shape[3], noise.shape[3])
        return nn.functional.mse_loss(pred_noise[:,:,:min_h,:min_w], noise[:,:,:min_h,:min_w])
    
    @torch.no_grad()
    def sample(self, condition, shape, num_steps=None):
        """生成多个样本量化不确定性"""
        device = condition.device
        T = num_steps or self.T
        x = torch.randn(shape, device=device)
        for t in reversed(range(T)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            betas_t = self.betas[t].to(device).view(-1, 1, 1, 1)
            sm_t = self.sqrt_one_minus_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
            sr_t = (1.0 / torch.sqrt(self.alphas[t])).to(device).view(-1, 1, 1, 1)
            model_out = self.model(x, t_batch, condition)
            min_h = min(model_out.shape[2], x.shape[2])
            min_w = min(model_out.shape[3], x.shape[3])
            model_out = model_out[:,:,:min_h,:min_w]
            x_a = x[:,:,:min_h,:min_w]
            mean = sr_t * (x_a - betas_t / sm_t * model_out)
            if t > 0:
                pv = self.betas[t] * (1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t])
                x = mean + torch.sqrt(pv.to(device)) * torch.randn_like(mean)
            else:
                x = mean
        return x


def train_diffusion(ddpm, train_loader, num_epochs=100, lr=2e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device} (扩散模型建议使用GPU)")
    ddpm.model = ddpm.model.to(device)
    optimizer = optim.Adam(ddpm.model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        ddpm.model.train()
        epoch_loss = 0.0
        for lr_b, hr_b in train_loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            t = torch.randint(0, ddpm.T, (hr_b.shape[0],), device=device)
            optimizer.zero_grad()
            loss = ddpm.p_losses(hr_b, lr_b, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.6f}")
            torch.save(ddpm.model.state_dict(), 'best_downscaling_diffusion.pth')
    return ddpm


if __name__ == '__main__':
    print("气象数据降尺度 - Diffusion Model 训练")
    from downscaling_cnn import ERA5DownscalingDataset
    
    LR_SIZE = (20, 35)
    HR_SIZE = (80, 140)
    NUM_TIMESTEPS = 200  # 演示用减少步数, 实际建议1000
    
    dataset = ERA5DownscalingDataset(n_samples=300, lr_size=LR_SIZE, hr_size=HR_SIZE)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    unet = ConditionedUNet(in_channels=2, base_channels=16, time_dim=64)
    ddpm = DDPM(unet, num_timesteps=NUM_TIMESTEPS)
    print(f"UNet参数量: {sum(p.numel() for p in unet.parameters()):,}")
    
    ddpm = train_diffusion(ddpm, train_loader, num_epochs=50)
    
    # 演示不确定性量化
    print("\\n生成5个样本演示不确定性量化...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr_s, _ = next(iter(train_loader))
    lr_s = lr_s[:1].to(device)
    samples = [ddpm.sample(lr_s, (1, 1, HR_SIZE[0], HR_SIZE[1])).cpu().numpy() for _ in range(5)]
    samples = np.array(samples)
    print(f"样本标准差 (不确定性): {samples.std():.4f}")
    print("训练完成!")
