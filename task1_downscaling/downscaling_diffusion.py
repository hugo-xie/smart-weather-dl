"""
气象数据降尺度 - Diffusion Model (DDPM)
==============================================
条件扩散模型用于气象场降尺度
特点: 生成概率性高分辨率预报, 量化不确定性
参考: Ho et al., 2020 "Denoising Diffusion Probabilistic Models"
依赖: pip install torch numpy scikit-image scipy tqdm
==============================================
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm

from download_data import (
    PROCESSED_DIR,
    build_dataloaders,
    preprocess_era5,
    processed_data_exists,
)
from evaluation_utils import evaluate_diffusion_noise_loss, print_metrics


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
        e1 = self.enc1(x_cond, t_emb)           # (B, c, H, W)
        e2 = self.enc2(self.down1(e1), t_emb)   # (B, 2c, H/2, W/2)
        e3 = self.enc3(self.down2(e2), t_emb)   # (B, 4c, H/4, W/4)
        b = self.bottleneck(e3, t_emb)           # (B, 8c, H/4, W/4)
        # up1 将 b 从 H/4 上采样到 H/2，而 e3 在 H/4，需要对 e3 进行插小或对 up1(b) 进行裁剪
        up1_b = self.up1(b)                      # (B, 4c, H/2, W/2)
        # 使用 interpolate 对齐 e3 到 up1_b 的尺寸
        e3_r = nn.functional.interpolate(e3, size=up1_b.shape[-2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([up1_b, e3_r], dim=1), t_emb)  # (B, 4c, H/2, W/2)
        up2_d1 = self.up2(d1)                    # (B, 2c, H, W)
        # 对齐 e2 到 up2_d1 的尺寸
        e2_r = nn.functional.interpolate(e2, size=up2_d1.shape[-2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([up2_d1, e2_r], dim=1), t_emb)  # (B, 2c, H, W)
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
        sa = self.sqrt_alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
        sm = self.sqrt_one_minus_alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
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
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} train", leave=False, ncols=100)
        for lr_b, hr_b in train_bar:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            t = torch.randint(0, ddpm.T, (hr_b.shape[0],), device=device)
            optimizer.zero_grad()
            loss = ddpm.p_losses(hr_b, lr_b, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        epoch_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.6f}")
        if (epoch + 1) % 20 == 0:
            torch.save(ddpm.model.state_dict(), 'best_downscaling_diffusion.pth')
    return ddpm


def _env_int(name, default):
    value = os.getenv(name)
    return default if value is None else int(value)


if __name__ == '__main__':
    print("=" * 60)
    print("气象数据降尺度 - Diffusion Model 训练")
    print("=" * 60)

    BATCH_SIZE = _env_int("DOWNSCALING_BATCH_SIZE", 2)
    NUM_EPOCHS = _env_int("DOWNSCALING_EPOCHS", 50)
    NUM_WORKERS = _env_int("DOWNSCALING_NUM_WORKERS", 0)
    MAX_SAMPLES = _env_int("DOWNSCALING_MAX_SAMPLES", 0)
    EVAL_MAX_BATCHES = _env_int("DOWNSCALING_EVAL_MAX_BATCHES", 0)
    NUM_TIMESTEPS = _env_int("DOWNSCALING_DIFFUSION_TIMESTEPS", 200)
    SAMPLE_STEPS = _env_int("DOWNSCALING_DIFFUSION_SAMPLE_STEPS", min(NUM_TIMESTEPS, 50))
    raw_dir = Path(os.getenv("DOWNSCALING_RAW_DIR", Path(__file__).resolve().parent))
    data_dir = Path(os.getenv("DOWNSCALING_DATA_DIR", PROCESSED_DIR))

    print("\n[1/5] 准备数据集...")
    if not processed_data_exists(data_dir):
        print(f"   未找到预处理数据: {data_dir}")
        print("   开始从 ERA5 NetCDF 文件生成 train/val/test 数据...")
        preprocess_era5(raw_dir=raw_dir, output_dir=data_dir)
    else:
        print(f"   使用预处理数据: {data_dir}")

    max_samples = None
    if MAX_SAMPLES > 0:
        max_samples = {"train": MAX_SAMPLES, "val": MAX_SAMPLES, "test": MAX_SAMPLES}
        print(f"   调试模式: 每个 split 最多使用 {MAX_SAMPLES} 个样本")

    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        max_samples=max_samples,
    )
    metadata = train_loader.dataset.metadata
    hr_h, hr_w = metadata["hr_shape"]
    lr_s, hr_s = train_loader.dataset[0]
    print(
        f"   训练: {len(train_loader.dataset)} 样本 | "
        f"验证: {len(val_loader.dataset)} 样本 | "
        f"测试: {len(test_loader.dataset)} 样本"
    )
    print(f"   LR shape: {tuple(lr_s.shape)} | HR shape: {tuple(hr_s.shape)}")

    print("\n[2/5] 初始化扩散模型...")
    unet = ConditionedUNet(in_channels=2, base_channels=16, time_dim=64)
    ddpm = DDPM(unet, num_timesteps=NUM_TIMESTEPS)
    print(f"   UNet参数量: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"   扩散步数: {NUM_TIMESTEPS}")

    print("\n[3/5] 开始训练...")
    ddpm = train_diffusion(ddpm, train_loader, num_epochs=NUM_EPOCHS)

    print("\n[4/5] 测试集噪声预测评估...")
    metrics = evaluate_diffusion_noise_loss(
        ddpm,
        test_loader,
        desc="Diffusion test",
        max_batches=EVAL_MAX_BATCHES,
    )
    print_metrics("测试集评估结果（扩散噪声预测）", metrics)

    print("\n[5/5] 生成5个样本演示不确定性量化...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ddpm.model.eval()
    lr_batch, _ = next(iter(test_loader))
    lr_batch = lr_batch[:1].to(device)
    samples = [
        ddpm.sample(
            lr_batch,
            (1, 1, hr_h, hr_w),
            num_steps=SAMPLE_STEPS,
        ).cpu().numpy()
        for _ in range(5)
    ]
    samples = np.array(samples)
    print(f"   采样步数: {SAMPLE_STEPS}")
    print(f"   样本标准差 (不确定性, normalized units): {samples.std():.4f}")
    print("训练完成!")


# ============================================================
# 思考题 & 动手练习
# Exercises & Hands-on Practice
# ============================================================
#
# ⭐ 入门题 1
# 将扩散步数（num_timesteps）分别设为 50、100、500、1000，
# 对比生成质量（RMSE）和推理速度的变化。
# 步数越多生成质量一定越好吗？实际应用中如何平衡两者？
#
# 💡 提示: 修改 DDPM(num_timesteps=N)，
#          用 time.time() 记录每次采样的耗时，同时记录生成样本的 RMSE
#
# ⭐⭐ 进阶题 2
# 扩散模型的独特优势是可以生成多个样本来量化不确定性。
# 请对同一个低分辨率输入生成 20 个高分辨率样本，
# 计算这 20 个样本的像素级标准差，并可视化展示不确定性分布。
#
# 💡 提示: 调用 ddpm.sample(lr_condition, shape) 20 次，
#          对结果列表计算 np.array(samples).std(axis=0)，
#          即为每个像素的不确定性估计
#
# ⭐⭐⭐ 挑战题 3
# 尝试将线性 Beta 调度改为 Cosine Beta 调度
# （参考 Improved DDPM 论文，Nichol & Dhariwal, 2021），
# 对比两种调度对训练稳定性和生成质量的影响。
# Cosine 调度为什么在开始和结束阶段加噪更平滑？
#
# 💡 提示: 将 self.betas = torch.linspace(1e-4, 0.02, T) 改为：
#          steps = torch.arange(T+1, dtype=torch.float)
#          alphas_cumprod = torch.cos(((steps/T)+0.008)/1.008 * pi/2)**2
#          alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#          betas = torch.clamp(1 - alphas_cumprod[1:]/alphas_cumprod[:-1], 0, 0.999)
