"""
台风路径预测 - Diffusion Model (条件扩散)
==============================================
条件扩散模型生成概率性台风轨迹
特点: 生成轨迹集合, 量化预报不确定性
依赖: pip install torch numpy pandas
==============================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math


class TrajectoryDenoiser(nn.Module):
    """
    轨迹去噪网络 (Transformer架构)
    
    输入: 带噪轨迹 + 时间步 + 历史条件
    输出: 预测的噪声
    """
    def __init__(self, output_len=4, output_size=2, input_len=12, input_size=4,
                 d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.output_len = output_len
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.condition_encoder = nn.Sequential(nn.Linear(input_size, d_model), nn.LayerNorm(d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*2, dropout=0.1, batch_first=True)
        self.cond_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.noisy_embed = nn.Sequential(nn.Linear(output_size, d_model), nn.LayerNorm(d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_model*2, dropout=0.1, batch_first=True)
        self.denoiser = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_size)
    
    def forward(self, x_t, t, condition):
        t_norm = (t.float() / 1000.0).unsqueeze(1)
        t_emb = self.time_embed(t_norm).unsqueeze(1)
        cond_memory = self.cond_transformer(self.condition_encoder(condition))
        noisy_emb = self.noisy_embed(x_t) + t_emb
        output = self.denoiser(noisy_emb, cond_memory)
        return self.output_proj(output)


class TyphoonDDPM:
    """台风轨迹预测扩散模型 (Cosine调度)"""
    def __init__(self, model, num_timesteps=500):
        self.model = model
        self.T = num_timesteps
        steps = torch.arange(num_timesteps + 1, dtype=torch.float)
        alphas_cumprod = torch.cos(((steps / num_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        self.betas = torch.clamp(1 - alphas_cumprod[1:] / alphas_cumprod[:-1], 0, 0.999)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        if noise is None: noise = torch.randn_like(x_0)
        device = x_0.device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        sa = sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sm = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sa * x_0 + sm * noise
    
    def p_losses(self, x_0, condition, t):
        noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(x_0, t, noise)
        return nn.functional.mse_loss(self.model(x_noisy, t, condition), noise)
    
    @torch.no_grad()
    def sample(self, condition, n_samples=10):
        """生成n_samples条轨迹 (概率性预报)"""
        device = condition.device
        alphas = self.alphas.to(device)
        betas = self.betas.to(device)
        alphas_cumprod = self.alphas_cumprod.to(device)
        alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        B = condition.shape[0]
        all_samples = []
        for _ in range(n_samples):
            x = torch.randn(B, self.model.output_len, 2, device=device)
            for t in reversed(range(self.T)):
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)
                pred_noise = self.model(x, t_batch, condition)
                alpha = alphas[t]
                alpha_cumprod = alphas_cumprod[t]
                beta = betas[t]
                mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise)
                if t > 0:
                    pv = beta * (1 - alphas_cumprod_prev[t]) / (1 - alpha_cumprod)
                    x = mean + torch.sqrt(pv) * torch.randn_like(mean)
                else:
                    x = mean
            all_samples.append(x)
        return torch.stack(all_samples)


def train_typhoon_diffusion(ddpm, train_loader, num_epochs=100, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ddpm.model = ddpm.model.to(device)
    optimizer = optim.AdamW(ddpm.model.parameters(), lr=lr, weight_decay=0.01)
    
    for epoch in range(num_epochs):
        ddpm.model.train()
        epoch_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            t = torch.randint(0, ddpm.T, (y_b.shape[0],), device=device)
            optimizer.zero_grad()
            loss = ddpm.p_losses(y_b, x_b, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.6f}")
    return ddpm


if __name__ == '__main__':
    print("台风路径预测 - Diffusion Model 训练")
    from typhoon_cnn import IBTrACSDataset, load_ibtracs_data
    
    tracks = load_ibtracs_data('/root/autodl-tmp/project/project/project/smart-weather-dl-main/dataset/ibtracs.WP.list.v04r00.csv')
    dataset = IBTrACSDataset(tracks=tracks, input_len=12, output_len=4)
    n_train = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, len(dataset)-n_train])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    denoiser = TrajectoryDenoiser(output_len=4, output_size=2, input_len=12, input_size=4, d_model=64, nhead=4, num_layers=3)
    ddpm = TyphoonDDPM(denoiser, num_timesteps=200)
    print(f"去噪网络参数量: {sum(p.numel() for p in denoiser.parameters()):,}")
    
    ddpm = train_typhoon_diffusion(ddpm, train_loader, num_epochs=100)
    
    ddpm.model.eval()

    from typhoon_utils import quick_ensemble_test_and_plot
    quick_ensemble_test_and_plot(
        val_loader,
        samples_fn=lambda x, n: ddpm.sample(x, n_samples=n),
        save_path='typhoon_diffusion_prediction.png',
        title='Diffusion Typhoon Ensemble Prediction',
        n_samples=20
    )
    print("训练完成!")


# ============================================================
# 思考题 & 动手练习
# Exercises & Hands-on Practice
# ============================================================
#
# ⭐ 入门题 1
# 对同一个历史轨迹输入，分别生成 5、10、20、50 条轨迹，
# 计算轨迹集合的平均误差和标准差。
# 随着样本数增加，误差和不确定性估计是否趋于收敛？
#
# 💡 提示: 多次调用 ddpm.sample(condition, n_samples=1)，
#          对返回的轨迹列表计算 np.array(trajectories).mean(0)
#          和 np.array(trajectories).std(0)，
#          用 matplotlib 画出轨迹集合图
#
# ⭐⭐ 进阶题 2
# 尝试将条件编码器从 Transformer 改为 LSTM，
# 对比两种条件编码器对扩散模型生成质量的影响。
# 对于台风轨迹这种时序条件，哪种编码器更合适？
#
# 💡 提示: 将 TrajectoryDenoiser 中的 cond_transformer 替换为
#          nn.LSTM(input_size, d_model, batch_first=True)，
#          对比训练损失和生成轨迹的 Haversine 误差
#
# ⭐⭐⭐ 挑战题 3
# 尝试将扩散模型的输出从仅预测位置（lat, lon）
# 扩展到同时预测强度（wind, pres），
# 实现台风轨迹+强度的联合概率预报。
# 联合预测相比单独预测位置有什么优势？
#
# 💡 提示: 将 output_size 从 2 改为 4，
#          同时修改 IBTrACSDataset 的 y 包含 wind 和 pres，
#          对比联合预测和单独预测位置的 Haversine 误差
