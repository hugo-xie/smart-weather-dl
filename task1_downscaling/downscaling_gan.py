"""
气象数据降尺度 - GAN (SRGAN风格)
==============================================
生成对抗网络用于气象场降尺度
生成器+判别器对抗训练, 生成真实感强的高分辨率场
参考: Ledig et al., 2017 "Photo-Realistic Single Image Super-Resolution Using a GAN"
依赖: pip install torch numpy scikit-image scipy
==============================================
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from download_data import (
    PROCESSED_DIR,
    build_dataloaders,
    preprocess_era5,
    processed_data_exists,
)
from evaluation_utils import evaluate_image_model, print_metrics


class ResidualBlockG(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels)
        )
    def forward(self, x): return x + self.block(x)


class PixelShuffleUpsampler(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor**2, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU()
    def forward(self, x): return self.act(self.pixel_shuffle(self.conv(x)))


class DownscalingGenerator(nn.Module):
    """
    降尺度生成器 (SRGAN风格)
    
    输入: 低分辨率场 (B, 1, H_lr, W_lr)
    输出: 高分辨率场 (B, 1, H_hr, W_hr)
    """
    def __init__(self, scale_factor=4, num_res_blocks=8, channels=64):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(1, channels, 9, padding=4), nn.PReLU())
        self.residual_blocks = nn.Sequential(*[ResidualBlockG(channels) for _ in range(num_res_blocks)])
        self.post_res = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels))
        upsampler_layers = []
        for _ in range(int(np.log2(scale_factor))):
            upsampler_layers.append(PixelShuffleUpsampler(channels, 2))
        self.upsamplers = nn.Sequential(*upsampler_layers)
        self.tail = nn.Conv2d(channels, 1, 9, padding=4)
    
    def forward(self, x):
        head_out = self.head(x)
        res_out = self.post_res(self.residual_blocks(head_out)) + head_out
        return self.tail(self.upsamplers(res_out))


class PatchDiscriminator(nn.Module):
    """PatchGAN判别器 - 对局部patch进行真/假判断"""
    def __init__(self, in_channels=1):
        super().__init__()
        def disc_block(in_ch, out_ch, stride=2, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)]
            if normalize: layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *disc_block(in_channels, 64, normalize=False), *disc_block(64, 128),
            *disc_block(128, 256), *disc_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    def forward(self, x): return self.model(x)


def train_gan(netG, netD, train_loader, num_epochs=100, lr_g=1e-4, lr_d=1e-4):
    """
    GAN训练策略:
    阶段1: 预训练生成器 (纯MSE, 稳定初始化)
    阶段2: 对抗训练 (MSE + 对抗损失)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG, netD = netG.to(device), netD.to(device)
    opt_G = optim.Adam(netG.parameters(), lr=lr_g, betas=(0.9, 0.999))
    opt_D = optim.Adam(netD.parameters(), lr=lr_d, betas=(0.9, 0.999))
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    
    # 阶段1: 预训练
    pretrain_epochs = min(10, num_epochs // 5)
    print(f"阶段1: 预训练生成器 ({pretrain_epochs} epochs)...")
    for epoch in range(pretrain_epochs):
        netG.train()
        train_bar = tqdm(train_loader, desc=f"Pretrain {epoch+1}/{pretrain_epochs}", leave=False, ncols=100)
        pretrain_loss = 0.0
        for lr_b, hr_b in train_bar:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            opt_G.zero_grad()
            fake_hr = netG(lr_b)
            min_h = min(fake_hr.shape[2], hr_b.shape[2])
            min_w = min(fake_hr.shape[3], hr_b.shape[3])
            loss = criterion_mse(fake_hr[:,:,:min_h,:min_w], hr_b[:,:,:min_h,:min_w])
            loss.backward()
            opt_G.step()
            pretrain_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"   预训练 Epoch [{epoch+1}/{pretrain_epochs}] G_MSE: {pretrain_loss/len(train_loader):.6f}")
    
    # 阶段2: 对抗训练
    print("阶段2: 对抗训练...")
    adversarial_epochs = num_epochs - pretrain_epochs
    for epoch in range(adversarial_epochs):
        netG.train(); netD.train()
        d_loss_total = g_loss_total = 0.0
        train_bar = tqdm(train_loader, desc=f"GAN {epoch+1}/{adversarial_epochs}", leave=False, ncols=100)
        
        for lr_b, hr_b in train_bar:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            fake_hr = netG(lr_b)
            min_h = min(fake_hr.shape[2], hr_b.shape[2])
            min_w = min(fake_hr.shape[3], hr_b.shape[3])
            fake_hr_a = fake_hr[:,:,:min_h,:min_w]
            hr_a = hr_b[:,:,:min_h,:min_w]
            
            # 训练判别器
            opt_D.zero_grad()
            real_pred = netD(hr_a)
            fake_pred = netD(fake_hr_a.detach())
            d_loss = (criterion_bce(real_pred, torch.ones_like(real_pred) * 0.9) +
                      criterion_bce(fake_pred, torch.zeros_like(fake_pred) + 0.1)) / 2
            d_loss.backward(); opt_D.step()
            
            # 训练生成器
            opt_G.zero_grad()
            fake_for_g = netD(fake_hr_a)
            g_adv = criterion_bce(fake_for_g, torch.ones_like(fake_for_g))
            g_content = criterion_mse(fake_hr_a, hr_a)
            g_loss = g_content + 0.001 * g_adv
            g_loss.backward(); opt_G.step()
            
            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
            train_bar.set_postfix(D=f"{d_loss.item():.4f}", G=f"{g_loss.item():.4f}")
        
        print(f"Epoch [{epoch+1}/{adversarial_epochs}] D: {d_loss_total/len(train_loader):.4f} | G: {g_loss_total/len(train_loader):.4f}")
        
        if (epoch + 1) % 20 == 0:
            torch.save(netG.state_dict(), 'best_downscaling_gan_g.pth')
    
    return netG, netD


def _env_int(name, default):
    value = os.getenv(name)
    return default if value is None else int(value)


if __name__ == '__main__':
    print("=" * 60)
    print("气象数据降尺度 - GAN 模型训练")
    print("=" * 60)

    BATCH_SIZE = _env_int("DOWNSCALING_BATCH_SIZE", 4)
    NUM_EPOCHS = _env_int("DOWNSCALING_EPOCHS", 60)
    NUM_WORKERS = _env_int("DOWNSCALING_NUM_WORKERS", 0)
    MAX_SAMPLES = _env_int("DOWNSCALING_MAX_SAMPLES", 0)
    EVAL_MAX_BATCHES = _env_int("DOWNSCALING_EVAL_MAX_BATCHES", 0)
    raw_dir = Path(os.getenv("DOWNSCALING_RAW_DIR", Path(__file__).resolve().parent))
    data_dir = Path(os.getenv("DOWNSCALING_DATA_DIR", PROCESSED_DIR))

    print("\n[1/4] 准备数据集...")
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
    scale_factor = int(metadata.get("scale_factor", 4))
    lr_s, hr_s = train_loader.dataset[0]
    print(
        f"   训练: {len(train_loader.dataset)} 样本 | "
        f"验证: {len(val_loader.dataset)} 样本 | "
        f"测试: {len(test_loader.dataset)} 样本"
    )
    print(f"   LR shape: {tuple(lr_s.shape)} | HR shape: {tuple(hr_s.shape)}")

    print("\n[2/4] 初始化 GAN...")
    netG = DownscalingGenerator(scale_factor=scale_factor, num_res_blocks=4, channels=32)
    netD = PatchDiscriminator()
    print(f"   生成器参数: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"   判别器参数: {sum(p.numel() for p in netD.parameters()):,}")

    print("\n[3/4] 开始训练...")
    netG, netD = train_gan(netG, netD, train_loader, num_epochs=NUM_EPOCHS)

    print("\n[4/4] 测试集评估...")
    metrics = evaluate_image_model(
        netG,
        test_loader,
        metadata,
        desc="GAN test",
        max_batches=EVAL_MAX_BATCHES,
    )
    print_metrics("测试集评估结果（反归一化到 K）", metrics)
    print("训练完成!")


# ============================================================
# 思考题 & 动手练习
# Exercises & Hands-on Practice
# ============================================================
#
# ⭐ 入门题 1
# 尝试将对抗损失的权重系数（当前为 0.001）分别设为
# 0、0.0001、0.001、0.01，观察生成结果的变化。
# 当权重为 0 时模型退化为什么？权重过大时又会出现什么问题？
#
# 💡 提示: 修改 train_gan() 中的 g_loss = g_content + λ * g_adv_loss，
#          分别用 λ=0/0.0001/0.001/0.01 训练，对比 RMSE 和视觉质量
#
# ⭐⭐ 进阶题 2
# GAN 训练中常见的问题是模式崩溃（Mode Collapse）。
# 请尝试将标准 GAN 损失改为 Wasserstein 损失
# （去掉判别器的 Sigmoid，将 BCELoss 改为直接计算期望），
# 观察训练稳定性的变化。
#
# 💡 提示: 将 criterion_bce 改为 Wasserstein 损失：
#          d_loss = -real_pred.mean() + fake_pred.mean()
#          并将判别器最后一层的 Sigmoid 去掉
#
# ⭐⭐⭐ 挑战题 3
# 尝试将亚像素卷积（Pixel Shuffle）上采样替换为
# 转置卷积（ConvTranspose2d），对比两种上采样方式
# 对生成图像中棋盘格伪影（Checkerboard Artifacts）的影响。
# 为什么 Pixel Shuffle 通常优于转置卷积？
#
# 💡 提示: 将 PixelShuffleUpsampler 替换为：
#          nn.ConvTranspose2d(channels, channels, 2, stride=2)
#          用 matplotlib 可视化生成图像中的棋盘格模式
