"""
气象数据降尺度 - GAN (SRGAN风格)
==============================================
生成对抗网络用于气象场降尺度
生成器+判别器对抗训练, 生成真实感强的高分辨率场
参考: Ledig et al., 2017 "Photo-Realistic Single Image Super-Resolution Using a GAN"
依赖: pip install torch numpy scikit-image scipy
==============================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


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
        self.tail = nn.Sequential(nn.Conv2d(channels, 1, 9, padding=4), nn.Tanh())
    
    def forward(self, x):
        head_out = self.head(x)
        res_out = self.post_res(self.residual_blocks(head_out)) + head_out
        out = self.tail(self.upsamplers(res_out))
        return (out + 1) / 2  # 映射到[0,1]


class PatchDiscriminator(nn.Module):
    """PatchGAN判别器 - 对局部patch进行真/假判断"""
    def __init__(self):
        super().__init__()
        def disc_block(in_ch, out_ch, stride=2, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)]
            if normalize: layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *disc_block(1, 64, normalize=False), *disc_block(64, 128),
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
        for lr_b, hr_b in train_loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            opt_G.zero_grad()
            fake_hr = netG(lr_b)
            min_h = min(fake_hr.shape[2], hr_b.shape[2])
            min_w = min(fake_hr.shape[3], hr_b.shape[3])
            criterion_mse(fake_hr[:,:,:min_h,:min_w], hr_b[:,:,:min_h,:min_w]).backward()
            opt_G.step()
        if (epoch + 1) % 5 == 0: print(f"   预训练 Epoch {epoch+1}/{pretrain_epochs}")
    
    # 阶段2: 对抗训练
    print("阶段2: 对抗训练...")
    for epoch in range(num_epochs - pretrain_epochs):
        netG.train(); netD.train()
        d_loss_total = g_loss_total = 0.0
        
        for lr_b, hr_b in train_loader:
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
            g_adv = criterion_bce(netD(fake_hr_a), torch.ones_like(netD(fake_hr_a)))
            g_content = criterion_mse(fake_hr_a, hr_a)
            g_loss = g_content + 0.001 * g_adv
            g_loss.backward(); opt_G.step()
            
            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs-pretrain_epochs}] D: {d_loss_total/len(train_loader):.4f} | G: {g_loss_total/len(train_loader):.4f}")
        
        if (epoch + 1) % 20 == 0:
            torch.save(netG.state_dict(), 'best_downscaling_gan_g.pth')
    
    return netG, netD


if __name__ == '__main__':
    print("气象数据降尺度 - GAN 模型训练")
    from downscaling_cnn import ERA5DownscalingDataset
    
    dataset = ERA5DownscalingDataset(n_samples=400, lr_size=(20, 35), hr_size=(80, 140))
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    netG = DownscalingGenerator(scale_factor=4, num_res_blocks=4, channels=32)
    netD = PatchDiscriminator()
    print(f"生成器参数: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"判别器参数: {sum(p.numel() for p in netD.parameters()):,}")
    
    netG, netD = train_gan(netG, netD, train_loader, num_epochs=60)
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
