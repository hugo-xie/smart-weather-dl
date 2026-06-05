"""
气象数据降尺度 - CNN 模型
==============================================
任务: 将ERA5低分辨率(1°)温度场降尺度到高分辨率(0.25°)
数据: ERA5再分析数据 (可通过CDS API免费获取)
运行: python downscaling_cnn.py
依赖: pip install torch numpy netCDF4 cdsapi matplotlib scikit-learn scikit-image
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
    ERA5DownscalingDataset,
    build_dataloaders,
    preprocess_era5,
    processed_data_exists,
)
from evaluation_utils import evaluate_image_model, print_metrics

# ============================================================
# 数据下载说明 (使用CDS API)
# ============================================================
# 1. 注册账号: https://cds.climate.copernicus.eu/
# 2. 安装: pip install cdsapi
# 3. 配置 ~/.cdsapirc:
#    url: https://cds.climate.copernicus.eu/api/v2
#    key: YOUR_UID:YOUR_API_KEY
#
# import cdsapi
# c = cdsapi.Client()
# c.retrieve('reanalysis-era5-single-levels', {
#     'product_type': 'reanalysis',
#     'variable': '2m_temperature',
#     'year': [str(y) for y in range(2010, 2021)],
#     'month': [f'{m:02d}' for m in range(1, 13)],
#     'day': [f'{d:02d}' for d in range(1, 32)],
#     'time': ['00:00', '06:00', '12:00', '18:00'],
#     'area': [55, 70, 15, 140],  # 中国区域 N,W,S,E
#     'grid': [0.25, 0.25],
#     'format': 'netcdf',
# }, 'era5_t2m_025deg.nc')

# ============================================================
# 数据集类
# ============================================================
# ERA5DownscalingDataset 在 download_data.py 中实现。
# 这里保留同名导入，方便其他脚本继续 from downscaling_cnn import ERA5DownscalingDataset。


# ============================================================
# CNN模型 (SRCNN + 残差块)
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class DownscalingCNN(nn.Module):
    """
    气象降尺度CNN模型
    
    输入: 低分辨率气象场 (B, 1, H_lr, W_lr)
    输出: 高分辨率气象场 (B, 1, H_hr, W_hr)
    """
    def __init__(self, scale_factor=4, num_residual_blocks=8):
        super(DownscalingCNN, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.head = nn.Sequential(nn.Conv2d(1, 64, 9, padding=4), nn.ReLU(inplace=True))
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(64, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 5, padding=2)
        )
    
    def forward(self, x):
        x_up = self.upsample(x)
        feat = self.head(x_up)
        feat = self.residual_blocks(feat)
        return x_up + self.tail(feat)


# ============================================================
# 训练函数
# ============================================================

def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4):
    """
    返回: (train_losses, val_losses) - 每个 epoch 的损失列表
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} train", leave=False, ncols=100)
        for lr_batch, hr_batch in train_bar:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            optimizer.zero_grad()
            pred = model(lr_batch)
            min_h = min(pred.shape[2], hr_batch.shape[2])
            min_w = min(pred.shape[3], hr_batch.shape[3])
            loss = criterion(pred[:,:,:min_h,:min_w], hr_batch[:,:,:min_h,:min_w])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} val", leave=False, ncols=100)
            for lr_batch, hr_batch in val_bar:
                lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
                pred = model(lr_batch)
                min_h = min(pred.shape[2], hr_batch.shape[2])
                min_w = min(pred.shape[3], hr_batch.shape[3])
                batch_loss = criterion(pred[:,:,:min_h,:min_w], hr_batch[:,:,:min_h,:min_w]).item()
                val_loss += batch_loss
                val_bar.set_postfix(loss=f"{batch_loss:.4f}")
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_downscaling_cnn.pth')
    
    return train_losses, val_losses


def calculate_metrics(pred, target):
    """计算RMSE, PSNR, SSIM"""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    max_val = target.max() - target.min()
    psnr = 20 * np.log10(max_val / (rmse + 1e-8))
    return {'RMSE': rmse, 'PSNR': psnr}


# ============================================================
# 主程序
# ============================================================

def _env_int(name, default):
    value = os.getenv(name)
    return default if value is None else int(value)


if __name__ == '__main__':
    print("=" * 60)
    print("气象数据降尺度 - CNN 模型训练")
    print("=" * 60)

    BATCH_SIZE = _env_int("DOWNSCALING_BATCH_SIZE", 4)
    NUM_EPOCHS = _env_int("DOWNSCALING_EPOCHS", 50)
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
    sample_lr, sample_hr = train_loader.dataset[0]
    scale_factor = int(metadata.get("scale_factor", 4))

    print(
        f"   训练: {len(train_loader.dataset)} 样本 | "
        f"验证: {len(val_loader.dataset)} 样本 | "
        f"测试: {len(test_loader.dataset)} 样本"
    )
    print(f"   LR shape: {tuple(sample_lr.shape)} | HR shape: {tuple(sample_hr.shape)}")
    print(
        "   年份划分: train=2010-2017 | val=2018-2019 | test=2020"
    )

    print("\n[2/4] 初始化CNN模型...")
    model = DownscalingCNN(scale_factor=scale_factor, num_residual_blocks=4)
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n[3/4] 开始训练...")
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
    )
    print(f"   最终训练损失: {train_losses[-1]:.6f} | 验证损失: {val_losses[-1]:.6f}")

    print("\n[4/4] 测试集评估...")
    metrics = evaluate_image_model(
        model,
        test_loader,
        metadata,
        desc="CNN test",
        max_batches=EVAL_MAX_BATCHES,
    )
    print_metrics("测试集评估结果（反归一化到 K）", metrics)

    print("\n训练完成! 最佳模型已保存为 best_downscaling_cnn.pth")


# ============================================================
# 思考题 & 动手练习
# Exercises & Hands-on Practice
# ============================================================
#
# ⭐ 入门题 1
# 尝试将残差块数量（num_residual_blocks）分别设为 2、4、8、16，
# 训练后对比验证集上的 RMSE。残差块越多效果一定越好吗？
# 讨论可能出现的过拟合现象。
#
# 💡 提示: 修改 DownscalingCNN(num_residual_blocks=N)，
#          用 calculate_metrics() 记录每种配置的 RMSE 和 PSNR
#
# ⭐ 入门题 2
# 将上采样方式从双线性插值（bilinear）改为双三次插值（bicubic）
# 和最近邻插值（nearest），对比三种方式对最终高分辨率结果的影响。
# 哪种上采样方式作为初始化最有利于后续卷积学习细节？
#
# 💡 提示: 修改 nn.Upsample(mode="bilinear") 中的 mode 参数，
#          尝试 "bicubic" 和 "nearest"
#
# ⭐⭐ 进阶题 3
# 尝试将损失函数从纯 MSE 改为 MSE + 0.1 × MAE 的组合损失，
# 观察训练曲线和生成结果的变化。为什么 MAE 对异常值更鲁棒？
#
# 💡 提示: 在 train_model() 中将 criterion = nn.MSELoss() 改为：
#          loss = mse_loss + 0.1 * mae_loss
