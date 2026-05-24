"""
气象数据降尺度 - CNN 模型
==============================================
任务: 将ERA5低分辨率(1°)温度场降尺度到高分辨率(0.25°)
数据: ERA5再分析数据 (可通过CDS API免费获取)
运行: python downscaling_cnn.py
依赖: pip install torch numpy netCDF4 cdsapi matplotlib scikit-learn scikit-image
==============================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

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

class ERA5DownscalingDataset(Dataset):
    """
    ERA5降尺度数据集
    
    使用模拟数据演示 - 实际使用时替换为真实ERA5数据
    真实数据读取示例:
        import xarray as xr
        ds = xr.open_dataset('era5_t2m_025deg.nc')
        t2m_hr = ds['t2m'].values  # (time, lat, lon)
        t2m_lr = ds['t2m'].coarsen(latitude=4, longitude=4).mean().values
    """
    def __init__(self, n_samples=500, lr_size=(20, 35), hr_size=(80, 140)):
        self.n_samples = n_samples
        self.lr_h, self.lr_w = lr_size
        self.hr_h, self.hr_w = hr_size
        
        print("注意: 当前使用模拟数据。实际使用时请替换为ERA5真实数据。")
        print("ERA5数据获取: https://cds.climate.copernicus.eu/")
        
        self.lr_data = []
        self.hr_data = []
        
        for _ in range(n_samples):
            hr_field = self._generate_realistic_field((self.hr_h, self.hr_w))
            lr_field = self._downsample(hr_field, (self.lr_h, self.lr_w))
            self.lr_data.append(lr_field)
            self.hr_data.append(hr_field)
        
        self.lr_data = np.array(self.lr_data, dtype=np.float32)
        self.hr_data = np.array(self.hr_data, dtype=np.float32)
        
        # 归一化到 [0, 1]
        self.t_min = self.hr_data.min()
        self.t_max = self.hr_data.max()
        self.lr_data = (self.lr_data - self.t_min) / (self.t_max - self.t_min)
        self.hr_data = (self.hr_data - self.t_min) / (self.t_max - self.t_min)
    
    def _generate_realistic_field(self, size):
        """生成具有空间相关性的模拟温度场"""
        from scipy.ndimage import gaussian_filter
        H, W = size
        x = np.linspace(0, 2*np.pi, W)
        y = np.linspace(0, 2*np.pi, H)
        X, Y = np.meshgrid(x, y)
        base = 285 + 15 * np.sin(Y * 0.5)
        noise = np.random.randn(H, W) * 3
        return gaussian_filter(base + noise, sigma=3)
    
    def _downsample(self, field, target_size):
        from skimage.transform import resize
        return resize(field, target_size, anti_aliasing=True)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        lr = torch.FloatTensor(self.lr_data[idx]).unsqueeze(0)
        hr = torch.FloatTensor(self.hr_data[idx]).unsqueeze(0)
        return lr, hr


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            optimizer.zero_grad()
            pred = model(lr_batch)
            min_h = min(pred.shape[2], hr_batch.shape[2])
            min_w = min(pred.shape[3], hr_batch.shape[3])
            loss = criterion(pred[:,:,:min_h,:min_w], hr_batch[:,:,:min_h,:min_w])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
                pred = model(lr_batch)
                min_h = min(pred.shape[2], hr_batch.shape[2])
                min_w = min(pred.shape[3], hr_batch.shape[3])
                val_loss += criterion(pred[:,:,:min_h,:min_w], hr_batch[:,:,:min_h,:min_w]).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_downscaling_cnn.pth')
    
    return model


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

if __name__ == '__main__':
    print("=" * 60)
    print("气象数据降尺度 - CNN 模型训练")
    print("=" * 60)
    
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LR_SIZE = (20, 35)
    HR_SIZE = (80, 140)
    
    print("\\n[1/4] 准备数据集...")
    dataset = ERA5DownscalingDataset(n_samples=500, lr_size=LR_SIZE, hr_size=HR_SIZE)
    n_train = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, len(dataset)-n_train])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"   训练: {n_train} 样本 | 验证: {len(dataset)-n_train} 样本")
    
    print("\\n[2/4] 初始化CNN模型...")
    model = DownscalingCNN(scale_factor=4, num_residual_blocks=4)
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\\n[3/4] 开始训练...")
    model = train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS)
    
    print("\\n[4/4] 评估...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        lr_s, hr_s = next(iter(val_loader))
        pred_s = model(lr_s.to(device))
        min_h = min(pred_s.shape[2], hr_s.shape[2])
        min_w = min(pred_s.shape[3], hr_s.shape[3])
        metrics = calculate_metrics(pred_s[0,0,:min_h,:min_w], hr_s[0,0,:min_h,:min_w].to(device))
    
    print("\\n评估结果:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")
    
    print("\\n训练完成! 下一步: 替换模拟数据为真实ERA5数据")


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
