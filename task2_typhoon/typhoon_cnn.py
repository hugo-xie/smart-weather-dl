"""
台风路径预测 - CNN 模型
==============================================
任务: 基于历史轨迹预测台风未来位置
数据: IBTrACS 西太平洋台风最佳路径数据
输入: 过去12个时次(72小时)的台风特征 [lat, lon, wind, pres]
输出: 未来4个时次(24小时)的台风中心位置
依赖: pip install torch numpy pandas scikit-learn matplotlib
==============================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

# ============================================================
# 数据下载说明
# ============================================================
# 下载IBTrACS西太平洋数据 (约50MB CSV):
# wget https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.WP.list.v04r00.csv
#
# 或Python下载:
# import urllib.request
# url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.WP.list.v04r00.csv"
# urllib.request.urlretrieve(url, "ibtracs_wp.csv")

def load_ibtracs_data(filepath='ibtracs_wp.csv'):
    if not os.path.exists(filepath):
        print(f"文件不存在, 使用模拟数据")
        print("下载: wget https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.WP.list.v04r00.csv -O ibtracs_wp.csv")
        return None
    
    df = pd.read_csv(filepath, skiprows=[1], low_memory=False)
    df = df[['SID', 'ISO_TIME', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'NATURE']].copy()
    for col in ['LAT', 'LON', 'WMO_WIND', 'WMO_PRES']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[df['NATURE'] == 'TS'].dropna(subset=['LAT', 'LON', 'WMO_WIND', 'WMO_PRES'])
    
    tracks = []
    for sid, group in df.groupby('SID'):
        group = group.sort_values('ISO_TIME').reset_index(drop=True)
        if len(group) >= 20:
            tracks.append(group[['LAT', 'LON', 'WMO_WIND', 'WMO_PRES']].values.astype(np.float32))
    
    print(f"加载台风数量: {len(tracks)}")
    return tracks


class IBTrACSDataset(Dataset):
    """
    IBTrACS台风轨迹预测数据集
    
    输入: 过去input_len个时次的特征 (lat, lon, wind, pres)
    输出: 未来output_len个时次的位置 (lat, lon)
    """
    def __init__(self, tracks=None, input_len=12, output_len=4, n_samples=2000):
        self.input_len = input_len
        self.output_len = output_len
        
        if tracks is None:
            print("使用模拟台风轨迹数据 (实际使用时请替换为IBTrACS数据)")
            tracks = self._generate_synthetic_tracks(max(50, n_samples // 20))
        
        self.samples = []
        for track in tracks:
            track_norm = self._normalize_track(track)
            for i in range(len(track_norm) - input_len - output_len + 1):
                x = track_norm[i:i+input_len]
                y = track_norm[i+input_len:i+input_len+output_len, :2]
                self.samples.append((x, y))
        
        print(f"总样本数: {len(self.samples)}")
    
    def _generate_synthetic_tracks(self, n_tracks=100):
        tracks = []
        for _ in range(n_tracks):
            n_steps = np.random.randint(30, 80)
            lat = np.random.uniform(5, 20)
            lon = np.random.uniform(120, 160)
            wind = np.random.uniform(35, 65)
            pres = np.random.uniform(960, 1000)
            track = []
            for t in range(n_steps):
                dlat = np.random.normal(0.3, 0.2) if lat <= 25 else np.random.normal(0.5, 0.2)
                dlon = np.random.normal(-0.5, 0.3) if lat <= 25 else np.random.normal(0.3, 0.3)
                lat = np.clip(lat + dlat, 0, 60)
                lon = np.clip(lon + dlon, 100, 180)
                wind = np.clip(wind + np.random.normal(0, 3), 20, 100)
                pres = np.clip(pres + np.random.normal(0, 2), 900, 1010)
                track.append([lat, lon, wind, pres])
            tracks.append(np.array(track, dtype=np.float32))
        return tracks
    
    def _normalize_track(self, track):
        t = track.copy()
        t[:, 0] = track[:, 0] / 60.0        # lat: [0,60] -> [0,1]
        t[:, 1] = (track[:, 1] - 100) / 80.0 # lon: [100,180] -> [0,1]
        t[:, 2] = track[:, 2] / 100.0         # wind: [0,100] -> [0,1]
        t[:, 3] = (track[:, 3] - 900) / 120.0 # pres: [900,1020] -> [0,1]
        return t
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class TyphoonCNN(nn.Module):
    """
    台风路径预测CNN模型 (1D卷积)
    
    输入: (B, seq_len, num_features)
    输出: (B, output_len, 2) - 未来位置
    """
    def __init__(self, input_len=12, num_features=4, output_len=4, channels=[64, 128, 256]):
        super().__init__()
        self.output_len = output_len
        layers = []
        in_ch = num_features
        for out_ch in channels:
            layers.extend([nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True)])
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, output_len * 2)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)  # (B, features, seq_len)
        x = self.pool(self.conv_layers(x)).squeeze(-1)
        return self.fc(x).view(x.shape[0], self.output_len, 2)


def haversine_distance(pred, target):
    """计算Haversine距离 (km)"""
    pred_lat = pred[:, :, 0] * 60.0
    pred_lon = pred[:, :, 1] * 80.0 + 100.0
    true_lat = target[:, :, 0] * 60.0
    true_lon = target[:, :, 1] * 80.0 + 100.0
    R = 6371.0
    dlat = torch.deg2rad(true_lat - pred_lat)
    dlon = torch.deg2rad(true_lon - pred_lon)
    a = torch.sin(dlat/2)**2 + torch.cos(torch.deg2rad(pred_lat)) * torch.cos(torch.deg2rad(true_lat)) * torch.sin(dlon/2)**2
    return R * 2 * torch.arcsin(torch.sqrt(a.clamp(0, 1)))


def train_typhoon_cnn(model, train_loader, val_loader, num_epochs=50, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b)
            loss = nn.functional.mse_loss(pred, y_b) + 0.01 * haversine_distance(pred, y_b).mean() / 1000
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += nn.functional.mse_loss(pred, y_b).item()
        
        model.eval()
        val_loss = val_hav = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                pred = model(x_b)
                val_loss += nn.functional.mse_loss(pred, y_b).item()
                val_hav += haversine_distance(pred, y_b).mean().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_hav /= len(val_loader)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train: {train_loss:.6f} | Val: {val_loss:.6f} | Haversine: {val_hav:.2f} km")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_typhoon_cnn.pth')
    return model


if __name__ == '__main__':
    print("=" * 60)
    print("台风路径预测 - CNN 模型训练")
    print("=" * 60)
    
    INPUT_LEN = 12
    OUTPUT_LEN = 4
    
    tracks = load_ibtracs_data('ibtracs_wp.csv')
    dataset = IBTrACSDataset(tracks=tracks, input_len=INPUT_LEN, output_len=OUTPUT_LEN)
    n_train = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, len(dataset)-n_train])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    
    model = TyphoonCNN(input_len=INPUT_LEN, num_features=4, output_len=OUTPUT_LEN)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    model = train_typhoon_cnn(model, train_loader, val_loader, num_epochs=50)
    print("训练完成! 下一步: 下载真实IBTrACS数据替换模拟数据")


# ============================================================
# 思考题 & 动手练习
# Exercises & Hands-on Practice
# ============================================================
#
# ⭐ 入门题 1
# 尝试将输入特征从 4 个（lat, lon, wind, pres）扩展到 8 个，
# 新增速度特征（dlat, dlon）和加速度特征（d2lat, d2lon）。
# 这些运动学特征能带来多大的性能提升？
#
# 💡 提示: 在 IBTrACSDataset._normalize_track() 中计算：
#          dlat = lat[t] - lat[t-1]，dlon = lon[t] - lon[t-1]
#          将特征维度从 4 改为 8，同时修改 TyphoonCNN(num_features=8)
#
# ⭐ 入门题 2
# 尝试将预测时长（output_len）从 4 个时次（24小时）
# 延长到 8 个时次（48小时），观察预测误差随预测时长增加的变化规律。
# 为什么长期预测误差增长更快？
#
# 💡 提示: 修改 IBTrACSDataset(output_len=8) 和 TyphoonCNN(output_len=8)，
#          分别记录 24h/48h 的 Haversine 误差（km）
#
# ⭐⭐ 进阶题 3
# 尝试将全局平均池化（AdaptiveAvgPool1d）改为全局最大池化
# （AdaptiveMaxPool1d），对比两种池化方式对台风路径预测的影响。
# 平均池化和最大池化分别保留了哪种时序信息？
#
# 💡 提示: 将 self.pool = nn.AdaptiveAvgPool1d(1) 改为
#          nn.AdaptiveMaxPool1d(1)，对比验证集 Haversine 误差
