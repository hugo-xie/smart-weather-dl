# Task 2 数据处理指南 — 台风路径预测

## 数据集：IBTrACS 最佳路径数据

**来源**：NOAA（美国国家海洋和大气管理局）  
**官网**：https://www.ncei.noaa.gov/products/international-best-track-archive  
**许可**：完全免费，无需注册，直接下载

### 数据特征

| 属性 | 值 |
|------|---|
| 覆盖范围 | 全球（本案例使用西太平洋 WP 盆地） |
| 时间跨度 | 1842 年至今 |
| 时间分辨率 | 6 小时 |
| 台风数量（WP） | ~1,700 条 |
| 关键变量 | 纬度、经度、最大持续风速、最小中心气压 |

---

## Step 1：下载数据

```bash
# 方法一：命令行下载（推荐）
wget https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.WP.list.v04r00.csv -O ibtracs_wp.csv

# 方法二：Python 下载
import urllib.request
url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.WP.list.v04r00.csv"
urllib.request.urlretrieve(url, "ibtracs_wp.csv")
print("下载完成，文件大小约 50 MB")
```

---

## Step 2：读取与清洗数据

```python
import pandas as pd
import numpy as np

# 读取数据（跳过第二行单位行）
df = pd.read_csv('ibtracs_wp.csv', skiprows=[1], low_memory=False)

# 选择关键字段
cols = ['SID', 'ISO_TIME', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'NATURE']
df = df[cols].copy()

# 数值化
for col in ['LAT', 'LON', 'WMO_WIND', 'WMO_PRES']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 只保留热带气旋（TS = Tropical Storm 及以上）
df = df[df['NATURE'] == 'TS'].copy()

# 删除缺失值
df = df.dropna(subset=['LAT', 'LON', 'WMO_WIND', 'WMO_PRES'])

print(f"台风总数: {df['SID'].nunique()}")
print(f"总记录数: {len(df)}")
print(f"时间范围: {df['ISO_TIME'].min()} ~ {df['ISO_TIME'].max()}")
```

---

## Step 3：构建轨迹数据集

```python
# 按台风 ID 分组，提取轨迹
tracks = []
for sid, group in df.groupby('SID'):
    group = group.sort_values('ISO_TIME').reset_index(drop=True)
    # 只保留足够长的轨迹（至少 20 个时次 = 5 天）
    if len(group) >= 20:
        track = group[['LAT', 'LON', 'WMO_WIND', 'WMO_PRES']].values.astype(np.float32)
        tracks.append(track)

print(f"有效台风轨迹数: {len(tracks)}")

# 归一化
def normalize_track(track):
    t = track.copy()
    t[:, 0] = track[:, 0] / 60.0          # lat: [0, 60] -> [0, 1]
    t[:, 1] = (track[:, 1] - 100) / 80.0  # lon: [100, 180] -> [0, 1]
    t[:, 2] = track[:, 2] / 100.0          # wind: [0, 100] -> [0, 1]
    t[:, 3] = (track[:, 3] - 900) / 120.0  # pres: [900, 1020] -> [0, 1]
    return t

# 滑动窗口构建样本
INPUT_LEN = 12   # 过去 72 小时（12 × 6h）
OUTPUT_LEN = 4   # 预测未来 24 小时（4 × 6h）

samples_x, samples_y = [], []
for track in tracks:
    track_norm = normalize_track(track)
    for i in range(len(track_norm) - INPUT_LEN - OUTPUT_LEN + 1):
        x = track_norm[i:i+INPUT_LEN]                           # (12, 4)
        y = track_norm[i+INPUT_LEN:i+INPUT_LEN+OUTPUT_LEN, :2]  # (4, 2) 只预测位置
        samples_x.append(x)
        samples_y.append(y)

samples_x = np.array(samples_x)  # (N, 12, 4)
samples_y = np.array(samples_y)  # (N, 4, 2)
print(f"总样本数: {len(samples_x)}")
```

---

## Step 4：数据集划分

```python
# 按台风 ID 划分（避免数据泄露）
# 建议：按年份划分，2019 年之前训练，2019–2020 年测试

# 简单随机划分示例
from sklearn.model_selection import train_test_split

n = len(samples_x)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)

X_train = samples_x[:n_train]
X_val   = samples_x[n_train:n_train+n_val]
X_test  = samples_x[n_train+n_val:]

y_train = samples_y[:n_train]
y_val   = samples_y[n_train:n_train+n_val]
y_test  = samples_y[n_train+n_val:]
```

| 集合 | 比例 | 样本数（约） |
|------|------|------------|
| 训练集 | 80% | ~40,000 |
| 验证集 | 10% | ~5,000 |
| 测试集 | 10% | ~5,000 |

---

## Step 5：将真实数据替换模拟数据

在各模型文件中，找到 `IBTrACSDataset` 类，将 `tracks=None` 改为传入真实轨迹：

```python
# 加载真实数据
tracks = load_ibtracs_data('ibtracs_wp.csv')

# 创建数据集
dataset = IBTrACSDataset(
    tracks=tracks,       # 传入真实轨迹列表
    input_len=12,
    output_len=4
)
```

---

## 评估指标

| 指标 | 说明 | 参考值（好） |
|------|------|------------|
| **Track Error (km)** | 路径误差（Haversine 距离） | < 100 km (24h) |
| **24h 误差** | 24 小时预报路径误差 | < 150 km |
| **48h 误差** | 48 小时预报路径误差 | < 250 km |
| **72h 误差** | 72 小时预报路径误差 | < 400 km |

> **参考**：业务数值预报模型（如 ECMWF）的 24h 路径误差约为 80–120 km。

---

## 扩展特征建议

加入以下环境场特征可显著提升预测精度：

| 特征 | 来源 | 说明 |
|------|------|------|
| 海表温度（SST） | ERA5 / OISST | 台风能量来源 |
| 850–200 hPa 垂直风切变 | ERA5 | 影响台风强度变化 |
| 相对涡度 | ERA5 | 大气旋转强度 |
| 运动学特征 | 计算 | dlat, dlon（速度）；d2lat, d2lon（加速度） |

```python
# 计算运动学特征示例
def add_kinematic_features(track):
    """在原始 4 特征基础上添加速度和加速度"""
    n = len(track)
    dlat = np.zeros(n)
    dlon = np.zeros(n)
    dlat[1:] = track[1:, 0] - track[:-1, 0]
    dlon[1:] = track[1:, 1] - track[:-1, 1]
    d2lat = np.zeros(n)
    d2lon = np.zeros(n)
    d2lat[2:] = dlat[2:] - dlat[1:-1]
    d2lon[2:] = dlon[2:] - dlon[1:-1]
    return np.column_stack([track, dlat, dlon, d2lat, d2lon])  # (n, 8)
```
