# Task 1 数据处理指南 — 气象数据降尺度

## 数据集：ERA5 再分析数据

**来源**：ECMWF（欧洲中期天气预报中心）  
**官网**：https://cds.climate.copernicus.eu/  
**许可**：免费开放，注册即可使用

### 数据特征

| 属性 | 低分辨率输入 | 高分辨率目标 |
|------|------------|------------|
| 空间分辨率 | 1.0° × 1.0°（约 100 km） | 0.25° × 0.25°（约 25 km） |
| 时间分辨率 | 6 小时 | 6 小时 |
| 变量 | 2m 温度（t2m）| 2m 温度（t2m）|
| 区域 | 中国区域（70°E–140°E，15°N–55°N）| 同左 |
| 时间范围 | 2010–2020 年 | 2010–2020 年 |

---

## Step 1：注册 CDS 账号

1. 访问 https://cds.climate.copernicus.eu/ 注册账号
2. 登录后在个人页面获取 API Key（UID 和 Key）
3. 在本地创建配置文件 `~/.cdsapirc`：

```
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
```

---

## Step 2：安装依赖

```bash
pip install cdsapi netCDF4 xarray numpy matplotlib scikit-image scipy
```

---

## Step 3：下载 ERA5 数据

```python
import cdsapi

c = cdsapi.Client()

for year in range(2010, 2021):
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': '2m_temperature',
            'year': str(year),
            'month': [f'{m:02d}' for m in range(1, 13)],
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'area': [55, 70, 15, 140],
            'grid': [0.25, 0.25],
            'format': 'netcdf',
        },
        f'era5_t2m_025deg_{year}.nc'
    )
```

> **提示**：完整下载约需 2–4 小时，文件大小约 50–100 GB（10 年数据）。  
> 建议先下载 1 年数据（约 5–10 GB）进行测试。

---

## Step 4：数据预处理

```python
import xarray as xr
import numpy as np
from skimage.transform import resize

# 读取高分辨率数据
ds = xr.open_dataset('era5_t2m_025deg.nc')
t2m_hr = ds['t2m'].values  # shape: (time, lat, lon)，单位 K

# 生成低分辨率数据（1°）：通过空间粗化
t2m_lr_xr = ds['t2m'].coarsen(latitude=4, longitude=4, boundary='trim').mean()
t2m_lr = t2m_lr_xr.values  # shape: (time, lat_lr, lon_lr)

# 全局归一化（使用训练集统计量）
t_mean = t2m_hr[:int(0.8*len(t2m_hr))].mean()
t_std  = t2m_hr[:int(0.8*len(t2m_hr))].std()

t2m_hr_norm = (t2m_hr - t_mean) / t_std
t2m_lr_norm = (t2m_lr - t_mean) / t_std

print(f"高分辨率形状: {t2m_hr.shape}")
print(f"低分辨率形状: {t2m_lr.shape}")
print(f"温度范围: {t2m_hr.min():.1f} K – {t2m_hr.max():.1f} K")
```

---

## Step 5：数据集划分

| 集合 | 时间范围 | 样本数（6h间隔） |
|------|---------|----------------|
| 训练集 | 2010–2017 年 | ~11,680 |
| 验证集 | 2018–2019 年 | ~2,920 |
| 测试集 | 2020 年 | ~1,460 |

```python
n_total = len(t2m_hr_norm)
n_train = int(0.8 * n_total)
n_val   = int(0.1 * n_total)

train_hr = t2m_hr_norm[:n_train]
val_hr   = t2m_hr_norm[n_train:n_train+n_val]
test_hr  = t2m_hr_norm[n_train+n_val:]

# 对应低分辨率
train_lr = t2m_lr_norm[:n_train]
val_lr   = t2m_lr_norm[n_train:n_train+n_val]
test_lr  = t2m_lr_norm[n_train+n_val:]
```

---

## Step 6：将真实数据替换模拟数据

在各模型文件中，找到 `ERA5DownscalingDataset` 类，将 `__init__` 中的模拟数据生成部分替换为：

```python
def __init__(self, lr_data, hr_data):
    """
    lr_data: numpy array, shape (N, H_lr, W_lr)
    hr_data: numpy array, shape (N, H_hr, W_hr)
    """
    self.lr_data = torch.FloatTensor(lr_data).unsqueeze(1)  # (N, 1, H_lr, W_lr)
    self.hr_data = torch.FloatTensor(hr_data).unsqueeze(1)  # (N, 1, H_hr, W_hr)

def __len__(self):
    return len(self.lr_data)

def __getitem__(self, idx):
    return self.lr_data[idx], self.hr_data[idx]
```

---

## 评估指标

| 指标 | 说明 | 参考值（好） |
|------|------|------------|
| **RMSE** | 均方根误差（K） | < 1.0 K |
| **PSNR** | 峰值信噪比（dB） | > 35 dB |
| **SSIM** | 结构相似性 | > 0.95 |
| **MAE** | 平均绝对误差（K） | < 0.8 K |

---

## 扩展变量建议

除 2m 温度外，还可尝试以下变量：

| 变量名 | CDS 名称 | 说明 |
|--------|---------|------|
| 10m 风速 U 分量 | `10m_u_component_of_wind` | 近地面东西风 |
| 10m 风速 V 分量 | `10m_v_component_of_wind` | 近地面南北风 |
| 地面气压 | `surface_pressure` | 地面大气压强 |
| 总降水量 | `total_precipitation` | 累计降水 |
| 500 hPa 位势高度 | `geopotential` | 大气环流指标 |
