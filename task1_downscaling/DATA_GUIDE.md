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

## Step 4：运行预处理脚本

下载完成后，年度 NetCDF 文件应位于 `task1_downscaling/` 目录下，文件名格式为：

```text
era5_t2m_025deg_2010.nc
era5_t2m_025deg_2011.nc
...
era5_t2m_025deg_2020.nc
```

从项目根目录运行：

```bash
python task1_downscaling/download_data.py
```

或者进入任务目录后运行：

```bash
cd task1_downscaling
python download_data.py
```

脚本会执行以下处理：

1. 读取 `2010-2020` 年的 ERA5 年度 NetCDF 文件。
2. 读取变量 `t2m`，自动兼容时间维度名 `time` 或 `valid_time`。
3. 将原始高分辨率网格裁剪到可被 `4` 整除的大小。
4. 使用 `4x4` block mean 从 HR 生成 LR。
5. 只使用训练集年份 `2010-2017` 计算归一化均值和标准差。
6. 对 train/val/test 的 LR 和 HR 使用相同的训练集均值、标准差归一化。
7. 将结果保存为 `.npy`，并写出 `metadata.json`。

当前数据的空间尺寸如下：

| 数据 | 原始尺寸 | 预处理后尺寸 | 说明 |
|------|---------|-------------|------|
| HR | `161 x 281` | `160 x 280` | 裁剪掉最后一行纬度和最后一列经度，使尺寸可被 4 整除 |
| LR | - | `40 x 70` | 由 HR 每 `4x4` 区域平均得到 |

默认输出目录：

```text
task1_downscaling/processed_era5_t2m_downscaling/
```

如果已经存在完整预处理结果，脚本会直接复用。需要强制重新生成时使用：

```bash
python task1_downscaling/download_data.py --overwrite
```

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--raw-dir` | `task1_downscaling/` | 原始年度 `.nc` 文件目录 |
| `--output-dir` | `task1_downscaling/processed_era5_t2m_downscaling/` | 预处理输出目录 |
| `--variable` | `t2m` | NetCDF 中读取的变量名 |
| `--scale-factor` | `4` | 降尺度倍率 |
| `--chunk-size` | `64` | 分块读取时间步数量，显存/内存紧张时可调小 |
| `--overwrite` | 关闭 | 重新生成已有预处理数据 |
| `--download` | 关闭 | 先调用 CDS API 下载，再预处理 |

---

## Step 5：预处理输出文件

预处理完成后，目录结构如下：

```text
processed_era5_t2m_downscaling/
├── metadata.json
├── train_lr.npy
├── train_hr.npy
├── train_time.npy
├── val_lr.npy
├── val_hr.npy
├── val_time.npy
├── test_lr.npy
├── test_hr.npy
├── test_time.npy
├── latitude_hr.npy
├── longitude_hr.npy
├── latitude_lr.npy
└── longitude_lr.npy
```

核心训练文件：

| 文件 | shape | 说明 |
|------|-------|------|
| `train_lr.npy` | `(11688, 40, 70)` | 训练集低分辨率输入 |
| `train_hr.npy` | `(11688, 160, 280)` | 训练集高分辨率目标 |
| `val_lr.npy` | `(2920, 40, 70)` | 验证集低分辨率输入 |
| `val_hr.npy` | `(2920, 160, 280)` | 验证集高分辨率目标 |
| `test_lr.npy` | `(1464, 40, 70)` | 测试集低分辨率输入 |
| `test_hr.npy` | `(1464, 160, 280)` | 测试集高分辨率目标 |

辅助文件：

| 文件 | 说明 |
|------|------|
| `metadata.json` | 保存 split 年份、样本数、空间尺寸、归一化均值和标准差 |
| `*_time.npy` | 每个样本对应的 ERA5 时间戳 |
| `latitude_hr.npy`, `longitude_hr.npy` | HR 网格每一行/列对应的真实纬度、经度 |
| `latitude_lr.npy`, `longitude_lr.npy` | LR 网格每一行/列对应的真实纬度、经度 |

默认年份划分：

| 集合 | 时间范围 | 样本数（6 小时间隔） |
|------|----------|--------------------|
| 训练集 | `2010-2017` | `11688` |
| 验证集 | `2018-2019` | `2920` |
| 测试集 | `2020` | `1464` |

归一化统计量来自训练集 HR 数据，当前预处理结果为：

```text
mean = 284.73544573983804 K
std  = 15.738599515322958 K
```

---

## Step 6：PyTorch Dataset 和 DataLoader

`download_data.py` 已经提供了 `ERA5DownscalingDataset` 和 `build_dataloaders()`，不需要再手动替换模型文件中的模拟数据。

单独使用 Dataset：

```python
from task1_downscaling.download_data import ERA5DownscalingDataset

train_set = ERA5DownscalingDataset(split="train")
lr, hr = train_set[0]

print(lr.shape)  # torch.Size([1, 40, 70])
print(hr.shape)  # torch.Size([1, 160, 280])
```

同时构建 train/val/test DataLoader：

```python
from task1_downscaling.download_data import build_dataloaders

train_loader, val_loader, test_loader = build_dataloaders(
    batch_size=4,
    num_workers=0,
)

for lr_batch, hr_batch in train_loader:
    print(lr_batch.shape)  # (B, 1, 40, 70)
    print(hr_batch.shape)  # (B, 1, 160, 280)
    break
```

Dataset 默认使用 `np.load(..., mmap_mode="r")` 读取 `.npy`，不会一次性把全部数据加载进内存。`__getitem__` 返回时会转换成 `torch.FloatTensor`，并自动增加 channel 维度：

```text
LR: (H_lr, W_lr) -> (1, H_lr, W_lr)
HR: (H_hr, W_hr) -> (1, H_hr, W_hr)
```

如果需要同时返回时间戳：

```python
test_set = ERA5DownscalingDataset(split="test", return_time=True)
lr, hr, time = test_set[0]
print(time)
```

---

## Step 7：运行 CNN 降尺度模型

`downscaling_cnn.py` 已经接入真实 ERA5 数据。直接运行：

```bash
python task1_downscaling/downscaling_cnn.py
```

脚本启动时会检查：

1. 如果 `processed_era5_t2m_downscaling/` 已存在，直接读取。
2. 如果预处理目录不存在，自动调用 `preprocess_era5()` 从年度 `.nc` 文件生成数据。
3. 使用 train split 训练，val split 验证，test split 做最终评估。
4. 保存验证集最优模型到 `best_downscaling_cnn.pth`。

可以用环境变量调整训练规模：

```bash
DOWNSCALING_BATCH_SIZE=4 \
DOWNSCALING_EPOCHS=50 \
DOWNSCALING_NUM_WORKERS=0 \
python task1_downscaling/downscaling_cnn.py
```

快速调试时可以限制每个 split 使用的样本数：

```bash
DOWNSCALING_EPOCHS=1 \
DOWNSCALING_MAX_SAMPLES=8 \
DOWNSCALING_BATCH_SIZE=2 \
python task1_downscaling/downscaling_cnn.py
```

可用环境变量：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `DOWNSCALING_BATCH_SIZE` | `4` | batch size |
| `DOWNSCALING_EPOCHS` | `50` | 训练 epoch 数 |
| `DOWNSCALING_NUM_WORKERS` | `0` | DataLoader worker 数 |
| `DOWNSCALING_MAX_SAMPLES` | `0` | 调试用；大于 0 时每个 split 最多读取该数量样本 |
| `DOWNSCALING_RAW_DIR` | `task1_downscaling/` | 原始 `.nc` 数据目录 |
| `DOWNSCALING_DATA_DIR` | `processed_era5_t2m_downscaling/` | 预处理数据目录 |

---

## Step 8：反归一化和地理坐标

模型训练时使用的是标准化后的温度：

```text
normalized = (temperature_K - mean) / std
```

评估或保存预测结果时，需要反归一化回开尔文：

```python
import json
import numpy as np
from pathlib import Path

data_dir = Path("task1_downscaling/processed_era5_t2m_downscaling")
metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))

mean = metadata["normalization"]["mean"]
std = metadata["normalization"]["std"]

pred_k = pred_norm * std + mean
target_k = target_norm * std + mean
```

将二维预测结果保存成带经纬度的 NetCDF：

```python
import numpy as np
import xarray as xr

lat = np.load(data_dir / "latitude_hr.npy")
lon = np.load(data_dir / "longitude_hr.npy")

da = xr.DataArray(
    pred_k,
    dims=("latitude", "longitude"),
    coords={"latitude": lat, "longitude": lon},
    name="t2m_downscaled",
)
da.to_netcdf("predicted_t2m_sample.nc")
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
