<div align="center">

# 🌤️ 智慧气象深度学习教学案例库

**Smart Weather Deep Learning — Course Baselines**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ERA5](https://img.shields.io/badge/Data-ERA5%20(ECMWF)-0066cc)](https://cds.climate.copernicus.eu/)
[![IBTrACS](https://img.shields.io/badge/Data-IBTrACS%20(NOAA)-00a86b)](https://www.ncei.noaa.gov/products/international-best-track-archive)

*面向「智慧气象」课程的深度学习教学案例，涵盖两大气象任务 × 五种主流模型*

[📖 在线教学平台](https://smartweather-4ppzq5rc.manus.space) · [Task 1: 降尺度](#-task-1--气象数据降尺度) · [Task 2: 台风预测](#-task-2--台风路径预测)

</div>

---

## 📋 项目简介

本仓库为**智慧气象**课程提供完整的深度学习教学案例，包含两个核心气象任务的 baseline 代码。每个案例均使用**公开免费数据集**，内置**模拟数据**可直接运行，同时提供详细的真实数据获取与替换指南。

### 设计原则

- **开箱即用**：所有代码内置模拟数据，无需下载数据集即可运行
- **真实可用**：提供完整的真实数据获取流程，替换模拟数据后效果更好
- **循序渐进**：五种模型从简单到复杂，适合逐步学习
- **对比学习**：相同任务、不同模型，便于横向对比分析

---

## 🗂️ 仓库结构

```
smart-weather-dl/
│
├── 📁 task1_downscaling/          # 任务一：气象数据降尺度
│   ├── downscaling_cnn.py         # CNN（残差超分辨率网络）
│   ├── downscaling_lstm.py        # LSTM（时序注意力网络）
│   ├── downscaling_transformer.py # Transformer（ViT 风格）
│   ├── downscaling_diffusion.py   # Diffusion Model（条件 DDPM）
│   ├── downscaling_gan.py         # GAN（SRGAN + PatchGAN）
│   ├── requirements.txt           # 依赖列表
│   └── DATA_GUIDE.md              # 数据获取与处理指南
│
├── 📁 task2_typhoon/              # 任务二：台风路径预测
│   ├── typhoon_cnn.py             # CNN（1D 卷积时序网络）
│   ├── typhoon_lstm.py            # LSTM（Seq2Seq + 注意力）
│   ├── typhoon_transformer.py     # Transformer（时序编解码器）
│   ├── typhoon_diffusion.py       # Diffusion Model（条件扩散）
│   ├── typhoon_gan.py             # GAN（WGAN-GP）
│   ├── requirements.txt           # 依赖列表
│   └── DATA_GUIDE.md              # 数据获取与处理指南
│
└── README.md                      # 本文件
```

---

## 🌡️ Task 1 · 气象数据降尺度

### 任务描述

将低分辨率（1°，约 100 km）气候模式输出转化为高分辨率（0.25°，约 25 km）气象场，即 **4 倍空间超分辨率**。

```
输入: 低分辨率温度场 (1°×1°)  →  输出: 高分辨率温度场 (0.25°×0.25°)
      [40×70 grid]                       [160×280 grid]
```

### 数据集

| 属性 | 详情 |
|------|------|
| **名称** | ERA5 再分析数据（ECMWF 第五代全球大气再分析） |
| **获取** | [CDS API](https://cds.climate.copernicus.eu/)（免费注册） |
| **变量** | 2m 温度（t2m），可扩展至风速、气压等 |
| **分辨率** | 0.25°（高分辨率目标），1.0°（低分辨率输入） |
| **时间** | 1940 年至今，6 小时间隔 |

### 模型一览

| 模型 | 文件 | 核心思路 | 参数量 | 推荐指数 |
|------|------|---------|--------|---------|
| **CNN** | `downscaling_cnn.py` | 残差块 + 双线性上采样 | ~200K | ⭐⭐⭐⭐⭐ |
| **LSTM** | `downscaling_lstm.py` | 时序注意力 + 全连接解码 | ~500K | ⭐⭐⭐⭐ |
| **Transformer** | `downscaling_transformer.py` | Patch Embedding + 全局自注意力 | ~800K | ⭐⭐⭐⭐ |
| **Diffusion** | `downscaling_diffusion.py` | 条件 DDPM + UNet 去噪 | ~300K | ⭐⭐⭐ |
| **GAN** | `downscaling_gan.py` | SRGAN 生成器 + PatchGAN 判别器 | ~400K | ⭐⭐⭐ |

### 快速开始

```bash
cd task1_downscaling
pip install -r requirements.txt

# 运行 CNN（推荐先从这个开始，约 5 分钟）
python downscaling_cnn.py

# 运行其他模型
python downscaling_lstm.py
python downscaling_transformer.py
python downscaling_diffusion.py   # 建议使用 GPU
python downscaling_gan.py
```

### 评估指标

| 指标 | 含义 | 越好 |
|------|------|------|
| **RMSE** | 均方根误差（K） | 越小 |
| **PSNR** | 峰值信噪比（dB） | 越大 |
| **SSIM** | 结构相似性 [0,1] | 越接近 1 |
| **MAE** | 平均绝对误差（K） | 越小 |

---

## 🌀 Task 2 · 台风路径预测

### 任务描述

基于台风历史轨迹（过去 72 小时），预测未来 24 小时的台风中心位置。

```
输入: 过去 12 个时次的 [lat, lon, wind, pres]  →  输出: 未来 4 个时次的 [lat, lon]
      (72 小时历史，6h 间隔)                           (24 小时预测)
```

### 数据集

| 属性 | 详情 |
|------|------|
| **名称** | IBTrACS v04r00（国际最佳路径档案） |
| **获取** | [直接下载 CSV](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.WP.list.v04r00.csv)（无需注册，约 50 MB） |
| **区域** | 西太平洋（WP 盆地） |
| **变量** | 纬度、经度、最大持续风速（kt）、最小中心气压（hPa） |
| **时间** | 1842 年至今，6 小时间隔 |

### 模型一览

| 模型 | 文件 | 核心思路 | 参数量 | 推荐指数 |
|------|------|---------|--------|---------|
| **CNN** | `typhoon_cnn.py` | 1D 卷积 + 全局平均池化 | ~150K | ⭐⭐⭐⭐⭐ |
| **LSTM** | `typhoon_lstm.py` | 双向编码器 + Bahdanau 注意力 | ~300K | ⭐⭐⭐⭐⭐ |
| **Transformer** | `typhoon_transformer.py` | 时序位置编码 + 自回归解码 | ~500K | ⭐⭐⭐⭐ |
| **Diffusion** | `typhoon_diffusion.py` | 条件扩散 + Transformer 去噪 | ~400K | ⭐⭐⭐ |
| **GAN** | `typhoon_gan.py` | WGAN-GP + 梯度惩罚 | ~350K | ⭐⭐⭐ |

### 快速开始

```bash
cd task2_typhoon
pip install -r requirements.txt

# （可选）下载真实 IBTrACS 数据
wget https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.WP.list.v04r00.csv -O ibtracs_wp.csv

# 运行 CNN（推荐先从这个开始）
python typhoon_cnn.py

# 运行其他模型
python typhoon_lstm.py
python typhoon_transformer.py
python typhoon_diffusion.py
python typhoon_gan.py
```

### 评估指标

| 指标 | 含义 | 参考值（业务模型） |
|------|------|-----------------|
| **Track Error (km)** | Haversine 路径误差 | 24h: ~100 km |
| **24h 误差** | 24 小时预报误差 | < 150 km |
| **48h 误差** | 48 小时预报误差 | < 250 km |
| **72h 误差** | 72 小时预报误差 | < 400 km |

---

## 🧠 五种模型对比

| 模型 | 优势 | 局限 | 适用场景 |
|------|------|------|---------|
| **CNN** | 简单高效，空间局部特征强 | 感受野有限，时序建模弱 | Baseline，快速验证 |
| **LSTM** | 时序依赖建模强，经典可靠 | 难以并行，长序列梯度问题 | 时序预测，Seq2Seq |
| **Transformer** | 全局注意力，可并行训练 | 需要大量数据，计算量大 | 大规模数据，多变量融合 |
| **Diffusion** | 概率性预报，量化不确定性 | 推理慢，训练复杂 | 集合预报，不确定性估计 |
| **GAN** | 生成质量高，细节真实 | 训练不稳定，模式崩溃风险 | 高质量生成，多样性预报 |

---

## 🧪 思考题（每个模型 3 道）

每个模型文件末尾均附有分级思考题，帮助学生通过修改参数主动探索：

- **⭐ 入门**：修改单个超参数（如残差块数量、序列长度），观察性能变化
- **⭐⭐ 进阶**：修改模型结构（如注意力机制、损失函数），分析原理
- **⭐⭐⭐ 挑战**：改变核心机制（如 Beta 调度、梯度惩罚），深入探究

---

## 🚀 在 Google Colab 运行

每个模型均可直接在 Google Colab 免费 GPU 上运行：

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 点击 **文件 → 上传笔记本**，上传对应的 `.ipynb` 文件（从[在线平台](https://smartweather-4ppzq5rc.manus.space)下载）
3. 点击 **运行时 → 更改运行时类型**，选择 **T4 GPU**（免费）
4. 按顺序运行每个代码块（`Shift+Enter`）

---

## 📦 环境配置

### 最低要求

```
Python >= 3.8
PyTorch >= 2.0
CUDA（可选，推荐用于 Diffusion/GAN 模型）
```

### 安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/smart-weather-dl.git
cd smart-weather-dl

# 安装降尺度任务依赖
pip install -r task1_downscaling/requirements.txt

# 安装台风预测任务依赖
pip install -r task2_typhoon/requirements.txt
```

---

## 📚 参考文献

| 模型 | 论文 |
|------|------|
| SRCNN (CNN) | Dong et al., 2015. *Learning a Deep Convolutional Network for Image Super-Resolution*. ECCV. |
| LSTM Seq2Seq | Sutskever et al., 2014. *Sequence to Sequence Learning with Neural Networks*. NeurIPS. |
| Bahdanau Attention | Bahdanau et al., 2015. *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR. |
| Vision Transformer | Dosovitskiy et al., 2021. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR. |
| DDPM | Ho et al., 2020. *Denoising Diffusion Probabilistic Models*. NeurIPS. |
| SRGAN | Ledig et al., 2017. *Photo-Realistic Single Image Super-Resolution Using a GAN*. CVPR. |
| WGAN-GP | Gulrajani et al., 2017. *Improved Training of Wasserstein GANs*. NeurIPS. |
| ERA5 | Hersbach et al., 2020. *The ERA5 global reanalysis*. QJRMS. |
| IBTrACS | Knapp et al., 2010. *The International Best Track Archive for Climate Stewardship (IBTrACS)*. BAMS. |

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。数据集版权归原始数据提供方所有：
- ERA5 数据：© ECMWF，遵循 [Copernicus 许可协议](https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf)
- IBTrACS 数据：NOAA 公开数据，无版权限制

---

<div align="center">

**🌤️ 智慧气象深度学习教学平台**

[在线访问](https://smartweather-4ppzq5rc.manus.space) · 如有问题，欢迎提 Issue

</div>
