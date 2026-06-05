"""
气象数据降尺度 - LSTM 模型
==============================================
利用时序信息将ERA5低分辨率温度场降尺度
输入: 过去12个时次(3天)的低分辨率场序列
输出: 当前时次的高分辨率场
依赖: pip install torch numpy scikit-image scipy
==============================================
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from download_data import (
    PROCESSED_DIR,
    load_metadata,
    preprocess_era5,
    processed_data_exists,
)
from evaluation_utils import evaluate_flat_model, print_metrics

class ERA5SequenceDataset(Dataset):
    """ERA5 sequence downscaling dataset built from preprocessed LR/HR arrays."""

    def __init__(
        self,
        split="train",
        data_dir=PROCESSED_DIR,
        seq_len=12,
        mmap=True,
        max_samples=None,
        return_time=False,
        n_samples=None,
        **_legacy_kwargs,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split 必须是 train/val/test，当前: {split}")
        if n_samples is not None and max_samples is None:
            max_samples = n_samples

        self.split = split
        self.data_dir = Path(data_dir)
        self.seq_len = int(seq_len)
        self.return_time = return_time

        if not processed_data_exists(self.data_dir):
            raise FileNotFoundError(
                f"未找到预处理数据: {self.data_dir}\n"
                "请先运行: python task1_downscaling/download_data.py"
            )

        self.metadata = load_metadata(self.data_dir)
        mmap_mode = "r" if mmap else None
        self.lr_data = np.load(self.data_dir / f"{split}_lr.npy", mmap_mode=mmap_mode)
        self.hr_data = np.load(self.data_dir / f"{split}_hr.npy", mmap_mode=mmap_mode)
        self.time = np.load(self.data_dir / f"{split}_time.npy", mmap_mode=mmap_mode)

        if len(self.lr_data) != len(self.hr_data):
            raise ValueError(
                f"{split} LR/HR 样本数不一致: "
                f"{len(self.lr_data)} vs {len(self.hr_data)}"
            )
        if len(self.lr_data) <= self.seq_len:
            raise ValueError(
                f"{split} 样本数 {len(self.lr_data)} 必须大于 seq_len={self.seq_len}"
            )

        dataset_len = len(self.lr_data) - self.seq_len
        if max_samples is not None:
            dataset_len = min(dataset_len, int(max_samples))
        self._len = dataset_len

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        lr_seq = self.lr_data[idx : idx + self.seq_len]
        hr_target = self.hr_data[idx + self.seq_len]

        seq = torch.from_numpy(
            np.array(lr_seq.reshape(self.seq_len, -1), dtype=np.float32, copy=True)
        )
        target = torch.from_numpy(
            np.array(hr_target.reshape(-1), dtype=np.float32, copy=True)
        )

        if self.return_time:
            return seq, target, str(self.time[idx + self.seq_len])
        return seq, target


def build_sequence_dataloaders(
    data_dir=PROCESSED_DIR,
    seq_len=12,
    batch_size=8,
    num_workers=0,
    pin_memory=None,
    max_samples=None,
):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    def max_for(split):
        if isinstance(max_samples, dict):
            return max_samples.get(split)
        return max_samples

    train_set = ERA5SequenceDataset(
        "train", data_dir=data_dir, seq_len=seq_len, max_samples=max_for("train")
    )
    val_set = ERA5SequenceDataset(
        "val", data_dir=data_dir, seq_len=seq_len, max_samples=max_for("val")
    )
    test_set = ERA5SequenceDataset(
        "test", data_dir=data_dir, seq_len=seq_len, max_samples=max_for("test")
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


class DownscalingLSTM(nn.Module):
    """
    气象降尺度LSTM模型 (带时序注意力)
    
    输入: (B, seq_len, lr_h*lr_w)
    输出: (B, hr_h*hr_w)
    """
    def __init__(self, lr_size, hr_size, hidden_size=256, num_layers=2, dropout=0.2):
        super(DownscalingLSTM, self).__init__()
        lr_h, lr_w = lr_size
        hr_h, hr_w = hr_size
        self.input_size = lr_h * lr_w
        self.output_size = hr_h * hr_w
        
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 时序注意力
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Softmax(dim=1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, self.output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.decoder(context)


def train_lstm(model, train_loader, val_loader, num_epochs=50, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} train", leave=False, ncols=100)
        for seq_b, tgt_b in train_bar:
            seq_b, tgt_b = seq_b.to(device), tgt_b.to(device)
            optimizer.zero_grad()
            pred = model(seq_b)
            loss = criterion(pred, tgt_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} val", leave=False, ncols=100)
            for seq_b, tgt_b in val_bar:
                seq_b, tgt_b = seq_b.to(device), tgt_b.to(device)
                batch_loss = criterion(model(seq_b), tgt_b).item()
                val_loss += batch_loss
                val_bar.set_postfix(loss=f"{batch_loss:.4f}")
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_downscaling_lstm.pth')
    return model


def _env_int(name, default):
    value = os.getenv(name)
    return default if value is None else int(value)


if __name__ == '__main__':
    print("=" * 60)
    print("气象数据降尺度 - LSTM 模型训练")
    print("=" * 60)

    BATCH_SIZE = _env_int("DOWNSCALING_BATCH_SIZE", 8)
    NUM_EPOCHS = _env_int("DOWNSCALING_EPOCHS", 50)
    NUM_WORKERS = _env_int("DOWNSCALING_NUM_WORKERS", 0)
    MAX_SAMPLES = _env_int("DOWNSCALING_MAX_SAMPLES", 0)
    EVAL_MAX_BATCHES = _env_int("DOWNSCALING_EVAL_MAX_BATCHES", 0)
    SEQ_LEN = _env_int("DOWNSCALING_SEQ_LEN", 12)
    raw_dir = Path(os.getenv("DOWNSCALING_RAW_DIR", Path(__file__).resolve().parent))
    data_dir = Path(os.getenv("DOWNSCALING_DATA_DIR", PROCESSED_DIR))

    print("\n[1/4] 准备时序数据集...")
    if not processed_data_exists(data_dir):
        print(f"   未找到预处理数据: {data_dir}")
        print("   开始从 ERA5 NetCDF 文件生成 train/val/test 数据...")
        preprocess_era5(raw_dir=raw_dir, output_dir=data_dir)
    else:
        print(f"   使用预处理数据: {data_dir}")

    max_samples = None
    if MAX_SAMPLES > 0:
        max_samples = {"train": MAX_SAMPLES, "val": MAX_SAMPLES, "test": MAX_SAMPLES}
        print(f"   调试模式: 每个 split 最多使用 {MAX_SAMPLES} 个序列样本")

    train_loader, val_loader, test_loader = build_sequence_dataloaders(
        data_dir=data_dir,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        max_samples=max_samples,
    )
    metadata = train_loader.dataset.metadata
    lr_size = tuple(metadata["lr_shape"])
    hr_size = tuple(metadata["hr_shape"])
    seq_s, tgt_s = train_loader.dataset[0]
    print(
        f"   训练: {len(train_loader.dataset)} 序列 | "
        f"验证: {len(val_loader.dataset)} 序列 | "
        f"测试: {len(test_loader.dataset)} 序列"
    )
    print(f"   seq_len: {SEQ_LEN}")
    print(f"   sequence shape: {tuple(seq_s.shape)} | target shape: {tuple(tgt_s.shape)}")

    print("\n[2/4] 初始化 LSTM...")
    model = DownscalingLSTM(lr_size=lr_size, hr_size=hr_size, hidden_size=256, num_layers=2)
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n[3/4] 开始训练...")
    model = train_lstm(model, train_loader, val_loader, num_epochs=NUM_EPOCHS)

    print("\n[4/4] 测试集评估...")
    metrics = evaluate_flat_model(
        model,
        test_loader,
        metadata,
        desc="LSTM test",
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
# 将输入序列长度（seq_len）分别设为 4、8、12、24，
# 对比不同历史时间窗口对降尺度效果的影响。
# 利用更长的历史序列一定能提升性能吗？为什么？
#
# 💡 提示: 修改 ERA5SequenceDataset(seq_len=N)，
#          同时调整 DownscalingLSTM 的输入层大小，记录验证集 RMSE
#
# ⭐⭐ 进阶题 2
# 尝试将单向 LSTM 改为双向 LSTM（bidirectional=True），
# 观察模型性能变化。对于降尺度任务（预测当前时刻的高分辨率场），
# 双向 LSTM 是否合理？讨论其适用性。
#
# 💡 提示: 在 nn.LSTM() 中添加 bidirectional=True，
#          注意需要将 hidden_proj 的输入维度乘以 2
#
# ⭐⭐ 进阶题 3
# 尝试将注意力机制去掉（直接使用最后一个时步的隐状态），
# 对比有无注意力的模型性能差异。
# 在哪种气象场景下注意力机制最有价值？
#
# 💡 提示: 在 DownscalingLSTM.forward() 中将注意力加权改为平均权重，
#          对比训练损失曲线和验证 RMSE
