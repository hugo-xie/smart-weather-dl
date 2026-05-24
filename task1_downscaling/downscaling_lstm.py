"""
气象数据降尺度 - LSTM 模型
==============================================
利用时序信息将ERA5低分辨率温度场降尺度
输入: 过去12个时次(3天)的低分辨率场序列
输出: 当前时次的高分辨率场
依赖: pip install torch numpy scikit-image scipy
==============================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ERA5SequenceDataset(Dataset):
    """ERA5时序降尺度数据集"""
    def __init__(self, n_samples=600, seq_len=12, lr_size=(20, 35), hr_size=(80, 140)):
        self.seq_len = seq_len
        lr_h, lr_w = lr_size
        hr_h, hr_w = hr_size
        
        print(f"生成时序数据集: {n_samples}样本, 序列长度={seq_len}")
        
        total_steps = n_samples + seq_len
        time_steps = np.arange(total_steps)
        
        all_hr_fields = []
        for t in time_steps:
            daily_cycle = 3 * np.sin(2 * np.pi * t / 4)
            annual_cycle = 15 * np.sin(2 * np.pi * t / 1460)
            x = np.linspace(0, np.pi, hr_w)
            y = np.linspace(0, np.pi, hr_h)
            X, Y = np.meshgrid(x, y)
            base = 285 + annual_cycle + daily_cycle
            spatial = 10 * np.sin(Y) + 5 * np.cos(X * 2)
            noise = np.random.randn(hr_h, hr_w) * 1.5
            all_hr_fields.append(base + spatial + noise)
        
        all_hr_fields = np.array(all_hr_fields, dtype=np.float32)
        
        from skimage.transform import resize
        all_lr_fields = np.array([
            resize(f, lr_size, anti_aliasing=True) for f in all_hr_fields
        ], dtype=np.float32)
        
        # 归一化
        t_min, t_max = all_hr_fields.min(), all_hr_fields.max()
        all_hr_fields = (all_hr_fields - t_min) / (t_max - t_min)
        all_lr_fields = (all_lr_fields - t_min) / (t_max - t_min)
        
        self.sequences = []
        self.targets = []
        for i in range(n_samples):
            seq = all_lr_fields[i:i+seq_len].reshape(seq_len, -1)
            target = all_hr_fields[i+seq_len].flatten()
            self.sequences.append(seq)
            self.targets.append(target)
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
    
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])


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
            nn.Linear(1024, self.output_size), nn.Sigmoid()
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
        for seq_b, tgt_b in train_loader:
            seq_b, tgt_b = seq_b.to(device), tgt_b.to(device)
            optimizer.zero_grad()
            pred = model(seq_b)
            loss = criterion(pred, tgt_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_b, tgt_b in val_loader:
                val_loss += criterion(model(seq_b.to(device)), tgt_b.to(device)).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_downscaling_lstm.pth')
    return model


if __name__ == '__main__':
    print("=" * 60)
    print("气象数据降尺度 - LSTM 模型训练")
    print("=" * 60)
    
    LR_SIZE = (20, 35)
    HR_SIZE = (80, 140)
    
    dataset = ERA5SequenceDataset(n_samples=600, seq_len=12, lr_size=LR_SIZE, hr_size=HR_SIZE)
    n_train = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, len(dataset)-n_train])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    model = DownscalingLSTM(lr_size=LR_SIZE, hr_size=HR_SIZE, hidden_size=256, num_layers=2)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    model = train_lstm(model, train_loader, val_loader, num_epochs=50)
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
