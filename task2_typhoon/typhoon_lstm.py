"""
台风路径预测 - LSTM Seq2Seq 模型
==============================================
序列到序列LSTM预测台风未来轨迹
特点: 双向编码器 + Bahdanau注意力 + Teacher Forcing
依赖: pip install torch numpy pandas
==============================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# 使用CNN案例中的数据集
# from typhoon_cnn import IBTrACSDataset, load_ibtracs_data


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.hidden_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_proj = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        hidden = torch.tanh(self.hidden_proj(torch.cat([hidden[-2], hidden[-1]], dim=1))).unsqueeze(0)
        cell = torch.tanh(self.cell_proj(torch.cat([cell[-2], cell[-1]], dim=1))).unsqueeze(0)
        return outputs, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, hidden_size, encoder_hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size + encoder_hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        dh = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat([dh, encoder_outputs], dim=2)))
        return torch.softmax(self.v(energy).squeeze(2), dim=1)


class LSTMDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, encoder_hidden_size, dropout=0.2):
        super().__init__()
        self.attention = Attention(hidden_size, encoder_hidden_size)
        self.lstm = nn.LSTM(output_size + encoder_hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size + encoder_hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_step, hidden, cell, encoder_outputs):
        input_step = input_step.unsqueeze(1)
        attn_weights = self.attention(hidden.squeeze(0), encoder_outputs).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        lstm_input = torch.cat([input_step, context], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        pred = self.fc(self.dropout(torch.cat([output.squeeze(1), context.squeeze(1), input_step.squeeze(1)], dim=1)))
        return pred, hidden, cell


class TyphoonSeq2SeqLSTM(nn.Module):
    """
    台风路径预测 Seq2Seq LSTM
    
    编码器: 双向LSTM压缩历史轨迹
    解码器: 带注意力的LSTM逐步预测
    """
    def __init__(self, input_size=4, hidden_size=128, num_encoder_layers=2,
                 output_size=2, output_len=4, dropout=0.2):
        super().__init__()
        self.output_len = output_len
        self.encoder = LSTMEncoder(input_size, hidden_size, num_encoder_layers, dropout)
        self.decoder = LSTMDecoder(output_size, hidden_size, hidden_size * 2, dropout)
    
    def forward(self, src, teacher_forcing_ratio=0.5, target=None):
        B = src.shape[0]
        encoder_outputs, (hidden, cell) = self.encoder(src)
        decoder_input = src[:, -1, :2]
        predictions = []
        for t in range(self.output_len):
            pred, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            predictions.append(pred)
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t, :]
            else:
                decoder_input = pred
        return torch.stack(predictions, dim=1)


def train_typhoon_lstm(model, train_loader, val_loader, num_epochs=80, lr=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                               steps_per_epoch=len(train_loader), epochs=num_epochs)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        tf_ratio = max(0.0, 0.5 - epoch * 0.005)
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b, teacher_forcing_ratio=tf_ratio, target=y_b)
            loss = nn.functional.mse_loss(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                pred = model(x_b, teacher_forcing_ratio=0.0)
                val_loss += nn.functional.mse_loss(pred, y_b).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train: {train_loss:.6f} | Val: {val_loss:.6f} | TF: {tf_ratio:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_typhoon_lstm.pth')
    return model


if __name__ == '__main__':
    print("台风路径预测 - LSTM Seq2Seq 训练")
    from typhoon_cnn import IBTrACSDataset, load_ibtracs_data
    
    tracks = load_ibtracs_data('ibtracs_wp.csv')
    dataset = IBTrACSDataset(tracks=tracks, input_len=12, output_len=4)
    n_train = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, len(dataset)-n_train])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    
    model = TyphoonSeq2SeqLSTM(input_size=4, hidden_size=128, num_encoder_layers=2, output_size=2, output_len=4)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    model = train_typhoon_lstm(model, train_loader, val_loader, num_epochs=80)
    print("训练完成!")
