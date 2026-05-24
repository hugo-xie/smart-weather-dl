"""
台风路径预测 - Transformer 模型
==============================================
使用时序Transformer预测台风路径
特点: 多头注意力 + Warmup学习率 + 自回归推理
依赖: pip install torch numpy pandas
==============================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1)])


class TyphoonTransformer(nn.Module):
    """
    台风路径预测Transformer
    
    编码器: 多头自注意力处理历史轨迹
    解码器: 自回归生成未来轨迹
    """
    def __init__(self, input_size=4, d_model=128, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=2,
                 output_size=2, output_len=4, dropout=0.1):
        super().__init__()
        self.output_len = output_len
        self.src_embedding = nn.Sequential(nn.Linear(input_size, d_model), nn.LayerNorm(d_model))
        self.tgt_embedding = nn.Sequential(nn.Linear(output_size, d_model), nn.LayerNorm(d_model))
        self.pos_encoder = TemporalPositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                           num_encoder_layers=num_encoder_layers,
                                           num_decoder_layers=num_decoder_layers,
                                           dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.output_proj = nn.Linear(d_model, output_size)
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    
    def generate_causal_mask(self, sz, device):
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
    
    def forward(self, src, tgt=None):
        src_emb = self.pos_encoder(self.src_embedding(src))
        if tgt is not None:
            start_token = src[:, -1:, :2]
            tgt_input = torch.cat([start_token, tgt[:, :-1, :]], dim=1)
            tgt_emb = self.pos_encoder(self.tgt_embedding(tgt_input))
            tgt_mask = self.generate_causal_mask(tgt_input.shape[1], src.device)
            output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
            return self.output_proj(output)
        else:
            predictions = []
            decoder_input = src[:, -1:, :2]
            memory = self.transformer.encoder(src_emb)
            for _ in range(self.output_len):
                tgt_emb = self.pos_encoder(self.tgt_embedding(decoder_input))
                tgt_mask = self.generate_causal_mask(decoder_input.shape[1], src.device)
                output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                next_pred = self.output_proj(output[:, -1:, :])
                predictions.append(next_pred)
                decoder_input = torch.cat([decoder_input, next_pred], dim=1)
            return torch.cat(predictions, dim=1)


def train_typhoon_transformer(model, train_loader, val_loader, num_epochs=80, lr=1e-4, warmup_steps=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    def lr_lambda(step):
        if step < warmup_steps: return step / warmup_steps
        progress = (step - warmup_steps) / max(1, num_epochs * len(train_loader) - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b, tgt=y_b)
            loss = nn.functional.mse_loss(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step(); scheduler.step()
            train_loss += loss.item(); global_step += 1
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                pred = model(x_b)
                val_loss += nn.functional.mse_loss(pred, y_b).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_typhoon_transformer.pth')
    return model


if __name__ == '__main__':
    print("台风路径预测 - Transformer 训练")
    from typhoon_cnn import IBTrACSDataset, load_ibtracs_data
    
    tracks = load_ibtracs_data('ibtracs_wp.csv')
    dataset = IBTrACSDataset(tracks=tracks, input_len=12, output_len=4)
    n_train = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, len(dataset)-n_train])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    
    model = TyphoonTransformer(input_size=4, d_model=128, nhead=8,
                                num_encoder_layers=4, num_decoder_layers=2, output_size=2, output_len=4)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    model = train_typhoon_transformer(model, train_loader, val_loader, num_epochs=80)
    print("训练完成!")
