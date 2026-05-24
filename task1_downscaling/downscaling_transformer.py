"""
气象数据降尺度 - Transformer (ViT风格) 模型
==============================================
使用Vision Transformer进行气象场降尺度
全局自注意力捕捉远程气候遥相关
依赖: pip install torch numpy scikit-image scipy
==============================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, L, D = x.shape
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class DownscalingTransformer(nn.Module):
    """
    气象降尺度Transformer (ViT风格)
    
    将低分辨率场分割为Patch序列, 用Transformer编码, 解码重建高分辨率场
    """
    def __init__(self, lr_size, hr_size, patch_size=4,
                 d_model=128, num_heads=4, num_layers=4, ff_dim=256, dropout=0.1):
        super().__init__()
        lr_h, lr_w = lr_size
        hr_h, hr_w = hr_size
        self.lr_h, self.lr_w = lr_h, lr_w
        self.hr_h, self.hr_w = hr_h, hr_w
        self.patch_size = patch_size
        
        n_patches_h = lr_h // patch_size
        n_patches_w = lr_w // patch_size
        self.n_patches = n_patches_h * n_patches_w
        
        patch_dim = patch_size * patch_size
        self.patch_embedding = nn.Sequential(nn.Linear(patch_dim, d_model), nn.LayerNorm(d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        self.encoder = nn.ModuleList([TransformerBlock(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model * self.n_patches, 2048), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2048, hr_h * hr_w), nn.Sigmoid()
        )
    
    def extract_patches(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        H_pad = (p - H % p) % p
        W_pad = (p - W % p) % p
        if H_pad > 0 or W_pad > 0:
            x = nn.functional.pad(x, (0, W_pad, 0, H_pad))
        B, C, H, W = x.shape
        nh, nw = H // p, W // p
        patches = x.reshape(B, C, nh, p, nw, p).permute(0, 2, 4, 1, 3, 5)
        return patches.reshape(B, nh * nw, C * p * p)
    
    def forward(self, x):
        B = x.shape[0]
        patches = self.extract_patches(x)
        tokens = self.patch_embedding(patches)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embedding[:, :tokens.shape[1]]
        for layer in self.encoder:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        flat = tokens[:, 1:, :].reshape(B, -1)
        return self.decoder(flat)


def train_transformer(model, train_loader, val_loader, num_epochs=50, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for lr_b, hr_b in train_loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            hr_flat = hr_b.view(hr_b.shape[0], -1)
            optimizer.zero_grad()
            pred = model(lr_b)
            min_size = min(pred.shape[1], hr_flat.shape[1])
            loss = criterion(pred[:, :min_size], hr_flat[:, :min_size])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_b, hr_b in val_loader:
                lr_b, hr_b = lr_b.to(device), hr_b.to(device)
                hr_flat = hr_b.view(hr_b.shape[0], -1)
                pred = model(lr_b)
                min_size = min(pred.shape[1], hr_flat.shape[1])
                val_loss += criterion(pred[:, :min_size], hr_flat[:, :min_size]).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_downscaling_transformer.pth')
    return model


if __name__ == '__main__':
    print("气象数据降尺度 - Transformer 模型训练")
    from downscaling_cnn import ERA5DownscalingDataset
    
    LR_SIZE = (20, 35)
    HR_SIZE = (80, 140)
    
    dataset = ERA5DownscalingDataset(n_samples=500, lr_size=LR_SIZE, hr_size=HR_SIZE)
    n_train = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, len(dataset)-n_train])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    
    model = DownscalingTransformer(lr_size=LR_SIZE, hr_size=HR_SIZE, patch_size=4,
                                    d_model=128, num_heads=4, num_layers=4)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    model = train_transformer(model, train_loader, val_loader, num_epochs=50)
    print("训练完成!")
