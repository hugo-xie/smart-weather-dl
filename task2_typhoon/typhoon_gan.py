"""
台风路径预测 - Conditional GAN (WGAN-GP)
==============================================
条件GAN生成台风未来轨迹
使用WGAN-GP提升训练稳定性
依赖: pip install torch numpy pandas
==============================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

torch.backends.cudnn.enabled = False

class TyphoonGenerator(nn.Module):
    """
    台风轨迹生成器
    
    输入: 历史轨迹 + 随机噪声
    输出: 未来轨迹
    """
    def __init__(self, input_len=12, input_size=4, noise_dim=32, output_len=4, output_size=2, hidden_size=128):
        super().__init__()
        self.noise_dim = noise_dim
        self.output_len = output_len
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.noise_proj = nn.Sequential(nn.Linear(noise_dim, hidden_size), nn.LeakyReLU(0.2))
        # decoder输入维度: output_size + hidden_size*2(context) + hidden_size(noise)
        decoder_input_size = output_size + hidden_size * 2 + hidden_size
        self.decoder = nn.LSTM(decoder_input_size, hidden_size, num_layers=2, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
    
    def forward(self, condition, noise=None):
        B, device = condition.shape[0], condition.device
        if noise is None: noise = torch.randn(B, self.noise_dim, device=device)
        enc_out, (h, c) = self.encoder(condition)
        # 合并双向编码器的隐状态作为 context
        context = torch.cat([h[-2], h[-1]], dim=1)  # (B, hidden_size*2)
        noise_feat = self.noise_proj(noise)           # (B, hidden_size)
        decoder_input = condition[:, -1:, :2]         # (B, 1, output_size)
        predictions = []
        h_dec = torch.zeros(2, B, self.hidden_size, device=device)
        c_dec = torch.zeros(2, B, self.hidden_size, device=device)
        for t in range(self.output_len):
            # combined: (B, 1, output_size + hidden_size*2 + hidden_size)
            combined = torch.cat([
                decoder_input,
                context.unsqueeze(1),
                noise_feat.unsqueeze(1)
            ], dim=2)
            out, (h_dec, c_dec) = self.decoder(combined, (h_dec, c_dec))
            pred = self.output_proj(out)
            predictions.append(pred)
            decoder_input = pred
        return torch.cat(predictions, dim=1)


class TyphoonDiscriminator(nn.Module):
    """台风轨迹判别器 (WGAN, 无Sigmoid)"""
    def __init__(self, input_len=12, input_size=4, output_len=4, output_size=2, hidden_size=128):
        super().__init__()
        self.cond_encoder = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.traj_encoder = nn.LSTM(output_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1)
        )
    
    def forward(self, condition, trajectory):
        _, (h_c, _) = self.cond_encoder(condition)
        cond_feat = torch.cat([h_c[-2], h_c[-1]], dim=1)
        _, (h_t, _) = self.traj_encoder(trajectory)
        traj_feat = torch.cat([h_t[-2], h_t[-1]], dim=1)
        return self.classifier(torch.cat([cond_feat, traj_feat], dim=1))


def compute_gradient_penalty(discriminator, condition, real_traj, fake_traj, device):
    alpha = torch.rand(real_traj.shape[0], 1, 1, device=device)
    interpolated = (alpha * real_traj + (1 - alpha) * fake_traj).requires_grad_(True)
    # WGAN-GP needs double backward; cuDNN RNN does not support it.
    with torch.backends.cudnn.flags(enabled=False):
        d_interpolated = discriminator(condition, interpolated)
    gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                     grad_outputs=torch.ones_like(d_interpolated),
                                     create_graph=True, retain_graph=True)[0]
    return ((gradients.reshape(gradients.shape[0], -1).norm(2, dim=1) - 1) ** 2).mean()


def train_typhoon_gan(netG, netD, train_loader, num_epochs=100, lr_g=1e-4, lr_d=1e-4, n_critic=5, lambda_gp=10):
    """WGAN-GP训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG, netD = netG.to(device), netD.to(device)
    opt_G = optim.Adam(netG.parameters(), lr=lr_g, betas=(0.0, 0.9))
    opt_D = optim.Adam(netD.parameters(), lr=lr_d, betas=(0.0, 0.9))
    
    for epoch in range(num_epochs):
        g_loss_total = d_loss_total = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            
            # 训练判别器 (n_critic次)
            for _ in range(n_critic):
                opt_D.zero_grad()
                fake_traj = netG(x_b).detach()
                gp = compute_gradient_penalty(netD, x_b, y_b, fake_traj, device)
                d_loss = -netD(x_b, y_b).mean() + netD(x_b, fake_traj).mean() + lambda_gp * gp
                d_loss.backward(); opt_D.step()
                d_loss_total += d_loss.item()
            
            # 训练生成器
            opt_G.zero_grad()
            fake_traj = netG(x_b)
            g_adv = -netD(x_b, fake_traj).mean()
            g_rec = nn.functional.mse_loss(fake_traj, y_b)
            g_loss = g_adv + 10.0 * g_rec
            g_loss.backward(); opt_G.step()
            g_loss_total += g_loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] G: {g_loss_total/len(train_loader):.4f} | D: {d_loss_total/(len(train_loader)*n_critic):.4f}")
            torch.save(netG.state_dict(), 'best_typhoon_gan_g.pth')
    return netG, netD


if __name__ == '__main__':
    print("台风路径预测 - Conditional GAN (WGAN-GP) 训练")
    from typhoon_cnn import IBTrACSDataset, load_ibtracs_data
    
    tracks = load_ibtracs_data('/root/autodl-tmp/project/project/project/smart-weather-dl-main/dataset/ibtracs.WP.list.v04r00.csv')
    dataset = IBTrACSDataset(tracks=tracks, input_len=12, output_len=4)
    n_train = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, len(dataset)-n_train])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    netG = TyphoonGenerator(input_len=12, input_size=4, noise_dim=32, output_len=4)
    netD = TyphoonDiscriminator(input_len=12, input_size=4, output_len=4)
    print(f"生成器: {sum(p.numel() for p in netG.parameters()):,} | 判别器: {sum(p.numel() for p in netD.parameters()):,}")
    
    netG, netD = train_typhoon_gan(netG, netD, train_loader, num_epochs=100, n_critic=5, lambda_gp=10)
    
    netG.eval()

    from typhoon_utils import quick_ensemble_test_and_plot
    quick_ensemble_test_and_plot(
        val_loader,
        samples_fn=lambda x, n: torch.stack([netG(x) for _ in range(n)]),
        save_path='typhoon_gan_prediction.png',
        title='GAN Typhoon Ensemble Prediction',
        n_samples=20
    )
    print("训练完成!")


# ============================================================
# 思考题 & 动手练习
# Exercises & Hands-on Practice
# ============================================================
#
# ⭐ 入门题 1
# 尝试将噪声维度（noise_dim）分别设为 4、16、64、256，
# 观察噪声维度对生成轨迹多样性和误差的影响。
# 噪声维度过小会导致什么问题？过大又会怎样？
#
# 💡 提示: 修改 TyphoonGenerator(noise_dim=N)，
#          对同一输入生成 20 条轨迹，
#          计算轨迹间的平均标准差衡量多样性
#
# ⭐⭐ 进阶题 2
# 尝试将 n_critic（每训练一次生成器训练判别器的次数）
# 分别设为 1、3、5、10，观察对 WGAN-GP 训练稳定性的影响。
# n_critic 过小或过大分别会导致什么问题？
#
# 💡 提示: 修改 train_typhoon_gan(n_critic=N)，
#          用 matplotlib 画出生成器损失和判别器损失的变化曲线，
#          观察训练稳定性
#
# ⭐⭐⭐ 挑战题 3
# WGAN-GP 中梯度惩罚系数 lambda_gp 当前为 10。
# 尝试将其设为 1、5、10、50，对比对训练稳定性和生成质量的影响。
# 梯度惩罚的理论依据是什么（Lipschitz 约束）？
# 为什么 lambda_gp=10 被广泛采用？
#
# 💡 提示: 修改 train_typhoon_gan(lambda_gp=N)，
#          记录判别器损失和梯度惩罚项的大小，
#          观察不同 lambda_gp 下的 Wasserstein 距离变化
