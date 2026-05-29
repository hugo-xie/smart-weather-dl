"""
台风路径预测 - 简单测试与可视化工具
==============================================
给各模型复用的最小评估/画图代码。
输入和输出均使用 IBTrACSDataset 中的归一化坐标。
"""

import os
import numpy as np
import torch


LAT_SCALE = 60.0
LON_MIN = 100.0
LON_SCALE = 80.0


def denormalize_positions(pos):
    """归一化位置 [lat, lon] -> 真实经纬度。支持 torch.Tensor 或 numpy.ndarray。"""
    if torch.is_tensor(pos):
        arr = pos.detach().cpu().numpy()
    else:
        arr = np.asarray(pos)
    out = arr.copy()
    out[..., 0] = out[..., 0] * LAT_SCALE
    out[..., 1] = out[..., 1] * LON_SCALE + LON_MIN
    return out


def haversine_km(pred, target):
    """计算平均路径误差，返回每个点的 Haversine 距离矩阵，单位 km。"""
    pred = denormalize_positions(pred)
    target = denormalize_positions(target)
    pred_lat = np.deg2rad(pred[..., 0])
    pred_lon = np.deg2rad(pred[..., 1])
    true_lat = np.deg2rad(target[..., 0])
    true_lon = np.deg2rad(target[..., 1])
    dlat = true_lat - pred_lat
    dlon = true_lon - pred_lon
    a = np.sin(dlat / 2) ** 2 + np.cos(pred_lat) * np.cos(true_lat) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@torch.no_grad()
def evaluate_model(model, data_loader, predict_fn=None, max_batches=10, device=None):
    """在少量 batch 上快速测试模型，返回 MSE 和平均路径误差。"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    total_mse, total_hav, total_count = 0.0, 0.0, 0

    for batch_idx, (x_b, y_b) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
        x_b, y_b = x_b.to(device), y_b.to(device)
        pred = predict_fn(model, x_b) if predict_fn is not None else model(x_b)
        batch_size = x_b.shape[0]
        total_mse += torch.nn.functional.mse_loss(pred, y_b).item() * batch_size
        total_hav += haversine_km(pred, y_b).mean() * batch_size
        total_count += batch_size

    return {
        'mse': total_mse / max(1, total_count),
        'haversine_km': total_hav / max(1, total_count),
    }


@torch.no_grad()
def get_one_prediction(model, data_loader, predict_fn=None, device=None):
    """取一个样本做可视化，返回历史轨迹、真实未来、预测未来。"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    x_b, y_b = next(iter(data_loader))
    x_b, y_b = x_b.to(device), y_b.to(device)
    pred = predict_fn(model, x_b[:1]) if predict_fn is not None else model(x_b[:1])
    return x_b[:1].detach().cpu(), y_b[:1].detach().cpu(), pred.detach().cpu()


def plot_prediction(history, target, pred, save_path, title='Typhoon track prediction'):
    """画历史轨迹、真实未来轨迹、预测未来轨迹。"""
    import matplotlib.pyplot as plt

    history_xy = denormalize_positions(history[0, :, :2])
    target_xy = denormalize_positions(target[0])
    pred_xy = denormalize_positions(pred[0])

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.plot(history_xy[:, 1], history_xy[:, 0], 'o-', label='History', color='#2563eb')
    plt.plot(target_xy[:, 1], target_xy[:, 0], 'o-', label='Target', color='#16a34a')
    plt.plot(pred_xy[:, 1], pred_xy[:, 0], 'o--', label='Prediction', color='#dc2626')
    plt.scatter(history_xy[-1, 1], history_xy[-1, 0], s=80, marker='*', color='#f59e0b', label='Forecast start')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def quick_test_and_plot(model, data_loader, save_path, predict_fn=None, title='Typhoon track prediction', max_batches=10):
    """打印快速测试指标，并保存一张单样本预测图。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = evaluate_model(model, data_loader, predict_fn=predict_fn, max_batches=max_batches, device=device)
    history, target, pred = get_one_prediction(model, data_loader, predict_fn=predict_fn, device=device)
    plot_prediction(history, target, pred, save_path, title=title)
    print(f"测试 MSE: {metrics['mse']:.6f} | 平均路径误差: {metrics['haversine_km']:.2f} km")
    print(f"预测图已保存: {save_path}")
    return metrics


def plot_ensemble_prediction(history, target, samples, save_path, title='Typhoon ensemble prediction'):
    """画概率模型的多条生成轨迹和集合平均轨迹。samples: (N, B, T, 2) 或 (N, T, 2)。"""
    import matplotlib.pyplot as plt

    history_xy = denormalize_positions(history[0, :, :2])
    target_xy = denormalize_positions(target[0])
    samples_np = denormalize_positions(samples)
    if samples_np.ndim == 4:
        samples_np = samples_np[:, 0]
    mean_xy = samples_np.mean(axis=0)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.plot(history_xy[:, 1], history_xy[:, 0], 'o-', label='History', color='#2563eb')
    plt.plot(target_xy[:, 1], target_xy[:, 0], 'o-', label='Target', color='#16a34a')
    for i, sample in enumerate(samples_np):
        plt.plot(sample[:, 1], sample[:, 0], '--', color='#dc2626', alpha=0.25, label='Samples' if i == 0 else None)
    plt.plot(mean_xy[:, 1], mean_xy[:, 0], 'o-', color='#7c3aed', linewidth=2, label='Sample mean')
    plt.scatter(history_xy[-1, 1], history_xy[-1, 0], s=80, marker='*', color='#f59e0b', label='Forecast start')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


@torch.no_grad()
def quick_ensemble_test_and_plot(data_loader, samples_fn, save_path, title='Typhoon ensemble prediction', n_samples=20):
    """给 GAN/Diffusion 这类概率模型保存集合轨迹图，并打印集合平均误差。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_b, y_b = next(iter(data_loader))
    x_one = x_b[:1].to(device)
    samples = samples_fn(x_one, n_samples).detach().cpu()
    mean_pred = samples.mean(dim=0)
    err = haversine_km(mean_pred, y_b[:1]).mean()
    spread = samples.std().item()
    plot_ensemble_prediction(x_b[:1], y_b[:1], samples, save_path, title=title)
    print(f"集合平均路径误差: {err:.2f} km | 集合标准差: {spread:.4f}")
    print(f"集合预测图已保存: {save_path}")
    return {'haversine_km': float(err), 'spread': spread}
