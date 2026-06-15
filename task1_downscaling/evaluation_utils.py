"""Shared evaluation helpers for task1 downscaling models."""

import math
from itertools import islice

import torch
from tqdm import tqdm


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalization_params(metadata):
    norm = metadata.get("normalization", {})
    return float(norm.get("mean", 0.0)), float(norm.get("std", 1.0))


def limited_loader(data_loader, max_batches=None):
    if max_batches is None or max_batches <= 0:
        return data_loader, len(data_loader)
    return islice(data_loader, max_batches), min(len(data_loader), max_batches)


def init_metric_state():
    return {
        "sse_k": 0.0,
        "sae_k": 0.0,
        "sse_norm": 0.0,
        "count": 0,
        "target_min_k": float("inf"),
        "target_max_k": float("-inf"),
    }


def update_regression_state(state, pred_norm, target_norm, mean, std):
    pred_k = pred_norm * std + mean
    target_k = target_norm * std + mean
    diff_k = pred_k - target_k
    diff_norm = pred_norm - target_norm

    state["sse_k"] += float((diff_k * diff_k).sum().item())
    state["sae_k"] += float(diff_k.abs().sum().item())
    state["sse_norm"] += float((diff_norm * diff_norm).sum().item())
    state["count"] += int(diff_k.numel())
    state["target_min_k"] = min(state["target_min_k"], float(target_k.min().item()))
    state["target_max_k"] = max(state["target_max_k"], float(target_k.max().item()))


def finalize_regression_metrics(state):
    count = max(state["count"], 1)
    rmse = math.sqrt(state["sse_k"] / count)
    mae = state["sae_k"] / count
    mse_norm = state["sse_norm"] / count
    data_range = state["target_max_k"] - state["target_min_k"]
    psnr = float("inf") if data_range <= 0 else 20 * math.log10(data_range / (rmse + 1e-8))
    return {
        "RMSE_K": rmse,
        "MAE_K": mae,
        "PSNR_dB": psnr,
        "MSE_norm": mse_norm,
    }


def evaluate_image_model(model, data_loader, metadata, device=None, desc="Test", max_batches=None):
    if device is None:
        device = model_device(model)
    mean, std = normalization_params(metadata)
    state = init_metric_state()
    model.eval()

    iterable, total = limited_loader(data_loader, max_batches)
    with torch.no_grad():
        for lr_b, hr_b in tqdm(iterable, total=total, desc=desc, leave=False, ncols=100):
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            pred = model(lr_b)
            min_h = min(pred.shape[-2], hr_b.shape[-2])
            min_w = min(pred.shape[-1], hr_b.shape[-1])
            update_regression_state(
                state,
                pred[..., :min_h, :min_w],
                hr_b[..., :min_h, :min_w],
                mean,
                std,
            )
    return finalize_regression_metrics(state)


def evaluate_flat_model(model, data_loader, metadata, device=None, desc="Test", max_batches=None):
    if device is None:
        device = model_device(model)
    mean, std = normalization_params(metadata)
    state = init_metric_state()
    model.eval()

    iterable, total = limited_loader(data_loader, max_batches)
    with torch.no_grad():
        for x_b, target_b in tqdm(iterable, total=total, desc=desc, leave=False, ncols=100):
            x_b, target_b = x_b.to(device), target_b.to(device)
            pred = model(x_b)
            target_flat = target_b.reshape(target_b.shape[0], -1)
            min_size = min(pred.shape[1], target_flat.shape[1])
            update_regression_state(
                state,
                pred[:, :min_size],
                target_flat[:, :min_size],
                mean,
                std,
            )
    return finalize_regression_metrics(state)


def evaluate_diffusion_noise_loss(ddpm, data_loader, device=None, desc="Test noise", max_batches=None):
    if device is None:
        device = model_device(ddpm.model)
    ddpm.model.eval()
    total_loss = 0.0
    total_samples = 0

    iterable, total = limited_loader(data_loader, max_batches)
    with torch.no_grad():
        for lr_b, hr_b in tqdm(iterable, total=total, desc=desc, leave=False, ncols=100):
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            t = torch.randint(0, ddpm.T, (hr_b.shape[0],), device=device)
            loss = ddpm.p_losses(hr_b, lr_b, t)
            batch_size = int(hr_b.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    return {"Noise_MSE": total_loss / max(total_samples, 1)}


def print_metrics(title, metrics):
    print(f"\n{title}:")
    for name, value in metrics.items():
        if isinstance(value, float) and math.isinf(value):
            print(f"   {name}: inf")
        else:
            print(f"   {name}: {value:.4f}")
