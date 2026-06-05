"""
ERA5 2m temperature data utilities for task1 downscaling.

This file has three responsibilities:
1. Optionally download yearly ERA5 NetCDF files through CDS API.
2. Preprocess downloaded 0.25 degree ERA5 data into normalized train/val/test
   arrays for 4x downscaling.
3. Provide a PyTorch Dataset/DataLoader wrapper used by downscaling_cnn.py.

Default split by year:
    train: 2010-2017
    val:   2018-2019
    test:  2020
"""

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset


THIS_DIR = Path(__file__).resolve().parent
RAW_FILE_TEMPLATE = "era5_t2m_025deg_{year}.nc"
PROCESSED_DIR = THIS_DIR / "processed_era5_t2m_downscaling"
REQUIRED_SPLITS = ("train", "val", "test")
DEFAULT_SPLIT_YEARS = {
    "train": tuple(range(2010, 2018)),
    "val": tuple(range(2018, 2020)),
    "test": (2020,),
}


def download_era5(raw_dir=THIS_DIR, years=range(2010, 2021), overwrite=False):
    """Download yearly ERA5 2m temperature files through CDS API."""
    import cdsapi

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    for year in years:
        target = raw_dir / RAW_FILE_TEMPLATE.format(year=year)
        if target.exists() and not overwrite:
            print(f"跳过已存在文件: {target}")
            continue

        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": "2m_temperature",
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": [55, 70, 15, 140],
                "grid": [0.25, 0.25],
                "format": "netcdf",
            },
            str(target),
        )


def processed_data_exists(data_dir=PROCESSED_DIR):
    """Return True when all processed split files and metadata exist."""
    data_dir = Path(data_dir)
    expected = [data_dir / "metadata.json"]
    for split in REQUIRED_SPLITS:
        expected.extend(
            [
                data_dir / f"{split}_lr.npy",
                data_dir / f"{split}_hr.npy",
                data_dir / f"{split}_time.npy",
            ]
        )
    return all(path.exists() for path in expected)


def load_metadata(data_dir=PROCESSED_DIR):
    metadata_path = Path(data_dir) / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_split_years(split_years):
    split_years = split_years or DEFAULT_SPLIT_YEARS
    missing = [split for split in REQUIRED_SPLITS if split not in split_years]
    if missing:
        raise ValueError(f"split_years 缺少集合: {missing}")
    return {
        split: tuple(int(year) for year in split_years[split])
        for split in REQUIRED_SPLITS
    }


def _raw_file_for_year(raw_dir, year):
    raw_dir = Path(raw_dir)
    exact = raw_dir / RAW_FILE_TEMPLATE.format(year=year)
    if exact.exists():
        return exact

    matches = sorted(raw_dir.glob(f"*{year}*.nc"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"找不到 {year} 年 ERA5 NetCDF 文件。期望路径: {exact}"
        )
    raise FileExistsError(f"{year} 年匹配到多个 NetCDF 文件: {matches}")


def _find_dim(data_array, candidates, label):
    for name in candidates:
        if name in data_array.dims:
            return name

    lowered = {dim.lower(): dim for dim in data_array.dims}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]

    raise ValueError(
        f"无法在变量 {data_array.name!r} 中找到 {label} 维度，"
        f"当前维度: {data_array.dims}"
    )


def _prepare_data_array(dataset, variable):
    if variable in dataset.data_vars:
        variable_name = variable
    elif len(dataset.data_vars) == 1:
        variable_name = next(iter(dataset.data_vars))
        print(f"变量 {variable!r} 不存在，自动使用唯一变量 {variable_name!r}")
    else:
        raise KeyError(
            f"变量 {variable!r} 不存在。可用变量: {list(dataset.data_vars)}"
        )

    data_array = dataset[variable_name]
    time_dim = _find_dim(data_array, ("time", "valid_time"), "time")
    lat_dim = _find_dim(data_array, ("latitude", "lat"), "latitude")
    lon_dim = _find_dim(data_array, ("longitude", "lon"), "longitude")

    extra_dims = [
        dim for dim in data_array.dims if dim not in (time_dim, lat_dim, lon_dim)
    ]
    for dim in extra_dims:
        if data_array.sizes[dim] != 1:
            raise ValueError(
                f"变量 {variable_name!r} 含有非单例额外维度 {dim}: "
                f"{data_array.sizes[dim]}"
            )
        data_array = data_array.isel({dim: 0})

    data_array = data_array.transpose(time_dim, lat_dim, lon_dim)
    return data_array, variable_name, time_dim, lat_dim, lon_dim


def _read_hr_chunk(data_array, time_dim, lat_dim, lon_dim, start, stop, hr_h, hr_w):
    chunk = data_array.isel(
        {
            time_dim: slice(start, stop),
            lat_dim: slice(0, hr_h),
            lon_dim: slice(0, hr_w),
        }
    ).values
    return np.asarray(chunk, dtype=np.float32)


def _coarsen_mean(hr_data, scale_factor):
    n, hr_h, hr_w = hr_data.shape
    lr_h = hr_h // scale_factor
    lr_w = hr_w // scale_factor
    return hr_data.reshape(
        n, lr_h, scale_factor, lr_w, scale_factor
    ).mean(axis=(2, 4), dtype=np.float32)


def _split_limit(max_samples_per_split, split):
    if max_samples_per_split is None:
        return None
    if isinstance(max_samples_per_split, Mapping):
        value = max_samples_per_split.get(split)
        return None if value is None else int(value)
    return int(max_samples_per_split)


def _scan_raw_files(raw_dir, split_years, variable, scale_factor):
    year_files = {}
    year_counts = {}
    first_info = None

    for split in REQUIRED_SPLITS:
        for year in split_years[split]:
            if year in year_files:
                continue

            path = _raw_file_for_year(raw_dir, year)
            year_files[year] = path
            with xr.open_dataset(path) as dataset:
                data_array, variable_name, time_dim, lat_dim, lon_dim = (
                    _prepare_data_array(dataset, variable)
                )
                year_counts[year] = int(data_array.sizes[time_dim])

                if first_info is None:
                    lat_size = int(data_array.sizes[lat_dim])
                    lon_size = int(data_array.sizes[lon_dim])
                    hr_h = (lat_size // scale_factor) * scale_factor
                    hr_w = (lon_size // scale_factor) * scale_factor
                    if hr_h <= 0 or hr_w <= 0:
                        raise ValueError(
                            f"空间尺寸过小，无法按 scale_factor={scale_factor} 降采样: "
                            f"{lat_size}x{lon_size}"
                        )

                    lat_hr = np.asarray(
                        data_array[lat_dim].values[:hr_h], dtype=np.float32
                    )
                    lon_hr = np.asarray(
                        data_array[lon_dim].values[:hr_w], dtype=np.float32
                    )
                    first_info = {
                        "variable": variable_name,
                        "units": data_array.attrs.get("units", ""),
                        "time_dim": time_dim,
                        "lat_dim": lat_dim,
                        "lon_dim": lon_dim,
                        "raw_shape": [lat_size, lon_size],
                        "hr_shape": [hr_h, hr_w],
                        "lr_shape": [hr_h // scale_factor, hr_w // scale_factor],
                        "lat_hr": lat_hr,
                        "lon_hr": lon_hr,
                        "lat_lr": lat_hr.reshape(-1, scale_factor).mean(
                            axis=1, dtype=np.float32
                        ),
                        "lon_lr": lon_hr.reshape(-1, scale_factor).mean(
                            axis=1, dtype=np.float32
                        ),
                    }

    return year_files, year_counts, first_info


def _compute_train_stats(
    raw_dir,
    year_files,
    split_years,
    variable,
    first_info,
    scale_factor,
    chunk_size,
    max_samples_per_split,
):
    hr_h, hr_w = first_info["hr_shape"]
    total = 0.0
    total_sq = 0.0
    count = 0
    remaining = _split_limit(max_samples_per_split, "train")

    print("计算训练集归一化统计量: train=2010-2017")
    for year in split_years["train"]:
        if remaining is not None and remaining <= 0:
            break

        path = year_files[year]
        used_this_year = 0
        with xr.open_dataset(path) as dataset:
            data_array, _, time_dim, lat_dim, lon_dim = _prepare_data_array(
                dataset, variable
            )
            n_time = int(data_array.sizes[time_dim])
            if remaining is not None:
                n_time = min(n_time, remaining)

            for start in range(0, n_time, chunk_size):
                stop = min(start + chunk_size, n_time)
                hr = _read_hr_chunk(
                    data_array,
                    time_dim,
                    lat_dim,
                    lon_dim,
                    start,
                    stop,
                    hr_h,
                    hr_w,
                )
                if not np.isfinite(hr).all():
                    raise ValueError(f"{path} 含有 NaN 或 Inf，需先清洗数据")

                total += float(hr.sum(dtype=np.float64))
                total_sq += float(
                    np.multiply(hr, hr, dtype=np.float64).sum(dtype=np.float64)
                )
                count += int(hr.size)
                used_this_year += int(hr.shape[0])

        if remaining is not None:
            remaining -= used_this_year
        print(f"  {year}: {used_this_year} samples")

    if count == 0:
        raise ValueError("训练集样本数为 0，无法计算归一化统计量")

    mean = total / count
    variance = max(total_sq / count - mean * mean, 1e-12)
    std = math.sqrt(variance)
    print(f"训练集均值: {mean:.6f} K")
    print(f"训练集标准差: {std:.6f} K")
    return mean, std


def _write_split_arrays(
    output_dir,
    split,
    raw_dir,
    year_files,
    years,
    variable,
    first_info,
    mean,
    std,
    scale_factor,
    chunk_size,
    sample_count,
):
    hr_h, hr_w = first_info["hr_shape"]
    lr_h, lr_w = first_info["lr_shape"]
    hr_path = output_dir / f"{split}_hr.npy"
    lr_path = output_dir / f"{split}_lr.npy"
    time_path = output_dir / f"{split}_time.npy"

    hr_out = np.lib.format.open_memmap(
        hr_path, mode="w+", dtype=np.float32, shape=(sample_count, hr_h, hr_w)
    )
    lr_out = np.lib.format.open_memmap(
        lr_path, mode="w+", dtype=np.float32, shape=(sample_count, lr_h, lr_w)
    )

    position = 0
    time_chunks = []
    print(f"写出 {split}: years={list(years)}, samples={sample_count}")

    for year in years:
        if position >= sample_count:
            break

        path = year_files[year]
        with xr.open_dataset(path) as dataset:
            data_array, _, time_dim, lat_dim, lon_dim = _prepare_data_array(
                dataset, variable
            )
            n_time = int(data_array.sizes[time_dim])
            remaining = sample_count - position
            n_time = min(n_time, remaining)
            used_this_year = 0

            for start in range(0, n_time, chunk_size):
                stop = min(start + chunk_size, n_time)
                hr = _read_hr_chunk(
                    data_array,
                    time_dim,
                    lat_dim,
                    lon_dim,
                    start,
                    stop,
                    hr_h,
                    hr_w,
                )
                if not np.isfinite(hr).all():
                    raise ValueError(f"{path} 含有 NaN 或 Inf，需先清洗数据")

                lr = _coarsen_mean(hr, scale_factor)
                hr -= mean
                hr /= std
                lr -= mean
                lr /= std

                n_chunk = int(hr.shape[0])
                hr_out[position : position + n_chunk] = hr
                lr_out[position : position + n_chunk] = lr

                times = data_array[time_dim].isel(
                    {time_dim: slice(start, stop)}
                ).values
                time_chunks.append(np.asarray(times))
                position += n_chunk
                used_this_year += n_chunk

        print(f"  {year}: {used_this_year} samples")

    hr_out.flush()
    lr_out.flush()

    if time_chunks:
        time_values = np.concatenate(time_chunks)
    else:
        time_values = np.asarray([], dtype="datetime64[ns]")
    np.save(time_path, time_values)

    if position != sample_count:
        raise RuntimeError(
            f"{split} 写出样本数不一致: expected={sample_count}, got={position}"
        )


def preprocess_era5(
    raw_dir=THIS_DIR,
    output_dir=PROCESSED_DIR,
    variable="t2m",
    scale_factor=4,
    split_years=None,
    chunk_size=64,
    overwrite=False,
    max_samples_per_split=None,
):
    """
    Preprocess yearly NetCDF files into normalized numpy arrays.

    The low-resolution input is produced by block averaging each 4x4 high-
    resolution patch. High-resolution data are cropped to a multiple of
    scale_factor, so the CNN output and target have matching sizes.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    split_years = _normalize_split_years(split_years)

    if processed_data_exists(output_dir) and not overwrite:
        print(f"发现已预处理数据，直接使用: {output_dir}")
        return load_metadata(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    year_files, year_counts, first_info = _scan_raw_files(
        raw_dir, split_years, variable, scale_factor
    )

    if first_info is None:
        raise ValueError("没有找到任何可处理的 ERA5 文件")

    split_counts = {}
    for split in REQUIRED_SPLITS:
        count = sum(year_counts[year] for year in split_years[split])
        limit = _split_limit(max_samples_per_split, split)
        if limit is not None:
            count = min(count, limit)
        if count <= 0:
            raise ValueError(f"{split} 样本数为 0")
        split_counts[split] = count

    mean, std = _compute_train_stats(
        raw_dir,
        year_files,
        split_years,
        variable,
        first_info,
        scale_factor,
        chunk_size,
        max_samples_per_split,
    )

    np.save(output_dir / "latitude_hr.npy", first_info["lat_hr"])
    np.save(output_dir / "longitude_hr.npy", first_info["lon_hr"])
    np.save(output_dir / "latitude_lr.npy", first_info["lat_lr"])
    np.save(output_dir / "longitude_lr.npy", first_info["lon_lr"])

    for split in REQUIRED_SPLITS:
        _write_split_arrays(
            output_dir,
            split,
            raw_dir,
            year_files,
            split_years[split],
            variable,
            first_info,
            mean,
            std,
            scale_factor,
            chunk_size,
            split_counts[split],
        )

    metadata = {
        "variable": first_info["variable"],
        "units": first_info["units"],
        "raw_dir": str(raw_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "scale_factor": scale_factor,
        "raw_shape": first_info["raw_shape"],
        "hr_shape": first_info["hr_shape"],
        "lr_shape": first_info["lr_shape"],
        "normalization": {
            "source_split": "train",
            "source_years": list(split_years["train"]),
            "mean": mean,
            "std": std,
        },
        "splits": {
            split: {
                "years": list(split_years[split]),
                "samples": split_counts[split],
                "lr_file": f"{split}_lr.npy",
                "hr_file": f"{split}_hr.npy",
                "time_file": f"{split}_time.npy",
            }
            for split in REQUIRED_SPLITS
        },
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"预处理完成: {output_dir}")
    print(f"高分辨率形状: {tuple(first_info['hr_shape'])}")
    print(f"低分辨率形状: {tuple(first_info['lr_shape'])}")
    return metadata


class ERA5DownscalingDataset(Dataset):
    """PyTorch Dataset for normalized ERA5 downscaling arrays."""

    def __init__(
        self,
        split="train",
        data_dir=PROCESSED_DIR,
        mmap=True,
        max_samples=None,
        return_time=False,
        n_samples=None,
        **_legacy_kwargs,
    ):
        if split not in REQUIRED_SPLITS:
            raise ValueError(f"split 必须是 {REQUIRED_SPLITS} 之一，当前: {split}")

        if n_samples is not None and max_samples is None:
            max_samples = n_samples

        self.split = split
        self.data_dir = Path(data_dir)
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

        dataset_len = len(self.lr_data)
        if max_samples is not None:
            dataset_len = min(dataset_len, int(max_samples))
        self._len = dataset_len

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        lr = torch.from_numpy(
            np.array(self.lr_data[idx], dtype=np.float32, copy=True)
        ).unsqueeze(0)
        hr = torch.from_numpy(
            np.array(self.hr_data[idx], dtype=np.float32, copy=True)
        ).unsqueeze(0)

        if self.return_time:
            return lr, hr, str(self.time[idx])
        return lr, hr


def build_dataloaders(
    data_dir=PROCESSED_DIR,
    batch_size=4,
    num_workers=0,
    pin_memory=None,
    max_samples=None,
):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    def max_for(split):
        if isinstance(max_samples, Mapping):
            return max_samples.get(split)
        return max_samples

    train_set = ERA5DownscalingDataset(
        "train", data_dir=data_dir, max_samples=max_for("train")
    )
    val_set = ERA5DownscalingDataset(
        "val", data_dir=data_dir, max_samples=max_for("val")
    )
    test_set = ERA5DownscalingDataset(
        "test", data_dir=data_dir, max_samples=max_for("test")
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


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess downloaded ERA5 NetCDF files for downscaling."
    )
    parser.add_argument("--raw-dir", type=Path, default=THIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--variable", default="t2m")
    parser.add_argument("--scale-factor", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--download",
        action="store_true",
        help="先使用 CDS API 下载 2010-2020 年数据，再预处理。",
    )
    args = parser.parse_args()

    if args.download:
        download_era5(raw_dir=args.raw_dir, years=range(2010, 2021))

    preprocess_era5(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        variable=args.variable,
        scale_factor=args.scale_factor,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
