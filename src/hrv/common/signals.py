from __future__ import annotations
from pathlib import Path
import math
import numpy as np


def _resample_to_grid(
    values: np.ndarray,
    times_ms: np.ndarray,
    grid_ms: np.ndarray,
    *,
    max_gap_ms: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """将不等间距的原始信号插值到等间距时间网格上。

    返回 (resampled_signal, valid_mask, valid_ratio)。
    如果离某个网格点最近的原始样本距离超过 *max_gap_ms* （500ms），
    则该网格点被标记为无效（置零 + mask=False）。
    """
    sig = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_ms, dtype=np.float64)

    # 只保留有限值
    valid = np.isfinite(sig) & np.isfinite(t)
    if valid.sum() < 2:
        empty = np.zeros(len(grid_ms), dtype=np.float32)
        return empty, np.zeros(len(grid_ms), dtype=bool), 0.0

    t = t[valid]
    sig = sig[valid]

    # 按时间排序并去重
    order = np.argsort(t)
    t = t[order]
    sig = sig[order]
    unique = np.r_[True, np.diff(t) > 0]
    t = t[unique]
    sig = sig[unique]

    if t.size < 2:
        empty = np.zeros(len(grid_ms), dtype=np.float32)
        return empty, np.zeros(len(grid_ms), dtype=bool), 0.0

    # 线性插值到网格，超出原始数据时间范围的部分填 NaN
    y = np.interp(grid_ms, t, sig, left=np.nan, right=np.nan)

    # 对每个网格点，找最近的原始样本距离
    nearest_idx = np.searchsorted(t, grid_ms)
    left_idx = np.clip(nearest_idx - 1, 0, t.size - 1)
    right_idx = np.clip(nearest_idx, 0, t.size - 1)
    nearest_dist = np.minimum(
        np.abs(grid_ms - t[left_idx]),
        np.abs(grid_ms - t[right_idx]),
    )

    # 有效 = 插值结果有限 且 最近原始样本在 max_gap_ms 范围内
    mask = np.isfinite(y) & (nearest_dist <= max_gap_ms)
    valid_ratio = float(mask.mean()) if mask.size else 0.0

    # 无效位置置零
    y = np.where(mask, y, 0.0)

    return y.astype(np.float32), mask.astype(bool), valid_ratio

def _infer_fs_ms(t_ms: np.ndarray) -> float:
    t = np.asarray(t_ms, dtype=np.float64)
    dt = np.diff(t[np.isfinite(t)])       # 相邻时间差
    dt = dt[(dt > 0) & np.isfinite(dt)]   # 过滤无效差值
    if dt.size == 0:
        return float("nan")
    return float(1000.0 / np.median(dt))   # 中位间隔 -> 采样率

def _ecg_fs_from_raw(ecg_raw: dict[str, np.ndarray]) -> float:
    t_ms = np.asarray(ecg_raw["time_ms"], dtype=np.float64)
    return _infer_fs_ms(t_ms)

def _object_array_cast(items: list[np.ndarray], dtype: np.dtype | type) -> np.ndarray:
    """Store ragged per-window arrays compactly while preserving object layout."""
    out = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        out[i] = np.asarray(item, dtype=dtype)
    return out

def _accel_gravity_reference(dev_raw: dict[str, np.ndarray]) -> float:
    mag = np.sqrt(
        np.asarray(dev_raw["accel_x"], dtype=np.float64) ** 2
        + np.asarray(dev_raw["accel_y"], dtype=np.float64) ** 2
        + np.asarray(dev_raw["accel_z"], dtype=np.float64) ** 2
    )
    med = float(np.nanmedian(mag))
    if 7.0 <= med <= 12.5:
        return 9.80665
    if 0.7 <= med <= 1.3:
        return 1.0
    return med

def _accel_mean_magnitude_window(
    dev_raw: dict[str, np.ndarray],
    *,
    t0: float,
    t1: float,
) -> float:
    t = np.asarray(dev_raw["timestamp"], dtype=np.float64)
    lo = int(np.searchsorted(t, t0, side="left"))
    hi = int(np.searchsorted(t, t1, side="right"))
    if hi <= lo:
        return float("nan")
    mag = np.sqrt(
        np.asarray(dev_raw["accel_x"][lo:hi], dtype=np.float64) ** 2
        + np.asarray(dev_raw["accel_y"][lo:hi], dtype=np.float64) ** 2
        + np.asarray(dev_raw["accel_z"][lo:hi], dtype=np.float64) ** 2
    )
    return float(np.nanmean(mag))

def _accel_motion_mean_magnitude_window(
    dev_raw: dict[str, np.ndarray],
    *,
    t0: float,
    t1: float,
    gravity_reference: float,
) -> float:
    t = np.asarray(dev_raw["timestamp"], dtype=np.float64)
    lo = int(np.searchsorted(t, t0, side="left"))
    hi = int(np.searchsorted(t, t1, side="right"))
    if hi <= lo:
        return float("nan")
    mag = np.sqrt(
        np.asarray(dev_raw["accel_x"][lo:hi], dtype=np.float64) ** 2
        + np.asarray(dev_raw["accel_y"][lo:hi], dtype=np.float64) ** 2
        + np.asarray(dev_raw["accel_z"][lo:hi], dtype=np.float64) ** 2
    )
    return float(np.nanmean(np.abs(mag - gravity_reference)))


def _rawslots_to_grid(
    values: np.ndarray,
    times_ms: np.ndarray,
    grid_ms: np.ndarray,
    *,
    slot_tolerance_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sig = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_ms, dtype=np.float64)
    valid = np.isfinite(sig) & np.isfinite(t)
    out_values = np.full(grid_ms.size, np.nan, dtype=np.float32)
    out_timestamps = np.full(grid_ms.size, np.nan, dtype=np.float64)
    out_mask = np.zeros(grid_ms.size, dtype=bool)
    if valid.sum() == 0:
        return out_values, out_mask, out_timestamps, 0.0

    t = t[valid]
    sig = sig[valid]
    step_ms = float(np.median(np.diff(grid_ms))) if grid_ms.size > 1 else 10.0
    slot_idx = np.rint((t - float(grid_ms[0])) / step_ms).astype(np.int64)
    in_range = (slot_idx >= 0) & (slot_idx < grid_ms.size)
    if not np.any(in_range):
        return out_values, out_mask, out_timestamps, 0.0

    t = t[in_range]
    sig = sig[in_range]
    slot_idx = slot_idx[in_range]
    dist = np.abs(t - grid_ms[slot_idx])
    close = dist <= slot_tolerance_ms
    if not np.any(close):
        return out_values, out_mask, out_timestamps, 0.0

    t = t[close]
    sig = sig[close]
    slot_idx = slot_idx[close]
    dist = dist[close]

    order = np.lexsort((dist, slot_idx))
    slot_sorted = slot_idx[order]
    first = np.r_[True, np.diff(slot_sorted) != 0]
    keep = order[first]
    chosen_slots = slot_idx[keep]
    out_values[chosen_slots] = sig[keep].astype(np.float32)
    out_timestamps[chosen_slots] = t[keep]
    out_mask[chosen_slots] = True
    return out_values, out_mask, out_timestamps, float(out_mask.mean())
