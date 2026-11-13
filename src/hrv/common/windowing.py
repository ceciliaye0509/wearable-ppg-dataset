from __future__ import annotations
from pathlib import Path
import math
import numpy as np


def _candidate_windows(
    ppg_raw: dict[str, dict[str, np.ndarray]],
    ecg_raw: dict[str, np.ndarray],
    *,
    devices: list[str],
    window_sec: int,
    stride_sec: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    # 收集所有设备 + ECG 的起止时间
    starts = [float(ppg_raw[d]["timestamp"][0]) for d in devices]
    ends = [float(ppg_raw[d]["timestamp"][-1]) for d in devices]
    starts.append(float(ecg_raw["time_ms"][0]))
    ends.append(float(ecg_raw["time_ms"][-1]))
    # 取各设备起始时间的最大值、结束时间的最小值 -> 公共重叠区间
    common_start = max(starts)
    common_end = min(ends)
    window_ms = window_sec * 1000.0
    stride_ms = stride_sec * 1000.0
    # 第一个窗口起点：对齐到 stride 的整数倍（向上取整）
    first_t0 = math.ceil(common_start / stride_ms) * stride_ms
    last_t0 = common_end - window_ms  # 最后一个完整窗口的起点
    if last_t0 < first_t0:
        # 公共区间不足以容纳一个完整窗口
        empty = np.array([], dtype=np.float64)
        return empty, empty, {
            "common_start_ms": common_start,
            "common_end_ms": common_end,
            "common_duration_sec": max(0.0, (common_end - common_start) / 1000.0),
        }
    # 生成所有候选窗口的起止时间数组
    t0 = np.arange(first_t0, last_t0 + 0.5 * stride_ms, stride_ms, dtype=np.float64)
    t1 = t0 + window_ms
    return t0, t1, {
        "common_start_ms": common_start,
        "common_end_ms": common_end,
        "common_duration_sec": max(0.0, (common_end - common_start) / 1000.0),
    }

def _boundary_offsets_ms(times_ms: np.ndarray, t0: float, t1: float) -> tuple[float, float, int]:
    t = np.asarray(times_ms, dtype=np.float64)
    lo = int(np.searchsorted(t, t0, side="left"))     # 窗口起点左侧第一个样本
    hi = int(np.searchsorted(t, t1, side="right"))     # 窗口终点右侧第一个样本
    if hi <= lo:
        return float("inf"), float("inf"), 0           # 窗口内无样本
    first = float(t[lo])
    last = float(t[hi - 1])
    return first - t0, t1 - last, hi - lo              # 起始偏移、结束偏移、样本计数
