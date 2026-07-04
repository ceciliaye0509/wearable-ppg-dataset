# ===========================================================================
# ★ 模块说明（重点）
# ===========================================================================
# 本脚本用于生成「基于原始时间线对齐」的四设备 PPG 数据集，以 ECG R-peak 为标签。
# 与早期基于 5 分钟窗口的 v2 生成器不同，本版本的核心流程为：
#   1. 从原始 PPG/ECG 时间序列出发；
#   2. 在共享的绝对时间网格上切分窗口；
#   3. 将四个 PPG 设备的两个通道统一重采样到 target_fs Hz；
#   4. 仅凭 PPG 采样覆盖率和 ECG 标签质控来决定窗口保留与否；
#   5. PPG SQI、运动指标、PPG 波峰及 PPG-HRV 仅作为元数据；
#   6. ECG R-peak 位置数组作为主要标签。
# ===========================================================================
"""
Generate raw-timeline aligned 4-device PPG dataset with ECG R-peak labels.

This generator is separate from the earlier 5min_windowed-based v2 generator.
It follows the updated dataset definition:

  - start from raw PPG/ECG timelines;
  - cut windows on one shared absolute-time grid;
  - resample all four PPG devices and both channels to a common target_fs Hz grid;
  - include windows using only PPG sample coverage and ECG label QC;
  - keep PPG SQI, motion, PPG peaks, and PPG-derived HRV as metadata only;
  - use ECG R-peak location arrays as the primary label.

Run with the project venv so SciPy and NeuroKit2 are available, e.g.

    /Users/jiqiyu/Desktop/Daily_HRV/venv/bin/python \
      src/heuristic_baselines/generate_rawaligned_4device_dataset.py \
      --dataset-name synced_4device_rawaligned_strict_reference \
      --stride-sec 300
"""
# ---------------------------------------------------------------------------
# 标准库与第三方库导入
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 项目内部模块导入
# 将当前脚本所在目录加入 sys.path，以便直接 import 同级模块
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import config  # noqa: E402  # 项目级配置（路径、常量等）
from algorithms import hrv  # noqa: E402  # HRV 相关算法（IBI 校正、RMSSD 等）
from algorithms.sqa import ppg_sqi  # noqa: E402  # PPG 信号质量指数计算
from preprocess import bandpass_filter  # noqa: E402  # 带通滤波
from io_utils import normalize_participant_id  # noqa: E402  # 统一参与者 ID 格式

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
CHANNELS: tuple[str, ...] = ("ppg_green", "ppg_ir")
DEVICES: tuple[str, ...] = ("Earring", "Ring", "Watch")

DEFAULT_MAX_IBI_CORRECTION_RATIO: float = 0.2 # ECG 每窗口纠正超过 20% 的 IBI → 标记 high_ecg_ibi_correction_ratio
DEFAULT_MAX_MOTION_FRACTION: float = 0.5 # PPG 每窗口运动占比超过 50% → 标记 motion_artifact
DEFAULT_MAX_PPG_IBI_CORRECTION_RATIO: float = 0.5 # PPG 每窗口纠正超过 50% 的 IBI → 标记 high_ppg_ibi_correction_ratio
DEFAULT_MAX_RMSSD_MS: float = 200.0 # PPG 的 RMSSD 超过 200ms → 标记 ppg_rmssd_too_high
DEFAULT_MIN_PPG_SQI: float = 0.4 # PPG 的 SQI 低于 0.4 → 标记 low_ppg_sqi

# ---------------------------------------------------------------------------
# NeuroKit2 导入检查
# neurokit2 用于 ECG 信号清洗和 R-peak 检测，若未安装则直接退出
# ---------------------------------------------------------------------------
try:
    import neurokit2 as nk
except ImportError as exc:  # pragma: no cover
    raise SystemExit("neurokit2 is required. Run with /Users/jiqiyu/Desktop/Daily_HRV/venv/bin/python") from exc

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------
RAW_ROOT = config.HEURISTIC_HF_SUBMISSION_ROOT / "raw_data"  # 原始数据根目录
MAX_PEAKS_PER_WINDOW = 1200  # 每个窗口允许的最大 R-peak 数量（人类心率的生理极限大约 220-240 bpm（极端运动或病理性心动过速）240 bpm × 5 min = 1200 次心跳）


# ---------------------------------------------------------------------------
# _resample_to_grid
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# _preprocess_for_peaks
# ---------------------------------------------------------------------------
def _preprocess_for_peaks(
    x_resampled: np.ndarray,
    fs: float,
    mode: str,
) -> np.ndarray:
    """PPG 波峰检测前的预处理。

    mode='raw' 直接返回原始信号；
    mode='bandpass' 做 0.7–3.5 Hz 带通滤波。
    """
    sig = np.asarray(x_resampled, dtype=np.float64)
    if mode == "raw":
        return sig
    if mode == "bandpass":
        return bandpass_filter(sig, 0.7, 3.5, fs)
    raise ValueError(f"Unknown preprocess mode: {mode}")


# ---------------------------------------------------------------------------
# _ppg_peaks_and_qc
# ---------------------------------------------------------------------------
def _ppg_peaks_and_qc(
    raw_resampled: np.ndarray,
    peak_signal: np.ndarray,
    *,
    fs: float,
    valid_sample_ratio: float,
) -> dict[str, object]:
    """PPG 波峰检测 + IBI/HRV 计算 + 信号质量评估。

    返回包含波峰位置、IBI、RMSSD、SDNN、SQI 等指标的字典。
    """
    try:
        peaks = hrv.detect_ppg_peaks(peak_signal, fs)
    except Exception:
        peaks = np.array([], dtype=np.int64)

    peak_times = peaks.astype(np.float64) / fs * 1000.0 if peaks.size else np.array([], dtype=np.float64)
    amps = raw_resampled[peaks].astype(np.float32) if peaks.size else np.array([], dtype=np.float32)

    # IBI 计算
    if peaks.size >= 2:
        ibi = np.diff(peak_times).astype(np.float64)
        valid_ibi = (ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)
        valid_ibi_ratio = float(valid_ibi.mean()) if ibi.size else 0.0
        ibi_valid = ibi[valid_ibi]
    else:
        ibi = np.array([], dtype=np.float64)
        ibi_valid = np.array([], dtype=np.float64)
        valid_ibi_ratio = 0.0

    # RMSSD / SDNN（需要至少 3 个有效 IBI）
    if ibi_valid.size >= 3:
        corrected, corr_ratio = hrv._correct_ibi_artifacts_with_ratio(ibi_valid)
        diff = np.diff(corrected)
        rmssd = float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")
        sdnn = float(np.std(corrected, ddof=1)) if corrected.size > 1 else float("nan")
    else:
        corrected = np.array([], dtype=np.float64)
        corr_ratio = float("nan")
        rmssd = float("nan")
        sdnn = float("nan")

    # SQI
    try:
        sqi = float(ppg_sqi(peak_signal, fs, peaks))
    except Exception:
        sqi = float("nan")

    return {
        "ppg_peak_times_rel_ms": peak_times.astype(np.float32),
        "ppg_peak_indices_grid": peaks.astype(np.int32),
        "ppg_peak_amplitudes_raw": np.asarray(amps, dtype=np.float32),
        "ppg_ibi_ms": ibi.astype(np.float32),
        "ppg_ibi_corrected_ms": corrected.astype(np.float32),
        "ppg_ibi_correction_ratio": corr_ratio,
        "ppg_rmssd_ms": rmssd,
        "ppg_sdnn_ms": sdnn,
        "ppg_valid_sample_ratio": valid_sample_ratio,
        "ppg_valid_ibi_ratio": valid_ibi_ratio,
        "ppg_sqi": sqi,
    }


# ---------------------------------------------------------------------------
# _ppg_channel_qc
# ---------------------------------------------------------------------------
def _ppg_channel_qc(
    *,
    valid_sample_ratio: float,
    valid_ibi_ratio: float,
    correction_ratio: float,
    rmssd_ms: float,
    sqi: float,
    motion_fraction: float,
    min_valid_sample_ratio: float,
    min_valid_ibi_ratio: float,
    min_sqi: float,
    max_motion_fraction: float,
) -> tuple[bool, str]:
    """PPG 单通道质控：返回 (pass, reason)。"""
    reasons: list[str] = []
    if valid_sample_ratio < min_valid_sample_ratio:
        reasons.append("low_ppg_sample_ratio")
    if valid_ibi_ratio < min_valid_ibi_ratio:
        reasons.append("low_ppg_valid_ibi_ratio")
    if np.isfinite(correction_ratio) and correction_ratio > DEFAULT_MAX_PPG_IBI_CORRECTION_RATIO:
        reasons.append("high_ppg_ibi_correction_ratio")
    if np.isfinite(rmssd_ms) and rmssd_ms > DEFAULT_MAX_RMSSD_MS:
        reasons.append("ppg_rmssd_too_high")
    if not np.isfinite(sqi) or sqi < min_sqi:
        reasons.append("low_ppg_sqi")
    if motion_fraction >= max_motion_fraction:
        reasons.append("motion_artifact")
    if not np.isfinite(rmssd_ms):
        reasons.append("invalid_ppg_hrv")

    return (len(reasons) == 0, "ok" if not reasons else ";".join(reasons))


# ---------------------------------------------------------------------------
# _simple_ecg_qrs_sqi
# ---------------------------------------------------------------------------
def _simple_ecg_qrs_sqi(
    ecg: np.ndarray,
    r_samples: np.ndarray,
    fs: float,
) -> float:
    """基于 QRS 突出度和一致性的简单 ECG 信号质量指数。

    返回 0.0-1.0 的综合评分 (0.7SNR + 0.3×一致性)。
    """
    sig = np.asarray(ecg, dtype=np.float64)
    samples = np.asarray(r_samples, dtype=np.int64)

    # R-peak 不足 3 个或采样率无效时无法评估
    if samples.size < 3 or not np.isfinite(fs) or fs <= 0:
        return float("nan")

    local_half = max(1, int(round(0.05 * fs)))   # QRS 局部窗口半宽（~50ms，覆盖一个 QRS 复合波）
    noise_half = max(1, int(round(0.3 * fs)))     # 噪声估计窗口半宽（~300ms，覆盖 QRS 周围的背景信号）

    # --- 对每个 R-peak 计算信噪比 ---
    scores: list[float] = []
    for s in samples:
        # 大窗口（±300ms）：用于估计背景噪声水平
        lo = max(0, int(s) - noise_half)
        hi = min(sig.size, int(s) + noise_half + 1)
        # 小窗口（±50ms）：只看 QRS 波本身
        local_lo = max(0, int(s) - local_half)
        local_hi = min(sig.size, int(s) + local_half + 1)

        noise_win = sig[lo:hi]              # 背景噪声窗口
        local_win = sig[local_lo:local_hi]  # QRS 局部窗口

        if noise_win.size < 5 or local_win.size < 3:
            continue

        # 用中位数估计基线水平，MAD×1.4826 估计噪声标准差（比 std 更抗异常值）
        # 对正态分布来说：MAD = median(|x - median(x)|) ≈ 0.6745 × σ，所以 σ ≈ MAD × (1 / 0.6745) = MAD × 1.4826
        baseline = float(np.nanmedian(noise_win))
        mad = float(np.nanmedian(np.abs(noise_win - baseline))) * 1.4826
        # QRS 波相对于基线的最大偏移幅度
        local_peak = float(np.nanmax(np.abs(local_win - baseline)))

        if not np.isfinite(local_peak) or not np.isfinite(mad):
            continue

        # 信噪比 = QRS 波高度 / 噪声水平（越大说明 QRS 波越突出）
        scores.append(local_peak / (mad + 1e-6))

    if len(scores) < 3:
        return float("nan")

    # --- 综合所有 R-peak 的信噪比，计算最终评分 ---
    scores_arr = np.asarray(scores, dtype=np.float64)
    # SNR 分量：中位信噪比 / 10，截断到 [0, 1]（信噪比 ≥ 10 得满分）
    snr_score = float(np.clip(np.nanmedian(scores_arr) / 10.0, 0.0, 1.0))
    # 一致性分量：各 R-peak 信噪比的变异系数越小 → 一致性越高
    consistency = 1.0 - float(np.clip(
        np.nanstd(scores_arr) / (np.nanmean(scores_arr) + 1e-6),
        0.0, 1.0,
    ))

    # 最终评分 = 70% 看信噪比够不够高 + 30% 看各心跳之间一不一致
    return float(np.clip(0.7 * snr_score + 0.3 * consistency, 0.0, 1.0))

# ---------------------------------------------------------------------------
# 辅助函数：获取参与者 ID 列表
# 若命令行指定了参与者，则按逗号分割；否则从原始数据目录自动扫描
# ---------------------------------------------------------------------------
def _participant_ids(raw_root: Path, raw: str | None) -> list[str]:
    if raw:
        return [normalize_participant_id(p) for p in raw.split(",") if p.strip()]
    return sorted((p.name for p in raw_root.iterdir() if p.is_dir()), key=lambda x: int(x[1:]))


# ---------------------------------------------------------------------------
# 辅助函数：构建原始数据文件路径
# 路径格式: raw_root / P{id} / P{id}_{device}_raw.npz
# ---------------------------------------------------------------------------
def _raw_path(raw_root: Path, pid: str, name: str) -> Path:
    return raw_root / pid / f"{pid}_{name}_raw.npz"


# ---------------------------------------------------------------------------
# 辅助函数：加载指定参与者的所有设备的原始 PPG 数据
# 返回嵌套字典：{设备名: {字段名: ndarray}}
# ---------------------------------------------------------------------------
def _load_ppg_raw(raw_root: Path, pid: str, devices: list[str]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for dev in devices:
        path = _raw_path(raw_root, pid, dev)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as z:
            out[dev] = {k: np.asarray(z[k]) for k in z.files}
    return out


# ---------------------------------------------------------------------------
# 辅助函数：加载指定参与者的原始 ECG 数据（Polar 设备）
# ---------------------------------------------------------------------------
def _load_ecg_raw(raw_root: Path, pid: str) -> dict[str, np.ndarray]:
    path = raw_root / pid / f"{pid}_polar_ecg_raw.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


# ---------------------------------------------------------------------------
# 辅助函数：从时间戳序列（毫秒）推断采样率 (Hz)
# 计算相邻有效时间戳的中位间隔，然后转换为频率
# ---------------------------------------------------------------------------
def _infer_fs_ms(t_ms: np.ndarray) -> float:
    t = np.asarray(t_ms, dtype=np.float64)
    dt = np.diff(t[np.isfinite(t)])       # 相邻时间差
    dt = dt[(dt > 0) & np.isfinite(dt)]   # 过滤无效差值
    if dt.size == 0:
        return float("nan")
    return float(1000.0 / np.median(dt))   # 中位间隔 -> 采样率


# ---------------------------------------------------------------------------
# 辅助函数：创建指定形状的空 object 数组
# 每个元素初始化为空的 float32 数组，用于存储变长的波峰信息
# ---------------------------------------------------------------------------
def _empty_object_array(shape: tuple[int, ...]) -> np.ndarray:
    arr = np.empty(shape, dtype=object)
    arr.fill(np.array([], dtype=np.float32))
    return arr


# ---------------------------------------------------------------------------
# ★ 辅助函数：计算窗口边界偏移量（重点）
# 给定一段时间戳和窗口 [t0, t1]，计算该设备在窗口起止处的偏移距离。
# 返回 (起始偏移_ms, 结束偏移_ms, 落入窗口的样本数)。
# 这些偏移会与 alignment_tolerance 比较，判断该设备是否在此窗口有效对齐。
# ---------------------------------------------------------------------------
def _boundary_offsets_ms(times_ms: np.ndarray, t0: float, t1: float) -> tuple[float, float, int]:
    t = np.asarray(times_ms, dtype=np.float64)
    lo = int(np.searchsorted(t, t0, side="left"))     # 窗口起点左侧第一个样本
    hi = int(np.searchsorted(t, t1, side="right"))     # 窗口终点右侧第一个样本
    if hi <= lo:
        return float("inf"), float("inf"), 0           # 窗口内无样本
    first = float(t[lo])
    last = float(t[hi - 1])
    return first - t0, t1 - last, hi - lo              # 起始偏移、结束偏移、样本计数


# ===========================================================================
# ★★ 核心函数：生成候选窗口列表（重点）
# ===========================================================================
# 找到所有 PPG 设备和 ECG 的公共时间重叠区间，然后按照 stride 在该区间上
# 生成等间距的窗口起止时间。这是整个数据集切分的基础。
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 辅助函数：从原始 ECG 数据推断采样率
# ---------------------------------------------------------------------------
def _ecg_fs_from_raw(ecg_raw: dict[str, np.ndarray]) -> float:
    t_ms = np.asarray(ecg_raw["time_ms"], dtype=np.float64)
    return _infer_fs_ms(t_ms)


# ===========================================================================
# ★★★ 核心函数：单窗口 ECG 标签生成与质控（最重点）
# ===========================================================================
# 对给定窗口 [t0, t1] 提取 ECG 片段，执行以下步骤：
#   1. 用 NeuroKit2 Pan-Tompkins 算法检测 R-peak
#   2. 计算 RR 间期、校正 IBI、RMSSD、SDNN、HR 等 HRV 指标
#   3. 综合多项质控条件判断该窗口的 ECG 标签是否可信
# 此函数的输出直接决定窗口是否被保留——是整个数据集生成的核心。
# ---------------------------------------------------------------------------
def _ecg_label_for_window(
    ecg_raw: dict[str, np.ndarray],
    ecg_fs: float,
    *,
    t0: float,
    t1: float,
    target_fs: float,
    min_valid_ibi_ratio: float,
    min_hr_bpm: float,
    max_hr_bpm: float,
    min_ecg_valid_sample_ratio: float,
    detector_method: str,
) -> dict[str, object]:
    window_ms = t1 - t0
    ecg = np.asarray(ecg_raw["ecg_uv"], dtype=np.float64)
    raw_t = np.asarray(ecg_raw["time_ms"], dtype=np.float64)
    # 用二分搜索定位窗口在原始 ECG 中的索引范围
    lo = int(np.searchsorted(raw_t, t0, side="left"))
    hi = int(np.searchsorted(raw_t, t1, side="right"))
    seg = ecg[lo:hi]  # 截取窗口内的 ECG 信号片段
    expected_n = max(1, int(round((window_ms / 1000.0) * ecg_fs)))  # 理论应有的样本数
    valid_ratio = float(np.isfinite(seg).sum() / expected_n) if seg.size else 0.0  # 有效样本覆盖率

    # ★ R-peak 检测：使用 NeuroKit2 Pan-Tompkins 算法
    peak_times = np.array([], dtype=np.float64)
    local_peak_idx = np.array([], dtype=np.int64)
    clean = np.array([], dtype=np.float64)
    if seg.size >= 50 and np.isfinite(ecg_fs) and ecg_fs > 0 and np.isfinite(seg).all():
        try:
            clean = nk.ecg_clean(seg, sampling_rate=ecg_fs, method="pantompkins1985")  # ECG 信号清洗
            _, info = nk.ecg_peaks(clean, sampling_rate=ecg_fs, method="pantompkins1985")  # R-peak 检测
            local_peak_idx = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64).ravel()
            local_peak_idx = local_peak_idx[(local_peak_idx >= 0) & (local_peak_idx < seg.size)]  # 过滤越界索引
            if local_peak_idx.size:
                peak_times = raw_t[lo + local_peak_idx]  # 转换为绝对时间戳
        except Exception:
            peak_times = np.array([], dtype=np.float64)
            local_peak_idx = np.array([], dtype=np.int64)

    # ★ HRV 指标计算
    rel_ms = peak_times - t0                         # R-peak 相对于窗口起点的时间（ms）
    peak_count = int(rel_ms.size)                     # R-peak 计数
    # 计算 RR 间期（相邻 R-peak 时间差）
    rr = np.diff(peak_times).astype(np.float64) if peak_times.size >= 2 else np.array([], dtype=np.float64)
    valid_rr = (rr >= hrv.IBI_MIN_MS) & (rr <= hrv.IBI_MAX_MS)  # 在生理合理范围内的 RR 间期
    valid_ibi_ratio = float(valid_rr.mean()) if rr.size else 0.0  # 有效 IBI 比例
    nn = rr[valid_rr]  # 仅保留有效的 RR 间期（NN 间期）
    if nn.size >= 3:
        # 校正 IBI 伪影（异常间期修正）
        corrected, corr_ratio = hrv._correct_ibi_artifacts_with_ratio(nn)
        diff = np.diff(corrected)
        rmssd = float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")  # 连续差值均方根
        sdnn = float(np.std(corrected, ddof=1)) if corrected.size > 1 else float("nan")  # NN 间期标准差
        hr_bpm = float(60000.0 / np.nanmean(corrected)) if np.nanmean(corrected) > 0 else float("nan")  # 心率 BPM
        rr_cv = float(np.nanstd(corrected) / np.nanmean(corrected)) if np.nanmean(corrected) > 0 else float("nan")  # 变异系数
    else:
        corrected = np.array([], dtype=np.float64)
        corr_ratio = float("nan")
        rmssd = float("nan")
        sdnn = float("nan")
        hr_bpm = float("nan")
        rr_cv = float("nan")

    # ★ QRS 信号质量指数与 R-peak 振幅
    qrs_sqi = _simple_ecg_qrs_sqi(clean if clean.size else seg, local_peak_idx, ecg_fs)
    peak_amp = []
    for p in local_peak_idx:
        if 0 <= int(p) < seg.size:
            peak_amp.append(float(seg[int(p)]))
    # ★★ ECG 标签质控判断（重点）
    # 综合以下条件决定该窗口的 ECG 标签是否可信：
    #   - ECG 有效样本覆盖率 >= 阈值（0.95）
    #   - 有效 IBI 比例 >= 阈值（1）
    #   - R-peak 数量足够（至少 3 个，且不低于基于最低心率计算的下限）
    #   - RMSSD 有限且不超标
    #   - IBI 校正比例不超标
    #   - 心率在合理范围内
    min_peak_count = int(math.floor((window_ms / 1000.0) * min_hr_bpm / 60.0))
    qc_pass = bool(
        valid_ratio >= min_ecg_valid_sample_ratio
        and valid_ibi_ratio >= min_valid_ibi_ratio
        and peak_count >= max(3, min_peak_count)
        and np.isfinite(rmssd)
        and (not np.isfinite(corr_ratio) or corr_ratio <= DEFAULT_MAX_IBI_CORRECTION_RATIO)
        and (not np.isfinite(hr_bpm) or (min_hr_bpm <= hr_bpm <= max_hr_bpm))
        and (not np.isfinite(rmssd) or rmssd <= DEFAULT_MAX_RMSSD_MS)
    )
    # 收集质控失败原因（用于诊断和调试）
    reasons: list[str] = []
    if valid_ratio < min_ecg_valid_sample_ratio:
        reasons.append("low_ecg_sample_ratio")          # ECG 采样覆盖率不足
    if valid_ibi_ratio < min_valid_ibi_ratio:
        reasons.append("low_ecg_valid_ibi_ratio")       # 有效 IBI 比例过低
    if peak_count < max(3, min_peak_count):
        reasons.append("too_few_ecg_peaks")             # R-peak 数量不足
    if not np.isfinite(rmssd):
        reasons.append("invalid_ecg_hrv")               # HRV 指标无效
    if np.isfinite(corr_ratio) and corr_ratio > DEFAULT_MAX_IBI_CORRECTION_RATIO:
        reasons.append("high_ecg_ibi_correction_ratio") # IBI 校正比例过高
    if np.isfinite(hr_bpm) and not (min_hr_bpm <= hr_bpm <= max_hr_bpm):
        reasons.append("ecg_hr_out_of_range")           # 心率超出合理范围
    if np.isfinite(rmssd) and rmssd > DEFAULT_MAX_RMSSD_MS:
        reasons.append("ecg_rmssd_too_high")            # RMSSD 过高
    if detector_method == "bad_ecg_fs":
        reasons.append(detector_method)                  # ECG 采样率异常

    # 返回完整的 ECG 标签和质控结果字典
    return {
        "ecg_r_peak_times_rel_ms": rel_ms.astype(np.float32),          # R-peak 相对时间（主标签）
        "ecg_r_peak_indices_grid": np.rint(rel_ms / 1000.0 * target_fs).astype(np.int32),  # 映射到 target_fs 网格的索引
        "ecg_r_peak_amplitudes_raw": np.asarray(peak_amp, dtype=np.float32),  # R-peak 原始振幅
        "ecg_rr_intervals_ms": rr.astype(np.float32),                  # RR 间期
        "ecg_rr_intervals_corrected_ms": corrected.astype(np.float32), # 校正后 RR 间期
        "ecg_rmssd_ms": rmssd,                                         # RMSSD (ms)
        "ecg_sdnn_ms": sdnn,                                           # SDNN (ms)
        "ecg_valid_sample_ratio": valid_ratio,                         # ECG 有效样本覆盖率
        "ecg_valid_ibi_ratio": valid_ibi_ratio,                        # 有效 IBI 比例
        "ecg_ibi_correction_ratio": corr_ratio,                        # IBI 校正比例
        "ecg_qc_pass": qc_pass,                                        # 质控是否通过
        "ecg_qc_reason": "ok" if qc_pass else ";".join(reasons),       # 质控原因
        "ecg_label_qc_pass": qc_pass,                                  # 标签质控是否通过
        "ecg_label_qc_reason": "ok" if qc_pass else ";".join(reasons), # 标签质控原因
        "ecg_qrs_sqi": qrs_sqi,                                        # QRS 信号质量指数
        "ecg_peak_count": peak_count,                                  # R-peak 计数
        "ecg_hr_bpm": hr_bpm,                                          # 心率 (BPM)
        "ecg_rr_cv": rr_cv,                                            # RR 变异系数
    }


# ---------------------------------------------------------------------------
# 辅助函数：从原始数据计算每个设备的运动检测阈值
# 将加速度计数据按段（seg_sec = 10s）切分，计算每段的加速度幅值标准差，
# 然后取指定百分位数（第 75 百分位数）作为该设备的运动阈值
# ---------------------------------------------------------------------------
def _motion_thresholds_from_raw(
    ppg_raw: dict[str, dict[str, np.ndarray]],
    *,
    percentile: float,
    seg_sec: float,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for dev, d in ppg_raw.items():
        t = np.asarray(d["timestamp"], dtype=np.float64)
        fs = _infer_fs_ms(t)
        seg_n = max(1, int(round(seg_sec * fs))) if np.isfinite(fs) else 1000
        # 计算三轴加速度的合成幅值
        mag = np.sqrt(
            np.asarray(d["accel_x"], dtype=np.float64) ** 2
            + np.asarray(d["accel_y"], dtype=np.float64) ** 2
            + np.asarray(d["accel_z"], dtype=np.float64) ** 2
        )
        # 按段计算标准差，然后取百分位数作为阈值
        n_seg = mag.size // seg_n
        if n_seg <= 0:
            vals = np.array([np.nanstd(mag)], dtype=np.float64)
        else:
            vals = np.array([np.nanstd(mag[i * seg_n:(i + 1) * seg_n]) for i in range(n_seg)], dtype=np.float64)
        thresholds[dev] = float(np.nanpercentile(vals, percentile)) if vals.size else float("nan")
    return thresholds


# ---------------------------------------------------------------------------
# 辅助函数：计算单个窗口内某设备的运动占比
# 将窗口内的加速度数据按段切分，计算每段的标准差是否超过阈值，
# 返回超阈值段数的占比（仅作为元数据，不影响窗口保留决策）
# ---------------------------------------------------------------------------
def _motion_fraction_window(
    dev_raw: dict[str, np.ndarray],
    *,
    t0: float,
    t1: float,
    threshold: float,
    seg_sec: float,
) -> float:
    t = np.asarray(dev_raw["timestamp"], dtype=np.float64)
    lo = int(np.searchsorted(t, t0, side="left"))
    hi = int(np.searchsorted(t, t1, side="right"))
    if hi <= lo:
        return float("nan")
    fs = _infer_fs_ms(t[lo:hi])
    seg_n = max(1, int(round(seg_sec * fs))) if np.isfinite(fs) else 1000
    mag = np.sqrt(
        np.asarray(dev_raw["accel_x"][lo:hi], dtype=np.float64) ** 2
        + np.asarray(dev_raw["accel_y"][lo:hi], dtype=np.float64) ** 2
        + np.asarray(dev_raw["accel_z"][lo:hi], dtype=np.float64) ** 2
    )
    n_seg = mag.size // seg_n
    if n_seg <= 0:
        stds = np.array([np.nanstd(mag)], dtype=np.float64)
    else:
        stds = np.array([np.nanstd(mag[i * seg_n:(i + 1) * seg_n]) for i in range(n_seg)], dtype=np.float64)
    return float(np.nanmean(stds > threshold)) if np.isfinite(threshold) else float("nan")


# ---------------------------------------------------------------------------
# 辅助函数：生成数据集的 README.md 文档
# 包含配置参数、窗口纳入规则、字段说明和汇总统计信息
# ---------------------------------------------------------------------------
def _write_readme(out_dir: Path, dataset_name: str, cfg: dict[str, object], summary: pd.DataFrame) -> None:
    total = int(summary["n_kept"].sum()) if "n_kept" in summary else 0
    participants = int((summary["n_kept"] > 0).sum()) if "n_kept" in summary else 0
    n_devices = len(cfg.get("devices", DEVICES))
    text = f"""# {dataset_name}

This dataset was generated from `snowballlab/Multisite-PPG/raw_data` by aligning
raw timelines first and then cutting shared absolute-time windows.

## Core Parameters

```json
{json.dumps(cfg, indent=2, ensure_ascii=False)}
```

## Inclusion Rule

A window is kept only if:

1. all selected PPG devices have samples near the shared window start and end within
   `alignment_tolerance_sec`;
2. every device/channel has `ppg_valid_sample_ratio >= min_valid_sample_ratio`
   after target_fs Hz resampling;
3. ECG label QC passes.

PPG SQI, motion fraction, PPG peak quality, and PPG-derived HRV are metadata
only. They are not used to remove windows.

## Current Output

- kept windows: {total}
- participants with kept windows: {participants}
- primary label: `ecg_r_peak_times_rel_ms`
- model input: `ppg_resampled` with shape `(N, {n_devices}, 2, target_len)`
- missing-sample mask: `ppg_valid_mask_resampled`

## Important Fields

| Field | Meaning |
|---|---|
| `ppg_resampled` | 4-device, 2-channel PPG resampled to the shared target_fs grid |
| `ppg_valid_mask_resampled` | True where a resampled point is supported by nearby raw samples |
| `ppg_valid_sample_ratio` | PPG sample coverage used for inclusion |
| `ecg_r_peak_times_rel_ms` | primary label: ECG R-peak times relative to window start |
| `ecg_rr_intervals_ms` | ECG RR intervals from Pan-Tompkins R-peaks |
| `ecg_rmssd_ms`, `ecg_sdnn_ms` | ECG-derived HRV labels from corrected RR intervals |
| `ecg_label_qc_pass`, `ecg_label_qc_reason` | final ECG label trustworthiness flag/reason |
| `ecg_qrs_sqi` | simple raw-ECG QRS prominence/consistency SQI |
| `ppg_peak_times_rel_ms`, `ppg_ibi_ms`, `ppg_rmssd_ms` | PPG-derived metadata, not inclusion criteria |
| `ppg_quality_flag`, `ppg_quality_reason` | metadata-only PPG quality indicators |
| `motion_fraction` | metadata-only accelerometer motion fraction |

## Summary

See `{dataset_name}_summary.csv` for per-participant counts and quality means.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


# ===========================================================================
# ★★★ 核心函数：单个参与者的数据集生成流水线（最重点）
# ===========================================================================
# 对单个参与者执行完整的数据集生成流程：
#   1. 加载原始 PPG + ECG 数据
#   2. 生成候选窗口并按边界对齐和 ECG 质控筛选
#   3. 对通过初筛的窗口进行 PPG 重采样和 PPG 波峰检测
#   4. 按 PPG 采样覆盖率做最终筛选
#   5. 保存结果为 .npz 文件并输出汇总 CSV
# ---------------------------------------------------------------------------
def generate_participant(
    pid: str,
    *,
    raw_root: Path,
    out_dir: Path,
    dataset_name: str,
    devices: list[str],
    window_sec: int,
    stride_sec: int,
    target_fs: float,
    alignment_tolerance_sec: float,
    min_valid_sample_ratio: float,
    max_source_gap_ms: float,
    preprocess_mode: str,
    motion_seg_sec: float,
    motion_percentile: float,
    ecg_min_valid_ibi_ratio: float,
    min_ecg_valid_sample_ratio: float,
    min_hr_bpm: float,
    max_hr_bpm: float,
    max_windows: int | None,
) -> Path | None:
    # --- 第 1 步：加载原始数据并计算基本参数 ---
    print(f"[{pid}] loading raw data")
    ppg_raw = _load_ppg_raw(raw_root, pid, devices)    # 加载四设备 PPG
    ecg_raw = _load_ecg_raw(raw_root, pid)              # 加载 ECG
    target_len = int(round(window_sec * target_fs))     # 窗口在 target_fs 下的样本数
    sample_step_ms = 1000.0 / target_fs                 # 每个样本间隔（ms）
    window_ms = window_sec * 1000.0
    tolerance_ms = alignment_tolerance_sec * 1000.0

    # --- 第 2 步：生成候选窗口 ---
    t0_all, t1_all, overlap_info = _candidate_windows(
        ppg_raw,
        ecg_raw,
        devices=devices,
        window_sec=window_sec,
        stride_sec=stride_sec,
    )
    if max_windows is not None:  # 调试用：限制最大窗口数
        t0_all = t0_all[:max_windows]
        t1_all = t1_all[:max_windows]

    out_dir.mkdir(parents=True, exist_ok=True)
    # 无候选窗口时直接跳过
    if t0_all.size == 0:
        summary = {
            "participant": pid,
            "n_candidate_windows": 0,
            "n_boundary_aligned": 0,
            "n_ppg_sample_pass": 0,
            "n_ecg_label_pass": 0,
            "n_kept": 0,
            **overlap_info,
            "skip_reason": "no_raw_common_overlap_for_window",
        }
        pd.DataFrame([summary]).to_csv(out_dir / f"{dataset_name}_{pid}_summary.csv", index=False)
        print(f"[SKIP] {pid}: no raw common overlap")
        return None

    # --- 推断 ECG 采样率并计算运动阈值 ---
    ecg_fs = _ecg_fs_from_raw(ecg_raw)
    ecg_detector = "window_pantompkins1985"
    print(f"[{pid}] ECG fs={ecg_fs:.3f}, detector={ecg_detector}")
    motion_threshold = _motion_thresholds_from_raw(
        ppg_raw,
        percentile=motion_percentile,
        seg_sec=motion_seg_sec,
    )

    # --- 第 3 步：★ 边界对齐检查 + ECG 质控筛选（重点） ---
    boundary_keep = []           # 每个窗口是否通过边界对齐
    max_start_diff_ms = []       # 最大起始偏移
    max_end_diff_ms = []         # 最大结束偏移
    ppg_sample_keep_pre = []     # PPG 采样初筛结果
    ecg_keep_pre = []            # ECG 质控初筛结果
    labels: list[dict[str, object]] = []  # ECG 标签字典列表

    # 遍历每个候选窗口，检查边界对齐和 ECG 标签质控
    print(f"[{pid}] screening {t0_all.size} candidate windows")
    for wi, (t0, t1) in enumerate(zip(t0_all, t1_all)):
        start_offsets = []
        end_offsets = []
        boundary_ok = True
        # 检查每个设备在窗口起止处的偏移是否在容许范围内
        for dev in devices:
            so, eo, _n = _boundary_offsets_ms(ppg_raw[dev]["timestamp"], float(t0), float(t1))
            start_offsets.append(so)
            end_offsets.append(eo)
            if so > tolerance_ms or eo > tolerance_ms:
                boundary_ok = False
        boundary_keep.append(boundary_ok)
        max_start_diff_ms.append(float(np.nanmax(start_offsets)))
        max_end_diff_ms.append(float(np.nanmax(end_offsets)))
        if not boundary_ok:
            labels.append({})
            ppg_sample_keep_pre.append(False)
            ecg_keep_pre.append(False)
            continue
        # 边界对齐通过后，执行 ECG 标签生成与质控
        label = _ecg_label_for_window(
            ecg_raw,
            ecg_fs,
            t0=float(t0),
            t1=float(t1),
            target_fs=target_fs,
            min_valid_ibi_ratio=ecg_min_valid_ibi_ratio,
            min_hr_bpm=min_hr_bpm,
            max_hr_bpm=max_hr_bpm,
            min_ecg_valid_sample_ratio=min_ecg_valid_sample_ratio,
            detector_method=ecg_detector,
        )
        labels.append(label)
        ecg_keep_pre.append(bool(label["ecg_label_qc_pass"]))
        ppg_sample_keep_pre.append(True)  # PPG 采样率的精确检查在重采样时进行

    # ★ 合并边界对齐 + ECG 质控结果，筛选出初步保留的窗口索引
    boundary_keep_arr = np.asarray(boundary_keep, dtype=bool)
    ecg_keep_arr = np.asarray(ecg_keep_pre, dtype=bool)
    prelim_idx = np.where(boundary_keep_arr & ecg_keep_arr)[0]
    if prelim_idx.size == 0:
        summary = {
            "participant": pid,
            "n_candidate_windows": int(t0_all.size),
            "n_boundary_aligned": int(boundary_keep_arr.sum()),
            "n_ppg_sample_pass": 0,
            "n_ecg_label_pass": int(ecg_keep_arr.sum()),
            "n_kept": 0,
            **overlap_info,
            "skip_reason": "no_window_after_boundary_and_ecg_qc",
        }
        pd.DataFrame([summary]).to_csv(out_dir / f"{dataset_name}_{pid}_summary.csv", index=False)
        print(f"[SKIP] {pid}: no windows after boundary + ECG QC")
        return None

    # --- 第 4 步：★ 预分配输出数组（重点） ---
    # ppg 形状: (n_windows, n_devices, n_channels, target_len)
    n_pre = int(prelim_idx.size)
    ppg = np.zeros((n_pre, len(devices), len(CHANNELS), target_len), dtype=np.float32)
    ppg_valid_mask = np.zeros_like(ppg, dtype=bool)
    ppg_valid_sample_ratio = np.zeros((n_pre, len(devices), len(CHANNELS)), dtype=np.float32)
    ppg_valid_ibi_ratio = np.zeros_like(ppg_valid_sample_ratio)
    ppg_sqi_arr = np.zeros_like(ppg_valid_sample_ratio)
    ppg_ibi_correction_ratio = np.full_like(ppg_valid_sample_ratio, np.nan)
    ppg_rmssd = np.full_like(ppg_valid_sample_ratio, np.nan)
    ppg_sdnn = np.full_like(ppg_valid_sample_ratio, np.nan)
    ppg_quality_flag = np.zeros((n_pre, len(devices), len(CHANNELS)), dtype=bool)
    ppg_quality_reason = np.empty((n_pre, len(devices), len(CHANNELS)), dtype=object)
    motion_fraction = np.zeros((n_pre, len(devices)), dtype=np.float32)

    # 变长数据用 object 数组存储（PPG 波峰、IBI 等）
    ppg_peak_times = _empty_object_array((n_pre, len(devices), len(CHANNELS)))
    ppg_peak_indices = _empty_object_array((n_pre, len(devices), len(CHANNELS)))
    ppg_peak_amplitudes = _empty_object_array((n_pre, len(devices), len(CHANNELS)))
    ppg_ibi = _empty_object_array((n_pre, len(devices), len(CHANNELS)))
    ppg_ibi_corrected = _empty_object_array((n_pre, len(devices), len(CHANNELS)))

    # --- 第 5 步：★★ PPG 重采样、波峰检测和质控（重点） ---
    final_keep = np.ones(n_pre, dtype=bool)
    print(f"[{pid}] resampling PPG for {n_pre} ECG-valid windows")
    for out_i, wi in enumerate(prelim_idx):
        t0 = float(t0_all[wi])
        grid_ms = t0 + np.arange(target_len, dtype=np.float64) * sample_step_ms  # target_fs 时间网格
        for di, dev in enumerate(devices):
            # 计算该设备在此窗口的运动占比（元数据）
            motion_fraction[out_i, di] = _motion_fraction_window(
                ppg_raw[dev],
                t0=t0,
                t1=float(t1_all[wi]),
                threshold=motion_threshold[dev],
                seg_sec=motion_seg_sec,
            )
            # 截取窗口附近的原始 PPG 数据（带边距容差）
            dev_t = np.asarray(ppg_raw[dev]["timestamp"], dtype=np.float64)
            seg_lo = int(np.searchsorted(dev_t, t0 - max_source_gap_ms, side="left"))
            seg_hi = int(np.searchsorted(dev_t, float(t1_all[wi]) + max_source_gap_ms, side="right"))
            seg_t = dev_t[seg_lo:seg_hi]
            for ci, ch in enumerate(CHANNELS):
                seg_x = np.asarray(ppg_raw[dev][ch][seg_lo:seg_hi], dtype=np.float64)
                # ★ 将原始 PPG 重采样到 target_fs 统一网格
                resampled, valid_mask, valid_ratio = _resample_to_grid(
                    seg_x,
                    seg_t,
                    grid_ms,
                    max_gap_ms=max_source_gap_ms,
                )
                ppg[out_i, di, ci] = resampled
                ppg_valid_mask[out_i, di, ci] = valid_mask
                ppg_valid_sample_ratio[out_i, di, ci] = valid_ratio
                # ★ PPG 采样覆盖率检查：不达标则标记为丢弃（阈值 0.9）
                if valid_ratio < min_valid_sample_ratio:
                    final_keep[out_i] = False
                # PPG 波峰检测与 HRV 计算（仅作为元数据）
                peak_signal = _preprocess_for_peaks(resampled, target_fs, preprocess_mode)
                peak_info = _ppg_peaks_and_qc(resampled, peak_signal, fs=target_fs, valid_sample_ratio=valid_ratio)
                ppg_peak_times[out_i, di, ci] = peak_info["ppg_peak_times_rel_ms"]
                ppg_peak_indices[out_i, di, ci] = peak_info["ppg_peak_indices_grid"]
                ppg_peak_amplitudes[out_i, di, ci] = peak_info["ppg_peak_amplitudes_raw"]
                ppg_ibi[out_i, di, ci] = peak_info["ppg_ibi_ms"]
                ppg_ibi_corrected[out_i, di, ci] = peak_info["ppg_ibi_corrected_ms"]
                ppg_ibi_correction_ratio[out_i, di, ci] = peak_info["ppg_ibi_correction_ratio"]
                ppg_rmssd[out_i, di, ci] = peak_info["ppg_rmssd_ms"]
                ppg_sdnn[out_i, di, ci] = peak_info["ppg_sdnn_ms"]
                ppg_valid_ibi_ratio[out_i, di, ci] = peak_info["ppg_valid_ibi_ratio"]
                ppg_sqi_arr[out_i, di, ci] = peak_info["ppg_sqi"]
                # PPG 单通道质控（元数据，不影响窗口保留）
                qflag, qreason = _ppg_channel_qc(
                    valid_sample_ratio=valid_ratio,
                    valid_ibi_ratio=peak_info["ppg_valid_ibi_ratio"],
                    correction_ratio=peak_info["ppg_ibi_correction_ratio"],
                    rmssd_ms=peak_info["ppg_rmssd_ms"],
                    sqi=peak_info["ppg_sqi"],
                    motion_fraction=float(motion_fraction[out_i, di]),
                    min_valid_sample_ratio=min_valid_sample_ratio,
                    min_valid_ibi_ratio=ecg_min_valid_ibi_ratio,
                    min_sqi=DEFAULT_MIN_PPG_SQI,
                    max_motion_fraction=DEFAULT_MAX_MOTION_FRACTION,
                )
                ppg_quality_flag[out_i, di, ci] = qflag
                ppg_quality_reason[out_i, di, ci] = qreason

    # --- 第 6 步：最终筛选 ---
    # 仅保留所有设备/通道的 PPG 采样覆盖率均达标的窗口
    keep_idx = np.where(final_keep)[0]
    if keep_idx.size == 0:
        summary = {
            "participant": pid,
            "n_candidate_windows": int(t0_all.size),
            "n_boundary_aligned": int(boundary_keep_arr.sum()),
            "n_ppg_sample_pass": 0,
            "n_ecg_label_pass": int(ecg_keep_arr.sum()),
            "n_kept": 0,
            **overlap_info,
            "skip_reason": "no_window_after_ppg_sample_coverage",
        }
        pd.DataFrame([summary]).to_csv(out_dir / f"{dataset_name}_{pid}_summary.csv", index=False)
        print(f"[SKIP] {pid}: no windows after PPG sample coverage")
        return None

    # --- 第 7 步：★ 保存结果为压缩 .npz 文件（重点） ---
    kept_wi = prelim_idx[keep_idx]
    kept_labels = [labels[int(wi)] for wi in kept_wi]
    out_path = out_dir / f"{dataset_name}_{pid}.npz"
    # 将配置信息嵌入到输出文件中
    cfg = {
        "dataset_name": dataset_name,
        "participant": pid,
        "source": "snowballlab/Multisite-PPG/raw_data local mirror",
        "devices": devices,
        "channels": CHANNELS,
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "target_fs": target_fs,
        "target_len": target_len,
        "alignment_tolerance_sec": alignment_tolerance_sec,
        "min_valid_sample_ratio": min_valid_sample_ratio,
        "max_source_gap_ms": max_source_gap_ms,
        "preprocess_mode_for_peak_metadata": preprocess_mode,
        "motion_seg_sec": motion_seg_sec,
        "motion_percentile": motion_percentile,
        "ecg_detector": "window_neurokit2_pantompkins1985",
        "ecg_fs_inferred_hz": ecg_fs,
        "ecg_min_valid_ibi_ratio": ecg_min_valid_ibi_ratio,
        "min_ecg_valid_sample_ratio": min_ecg_valid_sample_ratio,
        "primary_label": "ecg_r_peak_times_rel_ms",
        "inclusion_rule": "raw_boundary_aligned && ecg_label_qc_pass && ppg_valid_sample_ratio>=min_valid_sample_ratio for all device/channel",
        "ppg_quality_is_metadata_only": True,
    }
    print(f"[{pid}] saving {out_path} kept={keep_idx.size}/{t0_all.size}")
    np.savez_compressed(
        out_path,
        config_json=np.array(json.dumps(cfg, ensure_ascii=False)),
        participant=np.array(pid),
        devices=np.array(devices),
        channels=np.array(CHANNELS),
        t0_ms=t0_all[kept_wi],
        t1_ms=t1_all[kept_wi],
        max_start_diff_ms=np.asarray(max_start_diff_ms, dtype=np.float32)[kept_wi],
        max_end_diff_ms=np.asarray(max_end_diff_ms, dtype=np.float32)[kept_wi],
        ppg_resampled=ppg[keep_idx],
        ppg_valid_mask_resampled=ppg_valid_mask[keep_idx],
        ppg_valid_sample_ratio=ppg_valid_sample_ratio[keep_idx],
        ppg_valid_ibi_ratio=ppg_valid_ibi_ratio[keep_idx],
        ppg_sqi=ppg_sqi_arr[keep_idx],
        ppg_peak_times_rel_ms=ppg_peak_times[keep_idx],
        ppg_peak_indices_grid=ppg_peak_indices[keep_idx],
        ppg_peak_amplitudes_raw=ppg_peak_amplitudes[keep_idx],
        ppg_ibi_ms=ppg_ibi[keep_idx],
        ppg_ibi_corrected_ms=ppg_ibi_corrected[keep_idx],
        ppg_ibi_correction_ratio=ppg_ibi_correction_ratio[keep_idx],
        ppg_rmssd_ms=ppg_rmssd[keep_idx],
        ppg_sdnn_ms=ppg_sdnn[keep_idx],
        ppg_quality_flag=ppg_quality_flag[keep_idx],
        ppg_quality_reason=ppg_quality_reason[keep_idx],
        motion_fraction=motion_fraction[keep_idx],
        motion_threshold=np.array([motion_threshold[d] for d in devices], dtype=np.float32),
        ecg_r_peak_times_rel_ms=np.array([x["ecg_r_peak_times_rel_ms"] for x in kept_labels], dtype=object),
        ecg_r_peak_indices_grid=np.array([x["ecg_r_peak_indices_grid"] for x in kept_labels], dtype=object),
        ecg_r_peak_amplitudes_raw=np.array([x["ecg_r_peak_amplitudes_raw"] for x in kept_labels], dtype=object),
        ecg_rr_intervals_ms=np.array([x["ecg_rr_intervals_ms"] for x in kept_labels], dtype=object),
        ecg_rr_intervals_corrected_ms=np.array([x["ecg_rr_intervals_corrected_ms"] for x in kept_labels], dtype=object),
        ecg_rmssd_ms=np.array([x["ecg_rmssd_ms"] for x in kept_labels], dtype=np.float32),
        ecg_sdnn_ms=np.array([x["ecg_sdnn_ms"] for x in kept_labels], dtype=np.float32),
        ecg_valid_sample_ratio=np.array([x["ecg_valid_sample_ratio"] for x in kept_labels], dtype=np.float32),
        ecg_valid_ibi_ratio=np.array([x["ecg_valid_ibi_ratio"] for x in kept_labels], dtype=np.float32),
        ecg_ibi_correction_ratio=np.array([x["ecg_ibi_correction_ratio"] for x in kept_labels], dtype=np.float32),
        ecg_qc_pass=np.array([x["ecg_qc_pass"] for x in kept_labels], dtype=bool),
        ecg_qc_reason=np.array([x["ecg_qc_reason"] for x in kept_labels], dtype=object),
        ecg_label_qc_pass=np.array([x["ecg_label_qc_pass"] for x in kept_labels], dtype=bool),
        ecg_label_qc_reason=np.array([x["ecg_label_qc_reason"] for x in kept_labels], dtype=object),
        ecg_qrs_sqi=np.array([x["ecg_qrs_sqi"] for x in kept_labels], dtype=np.float32),
        ecg_peak_count=np.array([x["ecg_peak_count"] for x in kept_labels], dtype=np.int32),
        ecg_hr_bpm=np.array([x["ecg_hr_bpm"] for x in kept_labels], dtype=np.float32),
        ecg_rr_cv=np.array([x["ecg_rr_cv"] for x in kept_labels], dtype=np.float32),
        ecg_rpeak_detector_agreement=np.full(keep_idx.size, np.nan, dtype=np.float32),
    )

    # --- 第 8 步：输出汇总统计 ---
    summary = {
        "participant": pid,
        "n_candidate_windows": int(t0_all.size),
        "n_boundary_aligned": int(boundary_keep_arr.sum()),
        "n_ppg_sample_pass": int(final_keep.sum()),
        "n_ecg_label_pass": int(ecg_keep_arr.sum()),
        "n_kept": int(keep_idx.size),
        "kept_pct_of_candidate": float(100.0 * keep_idx.size / t0_all.size),
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "target_fs": target_fs,
        "target_len": target_len,
        "alignment_tolerance_sec": alignment_tolerance_sec,
        "max_source_gap_ms": max_source_gap_ms,
        "min_valid_sample_ratio": min_valid_sample_ratio,
        "mean_max_start_diff_ms": float(np.nanmean(np.asarray(max_start_diff_ms)[kept_wi])),
        "mean_max_end_diff_ms": float(np.nanmean(np.asarray(max_end_diff_ms)[kept_wi])),
        "mean_ppg_valid_sample_ratio": float(np.nanmean(ppg_valid_sample_ratio[keep_idx])),
        "mean_ppg_sqi": float(np.nanmean(ppg_sqi_arr[keep_idx])),
        "mean_ppg_ibi_correction_ratio": float(np.nanmean(ppg_ibi_correction_ratio[keep_idx])),
        "mean_motion_fraction": float(np.nanmean(motion_fraction[keep_idx])),
        "mean_ecg_valid_ibi_ratio": float(np.nanmean([x["ecg_valid_ibi_ratio"] for x in kept_labels])),
        "mean_ecg_ibi_correction_ratio": float(np.nanmean([x["ecg_ibi_correction_ratio"] for x in kept_labels])),
        "mean_ecg_qrs_sqi": float(np.nanmean([x["ecg_qrs_sqi"] for x in kept_labels])),
        "mean_ecg_hr_bpm": float(np.nanmean([x["ecg_hr_bpm"] for x in kept_labels])),
        **overlap_info,
        "skip_reason": "",
    }
    pd.DataFrame([summary]).to_csv(out_dir / f"{dataset_name}_{pid}_summary.csv", index=False)
    return out_path


# ===========================================================================
# ★ 主入口函数（重点）
# ===========================================================================
# 解析命令行参数 -> 遍历每个参与者调用 generate_participant -> 汇总结果
# ---------------------------------------------------------------------------
def main() -> None:
    # --- 命令行参数定义 ---
    ap = argparse.ArgumentParser(description="Generate raw-aligned 4-device ECG-label dataset.")
    ap.add_argument("--participants", default=None)          # 指定参与者（逗号分隔）
    ap.add_argument("--exclude", default="P2,P14,P16,P17")   # 排除的参与者
    ap.add_argument(
        "--devices",
        default=",".join(DEVICES),
        help="Comma-separated PPG devices to include. Default: all devices.",
    )
    ap.add_argument("--raw-root", default=str(RAW_ROOT))                  # 原始数据根目录
    ap.add_argument("--dataset-name", required=True)                       # 数据集名称（必填）
    ap.add_argument("--out-dir", default=None)                              # 输出目录
    ap.add_argument("--window-sec", type=int, default=300)                  # 窗口时长（秒）
    ap.add_argument("--stride-sec", type=int, default=300)                  # 窗口步长（秒）
    ap.add_argument("--target-fs", type=float, default=50.0)                # 目标采样率 (Hz)
    ap.add_argument("--alignment-tolerance-sec", type=float, default=2.0)   # 边界对齐容差（秒）
    ap.add_argument("--min-valid-sample-ratio", type=float, default=0.50)   # PPG 最低有效采样比
    ap.add_argument("--max-source-gap-ms", type=float, default=500.0)       # 重采样最大源间隔 (ms)
    ap.add_argument("--preprocess-mode", choices=("bandpass", "raw"), default="bandpass")  # 预处理模式
    ap.add_argument("--motion-seg-sec", type=float, default=10.0)           # 运动检测段长（秒）
    ap.add_argument("--motion-percentile", type=float, default=75.0)        # 运动阈值百分位数
    ap.add_argument("--ecg-min-valid-ibi-ratio", type=float, default=1.00)  # ECG 最低有效 IBI 比
    ap.add_argument("--min-ecg-valid-sample-ratio", type=float, default=0.95)  # ECG 最低有效采样比
    ap.add_argument("--min-hr-bpm", type=float, default=30.0)               # 最低心率 (BPM)
    ap.add_argument("--max-hr-bpm", type=float, default=200.0)              # 最高心率 (BPM)
    ap.add_argument("--max-windows", type=int, default=None)                # 调试用：最大窗口数
    args = ap.parse_args()

    # --- 解析参数并准备输入 ---
    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (config.HEURISTIC_RESULT_ROOT / args.dataset_name).resolve()
    participants = _participant_ids(raw_root, args.participants)
    excluded = {normalize_participant_id(p) for p in args.exclude.split(",") if p.strip()}
    participants = [p for p in participants if p not in excluded]  # 过滤排除的参与者
    # 设备名称校验
    device_lookup = {d.lower(): d for d in DEVICES}
    devices = [device_lookup.get(d.strip().lower(), d.strip()) for d in args.devices.split(",") if d.strip()]
    unknown_devices = sorted(set(devices) - set(DEVICES))
    if unknown_devices:
        raise SystemExit(f"Unknown devices: {unknown_devices}. Valid devices: {DEVICES}")

    # --- 保存全局配置并开始遍历参与者 ---
    cfg = {
        "dataset_name": args.dataset_name,
        "raw_root": str(raw_root),
        "participants": participants,
        "excluded": sorted(excluded),
        "devices": devices,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "target_fs": args.target_fs,
        "target_len": int(round(args.window_sec * args.target_fs)),
        "alignment_tolerance_sec": args.alignment_tolerance_sec,
        "min_valid_sample_ratio": args.min_valid_sample_ratio,
        "max_source_gap_ms": args.max_source_gap_ms,
        "preprocess_mode": args.preprocess_mode,
        "motion_seg_sec": args.motion_seg_sec,
        "motion_percentile": args.motion_percentile,
        "ecg_detector": "window_neurokit2_pantompkins1985",
        "ecg_min_valid_ibi_ratio": args.ecg_min_valid_ibi_ratio,
        "min_ecg_valid_sample_ratio": args.min_ecg_valid_sample_ratio,
        "primary_label": "ecg_r_peak_times_rel_ms",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- ★ 遍历每个参与者执行数据集生成 ---
    print(f"[rawaligned] participants={participants}")
    print(f"[rawaligned] devices={devices}")
    print(f"[rawaligned] out_dir={out_dir}")
    paths = []
    for pid in participants:
        try:
            path = generate_participant(
                pid,
                raw_root=raw_root,
                out_dir=out_dir,
                dataset_name=args.dataset_name,
                devices=devices,
                window_sec=args.window_sec,
                stride_sec=args.stride_sec,
                target_fs=args.target_fs,
                alignment_tolerance_sec=args.alignment_tolerance_sec,
                min_valid_sample_ratio=args.min_valid_sample_ratio,
                max_source_gap_ms=args.max_source_gap_ms,
                preprocess_mode=args.preprocess_mode,
                motion_seg_sec=args.motion_seg_sec,
                motion_percentile=args.motion_percentile,
                ecg_min_valid_ibi_ratio=args.ecg_min_valid_ibi_ratio,
                min_ecg_valid_sample_ratio=args.min_ecg_valid_sample_ratio,
                min_hr_bpm=args.min_hr_bpm,
                max_hr_bpm=args.max_hr_bpm,
                max_windows=args.max_windows,
            )
            if path is not None:
                paths.append(path)
        except Exception as exc:
            # 异常处理：记录错误信息并继续处理下一个参与者
            summary = {
                "participant": pid,
                "n_candidate_windows": 0,
                "n_boundary_aligned": 0,
                "n_ppg_sample_pass": 0,
                "n_ecg_label_pass": 0,
                "n_kept": 0,
                "skip_reason": f"exception:{type(exc).__name__}:{exc}",
            }
            pd.DataFrame([summary]).to_csv(out_dir / f"{args.dataset_name}_{pid}_summary.csv", index=False)
            print(f"[ERROR] {pid}: {exc!r}")

    # --- 汇总所有参与者的统计结果并生成 README ---
    summaries = []
    for csv_path in sorted(out_dir.glob(f"{args.dataset_name}_P*_summary.csv")):
        summaries.append(pd.read_csv(csv_path))
    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        summary_path = out_dir / f"{args.dataset_name}_summary.csv"
        combined.to_csv(summary_path, index=False)
        _write_readme(out_dir, args.dataset_name, cfg, combined)
        print(f"[SAVED] {summary_path}")
        print(f"[SAVED] {out_dir / 'README.md'}")
    print(f"[DONE] participant files={len(paths)}")


if __name__ == "__main__":
    main()
