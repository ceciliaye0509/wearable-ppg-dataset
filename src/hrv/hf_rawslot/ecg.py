from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import neurokit2 as nk
from hrv.common import rr as hrv
DEFAULT_MAX_IBI_CORRECTION_RATIO=.2
DEFAULT_MAX_RMSSD_MS=200.
MAX_PEAKS_PER_WINDOW=1200


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
        mean_corrected = float(np.nanmean(corrected))
        hr_bpm = float(60000.0 / mean_corrected) if mean_corrected > 0 else float("nan")  # 心率 BPM
        rr_cv = float(np.nanstd(corrected) / mean_corrected) if mean_corrected > 0 else float("nan")  # 变异系数
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
    #   - ECG 有效样本覆盖率 >= 阈值（1.0）
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
        and peak_count <= MAX_PEAKS_PER_WINDOW
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
    if peak_count > MAX_PEAKS_PER_WINDOW:
        reasons.append("too_many_ecg_peaks")            # R-peak 数量超过生理上限
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
