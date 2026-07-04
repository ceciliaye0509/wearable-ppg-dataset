"""
完整消融实验矩阵 v2 — 包含所有改进步骤的组合测试。

与 v1 相比的变化：
  1. 时间对齐：±10s 容差的跨 4 设备窗口匹配
  2. ECG 对称处理：同时测试插值方案（scheme A）和原始 RR 方案（scheme C）
  3. 运动伪差质控：10s 滑动窗口，逐设备阈值校准
  4. 25Hz 降采样模式，用于 Step 8 评估
  5. 波峰检测缓存：约 8 倍加速
  6. 多进程：4 个设备并行处理

使用方法：
    python full_step_matrix_v2.py --participant P7
"""
import argparse
import sys
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, decimate
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# 包路径设置与导入
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from algorithms import hrv
from preprocess import preprocess_ppg
from io_utils import merged_windows_npz, normalize_participant_id
import config

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------
DEVICES = ["Earring", "Ring", "Necklace", "Watch"]  # 四个 PPG 设备
CHANNELS = ["ppg_green", "ppg_ir"]                   # 两个 PPG 通道
ECG_FS = 130.0  # ECG 采样率（由 r_peak_samples / rr_intervals_ms 推导）

# 所有 2^3 = 8 种 (插值, IBI校正, 阈值门限) 组合
STEP_CONFIGS = [
    ("S0_baseline",         False, False, False),  # 纯基线，无任何改进
    ("S1_interp_only",      True,  False, False),  # 仅亚采样插值
    ("S2_ibi_only",         False, True,  False),  # 仅 IBI 伪差校正
    ("S7_thresh_only",      False, False, True),   # 仅有效比例/CV 门限
    ("S1+S2_interp_ibi",    True,  True,  False),  # 插值 + IBI 校正
    ("S1+S7_interp_thresh", True,  False, True),   # 插值 + 门限
    ("S2+S7_ibi_thresh",    False, True,  True),   # IBI 校正 + 门限
    ("S1+S2+S7_full",       True,  True,  True),   # 全部启用（完整流水线）
]


# ===================================================================
# ECG：真实 R 波峰定位与 RMSSD 计算
# ===================================================================
def find_true_r_peaks(ecg_signal, marked_peaks):
    """两遍 R 波峰校正，补偿 Pan-Tompkins 积分延迟。

    Pan-Tompkins 标记的位置通常在实际 R 波峰之后 ~11 个采样点（130Hz 下约 85ms）。
    延迟因 QRS 波形、心率和信噪比而略有变化。

    第一遍 — 宽搜索 [p-25, p+5]，估计每窗口的中位偏移量（对异常值鲁棒）。
    第二遍 — 窄搜索 [期望位置 ± 4]，精确定位每个 R 波峰，避免远处噪声极值引入抖动。

    这种两遍方法将系统性 RMSSD 误差从 ~5-7 ms MAE（单遍 [p-20, p-2]）
    降低到 ~1.2-1.7 ms MAE，提升 3-5 倍。
    """
    n = len(marked_peaks)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    # --- 第一遍：宽搜索估计全局中位偏移 ---
    offsets = np.zeros(n, dtype=np.int64)
    for i in range(n):
        mp = int(marked_peaks[i])
        lo = max(0, mp - 25)
        hi = min(len(ecg_signal), mp + 5)
        if lo >= hi:
            continue
        offsets[i] = (lo + int(np.argmax(ecg_signal[lo:hi]))) - mp

    median_offset = int(np.median(offsets))

    # --- 第二遍：基于期望位置的窄搜索精确定位 ---
    true_peaks = np.empty(n, dtype=np.int64)
    for i in range(n):
        mp = int(marked_peaks[i])
        expected = mp + median_offset
        lo = max(0, expected - 4)
        hi = min(len(ecg_signal), expected + 5)
        if lo >= hi:
            true_peaks[i] = mp
            continue
        true_peaks[i] = lo + int(np.argmax(ecg_signal[lo:hi]))

    return true_peaks


def ecg_rmssd(rr_ms, ecg_signal, r_peak_samples, n_rr_count,
              *, do_interp, do_ibi_correct, do_threshold):
    """计算 ECG RMSSD，支持可配置的处理步骤。

    do_interp=True（方案 A）：找到真实 R 波峰，应用亚采样插值，从插值位置重新计算 RR。
    do_interp=False（方案 C）：直接使用存储的 rr_intervals_ms。

    质控门限（与 PPG 侧对称）：
      - IBI 有效比例 < 80% → NaN（当 do_threshold=True）
      - IBI CV > IBI_CV_MAX → NaN（当 do_threshold=True）
      - RMSSD > RMSSD_MAX_MS → NaN（始终检查，校正后）
    """
    # 选择 RR 间期来源：插值或原始
    if do_interp and ecg_signal is not None and r_peak_samples is not None:
        # 从 ECG 信号中精确定位 R 波峰
        rp = r_peak_samples[:n_rr_count + 1].astype(np.int64)
        true_rp = find_true_r_peaks(ecg_signal, rp)
        # 亚采样插值 → 更精确的 RR 间期
        peaks_f = hrv._refine_peaks_parabolic(ecg_signal, true_rp)
        rr = np.diff(peaks_f) / ECG_FS * 1000.0
    else:
        # 直接使用存储的 RR 间期
        rr = np.asarray(rr_ms[:n_rr_count], dtype=np.float64)

    # --- Step 7：有效比例门限 ---
    valid_mask = (rr >= hrv.IBI_MIN_MS) & (rr <= hrv.IBI_MAX_MS)
    if do_threshold:
        if rr.size > 0 and valid_mask.sum() / len(rr) < 0.80:
            return float("nan")

    # 只保留生理范围内的 NN 间期
    nn = rr[valid_mask]
    if nn.size < 3:
        return float("nan")

    # --- IBI 变异系数门限（与 PPG 质控对称） ---
    if do_threshold:
        mean_nn = float(np.mean(nn))
        if mean_nn > 0:
            ibi_cv = float(np.std(nn, ddof=1) / mean_nn)
            if ibi_cv > hrv.IBI_CV_MAX:
                return float("nan")

    # --- Step 2：IBI 伪差校正 ---
    if do_ibi_correct:
        nn = hrv._correct_ibi_artifacts(nn)

    # 计算 RMSSD
    rmssd = float(np.sqrt(np.mean(np.diff(nn) ** 2)))

    # --- RMSSD 上限门限（与 PPG 质控对称） ---
    if rmssd > hrv.RMSSD_MAX_MS:
        return float("nan")

    return rmssd


# ===================================================================
# PPG RMSSD：使用缓存的 IBI 计算（避免重复检测波峰，8 倍加速）
# ===================================================================
def ppg_rmssd_from_cache(ibi_int, ibi_float, *,
                          do_interp, do_ibi_correct, do_threshold):
    """从预计算的 IBI 数组计算 PPG RMSSD（无需重新检测波峰）。

    质控门限（与 ECG 侧对称）：
      - IBI 有效比例 < 80% → NaN（当 do_threshold=True）
      - IBI CV > IBI_CV_MAX → NaN（当 do_threshold=True）
      - RMSSD > RMSSD_MAX_MS → NaN（始终检查，校正后）
    """
    # 根据是否插值选择 IBI 来源
    ibi = ibi_float if do_interp else ibi_int

    # --- 有效比例门限 ---
    if do_threshold:
        n_valid = int(((ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)).sum())
        if len(ibi) > 0 and n_valid / len(ibi) < 0.80:
            return float("nan")

    # 生理范围门限过滤
    nn = ibi[(ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)]
    if nn.size < 3:
        return float("nan")

    # --- IBI 变异系数门限（与 ECG 质控对称） ---
    if do_threshold:
        mean_nn = float(np.mean(nn))
        if mean_nn > 0:
            ibi_cv = float(np.std(nn, ddof=1) / mean_nn)
            if ibi_cv > hrv.IBI_CV_MAX:
                return float("nan")

    # --- IBI 伪差校正 ---
    if do_ibi_correct:
        nn = hrv._correct_ibi_artifacts(nn)

    rmssd = float(np.sqrt(np.mean(np.diff(nn) ** 2)))

    # --- RMSSD 上限门限 ---
    if rmssd > hrv.RMSSD_MAX_MS:
        return float("nan")

    return rmssd


# ===================================================================
# 运动质控
# ===================================================================
def compute_motion_stats(accel_x, accel_y, accel_z, fs, seg_sec=10):
    """计算单个窗口的逐片段运动标准差。

    将加速度合成幅值按 seg_sec 切段，返回每段的标准差数组。
    """
    # 三轴加速度合成幅值
    mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    seg_samples = int(seg_sec * fs)
    n_segs = len(mag) // seg_samples
    if n_segs == 0:
        return np.array([0.0])
    # 计算每段标准差
    stds = np.array([
        np.std(mag[i * seg_samples:(i + 1) * seg_samples])
        for i in range(n_segs)
    ])
    return stds


def calibrate_motion_thresholds(all_stds_by_device, percentile=75):
    """从标准差分布计算逐设备运动阈值（使用指定百分位数）。"""
    thresholds = {}
    for dev, stds in all_stds_by_device.items():
        if len(stds) == 0:
            thresholds[dev] = 0.5  # 回退默认值
            continue
        thresholds[dev] = float(np.percentile(stds, percentile))
    return thresholds


def motion_fraction(seg_stds, threshold):
    """计算超过运动阈值的 10s 片段占比。"""
    if len(seg_stds) == 0:
        return 0.0
    return float(np.mean(seg_stds > threshold))


# ===================================================================
# 降采样到 25 Hz
# ===================================================================
def downsample_to_25hz(signal_100hz):
    """抗混叠滤波 + 从 100Hz 降采样到 25Hz。"""
    sig = np.asarray(signal_100hz, dtype=np.float64)
    sig = sig[~np.isnan(sig)]
    if len(sig) < 20:
        return sig
    # scipy 的 decimate 内部自动应用抗混叠滤波
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return decimate(sig, 4, ftype='fir', zero_phase=True)


# ===================================================================
# 时间对齐：跨 4 设备窗口匹配
# ===================================================================
def align_windows(t0_dict, tol_ms=10000):
    """找到所有 4 个设备在容差范围内共有的窗口索引。

    以第一个可用设备为参考，对每个参考窗口在其他设备中找最近的匹配。
    返回 dict: {设备名: [对齐后的窗口索引列表]}
    """
    devs = [d for d in DEVICES if d in t0_dict]
    if len(devs) < 4:
        return {d: [] for d in devs}

    ref_dev = devs[0]  # 参考设备
    ref_t0 = t0_dict[ref_dev]
    aligned = {d: [] for d in devs}

    for ref_idx, t in enumerate(ref_t0):
        match_indices = {ref_dev: ref_idx}
        all_found = True
        # 在其他设备中寻找时间最接近的窗口
        for dev in devs[1:]:
            diffs = np.abs(t0_dict[dev] - t)
            best_idx = int(np.argmin(diffs))
            if diffs[best_idx] <= tol_ms:
                match_indices[dev] = best_idx
            else:
                all_found = False
                break
        # 只有所有设备都找到匹配时才保留
        if all_found:
            for dev, idx in match_indices.items():
                aligned[dev].append(idx)

    return aligned


# ===================================================================
# 单设备处理（可并行调用）
# ===================================================================
def process_device(args):
    """处理单个设备：所有通道 × 预处理方式 × 消融步骤 × ECG 插值选项。"""
    dev, pid, root, aligned_indices, motion_threshold = args

    raw_npz = merged_windows_npz(root, pid, dev)
    if not raw_npz.is_file():
        return []  # 文件不存在，跳过

    # --- 加载数据 ---
    with np.load(raw_npz, allow_pickle=True) as z:
        ppg_raw = {ch: np.asarray(z[ch]) for ch in CHANNELS if ch in z.files}
        fs = float(z["ppg_fs"])
        rr_all = np.asarray(z["rr_intervals_ms"])        # ECG RR 间期
        n_rr = np.asarray(z["n_rr"])                      # 每窗口有效 RR 数
        ecg_signal = np.asarray(z["ecg"])                  # ECG 原始波形
        r_peak_samples = np.asarray(z["r_peak_samples"])   # ECG R 波峰采样索引
        accel_x = np.asarray(z["accel_x"])                 # 加速度计数据
        accel_y = np.asarray(z["accel_y"])
        accel_z = np.asarray(z["accel_z"])

    n_windows = rr_all.shape[0]
    aligned_set = set(aligned_indices.get(dev, []))

    # --- 预计算带通滤波和 25Hz 降采样版本 ---
    ppg_bp = {}
    ppg_25hz = {}
    for ch in CHANNELS:
        if ch in ppg_raw:
            ppg_bp[ch] = preprocess_ppg(ppg_raw[ch], fs)
            # 对带通信号降采样到 25Hz
            ppg_25hz[ch] = np.array([
                downsample_to_25hz(ppg_bp[ch][i]) for i in range(n_windows)
            ], dtype=object)

    # --- 预计算每个窗口的运动统计 ---
    motion_stds = []
    motion_fracs = []
    for i in range(n_windows):
        seg_stds = compute_motion_stats(accel_x[i], accel_y[i], accel_z[i], fs)
        motion_stds.append(seg_stds)
        motion_fracs.append(motion_fraction(seg_stds, motion_threshold))

    print(f"  {dev}: {n_windows} 个窗口, {len(aligned_set)} 个已对齐, "
          f"运动阈值={motion_threshold:.3f}")

    rows = []
    preprocess_modes = ["bandpass", "raw", "25hz"]

    for ch in CHANNELS:
        if ch not in ppg_raw:
            continue

        for prep in preprocess_modes:
            # --- 选择信号源和对应采样率 ---
            if prep == "bandpass":
                ppg_data = ppg_bp[ch]
                ppg_fs = fs
            elif prep == "raw":
                ppg_data = ppg_raw[ch]
                ppg_fs = fs
            else:  # 25hz
                ppg_data = ppg_25hz[ch]
                ppg_fs = 25.0

            # --- 波峰检测缓存（每窗口只做一次，所有步骤共享） ---
            cached_peaks = []       # 整数波峰索引
            cached_peaks_f = []     # 浮点波峰位置（亚采样精度）
            cached_ibi_int = []     # 整数波峰计算的 IBI
            cached_ibi_float = []   # 浮点波峰计算的 IBI

            for i in range(n_windows):
                sig = np.asarray(ppg_data[i], dtype=np.float64)
                sig = sig[~np.isnan(sig)]
                peaks = hrv.detect_ppg_peaks(sig, ppg_fs)

                if peaks.size >= 3:
                    peaks_f = hrv._refine_peaks_parabolic(sig, peaks)
                    ibi_int = np.diff(peaks) / ppg_fs * 1000.0
                    ibi_float = np.diff(peaks_f) / ppg_fs * 1000.0
                else:
                    peaks_f = np.array([], dtype=np.float64)
                    ibi_int = np.array([], dtype=np.float64)
                    ibi_float = np.array([], dtype=np.float64)

                cached_peaks.append(peaks)
                cached_peaks_f.append(peaks_f)
                cached_ibi_int.append(ibi_int)
                cached_ibi_float.append(ibi_float)

            # --- 遍历所有消融步骤 × ECG 插值选项 ---
            for step_name, do_interp, do_ibi, do_thresh in STEP_CONFIGS:
                for ecg_interp in [False, True]:
                    # 初始化各子集的收集容器
                    ecg_vals = []      # 全部有效的 ECG RMSSD
                    ppg_vals = []      # 全部有效的 PPG RMSSD
                    errors = []        # 全部有效的绝对误差
                    n_valid = 0
                    n_aligned_valid = 0
                    n_clean_valid = 0
                    ecg_aligned = []   # 时间对齐子集
                    ppg_aligned = []
                    ecg_clean = []     # 低运动子集
                    ppg_clean = []
                    ecg_clean_aligned = []  # 低运动 ∩ 时间对齐子集
                    ppg_clean_aligned = []
                    n_clean_aligned_valid = 0

                    for i in range(n_windows):
                        # --- 计算 ECG RMSSD ---
                        ecg_v = ecg_rmssd(
                            rr_all[i], ecg_signal[i],
                            r_peak_samples[i], int(n_rr[i]),
                            do_interp=ecg_interp,
                            do_ibi_correct=do_ibi,
                            do_threshold=do_thresh,
                        )

                        # --- 计算 PPG RMSSD（使用缓存） ---
                        if cached_peaks[i].size < 3:
                            ppg_v = float("nan")
                        else:
                            ppg_v = ppg_rmssd_from_cache(
                                cached_ibi_int[i], cached_ibi_float[i],
                                do_interp=do_interp,
                                do_ibi_correct=do_ibi,
                                do_threshold=do_thresh,
                            )

                        # --- 收集有效配对结果 ---
                        if np.isfinite(ecg_v) and np.isfinite(ppg_v):
                            n_valid += 1
                            errors.append(abs(ppg_v - ecg_v))
                            ecg_vals.append(ecg_v)
                            ppg_vals.append(ppg_v)

                            is_aligned = i in aligned_set
                            is_clean = motion_fracs[i] < 0.5

                            # 时间对齐子集
                            if is_aligned:
                                n_aligned_valid += 1
                                ecg_aligned.append(ecg_v)
                                ppg_aligned.append(ppg_v)

                            # 低运动子集
                            if is_clean:
                                n_clean_valid += 1
                                ecg_clean.append(ecg_v)
                                ppg_clean.append(ppg_v)

                            # 低运动 ∩ 时间对齐子集
                            if is_aligned and is_clean:
                                n_clean_aligned_valid += 1
                                ecg_clean_aligned.append(ecg_v)
                                ppg_clean_aligned.append(ppg_v)

                    # --- 聚合统计量计算 ---
                    def _agg(err_list, ecg_list, ppg_list):
                        """计算 MAE、RMSE、Pearson r。"""
                        if not err_list:
                            return float("nan"), float("nan"), float("nan")
                        mae = float(np.mean(err_list))
                        rmse = float(np.sqrt(np.mean(np.array(err_list)**2)))
                        if len(ecg_list) >= 3:
                            r, _ = pearsonr(ecg_list, ppg_list)
                            r = float(r)
                        else:
                            r = float("nan")
                        return mae, rmse, r

                    # 全部窗口的统计
                    mae, rmse, r = _agg(errors, ecg_vals, ppg_vals)
                    cov = n_valid / n_windows * 100 if n_windows > 0 else 0.0

                    # 时间对齐子集的统计
                    if ecg_aligned:
                        err_al = [abs(e - p) for e, p in zip(ecg_aligned, ppg_aligned)]
                        mae_al, _, r_al = _agg(err_al, ecg_aligned, ppg_aligned)
                    else:
                        mae_al, r_al = float("nan"), float("nan")

                    # 低运动子集的统计
                    if ecg_clean:
                        err_cl = [abs(e - p) for e, p in zip(ecg_clean, ppg_clean)]
                        mae_cl, _, r_cl = _agg(err_cl, ecg_clean, ppg_clean)
                    else:
                        mae_cl, r_cl = float("nan"), float("nan")

                    # 低运动 ∩ 时间对齐子集的统计
                    if ecg_clean_aligned:
                        err_ca = [abs(e - p) for e, p in zip(ecg_clean_aligned, ppg_clean_aligned)]
                        mae_ca, _, r_ca = _agg(err_ca, ecg_clean_aligned, ppg_clean_aligned)
                    else:
                        mae_ca, r_ca = float("nan"), float("nan")

                    # --- 备注标签 ---
                    notes_parts = []
                    if not do_ibi and not do_interp and not do_thresh:
                        notes_parts.append("纯基线")
                    if do_thresh and cov == 100.0:
                        notes_parts.append("80% 阈值：所有窗口均通过")
                    if prep == "raw" and dev != "Earring":
                        notes_parts.append("非 Earring 使用原始信号（预期更差）")
                    if prep == "bandpass" and dev == "Earring":
                        notes_parts.append("Earring 使用带通（Step9: 原始更好）")
                    if prep == "25hz":
                        notes_parts.append("25Hz 降采样用于 Step8 评估")

                    # --- 组装结果行 ---
                    rows.append({
                        "participant": pid,
                        "device": dev,
                        "channel": ch,
                        "preprocess": prep,
                        "ppg_fs": ppg_fs,
                        "step": step_name,
                        "interp": do_interp,
                        "ibi_correct": do_ibi,
                        "threshold": do_thresh,
                        "ecg_interp": ecg_interp,
                        "rmssd_mae": round(mae, 2) if np.isfinite(mae) else float("nan"),
                        "rmssd_rmse": round(rmse, 2) if np.isfinite(rmse) else float("nan"),
                        "pearson_r": round(r, 4) if np.isfinite(r) else float("nan"),
                        "coverage_pct": round(cov, 1),
                        "n_valid": n_valid,
                        "n_total": n_windows,
                        "n_aligned": len(aligned_set),
                        "n_aligned_valid": n_aligned_valid,
                        "mae_aligned": round(mae_al, 2) if np.isfinite(mae_al) else float("nan"),
                        "r_aligned": round(r_al, 4) if np.isfinite(r_al) else float("nan"),
                        "n_clean_valid": n_clean_valid,
                        "mae_clean": round(mae_cl, 2) if np.isfinite(mae_cl) else float("nan"),
                        "r_clean": round(r_cl, 4) if np.isfinite(r_cl) else float("nan"),
                        "n_clean_aligned_valid": n_clean_aligned_valid,
                        "mae_clean_aligned": round(mae_ca, 2) if np.isfinite(mae_ca) else float("nan"),
                        "r_clean_aligned": round(r_ca, 4) if np.isfinite(r_ca) else float("nan"),
                        "motion_threshold": round(motion_threshold, 4),
                        "notes": "; ".join(notes_parts) if notes_parts else "",
                    })

    return rows


# ===================================================================
# 主函数
# ===================================================================
def main():
    # --- 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="完整消融实验矩阵 v2")
    parser.add_argument("--participant", action="append", default=None)
    parser.add_argument("--tol-sec", type=int, default=10,
                        help="时间对齐容差（秒）")
    parser.add_argument("--serial", action="store_true",
                        help="禁用多进程（用于调试）")
    args = parser.parse_args()

    participants = args.participant or ["P7"]
    tol_ms = args.tol_sec * 1000
    root = config.HEURISTIC_WINDOWS_ROOT
    out_dir = Path(__file__).resolve().parent / "outputs"

    for pid_raw in participants:
        pid = normalize_participant_id(pid_raw)
        print(f"\n{'='*70}")
        print(f"  完整消融实验矩阵 v2 — {pid}")
        print(f"{'='*70}")

        # --- 第 1 步：时间对齐 ---
        print(f"\n[1/4] 时间对齐 (±{args.tol_sec}s)...")
        t0_dict = {}
        for dev in DEVICES:
            npz = merged_windows_npz(root, pid, dev)
            if npz.is_file():
                with np.load(npz, allow_pickle=True) as z:
                    t0_dict[dev] = np.asarray(z["t0_ms"])
        aligned = align_windows(t0_dict, tol_ms)
        n_aligned = len(next(iter(aligned.values()))) if aligned else 0
        print(f"  公共窗口数: {n_aligned}")

        # --- 第 2 步：运动阈值校准（逐设备） ---
        print(f"\n[2/4] 运动阈值校准（逐设备）...")
        all_motion_stds = {}
        for dev in DEVICES:
            npz = merged_windows_npz(root, pid, dev)
            if not npz.is_file():
                continue
            with np.load(npz, allow_pickle=True) as z:
                ax = np.asarray(z["accel_x"])
                ay = np.asarray(z["accel_y"])
                az = np.asarray(z["accel_z"])
                fs = float(z["ppg_fs"])
            stds = []
            for i in range(ax.shape[0]):
                seg_stds = compute_motion_stats(ax[i], ay[i], az[i], fs)
                stds.extend(seg_stds.tolist())
            all_motion_stds[dev] = np.array(stds)
            p25, p50, p75, p90 = np.percentile(stds, [25, 50, 75, 90])
            print(f"  {dev:10s}: p25={p25:.3f} p50={p50:.3f} p75={p75:.3f} p90={p90:.3f}")

        motion_thresholds = calibrate_motion_thresholds(all_motion_stds, percentile=75)
        print(f"  阈值 (p75): {motion_thresholds}")

        # --- 第 3 步：处理所有设备 ---
        print(f"\n[3/4] 处理设备...")
        device_args = [
            (dev, pid, root, aligned, motion_thresholds.get(dev, 0.5))
            for dev in DEVICES
            if merged_windows_npz(root, pid, dev).is_file()
        ]

        # 串行或并行处理
        if args.serial or len(device_args) <= 1:
            all_rows = []
            for da in device_args:
                all_rows.extend(process_device(da))
        else:
            with Pool(min(4, len(device_args))) as pool:
                results = pool.map(process_device, device_args)
            all_rows = [r for batch in results for r in batch]

        # --- 第 4 步：保存结果 ---
        print(f"\n[4/4] 保存结果...")
        df = pd.DataFrame(all_rows)
        p_dir = out_dir / pid
        p_dir.mkdir(parents=True, exist_ok=True)
        out_path = p_dir / f"full_step_matrix_v2_{pid}.csv"
        df.to_csv(out_path, index=False)

        print(f"\n[已保存] {out_path}")
        print(f"  总行数: {len(df)}")
        n_combos = len(df.drop_duplicates(
            subset=["device", "channel", "preprocess", "step", "ecg_interp"]))
        print(f"  唯一组合数: {n_combos}")
        expected = len(device_args) * len(CHANNELS) * 3 * len(STEP_CONFIGS) * 2
        print(f"  预期组合数: {expected}")

        # --- 快速摘要：每设备最优配置 ---
        print(f"\n{'='*70}")
        print(f"  快速摘要 — {pid}")
        print(f"{'='*70}")
        full = df[(df["step"] == "S1+S2+S7_full") & (df["ecg_interp"] == False)]
        if not full.empty:
            print(f"\n  每设备最优 MAE (S1+S2+S7, ecg_interp=False):")
            for dev in DEVICES:
                sub = full[full["device"] == dev]
                if sub.empty:
                    continue
                best = sub.loc[sub["rmssd_mae"].idxmin()]
                print(f"    {dev:10s} {best['channel']:12s} {best['preprocess']:10s} "
                      f"MAE={best['rmssd_mae']:7.2f}  r={best['pearson_r']:.4f}  "
                      f"cov={best['coverage_pct']:.1f}%  "
                      f"MAE_aligned={best['mae_aligned']}  MAE_clean={best['mae_clean']}")

    print("\n完成。")


if __name__ == "__main__":
    main()
