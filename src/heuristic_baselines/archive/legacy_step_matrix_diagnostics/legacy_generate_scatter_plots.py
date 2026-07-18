"""
历史探索脚本：为旧版 step-matrix 消融生成 PPG RMSSD vs ECG RMSSD 散点图。

本脚本已归档，不属于正式 v1-primary unified heuristic baseline 主流程。
它依赖 4-device / Necklace 时代的 step 配置，只用于历史追溯。
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr

# 包路径设置与导入
ARCHIVE_ROOT = Path(__file__).resolve().parent
HEURISTIC_ROOT = ARCHIVE_ROOT.parents[1]
sys.path.insert(0, str(HEURISTIC_ROOT))
from algorithms import hrv
from io_utils import merged_windows_npz, normalize_participant_id
import config

ECG_FS = 130.0


# -- 核心函数，来自旧版 full_step_matrix_v2.py --
def find_true_r_peaks(ecg_signal, marked_peaks):
    n = len(marked_peaks)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    offsets = np.zeros(n, dtype=np.int64)
    for i in range(n):
        mp = int(marked_peaks[i])
        lo, hi = max(0, mp - 25), min(len(ecg_signal), mp + 5)
        if lo >= hi: continue
        offsets[i] = (lo + int(np.argmax(ecg_signal[lo:hi]))) - mp
    median_offset = int(np.median(offsets))
    true_peaks = np.empty(n, dtype=np.int64)
    for i in range(n):
        mp = int(marked_peaks[i])
        expected = mp + median_offset
        lo, hi = max(0, expected - 4), min(len(ecg_signal), expected + 5)
        if lo >= hi:
            true_peaks[i] = mp; continue
        true_peaks[i] = lo + int(np.argmax(ecg_signal[lo:hi]))
    return true_peaks


def bandpass(sig, fs, lo=0.5, hi=8.0, order=3):
    b, a = butter(order, [lo, hi], btype="band", fs=fs)
    return filtfilt(b, a, sig)


def compute_rmssd(peaks_int, peaks_float, fs, *, do_interp, do_ibi_correct, do_threshold):
    """从峰值数组计算通用 RMSSD，供 PPG 侧使用。"""
    ibi = (np.diff(peaks_float) / fs * 1000.0) if do_interp else (np.diff(peaks_int) / fs * 1000.0)
    valid = (ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)
    if do_threshold and len(ibi) > 0 and valid.sum() / len(ibi) < 0.80:
        return float("nan")
    nn = ibi[valid]
    if nn.size < 3:
        return float("nan")
    if do_threshold:
        m = float(np.mean(nn))
        if m > 0 and float(np.std(nn, ddof=1) / m) > hrv.IBI_CV_MAX:
            return float("nan")
    if do_ibi_correct:
        nn = hrv._correct_ibi_artifacts(nn)
    rmssd = float(np.sqrt(np.mean(np.diff(nn) ** 2)))
    if rmssd > hrv.RMSSD_MAX_MS:
        return float("nan")
    return rmssd


def compute_ecg_rmssd(rr_ms, n_rr, ecg_signal, r_peak_samples, *,
                      do_interp, do_ibi_correct, do_threshold):
    if do_interp and ecg_signal is not None and r_peak_samples is not None:
        rp = r_peak_samples[:n_rr + 1].astype(np.int64)
        true_rp = find_true_r_peaks(ecg_signal, rp)
        peaks_f = hrv._refine_peaks_parabolic(ecg_signal, true_rp)
        rr = np.diff(peaks_f) / ECG_FS * 1000.0
    else:
        rr = np.asarray(rr_ms[:n_rr], dtype=np.float64)
    valid = (rr >= hrv.IBI_MIN_MS) & (rr <= hrv.IBI_MAX_MS)
    if do_threshold and rr.size > 0 and valid.sum() / len(rr) < 0.80:
        return float("nan")
    nn = rr[valid]
    if nn.size < 3:
        return float("nan")
    if do_threshold:
        m = float(np.mean(nn))
        if m > 0 and float(np.std(nn, ddof=1) / m) > hrv.IBI_CV_MAX:
            return float("nan")
    if do_ibi_correct:
        nn = hrv._correct_ibi_artifacts(nn)
    rmssd = float(np.sqrt(np.mean(np.diff(nn) ** 2)))
    if rmssd > hrv.RMSSD_MAX_MS:
        return float("nan")
    return rmssd


# -- 旧版配置 --
PARTICIPANT = "P7"
DEVICES = ["Earring", "Ring", "Necklace", "Watch"]
BEST_CHANNEL = {"Earring": "ppg_ir", "Ring": "ppg_green",
                "Necklace": "ppg_green", "Watch": "ppg_green"}
STEPS = [
    ("S0_baseline",    False, False, False),
    ("S1_interp",      True,  False, False),
    ("S2_ibi",         False, True,  False),
    ("S7_threshold",   False, False, True),
    ("S1+S2+S7_full",  True,  True,  True),
]
COLORS = {"S0_baseline": "#888", "S1_interp": "#e67e22",
          "S2_ibi": "#2ecc71", "S7_threshold": "#9b59b6",
          "S1+S2+S7_full": "#e74c3c"}

OUT_DIR = HEURISTIC_ROOT / "outputs" / PARTICIPANT / "scatter_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

root = config.HEURISTIC_WINDOWS_ROOT
pid = normalize_participant_id(PARTICIPANT)

# Load data
print(f"Loading data for {PARTICIPANT}...")
device_data = {}
for dev in DEVICES:
    npz_path = merged_windows_npz(root, pid, dev)
    if not npz_path.is_file():
        print(f"  SKIP {dev}: not found"); continue
    ch = BEST_CHANNEL[dev]
    with np.load(npz_path, allow_pickle=True) as z:
        if ch not in z.files:
            print(f"  SKIP {dev}: no {ch}"); continue
        device_data[dev] = {
            "ppg": np.asarray(z[ch], dtype=np.float64),
            "rr": np.asarray(z["rr_intervals_ms"]),
            "n_rr": np.asarray(z["n_rr"]),
            "ecg_signal": np.asarray(z["ecg_signal"]) if "ecg_signal" in z.files else None,
            "r_peak_samples": np.asarray(z["r_peak_samples"]) if "r_peak_samples" in z.files else None,
            "fs": float(z["ppg_fs"]),
        }
    print(f"  {dev} ({ch}): {device_data[dev]['ppg'].shape[0]} windows")

# Compute per-window RMSSD
print("\nComputing per-window RMSSD...")
results = {}
for step_name, do_interp, do_ibi, do_thresh in STEPS:
    print(f"  {step_name}...", flush=True)
    results[step_name] = {}
    for dev in DEVICES:
        if dev not in device_data:
            continue
        d = device_data[dev]
        n_wins = d["ppg"].shape[0]
        ppg_arr, ecg_arr = np.full(n_wins, np.nan), np.full(n_wins, np.nan)
        for i in range(n_wins):
            raw = d["ppg"][i]
            raw = raw[~np.isnan(raw)]
            if raw.size < 50:
                continue
            try:
                bp = bandpass(raw, d["fs"])
            except:
                bp = raw
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                peaks = hrv.detect_ppg_peaks(bp, d["fs"])
            if peaks.size < 3:
                continue
            peaks_f = hrv._refine_peaks_parabolic(bp, peaks)
            ibi_int = np.diff(peaks) / d["fs"] * 1000.0
            ppg_arr[i] = compute_rmssd(peaks, peaks_f, d["fs"],
                                       do_interp=do_interp, do_ibi_correct=do_ibi,
                                       do_threshold=do_thresh)
            n_rr_i = int(d["n_rr"][i])
            ecg_arr[i] = compute_ecg_rmssd(
                d["rr"][i], n_rr_i,
                d["ecg_signal"][i] if d["ecg_signal"] is not None else None,
                d["r_peak_samples"][i] if d["r_peak_samples"] is not None else None,
                do_interp=do_interp, do_ibi_correct=do_ibi, do_threshold=do_thresh)
        mask = np.isfinite(ppg_arr) & np.isfinite(ecg_arr)
        results[step_name][dev] = {"ppg": ppg_arr, "ecg": ecg_arr}
        print(f"    {dev}: {mask.sum()}/{n_wins} valid")

# Generate scatter plots
print("\nGenerating scatter plots...")

def scatter_ax(ax, ex, px, color, alpha, title, ylabel):
    ax.scatter(ex, px, s=12, alpha=alpha, c=color, edgecolors="none")
    if len(ex) > 0:
        lim = max(np.max(ex), np.max(px)) * 1.1
    else:
        lim = 200
    lim = max(lim, 10)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    mae = float(np.mean(np.abs(px - ex))) if len(px) > 0 else float("nan")
    r = pearsonr(ex, px)[0] if len(px) > 2 else float("nan")
    ax.text(0.05, 0.95, f"MAE={mae:.1f}\nr={r:.3f}\nn={len(px)}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    if title: ax.set_title(title, fontsize=11, fontweight="bold")
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("ECG RMSSD (ms)", fontsize=9)
    ax.tick_params(labelsize=8)

# For each non-baseline step: 2×4 grid (baseline top, step bottom)
for step_name, do_interp, do_ibi, do_thresh in STEPS[1:]:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"PPG vs ECG RMSSD — {step_name} vs Baseline  ({PARTICIPANT}, bandpass, symmetric ECG)",
                 fontsize=14, fontweight="bold")
    for col, dev in enumerate(DEVICES):
        if dev not in results[step_name]: continue
        ch = BEST_CHANNEL[dev]
        for row, (sn, lbl) in enumerate([("S0_baseline", "Baseline"), (step_name, step_name)]):
            ppg = results[sn][dev]["ppg"]; ecg = results[sn][dev]["ecg"]
            m = np.isfinite(ppg) & np.isfinite(ecg)
            scatter_ax(axes[row, col], ecg[m], ppg[m],
                       COLORS[sn], 0.4 if sn == "S0_baseline" else 0.5,
                       f"{dev} ({ch})" if row == 0 else None,
                       f"{lbl}\nPPG RMSSD (ms)" if col == 0 else None)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fname = f"scatter_{step_name}.png"
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")

# Overlay plot: baseline (gray) + full pipeline (red)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(f"S1+S2+S7 Full Pipeline vs Baseline  ({PARTICIPANT}, bandpass, symmetric ECG)",
             fontsize=13, fontweight="bold")
for col, dev in enumerate(DEVICES):
    if dev not in results["S1+S2+S7_full"]: continue
    ax = axes[col]; ch = BEST_CHANNEL[dev]
    for sn, c, a, lbl in [("S0_baseline", "#ccc", 0.25, "Baseline"),
                           ("S1+S2+S7_full", "#e74c3c", 0.5, "S1+S2+S7")]:
        p, e = results[sn][dev]["ppg"], results[sn][dev]["ecg"]
        m = np.isfinite(p) & np.isfinite(e)
        ax.scatter(e[m], p[m], s=12, alpha=a, c=c, edgecolors="none", label=lbl)
    all_v = []
    for sn in ["S0_baseline", "S1+S2+S7_full"]:
        p, e = results[sn][dev]["ppg"], results[sn][dev]["ecg"]
        m = np.isfinite(p) & np.isfinite(e)
        if m.any(): all_v.extend([e[m].max(), p[m].max()])
    lim = max(all_v) * 1.1 if all_v else 200
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    # Stats
    for sn, prefix in [("S0_baseline", "Base"), ("S1+S2+S7_full", "Full")]:
        p, e = results[sn][dev]["ppg"], results[sn][dev]["ecg"]
        m = np.isfinite(p) & np.isfinite(e)
        mae = float(np.mean(np.abs(p[m] - e[m]))) if m.sum() > 0 else float("nan")
        r = pearsonr(e[m], p[m])[0] if m.sum() > 2 else float("nan")
    p, e = results["S1+S2+S7_full"][dev]["ppg"], results["S1+S2+S7_full"][dev]["ecg"]
    m = np.isfinite(p) & np.isfinite(e)
    mae_f = float(np.mean(np.abs(p[m] - e[m]))) if m.sum() > 0 else float("nan")
    r_f = pearsonr(e[m], p[m])[0] if m.sum() > 2 else float("nan")
    p0, e0 = results["S0_baseline"][dev]["ppg"], results["S0_baseline"][dev]["ecg"]
    m0 = np.isfinite(p0) & np.isfinite(e0)
    mae_b = float(np.mean(np.abs(p0[m0] - e0[m0]))) if m0.sum() > 0 else float("nan")
    r_b = pearsonr(e0[m0], p0[m0])[0] if m0.sum() > 2 else float("nan")
    ax.text(0.05, 0.95, f"Base: MAE={mae_b:.1f} r={r_b:.3f}\nFull: MAE={mae_f:.1f} r={r_f:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_title(f"{dev} ({ch})", fontsize=11, fontweight="bold")
    ax.set_xlabel("ECG RMSSD (ms)", fontsize=10)
    if col == 0: ax.set_ylabel("PPG RMSSD (ms)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(labelsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT_DIR / "scatter_full_pipeline_overlay.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: scatter_full_pipeline_overlay.png")

print(f"\nDone! All plots in: {OUT_DIR}")
