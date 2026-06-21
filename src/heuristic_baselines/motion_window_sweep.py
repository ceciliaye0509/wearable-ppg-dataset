"""
motion_window_sweep.py
----------------------
Sweep over motion-artifact segment sizes (seg_sec) to find the best
non-overlapping chunk size for PPG quality control.

Current implementation in full_step_matrix_v2.py uses 10s non-overlapping
chunks to compute per-segment accelerometer std, then flags windows where
>50% of chunks exceed the p75 motion threshold.

This script tests seg_sec in [2, 5, 10, 15, 20, 30, 60] and reports, for
each size, how MAE (PPG RMSSD vs ECG RMSSD) and coverage change on the
"clean" subset.

Pipeline used:
  - PPG: Optimized (interp + IBI correction + 80% validity gate)
  - ECG: raw rr_intervals_ms + 80% validity gate + IBI correction
  - Motion gate: motion_fraction < clean_fraction_threshold (swept over
    [0.3, 0.5, 0.7]) for each seg_sec

Usage:
    python motion_window_sweep.py
    python motion_window_sweep.py --participants P7,P11 --devices Watch,Ring
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.stats import pearsonr

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from algorithms.hrv import (
    detect_ppg_peaks,
    _refine_peaks_parabolic,
    _correct_ibi_artifacts,
    IBI_MIN_MS,
    IBI_MAX_MS,
)
from preprocess import bandpass_filter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "hf_upload" / "5min_windowed"
OUT_DIR = _ROOT / "outputs" / "motion_window_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICES = ["Earring", "Ring", "Necklace", "Watch"]
CHANNELS = ["ppg_green", "ppg_ir"]
NO_BANDPASS_DEVICES = {"Earring"}

SEG_SEC_VALUES = [2, 5, 10, 15, 20, 30, 60]
CLEAN_FRACTIONS = [0.3, 0.5, 0.7]
MOTION_CALIBRATION_PERCENTILE = 75


# ---------------------------------------------------------------------------
# ECG RMSSD (gold standard)
# ---------------------------------------------------------------------------
def ecg_rmssd(rr_ms: np.ndarray, n_rr: int) -> float:
    rr = np.asarray(rr_ms[:n_rr], dtype=np.float64)
    if rr.size == 0:
        return float("nan")
    valid = (rr >= IBI_MIN_MS) & (rr <= IBI_MAX_MS)
    if valid.sum() / len(rr) < 0.80:
        return float("nan")
    nn = rr[valid]
    if nn.size < 3:
        return float("nan")
    nn = _correct_ibi_artifacts(nn)
    return float(np.sqrt(np.mean(np.diff(nn) ** 2)))


# ---------------------------------------------------------------------------
# PPG RMSSD (Optimized Pipeline: interp + IBI correction + 80% gate)
# ---------------------------------------------------------------------------
def ppg_rmssd(ppg_raw: np.ndarray, fs: float, use_raw: bool) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if use_raw:
            sig = ppg_raw.astype(np.float64)
        else:
            x = detrend(ppg_raw.astype(np.float64), type="linear")
            sig = bandpass_filter(x, 0.7, 3.5, fs)
        peaks = detect_ppg_peaks(sig, fs)

    if peaks.size < 3:
        return float("nan")

    ibi = np.diff(peaks) / fs * 1000.0
    valid_ibi = (ibi >= IBI_MIN_MS) & (ibi <= IBI_MAX_MS)
    if len(ibi) == 0 or valid_ibi.sum() / len(ibi) < 0.80:
        return float("nan")

    peaks_f = _refine_peaks_parabolic(sig, peaks)
    ibi_f = np.diff(peaks_f) / fs * 1000.0
    nn = ibi_f[(ibi_f >= IBI_MIN_MS) & (ibi_f <= IBI_MAX_MS)]
    if nn.size < 3:
        return float("nan")
    nn = _correct_ibi_artifacts(nn)
    return float(np.sqrt(np.mean(np.diff(nn) ** 2)))


# ---------------------------------------------------------------------------
# Motion stats (non-overlapping chunks)
# ---------------------------------------------------------------------------
def compute_motion_stds(accel_x: np.ndarray, accel_y: np.ndarray,
                        accel_z: np.ndarray, fs: float, seg_sec: float) -> np.ndarray:
    mag = np.sqrt(accel_x ** 2 + accel_y ** 2 + accel_z ** 2)
    seg_n = int(seg_sec * fs)
    n_segs = len(mag) // seg_n
    if n_segs == 0:
        return np.array([float(np.std(mag))])
    return np.array([np.std(mag[i * seg_n:(i + 1) * seg_n]) for i in range(n_segs)])


def motion_fraction(seg_stds: np.ndarray, threshold: float) -> float:
    if len(seg_stds) == 0:
        return 0.0
    return float(np.mean(seg_stds > threshold))


# ---------------------------------------------------------------------------
# Process one participant / device / channel
# ---------------------------------------------------------------------------
def process_pdc(pid: str, device: str, channel: str) -> pd.DataFrame | None:
    npz_path = DATA_ROOT / pid / f"alignment_windows_{pid}_{device}.npz"
    if not npz_path.is_file():
        return None

    with np.load(npz_path, allow_pickle=True) as z:
        if channel not in z.files:
            return None
        ppg_all = np.asarray(z[channel], dtype=np.float64)
        rr_all = np.asarray(z["rr_intervals_ms"])
        n_rr = np.asarray(z["n_rr"])
        fs = float(np.asarray(z["ppg_fs"]).item())
        ax = np.asarray(z["accel_x"], dtype=np.float64)
        ay = np.asarray(z["accel_y"], dtype=np.float64)
        az = np.asarray(z["accel_z"], dtype=np.float64)

    n_windows = ppg_all.shape[0]
    use_raw = device in NO_BANDPASS_DEVICES

    # Compute PPG and ECG RMSSD for all windows (once, reused across sweeps)
    ppg_vals = np.array([ppg_rmssd(ppg_all[i], fs, use_raw) for i in range(n_windows)])
    ecg_vals = np.array([ecg_rmssd(rr_all[i], int(n_rr[i])) for i in range(n_windows)])

    # Compute per-window motion stds for each seg_sec (once per seg_sec)
    # Shape: {seg_sec: (n_windows,) array of motion stds arrays}
    motion_stds_all: dict[int, list[np.ndarray]] = {}
    for seg_sec in SEG_SEC_VALUES:
        stds_list = []
        for i in range(n_windows):
            stds_list.append(compute_motion_stds(ax[i], ay[i], az[i], fs, seg_sec))
        motion_stds_all[seg_sec] = stds_list

    rows = []
    paired = np.isfinite(ppg_vals) & np.isfinite(ecg_vals)
    n_paired = int(paired.sum())

    # Baseline (no motion filter)
    if n_paired >= 3:
        mae_base = float(np.mean(np.abs(ppg_vals[paired] - ecg_vals[paired])))
        r_base, _ = pearsonr(ppg_vals[paired], ecg_vals[paired])
    else:
        mae_base, r_base = float("nan"), float("nan")

    rows.append(dict(
        participant=pid, device=device, channel=channel,
        seg_sec="baseline", clean_frac_thresh="—",
        motion_thresh="—",
        n_total=n_windows, n_paired=n_paired,
        n_clean=n_paired,
        coverage_pct=round(n_paired / n_windows * 100, 1),
        mae=round(mae_base, 3) if np.isfinite(mae_base) else float("nan"),
        r=round(float(r_base), 4) if np.isfinite(r_base) else float("nan"),
    ))

    # Sweep seg_sec × clean_fraction
    for seg_sec in SEG_SEC_VALUES:
        stds_list = motion_stds_all[seg_sec]

        # Calibrate threshold from all segments across all windows (p75)
        all_stds = np.concatenate(stds_list)
        if len(all_stds) == 0:
            continue
        threshold = float(np.percentile(all_stds, MOTION_CALIBRATION_PERCENTILE))

        # Per-window motion fraction
        mfracs = np.array([motion_fraction(s, threshold) for s in stds_list])

        for clean_frac in CLEAN_FRACTIONS:
            clean_mask = mfracs < clean_frac
            combined = paired & clean_mask
            n_clean = int(combined.sum())
            cov = n_clean / n_windows * 100

            if n_clean >= 3:
                mae = float(np.mean(np.abs(ppg_vals[combined] - ecg_vals[combined])))
                r, _ = pearsonr(ppg_vals[combined], ecg_vals[combined])
                r = float(r)
            else:
                mae, r = float("nan"), float("nan")

            rows.append(dict(
                participant=pid, device=device, channel=channel,
                seg_sec=seg_sec, clean_frac_thresh=clean_frac,
                motion_thresh=round(threshold, 4),
                n_total=n_windows, n_paired=n_paired,
                n_clean=n_clean,
                coverage_pct=round(cov, 1),
                mae=round(mae, 3) if np.isfinite(mae) else float("nan"),
                r=round(r, 4) if np.isfinite(r) else float("nan"),
            ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default=None,
                    help="Comma-separated, e.g. P7,P11. Default: all available.")
    ap.add_argument("--devices", default=None,
                    help="Comma-separated, e.g. Watch,Ring. Default: all 4.")
    args = ap.parse_args()

    if args.participants:
        pids = [p.strip() for p in args.participants.split(",")]
    else:
        pids = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())

    devices = [d.strip() for d in args.devices.split(",")] if args.devices else DEVICES

    print(f"[motion_window_sweep] participants={pids}")
    print(f"[motion_window_sweep] devices={devices}")
    print(f"[motion_window_sweep] seg_sec values={SEG_SEC_VALUES}")
    print(f"[motion_window_sweep] clean_frac thresholds={CLEAN_FRACTIONS}")

    all_rows = []
    for pid in pids:
        for device in devices:
            for channel in CHANNELS:
                print(f"  {pid}/{device}/{channel} ...", end=" ", flush=True)
                df = process_pdc(pid, device, channel)
                if df is None or df.empty:
                    print("SKIP")
                    continue
                all_rows.append(df)
                n_base = df[df["seg_sec"] == "baseline"]["n_paired"].iloc[0]
                mae_base = df[df["seg_sec"] == "baseline"]["mae"].iloc[0]
                print(f"{n_base} paired windows, base MAE={mae_base:.2f} ms")

    if not all_rows:
        print("[WARN] No data processed.")
        return

    full_df = pd.concat(all_rows, ignore_index=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"motion_sweep_{ts}.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    # -----------------------------------------------------------------------
    # Summary: aggregate across participants — mean MAE and coverage per
    # (device, channel, seg_sec, clean_frac_thresh)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  SUMMARY: Mean MAE (ms) across all participants")
    print("  Gate: motion_fraction < clean_frac_thresh, threshold calibrated at p75")
    print("=" * 80)

    non_base = full_df[full_df["seg_sec"] != "baseline"].copy()
    non_base["seg_sec"] = non_base["seg_sec"].astype(int)
    non_base["clean_frac_thresh"] = non_base["clean_frac_thresh"].astype(float)

    base_df = full_df[full_df["seg_sec"] == "baseline"].copy()
    base_agg = (
        base_df.groupby(["device", "channel"])
        .agg(base_mae=("mae", "mean"), base_cov=("coverage_pct", "mean"))
        .reset_index()
    )

    sweep_agg = (
        non_base.groupby(["device", "channel", "seg_sec", "clean_frac_thresh"])
        .agg(mean_mae=("mae", "mean"), mean_cov=("coverage_pct", "mean"))
        .reset_index()
    )

    report_lines = []
    for device in devices:
        for channel in CHANNELS:
            sub_base = base_agg[(base_agg["device"] == device) & (base_agg["channel"] == channel)]
            sub_sweep = sweep_agg[(sweep_agg["device"] == device) & (sweep_agg["channel"] == channel)]
            if sub_base.empty or sub_sweep.empty:
                continue

            b_mae = sub_base["base_mae"].iloc[0]
            b_cov = sub_base["base_cov"].iloc[0]
            report_lines.append(f"\n{device} / {channel}")
            report_lines.append(f"  Baseline (no motion filter): MAE={b_mae:.2f} ms, coverage={b_cov:.1f}%")
            report_lines.append(f"  {'seg_sec':>8} {'gate<':>6} {'MAE (ms)':>10} {'MAE Δ':>8} {'Coverage':>10}")
            report_lines.append(f"  {'-'*46}")

            for _, row in sub_sweep.iterrows():
                delta = row["mean_mae"] - b_mae
                flag = " ✓" if delta < -0.5 else (" ✗" if delta > 0.5 else "")
                report_lines.append(
                    f"  {int(row['seg_sec']):>7}s {row['clean_frac_thresh']:>5.0%}"
                    f"  {row['mean_mae']:>9.2f}"
                    f"  {delta:>+7.2f}"
                    f"  {row['mean_cov']:>8.1f}%{flag}"
                )

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save markdown report
    md_path = OUT_DIR / f"motion_sweep_report_{ts}.md"
    md_lines = [
        "# Motion Artifact Segment-Size Sweep",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Participants**: {', '.join(pids)}",
        f"**Devices**: {', '.join(devices)}",
        f"**Channels**: {', '.join(CHANNELS)}",
        "",
        "## Method",
        "- Each 5-min PPG window is split into non-overlapping chunks of `seg_sec` seconds.",
        "- Per-chunk accelerometer magnitude std is computed.",
        "- Motion threshold is the p75 of all chunk stds for that participant/device.",
        "- A window is 'clean' if its motion fraction (fraction of chunks > threshold) is below the gate.",
        "- PPG RMSSD: Optimized pipeline (sub-sample interp + IBI correction + 80% validity gate).",
        "- ECG RMSSD: raw rr_intervals_ms + 80% validity gate + IBI correction.",
        "- Baseline: no motion filtering (all paired windows used).",
        "",
        "## Interpretation",
        "- **MAE Δ < -0.5 ms (✓)**: motion filter improves accuracy",
        "- **MAE Δ > +0.5 ms (✗)**: motion filter reduces accuracy (too many windows excluded)",
        "",
        "## Results",
        "",
        report_text,
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n[SAVED] {md_path}")

    # Best seg_sec per device/channel
    print("\n" + "=" * 80)
    print("  BEST seg_sec per device/channel (lowest mean MAE, coverage >= 50%)")
    print("=" * 80)
    print(f"  {'Device':<12} {'Channel':<12} {'Best seg_sec':>12} {'Gate':>6} {'MAE (ms)':>10} {'MAE Δ':>8} {'Coverage':>10}")
    print(f"  {'-'*70}")

    for device in devices:
        for channel in CHANNELS:
            sub_base = base_agg[(base_agg["device"] == device) & (base_agg["channel"] == channel)]
            sub_sweep = sweep_agg[(sweep_agg["device"] == device) & (sweep_agg["channel"] == channel)]
            if sub_base.empty or sub_sweep.empty:
                continue
            b_mae = sub_base["base_mae"].iloc[0]
            valid = sub_sweep[sub_sweep["mean_cov"] >= 50.0].copy()
            if valid.empty:
                continue
            best_idx = valid["mean_mae"].idxmin()
            best = valid.loc[best_idx]
            delta = best["mean_mae"] - b_mae
            print(
                f"  {device:<12} {channel:<12} {int(best['seg_sec']):>10}s"
                f" {best['clean_frac_thresh']:>5.0%}"
                f"  {best['mean_mae']:>9.2f}"
                f"  {delta:>+7.2f}"
                f"  {best['mean_cov']:>8.1f}%"
            )
    print()


if __name__ == "__main__":
    main()
