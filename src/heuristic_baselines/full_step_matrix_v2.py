"""
Full Step Matrix v2 — Complete experiment with all improvements.

Changes from v1:
  1. Time alignment: ±10s tolerance matching across 4 devices
  2. ECG symmetric processing: both interpolated (scheme A) and raw (scheme C)
  3. Motion artifact QC: 10s sliding window, per-device threshold calibration
  4. 25Hz downsample mode for Step 8 evaluation
  5. Peak detection caching: 8x speedup
  6. Multiprocessing: 4 devices in parallel

Usage:
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from algorithms import hrv
from preprocess import preprocess_ppg
from io_utils import merged_windows_npz, normalize_participant_id
import config

DEVICES = ["Earring", "Ring", "Necklace", "Watch"]
CHANNELS = ["ppg_green", "ppg_ir"]
ECG_FS = 130.0  # derived from data analysis: r_peak_samples / rr_intervals_ms

# All 2^3 = 8 combinations of (interp, ibi_correct, threshold)
STEP_CONFIGS = [
    ("S0_baseline",         False, False, False),
    ("S1_interp_only",      True,  False, False),
    ("S2_ibi_only",         False, True,  False),
    ("S7_thresh_only",      False, False, True),
    ("S1+S2_interp_ibi",    True,  True,  False),
    ("S1+S7_interp_thresh", True,  False, True),
    ("S2+S7_ibi_thresh",    False, True,  True),
    ("S1+S2+S7_full",       True,  True,  True),
]


# ===================================================================
# ECG: find true R-wave peaks and interpolate
# ===================================================================
def find_true_r_peaks(ecg_signal, marked_peaks):
    """Two-pass R-peak correction for Pan-Tompkins integration delay.

    Pan-Tompkins marks positions ~11 samples (85ms at 130Hz) AFTER the
    actual R-wave peak.  The delay varies slightly beat-to-beat due to
    QRS morphology, heart-rate, and SNR variation.

    Pass 1 — wide search [p-25, p+5] to estimate the per-window median
             offset (robust to outliers).
    Pass 2 — tight search [expected ± 4] to precisely locate each R-peak
             without introducing jitter from distant noise maxima.

    This two-pass approach reduces systematic RMSSD error from ~5-7 ms MAE
    (single-pass [p-20, p-2]) to ~1.2-1.7 ms MAE, a 3-5× improvement.
    """
    n = len(marked_peaks)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    # Pass 1: wide search to estimate per-window median offset
    offsets = np.zeros(n, dtype=np.int64)
    for i in range(n):
        mp = int(marked_peaks[i])
        lo = max(0, mp - 25)
        hi = min(len(ecg_signal), mp + 5)
        if lo >= hi:
            continue
        offsets[i] = (lo + int(np.argmax(ecg_signal[lo:hi]))) - mp

    median_offset = int(np.median(offsets))

    # Pass 2: tight search around expected position
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
    """Compute ECG RMSSD with configurable steps.

    When do_interp=True (scheme A): find true R-peaks, apply sub-sample
    interpolation, recompute RR from interpolated positions.
    When do_interp=False (scheme C): use stored rr_intervals_ms directly.

    QC gates (symmetric with PPG side):
      - IBI validity ratio < 80% → NaN  (when do_threshold)
      - IBI CV > IBI_CV_MAX → NaN       (when do_threshold)
      - RMSSD > RMSSD_MAX_MS → NaN      (always, post-correction)
    """
    if do_interp and ecg_signal is not None and r_peak_samples is not None:
        rp = r_peak_samples[:n_rr_count + 1].astype(np.int64)
        true_rp = find_true_r_peaks(ecg_signal, rp)
        # Use raw ECG for interpolation — true R-peaks are local maxima
        peaks_f = hrv._refine_peaks_parabolic(ecg_signal, true_rp)
        rr = np.diff(peaks_f) / ECG_FS * 1000.0
    else:
        rr = np.asarray(rr_ms[:n_rr_count], dtype=np.float64)

    # Step 7: validity threshold + IBI CV gate
    valid_mask = (rr >= hrv.IBI_MIN_MS) & (rr <= hrv.IBI_MAX_MS)
    if do_threshold:
        if rr.size > 0 and valid_mask.sum() / len(rr) < 0.80:
            return float("nan")

    nn = rr[valid_mask]
    if nn.size < 3:
        return float("nan")

    # IBI CV gate (symmetric with PPG QC)
    if do_threshold:
        mean_nn = float(np.mean(nn))
        if mean_nn > 0:
            ibi_cv = float(np.std(nn, ddof=1) / mean_nn)
            if ibi_cv > hrv.IBI_CV_MAX:
                return float("nan")

    # Step 2: IBI correction
    if do_ibi_correct:
        nn = hrv._correct_ibi_artifacts(nn)

    rmssd = float(np.sqrt(np.mean(np.diff(nn) ** 2)))

    # RMSSD upper bound gate (symmetric with PPG QC)
    if rmssd > hrv.RMSSD_MAX_MS:
        return float("nan")

    return rmssd


# ===================================================================
# PPG RMSSD with cached peaks (8x speedup)
# ===================================================================
def ppg_rmssd_from_cache(ibi_int, ibi_float, *,
                          do_interp, do_ibi_correct, do_threshold):
    """Compute PPG RMSSD from pre-computed IBI arrays (no re-detection).

    QC gates (symmetric with ECG side):
      - IBI validity ratio < 80% → NaN  (when do_threshold)
      - IBI CV > IBI_CV_MAX → NaN       (when do_threshold)
      - RMSSD > RMSSD_MAX_MS → NaN      (always, post-correction)
    """
    ibi = ibi_float if do_interp else ibi_int

    if do_threshold:
        n_valid = int(((ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)).sum())
        if len(ibi) > 0 and n_valid / len(ibi) < 0.80:
            return float("nan")

    nn = ibi[(ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)]
    if nn.size < 3:
        return float("nan")

    # IBI CV gate (symmetric with ECG QC)
    if do_threshold:
        mean_nn = float(np.mean(nn))
        if mean_nn > 0:
            ibi_cv = float(np.std(nn, ddof=1) / mean_nn)
            if ibi_cv > hrv.IBI_CV_MAX:
                return float("nan")

    if do_ibi_correct:
        nn = hrv._correct_ibi_artifacts(nn)

    rmssd = float(np.sqrt(np.mean(np.diff(nn) ** 2)))

    # RMSSD upper bound gate (symmetric with ECG QC)
    if rmssd > hrv.RMSSD_MAX_MS:
        return float("nan")

    return rmssd


# ===================================================================
# Motion quality control
# ===================================================================
def compute_motion_stats(accel_x, accel_y, accel_z, fs, seg_sec=10):
    """Return per-segment motion std values for one window."""
    mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    seg_samples = int(seg_sec * fs)
    n_segs = len(mag) // seg_samples
    if n_segs == 0:
        return np.array([0.0])
    stds = np.array([
        np.std(mag[i * seg_samples:(i + 1) * seg_samples])
        for i in range(n_segs)
    ])
    return stds


def calibrate_motion_thresholds(all_stds_by_device, percentile=75):
    """Compute per-device motion threshold from distribution."""
    thresholds = {}
    for dev, stds in all_stds_by_device.items():
        if len(stds) == 0:
            thresholds[dev] = 0.5  # fallback
            continue
        thresholds[dev] = float(np.percentile(stds, percentile))
    return thresholds


def motion_fraction(seg_stds, threshold):
    """Fraction of 10s segments exceeding motion threshold."""
    if len(seg_stds) == 0:
        return 0.0
    return float(np.mean(seg_stds > threshold))


# ===================================================================
# Downsample to 25 Hz
# ===================================================================
def downsample_to_25hz(signal_100hz):
    """Anti-alias filter + decimate from 100Hz to 25Hz."""
    sig = np.asarray(signal_100hz, dtype=np.float64)
    sig = sig[~np.isnan(sig)]
    if len(sig) < 20:
        return sig
    # scipy decimate applies anti-aliasing filter internally
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return decimate(sig, 4, ftype='fir', zero_phase=True)


# ===================================================================
# Time alignment
# ===================================================================
def align_windows(t0_dict, tol_ms=10000):
    """Find window indices common to all 4 devices within tolerance.

    Returns dict: {device: list of window indices} for aligned windows,
    using the first available device as the reference.
    """
    devs = [d for d in DEVICES if d in t0_dict]
    if len(devs) < 4:
        return {d: [] for d in devs}

    ref_dev = devs[0]
    ref_t0 = t0_dict[ref_dev]
    aligned = {d: [] for d in devs}

    for ref_idx, t in enumerate(ref_t0):
        match_indices = {ref_dev: ref_idx}
        all_found = True
        for dev in devs[1:]:
            diffs = np.abs(t0_dict[dev] - t)
            best_idx = int(np.argmin(diffs))
            if diffs[best_idx] <= tol_ms:
                match_indices[dev] = best_idx
            else:
                all_found = False
                break
        if all_found:
            for dev, idx in match_indices.items():
                aligned[dev].append(idx)

    return aligned


# ===================================================================
# Process one device (called in parallel)
# ===================================================================
def process_device(args):
    """Process one device: all channels × preprocess × steps × ecg_interp."""
    dev, pid, root, aligned_indices, motion_threshold = args

    raw_npz = merged_windows_npz(root, pid, dev)
    if not raw_npz.is_file():
        return []  # skip

    # Load data
    with np.load(raw_npz, allow_pickle=True) as z:
        ppg_raw = {ch: np.asarray(z[ch]) for ch in CHANNELS if ch in z.files}
        fs = float(z["ppg_fs"])
        rr_all = np.asarray(z["rr_intervals_ms"])
        n_rr = np.asarray(z["n_rr"])
        ecg_signal = np.asarray(z["ecg"])
        r_peak_samples = np.asarray(z["r_peak_samples"])
        accel_x = np.asarray(z["accel_x"])
        accel_y = np.asarray(z["accel_y"])
        accel_z = np.asarray(z["accel_z"])

    n_windows = rr_all.shape[0]
    aligned_set = set(aligned_indices.get(dev, []))

    # Precompute bandpass and 25Hz versions
    ppg_bp = {}
    ppg_25hz = {}
    for ch in CHANNELS:
        if ch in ppg_raw:
            ppg_bp[ch] = preprocess_ppg(ppg_raw[ch], fs)
            # 25Hz downsample of bandpass'd signal
            ppg_25hz[ch] = np.array([
                downsample_to_25hz(ppg_bp[ch][i]) for i in range(n_windows)
            ], dtype=object)

    # Precompute motion stats per window
    motion_stds = []
    motion_fracs = []
    for i in range(n_windows):
        seg_stds = compute_motion_stats(accel_x[i], accel_y[i], accel_z[i], fs)
        motion_stds.append(seg_stds)
        motion_fracs.append(motion_fraction(seg_stds, motion_threshold))

    print(f"  {dev}: {n_windows} windows, {len(aligned_set)} aligned, "
          f"motion_thresh={motion_threshold:.3f}")

    rows = []
    preprocess_modes = ["bandpass", "raw", "25hz"]

    for ch in CHANNELS:
        if ch not in ppg_raw:
            continue

        for prep in preprocess_modes:
            # Select signal and fs
            if prep == "bandpass":
                ppg_data = ppg_bp[ch]
                ppg_fs = fs
            elif prep == "raw":
                ppg_data = ppg_raw[ch]
                ppg_fs = fs
            else:  # 25hz
                ppg_data = ppg_25hz[ch]
                ppg_fs = 25.0

            # ---- PEAK DETECTION CACHE (one per window) ----
            cached_peaks = []
            cached_peaks_f = []
            cached_ibi_int = []
            cached_ibi_float = []

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

            # ---- STEP × ECG_INTERP LOOP ----
            for step_name, do_interp, do_ibi, do_thresh in STEP_CONFIGS:
                for ecg_interp in [False, True]:
                    ecg_vals = []
                    ppg_vals = []
                    errors = []
                    n_valid = 0
                    n_aligned_valid = 0
                    n_clean_valid = 0
                    ecg_aligned = []
                    ppg_aligned = []
                    ecg_clean = []
                    ppg_clean = []
                    ecg_clean_aligned = []
                    ppg_clean_aligned = []
                    n_clean_aligned_valid = 0

                    for i in range(n_windows):
                        # ECG RMSSD
                        ecg_v = ecg_rmssd(
                            rr_all[i], ecg_signal[i],
                            r_peak_samples[i], int(n_rr[i]),
                            do_interp=ecg_interp,
                            do_ibi_correct=do_ibi,
                            do_threshold=do_thresh,
                        )

                        # PPG RMSSD (from cache)
                        if cached_peaks[i].size < 3:
                            ppg_v = float("nan")
                        else:
                            ppg_v = ppg_rmssd_from_cache(
                                cached_ibi_int[i], cached_ibi_float[i],
                                do_interp=do_interp,
                                do_ibi_correct=do_ibi,
                                do_threshold=do_thresh,
                            )

                        if np.isfinite(ecg_v) and np.isfinite(ppg_v):
                            n_valid += 1
                            errors.append(abs(ppg_v - ecg_v))
                            ecg_vals.append(ecg_v)
                            ppg_vals.append(ppg_v)

                            is_aligned = i in aligned_set
                            is_clean = motion_fracs[i] < 0.5

                            if is_aligned:
                                n_aligned_valid += 1
                                ecg_aligned.append(ecg_v)
                                ppg_aligned.append(ppg_v)

                            if is_clean:
                                n_clean_valid += 1
                                ecg_clean.append(ecg_v)
                                ppg_clean.append(ppg_v)

                            if is_aligned and is_clean:
                                n_clean_aligned_valid += 1
                                ecg_clean_aligned.append(ecg_v)
                                ppg_clean_aligned.append(ppg_v)

                    # Compute aggregates
                    def _agg(err_list, ecg_list, ppg_list):
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

                    mae, rmse, r = _agg(errors, ecg_vals, ppg_vals)
                    cov = n_valid / n_windows * 100 if n_windows > 0 else 0.0

                    # Aligned subset
                    if ecg_aligned:
                        err_al = [abs(e - p) for e, p in zip(ecg_aligned, ppg_aligned)]
                        mae_al, _, r_al = _agg(err_al, ecg_aligned, ppg_aligned)
                    else:
                        mae_al, r_al = float("nan"), float("nan")

                    # Clean subset
                    if ecg_clean:
                        err_cl = [abs(e - p) for e, p in zip(ecg_clean, ppg_clean)]
                        mae_cl, _, r_cl = _agg(err_cl, ecg_clean, ppg_clean)
                    else:
                        mae_cl, r_cl = float("nan"), float("nan")

                    # Clean ∩ Aligned subset
                    if ecg_clean_aligned:
                        err_ca = [abs(e - p) for e, p in zip(ecg_clean_aligned, ppg_clean_aligned)]
                        mae_ca, _, r_ca = _agg(err_ca, ecg_clean_aligned, ppg_clean_aligned)
                    else:
                        mae_ca, r_ca = float("nan"), float("nan")

                    # Notes
                    notes_parts = []
                    if not do_ibi and not do_interp and not do_thresh:
                        notes_parts.append("Pure baseline")
                    if do_thresh and cov == 100.0:
                        notes_parts.append("80% threshold: all windows passed")
                    if prep == "raw" and dev != "Earring":
                        notes_parts.append("Raw on non-Earring (expect worse)")
                    if prep == "bandpass" and dev == "Earring":
                        notes_parts.append("Bandpass on Earring (Step9: raw better)")
                    if prep == "25hz":
                        notes_parts.append("25Hz downsample for Step8 eval")

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
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Full Step Matrix v2")
    parser.add_argument("--participant", action="append", default=None)
    parser.add_argument("--tol-sec", type=int, default=10,
                        help="Time alignment tolerance in seconds")
    parser.add_argument("--serial", action="store_true",
                        help="Disable multiprocessing (for debugging)")
    args = parser.parse_args()

    participants = args.participant or ["P7"]
    tol_ms = args.tol_sec * 1000
    root = config.HEURISTIC_WINDOWS_ROOT
    out_dir = Path(__file__).resolve().parent / "outputs"

    for pid_raw in participants:
        pid = normalize_participant_id(pid_raw)
        print(f"\n{'='*70}")
        print(f"  Full Step Matrix v2 — {pid}")
        print(f"{'='*70}")

        # --- Step 1: Time alignment ---
        print(f"\n[1/4] Time alignment (±{args.tol_sec}s)...")
        t0_dict = {}
        for dev in DEVICES:
            npz = merged_windows_npz(root, pid, dev)
            if npz.is_file():
                with np.load(npz, allow_pickle=True) as z:
                    t0_dict[dev] = np.asarray(z["t0_ms"])
        aligned = align_windows(t0_dict, tol_ms)
        n_aligned = len(next(iter(aligned.values()))) if aligned else 0
        print(f"  Common windows: {n_aligned}")

        # --- Step 2: Motion threshold calibration ---
        print(f"\n[2/4] Motion threshold calibration (per-device)...")
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
        print(f"  Thresholds (p75): {motion_thresholds}")

        # --- Step 3: Process all devices ---
        print(f"\n[3/4] Processing devices...")
        device_args = [
            (dev, pid, root, aligned, motion_thresholds.get(dev, 0.5))
            for dev in DEVICES
            if merged_windows_npz(root, pid, dev).is_file()
        ]

        if args.serial or len(device_args) <= 1:
            all_rows = []
            for da in device_args:
                all_rows.extend(process_device(da))
        else:
            with Pool(min(4, len(device_args))) as pool:
                results = pool.map(process_device, device_args)
            all_rows = [r for batch in results for r in batch]

        # --- Step 4: Save ---
        print(f"\n[4/4] Saving results...")
        df = pd.DataFrame(all_rows)
        p_dir = out_dir / pid
        p_dir.mkdir(parents=True, exist_ok=True)
        out_path = p_dir / f"full_step_matrix_v2_{pid}.csv"
        df.to_csv(out_path, index=False)

        print(f"\n[SAVED] {out_path}")
        print(f"  Total rows: {len(df)}")
        n_combos = len(df.drop_duplicates(
            subset=["device", "channel", "preprocess", "step", "ecg_interp"]))
        print(f"  Unique combos: {n_combos}")
        expected = len(device_args) * len(CHANNELS) * 3 * len(STEP_CONFIGS) * 2
        print(f"  Expected: {expected}")

        # Quick summary: best config per device
        print(f"\n{'='*70}")
        print(f"  Quick Summary — {pid}")
        print(f"{'='*70}")
        full = df[(df["step"] == "S1+S2+S7_full") & (df["ecg_interp"] == False)]
        if not full.empty:
            print(f"\n  Best MAE per device (S1+S2+S7, ecg_interp=False):")
            for dev in DEVICES:
                sub = full[full["device"] == dev]
                if sub.empty:
                    continue
                best = sub.loc[sub["rmssd_mae"].idxmin()]
                print(f"    {dev:10s} {best['channel']:12s} {best['preprocess']:10s} "
                      f"MAE={best['rmssd_mae']:7.2f}  r={best['pearson_r']:.4f}  "
                      f"cov={best['coverage_pct']:.1f}%  "
                      f"MAE_aligned={best['mae_aligned']}  MAE_clean={best['mae_clean']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
