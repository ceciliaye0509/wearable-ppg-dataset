"""
Experiment pipeline for two open questions:

Q1: Is 80% IBI validity threshold too extreme?
    → Sweep thresholds [0%, 50%, 60%, 70%, 80%, 90%, 95%]
    → Report MAE + coverage for each

Q3: Is detrend + bandpass (0.7-3.5Hz) helping or hurting baseline?
    → Run baseline with and without preprocessing
    → Compare MAE

Usage:
    python experiments.py --participant P7
"""
import sys, os, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from algorithms import hrv
from io_utils import merged_windows_npz, normalize_participant_id, participant_device_id
import config

VENV_PY = "/Users/jiqiyu/Desktop/Daily_HRV/venv/bin/python"
BASE = Path(__file__).resolve().parent
RR_GRID_FS = 1000.0
COMPARE_COLS = ("hr_mean", "HRV_RMSSD", "HRV_SDNN")


def ecg_hrv_corrected(rr_ms):
    """ECG HRV with IBI correction (same as eval v2)."""
    rr = np.asarray(rr_ms, dtype=np.float64)
    n_valid = int(((rr >= hrv.IBI_MIN_MS) & (rr <= hrv.IBI_MAX_MS)).sum())
    if rr.size > 0 and n_valid / len(rr) < 0.80:
        return {c: float("nan") for c in COMPARE_COLS}
    rr = rr[(rr >= hrv.IBI_MIN_MS) & (rr <= hrv.IBI_MAX_MS)]
    if rr.size < 3:
        return {c: float("nan") for c in COMPARE_COLS}
    rr = hrv._correct_ibi_artifacts(rr, threshold=0.20)
    diff = np.diff(rr)
    return {
        "hr_mean": 60000.0 / float(np.mean(rr)),
        "HRV_RMSSD": float(np.sqrt(np.mean(diff**2))),
        "HRV_SDNN": float(np.std(rr, ddof=1)),
    }


def ppg_hrv_with_threshold(ppg, fs, ibi_threshold):
    """Run PPG HRV pipeline with a specific IBI validity threshold."""
    ppg_clean = np.asarray(ppg, dtype=np.float64)
    ppg_clean = ppg_clean[~np.isnan(ppg_clean)]
    peaks = hrv.detect_ppg_peaks(ppg_clean, fs)

    if peaks.size < 3:
        return {c: float("nan") for c in COMPARE_COLS}

    # IBI validity gate with custom threshold
    if ibi_threshold > 0 and peaks.size >= 2:
        ibi_raw = np.diff(peaks) / fs * 1000.0
        n_valid = int(((ibi_raw >= hrv.IBI_MIN_MS) & (ibi_raw <= hrv.IBI_MAX_MS)).sum())
        if n_valid / len(ibi_raw) < ibi_threshold:
            return {c: float("nan") for c in COMPARE_COLS}

    # Sub-sample interpolation + IBI correction
    peaks_f = hrv._refine_peaks_parabolic(ppg_clean, peaks)
    ibi = np.diff(peaks_f) / fs * 1000.0
    nn = ibi[(ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)]
    if nn.size < 3:
        return {c: float("nan") for c in COMPARE_COLS}
    nn = hrv._correct_ibi_artifacts(nn, threshold=0.20)
    diff = np.diff(nn)
    mean_nn = float(np.mean(nn))
    return {
        "hr_mean": 60000.0 / mean_nn if mean_nn > 0 else float("nan"),
        "HRV_RMSSD": float(np.sqrt(np.mean(diff**2))),
        "HRV_SDNN": float(np.std(nn, ddof=1)),
    }


def ppg_hrv_no_preprocess(raw_ppg, fs, ibi_threshold=0.80):
    """Run PPG HRV without bandpass filtering (raw signal)."""
    ppg_clean = np.asarray(raw_ppg, dtype=np.float64)
    ppg_clean = ppg_clean[~np.isnan(ppg_clean)]
    # Skip bandpass — just detect peaks on raw signal
    peaks = hrv.detect_ppg_peaks(ppg_clean, fs)

    if peaks.size < 3:
        return {c: float("nan") for c in COMPARE_COLS}

    if ibi_threshold > 0 and peaks.size >= 2:
        ibi_raw = np.diff(peaks) / fs * 1000.0
        n_valid = int(((ibi_raw >= hrv.IBI_MIN_MS) & (ibi_raw <= hrv.IBI_MAX_MS)).sum())
        if n_valid / len(ibi_raw) < ibi_threshold:
            return {c: float("nan") for c in COMPARE_COLS}

    peaks_f = hrv._refine_peaks_parabolic(ppg_clean, peaks)
    ibi = np.diff(peaks_f) / fs * 1000.0
    nn = ibi[(ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)]
    if nn.size < 3:
        return {c: float("nan") for c in COMPARE_COLS}
    nn = hrv._correct_ibi_artifacts(nn, threshold=0.20)
    diff = np.diff(nn)
    mean_nn = float(np.mean(nn))
    return {
        "hr_mean": 60000.0 / mean_nn if mean_nn > 0 else float("nan"),
        "HRV_RMSSD": float(np.sqrt(np.mean(diff**2))),
        "HRV_SDNN": float(np.std(nn, ddof=1)),
    }


def experiment_q1_threshold_sweep(participant="P7"):
    """Q1: Sweep IBI validity thresholds and report MAE + coverage."""
    print(f"\n{'='*70}")
    print(f"  Q1: IBI Validity Threshold Sweep — {participant}")
    print(f"{'='*70}")

    thresholds = [0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    devices = ["Earring", "Ring", "Necklace", "Watch"]
    channel = "ppg_green"  # use green for all devices except Earring

    results = []
    for dev in devices:
        dev_id = participant_device_id(participant, dev)
        raw_npz = merged_windows_npz(participant, dev)
        prep_npz = BASE / "outputs" / participant / f"alignment_windows_{dev_id}_preprocess.npz"

        if not prep_npz.exists():
            print(f"  [SKIP] {dev} — no preprocessed data")
            continue

        with np.load(raw_npz, allow_pickle=True) as z:
            rr_all = np.asarray(z["rr_intervals_ms"])
            n_rr = np.asarray(z["n_rr"])

        with np.load(prep_npz, allow_pickle=True) as z:
            ch = "ppg_ir" if dev == "Earring" else channel
            if ch not in z.files:
                ch = channel
            ppg = z[ch]
            fs = float(z["ppg_fs"])

        n_windows = ppg.shape[0]
        for thresh in thresholds:
            valid = 0
            errors = []
            for i in range(n_windows):
                ecg_m = ecg_hrv_corrected(rr_all[i][:int(n_rr[i])])
                ppg_m = ppg_hrv_with_threshold(ppg[i], fs, thresh)
                ecg_rmssd = ecg_m.get("HRV_RMSSD", float("nan"))
                ppg_rmssd = ppg_m.get("HRV_RMSSD", float("nan"))
                if np.isfinite(ecg_rmssd) and np.isfinite(ppg_rmssd):
                    valid += 1
                    errors.append(abs(ppg_rmssd - ecg_rmssd))

            mae = np.mean(errors) if errors else float("nan")
            coverage = valid / n_windows * 100 if n_windows > 0 else 0
            results.append({
                "device": dev, "threshold": thresh,
                "mae": round(mae, 2), "coverage": round(coverage, 1),
                "n_valid": valid, "n_total": n_windows,
            })
            print(f"  {dev:10s}  thresh={thresh:.0%}  MAE={mae:8.2f}  coverage={coverage:5.1f}%  n={valid}/{n_windows}")

    df = pd.DataFrame(results)
    out_path = BASE / "outputs" / participant / "q1_threshold_sweep.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("  SUMMARY: Best threshold per device (MAE × Coverage trade-off)")
    print(f"{'='*70}")
    for dev in devices:
        sub = df[df["device"] == dev]
        if sub.empty:
            continue
        # Find threshold with best MAE where coverage > 50%
        viable = sub[sub["coverage"] > 50]
        if viable.empty:
            continue
        best = viable.loc[viable["mae"].idxmin()]
        print(f"  {dev:10s}  best_thresh={best['threshold']:.0%}  MAE={best['mae']:.2f}  coverage={best['coverage']:.1f}%")

    return df


def experiment_q3_bandpass(participant="P7"):
    """Q3: Compare baseline with and without bandpass preprocessing."""
    print(f"\n{'='*70}")
    print(f"  Q3: Bandpass vs No-Bandpass — {participant}")
    print(f"{'='*70}")

    devices = ["Earring", "Ring", "Necklace", "Watch"]
    results = []

    for dev in devices:
        dev_id = participant_device_id(participant, dev)
        raw_npz = merged_windows_npz(participant, dev)
        prep_npz = BASE / "outputs" / participant / f"alignment_windows_{dev_id}_preprocess.npz"

        if not prep_npz.exists():
            print(f"  [SKIP] {dev}")
            continue

        with np.load(raw_npz, allow_pickle=True) as z:
            rr_all = np.asarray(z["rr_intervals_ms"])
            n_rr = np.asarray(z["n_rr"])
            # Raw PPG (before bandpass)
            ch = "ppg_ir" if dev == "Earring" else "ppg_green"
            raw_ppg = z[ch] if ch in z.files else z["ppg_green"]
            raw_fs = float(z["ppg_fs"])

        with np.load(prep_npz, allow_pickle=True) as z:
            ch = "ppg_ir" if dev == "Earring" else "ppg_green"
            if ch not in z.files:
                ch = "ppg_green"
            bp_ppg = z[ch]
            bp_fs = float(z["ppg_fs"])

        n_windows = min(raw_ppg.shape[0], bp_ppg.shape[0])

        for mode in ["bandpass", "no_bandpass"]:
            valid = 0
            errors = []
            for i in range(n_windows):
                ecg_m = ecg_hrv_corrected(rr_all[i][:int(n_rr[i])])
                if mode == "bandpass":
                    ppg_m = ppg_hrv_with_threshold(bp_ppg[i], bp_fs, 0.80)
                else:
                    ppg_m = ppg_hrv_no_preprocess(raw_ppg[i], raw_fs, 0.80)

                ecg_rmssd = ecg_m.get("HRV_RMSSD", float("nan"))
                ppg_rmssd = ppg_m.get("HRV_RMSSD", float("nan"))
                if np.isfinite(ecg_rmssd) and np.isfinite(ppg_rmssd):
                    valid += 1
                    errors.append(abs(ppg_rmssd - ecg_rmssd))

            mae = np.mean(errors) if errors else float("nan")
            coverage = valid / n_windows * 100
            results.append({
                "device": dev, "mode": mode,
                "mae": round(mae, 2), "coverage": round(coverage, 1),
                "n_valid": valid,
            })
            print(f"  {dev:10s}  {mode:15s}  MAE={mae:8.2f}  coverage={coverage:5.1f}%  n={valid}/{n_windows}")

    df = pd.DataFrame(results)
    out_path = BASE / "outputs" / participant / "q3_bandpass_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", default="P7")
    parser.add_argument("--q1", action="store_true", help="Run Q1: threshold sweep")
    parser.add_argument("--q3", action="store_true", help="Run Q3: bandpass comparison")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    args = parser.parse_args()

    if args.all or args.q1:
        experiment_q1_threshold_sweep(args.participant)
    if args.all or args.q3:
        experiment_q3_bandpass(args.participant)
    if not (args.all or args.q1 or args.q3):
        print("Usage: python experiments.py --participant P7 --all")
        print("  --q1: Threshold sweep")
        print("  --q3: Bandpass comparison")
