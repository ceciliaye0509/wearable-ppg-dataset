"""
Full Step Matrix: All devices × all channels × bandpass/raw × 7 step configs.

Produces a CSV with one row per combination. Every cell is filled — no dashes.
The 'notes' column explains why a value is unchanged or N/A.

Usage:
    python full_step_matrix.py --participant P7
    python full_step_matrix.py --participant P7 --participant P5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from algorithms import hrv
from preprocess import preprocess_ppg
from io_utils import merged_windows_npz, normalize_participant_id, participant_device_id
import config

DEVICES = ["Earring", "Ring", "Necklace", "Watch"]
CHANNELS = ["ppg_green", "ppg_ir"]

# ---------------------------------------------------------------------------
# Step configurations: each is an independent combination of flags
# ---------------------------------------------------------------------------
STEP_CONFIGS = [
    # (name,               interp, ibi_correct, threshold)
    # All 2^3 = 8 combinations of 3 binary flags
    ("S0_baseline",         False,  False,       False),
    ("S1_interp_only",      True,   False,       False),
    ("S2_ibi_only",         False,  True,        False),
    ("S7_thresh_only",      False,  False,       True),
    ("S1+S2_interp_ibi",    True,   True,        False),
    ("S1+S7_interp_thresh", True,   False,       True),
    ("S2+S7_ibi_thresh",    False,  True,        True),
    ("S1+S2+S7_full",       True,   True,        True),
]


# ---------------------------------------------------------------------------
# ECG RMSSD computation — two versions
# ---------------------------------------------------------------------------
def ecg_rmssd_v1(rr_ms):
    """ECG RMSSD without IBI correction (raw RR intervals)."""
    rr = np.asarray(rr_ms, dtype=np.float64)
    nn = rr[(rr >= hrv.IBI_MIN_MS) & (rr <= hrv.IBI_MAX_MS)]
    if nn.size < 3:
        return float("nan")
    return float(np.sqrt(np.mean(np.diff(nn) ** 2)))


def ecg_rmssd_v2(rr_ms):
    """ECG RMSSD with IBI correction + 80% gate."""
    rr = np.asarray(rr_ms, dtype=np.float64)
    valid_mask = (rr >= hrv.IBI_MIN_MS) & (rr <= hrv.IBI_MAX_MS)
    if rr.size > 0 and valid_mask.sum() / len(rr) < 0.80:
        return float("nan")
    rr = rr[valid_mask]
    if rr.size < 3:
        return float("nan")
    rr = hrv._correct_ibi_artifacts(rr)
    return float(np.sqrt(np.mean(np.diff(rr) ** 2)))


# ---------------------------------------------------------------------------
# PPG RMSSD computation — parameterised by step flags
# ---------------------------------------------------------------------------
def ppg_rmssd_step(sig, fs, *, do_interp, do_ibi_correct, do_threshold):
    """Run PPG HRV pipeline with specific step configuration."""
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig[~np.isnan(sig)]
    peaks = hrv.detect_ppg_peaks(sig, fs)
    if peaks.size < 3:
        return float("nan")

    if do_interp:
        peaks_f = hrv._refine_peaks_parabolic(sig, peaks)
        ibi = np.diff(peaks_f) / fs * 1000.0
    else:
        ibi = np.diff(peaks) / fs * 1000.0

    # IBI validity gate (Step 7)
    if do_threshold:
        n_valid = int(((ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)).sum())
        if len(ibi) > 0 and n_valid / len(ibi) < 0.80:
            return float("nan")

    nn = ibi[(ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)]
    if nn.size < 3:
        return float("nan")

    if do_ibi_correct:
        nn = hrv._correct_ibi_artifacts(nn)

    return float(np.sqrt(np.mean(np.diff(nn) ** 2)))


# ---------------------------------------------------------------------------
# Run one participant
# ---------------------------------------------------------------------------
def run_participant(participant):
    pid = normalize_participant_id(participant)
    root = config.HEURISTIC_WINDOWS_ROOT
    rows = []

    for dev in DEVICES:
        raw_npz = merged_windows_npz(root, pid, dev)
        if not raw_npz.is_file():
            # Record skip for all combos
            for ch in CHANNELS:
                for prep in ["bandpass", "raw"]:
                    for step_name, *_ in STEP_CONFIGS:
                        rows.append({
                            "participant": pid, "device": dev, "channel": ch,
                            "preprocess": prep, "step": step_name,
                            "interp": False, "ibi_correct": False, "threshold": False,
                            "ecg_version": "n/a",
                            "rmssd_mae": float("nan"), "rmssd_rmse": float("nan"),
                            "pearson_r": float("nan"),
                            "coverage_pct": 0.0, "n_valid": 0, "n_total": 0,
                            "notes": f"NPZ file missing: {raw_npz.name}",
                        })
            continue

        # Load raw data
        with np.load(raw_npz, allow_pickle=True) as z:
            data_keys = list(z.files)
            ppg_raw = {}
            for ch in CHANNELS:
                if ch in z.files:
                    ppg_raw[ch] = np.asarray(z[ch])
                else:
                    ppg_raw[ch] = None
            fs = float(z["ppg_fs"])
            rr_all = np.asarray(z["rr_intervals_ms"])
            n_rr = np.asarray(z["n_rr"])

        n_windows = rr_all.shape[0]

        # Precompute bandpass versions
        ppg_bp = {}
        for ch in CHANNELS:
            if ppg_raw[ch] is not None:
                ppg_bp[ch] = preprocess_ppg(ppg_raw[ch], fs)
            else:
                ppg_bp[ch] = None

        print(f"  {dev} (n={n_windows}, keys={[k for k in data_keys if 'ppg' in k]})")

        for ch in CHANNELS:
            for prep in ["bandpass", "raw"]:
                # Select signal
                if prep == "bandpass":
                    ppg_data = ppg_bp[ch]
                else:
                    ppg_data = ppg_raw[ch]

                if ppg_data is None:
                    for step_name, do_interp, do_ibi, do_thresh in STEP_CONFIGS:
                        rows.append({
                            "participant": pid, "device": dev, "channel": ch,
                            "preprocess": prep, "step": step_name,
                            "interp": do_interp, "ibi_correct": do_ibi,
                            "threshold": do_thresh,
                            "ecg_version": "n/a",
                            "rmssd_mae": float("nan"), "rmssd_rmse": float("nan"),
                            "pearson_r": float("nan"),
                            "coverage_pct": 0.0, "n_valid": 0, "n_total": 0,
                            "notes": f"Channel {ch} not present in NPZ for {dev}",
                        })
                    continue

                # Ensure window count matches
                n = min(ppg_data.shape[0], n_windows)

                for step_name, do_interp, do_ibi, do_thresh in STEP_CONFIGS:
                    # ECG version: use v2 when PPG uses IBI correction, v1 otherwise
                    if do_ibi:
                        ecg_fn = ecg_rmssd_v2
                        ecg_ver = "v2_corrected"
                    else:
                        ecg_fn = ecg_rmssd_v1
                        ecg_ver = "v1_raw"

                    ecg_vals = []
                    ppg_vals = []
                    errors = []
                    n_valid = 0

                    for i in range(n):
                        ecg_v = ecg_fn(rr_all[i][:int(n_rr[i])])
                        ppg_v = ppg_rmssd_step(
                            ppg_data[i], fs,
                            do_interp=do_interp,
                            do_ibi_correct=do_ibi,
                            do_threshold=do_thresh,
                        )
                        if np.isfinite(ecg_v) and np.isfinite(ppg_v):
                            n_valid += 1
                            errors.append(abs(ppg_v - ecg_v))
                            ecg_vals.append(ecg_v)
                            ppg_vals.append(ppg_v)

                    mae = float(np.mean(errors)) if errors else float("nan")
                    rmse = float(np.sqrt(np.mean(np.array(errors)**2))) if errors else float("nan")
                    if len(ecg_vals) >= 3:
                        r, _ = pearsonr(ecg_vals, ppg_vals)
                        r = float(r)
                    else:
                        r = float("nan")
                    cov = n_valid / n * 100 if n > 0 else 0.0

                    # Generate notes
                    notes_parts = []
                    if do_thresh and cov == 100.0:
                        notes_parts.append("80% threshold had no effect: all windows passed")
                    if do_thresh and cov < 50.0:
                        notes_parts.append(f"WARNING: >50% windows rejected by 80% threshold")
                    if not do_ibi and not do_interp and not do_thresh:
                        notes_parts.append("Pure baseline: no improvements applied")
                    if prep == "raw" and dev != "Earring":
                        notes_parts.append("Raw signal on non-Earring device (expect worse MAE)")
                    if prep == "bandpass" and dev == "Earring":
                        notes_parts.append("Bandpass on Earring (Step 9 suggests raw is better)")

                    notes = "; ".join(notes_parts) if notes_parts else ""

                    rows.append({
                        "participant": pid, "device": dev, "channel": ch,
                        "preprocess": prep, "step": step_name,
                        "interp": do_interp, "ibi_correct": do_ibi,
                        "threshold": do_thresh, "ecg_version": ecg_ver,
                        "rmssd_mae": round(mae, 2) if np.isfinite(mae) else float("nan"),
                        "rmssd_rmse": round(rmse, 2) if np.isfinite(rmse) else float("nan"),
                        "pearson_r": round(r, 4) if np.isfinite(r) else float("nan"),
                        "coverage_pct": round(cov, 1),
                        "n_valid": n_valid, "n_total": n,
                        "notes": notes,
                    })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Full Step Matrix experiment")
    parser.add_argument("--participant", action="append", default=None,
                        help="Participant ID(s), e.g. P7. Can be repeated.")
    args = parser.parse_args()

    participants = args.participant or ["P7"]
    out_dir = Path(__file__).resolve().parent / "outputs"

    for pid_raw in participants:
        pid = normalize_participant_id(pid_raw)
        print(f"\n{'='*70}")
        print(f"  Full Step Matrix — {pid}")
        print(f"{'='*70}\n")

        df = run_participant(pid)

        p_dir = out_dir / pid
        p_dir.mkdir(parents=True, exist_ok=True)
        out_path = p_dir / f"full_step_matrix_{pid}.csv"
        df.to_csv(out_path, index=False)
        print(f"\n[SAVED] {out_path}")
        print(f"  Total rows: {len(df)}")
        print(f"  Unique combos: {len(df.drop_duplicates(subset=['device','channel','preprocess','step']))}")

        # Print summary
        print(f"\n{'='*70}")
        print(f"  Summary — {pid}")
        print(f"{'='*70}")
        print(f"\n{'device':<12} {'channel':<12} {'preprocess':<12} {'step':<20} {'MAE':>8} {'r':>8} {'cov%':>6} {'ecg':>14}")
        print("-" * 96)
        for _, row in df.iterrows():
            mae_s = f"{row['rmssd_mae']:.2f}" if np.isfinite(row['rmssd_mae']) else "NaN"
            r_s = f"{row['pearson_r']:.4f}" if np.isfinite(row['pearson_r']) else "NaN"
            print(f"{row['device']:<12} {row['channel']:<12} {row['preprocess']:<12} "
                  f"{row['step']:<20} {mae_s:>8} {r_s:>8} {row['coverage_pct']:>5.1f}% "
                  f"{row['ecg_version']:>14}")

    print("\nDone.")


if __name__ == "__main__":
    main()
