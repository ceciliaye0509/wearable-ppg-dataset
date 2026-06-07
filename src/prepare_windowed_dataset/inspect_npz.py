"""
Inspect alignment_windows NPZ files.

Usage:
    python inspect_npz.py                          # inspect all P8 devices
    python inspect_npz.py --pid P7                 # inspect all P7 devices
    python inspect_npz.py --pid P8 --device Ring   # inspect P8 Ring only
    python inspect_npz.py --pid P8 --window 5      # show details for window index 5
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

DEVICES = ("Earring", "Ring", "Necklace", "Watch")


def inspect_file(npz_path: Path, window_idx: int | None = None) -> None:
    """Print detailed inspection of a single NPZ file."""
    z = np.load(npz_path, allow_pickle=True)
    fsize_mb = os.path.getsize(npz_path) / 1024 / 1024

    n_windows = z["t0_ms"].shape[0]
    print(f"\n{'=' * 70}")
    print(f"  {npz_path.name}  ({fsize_mb:.1f} MB, {n_windows} windows)")
    print(f"{'=' * 70}")

    # ── Field summary ──
    print(f"\n  {'Field':<20} {'Shape':<20} {'Dtype':<12} {'Info'}")
    print(f"  {'-'*20} {'-'*20} {'-'*12} {'-'*30}")
    for k in sorted(z.files):
        arr = z[k]
        info = ""
        if arr.ndim == 0:
            info = f"value={arr}"
        elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.floating):
            valid = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
            if valid.size > 0:
                info = f"[{valid.min():.2f} .. {valid.max():.2f}], mean={valid.mean():.2f}"
        elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
            info = f"[{arr.min()} .. {arr.max()}], mean={arr.mean():.1f}"
        print(f"  {k:<20} {str(arr.shape):<20} {str(arr.dtype):<12} {info}")

    # ── Window duration ──
    dt_sec = (z["t1_ms"] - z["t0_ms"]) / 1000.0
    print(f"\n  Window duration: {dt_sec.mean():.1f} ± {dt_sec.std():.2f} sec")

    # ── HR stats ──
    if "hr_gt" in z.files:
        hr = z["hr_gt"]
        valid_hr = hr[~np.isnan(hr)]
        print(f"  HR ground truth:  {valid_hr.min():.1f} – {valid_hr.max():.1f} bpm, "
              f"mean={valid_hr.mean():.1f}, valid={len(valid_hr)}/{len(hr)}")

    # ── R-peak / RR stats ──
    if "n_rr" in z.files:
        n_rr = z["n_rr"]
        print(f"  RR intervals/win: min={n_rr.min()}, max={n_rr.max()}, mean={n_rr.mean():.0f}")

        # Compute HRV stats across all windows
        rr_data = z["rr_intervals_ms"]
        rmssd_list = []
        sdnn_list = []
        for i in range(n_windows):
            rr_i = rr_data[i, :n_rr[i]]
            if len(rr_i) > 1:
                sdnn_list.append(float(np.std(rr_i)))
                diff_rr = np.diff(rr_i)
                if len(diff_rr) > 0:
                    rmssd_list.append(float(np.sqrt(np.mean(diff_rr ** 2))))
        if rmssd_list:
            rmssd_arr = np.array(rmssd_list)
            sdnn_arr = np.array(sdnn_list)
            print(f"  RMSSD (all wins):  {rmssd_arr.min():.1f} – {rmssd_arr.max():.1f} ms, "
                  f"mean={rmssd_arr.mean():.1f}")
            print(f"  SDNN  (all wins):  {sdnn_arr.min():.1f} – {sdnn_arr.max():.1f} ms, "
                  f"mean={sdnn_arr.mean():.1f}")

    # ── Specific window detail ──
    if window_idx is not None:
        if window_idx < 0 or window_idx >= n_windows:
            print(f"\n  [ERROR] window_idx={window_idx} out of range [0, {n_windows - 1}]")
        else:
            _print_window_detail(z, window_idx)

    z.close()


def _print_window_detail(z, i: int) -> None:
    """Print detailed info for a single window."""
    print(f"\n  {'─' * 50}")
    print(f"  Window {i} detail")
    print(f"  {'─' * 50}")

    t0 = z["t0_ms"][i]
    t1 = z["t1_ms"][i]
    print(f"  Time range: {t0:.0f} – {t1:.0f} ms (duration: {(t1 - t0) / 1000:.1f} s)")

    if "hr_gt" in z.files:
        print(f"  HR ground truth: {z['hr_gt'][i]:.1f} bpm")

    if "n_peaks" in z.files:
        print(f"  R-peaks detected: {z['n_peaks'][i]}")

    if "ecg_valid_len" in z.files:
        ecg_vl = z["ecg_valid_len"][i]
        ecg_total = z["ecg"].shape[1]
        print(f"  ECG valid samples: {ecg_vl}/{ecg_total} ({100 * ecg_vl / ecg_total:.1f}%)")

    if "r_peak_samples" in z.files:
        rps = z["r_peak_samples"][i]
        valid_peaks = rps[rps >= 0]
        print(f"  R-peak positions (first 10): {valid_peaks[:10]}")
        print(f"  R-peak positions (last  10): {valid_peaks[-10:]}")

    if "n_rr" in z.files and "rr_intervals_ms" in z.files:
        n_rr_i = z["n_rr"][i]
        rr_i = z["rr_intervals_ms"][i, :n_rr_i]
        print(f"  RR intervals: {n_rr_i} total")
        print(f"    First 10 (ms): {np.round(rr_i[:10], 1)}")
        print(f"    Last  10 (ms): {np.round(rr_i[-10:], 1)}")
        print(f"    Mean: {rr_i.mean():.1f} ms, Std: {rr_i.std():.1f} ms")
        if len(rr_i) > 1:
            rmssd = float(np.sqrt(np.mean(np.diff(rr_i) ** 2)))
            sdnn = float(np.std(rr_i))
            print(f"    RMSSD: {rmssd:.1f} ms, SDNN: {sdnn:.1f} ms")

    # PPG signal stats
    for ch in ("ppg_ir", "ppg_green"):
        if ch in z.files:
            sig = z[ch][i]
            print(f"  {ch}: min={sig.min():.0f}, max={sig.max():.0f}, mean={sig.mean():.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect alignment_windows NPZ files")
    parser.add_argument("--pid", default="P8", help="Participant ID (default: P8)")
    parser.add_argument("--device", default=None, help="Device name (default: all)")
    parser.add_argument("--window", type=int, default=None, help="Show detail for window index")
    args = parser.parse_args()

    devices = [args.device] if args.device else list(DEVICES)

    for dev in devices:
        npz_path = OUTPUTS_DIR / args.pid / f"alignment_windows_{args.pid}_{dev}.npz"
        if not npz_path.exists():
            print(f"\n  [SKIP] {npz_path.name} not found")
            continue
        inspect_file(npz_path, window_idx=args.window)

    print()


if __name__ == "__main__":
    main()
