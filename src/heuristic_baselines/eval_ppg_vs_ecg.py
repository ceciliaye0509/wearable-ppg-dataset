"""
PPG (PRV) vs ECG (gold-standard HRV) per-window comparison.

For each 5-min window:
  * ECG side: rebuild R-peaks from the stored ``rr_intervals_ms`` (already in ms,
    so independent of the 130 Hz ECG rate) and compute HRV with the SAME NeuroKit2
    estimators used on PPG -> directly comparable HRV_* columns.
  * PPG side: read the existing per-window HRV CSV produced by hrv_runner.py.

Windows are aligned by ``t0_ms``. Outputs, under outputs/<Px>/:
  * ppg_vs_ecg_<dev>_<channel>.csv  -- per-window PPG, ECG, and error for each metric
  * ppg_vs_ecg_<dev>_<channel>_summary.csv -- bias / LoA / MAE / correlation per metric

Run from the package dir:  python eval_ppg_vs_ecg.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
import config  # noqa: E402

from algorithms import hrv  # noqa: E402
from io_utils import (  # noqa: E402
    merged_windows_npz, normalize_participant_id, participant_device_id,
)

# Metrics to compare (must exist in both PPG CSV and ECG computation).
COMPARE_COLS = ("hr_mean", "HRV_RMSSD", "HRV_SDNN", "HRV_pNN50", "HRV_LF", "HRV_HF", "HRV_SD1", "HRV_SD2")
# Reconstruct ECG peaks on a 1 ms grid so RR(ms) maps exactly to sample indices.
RR_GRID_FS = 1000.0


def ecg_hrv_from_rr(rr_ms: np.ndarray) -> dict[str, float]:
    """Gold-standard HRV from an ECG RR series (ms) via NeuroKit2.

    Applies the same IBI processing as the PPG side for fair comparison:
      - Physiological range gate [300, 2000] ms
      - Step 7: 80% IBI validity ratio gate (PMC11644394)
      - Step 2: IBI artifact correction (Lipponen & Tarvainen 2019)
      - Time-domain override from corrected IBI (same as PPG side)

    Steps 1/8 (sub-sample interpolation) are not applicable because ECG RR
    intervals are already at ms precision — no quantization issue.
    """
    import os
    baseline_mode = os.environ.get("BASELINE", "") == "1"

    rr = np.asarray(rr_ms, dtype=np.float64)

    # --- Step 7: IBI validity ratio gate (same as PPG side) ---
    if not baseline_mode and rr.size > 0:
        n_valid = int(((rr >= hrv.IBI_MIN_MS) & (rr <= hrv.IBI_MAX_MS)).sum())
        valid_ratio = n_valid / len(rr)
        if valid_ratio < 0.80:
            return {c: float("nan") for c in COMPARE_COLS}

    rr = rr[(rr >= hrv.IBI_MIN_MS) & (rr <= hrv.IBI_MAX_MS)]  # physiological gate
    if rr.size < 3:
        return {c: float("nan") for c in COMPARE_COLS}

    # --- Step 2: IBI artifact correction (same as PPG side) ---
    if not baseline_mode:
        rr = hrv._correct_ibi_artifacts(rr, threshold=0.20)

    # RR(ms) -> cumulative R-peak times -> integer sample indices at 1000 Hz.
    peaks = np.rint(np.concatenate([[0.0], np.cumsum(rr)])).astype(np.int64)
    m = hrv.hrv_metrics(peaks, RR_GRID_FS, freq=True, nonlinear=True)

    # Override time-domain with corrected IBI (consistent with PPG side).
    if not baseline_mode:
        diff = np.diff(rr)
        m["HRV_RMSSD"] = float(np.sqrt(np.mean(diff**2))) if len(diff) > 0 else float("nan")
        m["HRV_SDNN"] = float(np.std(rr, ddof=1)) if len(rr) > 1 else float("nan")
        m["HRV_MeanNN"] = float(np.mean(rr))
        if m["HRV_MeanNN"] > 0:
            m["hr_mean"] = 60000.0 / m["HRV_MeanNN"]

    return {c: m.get(c, float("nan")) for c in COMPARE_COLS}


def _bland_altman(ppg: np.ndarray, ecg: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(ppg) & np.isfinite(ecg)
    p, e = ppg[mask], ecg[mask]
    out = {"n": int(p.size)}
    if p.size < 2:
        return {**out, "bias": np.nan, "loa_lo": np.nan, "loa_hi": np.nan,
                "mae": np.nan, "r": np.nan}
    diff = p - e  # PPG minus ECG
    bias, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    out.update(
        bias=bias, loa_lo=bias - 1.96 * sd, loa_hi=bias + 1.96 * sd,
        mae=float(np.mean(np.abs(diff))),
        r=float(np.corrcoef(p, e)[0, 1]) if p.size > 2 else np.nan,
    )
    return out


def run_one(raw_npz: Path, ppg_csv: Path, out_dir: Path, dev: str, ch: str) -> None:
    with np.load(raw_npz, allow_pickle=True) as z:
        if "rr_intervals_ms" not in z.files:
            print(f"  [SKIP] {raw_npz.name} has no ECG rr_intervals_ms"); return
        rr_all = np.asarray(z["rr_intervals_ms"]); n_rr = np.asarray(z["n_rr"])
        t0 = np.asarray(z["t0_ms"], dtype=np.float64)

    ppg_df = pd.read_csv(ppg_csv)

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(t0)):
            ecg_m = ecg_hrv_from_rr(rr_all[i][: int(n_rr[i])])
            rows.append({"t0_ms": float(t0[i]), **{f"ecg_{k}": v for k, v in ecg_m.items()}})
    ecg_df = pd.DataFrame(rows)

    # Align PPG and ECG on t0_ms.
    merged = ppg_df.merge(ecg_df, on="t0_ms", how="inner", suffixes=("", "_dup"))
    for c in COMPARE_COLS:
        if c in merged and f"ecg_{c}" in merged:
            merged[f"err_{c}"] = merged[c] - merged[f"ecg_{c}"]

    out_dir.mkdir(parents=True, exist_ok=True)
    per_win = out_dir / f"ppg_vs_ecg_{dev}_{ch}.csv"
    merged.to_csv(per_win, index=False)

    summ = []
    for c in COMPARE_COLS:
        if c in merged and f"ecg_{c}" in merged:
            stats = _bland_altman(merged[c].to_numpy(float), merged[f"ecg_{c}"].to_numpy(float))
            summ.append({"metric": c, **stats})
    summ_df = pd.DataFrame(summ)
    summ_path = out_dir / f"ppg_vs_ecg_{dev}_{ch}_summary.csv"
    summ_df.to_csv(summ_path, index=False)

    print(f"  [SAVED] {per_win.name} ({len(merged)} aligned windows)")
    print(f"  [SAVED] {summ_path.name}")
    show = summ_df[summ_df["metric"].isin(["hr_mean", "HRV_RMSSD", "HRV_SDNN"])]
    for _, r in show.iterrows():
        print(f"    {r['metric']:12} bias={r['bias']:+8.2f}  MAE={r['mae']:7.2f}  r={r['r']:.3f}  n={int(r['n'])}")


def main() -> None:
    root = config.HEURISTIC_WINDOWS_ROOT
    participants = [normalize_participant_id(p) for p in config.HEURISTIC_PIPELINE_PARTICIPANTS]
    roles = list(config.HEURISTIC_DEVICE_ROLES)
    _ch = config.HEURISTIC_PPG_CHANNELS
    channels = [_ch] if isinstance(_ch, str) else list(_ch)
    result_root = config.HEURISTIC_RESULT_ROOT

    for pid in participants:
        out_dir = (result_root / pid).resolve()
        for role in roles:
            raw = merged_windows_npz(root, pid, role)
            dev = participant_device_id(pid, role)
            if not raw.is_file():
                continue
            for ch in channels:
                ppg_csv = out_dir / f"hrv_{dev}_{ch}.csv"
                if not ppg_csv.is_file():
                    print(f"[SKIP] no PPG HRV csv: {ppg_csv.name} (run hrv_runner.py first)")
                    continue
                print(f"\n=== {dev} / {ch} ===")
                run_one(raw, ppg_csv, out_dir, dev, ch)
    print("\n[eval] Done.")


if __name__ == "__main__":
    main()
