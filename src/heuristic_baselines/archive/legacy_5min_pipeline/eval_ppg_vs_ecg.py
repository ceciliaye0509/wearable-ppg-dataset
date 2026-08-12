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
# Physiological upper bound for ECG RMSSD — values above indicate ECG artifacts.
RMSSD_MAX_ECG_MS = 300.0
# Bootstrap resamples for 95% CI on MAE and r.
N_BOOTSTRAP = 1000
# 活动分层使用的运动片段长度，沿用旧消融实验中的 10 秒设置。
MOTION_SEG_SEC = 10


def _nan_ecg_metrics(valid_ratio: float = np.nan) -> dict[str, float]:
    out = {c: float("nan") for c in COMPARE_COLS}
    out["ecg_valid_ratio"] = valid_ratio
    out["ecg_ibi_correction_ratio"] = float("nan")
    return out


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
    if rr.size == 0:
        return _nan_ecg_metrics()

    valid_mask_all = (rr >= hrv.IBI_MIN_MS) & (rr <= hrv.IBI_MAX_MS)
    valid_ratio = float(valid_mask_all.sum() / len(rr))

    # --- Step 7: IBI validity ratio gate (same as PPG side) ---
    if not baseline_mode:
        if valid_ratio < 0.80:
            return _nan_ecg_metrics(valid_ratio)

    rr = rr[valid_mask_all]  # physiological gate
    if rr.size < 3:
        return _nan_ecg_metrics(valid_ratio)

    # --- Step 2: IBI artifact correction (same as PPG side) ---
    corr_ratio = 0.0
    if not baseline_mode:
        rr, corr_ratio = hrv._correct_ibi_artifacts_with_ratio(rr, threshold=0.20)

    # RR(ms) -> cumulative R-peak times -> integer sample indices at 1000 Hz.
    peaks = np.rint(np.concatenate([[0.0], np.cumsum(rr)])).astype(np.int64)
    m = hrv.hrv_metrics(peaks, RR_GRID_FS, freq=True, nonlinear=True)

    # Override time-domain with corrected IBI (consistent with PPG side).
    if not baseline_mode:
        diff = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(diff**2))) if len(diff) > 0 else float("nan")
        sdnn = float(np.std(rr, ddof=1)) if len(rr) > 1 else float("nan")
        m["HRV_RMSSD"] = rmssd
        m["HRV_SDNN"] = sdnn
        m["HRV_MeanNN"] = float(np.mean(rr))
        if m["HRV_MeanNN"] > 0:
            m["hr_mean"] = 60000.0 / m["HRV_MeanNN"]
        # Poincaré SD1/SD2 from corrected IBI (avoids quantization from integer peaks).
        if np.isfinite(rmssd) and np.isfinite(sdnn):
            sd1 = rmssd / (2.0 ** 0.5)
            sd2_sq = max(0.0, 2.0 * sdnn ** 2 - sd1 ** 2)
            m["HRV_SD1"] = sd1
            m["HRV_SD2"] = sd2_sq ** 0.5

    result = {c: m.get(c, float("nan")) for c in COMPARE_COLS}
    result["ecg_valid_ratio"] = valid_ratio
    result["ecg_ibi_correction_ratio"] = corr_ratio
    # ECG physiological upper bound: cap implausibly high RMSSD.
    if result.get("HRV_RMSSD", 0.0) > RMSSD_MAX_ECG_MS:
        result["HRV_RMSSD"] = float("nan")
    return result


def _bootstrap_ci(
    p: np.ndarray, e: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = 42
) -> dict[str, float]:
    """Bootstrap 95% CI for MAE and Pearson r."""
    if len(p) < 10:
        return {"mae_ci_lo": np.nan, "mae_ci_hi": np.nan,
                "r_ci_lo": np.nan, "r_ci_hi": np.nan}
    rng = np.random.default_rng(seed)
    mae_b, r_b = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(p), len(p))
        mae_b.append(float(np.mean(np.abs(p[idx] - e[idx]))))
        if len(idx) > 2:
            r_b.append(float(np.corrcoef(p[idx], e[idx])[0, 1]))
    return {
        "mae_ci_lo": float(np.percentile(mae_b, 2.5)),
        "mae_ci_hi": float(np.percentile(mae_b, 97.5)),
        "r_ci_lo":   float(np.percentile(r_b, 2.5))  if r_b else np.nan,
        "r_ci_hi":   float(np.percentile(r_b, 97.5)) if r_b else np.nan,
    }


def _bland_altman(ppg: np.ndarray, ecg: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(ppg) & np.isfinite(ecg)
    p, e = ppg[mask], ecg[mask]
    out: dict[str, float] = {"n": int(p.size)}
    nan_row = dict(bias=np.nan, loa_lo=np.nan, loa_hi=np.nan,
                   mae=np.nan, rmse=np.nan, medae=np.nan, mae_p80=np.nan, r=np.nan,
                   mae_ci_lo=np.nan, mae_ci_hi=np.nan,
                   r_ci_lo=np.nan, r_ci_hi=np.nan)
    if p.size < 2:
        return {**out, **nan_row}
    diff = p - e  # PPG minus ECG
    absdiff = np.abs(diff)
    bias, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    n80 = max(2, int(np.floor(0.8 * len(absdiff))))
    out.update(
        bias=bias,
        loa_lo=bias - 1.96 * sd,
        loa_hi=bias + 1.96 * sd,
        mae=float(np.mean(absdiff)),
        rmse=float(np.sqrt(np.mean(diff ** 2))),
        medae=float(np.median(absdiff)),
        mae_p80=float(np.mean(np.sort(absdiff)[:n80])),
        r=float(np.corrcoef(p, e)[0, 1]) if p.size > 2 else np.nan,
    )
    out.update(_bootstrap_ci(p, e))
    return out


def _motion_fraction_per_window(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    fs: float, seg_sec: float = MOTION_SEG_SEC, percentile: float = 75,
) -> np.ndarray:
    """Return per-window motion fraction (fraction of chunks > p75 threshold)."""
    seg_n = max(1, int(seg_sec * fs))
    all_stds: list[float] = []
    per_win_stds: list[np.ndarray] = []
    for i in range(ax.shape[0]):
        mag = np.sqrt(ax[i] ** 2 + ay[i] ** 2 + az[i] ** 2)
        n_segs = len(mag) // seg_n
        if n_segs == 0:
            stds = np.array([float(np.std(mag))])
        else:
            stds = np.array([np.std(mag[j * seg_n:(j + 1) * seg_n]) for j in range(n_segs)])
        per_win_stds.append(stds)
        all_stds.extend(stds.tolist())
    if not all_stds:
        return np.zeros(ax.shape[0])
    threshold = float(np.percentile(all_stds, percentile))
    return np.array([float(np.mean(s > threshold)) for s in per_win_stds])


def run_one(raw_npz: Path, ppg_csv: Path, out_dir: Path, dev: str, ch: str) -> None:
    with np.load(raw_npz, allow_pickle=True) as z:
        if "rr_intervals_ms" not in z.files:
            print(f"  [SKIP] {raw_npz.name} has no ECG rr_intervals_ms"); return
        rr_all  = np.asarray(z["rr_intervals_ms"])
        n_rr    = np.asarray(z["n_rr"])
        t0      = np.asarray(z["t0_ms"], dtype=np.float64)
        has_accel = all(k in z.files for k in ("accel_x", "accel_y", "accel_z"))
        if has_accel:
            ax = np.asarray(z["accel_x"], dtype=np.float64)
            ay = np.asarray(z["accel_y"], dtype=np.float64)
            az = np.asarray(z["accel_z"], dtype=np.float64)
            fs_ppg = float(np.asarray(z["ppg_fs"]).item()) if "ppg_fs" in z.files else 100.0

    ppg_df = pd.read_csv(ppg_csv)

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(t0)):
            ecg_m = ecg_hrv_from_rr(rr_all[i][: int(n_rr[i])])
            metric_part = {f"ecg_{k}": v for k, v in ecg_m.items() if k in COMPARE_COLS}
            qc_part = {k: v for k, v in ecg_m.items() if k not in COMPARE_COLS}
            rows.append({"t0_ms": float(t0[i]), **metric_part, **qc_part})
    ecg_df = pd.DataFrame(rows)

    # Align PPG and ECG on t0_ms.
    merged = ppg_df.merge(ecg_df, on="t0_ms", how="inner", suffixes=("", "_dup"))
    for c in COMPARE_COLS:
        if c in merged and f"ecg_{c}" in merged:
            merged[f"err_{c}"] = merged[c] - merged[f"ecg_{c}"]
            merged[f"abs_err_{c}"] = np.abs(merged[f"err_{c}"])

    # --- Activity stratification ---
    if has_accel:
        mfrac = _motion_fraction_per_window(ax, ay, az, fs_ppg)
        # Map t0 index → motion fraction for aligned windows.
        t0_to_mfrac = {float(t0[i]): float(mfrac[i]) for i in range(len(t0))}
        merged["motion_fraction"] = merged["t0_ms"].map(t0_to_mfrac)
    else:
        merged["motion_fraction"] = np.nan

    # --- Temporal split: first 50% = early half, second 50% = holdout ---
    merged_sorted = merged.sort_values("t0_ms").reset_index(drop=True)
    split_idx = len(merged_sorted) // 2
    merged_sorted["split"] = "early"
    merged_sorted.loc[split_idx:, "split"] = "holdout"
    merged = merged_sorted  # use sorted order for all outputs

    # --- Device reliability flag ---
    # Requires: n >= 10 paired windows, r > 0.5 (moderate correlation),
    # and PPG median RMSSD < 2× ECG median (not wildly inflated).
    # r > 0 was too lenient: a device with r ≈ 0.07 (Watch P7) passed.
    RELIABLE_R_MIN = 0.5
    RELIABLE_N_MIN = 10
    rmssd_col = "HRV_RMSSD"
    if rmssd_col in merged and f"ecg_{rmssd_col}" in merged:
        paired_mask = merged[rmssd_col].notna() & merged[f"ecg_{rmssd_col}"].notna()
        n_paired = int(paired_mask.sum())
        if n_paired >= RELIABLE_N_MIN:
            ppg_med = float(merged.loc[paired_mask, rmssd_col].median())
            ecg_med = float(merged.loc[paired_mask, f"ecg_{rmssd_col}"].median())
            r_val = float(np.corrcoef(
                merged.loc[paired_mask, rmssd_col],
                merged.loc[paired_mask, f"ecg_{rmssd_col}"]
            )[0, 1])
            device_reliable = (
                r_val > RELIABLE_R_MIN
                and ppg_med < 2.0 * ecg_med
            )
        else:
            device_reliable = False
    else:
        device_reliable = False

    # --- Summary: full / holdout / activity strata ---
    def _summ_rows(label: str, mask: pd.Series) -> list[dict]:
        sub = merged[mask]
        out_rows = []
        for c in COMPARE_COLS:
            if c in sub and f"ecg_{c}" in sub:
                stats = _bland_altman(sub[c].to_numpy(float), sub[f"ecg_{c}"].to_numpy(float))
                out_rows.append({"metric": c, "subset": label, **stats})
        return out_rows

    summ: list[dict] = []
    summ += _summ_rows("full", pd.Series([True] * len(merged), index=merged.index))
    summ += _summ_rows("holdout_50pct", merged["split"] == "holdout")
    if merged["motion_fraction"].notna().any():
        p33 = merged["motion_fraction"].quantile(0.33)
        p67 = merged["motion_fraction"].quantile(0.67)
        summ += _summ_rows("motion_low",  merged["motion_fraction"] <= p33)
        summ += _summ_rows("motion_mid",  (merged["motion_fraction"] > p33) & (merged["motion_fraction"] <= p67))
        summ += _summ_rows("motion_high", merged["motion_fraction"] > p67)

    summ_df = pd.DataFrame(summ)
    summ_df["device_reliable"] = device_reliable

    # Repeat full-subset agreement stats in the per-window file for traceability.
    # The dedicated *_summary.csv remains the canonical aggregate table.
    for _, srow in summ_df[summ_df["subset"] == "full"].iterrows():
        metric = str(srow["metric"])
        for stat in (
            "bias", "loa_lo", "loa_hi", "mae", "medae", "mae_p80", "r",
            "rmse", "mae_ci_lo", "mae_ci_hi", "r_ci_lo", "r_ci_hi",
        ):
            merged[f"full_{metric}_{stat}"] = srow.get(stat, np.nan)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_win = out_dir / f"ppg_vs_ecg_{dev}_{ch}.csv"
    merged.to_csv(per_win, index=False)

    summ_path = out_dir / f"ppg_vs_ecg_{dev}_{ch}_summary.csv"
    summ_df.to_csv(summ_path, index=False)

    print(f"  [SAVED] {per_win.name} ({len(merged)} aligned windows, "
          f"reliable={'YES' if device_reliable else 'NO ⚠'})")
    print(f"  [SAVED] {summ_path.name}")
    full_rows = summ_df[summ_df["subset"] == "full"]
    show = full_rows[full_rows["metric"].isin(["hr_mean", "HRV_RMSSD", "HRV_SDNN"])]
    for _, r in show.iterrows():
        print(f"    {r['metric']:12} bias={r['bias']:+8.2f}  "
              f"MAE={r['mae']:6.2f} [CI {r['mae_ci_lo']:.2f}–{r['mae_ci_hi']:.2f}]  "
              f"MedAE={r['medae']:6.2f}  MAE_p80={r['mae_p80']:6.2f}  "
              f"r={r['r']:.3f} [CI {r['r_ci_lo']:.3f}–{r['r_ci_hi']:.3f}]  "
              f"n={int(r['n'])}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default=None,
                    help="Comma-separated participant IDs, e.g. P7,P5. "
                         "Default: all from config.")
    args = ap.parse_args()

    root = config.HEURISTIC_WINDOWS_ROOT
    if args.participants:
        participants = [normalize_participant_id(p) for p in args.participants.split(",")]
    else:
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
