"""
Final four-device aligned HRV evaluation.

Reads per-device ``ppg_vs_ecg_<participant>_<device>_<channel>.csv`` files
created by eval_ppg_vs_ecg.py, keeps only windows where all four devices are
simultaneously available within a timestamp tolerance, and reports HRV accuracy
on that common-window denominator.

Outputs under outputs/<Px>/:
  aligned_eval_<Px>_<channel>_windows.csv
  aligned_eval_<Px>_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
import config  # noqa: E402
from eval_ppg_vs_ecg import COMPARE_COLS, _bland_altman  # noqa: E402
from io_utils import normalize_participant_id, participant_device_id  # noqa: E402

DEVICES = ("Earring", "Ring", "Necklace", "Watch")


def _align_indices(t0_by_device: dict[str, np.ndarray], tol_ms: float) -> list[dict[str, int]]:
    """Return matched row indices for timestamps present in all four devices."""
    if not all(d in t0_by_device for d in DEVICES):
        return []
    ref = DEVICES[0]
    matches: list[dict[str, int]] = []
    used = {d: set() for d in DEVICES}

    for ref_idx, t in enumerate(t0_by_device[ref]):
        row = {ref: ref_idx}
        ok = True
        for dev in DEVICES[1:]:
            diffs = np.abs(t0_by_device[dev] - t)
            idx = int(np.argmin(diffs))
            if float(diffs[idx]) > tol_ms or idx in used[dev]:
                ok = False
                break
            row[dev] = idx
        if ok:
            for dev, idx in row.items():
                used[dev].add(idx)
            matches.append(row)
    return matches


def _load_device_csv(out_dir: Path, pid: str, device: str, channel: str) -> pd.DataFrame | None:
    dev_id = participant_device_id(pid, device)
    path = out_dir / f"ppg_vs_ecg_{dev_id}_{channel}.csv"
    if not path.is_file():
        print(f"[SKIP] missing {path.name}; run eval_ppg_vs_ecg.py first")
        return None
    return pd.read_csv(path)


def _coverage_range(summary: pd.DataFrame, metric: str) -> str:
    sub = summary[summary["metric"] == metric]
    vals = sub["coverage_pct"].dropna().to_numpy(float)
    if vals.size == 0:
        return ""
    return f"{float(np.min(vals)):.1f}-{float(np.max(vals)):.1f}"


def run_participant(pid: str, channels: list[str], tol_ms: float) -> None:
    out_dir = (config.HEURISTIC_RESULT_ROOT / pid).resolve()
    all_summary: list[dict] = []

    for channel in channels:
        dfs: dict[str, pd.DataFrame] = {}
        for device in DEVICES:
            df = _load_device_csv(out_dir, pid, device, channel)
            if df is None:
                dfs = {}
                break
            dfs[device] = df.sort_values("t0_ms").reset_index(drop=True)
        if not dfs:
            continue

        t0_by_device = {d: dfs[d]["t0_ms"].to_numpy(float) for d in DEVICES}
        matches = _align_indices(t0_by_device, tol_ms)
        n_aligned = len(matches)
        if n_aligned == 0:
            print(f"[WARN] {pid}/{channel}: no four-device common windows")
            continue

        rows = []
        for align_id, idxs in enumerate(matches):
            row: dict[str, object] = {
                "participant": pid,
                "channel": channel,
                "aligned_window_id": align_id,
                "ref_t0_ms": float(dfs[DEVICES[0]].loc[idxs[DEVICES[0]], "t0_ms"]),
            }
            for device in DEVICES:
                src = dfs[device].loc[idxs[device]]
                prefix = device.lower()
                row[f"{prefix}_t0_ms"] = float(src["t0_ms"])
                for col in (
                    "HRV_RMSSD", "HRV_SDNN", "hr_mean",
                    "ecg_HRV_RMSSD", "ecg_HRV_SDNN", "ecg_hr_mean",
                    "sqi", "motion_fraction", "valid_ibi_ratio",
                    "ibi_correction_ratio", "ecg_valid_ratio",
                    "ecg_ibi_correction_ratio", "ppg_qc_reason",
                ):
                    if col in src.index:
                        row[f"{prefix}_{col}"] = src[col]
                for metric in COMPARE_COLS:
                    err_col = f"err_{metric}"
                    abs_col = f"abs_err_{metric}"
                    if err_col in src.index:
                        row[f"{prefix}_{err_col}"] = src[err_col]
                    if abs_col in src.index:
                        row[f"{prefix}_{abs_col}"] = src[abs_col]
            rows.append(row)

        aligned_df = pd.DataFrame(rows)
        win_path = out_dir / f"aligned_eval_{pid}_{channel}_windows.csv"
        aligned_df.to_csv(win_path, index=False)

        for device in DEVICES:
            prefix = device.lower()
            for metric in COMPARE_COLS:
                ppg_col = f"{prefix}_{metric}"
                ecg_col = f"{prefix}_ecg_{metric}"
                if ppg_col not in aligned_df or ecg_col not in aligned_df:
                    continue
                stats = _bland_altman(
                    aligned_df[ppg_col].to_numpy(float),
                    aligned_df[ecg_col].to_numpy(float),
                )
                n_valid = int(stats["n"])
                all_summary.append({
                    "participant": pid,
                    "channel": channel,
                    "device": device,
                    "metric": metric,
                    "n_aligned": n_aligned,
                    "n_valid": n_valid,
                    "coverage_pct": 100.0 * n_valid / n_aligned if n_aligned else np.nan,
                    **stats,
                    "mean_sqi": float(pd.to_numeric(
                        aligned_df.get(f"{prefix}_sqi"), errors="coerce"
                    ).mean()) if f"{prefix}_sqi" in aligned_df else np.nan,
                    "mean_motion_fraction": float(pd.to_numeric(
                        aligned_df.get(f"{prefix}_motion_fraction"), errors="coerce"
                    ).mean()) if f"{prefix}_motion_fraction" in aligned_df else np.nan,
                })

        print(f"[SAVED] {win_path.name} ({n_aligned} four-device windows)")

    if not all_summary:
        return

    summary = pd.DataFrame(all_summary)
    for metric in COMPARE_COLS:
        rng = _coverage_range(summary, metric)
        summary.loc[summary["metric"] == metric, "coverage_range_pct"] = rng

    summ_path = out_dir / f"aligned_eval_{pid}_summary.csv"
    summary.to_csv(summ_path, index=False)
    print(f"[SAVED] {summ_path.name}")

    show = summary[summary["metric"].isin(["HRV_RMSSD", "HRV_SDNN", "hr_mean"])]
    for _, r in show.iterrows():
        print(
            f"  {r['channel']:9s} {r['device']:9s} {r['metric']:10s} "
            f"MAE={r['mae']:6.2f} RMSE={r['rmse']:6.2f} "
            f"r={r['r']:.3f} cov={r['coverage_pct']:.1f}% n={int(r['n_valid'])}/{int(r['n_aligned'])}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Four-device aligned HRV evaluation.")
    ap.add_argument("--participants", default=None, help="Comma-separated IDs, e.g. P7,P11")
    ap.add_argument("--tol-sec", type=float, default=10.0)
    args = ap.parse_args()

    if args.participants:
        participants = [normalize_participant_id(p) for p in args.participants.split(",")]
    else:
        participants = [normalize_participant_id(p) for p in config.HEURISTIC_PIPELINE_PARTICIPANTS]

    ch_cfg = config.HEURISTIC_PPG_CHANNELS
    channels = [ch_cfg] if isinstance(ch_cfg, str) else list(ch_cfg)
    for pid in participants:
        run_participant(pid, channels, tol_ms=args.tol_sec * 1000.0)


if __name__ == "__main__":
    main()
