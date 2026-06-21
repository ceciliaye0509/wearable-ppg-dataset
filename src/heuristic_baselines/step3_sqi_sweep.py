"""
step3_sqi_sweep.py
------------------
Experiment: Can Step 3 (SQI gate) or stricter IBI quality metrics
improve upon the Optimized Pipeline?

Walkthrough note (Step 3): "SQA on preprocessed signal is ineffective —
bandpass already makes signal look clean. Try raw-signal SQI or IBI-based
quality metrics."

Gates tested (all applied ON TOP of Optimized Pipeline = Step 1+2+7+9):
  A. SQI on bandpass signal at thresholds 0.40-0.95 (current behaviour,
     but with thresholds above the current 0.40 default)
  B. SQI on RAW signal at thresholds 0.0-0.90
  C. Stricter IBI validity ratio 80%-95%
  D. IBI coefficient-of-variation < threshold (exclude noisy windows)
  E. IBI ectopic rate < threshold (exclude high consecutive-jump rate)

Baseline row = Optimized Pipeline with NO additional gate (threshold = 0.0
or 0.80 for the validity ratio).

Usage:
    python step3_sqi_sweep.py                      # P7 only
    python step3_sqi_sweep.py --participants P7,P5,P20,P11
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

import config
from algorithms.hrv import (
    detect_ppg_peaks,
    _refine_peaks_parabolic,
    _correct_ibi_artifacts,
    IBI_MIN_MS,
    IBI_MAX_MS,
    RMSSD_MAX_MS,
)
from algorithms.sqa import ppg_sqi
from preprocess import bandpass_filter
from io_utils import merged_windows_npz, normalize_participant_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICES    = ["Earring", "Ring", "Necklace", "Watch"]
CHANNELS   = ["ppg_green", "ppg_ir"]
# Earring uses raw signal for peak detection (Step 9)
NO_BANDPASS_DEVICES = {"Earring"}

# Gate sweeps
SQI_BP_THRESHOLDS    = [0.00, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
SQI_RAW_THRESHOLDS   = [0.00, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
IBI_VALID_THRESHOLDS = [0.80, 0.85, 0.90, 0.95]         # stricter Step 7
IBI_CV_THRESHOLDS    = [0.30, 0.25, 0.20, 0.15, 0.10]   # exclude HIGH CV (noisy)
IBI_ECTOPIC_THRESH   = [0.30, 0.25, 0.20, 0.15, 0.10]   # exclude HIGH ectopic rate

OUT_DIR = _ROOT / "outputs" / "step3_sqi_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# ECG gold standard (same as eval_ppg_vs_ecg.py — Step 2 + Step 7)
# ---------------------------------------------------------------------------
# ECG RMSSD upper bound: values above this indicate ECG artifact windows.
# eval_ppg_vs_ecg.py uses 300ms; we match it here for consistency.
ECG_RMSSD_MAX_MS = 300.0


def ecg_rmssd(rr_ms: np.ndarray) -> float:
    rr = np.asarray(rr_ms, dtype=np.float64)
    n_valid = int(((rr >= IBI_MIN_MS) & (rr <= IBI_MAX_MS)).sum())
    if rr.size == 0 or n_valid / len(rr) < 0.80:
        return float("nan")
    rr = rr[(rr >= IBI_MIN_MS) & (rr <= IBI_MAX_MS)]
    if rr.size < 3:
        return float("nan")
    rr = _correct_ibi_artifacts(rr)
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    # Fix #2: RMSSD upper bound — match eval_ppg_vs_ecg.py
    if rmssd > ECG_RMSSD_MAX_MS:
        return float("nan")
    return rmssd


# ---------------------------------------------------------------------------
# Optimized-Pipeline RMSSD + all quality side-channels for one window
# ---------------------------------------------------------------------------
def _preprocess_window(raw: np.ndarray, fs: float) -> np.ndarray:
    x = detrend(raw.astype(np.float64), type="linear")
    return bandpass_filter(x, 0.7, 3.5, fs)


def compute_window_metrics(
    ppg_raw: np.ndarray,
    ppg_bp: np.ndarray,
    fs: float,
    use_raw_for_detection: bool,
) -> dict:
    """
    Return a dict with:
      rmssd_opt        : Optimized-Pipeline RMSSD (NaN if gated)
      valid_ibi_ratio  : fraction of IBIs in [300,2000] ms
      sqi_bp           : composite SQI on bandpass signal
      sqi_raw          : composite SQI on raw signal
      ibi_cv           : coeff-of-variation of PRE-correction NN intervals
                         (matches hrv.py gate caliber — Fix #1)
      ibi_ectopic_rate : fraction of consec NN pairs differing > 20%
    """
    sig_for_detect = ppg_raw if use_raw_for_detection else ppg_bp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        peaks = detect_ppg_peaks(sig_for_detect, fs)

    result = dict(
        rmssd_opt=float("nan"),
        valid_ibi_ratio=0.0,
        sqi_bp=float("nan"),
        sqi_raw=float("nan"),
        ibi_cv=float("nan"),
        ibi_ectopic_rate=float("nan"),
    )

    if peaks.size < 3:
        return result

    ibi_raw = np.diff(peaks) / fs * 1000.0
    n_valid = int(((ibi_raw >= IBI_MIN_MS) & (ibi_raw <= IBI_MAX_MS)).sum())
    valid_ratio = n_valid / len(ibi_raw)
    result["valid_ibi_ratio"] = valid_ratio

    # SQI on both signal versions (using same detected peaks)
    result["sqi_bp"]  = ppg_sqi(ppg_bp,  fs, peaks)
    result["sqi_raw"] = ppg_sqi(ppg_raw, fs, peaks)

    # 80% IBI validity gate (Step 7)
    if valid_ratio < 0.80:
        return result

    # Step 1: cubic sub-sample interpolation
    peaks_f = _refine_peaks_parabolic(sig_for_detect, peaks)
    ibi = np.diff(peaks_f) / fs * 1000.0
    nn = ibi[(ibi >= IBI_MIN_MS) & (ibi <= IBI_MAX_MS)]
    if nn.size < 3:
        return result

    # Fix #1: IBI CV computed BEFORE artifact correction to match hrv.py
    # hrv.py gates on pre-correction CV, then corrects. Using post-correction
    # CV would underestimate the gate's actual filtering strength.
    mean_nn_raw = float(np.mean(nn))
    if mean_nn_raw > 0:
        result["ibi_cv"] = float(np.std(nn, ddof=1)) / mean_nn_raw
    # Ectopic rate also computed before correction (same IBI sequence)
    diff_nn_raw = np.diff(nn)
    if nn.size > 1:
        jumps = np.abs(diff_nn_raw) / nn[:-1]
        result["ibi_ectopic_rate"] = float(np.mean(jumps > 0.20))

    # Step 2: IBI artifact correction (AFTER computing CV/ectopic)
    nn = _correct_ibi_artifacts(nn)
    diff_nn = np.diff(nn)

    rmssd = float(np.sqrt(np.mean(diff_nn ** 2)))
    # RMSSD upper bound gate (match hrv.py)
    if rmssd > RMSSD_MAX_MS:
        return result  # rmssd_opt stays NaN
    result["rmssd_opt"] = rmssd

    return result


# ---------------------------------------------------------------------------
# Process one participant / device / channel
# ---------------------------------------------------------------------------
def process_participant_device_channel(
    pid: str, device: str, channel: str, windows_root: Path
) -> pd.DataFrame | None:
    """
    Return a DataFrame with one row per window:
      t0_ms, ecg_rmssd, rmssd_opt, valid_ibi_ratio, sqi_bp, sqi_raw,
      ibi_cv, ibi_ectopic_rate
    Returns None if NPZ not found or channel missing.
    """
    npz_path = merged_windows_npz(windows_root, pid, device)
    if not npz_path.is_file():
        return None

    with np.load(npz_path, allow_pickle=True) as z:
        if channel not in z.files:
            return None
        ppg_all  = np.asarray(z[channel], dtype=np.float64)
        rr_all   = np.asarray(z["rr_intervals_ms"])
        n_rr     = np.asarray(z["n_rr"])
        fs       = float(np.asarray(z["ppg_fs"]).item()) if "ppg_fs" in z.files else 100.0
        # Fix #3: include t0_ms for proper temporal split
        t0_all   = np.asarray(z["t0_ms"]) if "t0_ms" in z.files else None

    n_windows = ppg_all.shape[0]
    use_raw   = device in NO_BANDPASS_DEVICES

    rows = []
    for i in range(n_windows):
        raw_win = ppg_all[i]
        bp_win  = _preprocess_window(raw_win, fs)

        ecg_r   = ecg_rmssd(rr_all[i][: int(n_rr[i])])
        metrics = compute_window_metrics(raw_win, bp_win, fs, use_raw_for_detection=use_raw)
        row = {"ecg_rmssd": ecg_r, **metrics}
        if t0_all is not None:
            row["t0_ms"] = float(t0_all[i])
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gate application: filter by one extra quality criterion, compute stats
# ---------------------------------------------------------------------------
def _gate_stats(sub: pd.DataFrame, n_total: int) -> dict:
    """Compute MAE / MedAE / MAE_p80 / r / coverage for a filtered subset."""
    n_valid = len(sub)
    coverage = n_valid / n_total * 100 if n_total > 0 else 0.0
    if n_valid < 3:
        return dict(mae=float("nan"), medae=float("nan"), mae_p80=float("nan"),
                    r=float("nan"), coverage_pct=coverage, n_valid=n_valid, n_total=n_total)
    absdiff = np.abs(sub["rmssd_opt"].to_numpy(float) - sub["ecg_rmssd"].to_numpy(float))
    n80 = max(2, int(np.floor(0.8 * len(absdiff))))
    r, _ = pearsonr(sub["rmssd_opt"], sub["ecg_rmssd"])
    return dict(
        mae=float(np.mean(absdiff)),
        medae=float(np.median(absdiff)),
        mae_p80=float(np.mean(np.sort(absdiff)[:n80])),
        r=float(r),
        coverage_pct=coverage,
        n_valid=n_valid,
        n_total=n_total,
    )


def apply_gate(df: pd.DataFrame, gate_col: str, threshold: float,
               direction: str = "above") -> tuple[float, float, float, float, float, int, int]:
    """
    Apply an additional gate on top of the Optimized Pipeline.
    direction='above'  → keep windows where gate_col >= threshold (SQI, validity)
    direction='below'  → keep windows where gate_col <= threshold (CV, ectopic)

    Returns (mae, medae, mae_p80, r, coverage_pct, n_valid, n_total).
    """
    n_total = len(df)
    mask_opt = df["rmssd_opt"].notna() & df["ecg_rmssd"].notna()

    if gate_col is not None and threshold > 0.0:
        gate_col_vals = df[gate_col]
        if direction == "above":
            gate_mask = gate_col_vals >= threshold
        else:
            gate_mask = gate_col_vals <= threshold
        mask = mask_opt & gate_mask & gate_col_vals.notna()
    else:
        mask = mask_opt

    sub = df[mask]
    s = _gate_stats(sub, n_total)
    return s["mae"], s["medae"], s["mae_p80"], s["r"], s["coverage_pct"], s["n_valid"], s["n_total"]


# ---------------------------------------------------------------------------
# Build sweep rows for one participant / device / channel
# ---------------------------------------------------------------------------
def build_sweep_rows(df: pd.DataFrame, pid: str, device: str, channel: str) -> list[dict]:
    rows = []

    # Temporal split: tune on first half, report on second half (holdout).
    df_sorted = df.sort_values("t0_ms") if "t0_ms" in df.columns else df
    split_idx = len(df_sorted) // 2
    df_holdout = df_sorted.iloc[split_idx:].reset_index(drop=True)

    def _row(gate_type, threshold, direction="above"):
        mae, medae, mae_p80, r, cov, n_valid, n_total = apply_gate(
            df, gate_col, threshold, direction
        )
        # Same gate on holdout half
        mae_ho, medae_ho, mae_p80_ho, r_ho, cov_ho, n_ho, _ = apply_gate(
            df_holdout, gate_col, threshold, direction
        )
        return dict(
            participant=pid, device=device, channel=channel,
            gate_type=gate_type, threshold=threshold,
            mae=round(mae, 3) if np.isfinite(mae) else float("nan"),
            medae=round(medae, 3) if np.isfinite(medae) else float("nan"),
            mae_p80=round(mae_p80, 3) if np.isfinite(mae_p80) else float("nan"),
            r=round(r, 4) if np.isfinite(r) else float("nan"),
            coverage_pct=round(cov, 1),
            n_valid=n_valid, n_total=n_total,
            mae_holdout=round(mae_ho, 3) if np.isfinite(mae_ho) else float("nan"),
            medae_holdout=round(medae_ho, 3) if np.isfinite(medae_ho) else float("nan"),
            mae_p80_holdout=round(mae_p80_ho, 3) if np.isfinite(mae_p80_ho) else float("nan"),
            r_holdout=round(r_ho, 4) if np.isfinite(r_ho) else float("nan"),
            coverage_holdout=round(cov_ho, 1),
            n_holdout=n_ho,
        )

    # Baseline = Optimized Pipeline (no additional gate)
    gate_col = None
    rows.append({**_row("Optimized_baseline", 0.0), "gate_type": "Optimized_baseline"})

    # A. SQI on bandpass signal at higher thresholds
    gate_col = "sqi_bp"
    for thr in SQI_BP_THRESHOLDS:
        rows.append(_row(f"SQI_bandpass ≥ {thr}", thr, "above"))

    # B. SQI on RAW signal
    gate_col = "sqi_raw"
    for thr in SQI_RAW_THRESHOLDS:
        rows.append(_row(f"SQI_raw ≥ {thr}", thr, "above"))

    # C. Stricter IBI validity ratio
    gate_col = "valid_ibi_ratio"
    for thr in IBI_VALID_THRESHOLDS:
        rows.append(_row(f"IBI_validity ≥ {thr:.0%}", thr, "above"))

    # D. IBI coefficient-of-variation (exclude high-CV = noisy)
    gate_col = "ibi_cv"
    for thr in IBI_CV_THRESHOLDS:
        rows.append(_row(f"IBI_CV ≤ {thr}", thr, "below"))

    # E. IBI ectopic rate (exclude high ectopic = arrhythmia/noise)
    gate_col = "ibi_ectopic_rate"
    for thr in IBI_ECTOPIC_THRESH:
        rows.append(_row(f"IBI_ectopic ≤ {thr}", thr, "below"))

    return rows


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def _fmt(val, fmt=".2f"):
    return f"{val:{fmt}}" if np.isfinite(val) else "—"


def build_markdown_report(sweep_df: pd.DataFrame) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    participants = sweep_df["participant"].unique().tolist()
    lines = [
        f"# Step 3/4 SQI Sweep Report",
        f"",
        f"**Date**: {ts}  ",
        f"**Participants**: {', '.join(participants)}  ",
        f"**Baseline**: Optimized Pipeline (Step 1+2+7+9)  ",
        f"**Question**: Does SQI on raw signal or stricter IBI quality metrics improve HRV accuracy?",
        f"",
        f"> [!WARNING]",
        f"> This sweep's baseline differs from `eval_ppg_vs_ecg.py`: the sweep baseline",
        f"> does NOT include IBI\_CV ≤ 0.20 or RMSSD ≤ 200ms gates that are active in",
        f"> `hrv_from_ppg()`. MAE numbers here are NOT directly comparable to eval results.",
        f"> IBI\_CV is computed on PRE-correction IBI (matching `hrv.py` gate caliber).",
        f"",
    ]

    gate_families = {
        "A. SQI on preprocessed (bandpass) signal": "SQI_bandpass",
        "B. SQI on RAW signal (new)": "SQI_raw",
        "C. Stricter IBI validity ratio (Step 7 extension)": "IBI_validity",
        "D. IBI coefficient-of-variation gate": "IBI_CV",
        "E. IBI ectopic-rate gate": "IBI_ectopic",
    }

    for device in DEVICES:
        for channel in CHANNELS:
            sub = sweep_df[(sweep_df["device"] == device) & (sweep_df["channel"] == channel)]
            if sub.empty:
                continue
            lines.append(f"---")
            lines.append(f"## {device} / {channel}")
            lines.append(f"")

            # Baseline row first
            baseline = sub[sub["gate_type"] == "Optimized_baseline"]
            if not baseline.empty:
                b = baseline.iloc[0]
                lines.append(
                    f"**Optimized baseline**: MAE={_fmt(b['mae'])} ms, "
                    f"MedAE={_fmt(b.get('medae', float('nan')))} ms, "
                    f"MAE_p80={_fmt(b.get('mae_p80', float('nan')))} ms, "
                    f"r={_fmt(b['r'], '.4f')}, coverage={_fmt(b['coverage_pct'], '.1f')}%  "
                )
                lines.append(
                    f"**Baseline (holdout 50%)**:  MAE={_fmt(b.get('mae_holdout', float('nan')))} ms, "
                    f"MedAE={_fmt(b.get('medae_holdout', float('nan')))} ms, "
                    f"r={_fmt(b.get('r_holdout', float('nan')), '.4f')}, "
                    f"coverage={_fmt(b.get('coverage_holdout', float('nan')), '.1f')}%  "
                )
                lines.append(f"")

            for family_name, gate_prefix in gate_families.items():
                family_rows = sub[sub["gate_type"].str.startswith(gate_prefix)]
                if family_rows.empty:
                    continue
                lines.append(f"### {family_name}")
                lines.append(f"")
                lines.append(
                    f"| Threshold | MAE | MedAE | MAE_p80 | r | Coverage | "
                    f"MAE_ho | MedAE_ho | r_ho | Cov_ho | n_valid |"
                )
                lines.append(f"|---|---|---|---|---|---|---|---|---|---|---|")
                for _, row in family_rows.iterrows():
                    thr_str = row["gate_type"].split(" ", 1)[-1] if " " in row["gate_type"] else row["gate_type"]
                    better = ""
                    if np.isfinite(row["mae"]) and np.isfinite(b["mae"]) and row["mae"] < b["mae"] - 0.5:
                        better = " ✅"
                    elif np.isfinite(row["mae"]) and np.isfinite(b["mae"]) and row["mae"] > b["mae"] + 0.5:
                        better = " ❌"
                    lines.append(
                        f"| {thr_str} "
                        f"| {_fmt(row['mae'])}{better} "
                        f"| {_fmt(row.get('medae', float('nan')))} "
                        f"| {_fmt(row.get('mae_p80', float('nan')))} "
                        f"| {_fmt(row['r'], '.4f')} "
                        f"| {_fmt(row['coverage_pct'], '.1f')}% "
                        f"| {_fmt(row.get('mae_holdout', float('nan')))} "
                        f"| {_fmt(row.get('medae_holdout', float('nan')))} "
                        f"| {_fmt(row.get('r_holdout', float('nan')), '.4f')} "
                        f"| {_fmt(row.get('coverage_holdout', float('nan')), '.1f')}% "
                        f"| {int(row['n_valid']) if np.isfinite(row['n_valid']) else '—'} |"
                    )
                lines.append(f"")

    # Cross-device summary: best improvement per gate family
    lines.append(f"---")
    lines.append(f"## Summary: Best Gate per Device vs. Optimized Baseline")
    lines.append(f"")
    lines.append(
        f"| Device | Channel | Best gate | Threshold | "
        f"MAE Δ | MedAE Δ | MAE_p80 Δ | r Δ | Coverage | "
        f"MAE_ho Δ | r_ho Δ | Cov_ho |"
    )
    lines.append(f"|---|---|---|---|---|---|---|---|---|---|---|---|")

    for device in DEVICES:
        for channel in CHANNELS:
            sub = sweep_df[(sweep_df["device"] == device) & (sweep_df["channel"] == channel)]
            if sub.empty:
                continue
            baseline = sub[sub["gate_type"] == "Optimized_baseline"]
            if baseline.empty:
                continue
            b = baseline.iloc[0]
            non_base = sub[sub["gate_type"] != "Optimized_baseline"].copy()
            non_base = non_base.dropna(subset=["mae", "r"])
            if non_base.empty:
                continue
            non_base["mae_delta"] = non_base["mae"] - b["mae"]
            best_idx = non_base["mae_delta"].idxmin()
            best = non_base.loc[best_idx]

            def _delta(col):
                bv = b.get(col, float("nan"))
                rv = best.get(col, float("nan"))
                return rv - bv if np.isfinite(rv) and np.isfinite(bv) else float("nan")

            lines.append(
                f"| {device} | {channel} | {best['gate_type'].split(' ', 1)[0]} | "
                f"{best['gate_type'].split(' ', 1)[-1] if ' ' in best['gate_type'] else '—'} | "
                f"{_fmt(_delta('mae'), '+.2f')} ms | "
                f"{_fmt(_delta('medae'), '+.2f')} ms | "
                f"{_fmt(_delta('mae_p80'), '+.2f')} ms | "
                f"{_fmt(_delta('r'), '+.4f')} | "
                f"{_fmt(best['coverage_pct'], '.1f')}% | "
                f"{_fmt(_delta('mae_holdout'), '+.2f')} ms | "
                f"{_fmt(_delta('r_holdout'), '+.4f')} | "
                f"{_fmt(best.get('coverage_holdout', float('nan')), '.1f')}% |"
            )

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"## Interpretation")
    lines.append(f"")
    lines.append(f"- **MAE Δ < 0** (✅): gate improves accuracy vs Optimized baseline")
    lines.append(f"- **MAE Δ > 0** (❌): gate makes accuracy worse (too many valid windows excluded)")
    lines.append(f"- A gate is only useful if it improves MAE without collapsing coverage below ~80%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default=None,
                    help="Comma-separated participant IDs, e.g. P7,P5. "
                         "Default: config.HEURISTIC_PIPELINE_PARTICIPANTS")
    args = ap.parse_args()

    if args.participants:
        pids = [normalize_participant_id(p) for p in args.participants.split(",")]
    else:
        pids = [normalize_participant_id(p) for p in config.HEURISTIC_PIPELINE_PARTICIPANTS]

    windows_root = config.HEURISTIC_WINDOWS_ROOT
    print(f"[step3_sqi_sweep] participants={pids}")
    print(f"[step3_sqi_sweep] windows root={windows_root}")

    all_sweep_rows = []

    for pid in pids:
        print(f"\n{'='*60}\n  Participant: {pid}\n{'='*60}")
        for device in DEVICES:
            for channel in CHANNELS:
                print(f"\n  {device} / {channel} ...", end=" ", flush=True)
                df = process_participant_device_channel(pid, device, channel, windows_root)
                if df is None or df.empty:
                    print("SKIP (no data)")
                    continue

                # Save per-window intermediates for inspection
                pw_path = OUT_DIR / f"per_window_{pid}_{device}_{channel}.csv"
                df.to_csv(pw_path, index=False)

                n_opt = int(df["rmssd_opt"].notna().sum())
                n_ecg = int(df["ecg_rmssd"].notna().sum())
                n_both = int((df["rmssd_opt"].notna() & df["ecg_rmssd"].notna()).sum())
                print(f"{len(df)} windows | Opt valid={n_opt} | ECG valid={n_ecg} | paired={n_both}")

                # SQI distribution on raw signal (key diagnostic)
                raw_sqi = df["sqi_raw"].dropna()
                bp_sqi  = df["sqi_bp"].dropna()
                if len(raw_sqi):
                    print(f"    SQI_raw:  min={raw_sqi.min():.3f}  p25={raw_sqi.quantile(.25):.3f}  "
                          f"median={raw_sqi.median():.3f}  p75={raw_sqi.quantile(.75):.3f}  max={raw_sqi.max():.3f}")
                if len(bp_sqi):
                    print(f"    SQI_bp:   min={bp_sqi.min():.3f}  p25={bp_sqi.quantile(.25):.3f}  "
                          f"median={bp_sqi.median():.3f}  p75={bp_sqi.quantile(.75):.3f}  max={bp_sqi.max():.3f}")
                cv_vals = df["ibi_cv"].dropna()
                if len(cv_vals):
                    print(f"    IBI_CV:   min={cv_vals.min():.3f}  median={cv_vals.median():.3f}  max={cv_vals.max():.3f}")

                sweep_rows = build_sweep_rows(df, pid, device, channel)
                all_sweep_rows.extend(sweep_rows)

    if not all_sweep_rows:
        print("\n[WARN] No data processed.")
        return

    sweep_df = pd.DataFrame(all_sweep_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"threshold_sweep_{ts}.csv"
    sweep_df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    report = build_markdown_report(sweep_df)
    report_path = OUT_DIR / f"report_{ts}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[SAVED] {report_path}")

    # Print a condensed console summary
    print(f"\n{'='*70}")
    print(f"  Baseline vs. Best SQI/IBI gate — RMSSD MAE (ms)")
    print(f"{'='*70}")
    print(f"{'Device':<10} {'Channel':<12} {'Baseline':>9} {'Best gate':>9} {'Δ MAE':>8} {'Coverage':>10}")
    print("-" * 70)
    for device in DEVICES:
        for channel in CHANNELS:
            sub = sweep_df[(sweep_df["device"] == device) & (sweep_df["channel"] == channel)]
            if sub.empty:
                continue
            baseline = sub[sub["gate_type"] == "Optimized_baseline"]
            if baseline.empty:
                continue
            b_mae = baseline.iloc[0]["mae"]
            non_base = sub[(sub["gate_type"] != "Optimized_baseline") & sub["mae"].notna()].copy()
            if non_base.empty:
                continue
            non_base["delta"] = non_base["mae"] - b_mae
            best = non_base.loc[non_base["delta"].idxmin()]
            delta_str = f"{best['delta']:+.2f}" if np.isfinite(best["delta"]) else "—"
            print(f"{device:<10} {channel:<12} {_fmt(b_mae):>9} {_fmt(best['mae']):>9} "
                  f"{delta_str:>8} {_fmt(best['coverage_pct'], '.1f')+'%':>10}")
    print(f"{'='*70}")
    print(f"\n[step3_sqi_sweep] Done. Report: {report_path.name}")


if __name__ == "__main__":
    main()
