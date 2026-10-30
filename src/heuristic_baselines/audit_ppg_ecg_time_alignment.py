"""Audit PPG--ECG timing without modifying any training label.

The audit is deliberately diagnostic rather than a label-generation step.  It
uses the v3 ECG sidecar's QRS times and the raw Earring green PPG waveform,
then writes a fixed-seed stratified sample of 18 visual reviews plus four
quantitative checks:

* one-to-one PPG peak minus ECG R-peak delays and robust within-window drift;
* the common raw-support interval, excluding the first and last detected beat;
* raw and 20%-median-corrected RMSSD recomputed on that common interval;
* a PPG-source-window shift sensitivity analysis (-2 to +2 seconds).

The shift analysis is explicitly an edge/membership sensitivity test.  A
constant time shift cannot by itself change IBI/RMSSD away from window edges,
and it cannot distinguish a clock offset from a physiological PTT.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from algorithms import hrv, qppgfast  # noqa: E402
from preprocess import bandpass_filter  # noqa: E402


WINDOW_MS = 300_000.0
FS = 100.0
GRID_STEP_MS = 1000.0 / FS
STRICT_MAX_GAP_MS = 100.0
IBI_MIN_MS, IBI_MAX_MS = 300.0, 2000.0
MATCH_MIN_MS, MATCH_MAX_MS = 50.0, 800.0
SHIFTS_MS = np.arange(-2000.0, 2000.1, 500.0)
RNG_SEED = 20260925


def _rmssd(rr_ms: np.ndarray) -> float:
    rr = np.asarray(rr_ms, dtype=float)
    return float(np.sqrt(np.mean(np.diff(rr) ** 2))) if rr.size >= 3 else float("nan")


def _hrv_metrics(event_times_ms: np.ndarray) -> tuple[float, float, float]:
    """Return raw RMSSD, corrected RMSSD, and correction ratio."""
    event_times_ms = np.asarray(event_times_ms, dtype=float)
    ibi = np.diff(event_times_ms)
    valid = ibi[(ibi >= IBI_MIN_MS) & (ibi <= IBI_MAX_MS)]
    raw = _rmssd(valid)
    if valid.size < 3:
        return raw, float("nan"), float("nan")
    corrected, ratio = hrv._correct_ibi_artifacts_with_ratio(valid, threshold=0.20)
    return raw, _rmssd(corrected), float(ratio)


def _interior_events(event_times_ms: np.ndarray, start_ms: float, end_ms: float) -> np.ndarray:
    """Keep complete interior beats rather than beats that touch either edge."""
    events = np.asarray(event_times_ms, dtype=float)
    events = events[(events >= start_ms) & (events <= end_ms)]
    return events[1:-1] if events.size >= 3 else np.empty(0, dtype=float)


def _match_events(r_times_ms: np.ndarray, ppg_times_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily make one-to-one physiological forward matches."""
    r = np.asarray(r_times_ms, dtype=float)
    p = np.asarray(ppg_times_ms, dtype=float)
    ri: list[int] = []
    pi: list[int] = []
    cursor = 0
    for i, r_time in enumerate(r):
        while cursor < p.size and p[cursor] < r_time + MATCH_MIN_MS:
            cursor += 1
        if cursor < p.size and p[cursor] <= r_time + MATCH_MAX_MS:
            ri.append(i)
            pi.append(cursor)
            cursor += 1
    ri_array = np.asarray(ri, dtype=int)
    pi_array = np.asarray(pi, dtype=int)
    return ri_array, pi_array, p[pi_array] - r[ri_array]


def _delay_summary(r_times_ms: np.ndarray, ppg_times_ms: np.ndarray) -> dict[str, float]:
    ri, pi, delay = _match_events(r_times_ms, ppg_times_ms)
    out = {
        "matched_beats": int(delay.size),
        "ecg_r_peaks": int(len(r_times_ms)),
        "ppg_peaks": int(len(ppg_times_ms)),
        "match_rate_pct": float(100 * delay.size / len(r_times_ms)) if len(r_times_ms) else float("nan"),
        "delay_median_ms": float(np.median(delay)) if delay.size else float("nan"),
        "delay_iqr_ms": float(np.subtract(*np.percentile(delay, [75, 25]))) if delay.size >= 2 else float("nan"),
        "delay_slope_ms_per_min": float("nan"),
    }
    if delay.size >= 5:
        slope, _intercept, _low, _high = theilslopes(delay, r_times_ms[ri] / 60_000.0)
        out["delay_slope_ms_per_min"] = float(slope)
    return out


def _extract_raw_ppg(raw: dict[str, np.ndarray], start_ms: float, end_ms: float) -> tuple[np.ndarray, np.ndarray, float]:
    t = np.asarray(raw["timestamp"], dtype=float)
    x = np.asarray(raw["ppg_green"], dtype=float)
    lo = int(np.searchsorted(t, start_ms, side="left"))
    hi = int(np.searchsorted(t, end_ms, side="right"))
    t, x = t[lo:hi], x[lo:hi]
    finite = np.isfinite(t) & np.isfinite(x)
    t, x = t[finite], x[finite]
    if t.size < 2:
        return t, x, float("inf")
    order = np.argsort(t)
    t, x = t[order], x[order]
    unique = np.r_[True, np.diff(t) > 0]
    t, x = t[unique], x[unique]
    if t.size < 2:
        return t, x, float("inf")
    gap = max(float(t[0] - start_ms), float(end_ms - t[-1]), float(np.max(np.diff(t))))
    return t, x, gap


def _strict_ppg_peaks(raw: dict[str, np.ndarray], start_ms: float, end_ms: float) -> dict[str, object]:
    raw_t, raw_x, max_gap = _extract_raw_ppg(raw, start_ms, end_ms)
    grid = start_ms + np.arange(int(round((end_ms - start_ms) / GRID_STEP_MS)), dtype=float) * GRID_STEP_MS
    result: dict[str, object] = {"raw_t": raw_t, "raw_x": raw_x, "grid_t": grid, "max_gap_ms": max_gap,
                                 "strict_pass": bool(np.isfinite(max_gap) and max_gap <= STRICT_MAX_GAP_MS),
                                 "signal": np.empty(0), "peaks_t": np.empty(0), "feet_t": np.empty(0),
                                 "polarity": "invalid"}
    if not result["strict_pass"] or raw_t.size < 2 or grid.size < 50:
        return result
    x = np.interp(grid, raw_t, raw_x)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filt = bandpass_filter(x, 0.7, 3.5, FS)
        centered = filt - np.median(filt)
        # A waveform-only orientation rule: no ECG, IBI, peak, or foot input.
        polarity = "positive" if float(np.mean((centered / (np.std(centered) + 1e-12)) ** 3)) >= 0 else "negative"
        oriented = filt if polarity == "positive" else -filt
        peaks, feet = qppgfast._qppgfast_beat_detector(np.array(oriented, dtype=float, copy=True), FS)
        peaks = np.asarray(peaks, dtype=int)
        feet = np.asarray(feet, dtype=int)
        result.update(signal=oriented, peaks_t=grid[peaks[(peaks >= 0) & (peaks < grid.size)]],
                      feet_t=grid[feet[(feet >= 0) & (feet < grid.size)]], polarity=polarity)
    except Exception as error:  # One malformed waveform must remain an audit row.
        result["detector_error"] = type(error).__name__
    return result


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size < 2:
        return np.full_like(x, np.nan)
    scale = float(np.percentile(finite, 95) - np.percentile(finite, 5))
    return (x - np.median(finite)) / (scale if scale > 1e-12 else 1.0)


def _plot(row: dict[str, object], ppg: dict[str, object], ecg_raw: dict[str, np.ndarray], out_path: Path) -> None:
    start, end = float(row["t0_ms"]), float(row["t1_ms"])
    ecg_t = np.asarray(ecg_raw["time_ms"], dtype=float)
    ecg_x = np.asarray(ecg_raw["ecg_uv"], dtype=float)
    lo = int(np.searchsorted(ecg_t, start, side="left"))
    hi = int(np.searchsorted(ecg_t, end, side="right"))
    ecg_t, ecg_x = ecg_t[lo:hi], ecg_x[lo:hi]
    r_abs = np.asarray(row["r_abs_ms"], dtype=float)
    p_abs = np.asarray(ppg["peaks_t"], dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(16, 13), constrained_layout=True,
                             gridspec_kw={"height_ratios": [0.7, 1, 1, 1]})
    overview = axes[0]
    overview.eventplot([(r_abs - start) / 1000.0, (p_abs - start) / 1000.0],
                       lineoffsets=[1, 0], linelengths=0.75, colors=["#c73e2a", "#6a3d9a"])
    overview.set_yticks([0, 1], ["PPG peak", "ECG R"])
    overview.set_xlim(0, (end - start) / 1000.0)
    overview.grid(alpha=0.2)
    overview.set_title(
        f"{row['sample_group']} | {row['participant']} #{int(row['window_index'])} | "
        f"{row['ppg_device']} green strict={row['strict_ppg_pass']} | match={row['match_rate_pct']:.1f}% | "
        f"PTT median/IQR={row['delay_median_ms']:.0f}/{row['delay_iqr_ms']:.0f} ms"
    )
    centers = (60_000.0, 150_000.0, 240_000.0)
    raw_t, raw_x = np.asarray(ppg["raw_t"], float), np.asarray(ppg["raw_x"], float)
    for ax, center in zip(axes[1:], centers, strict=True):
        left, right = start + center - 7_500.0, start + center + 7_500.0
        em = (ecg_t >= left) & (ecg_t <= right)
        pm = (raw_t >= left) & (raw_t <= right)
        ax.plot((ecg_t[em] - start) / 1000.0, _normalise(ecg_x[em]) + 1.2, color="#263746", lw=0.7, label="raw ECG (normalised)")
        ax.plot((raw_t[pm] - start) / 1000.0, _normalise(raw_x[pm]) - 1.2, color="#536d8d", lw=0.7, label=f"raw PPG ({row['ppg_device']} green; normalised)")
        rr = r_abs[(r_abs >= left) & (r_abs <= right)]
        pp = p_abs[(p_abs >= left) & (p_abs <= right)]
        ax.scatter((rr - start) / 1000.0, np.full(rr.size, 1.75), s=20, color="#c73e2a", zorder=5, label="ECG R")
        ax.scatter((pp - start) / 1000.0, np.full(pp.size, -0.65), s=20, color="#6a3d9a", zorder=5, label="PPG peak")
        ax.set_xlim((left - start) / 1000.0, (right - start) / 1000.0)
        ax.set_ylim(-2.2, 2.2)
        ax.grid(alpha=0.2)
    axes[1].legend(loc="upper right", ncols=4, fontsize=8)
    axes[-1].set_xlabel("seconds in nominal ECG window")
    fig.text(0.01, 0.002,
             "Red: Emrich2023 + local-QRS ECG candidate. Purple: qppgfast systolic PPG peak. "
             "A missing purple series means the raw PPG failed the pre-specified 100 ms support-gap criterion.", fontsize=9)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _stratified_selection(meta: pd.DataFrame, n_windows: int) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    categories = [
        ("clean_random", 8, meta[meta.qc_pass & ~meta.review_required & (meta.max_edge_ms <= 10.0)]),
        ("edge_gap", 5, meta[meta.max_edge_ms >= 100.0]),
        ("ecg_review", 5, meta[(meta.review_required | ~meta.qc_pass) & (meta.max_edge_ms < 100.0)]),
    ]
    chosen: list[pd.DataFrame] = []
    used: set[tuple[str, int]] = set()
    for name, desired, pool in categories:
        pool = pool[~pool.apply(lambda r: (str(r.participant), int(r.window_index)) in used, axis=1)]
        if pool.empty:
            continue
        # Random within participant, then round-robin across participants.
        pieces = [g.sample(n=min(1, len(g)), random_state=int(rng.integers(1, 2**31 - 1)))
                  for _pid, g in pool.groupby("participant", sort=True)]
        candidate = pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=int(rng.integers(1, 2**31 - 1)))
        take = candidate.head(desired).copy()
        take["sample_group"] = name
        chosen.append(take)
        used.update((str(r.participant), int(r.window_index)) for r in take.itertuples())
    result = pd.concat(chosen, ignore_index=True)
    if len(result) < n_windows:
        remaining = meta[~meta.apply(lambda r: (str(r.participant), int(r.window_index)) in used, axis=1)]
        fill = remaining.sample(n=min(n_windows - len(result), len(remaining)), random_state=RNG_SEED).copy()
        fill["sample_group"] = "random_fill"
        result = pd.concat([result, fill], ignore_index=True)
    return result.head(n_windows).sort_values(["sample_group", "participant", "window_index"]).reset_index(drop=True)


def _metadata(windows_dir: Path, sidecar_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    peaks: dict[str, np.ndarray] = {}
    for window_path in sorted(windows_dir.glob("*.npz")):
        pid = window_path.stem.rsplit("_", 1)[-1]
        sidecar_path = sidecar_dir / f"ecg_ground_truth_v3_candidate_emrich_local_qrs_pan_ungated_{pid}.npz"
        if not sidecar_path.exists():
            continue
        with np.load(window_path, allow_pickle=True) as z:
            t0 = np.asarray(z["ppg_window_t0_ms"], dtype=float)
            t1 = np.asarray(z["ppg_window_t1_ms"], dtype=float)
            edge = np.maximum(np.asarray(z["max_start_diff_ms"], dtype=float), np.asarray(z["max_end_diff_ms"], dtype=float))
        with np.load(sidecar_path, allow_pickle=True) as z:
            qc = np.asarray(z["ecg_v3_qc_pass"], dtype=bool)
            review = np.asarray(z["ecg_review_required"], dtype=bool)
            reason = np.asarray(z["ecg_v3_qc_reason"], dtype=object)
            peak_array = np.asarray(z["ecg_r_peak_times_rel_ms"], dtype=object)
        for i in range(len(t0)):
            key = f"{pid}:{i}"
            peaks[key] = np.asarray(peak_array[i], dtype=float) + t0[i]
            rows.append({"participant": pid, "window_index": i, "t0_ms": t0[i], "t1_ms": t1[i],
                         "max_edge_ms": edge[i], "qc_pass": qc[i], "review_required": review[i], "qc_reason": str(reason[i])})
    return pd.DataFrame(rows), peaks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[mask], y[mask])[0, 1]) if mask.sum() >= 3 and np.std(x[mask]) and np.std(y[mask]) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("Multisite-PPG/raw_data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-windows", type=int, default=18)
    parser.add_argument("--ppg-device", type=str, default="Earring", help="Raw PPG device file stem, e.g. Earring, Ring, Watch")
    args = parser.parse_args()
    if not 10 <= args.n_windows <= 20:
        raise ValueError("--n-windows must be 10--20")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta, ecg_peaks = _metadata(args.windows_dir, args.sidecar_dir)
    selected = _stratified_selection(meta, args.n_windows)
    selected.to_csv(args.out_dir / "selected_windows.csv", index=False)
    rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    shift_rows: list[dict[str, object]] = []
    raw_cache: dict[str, dict[str, np.ndarray]] = {}
    ecg_cache: dict[str, dict[str, np.ndarray]] = {}
    for rank, source in selected.iterrows():
        row = source.to_dict()
        pid, wi = str(row["participant"]), int(row["window_index"])
        if pid not in raw_cache:
            with np.load(args.raw_root / pid / f"{pid}_{args.ppg_device}_raw.npz", allow_pickle=True) as z:
                raw_cache[pid] = {k: np.asarray(z[k]) for k in ("timestamp", "ppg_green", "ecg_offset_ms_used")}
            with np.load(args.raw_root / pid / f"{pid}_polar_ecg_raw.npz", allow_pickle=True) as z:
                ecg_cache[pid] = {k: np.asarray(z[k]) for k in ("time_ms", "ecg_uv")}
        raw = raw_cache[pid]
        start, end = float(row["t0_ms"]), float(row["t1_ms"])
        r_abs = ecg_peaks[f"{pid}:{wi}"]
        row["r_abs_ms"] = r_abs
        ppg = _strict_ppg_peaks(raw, start, end)
        p_abs = np.asarray(ppg["peaks_t"], dtype=float)
        delay = _delay_summary(r_abs, p_abs)
        raw_t = np.asarray(ppg["raw_t"], dtype=float)
        common_start = max(start, float(raw_t[0])) if raw_t.size else float("nan")
        common_end = min(end, float(raw_t[-1])) if raw_t.size else float("nan")
        ecg_interior = _interior_events(r_abs, common_start, common_end) if np.isfinite(common_start + common_end) else np.empty(0)
        ppg_interior = _interior_events(p_abs, common_start, common_end) if np.isfinite(common_start + common_end) else np.empty(0)
        ecg_raw, ecg_corr, ecg_corr_ratio = _hrv_metrics(ecg_interior)
        ppg_raw, ppg_corr, ppg_corr_ratio = _hrv_metrics(ppg_interior)
        ri, pi, delays = _match_events(r_abs, p_abs)
        for r_index, p_index, one_delay in zip(ri, pi, delays, strict=True):
            match_rows.append({"participant": pid, "window_index": wi, "ecg_r_time_ms": r_abs[r_index],
                               "ppg_peak_time_ms": p_abs[p_index], "ppg_minus_ecg_ms": one_delay})
        base = {k: v for k, v in row.items() if k != "r_abs_ms"}
        base["ppg_device"] = args.ppg_device
        out = {**base, **delay, "rank": int(rank + 1), "strict_ppg_pass": bool(ppg["strict_pass"]),
               "ppg_max_support_gap_ms": float(ppg["max_gap_ms"]), "ppg_polarity": ppg["polarity"],
               "raw_ecg_offset_ms_used": float(np.asarray(raw["ecg_offset_ms_used"]).item()),
               "common_start_offset_ms": common_start - start if np.isfinite(common_start) else np.nan,
               "common_end_offset_ms": end - common_end if np.isfinite(common_end) else np.nan,
               "ecg_rmssd_raw_common_ms": ecg_raw, "ecg_rmssd_corrected_common_ms": ecg_corr,
               "ecg_ibi_correction_ratio_common": ecg_corr_ratio, "ppg_rmssd_raw_common_ms": ppg_raw,
               "ppg_rmssd_corrected_common_ms": ppg_corr, "ppg_ibi_correction_ratio_common": ppg_corr_ratio,
               "rmssd_corrected_error_common_ms": ppg_corr - ecg_corr if np.isfinite(ppg_corr) and np.isfinite(ecg_corr) else np.nan}
        rows.append(out)
        for shift in SHIFTS_MS:
            shifted = _strict_ppg_peaks(raw, start + shift, end + shift)
            _raw, corr, corr_ratio = _hrv_metrics(_interior_events(np.asarray(shifted["peaks_t"], float), start + shift, end + shift))
            shift_rows.append({"participant": pid, "window_index": wi, "shift_ms": shift,
                               "strict_ppg_pass": bool(shifted["strict_pass"]), "ppg_max_support_gap_ms": float(shifted["max_gap_ms"]),
                               "ppg_rmssd_corrected_ms": corr, "ppg_ibi_correction_ratio": corr_ratio,
                               "ecg_rmssd_corrected_nominal_common_ms": ecg_corr,
                               "signed_error_ms": corr - ecg_corr if np.isfinite(corr) and np.isfinite(ecg_corr) else np.nan})
        _plot(out | {"r_abs_ms": r_abs}, ppg, ecg_cache[pid], args.out_dir / f"{rank + 1:02d}_{pid}_w{wi:04d}.png")
    result = pd.DataFrame(rows)
    shifts = pd.DataFrame(shift_rows)
    result.to_csv(args.out_dir / "window_alignment_metrics.csv", index=False)
    pd.DataFrame(match_rows).to_csv(args.out_dir / "matched_delays.csv", index=False)
    shifts.to_csv(args.out_dir / "shift_sensitivity_by_window.csv", index=False)
    summary_rows = []
    for shift, group in shifts.groupby("shift_ms", sort=True):
        err = group.signed_error_ms.to_numpy(float)
        ppg = group.ppg_rmssd_corrected_ms.to_numpy(float)
        ecg = group.ecg_rmssd_corrected_nominal_common_ms.to_numpy(float)
        summary_rows.append({"shift_ms": shift, "n_valid": int(np.isfinite(err).sum()), "mae_ms": float(np.nanmean(np.abs(err))),
                             "pearson_r": _pearson(ppg, ecg)})
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "shift_sensitivity_summary.csv", index=False)
    summary = {
        "audit": "stratified 18-window diagnostic only; no label was changed",
        "n_windows": int(len(result)), "rng_seed": RNG_SEED, "ppg_device_channel": f"{args.ppg_device}/ppg_green",
        "ppg_detector": "qppgfast peak; waveform-only skewness polarity; 0.7--3.5 Hz bandpass",
        "strict_ppg_support_gap_ms": STRICT_MAX_GAP_MS,
        "ptt_match_range_ms": [MATCH_MIN_MS, MATCH_MAX_MS],
        "raw_ecg_offset_ms_used_unique": sorted(result.raw_ecg_offset_ms_used.dropna().unique().tolist()),
        "strict_ppg_pass_windows": int(result.strict_ppg_pass.sum()),
        "matched_delay_median_ms": float(result.delay_median_ms.median()),
        "matched_delay_iqr_median_ms": float(result.delay_iqr_ms.median()),
        "delay_slope_median_ms_per_min": float(result.delay_slope_ms_per_min.median()),
        "common_rmssd_mae_ms": float(np.nanmean(np.abs(result.rmssd_corrected_error_common_ms))),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "# PPG--ECG timing audit\n\n"
        "This is a fixed-seed, stratified 18-window diagnostic. It does **not** modify ECG labels or select a model. "
        f"Each PNG shows raw ECG and raw {args.ppg_device}-green PPG in three 15-s excerpts, with ECG R events and qppgfast PPG peaks. "
        "`matched_delays.csv` contains one-to-one forward matches in the pre-specified 50--800 ms interval. "
        "`window_alignment_metrics.csv` recomputes raw/corrected RMSSD after intersecting raw ECG/PPG support and dropping first/last events. "
        "`shift_sensitivity_*` shifts the *source PPG interval* while holding ECG fixed; it tests edge membership sensitivity, not causal clock-offset correction.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
