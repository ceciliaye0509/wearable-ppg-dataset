"""Create manual-review plots for ECG R-peak placement and RMSSD labels.

The project derives its ECG labels with NeuroKit2's Pan--Tompkins 1985
detector.  A five-minute overview is useful for missed/extra beats, but it is
not enough to judge an 8--32 ms placement error.  This audit therefore writes
one figure per five-minute window with three raw-ECG excerpts.  Each excerpt
overlays the detector's peak with the nearest raw-QRS extremum (a review aid,
not a clinically annotated gold standard).

The default input is the locally available P2 Polar recording.  This local
copy has no matching PPG windows, so the PPG RMSSD/error fields are explicitly
recorded as unavailable rather than being inferred or borrowed from another
participant.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/daily_hrv_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd


WINDOW_SEC = 300.0
DETAIL_SEC = 10.0
DETAIL_CENTERS_SEC = (60.0, 150.0, 240.0)
IBI_MIN_MS = 300.0
IBI_MAX_MS = 2000.0


def _infer_fs(times_ms: np.ndarray) -> float:
    dt = np.diff(np.asarray(times_ms, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not dt.size:
        raise ValueError("ECG timestamps do not contain a positive finite interval")
    return float(1000.0 / np.median(dt))


def _rmssd(intervals_ms: np.ndarray) -> float:
    intervals = np.asarray(intervals_ms, dtype=float)
    if intervals.size < 3:
        return float("nan")
    return float(np.sqrt(np.mean(np.diff(intervals) ** 2)))


def _correct_intervals(intervals_ms: np.ndarray) -> tuple[np.ndarray, float]:
    """Use the project's correction routine when it is importable."""
    try:
        from algorithms.hrv import _correct_ibi_artifacts_with_ratio

        return _correct_ibi_artifacts_with_ratio(intervals_ms, threshold=0.20)
    except ImportError:  # Allows this single script to run outside package mode.
        return np.asarray(intervals_ms, dtype=float), 0.0


def _detect(seg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    clean = nk.ecg_clean(seg, sampling_rate=fs, method="pantompkins1985")
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="pantompkins1985")
    peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64)
    return clean, peaks[(peaks >= 0) & (peaks < seg.size)]


def _raw_qrs_extrema(raw: np.ndarray, peaks: np.ndarray, fs: float) -> np.ndarray:
    """Find each detected beat's closest raw-QRS apex within +/-80 ms.

    Pan--Tompkins positions are mapped back onto the raw waveform.  The
    extremum has the detector-inferred R polarity and gives reviewers a clear
    visual anchor for sample-level offset checks.
    """
    if not peaks.size:
        return np.empty(0, dtype=np.int64)
    baseline = float(np.median(raw))
    polarity = 1 if np.median(raw[peaks] - baseline) >= 0 else -1
    radius = max(1, int(round(0.080 * fs)))
    out = np.empty(peaks.size, dtype=np.int64)
    for i, peak in enumerate(peaks):
        lo, hi = max(0, int(peak) - radius), min(raw.size, int(peak) + radius + 1)
        chunk = raw[lo:hi]
        local = int(np.argmax(chunk) if polarity > 0 else np.argmin(chunk))
        out[i] = lo + local
    return out


def _read_activity_events(path: Path) -> list[tuple[float, str]]:
    if not path.exists():
        return []
    events: list[tuple[float, str]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                events.append((float(row["Unix Timestamp"]), row[" Activity"].strip()))
            except (KeyError, TypeError, ValueError):
                continue
    return events


def _candidate_window_starts(times_ms: np.ndarray, n_windows: int) -> np.ndarray:
    """Choose evenly distributed starts that remain within uninterrupted ECG runs.

    The recording spans multiple days.  It is not valid to use a simple global
    linspace because that can land in an inter-session gap with no ECG.
    """
    gap_ms = 250.0
    breaks = np.flatnonzero(np.diff(times_ms) > gap_ms) + 1
    bounds = np.r_[0, breaks, times_ms.size]
    candidates: list[float] = []
    stride_ms = 30_000.0
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        run_start, run_end = float(times_ms[lo]), float(times_ms[hi - 1])
        latest_start = run_end - WINDOW_SEC * 1000.0
        if latest_start < run_start:
            continue
        candidates.extend(np.arange(run_start, latest_start + 1.0, stride_ms).tolist())
        if not candidates or candidates[-1] != latest_start:
            candidates.append(latest_start)
    if len(candidates) < n_windows:
        raise ValueError(f"only {len(candidates)} uninterrupted {WINDOW_SEC:.0f}-s ECG windows are available")
    candidate_array = np.asarray(candidates, dtype=float)
    choose = np.linspace(0, candidate_array.size - 1, n_windows).round().astype(int)
    return candidate_array[choose]


def _events_in_window(events: list[tuple[float, str]], start_ms: float, end_ms: float) -> str:
    matched = [text for timestamp, text in events if start_ms <= timestamp < end_ms]
    return "; ".join(matched) if matched else "no logged transition"


def _plot_window(
    participant: str,
    raw: np.ndarray,
    absolute_times: np.ndarray,
    fs: float,
    start_ms: float,
    end_ms: float,
    rank: int,
    events: list[tuple[float, str]],
    out_path: Path,
) -> dict[str, object]:
    lo = int(np.searchsorted(absolute_times, start_ms, side="left"))
    hi = int(np.searchsorted(absolute_times, end_ms, side="right"))
    signal = raw[lo:hi]
    times = absolute_times[lo:hi]
    if signal.size < 50:
        raise ValueError(f"window {rank} has too few samples: {signal.size}")

    clean, peaks = _detect(signal, fs)
    raw_extrema = _raw_qrs_extrema(signal, peaks, fs)
    offsets_ms = (raw_extrema - peaks) * 1000.0 / fs
    peak_times = times[peaks]
    rr_ms = np.diff(peak_times)
    valid_rr = rr_ms[(rr_ms >= IBI_MIN_MS) & (rr_ms <= IBI_MAX_MS)]
    corrected_rr, correction_ratio = _correct_intervals(valid_rr)
    rmssd_raw = _rmssd(valid_rr)
    rmssd_corrected = _rmssd(corrected_rr)
    sample_ms = 1000.0 / fs
    signed_error_ms = np.nan
    ppg_rmssd_ms = np.nan

    fig = plt.figure(figsize=(16, 15), constrained_layout=True)
    grid = fig.add_gridspec(5, 1, height_ratios=(1.3, 0.9, 1.0, 1.0, 1.0))
    ax_overview = fig.add_subplot(grid[0])
    ax_rr = fig.add_subplot(grid[1])
    detail_axes = [fig.add_subplot(grid[i]) for i in range(2, 5)]

    t_sec = (times - start_ms) / 1000.0
    ax_overview.plot(t_sec, signal, color="#343a40", linewidth=0.45, label="raw ECG (Polar)")
    if peaks.size:
        ax_overview.scatter(t_sec[peaks], signal[peaks], s=9, color="#d62728", zorder=3, label="Pan--Tompkins R peak")
    ax_overview.set_xlim(0, WINDOW_SEC)
    ax_overview.set_ylabel("ECG (µV)")
    ax_overview.legend(loc="upper right", ncols=2)
    ax_overview.grid(alpha=0.2)
    ax_overview.set_title(
        f"#{rank:02d} 5-min ECG label audit | PT-1985 R peaks={peaks.size} | "
        f"ECG RMSSD={rmssd_corrected:.1f} ms (corrected; raw={rmssd_raw:.1f} ms)"
    )

    if rr_ms.size:
        ax_rr.plot(t_sec[peaks[1:]], rr_ms, color="#2ca02c", linewidth=0.8, label="detected RR")
    if corrected_rr.size:
        x_corr = t_sec[peaks[1 : 1 + corrected_rr.size]]
        ax_rr.plot(x_corr, corrected_rr, color="#ff7f0e", linewidth=1.0, alpha=0.85, label="RR used for ECG RMSSD")
    ax_rr.axhspan(IBI_MIN_MS, IBI_MAX_MS, color="#2ca02c", alpha=0.05, label="physiologic IBI range")
    ax_rr.set_xlim(0, WINDOW_SEC)
    ax_rr.set_ylabel("RR (ms)")
    ax_rr.set_xlabel("seconds in 5-min window")
    ax_rr.grid(alpha=0.2)
    ax_rr.legend(loc="upper right", ncols=3)

    for axis, center in zip(detail_axes, DETAIL_CENTERS_SEC, strict=True):
        d0 = max(0.0, center - DETAIL_SEC / 2.0)
        d1 = min(WINDOW_SEC, center + DETAIL_SEC / 2.0)
        select = (t_sec >= d0) & (t_sec <= d1)
        axis.plot(t_sec[select], signal[select], color="#343a40", linewidth=1.0, label="raw ECG")
        local_peaks = peaks[(t_sec[peaks] >= d0) & (t_sec[peaks] <= d1)]
        local_extrema = raw_extrema[(t_sec[peaks] >= d0) & (t_sec[peaks] <= d1)]
        if local_peaks.size:
            axis.scatter(t_sec[local_peaks], signal[local_peaks], marker="v", s=34, color="#d62728", zorder=5, label="detected location")
            axis.scatter(t_sec[local_extrema], signal[local_extrema], marker="x", s=34, linewidths=1.3, color="#1f77b4", zorder=6, label="nearest raw-QRS extremum")
        axis.set_xlim(d0, d1)
        axis.set_ylabel("ECG (µV)")
        axis.grid(alpha=0.25)
        axis.set_title(f"Raw-ECG placement check: {d0:.0f}–{d1:.0f} s")
    detail_axes[-1].set_xlabel("seconds in 5-min window")
    detail_axes[0].legend(loc="upper right", ncols=3)

    fig.text(
        0.01,
        0.002,
        "PPG RMSSD / PPG−ECG error: N/A (no synchronized PPG window is present locally). "
        "Blue × is a raw-waveform audit reference, not clinical ground truth. "
        f"Raw-QRS offset: median={np.median(offsets_ms):+.1f} ms; "
        f"p95 |offset|={np.quantile(np.abs(offsets_ms), .95):.1f} ms; sample period={sample_ms:.1f} ms.",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "participant": participant,
        "rank": rank,
        "window_start_unix_ms": round(start_ms, 3),
        "window_end_unix_ms": round(end_ms, 3),
        "n_detected_rpeaks": int(peaks.size),
        "ecg_rmssd_corrected_ms": rmssd_corrected,
        "ecg_rmssd_raw_valid_rr_ms": rmssd_raw,
        "ecg_ibi_correction_ratio": correction_ratio,
        "ppg_rmssd_ms": ppg_rmssd_ms,
        "ppg_minus_ecg_rmssd_ms": signed_error_ms,
        "raw_qrs_offset_median_ms": float(np.median(offsets_ms)) if offsets_ms.size else np.nan,
        "raw_qrs_offset_p95_abs_ms": float(np.quantile(np.abs(offsets_ms), 0.95)) if offsets_ms.size else np.nan,
        "raw_qrs_offset_max_abs_ms": float(np.max(np.abs(offsets_ms))) if offsets_ms.size else np.nan,
        "offset_within_one_sample_pct": float(100 * np.mean(np.abs(offsets_ms) <= sample_ms + 1e-9)) if offsets_ms.size else np.nan,
        "offset_within_two_samples_pct": float(100 * np.mean(np.abs(offsets_ms) <= 2 * sample_ms + 1e-9)) if offsets_ms.size else np.nan,
        "activity_log_transitions": _events_in_window(events, start_ms, end_ms),
        "plot_path": str(out_path),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-ecg", type=Path, default=here.parents[2] / "Multisite-PPG/raw_data/P2/P2_polar_ecg_raw.npz")
    parser.add_argument("--activity-log", type=Path, default=here.parents[2] / "Multisite-PPG/raw_data/P2/P2_activity_log.txt")
    parser.add_argument("--out-dir", type=Path, default=here / "outputs/ecg_rpeak_alignment_audit_p2")
    parser.add_argument("--n-windows", type=int, default=12)
    args = parser.parse_args()
    if not 10 <= args.n_windows <= 20:
        raise ValueError("--n-windows must be between 10 and 20 for a manual audit")

    with np.load(args.raw_ecg, allow_pickle=True) as z:
        raw = np.asarray(z["ecg_uv"], dtype=float)
        absolute_times = np.asarray(z["time_ms"], dtype=float)
    fs = _infer_fs(absolute_times)
    participant = args.raw_ecg.parent.name
    starts = _candidate_window_starts(absolute_times, args.n_windows)
    events = _read_activity_events(args.activity_log)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rank, start_ms in enumerate(starts, start=1):
        plot_path = args.out_dir / f"{participant.lower()}_ecg_audit_{rank:02d}.png"
        rows.append(_plot_window(participant, raw, absolute_times, fs, start_ms, start_ms + WINDOW_SEC * 1000.0, rank, events, plot_path))
    index = pd.DataFrame(rows)
    index.to_csv(args.out_dir / "audit_index.csv", index=False)
    summary = {
        "raw_ecg": str(args.raw_ecg),
        "sampling_rate_hz": fs,
        "window_seconds": WINDOW_SEC,
        "n_windows": args.n_windows,
        "detector": "NeuroKit2 ecg_clean/ecg_peaks, pantompkins1985",
        "ppg_status": "not locally available for P2; fields deliberately N/A",
        "median_of_window_median_raw_qrs_offset_ms": float(index["raw_qrs_offset_median_ms"].median()),
        "worst_window_p95_raw_qrs_offset_ms": float(index["raw_qrs_offset_p95_abs_ms"].max()),
    }
    (args.out_dir / "audit_summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")
    print(index[["rank", "n_detected_rpeaks", "ecg_rmssd_corrected_ms", "raw_qrs_offset_median_ms", "raw_qrs_offset_p95_abs_ms", "plot_path"]].to_string(index=False))


if __name__ == "__main__":
    main()
