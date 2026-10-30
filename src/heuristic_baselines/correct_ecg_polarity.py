"""Detect and apply high-confidence polarity corrections to raw Polar ECG.

Raw source NPZ files are never modified.  This writes an interval correction
manifest; downstream ECG loaders multiply ``ecg_uv`` by -1 only inside
high-confidence inverted intervals.  Ambiguous intervals stay unchanged and
are flagged for review.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_RAW_ROOT = REPO_ROOT.parent / "Multisite-PPG" / "raw_data"
DEFAULT_OUT_DIR = PACKAGE_ROOT / "outputs" / "ecg_polarity_correction_v1"


@dataclass
class Block:
    participant: str
    run_index: int
    block_index: int
    start_ms: float
    end_ms: float
    sample_count: int
    fs_hz: float
    raw_p01_p99_span_uv: float
    q005_uv: float
    q995_uv: float
    polarity_score: float
    event_count: int
    raw_polarity: str
    decision: str = "review"


def _infer_fs(times_ms: np.ndarray) -> float:
    delta = np.diff(np.asarray(times_ms, dtype=np.float64))
    delta = delta[np.isfinite(delta) & (delta > 0)]
    if not delta.size:
        return float("nan")
    return float(1000.0 / np.median(delta))


def _split_runs(times_ms: np.ndarray) -> list[tuple[int, int]]:
    if not times_ms.size:
        return []
    gap = np.diff(times_ms)
    # Split on actual time gaps and clock resets. Duplicate timestamps remain
    # within a run and are collapsed before filter-based measurements.
    breaks = np.flatnonzero((gap > 250.0) | (gap < 0.0)) + 1
    bounds = np.r_[0, breaks, times_ms.size]
    return [(int(left), int(right)) for left, right in zip(bounds[:-1], bounds[1:], strict=True) if right - left >= 2]


def _block_measurement(signal: np.ndarray, times_ms: np.ndarray, fs_hz: float) -> tuple[float, float, float, float, int, str]:
    """Return dominant event-QRS sign consensus and a coarse QRS-event count."""
    if signal.size < max(100, int(round(10.0 * fs_hz))) or not (60.0 <= fs_hz <= 1000.0):
        return float("nan"), float("nan"), float("nan"), float("nan"), 0, "low_signal"
    raw_q01, raw_q99 = np.quantile(signal, [0.01, 0.99])
    raw_span = float(raw_q99 - raw_q01)
    # Reuse the severe-amplitude-artifact boundary already used by the ECG
    # review workflow.  Such clips do not provide trustworthy QRS polarity.
    if raw_span > 5_000.0:
        return float("nan"), float("nan"), raw_span, float("nan"), 0, "severe_artifact"
    high = min(25.0, 0.45 * fs_hz)
    if high <= 5.0:
        return float("nan"), float("nan"), raw_span, float("nan"), 0, "low_signal"
    try:
        sos = butter(3, [5.0, high], btype="bandpass", fs=fs_hz, output="sos")
        filtered = sosfiltfilt(sos, signal.astype(np.float64, copy=False))
    except (ValueError, FloatingPointError):
        return float("nan"), float("nan"), raw_span, float("nan"), 0, "low_signal"
    if not np.isfinite(filtered).all():
        return float("nan"), float("nan"), raw_span, float("nan"), 0, "low_signal"
    q005, q995 = np.quantile(filtered, [0.005, 0.995])
    total = float(q995 - q005)
    if not np.isfinite(total) or total <= 1e-9:
        return float(q005), float(q995), raw_span, float("nan"), 0, "low_signal"
    robust_noise = float(np.median(np.abs(filtered - np.median(filtered))) * 1.4826)
    min_prominence = max(2.5 * robust_noise, 1e-9)
    events, _ = find_peaks(
        np.abs(filtered),
        distance=max(1, int(round(0.35 * fs_hz))),
        prominence=min_prominence,
    )
    required_events = max(8, int(round(0.20 * (times_ms[-1] - times_ms[0]) / 1000.0)))
    if len(events) < required_events or total < 3.0 * robust_noise:
        return float(q005), float(q995), raw_span, float("nan"), int(len(events)), "low_signal"
    event_signs = np.sign(filtered[events])
    positive_votes = int(np.count_nonzero(event_signs > 0.0))
    negative_votes = int(np.count_nonzero(event_signs < 0.0))
    vote_total = positive_votes + negative_votes
    if vote_total == 0:
        return float(q005), float(q995), raw_span, float("nan"), int(len(events)), "low_signal"
    score = float((positive_votes - negative_votes) / vote_total)
    if score >= 0.80:
        return float(q005), float(q995), raw_span, score, int(len(events)), "positive"
    if score <= -0.80:
        return float(q005), float(q995), raw_span, score, int(len(events)), "negative"
    return float(q005), float(q995), raw_span, score, int(len(events)), "ambiguous"


def _assign_stable_spans(
    run_blocks: list[Block],
    *,
    min_run_blocks: int,
    max_ambiguous_gap_blocks: int,
) -> list[dict[str, Any]]:
    """Accept sustained same-polarity spans; never infer a sign across another sign."""
    spans: list[dict[str, Any]] = []
    position = 0
    while position < len(run_blocks):
        first_label = run_blocks[position].raw_polarity
        if first_label not in {"positive", "negative"}:
            if first_label == "severe_artifact":
                run_blocks[position].decision = "severe_artifact"
            elif first_label == "low_signal":
                run_blocks[position].decision = "review_low_signal"
            else:
                run_blocks[position].decision = "review_ambiguous_polarity"
            position += 1
            continue
        end = position + 1
        ambiguous_gap = 0
        while end < len(run_blocks):
            label = run_blocks[end].raw_polarity
            if label == first_label:
                ambiguous_gap = 0
                end += 1
                continue
            if label in {"ambiguous", "low_signal"} and ambiguous_gap < max_ambiguous_gap_blocks:
                ambiguous_gap += 1
                end += 1
                continue
            break
        # A tentative gap is only accepted if the same sign resumes after it.
        if ambiguous_gap:
            resumes = end < len(run_blocks) and run_blocks[end].raw_polarity == first_label
            if not resumes:
                end -= ambiguous_gap
                ambiguous_gap = 0
        selected = run_blocks[position:end]
        signed = [block for block in selected if block.raw_polarity == first_label]
        if len(signed) >= min_run_blocks:
            for block in selected:
                block.decision = f"stable_{first_label}"
            spans.append({
                "participant": run_blocks[0].participant,
                "run_index": run_blocks[0].run_index,
                "start_ms": float(selected[0].start_ms),
                "end_ms": float(selected[-1].end_ms),
                "orientation": first_label,
                "correction_factor": -1 if first_label == "negative" else 1,
                "classified_blocks": len(signed),
                "span_blocks_including_bridged_gaps": len(selected),
                "median_abs_polarity_score": float(np.median([abs(b.polarity_score) for b in signed if np.isfinite(b.polarity_score)])),
                "median_qrs_event_count_per_block": float(np.median([b.event_count for b in signed])),
                "median_fs_hz": float(np.median([b.fs_hz for b in signed])),
            })
        else:
            for block in selected:
                block.decision = "review_short_polarity_run"
        position = end
    return spans


def _analyze_participant(
    participant: str,
    raw_path: Path,
    out_dir: Path,
    *,
    block_seconds: float,
    min_run_blocks: int,
    max_ambiguous_gap_blocks: int,
) -> tuple[dict[str, Any], list[Block], list[dict[str, Any]]]:
    with np.load(raw_path, allow_pickle=True) as source:
        if "ecg_uv" not in source.files or "time_ms" not in source.files:
            raise KeyError(f"{raw_path} needs ecg_uv and time_ms arrays; keys are {source.files}")
        signal = np.asarray(source["ecg_uv"], dtype=np.float64)
        times = np.asarray(source["time_ms"], dtype=np.float64)
    if signal.ndim != 1 or times.ndim != 1 or signal.shape != times.shape:
        raise ValueError(f"{participant}: expected equal-length one-dimensional ecg_uv and time_ms arrays")
    valid_samples = np.isfinite(signal) & np.isfinite(times)
    nonfinite_samples_excluded = int(signal.size - np.count_nonzero(valid_samples))
    signal, times = signal[valid_samples], times[valid_samples]
    if times.size < 2:
        raise ValueError(f"{participant}: no usable ECG samples")
    blocks: list[Block] = []
    spans: list[dict[str, Any]] = []
    runs = _split_runs(times)
    for run_index, (run_left, run_right) in enumerate(runs):
        run_t = times[run_left:run_right]
        run_x = signal[run_left:run_right]
        unique_time = np.r_[True, np.diff(run_t) > 0.0]
        run_t, run_x = run_t[unique_time], run_x[unique_time]
        if run_t.size < 2:
            continue
        fs_hz = _infer_fs(run_t)
        if not np.isfinite(fs_hz) or fs_hz < 60.0:
            continue
        run_blocks: list[Block] = []
        run_start, run_end = float(run_t[0]), float(run_t[-1] + 1000.0 / fs_hz)
        block_index = 0
        for block_start in np.arange(run_start, run_end, block_seconds * 1000.0):
            block_end = min(float(block_start + block_seconds * 1000.0), run_end)
            left = int(np.searchsorted(run_t, block_start, side="left"))
            right = int(np.searchsorted(run_t, block_end, side="left"))
            block_t, block_x = run_t[left:right], run_x[left:right]
            if block_t.size < max(int(10.0 * fs_hz), 100):
                block_index += 1
                continue
            q005, q995, raw_span, score, event_count, raw_polarity = _block_measurement(block_x, block_t, fs_hz)
            block = Block(
                participant=participant,
                run_index=run_index,
                block_index=block_index,
                start_ms=float(block_t[0]),
                end_ms=float(block_end),
                sample_count=int(block_t.size),
                fs_hz=float(fs_hz),
                raw_p01_p99_span_uv=raw_span,
                q005_uv=q005,
                q995_uv=q995,
                polarity_score=score,
                event_count=event_count,
                raw_polarity=raw_polarity,
            )
            blocks.append(block)
            run_blocks.append(block)
            block_index += 1
        spans.extend(_assign_stable_spans(
            run_blocks,
            min_run_blocks=min_run_blocks,
            max_ambiguous_gap_blocks=max_ambiguous_gap_blocks,
        ))

    stable_positive = sum(int(span["classified_blocks"]) for span in spans if span["orientation"] == "positive")
    stable_negative = sum(int(span["classified_blocks"]) for span in spans if span["orientation"] == "negative")
    stable_positive_minutes = sum((float(span["end_ms"]) - float(span["start_ms"])) / 60_000.0 for span in spans if span["orientation"] == "positive")
    applied_minutes = sum((float(span["end_ms"]) - float(span["start_ms"])) / 60_000.0 for span in spans if span["orientation"] == "negative")
    polarity_review_blocks = int(sum(
        block.decision.startswith("review") or block.decision == "low_signal"
        for block in blocks
    ))
    if stable_positive and stable_negative:
        status = "mixed_polarity_detected"
    elif stable_negative:
        status = "negative_only_detected"
    elif stable_positive:
        status = "positive_only_detected"
    else:
        status = "unclassified"
    summary = {
        "participant": participant,
        "source_file": str(raw_path.resolve()),
        "n_samples_total": int(signal.size),
        "n_duplicate_timestamp_samples_removed": int(sum(
            (right - left) - np.count_nonzero(np.r_[True, np.diff(times[left:right]) > 0.0])
            for left, right in runs
        )),
        "n_nonfinite_samples_excluded": nonfinite_samples_excluded,
        "n_recording_runs": len(runs),
        "sample_rate_hz_median": float(_infer_fs(times)),
        "n_analyzed_blocks": len(blocks),
        "n_stable_positive_blocks": stable_positive,
        "n_stable_negative_blocks": stable_negative,
        "n_review_blocks": int(sum(block.decision.startswith("review") for block in blocks)),
        "n_polarity_uncertain_blocks": polarity_review_blocks,
        "n_low_signal_blocks": int(sum(block.raw_polarity == "low_signal" for block in blocks)),
        "n_severe_artifact_blocks": int(sum(block.raw_polarity == "severe_artifact" for block in blocks)),
        "status": status,
        "stable_positive_minutes": round(stable_positive_minutes, 3),
        "auto_inverted_minutes": round(applied_minutes, 3),
        "correction_applied": stable_negative > 0,
        "review_required": status in {"mixed_polarity_detected", "unclassified"} or polarity_review_blocks > 0,
        "orientation_rule": "exclude raw p01-p99 spans >5000uV; 5-25Hz bandpass; per-60s >=80% same-sign QRS event extrema; >=3 sustained same-sign blocks; max one ambiguous/low-signal block bridged",
    }
    _write_example_plot(participant, signal, times, spans, out_dir)
    return summary, blocks, spans


def _write_example_plot(
    participant: str,
    signal: np.ndarray,
    times: np.ndarray,
    spans: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    if not spans:
        return
    negative = [span for span in spans if span["orientation"] == "negative"]
    selected = (negative or spans)[0]
    center_ms = (float(selected["start_ms"]) + float(selected["end_ms"])) / 2.0
    plot_start, plot_end = center_ms - 3_000.0, center_ms + 3_000.0
    center_index = int(np.argmin(np.abs(times - center_ms)))
    fs_hz = float(selected.get("median_fs_hz", 130.0))
    half_width = max(10, int(round(3.0 * fs_hz)))
    left, right = max(0, center_index - half_width), min(len(times), center_index + half_width + 1)
    keep = (times[left:right] >= plot_start) & (times[left:right] <= plot_end)
    x, t = signal[left:right][keep].copy(), times[left:right][keep]
    if right - left < 10 or x.size < 10:
        return
    corrected = x * int(selected["correction_factor"])
    # Import pyplot only after directing its cache to a temporary directory.
    cache_dir = Path("/private/tmp/daily_hrv_matplotlib")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("FC_CACHEDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, constrained_layout=True)
    seconds = (t - center_ms) / 1000.0
    axes[0].plot(seconds, x, color="#41566b", linewidth=0.7)
    axes[0].set_ylabel("Raw ECG (µV)")
    axes[1].plot(seconds, corrected, color="#087f8c", linewidth=0.7)
    axes[1].set_ylabel("Corrected ECG (µV)")
    axes[1].set_xlabel("Seconds from example midpoint")
    decision = "invert" if int(selected["correction_factor"]) < 0 else "retain"
    figure.suptitle(f"{participant}: {selected['orientation']} polarity · {decision} · {selected['classified_blocks']} accepted 60-s blocks")
    for axis in axes:
        axis.grid(alpha=0.22)
    path = out_dir / "polarity_examples" / f"{participant}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def apply_manifest_polarity(
    participant: str,
    signal: np.ndarray,
    times_ms: np.ndarray,
    intervals: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply accepted interval polarity and return corrected ECG plus flip mask."""
    corrected = np.asarray(signal, dtype=np.float64).copy()
    times = np.asarray(times_ms, dtype=np.float64)
    flipped = np.zeros(times.shape, dtype=bool)
    relevant = [
        row for row in intervals
        if row["participant"] == participant and int(row["correction_factor"]) == -1
    ]
    if not relevant:
        return corrected, flipped

    if all("run_index" in row for row in relevant):
        by_run: dict[int, list[dict[str, Any]]] = {}
        for row in relevant:
            by_run.setdefault(int(float(row["run_index"])), []).append(row)
        for run_index, (left, right) in enumerate(_split_runs(times)):
            run_times = times[left:right]
            for row in by_run.get(run_index, []):
                start = int(np.searchsorted(run_times, float(row["start_ms"]), side="left"))
                end = int(np.searchsorted(run_times, float(row["end_ms"]), side="left"))
                if end > start:
                    corrected[left + start:left + end] *= -1.0
                    flipped[left + start:left + end] = True
    else:
        # Backward-compatible slow path for hand-authored interval manifests
        # that predate run_index fields.
        for row in relevant:
            keep = (times >= float(row["start_ms"])) & (times < float(row["end_ms"]))
            corrected[keep] *= -1.0
            flipped[keep] = True
    return corrected, flipped


def read_correction_intervals(path: Path = DEFAULT_OUT_DIR / "correction_intervals.csv") -> list[dict[str, Any]]:
    """Load the versioned interval map written by this script."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_corrected_ecg(
    raw_path: Path,
    participant: str,
    manifest_path: Path = DEFAULT_OUT_DIR / "correction_intervals.csv",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return absolute sample times, corrected ECG, and a per-sample flip mask."""
    with np.load(raw_path, allow_pickle=True) as source:
        times = np.asarray(source["time_ms"], dtype=np.float64)
        signal = np.asarray(source["ecg_uv"], dtype=np.float64)
    corrected, flipped = apply_manifest_polarity(
        participant,
        signal,
        times,
        read_correction_intervals(manifest_path),
    )
    return times, corrected, flipped


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(out_dir: Path, summaries: list[dict[str, Any]], spans: list[dict[str, Any]]) -> None:
    negative_spans = [span for span in spans if span["orientation"] == "negative"]
    flipped_minutes = sum((float(span["end_ms"]) - float(span["start_ms"])) / 60_000.0 for span in negative_spans)
    negative_participants = sorted({str(span["participant"]) for span in negative_spans}, key=lambda p: int(p[1:]))
    status_cn = {
        "positive_only_detected": "已判定区间均正向",
        "negative_only_detected": "已判定区间均负向",
        "mixed_polarity_detected": "检测到正负混合区间",
        "unclassified": "无法判定",
    }
    lines = [
        "# ECG 波形极性诊断与纠正报告",
        "",
        f"本轮扫描本地找到的 `{len(summaries)}` 个 Polar ECG 原始文件，共确认 `{len(negative_spans)}` 段可自动取反的稳定负向区间，合计 `{flipped_minutes:.1f}` 分钟。候选 participant：{', '.join(negative_participants) if negative_participants else '无'}。",
        "",
        "检测按每个 ECG 时间段中的 60 秒块进行：先排除 1–99 分位跨度大于 5,000 µV 的严重伪迹块，再对 5–25 Hz 波形的局部绝对峰统计正负方向；至少 80% 事件同向，且连续至少 3 个块同向才进入纠正清单。至多跨过一个方向模糊/低信号块；严重伪迹不桥接。",
        "",
        "纠正清单按时间区间保存。原始 NPZ 没有被改写；`load_corrected_ecg(raw_path, participant)` 会按清单生成取反后的内存视图与 sample-level 翻转掩码。方向混合、质量差或证据不足的 participant/区间仍标记复核。",
        "",
        "| Participant | 判定 | 稳定正向分钟 | 自动取反分钟 | 极性待复核块 | 严重伪迹块 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['participant']} | {status_cn.get(str(row['status']), row['status'])} | {float(row['stable_positive_minutes']):.1f} | "
            f"{float(row['auto_inverted_minutes']):.1f} | {int(row['n_polarity_uncertain_blocks'])} | {int(row['n_severe_artifact_blocks'])} |"
        )
    lines.extend([
        "",
        "## 产物",
        "",
        "- `participant_polarity_manifest.csv`：participant 级结果和复核状态。",
        "- `correction_intervals.csv`：逐时间段的方向、翻转因子和置信度摘要。",
        "- `polarity_blocks.csv`：60 秒块级分位数、主极性票数分数、伪迹与判定记录。",
        "- `polarity_examples/`：每个 participant 一个原波形/纠正后波形示例。",
        "- `protocol.json`：冻结的参数和规则。",
        "",
        "R 峰算法比较时应统一使用同一份 `correction_intervals.csv`，并在波形读取处调用 `load_corrected_ecg()`；不得只对图形翻转、却让检测器读原向波形。",
    ])
    (out_dir / "REPORT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--participants", nargs="+", help="Default: every P*_polar_ecg_raw.npz found under raw-root")
    parser.add_argument("--block-seconds", type=float, default=60.0)
    parser.add_argument("--min-run-blocks", type=int, default=3)
    parser.add_argument("--max-ambiguous-gap-blocks", type=int, default=1)
    args = parser.parse_args()
    if args.block_seconds != 60.0:
        parser.error("this version freezes the polarity measurement block at 60 seconds")
    if args.min_run_blocks < 3 or args.max_ambiguous_gap_blocks < 0:
        parser.error("min-run-blocks must be >=3 and max-ambiguous-gap-blocks must be >=0")
    if args.participants:
        raw_paths = [args.raw_root / p / f"{p}_polar_ecg_raw.npz" for p in args.participants]
    else:
        raw_paths = sorted(args.raw_root.glob("P*/P*_polar_ecg_raw.npz"), key=lambda path: int(path.parent.name[1:]))
    missing = [str(path) for path in raw_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing ECG source file(s): " + ", ".join(missing))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    all_blocks: list[Block] = []
    all_spans: list[dict[str, Any]] = []
    for raw_path in raw_paths:
        participant = raw_path.parent.name
        summary, blocks, spans = _analyze_participant(
            participant,
            raw_path,
            args.out_dir,
            block_seconds=args.block_seconds,
            min_run_blocks=args.min_run_blocks,
            max_ambiguous_gap_blocks=args.max_ambiguous_gap_blocks,
        )
        summaries.append(summary)
        all_blocks.extend(blocks)
        all_spans.extend(spans)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    _write_csv(args.out_dir / "participant_polarity_manifest.csv", summaries, list(summaries[0].keys()))
    _write_csv(args.out_dir / "polarity_blocks.csv", [vars(block) for block in all_blocks], list(vars(all_blocks[0]).keys()))
    span_fields = [
        "participant", "run_index", "start_ms", "end_ms", "orientation", "correction_factor",
        "classified_blocks", "span_blocks_including_bridged_gaps", "median_abs_polarity_score", "median_qrs_event_count_per_block", "median_fs_hz",
    ]
    _write_csv(args.out_dir / "correction_intervals.csv", all_spans, span_fields)
    protocol = {
        "version": "ecg_polarity_correction_v1",
        "source_field": "ecg_uv",
        "time_field": "time_ms",
        "source_files_modified": False,
        "filter": "Butterworth order 3, zero-phase 5-25 Hz bandpass (upper cutoff capped at 0.45*fs)",
        "measurement": "per 60-s block, sign of each local maximum of absolute 5-25 Hz filtered ECG, with 350-ms candidate spacing and 2.5*MAD prominence",
        "block_orientation_threshold_abs": 0.80,
        "severe_raw_ecg_artifact_gate": "q99-q01 > 5000 uV excludes the block from polarity inference, matching the existing blinded ECG review QC",
        "minimum_qrs_candidates_per_block": "max(8, 0.20 * block duration in seconds)",
        "accepted_orientation": f"at least {args.min_run_blocks} consecutive classified blocks with the same orientation; at most {args.max_ambiguous_gap_blocks} ambiguous/low-signal block can be bridged when that same orientation resumes",
        "correction": "multiply ecg_uv by -1 only within accepted negative-polarity intervals; positive intervals retain +1; short/mixed/ambiguous intervals are unchanged and require review",
        "files": {
            "participants": "participant_polarity_manifest.csv",
            "blocks": "polarity_blocks.csv",
            "intervals": "correction_intervals.csv",
            "examples": "polarity_examples/",
        },
    }
    (args.out_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "# ECG polarity correction v1\n\n"
        "This creates an auditable correction manifest for the raw Polar ECG series. It never overwrites source NPZ files. "
        "Use `load_corrected_ecg(raw_path, participant)` from `correct_ecg_polarity.py` when loading samples; it returns times, the corrected ECG view, and a sample-level flip mask. "
        "Only sustained, high-confidence negative-QRS spans are inverted; uncertain spans remain unchanged and are flagged. "
        "See `protocol.json`, the participant manifest, block diagnostics, and before/after example plots.\n",
        encoding="utf-8",
    )
    _write_report(args.out_dir, summaries, all_spans)
    print(f"Wrote polarity manifest: {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
