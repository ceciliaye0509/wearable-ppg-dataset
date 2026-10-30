"""Compare historical and additional ECG R-peak detectors on polarity-corrected ECG.

The available target-domain annotations were made with algorithmic assistance,
so this is a development screen, not an independent validation study. It runs
each candidate against the same manual timing references on both the raw and
manifest-corrected signal views, and reports strict and conventional event
matching tolerances separately.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/daily_hrv_matplotlib")

import neurokit2 as nk
import numpy as np
import pandas as pd

from build_blinded_ecg_rpeak_review import _has_severe_amplitude_artifact, _independent_raw_r_candidates
from evaluate_assisted_human_confirmed_rpeaks import (
    EXCLUDED_PARTICIPANTS,
    UNRESOLVABLE_PREFIX,
    detector_indices,
    infer_fs,
    one_to_one_match,
    parse_clip_ranges,
    raw_first_pan_fill,
    refine_to_dominant_raw_extreme,
    refine_to_local_qrs_extreme,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUTS = PACKAGE_ROOT / "outputs"
DETECTORS = (
    # Previously compared in this project.
    "pantompkins1985",
    "neurokit",
    "emrich2023",
    "promac",
    "khamis2016",
    # Additional NeuroKit2 detectors not included in the earlier comparison.
    "hamilton2002",
    "zong2003",
    "martinez2004",
    "christov2004",
    "gamboa2008",
    "elgendi2010",
    "engzeemod2012",
    "manikandan2012",
    "kalidas2017",
    "nabian2018",
    "rodrigues2021",
)
TOLERANCES_MS = (10.0, 20.0, 40.0, 150.0)
CORRECTION_FACTOR = "correction_factor"


def detector_indices_method(signal: np.ndarray, fs: float, method: str) -> np.ndarray:
    """Use the project's established preprocessing for old methods and NK defaults for new ones."""
    if method in {"pantompkins1985", "neurokit", "emrich2023"}:
        return detector_indices(signal, fs, method)
    # The earlier evaluator passed NeuroKit-cleaned ECG to Khamis/ProMAC.
    # Keep that convention for consistency; these additional algorithms use
    # the same stable preprocessing rather than detector-specific tuning.
    clean = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit")
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method=method)
    idx = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    return idx[(idx >= 0) & (idx < len(signal))]


def _apply_corrections(
    signal: np.ndarray,
    times: np.ndarray,
    participant: str,
    corrections: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    corrected = np.asarray(signal, dtype=float).copy()
    flip_mask = np.zeros(len(times), dtype=bool)
    participant_rows = corrections.loc[corrections.participant.astype(str).eq(participant)]
    for row in participant_rows.itertuples(index=False):
        factor = int(getattr(row, CORRECTION_FACTOR))
        if factor == 1:
            continue
        interval = (times >= float(row.start_ms)) & (times < float(row.end_ms))
        flip_mask |= interval
    corrected[flip_mask] *= -1.0
    return corrected, int(np.count_nonzero(flip_mask))


def _run_detectors(signal: np.ndarray, fs: float) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    events: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    for method in DETECTORS:
        try:
            events[method] = detector_indices_method(signal, fs, method)
        except Exception as error:  # individual detector failures must not cancel the comparison
            events[method] = np.empty(0, dtype=int)
            failures[method] = f"{type(error).__name__}: {error}"
    return events, failures


def _candidate_sets(
    signal: np.ndarray,
    times: np.ndarray,
    fs: float,
    full_raw_signal: np.ndarray,
    full_raw_times: np.ndarray,
    clip_left_ms: float,
    clip_right_ms: float,
    detector_events: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    clip_lo = int(np.searchsorted(times, clip_left_ms, side="left"))
    clip_hi = int(np.searchsorted(times, clip_right_ms, side="right"))
    clip_signal, clip_times = signal[clip_lo:clip_hi], times[clip_lo:clip_hi]
    raw_morphology = clip_times[_independent_raw_r_candidates(clip_signal, clip_times)]

    # Deployment candidates retain the prior project's chunking rules.
    from evaluate_assisted_human_confirmed_rpeaks import deployment_candidates_for_clip

    raw_5s = deployment_candidates_for_clip(
        full_raw_signal, full_raw_times, clip_left_ms, clip_right_ms, chunk_ms=5_000.0
    )
    raw_20s = deployment_candidates_for_clip(
        full_raw_signal, full_raw_times, clip_left_ms, clip_right_ms, chunk_ms=20_000.0
    )

    candidates: dict[str, np.ndarray] = {
        "raw_morphology_clip": raw_morphology,
        "raw_morphology_deployment_5s": raw_5s,
        "raw_morphology_deployment_20s": raw_20s,
    }
    refined: dict[str, np.ndarray] = {}
    clipped_events: dict[str, np.ndarray] = {}
    for method, indices in detector_events.items():
        valid_idx = np.asarray(indices, dtype=int)
        valid_idx = valid_idx[(valid_idx >= 0) & (valid_idx < len(signal))]
        event_times = times[valid_idx]
        event_times = event_times[(event_times >= clip_left_ms) & (event_times <= clip_right_ms)]
        clipped_events[method] = event_times
        local_times = refine_to_local_qrs_extreme(signal, times, valid_idx, fs)
        local_times = local_times[(local_times >= clip_left_ms) & (local_times <= clip_right_ms)]
        refined[method] = local_times
        candidates[f"{method}_event"] = event_times
        candidates[f"{method}_local_qrs_extreme"] = local_times

    pan_idx = detector_events.get("pantompkins1985", np.empty(0, dtype=int))
    pan_global = refine_to_dominant_raw_extreme(signal, times, pan_idx, fs)
    pan_global = pan_global[(pan_global >= clip_left_ms) & (pan_global <= clip_right_ms)]
    candidates["pantompkins1985_legacy_raw_apex_refined"] = pan_global
    candidates["v3_5s_morphology_plus_pan_fill"] = raw_first_pan_fill(raw_5s, pan_global)
    candidates["raw_morphology_plus_emrich_local_fill"] = raw_first_pan_fill(
        raw_morphology, refined.get("emrich2023", np.empty(0))
    )
    candidates["emrich_local_plus_promac_local_fill"] = raw_first_pan_fill(
        refined.get("emrich2023", np.empty(0)), refined.get("promac", np.empty(0))
    )
    candidates["promac_local_plus_emrich_local_fill"] = raw_first_pan_fill(
        refined.get("promac", np.empty(0)), refined.get("emrich2023", np.empty(0))
    )

    # New, pre-defined conservative agreement rule: keep an Emrich location
    # only if ProMAC also places an event within 80 ms. No thresholds are fit
    # on these review clips.
    emrich, promac = refined.get("emrich2023", np.empty(0)), refined.get("promac", np.empty(0))
    if emrich.size and promac.size:
        agreement = [peak for peak in emrich if np.min(np.abs(promac - peak)) <= 80.0]
        candidates["new_emrich_promac_agreement_80ms"] = np.asarray(agreement, dtype=float)
    else:
        candidates["new_emrich_promac_agreement_80ms"] = np.empty(0, dtype=float)
    return candidates


def _clip_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    tolerance: float,
) -> dict[str, float | int]:
    reference = np.sort(np.asarray(reference, dtype=float))
    candidate = np.sort(np.asarray(candidate, dtype=float))
    matches, missed, extra = one_to_one_match(reference, candidate, tolerance_ms=tolerance)
    errors = np.asarray([candidate[j] - reference[i] for i, j in matches], dtype=float)
    abs_errors = np.abs(errors)
    return {
        "reference_peaks": int(len(reference)),
        "candidate_peaks": int(len(candidate)),
        "matched_peaks": int(len(matches)),
        "missed_peaks": int(len(missed)),
        "extra_peaks": int(len(extra)),
        "median_abs_timing_error_ms": float(np.median(abs_errors)) if abs_errors.size else float("nan"),
        "p95_abs_timing_error_ms": float(np.quantile(abs_errors, 0.95)) if abs_errors.size else float("nan"),
    }


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["signal_view", "candidate_method", "match_tolerance_ms"]
    for group_key, table in metrics.groupby(keys, dropna=False):
        ref = int(table.reference_peaks.sum())
        proposal = int(table.candidate_peaks.sum())
        matched = int(table.matched_peaks.sum())
        precision = matched / proposal if proposal else float("nan")
        recall = matched / ref if ref else float("nan")
        timing = table.loc[table.matched_peaks.gt(0), "median_abs_timing_error_ms"].dropna()
        p95 = table.loc[table.matched_peaks.gt(0), "p95_abs_timing_error_ms"].dropna()
        rows.append({
            **dict(zip(keys, group_key, strict=True)),
            "clips": int(table.clip_id.nunique()),
            "reference_peaks": ref,
            "candidate_peaks": proposal,
            "matched_peaks": matched,
            "missed_peaks": ref - matched,
            "extra_peaks": proposal - matched,
            "precision_pct": 100.0 * precision,
            "recall_pct": 100.0 * recall,
            "f1_pct": 200.0 * precision * recall / (precision + recall) if precision + recall else float("nan"),
            "median_clip_median_abs_timing_error_ms": float(timing.median()) if len(timing) else float("nan"),
            "median_clip_p95_abs_timing_error_ms": float(p95.median()) if len(p95) else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUTS / "assisted_ecg_rpeak_final_validation_v5_no_p5" / "review_manifest.csv",
    )
    parser.add_argument(
        "--auto-excluded-clips",
        type=Path,
        help="Optional manifest of severe-artifact clips that were not annotated.",
    )
    parser.add_argument(
        "--correction-intervals",
        type=Path,
        default=DEFAULT_OUTPUTS / "ecg_polarity_correction_v1" / "correction_intervals.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUTS / "ecg_rpeak_polarity_corrected_comparison_v1",
    )
    parser.add_argument("--event-tolerances-ms", default="10,20,40,150")
    parser.add_argument("--include-confidence", nargs="+", default=["high"])
    parser.add_argument("--reference-time-column", default="manual_rpeak_abs_unix_ms")
    parser.add_argument(
        "--reference-label",
        default="assisted human-confirmed QRS timings; not a fully blinded independent validation set",
    )
    args = parser.parse_args()

    tolerances = tuple(float(value.strip()) for value in args.event_tolerances_ms.split(",") if value.strip())
    if not tolerances or any(value <= 0 for value in tolerances):
        raise ValueError("event tolerances must be positive")
    annotations = pd.read_csv(args.annotations, dtype={"review_id": str, "participant": str, "notes": str})
    manifest = pd.read_csv(args.manifest, dtype={"review_id": str, "participant": str})
    corrections = pd.read_csv(args.correction_intervals, dtype={"participant": str})
    severe = pd.read_csv(args.auto_excluded_clips, dtype={"clip_id": str}) if args.auto_excluded_clips else pd.DataFrame(columns=["clip_id"])
    required = {"review_id", "participant", args.reference_time_column}
    if missing := required - set(annotations.columns):
        raise ValueError(f"annotations missing required columns: {sorted(missing)}")
    annotations[args.reference_time_column] = pd.to_numeric(annotations[args.reference_time_column], errors="coerce")
    if "notes" not in annotations:
        annotations["notes"] = ""
    else:
        annotations["notes"] = annotations.notes.fillna("")
    if args.include_confidence and "confidence" in annotations:
        annotations = annotations.loc[annotations.confidence.astype(str).isin(args.include_confidence)].copy()
    annotations = annotations.loc[
        annotations[args.reference_time_column].notna()
        & annotations.participant.notna()
        & ~annotations.participant.astype(str).isin(EXCLUDED_PARTICIPANTS)
    ].copy()

    unresolvable = {
        note[len(UNRESOLVABLE_PREFIX):]
        for note in annotations.notes.astype(str)
        if note.startswith(UNRESOLVABLE_PREFIX)
    }
    excluded = set(severe.clip_id.dropna().astype(str)) | unresolvable
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_cache: dict[str, tuple[np.ndarray, np.ndarray, float, np.ndarray, int]] = {}
    metric_rows: list[dict[str, object]] = []
    detector_failures: dict[str, list[str]] = {method: [] for method in DETECTORS}
    skipped_clips: list[dict[str, str]] = []
    corrected_windows: set[str] = set()
    corrected_annotated_clips: set[str] = set()
    raw_samples = corrected_samples = 0
    corrected_annotated_samples = 0

    for window in manifest.itertuples(index=False):
        review_id, participant = str(window.review_id), str(window.participant)
        if participant in EXCLUDED_PARTICIPANTS:
            continue
        raw_path = str(window.raw_ecg_path)
        if raw_path not in raw_cache:
            with np.load(raw_path, allow_pickle=True) as source:
                raw_signal = np.asarray(source["ecg_uv"], dtype=float)
                times = np.asarray(source["time_ms"], dtype=float)
            valid = np.isfinite(raw_signal) & np.isfinite(times)
            raw_signal, times = raw_signal[valid], times[valid]
            corrected_signal, flipped = _apply_corrections(raw_signal, times, participant, corrections)
            raw_cache[raw_path] = raw_signal, times, infer_fs(times), corrected_signal, flipped
        raw_signal, times, fs, corrected_signal, _ = raw_cache[raw_path]
        start, end = float(window.window_start_unix_ms), float(window.window_end_unix_ms)
        lo, hi = np.searchsorted(times, [start, end], side="left")
        if hi <= lo:
            continue
        window_flip_count = int(np.count_nonzero(raw_signal[lo:hi] != corrected_signal[lo:hi]))
        raw_samples += int(hi - lo)
        corrected_samples += window_flip_count
        if window_flip_count:
            corrected_windows.add(review_id)

        view_cache: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
        for view, all_signal in (("raw", raw_signal), ("polarity_corrected", corrected_signal)):
            signal = all_signal[lo:hi]
            signal_times = times[lo:hi]
            detector_events, failures = _run_detectors(signal, fs)
            view_cache[view] = signal, detector_events
            for method, message in failures.items():
                detector_failures[method].append(f"{review_id}/{view}: {message}")

        for clip_number, (left_sec, right_sec) in enumerate(parse_clip_ranges(window.detail_excerpt_seconds), start=1):
            clip_id = f"{review_id}_C{clip_number}"
            if clip_id in excluded:
                skipped_clips.append({"clip_id": clip_id, "reason": "severe_artifact_or_unresolvable"})
                continue
            left, right = start + left_sec * 1000.0, start + right_sec * 1000.0
            reference = np.sort(annotations.loc[
                annotations.review_id.eq(review_id)
                & annotations[args.reference_time_column].between(left, right),
                args.reference_time_column,
            ].to_numpy(float))
            if not len(reference):
                skipped_clips.append({"clip_id": clip_id, "reason": "no_high_confidence_annotation"})
                continue
            clip_lo = int(np.searchsorted(times, left, side="left"))
            clip_hi = int(np.searchsorted(times, right, side="right"))
            clip_flip_count = int(np.count_nonzero(raw_signal[clip_lo:clip_hi] != corrected_signal[clip_lo:clip_hi]))
            corrected_annotated_samples += clip_flip_count
            if clip_flip_count:
                corrected_annotated_clips.add(clip_id)
            for view, all_signal in (("raw", raw_signal), ("polarity_corrected", corrected_signal)):
                clip_candidates = _candidate_sets(
                    all_signal[lo:hi], times[lo:hi], fs,
                    all_signal, times, left, right, view_cache[view][1],
                )
                for method, candidate in clip_candidates.items():
                    for tolerance in tolerances:
                        score = _clip_metrics(reference, candidate, tolerance)
                        metric_rows.append({
                            "clip_id": clip_id,
                            "review_id": review_id,
                            "participant": participant,
                            "signal_view": view,
                            "candidate_method": method,
                            "match_tolerance_ms": tolerance,
                            **score,
                        })

    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise RuntimeError("no annotated review clips were available to evaluate")
    summary = _aggregate(metrics)
    summary.to_csv(args.out_dir / "candidate_summary.csv", index=False)
    metrics.to_csv(args.out_dir / "clip_metrics.csv", index=False)
    pd.DataFrame(skipped_clips).to_csv(args.out_dir / "skipped_clips.csv", index=False)

    corrected_rank = summary.loc[
        summary.signal_view.eq("polarity_corrected") & summary.match_tolerance_ms.eq(20.0)
    ].sort_values(["f1_pct", "precision_pct", "median_clip_p95_abs_timing_error_ms"], ascending=[False, False, True])
    report = {
        "reference_provenance": args.reference_label,
        "manifest": str(args.manifest.resolve()),
        "annotation_source": str(args.annotations.resolve()),
        "polarity_correction_manifest": str(args.correction_intervals.resolve()),
        "excluded_severe_or_unresolvable_clips": sorted(excluded),
        "skipped_unannotated_clips": [row["clip_id"] for row in skipped_clips if row["reason"] == "no_high_confidence_annotation"],
        "evaluated_clips": int(metrics.loc[metrics.match_tolerance_ms.eq(tolerances[0]), "clip_id"].nunique()),
        "participants_evaluated": sorted(metrics.participant.unique().tolist()),
        "windows_overlapping_applied_polarity_corrections": sorted(corrected_windows),
        "annotated_clips_overlapping_applied_polarity_corrections": sorted(corrected_annotated_clips),
        "raw_samples_in_review_windows": int(raw_samples),
        "polarity_flipped_samples_in_review_windows": int(corrected_samples),
        "polarity_flipped_samples_in_annotated_clips": int(corrected_annotated_samples),
        "detectors": list(DETECTORS),
        "tolerances_ms": list(tolerances),
        "primary_development_ranking": "corrected-view F1 at +/-20 ms; ties by precision then lower median clip p95 timing error",
        "top_corrected_view_at_20ms": corrected_rank.head(10).to_dict(orient="records"),
        "detector_failures": {method: values for method, values in detector_failures.items() if values},
        "interpretation_limit": "Use this to shortlist a frozen algorithm for the next independent blind review; do not call it independent target-domain accuracy. Five-second excerpts are not used for five-minute RMSSD validation.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out_dir / "REPORT_CN.md").write_text(_render_report(report, summary), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _render_report(report: dict[str, object], summary: pd.DataFrame) -> str:
    selected = summary.loc[
        summary.signal_view.eq("polarity_corrected") & summary.match_tolerance_ms.isin([20.0, 150.0])
    ]
    lines = [
        "# 极性纠正后 ECG R 峰算法比较（开发筛选）",
        "",
        "## 参考与限制",
        "",
        f"本比较参考为：{report['reference_provenance']}。结果仅作开发筛选。片段只用于 R 峰检出与定位比较，不用于五分钟 RMSSD 验证。极性纠正只在第一步清单标记的持续倒置区间内应用；原始 NPZ 未修改。",
        "",
        f"- 有效标注片段：{report['evaluated_clips']} 个。",
        f"- 覆盖到极性纠正区间的 review window：{', '.join(report['windows_overlapping_applied_polarity_corrections']) or '无'}。",
        f"- 实际被评估且与纠正区间重叠的片段：{', '.join(report['annotated_clips_overlapping_applied_polarity_corrections']) or '无'}。",
        f"- review window 中翻转样本：{report['polarity_flipped_samples_in_review_windows']} / {report['raw_samples_in_review_windows']}；有效标注片段中翻转：{report['polarity_flipped_samples_in_annotated_clips']}。",
        "- 主排序：纠正视图 ±20 ms F1；相同 F1 时依次看 precision 和片段内 p95 定位误差。另报 ±10/40/150 ms，避免只凭宽松事件匹配判断定位精度。",
        "",
        "## 纠正视图候选结果",
        "",
        "| 方法 | 容差 | F1 | Precision | Recall | 漏检 | 多检 | 中位片段 p95 定位误差 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected.sort_values(["match_tolerance_ms", "f1_pct"], ascending=[True, False]).itertuples(index=False):
        lines.append(
            f"| {row.candidate_method} | ±{row.match_tolerance_ms:g} ms | {row.f1_pct:.2f}% | "
            f"{row.precision_pct:.2f}% | {row.recall_pct:.2f}% | {row.missed_peaks} | {row.extra_peaks} | "
            f"{row.median_clip_p95_abs_timing_error_ms:.2f} ms |"
        )
    lines.extend([
        "",
        "## 下一步解释",
        "",
        "本表用于确定下一轮人工盲验要冻结的少数候选。即使某方法在这里排名第一，也不能据此宣称已找到真实数据上的最高准确度；下一步必须用不展示算法标记的 raw ECG 人工复核，之后才可定最终算法及 HRV 标签方法。",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
