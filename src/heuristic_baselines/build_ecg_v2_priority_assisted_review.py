"""Create an assisted ECG review page for the highest-priority QC chunks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_blinded_ecg_rpeak_review import _has_severe_amplitude_artifact, _write_annotation_app


EXCLUDED_PARTICIPANTS = {"P2", "P5", "P14", "P16", "P17"}


def closest_indices(times: np.ndarray, points: np.ndarray) -> list[int]:
    if not len(times) or not len(points):
        return []
    indices: list[int] = []
    for point in points:
        right = int(np.searchsorted(times, point, side="left"))
        candidates = [index for index in (right - 1, right) if 0 <= index < len(times)]
        if candidates:
            indices.append(min(candidates, key=lambda index: abs(times[index] - point)))
    return sorted(set(indices))


def deduplicate_nearby(points: list[np.ndarray], tolerance_ms: float = 150.0) -> np.ndarray:
    values = np.sort(np.concatenate(points)) if points else np.empty(0, dtype=float)
    kept: list[float] = []
    for value in values:
        if not kept or value - kept[-1] > tolerance_ms:
            kept.append(float(value))
    return np.asarray(kept, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-chunks", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument(
        "--sidecar-prefix", required=True,
        help="Filename prefix before _{participant}.npz for the candidate sidecars.",
    )
    parser.add_argument(
        "--candidate-method", default="Emrich2023 detector then per-event dominant raw QRS extreme within +/-50ms",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("Multisite-PPG/raw_data"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--page-seconds", type=float, default=5.0)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/ecg_priority_assisted_review"),
    )
    args = parser.parse_args()
    if args.limit < 1 or args.page_seconds <= 0:
        raise ValueError("--limit and --page-seconds must be positive")
    chunks = pd.read_csv(args.review_chunks)
    required = {"review_rank", "review_chunk_id", "participant", "chunk_start_ms", "chunk_end_ms", "priority", "distinct_qc_reasons"}
    if missing := required - set(chunks.columns):
        raise ValueError(f"review chunks missing columns: {sorted(missing)}")
    chosen = chunks.head(args.limit).copy()
    invalid = set(chosen.participant) & EXCLUDED_PARTICIPANTS
    if invalid:
        raise ValueError(f"excluded participants in review chunks: {sorted(invalid)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sidecar_cache: dict[str, dict[str, np.ndarray]] = {}
    manifest: list[dict[str, object]] = []
    clips: list[dict[str, object]] = []
    excluded_pages: list[dict[str, object]] = []
    page_ms = args.page_seconds * 1000.0

    for row in chosen.itertuples(index=False):
        participant = str(row.participant)
        start, end = float(row.chunk_start_ms), float(row.chunk_end_ms)
        review_id = str(row.review_chunk_id)
        raw_path = args.raw_root / participant / f"{participant}_polar_ecg_raw.npz"
        if participant not in raw_cache:
            with np.load(raw_path, allow_pickle=True) as source:
                signal = np.asarray(source["ecg_uv"], dtype=float)
                times = np.asarray(source["time_ms"], dtype=float)
            keep = np.isfinite(signal) & np.isfinite(times)
            raw_cache[participant] = signal[keep], times[keep]
        signal, times = raw_cache[participant]
        sidecar_path = args.sidecar_dir / f"{args.sidecar_prefix}_{participant}.npz"
        if participant not in sidecar_cache:
            with np.load(sidecar_path, allow_pickle=True) as source:
                sidecar_cache[participant] = {
                    "starts": np.asarray(source["ppg_window_t0_ms"], dtype=float),
                    "ends": np.asarray(source["ppg_window_t1_ms"], dtype=float),
                    "peaks": np.asarray(source["ecg_r_peak_times_rel_ms"], dtype=object),
                }
        sidecar = sidecar_cache[participant]
        coverage = np.minimum(sidecar["ends"], end) - np.maximum(sidecar["starts"], start)
        anchor = int(np.argmax(coverage))
        overlapping_sidecars = np.flatnonzero(coverage > 0)
        candidate_parts = []
        for source_index in overlapping_sidecars:
            candidate_abs = sidecar["starts"][source_index] + np.asarray(sidecar["peaks"][source_index], dtype=float)
            candidate_parts.append(candidate_abs[(candidate_abs >= start) & (candidate_abs <= end)])
        candidate_abs = deduplicate_nearby(candidate_parts)
        lo, hi = np.searchsorted(times, start, side="left"), np.searchsorted(times, end, side="right")
        local_t, local_x = times[lo:hi], signal[lo:hi]
        if len(local_t) < 20:
            raise ValueError(f"{review_id}: insufficient raw ECG samples")
        page_count = int(np.ceil((end - start) / page_ms))
        for page in range(page_count):
            page_start = start + page * page_ms
            page_end = min(end, page_start + page_ms)
            if page == page_count - 1:
                keep = (local_t >= page_start) & (local_t <= page_end)
            else:
                keep = (local_t >= page_start) & (local_t < page_end)
            clip_t, clip_x = local_t[keep], local_x[keep]
            if len(clip_t) < 10:
                continue
            severe = _has_severe_amplitude_artifact(clip_x)
            clip_id = f"{review_id}_C{page + 1:02d}"
            if severe:
                excluded_pages.append({
                    "clip_id": clip_id, "review_id": review_id, "participant": participant,
                    "start_sec_in_chunk": (page_start - start) / 1000.0,
                    "end_sec_in_chunk": (page_end - start) / 1000.0,
                    "reason": "severe_raw_ecg_artifact_1st_to_99th_percentile_span_gt_5000uv",
                })
            in_page = candidate_abs[(candidate_abs >= page_start) & (candidate_abs <= page_end)]
            clips.append({
                "id": clip_id, "reviewId": review_id, "participant": participant,
                "windowStart": round(start, 3),
                "startSec": (page_start - start) / 1000.0,
                "endSec": (page_end - start) / 1000.0,
                "t": (clip_t - start).round(3).tolist(), "x": clip_x.round(2).tolist(),
                "aidStatus": "suppressed_severe_artifact" if severe else "shown",
                "autoSkip": bool(severe), "suggested": [] if severe else closest_indices(clip_t, in_page),
            })
        manifest.append({
            "review_id": review_id, "review_rank": int(row.review_rank), "priority": int(row.priority),
            "participant": participant, "window_start_unix_ms": round(start, 3), "window_end_unix_ms": round(end, 3),
            "window_seconds": (end - start) / 1000.0, "qc_reasons": str(row.distinct_qc_reasons),
            "candidate_method": args.candidate_method,
            "candidate_sidecar_anchor_window_index": anchor,
            "candidate_sidecar_source_window_count": len(overlapping_sidecars),
            "candidate_sidecar_path": str(sidecar_path.resolve()),
            "raw_ecg_path": str(raw_path.resolve()),
        })

    pd.DataFrame(manifest).to_csv(args.out_dir / "review_manifest.csv", index=False)
    chosen.to_csv(args.out_dir / "source_review_chunks.csv", index=False)
    pd.DataFrame(excluded_pages).to_csv(args.out_dir / "auto_excluded_severe_artifact_pages.csv", index=False)
    pd.DataFrame(columns=[
        "reviewer_id", "review_id", "participant", "window_start_unix_ms",
        "manual_rpeak_abs_unix_ms", "confidence", "notes",
    ]).to_csv(args.out_dir / "rpeak_annotations_template.csv", index=False)
    protocol = {
        "version": "ecg_priority_assisted_review_v1",
        "scope": "assisted correction of high-risk ECG candidate regions; not blinded validation",
        "selected_chunks": len(manifest), "page_seconds": args.page_seconds, "pages": len(clips),
        "candidate_method": args.candidate_method,
        "excluded_participants": sorted(EXCLUDED_PARTICIPANTS),
    }
    (args.out_dir / "review_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "# ECG 高风险窗口 Assisted 复核包\n\n"
        f"红点是 {args.candidate_method} 的局部 QRS timing 候选，不是 Pan--Tompkins。"
        "逐页确认、删除或补加一个生理 QRS 对应的 timing fiducial；严重伪迹页自动跳过。"
        "本包用于纠正高风险标签，不能作为 detector 的独立盲法验证。\n",
        encoding="utf-8",
    )
    _write_annotation_app(
        args.out_dir, manifest, clips, assisted=True, aid_method="emrich2023",
        storage_key=f"ecg-priority-assisted-review:{args.out_dir.name}:v1",
    )
    print(f"Wrote {len(manifest)} review chunks and {len(clips)} pages to {args.out_dir}")


if __name__ == "__main__":
    main()
