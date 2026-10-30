"""Collapse overlapping five-minute ECG QC windows into review segments."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def joined_reasons(values: pd.Series) -> str:
    reasons = {reason for value in values for reason in str(value).split("|") if reason}
    return "|".join(sorted(reasons))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    queue = pd.read_csv(args.input_dir / "review_queue.csv")
    if queue.empty:
        pd.DataFrame().to_csv(args.input_dir / "review_segments.csv", index=False)
        pd.DataFrame().to_csv(args.input_dir / "review_segment_members.csv", index=False)
        return
    queue = queue.sort_values(["participant", "ppg_window_t0_ms", "ppg_window_t1_ms"]).copy()
    segment_ids: list[str] = []
    segment_number = 0
    for participant, group in queue.groupby("participant", sort=False):
        current_end = float("-inf")
        current_id = ""
        for row_index, row in group.iterrows():
            start, end = float(row.ppg_window_t0_ms), float(row.ppg_window_t1_ms)
            if start > current_end:
                segment_number += 1
                current_id = f"SEG_{segment_number:03d}_{participant}"
                current_end = end
            else:
                current_end = max(current_end, end)
            segment_ids.append((row_index, current_id))
    assignment = pd.Series(dict(segment_ids), name="review_segment_id")
    queue["review_segment_id"] = queue.index.map(assignment)
    queue["window_duration_s"] = (queue.ppg_window_t1_ms - queue.ppg_window_t0_ms) / 1000.0
    summary = (
        queue.groupby(["review_segment_id", "participant"], as_index=False)
        .agg(
            segment_start_ms=("ppg_window_t0_ms", "min"),
            segment_end_ms=("ppg_window_t1_ms", "max"),
            risk_window_count=("source_window_index", "size"),
            distinct_qc_reasons=("qc_reason", joined_reasons),
            min_valid_ibi_ratio=("ecg_valid_ibi_ratio", "min"),
            max_ibi_correction_ratio=("ecg_ibi_correction_ratio", "max"),
            max_rmssd_ms=("ecg_rmssd_corrected_ms", "max"),
            max_severe_artifact_chunks=("ecg_severe_artifact_chunk_count", "max"),
        )
    )
    summary["segment_duration_s"] = (summary.segment_end_ms - summary.segment_start_ms) / 1000.0
    summary["qc_reason_count"] = summary.distinct_qc_reasons.str.count("\\|") + 1
    summary["priority"] = (
        100 * summary.qc_reason_count
        + 50 * (summary.max_severe_artifact_chunks > 0).astype(int)
        + 25 * (summary.min_valid_ibi_ratio < 1.0).astype(int)
        + 10 * (summary.max_ibi_correction_ratio > 0.20).astype(int)
        + summary.risk_window_count.clip(upper=9)
    )
    summary = summary.sort_values(
        ["priority", "risk_window_count", "participant", "segment_start_ms"], ascending=[False, False, True, True]
    ).reset_index(drop=True)
    summary.insert(0, "review_rank", range(1, len(summary) + 1))
    summary.to_csv(args.input_dir / "review_segments.csv", index=False)
    queue.sort_values(["review_segment_id", "source_window_index"]).to_csv(
        args.input_dir / "review_segment_members.csv", index=False
    )
    chunks: list[dict[str, object]] = []
    for segment in summary.itertuples(index=False):
        chunk_start = float(segment.segment_start_ms)
        chunk_number = 0
        while chunk_start < float(segment.segment_end_ms):
            chunk_number += 1
            chunk_end = min(chunk_start + 300_000.0, float(segment.segment_end_ms))
            members = queue[
                queue.review_segment_id.eq(segment.review_segment_id)
                & queue.ppg_window_t0_ms.lt(chunk_end)
                & queue.ppg_window_t1_ms.gt(chunk_start)
            ]
            chunks.append({
                "review_rank": int(segment.review_rank),
                "review_chunk_id": f"{segment.review_segment_id}_C{chunk_number}",
                "review_segment_id": segment.review_segment_id,
                "participant": segment.participant,
                "chunk_start_ms": chunk_start,
                "chunk_end_ms": chunk_end,
                "chunk_duration_s": (chunk_end - chunk_start) / 1000.0,
                "overlapping_risk_window_count": len(members),
                "distinct_qc_reasons": joined_reasons(members.qc_reason),
                "min_valid_ibi_ratio": members.ecg_valid_ibi_ratio.min(),
                "max_ibi_correction_ratio": members.ecg_ibi_correction_ratio.max(),
                "max_rmssd_ms": members.ecg_rmssd_corrected_ms.max(),
                "priority": int(segment.priority),
            })
            chunk_start = chunk_end
    pd.DataFrame(chunks).sort_values(["priority", "review_rank", "chunk_start_ms"], ascending=[False, True, True]).to_csv(
        args.input_dir / "review_chunks_5min.csv", index=False
    )
    print(f"{len(queue)} risk windows collapsed into {len(summary)} review segments and {len(chunks)} <=5-minute chunks")


if __name__ == "__main__":
    main()
