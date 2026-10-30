"""Create review-queue and QC-reason tables from ECG candidate sidecars."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--sidecar-glob", default="*.npz",
        help="Glob relative to --input-dir for the sidecars to summarize (default: %(default)s).",
    )
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    sidecars = sorted(args.input_dir.glob(args.sidecar_glob))
    if not sidecars:
        raise FileNotFoundError(f"no sidecars matching {args.sidecar_glob!r} in {args.input_dir}")
    for path in sidecars:
        with np.load(path, allow_pickle=True) as source:
            participant = str(source["participant"].item())
            config = json.loads(str(source["config_json"].item()))
            for index, required in enumerate(np.asarray(source["ecg_review_required"], dtype=bool)):
                if not required:
                    continue
                correction_ratio = float(source["ecg_ibi_correction_ratio"][index])
                label_qc_reason = str(source["ecg_label_qc_reason"][index])
                # A non-zero correction is a review trigger even when the
                # automatic QC label otherwise passes.  Surface that fact in
                # the queue rather than emitting the misleading label "ok".
                review_reasons = [] if label_qc_reason == "ok" else label_qc_reason.split("|")
                if correction_ratio > 0.0:
                    review_reasons.append("ibi_correction_applied")
                rows.append({
                    "participant": participant,
                    "source_window_index": index,
                    "ppg_window_t0_ms": float(source["ppg_window_t0_ms"][index]),
                    "ppg_window_t1_ms": float(source["ppg_window_t1_ms"][index]),
                    "ecg_rmssd_corrected_ms": float(source["ecg_rmssd_corrected_ms"][index]),
                    "ecg_hr_bpm_corrected": float(source["ecg_hr_bpm_corrected"][index]),
                    "ecg_valid_ibi_ratio": float(source["ecg_valid_ibi_ratio"][index]),
                    "ecg_ibi_correction_ratio": correction_ratio,
                    "ecg_emrich_local_qrs_peak_count": int(source["ecg_emrich2023_local_qrs_peak_count"][index]),
                    "ecg_severe_artifact_chunk_count": int(source["ecg_severe_artifact_chunk_count"][index]),
                    "label_qc_reason": label_qc_reason,
                    "qc_reason": "|".join(sorted(set(review_reasons))),
                    "label_version": config["label_version"],
                    "sidecar_path": str(path.resolve()),
                })
    queue = pd.DataFrame(rows)
    if len(queue):
        queue = queue.sort_values(["participant", "source_window_index"]).reset_index(drop=True)
        reason_counts = (
            queue.assign(qc_reason=queue.qc_reason.str.split("\\|")).explode("qc_reason")
            .groupby("qc_reason", as_index=False).size().sort_values("size", ascending=False)
            .rename(columns={"size": "windows"})
        )
    else:
        reason_counts = pd.DataFrame(columns=["qc_reason", "windows"])
    queue.to_csv(args.input_dir / "review_queue.csv", index=False)
    reason_counts.to_csv(args.input_dir / "qc_reason_summary.csv", index=False)
    print(json.dumps({"review_windows": int(len(queue)), "reason_counts": reason_counts.to_dict("records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
