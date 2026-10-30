"""Build a full-window assisted ECG review packet from raw-morphology proposals.

The initial candidate marks use the same independent raw-morphology algorithm
as ``outputs/assisted_ecg_rpeak_review_v1/annotation_app.html``.  They are not
Pan--Tompkins output.  This package is for efficient human correction and must
not replace the detector-blinded validation set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_blinded_ecg_rpeak_review import (
    _has_severe_amplitude_artifact,
    _independent_raw_r_candidates,
    _write_annotation_app,
)


WINDOW_MS = 300_000.0
DEFAULT_PARTICIPANTS = ("P1", "P13", "P15")
EXCLUDED_PARTICIPANTS = {"P2", "P5", "P14", "P16", "P17"}


def _candidate_starts(times_ms: np.ndarray) -> np.ndarray:
    times = np.asarray(times_ms, dtype=float)
    breaks = np.flatnonzero(np.diff(times) > 250.0) + 1
    bounds = np.r_[0, breaks, len(times)]
    starts: list[float] = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        latest = float(times[hi - 1]) - WINDOW_MS
        if latest >= float(times[lo]):
            starts.extend(np.arange(times[lo], latest + 1.0, 30_000.0).tolist())
    return np.asarray(starts, dtype=float)


def _temporal_stratified_starts(times_ms: np.ndarray, count: int) -> np.ndarray:
    starts = _candidate_starts(times_ms)
    if starts.size < count:
        raise ValueError(f"only {starts.size} complete five-minute ECG windows are available")
    return starts[np.linspace(0, starts.size - 1, count).round().astype(int)]


def _write_readme(out_dir: Path, *, windows: int, chunks: int, chunk_seconds: float) -> None:
    (out_dir / "README.md").write_text(
        f"""# 完整五分钟 Assisted ECG R-apex 修正包

本包包含 {windows} 个完整五分钟窗口、共 {chunks} 个 {chunk_seconds:g} 秒 raw-ECG
页面。每页红色初始标记由与
`outputs/assisted_ecg_rpeak_review_v1/annotation_app.html` 相同的独立
raw-morphology proposal 算法产生：5–25 Hz 形态事件、自动主导极性和原始波形
局部极值定位。它**没有**调用 Pan–Tompkins。

## 正确用途

- 红点是可编辑的候选，不是 ground truth。点击红点附近可删除；点击未标的原始
  QRS fiducial 可补加；“Reset candidates”会恢复该页的原始 proposals。
- 若存在明显伪迹或候选不可信，人工应修正而非接受。该包用于快速构建
  human-corrected labels。
- 由于审阅者看见了候选点，本包**不能**用于独立比较该候选算法与人工 apex 的
  性能，也不能替代 `full_window_blinded_ecg_review_v2_training_cohort`。
- 仍需用 v2 盲法包的双审阅者共识，选择最终 detector/refinement/QC 流程并验收
  五分钟 corrected RMSSD error。

## 选窗范围

仅使用发布训练队列内、本机可得 raw ECG 的 `P1`、`P13`、`P15`；不含
`P2`。每位参与者选择记录早期及晚期各一个完整窗口。原始数据在本机没有活动
日志，因此不宣称活动类别分层。

## 输出

- `annotation_app.html`：离线 Assisted 审阅页面。
- `full_window_review_manifest.csv`：窗口和 proposal 状态。
- `rpeak_annotations_template.csv`：导出 CSV 格式。
- `review_protocol.json`：固定算法与抽样协议。
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("Multisite-PPG/raw_data"))
    parser.add_argument("--participants", nargs="+", default=list(DEFAULT_PARTICIPANTS))
    parser.add_argument("--windows-per-participant", type=int, default=2)
    parser.add_argument("--chunk-seconds", type=float, default=5.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/full_window_assisted_raw_morphology_v1_training_cohort"),
    )
    args = parser.parse_args()
    if args.windows_per_participant < 1:
        raise ValueError("--windows-per-participant must be positive")
    chunk_ms = float(args.chunk_seconds) * 1000.0
    if chunk_ms <= 0 or WINDOW_MS % chunk_ms != 0:
        raise ValueError("--chunk-seconds must divide 300 seconds exactly")
    participants = tuple(args.participants)
    invalid = set(participants) & EXCLUDED_PARTICIPANTS
    if invalid:
        raise ValueError(f"excluded participants cannot enter this training-cohort package: {sorted(invalid)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks_per_window = int(round(WINDOW_MS / chunk_ms))
    manifest: list[dict[str, object]] = []
    clips: list[dict[str, object]] = []
    for participant in participants:
        raw_path = args.raw_root / participant / f"{participant}_polar_ecg_raw.npz"
        with np.load(raw_path, allow_pickle=True) as raw_file:
            signal = np.asarray(raw_file["ecg_uv"], dtype=float)
            times = np.asarray(raw_file["time_ms"], dtype=float)
        for rank, start in enumerate(_temporal_stratified_starts(times, args.windows_per_participant), start=1):
            review_id = f"{participant}_AFW{rank:02d}"
            lo = int(np.searchsorted(times, start, side="left"))
            hi = int(np.searchsorted(times, start + WINDOW_MS, side="right"))
            local_t, local_x = times[lo:hi], signal[lo:hi]
            proposed = 0
            suppressed = 0
            for chunk_no in range(chunks_per_window):
                left, right = chunk_no * chunk_ms, (chunk_no + 1) * chunk_ms
                # Half-open pages avoid duplicating a peak on a shared boundary.
                keep = (local_t >= start + left) & (
                    local_t <= start + right if chunk_no == chunks_per_window - 1 else local_t < start + right
                )
                clip_t, clip_x = local_t[keep], local_x[keep]
                if clip_t.size < 10:
                    raise ValueError(f"{review_id}, chunk {chunk_no + 1}: insufficient raw ECG samples")
                severe = _has_severe_amplitude_artifact(clip_x)
                suggested = np.empty(0, dtype=int) if severe else _independent_raw_r_candidates(clip_x, clip_t)
                proposed += int(suggested.size)
                suppressed += int(severe)
                clips.append({
                    "id": f"{review_id}_C{chunk_no + 1:02d}",
                    "reviewId": review_id,
                    "participant": participant,
                    "windowStart": round(float(start), 3),
                    "startSec": left / 1000.0,
                    "endSec": right / 1000.0,
                    "t": (clip_t - start).round(3).tolist(),
                    "x": clip_x.round(2).tolist(),
                    "aidStatus": "suppressed_severe_artifact" if severe else "shown",
                    "suggested": suggested.tolist(),
                })
            manifest.append({
                "review_id": review_id,
                "participant": participant,
                "window_start_unix_ms": round(float(start), 3),
                "window_end_unix_ms": round(float(start + WINDOW_MS), 3),
                "window_seconds": 300.0,
                "chunk_seconds": args.chunk_seconds,
                "chunks_per_window": chunks_per_window,
                "selection": "temporally_stratified_without_activity_metadata",
                "proposal_method": "independent_raw_morphology_5to25Hz_polarity_aware_raw_extremum",
                "proposed_marks": proposed,
                "candidate_aid_suppressed_chunks": suppressed,
                "raw_ecg_path": str(raw_path.resolve()),
            })

    pd.DataFrame(manifest).to_csv(args.out_dir / "full_window_review_manifest.csv", index=False)
    pd.DataFrame(columns=[
        "reviewer_id", "review_id", "participant", "window_start_unix_ms",
        "manual_rpeak_abs_unix_ms", "confidence", "notes",
    ]).to_csv(args.out_dir / "rpeak_annotations_template.csv", index=False)
    protocol = {
        "version": "full_window_assisted_raw_morphology_v1",
        "scope": "assisted human correction; not independent detector validation",
        "participants": list(participants),
        "excluded_participants": sorted(EXCLUDED_PARTICIPANTS),
        "windows": len(manifest),
        "window_seconds": 300.0,
        "chunk_seconds": args.chunk_seconds,
        "chunks": len(clips),
        "proposal_method": "5-25Hz morphology events + global dominant polarity + raw local extreme",
        "pan_tompkins_used_for_proposals": False,
    }
    (args.out_dir / "review_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(args.out_dir, windows=len(manifest), chunks=len(clips), chunk_seconds=args.chunk_seconds)
    _write_annotation_app(
        args.out_dir,
        manifest,
        clips,
        assisted=True,
        storage_key="full-window-assisted-raw-morphology-v1",
    )
    print(f"Wrote {len(manifest)} full-window assisted windows and {len(clips)} chunks to {args.out_dir}")


if __name__ == "__main__":
    main()
