"""Attach the recorded polarity-screening decisions without changing ECG or marks."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import sys
import tempfile

APP_DIR = Path(__file__).resolve().parent
BASELINE_DIR = APP_DIR.parent
DEFAULT_PACKET = BASELINE_DIR / "outputs/ecg_rpeak_blind_review_remote_v1/review_data.json.gz"

LABELS = {
    "severe_artifact": "Severe artifact in the minute-level screen",
    "severe_segment_artifact": "Severe artifact in this segment",
    "ambiguous_polarity": "Unclear peak direction",
    "low_signal": "Insufficient signal for polarity screening",
    "short_polarity_run": "Same-direction run too short",
    "unclassified_polarity": "Polarity requires review",
    "unassessed_polarity": "No minute-level polarity decision",
}


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def screen_clip(clip, times, raw_signal, flip_mask, blocks):
    """Use raw amplitudes and existing minute decisions; never infer a new sign."""
    import numpy as np

    flags = []
    covered = np.zeros(len(times), dtype=bool)

    def flag(code, left, right, scope, reason, block=None):
        block = block or {}
        score = _number(block.get("polarity_score"))
        decision = block.get("decision", "")
        item = {
            "code": code, "label": LABELS[code], "scope": scope,
            "startIndex": int(left), "endIndex": int(right - 1),
            "startSec": round(float(clip["t"][left]), 4),
            "endSec": round(float(clip["t"][right - 1]), 4),
            "reason": reason, "decision": decision,
            "rawPolarity": block.get("raw_polarity", ""),
            "rawSpanUv": _number(block.get("raw_p01_p99_span_uv")),
            "signAgreement": (1 + abs(score)) / 2 if score is not None else None,
            "polarityScore": score,
            "bridged": decision in {"stable_positive", "stable_negative"} and block.get("raw_polarity") in {"ambiguous", "low_signal"},
            "flippedSampleCount": int(np.count_nonzero(flip_mask[left:right])),
        }
        flags.append(item)
        return item

    for block in blocks:
        if float(block["end_ms"]) <= times[0] or float(block["start_ms"]) > times[-1]:
            continue
        left = int(np.searchsorted(times, float(block["start_ms"]), side="left"))
        right = int(np.searchsorted(times, float(block["end_ms"]), side="left"))
        if left >= right:
            continue
        covered[left:right] = True
        raw = block["raw_polarity"]
        decision = block["decision"]
        bridged = decision in {"stable_positive", "stable_negative"} and raw in {"ambiguous", "low_signal"}
        if raw == "severe_artifact" or decision == "severe_artifact":
            code = "severe_artifact"
            reason = "Raw minute-level P01–P99 amplitude span exceeds 5,000 µV; excluded from polarity inference."
        elif raw == "ambiguous":
            code = "ambiguous_polarity"
            reason = "The minute did not meet the recorded same-direction vote threshold."
        elif raw == "low_signal":
            code = "low_signal"
            reason = "Signal strength, duration, or QRS-event count was insufficient for a reliable direction decision."
        elif decision == "review_short_polarity_run":
            code = "short_polarity_run"
            reason = "This same-direction run did not reach the required three classified minute blocks; no automatic polarity correction was accepted."
        elif not decision.startswith("stable_"):
            code = "unclassified_polarity"
            reason = "The recorded polarity decision requires manual review."
        else:
            continue
        if bridged:
            reason += " This minute was bridged by an accepted neighboring span in the existing correction manifest."
        elif code in {"ambiguous_polarity", "low_signal", "unclassified_polarity"}:
            reason += " No automatic polarity correction was accepted for this minute."
        flag(code, left, right, "minute", reason, block)

    missing = np.flatnonzero(~covered)
    if missing.size:
        groups = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
        for group in groups:
            flag("unassessed_polarity", int(group[0]), int(group[-1]) + 1, "minute",
                 "No recorded minute-level decision covers this interval; polarity is not confirmed by the screen.")

    span = float(np.quantile(raw_signal, .99) - np.quantile(raw_signal, .01))
    if span > 5000:
        item = flag("severe_segment_artifact", 0, len(times), "segment",
                    "Raw segment P01–P99 amplitude span exceeds 5,000 µV. Automatic morphology suggestions may be unreliable; review manually.")
        item["rawSpanUv"] = round(span, 2)
    flags.sort(key=lambda item: (item["startIndex"], item["endIndex"], item["code"]))
    return flags, {
        "available": True, "flagCount": len(flags), "requiresReview": bool(flags),
        "codes": sorted({item["code"] for item in flags}),
        "flippedSampleCount": int(np.count_nonzero(flip_mask)),
        "sampleCount": len(times), "rawP01P99SpanUv": round(span, 2),
    }


def attach_quality_flags(packet, *, raw_root: Path, correction_dir: Path):
    import numpy as np

    sys.path.insert(0, str(BASELINE_DIR))
    from correct_ecg_polarity import apply_manifest_polarity, read_correction_intervals

    blocks_path = correction_dir / "polarity_blocks.csv"
    blocks_by_participant = defaultdict(list)
    with blocks_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            blocks_by_participant[row["participant"]].append(row)
    corrections = read_correction_intervals(correction_dir / "correction_intervals.csv")
    groups = defaultdict(list)
    for clip in packet["clips"]:
        groups[clip["participant"]].append(clip)
    for participant, clips in groups.items():
        with np.load(raw_root / participant / f"{participant}_polar_ecg_raw.npz", allow_pickle=True) as source:
            times = np.asarray(source["time_ms"], dtype=float)
            signal = np.asarray(source["ecg_uv"], dtype=float)
        finite = np.isfinite(times) & np.isfinite(signal)
        times, signal = times[finite], signal[finite]
        corrected, flips = apply_manifest_polarity(participant, signal, times, corrections)
        unique = np.r_[True, np.diff(times) > 0]
        times, signal, corrected, flips = times[unique], signal[unique], corrected[unique], flips[unique]
        if np.any(np.diff(times) <= 0):
            raise ValueError(f"{participant}: raw timestamps do not match the packet's ordered source")
        for clip in clips:
            requested = clip["clipStart"] + np.asarray(clip["t"]) * 1000
            right = np.clip(np.searchsorted(times, requested), 0, len(times) - 1)
            left = np.maximum(0, right - 1)
            indices = np.where(np.abs(times[left] - requested) <= np.abs(times[right] - requested), left, right)
            if np.max(np.abs(times[indices] - requested)) > .1 or np.any(np.diff(indices) <= 0):
                raise ValueError(f"{clip['id']}: unable to align screening metadata to existing ECG samples")
            if not np.array_equal(np.round(corrected[indices], 2), np.asarray(clip["x"])):
                raise ValueError(f"{clip['id']}: existing ECG does not match the correction manifest")
            clip["qualityFlags"], clip["qualitySummary"] = screen_clip(
                clip, times[indices], signal[indices], flips[indices], blocks_by_participant[participant],
            )
    with (correction_dir / "protocol.json").open() as handle:
        protocol = json.load(handle)
    score = float(protocol["block_orientation_threshold_abs"])
    packet["qualityScreeningProtocol"] = {
        "source": str(blocks_path.resolve()), "rawAmplitudeThresholdUv": 5000,
        "netDirectionScoreThreshold": score, "equivalentSameSignFraction": (1 + score) / 2,
        "minimumClassifiedMinuteBlocks": 3, "maximumBridgedUncertainMinutes": 1,
        "note": "Read-only flags from the existing screening run. No waveform, peak, manual unreadable interval, or saved status is changed.",
    }
    codes = Counter(code for clip in packet["clips"] for code in clip["qualitySummary"]["codes"])
    return {
        "segments": len(packet["clips"]),
        "flagged_segments": sum(clip["qualitySummary"]["requiresReview"] for clip in packet["clips"]),
        "flagged_segments_by_reason": dict(codes),
        "bridged_uncertain_segments": sum(any(flag["bridged"] for flag in clip["qualityFlags"]) for clip in packet["clips"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.data_file.with_name("review_data.quality_candidate.json.gz")
    with gzip.open(args.data_file, "rt", encoding="utf-8") as handle:
        packet = json.load(handle)
    # Compare every original field after enrichment, including all ECG sample
    # arrays and original manual/candidate peak indices.
    original = json.loads(json.dumps(packet))
    protocol = json.loads(args.data_file.with_name("review_protocol.json").read_text())
    summary = attach_quality_flags(packet, raw_root=Path(protocol["raw_data_root"]), correction_dir=Path(protocol["polarity_correction_manifest"]).parent)
    assert packet["participants"] == original["participants"]
    for clip, original_clip in zip(packet["clips"], original["clips"], strict=True):
        assert all(clip.get(key) == value for key, value in original_clip.items() if key not in {"qualityFlags", "qualitySummary"}), clip["id"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".quality-", suffix=".gz", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        with gzip.open(temp_path, "wt", encoding="utf-8") as handle:
            json.dump(packet, handle, separators=(",", ":"), ensure_ascii=False)
        temp_path.replace(output)
    finally:
        temp_path.unlink(missing_ok=True)
    print(json.dumps({**summary, "output": str(output.resolve()), "original_fields_preserved": True}, indent=2))


if __name__ == "__main__":
    main()
