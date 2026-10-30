"""Prepare twenty fresh complete windows before reading new human annotations."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(BASE))
from build_blinded_ecg_rpeak_review import _candidate_starts
from correct_ecg_polarity import apply_manifest_polarity, read_correction_intervals
from evaluate_assisted_human_confirmed_rpeaks import detector_indices, infer_fs, refine_to_local_qrs_extreme
from remote_ecg_review.build_review_packet import _prior_manifests, _read_forbidden_starts
from remote_ecg_review.quality_flags import screen_clip
from remote_ecg_review.full_window import KIND, METHOD, prepare_full_window_supplement
from remote_ecg_review.peak_assistance import waveform_hash

LIVE = BASE / "outputs/ecg_rpeak_blind_review_remote_v1"
BATCH = "full_window_hrv_v1"
SEED = 20261029


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    path.chmod(0o600)


def block_summary(start, blocks):
    end = start + 300000
    totals = {"positive": 0., "negative": 0., "severe_artifact": 0., "uncertain": 0.}
    for block in blocks:
        overlap = max(0., min(end, float(block["end_ms"])) - max(start, float(block["start_ms"])))
        if not overlap:
            continue
        sign = block["raw_polarity"]
        key = sign if sign in totals else "uncertain"
        totals[key] += overlap / 1000
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw_root = ROOT / "Multisite-PPG/raw_data"
    correction = BASE / "outputs/ecg_polarity_correction_all20_v1/correction_intervals.csv"
    freeze = json.loads((ROOT / "analysis/ecg_detector_freeze_20261023_v1/frozen_protocol.json").read_text())
    if sha(correction) != freeze["polarity"]["sha256"]:
        raise ValueError("Existing frozen polarity manifest changed")
    for source in freeze["implementation_sources"]:
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Frozen detector implementation changed: " + source["path"])
    corrections = read_correction_intervals(correction)
    with gzip.open(LIVE / "review_data.json.gz", "rt") as handle:
        original = json.load(handle)
    manifests = _prior_manifests(BASE / "outputs")
    forbidden = _read_forbidden_starts(manifests)
    # Include every installed supplement, even if it lacks a manifest.
    packets = [LIVE / "review_data.json.gz", *sorted((LIVE / "supplements").glob("*.json.gz"))]
    for path in packets:
        with gzip.open(path, "rt") as handle:
            for clip in json.load(handle)["clips"]:
                forbidden.setdefault(clip["participant"], []).append(float(clip["windowStart"]))
    blocks = {}
    with (correction.parent / "polarity_blocks.csv").open() as handle:
        for row in csv.DictReader(handle):
            blocks.setdefault(row["participant"], []).append(row)
    # This is a small stratified validation cohort, not an unweighted estimate
    # of duration-population performance. No detector output selects a window.
    plan = {f"P{i}": ("negative" if i in {4, 17, 18, 19} else
                       "mixed_quality" if i in {2, 5, 8, 10, 13, 16} else "positive") for i in range(1, 21)}
    protocol = {"version": 1, "purpose": "full five-minute HRV reference annotation",
                "selection_seed": SEED, "planned_windows": 20, "windows_per_participant": 1,
                "stratum_preference_by_participant": plan,
                "selection": "Deterministic shuffled fresh contexts; prefer planned stratum using existing minute metadata only; record fallback. No detector performance or new annotation is used in selection.",
                "negative_preference": "at least 60 seconds of original negative minute evidence",
                "positive_preference": "at least 240 seconds positive; no severe or negative minute evidence",
                "mixed_quality_preference": "at least 15 seconds severe/uncertain evidence and at least 60 seconds signed evidence",
                "technical": "Original finite samples, strict timestamps, no gap >250ms; no deduplication, filling or interpolation",
                "no_prior_overlap": True, "polarity_manifest": str(correction), "polarity_sha256": sha(correction),
                "assistance_method": METHOD, "reviewer_scoped": True,
                "annotation": "Same immutable waveform and algorithm suggestions for each reviewer. Save peaks, manual unreadable intervals and 60 explicit page confirmations separately per reviewer. Keep all saved peaks, including those in unreadable intervals.",
                "future_evaluation": "Lock human references before scoring. Use same readable continuous runs for every algorithm; never bridge masks or sampling gaps when forming RR or adjacent RR differences. Report raw and IBI-corrected HRV separately; no production label replacement by this packet.",
                "acceptance_thresholds": "Numerical HRV acceptance thresholds remain to be agreed before viewing new validation scores; annotation completion does not itself qualify HRV accuracy.",
                "generated_at": datetime.now(timezone.utc).isoformat()}
    write_json(args.output / "review_protocol.json", protocol)
    selected, audit = [], []
    for pid, wanted in plan.items():
        source = raw_root / pid / f"{pid}_polar_ecg_raw.npz"
        stat = source.stat()
        with np.load(source, allow_pickle=True) as data:
            t, x = np.asarray(data["time_ms"], float), np.asarray(data["ecg_uv"], float)
        finite = np.flatnonzero(np.isfinite(t))
        t, x = t[finite[0]:finite[-1] + 1], x[finite[0]:finite[-1] + 1]
        if not np.isfinite(t).all() or np.any(np.diff(t) < 0):
            raise ValueError(pid + ": invalid original timebase")
        starts = _candidate_starts(t)
        blocked = np.unique(forbidden.get(pid, []))
        for prior in blocked:
            starts = starts[np.abs(starts - prior) >= 300000]
        bad = np.unique(np.r_[t[:-1][np.diff(t) <= 0], t[~np.isfinite(x)]])
        if len(bad):
            starts = starts[np.searchsorted(bad, starts) == np.searchsorted(bad, starts + 300000, side="right")]
        rng = np.random.default_rng(SEED + int(pid[1:]) * 1009)
        starts = rng.permutation(starts)
        choices = []
        for start in starts:
            summary = block_summary(float(start), blocks[pid])
            match = (summary["negative"] >= 60 if wanted == "negative" else
                     summary["positive"] >= 240 and summary["negative"] == 0 and summary["severe_artifact"] == 0 if wanted == "positive" else
                     summary["severe_artifact"] + summary["uncertain"] >= 15 and summary["positive"] + summary["negative"] >= 60)
            choices.append((not match, float(start), summary))
            if match:
                break
        if not choices:
            raise ValueError(pid + ": no fresh complete context remains")
        fallback, start, summary = min(choices, key=lambda c: c[0])
        a, b = np.searchsorted(t, start), np.searchsorted(t, start + 300000, side="right")
        wt, raw = t[a:b], x[a:b]
        if (len(wt) < 50 or not np.isfinite(raw).all() or np.any(np.diff(wt) <= 0)
                or np.max(np.diff(wt)) > 250 or wt[0] - start > 250 or start + 300000 - wt[-1] > 250):
            raise ValueError(pid + ": selected context failed original-sample integrity check")
        corrected, flips = apply_manifest_polarity(pid, raw, wt, corrections)
        try:
            seeds = detector_indices(corrected, infer_fs(wt), "rodrigues2021")
            timestamps = refine_to_local_qrs_extreme(corrected, wt, seeds, infer_fs(wt))
            indices = np.searchsorted(wt, timestamps).astype(int).tolist()
            status = "ok" if indices else "no_peaks"
        except Exception as error:
            indices, status = [], "detector_failure:" + type(error).__name__
        clip = {"id": f"full300_v1_{pid}_W01", "participant": pid, "reviewId": f"full300_v1_{pid}_W01",
                "reviewSet": KIND, "reviewProtocolKind": KIND, "reviewBatch": BATCH,
                "sourceLabel": "Full 5-minute HRV validation", "annotationMode": "algorithm_assisted",
                "blindReview": False, "annotationScope": "reviewer", "windowDurationSec": 300,
                "reviewPageSeconds": 5, "windowStart": float(start), "clipStart": float(wt[0]),
                "startSec": 0., "endSec": float((wt[-1] - start) / 1000),
                "t": np.round((wt - wt[0]) / 1000, 6).tolist(), "x": np.round(corrected, 2).tolist(),
                "suggested": indices, "restorePeaks": indices.copy(), "aidAvailable": bool(indices),
                "isPrevious": False, "hasPriorMarks": False, "priorUnreadable": False,
                "samplingGaps": [], "assistanceStatus": status}
        clip["samplingGaps"] = [[i, i + 1] for i, d in enumerate(np.diff(clip["t"])) if d > .2501]
        flags, quality = screen_clip(clip, wt, raw, flips, blocks[pid])
        clip["qualityFlags"], clip["qualitySummary"] = flags, quality
        clip["assistanceWaveformSha256"] = waveform_hash(clip)
        selected.append(clip)
        audit.append({"participant": pid, "clip_id": clip["id"], "window_start_unix_ms": start,
                      "preferred_stratum": wanted, "stratum_fallback": fallback, "minute_evidence_seconds": summary,
                      "fresh_candidate_count": len(starts), "source": str(source), "source_size": stat.st_size,
                      "source_mtime_ns": stat.st_mtime_ns, "original_sample_count": len(wt),
                      "source_window_sha256": hashlib.sha256(wt.tobytes() + raw.tobytes()).hexdigest(),
                      "waveform_sha256": clip["assistanceWaveformSha256"], "assistance_status": status,
                      "suggested_count": len(indices), "flipped_samples": int(flips.sum())})
        if (source.stat().st_size, source.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError("Raw source changed while preparing: " + pid)
        print(pid + ": complete window selected; " + wanted + (" (fallback)" if fallback else "") + f"; {len(indices)} suggestions", flush=True)
    packet = {"version": 2, "reviewProtocolKind": KIND, "participants": original["participants"],
              "reviewBatches": [{"id": BATCH, "label": "Full 5-minute HRV validation", "target": 20,
                                  "annotationScope": "reviewer", "default": True}],
              "assistanceProtocol": {"version": 1, "method": METHOD, "generatedAt": protocol["generated_at"]},
              "clips": selected}
    # Validate a decoded copy; loading enriches assistance metadata in memory only.
    prepare_full_window_supplement(json.loads(json.dumps(packet)), "validation")
    target = args.output / "supplement_full300_v1.json.gz"
    with gzip.open(target, "wt") as handle:
        json.dump(packet, handle, separators=(",", ":"), allow_nan=False)
    target.chmod(0o600)
    with (args.output / "review_manifest.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["participant", "window_start_unix_ms", "clip_id", "clip_seconds"])
        writer.writerows((c["participant"], c["windowStart"], c["id"], 300) for c in selected)
    write_json(args.output / "selection_audit_private.json", {"protocol_sha256": sha(args.output / "review_protocol.json"),
                "packet_sha256": sha(target), "selection": audit, "prior_manifests": [str(p) for p in manifests]})
    print(json.dumps({"packet": str(target), "windows": len(selected), "seconds": 300 * len(selected)}))


if __name__ == "__main__":
    main()
