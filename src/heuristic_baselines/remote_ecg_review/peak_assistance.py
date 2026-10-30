"""Explicit, hash-bound assistance overlay; never rewrite frozen blind packets."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

OVERLAY_NAME = "supplement_peak_assistance.json"
ASSISTED_SUPPLEMENT_METHODS = {
    "neurokit_local_qrs_extreme_50ms_full300s",
    "rodrigues2021_guarded_startup_local_qrs_extreme_50ms",
    "saved_human_marks_recheck_v1",
}
ASSISTED_BATCH_LABELS = {
    "independent_blind_v1": "Validation (algorithm-assisted; formerly blind)",
    "challenge_blind_v1": "Difficult / rule-boundary audit (algorithm-assisted)",
}


def waveform_hash(clip: dict) -> str:
    payload = json.dumps({"t": clip["t"], "x": clip["x"]}, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def prepare_assisted_supplement(extra: dict, packet_sha256: str) -> None:
    """Validate a separately versioned, explicitly assisted additive packet.

    No saved human data is read or changed here. Original candidate-free packets
    keep their existing stricter validation and original overlay hash.
    """
    protocol = extra.get("assistanceProtocol", {})
    if (extra.get("reviewProtocolKind") != "assisted_peak_validation_v1"
            or protocol.get("version") != 1
            or protocol.get("method") not in ASSISTED_SUPPLEMENT_METHODS
            or not isinstance(protocol.get("generatedAt"), str)):
        raise ValueError("Unrecognized assisted supplement protocol")
    batch_ids = {batch["id"] for batch in extra.get("reviewBatches", [])}
    if not batch_ids or len(batch_ids) != len(extra.get("reviewBatches", [])):
        raise ValueError("Missing or duplicate assisted supplement batches")
    for clip in extra["clips"]:
        t, x, indices = clip.get("t"), clip.get("x"), clip.get("suggested")
        if (not isinstance(t, list) or not isinstance(x, list) or not t or len(t) != len(x)
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in t + x)
                or any(a >= b for a, b in zip(t[:-1], t[1:]))):
            raise ValueError("Invalid assisted supplement waveform")
        if (clip.get("blindReview") is not False or clip.get("annotationMode") != "algorithm_assisted"
                or clip.get("reviewBatch") not in batch_ids
                or not isinstance(indices, list)
                or any(type(i) is not int or not 0 <= i < len(t) for i in indices)
                or indices != sorted(set(indices)) or clip.get("restorePeaks") != indices
                or clip.get("aidAvailable") is not bool(indices)):
            raise ValueError("Invalid assisted supplement peak suggestions")
        digest = waveform_hash(clip)
        if clip.get("assistanceWaveformSha256") != digest:
            raise ValueError("Assisted supplement waveform hash mismatch")
        ranges = clip.get("priorUnreadableRanges", [])
        if (not isinstance(ranges, list) or any(
                not isinstance(r, list) or len(r) != 2
                or any(type(i) is not int for i in r)
                or not 0 <= r[0] < r[1] < len(t) for r in ranges)):
            raise ValueError("Invalid preserved unreadable intervals")
        clip["screeningWithheld"] = True
        clip["assistance"] = {"method": protocol["method"], "overlaySha256": packet_sha256,
                              "waveformSha256": digest, "generatedAt": protocol["generatedAt"],
                              "status": clip.get("assistanceStatus", "ok"),
                              "notIndependentBlindValidation": True}
    for batch in extra["reviewBatches"]:
        if sum(c.get("reviewBatch") == batch["id"] for c in extra["clips"]) != batch.get("target"):
            raise ValueError("Assisted supplement batch count mismatch")


def apply_assistance_overlay(packet: dict, directory: Path) -> None:
    path = directory / OVERLAY_NAME
    if not path.exists():
        return
    overlay = json.loads(path.read_text(encoding="utf-8"))
    if overlay.get("version") != 1 or not isinstance(overlay.get("clips"), dict):
        raise ValueError("Malformed supplement assistance overlay")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    targets = {c["id"]: c for c in packet["clips"]
               if c.get("reviewBatch") in ASSISTED_BATCH_LABELS}
    if set(targets) != set(overlay["clips"]):
        raise ValueError("Assistance overlay must cover exactly the two supplement batches")
    for clip_id, clip in targets.items():
        item = overlay["clips"][clip_id]
        indices = item.get("suggested")
        if item.get("waveformSha256") != waveform_hash(clip):
            raise ValueError(f"Assistance waveform identity mismatch: {clip_id}")
        if (not isinstance(indices, list) or len(indices) > len(clip["t"])
                or any(type(i) is not int or not 0 <= i < len(clip["t"]) for i in indices)
                or indices != sorted(set(indices))):
            raise ValueError(f"Invalid assistance sample indices: {clip_id}")
        clip.update(
            wasBlindReview=True, blindReview=False, screeningWithheld=True,
            annotationMode="algorithm_assisted", suggested=indices, restorePeaks=indices.copy(),
            aidAvailable=bool(indices),
            sourceLabel=ASSISTED_BATCH_LABELS[clip["reviewBatch"]],
            assistance={"method": overlay["method"], "overlaySha256": digest,
                        "waveformSha256": item["waveformSha256"],
                        "generatedAt": overlay["generatedAt"],
                        "status": item.get("status", "ok"),
                        "notIndependentBlindValidation": True},
        )
    for batch in packet.get("reviewBatches", []):
        if batch["id"] in ASSISTED_BATCH_LABELS:
            batch["label"] = ASSISTED_BATCH_LABELS[batch["id"]]


def screening_withheld(clip: dict) -> bool:
    return bool(clip.get("blindReview") or clip.get("screeningWithheld"))
