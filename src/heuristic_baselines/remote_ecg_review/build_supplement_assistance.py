"""Generate an additive assistance overlay for the existing 40+20 clips.

Never edits source packets, human annotations, polarity decisions or sampling.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ecg-supplement-assistance-mpl")
APP = Path(__file__).resolve().parent
BASE = APP.parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(BASE))
from remote_ecg_review.peak_assistance import waveform_hash, ASSISTED_BATCH_LABELS


def main():
    import numpy as np
    from correct_ecg_polarity import read_correction_intervals, apply_manifest_polarity
    from evaluate_assisted_human_confirmed_rpeaks import detector_indices, infer_fs, refine_to_local_qrs_extreme

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement", type=Path, default=BASE / "outputs/ecg_rpeak_blind_review_remote_v1/supplements/supplement_blind60_v1.json.gz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists; choose a new path. No overwrite performed.")
    with gzip.open(args.supplement, "rt") as handle:
        source = json.load(handle)
    clips = source["clips"]
    if len(clips) != 60 or any(c.get("reviewBatch") not in ASSISTED_BATCH_LABELS for c in clips):
        raise ValueError("Expected the original 40+20 supplement")
    correction_path = BASE / "outputs/ecg_polarity_correction_all20_v1/correction_intervals.csv"
    corrections = read_correction_intervals(correction_path)
    output = {"version": 1, "method": "neurokit_local_qrs_extreme_50ms_full300s",
              "generatedAt": datetime.now(timezone.utc).isoformat(),
              "supplementSha256": hashlib.sha256(args.supplement.read_bytes()).hexdigest(),
              "correctionManifestSha256": hashlib.sha256(correction_path.read_bytes()).hexdigest(),
              "purpose": "User-requested algorithm-assisted review; not independent blind validation",
              "clips": {}}
    for pid in sorted({c["participant"] for c in clips}, key=lambda s: int(s[1:])):
        with np.load(ROOT / "Multisite-PPG/raw_data" / pid / f"{pid}_polar_ecg_raw.npz", allow_pickle=True) as data:
            times = np.asarray(data["time_ms"], float)
            raw = np.asarray(data["ecg_uv"], float)
        finite = np.flatnonzero(np.isfinite(times))
        times, raw = times[finite[0]:finite[-1] + 1], raw[finite[0]:finite[-1] + 1]
        if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            raise ValueError(f"{pid}: invalid source timeline")
        # The frozen manifest is indexed by full-record run number. Applying
        # it to a sliced window would renumber runs and miss valid flips.
        corrected_full, _ = apply_manifest_polarity(pid, raw, times, corrections)
        del raw
        windows = {}
        for clip in (c for c in clips if c["participant"] == pid):
            start = float(clip["windowStart"])
            if start not in windows:
                a = np.searchsorted(times, start, side="left")
                b = np.searchsorted(times, start + 300000, side="right")
                wt, wx = times[a:b], corrected_full[a:b]
                if (not np.isfinite(wt).all() or not np.isfinite(wx).all()
                        or np.any(np.diff(wt) <= 0)):
                    raise ValueError(f"{pid}: source context no longer technically eligible")
                corrected = wx
                status = "ok"
                try:
                    fs = infer_fs(wt)
                    candidates = refine_to_local_qrs_extreme(corrected, wt, detector_indices(corrected, fs, "neurokit"), fs)
                except Exception as error:
                    candidates = np.array([])
                    status = "detector_failed:" + type(error).__name__
                windows[start] = (wt, corrected, candidates, status)
            wt, corrected, candidates, status = windows[start]
            nearest = int(np.argmin(np.abs(wt - float(clip["clipStart"]))))
            ct = wt[nearest:nearest + len(clip["t"])]
            cx = corrected[nearest:nearest + len(clip["t"])]
            if (len(ct) != len(clip["t"]) or abs(ct[0] - clip["clipStart"]) > .1
                    or not np.allclose(np.round((ct - ct[0]) / 1000, 4), clip["t"], atol=1e-8, rtol=0)
                    or not np.array_equal(np.round(cx, 2), np.asarray(clip["x"]))):
                raise ValueError(f"{clip['id']}: frozen waveform identity mismatch")
            candidates = candidates[(candidates >= ct[0]) & (candidates <= ct[-1])]
            indices = sorted(set(int(np.argmin(np.abs(ct - peak))) for peak in candidates))
            output["clips"][clip["id"]] = {
                "suggested": indices, "waveformSha256": waveform_hash(clip),
                "status": status if status != "ok" or indices else "no_candidates",
            }
        del times, corrected_full, windows
        print(f"{pid}: assistance generated; frozen waveform identity verified", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    args.output.chmod(0o600)
    print(json.dumps({"clips": len(output["clips"]),
                      "suggested_peaks": sum(len(c["suggested"]) for c in output["clips"].values()),
                      "clips_without_suggestions": sum(not c["suggested"] for c in output["clips"].values()),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
