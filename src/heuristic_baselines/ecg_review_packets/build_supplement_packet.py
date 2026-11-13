"""Prepare 40 candidate-free validation and 20 separately sampled audit clips.

Only the final waveform-only packet is served. Selection evidence is retained
in a private sidecar, never in the browser payload. Existing packet/DB untouched.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/daily_hrv_matplotlib")
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
WORKSPACE = BASE.parents[2]
sys.path.insert(0, str(BASE))
from build_blinded_ecg_rpeak_review import _candidate_starts
from correct_ecg_polarity import apply_manifest_polarity, read_correction_intervals
from evaluate_assisted_human_confirmed_rpeaks import detector_indices, infer_fs, refine_to_local_qrs_extreme
from ecg_review_packets.build_review_packet import _prior_manifests, _read_forbidden_starts
from ecg_review_packets.quality_flags import screen_clip

FREEZE = WORKSPACE / "analysis/ecg_detector_freeze_20261023_v1"
BASE_PACKET = BASE / "outputs/ecg_rpeak_blind_review_remote_v1/review_data.json.gz"
OUT = BASE / "outputs/ecg_review_supplement_blind60_v1"
BATCHES = [
    {"id": "independent_blind_v1", "label": "Independent blind validation", "target": 40},
    {"id": "challenge_blind_v1", "label": "Difficult / rule-boundary audit", "target": 20},
]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    path.chmod(0o600)

def event_difference(a, b):
    i = j = matched = 0
    while i < len(a) and j < len(b):
        delta = a[i] - b[j]
        if abs(delta) <= 20:
            matched += 1; i += 1; j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return len(a) + len(b) - 2 * matched

def write_review_manifest(folder):
    """Make the new contexts visible to future no-overlap review builders."""
    evidence=json.loads((folder/"selection_audit_private.json").read_text())
    with gzip.open(folder/"supplement_blind60_v1.json.gz","rt") as handle:
        packet=json.load(handle)
    clips={c["id"]:c for c in packet["clips"]}
    path=folder/"review_manifest.csv"
    with path.open("x",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(["participant","window_start_unix_ms","clip_id","review_id","review_batch","clip_start_unix_ms","clip_seconds"])
        for row in evidence["sampling"]:
            clip=clips[row["clip_id"]]
            writer.writerow([clip["participant"],clip["windowStart"],clip["id"],clip["reviewId"],clip["reviewBatch"],clip["clipStart"],20])
    path.chmod(0o600)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--pool-per-participant", type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((FREEZE / "frozen_protocol.json").read_text())
    if digest(FREEZE / "frozen_protocol.json") != (FREEZE / "frozen_protocol.sha256").read_text().split()[0]:
        raise RuntimeError("Frozen protocol checksum mismatch")
    for source in protocol["implementation_sources"]:
        if digest(Path(source["path"])) != source["sha256"]:
            raise RuntimeError("Frozen implementation changed: " + source["path"])
    spec = importlib.util.spec_from_file_location("supplement_frozen_rules", FREEZE / "frozen_rules.py")
    rules = importlib.util.module_from_spec(spec); spec.loader.exec_module(rules)
    rules.self_test()
    with gzip.open(BASE_PACKET, "rt") as handle:
        original = json.load(handle)
    correction_path = Path(protocol["polarity"]["manifest"])
    if digest(correction_path) != protocol["polarity"]["sha256"]:
        raise RuntimeError("Polarity manifest changed")
    corrections = read_correction_intervals(correction_path)
    blocks_by_pid = {}
    with (correction_path.parent / "polarity_blocks.csv").open() as handle:
        for row in csv.DictReader(handle):
            blocks_by_pid.setdefault(row["participant"], []).append(row)
    manifests = _prior_manifests(BASE / "outputs")
    forbidden = _read_forbidden_starts(manifests)
    for clip in original["clips"]:
        forbidden.setdefault(clip["participant"], []).append(float(clip["windowStart"]))
    independent, pool, sampling = [], [], []
    for participant in [f"P{i}" for i in range(1,21)]:
        rng = np.random.default_rng(20261023 + int(participant[1:]) * 1009)
        raw_path = WORKSPACE / "Multisite-PPG/raw_data" / participant / f"{participant}_polar_ecg_raw.npz"
        with np.load(raw_path, allow_pickle=True) as data:
            times, raw = np.asarray(data["time_ms"], float), np.asarray(data["ecg_uv"], float)
        finite = np.flatnonzero(np.isfinite(times))
        lo, hi = int(finite[0]), int(finite[-1]) + 1
        times, raw = times[lo:hi], raw[lo:hi]
        if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            raise RuntimeError(f"{participant}: source time corruption")
        corrected, flips = apply_manifest_polarity(participant, raw, times, corrections)
        starts = _candidate_starts(times)
        blocked = np.asarray(forbidden.get(participant, []), float)
        if len(blocked):
            starts = starts[np.all(np.abs(starts[:,None]-blocked[None,:]) >= 300000, axis=1)]
        # G0 is evaluated on original samples; no timestamp deduplication or
        # nonfinite-signal repair is allowed inside an eligible source context.
        bad = np.unique(np.r_[times[:-1][np.diff(times) <= 0], times[~np.isfinite(raw)]])
        if len(bad):
            left = np.searchsorted(bad, starts, side="left")
            right = np.searchsorted(bad, starts + 300000, side="right")
            starts = starts[left == right]
        population = len(starts)
        chosen = []
        for rank in range(2 + args.pool_per_participant):
            available = starts if not chosen else starts[np.all(np.abs(starts[:,None]-np.asarray(chosen)[None,:]) >= 300000,axis=1)]
            if not len(available):
                if rank < 2:
                    raise RuntimeError(f"{participant}: insufficient fresh independent windows")
                break
            start = float(available[int(rng.integers(len(available)))])
            chosen.append(start)
            a,b = np.searchsorted(times,start,side="left"),np.searchsorted(times,start+300000,side="right")
            wt,wx = times[a:b],corrected[a:b]
            issues = rules.technical_reasons(wt,raw[a:b],start,start+300000)
            if issues:
                raise RuntimeError(f"{participant}: eligibility implementation mismatch: {issues}")
            offset = int(rng.integers(20,261))
            left,right = np.searchsorted(wt,start+offset*1000,side="left"),np.searchsorted(wt,start+(offset+20)*1000,side="left")
            ct,cx = wt[left:right],wx[left:right]
            blind = rank < 2
            batch = BATCHES[0] if blind else BATCHES[1]
            clip_id = f"blind60_{'ind' if blind else 'pool'}_{participant}_{rank+1:02d}"
            clip = {"id":clip_id,"participant":participant,"reviewId":f"blind60_{participant}_{rank+1:02d}",
                    "reviewSet":"ecg_independent_blind_v1" if blind else "ecg_challenge_blind_v1",
                    "reviewBatch":batch["id"],"batchLabel":batch["label"],"blindReview":True,"annotationMode":"manual_blind",
                    "sourceLabel":batch["label"],"isPrevious":False,"hasPriorMarks":False,"priorUnreadable":False,
                    "windowStart":round(start,3),"clipStart":round(float(ct[0]),3),
                    "startSec":round(float((ct[0]-start)/1000),3),"endSec":round(float((ct[-1]-start)/1000),3),
                    "t":np.round((ct-ct[0])/1000,4).tolist(),"x":np.round(cx,2).tolist(),
                    "suggested":[],"restorePeaks":[],"aidAvailable":False}
            flags, quality = screen_clip(clip,ct,raw[a+left:a+right],flips[a+left:a+right],blocks_by_pid[participant])
            clip["qualityFlags"],clip["qualitySummary"] = flags,quality
            gate = rules.waveform_gate_for_clip(times,raw,ct[0],ct[-1])
            spans = [block["span_uv"] for block in gate["blocks"] if np.isfinite(block["span_uv"])]
            evidence = {"clip_id":clip_id,"participant":participant,"window_start_unix_ms":start,
                        "excerpt_offset_sec":offset,"source_file":str(raw_path),"population_candidate_windows":population,
                        "available_windows_at_draw":len(available),"conditional_draw_probability":1/len(available),
                        "G0_pass":True,"G1_pass":gate["pass"],"G1_max_block_span_uv":max(spans) if spans else None,
                        "raw_segment_span_uv":quality["rawP01P99SpanUv"],"polarity_codes":quality["codes"],
                        "blocks_under_5000":sum(span<=5000 for span in spans),"source_blocks":len(spans),"purpose":"independent_blind" if blind else "candidate_pool"}
            if blind:
                # Selection was committed before reading any detector output.
                independent.append(clip); sampling.append(evidence)
                continue
            predicted, failures = {},[]
            for method in ("neurokit","emrich2023","rodrigues2021"):
                try:
                    idx = detector_indices(wx,infer_fs(wt),method)
                    peaks = refine_to_local_qrs_extreme(wx,wt,idx,infer_fs(wt))
                    predicted[method] = peaks[(peaks>=ct[0])&(peaks<=ct[-1])]
                except Exception as error:
                    failures.append(f"{method}:{type(error).__name__}")
                    predicted[method] = np.array([])
            names = list(predicted)
            evidence["detector_disagreement_count"] = max(event_difference(predicted[names[i]],predicted[names[j]]) for i in range(3) for j in range(i+1,3))
            evidence["detector_failures"] = failures
            pool.append((clip,evidence))
        print(f"{participant}: 2 independent clips committed; {len(chosen)-2} fresh challenge candidates screened",flush=True)
    if len(independent)!=40:
        raise RuntimeError("Independent batch must contain exactly 40 clips")
    selected,used,counts = [],set(),{}
    categories = ["rule_boundary","rejected_mixed_quality","detector_disagreement","low_amplitude","uncertain_polarity"]
    def score(e,category):
        span=e["G1_max_block_span_uv"] or 0
        if category=="rule_boundary":
            return abs(span-5000) if 3000<=span<=7000 else None
        if category=="rejected_mixed_quality":
            return span if not e["G1_pass"] and e["blocks_under_5000"]>=max(1,e["source_blocks"]//2) else None
        if category=="detector_disagreement":
            return -e["detector_disagreement_count"] if e["detector_disagreement_count"]>0 else None
        if category=="low_amplitude":
            return e["raw_segment_span_uv"] if e["G1_pass"] else None
        if category=="uncertain_polarity":
            return len(e["detector_failures"]) if any(code in e["polarity_codes"] for code in ["ambiguous_polarity","short_polarity_run","low_signal","unassessed_polarity","unclassified_polarity"]) else None
    # Scarcer waveform strata are allocated first; max two challenges/person.
    for category in categories:
        ranked = sorted([(score(e,category),i) for i,(_,e) in enumerate(pool) if score(e,category) is not None],key=lambda x:(x[0],x[1]))
        accepted = 0
        for _,index in ranked:
            clip,evidence = pool[index]
            if index in used or counts.get(clip["participant"],0)>=2:
                continue
            used.add(index);counts[clip["participant"]]=counts.get(clip["participant"],0)+1
            clip["id"] = f"blind60_challenge_{clip['participant']}_{len(selected)+1:02d}"
            evidence={**evidence,"clip_id":clip["id"],"purpose":"challenge_audit","selection_category":category}
            selected.append(clip);sampling.append(evidence);accepted+=1
            if accepted==4:
                break
        if accepted!=4:
            raise RuntimeError(f"Insufficient {category} candidates under participant cap: {accepted}/4; widen pool explicitly, do not silently change category")
    all_clips=independent+selected
    existing_ids={clip["id"] for clip in original["clips"]}
    assert len(all_clips)==60 and len({c["id"] for c in all_clips})==60
    assert not existing_ids.intersection(c["id"] for c in all_clips)
    for participant in [f"P{i}" for i in range(1,21)]:
        intervals=sorted((c["windowStart"],c["windowStart"]+300000) for c in all_clips if c["participant"]==participant)
        assert all(b<=c for (_,b),(c,_) in zip(intervals,intervals[1:]))
    payload={"version":2,"reviewBatches":BATCHES,"participants":original["participants"],"clips":all_clips}
    packet_path=args.output/"supplement_blind60_v1.json.gz"
    with gzip.open(packet_path,"wt",encoding="utf-8") as handle:
        json.dump(payload,handle,ensure_ascii=False,separators=(",",":"),allow_nan=False)
    packet_path.chmod(0o600)
    dump_json(args.output/"selection_audit_private.json",{"seed_scheme":"20261023 + participant_number*1009","base_packet_sha256":digest(BASE_PACKET),"frozen_protocol_sha256":digest(FREEZE/"frozen_protocol.json"),"source_manifests":[str(p) for p in manifests],"sampling":sampling,"challenge_pool_size":len(pool),"representative_sampling":"uniform sequential context draws with no temporal overlap; uniform integer excerpt offset20..260s; participant-balanced, not pooled duration-population prevalence","blinding":"No detector peaks, private reasons, G1 decisions or amplitude/polarity screening judgments exposed during first-pass review"})
    dump_json(args.output/"sampling_plan_amendment_v1.json",{"amends":"next_sampling counts only in ecg_detector_shortlist_and_validation_gate_20261023_v1","independent":40,"challenge":20,"seconds_per_clip":20,"algorithm_and_exclusion_parameters_changed":False,"authorization":"user requested adding the 40+20 scheme to the existing website","base_packet_unchanged":True,"source_annotations_unchanged":True,"packet_sha256":digest(packet_path)})
    write_review_manifest(args.output)
    print(json.dumps({"packet":str(packet_path),"independent":40,"challenge":20,"challenge_participants":len(counts),"existing_clips_unchanged":235},indent=2),flush=True)

if __name__=="__main__":
    main()
