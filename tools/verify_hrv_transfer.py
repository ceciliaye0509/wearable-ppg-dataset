"""Verify the frozen v4.3 data after copying it to another machine.

This checks transfer integrity and the saved participant split. It does not
regenerate ECG labels, preprocess PPG, fit scalers, or train a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO / "experiments" / "hrv_v4_3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_path(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe manifest path: {relative}")
    result = root.joinpath(*rel.parts).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path leaves dataset directory: {relative}")
    return result


def verify(data_dir: Path, quick: bool = False) -> dict:
    import numpy as np

    manifest = json.loads((EXPERIMENT / "dataset_manifest.json").read_text(encoding="utf-8"))
    folds = json.loads((EXPERIMENT / "folds.json").read_text(encoding="utf-8"))
    data_dir = data_dir.resolve()
    if sha256(data_dir / "checksums.json") != manifest["checksums_sha256"]:
        raise ValueError("Dataset checksum manifest differs from frozen v4.3")
    if sha256(data_dir / "ecg_reference_protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("ECG protocol differs from frozen v4.3")
    saved = json.loads((data_dir / "checksums.json").read_text(encoding="utf-8"))
    if saved != manifest["files_sha256"]:
        raise ValueError("Dataset file list differs from the committed transfer manifest")
    hashed, skipped = [], []
    for name, expected in saved.items():
        path = contained_path(data_dir, name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {name}")
        if quick and path.stat().st_size > 8 * 1024 * 1024:
            skipped.append(name)
            continue
        print(f"Checking {name}", flush=True)
        if sha256(path) != expected:
            raise ValueError(f"SHA256 mismatch: {name}")
        hashed.append(name)

    observed = {}
    for reference in manifest["participants"]:
        pid = reference["participant"]
        path = data_dir / f"{manifest['dataset_name']}_{pid}.npz"
        with np.load(path, allow_pickle=True) as z:
            qc = np.asarray(z["ecg_label_qc_pass"], dtype=bool)
            rmssd = np.asarray(z["ecg_rmssd_corrected_ms"], dtype=float)
            sdnn = np.asarray(z["ecg_sdnn_corrected_ms"], dtype=float)
            eligible = qc & np.isfinite(rmssd) & np.isfinite(sdnn)
            ratio = np.asarray(z["ppg_resampled_valid_sample_ratio"], dtype=float)[:, 0, 0]
            model_mask = eligible & (rmssd > 0) & (sdnn > 0)
            model_mask &= np.isfinite(ratio) & (ratio >= 0.95)
            peaks = z["ecg_r_peak_times_rel_ms"]
            valid_peaks = []
            for value in peaks:
                p = np.asarray(value, dtype=float).reshape(-1)
                valid_peaks.append(len(p) >= 4 and np.isfinite(p).all() and (np.diff(p) > 0).all())
            model_mask &= np.asarray(valid_peaks, dtype=bool)
            source = np.asarray(z["source_window_index"])
            start = np.asarray(z["ppg_window_t0_ms"])
            end = np.asarray(z["ppg_window_t1_ms"])
            if not all(len(a) == len(qc) for a in (rmssd, sdnn, source, start, end, ratio, peaks)):
                raise ValueError(f"Window metadata length mismatch: {pid}")
            if len(np.unique(source)) != len(source):
                raise ValueError(f"Duplicate source_window_index: {pid}")
            observed[pid] = {"candidates": len(qc), "valid": int(eligible.sum()),
                             "model_input_eligible": int(model_mask.sum())}
            for field in ("candidates", "valid", "model_input_eligible"):
                if observed[pid][field] != reference[field]:
                    raise ValueError(f"{pid}: {field} differs from migration snapshot")

    eligible_pids = {pid for pid, counts in observed.items() if counts["model_input_eligible"]}
    tests = []
    for fold in folds:
        train, val, test = (set(fold[key]) for key in ("train", "val", "test"))
        if train & val or train & test or val & test or train | val | test != eligible_pids:
            raise ValueError(f"Invalid participant split: {fold['name']}")
        tests.extend(test)
        for role in ("train", "val", "test"):
            n = sum(observed[pid]["model_input_eligible"] for pid in fold[role])
            if n != fold["model_input_eligible_windows"][role]:
                raise ValueError(f"Fold count differs: {fold['name']}/{role}")
    if len(tests) != len(set(tests)) or set(tests) != eligible_pids:
        raise ValueError("Four-fold test participants do not cover each eligible participant exactly once")
    candidates = sum(x["candidates"] for x in observed.values())
    valid = sum(x["valid"] for x in observed.values())
    if (candidates, valid) != (manifest["candidate_windows"], manifest["valid_ecg_label_windows"]):
        raise ValueError("Dataset aggregate counts differ")
    return {"status": "quick_check_passed" if quick else "full_transfer_verification_passed",
            "dataset_dir": str(data_dir), "dataset_name": manifest["dataset_name"],
            "protocol_sha256": manifest["protocol_sha256"], "candidate_windows": candidates,
            "valid_ecg_label_windows": valid, "participants": observed,
            "hashed_files": len(hashed), "unhashed_files": skipped,
            "folds_passed": len(folds), "training_ready": False,
            "next_step": "M1 runner/source-index/denominator adaptations and fresh-cache validation"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="Skip hashes for files larger than 8 MiB; not a full transfer verification")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify(args.data_dir, args.quick)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
