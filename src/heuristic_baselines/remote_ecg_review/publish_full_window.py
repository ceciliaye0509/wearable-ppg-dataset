"""Install the additive five-minute packet; retain service credentials in memory."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
from http.cookies import SimpleCookie
import http.client
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time

from serve_review import load_packet

APP = Path(__file__).resolve().parent
REPO = APP.parents[2]
DATA = APP.parent / "outputs/ecg_rpeak_blind_review_remote_v1"
SOURCE = DATA.parent / "full_window_hrv_validation_20261029_v1"
LEGACY_TABLES = ("annotations", "annotation_revisions", "algorithm_assistance_views")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(path):
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        return {name: [dict(row) for row in db.execute(f"SELECT * FROM {name} ORDER BY 1,2")]
                for name in LEGACY_TABLES}


def request(method, path, cookie=None, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", 8765, timeout=15)
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    connection.request(method, path, json.dumps(body) if body is not None else None, headers)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def listener():
    return subprocess.run(["/usr/sbin/lsof", "-nP", "-iTCP:8765", "-sTCP:LISTEN", "-t"],
                          capture_output=True, text=True).stdout.split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    target = DATA / "supplements/supplement_full300_v1.json.gz"
    if target.exists():
        raise RuntimeError("Full-window packet already installed; do not duplicate or overwrite it")
    if listener() != [str(args.pid)]:
        raise RuntimeError("Expected local review listener changed; service untouched")
    command = subprocess.run(["/bin/ps", "-p", str(args.pid), "-o", "command="],
                             capture_output=True, text=True, check=True).stdout
    argv = shlex.split(command)
    index = next((i for i, value in enumerate(argv) if value.endswith("/serve_review.py")), -1)
    if index not in (1, 2) or Path(argv[index]).resolve() != APP / "serve_review.py" or argv[1:index] not in ([], ["-B"]):
        raise RuntimeError("Unexpected process; service untouched")
    source = SOURCE / target.name
    audit = json.loads((SOURCE / "selection_audit_private.json").read_text())
    if sha(source) != audit["packet_sha256"]:
        raise RuntimeError("Candidate packet hash mismatch")
    old = load_packet(DATA / "review_data.json.gz")
    with tempfile.TemporaryDirectory(prefix="ecg-full300-preflight-") as temporary:
        root = Path(temporary)
        for name in ("review_data.json.gz", "supplement_peak_assistance.json"):
            (root / name).symlink_to(DATA / name)
        (root / "supplements").mkdir()
        for path in (DATA / "supplements").glob("*.json.gz"):
            (root / "supplements" / path.name).symlink_to(path)
        (root / "supplements" / source.name).symlink_to(source)
        prepared = load_packet(root / "review_data.json.gz")
    prior = {c["id"]: c for c in old["clips"]}
    added = [c for c in prepared["clips"] if c["id"] not in prior]
    if len(added) != 20 or len({c["participant"] for c in added}) != 20:
        raise RuntimeError("Incomplete twenty-participant five-minute packet")
    if any(c != prior[c["id"]] for c in prepared["clips"] if c["id"] in prior):
        raise RuntimeError("Prior packet content changed")
    originals = [DATA / "review_data.json.gz", DATA / "supplement_peak_assistance.json",
                 *sorted((DATA / "supplements").glob("*.json.gz")),
                 APP.parent / "outputs/ecg_polarity_correction_all20_v1/correction_intervals.csv"]
    hashes = {str(path): sha(path) for path in originals}
    environment_result = subprocess.run(["/bin/ps", "eww", "-p", str(args.pid), "-o", "command="], capture_output=True, text=True)
    credentials = {}
    for name in ("ECG_REVIEW_ACCESS_CODE", "ECG_REVIEW_SESSION_SECRET"):
        match = re.search(r"(?:^|\s)" + name + r"=(\S+)", environment_result.stdout)
        if environment_result.returncode or not match:
            raise RuntimeError("Existing service credentials unavailable; service untouched")
        credentials[name] = match.group(1)
    del environment_result
    status, headers, _ = request("POST", "/api/login", body={"reviewer": "Deployment verification", "accessCode": credentials["ECG_REVIEW_ACCESS_CODE"]})
    if status != 200:
        raise RuntimeError("Existing credentials did not verify; service untouched")
    cookies = SimpleCookie(); cookies.load(headers["Set-Cookie"])
    cookie = "ecg_review_session=" + cookies["ecg_review_session"].value
    status, _, body = request("GET", "/api/participants", cookie)
    if status != 200:
        raise RuntimeError("Existing progress API unavailable")
    previous_progress = json.loads(body)
    environment = dict(os.environ); environment.update(credentials)
    launch = argv if index == 2 else [argv[0], "-B", *argv[1:]]
    log_path = SOURCE / "service.log"
    def start():
        with log_path.open("ab") as log:
            log_path.chmod(0o600)
            return subprocess.Popen(launch, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    os.kill(args.pid, signal.SIGINT)
    for _ in range(80):
        if not listener():
            break
        time.sleep(.1)
    else:
        raise RuntimeError("Old service did not release the port; no second service started")
    database = DATA / "annotations.sqlite3"
    try:
        before = snapshot(database)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = DATA / "backups" / f"annotations_before_full300_{stamp}.sqlite3"
        with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as db, closing(sqlite3.connect(backup)) as dest:
            db.backup(dest)
        backup.chmod(0o600)
        if snapshot(backup) != before:
            raise RuntimeError("Backup verification failed")
        shutil.copyfile(source, target); target.chmod(0o600)
    except Exception:
        if target.exists():
            target.rename(target.with_name(target.name + ".disabled"))
        start()
        raise
    child = None
    try:
        child = start()
        live = None
        for _ in range(100):
            if child.poll() is not None:
                raise RuntimeError("Updated service exited")
            try:
                status, _, body = request("GET", "/api/participants", cookie)
                if status == 200:
                    live = json.loads(body); break
            except OSError:
                pass
            time.sleep(.1)
        if live is None:
            raise RuntimeError("Updated service did not become ready")
        if sum(p["total"] for p in live["participants"]) != len(prepared["clips"]):
            raise RuntimeError("Live packet count mismatch")
        for previous in previous_progress["participants"]:
            current = next(p for p in live["participants"] if p["participant"] == previous["participant"])
            if any(current["batches"][bid] != value for bid, value in previous["batches"].items()):
                raise RuntimeError("Prior batch progress changed")
            if current["batches"]["full_window_hrv_v1"] != {"total": 1, "saved": 0, "finished": 0, "label": "Not started",
                "qualityFlagged": sum(bool(c.get("qualityFlags")) for c in added if c["participant"] == current["participant"])}:
                raise RuntimeError("Initial full-window progress mismatch")
        for clip in added:
            status, _, body = request("GET", "/api/clip/" + clip["id"], cookie)
            value = json.loads(body)
            if (status != 200 or value["t"] != clip["t"] or value["x"] != clip["x"]
                    or value["peaks"] != clip["suggested"] or value["reviewedPages"] or value["unreadableRanges"]
                    or value["status"] != "unreviewed" or value["annotationScope"] != "reviewer"):
                raise RuntimeError("Full-window API identity mismatch")
        for row in before["annotations"]:
            status, _, body = request("GET", "/api/clip/" + row["clip_id"], cookie)
            value = json.loads(body)
            if (status != 200 or value["peaks"] != json.loads(row["peaks_json"])
                    or value["unreadableRanges"] != json.loads(row["unreadable_ranges_json"])
                    or value["version"] != row["version"] or value["status"] != row["status"]):
                raise RuntimeError("Previously saved annotation API changed")
        if snapshot(database) != before or any(sha(Path(path)) != digest for path, digest in hashes.items()):
            raise RuntimeError("Preservation verification failed; no database overwrite attempted")
        with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as db:
            if db.execute("SELECT COUNT(*) FROM full_window_annotations").fetchone()[0] != 0:
                raise RuntimeError("Unexpected saved full-window annotations")
        report = {"status": "live", "url": "http://127.0.0.1:8765/", "pid": child.pid,
                  "verified_at_utc": datetime.now(timezone.utc).isoformat(), "backup": str(backup),
                  "new_windows": len(added), "total_clips": len(prepared["clips"]),
                  "all_existing_saved_apis_verified": len(before["annotations"]),
                  "existing_peak_count": sum(len(json.loads(r["peaks_json"])) for r in before["annotations"]),
                  "existing_unreadable_intervals": sum(len(json.loads(r["unreadable_ranges_json"])) for r in before["annotations"]),
                  "legacy_annotation_revision_and_assistance_rows_unchanged": True,
                  "existing_credentials_and_binding_preserved": True, "existing_packet_hashes": hashes,
                  "new_packet_sha256": sha(target)}
        with (SOURCE / "deployment_report.json").open("x") as handle:
            json.dump(report, handle, indent=2); handle.write("\n")
        (SOURCE / "deployment_report.json").chmod(0o600)
        print(json.dumps(report, indent=2))
    except Exception:
        if child and child.poll() is None:
            child.send_signal(signal.SIGINT); child.wait(timeout=10)
        if target.exists():
            target.rename(target.with_name(target.name + ".disabled"))
        restored = start()
        print(json.dumps({"status": "prior_packet_service_restarted", "pid": restored.pid,
                          "database_overwritten": False}))
        raise


if __name__ == "__main__":
    main()
