"""Full-window isolation, completion coverage, exports and legacy regression."""
import copy
import csv
import gzip
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import test_intervals as shared
from full_window import KIND, METHOD, prepare_full_window_supplement
from peak_assistance import waveform_hash
from serve_review import load_packet, make_handler


def clip_fixture():
    return {"id": "test_clip", "participant": "P1", "reviewId": "full_window",
            "reviewSet": KIND, "reviewProtocolKind": KIND, "reviewBatch": "full_window_hrv_v1",
            "windowStart": 100000, "clipStart": 100000, "startSec": 0, "endSec": 299.9,
            "windowDurationSec": 300, "reviewPageSeconds": 5, "annotationScope": "reviewer",
            "blindReview": False, "annotationMode": "algorithm_assisted", "samplingGaps": [],
            "t": [i / 10 for i in range(3000)], "x": [0.] * 3000,
            "suggested": [2, 8], "restorePeaks": [2, 8], "aidAvailable": True}


class FullWindowAPITests(shared.IntervalReviewTests):
    def setUp(self):
        super().setUp()
        self.legacy = {**self.clip, "id": "legacy"}
        self.clip = clip_fixture()
        packet = {"participants": [{"participant": "P1"}], "clips": [self.legacy, self.clip],
                  "reviewBatches": [{"id": "existing", "label": "Existing", "target": 1},
                                    {"id": "full_window_hrv_v1", "label": "Full five minutes", "target": 1}]}
        handler = make_handler(packet, self.db, "test-access-code", b"test-session-secret-32-characters", False)
        handler.log_message = lambda *args: None
        self.server.RequestHandlerClass = handler

    def save_full(self, version=0, peaks=None, pages=None, status="in_progress", ranges=None):
        return self.request("POST", "/api/clip/test_clip/annotation", {
            "version": version, "peaks": [2, 8] if peaks is None else peaks, "reviewedPages": pages or [],
            "unreadableRanges": ranges or [], "status": status, "confidence": "high", "notes": ""})

    def login(self, name):
        status, headers, _ = self.request("POST", "/api/login", {"reviewer": name, "accessCode": "test-access-code"})
        self.assertEqual(status, 200)
        self.cookie = dict(headers)["Set-Cookie"].split(";", 1)[0]

    def test_two_reviewers_resume_distinct_peaks_masks_pages_and_revisions(self):
        self.assertEqual(self.save_full(peaks=[4, 6], pages=[0, 1], ranges=[[3, 7]])[0], 200)
        alice = self.cookie
        self.login("Second reviewer")
        self.assertEqual(self.load()["peaks"], [2, 8])
        self.assertEqual(self.load()["reviewedPages"], [])
        self.assertEqual(self.load()["unreadableRanges"], [])
        self.assertEqual(self.load()["version"], 0)
        self.assertEqual(self.save_full(peaks=[5], pages=[2], ranges=[[9, 12]])[0], 200)
        bob = self.cookie
        self.cookie = alice
        value = self.load()
        self.assertEqual((value["peaks"], value["unreadableRanges"], value["reviewedPages"]), ([4, 6], [[3, 7]], [0, 1]))
        self.assertEqual({r["reviewer"] for r in value["reviewerProgress"]}, {"Test reviewer", "Second reviewer"})
        self.assertEqual(self.save_full(version=0, peaks=[])[0], 409)
        self.assertEqual(self.save_full(version=1, peaks=[], pages=[0, 1], ranges=[[3, 7]])[0], 200)
        with sqlite3.connect(self.db) as db:
            snapshot = json.loads(db.execute("SELECT snapshot_json FROM full_window_revisions").fetchone()[0])
            self.assertEqual(json.loads(snapshot["peaks_json"]), [4, 6])
            self.assertEqual(db.execute("SELECT COUNT(*) FROM annotations").fetchone()[0], 0)
        self.cookie = bob
        self.assertEqual(self.load()["peaks"], [5])

    def test_cannot_complete_without_all_sixty_pages_or_spoof_coverage(self):
        self.assertEqual(self.save_full(pages=[0, 1], status="complete")[0], 400)
        self.assertEqual(self.save_full(pages=[60])[0], 400)
        self.assertEqual(self.save_full(pages=[True])[0], 400)
        self.assertEqual(self.save_full(pages=[1.5])[0], 400)
        self.assertEqual(self.save_full(pages=[0] * 60, status="complete")[0], 400)
        self.assertEqual(self.save_full(pages=list(range(60)), status="complete")[0], 200)
        self.assertEqual(self.load()["status"], "complete")
        self.assertEqual(len(self.load()["reviewedPages"]), 60)

    def test_own_progress_default_export_all_export_and_legacy_preservation(self):
        payload = {"version": 0, "peaks": [1, 5], "unreadableRanges": [[2, 4]],
                   "status": "complete", "confidence": "high", "notes": "old saved data"}
        self.assertEqual(self.request("POST", "/api/clip/legacy/annotation", payload)[0], 200)
        self.assertEqual(self.save_full(pages=list(range(60)), status="complete", ranges=[[1, 3]])[0], 200)
        alice = self.cookie
        self.login("Second reviewer")
        _, _, raw = self.request("GET", "/api/participants")
        progress = json.loads(raw)["participants"][0]["batches"]
        self.assertEqual(progress["existing"]["finished"], 1)
        self.assertEqual(progress["full_window_hrv_v1"]["saved"], 0)
        self.assertEqual(self.save_full(peaks=[5], pages=[0])[0], 200)
        _, _, raw = self.request("GET", "/api/export.csv")
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        self.assertEqual({r["reviewer_id"] for r in rows if r["annotation_scope"] == "reviewer"}, {"Second reviewer"})
        self.assertEqual(sum(r["clip_id"] == "legacy" for r in rows), 3)
        _, _, raw = self.request("GET", "/api/full-window-export.csv")
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        self.assertEqual({r["reviewer_id"] for r in rows}, {"Test reviewer", "Second reviewer"})
        self.assertTrue(all(r["clip_id"] == "test_clip" and r["window_duration_sec"] == "300" for r in rows))
        self.assertEqual(sum(r["peak_in_unreadable_interval"] == "true" for r in rows), 1)
        self.cookie = alice
        self.assertEqual(self.load()["peaks"], [2, 8])

    def test_unreadable_legacy_action_does_not_clear_peaks(self):
        self.assertEqual(self.save_full(peaks=[4, 6], pages=list(range(60)))[0], 200)
        self.assertEqual(self.save(1, [], "unresolvable")[0], 200)
        self.assertEqual(self.load()["peaks"], [4, 6])
        self.assertEqual(self.load()["unreadableRanges"], [[0, 2999]])


for name in shared.IntervalReviewTests.__dict__:
    if name.startswith("test_"):
        setattr(FullWindowAPITests, name, None)


class FullWindowPacketTests(unittest.TestCase):
    def packet(self):
        clip = clip_fixture()
        clip["assistanceWaveformSha256"] = waveform_hash(clip)
        return {"version": 2, "reviewProtocolKind": KIND, "clips": [clip],
                "reviewBatches": [{"id": "full_window_hrv_v1", "label": "Five minutes", "target": 1}],
                "assistanceProtocol": {"version": 1, "method": METHOD, "generatedAt": "2026-10-29"}}

    def test_full_context_hash_duration_and_gap_validation(self):
        packet = self.packet()
        prepare_full_window_supplement(copy.deepcopy(packet), "packetsha")
        for change in (lambda c: c.update(x=[9.] * 3000), lambda c: c.update(t=[v / 15 for v in c["t"]]),
                       lambda c: c.update(annotationScope="shared"), lambda c: c.update(samplingGaps=[[0, 1]])):
            altered = copy.deepcopy(packet)
            change(altered["clips"][0])
            with self.assertRaises(ValueError):
                prepare_full_window_supplement(altered, "packetsha")

    def test_full_packet_load_is_additive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "review_data.json.gz"
            with gzip.open(path, "wt") as handle:
                json.dump({"version": 2, "participants": [{"participant": "P1"}],
                           "clips": [{"id": "legacy", "participant": "P1"}]}, handle)
            original = path.read_bytes()
            (root / "supplements").mkdir()
            with gzip.open(root / "supplements/full.json.gz", "wt") as handle:
                json.dump(self.packet(), handle)
            result = load_packet(path)
            self.assertEqual([c["id"] for c in result["clips"]], ["legacy", "test_clip"])
            self.assertEqual(result["clips"][1]["assistance"]["method"], METHOD)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
