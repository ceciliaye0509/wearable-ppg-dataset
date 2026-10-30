"""Assistance is explicit; waveform, saved marks and provenance stay intact."""
import copy
from contextlib import closing
import csv
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import test_intervals as interval_tests
from serve_review import make_handler
from peak_assistance import apply_assistance_overlay, waveform_hash, OVERLAY_NAME


def add_overlay(packet, directory):
    overlay = {"version": 1, "method": "test_neurokit_local", "generatedAt": "2026-10-28",
               "clips": {c["id"]: {"suggested": [1, 5, 9], "waveformSha256": waveform_hash(c)}
                         for c in packet["clips"] if c.get("blindReview")}}
    path = Path(directory) / OVERLAY_NAME
    path.write_text(json.dumps(overlay))
    apply_assistance_overlay(packet, Path(directory))
    return path


class AssistedSupplementAPITests(interval_tests.IntervalReviewTests):
    def setUp(self):
        super().setUp()
        self.blind = {**self.clip, "reviewBatch": "independent_blind_v1", "blindReview": True,
                      "annotationMode": "manual_blind", "suggested": [], "restorePeaks": [], "aidAvailable": False}
        self.challenge = {**self.blind, "id": "challenge", "reviewBatch": "challenge_blind_v1"}
        self.legacy = {**self.clip, "id": "legacy"}
        self.original = {"participants": [{"participant": "P1"}], "clips": [self.legacy, self.blind, self.challenge],
                         "reviewBatches": [{"id": "existing", "label": "Existing review"},
                                           {"id": "independent_blind_v1", "label": "Independent blind validation"},
                                           {"id": "challenge_blind_v1", "label": "Difficult / rule-boundary audit"}]}
        self.packet = copy.deepcopy(self.original)
        add_overlay(self.packet, self.temp.name)
        self.install_handler(self.packet)

    def install_handler(self, packet):
        handler = make_handler(packet, self.db, "test-access-code", b"test-session-secret-32-characters", False)
        handler.log_message = lambda *args: None
        self.server.RequestHandlerClass = handler

    def test_both_batches_have_editable_suggestions_but_no_screening_answers(self):
        for clip_id in ["test_clip", "challenge"]:
            status, _, raw = self.request("GET", "/api/clip/" + clip_id)
            self.assertEqual(status, 200)
            clip = json.loads(raw)
            self.assertEqual(clip["peaks"], [1, 5, 9])
            self.assertEqual(clip["restorePeaks"], [1, 5, 9])
            self.assertFalse(clip["blindReview"])
            self.assertTrue(clip["wasBlindReview"])
            self.assertEqual(clip["annotationMode"], "algorithm_assisted")
            self.assertEqual(clip["qualityFlags"], [])
            self.assertEqual(clip["qualityRule"], {})
            self.assertEqual(clip["t"], self.clip["t"])
            self.assertEqual(clip["x"], self.clip["x"])
            self.assertEqual(self.request("POST", "/api/clip/" + clip_id + "/assistance-seen", {})[0], 200)
        with closing(sqlite3.connect(self.db)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM annotations").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM algorithm_assistance_views").fetchone()[0], 2)

    def test_saved_peaks_intervals_empty_lists_and_revision_history_win(self):
        self.assertEqual(self.save(0, [2, 8], "complete", [[3, 6]])[0], 200)
        loaded = self.load()
        self.assertEqual(loaded["peaks"], [2, 8])
        self.assertEqual(loaded["unreadableRanges"], [[3, 6]])
        self.assertEqual(loaded["restorePeaks"], [1, 5, 9])
        self.assertEqual(self.save(1, [], "in_progress", [[3, 6]])[0], 200)
        self.assertEqual(self.load()["peaks"], [])
        self.assertEqual(self.load()["unreadableRanges"], [[3, 6]])
        self.assertEqual(self.save(1, [4], "complete", [])[0], 409)
        with closing(sqlite3.connect(self.db)) as db:
            before = json.loads(db.execute("SELECT snapshot_json FROM annotation_revisions").fetchone()[0])
        self.assertEqual(json.loads(before["peaks_json"]), [2, 8])

    def test_export_tracks_exposure_even_after_overlay_removed(self):
        self.load()
        self.assertEqual(self.save(0, [2, 8], "complete", [[3, 6]])[0], 200)
        self.install_handler(self.original)
        _, _, raw = self.request("GET", "/api/export.csv")
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        self.assertEqual(sum(r["record_type"] == "peak" for r in rows), 2)
        self.assertTrue(all(r["annotation_mode"] == "algorithm_assisted" for r in rows))
        self.assertTrue(all(r["algorithm_assistance_seen"] == "true" for r in rows))
        self.assertTrue(all(r["algorithm_assistance_method"] == "test_neurokit_local" for r in rows))
        self.assertTrue(all(len(r["algorithm_assistance_overlay_sha256"]) == 64 for r in rows))

    def test_prior_annotations_remain_identical_and_progress_is_separate(self):
        body = {"peaks": [2, 8], "status": "complete", "confidence": "medium", "notes": "keep",
                "version": 0, "unreadableRanges": [[3, 6]]}
        self.assertEqual(self.request("POST", "/api/clip/legacy/annotation", body)[0], 200)
        with closing(sqlite3.connect(self.db)) as db:
            before = db.execute("SELECT * FROM annotations WHERE clip_id='legacy'").fetchone()
        self.load()
        _, _, raw = self.request("GET", "/api/participants")
        progress = json.loads(raw)["participants"][0]["batches"]
        self.assertEqual(progress["existing"]["finished"], 1)
        self.assertEqual(progress["independent_blind_v1"]["saved"], 0)
        with closing(sqlite3.connect(self.db)) as db:
            after = db.execute("SELECT * FROM annotations WHERE clip_id='legacy'").fetchone()
        self.assertEqual(before, after)


for name in list(interval_tests.IntervalReviewTests.__dict__):
    if name.startswith("test_"):
        setattr(AssistedSupplementAPITests, name, None)


class OverlayValidationTests(unittest.TestCase):
    def test_corrupt_or_wrong_identity_overlay_is_rejected(self):
        clip = {"id": "c", "reviewBatch": "independent_blind_v1", "blindReview": True,
                "t": [0, .1, .2], "x": [0, 1, 0]}
        packet = {"clips": [clip]}
        with tempfile.TemporaryDirectory() as directory:
            path = add_overlay({"clips": [{**clip, "t": list(range(11)), "x": list(range(11))}]}, directory)
            with self.assertRaises(ValueError):
                apply_assistance_overlay(copy.deepcopy(packet), Path(directory))
            value = json.loads(path.read_text())
            value["clips"]["c"]["waveformSha256"] = waveform_hash(clip)
            value["clips"]["c"]["suggested"] = [99]
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                apply_assistance_overlay(copy.deepcopy(packet), Path(directory))


if __name__ == "__main__":
    unittest.main()
