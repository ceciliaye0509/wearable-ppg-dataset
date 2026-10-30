"""Blind batch, additive packet and existing annotation regression tests."""
import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
import test_intervals as interval_tests
from serve_review import load_packet, make_handler


class SupplementAPITests(interval_tests.IntervalReviewTests):
    def setUp(self):
        super().setUp()
        self.legacy = {**self.clip, "id": "legacy", "reviewBatch": "existing"}
        self.blind = {**self.clip, "id": "test_clip", "reviewSet": "independent_v1",
                      "blindReview": True, "reviewBatch": "independent_blind_v1",
                      "annotationMode": "manual_blind", "suggested": [], "restorePeaks": [],
                      "aidAvailable": False, "sourceLabel": "Independent blind validation"}
        packet = {"participants": [{"participant": "P1"}], "clips": [self.legacy, self.blind],
                  "reviewBatches": [{"id": "existing", "label": "Existing review", "target": 1},
                                    {"id": "independent_blind_v1", "label": "Independent blind validation", "target": 1}]}
        handler = make_handler(packet, self.db, "test-access-code", b"test-session-secret-32-characters", False)
        handler.log_message = lambda *args: None
        self.server.RequestHandlerClass = handler

    # Inherited legacy-only tests run in their original test class. The three
    # blind tests below exercise this separate packet and database.
    def test_blind_empty_canvas_withholds_screening_and_restores_saved_only(self):
        value = self.load()
        self.assertEqual(value["peaks"], [])
        self.assertEqual(value["suggested"], [])
        self.assertEqual(value["restorePeaks"], [])
        self.assertEqual(value["qualityFlags"], [])
        self.assertEqual(value["qualityRule"], {})
        self.assertTrue(value["qualitySummary"]["withheldForBlindReview"])
        self.assertEqual(self.save(0, [2, 8], ranges=[[3, 6]])[0], 200)
        value = self.load()
        self.assertEqual(value["peaks"], [2, 8])
        self.assertEqual(value["unreadableRanges"], [[3, 6]])
        self.assertEqual(value["restorePeaks"], [])

    def test_batch_counts_and_export_keep_legacy_records(self):
        payload = {"version": 0, "peaks": [1, 5], "status": "complete", "confidence": "high", "notes": "original", "unreadableRanges": []}
        self.assertEqual(self.request("POST", "/api/clip/legacy/annotation", payload)[0], 200)
        _, _, body = self.request("GET", "/api/participants")
        progress = json.loads(body)["participants"][0]["batches"]
        self.assertEqual(progress["existing"]["finished"], 1)
        self.assertEqual(progress["independent_blind_v1"]["saved"], 0)
        self.assertEqual(self.save(0, [2, 8], "complete", [[3, 6]])[0], 200)
        _, _, body = self.request("GET", "/api/participants")
        self.assertEqual(json.loads(body)["participants"][0]["batches"]["independent_blind_v1"]["finished"], 1)
        _, _, body = self.request("GET", "/api/export.csv")
        rows = list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))
        self.assertEqual(sum(r["clip_id"] == "legacy" for r in rows), 2)
        self.assertEqual(sum(r["clip_id"] == "test_clip" and r["record_type"] == "peak" for r in rows), 2)
        self.assertTrue(all(r["annotation_mode"] == "manual_blind" for r in rows if r["clip_id"] == "test_clip"))

    def test_blind_clip_list_does_not_expose_screening_judgments(self):
        _, _, body = self.request("GET", "/api/participant/P1/clips")
        clip = next(c for c in json.loads(body)["clips"] if c["blindReview"])
        self.assertEqual(clip["peakCount"], 0)
        self.assertEqual(clip["qualityWarningCount"], 0)
        self.assertEqual(clip["qualityWarningLabels"], [])


# Avoid re-running the inherited assisted-only assertions on a blind packet.
for name in list(interval_tests.IntervalReviewTests.__dict__):
    if name.startswith("test_"):
        setattr(SupplementAPITests, name, None)


class SupplementPacketTests(unittest.TestCase):
    def test_append_is_in_memory_only_and_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            base = {"version": 2, "participants": [{"participant": "P1"}], "clips": [{"id": "old", "participant": "P1"}]}
            path = folder / "review_data.json.gz"
            with gzip.open(path, "wt") as handle:
                json.dump(base, handle)
            original = path.read_bytes()
            extras = folder / "supplements"; extras.mkdir()
            addition = {"version": 2, "reviewBatches": [{"id": "independent_blind_v1", "label": "Blind", "target": 1}],
                        "clips": [{"id": "new", "participant": "P1", "blindReview": True, "suggested": [], "restorePeaks": [], "aidAvailable": False}]}
            with gzip.open(extras / "new.json.gz", "wt") as handle:
                json.dump(addition, handle)
            self.assertEqual([c["id"] for c in load_packet(path)["clips"]], ["old", "new"])
            self.assertEqual(path.read_bytes(), original)
            addition["clips"][0]["id"] = "old"
            with gzip.open(extras / "new.json.gz", "wt") as handle:
                json.dump(addition, handle)
            with self.assertRaises(ValueError):
                load_packet(path)
            self.assertEqual(path.read_bytes(), original)

    def test_blind_supplement_with_candidate_peaks_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = folder / "review_data.json.gz"
            with gzip.open(path, "wt") as handle:
                json.dump({"version": 2, "participants": [{"participant": "P1"}], "clips": []}, handle)
            extras = folder / "supplements"; extras.mkdir()
            with gzip.open(extras / "bad.json.gz", "wt") as handle:
                json.dump({"version": 2, "clips": [{"id": "bad", "participant": "P1", "blindReview": True, "suggested": [2]}]}, handle)
            with self.assertRaises(ValueError):
                load_packet(path)


if __name__ == "__main__":
    unittest.main()
