"""New assisted batches must preserve saved marks and reject corrupt packets."""
import copy
from contextlib import closing
import gzip
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import test_intervals as interval_tests
from peak_assistance import waveform_hash
from serve_review import load_packet, make_handler


def assisted_packet(clip):
    clip = {**copy.deepcopy(clip), "reviewBatch": "peak_confirm_test",
            "blindReview": False, "annotationMode": "algorithm_assisted"}
    clip["assistanceWaveformSha256"] = waveform_hash(clip)
    return {"version": 2, "reviewProtocolKind": "assisted_peak_validation_v1",
            "assistanceProtocol": {"version": 1,
                "method": "neurokit_local_qrs_extreme_50ms_full300s", "generatedAt": "2026-10-29"},
            "reviewBatches": [{"id": "peak_confirm_test", "label": "Peak confirmation", "target": 1}],
            "clips": [clip]}


def write_packet(path, value):
    with gzip.open(path, "wt") as handle:
        json.dump(value, handle)


class AssistedAdditionPacketTests(unittest.TestCase):
    def test_identity_protocol_indices_and_batch_counts_are_enforced(self):
        clip = {"id": "new", "participant": "P1", "t": [0, .1, .2], "x": [0, 1, 0],
                "suggested": [1], "restorePeaks": [1], "aidAvailable": True}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            base = folder / "review_data.json.gz"
            write_packet(base, {"version": 2, "participants": [{"participant": "P1"}],
                                "clips": [{"id": "old", "participant": "P1"}]})
            original = base.read_bytes()
            supplements = folder / "supplements"
            supplements.mkdir()
            addition = supplements / "new.json.gz"
            good = assisted_packet(clip)
            write_packet(addition, good)
            loaded = load_packet(base)
            self.assertEqual([c["id"] for c in loaded["clips"]], ["old", "new"])
            self.assertEqual(loaded["clips"][1]["restorePeaks"], [1])
            self.assertEqual(len(loaded["clips"][1]["assistance"]["overlaySha256"]), 64)
            mutations = [
                lambda p: p.pop("reviewProtocolKind"),
                lambda p: p["assistanceProtocol"].update(method="unknown"),
                lambda p: p["clips"][0]["x"].__setitem__(0, 99),
                lambda p: p["clips"][0].update(suggested=[99], restorePeaks=[99]),
                lambda p: p["clips"][0].update(suggested=[True], restorePeaks=[True]),
                lambda p: p["reviewBatches"][0].update(target=2),
                lambda p: p["clips"][0].update(priorUnreadableRanges=[[0, 99]]),
            ]
            for mutate in mutations:
                with self.subTest(mutation=mutations.index(mutate)):
                    bad = copy.deepcopy(good)
                    mutate(bad)
                    write_packet(addition, bad)
                    with self.assertRaises(ValueError):
                        load_packet(base)
                    self.assertEqual(base.read_bytes(), original)


class AssistedAdditionAPITests(interval_tests.IntervalReviewTests):
    def setUp(self):
        super().setUp()
        folder = Path(self.temp.name)
        path = folder / "review_data.json.gz"
        write_packet(path, {"version": 2, "participants": [{"participant": "P1"}],
                            "clips": [{**self.clip, "id": "legacy", "reviewBatch": "existing"}]})
        (folder / "supplements").mkdir()
        write_packet(folder / "supplements/new.json.gz", assisted_packet(self.clip))
        handler = make_handler(load_packet(path), self.db, "test-access-code",
                               b"test-session-secret-32-characters", False)
        handler.log_message = lambda *args: None
        self.server.RequestHandlerClass = handler

    def test_human_recheck_preserves_initial_ranges_but_respects_saved_clearing(self):
        clip = {**self.clip, "isPrevious": True, "priorUnreadableRanges": [[3, 6]]}
        packet = {"participants": [{"participant": "P1"}], "clips": [clip]}
        handler = make_handler(packet, self.db, "test-access-code",
                               b"test-session-secret-32-characters", False)
        handler.log_message = lambda *args: None
        self.server.RequestHandlerClass = handler
        self.assertEqual(self.load()["unreadableRanges"], [[3, 6]])
        self.assertEqual(self.load()["peaks"], self.clip["suggested"])
        self.assertEqual(self.save(0, [2, 8], "complete", [])[0], 200)
        self.assertEqual(self.load()["unreadableRanges"], [])
        self.assertEqual(self.load()["peaks"], [2, 8])

    def test_new_batch_save_clear_and_unreadable_preserve_old_marks(self):
        payload = {"version": 0, "peaks": [2, 8], "status": "complete",
                   "confidence": "high", "notes": "keep", "unreadableRanges": [[3, 6]]}
        self.assertEqual(self.request("POST", "/api/clip/legacy/annotation", payload)[0], 200)
        with closing(sqlite3.connect(self.db)) as db:
            before = db.execute("SELECT * FROM annotations WHERE clip_id='legacy'").fetchone()
        fresh = self.load()
        self.assertEqual(fresh["version"], 0)
        self.assertEqual(fresh["status"], "unreviewed")
        self.assertEqual(fresh["peaks"], [1, 3, 5, 7, 9])
        self.assertEqual(self.save(0, [2, 8], "complete", [[3, 6]])[0], 200)
        self.assertEqual(self.save(1, [], "in_progress", [[3, 6]])[0], 200)
        self.assertEqual(self.load()["peaks"], [])
        self.assertEqual(self.load()["unreadableRanges"], [[3, 6]])
        self.assertEqual(self.load()["restorePeaks"], [1, 3, 5, 7, 9])
        self.assertEqual(self.save(2, [2, 8], "unresolvable", [[3, 6]])[0], 200)
        self.assertEqual(self.load()["peaks"], [2, 8])
        _, _, raw = self.request("GET", "/api/participants")
        batches = json.loads(raw)["participants"][0]["batches"]
        self.assertEqual(batches["existing"]["finished"], 1)
        self.assertEqual(batches["peak_confirm_test"]["saved"], 1)
        with closing(sqlite3.connect(self.db)) as db:
            self.assertEqual(db.execute("SELECT * FROM annotations WHERE clip_id='legacy'").fetchone(), before)
            snapshot = json.loads(db.execute(
                "SELECT snapshot_json FROM annotation_revisions WHERE clip_id='test_clip' AND version=1").fetchone()[0])
        self.assertEqual(json.loads(snapshot["peaks_json"]), [2, 8])


# Exercise only the new packet above; the original fixture covers its own tests.
for name in list(interval_tests.IntervalReviewTests.__dict__):
    if name.startswith("test_"):
        setattr(AssistedAdditionAPITests, name, None)


if __name__ == "__main__":
    unittest.main()
