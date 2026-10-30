"""Regression checks for upgrading saved marks to interval annotations."""
import csv
from contextlib import closing
import http.client
import io
import json
import sqlite3
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve_review import initialize_database, make_handler


class IntervalReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ecg-interval-test-")
        self.db = Path(self.temp.name) / "annotations.sqlite3"
        initialize_database(self.db)
        self.clip = {
            "id": "test_clip", "participant": "P1", "reviewId": "test_window",
            "reviewSet": "test", "windowStart": 100000, "clipStart": 100000,
            "startSec": 0, "endSec": 1, "t": [i / 10 for i in range(11)],
            "x": list(range(11)), "suggested": [1, 3, 5, 7, 9],
            "restorePeaks": [1, 3, 5, 7, 9], "aidAvailable": True,
            "qualityFlags": [{"code": "ambiguous_polarity", "label": "Unclear peak direction", "startIndex": 2, "endIndex": 6}],
            "qualitySummary": {"available": True, "requiresReview": True, "flagCount": 1},
        }
        packet = {"participants": [{"participant": "P1"}], "clips": [self.clip]}
        handler = make_handler(packet, self.db, "test-access-code", b"test-session-secret-32-characters", False)
        handler.log_message = lambda *args: None
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        status, headers, body = self.request("POST", "/api/login", {"reviewer": "Test reviewer", "accessCode": "test-access-code"})
        self.assertEqual(status, 200)
        self.cookie = dict(headers)["Set-Cookie"].split(";", 1)[0]

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()
        self.temp.cleanup()

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        headers = {"Cookie": self.cookie} if hasattr(self, "cookie") else {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        connection.request(method, path, json.dumps(body) if body is not None else None, headers)
        response = connection.getresponse()
        result = response.status, response.getheaders(), response.read()
        connection.close()
        return result

    def save(self, version, peaks, status="in_progress", ranges=None):
        payload = {"version": version, "peaks": peaks, "status": status, "confidence": "high", "notes": ""}
        if ranges is not None:
            payload["unreadableRanges"] = ranges
        return self.request("POST", "/api/clip/test_clip/annotation", payload)

    def load(self):
        status, _, raw = self.request("GET", "/api/clip/test_clip")
        self.assertEqual(status, 200)
        return json.loads(raw)

    def test_interval_roundtrip_keeps_peaks_and_exports_both(self):
        peaks = [1, 3, 5, 7, 9]
        self.assertEqual(self.save(0, peaks, ranges=[[2, 4], [4, 6]])[0], 200)
        saved = self.load()
        self.assertEqual(saved["peaks"], peaks)
        self.assertEqual(saved["unreadableRanges"], [[2, 6]])
        self.assertEqual(self.save(1, peaks, "complete", [[2, 6]])[0], 200)
        self.assertEqual(self.load()["peaks"], peaks)
        status, _, raw = self.request("GET", "/api/export.csv")
        self.assertEqual(status, 200)
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        self.assertEqual(sum(r["record_type"] == "peak" for r in rows), 5)
        self.assertEqual(sum(r["record_type"] == "unreadable_interval" for r in rows), 1)
        self.assertEqual(sum(r["peak_in_unreadable_interval"] == "true" for r in rows), 2)
        self.assertEqual(sum(r["peak_in_quality_flag"] == "true" for r in rows), 2)
        self.assertTrue(all(r["quality_flag_codes"] == "ambiguous_polarity" for r in rows))

    def test_screening_flags_never_relabel_or_remove_saved_marks(self):
        loaded = self.load()
        self.assertEqual(loaded["peaks"], [1, 3, 5, 7, 9])
        self.assertEqual(loaded["unreadableRanges"], [])
        self.assertEqual(loaded["status"], "unreviewed")
        self.assertTrue(loaded["qualitySummary"]["available"])
        self.assertEqual(self.save(0, [2, 8], "complete", [])[0], 200)
        loaded = self.load()
        self.assertEqual(loaded["peaks"], [2, 8])
        self.assertEqual(loaded["unreadableRanges"], [])
        self.assertEqual(loaded["status"], "complete")
        self.assertEqual(loaded["qualityFlags"], self.clip["qualityFlags"])
        _, _, data = self.request("GET", "/api/participant/P1/clips")
        self.assertEqual(json.loads(data)["clips"][0]["qualityWarningCount"], 1)

    def test_legacy_unreadable_request_preserves_custom_saved_peaks(self):
        self.assertEqual(self.save(0, [2, 8])[0], 200)
        self.assertEqual(self.save(1, [], "unresolvable")[0], 200)
        saved = self.load()
        self.assertEqual(saved["peaks"], [2, 8])
        self.assertEqual(saved["unreadableRanges"], [[0, 10]])
        # Old clients omit the new field; updating status must keep the ranges.
        self.assertEqual(self.save(2, [2, 8], "in_progress")[0], 200)
        self.assertEqual(self.load()["unreadableRanges"], [[0, 10]])
        # The upgraded client can explicitly remove an interval.
        self.assertEqual(self.save(3, [2, 8], "complete", [])[0], 200)
        self.assertEqual(self.load()["unreadableRanges"], [])

    def test_conflict_or_invalid_range_cannot_erase_saved_data(self):
        self.assertEqual(self.save(0, [2, 8], ranges=[[3, 6]])[0], 200)
        self.assertEqual(self.save(0, [], ranges=[])[0], 409)
        self.assertEqual(self.save(1, [], ranges=[[2, 99]])[0], 400)
        saved = self.load()
        self.assertEqual(saved["peaks"], [2, 8])
        self.assertEqual(saved["unreadableRanges"], [[3, 6]])

    def test_explicit_clear_keeps_previous_saved_revision(self):
        self.assertEqual(self.save(0, [2, 8], ranges=[])[0], 200)
        self.assertEqual(self.save(1, [], ranges=[])[0], 200)
        with closing(sqlite3.connect(self.db)) as db, db:
            snapshot = json.loads(db.execute("SELECT snapshot_json FROM annotation_revisions WHERE clip_id='test_clip' AND version=1").fetchone()[0])
        self.assertEqual(json.loads(snapshot["peaks_json"]), [2, 8])


class MigrationTests(unittest.TestCase):
    def test_old_saved_rows_survive_idempotent_migration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.sqlite3"
            original = ("saved_clip", "P1", "[2, 5, 9]", "complete", "medium", "original note", "reviewer", "2026-10-23", 7)
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("CREATE TABLE annotations (clip_id TEXT PRIMARY KEY,participant TEXT NOT NULL,peaks_json TEXT NOT NULL,status TEXT NOT NULL,confidence TEXT NOT NULL,notes TEXT NOT NULL,reviewer TEXT NOT NULL,updated_at TEXT NOT NULL,version INTEGER NOT NULL)")
                db.execute("INSERT INTO annotations VALUES (?,?,?,?,?,?,?,?,?)", original)
            initialize_database(path)
            initialize_database(path)
            with closing(sqlite3.connect(path)) as db, db:
                row = db.execute("SELECT * FROM annotations").fetchone()
            self.assertEqual(row[:9], original)
            self.assertEqual(row[9], "[]")


if __name__ == "__main__":
    unittest.main()
