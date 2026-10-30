"""Serve the shared, password-protected ECG review application."""
from __future__ import annotations

import argparse
import base64
import csv
from contextlib import closing
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
import ssl
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

try:
    from .peak_assistance import apply_assistance_overlay, screening_withheld, prepare_assisted_supplement
    from . import full_window
except ImportError:  # direct script entry point
    from peak_assistance import apply_assistance_overlay, screening_withheld, prepare_assisted_supplement
    import full_window


APP_DIR = Path(__file__).resolve().parent
BASELINE_DIR = APP_DIR.parent
DEFAULT_DATA = BASELINE_DIR / "outputs" / "ecg_rpeak_blind_review_remote_v1" / "review_data.json.gz"
DEFAULT_DB = BASELINE_DIR / "outputs" / "ecg_rpeak_blind_review_remote_v1" / "annotations.sqlite3"
COOKIE_NAME = "ecg_review_session"
SESSION_SECONDS = 8 * 60 * 60
MAX_BODY_BYTES = 2_000_000
ALLOWED_STATUSES = {"in_progress", "complete", "unresolvable"}


def load_packet(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Review packet not found: {path}. Run build_review_packet.py first.")
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        packet = json.load(handle)
    if packet.get("version") not in {1, 2} or not isinstance(packet.get("clips"), list):
        raise ValueError("Unsupported or malformed review packet")
    known_ids = {clip["id"] for clip in packet["clips"]}
    known_participants = {row["participant"] for row in packet["participants"]}
    packet.setdefault("reviewBatches", [{"id": "existing", "label": "Existing review", "target": len(packet["clips"])}])
    for addition in sorted((path.parent / "supplements").glob("*.json.gz")):
        with gzip.open(addition, "rt", encoding="utf-8") as handle:
            extra = json.load(handle)
        if extra.get("version") != 2 or not isinstance(extra.get("clips"), list):
            raise ValueError(f"Malformed supplement: {addition.name}")
        full = extra.get("reviewProtocolKind") == full_window.KIND
        assisted = extra.get("reviewProtocolKind") == "assisted_peak_validation_v1" or full
        if full:
            full_window.prepare_full_window_supplement(extra, hashlib.sha256(addition.read_bytes()).hexdigest())
        elif assisted:
            prepare_assisted_supplement(extra, hashlib.sha256(addition.read_bytes()).hexdigest())
        for clip in extra["clips"]:
            if clip["id"] in known_ids or clip["participant"] not in known_participants:
                raise ValueError(f"Duplicate or unknown supplement identity: {clip['id']}")
            if not assisted and (not clip.get("blindReview") or clip.get("suggested") or clip.get("restorePeaks") or clip.get("aidAvailable")):
                raise ValueError("Blind supplements cannot contain candidate peak suggestions")
            known_ids.add(clip["id"])
            packet["clips"].append(clip)
        for batch in extra.get("reviewBatches", []):
            if batch["id"] in {item["id"] for item in packet["reviewBatches"]}:
                raise ValueError("Duplicate review batch")
            packet["reviewBatches"].append(batch)
    apply_assistance_overlay(packet, path.parent)
    return packet


def initialize_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=15)) as connection, connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS annotations (
                clip_id TEXT PRIMARY KEY,
                participant TEXT NOT NULL,
                peaks_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('in_progress','complete','unresolvable')),
                confidence TEXT NOT NULL CHECK(confidence IN ('high','medium','low')),
                notes TEXT NOT NULL DEFAULT '',
                reviewer TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version INTEGER NOT NULL
            )"""
        )
        connection.execute("CREATE INDEX IF NOT EXISTS annotations_participant_idx ON annotations(participant)")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(annotations)")}
        if "unreadable_ranges_json" not in columns:
            connection.execute("ALTER TABLE annotations ADD COLUMN unreadable_ranges_json TEXT NOT NULL DEFAULT '[]'")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS annotation_revisions (
                clip_id TEXT NOT NULL, version INTEGER NOT NULL,
                snapshot_json TEXT NOT NULL, archived_at TEXT NOT NULL,
                PRIMARY KEY (clip_id, version)
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS algorithm_assistance_views (
                clip_id TEXT NOT NULL, reviewer TEXT NOT NULL,
                method TEXT NOT NULL, overlay_sha256 TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                PRIMARY KEY (clip_id, reviewer, overlay_sha256)
            )"""
        )
        full_window.initialize_full_window_tables(connection)


def normalize_unreadable_ranges(value: object, sample_count: int) -> list[list[int]]:
    if not isinstance(value, list) or len(value) > 1000:
        raise ValueError("Unreadable intervals must be a list of at most 1,000 ranges")
    ranges: list[list[int]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each unreadable interval needs a start and end sample")
        if any(isinstance(index, bool) or not isinstance(index, (int, float)) or not math.isfinite(index) or int(index) != index for index in item):
            raise ValueError("Unreadable interval endpoints must be sample indices")
        start, end = map(int, item)
        if not 0 <= start < end < sample_count:
            raise ValueError("Unreadable interval is outside this segment or has no duration")
        ranges.append([start, end])
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def annotation_ranges(clip: dict, row: sqlite3.Row | None) -> list[list[int]]:
    ranges = json.loads(row["unreadable_ranges_json"]) if row else clip.get("priorUnreadableRanges", [])
    if not ranges and ((row is not None and row["status"] == "unresolvable") or (row is None and clip.get("priorUnreadable"))):
        return [[0, len(clip["t"]) - 1]]
    return ranges


def normalize_reviewer(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Please enter a reviewer name.")
    value = " ".join(value.split()).strip()
    if not value or len(value) > 64 or any(ord(char) < 32 for char in value):
        raise ValueError("Reviewer name must be 1–64 characters.")
    return value


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def csv_safe(value: object) -> str:
    text = str(value)
    if text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def make_handler(packet: dict, db_path: Path, access_code: str, signing_secret: bytes, secure_cookie: bool):
    clips = {item["id"]: item for item in packet["clips"]}
    by_participant: dict[str, list[dict]] = {}
    for item in packet["clips"]:
        by_participant.setdefault(item["participant"], []).append(item)
    participant_order = [row["participant"] for row in packet["participants"]]
    def batch_of(clip):
        return clip.get("reviewBatch", "existing")

    class ReviewHandler(BaseHTTPRequestHandler):
        server_version = "ECGReview/1.0"

        def log_message(self, fmt: str, *args: object) -> None:
            # Avoid logging payloads, cookies, or participant-level API details.
            super().log_message("request handled")

        def _headers(self, status: int, content_type: str, *, no_store: bool = True) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            )
            if no_store:
                self.send_header("Cache-Control", "no-store")

        def _json(self, status: int, value: object, extra_headers: dict[str, str] | None = None) -> None:
            payload = json_bytes(value)
            self._headers(status, "application/json; charset=utf-8")
            for name, header_value in (extra_headers or {}).items():
                self.send_header(name, header_value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _body(self) -> dict:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as error:
                raise ValueError("Invalid request length") from error
            if length < 0 or length > MAX_BODY_BYTES:
                raise ValueError("Request is too large")
            parsed = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(parsed, dict):
                raise ValueError("Request must be a JSON object")
            return parsed

        def _session_value(self, reviewer: str, expires: int) -> str:
            payload = f"{expires}\n{reviewer}".encode("utf-8")
            signature = hmac.new(signing_secret, payload, hashlib.sha256).digest()
            return base64.urlsafe_b64encode(payload + b"\n" + signature).decode("ascii").rstrip("=")

        def _reviewer(self) -> str | None:
            cookie = self.headers.get("Cookie", "")
            values = dict(
                part.strip().split("=", 1)
                for part in cookie.split(";")
                if "=" in part
            )
            token = values.get(COOKIE_NAME)
            if not token:
                return None
            try:
                raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
                expiry_text, reviewer_bytes, signature = raw.split(b"\n", 2)
                payload = expiry_text + b"\n" + reviewer_bytes
                expected = hmac.new(signing_secret, payload, hashlib.sha256).digest()
                if not hmac.compare_digest(signature, expected) or int(expiry_text) < int(time.time()):
                    return None
                return normalize_reviewer(reviewer_bytes.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None

        def _require_reviewer(self) -> str | None:
            reviewer = self._reviewer()
            if reviewer is None:
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "Please sign in to continue."})
            return reviewer

        def _annotation_row(self, clip_id: str, reviewer: str):
            with closing(sqlite3.connect(db_path, timeout=15)) as connection, connection:
                connection.row_factory = sqlite3.Row
                return full_window.annotation_row(connection, clips[clip_id], reviewer)

        def _reviewer_progress(self, clip_id: str) -> list[dict]:
            if not full_window.is_full_window(clips[clip_id]):
                return []
            with closing(sqlite3.connect(db_path, timeout=15)) as connection:
                rows = connection.execute(
                    "SELECT reviewer,status,reviewed_pages_json FROM full_window_annotations WHERE clip_id=? ORDER BY reviewer",
                    (clip_id,),
                ).fetchall()
            return [{"reviewer": name, "status": status, "reviewedPages": len(json.loads(pages))}
                    for name, status, pages in rows]

        def _screening_reveal(self, clip, reviewer):
            with closing(sqlite3.connect(db_path, timeout=15)) as connection:
                connection.row_factory = sqlite3.Row
                return full_window.screening_reveal(connection, clip, reviewer)

        def _screening_withheld(self, clip, reviewer):
            return screening_withheld(clip) or (full_window.deferred_screening(clip) and self._screening_reveal(clip, reviewer) is None)

        def _record_assistance_view(self, clip: dict, reviewer: str) -> None:
            # Displaying suggestions is not a completed human annotation. Keep
            # immutable exposure provenance separately from saved peak rows.
            assistance = clip.get("assistance")
            if not assistance or not clip.get("suggested"):
                return
            with closing(sqlite3.connect(db_path, timeout=15)) as connection, connection:
                connection.execute(
                    "INSERT OR IGNORE INTO algorithm_assistance_views VALUES (?, ?, ?, ?, ?)",
                    (clip["id"], reviewer, assistance["method"], assistance["overlaySha256"],
                     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                )

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/" or path == "/index.html":
                content = (APP_DIR / "static" / "index.html").read_bytes()
                self._headers(HTTPStatus.OK, "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            if path in {"/app.js", "/styles.css", "/README.md"}:
                if path == "/README.md":
                    content_type = "text/markdown; charset=utf-8"
                    content = (APP_DIR / "README.md").read_bytes()
                else:
                    content_type = "text/javascript; charset=utf-8" if path.endswith(".js") else "text/css; charset=utf-8"
                    content = (APP_DIR / "static" / path.lstrip("/")).read_bytes()
                self._headers(HTTPStatus.OK, content_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            if not path.startswith("/api/"):
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            if path == "/api/me":
                reviewer = self._reviewer()
                if reviewer is None:
                    self._json(HTTPStatus.UNAUTHORIZED, {"error": "Please sign in to continue."})
                else:
                    self._json(HTTPStatus.OK, {"reviewer": reviewer})
                return
            reviewer = self._require_reviewer()
            if reviewer is None:
                return
            if path == "/api/participants":
                with closing(sqlite3.connect(db_path, timeout=15)) as connection, connection:
                    rows = connection.execute("""SELECT participant,status,COUNT(*) FROM (
                        SELECT participant,status FROM annotations UNION ALL
                        SELECT participant,status FROM full_window_annotations WHERE reviewer=?
                        ) GROUP BY participant,status""", (reviewer,)).fetchall()
                counts: dict[str, dict[str, int]] = {}
                for participant, status, count in rows:
                    counts.setdefault(participant, {})[status] = count
                result = []
                for participant in participant_order:
                    total = len(by_participant.get(participant, []))
                    state = counts.get(participant, {})
                    saved = sum(state.values())
                    finished = state.get("complete", 0) + state.get("unresolvable", 0)
                    label = "Not started" if saved == 0 else ("Complete" if total > 0 and finished == total else "In progress")
                    result.append({
                        "participant": participant,
                        "total": total,
                        "saved": saved,
                        "finished": finished,
                        "qualityFlagged": sum(bool(clip.get("qualityFlags")) and not self._screening_withheld(clip, reviewer) for clip in by_participant.get(participant, [])),
                        "label": label,
                        "reason": next((r.get("reason", "") for r in packet["participants"] if r["participant"] == participant), ""),
                        "batches": {
                            batch["id"]: self._batch_progress(participant, batch["id"], reviewer)
                            for batch in packet.get("reviewBatches", [])
                        },
                    })
                self._json(HTTPStatus.OK, {"participants": result, "reviewBatches": packet.get("reviewBatches", [])})
                return
            match = re.fullmatch(r"/api/participant/([^/]+)/clips", path)
            if match:
                participant = unquote(match.group(1))
                items = by_participant.get(participant)
                if items is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown participant"})
                    return
                result = []
                for clip in items:
                    row = self._annotation_row(clip["id"], reviewer)
                    restore_peaks = clip.get("restorePeaks", clip["suggested"])
                    result.append({
                        "id": clip["id"],
                        "reviewId": clip["reviewId"],
                        "startSec": clip["startSec"],
                        "endSec": clip["endSec"],
                        "status": row["status"] if row else "unreviewed",
                        "reviewer": row["reviewer"] if row else "",
                        "updatedAt": row["updated_at"] if row else "",
                        "peakCount": len(json.loads(row["peaks_json"])) if row else len(restore_peaks),
                        "isPrevious": bool(clip.get("isPrevious", False)),
                        "hasPriorMarks": bool(clip.get("hasPriorMarks", False)),
                        "priorUnreadable": bool(clip.get("priorUnreadable", False)),
                        "sourceLabel": clip.get("sourceLabel", "New review segment"),
                        "reviewBatch": batch_of(clip),
                        "blindReview": bool(clip.get("blindReview")),
                        "annotationMode": clip.get("annotationMode", "algorithm_assisted"),
                        "assistanceAvailable": bool(clip.get("assistance")),
                        "fullWindow": full_window.is_full_window(clip),
                        "reviewerProgress": self._reviewer_progress(clip["id"]),
                        "unreadableRangeCount": len(annotation_ranges(clip, row)),
                        "qualityWarningCount": 0 if self._screening_withheld(clip, reviewer) else len(clip.get("qualityFlags", [])),
                        "qualityWarningLabels": [] if self._screening_withheld(clip, reviewer) else sorted({flag["label"] for flag in clip.get("qualityFlags", [])}),
                    })
                self._json(HTTPStatus.OK, {"clips": result})
                return
            match = re.fullmatch(r"/api/clip/([^/]+)", path)
            if match:
                clip_id = unquote(match.group(1))
                clip = clips.get(clip_id)
                if clip is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown clip"})
                    return
                row = self._annotation_row(clip_id, reviewer)
                result = {
                    key: value for key, value in clip.items()
                    if key not in {"suggested", "windowStart", "clipStart"}
                }
                result["suggested"] = clip["suggested"]
                restore_peaks = clip.get("restorePeaks", clip["suggested"])
                result["restorePeaks"] = restore_peaks
                result["peaks"] = json.loads(row["peaks_json"]) if row else restore_peaks
                result["status"] = row["status"] if row else "unreviewed"
                result["confidence"] = row["confidence"] if row else clip.get("priorConfidence", "high")
                result["notes"] = row["notes"] if row else clip.get("priorNote", "")
                result["unreadableRanges"] = annotation_ranges(clip, row)
                result["qualityFlags"] = clip.get("qualityFlags", [])
                result["qualitySummary"] = clip.get("qualitySummary", {"available": False})
                result["qualityRule"] = {key: value for key, value in packet.get("qualityScreeningProtocol", {}).items() if key not in {"source", "note"}}
                result["version"] = row["version"] if row else 0
                result["reviewer"] = row["reviewer"] if row else ""
                if full_window.is_full_window(clip):
                    result["reviewedPages"] = json.loads(row["reviewed_pages_json"]) if row else []
                    result["reviewerProgress"] = self._reviewer_progress(clip_id)
                if clip.get("blindReview"):
                    # Candidate arrays and screening judgments are never sent to
                    # the blind reviewer, including after saving or restoring.
                    result["suggested"] = []
                    result["restorePeaks"] = []
                    result["peaks"] = json.loads(row["peaks_json"]) if row else []
                if full_window.deferred_screening(clip):
                    revealed = self._screening_reveal(clip, reviewer)
                    result["screeningRevealedAt"] = revealed["revealed_at"] if revealed else None
                    result["screeningWithheld"] = revealed is None
                    if revealed:
                        result["qualityRule"] = clip.get("screeningRule", {})
                if self._screening_withheld(clip, reviewer):
                    result["qualityFlags"] = []
                    result["qualitySummary"] = {"available": False, "withheldForBlindReview": bool(clip.get("blindReview")), "withheldForReview": True}
                    result["qualityRule"] = {}
                self._json(HTTPStatus.OK, result)
                return
            if path == "/api/export.csv":
                self._export_csv(reviewer)
                return
            if path == "/api/full-window-export.csv":
                self._export_csv(reviewer, all_full_reviewers=True)
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def _batch_progress(self, participant, batch_id, reviewer):
            items = [clip for clip in by_participant.get(participant, []) if batch_of(clip) == batch_id]
            rows = [self._annotation_row(clip["id"], reviewer) for clip in items]
            saved = sum(row is not None for row in rows)
            finished = sum(row is not None and row["status"] in {"complete", "unresolvable"} for row in rows)
            label = "Not started" if not saved else "Complete" if finished == len(items) else "In progress"
            return {"total": len(items), "saved": saved, "finished": finished, "label": label,
                    "qualityFlagged": sum(bool(clip.get("qualityFlags")) and not self._screening_withheld(clip, reviewer) for clip in items)}

        def _export_csv(self, reviewer: str, all_full_reviewers: bool = False) -> None:
            from io import StringIO

            with closing(sqlite3.connect(db_path, timeout=15)) as connection, connection:
                connection.row_factory = sqlite3.Row
                saved = {row["clip_id"]: row for row in connection.execute("SELECT * FROM annotations")}
                full_saved = {}
                query = "SELECT * FROM full_window_annotations" + ("" if all_full_reviewers else " WHERE reviewer=?")
                for row in connection.execute(query, () if all_full_reviewers else (reviewer,)):
                    full_saved.setdefault(row["clip_id"], []).append(row)
                assistance_views = {}
                for view in connection.execute("SELECT * FROM algorithm_assistance_views ORDER BY first_seen_at"):
                    assistance_views.setdefault((view["clip_id"], view["reviewer"]), view)
            buffer = StringIO(newline="")
            writer = csv.writer(buffer)
            writer.writerow([
                "reviewer_id", "review_set", "review_id", "participant", "clip_id", "window_start_unix_ms",
                "manual_rpeak_abs_unix_ms", "confidence", "review_status", "notes",
                "record_type", "unreadable_start_sec", "unreadable_end_sec",
                "unreadable_start_unix_ms", "unreadable_end_unix_ms", "peak_in_unreadable_interval",
                "quality_flag_codes", "quality_flags_json", "peak_in_quality_flag",
                "review_batch", "annotation_mode",
                "algorithm_assistance_seen", "algorithm_assistance_method",
                "algorithm_assistance_overlay_sha256", "first_algorithm_assistance_seen_at",
                "annotation_scope", "window_duration_sec", "peak_sample_index", "reviewed_pages_json",
                "sampling_gaps_json", "waveform_sha256",
                "screening_revealed_at", "first_pass_annotation_json",
            ])
            records = [(clip, row) for clip in packet["clips"]
                       if not all_full_reviewers or full_window.is_full_window(clip)
                       for row in (full_saved.get(clip["id"], []) if full_window.is_full_window(clip)
                                   else ([saved[clip["id"]]] if clip["id"] in saved else []))]
            for clip, row in records:
                if row is None:
                    continue
                peaks = json.loads(row["peaks_json"])
                ranges = annotation_ranges(clip, row)
                quality_flags = [] if self._screening_withheld(clip, reviewer) else clip.get("qualityFlags", [])
                reveal = self._screening_reveal(clip, row["reviewer"])
                quality_codes = ";".join(sorted({flag["code"] for flag in quality_flags}))
                exposure = assistance_views.get((clip["id"], row["reviewer"]))
                if exposure is None and not full_window.is_full_window(clip):
                    exposure = next((v for (cid, _), v in assistance_views.items() if cid == clip["id"]), None)
                mode = "algorithm_assisted" if exposure is not None else clip.get("annotationMode", "algorithm_assisted")
                assistance = clip.get("assistance", {})
                def write_record(timestamp: str, fields: list[object], inside_quality: str = "", sample_index: object = "") -> None:
                    writer.writerow([
                        csv_safe(row["reviewer"]), clip.get("reviewSet", "ecg_rpeak_blind_review_remote_v1"),
                        clip["reviewId"], clip["participant"], clip["id"],
                        clip["windowStart"], timestamp,
                        row["confidence"], row["status"], csv_safe(row["notes"]),
                    ] + fields + [quality_codes, json.dumps(quality_flags, ensure_ascii=False, separators=(",", ":")), inside_quality,
                                  batch_of(clip), mode, str(exposure is not None).lower(),
                                  exposure["method"] if exposure is not None else assistance.get("method", ""),
                                  exposure["overlay_sha256"] if exposure is not None else assistance.get("overlaySha256", ""),
                                  exposure["first_seen_at"] if exposure is not None else "",
                                  clip.get("annotationScope", "shared"), clip.get("windowDurationSec", ""), sample_index,
                                  row["reviewed_pages_json"] if full_window.is_full_window(clip) else "[]",
                                  json.dumps(clip.get("samplingGaps", [])),
                                  assistance.get("waveformSha256", ""),
                                  reveal["revealed_at"] if reveal else "",
                                  reveal["first_pass_json"] if reveal else ""])
                for index in peaks:
                    timestamp = clip["clipStart"] + clip["t"][index] * 1000.0
                    inside = any(start <= index <= end for start, end in ranges)
                    flagged = any(flag["startIndex"] <= index <= flag["endIndex"] for flag in quality_flags)
                    write_record(f"{timestamp:.3f}", ["peak", "", "", "", "", str(inside).lower()], str(flagged).lower(), index)
                for start, end in ranges:
                    left, right = clip["t"][start], clip["t"][end]
                    write_record("", ["unreadable_interval", left, right,
                                     f"{clip['clipStart'] + left * 1000.0:.3f}",
                                     f"{clip['clipStart'] + right * 1000.0:.3f}", ""])
                if not peaks and not ranges:
                    write_record("", ["status", "", "", "", "", ""])
            payload = buffer.getvalue().encode("utf-8-sig")
            self._headers(HTTPStatus.OK, "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="ecg_rpeak_annotations.csv"')
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            try:
                body = self._body()
            except (ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            if path == "/api/login":
                supplied_code = body.get("accessCode")
                try:
                    reviewer = normalize_reviewer(body.get("reviewer"))
                except ValueError as error:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                if not isinstance(supplied_code, str) or not hmac.compare_digest(
                    supplied_code.encode("utf-8"), access_code.encode("utf-8")
                ):
                    self._json(HTTPStatus.UNAUTHORIZED, {"error": "The access code is not correct."})
                    return
                expires = int(time.time()) + SESSION_SECONDS
                value = self._session_value(reviewer, expires)
                cookie = f"{COOKIE_NAME}={value}; Path=/; HttpOnly; SameSite=Strict; Max-Age={SESSION_SECONDS}"
                if secure_cookie:
                    cookie += "; Secure"
                self._json(HTTPStatus.OK, {"reviewer": reviewer}, {"Set-Cookie": cookie})
                return
            if path == "/api/logout":
                self._json(
                    HTTPStatus.OK,
                    {"ok": True},
                    {"Set-Cookie": f"{COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0" + ("; Secure" if secure_cookie else "")},
                )
                return
            reviewer = self._require_reviewer()
            if reviewer is None:
                return
            reveal_match = re.fullmatch(r"/api/clip/([^/]+)/reveal-screening", path)
            if reveal_match:
                clip = clips.get(unquote(reveal_match.group(1)))
                if clip is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown clip"})
                    return
                try:
                    with closing(sqlite3.connect(db_path, timeout=15)) as connection, connection:
                        connection.row_factory = sqlite3.Row
                        connection.execute("BEGIN IMMEDIATE")
                        result = full_window.reveal_screening(connection, clip, reviewer, int(body.get("version", -1)), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                        revealed_at = result["revealed_at"]
                except (TypeError, ValueError) as error:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                self._json(HTTPStatus.OK, {"ok": True, "revealedAt": revealed_at})
                return
            view_match = re.fullmatch(r"/api/clip/([^/]+)/assistance-seen", path)
            if view_match:
                clip = clips.get(unquote(view_match.group(1)))
                if clip is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown clip"})
                    return
                self._record_assistance_view(clip, reviewer)
                self._json(HTTPStatus.OK, {"ok": True})
                return
            match = re.fullmatch(r"/api/clip/([^/]+)/annotation", path)
            if not match:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            clip_id = unquote(match.group(1))
            clip = clips.get(clip_id)
            if clip is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown clip"})
                return
            status = body.get("status")
            if not isinstance(status, str) or status not in ALLOWED_STATUSES:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid review status"})
                return
            try:
                peaks = sorted({int(value) for value in body["peaks"]}) if "peaks" in body else None
                ranges = normalize_unreadable_ranges(body["unreadableRanges"], len(clip["t"])) if "unreadableRanges" in body else None
                expected_version = int(body.get("version", 0))
                confidence = body.get("confidence", "high")
                notes = body.get("notes", "")
                if not isinstance(notes, str) or len(notes) > 1000:
                    raise ValueError("Notes must be 1,000 characters or fewer")
                if not isinstance(confidence, str) or confidence not in {"high", "medium", "low"}:
                    raise ValueError("Invalid confidence value")
                if peaks is not None and any(value < 0 or value >= len(clip["t"]) for value in peaks):
                    raise ValueError("One or more peak marks are outside this clip")
                if expected_version < 0:
                    raise ValueError("Invalid annotation version")
            except (TypeError, ValueError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            try:
                with closing(sqlite3.connect(db_path, timeout=15)) as connection, connection:
                    connection.row_factory = sqlite3.Row
                    connection.execute("PRAGMA foreign_keys=ON")
                    connection.execute("BEGIN IMMEDIATE")
                    row = full_window.annotation_row(connection, clip, reviewer)
                    saved_peaks = json.loads(row["peaks_json"]) if row else clip.get("restorePeaks", clip["suggested"])
                    # Older pages sent an empty peak list for this action. Preserve the
                    # saved marks and interpret the old request as a full-clip interval.
                    if peaks is None or (status == "unresolvable" and "unreadableRanges" not in body):
                        peaks = saved_peaks
                    if ranges is None:
                        ranges = annotation_ranges(clip, row)
                    if status == "unresolvable" and not ranges:
                        ranges = [[0, len(clip["t"]) - 1]]
                    if status == "complete" and not peaks and not ranges:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": "Mark at least one peak or an unreadable interval before completing this segment."})
                        return
                    if full_window.is_full_window(clip):
                        try:
                            pages = full_window.normalize_pages(body.get("reviewedPages", json.loads(row["reviewed_pages_json"]) if row else []), clip)
                        except ValueError as error:
                            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                            return
                        if expected_version != (row["version"] if row else 0):
                            self._json(HTTPStatus.CONFLICT, {"error": "Your review changed on another device; reload before saving."})
                            return
                        if status in {"complete", "unresolvable"} and len(pages) != full_window.page_count(clip):
                            self._json(HTTPStatus.BAD_REQUEST, {"error": "Review and confirm all 60 five-second pages before completing this window."})
                            return
                        new_version = full_window.write_annotation(
                            connection, clip, reviewer, row, peaks=peaks, status=status, confidence=confidence,
                            notes=notes, updated_at=now, ranges=ranges, pages=pages)
                    elif row is None:
                        if expected_version != 0:
                            self._json(HTTPStatus.CONFLICT, {"error": "This clip changed elsewhere; reload before saving."})
                            return
                        connection.execute(
                            "INSERT INTO annotations (clip_id,participant,peaks_json,status,confidence,notes,reviewer,updated_at,version,unreadable_ranges_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
                            (clip_id, clip["participant"], json.dumps(peaks), status, confidence, notes, reviewer, now, json.dumps(ranges)),
                        )
                        new_version = 1
                    else:
                        current_version = int(row["version"])
                        if expected_version != current_version:
                            self._json(HTTPStatus.CONFLICT, {"error": "This clip changed elsewhere; reload before saving."})
                            return
                        new_version = current_version + 1
                        connection.execute(
                            "INSERT OR IGNORE INTO annotation_revisions VALUES (?, ?, ?, ?)",
                            (clip_id, current_version, json.dumps(dict(row)), now),
                        )
                        connection.execute(
                            "UPDATE annotations SET peaks_json=?, status=?, confidence=?, notes=?, reviewer=?, updated_at=?, version=?, unreadable_ranges_json=? WHERE clip_id=?",
                            (json.dumps(peaks), status, confidence, notes, reviewer, now, new_version, json.dumps(ranges), clip_id),
                        )
            except sqlite3.IntegrityError:
                self._json(HTTPStatus.CONFLICT, {"error": "This clip changed elsewhere; reload before saving."})
                return
            # Conservatively retain assisted provenance even for an older
            # client that saves without reporting its display exposure.
            self._record_assistance_view(clip, reviewer)
            self._json(HTTPStatus.OK, {
                "ok": True, "version": new_version, "status": status, "updatedAt": now, "reviewer": reviewer,
                "unreadableRanges": ranges,
                "reviewedPages": pages if full_window.is_full_window(clip) else [],
                "reviewerProgress": self._reviewer_progress(clip_id),
            })

    return ReviewHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 for network access")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--certfile", type=Path, help="TLS certificate for direct HTTPS access")
    parser.add_argument("--keyfile", type=Path, help="TLS private key for direct HTTPS access")
    parser.add_argument("--secure-cookie", action="store_true", help="Set Secure cookies when behind an HTTPS reverse proxy")
    parser.add_argument("--allow-insecure-http", action="store_true", help="Allow non-loopback plain HTTP; only for trusted VPN/LAN testing")
    args = parser.parse_args()

    access_code = os.environ.get("ECG_REVIEW_ACCESS_CODE", "")
    secret_value = os.environ.get("ECG_REVIEW_SESSION_SECRET", "")
    if len(access_code) < 12:
        raise SystemExit("Set ECG_REVIEW_ACCESS_CODE to a private access code of at least 12 characters.")
    if len(secret_value) < 32:
        raise SystemExit("Set ECG_REVIEW_SESSION_SECRET to a random secret of at least 32 characters.")
    if bool(args.certfile) != bool(args.keyfile):
        raise SystemExit("Provide both --certfile and --keyfile for direct HTTPS.")
    tls_enabled = bool(args.certfile and args.keyfile)
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not tls_enabled and not args.allow_insecure_http:
        raise SystemExit("Remote binding requires HTTPS. Provide a TLS certificate or explicitly allow trusted-LAN HTTP.")

    packet = load_packet(args.data_file)
    initialize_database(args.database)
    secure_cookie = tls_enabled or args.secure_cookie
    server = ThreadingHTTPServer((args.host, args.port), make_handler(
        packet, args.database, access_code, secret_value.encode("utf-8"), secure_cookie
    ))
    server.daemon_threads = True
    if tls_enabled:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(str(args.certfile), str(args.keyfile))
        server.socket = context.wrap_socket(server.socket, server_side=True)
    scheme = "https" if tls_enabled else "http"
    print(f"ECG review app listening at {scheme}://{args.host}:{args.port}")
    print(f"Shared review database: {args.database.resolve()}")
    print("Keep the data, database, access code, and session secret on an approved, access-controlled host.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ECG review server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
