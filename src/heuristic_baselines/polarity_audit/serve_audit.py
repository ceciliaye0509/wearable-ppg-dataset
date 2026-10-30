"""Serve independent human QRS-readability and direction assessments."""
from __future__ import annotations

import argparse
import base64
from contextlib import closing
import csv
import gzip
import hashlib
import hmac
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
import json
import os
from pathlib import Path
import secrets
import sqlite3
import ssl
import sys
import time
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from remote_ecg_review.serve_review import csv_safe, json_bytes, normalize_reviewer
from peak_assistance import suggest_peaks
from comparison_review import public_task, validate_comparison, comparison_csv

DEFAULT_DIR = HERE.parent / 'outputs/beatwise_polarity_audit_v2'
COMPARISON_DIR = HERE.parent / 'outputs/polarity_pairwise_disagreements_v1'
COOKIE = 'ecg_polarity_audit_session'


def initialize_database(path, packet_digest):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=15)) as db, db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        saved = db.execute("SELECT value FROM metadata WHERE key='packet_sha256'").fetchone()
        if saved and saved[0] != packet_digest:
            raise ValueError('This database belongs to a different audit packet; use its original packet or a new database.')
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('packet_sha256', ?)", (packet_digest,))
        db.execute('CREATE TABLE IF NOT EXISTS ratings (id TEXT PRIMARY KEY, answer_json TEXT NOT NULL, version INTEGER NOT NULL, reviewer TEXT NOT NULL, updated_at TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS revisions (id TEXT NOT NULL, version INTEGER NOT NULL, snapshot_json TEXT NOT NULL, PRIMARY KEY(id,version))')


def validate_answer(body, task=None):
    answer = {key: body.get(key, '') for key in ['qrs_readability', 'direction_determinability', 'direction', 'confidence', 'notes', 'status']}
    allowed = {
        'qrs_readability': {'', 'readable', 'partially_readable', 'unreadable', 'unsure'},
        'direction_determinability': {'', 'yes', 'no', 'unsure'},
        'direction': {'', 'positive', 'negative', 'undetermined'},
        'confidence': {'high', 'medium', 'low'}, 'status': {'in_progress', 'complete'},
    }
    for key, values in allowed.items():
        if not isinstance(answer[key], str) or answer[key] not in values:
            raise ValueError(f'Invalid {key}')
    if not isinstance(answer['notes'], str) or len(answer['notes']) > 2000:
        raise ValueError('Notes must be 2,000 characters or fewer')
    if answer['direction_determinability'] == 'yes':
        if answer['direction'] not in {'positive', 'negative'}:
            raise ValueError('Choose positive or negative when direction is determinable')
    else:
        answer['direction'] = 'undetermined'
    if answer['status'] == 'complete' and (not answer['qrs_readability'] or not answer['direction_determinability']):
        raise ValueError('Answer both independent questions before completing this minute')
    version = body.get('version')
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError('Invalid saved version')
    if 'peak_indices' in body:
        peaks = body['peak_indices']
        if task is None or not isinstance(peaks, list) or len(peaks) > 2000:
            raise ValueError('Invalid peak marks')
        for index in peaks:
            if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(task['t'])
                    or not task['focusStartSec'] <= task['t'][index] < task['focusEndSec']):
                raise ValueError('Peak marks must reference samples inside the focus interval')
        if len(set(peaks)) != len(peaks):
            raise ValueError('Duplicate peak marks')
        if body.get('peak_origin') not in {'algorithm', 'manual', 'mixed', 'unreviewed'}:
            raise ValueError('Invalid peak origin')
        for key in ['peaks_reviewed', 'assistance_shown']:
            if not isinstance(body.get(key), bool):
                raise ValueError(f'Invalid {key}')
        answer.update(peak_indices=sorted(peaks), peak_origin=body['peak_origin'],
                      peaks_reviewed=body['peaks_reviewed'], assistance_shown=body['assistance_shown'])
    return answer, version


def save_rating(path, task_id, answer, version, reviewer):
    with closing(sqlite3.connect(path, timeout=15)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        db.row_factory = sqlite3.Row
        old = db.execute('SELECT * FROM ratings WHERE id=?', (task_id,)).fetchone()
        if (old['version'] if old else 0) != version:
            return None
        if old:
            db.execute('INSERT OR IGNORE INTO revisions VALUES (?,?,?)', (task_id, old['version'], json.dumps(dict(old))))
            previous = json.loads(old['answer_json'])
            # Older clients may save ratings without sending peak fields. Never erase them.
            answer = dict(previous, **answer)
            if previous.get('assistance_shown'):
                answer['assistance_shown'] = True
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        db.execute('INSERT INTO ratings VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET answer_json=excluded.answer_json,version=excluded.version,reviewer=excluded.reviewer,updated_at=excluded.updated_at',
                   (task_id, json.dumps(answer), version + 1, reviewer, now))
        return dict(answer, version=version + 1, reviewer=reviewer, updated_at=now)


def read_ratings(path):
    with closing(sqlite3.connect(path, timeout=15)) as db:
        db.row_factory = sqlite3.Row
        return {r['id']: dict(json.loads(r['answer_json']), version=r['version'], reviewer=r['reviewer'], updated_at=r['updated_at']) for r in db.execute('SELECT * FROM ratings')}


def make_handler(packet, db_path, access_code, secret, secure_cookie=False, comparison_packet=None, comparison_db=None):
    tasks = {task['id']: task for task in packet['tasks']}
    paired = {task['id']:task for task in comparison_packet['tasks']} if comparison_packet else {}

    @lru_cache(maxsize=256)
    def assistance(task_id):
        return suggest_peaks(tasks[task_id])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def headers_out(self, status, content_type):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")

        def respond(self, status, value, cookie=None):
            payload = json_bytes(value)
            self.headers_out(status, 'application/json; charset=utf-8')
            if cookie:
                self.send_header('Set-Cookie', cookie)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def reviewer(self):
            try:
                cookies = dict(part.strip().split('=', 1) for part in self.headers.get('Cookie', '').split(';') if '=' in part)
                encoded, signature = cookies[COOKIE].split('.', 1)
                if not hmac.compare_digest(signature, hmac.new(secret, encoded.encode(), hashlib.sha256).hexdigest()):
                    return None
                data = json.loads(base64.urlsafe_b64decode(encoded + '=' * (-len(encoded) % 4)))
                if data['expires'] <= time.time():
                    return None
                return normalize_reviewer(data['reviewer'])
            except (KeyError, ValueError, TypeError, UnicodeDecodeError):
                return None

        def require_reviewer(self):
            reviewer = self.reviewer()
            if not reviewer:
                self.respond(401, {'error': 'Please sign in to continue.'})
            return reviewer

        def do_GET(self):
            path = urlsplit(self.path).path
            assets = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'), '/styles.css': ('styles.css', 'text/css')}
            assets.update({'/legacy/':('index.html','text/html'),'/legacy/app.js':('app.js','text/javascript'),
                           '/legacy/styles.css':('styles.css','text/css')})
            if comparison_packet:
                assets.update({'/':('comparison.html','text/html'),'/comparison.js':('comparison.js','text/javascript'),
                               '/comparison.css':('comparison.css','text/css')})
            if path in assets:
                name, kind = assets[path]
                payload = (HERE / 'static' / name).read_bytes()
                self.headers_out(200, kind + '; charset=utf-8')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            reviewer = self.require_reviewer()
            if not reviewer:
                return
            if path.startswith('/api/comparison/'):
                if not comparison_packet:
                    self.respond(404, {'error':'Comparison packet is not installed'})
                    return
                ratings=read_ratings(comparison_db)
                if path == '/api/comparison/tasks':
                    self.respond(200,dict(tasks=[dict(id=t['id'],participant=t['participant'],rating=ratings.get(t['id'])) for t in comparison_packet['tasks']],
                                          participants=comparison_packet['protocol']['strata']))
                elif path.startswith('/api/comparison/task/'):
                    task=paired.get(unquote(path[len('/api/comparison/task/'):]))
                    self.respond(200,public_task(task,ratings.get(task['id']))) if task else self.respond(404,{'error':'Unknown comparison minute'})
                elif path == '/api/comparison/export.csv':
                    payload=comparison_csv(comparison_packet,ratings,csv_safe)
                    self.headers_out(200,'text/csv; charset=utf-8')
                    self.send_header('Content-Disposition','attachment; filename="direction_paired_preferences.csv"')
                    self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
                else:
                    self.respond(404,{'error':'Not found'})
                return
            if path == '/api/me':
                self.respond(200, {'reviewer': reviewer})
            elif path == '/api/tasks':
                ratings = read_ratings(db_path)
                self.respond(200, {'tasks': [dict(id=t['id'], participant=t['participant'], rating=ratings.get(t['id'])) for t in packet['tasks']],
                                   'participants': [s['participant'] for s in packet['protocol']['strata']]})
            elif path.startswith('/api/task/'):
                task = tasks.get(unquote(path[len('/api/task/'):]))
                if not task:
                    self.respond(404, {'error': 'Unknown audit minute'})
                    return
                # Hide direction predictions/votes; expose optional, non-authoritative timing assistance.
                allowed = ['id', 'participant', 't', 'x', 'focusStartSec', 'focusEndSec']
                self.respond(200, dict({k: task[k] for k in allowed}, rating=read_ratings(db_path).get(task['id']),
                                       assistance=assistance(task['id'])))
            elif path == '/api/export.csv':
                ratings = read_ratings(db_path)
                strata = {s['participant']: s for s in packet['protocol']['strata']}
                buffer = StringIO(newline='')
                fields = ['id', 'participant', 'sampling_stratum', 'population_minutes', 'sampled_minutes', 'inverse_probability_weight',
                          'qrs_readability', 'direction_determinability', 'direction', 'confidence', 'status', 'reviewer', 'updated_at', 'notes',
                          'assistance_shown', 'assistance_method', 'assistance_source_sha256', 'peak_origin', 'peaks_reviewed',
                          'peak_indices', 'peak_times_sec', 'suggested_peak_indices']
                writer = csv.DictWriter(buffer, fieldnames=fields)
                writer.writeheader()
                for task in packet['tasks']:
                    row = dict(id=task['id'], participant=task['participant'])
                    sampling = task.get('sampling', strata[task['participant']])
                    row.update({k: sampling[k] for k in ['population_minutes', 'sampled_minutes', 'inverse_probability_weight']})
                    row['sampling_stratum'] = sampling.get('sampling_stratum', 'old_ambiguous')
                    row.update({k: v for k, v in ratings.get(task['id'], {}).items() if k in fields})
                    if 'peak_indices' in row:
                        row['peak_times_sec'] = json.dumps([task['t'][i] for i in row['peak_indices']])
                    for key in ['peak_indices', 'suggested_peak_indices']:
                        if key in row:
                            row[key] = json.dumps(row[key])
                    writer.writerow({k: csv_safe(v) for k, v in row.items()})
                payload = buffer.getvalue().encode('utf-8-sig')
                self.headers_out(200, 'text/csv; charset=utf-8')
                self.send_header('Content-Disposition', 'attachment; filename="polarity_human_audit.csv"')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.respond(404, {'error': 'Not found'})

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and (urlsplit(origin).scheme not in {'http', 'https'} or urlsplit(origin).netloc != self.headers.get('Host')):
                self.respond(403, {'error': 'Cross-origin request rejected'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 <= size <= 64000 or self.headers.get_content_type() != 'application/json':
                    raise ValueError('Send a JSON request smaller than 64 KB')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('Request must be a JSON object')
            except (ValueError, json.JSONDecodeError) as error:
                self.respond(400, {'error': str(error)})
                return
            path = urlsplit(self.path).path
            if path == '/api/login':
                try:
                    name = normalize_reviewer(body.get('reviewer'))
                    code = body.get('accessCode')
                    if not isinstance(code, str) or not hmac.compare_digest(code.encode(), access_code.encode()):
                        self.respond(401, {'error': 'Incorrect access code'})
                        return
                except ValueError as error:
                    self.respond(400, {'error': str(error)})
                    return
                encoded = base64.urlsafe_b64encode(json_bytes(dict(reviewer=name, expires=int(time.time()) + 28800))).decode().rstrip('=')
                token = encoded + '.' + hmac.new(secret, encoded.encode(), hashlib.sha256).hexdigest()
                cookie = f'{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=28800' + ('; Secure' if secure_cookie else '')
                self.respond(200, {'reviewer': name}, cookie)
                return
            if not (reviewer := self.require_reviewer()):
                return
            if path.startswith('/api/comparison/task/') and path.endswith('/review'):
                task_id=unquote(path[len('/api/comparison/task/'):-len('/review')])
                if task_id not in paired:
                    self.respond(404,{'error':'Unknown comparison minute'});return
                try:
                    answer,version=validate_comparison(body)
                except ValueError as error:
                    self.respond(400,{'error':str(error)});return
                result=save_rating(comparison_db,task_id,answer,version,reviewer)
                self.respond(200 if result else 409,result or {'error':'This minute was changed on another device. Reload before saving.'})
                return
            if not path.startswith('/api/task/') or not path.endswith('/review'):
                self.respond(404, {'error': 'Not found'})
                return
            task_id = unquote(path[len('/api/task/'):-len('/review')])
            if task_id not in tasks:
                self.respond(404, {'error': 'Unknown audit minute'})
                return
            try:
                answer, version = validate_answer(body, tasks[task_id])
                if 'peak_indices' in answer:
                    suggestions = assistance(task_id)
                    answer.update(assistance_method=suggestions['method'],
                                  assistance_source_sha256=suggestions['source_sha256'],
                                  suggested_peak_indices=suggestions['peak_indices'])
            except ValueError as error:
                self.respond(400, {'error': str(error)})
                return
            result = save_rating(db_path, task_id, answer, version, reviewer)
            self.respond(200 if result else 409, result or {'error': 'This minute was changed on another device. Reload before saving.'})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-file', type=Path, default=DEFAULT_DIR / 'audit_packet.json.gz')
    parser.add_argument('--database', type=Path, default=DEFAULT_DIR / 'human_audit.sqlite3')
    parser.add_argument('--comparison-file',type=Path,default=COMPARISON_DIR/'comparison_packet.json.gz')
    parser.add_argument('--comparison-database',type=Path,default=COMPARISON_DIR/'paired_preferences.sqlite3')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--certfile', type=Path)
    parser.add_argument('--keyfile', type=Path)
    parser.add_argument('--secure-cookie', action='store_true')
    parser.add_argument('--local-session', action='store_true', help='Generate a temporary local-only access code and session secret')
    args = parser.parse_args()
    if args.local_session:
        if args.host not in {'127.0.0.1', 'localhost'}:
            parser.error('Temporary local sessions require a loopback address')
        access, secret = secrets.token_urlsafe(18), secrets.token_bytes(32)
    else:
        access = os.environ.get('ECG_AUDIT_ACCESS_CODE', '')
        secret = os.environ.get('ECG_AUDIT_SESSION_SECRET', '').encode()
        if len(access) < 12 or len(secret) < 32:
            parser.error('Set ECG_AUDIT_ACCESS_CODE (12+ characters) and ECG_AUDIT_SESSION_SECRET (32+ characters)')
    if bool(args.certfile) != bool(args.keyfile):
        parser.error('Provide both TLS certificate and key')
    if args.host not in {'127.0.0.1', 'localhost'} and not args.certfile:
        parser.error('Remote access requires HTTPS')
    with gzip.open(args.data_file, 'rt') as handle:
        packet = json.load(handle)
    if packet.get('version') != 1 or not packet.get('tasks'):
        parser.error('Missing or unsupported audit packet; run build_audit.py first')
    initialize_database(args.database, hashlib.sha256(args.data_file.read_bytes()).hexdigest())
    comparison_packet=None
    if args.comparison_file.exists():
        with gzip.open(args.comparison_file,'rt') as handle:comparison_packet=json.load(handle)
        if comparison_packet.get('version')!=1 or not comparison_packet.get('tasks'):
            parser.error('Unsupported comparison packet')
        if len({t['id'] for t in comparison_packet['tasks']})!=len(comparison_packet['tasks']):
            parser.error('Duplicate comparison IDs')
        initialize_database(args.comparison_database,hashlib.sha256(args.comparison_file.read_bytes()).hexdigest())
    server = ThreadingHTTPServer((args.host, args.port), make_handler(packet, args.database, access, secret, bool(args.certfile or args.secure_cookie),comparison_packet,args.comparison_database))
    if args.certfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.certfile, args.keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    print(f"Audit workspace: {'https' if args.certfile else 'http'}://{args.host}:{args.port}/", flush=True)
    if args.local_session:
        print(f'Local audit access code: {access}', flush=True)
    print(f'Separate audit database: {args.database.resolve()}', flush=True)
    if comparison_packet:
        print(f'Paired comparison: {len(comparison_packet["tasks"])} minutes; original audit retained at /legacy/',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
