"""Legacy offline database migration schema, with no HTTP server."""
from pathlib import Path
from contextlib import closing
import sqlite3

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

import json

def read_ratings(path):
    with closing(sqlite3.connect(path, timeout=15)) as db:
        db.row_factory = sqlite3.Row
        return {r['id']: dict(json.loads(r['answer_json']), version=r['version'], reviewer=r['reviewer'], updated_at=r['updated_at']) for r in db.execute('SELECT * FROM ratings')}
