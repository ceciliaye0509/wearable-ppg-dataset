"""Isolated three-method follow-up; does not restart or write older workspaces."""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from io import StringIO
import json
import os
from pathlib import Path
import secrets
import sys
import time
from urllib.parse import unquote,urlsplit

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from serve_audit import initialize_database,read_ratings,save_rating
from remote_ecg_review.serve_review import normalize_reviewer,csv_safe,json_bytes
from build_targeted_followup import DEFAULT_OUT

COOKIE='ecg_direction_followup_session'
CHOICES={'','A','B','C','AB','AC','BC','same','neither','unsure'}
SIDE_FIELDS=('direction','support','decision','flip')


def public_task(task,rating=None):
    return dict(id=task['id'],participant=task['participant'],queue=task['queue'],
                t=task['t'],x=task['x'],focusStartSec=task['focusStartSec'],focusEndSec=task['focusEndSec'],
                sides={side:{key:value[key] for key in SIDE_FIELDS} for side,value in task['sides'].items()},rating=rating)


def validate_answer(body):
    choice,status,confidence=body.get('choice',''),body.get('status'),body.get('confidence','medium')
    notes,version=body.get('notes',''),body.get('version')
    if not isinstance(choice,str) or choice not in CHOICES:raise ValueError('Choose a method, a tie, none reliable, or cannot judge.')
    if status not in {'in_progress','complete'} or confidence not in {'high','medium','low'}:raise ValueError('Invalid status or confidence.')
    if status=='complete' and not choice:raise ValueError('Choose a result before completing this minute.')
    if not isinstance(notes,str) or len(notes)>2000:raise ValueError('Notes must be at most 2,000 characters.')
    if type(version) is not int or version<0:raise ValueError('Invalid saved version.')
    return dict(choice=choice,status=status,confidence=confidence,notes=notes,
                review_mode='assisted_targeted_three_method_preference'),version


def export_csv(packet,ratings):
    buffer=StringIO(newline='')
    fields=['id','participant','queue','choice','confidence','status','reviewer','updated_at','version','notes','review_mode']
    writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader()
    for task in packet['tasks']:
        if task['id'] not in ratings:continue
        row=dict(id=task['id'],participant=task['participant'],queue=task['queue'],**ratings[task['id']])
        writer.writerow({key:csv_safe(row.get(key,'')) for key in fields})
    return buffer.getvalue().encode('utf-8-sig')


def make_handler(packet,db_path,access,secret,*,public_fn=public_task,validate_fn=validate_answer,
                 csv_fn=export_csv,cookie_name=COOKIE,assets_override=None,summary_fn=None,results_fn=None,export_filename=None):
    tasks={t['id']:t for t in packet['tasks']}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass

        def headers_out(self,status,kind):
            self.send_response(status);self.send_header('Content-Type',kind)
            for key,val in [('Cache-Control','no-store'),('X-Content-Type-Options','nosniff'),('X-Frame-Options','DENY'),
                            ('Referrer-Policy','no-referrer'),('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")]:
                self.send_header(key,val)

        def respond(self,status,value,cookie=None):
            raw=json_bytes(value);self.headers_out(status,'application/json; charset=utf-8')
            if cookie:self.send_header('Set-Cookie',cookie)
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)

        def reviewer(self):
            try:
                cookies=dict(p.strip().split('=',1) for p in self.headers.get('Cookie','').split(';') if '=' in p)
                encoded,signature=cookies[cookie_name].split('.',1)
                expected=hmac.new(secret,encoded.encode(),hashlib.sha256).hexdigest()
                if not hmac.compare_digest(signature,expected):return None
                data=json.loads(base64.urlsafe_b64decode(encoded+'='*(-len(encoded)%4)))
                return normalize_reviewer(data['reviewer']) if data['expires']>time.time() else None
            except (KeyError,ValueError,TypeError,UnicodeError):return None

        def do_GET(self):
            path=urlsplit(self.path).path
            assets={'/':('triad.html','text/html'),'/triad.js':('triad.js','text/javascript'),
                    '/triad.css':('triad.css','text/css'),'/styles.css':('styles.css','text/css')}
            if assets_override is not None:assets=assets_override
            if path in assets:
                name,kind=assets[path];raw=(HERE/'static'/name).read_bytes()
                self.headers_out(200,kind+'; charset=utf-8');self.send_header('Content-Length',str(len(raw)))
                self.end_headers();self.wfile.write(raw);return
            reviewer=self.reviewer()
            if not reviewer:self.respond(401,dict(error='Please sign in.'));return
            if path=='/api/me':self.respond(200,dict(reviewer=reviewer));return
            ratings=read_ratings(db_path)
            if path=='/api/tasks':
                if summary_fn is not None:self.respond(200,summary_fn(packet,ratings))
                else:self.respond(200,dict(tasks=[dict(id=t['id'],participant=t['participant'],queue=t['queue'],rating=ratings.get(t['id'])) for t in packet['tasks']],
                    participants=packet['protocol']['participants'],evidence=packet['protocol']['evidence'],
                    priority_minutes=packet['protocol']['priority_minutes'],optional_minutes=packet['protocol']['optional_minutes']))
            elif path.startswith('/api/task/'):
                task=tasks.get(unquote(path[len('/api/task/'):]))
                self.respond(200,public_fn(task,ratings.get(task['id']))) if task else self.respond(404,dict(error='Unknown minute.'))
            elif path=='/api/export.csv':
                raw=csv_fn(packet,ratings);self.headers_out(200,'text/csv; charset=utf-8')
                filename=export_filename or ('direction_reference_labels.csv' if public_fn is not public_task else 'direction_targeted_choices.csv')
                self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
                self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            elif path=='/api/results' and results_fn is not None:
                try:self.respond(200,results_fn(packet,ratings))
                except ValueError as error:self.respond(409,dict(error=str(error)))
            else:self.respond(404,dict(error='Not found.'))

        def do_POST(self):
            origin=self.headers.get('Origin')
            if origin and (urlsplit(origin).scheme not in {'http','https'} or urlsplit(origin).netloc!=self.headers.get('Host')):
                self.respond(403,dict(error='Cross-origin request rejected.'));return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=64000 or self.headers.get_content_type()!='application/json':raise ValueError('Send a JSON request below 64 KB.')
                body=json.loads(self.rfile.read(size))
                if not isinstance(body,dict):raise ValueError('Expected a JSON object.')
            except (ValueError,json.JSONDecodeError) as error:self.respond(400,dict(error=str(error)));return
            path=urlsplit(self.path).path
            if path=='/api/login':
                try:
                    reviewer=normalize_reviewer(body.get('reviewer'));code=body.get('accessCode')
                    if not isinstance(code,str) or not hmac.compare_digest(code.encode(),access.encode()):
                        self.respond(401,dict(error='Incorrect access code.'));return
                except ValueError as error:self.respond(400,dict(error=str(error)));return
                encoded=base64.urlsafe_b64encode(json_bytes(dict(reviewer=reviewer,expires=int(time.time())+28800))).decode().rstrip('=')
                token=encoded+'.'+hmac.new(secret,encoded.encode(),hashlib.sha256).hexdigest()
                self.respond(200,dict(reviewer=reviewer),f'{cookie_name}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=28800');return
            reviewer=self.reviewer()
            if not reviewer:self.respond(401,dict(error='Please sign in.'));return
            if not path.startswith('/api/task/') or not path.endswith('/review'):
                self.respond(404,dict(error='Not found.'));return
            key=unquote(path[len('/api/task/'):-len('/review')])
            if key not in tasks:self.respond(404,dict(error='Unknown minute.'));return
            try:answer,version=validate_fn(body)
            except (ValueError,TypeError) as error:self.respond(400,dict(error=str(error)));return
            saved=save_rating(db_path,key,answer,version,reviewer)
            self.respond(200 if saved else 409,saved or dict(error='Another device saved this minute. Reload before saving.'))
    return Handler


def main():
    if (DEFAULT_OUT/'retired_from_current_evaluation.json').exists():
        from serve_direction_truth import main as truth_main
        return truth_main()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-file',type=Path,default=DEFAULT_OUT/'followup_packet.json.gz')
    parser.add_argument('--database',type=Path,default=DEFAULT_OUT/'followup_preferences.sqlite3')
    parser.add_argument('--port',type=int,default=8767)
    parser.add_argument('--local-session',action='store_true')
    args=parser.parse_args()
    if args.local_session:access,secret=secrets.token_urlsafe(18),secrets.token_bytes(32)
    else:access,secret=os.environ.get('ECG_FOLLOWUP_ACCESS_CODE',''),os.environ.get('ECG_FOLLOWUP_SESSION_SECRET','').encode()
    if len(access)<12 or len(secret)<32:parser.error('Set private ECG_FOLLOWUP_ACCESS_CODE and ECG_FOLLOWUP_SESSION_SECRET, or use --local-session.')
    with gzip.open(args.data_file,'rt') as f:packet=json.load(f)
    if packet.get('version')!=1 or not packet.get('tasks'):parser.error('Build the follow-up packet first.')
    if len({t['id'] for t in packet['tasks']})!=len(packet['tasks']):parser.error('Duplicate task IDs.')
    # Bind before opening storage, so a port collision cannot affect a database.
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(packet,args.database,access,secret))
    initialize_database(args.database,hashlib.sha256(args.data_file.read_bytes()).hexdigest())
    server.daemon_threads=True
    print(f'Follow-up: http://127.0.0.1:{args.port}/',flush=True)
    if args.local_session:print(f'Local access code: {access}',flush=True)
    print(f'{packet["protocol"]["priority_minutes"]} priority minutes; {packet["protocol"]["optional_minutes"]} optional checks. Existing services unchanged.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
