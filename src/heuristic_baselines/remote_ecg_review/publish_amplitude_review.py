"""Install a tested additive amplitude-audit packet on the existing local host."""
from contextlib import closing
from datetime import datetime,timezone
from http.cookies import SimpleCookie
from pathlib import Path
import argparse,json,os,re,shlex,shutil,signal,sqlite3,subprocess,tempfile,time
from publish_full_window import request,listener,sha
from serve_review import load_packet

APP=Path(__file__).resolve().parent;DATA=APP.parent/'outputs/ecg_rpeak_blind_review_remote_v1'
ROOT=APP.parents[3];OUT=ROOT/'analysis/ecg_amplitude_fullwindow_review_20261029_side_v1'

def snapshot(path):
 with closing(sqlite3.connect(f'file:{path}?mode=ro',uri=True)) as db:
  db.row_factory=sqlite3.Row
  names=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
  rows={name:sorted([dict(row) for row in db.execute('SELECT * FROM "'+name+'"')],key=lambda r:json.dumps(r,sort_keys=True)) for name in names}
  if not rows.get('screening_review_reveals'):rows.pop('screening_review_reveals',None)
  return rows

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--reload-only',action='store_true');args=parser.parse_args()
 pids=listener()
 if len(pids)!=1:raise RuntimeError('Expected exactly one existing annotation server; no service changed')
 pid=int(pids[0]);source=OUT/'supplement_amplitude_full300_v1.json.gz';target=DATA/'supplements'/source.name
 if target.exists() and not args.reload_only:raise RuntimeError('Packet already installed; inspect existing deployment before repeating')
 if args.reload_only:assert target.exists() and sha(target)==sha(source)
 assert sha(source)==json.loads((OUT/'packet_verification.json').read_text())['packet_sha256']
 command=subprocess.run(['/bin/ps','-p',str(pid),'-o','command='],capture_output=True,text=True,check=True).stdout
 argv=shlex.split(command);index=next((i for i,v in enumerate(argv) if v.endswith('/serve_review.py')),-1)
 if index not in [1,2] or Path(argv[index]).resolve()!=APP/'serve_review.py':raise RuntimeError('Unexpected listener; untouched')
 env_result=subprocess.run(['/bin/ps','eww','-p',str(pid),'-o','command='],capture_output=True,text=True,check=True)
 secrets={}
 for name in ['ECG_REVIEW_ACCESS_CODE','ECG_REVIEW_SESSION_SECRET']:
  match=re.search(r'(?:^|\s)'+name+r'=(\S+)',env_result.stdout)
  if not match:raise RuntimeError('Existing service credentials unavailable; service untouched')
  secrets[name]=match.group(1)
 del env_result
 status,headers,_=request('POST','/api/login',body=dict(reviewer='Deployment verification',accessCode=secrets['ECG_REVIEW_ACCESS_CODE']))
 assert status==200
 cookies=SimpleCookie();cookies.load(headers['Set-Cookie']);cookie='ecg_review_session='+cookies['ecg_review_session'].value
 status,_,body=request('GET','/api/participants',cookie);assert status==200;previous_progress=json.loads(body)
 old=load_packet(DATA/'review_data.json.gz')
 with tempfile.TemporaryDirectory(prefix='ecg-amplitude-preflight-') as temporary:
  temporary=Path(temporary);(temporary/'supplements').mkdir()
  for name in ['review_data.json.gz','supplement_peak_assistance.json']:(temporary/name).symlink_to(DATA/name)
  for path in (DATA/'supplements').glob('*.json.gz'):(temporary/'supplements'/path.name).symlink_to(path)
  if not args.reload_only:(temporary/'supplements'/source.name).symlink_to(source)
  prepared=load_packet(temporary/'review_data.json.gz')
 old_by_id={c['id']:c for c in old['clips']};added=[c for c in prepared['clips'] if c.get('reviewBatch')=='amplitude_fullwindow_review_20261029_v1']
 assert len(added)==8 and all(c==old_by_id[c['id']] for c in prepared['clips'] if c['id'] in old_by_id)
 hashes={str(p):sha(p) for p in [DATA/'review_data.json.gz',DATA/'supplement_peak_assistance.json',*list((DATA/'supplements').glob('*.json.gz'))]}
 environment=dict(os.environ);environment.update(secrets);launch=argv if index==2 else [argv[0],'-B',*argv[1:]]
 def start():
  with (OUT/'service.log').open('ab') as log:
   (OUT/'service.log').chmod(0o600)
   return subprocess.Popen(launch,cwd=APP.parents[2],env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 os.kill(pid,signal.SIGINT)
 for _ in range(100):
  if not listener():break
  time.sleep(.1)
 else:raise RuntimeError('Existing server did not stop; no second server started')
 database=DATA/'annotations.sqlite3';before=snapshot(database)
 backup=DATA/'backups'/('annotations_before_amplitude_review_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'.sqlite3')
 child=None
 try:
  with closing(sqlite3.connect(f'file:{database}?mode=ro',uri=True)) as db,closing(sqlite3.connect(backup)) as saved:db.backup(saved)
  backup.chmod(0o600);assert snapshot(backup)==before
  if not args.reload_only:shutil.copyfile(source,target);target.chmod(0o600)
  child=start()
  for _ in range(100):
   if child.poll() is not None:raise RuntimeError('Updated server exited')
   try:
    status,_,body=request('GET','/api/participants',cookie)
    if status==200:current=json.loads(body);break
   except OSError:pass
   time.sleep(.1)
  else:raise RuntimeError('Updated server did not become ready')
  assert sum(p['total'] for p in current['participants'])==len(prepared['clips'])
  for previous in previous_progress['participants']:
   now=next(p for p in current['participants'] if p['participant']==previous['participant'])
   assert all(now['batches'][k]==v for k,v in previous['batches'].items())
  for clip in added:
   status,_,body=request('GET','/api/clip/'+clip['id'],cookie);value=json.loads(body)
   assert status==200 and value['x']==clip['x'] and value['t']==clip['t'] and value['peaks']==clip['suggested']
   if not args.reload_only:
    assert value['screeningWithheld'] and not value['qualityFlags'] and not value['qualitySummary']['available']
    assert value['reviewedPages']==[] and value['status']=='unreviewed'
  assert snapshot(database)==before
  assert all(sha(Path(p))==value for p,value in hashes.items())
  result=dict(status='live',url='http://127.0.0.1:8765/',pid=child.pid,new_windows=8,total_clips=len(prepared['clips']),
   backup=str(backup),all_earlier_packets_and_database_rows_preserved=True,new_screening_hidden_before_first_pass=True,
   waveform_and_candidate_indices_verified=True,existing_binding_and_credentials_preserved=True,
   packet_sha256=sha(target),verified_at_utc=datetime.now(timezone.utc).isoformat())
  (OUT/'deployment_report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 except Exception:
  if child and child.poll() is None:child.send_signal(signal.SIGINT);child.wait(timeout=10)
  if target.exists() and not args.reload_only:target.rename(target.with_name(target.name+'.disabled'))
  restored=start();print(json.dumps(dict(status='old_packet_service_restored',pid=restored.pid,database_not_overwritten=True)))
  raise

if __name__=='__main__':main()
