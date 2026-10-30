"""Freeze two methods, scan all source minutes and prepare unseen 3-minute audits."""
import ast
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv,gzip,hashlib,importlib.util,json,os
from pathlib import Path
import random,sys,time
import numpy as np

HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[2]
sys.path[:0]=[str(HERE),str(BASE)]
from build_direction_random20 import load_raw,indices,eligibility_reasons,MANIFEST
from build_p17_replacement import DEFAULT_OUT as PREVIOUS

DEFAULT_OUT=ROOT/'analysis/ecg_direction_flip_acceptance_20261029_v1'
SEED='20261029:unseen_flip_acceptance_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def key(r):return f"{r['participant']}_R{int(r['run_index']):03d}_B{int(r['block_index']):05d}"
def load_module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def worker(pid,out_dir):
 out=Path(out_dir);cache=out/f'checkpoint_{pid}.json'
 if cache.exists():return pid,json.loads(cache.read_text())
 helper=load_module(out/'frozen_sources/methods.py','frozen_flip_helpers')
 v3=load_module(out/'frozen_sources/direction_v3.py','frozen_v3').measure_direction_v3
 refined=helper.refined_method()
 protocol=json.loads((out/'protocol.json').read_text())
 source=[r for r in csv.DictReader(MANIFEST.open()) if r['participant']==pid]
 t,x=load_raw(pid);rows=[]
 for r in source:
  a,b=indices(t,r);start,end=float(r['start_ms']),float(r['end_ms']);issues=eligibility_reasons(t,x,r)
  if pid=='P17' and start<protocol['excluded_P17_interval'][1] and end>protocol['excluded_P17_interval'][0]:issues.append('shared_P19_source_excluded')
  row=dict(id=key(r),participant=pid,run_index=int(r['run_index']),block_index=int(r['block_index']),start_ms=start,end_ms=end,technical_pass=not issues,technical_reasons=issues)
  original=combined=dict(direction='technical_invalid',support='unavailable',reason=';'.join(issues))
  if not issues:
   mt,mx=t[a:b],x[a:b];fs=float(r['fs_hz']);full=v3(mx,mt,fs);original=helper.simple(full);combined=dict(original)
   if original['direction']=='severe_artifact':
    local=helper.local_recovery(mx,mt,fs,end-60000.,v3)
    if local['direction'] in {'positive','negative'}:combined={k:local[k] for k in ['direction','support','reason']}
   elif original['direction'] in {'ambiguous','low_signal'}:
    align=refined(mx,mt,fs)
    if align['raw_polarity'] in {'positive','negative'}:combined=helper.simple(align)
  row.update(v3_original=original,combined_candidate=combined);rows.append(row)
 dump(cache,rows);return pid,rows

def freeze(out):
 out.mkdir(mode=0o700);f=out/'frozen_sources';f.mkdir()
 original=ROOT/'analysis/ecg_direction_stability_diagnostics_20261029_v1/evaluate.py'
 gate=HERE/'direction_gate_review.py'
 pieces=[]
 for path,names in [(original,{'refined_method','simple','local_recovery','stable'}),(gate,{'aggregate_quarters'})]:
  text=path.read_text();tree=ast.parse(text)
  pieces.extend(ast.get_source_segment(text,node) for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names)
 (f/'direction_v3.py').write_bytes((HERE/'direction_v3.py').read_bytes())
 (f/'methods.py').write_text('from pathlib import Path\nimport numpy as np\nAUDIT=Path(__file__).resolve().parent\nSIGNED={"positive","negative"}\n\n'+'\n\n'.join(pieces)+'\n')
 old=json.load(gzip.open(PREVIOUS/'reference_packet.json.gz','rt'))
 source=list(csv.DictReader(MANIFEST.open()));lookup={key(r):r for r in source}
 seen={r['id'] for r in csv.DictReader((ROOT/'analysis/ecg_rodrigues_polarity_shadow_20261029_v1/results/minute_results.csv').open())}
 seen.update(t['id'].removeprefix('truth_') for t in old['tasks'])
 # Whole development neighborhoods are excluded, including previous peak contexts.
 for cid in list(seen):
  if cid not in lookup:continue
  row=lookup[cid]
  for delta in (-2,-1,1,2):
   candidate=key(dict(row,block_index=int(row['block_index'])+delta))
   if candidate in lookup:seen.add(candidate)
 dup=json.loads((ROOT/'analysis/ecg_direction_stability_diagnostics_20261029_v1/results/duplicate_recording_audit.json').read_text())
 hashes={str(p):sha(p) for p in [HERE/'direction_v3.py',original,gate,MANIFEST,PREVIOUS/'reference_packet.json.gz',BASE/'outputs/ecg_polarity_correction_all20_v1/correction_intervals.csv']}
 protocol=dict(version='unseen_three_minute_flip_acceptance_v1',seed=SEED,methods_frozen=True,
  frozen_source_hashes={p.name:sha(p) for p in f.iterdir()},protected_source_hashes=hashes,
  excluded_P17_interval=[dup['start_ms'],dup['end_ms']],excluded_seen_ids=sorted(seen),
  scan='All source minutes from all 20 participants. No raw repair or interpolation; shared P17/P19 source excluded from P17.',
  sample='At most 12 nonoverlapping unseen 3-minute windows: six differing flip decisions, two agreeing negative, two agreeing positive, two algorithm-unclear controls. Maximum two per participant. Fixed random seed; prediction strata hidden while labeling.',
  blind_review='Label each of three minutes positive, negative, balanced, mixed, unreadable or unsure on original raw ECG; no method outputs or peak marks shown.',
  acceptance='Do not promote the combined candidate if any newly flipped minute is positive/balanced/mixed/unreadable/unsure. Require at least one confirmed additional negative correction. Small targeted acceptance, not a population accuracy estimate.',
  production_corrections_modified=False,prior_labels_reused_as_new_answers=False,builder_sha256=sha(Path(__file__)))
 dump(out/'protocol.json',protocol)

def prepare(out,rows):
 protocol=json.loads((out/'protocol.json').read_text());helper=load_module(out/'frozen_sources/methods.py','frozen_helpers_prepare')
 accepted={m:helper.stable(rows,m) for m in ['v3_original','combined_candidate']}
 for r in rows:
  r['v3_flip']=accepted['v3_original'].get(r['id'])=='negative'
  r['combined_flip']=accepted['combined_candidate'].get(r['id'])=='negative'
 seen=set(protocol['excluded_seen_ids']);ordered=sorted(rows,key=lambda r:(int(r['participant'][1:]),r['run_index'],r['block_index']))
 pools={k:[] for k in ['difference','negative','positive','unclear']}
 for i in range(len(ordered)-2):
  tri=ordered[i:i+3]
  if not all(r['technical_pass'] and r['id'] not in seen for r in tri):continue
  if any(b['participant']!=a['participant'] or b['run_index']!=a['run_index'] or b['block_index']!=a['block_index']+1 or abs(b['start_ms']-a['end_ms'])>250 for a,b in zip(tri,tri[1:])):continue
  base=[r['v3_flip'] for r in tri];combo=[r['combined_flip'] for r in tri]
  kind=None
  if base!=combo and all(accepted['combined_candidate'].get(r['id'])=='negative' for r in tri):kind='difference'
  elif all(base) and base==combo:kind='negative'
  elif all(accepted[m].get(r['id'])=='positive' for r in tri for m in accepted):kind='positive'
  elif not any(base+combo) and sum(r['v3_original']['direction'] in {'severe_artifact','ambiguous','low_signal'} for r in tri)>=2:kind='unclear'
  if kind:pools[kind].append(tri)
 rng=random.Random(SEED);chosen=[];used=set();per_pid={}
 for kind,n in [('difference',6),('negative',2),('positive',2),('unclear',2)]:
  candidates=list(pools[kind]);rng.shuffle(candidates);taken=0
  for tri in candidates:
   pid=tri[0]['participant']
   if per_pid.get(pid,0)>=2 or any(r['id'] in used for r in tri):continue
   chosen.append((kind,tri));used.update(r['id'] for r in tri);per_pid[pid]=per_pid.get(pid,0)+1;taken+=1
   if taken==n:break
 rng.shuffle(chosen)
 # Selections are fixed before extracting/displaying any new waveform.
 dump(out/'private_selection.json',dict(pool_sizes={k:len(v) for k,v in pools.items()},selected=[dict(kind=k,minute_ids=[r['id'] for r in tri]) for k,tri in chosen]))
 tasks=[];private={};fingerprints=set()
 for pid in sorted({tri[0]['participant'] for _,tri in chosen},key=lambda p:int(p[1:])):
  t,x=load_raw(pid)
  for kind,tri in chosen:
   if tri[0]['participant']!=pid:continue
   start=tri[0]['end_ms']-60000.;end=tri[-1]['end_ms'];a,b=np.searchsorted(t,[start,end]);mt,mx=t[a:b],x[a:b]
   fp=hashlib.sha256(mt.astype('<f8').tobytes()+mx.astype('<f8').tobytes()).hexdigest()
   if fp in fingerprints:raise ValueError('Duplicate selected waveform; selection must be reviewed before labeling.')
   fingerprints.add(fp);task_id='flip_'+tri[0]['id']
   tasks.append(dict(id=task_id,participant=pid,queue='flip_acceptance',t=np.round((mt-start)/1000,4).tolist(),x=np.round(mx,2).tolist(),focusStartSec=0.,focusEndSec=180.))
   private[task_id]=dict(kind=kind,minute_ids=[r['id'] for r in tri],v3_flip=[r['v3_flip'] for r in tri],combined_flip=[r['combined_flip'] for r in tri],raw_fingerprint=fp)
 order={ 'flip_'+tri[0]['id']:i for i,(_,tri) in enumerate(chosen)};tasks.sort(key=lambda t:order[t['id']])
 dump(out/'private_predictions.json',private)
 with gzip.open(out/'review_packet.json.gz','wt') as f:json.dump(dict(version=1,tasks=tasks,protocol=protocol),f,separators=(',',':'),allow_nan=False)
 dump(out/'scan_summary.json',dict(total_source_minutes=len(rows),technical_valid=sum(r['technical_pass'] for r in rows),different_flip_minutes=sum(r['v3_flip']!=r['combined_flip'] for r in rows),pool_sizes={k:len(v) for k,v in pools.items()},review_windows=len(tasks),production_changed=False))
 print('Prepared blind acceptance:',json.loads((out/'scan_summary.json').read_text()),flush=True)

def main():
 started=time.monotonic()
 if not DEFAULT_OUT.exists():freeze(DEFAULT_OUT)
 protocol=json.loads((DEFAULT_OUT/'protocol.json').read_text())
 if (DEFAULT_OUT/'review_packet.json.gz').exists():raise SystemExit('Acceptance packet already frozen.')
 assert all(sha(DEFAULT_OUT/'frozen_sources'/p)==h for p,h in protocol['frozen_source_hashes'].items())
 rows=[]
 with ProcessPoolExecutor(max_workers=2) as executor:
  futures=[executor.submit(worker,f'P{i}',str(DEFAULT_OUT)) for i in range(1,21)]
  for future in as_completed(futures):
   pid,result=future.result();rows.extend(result);print(f'{pid}: {len(result)} minutes scanned; elapsed {time.monotonic()-started:.0f}s',flush=True)
 assert all(sha(Path(p))==h for p,h in protocol['protected_source_hashes'].items())
 prepare(DEFAULT_OUT,rows)

if __name__=='__main__':main()
