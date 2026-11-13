"""Build a read-only history package, excluding the active 36-window review.
Only immutable existing annotation exports and their bound waveform packets are read.
"""
from pathlib import Path
import ast,collections,csv,gzip,hashlib,io,json,math,zipfile
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True,help='Private Daily_HRV research root; not the public code repository')
parser.add_argument('--source',type=Path,required=True,help='Existing complete historical annotation export ZIP')
parser.add_argument('--packet-candidates',type=Path,required=True,help='JSON list of bound private waveform packet paths')
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--correction',type=Path,help='Explicit newer corrected annotation CSV')
parser.add_argument('--correction-before',type=Path,help='Its preceding CSV, used to identify changed tasks only')
args=parser.parse_args()
R=args.root.resolve();HERE=Path(__file__).resolve().parent;OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
SOURCE=args.source.resolve()

sha=lambda b:hashlib.sha256(b).hexdigest()
def encoded(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def mapped(p):
 p=Path(p);old=R/'wearable-ppg-dataset/src'
 return R/'research_workspace/archive/legacy_source/src'/p.relative_to(old) if str(p).startswith(str(old)+'/') else p
# Preserve the previously delivered duration and sampling classification policy.
def classify(r):
 mode=r.get('review_mode','');queue=r.get('queue','');batch=r.get('review_batch','');review=r.get('review_set','')
 if mode=='raw_direction_labeling_v1':
  if queue=='main':return 'Random','confirmed_random','queue=main; source protocol: participant-stratified simple random sampling'
  if queue in ('difficult','optional'):return 'Difficult','confirmed_targeted','queue='+queue+'; original truth protocol: core flip-change minutes and optional targeted checks'
  if queue=='replacement':return 'Random','replacement_of_random','replacement in corrected main sample; preserve replaces_id; not an additional independent random draw'
 if mode=='beatwise_polarity_audit_v2_assisted':return 'Difficult','random_within_difficult_pool','Protocol samples raw_polarity=ambiguous minutes; random sampling within a restricted difficult pool'
 if mode=='amplitude_gate_interval_review_v1':return 'Difficult','confirmed_targeted','amplitude gate interval follow-up protocol'
 if mode=='three_minute_flip_acceptance_v1':return 'Difficult','confirmed_targeted','three-minute direction-flip acceptance follow-up'
 if mode in ('assisted_paired_direction_preference','assisted_targeted_three_method_preference'):return 'Difficult','confirmed_targeted','Method preference from disagreement/targeted comparison; not an absolute direction label'
 if batch=='full_window_hrv_v1':return 'Random','stratified_random','Builder uses seeded shuffled fresh contexts with participant polarity/quality stratum preferences; not unrestricted population SRS'
 if batch in ('random_ECG_audit_v4_2_v1','independent_blind_v1') or review=='random_v4_2_population_audit_v1':return 'Random','named_random_or_independent_batch','Original random audit / independent blind sampling batch; original protocol identity retained'
 rules={'priority_warning_review_v1':'warning-priority follow-up','challenge_blind_v1':'difficult/rule-boundary audit','startup_context_review_20261029_v1':'startup/context targeted follow-up','peak_targeted_20261029_v1':'targeted peak follow-up','peak_confirm_20261029_v1':'peak confirmation follow-up','v45_gt_high_correction_20261105_v1':'high RR correction burden follow-up','amplitude_fullwindow_review_20261029_v1':'amplitude-related full-window follow-up','rr_correction_discrepancy_20261029_v1':'RR correction discrepancy follow-up','saved_short_RR_recheck_v1':'saved short RR/boundary recheck','bidmc_icu_four_discrepant_terminal_windows_20261105':'four discrepant terminal windows'}
 if batch in rules:return 'Difficult','confirmed_targeted',rules[batch]+'; review_batch='+batch
 return 'Random','uncertain_default_random','Sampling category not established from retained record/protocol; placed in Random by user instruction; NOT proof of random sampling'

def duration(r,n):
 if '三分钟任务的逐分钟展开' in n:return '180s','parent_180s_derived_view'
 for key in ('nominal_duration_group','duration_group'):
  v=r.get(key,'')
  if v and v!='variable_duration':return v,'original_'+key
 sec=float(r.get('task_duration_sec') or 0)
 if sec>0:
  nearest=round(sec)
  if abs(sec-nearest)<=.05:return f'{nearest:03d}s','nominal_duration_nearest_second_within_50ms; exact task_duration_sec unchanged'
  return f'{sec:08.4f}'.rstrip('0').rstrip('.')+'s','task_duration_sec_display_rounded_to_0.1ms; original precision retained'
 return 'Unknown_duration','no_duration_in_original_record'

tables={};groups={};derived=[]
with zipfile.ZipFile(SOURCE) as z:
 for name in z.namelist():
  if not name.endswith('.csv') or not any(x in name for x in ['/01_当前有效人工标注/','/03_待完成与待核对/']):continue
  reader=csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig')));rows=list(reader)
  if not rows:continue
  if '三分钟任务的逐分钟展开' in name:derived.extend(rows);continue
  fields=reader.fieldnames;tables[name]=fields
  for row in rows:
   semantic='rpeak_events' if row.get('annotation_group_id') else ('reviewed_peak_indices_separate_coordinates' if '/R峰/' in name else ('method_preference_not_absolute_direction' if '方法偏好' in name else 'direction_decision'))
   key=(semantic,row.get('annotation_group_id') or row.get('task_id'),row.get('reviewer_id') or row.get('reviewer',''))
   groups.setdefault(key,dict(semantic=semantic,table=name,rows=[]))['rows'].append(row)
# The already verified 2026-11-13 single-mark correction replaces its older value.
correction=args.correction;correction_before=args.correction_before;patches=[]
if correction and correction_before and correction.exists() and correction_before.exists():
 old=list(csv.DictReader(correction_before.open(encoding='utf-8-sig')));new=list(csv.DictReader(correction.open(encoding='utf-8-sig')));assert len(old)==len(new)
 changed={a['clip_id'] for a,b in zip(old,new) if a!=b}
 for group in groups.values():
  rows=group['rows'];first=rows[0]
  if group['semantic']!='rpeak_events' or first['clip_id'] not in changed:continue
  latest=[x for x in new if x['clip_id']==first['clip_id'] and x['reviewer_id']==first['reviewer_id']];assert len(latest)==len(rows)
  merged=[]
  for prior,updated in zip(rows,latest):
   assert prior['clip_id']==updated['clip_id'] and prior['record_type']==updated['record_type']
   merged.append({**prior,**updated})
  group['rows']=merged;group['correction_source']=str(correction);patches.append(dict(clip_id=first['clip_id'],reviewer=first['reviewer_id'],source_sha256=sha(correction.read_bytes()),reason='Newer verified single native-sample correction; replaces older exported mark.'))
# Logical identity retains distinct reviewers, protocols and native coordinate systems.
latest={};dedup=[]
for group in groups.values():
 row=group['rows'][0];identity=(group['semantic'],row.get('dataset',''),row.get('clip_id') or row.get('task_id'),row.get('reviewer_id') or row.get('reviewer'),row.get('task_duration_sec'),row.get('waveform_sha256') or row.get('packet_sha256') or row.get('clip_start_unix_ms'))
 rank=(bool(group.get('correction_source')),row.get('annotation_updated_at') or row.get('updated_at') or '',int(row.get('source_version') or row.get('version') or 0))
 group['rank']=rank
 if identity in latest:
  a=latest[identity];keep,drop=(group,a) if rank>a['rank'] else (a,group);latest[identity]=keep;dedup.append(dict(identity=list(identity),kept_rank=list(keep['rank']),removed_rank=list(drop['rank'])))
 else:latest[identity]=group
# Collect the original packets; never read the active 36-window app or its reviews.
paths=set(map(Path,json.loads(args.packet_candidates.read_text())))
base=R/'research_workspace/archive/legacy_source/src/heuristic_baselines/outputs/ecg_rpeak_blind_review_remote_v1'
paths.update((base/'supplements').glob('*.json.gz'))
for group in latest.values():
 p=group['rows'][0].get('packet_source')
 if p:paths.add(mapped(p))
packets=collections.defaultdict(list);packet_provenance=[]
for path in sorted(paths):
 if not path.exists():continue
 raw=path.read_bytes()
 try:value=json.loads(gzip.decompress(raw) if path.name.endswith('.gz') else raw)
 except Exception:continue
 tasks=next((value[k] for k in ['clips','tasks','items'] if isinstance(value.get(k),list)),[])
 if not tasks:continue
 digest=sha(raw);packet_provenance.append(dict(path=str(path),sha256=digest,tasks=len(tasks)))
 for task in tasks:
  if 'id' in task and isinstance(task.get('t'),list) and isinstance(task.get('x'),list):packets[task['id']].append((task,path,digest))

def bind(group):
 row=group['rows'][0];cid=row.get('clip_id') or row.get('task_id');candidates=packets.get(cid,[]);expected=row.get('waveform_sha256');packet_hash=row.get('packet_sha256');good=[]
 for clip,path,digest in candidates:
  raw=encoded(dict(t=clip['t'],x=clip['x']));h=sha(raw)
  if expected and h!=expected:
   # Alignment packets hash original absolute <f8 time/amplitude bytes, not displayed JSON.
   # Their complete packet hash and embedded native hash must both match the saved record.
   if not (row.get('review_mode')=='alignment_direction_acceptance_v1' and packet_hash==digest and clip.get('waveform_sha256')==expected):continue
  if packet_hash and digest!=packet_hash:continue
  if group['semantic']=='rpeak_events':
   origin=float(row.get('clip_start_unix_ms') or clip.get('clipStart') or 0)
   if 'clipStart' in clip and abs(float(clip['clipStart'])-origin)>.001:continue
   ok=True
   for point in group['rows']:
    if point['record_type']!='peak':continue
    index=point.get('peak_sample_index')
    if not index:ok=False;break
    idx=int(index)
    if not 0<=idx<len(clip['t']):ok=False;break
    if abs(origin+1000*clip['t'][idx]-float(point['manual_rpeak_abs_unix_ms']))>.06:ok=False;break
   if not ok:continue
  good.append((clip,path,digest,h,raw))
 if not good:return None,'No packet satisfied task identity, waveform hash and native peak coordinates.'
 unique={x[3] for x in good}
 if len(unique)>1:return None,'Multiple differing waveforms share this task identity; no silent selection.'
 return good[0],''
items=[];payloads={};csv_buckets=collections.defaultdict(list);unavailable=[];bound=0
for group in sorted(latest.values(),key=lambda g:(g['table'],g['rows'][0].get('clip_id') or g['rows'][0].get('task_id'),g['rows'][0].get('reviewer_id') or g['rows'][0].get('reviewer',''))):
 rows=group['rows'];row=rows[0];category,basis,reason=classify(row);dur,dur_basis=duration(row,group['table']);kind='R_peak' if group['semantic'] in ['rpeak_events','reviewed_peak_indices_separate_coordinates'] else ('Method_preference' if group['semantic']=='method_preference_not_absolute_direction' else 'Direction')
 cid=row.get('clip_id') or row.get('task_id');reviewer=row.get('reviewer_id') or row.get('reviewer','');ident=sha(encoded([group['semantic'],cid,reviewer,row.get('packet_sha256'),row.get('clip_start_unix_ms')]))[:20];folder=f'{dur}/{category}/{kind}'
 records_path=f'{folder}/records/{ident}.json.gz';payloads[records_path]=gzip.compress(encoded(dict(rows=rows,semantic=group['semantic'],classificationBasis=basis,classificationReason=reason,sourceTable=Path(group['table']).name,correctionApplied=bool(group.get('correction_source')))),mtime=0)
 binding,error=bind(group);waveform=None;peaks=[];unreadable=[]
 if binding:
  clip,path,packet_hash,wave_hash,raw=binding
  t,x=clip['t'],clip['x']
  if len(t)!=len(x) or any(not math.isfinite(v) for v in t+x):raise ValueError('Non-finite or inconsistent historical waveform: '+cid)
  # Historical time anomalies remain visible; never sort, fill or invent samples.
  wave_path=f'{folder}/waveforms/{ident}.json.gz';compressed=gzip.compress(raw,mtime=0);payloads[wave_path]=compressed
  waveform=dict(file=wave_path,sha256=sha(compressed),sampleCount=len(t),durationSec=t[-1]-t[0],sourcePacketSha256=packet_hash,sourceWaveformSha256=wave_hash)
  if group['semantic']=='rpeak_events':
   peaks=sorted(set(int(p['peak_sample_index']) for p in rows if p['record_type']=='peak'))
   unreadable=[[float(p['unreadable_start_sec']),float(p['unreadable_end_sec'])] for p in rows if p['record_type'] in ['unreadable','unreadable_interval'] and p.get('unreadable_start_sec') and p.get('unreadable_end_sec')]
  elif group['semantic']=='reviewed_peak_indices_separate_coordinates':peaks=json.loads(row.get('peak_indices_json') or '[]')
  bound+=1
 else:unavailable.append(dict(id=ident,clip_id=cid,semantic=group['semantic'],reason=error))
 status=row.get('review_status') or row.get('status') or 'unknown';summary={k:row[k] for k in ['direction','choice','confidence','notes','review_mode','review_batch','minute_directions_json','qrs_readability','direction_determinability','peaks_reviewed','annotation_updated_at','updated_at','source_version','version'] if row.get(k)}
 items.append(dict(id=ident,kind=kind,semantic=group['semantic'],clipId=cid,participant=row.get('participant',''),reviewer=reviewer,durationGroup=dur,samplingGroup=category,status=status,recordsFile=records_path,recordCount=len(rows),waveform=waveform,waveformUnavailableReason=error,peaks=peaks,unreadable=unreadable,summary=summary,classificationBasis=basis))
 csv_buckets[f'{folder}/{Path(group["table"]).name}'].extend(rows)
for name,rows in csv_buckets.items():
 fields=list(dict.fromkeys(k for r in rows for k in r));f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);payloads[name]=('\ufeff'+f.getvalue()).encode()
# Derived minute views remain attached to their parent, not extra independent tasks.
if derived:payloads['Provenance/derived_minute_views.json']=encoded(derived)
manifest=dict(format='ecg-annotation-archive',version=1,title='历史人工ECG标注 · 时长 / Random / Difficult',items=items,sourceSnapshotSha256=sha(SOURCE.read_bytes()),active36WindowReviewIncluded=False)
payloads['history_manifest.json']=encoded(manifest)
receipt=dict(sourceSnapshot=str(SOURCE),sourceSnapshotSha256=sha(SOURCE.read_bytes()),tasks=len(items),tasksByKind=dict(collections.Counter(i['kind'] for i in items)),tasksByStatus=dict(collections.Counter(i['status'] for i in items)),waveformBound=bound,waveformUnavailable=len(unavailable),deduplicatedOldVersions=len(dedup),newerCorrectionsApplied=patches,active36WindowReviewIncluded=False,current59ReviewPackagesIncluded=False,derivedMinuteRows=len(derived),originalSourceModified=False)
payloads['Provenance/verification.json']=encoded(receipt);payloads['Provenance/deduplication.json']=encoded(dedup);payloads['Provenance/unavailable_waveforms.json']=encoded(unavailable);payloads['Provenance/source_packets.json']=encoded(packet_provenance)
payloads['开始查看.txt']=('打开Cloudflare网页，拖入整个ZIP。按时长→Random/Difficult→类别/参与者筛选。\n这是历史只读标注，不会修改人工数据库、GT或当前36窗审核。\n仅保留当前有效最新记录；含1个BIDMC未完成任务，状态明确保留。\nRandom内部分旧任务的抽样依据不明确，沿用整理版归类并显示uncertain_default_random，不据此认定为简单随机样本。\n方法偏好不等于方向真值；独立坐标峰不混为主R峰标注。无可靠波形绑定的任务仍可查看全部原始标注记录。\n').encode()
archive=OUT/'全部已有人工标注_历史查看包_按时长及难易_20261113.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in sorted(payloads.items()):z.writestr('ECG_History_20261113/'+name,data)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
receipt.update(zip=str(archive),zipBytes=archive.stat().st_size,zipSha256=sha(archive.read_bytes()))
(OUT/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False,indent=2));print('UNAVAILABLE',json.dumps(unavailable[:8],ensure_ascii=False))
