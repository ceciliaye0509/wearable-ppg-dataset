"""Small fixed whole-window audit of the single-source-block amplitude gate."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, gzip, hashlib, json, sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; BASE=HERE.parent; ROOT=BASE.parents[2]
sys.path[:0]=[str(BASE),str(ROOT/'analysis/ecg_detector_freeze_20261023_v1')]
from correct_ecg_polarity import apply_manifest_polarity,read_correction_intervals
from evaluate_polarity_corrected_ecg_rpeaks import detector_indices_method
from evaluate_assisted_human_confirmed_rpeaks import infer_fs,refine_to_local_qrs_extreme
from frozen_rules import technical_reasons,waveform_gate_for_clip
from remote_ecg_review.full_window import KIND,METHOD,prepare_full_window_supplement
from remote_ecg_review.peak_assistance import waveform_hash

LIVE=BASE/'outputs/ecg_rpeak_blind_review_remote_v1'
RELEASE=ROOT/'analysis/ecg_gt_release_20261029_v1'
DIRECTION=ROOT/'analysis/ecg_direction_combination_freeze_20261029_v1'
DEFAULT=ROOT/'analysis/ecg_amplitude_fullwindow_review_20261029_side_v1'
BATCH='amplitude_fullwindow_review_20261029_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');p.chmod(0o600)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=DEFAULT);args=ap.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=False)
 frames=[pd.read_csv(p,keep_default_na=False) for p in (RELEASE/'labels_final').glob('*_audit.csv')]
 data=pd.concat(frames,ignore_index=True)
 cases=data[data.ecg_label_qc_reason.eq('G1_severe_source_block')&data.severe_source_blocks.eq(1)]
 controls=data[data.ecg_label_qc_pass&data.severe_source_blocks.eq(0)]
 assert len(cases)==181,'Source cohort changed; review selection contract before proceeding'
 seen={};old_ids=set();packets=[LIVE/'review_data.json.gz',*sorted((LIVE/'supplements').glob('*.json.gz'))]
 for path in packets:
  with gzip.open(path,'rt') as f:packet=json.load(f)
  for clip in packet['clips']:
   seen.setdefault(clip['participant'],[]).append(float(clip['windowStart']));old_ids.add(clip['id'])
 original=json.load(gzip.open(LIVE/'review_data.json.gz','rt'))
 protocol=dict(version=1,seed=20261029,purpose='Audit whether a 5-second amplitude flag justifies rejecting a whole 300-second ECG reference',
  case_population='181 windows whose only rejection reason is one severe 5-second source block',
  selection='Fixed seed, six distinct participants from this population; two clean-control windows, at most two windows per participant; no overlap with existing installed review contexts or within this cohort; no new human labels or detector performance used',
  target_cases=6,target_controls=2,cohorts_not_population_weighted=True,
  first_pass='Rodrigues suggestions editable; automatic quality flags and case/control assignment withheld by server until reviewer saves complete 60-page first pass',
  reveal='Explicit reveal after completion; archive immutable first-pass snapshot per reviewer before showing flags; subsequent edits remain in revision history',
  future_analysis='Compare automatic and manual peaks on identical readable continuous runs; do not bridge unreadable intervals or gaps. Report usable seconds and beat coverage. Partial-window HRV must not be called complete 5-minute HRV. This audit does not change ECG ground truth or frozen thresholds.',
  polarity_manifest_sha256=sha(DIRECTION/'correction_intervals.csv'),label_protocol_sha256=sha(RELEASE/'release_protocol.json'),
  original_waveforms_and_annotations_modified=False)
 dump(out/'protocol.json',protocol)
 rng=np.random.default_rng(20261029);selected=[];counts={}
 def fresh(row):
  return all(abs(row.start_ms-start)>=300000 for start in seen.get(row.participant,[]))
 def choose(pool,n,kind,distinct):
  chosen=0
  for i in rng.permutation(len(pool)):
   row=pool.iloc[int(i)]
   if not fresh(row) or counts.get(row.participant,0)>=(1 if distinct else 2):continue
   selected.append(dict(row.to_dict(),private_stratum=kind));counts[row.participant]=counts.get(row.participant,0)+1
   seen.setdefault(row.participant,[]).append(row.start_ms);chosen+=1
   if chosen==n:return
  raise ValueError(f'Insufficient fresh non-overlapping {kind} windows; no automatic relaxation')
 choose(cases,6,'single_block_alert',True);choose(controls,2,'control',False)
 rng.shuffle(selected)
 dump(out/'selection_audit_private.json',dict(protocol_sha256=sha(out/'protocol.json'),selection=selected))
 clips=[];evidence=[];corrections=read_correction_intervals(DIRECTION/'correction_intervals.csv')
 for pid in sorted({r['participant'] for r in selected},key=lambda p:int(p[1:])):
  source=ROOT/'Multisite-PPG/raw_data'/pid/f'{pid}_polar_ecg_raw.npz';before=source.stat()
  with np.load(source,allow_pickle=True) as z:t=np.asarray(z['time_ms'],float);x=np.asarray(z['ecg_uv'],float)
  finite=np.flatnonzero(np.isfinite(t));t,x=t[finite[0]:finite[-1]+1],x[finite[0]:finite[-1]+1]
  assert np.isfinite(t).all() and not np.any(np.diff(t)<0)
  corrected,flips=apply_manifest_polarity(pid,x,t,corrections)
  for number,row in enumerate([r for r in selected if r['participant']==pid],1):
   start,end=row['start_ms'],row['end_ms'];a,b=np.searchsorted(t,start,'left'),np.searchsorted(t,end,'right')
   wt,wx=t[a:b],corrected[a:b]
   assert not technical_reasons(wt,x[a:b],start,end)
   gate=waveform_gate_for_clip(t,x,start,end)
   expected=1 if row['private_stratum']=='single_block_alert' else 0
   assert gate['assessable'] and gate['severe_blocks']==expected
   seeds=detector_indices_method(wx,infer_fs(wt),'rodrigues2021')
   times=refine_to_local_qrs_extreme(wx,wt,seeds,infer_fs(wt));peaks=np.searchsorted(wt,times).astype(int).tolist()
   cid=f'amp300_v1_{pid}_W{number:02d}';assert cid not in old_ids
   flags=[]
   for block in gate['blocks']:
    if not block['severe']:continue
    left=max(0,int(np.searchsorted(wt,block['start_ms'],'left')));right=min(len(wt)-1,int(np.searchsorted(wt,block['end_ms'],'left'))-1)
    if right<left:continue
    flags.append(dict(code='severe_source_block',label='5-second raw amplitude alert',scope='block',
     startIndex=left,endIndex=right,startSec=round((wt[left]-wt[0])/1000,6),endSec=round((wt[right]-wt[0])/1000,6),
     rawSpanUv=block['span_uv'],flippedSampleCount=int(flips[a+left:a+right+1].sum()),
     reason='The full source block P99−P01 exceeds 5,000 µV. This is an automatic amplitude alert, not a human unreadability label.'))
   clip=dict(id=cid,participant=pid,reviewId=cid,reviewSet=KIND,reviewProtocolKind=KIND,reviewBatch=BATCH,
    sourceLabel='Amplitude-rule full-window review',annotationMode='algorithm_assisted',blindReview=False,
    annotationScope='reviewer',windowDurationSec=300,reviewPageSeconds=5,deferredScreeningReview=True,
    windowStart=float(start),clipStart=float(wt[0]),startSec=0.,endSec=float((wt[-1]-wt[0])/1000),
    t=np.round((wt-wt[0])/1000,6).tolist(),x=np.round(wx,2).tolist(),suggested=peaks,restorePeaks=peaks.copy(),
    aidAvailable=bool(peaks),isPrevious=False,hasPriorMarks=False,priorUnreadable=False,samplingGaps=[],assistanceStatus='ok' if peaks else 'no_peaks',
    qualityFlags=flags,qualitySummary=dict(available=True,requiresReview=bool(flags),flagCount=len(flags),
     rawP01P99SpanUv=float(np.quantile(x[a:b],.99)-np.quantile(x[a:b],.01)),flippedSampleCount=int(flips[a:b].sum())),
    screeningRule=dict(rawAmplitudeThresholdUv=5000,blockSeconds=5,polarityVersion='direction_combination_frozen_20261029_v1'))
   clip['assistanceWaveformSha256']=waveform_hash(clip);clips.append(clip)
   evidence.append(dict(clip_id=cid,private_stratum=row['private_stratum'],source=str(source),source_size=before.st_size,
    source_mtime_ns=before.st_mtime_ns,source_window_index=row['source_window_index'],window_start_ms=start,
    waveform_sha256=clip['assistanceWaveformSha256'],source_blocks=gate['blocks'],suggested_peaks=len(peaks)))
  assert (source.stat().st_size,source.stat().st_mtime_ns)==(before.st_size,before.st_mtime_ns)
  print(pid,'complete 300-second waveform prepared',flush=True)
 packet=dict(version=2,reviewProtocolKind=KIND,participants=original['participants'],
  reviewBatches=[dict(id=BATCH,label='Amplitude-rule review · full 5 minutes',target=8,annotationScope='reviewer',default=True)],
  assistanceProtocol=dict(version=1,method=METHOD,generatedAt=datetime.now(timezone.utc).isoformat()),clips=clips)
 prepare_full_window_supplement(json.loads(json.dumps(packet)),'preflight')
 target=out/'supplement_amplitude_full300_v1.json.gz'
 with gzip.open(target,'wt') as f:json.dump(packet,f,separators=(',',':'),allow_nan=False)
 target.chmod(0o600)
 dump(out/'packet_verification.json',dict(packet_sha256=sha(target),windows=len(clips),cases=6,controls=2,evidence=evidence))
 print(json.dumps(dict(output=str(out),windows=len(clips),cases=6,controls=2)),flush=True)

if __name__=='__main__':main()
