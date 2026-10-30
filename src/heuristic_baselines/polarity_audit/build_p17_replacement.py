"""Replace four provenance-affected P17 main tasks without erasing saved records."""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
from importlib.util import module_from_spec,spec_from_file_location
import json
from pathlib import Path
import random
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parents[2]
sys.path[:0]=[str(BASE),str(HERE)]
from build_direction_random20 import DEFAULT_OUT as PREVIOUS,MANIFEST,load_raw,indices,eligibility_reasons
from build_direction_truth import sha,write_json,METHODS
from correct_ecg_polarity import _block_measurement
from polarity_audit.direction_v3 import measure_direction_v3
from assisted_direction import context_candidates,minute_anchors,measure_at_anchors
from frozen_rules import technical_reasons

DEFAULT_OUT=ROOT/'analysis/ecg_direction_p17_replacement_20261029_v1'
DIAGNOSTICS=ROOT/'analysis/ecg_direction_stability_diagnostics_20261029_v1'
SEED='20261029:P17:provenance_replacement_v1'
BAD_IDS=['truth_P17_R015_B00022','truth_P17_R015_B00039',
         'truth_P17_R015_B00473','truth_P17_R015_B00560']


def key(row):return f"{row['participant']}_R{int(row['run_index']):03d}_B{int(row['block_index']):05d}"


def overlaps(a,b):return a[0]<b[1] and b[0]<a[1]


def choose(pool):
    ordered=sorted(pool,key=lambda r:(int(r['run_index']),int(r['block_index'])))
    if len(ordered)<4:raise ValueError('Fewer than four eligible independent P17 minutes.')
    return random.Random(SEED).sample(ordered,4)


def main():
    if DEFAULT_OUT.exists():raise SystemExit('Existing replacement sets are never overwritten.')
    old_packet_path=PREVIOUS/'reference_packet.json.gz'
    old=json.load(gzip.open(old_packet_path,'rt'))
    predictions=json.loads((PREVIOUS/'private_predictions.json').read_text())['predictions']
    if sha(PREVIOUS/'private_predictions.json')!=old['protocol']['private_predictions_sha256']:
        raise ValueError('Original frozen predictions changed.')
    source=list(csv.DictReader(MANIFEST.open()));lookup={key(r):r for r in source}
    provenance_path=DIAGNOSTICS/'results/source_provenance.json'
    provenance=json.loads(provenance_path.read_text())
    duplicate_path=DIAGNOSTICS/'results/duplicate_recording_audit.json'
    duplicate=json.loads(duplicate_path.read_text())
    blocked_file=(float(duplicate['start_ms']),float(duplicate['end_ms']))
    existing={t['id'] for t in old['tasks']}
    p19=[lookup[t['id'].removeprefix('truth_')] for t in old['tasks'] if t['participant']=='P19']
    p19_intervals=[(float(r['end_ms'])-60000.,float(r['end_ms'])) for r in p19]
    p19_waveforms={hashlib.sha256(np.asarray(t['x'],dtype='<f8').tobytes()).hexdigest() for t in old['tasks'] if t['participant']=='P19'}
    assert len(BAD_IDS)==4 and all(t in existing for t in BAD_IDS)
    protected=[old_packet_path,PREVIOUS/'private_predictions.json',MANIFEST,
        BASE/'correct_ecg_polarity.py',HERE/'direction_v3.py',HERE/'beatwise.py',
        BASE/'outputs/ecg_polarity_correction_all20_v1/correction_intervals.csv',
        ROOT/'Multisite-PPG/raw_data/P17/P17_polar_ecg_raw.npz',provenance_path,duplicate_path]
    hashes={str(p):sha(p) for p in protected}
    t,x=load_raw('P17');eligible=[];excluded=[]
    for r in [r for r in source if r['participant']=='P17']:
        interval=(float(r['end_ms'])-60000.,float(r['end_ms']))
        reasons=[]
        if 'truth_'+key(r) in existing:reasons.append('previously_available_P17_task')
        if overlaps(interval,blocked_file):reasons.append('entire_shared_P19_source_file_excluded')
        if any(overlaps(interval,b) for b in p19_intervals):reasons.append('overlaps_P19_selected_interval')
        if not reasons:reasons.extend(eligibility_reasons(t,x,r))
        if not reasons:
            a,b=indices(t,r)
            digest=hashlib.sha256(np.round(x[a:b],2).astype('<f8').tobytes()).hexdigest()
            if digest in p19_waveforms:reasons.append('same_waveform_as_selected_P19_task')
        if reasons:excluded.append(dict(id=key(r),reasons=reasons))
        else:eligible.append(r)
    selected=choose(eligible)
    DEFAULT_OUT.mkdir(mode=0o700)
    write_json(DEFAULT_OUT/'replacement_selection.json',dict(seed=SEED,selected=[dict(id=key(r),**r) for r in selected],
        eligible_pool=len(eligible),exclusions=excluded,algorithm_outputs_used_for_selection=False,
        human_answers_used_for_selection=False,shared_source_file=provenance['shared_source_filename'],
        excluded_entire_source_interval_ms=blocked_file,
        excluded_P19_task_ids=[t['id'] for t in old['tasks'] if t['participant']=='P19']))
    print('Four P17 replacements fixed before prediction:',[key(r) for r in selected],flush=True)
    tasks=[dict(v) for v in old['tasks']]
    for task in tasks:
        task['in_corrected_main']=task['queue']=='main' and task['id'] not in BAD_IDS
        task['superseded_in_corrected_main']=task['id'] in BAD_IDS
    raw_fingerprints={};experimental={}
    spec=spec_from_file_location('frozen_offline_direction_candidate',DIAGNOSTICS/'evaluate.py')
    module=module_from_spec(spec);spec.loader.exec_module(module)
    refined=module.refined_method()
    for old_id,r in zip(BAD_IDS,selected):
        start,end,fs=float(r['start_ms']),float(r['end_ms']),float(r['fs_hz'])
        a,b=indices(t,r);mt,mx=t[a:b],x[a:b]
        task_id='truth_'+key(r)
        assert not overlaps((end-60000.,end),blocked_file)
        assert not any(overlaps((end-60000.,end),q) for q in p19_intervals)
        signal_hash=hashlib.sha256(np.round(mx,2).astype('<f8').tobytes()).hexdigest()
        assert signal_hash not in p19_waveforms
        tasks.append(dict(id=task_id,participant='P17',queue='replacement',
            t=np.round((mt-(end-60000.))/1000,4).tolist(),x=np.round(mx,2).tolist(),
            focusStartSec=0.,focusEndSec=60.,in_corrected_main=True,
            superseded_in_corrected_main=False,replaces_id=old_id))
        old_result=_block_measurement(mx,mt,fs)
        assert old_result[-1]==r['raw_polarity']
        v3=measure_direction_v3(mx,mt,fs)
        run=[z for z in source if z['participant']=='P17' and z['run_index']==r['run_index']]
        rs,re=min(float(z['start_ms']) for z in run),max(float(z['end_ms']) for z in run)
        cs=max(rs,min(start-120000.,re-300000.));ce=cs+300000.
        ca,cb=np.searchsorted(t,[cs,ce]);ct,cx=t[ca:cb],x[ca:cb]
        issues=technical_reasons(ct,cx,cs,ce)
        if re-rs<300000:issues.append('continuous_run_shorter_than_300s')
        if overlaps((cs,ce),blocked_file):issues.append('context_intersects_excluded_source_file')
        rod=dict(raw_polarity='technical_unavailable',cross_check_status='unavailable',reason=';'.join(issues))
        if not issues:
            if v3['raw_polarity']=='severe_artifact':rod=measure_at_anchors(mx,mt,fs,[])
            else:
                info=context_candidates(cx,ct,fs)
                if info['failures']:
                    rod=dict(raw_polarity='detector_failure',cross_check_status='unavailable',reason=';'.join(info['failures']))
                else:rod=measure_at_anchors(mx,mt,fs,minute_anchors(info['rodrigues'],ct,mt,a-ca,fs))
        predictions[task_id]={
            'old_v1':dict(direction=old_result[-1],support='historical_rule',flip=None),
            'v3_minute':dict(direction=v3['raw_polarity'],support=v3['cross_check_status'],flip=None),
            'rod300':dict(direction=rod['raw_polarity'],support=rod['cross_check_status'],flip=None)}
        align=refined(mx,mt,fs);candidate=module.simple(v3)
        if v3['raw_polarity']=='severe_artifact':
            local=module.local_recovery(mx,mt,fs,end-60000.,measure_direction_v3)
            if local['direction'] in {'positive','negative'}:candidate={k:local[k] for k in ['direction','support','reason']}
        elif v3['raw_polarity'] in {'ambiguous','low_signal'} and align['raw_polarity'] in {'positive','negative'}:
            candidate=module.simple(align)
        experimental[task_id]=dict(v3_original=module.simple(v3),alignment_diagnostic=module.simple(align),combined_candidate=candidate)
        raw_fingerprints[task_id]=dict(signal_sha256=hashlib.sha256(mx.astype('<f8').tobytes()).hexdigest(),
            absolute_time_sha256=hashlib.sha256(mt.astype('<f8').tobytes()).hexdigest())
    corrected=[v for v in tasks if v['in_corrected_main']]
    assert len(tasks)==179 and len(corrected)==160
    assert Counter(v['participant'] for v in corrected)=={f'P{i}':8 for i in range(1,21)}
    protocol=dict(old['protocol'],version='raw_direction_truth_v3',source_packet_sha256=sha(old_packet_path),
        source_packet=str(old_packet_path),main_minutes=160,corrected_main_minutes=160,replacement_minutes=4,
        superseded_main_ids=BAD_IDS,replacement_ids=[t['id'] for t in tasks if t['queue']=='replacement'],
        replacement_seed=SEED,replacement_pool_minutes=len(eligible),
        repair='Retain the historical 160 main records and every saved answer. Corrected main = 156 retained originals + four newly labeled P17 tasks. The whole shared P19 source file, all existing P17 tasks and all selected P19 intervals/waveforms are excluded before random sampling. Source files and participant assignments are not edited.',
        corrected_scope='Provenance repair of the existing development benchmark, not a new independent all-participant validation. Complete all four fresh replacement labels before showing corrected scores. Historical affected records remain available but never enter corrected scores.',
        source_note='Uses the unchanged original unflipped local ECG mirror. New HF polarity-corrected downloads are not substituted.',
        protected_repair_hashes=hashes,builder_sha256=sha(Path(__file__)))
    write_json(DEFAULT_OUT/'private_predictions.json',dict(methods=list(METHODS),predictions=predictions))
    protocol['private_predictions_sha256']=sha(DEFAULT_OUT/'private_predictions.json')
    with gzip.open(DEFAULT_OUT/'reference_packet.json.gz','wt') as f:
        json.dump(dict(version=1,tasks=tasks,protocol=protocol),f,separators=(',',':'),allow_nan=False)
    write_json(DEFAULT_OUT/'protocol.json',protocol)
    write_json(DEFAULT_OUT/'private_candidate_predictions.json',dict(source= str(DIAGNOSTICS/'evaluate.py'),source_sha256=sha(DIAGNOSTICS/'evaluate.py'),predictions=experimental,raw_fingerprints=raw_fingerprints,labels_consulted=False))
    assert all(sha(Path(p))==digest for p,digest in hashes.items())
    print(json.dumps(dict(output=str(DEFAULT_OUT),replacement_minutes=4,corrected_main_minutes=160,
        eligible_pool=len(eligible),P19_overlap_count=0,shared_source_file_selected=False,old_records_retained=True),indent=2),flush=True)


if __name__=='__main__':main()
