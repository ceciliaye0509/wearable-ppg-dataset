"""Start a fresh waveform-only reference set. Retire, never import old answers."""
from __future__ import annotations

import argparse
from contextlib import closing
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parents[2]
sys.path.insert(0,str(BASE))
from correct_ecg_polarity import _split_runs
from polarity_audit.build_targeted_followup import DEFAULT_OUT as PREVIOUS,SHADOW,read_rows,accepted_direction

DEFAULT_OUT=ROOT/'analysis/ecg_direction_truth_20261029_v1'
METHODS=('old_v1','v3_minute','rod300')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if args.out_dir.exists():raise SystemExit('Existing reference sets are never overwritten; choose a new directory.')
    source=PREVIOUS/'followup_packet.json.gz'
    with gzip.open(source,'rt') as f:previous=json.load(f)
    rows={r['id']:r for r in read_rows(SHADOW/'minute_results.csv')}
    decisions={r['id']:r for r in read_rows(SHADOW/'v1_vs_shadow_interval_decisions_by_minute.csv')}
    intervals={m:read_rows(SHADOW/f'candidate_intervals_{m}.csv') for m in ['v3_minute','rod300']}
    # Restore ALL 12 changed minutes, including the minute formerly omitted
    # because it had an A/B preference. Old answers cannot affect eligibility.
    queues={r['id']:'priority' for r in read_rows(SHADOW/'core_flip_changes_for_review.csv')}
    for task in previous['tasks']:
        if task['queue']=='optional':queues[task['id'].removeprefix('followup_')]='optional'
    old_waveforms={t['id'].removeprefix('followup_'):t for t in previous['tasks']}
    tasks,predictions=[],{}
    for key,queue in sorted(queues.items(),key=lambda kv:(kv[1]!='priority',int(rows[kv[0]]['participant'][1:]),kv[0])):
        row=rows[key]
        if row['technical_pass']!='True':raise ValueError('An ineligible source minute entered the reference set.')
        if key in old_waveforms:
            old=old_waveforms[key];t,x=old['t'],old['x'];duration=old['focusEndSec']
        else:
            pid=row['participant']
            with np.load(ROOT/'Multisite-PPG/raw_data'/pid/f'{pid}_polar_ecg_raw.npz',allow_pickle=False) as data:
                times,signal=np.asarray(data['time_ms'],float),np.asarray(data['ecg_uv'],float)
            finite=np.flatnonzero(np.isfinite(times));times,signal=times[finite[0]:finite[-1]+1],signal[finite[0]:finite[-1]+1]
            a,b=_split_runs(times)[int(row['run_index'])];rt,rx=times[a:b],signal[a:b]
            lo,hi=np.searchsorted(rt,[float(row['start_ms']),float(row['end_ms'])],side='left');rt,rx=rt[lo:hi],rx[lo:hi]
            if not np.isfinite(rx).all() or not np.all(np.diff(rt)>0):raise ValueError('Invalid raw waveform.')
            t=np.round((rt-float(row['start_ms']))/1000,4).tolist();x=np.round(rx,2).tolist()
            duration=(float(row['end_ms'])-float(row['start_ms']))/1000
        task_id='truth_'+key
        tasks.append(dict(id=task_id,participant=row['participant'],queue=queue,t=t,x=x,focusStartSec=0.,focusEndSec=duration))
        predictions[task_id]={}
        for method in METHODS:
            accepted=decisions[key]['old_accepted_direction'] if method=='old_v1' else accepted_direction(row,intervals[method])
            predictions[task_id][method]=dict(direction=row[method],flip=accepted=='negative',
                                             support='historical_rule' if method=='old_v1' else row[method+'_support'])
    protected=[BASE/'correct_ecg_polarity.py',HERE/'direction_v3.py',HERE/'beatwise.py',
               BASE/'outputs/ecg_polarity_correction_all20_v1/correction_intervals.csv',source,
               SHADOW/'minute_results.csv',SHADOW/'summary.json']
    protected_hashes={str(p):sha(p) for p in protected}
    args.out_dir.mkdir(parents=True,mode=0o700)
    archives=args.out_dir/'archived_previous_records';archives.mkdir(mode=0o700)
    archived=[]
    for name,path in [('old_direction_audit',BASE/'outputs/beatwise_polarity_audit_v2/human_audit.sqlite3'),
                      ('old_paired_preferences',BASE/'outputs/polarity_pairwise_disagreements_v1/paired_preferences.sqlite3'),
                      ('old_three_method_preferences',PREVIOUS/'followup_preferences.sqlite3')]:
        destination=archives/(name+'.sqlite3')
        with closing(sqlite3.connect(f'file:{path}?mode=ro',uri=True)) as src,closing(sqlite3.connect(destination)) as dst:
            count=src.execute('SELECT COUNT(*) FROM ratings').fetchone()[0];src.backup(dst)
        archived.append(dict(source=str(path),backup=str(destination),ratings=count,used_in_new_scoring=False))
    download=Path('/Users/jiqiyu/Downloads/direction_targeted_choices.csv')
    if download.exists():shutil.copyfile(download,archives/'previous_targeted_choices.csv')
    protocol=dict(version='raw_direction_truth_v1',reference_definition='dominant_QRS_amplitude_relative_to_local_baseline',
        priority_minutes=sum(t['queue']=='priority' for t in tasks),optional_minutes=sum(t['queue']=='optional' for t in tasks),
        participants=[dict(participant=f'P{i}',priority_minutes=sum(t['participant']==f'P{i}' and t['queue']=='priority' for t in tasks),
            optional_minutes=sum(t['participant']==f'P{i}' and t['queue']=='optional' for t in tasks)) for i in range(1,21)],
        old_answers_consulted_by_this_builder=False,old_records_used_for_scoring=False,old_records_prefilled=False,
        algorithm_outputs_visible_during_labeling=False,peak_marks_modified=False,
        sampling='All 12 core flip-change minutes plus the same 3 optional checks. The inherited target pool was selected during prior development; old answers do not exclude any changed minute in this new round. This is targeted re-review, not a representative population sample.',
        prior_exposure='Existing snippets were previously available in algorithm-comparison pages. Hiding predictions now does not make this an unseen independent validation set.',
        scoring='Only completed raw_direction_labeling_v1 records in the new packet-bound database. Signed reference labels assess correct-and-decided, opposite, abstained and coverage. Balanced/mixed are reported separately; unreadable/unsure are excluded from signed accuracy.',
        protected_hashes=protected_hashes,builder_sha256=sha(Path(__file__)))
    write_json(args.out_dir/'private_predictions.json',dict(methods=list(METHODS),predictions=predictions,source_sha256=sha(SHADOW/'minute_results.csv')))
    protocol['private_predictions_sha256']=sha(args.out_dir/'private_predictions.json')
    with gzip.open(args.out_dir/'reference_packet.json.gz','wt') as f:json.dump(dict(version=1,tasks=tasks,protocol=protocol),f,separators=(',',':'),allow_nan=False)
    write_json(args.out_dir/'protocol.json',protocol)
    write_json(args.out_dir/'archive_manifest.json',dict(archived=archived,old_records_retired_from_this_benchmark=True,peak_annotations_untouched=True))
    if any(sha(Path(p))!=value for p,value in protected_hashes.items()):raise RuntimeError('A protected source changed during preparation.')
    print(json.dumps(dict(output=str(args.out_dir),priority=protocol['priority_minutes'],optional=protocol['optional_minutes'],archived=archived),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
