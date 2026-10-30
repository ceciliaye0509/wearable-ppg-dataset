"""Reuse recorded evidence and prepare only unresolved three-method decisions.

No new human answers, production corrections, old packets or databases are
written. A prior paired preference can remove a duplicate review burden but is
never converted into an absolute direction label or a three-method preference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[2]
sys.path.insert(0,str(BASE))
from correct_ecg_polarity import _split_runs
from polarity_audit.build_audit import read_rows, write_rows, json_clean

SHADOW = ROOT/'analysis/ecg_rodrigues_polarity_shadow_20261029_v1/results'
DEFAULT_OUT = ROOT/'analysis/ecg_direction_targeted_followup_20261029_v1'
METHODS = ('old_v1','v3_minute','rod300')
METHOD_NAMES = {'old_v1':'Old v1 rule','v3_minute':'Template v3','rod300':'Rodrigues-assisted rule'}
SIGNED = {'positive','negative'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_saved(path):
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as con:
        return {key:dict(json.loads(answer),version=version,reviewer=reviewer,updated_at=updated)
                for key,answer,version,reviewer,updated in con.execute('SELECT id,answer_json,version,reviewer,updated_at FROM ratings')}


def direction_evidence(answer):
    return (answer.get('status')=='complete' and answer.get('direction_determinability')=='yes'
            and answer.get('direction') in SIGNED)


def reusable_preference(answer):
    return answer.get('status')=='complete' and answer.get('choice') in {'A','B','same'}


def select_followup(rows, core_ids, flip_changes, directions, preferences, seed=20261029):
    by_id = {r['id']:r for r in rows}
    keys = set(flip_changes)
    keys.update(r['id'] for r in rows if r['id'] in core_ids and len({r[m] for m in METHODS}&SIGNED)>1)
    priority, skipped = [], []
    for key in sorted(keys):
        row = by_id[key]
        if direction_evidence(directions.get(key,{})):
            skipped.append(dict(id=key,participant=row['participant'],evidence='explicit_direction',
                                note='Reuse saved absolute direction; no repeat review.'))
        elif reusable_preference(preferences.get('pair_'+key,{})):
            skipped.append(dict(id=key,participant=row['participant'],evidence='paired_preference_only',
                                note='Prior preference retained; not an absolute direction or three-method answer. Proposed correction remains unconfirmed.'))
        elif row.get('technical_pass') != 'True':
            skipped.append(dict(id=key,participant=row['participant'],evidence='technical_exclusion',note='Not eligible for new direction review.'))
        else:
            priority.append(dict(row,queue='priority',selection_reason='changed_flip' if key in flip_changes else 'opposite_signed_predictions'))
    # Small, explicitly optional check set. Never inflate the priority workload.
    pools = defaultdict(list)
    for row in rows:
        key = row['id']
        if (key in core_ids and key not in keys and row.get('technical_pass')=='True'
                and not direction_evidence(directions.get(key,{}))
                and not reusable_preference(preferences.get('pair_'+key,{}))
                and row['old_v1'] in SIGNED and len({row[m] for m in METHODS})==1):
            pools[row['participant']].append(row)
    pids = sorted(pools,key=lambda p:int(p[1:]))
    rng = random.Random(seed)
    optional = []
    for pid in rng.sample(pids,min(3,len(pids))):
        row = rng.choice(sorted(pools[pid],key=lambda r:r['id']))
        optional.append(dict(row,queue='optional',selection_reason='agreement_spot_check'))
    return priority,optional,skipped


def accepted_direction(row, intervals):
    midpoint = (float(row['start_ms'])+float(row['end_ms']))/2
    matches = [r for r in intervals if r['participant']==row['participant']
               and int(r['run_index'])==int(row['run_index'])
               and float(r['start_ms'])<=midpoint<float(r['end_ms'])]
    if len(matches)>1:raise ValueError('Overlapping candidate correction intervals')
    return matches[0]['orientation'] if matches else 'unaccepted'


def summarise_evidence(rows, directions):
    labeled = [r for r in rows if direction_evidence(directions.get(r['id'],{}))]
    metrics = []
    for method in METHODS:
        signed = [r for r in labeled if r[method] in SIGNED]
        same = sum(r[method]==directions[r['id']]['direction'] for r in signed)
        metrics.append(dict(method=method,label=METHOD_NAMES[method],reference_minutes=len(labeled),
                            same_direction=same,opposite_direction=len(signed)-same,
                            abstained=len(labeled)-len(signed)))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--seed',type=int,default=20261029)
    args = parser.parse_args()
    if args.out_dir.exists():raise SystemExit('Existing follow-up packets are never replaced; use a new output directory.')
    rows = read_rows(SHADOW/'minute_results.csv')
    decisions = read_rows(SHADOW/'v1_vs_shadow_interval_decisions_by_minute.csv')
    decision_by_id = {r['id']:r for r in decisions}
    core_ids = {r['id'] for r in decisions if r['is_core']=='True'}
    changes = {r['id']:r for r in read_rows(SHADOW/'core_flip_changes_for_review.csv')}
    old_db = BASE/'outputs/beatwise_polarity_audit_v2/human_audit.sqlite3'
    paired_db = BASE/'outputs/polarity_pairwise_disagreements_v1/paired_preferences.sqlite3'
    directions,preferences = read_saved(old_db),read_saved(paired_db)
    priority,optional,skipped = select_followup(rows,core_ids,changes,directions,preferences,args.seed)
    intervals = {m:read_rows(SHADOW/f'candidate_intervals_{m}.csv') for m in ['v3_minute','rod300']}
    protected = [SHADOW/'summary.json',SHADOW/'minute_results.csv',SHADOW/'human_review_snapshot.json',
                 BASE/'correct_ecg_polarity.py',HERE/'direction_v3.py',HERE/'beatwise.py',
                 BASE/'outputs/ecg_polarity_correction_all20_v1/correction_intervals.csv',
                 BASE/'outputs/polarity_pairwise_disagreements_v1/comparison_packet.json.gz',
                 BASE/'outputs/beatwise_polarity_audit_v2/audit_packet.json.gz']
    before = {str(p):digest(p) for p in protected}
    selected = priority+optional
    by_pid = defaultdict(list)
    for row in selected:by_pid[row['participant']].append(row)
    tasks, manifest = [], []
    for pid in sorted(by_pid,key=lambda p:int(p[1:])):
        with np.load(ROOT/'Multisite-PPG/raw_data'/pid/f'{pid}_polar_ecg_raw.npz',allow_pickle=False) as data:
            times,signal = np.asarray(data['time_ms'],float),np.asarray(data['ecg_uv'],float)
        finite = np.flatnonzero(np.isfinite(times))
        times,signal = times[finite[0]:finite[-1]+1],signal[finite[0]:finite[-1]+1]
        runs = _split_runs(times)
        for row in sorted(by_pid[pid],key=lambda r:r['id']):
            a,b = runs[int(row['run_index'])];rt,rx = times[a:b],signal[a:b]
            lo,hi = np.searchsorted(rt,[float(row['start_ms']),float(row['end_ms'])],side='left')
            t,x = rt[lo:hi],rx[lo:hi]
            if not (len(x)>=100 and np.isfinite(x).all() and np.all(np.diff(t)>0)):
                raise ValueError('Follow-up source is not strictly valid: '+row['id'])
            sides = {}
            ordering = list(METHODS);random.Random(f'{args.seed}:layout:{row["id"]}').shuffle(ordering)
            for side,method in zip('ABC',ordering):
                accepted = decision_by_id[row['id']]['old_accepted_direction'] if method=='old_v1' else accepted_direction(row,intervals[method])
                sides[side] = dict(method_id=method,direction=row[method],
                    support='historical_rule' if method=='old_v1' else row[method+'_support'],
                    decision={'positive':'stable_positive','negative':'stable_negative'}.get(accepted,'review_required'),
                    flip=accepted=='negative')
            tasks.append(dict(id='followup_'+row['id'],participant=pid,queue=row['queue'],
                t=np.round((t-float(row['start_ms']))/1000,4).tolist(),x=np.round(x,2).tolist(),
                focusStartSec=0.,focusEndSec=(float(row['end_ms'])-float(row['start_ms']))/1000,sides=sides))
            manifest.append(dict(id='followup_'+row['id'],source_id=row['id'],participant=pid,queue=row['queue'],
                selection_reason=row['selection_reason'],start_ms=row['start_ms'],end_ms=row['end_ms'],
                method_A=ordering[0],method_B=ordering[1],method_C=ordering[2]))
        print(f'{pid}: {len(by_pid[pid])} original unflipped minutes verified',flush=True)
        del times,signal,finite
    tasks.sort(key=lambda t:(t['queue']!='priority',int(t['participant'][1:]),t['id']))
    evidence_metrics = summarise_evidence(rows,directions)
    all_explicit = {key:a for key,a in directions.items() if direction_evidence(a)}
    provenance = dict(explicit_direction_records=len(all_explicit),explicit_records_scored=evidence_metrics[0]['reference_minutes'],
                      paired_preference_records=len(preferences),paired_preferences_used_as_direction_truth=False,
                      metrics=evidence_metrics,skipped_prior_records=len(skipped),
                      limitation='Existing algorithm-assisted direction ratings; selected development evidence, not independent accuracy or population prevalence.')
    protocol = dict(version='targeted_three_method_followup_v1',seed=args.seed,
        priority_minutes=len(priority),optional_minutes=len(optional),methods=list(METHODS),
        selection='Core flip-state changes or opposing signed predictions; reuse explicit direction or completed decisive prior paired preference. Three agreement checks are optional.',
        existing_records_reused=True,prior_preference_is_absolute_truth=False,
        new_human_answers_created=False,production_correction_modified=False,
        source_hashes=before,builder_sha256=digest(Path(__file__)),
        evidence=provenance,participants=[dict(participant=f'P{i}',
            priority_minutes=sum(t['participant']==f'P{i}' and t['queue']=='priority' for t in tasks),
            optional_minutes=sum(t['participant']==f'P{i}' and t['queue']=='optional' for t in tasks),
            reused_direction_records=sum(key.startswith(f'P{i}_') for key in all_explicit)) for i in range(1,21)])
    args.out_dir.mkdir(parents=True,mode=0o700)
    with gzip.open(args.out_dir/'followup_packet.json.gz','wt') as f:
        json.dump(json_clean(dict(version=1,tasks=tasks,protocol=protocol)),f,allow_nan=False,separators=(',',':'))
    for name,value in [('protocol',protocol),('reused_evidence',dict(provenance=provenance,direction_records=directions,paired_preferences=preferences))]:
        (args.out_dir/(name+'.json')).write_text(json.dumps(json_clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_rows(args.out_dir/'reused_direction_comparison.csv',evidence_metrics)
    write_rows(args.out_dir/'skipped_prior_records.csv',skipped)
    write_rows(args.out_dir/'sampling_manifest_private.csv',manifest)
    labeled = []
    for row in rows:
        if direction_evidence(directions.get(row['id'],{})):
            labeled.append(dict(id=row['id'],participant=row['participant'],human_direction=directions[row['id']]['direction'],
                reviewer=directions[row['id']].get('reviewer',''),assistance_shown=directions[row['id']].get('assistance_shown','unknown'),
                **{m:row[m] for m in METHODS}))
    write_rows(args.out_dir/'reused_direction_by_minute.csv',labeled)
    after = {str(p):digest(p) for p in protected}
    if before!=after:raise RuntimeError('A protected source changed during building; inspect before serving.')
    # A byte-for-byte logical snapshot comparison does not touch old rows.
    if directions!=read_saved(old_db) or preferences!=read_saved(paired_db):
        raise RuntimeError('Existing reviews changed during preparation; rebuild from the current snapshot.')
    verification = dict(priority_minutes=len(priority),optional_minutes=len(optional),
        new_tasks_with_existing_explicit_answer=0,new_tasks_with_existing_decisive_preference=0,
        protected_sources_unchanged=True,prior_rating_rows_unchanged=True,new_ratings_written=0,
        task_ids_unique=len({t['id'] for t in tasks})==len(tasks),new_packet_sha256=digest(args.out_dir/'followup_packet.json.gz'))
    (args.out_dir/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.out_dir),verification=verification,reused=provenance),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
