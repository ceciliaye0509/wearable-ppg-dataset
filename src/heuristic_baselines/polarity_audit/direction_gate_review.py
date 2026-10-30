"""Separate amplitude-gate diagnostics and interval review; never edits main labels."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import csv
import gzip
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
import json
from pathlib import Path
import sqlite3
import sys
from urllib.parse import urlsplit

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(HERE))
from build_direction_random20 import DEFAULT_OUT as SOURCE, MANIFEST, load_raw, indices
from build_direction_truth import sha, write_json
from serve_audit import initialize_database, read_ratings
from serve_targeted_followup import make_handler as make_base_handler
from remote_ecg_review.serve_review import csv_safe

DEFAULT_OUT = ROOT/'analysis/ecg_direction_gate_review_20261029_v1'
REVIEW_MODE = 'amplitude_gate_interval_review_v1'
READABILITY = {'', 'readable', 'unreadable', 'unsure'}
DIRECTIONS = {'', 'positive', 'negative', 'balanced', 'mixed', 'unreadable', 'unsure'}


def private_method(bypass=False):
    spec = spec_from_file_location('isolated_gate_diagnostic', HERE/'direction_v3.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    if bypass:
        module.PARAMETERS = dict(module.PARAMETERS, severe_raw_p01_p99_span_uv=float('inf'))
    return module.measure_direction_v3


def aggregate_quarters(quarters):
    """Predeclared 45/60-second agreement, with any explicit opposite veto."""
    signs = [q['direction'] for q in quarters if q['direction'] in {'positive','negative'}]
    if len(set(signs)) > 1:
        return 'ambiguous'
    return signs[0] if len(signs) >= 3 else 'ambiguous'


def validate_interval_answer(body):
    tiles, direction = body.get('tile_readability'), body.get('direction','')
    status, version = body.get('status'), body.get('version')
    confidence, notes = body.get('confidence','medium'), body.get('notes','')
    if not isinstance(tiles,list) or len(tiles)!=12 or any(not isinstance(x,str) or x not in READABILITY for x in tiles):
        raise ValueError('Provide one readability label for each of the twelve 5-second intervals.')
    if not isinstance(direction,str) or direction not in DIRECTIONS:
        raise ValueError('Invalid direction for the readable intervals.')
    if status not in {'in_progress','complete'} or confidence not in {'high','medium','low'}:
        raise ValueError('Invalid status or confidence.')
    if not isinstance(notes,str) or len(notes)>2000 or type(version) is not int or version<0:
        raise ValueError('Invalid notes or saved version.')
    if status=='complete':
        if '' in tiles or not direction:
            raise ValueError('Label all twelve intervals and choose a direction before completing.')
        if 'readable' not in tiles and direction not in {'unreadable','unsure'}:
            raise ValueError('A signed direction requires at least one readable interval.')
        if 'readable' in tiles and direction=='unreadable':
            raise ValueError('Choose the direction of the readable intervals, or Unsure.')
    return dict(tile_readability=tiles, direction=direction, status=status, confidence=confidence,
        notes=notes, review_mode=REVIEW_MODE,
        readable_intervals_sec=[[i*5,(i+1)*5] for i,label in enumerate(tiles) if label=='readable']),version


def complete(rating):
    try:
        answer,_=validate_interval_answer(dict(rating,version=rating.get('version',0)))
        return answer['status']=='complete' and rating.get('review_mode')==REVIEW_MODE
    except (TypeError,ValueError):
        return False


def public_task(task,rating=None):
    # Amplitude diagnostics only; candidate directions and prior reference labels
    # remain private. These outcome-selected checks are development, not validation.
    return {**{k:task[k] for k in ['id','source_id','participant','queue','t','x',
        'focusStartSec','focusEndSec','minute_span_uv','tiles']},'rating':rating}


def summary(packet,ratings):
    tasks=packet['tasks']
    return dict(tasks=[dict(id=t['id'],source_id=t['source_id'],participant=t['participant'],
        rating=ratings.get(t['id'])) for t in tasks],
        participants=sorted({t['participant'] for t in tasks},key=lambda p:int(p[1:])),
        total=len(tasks),completed=sum(complete(ratings.get(t['id'],{})) for t in tasks),
        main_labels_unchanged=True,review_mode=REVIEW_MODE)


def export_csv(packet,ratings):
    buffer=StringIO(newline='')
    fields=['id','source_id','participant','direction','confidence','status','tile_readability',
            'readable_intervals_sec','notes','reviewer','updated_at','version','review_mode']
    writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader()
    for task in packet['tasks']:
        r=ratings.get(task['id'])
        if not r or r.get('review_mode')!=REVIEW_MODE:continue
        row=dict(id=task['id'],source_id=task['source_id'],participant=task['participant'],**r)
        for key in ('tile_readability','readable_intervals_sec'):
            row[key]=json.dumps(row.get(key,[]),separators=(',',':'))
        writer.writerow({k:csv_safe(row.get(k,'')) for k in fields})
    return buffer.getvalue().encode('utf-8-sig')


def attach_routes(MainHandler,packet,db_path,access,secret,cookie_name):
    assets={'/gate-review':('gate_review.html','text/html'),'/gate_review.js':('gate_review.js','text/javascript'),
            '/gate_review.css':('gate_review.css','text/css'),'/styles.css':('styles.css','text/css')}
    GateHandler=make_base_handler(packet,db_path,access,secret,public_fn=public_task,
        validate_fn=validate_interval_answer,csv_fn=export_csv,cookie_name=cookie_name,
        assets_override=assets,summary_fn=summary,export_filename='direction_gate_interval_labels.csv')
    class Handler(MainHandler):
        def dispatch_gate(self,method):
            original=self.path
            if original.startswith('/gate-api/'):
                self.path='/api/'+original[len('/gate-api/'):]
            try:getattr(GateHandler,method)(self)
            finally:self.path=original

        def do_GET(self):
            path=urlsplit(self.path).path
            if path in {'/gate-review','/gate_review.js','/gate_review.css'} or path.startswith('/gate-api/'):
                self.dispatch_gate('do_GET')
            else:super().do_GET()

        def do_POST(self):
            if urlsplit(self.path).path.startswith('/gate-api/'):
                self.dispatch_gate('do_POST')
            else:super().do_POST()
    return Handler


def build(args):
    if args.out_dir.exists():
        raise ValueError('Existing diagnostic sets are never overwritten.')
    packet_path=SOURCE/'reference_packet.json.gz'
    with gzip.open(packet_path,'rt') as f:packet=json.load(f)
    prediction_path=SOURCE/'private_predictions.json'
    assert sha(prediction_path)==packet['protocol']['private_predictions_sha256']
    predictions=json.loads(prediction_path.read_text())['predictions']
    labels=list(csv.DictReader(args.labels.open(encoding='utf-8-sig',newline='')))
    main=[t for t in packet['tasks'] if t['queue']=='main']
    assert len(labels)==160 and {r['id'] for r in labels}=={t['id'] for t in main}
    assert all(r['review_mode']=='raw_direction_labeling_v1' and r['status']=='complete' for r in labels)
    references={r['id']:r for r in labels}
    selected=json.loads((SOURCE/'frozen_selection.json').read_text())['selected']
    source={r['id']:r for r in selected}
    protected={str(p):sha(p) for p in [HERE/'direction_v3.py',BASE/'correct_ecg_polarity.py',
        SOURCE/'reference_packet.json.gz',prediction_path,
        BASE/'outputs/ecg_polarity_correction_all20_v1/correction_intervals.csv']}
    args.out_dir.mkdir(parents=True,mode=0o700)
    protocol=dict(version='amplitude_gate_review_v1',source_packet=str(packet_path),source_packet_sha256=sha(packet_path),
        labels_source=str(args.labels),labels_sha256=sha(args.labels),protected_hashes=protected,
        experiment_fixed_before_evaluation=True,
        experiment='All 160 fixed random minutes, no resampling. Original full-minute v3; isolated whole-minute amplitude-gate bypass as a diagnostic only; four separate 15-second quarters each using unchanged v3 and the unchanged 5000 uV gate. Local candidate requires at least three agreeing signed quarters, no confidently opposite quarter. Never concatenate disconnected intervals.',
        amplitude_tiles='Twelve raw 5-second P99-P01 spans for display only. A small span is not proof of QRS readability.',
        human_review='All nine signed human-reference minutes where frozen full-minute v3 returned severe_artifact. Raw ECG only, no candidate directions or prefilled interval answers. Separate interval-label database; original absolute labels and peak marks untouched.',
        limitations='Development diagnostics after seeing the main labels. Not a new independent accuracy estimate or authority to change production correction rules. Three-minute automatic flips not assessed.',
        builder_sha256=sha(Path(__file__)))
    write_json(args.out_dir/'protocol.json',protocol)
    (args.out_dir/'reference_labels_snapshot.csv').write_bytes(args.labels.read_bytes())
    measure,bypass=private_method(),private_method(True)
    results,tasks=[],[]
    for pid in [f'P{i}' for i in range(1,21)]:
        t,x=load_raw(pid)
        for task in [v for v in main if v['participant']==pid]:
            row=source[task['id'].removeprefix('truth_')]
            a,b=indices(t,row);mt,mx=t[a:b],x[a:b]
            fs=float(row['fs_hz']);origin=float(row['end_ms'])-60000.
            original=measure(mx,mt,fs)
            assert original['raw_polarity']==predictions[task['id']]['v3_minute']['direction']
            ungated=bypass(mx,mt,fs)
            quarters=[]
            for i in range(4):
                lo,hi=np.searchsorted(mt,[origin+i*15000,origin+(i+1)*15000])
                part=measure(mx[lo:hi],mt[lo:hi],fs)
                quarters.append(dict(start_sec=i*15,end_sec=(i+1)*15,
                    direction=part['raw_polarity'],reason=part['reason'],support=part['cross_check_status'],
                    span_uv=part['raw_p01_p99_span_uv'] if np.isfinite(part['raw_p01_p99_span_uv']) else None))
            tiles=[]
            for i in range(12):
                lo,hi=np.searchsorted(mt,[origin+i*5000,origin+(i+1)*5000])
                span=float(np.quantile(mx[lo:hi],.99)-np.quantile(mx[lo:hi],.01))
                tiles.append(dict(index=i,start_sec=i*5,end_sec=(i+1)*5,span_uv=span,over_threshold=span>5000.))
            ref=references[task['id']]['direction']
            results.append(dict(id=task['id'],participant=pid,reference=ref,
                original=original['raw_polarity'],gate_bypass_diagnostic=ungated['raw_polarity'],
                local_quarters_candidate=aggregate_quarters(quarters),quarters=quarters,amplitude_tiles=tiles,
                minute_span_uv=float(original['raw_p01_p99_span_uv'])))
            if ref in {'positive','negative'} and original['raw_polarity']=='severe_artifact':
                tasks.append(dict(task,id='gate_'+task['id'].removeprefix('truth_'),source_id=task['id'],queue='gate_review',
                                  minute_span_uv=float(original['raw_p01_p99_span_uv']),tiles=tiles))
        print(f'{pid}: original gate, isolated bypass and fixed local-window diagnostic completed',flush=True)
        del t,x
    metrics={}
    signed=[r for r in results if r['reference'] in {'positive','negative'}]
    unreadable=[r for r in results if r['reference']=='unreadable']
    for method in ('original','gate_bypass_diagnostic','local_quarters_candidate'):
        decided=[r for r in signed if r[method] in {'positive','negative'}]
        metrics[method]=dict(correct=sum(r[method]==r['reference'] for r in signed),
            opposite=sum(r[method]!=r['reference'] for r in decided),abstained=len(signed)-len(decided),
            signed_references=len(signed),unreadable_references=len(unreadable),
            signed_outputs_on_unreadable=sum(r[method] in {'positive','negative'} for r in unreadable),
            recovered_gate_review_minutes=sum(r['original']=='severe_artifact' and r[method]==r['reference'] for r in signed))
    assert len(tasks)==9
    with gzip.open(args.out_dir/'review_packet.json.gz','wt') as f:
        json.dump(dict(version=1,tasks=tasks,protocol=protocol),f,separators=(',',':'),allow_nan=False)
    write_json(args.out_dir/'private_diagnostics.json',dict(results=results,metrics=metrics))
    summary_result=dict(metrics=metrics,review_minutes=9,main_minutes_unchanged=160,production_rules_changed=False,
                        nine_gate_cases=[{k:r[k] for k in ['id','participant','reference','original','gate_bypass_diagnostic','local_quarters_candidate','minute_span_uv']} for r in results if r['id'] in {v['source_id'] for v in tasks}])
    write_json(args.out_dir/'diagnostic_summary.json',summary_result)
    if any(sha(Path(p))!=digest for p,digest in protected.items()):
        raise RuntimeError('A protected source changed during the diagnostic')
    print(json.dumps(summary_result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,default=Path('/Users/jiqiyu/Downloads/direction_reference_labels.csv'))
    parser.add_argument('--out-dir',type=Path,default=DEFAULT_OUT)
    build(parser.parse_args())


if __name__=='__main__':main()
