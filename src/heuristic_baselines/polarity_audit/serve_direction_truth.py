"""Waveform-only direction reference labels; score ONLY this fresh label set."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
from http.server import ThreadingHTTPServer
from io import StringIO
import json
import os
from pathlib import Path
import secrets
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from serve_targeted_followup import make_handler as make_base_handler
from serve_audit import initialize_database,read_ratings
from remote_ecg_review.serve_review import csv_safe
from build_direction_truth import METHODS
from build_direction_random20 import DEFAULT_OUT

COOKIE='ecg_direction_truth_session'
REVIEW_MODE='raw_direction_labeling_v1'
LABELS={'','positive','negative','balanced','mixed','unreadable','unsure'}
SIGNED={'positive','negative'}
METHOD_NAMES={'old_v1':'Old v1 rule','v3_minute':'Template v3','rod300':'Rodrigues-assisted rule'}
COHORT_NAMES={'main':'Random main evaluation','difficult':'Difficult audit','optional':'Additional optional checks'}
REPAIR_VERSION='raw_direction_truth_v3'


def cohort_definitions(packet):
    groups=[(key,label,[t for t in packet['tasks'] if t['queue']==key]) for key,label in COHORT_NAMES.items()]
    if packet['protocol'].get('version')==REPAIR_VERSION:
        groups[0]=('main','Original main evaluation (historical)',groups[0][2])
        groups=[('replacement','P17 replacement review',[t for t in packet['tasks'] if t['queue']=='replacement']),
                ('corrected','Corrected main evaluation',[t for t in packet['tasks'] if t.get('in_corrected_main')]),*groups]
    return groups


def public_task(task,rating=None):
    # A whitelist is essential: do not send method outputs, flip actions,
    # prior preferences or earlier human labels to the labeling page.
    return {**{k:task[k] for k in ['id','participant','queue','t','x','focusStartSec','focusEndSec']},
            **{k:task[k] for k in ['in_corrected_main','superseded_in_corrected_main','replaces_id'] if k in task},'rating':rating}


def validate_answer(body):
    direction=body.get('direction','');status=body.get('status');confidence=body.get('confidence','medium')
    notes=body.get('notes','');version=body.get('version')
    if not isinstance(direction,str) or direction not in LABELS:raise ValueError('Choose a direction or an explicit uncertain category.')
    if not isinstance(status,str) or status not in {'in_progress','complete'}:raise ValueError('Invalid status.')
    if not isinstance(confidence,str) or confidence not in {'high','medium','low'}:raise ValueError('Invalid confidence.')
    if status=='complete' and not direction:raise ValueError('Choose a label before completing this minute.')
    if not isinstance(notes,str) or len(notes)>2000:raise ValueError('Notes must be at most 2,000 characters.')
    if type(version) is not int or version<0:raise ValueError('Invalid saved version.')
    return dict(direction=direction,status=status,confidence=confidence,notes=notes,review_mode=REVIEW_MODE),version


def is_complete(rating):
    return (rating.get('status')=='complete' and rating.get('review_mode')==REVIEW_MODE
            and rating.get('direction') in LABELS-{''})


def summary(packet,ratings):
    if packet['protocol'].get('version') in {'raw_direction_truth_v2',REPAIR_VERSION}:
        cohorts=[]
        for key,label,tasks in cohort_definitions(packet):
            done=sum(is_complete(ratings.get(t['id'],{})) for t in tasks)
            cohorts.append(dict(id=key,label=label,minutes=len(tasks),completed=done,results_ready=bool(tasks) and done==len(tasks),task_ids=[t['id'] for t in tasks]))
        return dict(tasks=[dict(id=t['id'],participant=t['participant'],queue=t['queue'],rating=ratings.get(t['id']),
            **{k:t[k] for k in ['in_corrected_main','superseded_in_corrected_main','replaces_id'] if k in t}) for t in packet['tasks']],
            participants=packet['protocol']['participants'],cohorts=cohorts,
            main_minutes=packet['protocol']['main_minutes'],difficult_minutes=packet['protocol']['difficult_minutes'],
            optional_minutes=packet['protocol']['optional_minutes'],results_ready=any(c['results_ready'] for c in cohorts),
            corrected_main_minutes=packet['protocol'].get('corrected_main_minutes',0),
            replacement_minutes=packet['protocol'].get('replacement_minutes',0),
            old_records_used=False,review_mode=REVIEW_MODE)
    priority=[t for t in packet['tasks'] if t['queue']=='priority']
    return dict(tasks=[dict(id=t['id'],participant=t['participant'],queue=t['queue'],rating=ratings.get(t['id'])) for t in packet['tasks']],
        participants=packet['protocol']['participants'],priority_minutes=len(priority),
        optional_minutes=sum(t['queue']=='optional' for t in packet['tasks']),
        results_ready=all(is_complete(ratings.get(t['id'],{})) for t in priority),
        old_records_used=False,review_mode=REVIEW_MODE)


def compare_labels(packet,predictions,ratings):
    if packet['protocol'].get('version') in {'raw_direction_truth_v2',REPAIR_VERSION}:
        cohorts=[]
        for key,label,tasks in cohort_definitions(packet):
            result=compare_labels(dict(tasks=tasks,protocol={}),predictions,ratings)
            result.update(id=key,label=label,total_minutes=len(tasks),
                          results_ready=bool(tasks) and result['completed_new_labels']==len(tasks))
            result['scope']=('Random complete minutes, eight per participant. Results describe technically eligible minutes with equal participant allocation; not an unseen external validation.'
                if key=='main' else 'Previously selected checks. Report separately from the random main evaluation; not a population estimate.')
            if packet['protocol'].get('version')==REPAIR_VERSION:
                result['scope']={
                    'main':'Historical original allocation includes four provenance-affected P17 minutes. Retained for audit; use the corrected main set for the repaired comparison.',
                    'corrected':packet['protocol']['corrected_scope'],
                    'replacement':'Four newly labeled P17 replacements, scored separately. Sampling excluded the entire shared P19 source and all selected P19 intervals/waveforms.',
                    'difficult':'Historical targeted difficult checks include two provenance-flagged P17 records. Keep separate; this is not an independent population estimate.',
                    'optional':'Three historical optional checks, reported separately.'}[key]
            cohorts.append(result)
        return dict(cohorts=cohorts,old_records_used=False,pooled_score_prohibited=True,
                    scoring_note='Random main, difficult and optional cohorts are never pooled. Algorithm failures and missing detector context count as abstentions on signed references.')
    references=[(t,ratings[t['id']]) for t in packet['tasks'] if is_complete(ratings.get(t['id'],{}))]
    signed=[(t,r) for t,r in references if r['direction'] in SIGNED]
    non_single=[(t,r) for t,r in references if r['direction'] in {'balanced','mixed'}]
    rows=[]
    for method in METHODS:
        decided=[(t,r) for t,r in signed if predictions[t['id']][method]['direction'] in SIGNED]
        correct=sum(predictions[t['id']][method]['direction']==r['direction'] for t,r in decided)
        opposite=len(decided)-correct;n=len(signed)
        rows.append(dict(method=method,label=METHOD_NAMES[method],signed_reference_minutes=n,
            correct_and_decided=correct,opposite_direction=opposite,abstained=n-len(decided),
            correct_and_decided_fraction=correct/n if n else None,
            direction_accuracy_when_decided=correct/len(decided) if decided else None,
            coverage=len(decided)/n if n else None,
            readable_non_single_direction_minutes=len(non_single),
            signed_output_on_balanced_or_mixed=sum(predictions[t['id']][method]['direction'] in SIGNED for t,r in non_single),
            flip_policy_evaluable_minutes=sum(predictions[t['id']][method].get('flip') is not None for t,r in signed),
            proposed_reversals_on_positive=sum(r['direction']=='positive' and bool(predictions[t['id']][method].get('flip')) for t,r in signed),
            negative_references_proposed_reversed=sum(r['direction']=='negative' and bool(predictions[t['id']][method].get('flip')) for t,r in signed)))
    return dict(completed_new_labels=len(references),signed_reference_minutes=len(signed),
        label_counts=dict(Counter(r['direction'] for _,r in references)),methods=rows,
        old_records_used=False,scope='Current targeted re-review only; snippets were previously available for comparison. Not a population estimate or unseen independent validation.',
        scoring_note='All completed positive/negative labels are included. Balanced/mixed are reported separately; unreadable/unsure are excluded from signed direction accuracy. Minute labels do not by themselves validate the three-minute automatic-flip policy.')


def export_csv(packet,ratings):
    buffer=StringIO(newline='');fields=['id','participant','queue','direction','confidence','status','reviewer','updated_at','version','notes','review_mode',
        'in_corrected_main','superseded_in_corrected_main','replaces_id']
    writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader()
    for task in packet['tasks']:
        rating=ratings.get(task['id'])
        if not rating or rating.get('review_mode')!=REVIEW_MODE:continue
        row=dict(id=task['id'],participant=task['participant'],queue=task['queue'],**rating)
        row.update(in_corrected_main=task.get('in_corrected_main',task['queue']=='main'),
                   superseded_in_corrected_main=task.get('superseded_in_corrected_main',False),replaces_id=task.get('replaces_id',''))
        writer.writerow({k:csv_safe(row.get(k,'')) for k in fields})
    return buffer.getvalue().encode('utf-8-sig')


def make_handler(packet,predictions,db_path,access,secret,*,gate_packet=None,gate_db_path=None,acceptance_packet=None,acceptance_predictions=None,acceptance_db_path=None):
    def results(pkt,ratings):
        if not summary(pkt,ratings)['results_ready']:
            raise ValueError('Complete a review set before viewing its algorithm comparison. Predictions stay hidden while labeling.')
        result=compare_labels(pkt,predictions,ratings)
        if 'cohorts' in result:
            # Never reveal unfinished-set predictions through a completed set.
            result['cohorts']=[c for c in result['cohorts'] if c['results_ready']]
        return result
    assets={'/':('direction_truth.html','text/html'),'/direction_truth.js':('direction_truth.js','text/javascript'),
            '/direction_truth.css':('direction_truth.css','text/css'),'/styles.css':('styles.css','text/css')}
    handler=make_base_handler(packet,db_path,access,secret,public_fn=public_task,validate_fn=validate_answer,
        csv_fn=export_csv,cookie_name=COOKIE,assets_override=assets,summary_fn=summary,results_fn=results)
    if gate_packet is not None:
        from direction_gate_review import attach_routes
        handler=attach_routes(handler,gate_packet,gate_db_path,access,secret,COOKIE)
    if acceptance_packet is not None:
        from flip_acceptance import attach
        handler=attach(handler,acceptance_packet,acceptance_predictions,acceptance_db_path,access,secret,COOKIE)
    return handler


def load_sources(data_file,prediction_file):
    with gzip.open(data_file,'rt') as f:packet=json.load(f)
    if packet.get('version')!=1 or packet.get('protocol',{}).get('version') not in {'raw_direction_truth_v1','raw_direction_truth_v2',REPAIR_VERSION}:raise ValueError('Expected the new waveform-only reference packet.')
    if not packet.get('tasks') or len({t['id'] for t in packet['tasks']})!=len(packet['tasks']):raise ValueError('Missing or duplicate reference tasks.')
    if hashlib.sha256(prediction_file.read_bytes()).hexdigest()!=packet['protocol']['private_predictions_sha256']:raise ValueError('Frozen prediction checksum changed.')
    predictions=json.loads(prediction_file.read_text())['predictions']
    if set(predictions)!={t['id'] for t in packet['tasks']}:raise ValueError('Prediction and reference task identities do not match.')
    for value in predictions.values():
        if set(value)!=set(METHODS):raise ValueError('Missing a frozen method.')
    return packet,predictions


def main():
    from build_flip_acceptance import DEFAULT_OUT as ACCEPT_OUT
    from direction_gate_review import DEFAULT_OUT as GATE_OUT
    from build_p17_replacement import DEFAULT_OUT as REPLACEMENT_OUT
    current_out=REPLACEMENT_OUT if (REPLACEMENT_OUT/'reference_packet.json.gz').exists() else DEFAULT_OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-file',type=Path,default=current_out/'reference_packet.json.gz')
    parser.add_argument('--predictions-file',type=Path,default=current_out/'private_predictions.json')
    parser.add_argument('--database',type=Path,default=current_out/'direction_reference.sqlite3')
    parser.add_argument('--gate-data-file',type=Path,default=GATE_OUT/'review_packet.json.gz')
    parser.add_argument('--gate-database',type=Path,default=GATE_OUT/'interval_reviews.sqlite3')
    parser.add_argument('--acceptance-data-file',type=Path,default=ACCEPT_OUT/'review_packet.json.gz')
    parser.add_argument('--acceptance-database',type=Path,default=ACCEPT_OUT/'acceptance_labels.sqlite3')
    parser.add_argument('--port',type=int,default=8767)
    parser.add_argument('--local-session',action='store_true')
    parser.add_argument('--score-only',action='store_true')
    args=parser.parse_args();packet,predictions=load_sources(args.data_file,args.predictions_file)
    if args.score_only:
        ratings=read_ratings(args.database) if args.database.exists() else {}
        print(json.dumps(compare_labels(packet,predictions,ratings),indent=2));return
    access=os.environ.get('ECG_DIRECTION_TRUTH_ACCESS_CODE','')
    if args.local_session:access,secret=access or secrets.token_urlsafe(18),secrets.token_bytes(32)
    else:secret=os.environ.get('ECG_DIRECTION_TRUTH_SESSION_SECRET','').encode()
    if len(access)<12 or len(secret)<32:parser.error('Use --local-session or set private ECG_DIRECTION_TRUTH_ACCESS_CODE and ECG_DIRECTION_TRUTH_SESSION_SECRET.')
    gate_packet=None
    if args.gate_data_file.exists():
        with gzip.open(args.gate_data_file,'rt') as f:gate_packet=json.load(f)
        source_digests={hashlib.sha256(args.data_file.read_bytes()).hexdigest(),packet['protocol'].get('source_packet_sha256')}
        source_tasks={t['id']:t for t in packet['tasks']}
        if (gate_packet.get('protocol',{}).get('version')!='amplitude_gate_review_v1'
                or gate_packet['protocol']['source_packet_sha256'] not in source_digests
                or any(t['source_id'] not in source_tasks or any(t[k]!=source_tasks[t['source_id']][k] for k in ['participant','t','x']) for t in gate_packet['tasks'])):
            parser.error('The interval review belongs to a different main packet.')
        initialize_database(args.gate_database,hashlib.sha256(args.gate_data_file.read_bytes()).hexdigest())
    acceptance=None;acceptance_predictions=None
    if args.acceptance_data_file.exists():
        integrity=json.loads((args.acceptance_data_file.parent/'packet_integrity.json').read_text())
        if any(hashlib.sha256((args.acceptance_data_file.parent/name).read_bytes()).hexdigest()!=digest for name,digest in integrity.items()):parser.error('Frozen acceptance files changed.')
        with gzip.open(args.acceptance_data_file,'rt') as f:acceptance=json.load(f)
        acceptance_predictions=json.loads((args.acceptance_data_file.parent/'private_predictions.json').read_text())
        if acceptance['protocol']['version']!='unseen_three_minute_flip_acceptance_v1' or set(acceptance_predictions)!={t['id'] for t in acceptance['tasks']}:parser.error('Invalid acceptance packet or prediction identities.')
        initialize_database(args.acceptance_database,hashlib.sha256(args.acceptance_data_file.read_bytes()).hexdigest())
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(packet,predictions,args.database,access,secret,
        gate_packet=gate_packet,gate_db_path=args.gate_database,acceptance_packet=acceptance,
        acceptance_predictions=acceptance_predictions,acceptance_db_path=args.acceptance_database))
    initialize_database(args.database,hashlib.sha256(args.data_file.read_bytes()).hexdigest());server.daemon_threads=True
    print(f'Direction labeling: http://127.0.0.1:{args.port}/',flush=True)
    if args.local_session:print(f'Local access code: {access}',flush=True)
    print(json.dumps({k:packet['protocol'].get(k) for k in ['main_minutes','difficult_minutes','priority_minutes','optional_minutes']}),flush=True)
    if gate_packet is not None:print(f'Interval follow-up: http://127.0.0.1:{args.port}/gate-review · {len(gate_packet["tasks"])} minutes',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
