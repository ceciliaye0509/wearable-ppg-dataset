"""Freeze a participant-balanced random direction benchmark, separate from hard cases."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
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
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT/'analysis/ecg_rodrigues_polarity_shadow_20261029_v1'))
sys.path.insert(0, str(ROOT/'analysis/ecg_detector_freeze_20261023_v1'))
from polarity_audit.build_direction_truth import DEFAULT_OUT as PREVIOUS, METHODS, sha, write_json
from polarity_audit.direction_v3 import measure_direction_v3
from assisted_direction import context_candidates, minute_anchors, measure_at_anchors
from frozen_rules import technical_reasons

DEFAULT_OUT = ROOT/'analysis/ecg_direction_truth_random20_20261029_v2'
MANIFEST = BASE/'outputs/ecg_polarity_correction_all20_v1/polarity_blocks.csv'
RAW = ROOT/'Multisite-PPG/raw_data'
SEED = 20261029
PER_PARTICIPANT = 8


def key(row):
    return f"{row['participant']}_R{int(row['run_index']):03d}_B{int(row['block_index']):05d}"


def select_random(eligible, participant, seed=SEED):
    """SRS without replacement; inputs never include predictions or human answers."""
    ordered = sorted(eligible, key=lambda r:(int(r['run_index']), int(r['block_index'])))
    if len(ordered) < PER_PARTICIPANT:
        raise ValueError(f'{participant}: fewer than eight eligible complete minutes')
    return random.Random(f'{seed}:{participant}').sample(ordered, PER_PARTICIPANT)


def load_raw(pid):
    path = RAW/pid/f'{pid}_polar_ecg_raw.npz'
    with np.load(path, allow_pickle=False) as data:
        t, x = np.asarray(data['time_ms'], float), np.asarray(data['ecg_uv'], float)
    finite = np.flatnonzero(np.isfinite(t))
    if not len(finite):
        raise ValueError(pid+': missing source times')
    t, x = t[finite[0]:finite[-1]+1], x[finite[0]:finite[-1]+1]
    if not np.isfinite(t).all() or np.any(np.diff(t) < 0):
        raise ValueError(pid+': source timebase cannot be searched without repair')
    return t, x


def indices(t, row):
    return np.searchsorted(t, [float(row['start_ms']), float(row['end_ms'])], side='left')


def eligibility_reasons(t, x, row):
    # The legacy manifest stores the first actual sample as start_ms. It is
    # normally a few milliseconds AFTER the nominal minute boundary.
    # Do not mistake this ordinary sample-grid offset for a partial minute.
    start, end = float(row['start_ms']), float(row['end_ms'])
    if end-start < 59500.:
        return ['partial_minute_under_59.5s']
    a, b = indices(t, row)
    reasons = technical_reasons(t[a:b], x[a:b], end-60000., end)
    if b-a != int(row['sample_count']):
        reasons.append('legacy_sample_count_differs_without_repair')
    return reasons


def migrate_labels(old_packet_file, new_packet_file, source_db, destination_db):
    """Copy only this round's absolute labels verbatim, including their history."""
    sys.path.insert(0, str(HERE))
    from ecg_review_packets.legacy_labels import initialize_database
    if destination_db.exists():
        raise ValueError('Refusing to overwrite an existing label database')
    with gzip.open(old_packet_file, 'rt') as f:
        old = {t['id']:t for t in json.load(f)['tasks']}
    with gzip.open(new_packet_file, 'rt') as f:
        new = {t['id']:t for t in json.load(f)['tasks']}
    for task_id, task in old.items():
        if task_id not in new or any(task[k] != new[task_id][k] for k in ('participant','t','x','focusStartSec','focusEndSec')):
            raise ValueError('Saved task identity or waveform changed: '+task_id)
    # The serving process must be stopped at cutover; backup also captures WAL.
    backup = destination_db.parent/'previous_absolute_labels_backup.sqlite3'
    with closing(sqlite3.connect(f'file:{source_db}?mode=ro', uri=True)) as src, closing(sqlite3.connect(backup)) as dst:
        expected = src.execute("SELECT value FROM metadata WHERE key='packet_sha256'").fetchone()
        if expected != (sha(old_packet_file),):
            raise ValueError('Source labels belong to a different packet')
        src.backup(dst)
    initialize_database(destination_db, sha(new_packet_file))
    with closing(sqlite3.connect(backup)) as src, closing(sqlite3.connect(destination_db)) as dst, dst:
        rows = src.execute('SELECT * FROM ratings').fetchall()
        if any(row[0] not in old or json.loads(row[1]).get('review_mode') != 'raw_direction_labeling_v1' for row in rows):
            raise ValueError('Only the current absolute-direction label round may migrate')
        revisions = src.execute('SELECT * FROM revisions').fetchall()
        if any(row[0] not in old for row in revisions):
            raise ValueError('Unexpected revision identity')
        dst.executemany('INSERT INTO ratings VALUES (?,?,?,?,?)', rows)
        dst.executemany('INSERT INTO revisions VALUES (?,?,?)', revisions)
        assert dst.execute('SELECT * FROM ratings').fetchall() == rows
        assert dst.execute('SELECT * FROM revisions').fetchall() == revisions
    result = dict(migrated_labels=len(rows), migrated_revisions=len(revisions),
                  old_preference_records_imported=False, backup=str(backup))
    write_json(destination_db.parent/'label_migration.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--migrate-labels', action='store_true')
    args = parser.parse_args()
    if args.migrate_labels:
        print(json.dumps(migrate_labels(PREVIOUS/'reference_packet.json.gz', args.out_dir/'reference_packet.json.gz',
            PREVIOUS/'direction_reference.sqlite3', args.out_dir/'direction_reference.sqlite3'), indent=2)); return
    if args.out_dir.exists():
        raise SystemExit('Existing datasets are never overwritten.')
    with gzip.open(PREVIOUS/'reference_packet.json.gz', 'rt') as f:
        previous = json.load(f)
    predictions = json.loads((PREVIOUS/'private_predictions.json').read_text())['predictions']
    excluded_ids = {t['id'].removeprefix('truth_') for t in previous['tasks']}
    source = list(csv.DictReader(MANIFEST.open()))
    participants = [f'P{i}' for i in range(1,21)]
    assert set(r['participant'] for r in source) == set(participants)
    protected = [MANIFEST, BASE/'correct_ecg_polarity.py', HERE/'direction_v3.py', HERE/'beatwise.py',
                 ROOT/'analysis/ecg_rodrigues_polarity_shadow_20261029_v1/assisted_direction.py',
                 BASE/'outputs/ecg_polarity_correction_all20_v1/correction_intervals.csv']
    hashes = {str(p):sha(p) for p in protected}
    args.out_dir.mkdir(parents=True, mode=0o700)
    selected, sampling, exclusions, raw_sources = [], [], [], []
    # Pass one fixes ALL selections before any new method is evaluated.
    for pid in participants:
        t, x = load_raw(pid)
        eligible = []
        rows = [r for r in source if r['participant'] == pid]
        for row in rows:
            cid = key(row)
            reasons = []
            if cid in excluded_ids:
                reasons.append('already_in_separate_difficult_or_optional_set')
            if not reasons:
                reasons.extend(eligibility_reasons(t, x, row))
            if reasons:
                exclusions.append(dict(id=cid, participant=pid, reasons=';'.join(reasons)))
            else:
                eligible.append(row)
        chosen = select_random(eligible, pid)
        selected.extend(chosen)
        sampling.append(dict(participant=pid, source_minutes=len(rows), eligible_minutes=len(eligible),
                             selected_minutes=len(chosen), inclusion_probability=len(chosen)/len(eligible)))
        path = RAW/pid/f'{pid}_polar_ecg_raw.npz'
        stat = path.stat()
        raw_sources.append(dict(participant=pid, path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
        print(f'{pid}: fixed {len(chosen)} random minutes from {len(eligible)} technically eligible minutes', flush=True)
        del t, x
    write_json(args.out_dir/'frozen_selection.json', dict(seed=SEED, per_participant=PER_PARTICIPANT,
        selected=[dict(id=key(r), **r) for r in selected], sampling=sampling,
        exclusions=exclusions, algorithm_outputs_used=False, human_answers_used=False))
    print('All 160 selections frozen. Evaluating the unchanged three methods.', flush=True)
    tasks, diagnostics = [], []
    from correct_ecg_polarity import _block_measurement
    for pid in participants:
        t, x = load_raw(pid)
        run_bounds = defaultdict(list)
        for r in source:
            if r['participant'] == pid:
                run_bounds[int(r['run_index'])].extend([float(r['start_ms']), float(r['end_ms'])])
        for row in [r for r in selected if r['participant'] == pid]:
            cid, task_id = key(row), 'truth_'+key(row)
            start, end, fs = float(row['start_ms']), float(row['end_ms']), float(row['fs_hz'])
            a, b = indices(t, row); mt, mx = t[a:b], x[a:b]
            old = _block_measurement(mx, mt, fs)
            if old[-1] != row['raw_polarity']:
                raise ValueError(cid+': frozen v1 reproduction changed')
            v3 = measure_direction_v3(mx, mt, fs)
            rs, re = min(run_bounds[int(row['run_index'])]), max(run_bounds[int(row['run_index'])])
            cs = max(rs, min(start-120000., re-300000.)); ce = cs+300000.
            ca, cb = np.searchsorted(t, [cs, ce], side='left'); ct, cx = t[ca:cb], x[ca:cb]
            issues = technical_reasons(ct, cx, cs, ce)
            if re-rs < 300000.:
                issues.append('continuous_run_shorter_than_300s')
            rod = dict(raw_polarity='technical_unavailable', reason='context:'+';'.join(issues), cross_check_status='unavailable')
            if not issues:
                if v3['raw_polarity'] == 'severe_artifact':
                    rod = measure_at_anchors(mx, mt, fs, [])
                else:
                    info = context_candidates(cx, ct, fs)
                    if info['failures']:
                        rod = dict(raw_polarity='detector_failure', reason=';'.join(info['failures']), cross_check_status='unavailable')
                    else:
                        anchors = minute_anchors(info['rodrigues'], ct, mt, a-ca, fs)
                        rod = measure_at_anchors(mx, mt, fs, anchors)
            tasks.append(dict(id=task_id, participant=pid, queue='main', t=np.round((mt-(end-60000.))/1000,4).tolist(),
                              x=np.round(mx,2).tolist(), focusStartSec=0., focusEndSec=60.))
            predictions[task_id] = {
                'old_v1':dict(direction=old[-1], support='historical_rule', flip=None),
                'v3_minute':dict(direction=v3['raw_polarity'], support=v3.get('cross_check_status','not_applicable'), flip=None),
                'rod300':dict(direction=rod['raw_polarity'], support=rod.get('cross_check_status','not_applicable'), flip=None)}
            diagnostics.append(dict(id=cid, participant=pid, context_issues=issues,
                directions={m:p['direction'] for m,p in predictions[task_id].items()},
                v3_reason=v3['reason'], rod_reason=rod['reason']))
        print(f'{pid}: three methods frozen for all eight random minutes', flush=True)
        del t, x
    # Retain every task from the new absolute-label round, byte-equivalent waveforms.
    for old in previous['tasks']:
        tasks.append(dict(old, queue='difficult' if old['queue']=='priority' else 'optional'))
    protocol = dict(version='raw_direction_truth_v2', seed=SEED, main_minutes=160, difficult_minutes=12,
        priority_minutes=160, optional_minutes=3,
        participants=[dict(participant=p, main_minutes=8,
            difficult_minutes=sum(t['participant']==p and t['queue']=='difficult' for t in tasks),
            optional_minutes=sum(t['participant']==p and t['queue']=='optional' for t in tasks)) for p in participants],
        sampling='Simple random sampling without replacement within each participant, fixed seed. Only complete technically valid raw minutes are eligible. Existing separate review tasks excluded to avoid duplicate cohorts. No amplitude, polarity, agreement, prediction, RR or human-answer filters.',
        population='Technically eligible complete raw minutes within each participant. Equal participant allocation is not duration weighting across participants. No claim of unseen external validation.',
        technical_policy='No repair, interpolation or deduplication. Legacy duration >=59.5 s; display uses the nominal 60-second boundary at end_ms minus 60000. Finite raw samples, strictly increasing timestamps, gaps and edge coverage <=250 ms, exact legacy sample count. Ordinary millisecond offsets of the first actual sample are not excluded. Severe amplitude artifacts remain eligible. Unavailable 300-second detector context is an abstention, never a reason to resample.',
        sampling_counts=sampling, old_records_used_for_scoring=False, old_records_prefilled=False,
        current_absolute_labels_preserved=True, algorithm_outputs_visible_during_labeling=False,
        peak_marks_modified=False, source_waveforms_modified=False,
        scoring='Main random, difficult targeted and optional cohorts scored separately, never pooled. Only completed current-round absolute labels count. Positive/negative assess correct, opposite, abstained and coverage. Balanced/mixed separate; unreadable/unsure excluded from signed accuracy. No three-minute flip-policy score from isolated random minutes.',
        protected_hashes=hashes, raw_sources=raw_sources, builder_sha256=sha(Path(__file__)),
        prior_packet=str(PREVIOUS/'reference_packet.json.gz'), prior_packet_sha256=sha(PREVIOUS/'reference_packet.json.gz'))
    write_json(args.out_dir/'private_predictions.json', dict(methods=list(METHODS), predictions=predictions))
    protocol['private_predictions_sha256'] = sha(args.out_dir/'private_predictions.json')
    with gzip.open(args.out_dir/'reference_packet.json.gz','wt') as f:
        json.dump(dict(version=1, tasks=tasks, protocol=protocol),f,separators=(',',':'),allow_nan=False)
    write_json(args.out_dir/'protocol.json', protocol)
    write_json(args.out_dir/'method_diagnostics.json', diagnostics)
    if any(sha(Path(p)) != value for p,value in hashes.items()):
        raise RuntimeError('A frozen method or production source changed')
    for entry in raw_sources:
        stat = Path(entry['path']).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (entry['size'],entry['mtime_ns'])
    print(json.dumps(dict(output=str(args.out_dir), main_minutes=160, difficult_minutes=12, optional=3,
                         labels_not_imported_until_cutover=True)), flush=True)


if __name__ == '__main__':
    main()
