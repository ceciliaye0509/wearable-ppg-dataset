"""Run the candidate direction method and draw a participant-stratified audit.

Outputs are versioned and separate from the active peak-review packet/database.
Human judgments are deliberately left blank and hidden from automated outcomes.
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
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from correct_ecg_polarity import Block, _assign_stable_spans, _split_runs
from beatwise import PARAMETERS, measure_beatwise

DEFAULT_SOURCE = BASE / 'outputs/ecg_polarity_correction_all20_v1/polarity_blocks.csv'
DEFAULT_RAW = BASE.parents[2] / 'Multisite-PPG/raw_data'
DEFAULT_OUT = BASE / 'outputs/beatwise_polarity_audit_v2'


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def block_key(row):
    return f"{row['participant']}_R{int(row['run_index']):03d}_B{int(row['block_index']):05d}"


def stratified_sample(rows, per_participant=8, seed=20261023):
    if per_participant < 2:
        raise ValueError('Request at least two audit minutes per non-census participant')
    population = defaultdict(list)
    for row in rows:
        if row['raw_polarity'] == 'ambiguous':
            population[row['participant']].append(row)
    selected, strata = {}, []
    for pid in sorted({r['participant'] for r in rows}, key=lambda p: int(p[1:])):
        pool = sorted(population[pid], key=lambda r: (int(r['run_index']), int(r['block_index'])))
        n = min(per_participant, len(pool))
        # Each participant has its own seed: adding a participant cannot reshuffle others.
        rng = random.Random(f'{seed}:{pid}')
        draw = rng.sample(pool, n)
        strata.append(dict(participant=pid, population_minutes=len(pool), sampled_minutes=n,
                           inclusion_probability=n / len(pool) if pool else 0,
                           inverse_probability_weight=len(pool) / n if n else 0))
        for row in draw:
            selected[block_key(row)] = dict(row)
    return selected, strata


def json_clean(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-blocks', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--sample-per-participant', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261023)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit('Output directory is not empty; choose a new versioned output directory.')
    source_rows = read_rows(args.source_blocks)
    sampled, strata = stratified_sample(source_rows, args.sample_per_participant, args.seed)
    stratum_by_pid = {s['participant']: s for s in strata}
    by_pid = defaultdict(list)
    for row in source_rows:
        by_pid[row['participant']].append(row)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, all_spans, tasks, comparisons, sampling_rows = [], [], [], [], []
    known_example = None
    for pid in sorted(by_pid, key=lambda p: int(p[1:])):
        path = args.raw_root / pid / f'{pid}_polar_ecg_raw.npz'
        with np.load(path, allow_pickle=False) as source:
            times, signal = np.asarray(source['time_ms'], dtype=float), np.asarray(source['ecg_uv'], dtype=float)
        finite = np.isfinite(times) & np.isfinite(signal)
        if not finite.all():
            times, signal = times[finite], signal[finite]
        runs = _split_runs(times)
        by_run = defaultdict(list)
        for old in by_pid[pid]:
            by_run[int(old['run_index'])].append(old)
        participant_rows = []
        for run_index, old_blocks in sorted(by_run.items()):
            a, b = runs[run_index]
            rt, rx = times[a:b], signal[a:b]
            keep = np.r_[True, np.diff(rt) > 0]
            rt, rx = rt[keep], rx[keep]
            new_blocks, run_rows = [], []
            for old in sorted(old_blocks, key=lambda r: int(r['block_index'])):
                lo = int(np.searchsorted(rt, float(old['start_ms']), side='left'))
                hi = int(np.searchsorted(rt, float(old['end_ms']), side='left'))
                bt, bx = rt[lo:hi], rx[lo:hi]
                if len(bx) != int(old['sample_count']):
                    raise ValueError(f'{block_key(old)}: source sample count no longer matches the old manifest')
                result = measure_beatwise(bx, bt, float(old['fs_hz']))
                if not np.isclose(result['raw_p01_p99_span_uv'], float(old['raw_p01_p99_span_uv']), atol=.01, rtol=0, equal_nan=True):
                    raise ValueError(f'{block_key(old)}: source amplitude no longer matches the old manifest')
                block = Block(participant=pid, run_index=run_index, block_index=int(old['block_index']),
                              start_ms=float(old['start_ms']), end_ms=float(old['end_ms']),
                              sample_count=len(bx), fs_hz=float(old['fs_hz']),
                              raw_p01_p99_span_uv=result['raw_p01_p99_span_uv'],
                              q005_uv=result['q005_uv'], q995_uv=result['q995_uv'],
                              polarity_score=result['polarity_score'], event_count=result['event_count'],
                              raw_polarity=result['raw_polarity'])
                new_blocks.append(block)
                row = dict(vars(block), old_raw_polarity=old['raw_polarity'], old_decision=old['decision'],
                           old_polarity_score=old['polarity_score'], old_event_count=old['event_count'])
                row.update({k: v for k, v in result.items() if k not in row and k != 'beat_indices'})
                run_rows.append(row)
                key = block_key(old)
                if pid == 'P2' and run_index == 1 and int(old['block_index']) == 6:
                    known_example = dict(id=key, old=old, candidate=json_clean(result),
                                         purpose='Previously inspected diagnostic example; not independent validation')
                if key in sampled:
                    left = int(np.searchsorted(rt, float(old['start_ms']) - 20000, side='left'))
                    right = int(np.searchsorted(rt, float(old['end_ms']) + 20000, side='left'))
                    ct, cx = rt[left:right], rx[left:right]
                    duration = (float(old['end_ms']) - float(old['start_ms'])) / 1000
                    tasks.append(dict(id=key, participant=pid,
                                      t=np.round((ct - float(old['start_ms'])) / 1000, 4).tolist(),
                                      x=np.round(cx, 2).tolist(), focusStartSec=0., focusEndSec=duration))
                    sampling_rows.append(dict(id=key, participant=pid, run_index=run_index,
                                              block_index=int(old['block_index']), start_ms=old['start_ms'],
                                              end_ms=old['end_ms'], population_minutes=stratum_by_pid[pid]['population_minutes'],
                                              sampled_minutes=stratum_by_pid[pid]['sampled_minutes'],
                                              inverse_probability_weight=stratum_by_pid[pid]['inverse_probability_weight'],
                                              new_raw_polarity=result['raw_polarity'], old_raw_polarity=old['raw_polarity'],
                                              qrs_readability='', direction_determinability='', direction='',
                                              confidence='', reviewer='', notes=''))
            # No unknown block is bridged into a new automatic-flip interval.
            spans = _assign_stable_spans(new_blocks, min_run_blocks=3, max_ambiguous_gap_blocks=0)
            all_spans.extend(spans)
            for row, block in zip(run_rows, new_blocks, strict=True):
                row['decision'] = block.decision
            participant_rows.extend(run_rows)
            del rt, rx, keep
        all_rows.extend(participant_rows)
        old_counts = Counter(row['old_raw_polarity'] for row in participant_rows)
        new_counts = Counter(row['raw_polarity'] for row in participant_rows)
        changed = sum(row['old_raw_polarity'] == 'ambiguous' and row['raw_polarity'] in {'positive', 'negative'} for row in participant_rows)
        comparison = dict(participant=pid, total_minutes=len(participant_rows), old_ambiguous=old_counts['ambiguous'],
                          new_ambiguous=new_counts['ambiguous'], new_low_signal=new_counts['low_signal'],
                          severe_artifact=new_counts['severe_artifact'], old_ambiguous_now_determinate=changed,
                          old_determinate_now_uncertain=sum(row['old_raw_polarity'] in {'positive', 'negative'} and row['raw_polarity'] not in {'positive', 'negative'} for row in participant_rows),
                          old_determinate_opposite_direction=sum(row['old_raw_polarity'] in {'positive', 'negative'} and row['raw_polarity'] in {'positive', 'negative'} and row['old_raw_polarity'] != row['raw_polarity'] for row in participant_rows),
                          audit_sample_minutes=stratum_by_pid[pid]['sampled_minutes'])
        comparisons.append(comparison)
        print(json.dumps(comparison), flush=True)
        del times, signal, finite
    if len(tasks) != len(sampled) or len({t['id'] for t in tasks}) != len(tasks):
        raise RuntimeError('Sampling packet is incomplete or has duplicate IDs')
    tasks.sort(key=lambda r: (int(r['participant'][1:]), r['id']))
    sampling_rows.sort(key=lambda r: (int(r['participant'][1:]), r['id']))
    protocol = dict(version='beatwise_polarity_audit_v2_candidate', seed=args.seed,
                    population='All raw_polarity=ambiguous blocks in the frozen all20 v1 manifest; participant-stratified SRS without replacement',
                    population_minutes=sum(s['population_minutes'] for s in strata), sample_minutes=len(tasks),
                    requested_per_participant=args.sample_per_participant, strata=strata,
                    source_blocks=str(args.source_blocks.resolve()),
                    source_blocks_sha256=hashlib.sha256(args.source_blocks.read_bytes()).hexdigest(),
                    method_source_sha256=hashlib.sha256((HERE / 'beatwise.py').read_bytes()).hexdigest(),
                    parameters=PARAMETERS, waveform='Original raw ECG, no sign flip or filtering; 60-s focus plus up to 20 s of context on either side',
                    assessment='Independent QRS readability and direction determinability; automated predictions/votes hidden from review API',
                    source_and_active_annotations_modified=False, automatic_flip_status='Candidate intervals only; not promoted to the active correction manifest',
                    estimates='Use inverse-probability weights. No prevalence estimate until the planned sample is fully rated; readable/partial/unreadable are separate categories. No automated result is a human rating.')
    write_rows(args.out_dir / 'polarity_blocks_v2.csv', all_rows)
    write_rows(args.out_dir / 'participant_comparison.csv', comparisons)
    write_rows(args.out_dir / 'sampling_manifest.csv', sampling_rows)
    span_fields = ['participant', 'run_index', 'start_ms', 'end_ms', 'orientation', 'correction_factor',
                   'classified_blocks', 'span_blocks_including_bridged_gaps', 'median_abs_polarity_score',
                   'median_qrs_event_count_per_block', 'median_fs_hz']
    write_rows(args.out_dir / 'correction_intervals_candidate.csv', all_spans, span_fields)
    with gzip.open(args.out_dir / 'audit_packet.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(dict(version=1, tasks=tasks, protocol=protocol), handle, separators=(',', ':'), allow_nan=False)
    (args.out_dir / 'audit_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (args.out_dir / 'known_p2_example.json').write_text(json.dumps(known_example, indent=2, allow_nan=False) + '\n')
    from safety_controls import add_controls
    controls = add_controls(args.out_dir, args.raw_root)
    print(json.dumps(dict(output=str(args.out_dir.resolve()), **controls, analyzed_minutes=len(all_rows))), flush=True)


if __name__ == '__main__':
    main()
