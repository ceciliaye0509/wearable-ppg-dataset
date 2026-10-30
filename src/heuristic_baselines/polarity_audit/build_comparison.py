"""Sample frozen v1/v2 direction disagreements for anonymous paired review.

No re-fitting, active sign correction, prior packet, or annotation modification.
The packet is private server input; method identities are stripped by the API.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from correct_ecg_polarity import _block_measurement, _split_runs
from polarity_audit.beatwise import measure_beatwise
from polarity_audit.build_audit import block_key, json_clean, read_rows, write_rows

SOURCE = BASE / 'outputs/beatwise_polarity_audit_v2'
DEFAULT_OUT = BASE / 'outputs/polarity_pairwise_disagreements_v1'
RAW = BASE.parents[2] / 'Multisite-PPG/raw_data'
RESERVED = BASE / 'outputs/ecg_review_supplement_blind60_v1/review_manifest.csv'
LABELS = {'positive', 'negative', 'ambiguous', 'low_signal', 'severe_artifact'}


def overlaps_reserved(row, reserved):
    return any(float(row['start_ms']) < end and float(row['end_ms']) > start
               for start, end in reserved.get(row['participant'], []))


def sample_disagreements(rows, reserved, count=8, seed=20261028):
    if count < 1:
        raise ValueError('At least one minute per participant is required')
    groups = defaultdict(list)
    for row in rows:
        if row['raw_polarity'] not in LABELS or row['old_raw_polarity'] not in LABELS:
            raise ValueError('Unknown frozen direction label')
        groups[row['participant']].append(row)
    selected, strata = [], []
    for pid in sorted(groups, key=lambda p: int(p[1:])):
        population = [r for r in groups[pid] if r['raw_polarity'] != r['old_raw_polarity']]
        complete = [r for r in population if float(r['end_ms']) - float(r['start_ms']) >= 59500]
        pool = sorted((r for r in complete if not overlaps_reserved(r, reserved)), key=block_key)
        n = min(count, len(pool))
        rng = random.Random(f'{seed}:draw:{pid}')
        draw = sorted(rng.sample(pool, n), key=block_key)
        # Balance the hidden layout within participant; never expose this key.
        assignments = ['old'] * (n // 2) + ['new'] * (n - n // 2)
        random.Random(f'{seed}:layout:{pid}').shuffle(assignments)
        selected.extend(dict(r, method_a=method) for r, method in zip(draw, assignments))
        strata.append(dict(participant=pid, differing_blocks=len(population),
                           complete_differing_minutes=len(complete),
                           excluded_reserved_minutes=len(complete)-len(pool),
                           eligible_minutes=len(pool), sampled_minutes=n,
                           inclusion_probability=n/len(pool) if pool else 0,
                           inverse_probability_weight=len(pool)/n if n else 0))
    return selected, strata


def trace_old(x, t, fs):
    q005, q995, span, score, count, direction = _block_measurement(x, t, fs)
    indices = np.array([], dtype=int)
    signs = np.array([], dtype=int)
    if direction != 'severe_artifact' and len(x) >= max(100, round(10*fs)) and 60 <= fs <= 1000:
        filtered = sosfiltfilt(butter(3, [5, min(25, .45*fs)], btype='bandpass', fs=fs, output='sos'), x)
        noise = float(np.median(np.abs(filtered-np.median(filtered))) * 1.4826)
        indices, _ = find_peaks(np.abs(filtered), distance=max(1, round(.35*fs)), prominence=max(2.5*noise, 1e-9))
        signs = np.sign(filtered[indices]).astype(int)
    if len(indices) != count:
        raise ValueError('Old vote trace does not reproduce frozen event count')
    return dict(direction=direction, score=score, event_count=count,
                positive_votes=int(np.sum(signs > 0)), negative_votes=int(np.sum(signs < 0)),
                uncertain_beats=int(np.sum(signs == 0)),
                votes=[dict(index=int(i), sign=int(v)) for i,v in zip(indices, signs)])


def trace_new(x, t, fs):
    """Trace unchanged v2 candidate votes; assert parity with its frozen function.

    The frozen implementation deliberately remains untouched. This diagnostic
    repeats only its candidate/anchor stage to retain per-event vote signs.
    """
    result = measure_beatwise(x, t, fs)
    if result['reason'] in {'insufficient_signal', 'severe_amplitude_gate'}:
        anchors, votes = [], []
    else:
        filtered = sosfiltfilt(butter(3, [5, min(25, .45*fs)], btype='bandpass', fs=fs, output='sos'), x)
        derivative = np.diff(filtered, prepend=filtered[0])
        width = max(3, round(.080*fs))
        energy = np.convolve(derivative**2, np.ones(width)/width, mode='same')
        median = float(np.median(energy))
        noise = float(np.median(np.abs(energy-median))*1.4826)
        scale = float(np.quantile(energy, .95))
        seeds, _ = find_peaks(energy, distance=max(1, round(.250*fs)),
                             height=max(median+5*noise, .15*scale), prominence=max(4*noise, .10*scale))
        radius, outer = max(2, round(.080*fs)), max(max(2, round(.080*fs))+1, round(.200*fs))
        half_template = max(radius, round(.120*fs))
        anchors, votes = [], []
        for seed in seeds:
            if seed < outer or seed+outer >= len(x):
                continue
            shoulders = np.r_[x[seed-outer:seed-radius], x[seed+radius+1:seed+outer+1]]
            local = x[seed-radius:seed+radius+1] - float(np.median(shoulders))
            up, down = max(0., float(local.max())), max(0., float(-local.min()))
            dominance = (up-down)/max(up+down, 1e-12)
            anchor = int(seed-radius+int(np.argmax(np.abs(local))))
            if anchors and t[anchor]-t[anchors[-1]] < 200:
                continue
            if anchor < half_template or anchor+half_template >= len(x):
                continue
            anchors.append(anchor)
            votes.append(1 if dominance >= .20 else -1 if dominance <= -.20 else 0)
    if anchors != result['beat_indices']:
        raise ValueError('New diagnostic anchors differ from frozen implementation')
    positive, negative = sum(v > 0 for v in votes), sum(v < 0 for v in votes)
    if result['reason'] != 'too_few_qrs_candidates' and (positive != result['positive_votes'] or negative != result['negative_votes']):
        raise ValueError('New diagnostic votes differ from frozen implementation')
    return dict(direction=result['raw_polarity'], score=result['polarity_score'], event_count=len(anchors),
                positive_votes=positive, negative_votes=negative, uncertain_beats=sum(v == 0 for v in votes),
                votes=[dict(index=int(i), sign=int(v)) for i,v in zip(anchors, votes)])


def check_match(row, result, prefix=''):
    direction = row['old_raw_polarity'] if prefix else row['raw_polarity']
    score = float(row['old_polarity_score'] if prefix else row['polarity_score'])
    count = int(row['old_event_count'] if prefix else row['event_count'])
    if direction != result['direction'] or count != result['event_count'] or not np.isclose(score, result['score'], atol=1e-9, equal_nan=True):
        raise ValueError(f'{block_key(row)}: raw samples no longer reproduce frozen {prefix or "new"} result')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--per-participant', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261028)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit('Output exists; use a new directory, never replace saved comparison packets.')
    protocol = json.loads((SOURCE/'audit_protocol.json').read_text())
    if hashlib.sha256((HERE/'beatwise.py').read_bytes()).hexdigest() != protocol['method_source_sha256']:
        raise RuntimeError('Frozen candidate source changed; cannot trace this version')
    rows = read_rows(SOURCE/'polarity_blocks_v2.csv')
    reserved = defaultdict(list)
    # Protect both batches' complete five-minute detection contexts, not merely
    # the displayed 20-second excerpts. Do not send selection evidence to UI.
    for row in read_rows(RESERVED):
        start = float(row['window_start_unix_ms'])
        reserved[row['participant']].append((start, start+300000))
    chosen, strata = sample_disagreements(rows, reserved, args.per_participant, args.seed)
    tasks = []
    for pid in sorted({r['participant'] for r in chosen}, key=lambda p: int(p[1:])):
        with np.load(RAW/pid/f'{pid}_polar_ecg_raw.npz', allow_pickle=False) as data:
            times, signal = np.asarray(data['time_ms'], float), np.asarray(data['ecg_uv'], float)
        finite = np.isfinite(times) & np.isfinite(signal)
        times, signal = times[finite], signal[finite]
        runs = _split_runs(times)
        for row in [r for r in chosen if r['participant'] == pid]:
            a,b = runs[int(row['run_index'])]
            rt,rx = times[a:b],signal[a:b]
            keep = np.r_[True, np.diff(rt)>0]
            rt,rx = rt[keep],rx[keep]
            left,right = np.searchsorted(rt,[float(row['start_ms']),float(row['end_ms'])],side='left')
            t,x = rt[left:right],rx[left:right]
            if len(x) != int(row['sample_count']):
                raise ValueError(f'{block_key(row)}: source sample count mismatch')
            old,new = trace_old(x,t,float(row['fs_hz'])),trace_new(x,t,float(row['fs_hz']))
            check_match(row, old, 'old');check_match(row,new)
            old['decision'],new['decision'] = row['old_decision'],row['decision']
            methods = {'old':old,'new':new}
            mapping = dict(A=row['method_a'],B='new' if row['method_a']=='old' else 'old')
            tasks.append(dict(id='pair_'+block_key(row),participant=pid,
                              t=np.round((t-float(row['start_ms']))/1000,4).tolist(),x=np.round(x,2).tolist(),
                              focusStartSec=0.,focusEndSec=(float(row['end_ms'])-float(row['start_ms']))/1000,
                              sides={side:dict(method_id=method,**methods[method]) for side,method in mapping.items()}))
        print(f'{pid}: {sum(t["participant"]==pid for t in tasks)} paired minutes verified',flush=True)
        del times,signal,finite
    tasks.sort(key=lambda t:(int(t['participant'][1:]),t['id']))
    args.out_dir.mkdir(mode=0o700, parents=True)
    amended = dict(version='direction_disagreement_pairwise_v1',seed=args.seed,
                   requested_per_participant=args.per_participant,strata=strata,sampled_minutes=len(tasks),
                   selection='SRS without replacement within each participant; old_raw_polarity != raw_polarity; duration >=59.5 seconds; exclude all reserved blind/challenge five-minute contexts',
                   purpose='Assisted paired preference on direction disagreements, NOT independent ground-truth accuracy or whole-dataset prevalence',
                   assignment='Stable participant-balanced randomized A/B; identities withheld by review API',
                   source_csv_sha256=hashlib.sha256((SOURCE/'polarity_blocks_v2.csv').read_bytes()).hexdigest(),
                   beatwise_source_sha256=hashlib.sha256((HERE/'beatwise.py').read_bytes()).hexdigest(),
                   old_source_sha256=hashlib.sha256((BASE/'correct_ecg_polarity.py').read_bytes()).hexdigest(),
                   reserved_manifest_sha256=hashlib.sha256(RESERVED.read_bytes()).hexdigest(),
                   builder_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   active_correction_modified=False,previous_audit_modified=False)
    with gzip.open(args.out_dir/'comparison_packet.json.gz','wt',encoding='utf-8') as f:
        json.dump(json_clean(dict(version=1,tasks=tasks,protocol=amended)),f,separators=(',',':'),allow_nan=False)
    (args.out_dir/'comparison_protocol.json').write_text(json.dumps(amended,indent=2)+'\n')
    write_rows(args.out_dir/'participant_sampling.csv',strata)
    write_rows(args.out_dir/'sampling_manifest_private.csv',chosen)
    for path in args.out_dir.iterdir():path.chmod(0o600)
    print(json.dumps(dict(output=str(args.out_dir),sampled=len(tasks),strata=strata)),flush=True)


if __name__ == '__main__':
    main()
