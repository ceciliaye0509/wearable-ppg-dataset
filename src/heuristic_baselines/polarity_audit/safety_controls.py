"""Add separately weighted safety controls without changing the primary sample."""
import csv
from collections import defaultdict
import gzip
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from correct_ecg_polarity import _split_runs


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def add_controls(folder, raw_root):
    folder, raw_root = Path(folder), Path(raw_root)
    protocol = json.loads((folder / 'audit_protocol.json').read_text())
    if protocol.get('safety_strata'):
        raise ValueError('Safety controls were already attached')
    if (folder / 'human_audit.sqlite3').exists():
        raise ValueError('Attach safety controls before starting the audit database')
    with gzip.open(folder / 'audit_packet.json.gz', 'rt') as f:
        packet = json.load(f)
    primary = rows(folder / 'sampling_manifest.csv')
    population = defaultdict(list)
    for row in rows(folder / 'polarity_blocks_v2.csv'):
        if row['old_raw_polarity'] in {'positive', 'negative'}:
            if row['raw_polarity'] not in {'positive', 'negative'}:
                population[(row['participant'], 'new_uncertainty')].append(row)
            elif row['old_raw_polarity'] != row['raw_polarity']:
                population[(row['participant'], 'direction_reversal')].append(row)
    sampled, strata = defaultdict(list), []
    for (pid, kind), pool in sorted(population.items(), key=lambda item: (int(item[0][0][1:]), item[0][1])):
        pool.sort(key=lambda r: (int(r['run_index']), int(r['block_index'])))
        n = len(pool) if kind == 'direction_reversal' else min(2, len(pool))
        chosen = random.Random(f"{protocol['seed']}:{pid}:{kind}").sample(pool, n)
        stratum = dict(participant=pid, sampling_stratum=kind, population_minutes=len(pool),
                       sampled_minutes=n, inverse_probability_weight=len(pool) / n)
        strata.append(stratum)
        sampled[pid].extend((row, stratum) for row in chosen)
    for row in primary:
        row['sampling_stratum'] = 'old_ambiguous'
    old_by_id = {r['id']: r for r in primary}
    for task in packet['tasks']:
        task['sampling'] = {k: old_by_id[task['id']][k] for k in ['sampling_stratum', 'population_minutes', 'sampled_minutes', 'inverse_probability_weight']}
    extras = []
    for pid in sorted(sampled, key=lambda p: int(p[1:])):
        with np.load(raw_root / pid / f'{pid}_polar_ecg_raw.npz', allow_pickle=False) as src:
            t, x = np.asarray(src['time_ms'], dtype=float), np.asarray(src['ecg_uv'], dtype=float)
        valid = np.isfinite(t) & np.isfinite(x)
        if not valid.all():
            t, x = t[valid], x[valid]
        bounds = _split_runs(t)
        for row, stratum in sampled[pid]:
            a, b = bounds[int(row['run_index'])]
            rt, rx = t[a:b], x[a:b]
            keep = np.r_[True, np.diff(rt) > 0]
            rt, rx = rt[keep], rx[keep]
            start, end = float(row['start_ms']), float(row['end_ms'])
            lo, hi = np.searchsorted(rt, [start - 20000, end + 20000], side='left')
            key = f"{pid}_R{int(row['run_index']):03d}_B{int(row['block_index']):05d}"
            packet['tasks'].append(dict(id=key, participant=pid, t=np.round((rt[lo:hi] - start) / 1000, 4).tolist(),
                                        x=np.round(rx[lo:hi], 2).tolist(), focusStartSec=0., focusEndSec=(end - start) / 1000,
                                        sampling={k: stratum[k] for k in ['sampling_stratum', 'population_minutes', 'sampled_minutes', 'inverse_probability_weight']}))
            extras.append(dict(id=key, participant=pid, run_index=row['run_index'], block_index=row['block_index'],
                               start_ms=row['start_ms'], end_ms=row['end_ms'],
                               population_minutes=stratum['population_minutes'], sampled_minutes=stratum['sampled_minutes'],
                               inverse_probability_weight=stratum['inverse_probability_weight'],
                               new_raw_polarity=row['raw_polarity'], old_raw_polarity=row['old_raw_polarity'],
                               qrs_readability='', direction_determinability='', direction='', confidence='', reviewer='', notes='',
                               sampling_stratum=stratum['sampling_stratum']))
        del t, x, rt, rx, keep, valid
    combined = primary + extras
    assert len({r['id'] for r in combined}) == len(combined)
    combined.sort(key=lambda r: (int(r['participant'][1:]), r['id']))
    packet['tasks'].sort(key=lambda r: (int(r['participant'][1:]), r['id']))
    protocol.update(primary_sample_minutes=len(primary), safety_sample_minutes=len(extras),
                    sample_minutes=len(combined), safety_strata=strata,
                    safety_sampling='Up to two SRS new-uncertainty minutes per participant; every old/new sign reversal. Kept separate from old-alert prevalence estimates.')
    packet['protocol'] = protocol
    with (folder / 'sampling_manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    with gzip.open(folder / 'audit_packet.json.gz', 'wt') as f:
        json.dump(packet, f, separators=(',', ':'), allow_nan=False)
    (folder / 'audit_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    return dict(primary_minutes=len(primary), safety_minutes=len(extras), total_minutes=len(combined))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(add_controls(args.audit_dir, args.raw_root)), flush=True)
