"""Render a reproducible, prediction-blinded visual first-pass sample."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import random

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    args = parser.parse_args()
    folder = args.audit_dir / 'visual_first_pass'
    folder.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(folder / 'mpl_cache'))
    os.environ.setdefault('XDG_CACHE_HOME', str(folder / 'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    with (args.audit_dir / 'sampling_manifest.csv').open() as f:
        primary = [r for r in csv.DictReader(f) if r['sampling_stratum'] == 'old_ambiguous']
    with gzip.open(args.audit_dir / 'audit_packet.json.gz', 'rt') as f:
        packet = json.load(f)
    tasks = {t['id']: t for t in packet['tasks']}
    ids = []
    for pid in sorted({r['participant'] for r in primary}, key=lambda p: int(p[1:])):
        pool = sorted(r['id'] for r in primary if r['participant'] == pid)
        ids.append(random.Random('20261023:visual:' + pid).choice(pool))
    (folder / 'selection.json').write_text(json.dumps(dict(
        purpose='One randomly selected primary audit minute per participant for AI visual first pass only; no population prevalence estimate or human ground truth',
        ids=ids), indent=2) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    for offset in range(0, len(ids), 3):
        group = ids[offset:offset + 3]
        fig, axes = plt.subplots(len(group), 3, figsize=(18, 2.9 * len(group) + .5), squeeze=False,
                                 gridspec_kw={'width_ratios':[1.8,1,1]}, layout='constrained')
        for row, key in enumerate(group):
            task = tasks[key]
            t, x = np.asarray(task['t']), np.asarray(task['x'])
            duration = task['focusEndSec']
            intervals = [(0,duration),(.20*duration,.20*duration+5),(.70*duration,min(duration,.70*duration+5))]
            for col, (a,b) in enumerate(intervals):
                axis = axes[row,col]
                keep = (t>=a)&(t<=b)
                axis.plot(t[keep],x[keep],color='#224d58',linewidth=.65)
                axis.set_xlim(a,b)
                axis.set_title((key+' | whole focus minute') if col==0 else f'{a:.1f}–{b:.1f} s detail (own y-axis)',fontsize=10)
                axis.set_xlabel('Seconds within focus minute')
                axis.set_ylabel('Raw ECG (uV)')
                axis.yaxis.set_major_locator(MaxNLocator(4))
                axis.grid(alpha=.20)
        fig.suptitle('Prediction-blinded visual first pass | raw source ECG | all assessments require human confirmation',fontsize=12)
        path=folder/f'preview_{offset//3+1:02d}.png'
        fig.savefig(path,dpi=140)
        plt.close(fig)
        print(path,flush=True)


if __name__=='__main__': main()
