"""Reproducible, isolated development checks for ECG direction v3.

Reads raw ECG, frozen direction outputs and existing reviews. Writes only a new
output directory. Human-assisted preferences are development evidence, never
independent ground truth. Reserved peak-review contexts are not inspected.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
import random
import sqlite3
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from correct_ecg_polarity import _split_runs
from polarity_audit.beatwise import measure_beatwise
from polarity_audit.build_audit import block_key, json_clean, read_rows, write_rows
from polarity_audit.build_comparison import RAW, RESERVED, overlaps_reserved, check_match, trace_old
from polarity_audit.direction_v3 import PARAMETERS, measure_direction_v3, measure_direction_variant

OUTPUTS = BASE/'outputs'
FROZEN = OUTPUTS/'beatwise_polarity_audit_v2'
PAIR = OUTPUTS/'polarity_pairwise_disagreements_v1'
SIGNED = {'positive', 'negative'}
VARIANTS = {
    'minute_primary': dict(aggregate='minute', bands='primary'),
    'minute_dual': dict(aggregate='minute', bands='both'),
    'tiles_primary': dict(aggregate='tiles', bands='primary'),
    'tiles_dual': dict(aggregate='tiles', bands='both'),
}
METHODS = ['old','new',*VARIANTS,'v3']
LITERATURE = [
    dict(title='Introduction to electrocardiogram signal quality assessment and estimated accuracy for textile electrodes (2025)',
         url='https://doi.org/10.1038/s41598-025-25365-x',
         use='借鉴短区间内 QRS 模板和逐搏形态相关性；原论文用于 RR 数据质量，不是方向校正。本文 0.70 相关性和方向门槛是项目开发参数，没有照搬其 0.98 质量阈值。'),
    dict(title='Automated differentiation of wide QRS complex tachycardia using QRS complex polarity (2024)',
         url='https://www.nature.com/articles/s43856-024-00725-2',
         use='参考主导 QRS 模板、等电位基线和正负振幅/面积的定义。原文按带符号面积定义方向，本项目按主峰振幅定义，二者不能混用；未复现其临床分类模型。'),
    dict(title='A reproducible benchmark of QRS detection algorithms across diverse ECG datasets and noise conditions (2026)',
         url='https://pmc.ncbi.nlm.nih.gov/articles/PMC13190806/',
         use='参考统一条件比较及跨数据集检验；该文评测 QRS 检测，不证明任何算法在本数据上的方向判定最优。'),
    dict(title='Accelerated Sample-Accurate R-Peak Detectors Based on Visibility Graphs (2023)',
         url='https://www.eurasip.org/Proceedings/Eusipco/Eusipco2023/pdfs/0001090.pdf',
         use='区分 R 峰精确定位和本任务的方向判断。FastNVG 的可见图偏向正峰，本轮不直接用它决定倒置波方向，未调用其检测器。'),
    dict(title='Unsupervised Transformer-Based Anomaly Detection in ECG Signals (2023)',
         url='https://www.mdpi.com/1999-4893/16/3/152',
         use='仅参考连续 200/600 ms 中值滤波作为原始波形基线诊断视图；未训练或复现 Transformer。'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_protected():
    paths = [BASE/'correct_ecg_polarity.py', HERE/'beatwise.py', RESERVED,
             FROZEN/'audit_packet.json.gz', PAIR/'comparison_packet.json.gz']
    paths += sorted((OUTPUTS/'ecg_polarity_correction_all20_v1').glob('*.csv'))
    return {str(p): sha(p) for p in paths}


def add_controls(rows, selected, reserved, seed):
    groups = defaultdict(list)
    for row in rows:
        if block_key(row) in selected or overlaps_reserved(row,reserved):
            continue
        if float(row['end_ms'])-float(row['start_ms']) < 59500:
            continue
        old, new = row['old_raw_polarity'], row['raw_polarity']
        if old == new and old in {'positive','negative','ambiguous','severe_artifact','low_signal'}:
            groups[(row['participant'],old)].append(row)
    counts = dict(positive=6, negative=6, ambiguous=6, severe_artifact=2, low_signal=2)
    for (pid, label), pool in sorted(groups.items()):
        draw = random.Random(f'{seed}:{pid}:{label}').sample(sorted(pool,key=block_key),min(counts[label],len(pool)))
        for row in draw:
            selected[block_key(row)].add('control_'+label)


def target_summary(rows, field, target):
    pairs = [r for r in rows if r.get(target) in SIGNED]
    decisive = [r for r in pairs if r[field] in SIGNED]
    return dict(n=len(pairs), decisive=len(decisive),
                same_direction=sum(r[field] == r[target] for r in decisive),
                opposite_direction=sum(r[field] != r[target] for r in decisive),
                abstained=len(pairs)-len(decisive))


def make_summary(results):
    methods = METHODS
    reviewed = [r for r in results if r.get('preference_choice')]
    explicit = [r for r in results if r.get('human_status') == 'complete']
    paired = [r for r in results if 'paired' in r['groups'].split(';')]
    controls = [r for r in results if any(g.startswith('control_') for g in r['groups'].split(';'))]
    summary = dict(total_minutes=len(results), participants=sorted({r['participant'] for r in results},key=lambda p:int(p[1:])),
                   paired_minutes=len(paired), reviewed_preferences=len(reviewed),
                   explicit_direction_reviews=len(explicit), control_minutes=len(controls),
                   preference_comparison={m:target_summary(reviewed,m,'preference_direction_proxy') for m in methods},
                   explicit_review_comparison={m:target_summary(explicit,m,'human_direction') for m in methods},
                   output_distributions={group:{m:dict(Counter(r[m] for r in rows)) for m in methods}
                                         for group,rows in [('all',results),('paired',paired),('controls',controls)]},
                   per_participant=[], opposite_from_agreeing_controls=[],
                   cross_check_status=dict(Counter(r['cross_check_status'] for r in results)),
                   undecided_preferences=[dict(id=r['id'],v3=r['v3'],cross_check_status=r['cross_check_status']) for r in reviewed if r['preference_choice'] not in {'A','B'}])
    for pid in summary['participants']:
        subset = [r for r in reviewed if r['participant'] == pid]
        if subset:
            summary['per_participant'].append(dict(participant=pid, reviewed=len(subset),
                **{m:target_summary(subset,m,'preference_direction_proxy') for m in ['old','new','v3']}))
    summary['opposite_from_agreeing_controls'] = [r['id'] for r in controls if r['old'] == r['new'] and r['old'] in SIGNED and r['v3'] in SIGNED and r['v3'] != r['old']]
    return summary


def plot_examples(out, examples, result_by_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(examples),3,figsize=(18,3.1*len(examples)),squeeze=False)
    for line, (key,x,t,fs,details) in enumerate(examples):
        r = result_by_id[key]
        secs = (t-t[0])/1000
        d = details['diagnostics']
        visible = (secs >= 20) & (secs < 25)
        ax = axes[line,0]
        ax.plot(secs[visible],x[visible],label='Original raw',lw=.8,color='#9b9b9b')
        ax.plot(secs[visible],d['baseline_corrected_raw'][visible],label='Median-baseline diagnostic',lw=.9,color='#166479')
        ax.set_title(f"{key}\nv1: {r['old']} | v2: {r['new']} | v3: {r['v3']}",fontsize=10)
        ax.set_ylabel('ECG (uV)');ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.15)
        ax = axes[line,1]
        for name,col in [('primary_filtered','#177a60'),('confirmation_filtered','#9569b2')]:
            ax.plot(secs[visible],d[name][visible],label='5-25 Hz' if name=='primary_filtered' else '8-35 Hz',lw=.9,color=col)
        anchor = np.array(details['beat_indices'],dtype=int)
        anchor = anchor[(secs[anchor] >= 20) & (secs[anchor] < 25)]
        ax.scatter(secs[anchor],d['primary_filtered'][anchor],s=16,color='#d17927',label='Slope anchors (not human peaks)',zorder=3)
        ax.set_title('QRS direction channels; slow drift removed',fontsize=10)
        ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.15)
        ax = axes[line,2]
        for template,label,color in zip(d['templates'],['5-25 Hz template','8-35 Hz template'],['#177a60','#9569b2']):
            ax.plot(np.array(d['template_seconds'])*1000,template,label=label,color=color)
        ax.axhline(0,color='gray',lw=.6);ax.set_title(f"15s tiles: {', '.join(z['direction'] for z in details['tiles'])}\n{details['reason']}",fontsize=9)
        ax.legend(fontsize=8);ax.grid(alpha=.15)
        for ax in axes[line,:2]:ax.set_xlabel('Seconds within minute')
        axes[line,2].set_xlabel('ms from polarity-independent slope anchor')
    fig.tight_layout();fig.savefig(out/'component_examples.png',dpi=150);plt.close(fig)


def render_report(out, summary, protocol):
    def table(headers, rows):
        return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
    def metrics_table(name):
        return table(['方法','可对照方向数','给出方向','与记录方向一致','方向相反','未给出方向'],
                     [[m,*summary[name][m].values()] for m in summary[name]])
    body = '<h1>ECG 朝向方法 v3：开发检查</h1><p>目标：在方向正确的前提下，减少不必要的“不明确”。方向指 QRS 主峰的振幅朝向，不代表导联放反的临床诊断。此版本只输出候选判定，未翻转数据、未改网页、未改已有人工标记。</p>'
    body += '<h2>这次改了什么</h2><ol><li>用 QRS 能量找候选，用最大斜率对齐，避免原始波形漂移和 R/S 振幅交替使模板错位。</li><li>在 5–25 Hz 去漂移通道判断 QRS 正负振幅，用 8–35 Hz 通道交叉检查；原始波形仍用于严重伪迹门槛。</li><li>不再要求每一搏都达到 0.20 的强振幅优势。整分钟看模板、连续振幅优势和至少 80% 的逐搏同向性；相关性≥0.70、优势≥0.05，得到主方向。</li><li>独立在每约 15 秒、两个滤波通道做更严格检查。至少 75% 时间同向则显示 supported；支持不足显示 limited_support，保留主方向；明确的反向区间或滤波通道冲突会使主方向拒判。这样把方向与交叉证据充分性分开。参数均为项目开发选择，需要新人工方向标记检验。</li></ol>'
    body += '<h2>样本与证据的性质</h2><p>'+html.escape(f"共 {summary['total_minutes']} 个分钟，{len(summary['participants'])} 位 participant；其中新旧分歧样本 {summary['paired_minutes']}，已有成对选择 {summary['reviewed_preferences']}，已有明确方向审阅记录 {summary['explicit_direction_reviews']}，额外分层控制样本 {summary['control_minutes']}。")+'</p>'
    body += '<p>所有这些都是开发检查；29 条选择已用于发现问题。选了某算法，只能推导“偏好算法给出的方向”，不能当成独立方向真值。旧方向审阅也含算法辅助。尚未审阅的样本只比较输出变化，不判断谁正确。样本按类别分层，不能据此估计全数据错误率或明确率。原方向审阅 149 段中有 1 段与保留的峰值盲验上下文重叠，本轮排除，故只纳入 148 段；对应已有方向记录由 36 条降至 35 条。各来源有重叠，不能直接相加。保留的 40+20 峰值盲验五分钟上下文全部排除。</p>'
    body += '<h2>相对于已有选择的方向代理</h2>'+metrics_table('preference_comparison')
    body += '<p>new=冻结的 v2。minute_primary=整分钟单滤波通道；minute_dual=整分钟双通道；tiles_primary=分区间单通道；tiles_dual=分区间双通道；v3=主方向与交叉支持分开，实际方向冲突会阻止输出。这些消融均用同一套新候选与对齐流程，不能把 v2→v3 的全部变化归因于某一项。</p>'
    body += '<p>根据目前目标，优先保留能给出稳定主方向的 minute_primary，并把严格门槛作为独立支持信息；当前开发样本尚未显示严格门槛减少了方向相反的输出。这个策略选择使用了开发数据，不能据此声称泛化性能最优。</p>'
    body += '<h2>已有直接方向记录（含算法辅助，仍非独立验证）</h2>'+metrics_table('explicit_review_comparison')
    body += '<h2>按 participant 看已选择样本</h2>'+table(['participant','已选分钟','旧版明确/同向/反向','v2 明确/同向/反向','v3 明确/同向/反向'],[[r['participant'],r['reviewed'],*[f"{r[m]['decisive']}/{r[m]['same_direction']}/{r[m]['opposite_direction']}" for m in ['old','new','v3']]] for r in summary['per_participant']])
    body += '<h2>新旧分歧 127 分钟上的输出分布</h2>'+table(['方法','positive','negative','ambiguous','low_signal','severe_artifact'],[[m,*[counts.get(k,0) for k in ['positive','negative','ambiguous','low_signal','severe_artifact']]] for m,counts in summary['output_distributions']['paired'].items()])
    body += '<h2>控制检查</h2><p>旧版与 v2 均给相同正负方向、但 v3 给反向的控制样本：'+html.escape(', '.join(summary['opposite_from_agreeing_controls']) or '无')+'。这只表示算法是否发生符号冲突，不表示两版共同方向就是真值。</p>'
    body += '<p>交叉检查支持状态（514 段分层开发样本，非全数据比例）：'+html.escape(str(summary['cross_check_status']))+'。</p><p>你选 unsure 的片段仍须复查，v3 的输出不是你的确认：'+html.escape(str(summary['undecided_preferences']))+'。</p>'
    body += '<h2>典型组件图</h2><p>左：原始波形与中值基线诊断；中：去漂移后的两个 QRS 通道与斜率锚点；右：模板。锚点用于方向诊断，不是人工 R 峰标签。</p><img src="component_examples.png" style="width:100%;max-width:1600px">'
    body += '<h2>文献依据与边界</h2><ul>'+''.join('<li><a href="'+s['url']+'">'+html.escape(s['title'])+'</a>：'+html.escape(s['use'])+'</li>' for s in LITERATURE)+'</ul>'
    body += '<h2>现在能得出的结论</h2><p>可以检查新版是否恢复了人能判断的方向，以及改动发生在定位、振幅、模板还是区间汇总。还不能据此宣布最优算法或全数据准确率。下一轮应在新旧新三版分歧处，先隐藏算法结论，直接标“朝上 / 朝下 / 双相接近 / 信号无法判断”；按 participant 和失败形态抽样、拉开时间距离，再冻结参数。一分钟方向确认之后，才独立验证连续三分钟的自动翻转策略。</p>'
    body += '<h2>可追溯文件</h2><ul>'+''.join(f'<li><a href="{p}">{p}</a></li>' for p in ['summary.json','protocol.json','minute_results.csv','tile_diagnostics.csv','beat_diagnostics.json.gz','source_protection.json','human_review_snapshot.json'])+'</ul>'
    (out/'development_report_zh.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>ECG direction v3 development</title><style>body{font:17px/1.7 system-ui,sans-serif;max-width:1400px;margin:40px auto;padding:0 28px;color:#172c36}table{border-collapse:collapse;font-size:15px}th,td{border:1px solid #bdcdd2;padding:8px 13px;text-align:left}th{background:#eef5f6}h2{margin-top:36px}a{color:#126b86}</style>'+body+'</html>',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preferences',type=Path,default=Path('/Users/jiqiyu/Downloads/direction_paired_preferences.csv'))
    parser.add_argument('--out-dir',type=Path,default=OUTPUTS/'ecg_direction_v3_development_20261028_r1')
    parser.add_argument('--seed',type=int,default=20261028)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit('Output already exists; choose a new directory. Saved results are never replaced.')
    before = snapshot_protected()
    if sha(HERE/'beatwise.py') != json.loads((FROZEN/'audit_protocol.json').read_text())['method_source_sha256']:
        raise RuntimeError('Frozen v2 implementation no longer matches its protocol')
    all_rows = read_rows(FROZEN/'polarity_blocks_v2.csv')
    by_key = {block_key(r):r for r in all_rows}
    if len(by_key) != len(all_rows):raise ValueError('Duplicate frozen minute keys')
    reserved = defaultdict(list)
    for r in read_rows(RESERVED):
        start = float(r['window_start_unix_ms']);reserved[r['participant']].append((start,start+300000))
    selected = defaultdict(set)
    packet = json.load(gzip.open(PAIR/'comparison_packet.json.gz','rt'))
    paired = {t['id']:t for t in packet['tasks']}
    for key in paired:selected[key.removeprefix('pair_')].add('paired')
    for r in read_rows(FROZEN/'sampling_manifest.csv'):
        if not overlaps_reserved(by_key[r['id']],reserved):selected[r['id']].add('prior_audit')
    add_controls(all_rows,selected,reserved,args.seed)
    if any(overlaps_reserved(by_key[key],reserved) for key in selected):
        raise RuntimeError('Reserved context would be inspected')
    pref_rows = read_rows(args.preferences)
    if len({r['id'] for r in pref_rows}) != len(pref_rows):raise ValueError('Duplicate human preference IDs')
    if set(r['id'] for r in pref_rows)-paired.keys():raise ValueError('Unknown human preference ID')
    prefs = {r['id']:r for r in pref_rows if r['status']=='complete'}
    with sqlite3.connect('file:'+str(FROZEN/'human_audit.sqlite3')+'?mode=ro',uri=True) as con:
        human = {key:json.loads(answer) for key,answer in con.execute('select id,answer_json from ratings')}
    human_snapshot = {key:{k:v for k,v in answer.items() if k in ['status','direction','direction_determinability','qrs_readability','confidence','assistance_shown']} for key,answer in human.items()}
    args.out_dir.mkdir(parents=True,mode=0o700)
    results, tiles, beats, examples = [], [], [], []
    example_ids = {next(t['id'][5:] for t in packet['tasks'] if t['participant']==pid) for pid in ['P2','P3','P4','P5']}
    for pid in sorted({by_key[k]['participant'] for k in selected},key=lambda p:int(p[1:])):
        with np.load(RAW/pid/f'{pid}_polar_ecg_raw.npz',allow_pickle=False) as data:
            times, signal = np.asarray(data['time_ms'],float),np.asarray(data['ecg_uv'],float)
        finite = np.isfinite(times)&np.isfinite(signal)
        times,signal = times[finite],signal[finite]
        runs = _split_runs(times)
        rows = [by_key[k] for k in selected if by_key[k]['participant']==pid]
        for run_i in sorted({int(r['run_index']) for r in rows}):
            a,b = runs[run_i];rt,rx = times[a:b],signal[a:b]
            keep = np.r_[True,np.diff(rt)>0];rt,rx = rt[keep],rx[keep]
            for source in sorted([r for r in rows if int(r['run_index'])==run_i],key=block_key):
                key = block_key(source);fs = float(source['fs_hz'])
                left,right = np.searchsorted(rt,[float(source['start_ms']),float(source['end_ms'])],side='left')
                t,x = rt[left:right],rx[left:right]
                if len(x) != int(source['sample_count']):raise RuntimeError('Raw minute sample count changed: '+key)
                new = measure_beatwise(x,t,fs)
                check_match(source,dict(direction=new['raw_polarity'],event_count=new['event_count'],score=new['polarity_score']))
                old = trace_old(x,t,fs);check_match(source,old,'old')
                variants = {m:measure_direction_variant(x,t,fs,**config) for m,config in VARIANTS.items()}
                variants['v3'] = measure_direction_v3(x,t,fs,diagnostics=True)
                inv = measure_direction_v3(-x,t,fs)
                v3 = variants['v3']
                expected = {'positive':'negative','negative':'positive'}.get(v3['raw_polarity'],v3['raw_polarity'])
                if inv['raw_polarity'] != expected or inv['beat_indices'] != v3['beat_indices']:
                    raise RuntimeError('Sign-equivariance failed: '+key)
                row = dict(id=key,participant=pid,groups=';'.join(sorted(selected[key])),
                           start_ms=source['start_ms'],end_ms=source['end_ms'],fs_hz=fs,
                           old=source['old_raw_polarity'],new=source['raw_polarity'],
                           new_reason=new['reason'],old_count=old['event_count'],new_count=new['event_count'],
                           **{m:r['raw_polarity'] for m,r in variants.items()},
                           v3_reason=v3['reason'],v3_count=v3['event_count'],supported_fraction=v3['supported_fraction'],
                           cross_check_status=v3['cross_check_status'],cross_check_direction=v3['cross_check_direction'],cross_check_reason=v3['cross_check_reason'],
                           preference_choice='',preference_method='',preference_direction_proxy='',
                           human_status='',human_direction='',human_determinability='',human_assisted='')
                if 'pair_'+key in prefs:
                    pref = prefs['pair_'+key];row['preference_choice']=pref['choice']
                    chosen = paired['pair_'+key]['sides'].get(pref['choice'])
                    if chosen:
                        row['preference_method']=chosen['method_id'];row['preference_direction_proxy']=chosen['direction']
                if key in human:
                    h = human[key];row.update(human_status=h.get('status',''),human_determinability=h.get('direction_determinability',''),
                        human_direction=h.get('direction','') if h.get('direction_determinability')=='yes' else '',human_assisted=h.get('assistance_shown','unknown'))
                results.append(row)
                for m,v in variants.items():
                    for tile_i,tile in enumerate(v['tiles']):
                        for band_i,band in enumerate(tile['bands']):
                            tiles.append(dict(id=key,method=m,tile=tile_i,band=band_i,start_sec=tile['start_sec'],end_sec=tile['end_sec'],
                                              count=tile['count'],tile_direction=tile['direction'],tile_reason=tile['reason'],**band))
                d = v3.get('diagnostics',{})
                beats.append(dict(id=key,beat_indices=v3['beat_indices'],
                                  relative_beat_seconds=d.get('relative_beat_seconds',[]),
                                  per_beat_dominance=d.get('per_beat_dominance',[]),tiles=v3['tiles']))
                if key in example_ids:examples.append((key,x.copy(),t.copy(),fs,v3))
        print(f'{pid}: {len(rows)} raw minutes, frozen parity and sign reversal verified',flush=True)
        del times,signal,finite
    after = snapshot_protected()
    if before != after:raise RuntimeError('Protected source or correction artifact changed during evaluation')
    results.sort(key=lambda r:(int(r['participant'][1:]),r['id']))
    summary = make_summary(results)
    protocol = dict(version='ECG_direction_v3_development_20261028_r1',parameters=PARAMETERS,ablations=VARIANTS,
                    seed=args.seed,source_sha256=sha(HERE/'direction_v3.py'),evaluator_sha256=sha(Path(__file__)),
                    preferences_source=str(args.preferences),preferences_sha256=sha(args.preferences),
                    reserved_manifest_sha256=sha(RESERVED),literature=LITERATURE,
                    purpose='Development, not independent validation; no production correction or annotation writes',
                    sampling='All paired disagreement and non-reserved original direction-audit tasks, plus up to 6 agreeing positive/negative/ambiguous and 2 severe/low-signal control minutes per participant; no prevalence inference',
                    sign_reversal_checks=len(results),frozen_v1_v2_reproduced=len(results),
                    excluded_reserved_contexts=True,production_modified=False)
    for name,obj in [('summary',summary),('protocol',protocol),('source_protection',dict(before=before,after=after,unchanged=before==after)),
                     ('human_review_snapshot',dict(paired_preferences=pref_rows,prior_direction_reviews=human_snapshot))]:
        (args.out_dir/(name+'.json')).write_text(json.dumps(json_clean(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_rows(args.out_dir/'minute_results.csv',results);write_rows(args.out_dir/'tile_diagnostics.csv',tiles)
    with gzip.open(args.out_dir/'beat_diagnostics.json.gz','wt') as f:json.dump(json_clean(beats),f,allow_nan=False,separators=(',',':'))
    plot_examples(args.out_dir,examples,{r['id']:r for r in results})
    render_report(args.out_dir,summary,protocol)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('Report: '+str(args.out_dir/'development_report_zh.html'),flush=True)


if __name__ == '__main__':
    main()
