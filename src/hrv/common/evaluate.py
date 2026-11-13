"""One explicit denominator and named HRV targets for both recipes."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from .dataset import iter_rows


def summarize(records):
    output=[]
    keys=sorted({(r['device'],r['channel']) for r in records})
    for device,channel in keys:
        rows=[r for r in records if r['device']==device and r['channel']==channel]
        eligible=[r for r in rows if r['ecg_eligible']]
        good=[r for r in eligible if np.isfinite(r['pred_rmssd_ms']) and np.isfinite(r['pred_sdnn_ms'])]
        item=dict(device=device,channel=channel,candidates=len(rows),ecg_eligible=len(eligible),successful=len(good),
            coverage_all_candidates_pct=100*len(good)/len(rows) if rows else 0.,coverage_eligible_pct=100*len(good)/len(eligible) if eligible else 0.)
        for name in ['rmssd','sdnn']:
            pred=np.array([r['pred_'+name+'_ms'] for r in good]);truth=np.array([r['ecg_'+name+'_ms'] for r in good])
            item[name+'_mae_ms']=float(np.mean(abs(pred-truth))) if len(good) else None
            item[name+'_r']=float(np.corrcoef(pred,truth)[0,1]) if len(good)>1 and np.std(pred)>0 and np.std(truth)>0 else None
        output.append(item)
    return output


def baseline_main(predict,reporter=None):
    p=argparse.ArgumentParser();p.add_argument('--dataset-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--detector',choices=['qppgfast','scipy'],default='qppgfast');p.add_argument('--participants');p.add_argument('--limit',type=int);a=p.parse_args()
    ids=set(a.participants.split(',')) if a.participants else None
    records=[];count=0
    for row in iter_rows(a.dataset_dir):
        pid=str(row['participant'])
        if ids is not None and pid not in ids:continue
        eligible=bool(row['ecg_label_qc_pass']) and np.isfinite(row['ecg_rmssd_corrected_ms']) and np.isfinite(row['ecg_sdnn_corrected_ms'])
        for di,d in enumerate(row['devices']):
            for ci,c in enumerate(row['channels']):
                result=predict(row,di,ci,detector=a.detector) if eligible else dict(rmssd_ms=np.nan,sdnn_ms=np.nan)
                records.append(dict(method=a.detector,participant=pid,source_window_index=int(row['source_window_index']),start_ms=float(row['ppg_window_t0_ms']),
                    device=str(d),channel=str(c),ecg_eligible=eligible,ecg_rmssd_ms=float(row['ecg_rmssd_corrected_ms']),ecg_sdnn_ms=float(row['ecg_sdnn_corrected_ms']),
                    pred_rmssd_ms=float(result['rmssd_ms']),pred_sdnn_ms=float(result['sdnn_ms'])))
        count+=1
        if a.limit and count>=a.limit:break
    a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]) if records else ['participant']);writer.writeheader();writer.writerows(records)
    (a.output_dir/'summary.json').write_text(json.dumps(summarize(records),indent=2))
    if reporter is not None:reporter(records,a.output_dir)
