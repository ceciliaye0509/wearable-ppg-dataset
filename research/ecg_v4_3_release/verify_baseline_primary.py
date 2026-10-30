"""Recompute the primary directly from raw PPG predictions and actual NPZ GT."""
from pathlib import Path
import numpy as np
import pandas as pd
from build_release import ROOT,HERE,DEST,dump
def main():
 chunks=[];total=valid=0
 for file in DEST.glob('*.npz'):
  pid=file.stem.rsplit('_',1)[1]
  with np.load(file,allow_pickle=True) as z:
   q=z['ecg_label_qc_pass'];ref=pd.DataFrame(dict(window_index=z['source_window_index'],qc=q,corr_rmssd=z['ecg_rmssd_corrected_ms'],corr_sdnn=z['ecg_sdnn_corrected_ms'],raw_rmssd=z['ecg_rmssd_uncorrected_ms'],raw_sdnn=z['ecg_sdnn_uncorrected_ms']));total+=len(q);valid+=int(q.sum())
  raw=pd.read_csv(ROOT/'analysis/ecg_gt_release_20261029_v1/baseline_rerun'/f'{pid}.csv.gz',usecols=['window_index','device','channel','detector','ibi_correction','fiducial','polarity_mode','ppg_rmssd_ms','ppg_sdnn_ms'])
  mask=(raw.detector.eq('qppgfast')&raw.ibi_correction&raw.polarity_mode.eq('peak_train')&raw.fiducial.eq(raw.device.map({'Earring':'peak','Ring':'foot','Watch':'foot'})))
  take=raw[mask].merge(ref,on='window_index',validate='many_to_one');assert len(take)==6*len(ref);chunks.append(take)
 allrows=pd.concat(chunks,ignore_index=True);stored=pd.read_csv(HERE/'baseline/by_channel.csv');checks=0;rows=[]
 for definition,prefix,scope in [('corrected','corr','eligible_v4_3_GT'),('observed','raw','eligible_v4_3_raw_secondary')]:
  for (device,channel),g in allrows.groupby(['device','channel']):
   ok=g.qc&np.isfinite(g.ppg_rmssd_ms)&np.isfinite(g.ppg_sdnn_ms)&np.isfinite(g[prefix+'_rmssd'])&np.isfinite(g[prefix+'_sdnn']);g=g[ok]
   row=dict(definition=definition,device=device,channel=channel,n_valid=len(g),coverage_eligible_pct=100*len(g)/valid,coverage_all_pct=100*len(g)/total)
   expected=stored[stored.scope.eq('all__'+scope)&stored.baseline_method.eq('qppgfast_devicewise_corr')&stored.device.eq(device)&stored.channel.eq(channel)].iloc[0]
   assert len(g)==expected.n_valid;np.testing.assert_allclose(row['coverage_eligible_pct'],expected.coverage_pct,atol=1e-10)
   for name in ['rmssd','sdnn']:
    x=g['ppg_'+name+'_ms'].to_numpy(float);y=g[prefix+'_'+name].to_numpy(float);mae=float(np.abs(x-y).sum()/len(x));xc=x-x.mean();yc=y-y.mean();r=float(np.dot(xc,yc)/np.sqrt(np.dot(xc,xc)*np.dot(yc,yc)))
    np.testing.assert_allclose([mae,r],[expected[name.upper()+'_MAE_ms'],expected[name.upper()+'_r']],atol=1e-10,rtol=1e-10);checks+=2
    row[name.upper()+'_MAE_ms']=mae;row[name.upper()+'_r']=r
   rows.append(row)
 assert (total,valid,checks)==(14036,10271,48)
 pd.DataFrame(rows).to_csv(HERE/'baseline/independent_primary_by_channel.csv',index=False)
 dump(HERE/'baseline/independent_verification.json',dict(status='independent_primary_pass',source='Direct raw PPG columns; separately implemented selector and centered-dot correlation; references from actual final NPZ files',PPG_outputs_reused=True,ECG_cache_columns_ignored=True,targets=2,channels=6,numeric_error_and_correlation_checks=checks,candidate_windows=total,valid_reference_windows=valid))
 print('Independent current primary verification passed:48 error/correlation values and12 target/channel coverage checks.',flush=True)
if __name__=='__main__':main()
