"""E4: same PPG predictions, three label versions, fixed nine methods."""
from pathlib import Path
import sys,json,shutil
import numpy as np
import pandas as pd
from build_release import ROOT,HERE,PARENT,DEST,NAME,sha,dump
RELEASE=ROOT/'analysis/ecg_gt_release_20261029_v1';sys.path.insert(0,str(RELEASE))
from compare_references import select_variants,summarize,METHODS,HOLDOUT
OLD=ROOT/'analysis/ecg_manual_qc_update_20261029_side_v1/dataset/synced_3device_rawaligned_training_v4_1_stride30_rawslots_manualqc_20261029'
def main():
 out=HERE/'baseline';out.mkdir(exist_ok=False)
 proof=json.loads((ROOT/'analysis/ecg_guarded_startup_full_eval_20261029_side_v1/baseline_summary.json').read_text());protected={p:h for p,h in proof['source_hashes'].items() if '/baseline_rerun/' in p}
 frames=[];pred=[];raw_count=0
 for target in sorted(DEST.glob('*.npz')):
  pid=target.stem.rsplit('_',1)[1];r=None
  for label,folder in [('v4_1',OLD),('v4_2',PARENT),('v4_3',DEST)]:
   source=next(folder.glob(f'*_{pid}.npz'))
   with np.load(source,allow_pickle=True) as z:
    if r is None:r=pd.DataFrame(dict(participant=pid,window_index=z['source_window_index'],start_ms=z['ppg_window_t0_ms']))
    else:np.testing.assert_array_equal(r.window_index,z['source_window_index']);np.testing.assert_array_equal(r.start_ms,z['ppg_window_t0_ms'])
    r[label+'_qc']=z['ecg_label_qc_pass'];r[label+'_rmssd_ms']=z['ecg_rmssd_corrected_ms'];r[label+'_sdnn_ms']=z['ecg_sdnn_corrected_ms']
    if label=='v4_3':r['v4_3_raw_rmssd_ms']=z['ecg_rmssd_uncorrected_ms'];r['v4_3_raw_sdnn_ms']=z['ecg_sdnn_uncorrected_ms']
  path=RELEASE/'baseline_rerun'/f'{pid}.csv.gz';assert sha(path)==protected[str(path)]
  raw=pd.read_csv(path,usecols=lambda c:not c.startswith('ecg_'));assert len(raw)==len(r)*60
  selected=select_variants(raw);assert len(selected)==len(r)*54 and not selected.duplicated(['window_index','device','channel','baseline_method']).any()
  np.testing.assert_allclose(selected.start_ms,selected.window_index.map(r.set_index('window_index').start_ms),rtol=0,atol=.001)
  pred.append(selected.merge(r,on=['participant','window_index','start_ms'],validate='many_to_one'));frames.append(r);raw_count+=len(raw)
  print(pid,'PPG/reference rows rebound',len(raw),flush=True)
 ref=pd.concat(frames,ignore_index=True);chosen=pd.concat(pred,ignore_index=True);assert len(ref)==14036 and ref.v4_3_qc.sum()==10271
 ref.to_csv(out/'reference_versions_by_window.csv.gz',index=False)
 scopes={}
 for label in ['v4_1','v4_2','v4_3']:
  scopes[f'all_candidates_{label}_GT']=(pd.Series(True,index=ref.index),label)
  scopes[f'eligible_{label}_GT']=(ref[label+'_qc'],label)
 scopes.update(all_candidates_v4_3_raw_secondary=(pd.Series(True,index=ref.index),'v4_3_raw'),eligible_v4_3_raw_secondary=(ref.v4_3_qc,'v4_3_raw'),paired_v4_1_GT=(ref.v4_1_qc&ref.v4_2_qc,'v4_1'),paired_v4_2_for_v4_1_GT=(ref.v4_1_qc&ref.v4_2_qc,'v4_2'),paired_v4_2_GT=(ref.v4_2_qc&ref.v4_3_qc,'v4_2'),paired_v4_3_GT=(ref.v4_2_qc&ref.v4_3_qc,'v4_3'),added_full300_manual_v4_3_GT=(ref.v4_3_qc&~ref.v4_2_qc,'v4_3'))
 tables=[];people=[]
 for split,pids in [('all',set(ref.participant)),('train',set(ref.participant)-HOLDOUT),('holdout',HOLDOUT)]:
  for name,(mask,prefix) in scopes.items():
   r=ref[mask&ref.participant.isin(pids)]
   if r.empty:continue
   take=chosen.merge(r[['participant','window_index']],on=['participant','window_index'],validate='many_to_one')
   denominators={(d,c):len(r) for d in ['Earring','Ring','Watch'] for c in ['ppg_green','ppg_ir']}
   tables.append(summarize(take,split+'__'+name,prefix,denominators))
   if split=='all' and name=='eligible_v4_3_GT':
    for pid,g in take.groupby('participant'):
     denom={(d,c):int(r.participant.eq(pid).sum()) for d in ['Earring','Ring','Watch'] for c in ['ppg_green','ppg_ir']};part=summarize(g,split+'__'+name,prefix,denom);part['participant']=pid;people.append(part)
  print(split,'all predefined scopes scored',flush=True)
 channel=pd.concat(tables,ignore_index=True);channel.to_csv(out/'by_channel.csv',index=False);pd.concat(people,ignore_index=True).to_csv(out/'by_participant_channel.csv',index=False)
 macro=channel.groupby(['scope','baseline_method'],as_index=False).agg(channels=('channel','size'),coverage_pct=('coverage_pct','mean'),RMSSD_MAE_ms=('RMSSD_MAE_ms','mean'),RMSSD_r=('RMSSD_r','mean'),SDNN_MAE_ms=('SDNN_MAE_ms','mean'),SDNN_r=('SDNN_r','mean'))
 macro.to_csv(out/'six_channel_macro.csv',index=False)
 stored=pd.read_csv(OLD/'baseline_by_channel.csv');reproduced=0
 for split in ['all','train','holdout']:
  for scope in ['all_candidates_v4_1_GT','eligible_v4_1_GT']:
   name=split+'__'+scope;a=channel[channel.scope.eq(name)].sort_values(['baseline_method','device','channel']);b=stored[stored.scope.eq(name)].sort_values(['baseline_method','device','channel']);assert len(a)==len(b)==54
   for key in ['n_total','n_valid','coverage_pct','RMSSD_MAE_ms','RMSSD_r','SDNN_MAE_ms','SDNN_r']:np.testing.assert_allclose(a[key],b[key],rtol=1e-10,atol=1e-10,equal_nan=True)
   reproduced+=len(a)
 for split in ['all','train','holdout']:
  a=channel[channel.scope.eq(split+'__paired_v4_2_GT')].sort_values(['baseline_method','device','channel']);b=channel[channel.scope.eq(split+'__paired_v4_3_GT')].sort_values(['baseline_method','device','channel'])
  for key in ['n_total','n_valid','coverage_pct','RMSSD_MAE_ms','RMSSD_r','SDNN_MAE_ms','SDNN_r']:np.testing.assert_allclose(a[key],b[key],atol=1e-12,rtol=1e-12,equal_nan=True)
 primary='qppgfast_devicewise_corr';train=macro[macro.scope.eq('train__eligible_v4_3_GT')].set_index('baseline_method');hold=macro[macro.scope.eq('holdout__eligible_v4_3_GT')].set_index('baseline_method')
 def accept(r,b):return dict(coverage_retained=bool(r.coverage_pct>=b.coverage_pct-1),RMSSD_MAE_improved=bool(r.RMSSD_MAE_ms<b.RMSSD_MAE_ms),SDNN_MAE_improved=bool(r.SDNN_MAE_ms<b.SDNN_MAE_ms),RMSSD_r_not_worse=bool(r.RMSSD_r>=b.RMSSD_r),SDNN_r_not_worse=bool(r.SDNN_r>=b.SDNN_r))
 tests=[]
 for method in METHODS:
  if method==primary:continue
  criteria=accept(train.loc[method],train.loc[primary]);tests.append(dict(method=method,**criteria,train_pass=all(criteria.values())))
 tests=pd.DataFrame(tests);tests.to_csv(out/'train_acceptance.csv',index=False);passed=tests.loc[tests.train_pass,'method'].tolist();alternative=min(passed,key=lambda x:train.loc[x,'RMSSD_MAE_ms']) if passed else None;holdcriteria=accept(hold.loc[alternative],hold.loc[primary]) if alternative else {};selected=alternative if alternative and all(holdcriteria.values()) else primary
 decision=dict(original_primary=primary,train_selected_alternative=alternative,holdout_criteria=holdcriteria,recommended_primary=selected,change_recommended=selected!=primary,predeclared_acceptance_unchanged=True,ECG_target_not_selected_by_PPG_scores=True)
 dump(out/'freeze_recommendation.json',decision)
 current=macro[macro.baseline_method.eq(selected)&macro.scope.isin(['all__all_candidates_v4_3_GT','all__eligible_v4_3_GT'])].copy();current.insert(0,'dataset_version','v4.3');current['n_total_per_channel']=current.scope.map({'all__all_candidates_v4_3_GT':14036,'all__eligible_v4_3_GT':10271});current['coverage_denominator']=current.scope.map({'all__all_candidates_v4_3_GT':'all_PPG_candidate_windows','all__eligible_v4_3_GT':'eligible_ECG_reference_windows'});current.to_csv(out/'current_primary_scores.csv',index=False)
 comparisons=macro[macro.baseline_method.eq(primary)&macro.scope.str.startswith('all__')].copy();comparisons.to_csv(out/'primary_version_comparison.csv',index=False)
 verification=dict(status='baseline_rescoring_verified',dataset_name=NAME,PPG_raw_prediction_rows=raw_count,PPG_variant_rows=len(chosen),PPG_predictions_rerun=False,preexisting_cache_hashes_unchanged=True,source_hashes=protected,methods=9,prior_v4_1_channel_rows_independently_reproduced=reproduced,common_v4_2_v4_3_all_metrics_identical=True,corrected_NN_proxy_primary=True,observed_RR_secondary_same_windows=True,selection=decision,train_holdout=sorted(HOLDOUT))
 dump(out/'verification.json',verification)
 for source,target in [('six_channel_macro.csv','baseline_six_channel_macro.csv'),('by_channel.csv','baseline_by_channel.csv'),('train_acceptance.csv','baseline_train_acceptance.csv'),('freeze_recommendation.json','baseline_freeze_recommendation.json'),('current_primary_scores.csv','baseline_primary_scores.csv'),('verification.json','baseline_verification.json')]:shutil.copy2(out/source,DEST/target)
 print(current.to_string(index=False),flush=True);print(json.dumps(decision,indent=2),flush=True)
if __name__=='__main__':main()
