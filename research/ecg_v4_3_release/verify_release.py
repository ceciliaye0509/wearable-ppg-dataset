"""Verify actual v4.3 NPZ files independently of the build and QC ledger."""
from pathlib import Path
import sys,json,zipfile,gc
import numpy as np
from build_release import ROOT,HERE,QA,FOLLOW,PARENT,DEST,NAME,sha,dump,SCALARS
sys.path.insert(0,str(ROOT/'analysis/ecg_training_release_v4_2_20261029_v1'))
from verify import entry_digest,fix_rr,scalar_stats
sys.path.insert(0,str(ROOT/'wearable-ppg-dataset/src'))
from ml_hrv.data.schema import validate_npz_schema
def main():
 build=json.loads((HERE/'build_report.json').read_text());protocol=sha(DEST/'ecg_reference_protocol.json');assert protocol==build['protocol_sha256']
 refs=json.loads((DEST/'annotation_snapshot.json').read_text());byid={a['clip_id']:a for a in refs['annotations']};assert len(byid)==394 and refs['effective_interval_records']==137
 for cid,expected in [('rr_recheck_v1_P3',[223.,292.,192.5]),('rr_recheck_v1_P9',[])]:
  rr=np.diff(byid[cid]['peak_abs_unix_ms']);np.testing.assert_array_equal(rr[rr<300],expected);assert byid[cid]['unreadable_intervals_abs_ms']==[]
 parentP3=byid['full300_v1_P3_W01'];assert len(parentP3['peak_sample_indices'])==559 and len(parentP3['unreadable_intervals_abs_ms'])==1 and parentP3['effective_reference_superseded_by']=='rr_recheck_v1_P3'
 assert sha(DEST/'manual_annotations_export12.csv')==json.loads((QA/'evaluation.json').read_text())['source_sha256']
 count=changed=valid=non_ecg=old_rejections=partial=0;hashes={};details=[]
 for row in build['participants']:
  pid=row['participant'];source=next(PARENT.glob(f'*_{pid}.npz'));target=DEST/row['file'];assert sha(source)==build['source_hashes'][str(source)] and sha(target)==build['NPZ_sha256'][target.name]
  with zipfile.ZipFile(source) as a,zipfile.ZipFile(target) as b,source.open('rb') as ar,target.open('rb') as br:
   for info in a.infolist():
    key=info.filename.removesuffix('.npy')
    if key.startswith('ecg_') or key=='config_json':continue
    other=b.getinfo(info.filename);assert entry_digest(ar,info)==entry_digest(br,other),(pid,key);non_ecg+=1
  validate_npz_schema(target)
  with np.load(source,allow_pickle=True) as a,np.load(target,allow_pickle=True) as b:
   n=len(b['source_window_index']);np.testing.assert_array_equal(a['source_window_index'],b['source_window_index']);np.testing.assert_array_equal(a['ppg_window_t0_ms'],b['ppg_window_t0_ms']);np.testing.assert_array_equal(a['ppg_window_t1_ms'],b['ppg_window_t1_ms'])
   q=b['ecg_label_qc_pass'];prior=a['ecg_label_qc_pass'];full=b['ecg_manual_full300_G1_exception'];assert not (prior&~q).any() and np.array_equal(q&~prior,full)
   cfg=json.loads(str(b['config_json']));assert cfg['dataset_name']==NAME and cfg['ecg_label_protocol_sha256']==protocol and cfg['manual_annotation_snapshot_sha256']==sha(DEST/'annotation_snapshot.json')
   np.testing.assert_array_equal(b['ecg_manual_repaired'],a['ecg_manual_repaired']);np.testing.assert_array_equal(b['ecg_manual_resolution_rejected'],a['ecg_manual_resolution_rejected']);np.testing.assert_array_equal(b['ecg_latest_saved_manual_unreadable_overlap'],a['ecg_latest_saved_manual_unreadable_overlap']);np.testing.assert_array_equal(b['ecg_polarity_flipped_sample_ratio'],a['ecg_polarity_flipped_sample_ratio'])
   assert not q[b['ecg_manual_resolution_rejected']].any()
   pks=b['ecg_r_peak_times_abs_ms'];old=a['ecg_r_peak_times_abs_ms'];rrs=b['ecg_rr_intervals_uncorrected_ms'];cos=b['ecg_rr_intervals_corrected_ms'];scalar={key:b[key] for key in SCALARS}
   oldrr=a['ecg_rr_intervals_uncorrected_ms'];oldco=a['ecg_rr_intervals_corrected_ms'];oldscalar={key:a[key] for key in SCALARS}
   for i in range(n):
    peak=np.asarray(pks[i],float)
    if not full[i]:
     np.testing.assert_array_equal(peak,old[i]);np.testing.assert_array_equal(rrs[i],oldrr[i]);np.testing.assert_array_equal(cos[i],oldco[i])
     for k in SCALARS:np.testing.assert_equal(scalar[k][i],oldscalar[k][i])
    else:
     cid=str(b['ecg_manual_full300_source_clip_id'][i]);np.testing.assert_allclose(peak,byid[cid]['peak_abs_unix_ms'],rtol=0,atol=.11);assert b['ecg_severe_source_blocks'][i]>0 and b['ecg_source_eligible'][i] and str(b['ecg_automatic_source_gate_reasons'][i])=='G1_severe_source_block'
    if q[i]:
     rr=np.diff(peak);co=fix_rr(rr);assert np.all((rr>=300)&(rr<=2000)) and 200<=len(peak)<=1200
     np.testing.assert_array_equal(rrs[i],rr.astype(np.float32));np.testing.assert_array_equal(cos[i],co.astype(np.float32))
     assert (co!=rr).mean()<=.2 and 40<=60000/co.mean()<=200 and np.sqrt(np.mean(np.diff(co)**2))<=200
     for state,values in [('uncorrected',rr),('corrected',co)]:
      for name,value in scalar_stats(values).items():
       key=f'ecg_{name}_{state}_ms' if name in ['rmssd','sdnn','mean_ibi'] else (f'ecg_hr_bpm_{state}' if name=='hr_bpm' else f'ecg_rr_cv_{state}')
       np.testing.assert_allclose(scalar[key][i],value,atol=1e-5,rtol=1e-7)
    else:
     for k in SCALARS:assert np.isnan(scalar[k][i])
   if pid=='P6':assert not q.any()
   count+=n;valid+=int(q.sum());changed+=int(full.sum());partial+=int(b['ecg_manual_repaired'].sum());old_rejections+=int(b['ecg_manual_resolution_rejected'].sum())
   details.append(dict(participant=pid,candidates=n,valid=int(q.sum()),full300_manual_exceptions=int(full.sum()),schema=True))
  hashes[target.name]=sha(target);print(pid,'verified',row['valid'],'valid',flush=True);gc.collect()
 assert (count,valid,changed,partial,old_rejections,non_ecg)==(14036,10271,4,97,7,330)
 verification=dict(status='verified_v4_3_export12_fullmanual',dataset_dir=str(DEST),dataset_name=NAME,protocol_sha256=protocol,manual_snapshot_sha256=sha(DEST/'annotation_snapshot.json'),candidate_windows=count,valid_ecg_label_windows=valid,invalid_ecg_label_windows=count-valid,participant_files=15,participants_with_valid_labels=14,all10271_full_HRV_independently_recomputed=True,all330_non_ECG_compressed_records_byte_identical=True,all_old10267_training_scalars_and_peaks_unchanged=True,new_full300_manual_exceptions=4,old97_local_repairs_retained=True,old7_manual_rejections_retained=True,all_P3_latest_marks_and_native_scope_supersession_retained=True,no_old_source_annotation_files_modified=True,all_model_schemas_passed=True,all_source_dataset_hashes_unchanged=True,NPZ_sha256=hashes,participants=details,model_training_started=False,HF_uploaded=False)
 dump(HERE/'verification.json',verification);dump(DEST/'integration_verification.json',verification);print('V4.3 VERIFICATION PASSED',flush=True)
if __name__=='__main__':main()
