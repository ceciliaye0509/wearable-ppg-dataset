"""Package the verified export12/manual-full300 decisions without overwrites."""
from pathlib import Path
import sys,json,shutil,subprocess
import numpy as np
import pandas as pd
ROOT=Path('/Users/jiqiyu/Desktop/Daily_HRV');HERE=Path(__file__).resolve().parent
QA=ROOT/'analysis/ecg_export12_validation_20261029_v1';FOLLOW=ROOT/'analysis/ecg_current_questions_20261029_v1/followup_01'
DECISION=json.loads((QA/'joint_decision.json').read_text());PARENT=Path(DECISION['parent_dataset_dir'])
NAME='synced_3device_rawaligned_training_v4_3_stride30_rawslots_fullmanual_export12_20261029'
DEST=HERE/'dataset'/NAME
sys.path.insert(0,str(ROOT/'analysis/ecg_training_release_v4_2_20261029_v1'))
from build import sha,dump,append_replace,SCALARS

def main():
 assert not (HERE/'build_report.json').exists() and not DEST.exists(),'Never replace an existing complete or partial release'
 assert shutil.disk_usage(HERE).free>4*2**30
 DEST.mkdir(parents=True)
 excludes={'config.json','README.md','checksums.json','ecg_reference_protocol.json','annotation_snapshot.json','manual_annotations_export10.csv','latest_saved_manual_unreadable_intervals.csv','ecg_label_audit.csv.gz','integration_verification.json','dataset_freeze_manifest.json'}
 for p in PARENT.iterdir():
  if p.is_file() and p.suffix!='.npz' and p.name not in excludes and not p.name.endswith('_summary.csv'):shutil.copy2(p,DEST/p.name)
 for source,target in [(QA/'inputs/ecg_rpeak_annotations-12.csv','manual_annotations_export12.csv'),(QA/'annotation_snapshot_export12.json','annotation_source_snapshot_export12.json'),(QA/'effective_manual_references.json','effective_manual_references.json'),(QA/'manual_reference_supersession.json','manual_reference_supersession.json'),(QA/'effective_manual_unreadable_intervals.csv','latest_saved_manual_unreadable_intervals.csv'),(QA/'short_RR_latest_review.json','manual_short_RR_latest_review.json'),(QA/'random_audit_summary.json','random_audit_summary.json'),(QA/'random_clip_scores.csv','random_audit_clip_scores.csv'),(QA/'joint_decision.json','research_joint_decision.json')]:shutil.copy2(source,DEST/target)
 shutil.copytree(FOLLOW/'manual_G1_exception_candidates',DEST/'manual_full300_G1_exceptions')
 effective=json.loads((QA/'effective_manual_references.json').read_text())['annotations'];snap=[]
 for r in effective:
  row=dict(clip_id=r['clip_id'],reviewer=r['reviewer'],participant=r['participant'],status=r['status'],confidence=r['confidence'],review_batch=r['batch'],annotation_scope='reviewer' if r['full300'] else 'shared',peak_sample_indices=r['peak_sample_indices'],peak_abs_unix_ms=r['peak_abs_ms'],unreadable_intervals_abs_ms=r['unreadable_intervals_abs_ms'],reviewed_pages=r['reviewed_pages'])
  if 'superseded_by' in r:row.update(effective_reference_superseded_by=r['superseded_by'],parent_native_sample_range=r['superseded_parent_sample_range'])
  snap.append(row)
 dump(DEST/'annotation_snapshot.json',dict(source='export12 with two explicit recheck subrange overrides',source_export_sha256=sha(QA/'inputs/ecg_rpeak_annotations-12.csv'),clips=394,shared_annotations=362,full_window_annotations=32,raw_export_interval_records=138,effective_interval_records=137,annotations=snap))
 snapshotsha=sha(DEST/'annotation_snapshot.json')
 protocol=dict(version='ecg_v4_3_export12_full300_manual_G1_20261029',parent_dataset=str(PARENT),parent_protocol_sha256=sha(PARENT/'ecg_reference_protocol.json'),dataset_name=NAME,
  automatic_detector='Unchanged Rodrigues2021 + local50ms + guarded startup',manual_partial_repairs=97,manual_full300_G1_exceptions=4,
  manual_reference=dict(export_sha256=sha(DEST/'manual_annotations_export12.csv'),effective_snapshot_sha256=snapshotsha,supersession_sha256=sha(DEST/'manual_reference_supersession.json'),effective_interval_records=137,policy='Explicit latest recheck replaces only its source parent native crop; old source records retained, unreadable tails outside crop retained.'),
  RR=dict(bounds_ms=[300,2000],scope='Conservative full300s corrected-NN-proxy training eligibility, not a QRS truth test',correction='median11 reflect; >20% replacement',raw_observations='Preserve all latest human marks including P3 223/292/192.5ms; no deletion by bounds'),
  G1=dict(global_threshold_uv=5000,automatic_gate_unchanged=True,manual_exception='Only4 exact fully readable high-confidence human-reviewed300s windows; replace all peaks, verify native samples, preserve amplitude alert diagnostics; no neighbour recovery'),
  G2='Effective export12 intervals: same window mask asv4.2 despite scope updates; preserve all other vetoes and7 previous manual rejections',
  quality=dict(peak_count=[200,1200],HR_bpm=[40,200],max_correction_ratio=.2,max_RMSSD_ms=200,no_gap_bridging=True),
  label_targets=dict(primary='fixed corrected NN proxy: RMSSD/SDNN ms',secondary='observed raw RR HRV on identical windows',clinical_NN_truth_claimed=False),
  polarity=dict(manifest_sha256=sha(PARENT/'ecg_polarity_correction_intervals.csv'),accepted_negative_minutes=2717,flip_intervals=34,physical_device_switch_validated=False),
  random_audit=dict(participants=14,clips=28,seconds=560,human_peaks=754,FN20=0,FP20=0,scope='Pre-addition v4.2 retained window-context sample; not all300s/all-population zero error'),
  non_ECG='All compressed non-ECG/config NPZ records byte-identical to parent',baseline='Evaluated separately with same PPG predictions and frozen9-candidate acceptance protocol',model_cache_rebuild_required=True,HF_uploaded=False)
 dump(DEST/'ecg_reference_protocol.json',protocol);ph=sha(DEST/'ecg_reference_protocol.json')
 def config(c):
  c=dict(c);c.update(dataset_name=NAME,ecg_label_version=protocol['version'],parent_local_dataset=PARENT.name,parent_ecg_label_protocol_sha256=protocol['parent_protocol_sha256'],ecg_label_protocol_sha256=ph,manual_annotation_snapshot_sha256=snapshotsha,manual_export_sha256=protocol['manual_reference']['export_sha256'],manual_reference_supersession_sha256=protocol['manual_reference']['supersession_sha256'],manual_repaired_windows=97,manual_full300_G1_exception_windows=4,new_manual_rejected_windows=0,parent_v4_2_manual_rejections_retained=7,valid_ecg_label_windows=10271,invalid_ecg_label_windows=3765,baseline_status='separate_version_bound_evaluation',ecg_label_provenance='97 local anchored human repairs retained;4 full300 human substitutions; all other peak sequences unchanged fromv4.2. Inherited automatic diagnostics remain historical.',primary_reference_definition='corrected_NN_proxy',required_secondary_reference_definition='observed_RR',local_integration_status='verified_by_release_workflow',HF_uploaded=False)
  return c
 parentchecks=json.loads((PARENT/'checksums.json').read_text());ledger=pd.read_csv(QA/'prospective_QC_ledger.csv.gz');summaries=[];sourcehashes={};newhashes={};fields=['ecg_r_peak_times_abs_ms','ecg_r_peak_times_rel_ms','ecg_r_peak_indices_grid','ecg_r_peak_amplitudes_raw','ecg_r_peak_amplitudes_corrected','ecg_rr_intervals_uncorrected_ms','ecg_rr_intervals_corrected_ms','ecg_peak_count','ecg_valid_ibi_ratio','ecg_ibi_correction_ratio','ecg_estimate_rmssd_ms','ecg_estimate_sdnn_ms',*SCALARS]
 for source in sorted(PARENT.glob('*.npz')):
  pid=source.stem.rsplit('_',1)[1];assert sha(source)==parentchecks[source.name];sourcehashes[str(source)]=parentchecks[source.name]
  with np.load(source,allow_pickle=True) as z:
   values={k:z[k].copy() for k in z.files if k.startswith('ecg_')};cfg=config(json.loads(str(z['config_json'])));indices=z['source_window_index'];oldqc=z['ecg_label_qc_pass'];n=len(indices)
  strings=[k for k,v in values.items() if v.dtype.kind=='U']
  for k in strings:values[k]=values[k].astype(object)
  values['ecg_parent_v4_2_label_qc_pass']=oldqc.copy();values['ecg_parent_v4_2_source_gate_reasons']=values['ecg_source_gate_reasons'].copy()
  values['ecg_parent_v4_2_new_manual_qc_rejection']=values['ecg_new_manual_qc_rejection'].copy();values['ecg_parent_v4_2_new_G2_manual_rejection']=values['ecg_new_G2_manual_rejection'].copy()
  values['ecg_manual_full300_G1_exception']=np.zeros(n,bool);values['ecg_manual_full300_source_clip_id']=np.full(n,'',dtype=object);values['ecg_automatic_source_gate_reasons']=values['ecg_source_gate_reasons'].copy()
  strings+=['ecg_parent_v4_2_source_gate_reasons','ecg_manual_full300_source_clip_id','ecg_automatic_source_gate_reasons']
  rows=ledger[ledger.participant.eq(pid)].sort_values('source_window_index');np.testing.assert_array_equal(rows.source_window_index,indices)
  np.testing.assert_array_equal(rows.effective_G2,values['ecg_latest_saved_manual_unreadable_overlap'])
  for c in json.loads((DEST/'manual_full300_G1_exceptions/verification.json').read_text())['records']:
   if c['participant']!=pid:continue
   i=c['source_window_index']
   with np.load(DEST/'manual_full300_G1_exceptions'/Path(c['candidate_file']).name,allow_pickle=True) as z:
    assert int(z['source_window_index'][0])==i
    for key in fields:values[key][i]=z[key][0]
   values['ecg_manual_full300_G1_exception'][i]=True;values['ecg_manual_full300_source_clip_id'][i]=c['clip_id'];values['ecg_label_origin'][i]='full300_human_G1_exception'
  values['ecg_label_qc_pass']=rows.prospective_qc.to_numpy(bool);values['ecg_source_gate_reasons']=rows.effective_source_reasons.fillna('').to_numpy(str)
  values['ecg_source_eligible']=values['ecg_source_gate_reasons']==''
  values['ecg_label_qc_reason'][values['ecg_label_qc_pass']]=''
  values['ecg_new_manual_qc_rejection']=oldqc&~values['ecg_label_qc_pass'];values['ecg_new_G2_manual_rejection']=np.zeros(n,bool)
  for key in strings:values[key]=np.asarray(values[key],dtype=str)
  values['config_json']=np.array(json.dumps(cfg,ensure_ascii=False));target=DEST/f'{NAME}_{pid}.npz'
  subprocess.run(['/bin/cp','-c',str(source),str(target)],check=True);assert target.stat().st_ino!=source.stat().st_ino
  append_replace(target,values);newhashes[target.name]=sha(target)
  count=dict(participant=pid,file=target.name,candidates=n,valid=int(values['ecg_label_qc_pass'].sum()),parent_valid=int(oldqc.sum()),manual_full300_exceptions=int(values['ecg_manual_full300_G1_exception'].sum()))
  summaries.append(count);print(pid,'built',count['valid'],'valid',flush=True)
 pd.DataFrame(summaries).to_csv(DEST/f'{NAME}_summary.csv',index=False);shutil.copy2(QA/'prospective_QC_ledger.csv.gz',DEST/'ecg_label_audit.csv.gz');dump(DEST/'config.json',config(json.loads((PARENT/'config.json').read_text())))
 (DEST/'README.md').write_text('''# ECG training dataset v4.3

15 participant files,14,036 candidates,10,271 valid references across14 participants. Parentv4.2 remains intact.

Unchanged automatic Rodrigues/local50ms/guarded-startup detections,97 prior local human repairs and7 prior manual rejections are retained. Four exact300s windows use complete saved human peaks as explicit G1 exceptions. Automatic5000uV screening diagnostics remain; no global threshold relaxation or neighbour recovery.

Export12 is preserved alongside an effective reference snapshot. Explicit P3/P9 rechecks supersede only their source native-sample crops. The P3 223/292/192.5ms marks and newly readable scope are retained. Old unreadable tails outside that crop remain. These excerpts do not map to newly eligible main training references.

Primary target is the declared fixed-corrected NN proxy, not clinically adjudicated NN. Observed-RR HRV is retained and reported secondarily.300..2000ms is a conservative complete-window eligibility gate; never a criterion for deleting human peaks.

Training mask: ecg_label_qc_pass and finite corrected RMSSD/SDNN. All non-ECG/config NPZ members and window identities are byte-preserved. New model caches/scalers must bind this version. Baseline evaluation is version-bound separately; no model training or HF upload occurred.
''')
 assert sum(r['valid'] for r in summaries)==10271
 dump(HERE/'build_report.json',dict(dataset_name=NAME,dataset_dir=str(DEST),parent_dataset_dir=str(PARENT),protocol_sha256=ph,manual_snapshot_sha256=snapshotsha,source_hashes=sourcehashes,NPZ_sha256=newhashes,participants=summaries,candidate_windows=14036,valid_windows=10271,invalid_windows=3765,status='built_pending_verification'))
 print('V4.3 BUILD COMPLETE; active entry remainsv4.2.',flush=True)
if __name__=='__main__':main()
