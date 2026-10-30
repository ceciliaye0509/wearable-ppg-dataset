"""Seal v4.3 plus verified E4 scores and advance the unique project state."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,csv
import numpy as np
import pandas as pd
from build_release import ROOT,HERE,QA,FOLLOW,PARENT,DEST,NAME,sha,dump
CP=ROOT/'analysis/project_checkpoint_20261029_v2';ENTRY=ROOT/'analysis/current_ecg_training_dataset.json';POINTER=ROOT/'analysis/current_project_checkpoint.json';MODEL=ROOT/'analysis/current_model_plan.json';REPORT=CP/'唯一当前状态与下一步_2026-10-29.md'
def atomic_json(path,value):
 temp=path.with_name(path.name+'.release_tmp');dump(temp,value);temp.replace(path)
def main():
 assert not (HERE/'activation_receipt.json').exists(),'Already activated'
 v=json.loads((HERE/'verification.json').read_text());b=json.loads((HERE/'baseline/verification.json').read_text());iv=json.loads((HERE/'baseline/independent_verification.json').read_text());research=json.loads((QA/'joint_decision.json').read_text())
 assert v['valid_ecg_label_windows']==10271 and b['selection']['recommended_primary']=='qppgfast_devicewise_corr' and iv['numeric_error_and_correlation_checks']==48
 entry=json.loads(ENTRY.read_text());pointer=json.loads(POINTER.read_text());assert entry['dataset_dir']==str(PARENT)
 oldmanifest=Path(pointer['project_manifest']);assert sha(oldmanifest)==pointer['project_manifest_sha256']
 for p,h in json.loads(oldmanifest.read_text())['file_sha256'].items():assert sha(p)==h,p
 now=datetime.now(timezone.utc).isoformat();hist=CP/'history/before_v4_3_export12_r7';assert not hist.exists();hist.mkdir(parents=True);snapshots={}
 for p in [ENTRY,POINTER,MODEL,REPORT,CP/'current_state.json',CP/'action_plan.json',CP/'下一步清单.csv',oldmanifest]:
  q=hist/p.name;shutil.copy2(p,q);snapshots[str(p)]=dict(path=str(q),sha256=sha(q))
 # Complete all stable dataset evidence before computing its final checksums.
 shutil.copy2(HERE/'baseline/independent_verification.json',DEST/'baseline_independent_verification.json')
 freeze=DEST/'dataset_freeze_manifest.json';assert not freeze.exists()
 metadata={str(p.relative_to(DEST)):sha(p) for p in DEST.rglob('*') if p.is_file() and not (p.parent==DEST and p.suffix=='.npz')}
 f=dict(freeze_id='daily_hrv_v4_3_export12_fullmanual_frozen_20261029',created_utc=now,status='frozen_verified_local_E4_complete',dataset_name=NAME,dataset_version='v4.3',protocol_sha256=v['protocol_sha256'],annotation_snapshot_sha256=v['manual_snapshot_sha256'],candidate_windows=14036,valid_ecg_label_windows=10271,invalid_ecg_label_windows=3765,participant_files=15,participants_with_valid_labels=14,parent_dataset=str(PARENT),parent_protocol_sha256=entry['protocol_sha256'],parent_freeze_manifest=entry['project_freeze_manifest'],parent_freeze_manifest_sha256=entry['project_freeze_manifest_sha256'],NPZ_sha256=v['NPZ_sha256'],metadata_sha256=metadata,baseline_primary=b['selection']['recommended_primary'],baseline_validation='baseline_verification.json and baseline_independent_verification.json',local_manual_full300_exceptions=4,original97_repairs_and7_rejections_retained=True,clinical_NN_or_physical_switch_truth_claimed=False,HF_uploaded=False,model_training_started=False)
 dump(freeze,f);dump(DEST/'checksums.json',{**v['NPZ_sha256'],**metadata,freeze.name:sha(freeze)})
 scores=pd.read_csv(HERE/'baseline/current_primary_scores.csv');eligible=scores[scores.scope.eq('all__eligible_v4_3_GT')].iloc[0];allscore=scores[scores.scope.eq('all__all_candidates_v4_3_GT')].iloc[0]
 scorefile=CP/'frozen_primary_scores_v4_3.csv';shutil.copy2(HERE/'baseline/current_primary_scores.csv',scorefile)
 scorevalues=dict(primary='qppgfast_devicewise_corr',dataset_version='v4.3',RMSSD_MAE_ms=float(eligible.RMSSD_MAE_ms),RMSSD_r=float(eligible.RMSSD_r),SDNN_MAE_ms=float(eligible.SDNN_MAE_ms),SDNN_r=float(eligible.SDNN_r),coverage_eligible_pct=float(eligible.coverage_pct),coverage_all_candidates_pct=float(allscore.coverage_pct),scores_file=str(scorefile),model_training_scores=False,reference_definition='fixed_corrected_NN_proxy',raw_RR_secondary=str(HERE/'baseline/six_channel_macro.csv'))
 dump(QA/'verification.json',dict(status='export12_human_review_and_joint_decision_verified',source_csv_sha256=research['source_export_sha256'],complete_clips=394,new_completed_clips=30,original364_source_clips_unchanged=True,explicit_recheck_reference_supersession=2,P3_latest_peaks=18,P3_short_intervals_ms=[223.,292.,192.5],P3_latest_review_unreadable_intervals=0,P3_parent_effective_peaks=559,old_unreadable_tail_outside_crop_retained=True,P9_recheck_peaks=9,P9_short_intervals_under300=0,random_audit_participants=14,random_audit_clips=28,random_audit_seconds=560,random_audit_human_peaks=754,FN20=0,FP20=0,global_zero_error_claimed=False,all_five_current_research_dispositions_recorded=True,old_source_csv_and_web_database_modified_by_agent=False,integrated_release='v4.3'))
 action=json.loads((CP/'action_plan.json').read_text());tasks={t['id']:t for t in action['tasks']}
 tasks['F0']['action']='保留v4.1/v4.2版本链；当前正式本地数据与baseline冻结为v4.3'
 tasks['R7'].update(status='done_with_training_eligibility_scope_limit',action='已纳入最新P3/P9复核；保留300–2000ms作主标签保守纳入条件，不按阈值删除人工峰，不宣称范围普适最优',evidence=str(QA/'short_RR_latest_review.json'),result=research['R7'])
 tasks['R3'].update(status='done_initial28_probability_audit_with_uncertainty',action='28随机片段/14人/560秒/754峰：±20ms零漏零多；已报告抽样权重和不确定性，不外推整窗或全库零错误',evidence=str(QA/'random_audit_summary.json'),result=research['R3'])
 tasks['R5'].update(status='done_four_full300_exceptions_integrated_v4_3',action='保留全局5000µV；4个完整人工窗单窗例外已整合v4.3，未恢复相邻窗',evidence=str(HERE/'verification.json'))
 tasks['R6'].update(status='done_primary_and_secondary_scored',action='固定校正NN代理为主，原始RR为必报次级参考；两套定义已在同一窗口重评，不把统计校正说成错峰修复',evidence=str(HERE/'baseline/six_channel_macro.csv'))
 tasks['ECG_DECISION'].update(status='done',action='五项当前研究已按证据范围结案，最新参考和4窗人工例外已联合冻结为v4.3',evidence=str(QA/'joint_decision.json'))
 tasks['E3R']=dict(id='E3R',status='done',priority='current_required',depends_on=['ECG_DECISION'],action='v4.3完整包封装并通过15文件schema、10,271窗HRV及330项非ECG字节核验',evidence=str(HERE/'verification.json'),output=str(DEST))
 tasks['E4'].update(status='done',depends_on=['E3R'],action='同一PPG输出重评9配置和v4.1/v4.2/v4.3；无替代通过原验收，保留qppgfast_devicewise_corr',evidence=str(HERE/'baseline/verification.json'),scores=str(scorefile))
 tasks['M1'].update(status='ready_configuration_and_cache_rebuild',depends_on=['E4'],action='按v4.3配置并重建缓存/scaler，完成同事固定四折、Earring绿光和50Hz处理对齐；小样本核验后进入M2',target_dataset=str(DEST),target_protocol_sha256=v['protocol_sha256'])
 tasks['W1'].update(status='all30_current_reviews_complete',action='本轮2个短RR及28个随机片段均已Complete；394条原始标注已纳入export12快照，无新增人工任务',evidence=str(QA/'verification.json'))
 order=[t['id'] for t in action['tasks']];order.insert(order.index('E4'),'E3R');action.update(revision='v4_3_export12_ECG_E4_complete_r7',updated_utc=now,priority='Current ECG closeout and E4 complete; next M1 configuration/cache, with BIDMC protocol preparation independently available',tasks=[tasks[k] for k in order]);dump(CP/'action_plan.json',action)
 with (CP/'下一步清单.csv').open('w',encoding='utf-8-sig',newline='') as fh:
  writer=csv.writer(fh);writer.writerow(['ID','状态','优先级','依赖','动作','验收/结束条件'])
  for t in action['tasks']:writer.writerow([t['id'],t['status'],t.get('priority',''),','.join(t['depends_on']),t['action'],t.get('acceptance','')])
 state=json.loads((CP/'current_state.json').read_text());priorheur=state['heuristic'];state.update(status='v4_3_frozen_ECG_and_E4_complete_M1_ready',as_of_utc=now,full_training_container=dict(version='v4.3',dataset_name=NAME,dataset_dir=str(DEST),protocol_sha256=v['protocol_sha256'],candidate_windows=14036,valid_windows=10271,invalid_windows=3765,participant_files=15,participants_with_valid_labels=14,immutable_dataset_freeze_manifest=str(freeze),dataset_freeze_sha256=sha(freeze),checksums=str(DEST/'checksums.json'),verification=str(HERE/'verification.json')),heuristic=scorevalues,current_research_questions=[{k:t[k] for k in ['id','status','action','evidence']} for t in action['tasks'] if t['id'] in ['R7','R5','R6','R3','R4']],research_dispositions=str(QA/'joint_decision.json'),latest_export_verification=str(QA/'verification.json'),next_action='M1: bind v4.3, finish fixed-colleague-fold/Earring-green50Hz configuration and rebuild cache/scalers. No training started.',E4_status='done',current_human_work=dict(status='all30_complete',new_manual_tasks_needed=False,complete_total=394,source_export=str(DEST/'manual_annotations_export12.csv')),manual_G1_exception_candidates=dict(count=4,status='integrated_in_v4.3',verification=str(HERE/'verification.json')))
 state['accepted_refinement_sidecar'].update(packaged_in_full_training_container=True,packaged_dataset_version='v4.3')
 state['annotation_reference'].update(source_csv=str(DEST/'manual_annotations_export12.csv'),sha256=research['source_export_sha256'],clips_available=394,full5min_available=32,manual_interval_records=137,raw_export_interval_records=138,effective_interval_records=137,new_manual_annotations_needed_now=False,website_database_changed=False,live_website_total_clips=394,current_tasks=0,frozen_annotation_clips=394)
 state['model_line'].update(priority='M1_ready_after_completed_ECG_E4',target_dataset_version='v4.3',target_protocol_sha256=v['protocol_sha256'],cache_rebuild_required=True,training_started=False)
 state['not_started']=['M1 final configuration and v4.3 cache rebuild','formal model retraining','harmonized BIDMC experiment','HF publication'];dump(CP/'current_state.json',state)
 model=json.loads(MODEL.read_text());model.update(priority='M1: bind v4.3 and build colleague-matched cache/configuration, then M2; ECG/E4 complete, model training not started',target_dataset=str(DEST),target_protocol_sha256=v['protocol_sha256']);dump(MODEL,model)
 detail=f'''# 导出12复核、v4.3与baseline结果

## P3更新已采用

P3短RR复核为18个峰、Complete/high、复核片段内无不可读区间。223、292、192.5ms三个间隔全部保留在原始人工参考；旧父片段的有效峰由558变为559，重新查看范围内的旧不可读状态已被覆盖。范围外仍有一段旧不可读尾部，未擅自清除。P9删除了造成246ms短间隔的一个标记，最新片段无小于300ms间隔。

两处原始标注行/历史文件均保留，以有效参考覆盖关系表达最新判断。你重标的是短RR P3；极性方向页面的另一个P3仍是原来的unreadable记录，两个任务没有混用。

## 随机审计

28段、14人、560秒、754个人工峰，在±20ms标准下TP754、漏0、多0；±10/150ms结果亦已保存。加权错误率点估计为0，不能证明全库零错误。每层仅一个随机片段，未报告虚假的零宽度bootstrap区间；保守的20秒片段含错率95%界为0–32.019%，这不是峰级错误率。现有合格训练窗未出现需要按这批标注修复的分歧。

## 当前采用决定

- 300–2000ms保留为完整300秒校正NN代理标签的保守纳入条件，不用它删人工峰或断言范围外峰为假。P3已证明人工测量间隔可落在范围外；250ms仍不能纳入192.5/223ms，且会额外放行233个未充分验证窗口，所以本轮不全局放宽。不宣称该范围普适最优。
- 全局5000µV保持。仅4个完整人工窗作为例外进入新版本：P4索引563、P9索引723、P12索引685、P18索引48；不扩展到相邻窗。
- 主参考是明确命名的固定校正NN代理；原始RR是必报次级结果，不声称已经人工确认临床NN。
- 极性依用户决定按现有波形证据结案，保留未验证真实物理切换的限制。

## v4.3

14,036候选、10,271合格、3,765不合格，15文件、14人有合格标签。较v4.2新增4个完整人工例外，无新增拒绝；原97个局部修复和7个拒绝窗全部保留。2,717负向分钟、34翻转区间不变。394条原始标注、138条原始不可读记录与137条有效区间分别保存；新旧G2窗口布尔掩码相同。

全部10,271窗HRV独立复算、15文件模型schema和330项非ECG压缩记录逐字节核验通过。v4.2及所有原数据/人工记录保留。

## E4结果

使用相同842,160行PPG预测，重评原9配置；原v4.1的324行通道指标独立复现；当前主方法两套参考的48个误差/相关系数另行独立复算。所有替代配置未通过原train验收，保留qppgfast_devicewise_corr。以下为六通道等权平均，均为heuristic相对ECG参考，不是模型重训成绩。

| 版本 | 合格ECG窗 | RMSSD MAE / r | SDNN MAE / r | 合格参考coverage | 全候选coverage |
|---|---:|---|---|---:|---:|
| v4.1 | 10,274 | 41.431389 / 0.263846 | 45.078363 / 0.459436 | 80.771851% | 59.122970% |
| v4.2 | 10,267 | 41.434195 / 0.263884 | 45.072590 / 0.459610 | 80.761988% | 59.075473% |
| v4.3 | 10,271 | {scorevalues['RMSSD_MAE_ms']:.6f} / {scorevalues['RMSSD_r']:.6f} | {scorevalues['SDNN_MAE_ms']:.6f} / {scorevalues['SDNN_r']:.6f} | {scorevalues['coverage_eligible_pct']:.6f}% | {scorevalues['coverage_all_candidates_pct']:.6f}% |

MAE单位ms。v4.3与v4.2共有的10,267窗中，标签、PPG输出和全部同窗指标一致；汇总变化来自新增4窗。原始RR次级参考同窗RMSSD MAE39.781643ms、r0.263974，SDNN MAE44.294231ms、r0.457109；它使用另一参考定义，不能据更低MAE反选ECG真值。

下一步M1：按v4.3绑定标签，完成同事固定四折/Earring绿光/50Hz处理对齐，重建缓存和训练集scaler，再进行M2。BIDMC统一协议可并行。模型尚未训练，HF未上传。
'''
 (QA/'导出12与v4.3结果.md').write_text(detail)
 REPORT.write_text(f'''# 当前进度与下一步

**本轮ECG研究、v4.3封装和E4 baseline复算已完成。下一步是M1：按v4.3完成同事对照配置与新缓存。**

## 当前版本与成绩

正式本地版本为v4.3：14,036候选、10,271合格、3,765不合格；15文件、14人有合格标签。保留97个局部人工修复窗和7个拒绝窗，新增4个完整人工G1例外。极性仍为2,717负向分钟／34翻转区间。

主heuristic继续用qppgfast_devicewise_corr：Earring用peak，Ring/Watch用foot，开启IBI correction。九个配置重评后没有替代通过原验收。六通道等权平均：

| 指标 | v4.3 |
|---|---:|
| RMSSD MAE / r | {scorevalues['RMSSD_MAE_ms']:.6f}ms / {scorevalues['RMSSD_r']:.6f} |
| SDNN MAE / r | {scorevalues['SDNN_MAE_ms']:.6f}ms / {scorevalues['SDNN_r']:.6f} |
| Coverage／10,271合格参考 | {scorevalues['coverage_eligible_pct']:.6f}% |
| Coverage／14,036全部候选 | {scorevalues['coverage_all_candidates_pct']:.6f}% |

这是heuristic成绩，模型尚未重训。旧版本、原数据和人工历史保留；15文件schema、10,271窗HRV独立复算、330项非ECG记录字节核验均通过。

## 这次人工复核如何采用

- **P3已更新。** 最新短RR片段18峰、无不可读区间；223/292/192.5ms间隔全部保留在人工原始参考。有效父片段558→559峰；重看范围内旧不可读标记已覆盖，范围外旧尾部仍保留。未把它和极性页面的另一个P3混用。
- **P9已更新。** 删除了造成246ms短RR的标记，最新复核片段无小于300ms间隔。
- **随机审计完成。** 28段、14人、560秒、754峰，±20ms零漏零多；不能外推全库零错误，报告保留抽样不确定性。当前无需新增人工任务。

## 五项研究的当前结论

| 项目 | 采用决定与限制 |
|---|---|
| RR范围 | 300–2000ms保留作完整主标签保守纳入条件；不按它删人工峰，不宣称普适最优 |
| 幅度门槛 | 保留全局5000µV；仅4个完整人工窗例外已进入v4.3，不恢复相邻窗 |
| 原始/校正目标 | 校正NN代理为主，原始RR同窗必报次级；不把统计校正称为错峰修复或临床NN确认 |
| 剩余峰错误率 | 本轮分层随机审计点估计零错误，样本有限；不声称整窗/全库人工认证 |
| 极性切换 | 依用户决定按现有波形结案，保留未验证真实物理切换的限制 |

## 接下来

1. **M1：配置和新缓存。** 绑定v4.3，完成同事固定四折、Earring绿光及50Hz处理对齐，重建缓存与训练集scaler，先核验小样本。
2. **M2：模型。** 复跑原模型，再做少量固定优化；与同事峰检测任务的差别单列。
3. **BIDMC协议可并行。** 统一ECG参考、单位、上下文、窗口、HRV、QC及coverage分母后，再做跨数据集比较。

旧v4.1缓存数组已清理，不能复用旧标签/scaler。模型训练和HF上传尚未开始。时间戳及共享来源问题仍维持排除；中段恢复/拼接/部分HRV另立版本研究。

'''+ '文件：[导出12与版本结果]('+str(QA/'导出12与v4.3结果.md')+') · [当前数据入口]('+str(ENTRY)+') · [冻结成绩]('+str(scorefile)+') · [完整包验证]('+str(HERE/'verification.json')+') · [机器清单]('+str(CP/'action_plan.json')+')\n')
 manifestpath=CP/'project_manifest_v4_3_export12_r7.json'
 tracked=[REPORT,CP/'current_state.json',CP/'action_plan.json',CP/'下一步清单.csv',scorefile,MODEL,freeze,DEST/'checksums.json',HERE/'verification.json',HERE/'baseline/verification.json',HERE/'baseline/independent_verification.json',HERE/'baseline/primary_version_comparison.csv',*hist.iterdir(),*[p for p in QA.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('-shm')]]
 tracked += [p for p in HERE.glob('*.py')]
 manifest=dict(checkpoint_id='daily_hrv_v4_3_export12_ECG_E4_complete_r7',created_utc=now,dataset_version='v4.3',dataset_name=NAME,dataset_protocol_sha256=v['protocol_sha256'],dataset_freeze_manifest=str(freeze),dataset_freeze_sha256=sha(freeze),parent_project_manifest=str(oldmanifest),parent_project_manifest_sha256=sha(oldmanifest),prior_file_snapshots=snapshots,file_sha256={str(p):sha(p) for p in tracked},score_file=str(scorefile),model_training_started=False,HF_uploaded=False);dump(manifestpath,manifest)
 newentry=dict(entry);newentry.update(dataset_version='v4.3',dataset_name=NAME,dataset_dir=str(DEST),status='frozen_verified_local_ECG_E4_complete_M1_ready',verification_file=str(HERE/'verification.json'),protocol_sha256=v['protocol_sha256'],manual_annotation_snapshot_sha256=v['manual_snapshot_sha256'],valid_ecg_label_windows=10271,invalid_ecg_label_windows=3765,parent_dataset=str(PARENT),baseline_status='verified_frozen_v4_3',baseline_summary_file=str(DEST/'baseline_primary_scores.csv'),baseline_detail_file=str(DEST/'baseline_six_channel_macro.csv'),previous_dataset_baseline_summary_file=str(HERE/'baseline/primary_version_comparison.csv'),previous_baseline_dataset_version='v4.2',project_freeze_id=f['freeze_id'],project_freeze_manifest=str(freeze),project_freeze_manifest_sha256=sha(freeze),project_checkpoint_manifest=str(manifestpath),project_checkpoint_manifest_sha256=sha(manifestpath),manual_full300_G1_exception_windows=4,new_rejected_windows=0,parent_v4_2_manual_rejections_retained=7,new_manual_rejected_windows=0,label_cache_rebuild_required=True);newentry['baseline_primary']=scorevalues['primary']
 newpointer=dict(pointer);newpointer.update(status=state['status'],checkpoint_id=manifest['checkpoint_id'],dataset_version='v4.3',scores=str(scorefile),scores_status='verified_frozen_v4_3',freeze_manifest=str(freeze),project_manifest=str(manifestpath),project_manifest_sha256=sha(manifestpath),parent_project_manifest=str(oldmanifest),previous_dataset_scores=str(HERE/'baseline/primary_version_comparison.csv'),previous_scores_dataset_version='v4.2',accepted_refinement_packaged_dataset='v4.3')
 for p,h in manifest['file_sha256'].items():assert sha(p)==h,p
 atomic_json(ENTRY,newentry);atomic_json(POINTER,newpointer)
 receipt=dict(status='v4_3_verified_frozen_and_activated_E4_complete',dataset_name=NAME,dataset_dir=str(DEST),protocol_sha256=v['protocol_sha256'],dataset_checksums_sha256=sha(DEST/'checksums.json'),project_manifest=str(manifestpath),project_manifest_sha256=sha(manifestpath),entry_after_sha256=sha(ENTRY),pointer_after_sha256=sha(POINTER),scores=scorevalues,old_versions_preserved=True,model_training_started=False,HF_uploaded=False);dump(HERE/'activation_receipt.json',receipt)
 dump(HERE/'artifact_checksums.json',{str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'dataset' not in p.relative_to(HERE).parts and p.name!='artifact_checksums.json'})
 print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
