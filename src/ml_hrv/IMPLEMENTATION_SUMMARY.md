# Raw continuous HRV 第一版实现总结

更新日期：2026-09-11

## 结论与当前边界

已在 `src/model_baselines` 同级建立独立的 `src/ml_hrv`。这一版已把数据契约、
模型、delay-aware beat 训练、RR 修正、participant-disjoint 评估、checkpoint、
逐窗预测和部署包装连通。全量 SegNet 四折已经完成并冻结；结果显示相关、CCC 近零且 R²
为负，而且 RMSSD/SDNN MAE 均差于每折 train-only median 常数，**不能称为 SOTA**。完整数值、冻结清单与加速实测见
[`BASELINE_RESULTS.md`](BASELINE_RESULTS.md)。

主部署仍是一台设备的 green + IR。三设备只在训练 batch 中作为三个独立样本，
用于共享参数和 matched-window consistency；评估另外计算三设备 precision fusion
upper bound，不把三设备同时输入主部署模型。

## 已固定的数据契约

| 角色 | 字段/定义 |
|---|---|
| 主波形 | `ppg_rawslot_values`，绝不换成 `ppg_resampled_values` |
| 缺失 | `ppg_rawslot_mask`；缺失值归一化后置零，但 mask 始终单独输入 |
| 时间 | `(rawslot_timestamp-grid_timestamp)/(1000/fs)`，裁剪到 ±2 slots |
| 主标签顺序 | `ecg_rmssd_corrected_ms`, `ecg_sdnn_corrected_ms` |
| 主窗口 | causal trailing 300 s = 30 × 10 s |
| 更新 | 每 30 s，一次加入 3 个 segment；5 min warm-up 前拒绝输出 |
| 一期运动量 | `accel_motion_mean_mag` 当前窗单标量 |
| 短窗辅助 | 30/60/120 s；只作为辅助结果，标签由该 trailing crop 的 ECG R 峰经相邻性保持修正后重算 |

这里把“一期加速度标量”具体解释为 `accel_motion_mean_mag`，因为
`accel_mean_mag` 主要接近重力模长（约 9.8 m/s²），对运动强度区分较弱。后续消融仍应
严格使用同一 split 比较：无 accel / motion scalar / 原始 25 Hz 三轴 accel。

当前源数据本身已带 PPG coverage 筛选；本实现不额外提高 coverage 门槛，只保留 ECG
label QC。按要求，本轮没有制造“无 PPG coverage 预筛”的 challenge set。建议未来从
更上游、未筛选 raw session 构建并冻结该集合，而不是从当前训练窗口反向拼一个伪
challenge set。

## 模型

```text
single-device raw green/IR + mask + relative timestamp jitter
        ↓
30 × 10 s mask-aware shared dilated encoder
        ├─ causal segment-token TCN → log RMSSD/SDNN mean + variance
        └─ beat heatmap + sub-slot offset + SQI + latent pulse delay
                       ↓
            adjacency-preserving RR correction
                       ↓
                 physiological HRV
        ↓
conditional uncertainty fusion
        ↓
RMSSD / SDNN / confidence / reject_reason
```

关键实现选择：

1. encoder 只对 `mask=True` 的观测做归一化，mask 与 jitter 保留为独立通道。
2. direct head 用左侧 padding 的 causal TCN 聚合 30 个 segment token，并在 log-ms
   标准化空间输出异方差均值/方差。
3. beat target 不再把 ECG R 峰直接当 PPG 峰。训练 loss 在 100–700 ms 内搜索 ECG→PPG
   正延迟，选择与预测 pulse heatmap 最匹配的 delay，再监督 heatmap、offset 和 delay。
4. RR 修正优先合并疑似 extra beat、拆分疑似 missed beat；无法修正的 interval 变成
   NaN separator。RMSSD 只对 separator 同侧的真正相邻 RR 求差，杜绝旧实现先过滤 RR
   后把非相邻 interval 拼接的问题。
5. fusion 默认关闭。只有 held-out direct 与 beat 的错误不高度同向，且融合相对最佳
   单路径至少降低 0.5 ms MAE 时才生成 approval；否则生产输出保持 direct。

## 伙伴 SegNet 的可信复现

`configs/segnet_rawslot_baseline.json` 保留伙伴 SegNet 的“共享 10 s CNN token + 30 段
mean aggregation”核心，但修复为：

- rawslot green/IR + mask，不读取 resampled PPG；
- RMSSD、SDNN 顺序统一且来自 corrected ECG fields；
- participant-disjoint folds；
- participant-balanced、matched-device batches；
- log-ms scaler 只拟合 train participants；
- checkpoint 带 config、fold、scaler、target order 和完整 history；
- test 每一个 window/device 都保存预测、confidence、coverage、motion 和 reject reason。

这应先于 causal TCN 和 beat joint 模型运行，作为新的 direct baseline。

## 训练与评估产物

每 fold 写入：

```text
checkpoints/fold_N_<stage>.pt
fold_N/window_predictions.csv
fold_N/metrics.json
fold_N/run_metadata.json
fold_N/summary.md
combined/window_predictions.csv
combined/metrics.json
combined/summary.md
```

`metrics.json` 包含 micro 和 participant-macro MAE/RMSE/bias/Pearson r/R²/CCC、Bland–Altman
limits、participant bootstrap 95% CI、per-device、high/middle/low motion、coverage strata、
device × coverage matched comparison、risk–coverage、top-10%-error 高置信率，以及三设备
融合 upper bound。主要论文表应使用 participant-macro 和 participant bootstrap CI，
不能把 30 s stride 的重叠窗口当独立样本来做普通 window bootstrap。

## 推荐运行顺序

从仓库根目录运行；cache 是 lossless/mmap，不做波形插值。全数据 cache 预计约 15 GB。

```bash
PYTHONPATH=src python -m ml_hrv.scripts.validate_data \
  src/heuristic_baselines/outputs/synced_3device_rawaligned_training_v1_stride30_rawslots

PYTHONPATH=src python -m ml_hrv.scripts.prepare_cache \
  src/heuristic_baselines/outputs/synced_3device_rawaligned_training_v1_stride30_rawslots \
  artifacts/ml_hrv/rawslot_cache

# Existing cache: backfill per-window statistics for main and beat-pretrain crops
PYTHONPATH=src python -m ml_hrv.scripts.prepare_stats \
  artifacts/ml_hrv/rawslot_cache --seconds 300 60

# 1. rawslot + mask SegNet reproduction
PYTHONPATH=src python -m ml_hrv.scripts.train \
  --config src/ml_hrv/configs/segnet_rawslot_baseline.json \
  --output-dir artifacts/ml_hrv/segnet_rawslot

# 2. mask + jitter + scalar accel causal direct model
PYTHONPATH=src python -m ml_hrv.scripts.train \
  --config src/ml_hrv/configs/phase1.json \
  --output-dir artifacts/ml_hrv/phase1_direct

# 3. ECG-event-only 60 s beat pretraining; no outer-test decode
PYTHONPATH=src python -m ml_hrv.scripts.train \
  --config src/ml_hrv/configs/beat_pretrain_60s.json \
  --short-window-seconds 60 \
  --init-checkpoint 'artifacts/ml_hrv/phase1_direct/checkpoints/{fold}_direct.pt' \
  --freeze-encoder-epochs 1 \
  --output-dir artifacts/ml_hrv/beat_pretrain_60s

# 4. 5 min delay-aware beat + direct joint fine-tuning; fusion stays disabled
PYTHONPATH=src python -m ml_hrv.scripts.train \
  --config src/ml_hrv/configs/joint_beats.json \
  --init-checkpoint 'artifacts/ml_hrv/beat_pretrain_60s/checkpoints/{fold}_beat_pretrain.pt' \
  --output-dir artifacts/ml_hrv/joint_beats

# 5. same fold protocol, 30-second low-latency auxiliary result
PYTHONPATH=src python -m ml_hrv.scripts.train \
  --config src/ml_hrv/configs/phase1.json \
  --short-window-seconds 30 \
  --output-dir artifacts/ml_hrv/aux_30s
```

快速验证：

```bash
PYTHONPATH=src python -m unittest discover -s tests/ml_hrv -v
PYTHONPATH=src python -m ml_hrv.scripts.smoke_test
```

开发期单折短跑必须加 `--skip-test-evaluation`；该模式不会构造 outer-test dataset，也不会
生成 outer-test 预测。正式冻结候选才移除该参数。

## 仍需按顺序完成的实验

1. ~~跑完 SegNet 四折并冻结结果，确认字段、scaler、checkpoint 和逐窗 CSV。~~ 已完成。
2. 跑 causal direct，并只用 inner validation 决定容量、dropout 和 confidence threshold。
3. 先用 30–60 s ECG event crop 预训练 beat head，再跑 5 min joint beat；正式补充 beat
   event F1/timing MAE/count MAE 后才判断互补性。
4. 只有互补性门槛通过才启用 direct–beat fusion；三设备 fusion 始终只报 upper bound。
5. 复用完全相同 folds 做无 accel / scalar / raw 25 Hz XYZ 消融；当前 rawslot NPZ 没有
   XYZ 序列，第三项需从上游 raw data 按 timestamp 对齐，不能伪造上采样。
6. 最后冻结从未用于调参的 participant test 或外部数据集，并加入未做 PPG coverage
   预筛的真实 challenge set。

## 文献依据与可比性边界

- Kazemi et al., “Robust PPG Peak Detection Using Dilated Convolutional Neural Networks,”
  *Sensors* 22(16), 6054 (2022), DOI [10.3390/s22166054](https://doi.org/10.3390/s22166054)。
  支持用 dilated CNN 做噪声下 beat detection，但该文把 20 Hz PPG 线性上采样到 100 Hz；
  本实现不能照搬该输入处理，因此保留 rawslot mask/timestamp。
- Sarhaddi et al., “A comprehensive accuracy assessment of Samsung smartwatch heart rate and
  heart rate variability,” *PLOS ONE* 17(12), e0268361 (2022),
  DOI [10.1371/journal.pone.0268361](https://doi.org/10.1371/journal.pone.0268361)。
  其 24 h free-living 结果显示 awake/activity 条件的多数 HRV 指标误差明显高于 sleep，
  因此 motion-stratified 结果不能省略。
- O’Grady et al., “The Validity of Apple Watch Series 9 and Ultra 2 for Serial Measurements of
  Heart Rate Variability and Resting Heart Rate,” *Sensors* 24(19), 6220 (2024),
  DOI [10.3390/s24196220](https://doi.org/10.3390/s24196220)。其评价的是 Apple Watch
  导出的 SDNN，不能当成本项目 RMSSD benchmark。
- Flatt et al., “Biostrap Kairos Wristband Versus Electrocardiography for Resting Heart Rate
  Variability Assessment,” *Sensors* 25(10), 3165 (2025),
  DOI [10.3390/s25103165](https://doi.org/10.3390/s25103165)。该文强调随 HRV 增大而出现
  proportional bias/heteroscedasticity，支持 log target、uncertainty 和 Bland–Altman。
- Sinichi et al., “Right Place, Right Time: Validation of a Consumer-Grade Wearable for Heart
  Rate and Heart Rate Variability Across Sleep-Wake Cycles, Physical Activity, and Postures in
  an Ambulatory Study” (PsyArXiv preprint, 2026),
  DOI [10.31234/osf.io/qc2b3_v1](https://doi.org/10.31234/osf.io/qc2b3_v1)。支持把
  “何时可信”本身作为输出与评估任务；该文是 preprint，证据等级需单独注明。
- Tao et al., “Method comparison of a ring-based PPG device and a chest-strap monitor for heart
  rate and heart rate variability in shooting athletes,” *BMC Sports Science, Medicine and
  Rehabilitation* (2026), DOI
  [10.1186/s13102-026-01960-x](https://doi.org/10.1186/s13102-026-01960-x)。其 90 s、
  seated/low-motion、participant-mean 结果不能与本项目 free-living 5 min window-level 主结果
  直接横比；它适合作为短窗辅助结果的受控场景参照。

因此，对外措辞应是“single-device ECG-referenced continuous HRV estimation candidate”。
只有在上述严格协议中同时报告误差、相关/一致性、coverage、tail risk 和 calibration，且
超过冻结 baseline 与可比公开方法后，才升级为 SOTA claim。
