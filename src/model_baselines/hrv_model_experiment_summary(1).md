# PPG→HRV Model Experiments

**Current reproducible summary / 当前可复现实验总结**

## 0. Executive summary

| Pipeline | Current status | Main conclusion |
|---|---|---|
| qppg heuristic | Strong reference baseline | Must be compared on the same accepted windows because its coverage is below 100%. |
| Own PeakNet | Updated 16-fold rerun pending | Post-fix P1 diagnostics show that peak localization remains the bottleneck; legacy full-run results have been removed. |
| PulsePPG frozen + nested Ridge | **Primary foundation-model baseline** | Most stable verified PulsePPG result; strongest pooled SDNN R²/r and RMSSD r. |
| PulsePPG head-only | Control ablation | Replacing Ridge with a neural head does not improve SDNN and weakens correlation. |
| PulsePPG partial fine-tuning | Fine-tuning ablation | Improves SDNN relative to head-only, but does not consistently beat Frozen Ridge and does not improve RMSSD. |
| PaPaGei frozen + Ridge | Secondary foundation baseline | Runs completed, but exact aggregate metrics still need regeneration from prediction CSVs. |

**Current model decision（当前模型选择）:** retain **Frozen PulsePPG + nested participant-grouped Ridge** as the main foundation-model baseline. Report head-only and partial fine-tuning as ablations, not as superior models.

## 1. Scope（实验范围）

Dataset: synchronized 5-minute PPG/ECG windows, 30-second stride, 16 participants, three devices (Earring, Ring, Watch), Green and IR channels. ECG targets are corrected SDNN and corrected RMSSD in milliseconds.

This document separates the following model pipelines:

1. **Own PeakNet**: PPG → dense peak probabilities → event decoding → raw/corrected IBI → SDNN/RMSSD.
2. **PulsePPG frozen**: pretrained PulsePPG encoder → 512-D window embedding → nested participant-grouped Ridge → SDNN/RMSSD.
3. **PaPaGei frozen**: pretrained PaPaGei-S encoder → segment embeddings → 5-minute aggregation → nested participant-grouped Ridge → SDNN/RMSSD.
4. **PulsePPG head-only / partial fine-tuning**: frozen or partially unfrozen PulsePPG encoder → neural regression head → SDNN/RMSSD.

The heuristic qppgfast/SciPy results are maintained by the parallel heuristic pipeline and are not recomputed here.

## 2. Evaluation definitions（指标怎么读）

| Quantity | Meaning |
|---|---|
| MAE ↓ | Mean absolute difference between predicted and corrected ECG HRV, in ms. 越低越好。 |
| Pearson `r` ↑ | Linear association between prediction and target. It measures whether predictions follow window-to-window changes; it is not “Pearson correction.” |
| R² ↑ | Improvement over predicting the test participant's mean. `R² < 0` means the squared error is worse than that mean reference，不代表代码一定有 bug。 |
| Participant mean ± std | Each held-out participant receives equal weight（每个人权重相同）. |
| Pooled metric | All held-out windows are concatenated; participants with more windows receive greater weight（每个窗口权重相同）. |
| Peak F1 | Event-level peak matching F1 within the configured temporal tolerance. |
| IBI correction ratio | Fraction of detected IBI replaced/flagged by the local-median correction rule. |
| HRV coverage | Fraction of attempted windows producing finite SDNN and RMSSD. |

MAE, R², and Pearson r were independently recomputed from saved predictions using both their explicit formulas and NumPy/scikit-learn. The values matched exactly, so the unusual negative R² and weak correlations are genuine model behavior rather than an evaluation implementation error.

## 3. Own PeakNet（自己的 peak model）

### 3.1 Pipeline

The current pipeline uses preprocessed PPG, a U-Net or literature-inspired dilated CNN, `scipy.signal.find_peaks`, physiological IBI limits, and optional local-median IBI correction. Training uses dense Gaussian ECG R-peak targets with weighted BCE + Dice loss. The checkpoint originally was selected by validation loss; a new diagnostic version selects it by validation event-level F1.

### 3.2 Current result status（当前结果状态）

The earlier three-device 16-fold table was produced before the final preprocessing, valid-pair handling, architecture, and event-F1 checkpoint-selection updates. It has therefore been removed from this summary and must not be used as the current Own PeakNet result.

At present, only post-fix P1 diagnostics are retained below. A new Earring/Green 16-fold LOSO run using the finalized configuration is pending. Until that run finishes, Own PeakNet has no formal population-level result in this report.

### 3.3 Raw versus corrected IBI（P1 diagnostics）

Across inspected folds, raw-I​​BI HRV errors were much larger than corrected-I​​BI errors. For example, in P1 Earring runs, raw RMSSD MAE was approximately 84–130 ms, while median-corrected RMSSD MAE was approximately 9–13 ms. Typical correction ratios were 0.34–0.43.

Interpretation: correction is effective, but it is compensating for a large number of detection errors. Moderate corrected MAE accompanied by low correlation and negative R² does not demonstrate reliable beat-to-beat recovery.

### 3.4 Architecture and IBI-method smoke tests（P1 Earring）

| Peak backbone | IBI method | Peak F1 | Predicted/true mean peaks | Coverage | SDNN MAE | SDNN r | RMSSD MAE | RMSSD r |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net | median | 0.381 | 696.8 / 442.2 | 100.0% | 15.53 | -0.067 | 9.36 | -0.050 |
| Dilated | median | 0.391 | 719.4 / 442.2 | 100.0% | 13.26 | 0.040 | 9.12 | 0.245 |
| U-Net | remove/reject | 0.381 | 696.8 / 442.2 | 45.4% | 19.73 | -0.017 | 9.99 | -0.062 |
| Dilated | remove/reject | 0.391 | 719.4 / 442.2 | 47.0% | 17.10 | 0.062 | 10.30 | 0.291 |

The dilated model produced a small smoke-test improvement. Removing abnormal IBI reduced coverage by more than half and did not improve MAE, so median replacement remains the current main correction method.

### 3.5 Peak-channel and checkpoint diagnostic

The completed P1 Earring Green-only run trained for 20 epochs. Validation event F1 was highest at epoch 19 (`0.297`), whereas validation loss was already best at epoch 1. The selected checkpoint produced test event F1 `0.266`, predicted/true mean peak counts `479.3/440.5`, SDNN MAE `28.06 ms` (`r=0.310`), and RMSSD MAE `15.38 ms` (`r=-0.234`). This confirms that validation loss and event-level F1 select different checkpoints, but Green-only training still did not improve the final HRV result.

### 3.6 Current bottleneck

The principal bottleneck is event decoding, not IBI correction. PeakNet often predicts about 700 events where ECG contains about 440. Even when a fixed threshold makes the total count similar, event F1 remains low. The dense sample-wise loss does not explicitly enforce one event per beat or stable adjacent intervals. ECG R-peak targets also impose device-dependent timing offsets on a PPG-fiducial task.

In short: IBI correction can hide many false/missed detections in the final MAE, but it cannot recover reliable beat-to-beat timing. Therefore PeakNet should currently be presented as an interpretable failure analysis rather than the strongest HRV model.

## 4. PulsePPG experiments（Frozen、Head-only、Partial FT）

### 4.1 Frozen Ridge protocol

Input is one PPG device/channel at 100 Hz, decimated to 50 Hz, normalized per 5-minute window, encoded into a 512-D PulsePPG representation, and evaluated with outer participant LOSO. Ridge alpha is selected by inner GroupKFold over training participants; feature and target scalers are fitted only inside each training partition.

### 4.2 Pooled results across 11,788 held-out windows

| Device/Channel | SDNN MAE | SDNN R² | SDNN r | RMSSD MAE | RMSSD R² | RMSSD r |
|---|---:|---:|---:|---:|---:|---:|
| Earring/Green | 14.269 | 0.139 | 0.418 | 11.265 | -0.185 | 0.153 |
| Ring/Green | 14.079 | 0.124 | 0.398 | 10.121 | 0.017 | 0.326 |
| Watch/Green | 14.821 | 0.026 | 0.297 | 11.203 | -0.178 | 0.090 |

LOSO training-mean baselines were SDNN MAE 15.668 ms and RMSSD MAE 10.393 ms. PulsePPG improved SDNN MAE by 0.85–1.59 ms. For RMSSD, only Ring slightly improved over the mean baseline; Earring and Watch were worse.

All 16 folds selected the maximum searched Ridge alpha (`1000`), so the original alpha grid ended at its boundary. The frozen baseline should be rerun with a wider regularization grid before being considered fully tuned.

### 4.3 PulsePPG head-only and partial fine-tuning

Two neural-head variants were evaluated on Earring/Green with complete participant-level outer evaluation:

- **Head-only**: the pretrained encoder is frozen and only a LayerNorm–Dropout–Linear regression head is trained.
- **Partial FT**: the final PulsePPG residual block and the same regression head are trained. The encoder learning rate is `1e-5`, head learning rate is `1e-3`, weight decay is `1e-4`, and early stopping uses validation mean MAE.

For most folds, P20 was held out for validation and the current participant for testing; for the P20 test fold, P19 was used for validation. Consequently, these neural-head runs use 14 training participants plus one validation participant, whereas the nested Ridge baseline can train on all 15 non-test participants after selecting alpha. This protocol difference must be retained as a reporting caveat.

#### Pooled results across 11,788 held-out windows

| PulsePPG method | SDNN MAE | SDNN R² | SDNN r | RMSSD MAE | RMSSD R² | RMSSD r |
|---|---:|---:|---:|---:|---:|---:|
| Frozen + nested Ridge | **14.269** | **0.139** | **0.418** | 11.265 | -0.185 | **0.153** |
| Frozen + neural head | 14.833 | 0.062 | 0.300 | **11.108** | **-0.163** | 0.068 |
| Partial FT + neural head | 14.478 | 0.091 | 0.351 | 11.297 | -0.195 | 0.076 |

#### Participant-mean LOSO results

| PulsePPG method | SDNN MAE | SDNN R² | SDNN r | RMSSD MAE | RMSSD R² | RMSSD r |
|---|---:|---:|---:|---:|---:|---:|
| Frozen + neural head | 14.326 ± 4.450 | -0.812 ± 1.697 | 0.338 ± 0.174 | **12.027 ± 3.862** | **-2.023 ± 2.836** | **0.157 ± 0.271** |
| Partial FT + neural head | **13.903 ± 4.207** | **-0.646 ± 1.230** | **0.339 ± 0.185** | 12.334 ± 4.050 | -2.193 ± 2.962 | 0.144 ± 0.286 |

Partial fine-tuning improves SDNN relative to the matched head-only control, but does not improve RMSSD. Relative to Frozen Ridge, it does not improve the pooled results overall. In the participant-paired comparison against Frozen Ridge, SDNN improved in 8/16 participants and worsened in 8/16 (median MAE change `+0.023 ms`); RMSSD improved in 9/16 and worsened in 7/16 (median change `-0.138 ms`). These near-zero median changes show no consistent participant-level fine-tuning benefit.

Prediction shrinkage remains substantial. For Partial FT, pooled predicted-versus-true standard deviations were `10.464/20.693 ms` for SDNN and `6.541/12.598 ms` for RMSSD. The model therefore recovers only about half of the observed HRV variation. Independent manual formula checks exactly reproduced the NumPy/scikit-learn MAE, R², and Pearson r values, ruling out metric implementation error.

## 5. PaPaGei frozen baseline（待正式汇总）

PaPaGei-S uses 125-Hz, 10-second segments; 30 segment embeddings are averaged to form each 5-minute 512-D representation. It uses the same outer LOSO and nested grouped Ridge design as PulsePPG.

Available run summaries indicate approximate fold-level MAE:

| Device/Channel | SDNN MAE | RMSSD MAE |
|---|---:|---:|
| Earring/Green | ~15.16 | ~12.19 |
| Ring/Green | ~13.75 | ~10.45 |
| Watch/Green | ~16.68 | ~11.87 |

These PaPaGei aggregate values should be regenerated from its prediction CSV before formal reporting. As with PulsePPG, every displayed fold selected alpha `1000`, indicating boundary-limited regularization search.

## 6. Cross-method interpretation

| Method | Main strength | Main limitation |
|---|---|---|
| Own PeakNet | Produces explicit peaks and supports raw/corrected IBI ablation | Updated full LOSO result pending; current evidence is limited to P1 diagnostics |
| PulsePPG frozen | Best currently verified model-level overall stability; cached embeddings make evaluation cheap | Weak RMSSD correlation and prediction shrinkage |
| PulsePPG head-only | Matched control for isolating encoder adaptation | Neural head does not improve SDNN and further reduces correlation |
| PulsePPG partial FT | Improves SDNN relative to the matched head-only control | Participant-specific effect; no overall gain over Frozen Ridge and no RMSSD improvement |
| PaPaGei frozen | Competitive Ring SDNN result | Slightly weaker overall and also boundary-limited Ridge tuning |

The heuristic baseline and model results must not be ranked by MAE alone when their evaluated window sets differ. A fair comparison requires the identical participant/window subset and simultaneous reporting of coverage.

### 6.1 What the PulsePPG comparison establishes

1. **Head-only vs Frozen Ridge:** a neural regression head alone is not better than the simpler nested Ridge baseline.
2. **Partial FT vs Head-only:** adapting the final residual block helps SDNN, so encoder adaptation has some signal.
3. **Partial FT vs Frozen Ridge:** the benefit is participant-specific and does not translate into a consistent overall improvement.
4. **RMSSD remains harder:** every PulsePPG variant substantially compresses prediction variance, suggesting that a pooled 5-minute embedding does not preserve enough beat-to-beat information.

### 6.2 Main reporting sentence

> Frozen PulsePPG with nested Ridge remains the strongest and most stable foundation-model baseline. Partial fine-tuning improves SDNN relative to a matched head-only control, but provides no consistent participant-level advantage over Frozen Ridge and does not improve RMSSD.

## 7. Next experiments（按优先级）

1. **Highest priority:** rerun finalized Own PeakNet on Earring/Green with complete 16-fold LOSO, Dilated backbone, median IBI correction, Green input, and validation event-F1 checkpoint selection.
2. **Highest priority:** regenerate exact PaPaGei pooled and participant-level metrics from its prediction CSVs.
3. Align qppg, PeakNet, PaPaGei, and PulsePPG on the identical participant/window subset and report coverage alongside MAE, R², and r.
4. **Recommended new experiment:** combine qppg HRV/quality features with frozen PulsePPG embeddings using nested Ridge, and test a residual-correction variant: `final HRV = qppg HRV + predicted error`.
5. Widen the Frozen PulsePPG/PaPaGei Ridge alpha grid beyond `1000` and verify that selection no longer stops at the grid boundary.
6. Treat PulsePPG head-only and partial fine-tuning as completed ablations; do not begin a large fine-tuning sweep unless a new hypothesis targets participant calibration or temporal aggregation.
7. Investigate temporal or beat-aware aggregation for RMSSD, since global 5-minute embeddings and current regressors substantially shrink within-participant variability.

## 8. Reporting cautions

- P1 smoke tests are diagnostic and cannot establish population-level superiority.
- Thresholds, checkpoint epochs, channel choices, and IBI gates must be chosen without using the held-out test participant.
- Overlapping 5-minute windows are not independent observations; all splitting remains participant-level.
- Coverage and QC rules must accompany every heuristic-versus-model comparison.
