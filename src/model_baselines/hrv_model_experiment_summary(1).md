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

## 5. PaPaGei frozen baseline（正式 16-fold LOSO）

PaPaGei-S (commit `0c537dad4d2850e15b724260de820dd68d77f0b0`) resamples PPG from 100 Hz to 125 Hz and applies pyPPG filtering at 0.5–12 Hz (order 4). Each 5-minute window is split into 30 ten-second segments, and their 512-D embeddings are mean-pooled. Evaluation uses outer participant LOSO, five-fold inner GroupKFold for Ridge alpha selection, and training-only feature/target scaling. Each device contains 11,788 held-out windows.

### 5.1 Pooled results across 11,788 held-out windows

| Device/Channel | SDNN MAE | SDNN R² | SDNN r | RMSSD MAE | RMSSD R² | RMSSD r |
|---|---:|---:|---:|---:|---:|---:|
| Earring/Green | 15.04 | -0.004 | 0.215 | 11.29 | -0.203 | 0.053 |
| Ring/Green | **14.82** | **0.029** | **0.256** | **9.84** | **0.064** | **0.325** |
| Watch/Green | 17.36 | -0.269 | -0.229 | 10.83 | -0.152 | -0.141 |

### 5.2 Participant-mean LOSO results

| Device/Channel | SDNN MAE | SDNN R² | SDNN r | RMSSD MAE | RMSSD R² | RMSSD r |
|---|---:|---:|---:|---:|---:|---:|
| Earring/Green | 15.16 ± 4.62 | -0.860 ± 1.274 | 0.235 ± 0.209 | 12.19 ± 3.72 | -2.207 ± 2.864 | 0.228 ± 0.313 |
| Ring/Green | **13.75 ± 3.98** | **-0.570 ± 0.817** | **0.202 ± 0.190** | **10.45 ± 3.34** | **-1.462 ± 2.399** | **0.166 ± 0.297** |
| Watch/Green | 16.68 ± 5.55 | -1.300 ± 1.766 | 0.081 ± 0.227 | 11.87 ± 3.98 | -2.099 ± 3.270 | 0.027 ± 0.136 |

### 5.3 Interpretation

Ring/Green is PaPaGei’s strongest configuration.

Compared with Frozen PulsePPG:

- PaPaGei is weaker for SDNN on all three devices.
- For Earring RMSSD, the two methods have almost identical MAE, but PulsePPG has better R² and Pearson r.
- For Ring RMSSD, PaPaGei achieves lower MAE and higher R² (`MAE=9.84 ms`, `R²=0.064`), while its correlation is nearly identical to PulsePPG (`0.325` versus `0.326`).
- For Watch RMSSD, PaPaGei has slightly lower MAE, but its pooled correlation is negative (`r=-0.141`). Therefore, it should not be described as reliably tracking window-to-window HRV variation.

The Watch result also shows why pooled and participant-mean metrics should both be reported: its fold-mean correlation is slightly positive, while its pooled correlation is negative. This may be caused by participant-level calibration differences and unequal participant sample sizes; it is not by itself evidence of an evaluation bug.

The alpha search ended at the maximum tested value (`1000`), indicating that Ridge regularization may still be boundary-limited. A wider alpha grid should therefore be tested before treating PaPaGei as fully tuned.

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


## 7. Literature basis（文献与方法依据）

Among the five supplied papers, Kazemi et al. (2022) is the only study that directly proposes an open-source machine-learning method for PPG peak detection. The other papers primarily validate wearable-derived HRV against ECG and therefore support our HRV extraction and evaluation design rather than our trainable models.

### Open-source Kazemi implementation

Code: https://github.com/HealthSciTech/Robust_PPG_PD

The official implementation uses 100-Hz, 15-second inputs and seven Conv1D layers with kernel size 3. Dilation rates are `1, 2, 4, 8, 16, 32, 64`, hidden activations are ELU, and the final layer outputs sigmoid peak probabilities. Training uses Adam and ordinary binary cross-entropy. Its peak finder applies thresholding, local-maximum detection, and a 0.35-second minimum-distance rule that retains the stronger of two nearby candidates.

The repository provides TensorFlow research code and data-processing examples but not a directly reusable PyTorch package or clearly packaged pretrained checkpoint. We should therefore reproduce the architecture in PyTorch rather than copy the script directly.

### Methodological Basis and Study-Specific Adaptations

Our experiments distinguish between three levels of methodological support:

1. **Literature-derived implementations** reproduce the principal architecture
   and training choices reported in a published method.
2. **Literature-supported adaptations** apply established methods to our
   five-minute PPG-to-HRV setting but require dataset-specific implementation
   choices.
3. **Study-specific ablations** are evaluated experimentally and are not
   presented as published methods.

| Component | Methodological basis | Status in this study |
|---|---|---|
| Kazemi-style dilated CNN, ELU activations, binary peak labels, and ordinary BCE | Kazemi et al. proposed a dilated 1D CNN for noise-robust PPG peak detection | Closest literature-derived reproduction |
| U-Net/dilated peak segmentation | Supported by supervised PPG peak-detection work, including Kazemi et al. and TAU | Literature-supported model family |
| Peak detection → IBI sequence → SDNN/RMSSD | Standard PPG-derived PRV workflow used throughout the HRV literature | Literature-supported physiological pipeline |
| Physiological RR limits and local-median artifact detection | HRV preprocessing studies commonly detect abnormal intervals using physiological limits and deviation from a local median | Literature-supported adaptation; exact thresholds are dataset-specific |
| Weighted BCE + Dice loss | Weighted cross-entropy and Dice-based objectives are established approaches for highly imbalanced segmentation targets | Literature-supported adaptation, not part of the Kazemi reproduction |
| Gaussian peak labels | Soft or distance-based peak targets are motivated by temporal uncertainty in event localization, including distance-transform supervision in TAU | Study-specific implementation inspired by prior work |
| `scipy.signal.find_peaks` decoding | Converts dense model probabilities into discrete pulse events using minimum-distance and threshold constraints | Study-specific implementation |
| Validation-selected peak threshold | Prevents selection using the held-out test participant | Study-specific model-selection procedure |
| Frozen PulsePPG/PaPaGei embeddings | Foundation-model and self-supervised PPG studies support transferring pretrained representations to downstream physiological tasks | Literature-supported transfer-learning baseline |
| Ridge regression on frozen embeddings | A regularized linear probe tests whether HRV information is already encoded while limiting overfitting | Study-specific downstream estimator using a standard probing method |
| Mean pooling of 10-second embeddings into one five-minute representation | Required because the pretrained encoder operates on shorter segments than the HRV analysis window | Study-specific aggregation design |
| Regression-head training and partial fine-tuning | Standard alternatives to frozen linear probing for adapting pretrained encoders | Transfer-learning ablations |
| Participant-level outer LOSO with participant-grouped inner validation | Prevents windows from the same participant appearing in both training and evaluation partitions | Leakage-control protocol |
| Heuristic HRV + embedding-based residual correction | Motivated by knowledge-informed PPG models and morphology-aware HRV estimation | Proposed hybrid method; must be validated against heuristic-only and embedding-only baselines |

### Interpretation

The Kazemi-style experiment is the closest reproduction of a published peak
detector within the constraints of our dataset. The original U-Net/dilated
models, loss variants, threshold selection, and IBI-correction variants are
literature-informed adaptations rather than strict reproductions.

The PulsePPG and PaPaGei experiments use published pretrained encoders, while
the five-minute aggregation, Ridge regression, participant-level nested LOSO,
and partial-fine-tuning protocol are study-specific downstream evaluation
choices.

The proposed hybrid model is therefore described as a
**literature-motivated method**:

PPG waveform
→ heuristic peak/IBI estimation
→ initial SDNN/RMSSD and signal-quality features
→ pretrained PPG embedding
→ regularized residual correction
→ final SDNN/RMSSD prediction.

It is not assumed to be superior by construction. Its contribution must be
established through leakage-safe comparison with:

1. heuristic HRV alone;
2. pretrained embedding alone;
3. heuristic features plus embedding;
4. heuristic features plus embedding-based residual correction.

### Recommended reproduction experiment

We will implement the official Kazemi seven-layer dilated CNN in PyTorch and compare:

1. ordinary BCE versus weighted BCE + Dice;
2. the paper-style 0.35-second peak finder versus SciPy `find_peaks`;
3. event F1, peak-count error, HRV MAE/R²/r, and coverage under the same participant-level split.

This experiment will determine whether the current bottleneck is caused primarily by the architecture, training objective, or peak decoder.


## Current Status and Next Steps

### In progress

1. **Kazemi-style PeakNet, complete 16-fold LOSO**

   A literature-derived Kazemi-style dilated CNN is currently being evaluated
   on Earring/Green using binary ±2-sample peak labels, ordinary BCE,
   validation event-F1 checkpoint selection, and median IBI correction.

   The evaluation will report peak-detection performance, peak-count error,
   HRV coverage, raw-versus-corrected IBI results, and SDNN/RMSSD MAE, R²,
   and Pearson r.

2. **Finalized Own PeakNet evaluation**

   The finalized study-specific PeakNet configuration should be evaluated
   separately from the Kazemi reproduction:

   - Earring/Green input
   - Dilated backbone
   - weighted BCE + Dice loss
   - Gaussian peak labels
   - validation event-F1 checkpoint selection
   - validation-selected event threshold
   - local-median IBI correction
   - complete 16-fold participant LOSO

   This experiment is required for a fair comparison between the original
   study-specific PeakNet and the literature-derived Kazemi-style model.

### Results requiring recalculation

3. **Construct an identical evaluation subset**

   Match qPPG, Own PeakNet, Kazemi PeakNet, PaPaGei, and PulsePPG predictions
   by:

   `participant + window_index + device + channel`

   Report both:

   - each method's native coverage; and
   - performance on the common intersection of windows available to every
     compared method.

   This distinction is necessary because a method can obtain apparently
   better error values by rejecting difficult windows.

### Recommended next experiment

4. **Physiology-informed embedding correction**

   Combine qPPG-derived HRV and quality features with frozen PulsePPG
   embeddings using leakage-safe nested Ridge.

   Evaluate two variants:

   **Feature-fusion model**

   `predicted HRV = Ridge(qPPG features + PulsePPG embedding)`

   **Residual-correction model**

   `final HRV = qPPG HRV + Ridge(target HRV − qPPG HRV)`

   Required comparisons:

   - qPPG alone;
   - frozen PulsePPG embedding alone;
   - qPPG features + embedding;
   - qPPG residual correction + embedding.

   All variants must use the same outer participant LOSO folds, grouped inner
   validation, and identical held-out windows.

5. **Extend the Ridge regularization search**

   The previous frozen-embedding experiments frequently selected
   `alpha = 1000`, the largest candidate in the original grid. This indicates
   that the optimum may lie outside the tested range.

   Extend the grid to:

   `[0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]`

   Confirm whether the selected alpha remains at the upper boundary. If it
   does, inspect feature scaling and compare performance with a
   mean-prediction baseline before extending the grid further.

### Completed ablations

6. **PulsePPG adaptation experiments — completed**

   Frozen Ridge, head-only training, and partial fine-tuning have been
   evaluated with complete 16-fold participant LOSO.

   Partial fine-tuning did not consistently outperform the frozen encoder
   across participants. Therefore, no large fine-tuning sweep is currently
   planned unless a specific hypothesis addresses:

   - participant-level calibration;
   - temporal aggregation;
   - device/domain shift; or
   - insufficient RMSSD-sensitive representations.

7. **Own PeakNet diagnostic ablations — completed**

   The following components have already been investigated on the diagnostic
   participant:

   - U-Net versus dilated backbone;
   - median replacement versus IBI removal;
   - learning-rate sweep;
   - positive-class-weight sweep;
   - peak-threshold sweep;
   - raw versus corrected IBI;
   - visual best/worst-window spot checks.

   These experiments identified over-detection, peak-count mismatch, and
   sensitivity of downstream HRV to beat errors as the main limitations.

### Longer-term methodological direction

8. **Temporal or beat-aware aggregation for RMSSD**

   Current global five-minute embeddings substantially compress prediction
   variability, particularly for RMSSD. Future models should preserve
   short-range beat-to-beat information through one of the following:

   - sequence modeling over 10-second embeddings;
   - attention-based segment aggregation;
   - beat-level or IBI-level embeddings;
   - prediction of local RMSSD contributions followed by aggregation.

   This experiment should follow the harmonized baseline comparison and
   hybrid Ridge experiment rather than precede them.

## 9. Cautions

- P1 smoke tests are diagnostic and cannot establish population-level superiority.
- Thresholds, checkpoint epochs, channel choices, and IBI gates must be chosen without using the held-out test participant.
- Overlapping 5-minute windows are not independent observations; all splitting remains participant-level.
- Coverage and QC rules must accompany every heuristic-versus-model comparison.
