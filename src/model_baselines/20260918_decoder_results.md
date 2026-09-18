# PPG→HRV Weekly Progress Update — Final Decoder Results Included

**Reporting period:** 2026-09-12 to 2026-09-18  
**Purpose:** summarize this week's new experiments while retaining last week's verified baselines for comparison.

## 1. Executive summary（本周核心进展）

This week we completed the previously pending population-level evaluations of our own Dilated PeakNet and the literature-derived Kazemi PeakNet, diagnosed the source of the peak-model failures, and tested a validation-based decoder objective.

Main findings:

1. The finalized **Own Dilated PeakNet 16-fold LOSO** run is complete. It systematically over-detects peaks (`predicted/reference count ratio = 1.593 ± 0.192`) and performs poorly for population-level HRV.
2. The **Kazemi PeakNet grouped 4-fold** evaluation is complete on all 11,788 held-out windows. It captures HRV variation better than the original PeakNet (`pooled r=0.513` for SDNN and `0.376` for RMSSD), but has severe fold-dependent calibration errors.
3. A new **validation decoder score** nearly eliminates peak-count bias. In the completed grouped 4-fold Kazemi run, it reduces pooled SDNN MAE from `27.879` to `10.560 ms` and raises pooled SDNN r from `0.513` to `0.767`.
4. A **hard-negative loss** gives only a small additional P1 SDNN improvement and does not improve RMSSD consistently.
5. An external **BIDMC sanity check** confirms that the model detects true cardiac cycles but also decodes secondary PPG waveform maxima as extra events.
6. **Kazemi + decoder score is now the strongest learned SDNN result**, although it uses grouped 4-fold rather than LOSO. Frozen PulsePPG remains the strongest stable foundation-model baseline and is substantially better for RMSSD.

## 2. Last week's verified reference results（上周基线）

### 2.1 Pooled held-out windows

| Method | Device | Protocol | SDNN MAE | SDNN R² | SDNN r | RMSSD MAE | RMSSD R² | RMSSD r |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **PulsePPG frozen + nested Ridge** | Earring/Green | 16-fold LOSO | **14.269** | **0.139** | **0.418** | 11.265 | −0.185 | 0.153 |
| PulsePPG frozen + nested Ridge | Ring/Green | 16-fold LOSO | 14.079 | 0.124 | 0.398 | 10.121 | 0.017 | **0.326** |
| PulsePPG frozen + nested Ridge | Watch/Green | 16-fold LOSO | 14.821 | 0.026 | 0.297 | 11.203 | −0.178 | 0.090 |
| PulsePPG head-only | Earring/Green | 16-fold LOSO | 14.833 | 0.062 | 0.300 | 11.108 | −0.163 | 0.068 |
| PulsePPG partial fine-tuning | Earring/Green | 16-fold LOSO | 14.478 | 0.091 | 0.351 | 11.297 | −0.195 | 0.076 |
| PaPaGei-S frozen + Ridge | Earring/Green | 16-fold LOSO | 15.04 | −0.004 | 0.215 | 11.29 | −0.203 | 0.053 |
| PaPaGei-S frozen + Ridge | Ring/Green | 16-fold LOSO | 14.82 | 0.029 | 0.256 | **9.84** | **0.064** | 0.325 |
| PaPaGei-S frozen + Ridge | Watch/Green | 16-fold LOSO | 17.36 | −0.269 | −0.229 | 10.83 | −0.152 | −0.141 |

### 2.2 Interpretation retained from last week

- Frozen PulsePPG + nested Ridge is the strongest stable foundation-model baseline.
- Partial fine-tuning helps SDNN relative to the matched neural-head control, but does not consistently outperform Frozen Ridge.
- PaPaGei Ring/Green is competitive for RMSSD.
- Global five-minute embeddings compress prediction variance, especially for RMSSD.
- qppg remains a strong heuristic reference, but a formal comparison requires identical accepted windows and coverage reporting.

## 3. New result: finalized Own Dilated PeakNet LOSO

The previously pending complete 16-fold Earring/Green LOSO run is now finished.

### 3.1 Participant-fold mean ± std

| Quantity | Result |
|---|---:|
| Peak precision | 0.294 ± 0.050 |
| Peak recall | 0.467 ± 0.084 |
| Peak F1 | 0.360 ± 0.061 |
| Predicted/reference peak-count ratio | **1.593 ± 0.192** |
| RR correction ratio | 0.356 ± 0.051 |
| SDNN MAE | 29.08 ± 10.47 ms |
| SDNN R² | −8.452 ± 7.686 |
| SDNN r | 0.005 ± 0.239 |
| RMSSD MAE | 13.26 ± 5.19 ms |
| RMSSD R² | −2.565 ± 2.686 |
| RMSSD r | 0.029 ± 0.355 |

### 3.2 Conclusion

The original model frequently decodes more than one event per cardiac cycle. Median IBI correction reduces extreme HRV errors but cannot reconstruct reliable beat-to-beat timing. The finalized result confirms that this model is currently an interpretable failure-analysis baseline rather than the best HRV estimator.

## 4. New result: decoder experiments

We tested whether event-decoding constraints could reduce false peaks independently of changing the backbone.

### 4.1 Fixed prominence and minimum-distance sweep on P1

| Decoder | Peak F1 | Predicted / true peaks | RR correction | SDNN MAE / r | RMSSD MAE / r |
|---|---:|---:|---:|---:|---:|
| Original Dilated | 0.355 | 678.7 / 442.2 | 0.356 | 31.01 / −0.180 | **7.83 / 0.297** |
| Prominence 0.10, distance 300 ms | 0.331 | 613.6 / 442.1 | 0.416 | 34.30 / −0.187 | 7.56 / 0.293 |
| Prominence 0.20, distance 300 ms | 0.291 | 529.5 / 442.1 | 0.414 | 53.56 / 0.216 | 14.76 / −0.124 |
| Prominence 0.10, distance 350 ms | 0.282 | 508.3 / 441.8 | 0.361 | 37.33 / 0.227 | 14.50 / −0.250 |
| Prominence 0.10, distance 400 ms | 0.269 | 468.6 / 440.4 | 0.308 | 23.33 / 0.311 | 14.57 / −0.194 |

Increasing the minimum distance reduces the number of predictions, but a fixed decoder constraint does not jointly improve SDNN and RMSSD.

### 4.2 Validation decoder-score selection

We introduced a validation-only decoder score:

\[
S = F1 - 0.5\,|\text{count ratio}-1| - 0.25\,(\text{RR correction ratio}).
\]

| Method | Peak F1 | Predicted / true peaks | RR correction | SDNN MAE / r | RMSSD MAE / r |
|---|---:|---:|---:|---:|---:|
| Original Dilated | 0.355 | 678.7 / 442.2 | 0.356 | 31.01 / −0.180 | **7.83 / 0.297** |
| Decoder score | 0.294 | **439.8 / 440.5** | 0.272 | 13.43 / 0.612 | 15.06 / −0.348 |
| Decoder score + hard-negative loss | 0.294 | 427.6 / 440.5 | **0.258** | **12.44 / 0.668** | 15.81 / −0.271 |

The decoder score largely removes P1 over-detection and substantially improves SDNN. However, RMSSD becomes worse, showing that matching the total event count does not ensure accurate adjacent IBIs.

The hard-negative loss provides only a small additional SDNN gain and is retained as an exploratory ablation rather than a confirmed improvement.

## 5. New result: Kazemi PeakNet grouped 4-fold

We implemented the literature-derived seven-layer Kazemi dilated CNN with ELU activations, binary ±2-sample peak labels, ordinary BCE, validation checkpoint selection, and median IBI correction.

To reduce runtime while preserving participant independence, we used a leakage-safe grouped 4-fold protocol:

- 10 training participants
- 2 validation participants
- 4 test participants
- every participant appears in the test set exactly once
- 11,788 total held-out windows
- zero duplicate participant/window pairs

### 5.1 Overall results

| Aggregation | Target | MAE | R² | Pearson r |
|---|---|---:|---:|---:|
| Pooled windows | SDNN | 27.879 ms | −3.714 | **0.513** |
| Pooled windows | RMSSD | 15.445 ms | −1.903 | **0.376** |
| Participant mean ± std | SDNN | 25.61 ± 23.11 ms | −5.405 ± 6.884 | 0.432 ± 0.425 |
| Participant mean ± std | RMSSD | 17.04 ± 11.09 ms | −6.631 ± 9.555 | 0.162 ± 0.347 |

Pooled event detection:

- Precision: `0.308`
- Recall: `0.379`
- F1: `0.340`

### 5.2 Fold-level results

| Fold | Test participants | Threshold | SDNN MAE | SDNN r | RMSSD MAE | RMSSD r |
|---:|---|---:|---:|---:|---:|---:|
| 0 | P3, P5, P11, P20 | 0.35 | 31.60 | 0.064 | 12.45 | −0.067 |
| 1 | P18, P1, P13, P7 | 0.45 | 61.53 | 0.465 | 17.69 | 0.323 |
| 2 | P9, P6, P15, P19 | 0.55 | **7.41** | **0.834** | 15.50 | 0.237 |
| 3 | P4, P12, P8, P10 | 0.10 | **7.92** | **0.847** | 16.46 | **0.479** |

### 5.3 Interpretation

Kazemi captures substantially more window-to-window HRV variation than the original Dilated PeakNet, particularly for SDNN. However, calibration varies sharply across folds:

- Fold 0 SDNN mean error: `+24.72 ms`
- Fold 1 SDNN mean error: `+58.94 ms`
- Fold 2 SDNN mean error: `−0.18 ms`
- Fold 3 SDNN mean error: `−3.01 ms`

The high pooled correlation but negative R² is therefore not contradictory. The predictions track variation, but large fold-specific offsets make the absolute estimates inaccurate.

## 6. New result: BIDMC external sanity check

An older P1 Dilated checkpoint was evaluated on one relatively stable BIDMC clinical PPG record.

| Metric | Result |
|---|---:|
| Strict peak F1 | 0.691 |
| Delay-aligned peak F1 | 0.692 |
| Precision | 0.535 |
| Recall | 0.976 |
| Predicted/reference count ratio | 1.829 |

The network recognizes nearly all true cardiac cycles, but it also assigns high probabilities to secondary PPG waveform maxima. This confirms that the bottleneck is event decoding and false-positive control rather than complete failure to learn pulse morphology.

## 7. Updated cross-method comparison

| Method | Evaluation | Main strength | Main limitation |
|---|---|---|---|
| qppg heuristic | Native accepted windows | Strong reference HRV estimate | Coverage differs; common-window comparison pending |
| Own Dilated PeakNet | 16-fold LOSO | Explicit peaks and interpretable error analysis | Systematic over-detection; weak HRV correlation |
| Kazemi PeakNet | Grouped 4-fold | Highest new peak-model HRV correlations | Severe fold-dependent calibration and high MAE |
| PulsePPG frozen + Ridge | 16-fold LOSO | Best stable Earring model-level baseline | RMSSD correlation remains modest |
| PulsePPG partial FT | 16-fold LOSO | Some SDNN improvement over head-only | No consistent gain over Frozen Ridge |
| PaPaGei frozen + Ridge | 16-fold LOSO | Strong Ring/Green RMSSD | Weaker Earring SDNN; alpha search boundary issue |

The protocols differ: the foundation-model results use LOSO, whereas the new Kazemi result uses grouped 4-fold. Both are participant-independent, but they should not be described as an exact controlled comparison.

## 8. Current scientific interpretation

1. **Peak-model failure is primarily a decoding/calibration problem.** Both BIDMC and wearable results show that cardiac activity is recognized, but secondary maxima are often converted into extra events.
2. **SDNN benefits strongly from correct peak count and RR-distribution calibration.** This is demonstrated by the P1 decoder-score result and Kazemi folds 2/3.
3. **RMSSD requires accurate local timing, not only the correct number of beats.** Occasional missed or extra peaks create alternating short/long intervals and strongly distort successive differences.
4. **Strict peak F1 and HRV performance measure different properties.** A model may consistently select a delayed PPG landmark, producing modest ECG-centered F1 while preserving RR intervals.
5. **Embedding models are better calibrated but shrink variability.** Peak-based Kazemi has higher pooled correlation, while PulsePPG has much lower MAE and better R².

## 9. New result: Kazemi + validation decoder score

The grouped 4-fold decoder-score experiment is complete. Checkpoint and threshold were selected using validation participants only:

\[
S = F1 - 0.5\,|\text{count ratio}-1| - 0.25\,(\text{RR correction ratio}).
\]

Configuration:

- Kazemi architecture and binary targets
- Earring/Green input
- median IBI correction
- hard-negative weight `0.0`
- separate output directory; previous results are preserved

### 9.1 Final results

| Aggregation | Target | MAE | R² | Pearson r |
|---|---|---:|---:|---:|
| Pooled 11,788 windows | **SDNN** | **10.560 ms** | **0.262** | **0.767** |
| Pooled 11,788 windows | RMSSD | 17.316 ms | −1.821 | 0.385 |
| Participant mean ± std | **SDNN** | **9.338 ± 4.125 ms** | **0.084 ± 0.593** | **0.781 ± 0.125** |
| Participant mean ± std | RMSSD | 17.622 ± 7.461 ms | −6.441 ± 7.586 | 0.310 ± 0.357 |

All 11,788 held-out windows were present, with zero duplicate participant/window pairs and 100% HRV coverage.

### 9.2 Paired comparison against original Kazemi

| Target | Original pooled MAE | Decoder-score pooled MAE | Original pooled R² | Decoder-score pooled R² | Original pooled r | Decoder-score pooled r |
|---|---:|---:|---:|---:|---:|---:|
| SDNN | 27.879 | **10.560** | −3.714 | **0.262** | 0.513 | **0.767** |
| RMSSD | **15.445** | 17.316 | −1.903 | **−1.821** | 0.376 | **0.385** |

SDNN MAE decreases by approximately **62.1%**. The new validation objective repairs the catastrophic SDNN bias in folds 0 and 1 without degrading folds 2 and 3:

| Fold | Original SDNN MAE | Decoder-score SDNN MAE | Original SDNN r | Decoder-score SDNN r |
|---:|---:|---:|---:|---:|
| 0 | 31.60 | **14.36** | 0.064 | **0.608** |
| 1 | 61.53 | **12.50** | 0.465 | **0.742** |
| 2 | 7.41 | **7.27** | **0.834** | 0.812 |
| 3 | 7.92 | **7.38** | 0.847 | **0.883** |

Every fold selected threshold `0.60`, and validation peak-count ratios ranged only from `0.999` to `1.009`. This indicates that the decoder score substantially improves cross-fold calibration.

### 9.3 Comparison with last week's learned baselines

| Method | Protocol | SDNN MAE | SDNN R² | SDNN r | RMSSD MAE | RMSSD R² | RMSSD r |
|---|---|---:|---:|---:|---:|---:|---:|
| **Kazemi + decoder score** | Grouped 4-fold | **10.560** | **0.262** | **0.767** | 17.316 | −1.821 | **0.385** |
| PulsePPG frozen + Ridge | 16-fold LOSO | 14.269 | 0.139 | 0.418 | 11.265 | −0.185 | 0.153 |
| PulsePPG partial FT | 16-fold LOSO | 14.478 | 0.091 | 0.351 | 11.297 | −0.195 | 0.076 |
| PaPaGei Ring frozen + Ridge | 16-fold LOSO | 14.82 | 0.029 | 0.256 | **9.84** | **0.064** | 0.325 |

Because Kazemi uses grouped 4-fold and the foundation models use LOSO, this table is contextual rather than a strict controlled ranking. Nevertheless, Kazemi + decoder score is the strongest current SDNN result across all three reported metrics. It does not improve RMSSD.

## 10. Next steps

1. Add strict and delay-aligned event metrics to all peak-model reports.
2. Develop an IBI-sensitive validation objective for RMSSD; the count-based decoder score solves SDNN but not adjacent-interval accuracy.
3. Match qppg, Own PeakNet, Kazemi, PulsePPG, and PaPaGei on identical held-out windows and report native/common coverage.
4. Run the proposed hybrid baseline:

   `qppg HRV + quality features + frozen embedding → nested Ridge residual correction`

5. Extend the frozen-embedding Ridge alpha grid beyond `1000` and check whether selection remains boundary-limited.

## 11. Meeting take-away

> This week established that cross-participant event decoding and calibration—not complete failure to learn cardiac morphology—was the main limitation of our peak models. Validation decoder-score selection transforms Kazemi PeakNet into the strongest current learned SDNN estimator, reducing pooled MAE by 62.1% to 10.56 ms and increasing pooled r to 0.767. However, RMSSD remains poor because correct total peak count does not guarantee accurate adjacent IBIs. The next priority is therefore an IBI-sensitive objective and a leakage-safe hybrid combining heuristic HRV with learned residual correction.
