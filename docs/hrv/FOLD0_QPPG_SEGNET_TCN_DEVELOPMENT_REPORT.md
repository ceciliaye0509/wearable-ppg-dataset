# Fold-0 qPPG × SegNet/TCN development comparison

## Scope and status

This is a **fold-0 inner-validation development report**, not a four-fold or outer-test result.  It compares four model variants under one frozen raw-slot contract:

1. raw causal TCN;
2. qPPGFast + causal-TCN residual;
3. raw partner-style SegNet;
4. qPPGFast + partner-style-SegNet residual.

No outer-test participant was loaded for these development runs, and no formal four-fold run was started.  The raw NPZ download and staged raw-slot cache were already complete before this work; neither was re-run.  No Hugging Face token was read, displayed, or recreated.

## Runtime environment

| Item | Value |
|---|---|
| Compute | Purdue A30 node `gilbreth-d005` |
| Main training runtime | `/scratch/gilbreth/qiuyue/daily_hrv/envs/water-legacy/bin/python` |
| Python | 3.8.20 |
| NumPy | 1.24.3 |
| PyTorch / CUDA | 2.0.0 / 11.8; CUDA available |
| Training environment policy | Existing `water-legacy` used unchanged; no package installation or upgrade |
| qPPG feature extraction only | Existing `anaconda/2024.10-py312` with SciPy 1.13.1, because `water-legacy` has no SciPy; no environment was modified |

## Frozen experimental contract

| Dimension | Contract |
|---|---|
| Raw source | `synced_3device_rawaligned_training_v1_stride30_rawslots` |
| Cached source | `artifacts/ml_hrv/rawslot_cache` |
| Device / PPG input | Earring, green channel only, raw-slot 100 Hz |
| Window / update | 5 min window, 30 s stride; 10 s encoder segments |
| Labels | ECG-corrected RMSSD and SDNN; ECG label QC only |
| Fold-0 training participants | P20, P9, P7, P5, P18, P15, P19, P1, P6, P4 |
| Fold-0 inner validation | P10, P12 (`n=1,178` Earring windows) |
| Held-out outer test, not loaded | P8, P11, P3, P13 |
| Training unit | 4 windows/batch; 128 updates/epoch |

The qPPG feature CSV was generated from the same raw-slot contract and is PPG-only.  It contains qPPG RMSSD/SDNN base estimates plus nine quality features (validity, peak count, valid IBI ratio, IBI CV, IBI-correction ratio, SQI, valid-sample ratio, maximum gap, and motion).  The qPPG valid fraction was 75.1% in the fold-0 training rows and 95.1% in the validation rows.  All qPPG feature imputation/scaling and invalid-base fallback medians were fitted on the training rows only.

## What was compared with the original SegNet path

### Original partner-style SegNet path

`PartnerSegNetEncoder` is the repository's faithful partner-style 10-second CNN: three stride-2 `Conv1d → BatchNorm → ReLU` blocks operating on selected raw PPG channels plus their masks.  `SegNetMeanHead` averages the 30 segment tokens and predicts two HRV outputs.  The pre-existing broad baseline configuration was `src/ml_hrv/configs/segnet_rawslot_baseline.json` (three devices and default green+IR inputs).

### Controlled SegNet adaptation for this comparison

The SegNet **architecture was retained** (`direct_architecture: segnet_mean`, token dimension 64, no timestamp jitter).  The comparison configuration changed only the input/training contract required for a like-for-like fold-0 comparison with TCN:

- restrict input to Earring + green at 100 Hz, instead of the broad three-device default;
- retain 5-minute windows and the frozen fold-0 split;
- use direct Huber loss in standardized log-HRV space, rather than learned heteroscedastic NLL, because the earlier direct-TCN gate revealed an uncertainty/near-constant-prediction failure mode;
- use a time-bounded 6-epoch × 128-update development gate with early stopping, not a production training schedule.

### qPPG residual treatment

For either architecture, qPPG residual means:

```text
final normalized log-HRV = normalized qPPG base + neural correction (residual)
```

The neural model still sees raw PPG.  It also receives the train-standardized PPG-only qPPG quality vector, including the `qppg_valid` flag.  On an invalid qPPG window, the qPPG base is replaced by the train-only qPPG median; invalid windows are retained rather than dropped.  This tests whether the neural network can correct a physiological qPPG estimate, rather than asking it to relearn the entire HRV calculation from scratch.

## Created and modified code

All paths below are repository-relative.

| Path | Change / purpose |
|---|---|
| `src/ml_hrv/models/encoder.py` | Existing `PartnerSegNetEncoder`; used unchanged as the partner-style SegNet encoder for this comparison. |
| `src/ml_hrv/models/direct.py` | Extended `DirectHRVHead` and `SegNetMeanHead` with optional qPPG-quality conditioning. |
| `src/ml_hrv/models/pipeline.py` | Added residual assembly: normalized qPPG base + learned direct correction. |
| `src/ml_hrv/data/qppg.py` | **Created.** Loads keyed qPPG feature CSV rows and implements train-only median imputation / standardization. |
| `src/ml_hrv/data/dataset.py` | Joins qPPG fields to raw-slot samples and preserves qPPG validity/base information in batches. |
| `src/ml_hrv/config.py` | Added qPPG feature-table and residual enablement configuration. |
| `src/ml_hrv/training/losses.py` | Added development Huber direct-loss option (existing code change used by all raw/qPPG gates). |
| `src/ml_hrv/training/trainer.py` | Encodes qPPG base with the train-only target scaler and saves qPPG scaler state in checkpoints. |
| `src/ml_hrv/training/checkpointing.py` | Stores optional qPPG feature-scaler state. |
| `src/ml_hrv/evaluation/evaluator.py` | Feeds qPPG residual inputs at evaluation and writes qPPG base/validity/residual fields. |
| `src/ml_hrv/scripts/train.py` | Fits qPPG scaler on training rows only and supplies it to validation data. |
| `src/ml_hrv/scripts/evaluate_inner_validation.py` | Recomputes and verifies train-only target/qPPG scaler state before inner-validation evaluation. |
| `src/ml_hrv/scripts/build_qppgfast_features.py` | **Created earlier in this development line.** Builds the frozen raw-slot qPPGFast feature table without labels. |
| `src/ml_hrv/scripts/analyze_qppg_residual.py` | **Created.** Read-only qPPG base vs residual audit by all/valid/invalid qPPG windows. |
| `src/ml_hrv/configs/teacher_green_earring_causal_tcn_huber_dev.json` | Raw-TCN development gate configuration. |
| `src/ml_hrv/configs/teacher_green_earring_causal_tcn_qppg_residual_dev.json` | qPPG + TCN residual configuration. |
| `src/ml_hrv/configs/teacher_green_earring_segnet_huber_dev.json` | **Created.** Raw SegNet fold-0 development gate configuration. |
| `src/ml_hrv/configs/teacher_green_earring_segnet_qppg_residual_dev.json` | **Created.** qPPG + SegNet residual fold-0 development gate configuration. |

Relevant development commits: `e3ce855` (Huber gate), `3198888` (inner-validation evaluator), `1d29ffa` (qPPG residual path), `76b31f9` (SegNet gates), and `dfc07bb` (qPPG stratification audit).

## Results: fold-0 inner validation

The train-median baseline predicts the fold-0 training-label median for every validation window.  It is not a partner model and does not use PPG.

| Model | RMSSD MAE (ms) | RMSSD r | RMSSD R² | RMSSD prediction SD (ms) | SDNN MAE (ms) | SDNN r | SDNN R² | SDNN prediction SD (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train-median baseline | 7.620 | n/a | ~0 | 0.000 | 15.301 | n/a | ~0 | 0.000 |
| Raw TCN (Huber gate) | 7.371 | -0.104 | -0.142 | 0.026 | 14.942 | 0.106 | -0.015 | 0.038 |
| qPPG + TCN residual | 14.197 | 0.233 | -3.756 | 14.684 | 7.067 | 0.889 | 0.767 | 16.812 |
| Raw SegNet (Huber gate) | 7.373 | -0.066 | -0.171 | 2.853 | 14.215 | 0.217 | 0.040 | 4.702 |
| qPPG + SegNet residual | 13.310 | 0.231 | -3.443 | 14.599 | **6.830** | **0.889** | **0.778** | 15.872 |

For scale reference, validation target SD is 8.135 ms for RMSSD and 20.385 ms for SDNN.

## Interpretation

1. **Raw direct models are not sufficient.**  Raw TCN is essentially constant for both outputs; raw SegNet has more variation but only weak SDNN signal and does not beat the RMSSD median baseline.
2. **qPPG augmentation is strongly effective for SDNN.**  qPPG + SegNet gives the best SDNN result (MAE 6.830 ms, `r=0.889`, `R²=0.778`), reducing MAE by about 55% relative to the train-median baseline.  qPPG + TCN is close (MAE 7.067 ms).  The 0.237 ms gap is too small to claim an architecture winner from one inner-validation split.
3. **Current qPPG residual design is unsuitable for RMSSD.**  Both qPPG residual models have strongly negative RMSSD R² and prediction SD about 14.6 ms, much larger than the 8.1 ms target SD.  This is an over-variable RMSSD correction, not the earlier constant-prediction collapse.
4. **Do not average RMSSD and SDNN into one score.**  Doing so would hide the SDNN gain and RMSSD failure.  Report them separately.

## Decision and next steps

- Do **not** launch formal four-fold training yet.
- Preserve these four fold-0 development outputs and run the read-only `analyze_qppg_residual.py` audit for all/qPPG-valid/qPPG-invalid windows.
- Diagnose whether RMSSD degradation originates in the qPPG base, the neural residual correction, or invalid-qPPG fallback.
- Only after that audit, consider a separate RMSSD head/loss or a separate RMSSD model.  Any later four-fold decision must use the same frozen contract and be reported separately for RMSSD and SDNN.

## Artifact locations

| Variant | Run directory | Inner-validation outputs |
|---|---|---|
| Raw TCN | `artifacts/ml_hrv/dev_tcn_huber_fold0` | `inner_validation/inner_validation_metrics.json`, `inner_validation/window_predictions.csv` |
| qPPG + TCN | `artifacts/ml_hrv/dev_tcn_qppg_residual_fold0` | `inner_validation/inner_validation_metrics.json`, `inner_validation/window_predictions.csv` |
| Raw SegNet | `artifacts/ml_hrv/dev_segnet_raw_fold0` | `inner_validation/inner_validation_metrics.json`, `inner_validation/window_predictions.csv` |
| qPPG + SegNet | `artifacts/ml_hrv/dev_segnet_qppg_residual_fold0` | `inner_validation/inner_validation_metrics.json`, `inner_validation/window_predictions.csv` |
