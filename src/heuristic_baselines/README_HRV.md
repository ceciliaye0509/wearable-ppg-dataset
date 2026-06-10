# HRV Pipeline (follow-up)

PPG-based heart rate variability (HRV) for the multi-site wearable dataset, with
optional validation against synchronized ECG. Built on **NeuroKit2**.

This is an additive layer on top of the published HR-baseline code.

> **Note:** HRV computed from PPG peaks is technically *pulse rate variability
> (PRV)*, an approximation of ECG-derived HRV. We call it HRV for short.

---

## 1. Install

From the repo:

```bash
cd src/heuristic_baselines
pip install -r requirements.txt        # numpy, scipy, pandas, neurokit2
```

NeuroKit2 (>=0.2.0) is already listed in `requirements.txt`. Python 3.10+.

## 2. Get the data

The 5-minute windowed data lives in the **private** HuggingFace dataset
`snowballlab/HRV`. Download it as a **sibling folder of the repo**:

```bash
# from the folder that CONTAINS the repo (e.g. ~/Desktop)
huggingface-cli login                  # paste your HF token (needs access to snowballlab)
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="snowballlab/HRV", repo_type="dataset", local_dir="HRV")'
```

Expected layout — `HRV/` must sit next to `wearable-ppg-dataset/`:

```
<parent>/
├── wearable-ppg-dataset/                 # this repo
└── HRV/
    └── 5min_windowed/
        └── P7/
            ├── alignment_windows_P7_Earring.npz
            ├── alignment_windows_P7_Ring.npz
            ├── alignment_windows_P7_Necklace.npz
            └── alignment_windows_P7_Watch.npz
```

Each `.npz` holds the windowed PPG (`ppg_green`, `ppg_ir`), ground-truth HR
(`hr_gt`), timestamps (`t0_ms`), and synchronized ECG (`rr_intervals_ms`,
`r_peak_samples`) used for validation.

## 3. Run

One command runs the whole pipeline:

```bash
cd src/heuristic_baselines
python run_all.py
```

This clears stale caches, computes PPG HRV, then compares it to ECG. Options:

```bash
python run_all.py --no-eval       # PPG HRV only, skip the ECG comparison (faster)
python run_all.py --keep-cache    # reuse preprocess caches (faster reruns, same data)
```

You can also run the steps individually:

```bash
python hrv_runner.py              # step 1: PPG -> HRV
python eval_ppg_vs_ecg.py         # step 2: PPG HRV vs ECG gold standard
```

Which participants / devices / channels run is controlled in `config.py`
(`HEURISTIC_PIPELINE_PARTICIPANTS`, `HEURISTIC_DEVICE_ROLES`,
`HEURISTIC_PPG_CHANNELS`). No code changes are needed to add more participants —
just download their data and rerun.

## 4. Outputs

Everything lands in `outputs/<Px>/` (git-ignored):

| File | Contents |
| --- | --- |
| `hrv_<device>_<channel>.csv` | One row per 5-min window: time-domain (`HRV_RMSSD`, `HRV_SDNN`, `HRV_pNN50`, …), frequency-domain (`HRV_LF`, `HRV_HF`, `HRV_LFHF`), Poincaré (`HRV_SD1`, `HRV_SD2`), plus `hr_gt`, `hr_mean`, `n_peaks` |
| `ppg_vs_ecg_<device>_<channel>.csv` | Per-window PPG value, ECG value, and error for each metric |
| `ppg_vs_ecg_<device>_<channel>_summary.csv` | Per-metric agreement: bias, 95% limits of agreement (Bland-Altman), MAE, correlation |
| `*_preprocess.npz` | Cached band-pass-filtered signal (auto-generated, safe to delete) |

**Primary endpoints:** `HRV_RMSSD` (short-term / vagal) and `HRV_SDNN` (overall),
reported alongside mean HR. Frequency-domain indices are exploratory.

## 5. Important: quality control

`hr_mean` vs `hr_gt` (and the ECG comparison) tells you whether beat detection
worked. When they disagree, the HRV values are **artifacts of failed detection,
not real variability** — a large RMSSD/SDNN is a red flag, not a finding.
Filter windows by agreement (e.g. keep `|hr_mean - hr_gt| < 5 bpm`) before
aggregating. In our P7 run, the **earring** channel matched ECG closely
(HR r ≈ 1.0, SDNN r ≈ 0.998, RMSSD r ≈ 0.982), while **wrist/necklace green**
channels often failed detection and should be screened out.

## 6. Common gotchas

- **Always clear caches after changing the data or window length.** `run_all.py`
  does this automatically; if running steps manually, delete
  `outputs/*/*_preprocess.npz` first, or stale windows will be silently reused.
- **Run `eval_ppg_vs_ecg.py` after `hrv_runner.py`** — it reads the `hrv_*.csv`
  files. `run_all.py` already orders this correctly.
- **The ECG comparison is slow** (~7–8 min per device/channel, since it runs
  NeuroKit on each window). Use `--no-eval` while iterating on PPG HRV.

## 7. Files

| File | Role |
| --- | --- |
| `algorithms/hrv.py` | Peak detection + NeuroKit2 HRV computation (one window) |
| `hrv_runner.py` | Batch PPG → HRV over all participants/devices/channels |
| `eval_ppg_vs_ecg.py` | PPG HRV vs ECG gold-standard comparison |
| `run_all.py` | Single-command entry that runs the above in order |
| `config.py` | Settings (HRV block appended; existing HR settings unchanged) |

## 8. Method summary

PPG is band-pass filtered (0.7–3.5 Hz), beats are detected with NeuroKit2's
Elgendi pulse detector (with artifact correction), inter-beat (NN) intervals are
gated to 300–2000 ms, and HRV is computed with NeuroKit2's `hrv_time`,
`hrv_frequency`, and `hrv_nonlinear`. Frequency bands and 5-minute window length
follow the 1996 ESC/NASPE Task Force standard. ECG ground truth is computed from
the stored RR intervals with the same NeuroKit2 estimators for a like-for-like
comparison.
