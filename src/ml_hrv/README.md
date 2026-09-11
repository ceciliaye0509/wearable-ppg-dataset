# `ml_hrv`

`ml_hrv` is the raw-slot continuous HRV track. It lives beside the partner
`model_baselines` package and does not alter that code.

The primary contract is fixed:

- input waveform: `ppg_rawslot_values`, one device, green + IR;
- missingness/timing: `ppg_rawslot_mask` and relative timestamp jitter;
- target order: `ecg_rmssd_corrected_ms`, then `ecg_sdnn_corrected_ms`;
- main protocol: causal trailing 5 minutes, update every 30 seconds;
- phase-one motion input: `accel_motion_mean_mag`, one scalar per window/device;
- no participant ID and no ECG-derived inference feature;
- no `ppg_resampled_values` in the primary model.

See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for architecture and
experiment order, and [BASELINE_RESULTS.md](BASELINE_RESULTS.md) for the frozen
four-fold result, acceleration benchmarks, and current go/no-go decisions.
