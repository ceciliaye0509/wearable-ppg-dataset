"""Recompute Steps 0-8 MAE against v2 ECG (with IBI correction + 80% gate)."""
import numpy as np, sys
from pathlib import Path
from scipy.signal import detrend

sys.path.insert(0, str(Path(__file__).resolve().parent))
from algorithms import hrv
from preprocess import preprocess_ppg
from io_utils import merged_windows_npz
import config

DEVICES = ["Earring", "Ring", "Necklace", "Watch"]
CHANNELS = {"Earring": "ppg_ir", "Ring": "ppg_green", "Necklace": "ppg_green", "Watch": "ppg_green"}


def ecg_rmssd_v2(rr_ms):
    """ECG HRV with v2 correction (IBI correction + 80% gate)."""
    rr = np.asarray(rr_ms, dtype=np.float64)
    valid_mask = (rr >= hrv.IBI_MIN_MS) & (rr <= hrv.IBI_MAX_MS)
    if rr.size > 0 and valid_mask.sum() / len(rr) < 0.80:
        return float("nan")
    rr = rr[valid_mask]
    if rr.size < 3:
        return float("nan")
    rr = hrv._correct_ibi_artifacts(rr)
    return float(np.sqrt(np.mean(np.diff(rr) ** 2)))


def ppg_rmssd_step(sig, fs, *, do_interp=False, do_ibi_correct=False,
                    do_threshold=False, cubic=False):
    """Run PPG HRV pipeline with specific step configuration."""
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig[~np.isnan(sig)]
    peaks = hrv.detect_ppg_peaks(sig, fs)
    if peaks.size < 3:
        return float("nan")

    if do_interp:
        peaks_f = hrv._refine_peaks_parabolic(sig, peaks)  # always cubic in code
        ibi = np.diff(peaks_f) / fs * 1000.0
    else:
        ibi = np.diff(peaks) / fs * 1000.0

    # IBI validity gate
    if do_threshold:
        n_valid = int(((ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)).sum())
        if len(ibi) > 0 and n_valid / len(ibi) < 0.80:
            return float("nan")

    nn = ibi[(ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)]
    if nn.size < 3:
        return float("nan")

    if do_ibi_correct:
        nn = hrv._correct_ibi_artifacts(nn)

    return float(np.sqrt(np.mean(np.diff(nn) ** 2)))


def run_step(name, ppg_data, fs, rr_all, n_rr, n_windows, **kwargs):
    """Compute MAE for a given step config."""
    errors = []
    valid = 0
    for i in range(n_windows):
        ecg_v = ecg_rmssd_v2(rr_all[i][:int(n_rr[i])])
        ppg_v = ppg_rmssd_step(ppg_data[i], fs, **kwargs)
        if np.isfinite(ecg_v) and np.isfinite(ppg_v):
            valid += 1
            errors.append(abs(ppg_v - ecg_v))
    mae = np.mean(errors) if errors else float("nan")
    return mae, valid, n_windows


def main():
    participant = sys.argv[1] if len(sys.argv) > 1 else "P7"
    root = config.HEURISTIC_WINDOWS_ROOT

    steps = [
        ("Step 0: Baseline",        dict(do_interp=False, do_ibi_correct=False, do_threshold=False)),
        ("Step 1: +Interpolation",   dict(do_interp=True,  do_ibi_correct=False, do_threshold=False)),
        ("Step 2: +IBI Correction",  dict(do_interp=True,  do_ibi_correct=True,  do_threshold=False)),
        ("Step 7: +80% Threshold",   dict(do_interp=True,  do_ibi_correct=True,  do_threshold=True)),
        # Step 8 = same as Step 7 (cubic is always on in current code)
    ]

    print(f"\n{'='*80}")
    print(f"  Recompute Steps vs v2 ECG — {participant}")
    print(f"{'='*80}\n")

    for dev in DEVICES:
        ch = CHANNELS[dev]
        raw_path = merged_windows_npz(root, participant, dev)
        if not raw_path.is_file():
            print(f"  [SKIP] {dev}")
            continue

        with np.load(raw_path, allow_pickle=True) as z:
            ppg_raw = np.asarray(z[ch])
            fs = float(z["ppg_fs"])
            rr_all = np.asarray(z["rr_intervals_ms"])
            n_rr = np.asarray(z["n_rr"])

        # Preprocessed (detrend+bandpass)
        out_dir = Path(__file__).resolve().parent / "outputs" / participant
        prep_path = out_dir / f"alignment_windows_{participant}_{dev}_preprocess.npz"
        if prep_path.is_file():
            with np.load(prep_path, allow_pickle=True) as z:
                ppg_bp = np.asarray(z[ch]) if ch in z.files else np.asarray(z["ppg_green"])
        else:
            # Generate on-the-fly
            ppg_bp = preprocess_ppg(ppg_raw, fs)

        n = ppg_raw.shape[0]
        print(f"  {dev} ({ch}, n={n})")

        for step_name, kwargs in steps:
            # Steps 0-2 use bandpass'd signal; Step 7+ also bandpass
            mae, valid, total = run_step(step_name, ppg_bp, fs, rr_all, n_rr, n, **kwargs)
            cov = valid / total * 100 if total > 0 else 0
            print(f"    {step_name:30s}  MAE={mae:8.2f}  coverage={cov:5.1f}%  n={valid}/{total}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
