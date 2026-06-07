"""Quick sanity check: verify every NPZ can be opened and has expected keys."""
import os, sys, numpy as np
from pathlib import Path

EXPECTED_KEYS = {
    "t0_ms", "t1_ms", "ecg", "ecg_valid_len", "ppg_fs",
    "ppg_ir", "ppg_green", "ppg_t_ms",
    "accel_x", "accel_y", "accel_z",
    "hr_gt", "n_peaks",
    "r_peak_samples", "rr_intervals_ms", "n_rr",
}

root = Path(__file__).resolve().parent
ok = fail = 0

for npz_path in sorted(root.rglob("*.npz")):
    try:
        z = np.load(npz_path, allow_pickle=True)
        keys = set(z.files)
        missing = EXPECTED_KEYS - keys
        n = z["t0_ms"].shape[0]
        if missing:
            print(f"[WARN] {npz_path.relative_to(root)}: missing keys {missing}")
            fail += 1
        else:
            print(f"[OK]   {npz_path.relative_to(root)}: {n} windows, {len(keys)} keys")
            ok += 1
        z.close()
    except Exception as e:
        print(f"[FAIL] {npz_path.relative_to(root)}: {e}")
        fail += 1

print(f"\n{'='*40}")
print(f"Total: {ok + fail} files, {ok} OK, {fail} FAIL")
if fail == 0:
    print("All checks passed!")
else:
    print(f"WARNING: {fail} file(s) had issues!")
    sys.exit(1)
