"""
One-command HRV pipeline.

Runs the whole thing in the correct order so you don't have to remember it:

    1. (optional) clear stale preprocess caches  -- on by default when --fresh
    2. hrv_runner.py   : PPG -> HRV per window      (outputs/<Px>/hrv_*.csv)
    3. eval_ppg_vs_ecg : PPG HRV vs ECG gold-std    (outputs/<Px>/ppg_vs_ecg_*.csv)

Usage (from src/heuristic_baselines/):

    python run_all.py                # full pipeline, fresh caches
    python run_all.py --no-eval      # only compute PPG HRV, skip ECG comparison
    python run_all.py --keep-cache   # reuse existing *_preprocess.npz (faster reruns)

All participants / devices / channels come from config.py, same as before.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
import config  # noqa: E402


def _clear_caches() -> None:
    """Delete stale *_preprocess.npz so a data/window change can't be masked."""
    result_root = config.HEURISTIC_RESULT_ROOT
    removed = 0
    if result_root.is_dir():
        for f in result_root.glob("*/*_preprocess.npz"):
            try:
                f.unlink(); removed += 1
            except OSError:
                pass
    print(f"[run_all] cleared {removed} cached *_preprocess.npz file(s)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full HRV pipeline in one command.")
    ap.add_argument("--no-eval", action="store_true", help="skip PPG-vs-ECG comparison")
    ap.add_argument("--keep-cache", action="store_true", help="reuse existing preprocess caches")
    args = ap.parse_args()

    t_start = time.time()
    print("=" * 64)
    print(f"[run_all] data_source={config.HEURISTIC_DATA_SOURCE!r}  root={config.HEURISTIC_WINDOWS_ROOT}")
    print(f"[run_all] participants={list(config.HEURISTIC_PIPELINE_PARTICIPANTS)} "
          f"devices={list(config.HEURISTIC_DEVICE_ROLES)}")
    print("=" * 64)

    # Step 1: caches
    if args.keep_cache:
        print("[run_all] keeping existing preprocess caches (--keep-cache)")
    else:
        _clear_caches()

    # Step 2: PPG -> HRV
    print("\n" + "#" * 64 + "\n# STEP 1/2: computing PPG HRV (hrv_runner)\n" + "#" * 64)
    import hrv_runner
    hrv_runner.main()

    # Step 3: PPG vs ECG (optional)
    if args.no_eval:
        print("\n[run_all] skipping PPG-vs-ECG comparison (--no-eval)")
    else:
        print("\n" + "#" * 64 + "\n# STEP 2/2: PPG vs ECG comparison (eval_ppg_vs_ecg)\n" + "#" * 64)
        try:
            import eval_ppg_vs_ecg
            old_argv = sys.argv[:]
            try:
                sys.argv = [sys.argv[0]]
                eval_ppg_vs_ecg.main()
            finally:
                sys.argv = old_argv
        except FileNotFoundError as e:
            print(f"[run_all] eval skipped: {e}")
        except ImportError:
            print("[run_all] eval_ppg_vs_ecg.py not found -- skipping comparison")

    mins = (time.time() - t_start) / 60.0
    print("\n" + "=" * 64)
    print(f"[run_all] ALL DONE in {mins:.1f} min. Results in {config.HEURISTIC_RESULT_ROOT}")
    print("=" * 64)


if __name__ == "__main__":
    main()
