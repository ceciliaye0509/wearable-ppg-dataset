"""Run pipeline for a specific participant: HRV + eval + metrics.

Usage:  python bench_participant.py P5
        python bench_participant.py P5 --baseline
"""
import sys, os, subprocess, json
from pathlib import Path

VENV_PY = "/Users/jiqiyu/Desktop/Daily_HRV/venv/bin/python"
BASE = Path(__file__).resolve().parent


def run(script, participant, baseline=False):
    """Run a sub-script with PARTICIPANT env var set."""
    env = os.environ.copy()
    env["PARTICIPANT"] = participant
    if baseline:
        env["BASELINE"] = "1"
    subprocess.run([VENV_PY, script], cwd=str(BASE), check=True, env=env)


def collect_metrics(participant):
    """Return dict of eval summary metrics for given participant."""
    import pandas as pd
    results = {}
    out = BASE / "outputs" / participant
    for f in sorted(out.glob("ppg_vs_ecg_*_summary.csv")):
        name = f.stem.replace(f"ppg_vs_ecg_{participant}_", "").replace("_summary", "")
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            key = f"{name}/{r['metric']}"
            results[key] = {
                "r": round(r["r"], 4),
                "mae": round(r["mae"], 2),
                "bias": round(r["bias"], 2),
                "n": int(r["n"]),
            }
    return results


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    participant = args[0] if args else "P7"
    baseline = "--baseline" in sys.argv

    mode = "BASELINE" if baseline else "OPTIMIZED"
    print(f"[bench] Running {mode} pipeline for {participant}")

    out = BASE / "outputs" / participant
    out.mkdir(parents=True, exist_ok=True)

    # Clear old outputs
    for f in out.glob("hrv_*.csv"):
        f.unlink()
    for f in out.glob("ppg_vs_ecg_*"):
        f.unlink()

    # Run HRV + eval
    run("hrv_runner.py", participant, baseline=baseline)
    run("eval_ppg_vs_ecg.py", participant)

    # Collect and print metrics
    metrics = collect_metrics(participant)
    print("\n" + "=" * 60)
    print(f"METRICS SUMMARY — {participant} ({mode})")
    print("=" * 60)
    for key in sorted(metrics.keys()):
        if "HRV_RMSSD" in key or "hr_mean" in key:
            m = metrics[key]
            print(f"  {key:45s}  MAE={m['mae']:8.2f}  r={m['r']:.4f}  bias={m['bias']:+.2f}  n={m['n']}")

    # Save to JSON
    suffix = "_baseline" if baseline else ""
    json_path = out / f"metrics_snapshot{suffix}.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n[SAVED] {json_path}")
    print(f"\n[DONE] {participant} ({mode})")
