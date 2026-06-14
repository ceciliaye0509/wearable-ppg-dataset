"""Quick P7-only pipeline: HRV + eval + scatter, collecting metrics."""
import sys, subprocess, json
from pathlib import Path

VENV_PY = "/Users/jiqiyu/Desktop/Daily_HRV/venv/bin/python"
BASE = Path(__file__).resolve().parent

def run(script):
    subprocess.run([VENV_PY, script], cwd=str(BASE), check=True)

def collect_metrics():
    """Return dict of eval summary metrics for P7."""
    import pandas as pd
    results = {}
    out = BASE / "outputs" / "P7"
    for f in sorted(out.glob("ppg_vs_ecg_*_summary.csv")):
        name = f.stem.replace("ppg_vs_ecg_P7_", "").replace("_summary", "")
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
    import shutil
    # Clear old eval outputs (keep HRV CSVs if --eval-only)
    out = BASE / "outputs" / "P7"
    eval_only = "--eval-only" in sys.argv
    if not eval_only:
        # Remove HRV CSVs + eval CSVs
        for f in out.glob("hrv_*.csv"):
            f.unlink()
    for f in out.glob("ppg_vs_ecg_*"):
        f.unlink()

    if not eval_only:
        run("hrv_runner.py")
    run("eval_ppg_vs_ecg.py")

    metrics = collect_metrics()
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    # Print key metrics
    for key in ["Earring_ppg_ir/HRV_RMSSD", "Earring_ppg_ir/HRV_SDNN",
                 "Earring_ppg_green/HRV_RMSSD", "Earring_ppg_green/HRV_SDNN",
                 "Earring_ppg_ir/hr_mean", "Earring_ppg_green/hr_mean",
                 "Ring_ppg_green/HRV_RMSSD", "Necklace_ppg_green/HRV_RMSSD",
                 "Watch_ppg_green/HRV_RMSSD"]:
        if key in metrics:
            m = metrics[key]
            print(f"  {key:40s}  MAE={m['mae']:8.2f}  r={m['r']:.4f}  bias={m['bias']:+.2f}")

    # Save to JSON
    json_path = BASE / "outputs" / "P7" / "metrics_snapshot.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n[SAVED] {json_path}")

    # Generate scatter plots
    from summarize_hrv import plot_scatter
    plot_scatter("ppg_ir")
    plot_scatter("ppg_green")
    print("\n[DONE]")
