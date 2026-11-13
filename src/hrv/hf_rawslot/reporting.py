import numpy as np
import pandas as pd

def _primary_channel_summary(selected: pd.DataFrame) -> pd.DataFrame:
    """Summarize the two production methods at participant/device/channel level."""
    rows: list[dict[str, object]] = []
    for keys, group in selected.groupby(["baseline_method", "participant", "device", "channel"], dropna=False):
        pred = group["ppg_rmssd_ms"].to_numpy(float)
        ref = group["ecg_rmssd_ms"].to_numpy(float)
        sdnn = group["ppg_sdnn_ms"].to_numpy(float)
        valid = np.isfinite(pred) & np.isfinite(ref) & np.isfinite(sdnn)
        pred, ref = pred[valid], ref[valid]
        n_total = int(group.shape[0])
        n_valid = int(pred.size)
        rows.append({
            "baseline_method": keys[0],
            "participant": keys[1],
            "device": keys[2],
            "channel": keys[3],
            "n_total": n_total,
            "n_valid": n_valid,
            "coverage_pct": 100.0 * n_valid / n_total if n_total else np.nan,
            "MAE": float(np.mean(np.abs(pred - ref))) if n_valid else np.nan,
            "R": float(np.corrcoef(pred, ref)[0, 1]) if n_valid >= 3 and np.std(pred) > 0 and np.std(ref) > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["baseline_method", "participant", "device", "channel"])

def _primary_cohort_summary(channel: pd.DataFrame) -> pd.DataFrame:
    """Each participant and each of its six device/channel results have equal weight."""
    participant = (
        channel.groupby(["baseline_method", "participant"], dropna=False)
        .agg(
            channels=("channel", "size"),
            mean_MAE=("MAE", "mean"),
            mean_R=("R", "mean"),
            mean_coverage_pct=("coverage_pct", "mean"),
            min_channel_coverage_pct=("coverage_pct", "min"),
        )
        .reset_index()
    )
    cohort = (
        participant.groupby("baseline_method", dropna=False)
        .agg(
            participants=("participant", "size"),
            mean_participant_MAE=("mean_MAE", "mean"),
            median_participant_MAE=("mean_MAE", "median"),
            mean_participant_R=("mean_R", "mean"),
            median_participant_R=("mean_R", "median"),
            mean_participant_coverage_pct=("mean_coverage_pct", "mean"),
            min_channel_coverage_pct=("min_channel_coverage_pct", "min"),
        )
        .reset_index()
        .sort_values("mean_participant_MAE")
    )
    return participant.sort_values(["baseline_method", "participant"]), cohort

def write_reports(records,output_dir):
    if not records:return
    df=pd.DataFrame(records).rename(columns={'pred_rmssd_ms':'ppg_rmssd_ms','pred_sdnn_ms':'ppg_sdnn_ms'})
    df['baseline_method']=df['method'].map({'qppgfast':'qppgfast_devicewise_v1','scipy':'scipy_peak_v1'})
    channel=_primary_channel_summary(df)
    participant,cohort=_primary_cohort_summary(channel)
    channel.to_csv(output_dir/'historical_channel_summary.csv',index=False)
    participant.to_csv(output_dir/'historical_participant_summary.csv',index=False)
    cohort.to_csv(output_dir/'historical_cohort_summary.csv',index=False)
