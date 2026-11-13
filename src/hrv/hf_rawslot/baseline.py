"""Published raw-slot baseline: strict100ms, frozen qppg device fiducials."""
import numpy as np
from hrv.common.ppg import detect
from hrv.common.rr import _correct_ibi_artifacts_with_ratio
def _strict_interp100_channel(
    arrays: dict[str, np.ndarray],
    wi: int,
    di: int,
    ci: int,
    *,
    max_gap_ms: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Build a uniform grid without extrapolating beyond raw PPG support.

    A window is rejected when an internal raw-sample gap, or either boundary
    support gap, exceeds ``max_gap_ms``. The returned grid is the supported
    part of the 100 Hz window, so every detector sees a genuinely uniform
    sequence and no value is extrapolated at the boundaries.
    """
    grid_t = np.asarray(arrays["ppg_grid_timestamp_ms"][wi], dtype=np.float64)
    values = np.asarray(arrays["ppg_rawslot_values"][wi, di, ci], dtype=np.float64)
    timestamps = np.asarray(arrays["ppg_rawslot_timestamp_ms"][wi, di, ci], dtype=np.float64)
    mask = (
        np.asarray(arrays["ppg_rawslot_mask"][wi, di, ci], dtype=bool)
        & np.isfinite(values)
        & np.isfinite(timestamps)
    )
    raw_ratio = float(arrays["ppg_rawslot_valid_sample_ratio"][wi, di, ci])
    if int(mask.sum()) < 2 or grid_t.size < 50:
        return np.empty(0), np.empty(0), 100.0, raw_ratio, float("inf")

    raw_t = timestamps[mask]
    raw_x = values[mask]
    order = np.argsort(raw_t)
    raw_t = raw_t[order]
    raw_x = raw_x[order]
    unique = np.r_[True, np.diff(raw_t) > 0]
    raw_t = raw_t[unique]
    raw_x = raw_x[unique]
    if raw_t.size < 2:
        return np.empty(0), np.empty(0), 100.0, raw_ratio, float("inf")

    internal_gaps = np.diff(raw_t)
    leading_gap = max(0.0, float(raw_t[0] - grid_t[0]))
    trailing_gap = max(0.0, float(grid_t[-1] - raw_t[-1]))
    max_support_gap = float(max(np.max(internal_gaps), leading_gap, trailing_gap))
    if not np.isfinite(max_support_gap) or max_support_gap > max_gap_ms:
        return np.empty(0), np.empty(0), 100.0, raw_ratio, max_support_gap

    inside = (grid_t >= raw_t[0]) & (grid_t <= raw_t[-1])
    uniform_t = grid_t[inside]
    if uniform_t.size < 50:
        return np.empty(0), np.empty(0), 100.0, raw_ratio, max_support_gap
    uniform_x = np.interp(uniform_t, raw_t, raw_x)
    return uniform_x, uniform_t, 100.0, raw_ratio, max_support_gap

def predict(row,di,ci,detector="qppgfast"):
    arrays={k:np.asarray(v)[None] for k,v in row.items()}
    x,t,*_=_strict_interp100_channel(arrays,0,di,ci,max_gap_ms=100.)
    if len(x)<50:return dict(rmssd_ms=np.nan,sdnn_ms=np.nan)
    peaks=detect(x,t,str(row['devices'][di]),detector=detector);rr=np.diff(peaks)
    nn=rr[np.isfinite(rr)&(rr>=300)&(rr<=2000)]
    if len(nn)<3:return dict(rmssd_ms=np.nan,sdnn_ms=np.nan)
    corrected,_=_correct_ibi_artifacts_with_ratio(nn,threshold=.2)
    return dict(rmssd_ms=float(np.sqrt(np.mean(np.diff(corrected)**2))),sdnn_ms=float(np.std(corrected,ddof=1)))

if __name__=='__main__':
    from hrv.common.evaluate import baseline_main
    from hrv.hf_rawslot.reporting import write_reports
    baseline_main(predict,reporter=write_reports)
