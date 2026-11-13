import numpy as np
MAX_GAP_MS=250.


def technical_reasons(times, signal, start_ms, end_ms):
    t, x = np.asarray(times, float), np.asarray(signal, float)
    reasons = []
    if len(t) != len(x) or len(t) < 50:
        return ["insufficient_or_mismatched_samples"]
    if not np.isfinite(t).all() or not np.isfinite(x).all():
        return ["nonfinite_time_or_signal"]
    if np.any(np.diff(t) <= 0):
        reasons.append("duplicate_or_nonmonotonic_timestamps")
    if np.any(np.diff(t) > MAX_GAP_MS):
        reasons.append("sample_gap_over_250ms")
    if t[0] - start_ms > MAX_GAP_MS or end_ms - t[-1] > MAX_GAP_MS:
        reasons.append("incomplete_context")
    return reasons


class CentralGate:
    def __init__(self,t,raw):
        self.t=t;self.raw=raw;self.bounds=np.r_[0,np.flatnonzero(np.diff(t)>250)+1,len(t)];self.cache={}
    def assess(self,start,end):
        t,x=self.t,self.raw;a,b=np.searchsorted(t,[start,end]);a=int(a);b=int(np.searchsorted(t,end,'right'))
        issues=technical_reasons(t[a:b],x[a:b],start,end)
        result=dict(reasons=issues,assessable=False,severe_blocks=0,max_span_uv=np.nan)
        if issues:return result
        run=int(np.searchsorted(self.bounds,a,'right')-1);lo,hi=self.bounds[run:run+2];rt=t[lo:hi];origin=rt[0];spans=[];assess=True
        for block in range(max(0,int(np.floor((start-origin)/5000))),min(int(np.floor((end-origin)/5000)),int(np.floor((rt[-1]-origin)/5000)))+1):
            key=(run,block)
            if key not in self.cache:
                ia,ib=np.searchsorted(rt,[origin+block*5000,origin+(block+1)*5000]);v=x[lo+ia:lo+ib];ok=len(v)>=30 and np.isfinite(v).all();span=float(np.quantile(v,.99)-np.quantile(v,.01)) if ok else np.nan;self.cache[key]=(ok,span)
            ok,span=self.cache[key];assess &=ok;spans.append(span)
        assess &= bool(spans);bad=sum(np.isfinite(v) and v>5000 for v in spans)
        if not assess:issues.append('G1_unassessable_source_block')
        if bad:issues.append('G1_severe_source_block')
        result.update(assessable=bool(assess),severe_blocks=int(bad),max_span_uv=float(np.nanmax(spans)) if spans else np.nan)
        return result


def infer_fs(times_ms: np.ndarray) -> float:
    steps = np.diff(np.asarray(times_ms, dtype=float))
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if not len(steps):
        raise ValueError("cannot infer ECG sampling frequency")
    return float(1000.0 / np.median(steps))

def refine_to_local_qrs_extreme(signal: np.ndarray, times_ms: np.ndarray, indices: np.ndarray, fs: float) -> np.ndarray:
    """Snap each detector event to its local dominant QRS extreme within ±50 ms.

    Unlike the legacy global-polarity refinement, polarity is chosen per event.
    This avoids moving an upright R apex to its following S trough when another
    recording segment happens to have larger negative excursions.
    """
    x = np.asarray(signal, dtype=float)
    radius = max(1, int(round(0.050 * fs)))
    refined: list[int] = []
    for index in np.asarray(indices, dtype=int):
        left, right = max(0, index - radius), min(len(x), index + radius + 1)
        segment = x[left:right]
        if not len(segment):
            continue
        baseline = float(np.median(segment))
        upward = left + int(np.argmax(segment))
        downward = left + int(np.argmin(segment))
        refined.append(upward if abs(x[upward] - baseline) >= abs(x[downward] - baseline) else downward)
    return times_ms[np.unique(np.asarray(refined, dtype=int))] if refined else np.empty(0, dtype=float)

from pathlib import Path
import csv,json
from hrv.common.rr import compute
from .context import PaddingGuard,select

class ECGReference:
    """Detect from native ECG; reference inputs contain only reviewed decisions."""
    def __init__(self,participant,raw,reference_dir):
        self.participant=participant
        self.t=np.asarray(raw['time_ms'],float)
        self.raw=np.asarray(raw['ecg_uv'],float)
        finite=np.flatnonzero(np.isfinite(self.t))
        if not len(finite):raise ValueError('ECG has no finite timestamps')
        sl=slice(finite[0],finite[-1]+1);self.t=self.t[sl];self.raw=self.raw[sl]
        ref=json.loads((Path(reference_dir)/'reference.json').read_text())
        self.ref=ref['participants'][participant]
        self.exclusions=[r for r in ref['source_exclusions'] if r['participant']==participant]
        self.acquisition=self.raw.copy();self.corrected=self.raw.copy()
        for r in self.ref['sign_intervals']:
            a=np.searchsorted(self.t,r['start_ms'],'left');b=np.searchsorted(self.t,r['end_ms'],'right')
            if r['acquisition_flip']:self.acquisition[a:b]*=-1
            self.corrected[a:b]=self.acquisition[a:b]*(-1 if r['target_flip'] else 1)
        self.central=CentralGate(self.t,self.acquisition)
        self.padding=PaddingGuard(self.t,self.acquisition)
        self.fs0=infer_fs(self.t)
        with (Path(reference_dir)/'latest_saved_manual_unreadable_intervals.csv').open() as f:
            self.unreadable=[(float(r['start_ms']),float(r['end_ms'])) for r in csv.DictReader(f) if r['participant']==participant]

    def detect(self,start,end,fs):
        import neurokit2 as nk
        a=np.searchsorted(self.t,start,'left');b=np.searchsorted(self.t,end,'right')
        clean=nk.ecg_clean(self.corrected[a:b],sampling_rate=fs,method='neurokit')
        _,info=nk.ecg_peaks(clean,sampling_rate=fs,method='rodrigues2021')
        ids=np.asarray(info.get('ECG_R_Peaks',[]),int)
        ids=ids[(ids>=0)&(ids<b-a)]
        return refine_to_local_qrs_extreme(self.corrected[a:b],self.t[a:b],ids,fs)

    def label(self,start,end):
        a=np.searchsorted(self.t,start,'left');b=np.searchsorted(self.t,end,'right')
        reasons=self.central.assess(start,end)['reasons'].copy()
        ratio=min(1.,np.isfinite(self.acquisition[a:b]).sum()/max(1,round((end-start)*self.fs0/1000)))
        if ratio<1.:reasons.append('low_ecg_sample_ratio')
        if any(r['start_ms']<=end and r['end_ms']>=start for r in self.exclusions):reasons.append('confirmed_source_exclusion')
        if any(l<=end and r>=start for l,r in self.unreadable):reasons.append('G2_saved_manual_unreadable_interval')
        manual=[r for r in self.ref['manual_rejections'] if r['start_ms']==start and r['end_ms']==end]
        if manual:reasons.extend(manual[0]['reason'].split('|'))
        exception=next((r for r in self.ref['full300_exceptions'] if r['start_ms']==start and r['end_ms']==end),None)
        peaks=np.empty(0);route='source_rejected'
        if exception:
            # Only the four explicitly reviewed windows may bypass amplitude G1.
            reasons=[r for r in reasons if r not in {'G1_severe_source_block','G1_unassessable_source_block'}]
            if not reasons:peaks=np.asarray(exception['peaks'],float);route='manual_full300_exception'
        elif not reasons:
            try:
                fs=infer_fs(self.t[a:b]);original=self.detect(start,end,fs)
                pa=np.searchsorted(self.t,start-10000,'left');pb=np.searchsorted(self.t,end+10000,'right')
                technical=not technical_reasons(self.t[pa:pb],self.acquisition[pa:pb],start-10000,end+10000)
                if any(r['start_ms']<=end+10000 and r['end_ms']>=start-10000 for r in self.exclusions):technical=False
                guard=bool(self.padding.assess(start,end)['pass_guard']) if technical else False
                padded=self.detect(start-10000,end+10000,fs) if technical and guard else np.empty(0)
                padded=padded[(padded>=start)&(padded<=end)]
                peaks,decision=select(original,padded,start,end,technical,guard,True);route=decision['route']
                for r in self.ref['patches']:
                    if r['start_ms']!=start or r['end_ms']!=end:continue
                    left,right=r['left_anchor_ms'],r['right_anchor_ms']
                    replacement=np.asarray(r['manual_peak_times_ms'],float)
                    # Exact human anchor identity is an essential label condition.
                    if not np.any(peaks==left) or not np.any(peaks==right):
                        reasons.append('manual_patch_anchor_mismatch');continue
                    peaks=np.unique(np.r_[peaks[peaks<left],replacement,peaks[peaks>right]])
            except (ValueError, IndexError, ZeroDivisionError, FloatingPointError, RuntimeError) as exc:
                peaks=np.empty(0)
                reasons.append('detector_failure')
                route=f'detector_failure:{type(exc).__name__}:{exc}'

        stats=compute(peaks)
        reasons=list(dict.fromkeys(reasons+stats['reasons']))
        passed=not reasons
        ids=np.searchsorted(self.t,peaks)
        if len(ids) and (np.any(ids>=len(self.t)) or not np.array_equal(self.t[ids],peaks)):
            raise ValueError('ECG reference peak is not a native source sample')
        out=dict(ecg_label_qc_pass=passed,ecg_label_qc_reason='|'.join(reasons),ecg_startup_route=route,
            ecg_r_peak_times_abs_ms=peaks,ecg_r_peak_times_rel_ms=(peaks-start).astype(np.float32),
            ecg_r_peak_indices_grid=np.rint((peaks-start)/10).astype(np.int32),
            ecg_r_peak_amplitudes_raw=self.acquisition[ids].astype(np.float32),
            ecg_r_peak_amplitudes_corrected=self.corrected[ids].astype(np.float32),
            ecg_rr_intervals_uncorrected_ms=stats['rr'].astype(np.float32),
            ecg_rr_intervals_corrected_ms=stats['corr'].astype(np.float32),
            ecg_valid_sample_ratio=ratio,ecg_valid_ibi_ratio=stats['valid_ratio'],ecg_ibi_correction_ratio=stats['correction_ratio'],ecg_peak_count=len(peaks))
        for state in ['uncorrected','corrected']:
            for name,value in stats[state].items():
                key=f'ecg_{name}_{state}_ms' if name in {'rmssd','sdnn','mean_ibi'} else f'ecg_{name}_{state}'
                out[key]=np.float32(value if passed else np.nan)
        return out
