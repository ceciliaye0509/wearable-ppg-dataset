import numpy as np
IBI_MIN_MS=300.0
IBI_MAX_MS=2000.0


def _correct_ibi_artifacts_with_ratio(
    ibi_ms: np.ndarray,
    threshold: float = 0.20,
) -> tuple[np.ndarray, float]:
    """与 _correct_ibi_artifacts 相同，额外返回被替换的 IBI 比例。"""
    if ibi_ms.size < 5:
        return ibi_ms.copy(), 0.0
    from scipy.ndimage import median_filter

    corrected = ibi_ms.copy()
    med = median_filter(ibi_ms, size=11, mode="reflect")
    outlier = np.abs(corrected - med) > threshold * med
    corrected[outlier] = med[outlier]
    return corrected, float(np.mean(outlier))


def stat(rr):
 if len(rr)<3:return dict(rmssd=np.nan,sdnn=np.nan,hr_bpm=np.nan,mean_ibi=np.nan,rr_cv=np.nan)
 mean=float(np.mean(rr));sd=float(np.std(rr,ddof=1))
 return dict(rmssd=float(np.sqrt(np.mean(np.diff(rr)**2))),sdnn=sd,hr_bpm=60000/mean,mean_ibi=mean,rr_cv=sd/mean)

def compute(peaks):
 rr=np.diff(peaks);valid=(rr>=300)&(rr<=2000);idx=np.flatnonzero(valid)
 parts=[rr[v] for v in np.split(idx,np.flatnonzero(np.diff(idx)>1)+1) if len(v)]
 pairs=[_correct_ibi_artifacts_with_ratio(v) for v in parts]
 raw=np.concatenate(parts) if parts else np.empty(0);corr=np.concatenate([v for v,_ in pairs]) if pairs else np.empty(0)
 ratio=sum(len(v)*r for v,r in pairs)/len(corr) if len(corr) else np.nan
 un,co=stat(raw),stat(corr)
 if len(parts)>1:
  for chunks,s in [(parts,un),([v for v,_ in pairs],co)]:
   diffs=np.concatenate([np.diff(v) for v in chunks if len(v)>1]) if any(len(v)>1 for v in chunks) else np.empty(0)
   s['rmssd']=float(np.sqrt(np.mean(diffs**2))) if len(diffs) else np.nan
 reasons=[];vr=float(valid.mean()) if len(rr) else 0.
 if vr<1:reasons.append('invalid_ibi')
 if len(peaks)<200:reasons.append('too_few_peaks')
 if len(peaks)>1200:reasons.append('too_many_peaks')
 if not np.isfinite(ratio) or ratio>.20:reasons.append('high_or_invalid_ibi_correction_ratio')
 if not np.isfinite(co['hr_bpm']) or not 40<=co['hr_bpm']<=200:reasons.append('hr_out_of_range')
 if not np.isfinite(co['rmssd']) or co['rmssd']>200:reasons.append('invalid_or_high_rmssd')
 return dict(rr=rr,corr=corr,uncorrected=un,corrected=co,valid_ratio=vr,correction_ratio=ratio,reasons=reasons,bad_RR=int((~valid).sum()))
