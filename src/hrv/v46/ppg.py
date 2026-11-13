"""The fixed segment250 policy, independent of another version's reader."""
import numpy as np
from hrv.common.signals import _rawslots_to_grid
from .policies import canonical_source,bounded_resample,flat,support_gap,segment_support


def source(device):
    t,x,bad,events,info=canonical_source(device['timestamp'],[device['ppg_green'],device['ppg_ir']])
    for i in range(2):x[i][bad[i]]=np.nan
    return t,x


def channel_plan(times,values,grid):
    a=np.searchsorted(times,grid[0]-500,'left');b=np.searchsorted(times,grid[-1]+510,'right')
    raw,observed,raw_ts,ratio=_rawslots_to_grid(values[a:b],times[a:b],grid,slot_tolerance_ms=5.)
    # A source-flat window must stay missing after source QC as in the frozen arm.
    if flat(raw,observed):raw[:]=np.nan;observed[:]=False;raw_ts[:]=np.nan
    valid=observed&np.isfinite(raw)&np.isfinite(raw_ts)
    t,x=raw_ts[valid],raw[valid]
    order=np.argsort(t,kind='stable');t,x=t[order],x[order]
    unique=np.r_[True,np.diff(t)>0] if len(t) else np.zeros(0,bool);t,x=t[unique],x[unique]
    gap=support_gap(raw_ts,observed,grid)
    segments,keep=segment_support(t,grid,250.)
    usable=len(t)>=2 and observed.mean()>=.9 and not flat(raw,observed) and keep.mean()>=.9
    full=np.full(len(grid),np.nan,np.float32);mask=np.zeros(len(grid),bool)
    model_segments=[];model_keep=np.zeros(len(grid),bool)
    if gap<=250.:
        full,mask,_=bounded_resample(values[a:b],times[a:b],grid,max_gap_ms=250.)
        model_segments=segments;model_keep=keep.copy();model_usable=bool(usable and mask.mean()>=.95)
    else:
        for s in segments:
            inside=(grid>=s['lo'])&(grid<=s['hi']);g=grid[inside]
            left=np.searchsorted(times,s['lo'],'left');right=np.searchsorted(times,s['hi'],'right')
            y,good,_=bounded_resample(values[left:right],times[left:right],g,max_gap_ms=250.)
            full[inside]=y;mask[inside]=good
            if good.mean()>=.95:
                model_segments.append(s);model_keep|=(grid>=s['keep_lo'])&(grid<=s['keep_hi'])
        model_usable=bool(usable and model_keep.mean()>=.9)
    return dict(raw=raw,observed=observed,raw_timestamp=raw_ts,resampled=full,resampled_mask=mask,
        baseline_keep=keep,model_keep=model_keep,baseline_usable=bool(usable),model_usable=model_usable,
        baseline_segments=segments,model_segments=model_segments,requires_split=bool(gap>250.))
