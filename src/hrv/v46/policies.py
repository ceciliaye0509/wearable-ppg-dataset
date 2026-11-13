import numpy as np
GAP_MS=100.
PLATEAU_MS=2000.
RAIL_MS=20.
RAIL=262143.


def canonical_source(t,values):
 """Stable global order, deterministic dedup, per-channel conflict masking.

 Equal optical observations at one timestamp collapse to one. Conflicting
 optical observations are marked missing instead of inventing an average.
 Raw order/clock units are preserved in the untouched source and audit ledger.
 """
 t=np.asarray(t,float);finite=np.isfinite(t);order=np.argsort(t[finite],kind='stable');st=t[finite][order]
 first=np.r_[True,np.diff(st)>0];ix=np.flatnonzero(first);ut=st[ix];counts=np.diff(np.r_[ix,len(st)])
 result=[];bad=[];events=[]
 for ci,x in enumerate(values):
  sx=np.asarray(x,float)[finite][order];v=sx[ix].copy();conflict=np.zeros(len(ut),bool)
  for j in np.flatnonzero(counts>1):
   obs=sx[ix[j]:ix[j]+counts[j]];f=obs[np.isfinite(obs)]
   if not len(f) or np.any(f!=f[0]):conflict[j]=True;v[j]=np.nan
   else:v[j]=f[0]
  invalid=conflict|~np.isfinite(v);known_rail=bool(np.any(np.isfinite(v)) and np.nanmax(v)==RAIL)
  connected=np.isfinite(v[1:])&np.isfinite(v[:-1])&(np.diff(ut)<=GAP_MS)&(v[1:]==v[:-1])
  edges=np.r_[0,np.flatnonzero(~connected)+1,len(v)];duration=ut[edges[1:]-1]-ut[edges[:-1]]
  for j in np.flatnonzero((duration>=PLATEAU_MS)|((v[edges[:-1]]==RAIL)&known_rail&(duration>=RAIL_MS))):
   a,b=edges[j:j+2];invalid[a:b]=True
   events.append(dict(channel_index=ci,start_ms=float(ut[a]),end_ms=float(ut[b-1]),samples=int(b-a),value=float(v[a]),duration_ms=float(duration[j]),reason='constant_at_least2s' if duration[j]>=PLATEAU_MS else 'observed262143_ceiling_plateau_at_least20ms'))
  for j in np.flatnonzero(conflict):events.append(dict(channel_index=ci,start_ms=float(ut[j]),end_ms=float(ut[j]),samples=int(counts[j]),value=np.nan,duration_ms=0.,reason='conflicting_values_at_same_timestamp'))
  result.append(v);bad.append(invalid)
 return ut,result,bad,events,dict(raw_samples=len(t),nonfinite_time=int((~finite).sum()),backward_steps=int((np.diff(t[finite])<0).sum()),duplicate_excess=int((counts-1).sum()),duplicate_time_groups=int((counts>1).sum()))

def bounded_resample(x,t,grid,max_gap_ms=GAP_MS):
 """No extrapolation; never interpolate between anchors >100ms apart."""
 good=np.isfinite(x)&np.isfinite(t);tt=t[good];xx=x[good];out=np.full(len(grid),np.nan,np.float32);mask=np.zeros(len(grid),bool)
 if len(tt)<2:return out,mask,0.
 right=np.searchsorted(tt,grid);r=np.minimum(right,len(tt)-1);l=np.maximum(right-1,0)
 exact=(right<len(tt))&(tt[r]==grid)
 inside=(right>0)&(right<len(tt));short=(tt[r]-tt[l])<=max_gap_ms
 mask=exact|(inside&short);out[mask]=np.interp(grid[mask],tt,xx).astype(np.float32)
 return out,mask,float(mask.mean())

def support_gap(t,mask,grid):
 v=np.sort(t[mask&np.isfinite(t)])
 if len(v)<2:return float('inf')
 return float(max(np.max(np.diff(v)),max(0.,v[0]-grid[0]),max(0.,grid[-1]-v[-1])))

def flat(x,mask):
 v=np.asarray(x,float)[mask&np.isfinite(x)]
 if len(v)<2:return False
 low,high=v.min(),v.max();return bool(high-low<=64*np.finfo(float).eps*max(abs(low),abs(high),np.finfo(float).tiny))


def segment_support(t, grid, cap, guard_ms=3000.):
    """Split on long gaps; reserve 3 s on both sides of each internal new boundary."""
    if len(t) < 2:
        return [], np.zeros(len(grid), bool)
    cuts = np.r_[0, np.flatnonzero(np.diff(t) > cap)+1, len(t)]
    segments = []; retained = np.zeros(len(grid), bool)
    for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        lo, hi = float(t[a]), float(t[b-1])
        # Outer window edges keep historical treatment; newly created/long-missing
        # edges need a guard. An incomplete outer edge >cap is also a new boundary.
        left_guard = guard_ms if k > 0 or lo-grid[0] > cap else 0.
        right_guard = guard_ms if k < len(cuts)-2 or grid[-1]-hi > cap else 0.
        inside = (grid >= lo) & (grid <= hi)
        valid = inside & (grid >= lo+left_guard) & (grid <= hi-right_guard)
        if valid.any() and inside.sum() >= 100:
            segments.append(dict(a=int(a), b=int(b), lo=lo, hi=hi,
                                 keep_lo=lo+left_guard, keep_hi=hi-right_guard))
            retained |= valid
    return segments, retained
