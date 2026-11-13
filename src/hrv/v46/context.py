import numpy as np
PAD_MS=10000.
STARTUP_MS=5000.
BLOCK_MS=5000.
SEVERE_UV=5000.


class PaddingGuard:
 def __init__(self,times,raw):
  self.t=np.asarray(times,float);self.x=np.asarray(raw,float)
  self.runs=np.r_[0,np.flatnonzero(np.diff(self.t)>250)+1,len(self.t)]
  self.cache={}

 def assess(self,start,end):
  evidence={}
  # Inspect full, existing run-anchored source blocks touching ONLY the added
  # context. Boundary-overlapping blocks deliberately follow the old G1 policy.
  for left,right in [(start-PAD_MS,start),(end,end+PAD_MS)]:
   ia,ib=np.searchsorted(self.t,left,'left'),np.searchsorted(self.t,right,'right')
   if ib<=ia:continue
   run0=int(np.searchsorted(self.runs,ia,'right')-1)
   run1=int(np.searchsorted(self.runs,ib-1,'right')-1)
   for run in range(run0,run1+1):
    lo,hi=self.runs[run:run+2];origin=self.t[lo];lasttime=self.t[hi-1]
    first=max(0,int(np.floor((left-origin)/BLOCK_MS)))
    last=min(int(np.floor((right-origin)/BLOCK_MS)),int(np.floor((lasttime-origin)/BLOCK_MS)))
    for block in range(first,last+1):
     key=(run,block)
     if key not in self.cache:
      a=origin+block*BLOCK_MS;b=a+BLOCK_MS
      i,j=np.searchsorted(self.t,[a,b],'left');j=min(j,hi);i=max(i,lo)
      v=self.x[i:j];ok=len(v)>=30 and bool(np.isfinite(v).all())
      span=float(np.quantile(v,.99)-np.quantile(v,.01)) if ok else np.nan
      self.cache[key]=dict(run=int(run),block=int(block),start_ms=float(a),end_ms=float(b),samples=len(v),assessable=ok,span_uv=span,severe=bool(ok and span>SEVERE_UV))
     evidence[key]=self.cache[key]
  bad=[v for v in evidence.values() if not v['assessable'] or v['severe']]
  return dict(pass_guard=bool(evidence) and not bad,bad_blocks=bad,blocks_checked=len(evidence))

def select(original,padded,start,end,technical_pass,guard_pass,startup_only=True):
 old=np.asarray(original,float);pad=np.asarray(padded,float)
 if not technical_pass:return old.copy(),dict(route='original_context_unavailable',anchor_ms=None)
 if not guard_pass:return old.copy(),dict(route='original_padding_artifact',anchor_ms=None)
 if not startup_only:return pad.copy(),dict(route='guarded_full_context',anchor_ms=None)
 shared=np.intersect1d(old,pad,assume_unique=True)
 shared=shared[(shared>=start)&(shared<=min(end,start+STARTUP_MS))]
 if not len(shared):return old.copy(),dict(route='original_no_exact_startup_anchor',anchor_ms=None)
 anchor=float(shared[0]);out=np.r_[pad[pad<anchor],old[old>=anchor]]
 assert np.array_equal(out[out>=anchor],old[old>=anchor])
 assert not len(out) or (np.all(np.diff(out)>0) and out[0]>=start and out[-1]<=end)
 return out,dict(route='guarded_startup_to_first_exact_common_peak',anchor_ms=anchor)
