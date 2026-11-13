"""Segment250 qppg baseline with original RR adjacency."""
import json
import numpy as np
from hrv.common.ppg import detect
from hrv.common.rr import _correct_ibi_artifacts_with_ratio
from hrv.common.metrics import summarize_nn

def baseline_nn(peaks):
    rr=np.diff(peaks);valid=np.isfinite(rr)&(rr>=300)&(rr<=2000)
    positions=np.flatnonzero(valid);nn=rr[valid]
    if not len(nn):return np.empty(0),np.empty(0)
    corrected,_=_correct_ibi_artifacts_with_ratio(nn,threshold=.2)
    return corrected,np.diff(corrected)[np.diff(positions)==1]

def predict(row,di,ci,detector="qppgfast"):
    if not row['ppg_baseline_usable'][di,ci]:return summarize_nn([])
    grid=row['ppg_grid_timestamp_ms'];mask=row['ppg_rawslot_mask'][di,ci]
    t=row['ppg_rawslot_timestamp_ms'][di,ci][mask];x=row['ppg_rawslot_values'][di,ci][mask]
    order=np.argsort(t,kind='stable');t,x=t[order],x[order];first=np.r_[True,np.diff(t)>0];t,x=t[first],x[first]
    specs=json.loads(str(row['ppg_segments_json']))[di][ci]['baseline'];groups=[]
    for s in specs:
        g=grid[(grid>=s['lo'])&(grid<=s['hi'])]
        inside=(t>=s['lo'])&(t<=s['hi'])
        values=np.interp(g,t[inside],x[inside])
        peaks=detect(values,g,str(row['devices'][di]),detector=detector)
        peaks=peaks[(peaks>=s['keep_lo'])&(peaks<=s['keep_hi'])]
        groups.append(baseline_nn(peaks))
    return summarize_nn(groups)

if __name__=='__main__':
    from hrv.common.evaluate import baseline_main
    baseline_main(predict)
