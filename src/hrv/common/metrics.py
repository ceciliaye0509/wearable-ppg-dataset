import numpy as np


def summarize_nn(groups):
    nn=[np.asarray(g[0],float) for g in groups if len(g[0])]
    differences=[np.asarray(g[1],float) for g in groups if len(g[1])]
    all_nn=np.concatenate(nn) if nn else np.empty(0)
    diff=np.concatenate(differences) if differences else np.empty(0)
    if len(all_nn)<3 or not len(diff):
        return dict(rmssd_ms=np.nan,sdnn_ms=np.nan,n_valid_rr=len(all_nn),n_pairs=len(diff))
    return dict(rmssd_ms=float(np.sqrt(np.mean(diff**2))),sdnn_ms=float(np.std(all_nn,ddof=1)),
                n_valid_rr=len(all_nn),n_pairs=len(diff))
