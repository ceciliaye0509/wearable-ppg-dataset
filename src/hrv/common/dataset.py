"""Read HF participant NPZs or this package's bounded-size window shards."""
from pathlib import Path
import json
import numpy as np

def iter_rows(directory,participant=None):
    directory=Path(directory)
    manifest=directory/"dataset_manifest.json"
    if manifest.exists():
        data=json.loads(manifest.read_text())
        files=[directory/f for pid,entries in data["participants"].items() if participant is None or pid==participant for f in entries]
    else:
        files=sorted(directory.glob(f"*_{participant or 'P*'}.npz"))
    for path in files:
        with np.load(path,allow_pickle=True) as z:
            count=len(z["ppg_window_t0_ms"])
            small={k:z[k] for k in z.files}
        for i in range(count):
            row={k:(v[i] if v.ndim and v.shape[0]==count and k not in {"devices","channels"} else v) for k,v in small.items()}
            if "source_window_index" not in row:row["source_window_index"]=i
            yield row

def stack_rows(rows):
    result={}
    for key in rows[0]:
        values=[r[key] for r in rows]
        if key in {"devices","channels","participant","target_fs","window_sec","stride_sec"}:result[key]=np.asarray(values[0])
        elif key.startswith("ecg_r_peak") or key.startswith("ecg_rr_intervals"):
            array=np.empty(len(values),object)
            for i,v in enumerate(values):array[i]=v
            result[key]=array
        else:result[key]=np.asarray(values)
    return result
