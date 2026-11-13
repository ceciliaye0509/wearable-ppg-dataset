"""Generate v4.6 from native PPG/ECG, never from another version's windows."""
from pathlib import Path
import argparse,json
import numpy as np
from hrv.common.settings import DEVICES,CHANNELS
from hrv.common.rawio import _load_ppg_raw,_load_ecg_raw
from hrv.common.windowing import _candidate_windows,_boundary_offsets_ms
from hrv.common.signals import _rawslots_to_grid,_accel_gravity_reference,_accel_mean_magnitude_window,_accel_motion_mean_magnitude_window
from hrv.common.dataset import stack_rows
from hrv.common.metrics import summarize_nn
from .ecg import ECGReference
from .ppg import source,channel_plan


def membership(ppg,ecg):
    starts,ends,_=_candidate_windows(ppg,ecg,devices=DEVICES,window_sec=300,stride_sec=30)
    source_index=0
    for candidate_index,(start,end) in enumerate(zip(starts,ends)):
        offsets=[_boundary_offsets_ms(ppg[d]['timestamp'],start,end) for d in DEVICES]
        if any(a>2000 or b>2000 for a,b,n in offsets):continue
        grid=start+np.arange(30000)*10.
        ratios=[]
        for d in DEVICES:
            t=ppg[d]['timestamp'];a=np.searchsorted(t,start-500,'left');b=np.searchsorted(t,end+500,'right')
            for c in CHANNELS:
                *_,ratio=_rawslots_to_grid(ppg[d][c][a:b],t[a:b],grid,slot_tolerance_ms=5.)
                ratios.append(ratio)
        if min(ratios)<.9:continue
        yield source_index,candidate_index,float(start),float(end),grid
        source_index+=1


def observation_flag(label,grid,keep):
    if not label['ecg_label_qc_pass']:return False
    peaks=label['ecg_r_peak_times_abs_ms'];rr=label['ecg_rr_intervals_corrected_ms'];groups=[]
    edges=np.flatnonzero(np.diff(np.r_[False,keep,False].astype(np.int8)))
    for a,b in zip(edges[::2],edges[1::2]):
        valid=(peaks[:-1]>=grid[a])&(peaks[1:]<=grid[b-1]);pos=np.flatnonzero(valid);nn=rr[valid]
        groups.append((nn,np.diff(nn)[np.diff(pos)==1]))
    stats=summarize_nn(groups)
    return any(np.isfinite(stats[k]) and abs(stats[k]-label['ecg_'+k.removesuffix('_ms')+'_corrected_ms'])>10 for k in ['rmssd_ms','sdnn_ms'])


def iter_participant(raw_root,pid,reference_dir,*,window_start_ms=None):
    ppg=_load_ppg_raw(Path(raw_root),pid,list(DEVICES));ecg=_load_ecg_raw(Path(raw_root),pid)
    reference=ECGReference(pid,ecg,reference_dir)
    native={d:source(ppg[d]) for d in DEVICES}
    gravity={d:_accel_gravity_reference(ppg[d]) for d in DEVICES}
    for wi,ci,start,end,grid in membership(ppg,ecg):
        if window_start_ms is not None and start!=window_start_ms:continue
        row=dict(participant=np.array(pid),devices=np.asarray(DEVICES),channels=np.asarray(CHANNELS),
            target_fs=np.array(100.),window_sec=np.array(300),stride_sec=np.array(30),
            source_window_index=wi,original_candidate_window_index=ci,
            ppg_window_t0_ms=start,ppg_window_t1_ms=end,ppg_grid_timestamp_ms=grid)
        row.update(reference.label(start,end))
        fields={k:[] for k in ['raw','observed','raw_timestamp','resampled','resampled_mask','baseline_keep','model_keep','baseline_usable','model_usable','requires_split']}
        specs=[];flags=[]
        for d in DEVICES:
            plans=[channel_plan(native[d][0],native[d][1][c],grid) for c in range(2)]
            for key in fields:fields[key].append([p[key] for p in plans])
            specs.append([dict(baseline=p['baseline_segments'],model=p['model_segments']) for p in plans])
            flags.append([observation_flag(row,grid,p['model_keep']) for p in plans])
        names=dict(raw='ppg_rawslot_values',observed='ppg_rawslot_mask',raw_timestamp='ppg_rawslot_timestamp_ms',
            resampled='ppg_resampled_values',resampled_mask='ppg_resampled_mask',baseline_keep='ppg_baseline_retained_mask',
            model_keep='ppg_model_retained_mask',baseline_usable='ppg_baseline_usable',model_usable='ppg_model_usable',requires_split='ppg_requires_split')
        for key,name in names.items():row[name]=np.asarray(fields[key])
        row['ppg_rawslot_valid_sample_ratio']=row['ppg_rawslot_mask'].mean(axis=-1).astype(np.float32)
        row['ppg_resampled_valid_sample_ratio']=row['ppg_resampled_mask'].mean(axis=-1).astype(np.float32)
        row['ppg_segments_json']=json.dumps(specs)
        row['ppg_observation_review_required']=np.asarray(flags,bool)
        row['accel_mean_mag']=np.asarray([_accel_mean_magnitude_window(ppg[d],t0=start,t1=end) for d in DEVICES],np.float32)
        row['accel_motion_mean_mag']=np.asarray([_accel_motion_mean_magnitude_window(ppg[d],t0=start,t1=end,gravity_reference=gravity[d]) for d in DEVICES],np.float32)
        yield row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path,default=Path(__file__).resolve().parents[3]/'configs/v46_reference')
    p.add_argument('--participants',default=','.join(f'P{i}' for i in range(1,21)))
    p.add_argument('--limit',type=int);p.add_argument('--window-start-ms',type=float)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if (a.output_dir/'dataset_manifest.json').exists():raise FileExistsError('Use a new output directory for a new generation.')
    manifest=dict(version='v4.6',window_sec=300,stride_sec=30,target_fs=100,independent_raw_generation=True,participants={})
    for pid in a.participants.split(','):
        folder=a.output_dir/pid;folder.mkdir(exist_ok=True);buffer=[];files=[];count=0;valid=0
        def flush():
            if not buffer:return
            file=folder/f'part_{len(files):04d}.npz'
            np.savez_compressed(file,**stack_rows(buffer));files.append(str(file.relative_to(a.output_dir)));buffer.clear()
        for row in iter_participant(a.raw_root,pid,a.reference_dir,window_start_ms=a.window_start_ms):
            buffer.append(row);count+=1;valid+=int(row['ecg_label_qc_pass'])
            if len(buffer)>=32:flush()
            if a.limit and count>=a.limit:break
        flush();manifest['participants'][pid]=files
        print(f'{pid}: {count} candidates, {valid} valid ECG labels',flush=True)
    (a.output_dir/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
