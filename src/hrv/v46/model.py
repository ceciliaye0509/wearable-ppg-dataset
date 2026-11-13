"""Segment-aware adapter around the unchanged colleague Kazemi model."""
from dataclasses import dataclass
from pathlib import Path
import argparse,csv,json,copy,random
from scipy.signal import find_peaks
import numpy as np
from scipy.signal import resample_poly
import torch
from hrv.common.partner import load
from hrv.common.dataset import iter_rows
from hrv.common.metrics import summarize_nn
from .loss import masked_window_loss

@dataclass
class Segment:
    features: np.ndarray
    target: np.ndarray
    valid: np.ndarray
    observed: np.ndarray
    start_ms: float



def peak_targets(peaks_rel_ms,length):
    """The frozen Kazemi target: the peak bin and two neighbors on each side."""
    target=np.zeros(length,bool)
    for t_ms in np.asarray(peaks_rel_ms,float):
        center=int(round(t_ms/1000.*50.))
        target[max(0,center-2):min(length,center+3)]=True
    return target


def peak_indices(prob,threshold):
    # Frozen decoder values, independent of another entry's environment.
    indices,_=find_peaks(prob,height=threshold,distance=15,prominence=0.)
    return indices.astype(np.int64)


def validate_folds(records):
    assignments={}
    if sorted(r['fold'] for r in records)!=list(range(4)):raise ValueError('Expected the four frozen folds')
    for rec in records:
        train,val,test=map(set,[rec['train'],rec['val'],rec['test']])
        if train&val or train&test or val&test:raise ValueError('Participant split leakage')
        for pid in test:
            if pid in assignments:raise ValueError('Duplicate held-out participant: '+pid)
            assignments[pid]=rec['fold']
    return assignments


def prepare(row,H,di=0,ci=0):
    if not row['ppg_model_usable'][di,ci]:return []
    grid=row['ppg_grid_timestamp_ms'];parts=[]
    if not row['ppg_requires_split'][di,ci]:ranges=[np.ones(len(grid),bool)]
    else:
        specs=json.loads(str(row['ppg_segments_json']))[di][ci]['model']
        ranges=[(grid>=s['lo'])&(grid<=s['hi']) for s in specs]
    for inside in ranges:
        g=grid[inside];v=row['ppg_resampled_values'][di,ci][inside]
        filtered=H._preprocess_peak_ppg(np.asarray(v,np.float32)[None,None,:],100.,low_hz=.5,high_hz=8.)
        parts.append((resample_poly(filtered,1,2,axis=-1).astype(np.float32,copy=False),g,inside))
    if not parts:return []
    pooled=np.concatenate([p[0] for p in parts],axis=-1);mean=np.nanmean(pooled,axis=-1,keepdims=True);std=np.nanstd(pooled,axis=-1,keepdims=True)
    segments=[]
    for values,g,inside in parts:
        x=np.nan_to_num((values-mean)/(std+1e-6),nan=0.,posinf=0.,neginf=0.)
        x=np.concatenate([x,np.diff(x,axis=-1,prepend=x[:,:,:1])],axis=1)[0].astype(np.float32)
        t=g[0]+np.arange(x.shape[-1])*20.;ix=np.searchsorted(g,t)
        valid=row['ppg_model_retained_mask'][di,ci][inside][ix]&np.isfinite(x).all(axis=0)
        observed=row['ppg_rawslot_mask'][di,ci][inside][ix]
        peaks=np.asarray(row['ecg_r_peak_times_abs_ms'],float)
        peaks=peaks[(peaks>=g[0]-40.)&(peaks<=g[-1]+40.)]
        target=peak_targets(peaks-g[0],x.shape[-1])
        segments.append(Segment(x,target,valid,observed,float(g[0])))
    return segments


def forward(model,segments):
    return [model(torch.from_numpy(s.features)[None])[0][0,0] for s in segments]


def decode(H,segments,logits,threshold,*,split=False):
    groups=[];valid_total=pairs_total=0
    for s,logit in zip(segments,logits):
        prob=logit.detach().sigmoid().cpu().numpy();indices=peak_indices(prob,threshold)
        indices=indices[s.valid[indices]]
        labels=np.cumsum(np.r_[True,np.diff(np.asarray(s.valid,np.int8))!=0])
        for label in np.unique(labels[indices]):
            idx=indices[labels[indices]==label]
            times=idx.astype(np.float64)*20.+s.start_ms if split else idx/50.*1000.
            rr=np.diff(times)
            if not len(rr):continue
            good=np.isfinite(rr)&(rr>=300)&(rr<=2000)
            corrected,_=H._correct_rr_intervals(rr,threshold=.20,min_rr_ms=300.,max_rr_ms=2000.)
            groups.append((corrected,np.diff(corrected)));valid_total+=int(good.sum());pairs_total+=int((good[:-1]&good[1:]).sum())
    result=summarize_nn(groups)
    if valid_total<3 or pairs_total<1:result.update(rmssd_ms=np.nan,sdnn_ms=np.nan)
    return result


def eligible(row):
    return bool(row['ecg_label_qc_pass']) and np.isfinite(row['ecg_rmssd_corrected_ms']) and np.isfinite(row['ecg_sdnn_corrected_ms'])


def evaluate(args,H):
    models={};records=[]
    folds=json.loads(args.folds.read_text());assignment=validate_folds(folds)
    for i,row in enumerate(iter_rows(args.dataset_dir)):
        pid=str(row['participant']);fold=assignment.get(pid)
        value=dict(rmssd_ms=np.nan,sdnn_ms=np.nan)
        if fold is not None and eligible(row):
            if fold not in models:
                model=H.KazemiPeakNet(2);state=torch.load(args.checkpoint_dir/f'fold_{fold}.pt',map_location='cpu',weights_only=True)
                model.load_state_dict(state['model_state_dict'],strict=True);model.eval()
                cfg=json.loads((args.decoder_dir/f'decoder_fold{fold}.json').read_text());models[fold]=(model,cfg['threshold'])
            model,threshold=models[fold];segments=prepare(row,H)
            with torch.no_grad():logits=forward(model,segments)
            value=decode(H,segments,logits,threshold,split=bool(row['ppg_requires_split'][0,0]))
        records.append(dict(participant=pid,source_window_index=int(row['source_window_index']),fold=fold,
            start_ms=float(row['ppg_window_t0_ms']),device='Earring',channel='ppg_green',ecg_eligible=eligible(row),
            ecg_rmssd_ms=float(row['ecg_rmssd_corrected_ms']),ecg_sdnn_ms=float(row['ecg_sdnn_corrected_ms']),pred_rmssd_ms=value['rmssd_ms'],pred_sdnn_ms=value['sdnn_ms']))
        if args.limit and i+1>=args.limit:break
    from hrv.common.evaluate import summarize
    with (args.output_dir/'predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]) if records else ['participant']);w.writeheader();w.writerows(records)
    (args.output_dir/'summary.json').write_text(json.dumps(summarize(records),indent=2))


def training_windows(dataset_dir,pid,H):
    for row in iter_rows(dataset_dir,participant=pid):
        if not eligible(row):continue
        segments=prepare(row,H)
        if segments:yield str(row['participant']),int(row['source_window_index']),segments


def train(args,H):
    records=json.loads(args.folds.read_text());validate_folds(records)
    fold=next(f for f in records if f['fold']==args.fold)
    # The historical bridge instantiated its checkpoint network before creating
    # the training network. Preserve that seed-1 RNG progression without a
    # dependency on an old checkpoint or old dataset directory.
    H.KazemiPeakNet(2)
    model=H.KazemiPeakNet(2)
    initial_checkpoint=getattr(args,'init_checkpoint',None)
    smoke_steps=getattr(args,'train_smoke_steps',0)
    if initial_checkpoint:
        state=torch.load(initial_checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model_state_dict'],strict=True)
    if smoke_steps and not initial_checkpoint:raise ValueError('--train-smoke-steps requires --init-checkpoint')
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    best=float('inf');stale=0;steps=0;trace=[]
    smoke_anchor=None
    if smoke_steps:
        for pid in fold['train']:
            for _,index,segments in training_windows(args.dataset_dir,pid,H):
                if len(segments)>1:smoke_anchor=(pid,index);break
            if smoke_anchor:break
        if smoke_anchor is None:raise ValueError('No multi-segment training window for smoke acceptance')
    for epoch in range(args.epochs):
        rng=random.Random(1+epoch);pids=list(fold['train']);rng.shuffle(pids);model.train();epoch_steps=0
        if smoke_anchor:pids.remove(smoke_anchor[0]);pids.insert(0,smoke_anchor[0])
        for pid in pids:
            windows=list(training_windows(args.dataset_dir,pid,H))
            indices=list(range(len(windows)));rng.shuffle(indices)
            if smoke_anchor and pid==smoke_anchor[0]:
                chosen=next(i for i,w in enumerate(windows) if w[1]==smoke_anchor[1]);indices.remove(chosen);indices.insert(0,chosen)
            for offset in range(0,len(indices),args.batch_size):
                batch=[windows[i] for i in indices[offset:offset+args.batch_size]]
                optimizer.zero_grad()
                loss=torch.stack([masked_window_loss(forward(model,s),s) for _,_,s in batch]).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();steps+=1;epoch_steps+=1
                trace.append(dict(epoch=epoch,step=steps,participant=pid,source_windows=[w[1] for w in batch],
                    segments=[len(w[2]) for w in batch],valid_bins=[sum(int(s.valid.sum()) for s in w[2]) for w in batch],loss=float(loss.detach())))
                cap=smoke_steps or args.steps
                if cap and steps>=cap:break
            if (smoke_steps or args.steps) and steps>=(smoke_steps or args.steps):break
        if not epoch_steps:raise ValueError('No eligible training windows')
        model.eval();losses=[]
        with torch.no_grad():
            for pid in fold['val']:
                for _,_,segments in training_windows(args.dataset_dir,pid,H):
                    losses.append(float(masked_window_loss(forward(model,segments),segments)))
                    if args.limit and len(losses)>=args.limit:break
                if args.limit and len(losses)>=args.limit:break
        if not losses:raise ValueError('No eligible validation windows')
        score=float(np.mean(losses));print(f'epoch={epoch+1} val_loss={score:.6f} steps={steps}',flush=True)
        if score<best:
            best=score;stale=0
            torch.save(dict(model_state_dict=copy.deepcopy(model.state_dict()),optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch+1,steps=steps,validation_loss=score,seed=1,training_contract='segment250_retained_mask_v2',
                train_participants=fold['train'],validation_participants=fold['val']),args.output_dir/'checkpoint.pt')
        else:stale+=1
        if (args.patience and stale>=args.patience) or ((smoke_steps or args.steps) and steps>=(smoke_steps or args.steps)):break
    (args.output_dir/'training_receipt.json').write_text(json.dumps(dict(steps=steps,validation_loss=best,trace=trace,
        mode='smoke' if smoke_steps else 'training',train_participants=fold['train'],validation_participants=fold['val'],
        test_participants_not_loaded=fold['test'],initial_checkpoint=str(initial_checkpoint) if initial_checkpoint else None),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['eval','train'])
    p.add_argument('--dataset-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--folds',type=Path,required=True);p.add_argument('--checkpoint-dir',type=Path);p.add_argument('--decoder-dir',type=Path)
    p.add_argument('--fold',type=int,default=0);p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--patience',type=int,default=5);p.add_argument('--limit',type=int);p.add_argument('--steps',type=int);p.add_argument('--init-checkpoint',type=Path);p.add_argument('--train-smoke-steps',type=int,default=0)
    a=p.parse_args()
    if a.mode=='eval' and (a.checkpoint_dir is None or a.decoder_dir is None):p.error('eval requires --checkpoint-dir and --decoder-dir')
    a.output_dir.mkdir(parents=True,exist_ok=True);random.seed(1);np.random.seed(1);torch.manual_seed(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    H=load()
    if a.mode=='eval':evaluate(a,H)
    else:train(a,H)

if __name__=='__main__':main()
