# encoding=utf-8
"""
hrv_ext.py
==========
Drop-in extension that wires "HRV regression (Path A)" and "R-peak detection
(Path B)" into your existing main_supervised_baseline.py. The key piece is a
loader with the SAME return signature as your setup_dataloaders, so your
train() / train_sup() / LOSO scaffolding does not need to change.

Deps: numpy, torch. Data = the snowballlab *_P*.npz files.

args fields this module expects (add them via argparse in main):
    args.task           'hrv' | 'peak'
    args.data_dir       directory containing *_P*.npz
    args.device_name    'earring' | 'ring' | 'watch'   (named device_name to
                        avoid clashing with args.cuda)
    args.use_deriv      0/1
    args.resample_hz    0 = keep 100Hz, else e.g. 50
    args.src_hz         100.0
    args.batch_size
    args.target_domain  the held-out test participant, e.g. 'P5'
    args.limit          cap windows per subject (smoke test), 0 = no cap
setup_dataloaders_hrv writes onto args: args._mu / args._sd / args._logmask
(used by test_hrv to invert the target transform).
"""
import os, re, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

FIELD_PPG   = "ppg_resampled"
FIELD_QC    = "ecg_label_qc_pass"
LABEL_KEYS  = ["ecg_sdnn_ms", "ecg_rmssd_ms"]
LOG_TARGET  = np.array([True, True])
PPG_BASE    = ["ppg_sdnn_ms", "ppg_rmssd_ms"]
RPEAK_KEY   = "ecg_r_peak_times_rel_ms"          # object array, one set of ms per window
DEVICE_IDX  = {"earring": 0, "ring": 1, "watch": 2}

_CACHE = {}          # pid -> loaded dict, avoids re-reading from disk each LOSO fold


# -- discover participants ----------------------------------------------------
def discover_hrv_participants(data_dir):
    pids = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*_P*.npz"))):
        m = re.search(r"_P(\d+)\.npz$", os.path.basename(f))
        if m:
            pids.append("P" + m.group(1))
    return pids


def _npz_path(data_dir, pid):
    hits = glob.glob(os.path.join(data_dir, f"*_{pid}.npz"))
    if not hits:
        raise FileNotFoundError(f"No npz for {pid} in {data_dir}")
    return hits[0]


# -- load one participant (task decides the target form) ----------------------
def _load_one(data_dir, pid, args):
    key = (pid, args.task, args.device_name, int(args.use_deriv),
           float(args.resample_hz), int(args.limit))
    if key in _CACHE:
        return _CACHE[key]

    z = np.load(_npz_path(data_dir, pid), allow_pickle=True)
    di = DEVICE_IDX[args.device_name]

    ppg = np.asarray(z[FIELD_PPG], dtype=np.float32)[:, di, :, :]      # (N,2,T)
    y   = np.stack([np.asarray(z[k], float) for k in LABEL_KEYS], 1)   # (N,K) scalar truth (ms)

    keep = np.isfinite(y).all(1) & (y > 0).all(1)
    if FIELD_QC in z.files:
        keep &= np.asarray(z[FIELD_QC]).astype(bool)

    # PPG-derived baseline (N,3,2) -> select device, nanmean over 2 channels -> (N,K)
    base = None
    if all(k in z.files for k in PPG_BASE):
        cols = []
        for k in PPG_BASE:
            b = np.asarray(z[k], float)[:, di, :]
            with np.errstate(invalid="ignore"):
                cols.append(np.nanmean(b, axis=1))
        base = np.stack(cols, 1)

    rpeaks = z[RPEAK_KEY] if RPEAK_KEY in z.files else None            # object (N,)

    idx = np.where(keep)[0]
    if args.limit and len(idx) > args.limit:
        idx = idx[:args.limit]

    ppg, y = ppg[idx], y[idx]
    if base is not None:   base   = base[idx]
    if rpeaks is not None: rpeaks = rpeaks[idx]

    # downsample
    tgt_fs = args.src_hz
    if args.resample_hz and args.resample_hz < args.src_hz:
        step = int(round(args.src_hz / args.resample_hz))
        ppg = ppg[:, :, ::step]
        tgt_fs = args.src_hz / step
    T = ppg.shape[-1]

    # per-window per-channel z-norm
    ppg = (ppg - ppg.mean(-1, keepdims=True)) / (ppg.std(-1, keepdims=True) + 1e-6)
    if args.use_deriv:
        d = np.diff(ppg, axis=-1, prepend=ppg[:, :, :1])
        ppg = np.concatenate([ppg, d], axis=1)                        # (N,2C,T)

    out = {"X": ppg.astype(np.float32), "y_ms": y.astype(np.float32),
           "base": base, "tgt_fs": tgt_fs, "T": T}

    if args.task == "peak":
        # dense soft label: a Gaussian bump at each R-peak position
        out["peak"] = _peak_targets(rpeaks, T, tgt_fs)                # (N,T) in [0,1]

    _CACHE[key] = out
    return out


def _peak_targets(rpeaks, T, fs, sigma_ms=25.0):
    N = len(rpeaks)
    mask = np.zeros((N, T), np.float32)
    sig = max(1.0, sigma_ms / 1000.0 * fs)
    half = int(3 * sig)
    bump = np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2).astype(np.float32)
    for i in range(N):
        arr = rpeaks[i]
        if arr is None:
            continue
        for t_ms in np.asarray(arr, float):
            c = int(round(t_ms / 1000.0 * fs))
            a, b = max(0, c - half), min(T, c + half + 1)
            ba, bb = a - (c - half), (b - (c - half))
            mask[i, a:b] = np.maximum(mask[i, a:b], bump[ba:bb])
    return mask


def n_input_channels(args):
    return 2 * (2 if args.use_deriv else 1)   # green+ir, (x2 if derivative added)


# -- setup_dataloaders: same signature as yours (train_loaders, val, test) ----
def setup_dataloaders_hrv(args):
    pids = discover_hrv_participants(args.data_dir)
    test_pid = args.target_domain
    train_pids = [p for p in pids if p != test_pid]
    if len(train_pids) < 2:
        raise ValueError("Need >=3 participants for LOSO (train/val/test)")
    val_pid = train_pids[-1]
    fit_pids = train_pids[:-1]

    tr = [_load_one(args.data_dir, p, args) for p in fit_pids]
    va = _load_one(args.data_dir, val_pid, args)
    te = _load_one(args.data_dir, test_pid, args)

    Xtr = np.concatenate([d["X"] for d in tr]);  Xva, Xte = va["X"], te["X"]
    if args.task == "hrv":
        Xtr = np.transpose(Xtr, (0, 2, 1))
        Xva = np.transpose(Xva, (0, 2, 1))
        Xte = np.transpose(Xte, (0, 2, 1))

    if args.task == "hrv":
        Ytr_ms = np.concatenate([d["y_ms"] for d in tr])
        mu, sd, logm = _fit_scaler(Ytr_ms)
        args._mu, args._sd, args._logmask = mu, sd, logm
        args._tgt_fs = te["tgt_fs"]
        ytr = _encode(Ytr_ms, mu, sd, logm)
        yva = _encode(va["y_ms"], mu, sd, logm)
        # test set: store RAW ms + baseline; test_hrv inverts and compares
        yte = te["y_ms"]
        args._test_base = te["base"]
        def to_ds(X, Y):  return TensorDataset(torch.tensor(X), torch.tensor(Y, dtype=torch.float32))
        tr_ds, va_ds, te_ds = to_ds(Xtr, ytr), to_ds(Xva, yva), to_ds(Xte, yte)

    elif args.task == "peak":
        Ptr = np.concatenate([d["peak"] for d in tr])
        args._tgt_fs = te["tgt_fs"]
        args._test_y_ms = te["y_ms"]                 # to score the decoded HRV
        args._test_base = te["base"]
        def to_ds(X, P):  return TensorDataset(torch.tensor(X), torch.tensor(P))
        tr_ds, va_ds, te_ds = to_ds(Xtr, Ptr), to_ds(Xva, va["peak"]), to_ds(Xte, te["peak"])
    else:
        raise ValueError(args.task)

    bs = args.batch_size
    mk = lambda ds, sh: DataLoader(_TripletDS(ds), batch_size=bs, shuffle=sh)
    return [mk(tr_ds, True)], mk(va_ds, False), mk(te_ds, False)


class _TripletDS(torch.utils.data.Dataset):
    """Wrap so each item yields (sample, target, dummy), matching your
    train()/test() 3-tuple unpacking."""
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, y, 0


# -- target transform ---------------------------------------------------------
def _fit_scaler(y_ms):
    logm = LOG_TARGET.copy()
    yl = y_ms.copy(); yl[:, logm] = np.log(yl[:, logm])
    return yl.mean(0), yl.std(0) + 1e-8, logm

def _encode(y_ms, mu, sd, logm):
    yl = y_ms.copy(); yl[:, logm] = np.log(yl[:, logm]); return (yl - mu) / sd

def _decode(yn, mu, sd, logm):
    yl = yn * sd + mu; yl[:, logm] = np.exp(yl[:, logm]); return yl


# -- Path B model: 1D U-Net emitting per-sample R-peak probability ------------
class _Enc(nn.Module):
    def __init__(s, ci, co):
        super().__init__()
        s.net = nn.Sequential(nn.Conv1d(ci, co, 7, 2, 3), nn.BatchNorm1d(co), nn.ReLU(True),
                              nn.Conv1d(co, co, 7, 1, 3), nn.BatchNorm1d(co), nn.ReLU(True))
    def forward(s, x): return s.net(x)

class _Dec(nn.Module):
    def __init__(s, ci, sk, co):
        super().__init__()
        s.up = nn.ConvTranspose1d(ci, co, 4, 2, 1)
        s.net = nn.Sequential(nn.Conv1d(co + sk, co, 7, 1, 3), nn.BatchNorm1d(co), nn.ReLU(True))
    def forward(s, x, skip):
        x = s.up(x)
        if x.shape[-1] != skip.shape[-1]:                  # length align
            x = nn.functional.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        return s.net(torch.cat([x, skip], 1))

class PeakNet(nn.Module):
    """(B,C,T) -> (B,1,T) per-sample logit. Returns (logit, None) to match
    your train()'s `out, _ = model(...)` unpacking."""
    def __init__(self, cin, w=24):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(cin, w, 7, 1, 3), nn.BatchNorm1d(w), nn.ReLU(True))
        self.e1, self.e2, self.e3 = _Enc(w, w*2), _Enc(w*2, w*4), _Enc(w*4, w*8)
        self.d3 = _Dec(w*8, w*4, w*4)
        self.d2 = _Dec(w*4, w*2, w*2)
        self.d1 = _Dec(w*2, w,   w)
        self.head = nn.Conv1d(w, 1, 1)
    def forward(self, x):
        s0 = self.stem(x); s1 = self.e1(s0); s2 = self.e2(s1); s3 = self.e3(s2)
        x = self.d3(s3, s2); x = self.d2(x, s1); x = self.d1(x, s0)
        return self.head(x), None


def make_criterion(args):
    return nn.MSELoss() if args.task == "hrv" else nn.BCEWithLogitsLoss()


# -- task-specific test (use instead of your test()) --------------------------
def test_hrv(test_loader, model, device, args, participant_id=None):
    model.eval(); preds, tgts = [], []
    with torch.no_grad():
        for x, y, _ in test_loader:
            out, _ = model(x.to(device))
            preds.append(out.cpu().numpy()); tgts.append(y.numpy())
    pn = np.concatenate(preds); yt = np.concatenate(tgts)              # yt already in ms
    yp = _decode(pn, args._mu, args._sd, args._logmask)               # invert -> ms
    res = _regress_metrics(yt, yp)
    base = getattr(args, "_test_base", None)
    bres = None
    if base is not None:
        m = np.isfinite(base).all(1)
        bres = _regress_metrics(yt[m], base[m])
    return res, bres


def test_peak(test_loader, model, device, args, participant_id=None):
    model.eval(); yhrv_pred = []
    fs = args._tgt_fs
    with torch.no_grad():
        for x, _, _ in test_loader:
            logit, _ = model(x.to(device))
            prob = torch.sigmoid(logit).squeeze(1).cpu().numpy()      # (B,T)
            for p in prob:
                yhrv_pred.append(_hrv_from_prob(p, fs))
    yt = args._test_y_ms                                              # (N,K) ms
    yp = np.array(yhrv_pred)                                          # (N,K) ms
    m = np.isfinite(yp).all(1)
    return _regress_metrics(yt[m], yp[m]), None


def _hrv_from_prob(prob, fs, thr=0.3, min_rr_ms=300):
    # peak picking via scipy: height threshold + min distance (in samples)
    from scipy.signal import find_peaks
    dist = max(1, int(min_rr_ms / 1000 * fs))
    idx, _ = find_peaks(prob, height=thr, distance=dist)
    if len(idx) < 4:
        return np.array([np.nan, np.nan])
    t = idx / fs * 1000.0
    rr = np.diff(t)
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 3:
        return np.array([np.nan, np.nan])
    sdnn = rr.std(ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    return np.array([sdnn, rmssd])


def _regress_metrics(yt, yp):
    out = {}
    for j, n in enumerate(LABEL_KEYS):
        t, p = yt[:, j], yp[:, j]
        if t.size < 2:                       # too few samples (e.g. untrained degenerate)
            out[n] = dict(r2=float("nan"), me=float("nan"),
                          sde=float("nan"), mae=float("nan"), n=int(t.size))
            continue
        ss = np.sum((t - p) ** 2)
        r2 = 1 - ss / (np.sum((t - t.mean()) ** 2) + 1e-12)
        out[n] = dict(r2=r2, me=float(np.mean(p - t)),
                      sde=float(np.std(p - t)), mae=float(np.mean(np.abs(p - t))),
                      n=int(t.size))
    return out
