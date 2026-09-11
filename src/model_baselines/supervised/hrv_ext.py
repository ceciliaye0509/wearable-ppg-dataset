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
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

FIELD_PPG = "ppg_resampled_values"
FIELD_QC = "ecg_label_qc_pass"
FIELD_PPG_VALID_RATIO = "ppg_resampled_valid_sample_ratio"
LABEL_KEYS = [
    "ecg_sdnn_corrected_ms",
    "ecg_rmssd_corrected_ms",
]
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
    key = (pid, args.task, args.device_name,
           getattr(args, "peak_channels", "both"), int(args.use_deriv),
           float(args.resample_hz), int(args.limit),
           os.environ.get("PEAK_MIN_PPG_VALID_RATIO", "0.95"),
           os.environ.get("PEAK_BANDPASS_LOW_HZ", "0.5"),
           os.environ.get("PEAK_BANDPASS_HIGH_HZ", "8.0"))
    if key in _CACHE:
        return _CACHE[key]

    z = np.load(_npz_path(data_dir, pid), allow_pickle=True)
    di = DEVICE_IDX[args.device_name]

    ppg = np.asarray(z[FIELD_PPG], dtype=np.float32)[:, di, :, :]      # (N,2,T)
    if args.task == "peak":
        channel_indices = {
            "green": [0], "ir": [1], "both": [0, 1],
        }[getattr(args, "peak_channels", "both")]
        ppg = ppg[:, channel_indices, :]
    y   = np.stack([np.asarray(z[k], float) for k in LABEL_KEYS], 1)   # (N,K) scalar truth (ms)

    keep = np.isfinite(y).all(1) & (y > 0).all(1)
    if FIELD_QC in z.files:
        keep &= np.asarray(z[FIELD_QC]).astype(bool)

    # Peak detection requires usable PPG as well as usable ECG labels.
    # Require both GREEN and IR channels for the selected device because
    # PeakNet consumes both channels.
    if args.task == "peak" and FIELD_PPG_VALID_RATIO in z.files:
        minimum_ppg_ratio = float(
            os.environ.get("PEAK_MIN_PPG_VALID_RATIO", "0.95")
        )
        ppg_valid_ratio = np.asarray(
            z[FIELD_PPG_VALID_RATIO],
            dtype=np.float32,
        )[:, di, channel_indices]
        keep &= np.isfinite(ppg_valid_ratio).all(axis=1)
        keep &= (ppg_valid_ratio >= minimum_ppg_ratio).all(axis=1)

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

    if args.task == "peak":
        if rpeaks is None:
            raise KeyError(f"Missing required peak field: {RPEAK_KEY}")
        valid_peak_labels = np.asarray([
            _valid_rpeak_array(value)
            for value in rpeaks
        ])
        keep &= valid_peak_labels

    idx = np.where(keep)[0]
    if args.limit and len(idx) > args.limit:
        idx = idx[:args.limit]

    ppg, y = ppg[idx], y[idx]
    if base is not None:   base   = base[idx]
    if rpeaks is not None: rpeaks = rpeaks[idx]

    # PeakNet needs pulse morphology rather than slow baseline drift.
    # Fill short unsupported regions, then band-pass before resampling.
    if args.task == "peak":
        ppg = _preprocess_peak_ppg(
            ppg,
            fs=float(args.src_hz),
            low_hz=float(os.environ.get("PEAK_BANDPASS_LOW_HZ", "0.5")),
            high_hz=float(os.environ.get("PEAK_BANDPASS_HIGH_HZ", "8.0")),
        )

    # Downsample. Peak detection uses polyphase resampling so frequencies above
    # the new Nyquist limit are removed instead of aliased into the pulse band.
    tgt_fs = args.src_hz
    if args.resample_hz and args.resample_hz < args.src_hz:
        if args.task == "peak":
            from fractions import Fraction
            from scipy.signal import resample_poly

            ratio = Fraction(
                float(args.resample_hz) / float(args.src_hz)
            ).limit_denominator(1000)
            ppg = resample_poly(
                ppg,
                up=ratio.numerator,
                down=ratio.denominator,
                axis=-1,
            ).astype(np.float32, copy=False)
            tgt_fs = float(args.src_hz) * ratio.numerator / ratio.denominator
        else:
            step = int(round(args.src_hz / args.resample_hz))
            ppg = ppg[:, :, ::step]
            tgt_fs = args.src_hz / step
    T = ppg.shape[-1]

    # per-window per-channel z-norm
    # Normalize while ignoring unsupported NaN samples
    ppg_mean = np.nanmean(ppg, axis=-1, keepdims=True)
    ppg_std = np.nanstd(ppg, axis=-1, keepdims=True)

    ppg = (ppg - ppg_mean) / (ppg_std + 1e-6)

    # After normalization, missing samples are replaced by the window-channel mean
    # because zero is the normalized mean.
    ppg = np.nan_to_num(
        ppg,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

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


def _valid_rpeak_array(value):
    """Return True for a nonempty, finite, strictly increasing peak sequence."""
    if value is None:
        return False
    peaks = np.asarray(value, dtype=np.float64).reshape(-1)
    return (
        peaks.size >= 4 and
        np.isfinite(peaks).all() and
        np.all(np.diff(peaks) > 0)
    )


def _fill_missing_linear(signal):
    """Linearly fill non-finite samples along the last axis."""
    signal = np.asarray(signal, dtype=np.float32).copy()
    flat = signal.reshape(-1, signal.shape[-1])
    sample_index = np.arange(signal.shape[-1])

    for row in flat:
        finite = np.isfinite(row)
        if finite.all():
            continue
        if finite.sum() < 2:
            row[:] = 0.0
            continue
        row[~finite] = np.interp(
            sample_index[~finite],
            sample_index[finite],
            row[finite],
        )
    return signal


def _preprocess_peak_ppg(ppg, fs, low_hz=0.5, high_hz=8.0):
    """Fill missing samples and apply a zero-phase Butterworth band-pass."""
    from scipy.signal import butter, sosfiltfilt

    if not 0.0 < low_hz < high_hz < fs / 2.0:
        raise ValueError(
            f"Invalid peak band-pass [{low_hz}, {high_hz}] Hz for fs={fs}"
        )

    filled = _fill_missing_linear(ppg)
    sos = butter(
        4,
        [low_hz, high_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos, filled, axis=-1).astype(np.float32, copy=False)


def _peak_targets(rpeaks, T, fs, sigma_ms=25.0):
    N = len(rpeaks)
    mask = np.zeros((N, T), np.float32)
    target_mode = os.environ.get("PEAK_TARGET_MODE", "gaussian").strip().lower()
    if target_mode == "kazemi":
        # Official code labels the peak sample and two samples on each side.
        for i, arr in enumerate(rpeaks):
            if arr is None:
                continue
            for t_ms in np.asarray(arr, float):
                center = int(round(t_ms / 1000.0 * fs))
                start, stop = max(0, center - 2), min(T, center + 3)
                mask[i, start:stop] = 1.0
        return mask
    if target_mode != "gaussian":
        raise ValueError(f"Unknown PEAK_TARGET_MODE: {target_mode!r}")
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
    channel_mode = (
        getattr(args, "peak_channels", "both")
        if args.task == "peak" else "both"
    )
    base_channels = 2 if channel_mode == "both" else 1
    return base_channels * (2 if args.use_deriv else 1)


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

    if args.task == "hrv_seg":                     # C: cut each 5-min into 30 x 10s
        Xtr = _segment(Xtr); Xva = _segment(Xva); Xte = _segment(Xte)  # (N, S, C, L)

    if args.task in ("hrv", "hrv_seg"):
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

def _segment(X, n_seg=30):
    # (N, C, T) -> (N, n_seg, C, L)  ; L = T // n_seg  (10s each for a 5-min window)
    N, C, T = X.shape
    L = T // n_seg
    X = X[:, :, :L * n_seg]
    return X.reshape(N, C, n_seg, L).transpose(0, 2, 1, 3).copy()


class SegNet(nn.Module):
    """Path C: a shared small CNN encodes each 10s segment, then aggregates across
    the 30 segments (mean or LSTM) -> [SDNN, RMSSD]. Input (B, S, C, L).
    Returns (out, None) to match your train()'s `out, _ = model(...)`."""
    def __init__(self, cin, agg="mean", feat=64, k_out=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(cin, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.ReLU(True),
            nn.Conv1d(32, 64, 7, 2, 3),  nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Conv1d(64, feat, 7, 2, 3), nn.BatchNorm1d(feat), nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1))
        self.agg = agg
        if agg == "lstm":
            self.rnn = nn.LSTM(feat, feat, batch_first=True)
        self.head = nn.Linear(feat, k_out)
    def forward(self, x):
        B, S, C, L = x.shape
        f = self.enc(x.reshape(B * S, C, L)).squeeze(-1)   # (B*S, feat)
        f = f.reshape(B, S, -1)                            # (B, S, feat)
        if self.agg == "lstm":
            _, (h, _) = self.rnn(f); g = h[-1]             # (B, feat)
        else:
            g = f.mean(1)                                  # (B, feat)
        return self.head(g), None


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


class DilatedResidualBlock(nn.Module):
    """Residual 1-D block that expands temporal context without pooling."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.net(x))


class DilatedPeakNet(nn.Module):
    """Literature-inspired dilated CNN for per-sample PPG peak logits.

    This is an independent implementation, not an exact reproduction of the
    Kazemi et al. model. It preserves temporal resolution and uses dilations to
    observe multiple neighboring beats.
    """

    def __init__(self, cin, width=48, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(cin, width, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[
            DilatedResidualBlock(width, dilation)
            for dilation in dilations
        ])
        self.head = nn.Conv1d(width, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x), None


class KazemiPeakNet(nn.Module):
    """PyTorch reproduction of the seven-layer Kazemi et al. dilated CNN.

    The published TensorFlow model uses kernel size 3, ELU activations,
    filters 4/8/8/16/16/32/1, and dilations 1/2/4/8/16/32/64.
    This module returns logits; sigmoid is applied by the loss/evaluation code.
    """

    def __init__(self, cin):
        super().__init__()
        filters = (4, 8, 8, 16, 16, 32)
        dilations = (1, 2, 4, 8, 16, 32)
        layers = []
        input_channels = cin
        for output_channels, dilation in zip(filters, dilations):
            layers.extend([
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    dilation=dilation,
                    padding=dilation,
                ),
                nn.ELU(inplace=True),
            ])
            input_channels = output_channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv1d(
            input_channels,
            1,
            kernel_size=3,
            dilation=64,
            padding=64,
        )

    def forward(self, x):
        return self.head(self.features(x)), None


class PeakLoss(nn.Module):
    """
    Weighted BCE + soft Dice loss.

    Weighted BCE reduces the effect of the large number of non-peak samples.
    Dice loss encourages overlap between predicted and true peak regions.
    """
    def __init__(self, pos_weight=10.0, dice_weight=1.0):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits, target):
        pos_weight = torch.tensor(
            self.pos_weight,
            device=logits.device,
            dtype=logits.dtype,
        )

        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
        )

        prob = torch.sigmoid(logits)

        intersection = (prob * target).sum(dim=1)
        denominator = prob.sum(dim=1) + target.sum(dim=1)

        dice = (
            (2.0 * intersection + 1e-6) /
            (denominator + 1e-6)
        ).mean()

        return bce + self.dice_weight * (1.0 - dice)


def make_criterion(args):
    if args.task == "peak":
        return PeakLoss(
            pos_weight=float(
                os.environ.get("PEAK_POS_WEIGHT", "10.0")
            ),
            dice_weight=float(
                os.environ.get("PEAK_DICE_WEIGHT", "1.0")
            ),
        )
    return nn.MSELoss()


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


def test_peak(
    test_loader,
    model,
    device,
    args,
    participant_id=None,
):
    model.eval()

    fs = args._tgt_fs
    threshold = getattr(args, "_peak_threshold", 0.30)

    predicted_hrv = []
    predicted_hrv_raw = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_windows = 0
    valid_hrv_windows = 0

    predicted_peak_counts = []
    true_peak_counts = []
    correction_ratios = []

    # Optional visual audit. Disabled unless PEAK_AUDIT_DIR is set.
    # When enabled, only the requested held-out participant is exported.
    audit_dir = os.environ.get("PEAK_AUDIT_DIR", "").strip()
    audit_participant = os.environ.get(
        "PEAK_AUDIT_PARTICIPANT",
        "P1",
    ).strip()
    current_participant = str(
        participant_id or getattr(args, "target_domain", "unknown")
    )
    collect_audit = bool(audit_dir) and current_participant == audit_participant
    audit_rows = []
    test_row_index = 0

    with torch.no_grad():
        for x, target_peak, _ in test_loader:
            logits, _ = model(x.to(device))

            probabilities = (
                torch.sigmoid(logits)
                .squeeze(1)
                .cpu()
                .numpy()
            )

            target_peak = target_peak.numpy()

            for prob, target in zip(probabilities, target_peak):
                total_windows += 1

                predicted_peaks = _find_peak_indices(
                    prob,
                    fs,
                    threshold,
                )

                true_peaks = _find_peak_indices(
                    target,
                    fs,
                    threshold=0.50,
                )

                tp, fp, fn = _match_peak_counts(
                    predicted_peaks,
                    true_peaks,
                    fs,
                    tolerance_ms=100.0,
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn

                true_peak_counts.append(len(true_peaks))

                (
                    raw_hrv,
                    hrv,
                    predicted_count,
                    correction_ratio,
                ) = _hrv_from_prob(
                    prob,
                    fs,
                    thr=threshold,
                    ibi_method=args.ibi_method,
                    min_rr_ms=args.min_rr_ms,
                    max_rr_ms=args.max_rr_ms,
                    deviation_threshold=args.ibi_deviation_threshold,
                    reject_ratio=args.ibi_reject_ratio,
                )

                predicted_peak_counts.append(predicted_count)
                correction_ratios.append(correction_ratio)
                predicted_hrv_raw.append(raw_hrv)
                predicted_hrv.append(hrv)

                if collect_audit:
                    window_tp, window_fp, window_fn = _match_peak_counts(
                        predicted_peaks,
                        true_peaks,
                        fs,
                        tolerance_ms=100.0,
                    )
                    window_precision = window_tp / max(
                        window_tp + window_fp,
                        1,
                    )
                    window_recall = window_tp / max(
                        window_tp + window_fn,
                        1,
                    )
                    window_f1 = (
                        2.0 * window_precision * window_recall /
                        max(window_precision + window_recall, 1e-12)
                    )
                    matched_offsets_ms = _matched_peak_offsets_ms(
                        predicted_peaks,
                        true_peaks,
                        fs,
                        tolerance_ms=100.0,
                    )
                    audit_rows.append({
                        "test_row_index": int(test_row_index),
                        "peak_f1": float(window_f1),
                        "precision": float(window_precision),
                        "recall": float(window_recall),
                        "tp": int(window_tp),
                        "fp": int(window_fp),
                        "fn": int(window_fn),
                        "true_peak_count": int(len(true_peaks)),
                        "predicted_peak_count": int(len(predicted_peaks)),
                        "median_matched_offset_ms": (
                            float(np.median(matched_offsets_ms))
                            if len(matched_offsets_ms)
                            else np.nan
                        ),
                        "mean_matched_abs_offset_ms": (
                            float(np.mean(np.abs(matched_offsets_ms)))
                            if len(matched_offsets_ms)
                            else np.nan
                        ),
                        "correction_ratio": float(correction_ratio),
                        "raw_sdnn_ms": float(raw_hrv[0]),
                        "raw_rmssd_ms": float(raw_hrv[1]),
                        "corrected_sdnn_ms": float(hrv[0]),
                        "corrected_rmssd_ms": float(hrv[1]),
                    })

                test_row_index += 1

                if np.isfinite(hrv).all():
                    valid_hrv_windows += 1

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)

    f1 = (
        2.0 * precision * recall /
        max(precision + recall, 1e-12)
    )

    coverage = valid_hrv_windows / max(total_windows, 1)

    correction_ratios = np.asarray(correction_ratios, dtype=float)

    if collect_audit and audit_rows:
        _export_peak_spotcheck(
            test_loader=test_loader,
            model=model,
            device=device,
            args=args,
            participant_id=current_participant,
            threshold=threshold,
            audit_rows=audit_rows,
            output_root=audit_dir,
        )

    print(
        f"    Peak detection: "
        f"threshold={threshold:.2f} | "
        f"precision={precision:.3f} | "
        f"recall={recall:.3f} | "
        f"F1={f1:.3f}"
    )

    print(
        f"    Peak counts: "
        f"true mean={np.mean(true_peak_counts):.1f} | "
        f"predicted mean={np.mean(predicted_peak_counts):.1f}"
    )

    print(
        f"    HRV coverage: "
        f"{valid_hrv_windows}/{total_windows} "
        f"({100.0 * coverage:.1f}%)"
    )

    if np.isfinite(correction_ratios).any():
        print(
            f"    Mean RR correction ratio: "
            f"{np.nanmean(correction_ratios):.3f}"
        )

    true_hrv = args._test_y_ms
    predicted_hrv_raw = np.asarray(predicted_hrv_raw)
    predicted_hrv = np.asarray(predicted_hrv)

    paired_valid = (
        np.isfinite(predicted_hrv_raw).all(axis=1) &
        np.isfinite(predicted_hrv).all(axis=1) &
        np.isfinite(true_hrv).all(axis=1)
    )

    if paired_valid.sum() >= 2:
        raw_metrics = _regress_metrics(
            true_hrv[paired_valid],
            predicted_hrv_raw[paired_valid],
        )
        corrected_metrics = _regress_metrics(
            true_hrv[paired_valid],
            predicted_hrv[paired_valid],
        )

        print(
            "    IBI correction ablation "
            f"(paired windows={paired_valid.sum()}):"
        )

        for key, short_name in zip(
            LABEL_KEYS,
            ["SDNN", "RMSSD"],
        ):
            raw_result = raw_metrics[key]
            corrected_result = corrected_metrics[key]

            improvement = (
                raw_result["mae"] -
                corrected_result["mae"]
            )

            print(
                f"      {short_name}: "
                f"raw MAE={raw_result['mae']:.3f} ms | "
                f"corrected MAE={corrected_result['mae']:.3f} ms | "
                f"improvement={improvement:+.3f} ms | "
                f"raw r={raw_result['r']:.3f} | "
                f"corrected r={corrected_result['r']:.3f}"
            )

    valid = (
        np.isfinite(predicted_hrv).all(axis=1) &
        np.isfinite(true_hrv).all(axis=1)
    )

    if valid.sum() < 2:
        empty_result = {
            key: {
                "r2": np.nan,
                "r": np.nan,
                "me": np.nan,
                "sde": np.nan,
                "mae": np.nan,
                "n": int(valid.sum()),
            }
            for key in LABEL_KEYS
        }

        return empty_result, None

    return (
        _regress_metrics(
            true_hrv[valid],
            predicted_hrv[valid],
        ),
        None,
    )

def _find_peak_indices(prob, fs, threshold):
    """
    Convert one probability sequence into predicted peak indices.
    A minimum distance of 300 ms corresponds to a maximum HR of 200 bpm.
    """
    from scipy.signal import find_peaks

    min_distance = max(1, int(round(float(os.environ.get("PEAK_MIN_DISTANCE_MS", "300")) / 1000.0 * fs)))

    peaks, _ = find_peaks(
        prob,
        height=threshold,
        distance=min_distance,
        prominence=float(os.environ.get("PEAK_PROMINENCE", "0.0")),
    )

    return peaks.astype(np.int64)


def _match_peak_counts(predicted, truth, fs, tolerance_ms=100.0):
    """
    Greedily match predicted peaks to ECG peaks within a time tolerance.
    Returns TP, FP, FN.
    """
    predicted = np.asarray(predicted, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)

    tolerance = max(
        1,
        int(round(tolerance_ms / 1000.0 * fs)),
    )

    used = np.zeros(len(truth), dtype=bool)
    tp = 0

    for peak in predicted:
        candidates = np.where(
            (~used) & (np.abs(truth - peak) <= tolerance)
        )[0]

        if len(candidates) == 0:
            continue

        nearest = candidates[
            np.argmin(np.abs(truth[candidates] - peak))
        ]

        used[nearest] = True
        tp += 1

    fp = len(predicted) - tp
    fn = len(truth) - tp

    return tp, fp, fn


def _matched_peak_offsets_ms(
    predicted,
    truth,
    fs,
    tolerance_ms=100.0,
):
    """Return one-to-one matched offsets: predicted time minus ECG time."""
    predicted = np.asarray(predicted, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    tolerance = max(1, int(round(tolerance_ms / 1000.0 * fs)))
    used = np.zeros(len(truth), dtype=bool)
    offsets = []

    for peak in predicted:
        candidates = np.where(
            (~used) & (np.abs(truth - peak) <= tolerance)
        )[0]
        if len(candidates) == 0:
            continue
        nearest = candidates[
            np.argmin(np.abs(truth[candidates] - peak))
        ]
        used[nearest] = True
        offsets.append((peak - truth[nearest]) / fs * 1000.0)

    return np.asarray(offsets, dtype=np.float64)


def _select_peak_spotcheck_rows(audit_rows, count=6):
    """Select two worst, two median, and two best windows by peak F1."""
    ordered = sorted(
        audit_rows,
        key=lambda row: (row["peak_f1"], row["test_row_index"]),
    )
    n = len(ordered)
    if n <= count:
        return [("all", row) for row in ordered]

    candidates = [
        ("worst", ordered[0]),
        ("worst", ordered[1]),
        ("middle", ordered[max(0, n // 2 - 1)]),
        ("middle", ordered[min(n - 1, n // 2)]),
        ("best", ordered[-2]),
        ("best", ordered[-1]),
    ]

    selected = []
    used = set()
    for category, row in candidates:
        index = row["test_row_index"]
        if index not in used:
            selected.append((category, row))
            used.add(index)
    return selected


def _export_peak_spotcheck(
    test_loader,
    model,
    device,
    args,
    participant_id,
    threshold,
    audit_rows,
    output_root,
):
    """Export six representative peak windows without changing evaluation."""
    import csv
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = float(args._tgt_fs)
    device_name = str(getattr(args, "device_name", "unknown"))
    output_dir = Path(output_root) / device_name / str(participant_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_peak_spotcheck_rows(audit_rows, count=6)
    selected_by_index = {
        row["test_row_index"]: (category, row)
        for category, row in selected
    }

    true_hrv = np.asarray(args._test_y_ms)
    exported_rows = []
    global_index = 0
    model.eval()

    with torch.no_grad():
        for x, target_peak, _ in test_loader:
            batch_indices = range(global_index, global_index + len(x))
            wanted_positions = [
                position
                for position, row_index in enumerate(batch_indices)
                if row_index in selected_by_index
            ]

            if wanted_positions:
                logits, _ = model(x.to(device))
                probability = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                x_numpy = x.cpu().numpy()
                target_numpy = target_peak.cpu().numpy()

                for position in wanted_positions:
                    row_index = global_index + position
                    category, summary = selected_by_index[row_index]
                    prob = probability[position]
                    target = target_numpy[position]
                    signal = x_numpy[position]

                    predicted_peaks = _find_peak_indices(
                        prob,
                        fs,
                        threshold,
                    )
                    true_peaks = _find_peak_indices(
                        target,
                        fs,
                        threshold=0.50,
                    )

                    reference_mask = np.zeros(len(prob), dtype=np.int8)
                    prediction_mask = np.zeros(len(prob), dtype=np.int8)
                    reference_mask[true_peaks] = 1
                    prediction_mask[predicted_peaks] = 1

                    stem = f"{category}_row_{row_index:04d}"
                    csv_path = output_dir / f"{stem}.csv"
                    time_s = np.arange(len(prob), dtype=float) / fs
                    green = signal[0]
                    ir = signal[1] if signal.shape[0] > 1 else np.full_like(green, np.nan)

                    with csv_path.open("w", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow([
                            "time_s",
                            "ppg_green_normalized",
                            "ppg_ir_normalized",
                            "reference_target",
                            "predicted_probability",
                            "is_reference_peak",
                            "is_predicted_peak",
                        ])
                        writer.writerows(zip(
                            time_s,
                            green,
                            ir,
                            target,
                            prob,
                            reference_mask,
                            prediction_mask,
                        ))

                    display_n = min(len(prob), int(round(30.0 * fs)))
                    display_t = time_s[:display_n]
                    display_predicted = predicted_peaks[predicted_peaks < display_n]
                    display_true = true_peaks[true_peaks < display_n]

                    figure, axes = plt.subplots(
                        3,
                        1,
                        figsize=(16, 9),
                        sharex=True,
                    )
                    channels = [
                        (green, "Green PPG (normalized)", "tab:green"),
                        (ir, "IR PPG (normalized)", "tab:purple"),
                    ]
                    for axis, (wave, label, color) in zip(axes[:2], channels):
                        axis.plot(display_t, wave[:display_n], color=color, linewidth=0.8)
                        for peak in display_true:
                            axis.axvline(peak / fs, color="tab:blue", alpha=0.35, linewidth=0.8)
                        for peak in display_predicted:
                            axis.axvline(peak / fs, color="tab:red", alpha=0.35, linewidth=0.8)
                        axis.set_ylabel(label)
                        axis.grid(alpha=0.15)

                    axes[2].plot(display_t, prob[:display_n], color="black", linewidth=0.9)
                    axes[2].axhline(
                        threshold,
                        color="tab:red",
                        linestyle="--",
                        label=f"threshold={threshold:.2f}",
                    )
                    axes[2].scatter(
                        display_true / fs,
                        target[display_true],
                        color="tab:blue",
                        s=24,
                        label="ECG reference event",
                        zorder=3,
                    )
                    axes[2].scatter(
                        display_predicted / fs,
                        prob[display_predicted],
                        color="tab:red",
                        marker="x",
                        s=32,
                        label="predicted event",
                        zorder=3,
                    )
                    axes[2].set_ylabel("Peak probability")
                    axes[2].set_xlabel("Time (seconds)")
                    axes[2].set_ylim(-0.03, 1.03)
                    axes[2].legend(loc="upper right")
                    axes[2].grid(alpha=0.15)

                    figure.suptitle(
                        f"{device_name}/{participant_id} row={row_index} "
                        f"({category}) | F1={summary['peak_f1']:.3f} | "
                        f"true={summary['true_peak_count']} pred={summary['predicted_peak_count']} | "
                        f"median offset={summary['median_matched_offset_ms']:.1f} ms"
                    )
                    figure.tight_layout()
                    figure.savefig(output_dir / f"{stem}.png", dpi=160)
                    plt.close(figure)

                    result = dict(summary)
                    result.update({
                        "category": category,
                        "participant": participant_id,
                        "device": device_name,
                        "threshold": float(threshold),
                        "true_sdnn_ms": float(true_hrv[row_index, 0]),
                        "true_rmssd_ms": float(true_hrv[row_index, 1]),
                        "plot_path": str(output_dir / f"{stem}.png"),
                        "sample_csv_path": str(csv_path),
                    })
                    exported_rows.append(result)

            global_index += len(x)

    summary_path = output_dir / "spotcheck_summary.csv"
    if exported_rows:
        fieldnames = list(exported_rows[0].keys())
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(exported_rows)

    print(
        f"    Peak spot-check: exported {len(exported_rows)} windows "
        f"to {output_dir}"
    )

def select_peak_threshold(
    val_loader,
    model,
    device,
    args,
    thresholds=None,
    return_score=False,
    quiet=False,
):
    """
    Select the peak probability threshold using validation-set peak F1.
    The test participant is not used for threshold selection.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.81, 0.05)
        fixed_threshold = os.environ.get("PEAK_FIXED_THRESHOLD", "").strip()
        if fixed_threshold:
            thresholds = [float(fixed_threshold)]

    model.eval()
    fs = args._tgt_fs

    probabilities = []
    targets = []

    with torch.no_grad():
        for x, y, _ in val_loader:
            logits, _ = model(x.to(device))

            prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            probabilities.extend(prob)
            targets.extend(y.numpy())

    best_threshold = 0.30
    best_f1 = -1.0

    for threshold in thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for prob, target in zip(probabilities, targets):
            predicted_peaks = _find_peak_indices(
                prob,
                fs,
                threshold,
            )

            true_peaks = _find_peak_indices(
                target,
                fs,
                threshold=0.50,
            )

            tp, fp, fn = _match_peak_counts(
                predicted_peaks,
                true_peaks,
                fs,
                tolerance_ms=100.0,
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)

        f1 = (
            2.0 * precision * recall /
            max(precision + recall, 1e-12)
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    if not quiet:
        print(
            f"    Selected peak threshold: {best_threshold:.2f} "
            f"| validation F1: {best_f1:.3f}"
        )

    if return_score:
        return best_threshold, best_f1
    return best_threshold

def _correct_rr_intervals(
    rr,
    threshold=0.20,
    min_rr_ms=300.0,
    max_rr_ms=2000.0,
):
    """
    Apply the same median-filter correction used for the corrected ECG labels.
    """
    from scipy.ndimage import median_filter

    rr = np.asarray(rr, dtype=np.float64)

    if rr.size < 5:
        return rr.copy(), 0.0

    local_median = median_filter(
        rr,
        size=11,
        mode="reflect",
    )

    physiological_artifact = (
        (rr < min_rr_ms) |
        (rr > max_rr_ms) |
        ~np.isfinite(rr)
    )
    local_artifact = (
        np.abs(rr - local_median) >
        threshold * local_median
    )
    artifact = physiological_artifact | local_artifact

    corrected = rr.copy()
    corrected[artifact] = local_median[artifact]

    return corrected, float(np.mean(artifact))


def _remove_abnormal_rr_intervals(
    rr,
    threshold=0.20,
    min_rr_ms=300.0,
    max_rr_ms=3000.0,
    reject_ratio=0.50,
):
    """Remove abnormal IBIs and reject windows with excessive removal.

    This implements the rule described by Sarhaddi et al.: retain intervals in
    the physiological range and within a specified fraction of the window's
    mean normal interval; reject the window when too many intervals are lost.
    """
    rr = np.asarray(rr, dtype=np.float64)
    physiological = (
        np.isfinite(rr) &
        (rr >= min_rr_ms) &
        (rr <= max_rr_ms)
    )

    initial = rr[physiological]
    if initial.size < 3:
        return None, 1.0, None

    mean_rr = float(np.mean(initial))
    normal = physiological & (
        np.abs(rr - mean_rr) <= threshold * mean_rr
    )
    removal_ratio = float(1.0 - np.mean(normal))

    cleaned = rr[normal]
    if removal_ratio > reject_ratio or cleaned.size < 3:
        return None, removal_ratio, None

    return cleaned, removal_ratio, normal


def _hrv_from_rr(rr, original_valid_mask=None):
    """Return HRV without joining intervals separated by a removed IBI."""
    rr = np.asarray(rr, dtype=np.float64)
    if rr.size < 3 or not np.isfinite(rr).all():
        return np.array([np.nan, np.nan])
    if original_valid_mask is None:
        differences = np.diff(rr)
    else:
        original_valid_mask = np.asarray(original_valid_mask, dtype=bool)
        # Reconstruct the original sequence through the caller-provided mask.
        # The cleaned values retain their original order; only consecutive
        # retained intervals contribute to RMSSD.
        original_positions = np.flatnonzero(original_valid_mask)
        consecutive = np.diff(original_positions) == 1
        differences = np.diff(rr)[consecutive]
    if differences.size < 1:
        return np.array([np.std(rr, ddof=1), np.nan])
    return np.array([
        np.std(rr, ddof=1),
        np.sqrt(np.mean(differences ** 2)),
    ])


def _hrv_from_prob(
    prob,
    fs,
    thr=0.30,
    min_rr_ms=300.0,
    max_rr_ms=2000.0,
    ibi_method='median',
    deviation_threshold=0.20,
    reject_ratio=0.50,
):
    predicted_peaks = _find_peak_indices(
        prob,
        fs,
        threshold=thr,
    )

    if len(predicted_peaks) < 4:
        missing = np.array([np.nan, np.nan])
        return missing.copy(), missing.copy(), len(predicted_peaks), np.nan

    peak_times_ms = predicted_peaks / fs * 1000.0
    rr = np.diff(peak_times_ms)

    valid = (
        (rr >= min_rr_ms) &
        (rr <= max_rr_ms)
    )

    valid_rr = rr[valid]

    if len(valid_rr) < 3:
        missing = np.array([np.nan, np.nan])
        return missing.copy(), missing.copy(), len(predicted_peaks), np.nan

    raw_sdnn = np.std(valid_rr, ddof=1)

    # RMSSD is defined only over originally adjacent valid intervals. Do not
    # concatenate across an interval removed by physiological range filtering.
    valid_pairs = valid[:-1] & valid[1:]
    if valid_pairs.sum() < 1:
        missing = np.array([np.nan, np.nan])
        return missing.copy(), missing.copy(), len(predicted_peaks), np.nan
    raw_rmssd = np.sqrt(np.mean(np.diff(rr)[valid_pairs] ** 2))
    raw_hrv = np.array([raw_sdnn, raw_rmssd])

    if ibi_method == 'raw':
        physiological_ratio = float(1.0 - np.mean(valid))
        return raw_hrv, raw_hrv.copy(), len(predicted_peaks), physiological_ratio

    if ibi_method == 'remove':
        cleaned_rr, removal_ratio, normal_mask = _remove_abnormal_rr_intervals(
            rr,
            threshold=deviation_threshold,
            min_rr_ms=min_rr_ms,
            max_rr_ms=max_rr_ms,
            reject_ratio=reject_ratio,
        )
        if cleaned_rr is None:
            missing = np.array([np.nan, np.nan])
            return raw_hrv, missing, len(predicted_peaks), removal_ratio
        return (
            raw_hrv,
            _hrv_from_rr(cleaned_rr, original_valid_mask=normal_mask),
            len(predicted_peaks),
            removal_ratio,
        )

    if ibi_method != 'median':
        raise ValueError(f"Unknown ibi_method: {ibi_method!r}")

    corrected_rr, correction_ratio = _correct_rr_intervals(
        rr,
        threshold=0.20,
        min_rr_ms=min_rr_ms,
        max_rr_ms=max_rr_ms,
    )

    if len(corrected_rr) < 3:
        missing = np.array([np.nan, np.nan])
        return raw_hrv, missing, len(predicted_peaks), correction_ratio

    sdnn = np.std(corrected_rr, ddof=1)
    rmssd = np.sqrt(
        np.mean(np.diff(corrected_rr) ** 2)
    )

    return (
        raw_hrv,
        np.array([sdnn, rmssd]),
        len(predicted_peaks),
        correction_ratio,
    )


def _regress_metrics(yt, yp):
    out = {}
    for j, n in enumerate(LABEL_KEYS):
        t, p = yt[:, j], yp[:, j]
        if t.size < 2:
            out[n] = dict(r2=float("nan"), r=float("nan"), me=float("nan"),
                          sde=float("nan"), mae=float("nan"), n=int(t.size))
            continue
        ss = np.sum((t - p) ** 2)
        r2 = 1 - ss / (np.sum((t - t.mean()) ** 2) + 1e-12)
        if p.std() < 1e-8 or t.std() < 1e-8:
            r = float("nan")
        else:
            r = float(np.corrcoef(t, p)[0, 1])
        out[n] = dict(r2=r2, r=r, me=float(np.mean(p - t)),
                      sde=float(np.std(p - t)), mae=float(np.mean(np.abs(p - t))),
                      n=int(t.size))
    return out
