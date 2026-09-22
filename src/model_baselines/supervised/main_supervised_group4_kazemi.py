# encoding=utf-8
"""
main_supervised_baseline.py
----------------------------
Supervised baseline for PPG heart rate regression.

- Always performs regression (predicts bpm directly).
- Participants are discovered automatically from --data_dir subfolders.
- Leave-one-subject-out evaluation.

Usage:
    # Single-device (one wearable position)
    python main_supervised_baseline.py \\
        --dataset ppg --position ring \\
        --backbone DCL --data_dir /path/to/data --cuda 0

    # Multi-site (4 devices fused)
    python main_supervised_baseline.py \\
        --dataset multisite \\
        --backbone DCL --data_dir /path/to/data --cuda 0
"""

import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

from models.backbones import FCN, DeepConvLSTM, cnn_lstm, LSTM, Transformer
from models.models_nc import ResNet1D
from data_preprocess.data_prep import setup_dataloaders
from data_preprocess.participants_config import discover_participants
from data_preprocess.data_preprocess_multisite import discover_multisite_participants
import hrv_ext_group4_kazemi as H


# ── Argument parser ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Supervised baseline — PPG HR regression')

# Hardware
parser.add_argument('--cuda', default=0, type=int,
                    help='CUDA device index (0, 1, …). Falls back to CPU if unavailable.')

# Training hyperparameters
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epoch',    type=int, default=20)
parser.add_argument('--lr',         type=float, default=5e-4)
parser.add_argument('--patience',   type=int, default=5)

# Dataset
parser.add_argument('--dataset', default='ppg', type=str,
                    choices=['ppg', 'multisite'],
                    help='Dataset name. dataset=one device per run; multisite=4 devices fused.')
parser.add_argument('--position', default=None, type=str,
                    choices=['ring', 'earring', 'necklace', 'watch', None],
                    help='Wearable body position. Required for ppg, not used for multisite.')
parser.add_argument('--data_dir', default='../../../Multisite-PPG/ppg_windowed_data', type=str,
                    help='Root directory containing per-participant subfolders'
                         '(e.g. P1/, P2/, …).')
parser.add_argument('--split_ratio', default=0.2, type=float,
                    help='Fraction of source-participant windows held out for validation.')
parser.add_argument('--cases', default='subject_val', type=str,
                    choices=['subject_val'],
                    help='Evaluation protocol. Only subject_val (leave-one-out) is supported.')
parser.add_argument('--single_device', type=int, default=None,
                    choices=[None, 0, 1, 2, 3],
                    help='For multisite dataset: None=all 4 devices; '
                         '0=earring, 1=ring, 2=watch, 3=necklace (single device on multisite split)')

# Backbone
parser.add_argument('--backbone', default='DCL', type=str,
                    choices=['FCN', 'DCL', 'cnn_lstm', 'LSTM', 'Transformer', 'resnet'],
                    help='Encoder architecture.')

# Logging
parser.add_argument('--logdir', default='log/', type=str,
                    help='Directory for per-run log files.')
parser.add_argument('--use_preprocess', action='store_true')

parser.add_argument('--task', default='hr', choices=['hr', 'hrv', 'peak', 'hrv_seg'])
parser.add_argument('--device_name', default='watch', choices=['earring','ring','watch'])
parser.add_argument('--use_deriv', type=int, default=1)
parser.add_argument(
    '--peak_channels', default='both', choices=['green', 'ir', 'both'],
    help='PPG channels used by the peak task. Ignored for other tasks.',
)
parser.add_argument('--resample_hz', type=float, default=0)   # 0 = keep 100Hz; 50 recommended first
parser.add_argument('--src_hz', type=float, default=100.0)
parser.add_argument('--limit', type=int, default=0)           # cap windows/subject (smoke); 0 = no cap
parser.add_argument('--agg', default='mean', choices=['mean','lstm'])
parser.add_argument(
    '--peak_backbone',
    default='unet',
    choices=['unet', 'dilated', 'kazemi'],
    help='Peak detector: U-Net, our dilated model, or Kazemi et al. architecture.',
)
parser.add_argument(
    '--ibi_method',
    default='median',
    choices=['raw', 'median', 'remove'],
    help='IBI handling: none, local-median replacement, or remove/reject ablation.',
)
parser.add_argument('--min_rr_ms', type=float, default=300.0)
parser.add_argument('--max_rr_ms', type=float, default=2000.0)
parser.add_argument('--ibi_deviation_threshold', type=float, default=0.20)
parser.add_argument('--ibi_reject_ratio', type=float, default=0.50)
parser.add_argument(
    "--only_target",
    default=None,
    help="Run only one held-out participant, e.g. P1.",
)
parser.add_argument(
    "--group_fold",
    type=int,
    choices=[0, 1, 2, 3],
    default=None,
    help="Run one leakage-safe participant-grouped 4-fold split.",
)
parser.add_argument(
    "--group_output_dir",
    default="group4_results",
    help="Output directory for split manifest, predictions, and metrics.",
)
# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(args):
    """
    Instantiate the chosen backbone in regression mode (n_classes=1).
    All models receive n_channels=args.n_feature
    (1 for single, 4 for multisite, 8 for multisite+accel).
    """
    if args.task == 'peak':
        if args.peak_backbone == 'kazemi':
            return H.KazemiPeakNet(H.n_input_channels(args))
        if args.peak_backbone == 'dilated':
            return H.DilatedPeakNet(H.n_input_channels(args))
        return H.PeakNet(H.n_input_channels(args))
    if args.task == 'hrv_seg':
        return H.SegNet(H.n_input_channels(args), agg=args.agg) # Path C
    n_cls = 2 if args.task == 'hrv' else 1          # Path A: 2 HRV scalars
    n_ch = args.n_feature
    len_ = args.len_sw 

    if args.backbone == 'FCN':
        return FCN(n_channels=n_ch, in_dim=len_, n_classes=n_cls,
                   backbone=False, regress=True)
    elif args.backbone == 'DCL':
        return DeepConvLSTM(n_channels=n_ch, n_classes=n_cls,
                            conv_kernels=64, kernel_size=5, LSTM_units=128,
                            backbone=False, regress=True)
    elif args.backbone == 'cnn_lstm':
        return cnn_lstm(n_channels=n_ch, n_classes=n_cls,
                        backbone=False, regress=True)
    elif args.backbone == 'LSTM':
        return LSTM(n_channels=n_ch, n_classes=n_cls, LSTM_units=256,
                    backbone=False, regress=True)
    elif args.backbone == 'Transformer':
        return Transformer(n_channels=n_ch, len_sw=len_, n_classes=n_cls,
                           dim=128, depth=4, heads=4, mlp_dim=64,
                           dropout=0.1, backbone=False, regress=True)
    elif args.backbone == 'resnet':
        return ResNet1D(in_channels=n_ch, base_filters=32, kernel_size=5,
                        stride=2, groups=1, n_block=8, n_classes=n_cls,
                        downsample_gap=2, increasefilter_gap=4,
                        backbone=False, regress=True)
    else:
        raise NotImplementedError(f"Unknown backbone: {args.backbone}")


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args, train_loaders, val_loader, model, device, optimizer, criterion,
          save_dir='results/'):
    """
    Train with early stopping based on validation event F1 for peak detection,
    and validation loss for the other tasks.
    Saves the best checkpoint to save_dir/{model_name}.pt.

    Returns the best model state dict.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, args.model_name + '.pt')

    min_val_loss  = float('inf')
    best_val_f1   = -float('inf')
    best_state    = None
    epochs_no_imp = 0

    for epoch in range(args.n_epoch):

        # ── training
        model.train()
        train_loss = 0.0
        for sample, target, _ in train_loaders[0]:
            sample = sample.to(device)
            target = target.to(device)

            out, _ = model(sample)
            loss   = criterion(out.squeeze(1), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loaders[0])

        # ── validation
        if val_loader is None:
            best_state = deepcopy(model.state_dict())
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       checkpoint_path)
            continue

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for sample, target, _ in val_loader:
                sample = sample.to(device)
                target = target.to(device)
                out, _ = model(sample)
                val_loss += criterion(out.squeeze(1), target).item()
                n_val_batches += 1
        val_loss /= n_val_batches

        val_f1 = None
        if args.task == 'peak':
            _, val_f1 = H.select_peak_threshold(
                val_loader, model, device, args,
                return_score=True, quiet=True,
            )

        improved = (
            val_f1 > best_val_f1
            if args.task == 'peak'
            else val_loss < min_val_loss
        )

        if improved:
            min_val_loss  = val_loss
            if val_f1 is not None:
                best_val_f1 = val_f1
            best_state    = deepcopy(model.state_dict())
            epochs_no_imp = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       checkpoint_path)
            metric_text = f" | Val event F1: {val_f1:.3f}" if val_f1 is not None else ""
            print(f"    Epoch {epoch+1:>3}/{args.n_epoch}  |  "
                  f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}"
                  f"{metric_text} *")
        else:
            epochs_no_imp += 1
            metric_text = f" | Val event F1: {val_f1:.3f}" if val_f1 is not None else ""
            print(f"    Epoch {epoch+1:>3}/{args.n_epoch}  |  "
                  f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}"
                  f"{metric_text}")
            if args.patience > 0 and epochs_no_imp >= args.patience:
                print(f"    Early stopping at epoch {epoch + 1} "
                      f"(no improvement for {args.patience} epochs)")
                break

    return best_state


# ── Evaluation ────────────────────────────────────────────────────────────────

def test(test_loader, model, device, criterion,
         save_path=None, append=False, participant_id=None):
    """
    Evaluate on the test set.

    Returns:
        mae  : Mean Absolute Error (bpm)
        rmse : Root Mean Squared Error (bpm)
        r    : Pearson correlation coefficient
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for sample, target, _ in test_loader:
            sample = sample.to(device)
            target = target.to(device)
            out, _ = model(sample)
            all_preds.append(out.squeeze(1).cpu())
            all_targets.append(target.cpu())

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    r    = float(np.corrcoef(preds, targets)[0, 1])
    if np.isnan(r):
        r = 0.0

    # save predictions (appending across participants)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mode = 'a' if append else 'w'
        file_exists = os.path.exists(save_path)

        with open(save_path, mode) as f:
            if mode == 'w' or not file_exists:
                f.write('participant ground_truth_bpm predicted_bpm\n')
            tag = participant_id if participant_id is not None else 'unknown'
            for t, p in zip(targets, preds):
                f.write(f'{tag} {t:.3f} {p:.3f}\n')

    return mae, rmse, r


# ── Per-participant training run ──────────────────────────────────────────────

def train_sup(args, seed_idx: int):
    """
    Full train->test cycle for one (participant, seed) combination.

    args.target_domain must be set to the held-out participant ID (e.g. "P3")
    before calling this function.

    Returns:
        (mae, rmse, r)  — single test result
    """
    set_seed(seed_idx * 10 + np.random.randint(0, 10))

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    if args.task in ('hrv', 'peak', 'hrv_seg'):
        train_loaders, val_loader, test_loader = H.setup_dataloaders_hrv(args)
    else:
        train_loaders, val_loader, test_loader = setup_dataloaders(args)

    model = build_model(args).to(device)

    position_tag = args.position if args.dataset == 'ppg' else 'multisite'
    args.model_name = (
        f"{args.backbone}_{args.dataset}_{position_tag}"
        f"_target{args.target_domain}"
        f"_seed{seed_idx}"
        f"_lr{args.lr}_bs{args.batch_size}"
    )
    if args.task == 'peak':
        args.model_name += (
            f"_peak{args.peak_backbone}_{args.peak_channels}"
            f"_ibi{args.ibi_method}"
        )

    os.makedirs(args.logdir, exist_ok=True)
    log_path = os.path.join(args.logdir, args.model_name + '.log')
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s %(message)s')

    criterion = H.make_criterion(args) if args.task in ('hrv','peak','hrv_seg') else nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state = train(args, train_loaders, val_loader, model, device, optimizer, criterion)

    model_test = build_model(args).to(device)
    model_test.load_state_dict(best_state)

    if args.task == 'peak':
        args._peak_threshold = H.select_peak_threshold(
            val_loader,
            model_test,
            device,
            args,
        )

    pred_path = f"predictions/{args.backbone}_{args.dataset}_{position_tag}.txt"

    if args.task in ('hrv', 'hrv_seg'):
        res, bres = H.test_hrv(test_loader, model_test, device, args, args.target_domain)
        return ('hrv', res, bres)
    elif args.task == 'peak':
        res, bres = H.test_peak(test_loader, model_test, device, args, args.target_domain)
        return ('peak', res, bres)
    else:
        mae, rmse, r = test(test_loader, model_test, device, criterion,
                            save_path=pred_path, append=True, participant_id=args.target_domain)
        return ('hr', mae, rmse, r)


# ── Dataset config ────────────────────────────────────────────────────────────

def configure_dataset(args):
    if args.task in ('hrv', 'peak', 'hrv_seg'):
        step = int(round(args.src_hz / args.resample_hz)) if args.resample_hz else 1
        args.len_sw    = (30000 + step - 1) // step      # length after downsampling
        args.out_dim   = args.len_sw
        args.n_feature = H.n_input_channels(args)         # = 4 (green+ir+derivative)
        args.position  = None
        parts = H.discover_hrv_participants(args.data_dir)
        if not parts:
            raise FileNotFoundError(f"No *_P*.npz in {args.data_dir}")
        print(f"[{args.task}] {len(parts)} participants: {parts}")
        return parts

# ---- keep original hr logic below unchanged ----
    args.len_sw  = 200
    args.out_dim = 200
    if args.dataset == 'ppg':
        args.n_feature = 1      # green only, single position
        if args.position is None:
            raise ValueError("--position is required for --dataset ppg.")
        participants = discover_participants(args.data_dir, args.position)
        if not participants:
            raise FileNotFoundError(
                f"No valid participant folders found in '{args.data_dir}' "
                f"for position '{args.position}'."
            )
        print(f"Found {len(participants)} participants for '{args.position}': {participants}")

    elif args.dataset == 'multisite':
        # n_feature: 4 if all devices fused, 1 if isolating a single device
        args.n_feature = 1 if args.single_device is not None else 4
        args.position  = None   # not used for multisite
        participants   = discover_multisite_participants(args.data_dir)
        if not participants:
            raise FileNotFoundError(
                f"No participants with aligned_4device.npz found in '{args.data_dir}'. "
                f"Run align.py --modality green first."
            )
        print(f"Found {len(participants)} multisite participants: {participants}")

    else:
        raise ValueError(f"Unknown dataset: '{args.dataset}'.")

    return participants


GROUP4_TEST_FOLDS = {
    0: ["P3", "P5", "P11", "P20"],
    1: ["P18", "P1", "P13", "P7"],
    2: ["P9", "P6", "P15", "P19"],
    3: ["P4", "P12", "P8", "P10"],
}

GROUP4_VALIDATION = {
    0: ["P7", "P12"],
    1: ["P10", "P15"],
    2: ["P4", "P8"],
    3: ["P13", "P19"],
}


def _group4_split(participants, fold_index):
    all_pids = set(participants)
    expected = set().union(*map(set, GROUP4_TEST_FOLDS.values()))
    if all_pids != expected:
        raise ValueError(
            "Grouped split participant mismatch: "
            f"data={sorted(all_pids)} expected={sorted(expected)}"
        )

    test_pids = GROUP4_TEST_FOLDS[fold_index]
    val_pids = GROUP4_VALIDATION[fold_index]
    train_pids = sorted(all_pids - set(test_pids) - set(val_pids))

    train_set, val_set, test_set = map(set, (train_pids, val_pids, test_pids))
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Participant leakage detected in grouped split")
    if train_set | val_set | test_set != all_pids:
        raise RuntimeError("Grouped split does not cover all participants")
    return train_pids, list(val_pids), list(test_pids)


def run_group4_peak(args, participants, seed_idx=0):
    if args.task != "peak":
        parser.error("--group_fold currently supports --task peak only")

    fold_index = int(args.group_fold)
    train_pids, val_pids, test_pids = _group4_split(participants, fold_index)

    print(f"Group fold: {fold_index}")
    print(f"Train participants ({len(train_pids)}): {train_pids}")
    print(f"Validation participants ({len(val_pids)}): {val_pids}")
    print(f"Test participants ({len(test_pids)}): {test_pids}")
    print("Leakage check: PASS")

    output_dir = Path(args.group_output_dir) / f"fold_{fold_index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "fold": fold_index,
        "train_participants": train_pids,
        "validation_participants": val_pids,
        "test_participants": test_pids,
        "seed": int(seed_idx),
    }
    (output_dir / "split.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n"
    )

    set_seed(seed_idx * 10 + 1)
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    )
    train_loaders, val_loader, test_loaders, test_metadata = (
        H.setup_dataloaders_peak_group(
            args,
            train_pids=train_pids,
            val_pids=val_pids,
            test_pids=test_pids,
        )
    )

    args.target_domain = f"group4_fold{fold_index}"
    args.model_name = (
        f"{args.backbone}_{args.dataset}_group4fold{fold_index}"
        f"_seed{seed_idx}_lr{args.lr}_bs{args.batch_size}"
        f"_peak{args.peak_backbone}_{args.peak_channels}"
        f"_ibi{args.ibi_method}"
    )
    args.logdir = str(output_dir / "train_log")

    model = build_model(args).to(device)
    criterion = H.make_criterion(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state = train(
        args,
        train_loaders,
        val_loader,
        model,
        device,
        optimizer,
        criterion,
        save_dir=str(output_dir / "checkpoints"),
    )

    model_test = build_model(args).to(device)
    model_test.load_state_dict(best_state)
    args._peak_threshold = H.select_peak_threshold(
        val_loader,
        model_test,
        device,
        args,
    )

    participant_metrics = {}
    prediction_rows = []
    pooled_tp = pooled_fp = pooled_fn = 0
    for pid in test_pids:
        print(f"\n  Test participant: {pid}")
        args._test_y_ms = test_metadata[pid]["y_ms"]
        args._test_base = test_metadata[pid]["base"]
        metrics, _ = H.test_peak(
            test_loaders[pid],
            model_test,
            device,
            args,
            participant_id=pid,
        )
        participant_metrics[pid] = metrics

        evaluation = args._last_peak_eval
        pooled_tp += evaluation["tp"]
        pooled_fp += evaluation["fp"]
        pooled_fn += evaluation["fn"]
        true_hrv = evaluation["true_hrv"]
        predicted_hrv = evaluation["predicted_hrv"]
        for window_index, (true_values, predicted_values) in enumerate(
            zip(true_hrv, predicted_hrv)
        ):
            prediction_rows.append({
                "fold": fold_index,
                "participant": pid,
                "window_index": window_index,
                "true_sdnn_ms": float(true_values[0]),
                "pred_sdnn_ms": float(predicted_values[0]),
                "true_rmssd_ms": float(true_values[1]),
                "pred_rmssd_ms": float(predicted_values[1]),
            })

    prediction_path = output_dir / "predictions.csv"
    with prediction_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    true_all = np.asarray([
        [row["true_sdnn_ms"], row["true_rmssd_ms"]]
        for row in prediction_rows
    ])
    pred_all = np.asarray([
        [row["pred_sdnn_ms"], row["pred_rmssd_ms"]]
        for row in prediction_rows
    ])
    pooled_valid = np.isfinite(true_all).all(1) & np.isfinite(pred_all).all(1)
    pooled_metrics = H._regress_metrics(
        true_all[pooled_valid],
        pred_all[pooled_valid],
    )
    peak_precision = pooled_tp / max(pooled_tp + pooled_fp, 1)
    peak_recall = pooled_tp / max(pooled_tp + pooled_fn, 1)
    peak_f1 = 2 * peak_precision * peak_recall / max(
        peak_precision + peak_recall, 1e-12
    )

    summary = {
        **split_manifest,
        "threshold": float(args._peak_threshold),
        "participant_metrics": participant_metrics,
        "pooled_metrics": pooled_metrics,
        "pooled_peak_detection": {
            "tp": pooled_tp,
            "fp": pooled_fp,
            "fn": pooled_fn,
            "precision": peak_precision,
            "recall": peak_recall,
            "f1": peak_f1,
        },
        "pooled_valid_windows": int(pooled_valid.sum()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(f"\nGroup fold {fold_index} pooled held-out windows: {pooled_valid.sum()}")
    print(
        f"Group fold {fold_index} pooled peak detection: "
        f"precision={peak_precision:.3f} recall={peak_recall:.3f} "
        f"F1={peak_f1:.3f}"
    )
    for key in H.LABEL_KEYS:
        result = pooled_metrics[key]
        print(
            f"Group fold {fold_index} pooled {key}: "
            f"R2={result['r2']:.3f} r={result['r']:.3f} "
            f"MAE={result['mae']:.2f} ms"
        )
    print(f"Group fold results: {output_dir}")
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    set_seed(40)
    args = parser.parse_args()

    participants = configure_dataset(args)

    if args.group_fold is not None:
        run_group4_peak(args, participants, seed_idx=0)
        raise SystemExit(0)

    if args.only_target is not None:
        if args.only_target not in participants:
            parser.error(
                f"--only_target {args.only_target!r} is not in "
                f"available participants: {participants}"
            )

        participants = [args.only_target]

        print(
            f"Only target participant: "
            f"{args.only_target}"
        )
    all_seed_results = []

    for seed_idx in range(1):  # change to range(3) to run multiple seeds
        print(f"\n{'='*60}")
        print(f"Seed {seed_idx + 1}")
        print('='*60)

        fold_results = []
        for pid in participants:
            print(f"\n  Target participant: {pid}")
            args.target_domain = pid
            out = train_sup(args, seed_idx)

            if out[0] == 'hr':
                _, mae, rmse, r = out
                fold_results.append([mae, rmse, r])
                print(f"    MAE {mae:.2f} | RMSE {rmse:.2f} | R {r:.4f}")
            else:                                   # hrv / peak: per-metric R2/MAE
                _, res, bres = out
                fold_results.append(res)
                for k, v in res.items():
                    print(f"    {k:12s} R2={v['r2']:6.3f}  r={v['r']:6.3f}  MAE={v['mae']:6.2f} ms")
                if bres:
                    for k, v in bres.items():
                        print(f"    (PRV) {k:12s} R2={v['r2']:6.3f}  r={v['r']:6.3f}  MAE={v['mae']:6.2f} ms")
        all_seed_results.append(fold_results)


    # ── Final summary across all participants
    if args.task in ('hrv', 'peak', 'hrv_seg'):
        flat = [r for seed in all_seed_results for r in seed]   # list of dict
        print(f"\n{'='*60}\nFinal ({len(flat)} folds, mean +/- std):")
        for k in H.LABEL_KEYS:
            r2  = np.array([f[k]['r2']  for f in flat])
            rr  = np.array([f[k]['r']   for f in flat])
            mae = np.array([f[k]['mae'] for f in flat])
            print(f"  {k:12s} R2 {np.nanmean(r2):.3f}+/-{np.nanstd(r2):.3f}  "
                  f"r {np.nanmean(rr):.3f}+/-{np.nanstd(rr):.3f}  "
                  f"MAE {np.nanmean(mae):.2f}+/-{np.nanstd(mae):.2f} ms")
    else:
        flat = np.array(all_seed_results).reshape(-1, 3)   # ← (n_seeds * n_participants, 3)
        overall_mean = flat.mean(axis=0)
        overall_std  = flat.std(axis=0)
        print(f"\n{'='*60}")
        print(f"Final results (mean ± std across {flat.shape[0]} participant-runs)")

    # ── Save summary to file ─────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    summary_path = (
        f"results/summary_{args.backbone}_{args.dataset}_"
        f"{args.position or 'multisite'}.txt"
    )

    with open(summary_path, "w") as f:
        f.write(f"Backbone     : {args.backbone}\n")
        f.write(f"Dataset      : {args.dataset}\n")
        f.write(f"Position     : {args.position or 'multisite'}\n")

        if args.task in ("hrv", "peak", "hrv_seg"):
            f.write(f"Participants : {len(flat)}\n\n")

            for key in H.LABEL_KEYS:
                r2_values = np.array(
                    [result[key]["r2"] for result in flat]
                )
                r_values = np.array(
                    [result[key]["r"] for result in flat]
                )
                mae_values = np.array(
                    [result[key]["mae"] for result in flat]
                )

                f.write(f"{key}\n")
                f.write(
                    f"  R2  : {np.nanmean(r2_values):.3f} "
                    f"+/- {np.nanstd(r2_values):.3f}\n"
                )
                f.write(
                    f"  r   : {np.nanmean(r_values):.3f} "
                    f"+/- {np.nanstd(r_values):.3f}\n"
                )
                f.write(
                    f"  MAE : {np.nanmean(mae_values):.2f} "
                    f"+/- {np.nanstd(mae_values):.2f} ms\n\n"
                )
        else:
            f.write(f"Participants : {flat.shape[0]}\n\n")
            f.write(
                f"MAE  : {overall_mean[0]:.2f} "
                f"+/- {overall_std[0]:.2f} bpm\n"
            )
            f.write(
                f"RMSE : {overall_mean[1]:.2f} "
                f"+/- {overall_std[1]:.2f} bpm\n"
            )
            f.write(
                f"R    : {overall_mean[2]:.4f} "
                f"+/- {overall_std[2]:.4f}\n"
            )

    print(f"Summary saved to {summary_path}")
