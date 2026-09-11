from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ml_hrv.config import ExperimentConfig
from ml_hrv.data.schema import validate_npz_schema
from ml_hrv.data.dataset import RawslotWindowDataset
from ml_hrv.data.staging import write_robust_stats_from_cache
from ml_hrv.data.splits import make_group_folds
from ml_hrv.evaluation.evaluator import evaluate_model
from ml_hrv.evaluation.beat_events import match_beat_events
from ml_hrv.evaluation.reporting import write_run_artifacts
from ml_hrv.inference import CausalHRVMonitor
from ml_hrv.models.pipeline import RawContinuousHRVModel
from ml_hrv.physiology import correct_rr_adjacency, hrv_from_corrected_rr
from ml_hrv.training.checkpointing import load_checkpoint, save_checkpoint
from ml_hrv.training.losses import DelayAwareBeatLoss, PipelineLoss, TargetScaler
from ml_hrv.training.targets import build_ecg_beat_targets, shift_candidates_with_zeros, shift_with_zeros


class PipelineTests(unittest.TestCase):
    def tiny_config(self) -> ExperimentConfig:
        config = ExperimentConfig()
        config.data.fs_hz = 20.0
        config.data.window_seconds = 20
        config.data.segment_seconds = 10
        config.data.update_seconds = 10
        config.model.width = 8
        config.model.token_dim = 16
        config.model.tcn_layers = 2
        config.train.stage = "joint"
        return config

    def test_model_shapes_and_backward(self):
        config = self.tiny_config()
        model = RawContinuousHRVModel(config)
        shape = (3, 2, 2, 200)
        output = model(
            torch.randn(shape),
            torch.ones(shape, dtype=torch.bool),
            torch.zeros(shape),
            torch.arange(3),
            torch.rand(3, 1),
        )
        self.assertEqual(output["direct_loc"].shape, (3, 2))
        self.assertEqual(output["beat_heatmap_logits"].shape, (3, 2, 200))
        output["direct_loc"].sum().backward()

    def test_partner_segnet_rawslot_baseline(self):
        config = self.tiny_config()
        config.model.direct_architecture = "segnet_mean"
        config.model.use_timestamp_jitter = False
        config.model.token_dim = 64
        config.train.stage = "direct"
        model = RawContinuousHRVModel(config)
        shape = (2, 2, 2, 200)
        output = model(
            torch.randn(shape),
            torch.ones(shape, dtype=torch.bool),
            torch.zeros(shape),
            torch.tensor([0, 1]),
            None,
            compute_beat=False,
        )
        self.assertEqual(output["direct_loc"].shape, (2, 2))
        self.assertNotIn("beat_heatmap_logits", output)

    def test_evaluator_and_causal_monitor_contract(self):
        config = self.tiny_config()
        config.train.stage = "direct"
        model = RawContinuousHRVModel(config)
        scaler = TargetScaler.fit(np.array([[20, 40], [30, 50], [40, 60]], dtype=float))
        shape = (1, 2, 2, 200)
        raw_batch = {
            "values": torch.randn(shape),
            "mask": torch.ones(shape, dtype=torch.bool),
            "jitter": torch.zeros(shape),
            "device_id": torch.tensor([0]),
            "accel": torch.tensor([[0.2]]),
            "motion_scalar": torch.tensor([[0.2]]),
            "target_ms": torch.tensor([[30.0, 50.0]]),
            "rpeaks_ms": [torch.tensor([1000.0, 2000.0, 3000.0, 4000.0])],
            "participant": ["P1"],
            "window_index": [0],
            "window_start_ms": [0.0],
            "device": ["Earring"],
            "group_id": torch.tensor([0]),
            "coverage": torch.tensor([0.9], dtype=torch.float32),
            "label_source": ["ecg_corrected_5min_fields"],
        }
        rows = evaluate_model(model, [raw_batch], scaler, config, torch.device("cpu"), "fold_0", False)
        self.assertEqual(rows[0]["prediction_source"], "direct")
        self.assertIn("reject_reason", rows[0])
        self.assertTrue(rows[0]["accepted"])
        beat_rows = evaluate_model(model, [raw_batch], scaler, config, torch.device("cpu"), "fold_0", True)
        self.assertIn("beat_event_f1", beat_rows[0])
        self.assertIn("beat_timing_mae_ms", beat_rows[0])

        monitor = CausalHRVMonitor(model, scaler, config, "Earring")
        update_shape = (1, 2, 200)
        first = monitor.update(
            np.random.randn(*update_shape).astype(np.float32),
            np.ones(update_shape, dtype=bool),
            np.zeros(update_shape, dtype=np.float32),
            0.2,
        )
        second = monitor.update(
            np.random.randn(*update_shape).astype(np.float32),
            np.ones(update_shape, dtype=bool),
            np.zeros(update_shape, dtype=np.float32),
            0.2,
        )
        self.assertFalse(first.ready)
        self.assertTrue(second.ready)
        self.assertEqual(second.source, "direct")

    def test_delay_search_recovers_later_ppg_pulse(self):
        fs = 100.0
        rpeaks = [torch.tensor([500.0, 1500.0, 2500.0])]
        target, _ = build_ecg_beat_targets(rpeaks, 350, fs)
        logits = torch.full((1, 1, 350), -8.0)
        shift = 30  # 300 ms pulse transit delay
        logits[0, 0, shift:] = torch.maximum(logits[0, 0, shift:], target[0, :-shift] * 16 - 8)
        loss = DelayAwareBeatLoss(fs, 100, 700, 10)
        _, _, selected = loss(logits, torch.zeros_like(logits), torch.tensor([300.0]), rpeaks)
        self.assertAlmostEqual(float(selected[0]), 300.0, delta=20.0)

    def test_one_to_one_beat_event_metrics(self):
        reference = np.array([500.0, 1500.0, 2500.0, 3500.0])
        predicted = np.array([805.0, 1790.0, 2815.0, 3800.0, 4400.0])
        result = match_beat_events(predicted, reference, pulse_delay_ms=300.0, tolerance_ms=50.0)
        self.assertEqual((result.tp, result.fp, result.fn), (4, 1, 0))
        self.assertAlmostEqual(result.precision, 0.8)
        self.assertAlmostEqual(result.recall, 1.0)
        self.assertAlmostEqual(result.timing_mae_ms, 7.5)
        self.assertEqual(result.count_error, 1)

    def test_vectorized_delay_candidates_match_individual_shifts(self):
        values = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        shifts = torch.tensor([0, 2, 7, 20])
        actual = shift_candidates_with_zeros(values, shifts)
        expected = torch.stack(
            [shift_with_zeros(values, shift.repeat(len(values))) for shift in shifts], dim=1
        )
        torch.testing.assert_close(actual, expected)

    def test_beat_pretrain_does_not_use_window_hrv_scalar(self):
        config = self.tiny_config()
        config.train.stage = "beat_pretrain"
        model = RawContinuousHRVModel(config)
        shape = (1, 2, 2, 200)
        outputs = model(
            torch.randn(shape),
            torch.ones(shape, dtype=torch.bool),
            torch.zeros(shape),
            torch.tensor([0]),
            torch.tensor([[0.1]]),
        )
        batch = {
            "target_ms": torch.tensor([[float("nan"), float("nan")]]),
            "group_id": torch.tensor([0]),
            "rpeaks_ms": [torch.tensor([1000.0, 2000.0, 3000.0])],
        }
        scaler = TargetScaler.fit(np.array([[20.0, 40.0], [30.0, 50.0]]))
        losses = PipelineLoss(config, scaler)(outputs, batch)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(float(losses["direct"].detach()), 0.0)

    def test_cached_robust_stats_match_legacy_window_normalization(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=(3, 3, 2, 40)).astype(np.float32)
        mask = rng.random(values.shape) > 0.15
        values[~mask] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "values.npy", values)
            np.save(root / "mask.npy", mask)
            np.savez(root / "metadata.npz", target_fs=2.0, window_sec=20)
            path = write_robust_stats_from_cache(root, 20)
            stats = np.load(path)
            for window in range(3):
                for device in range(3):
                    legacy = RawslotWindowDataset._robust_normalize(
                        values[window, device], mask[window, device]
                    )
                    cached = RawslotWindowDataset._robust_normalize_from_stats(
                        values[window, device], mask[window, device], stats[window, device]
                    )
                    np.testing.assert_allclose(cached, legacy, rtol=0.0, atol=1e-6)

    def test_rr_invalid_interval_is_an_adjacency_separator(self):
        beats = np.array([0, 700, 950, 1950, 3050, 3950], dtype=float)
        corrected = correct_rr_adjacency(beats)
        self.assertTrue(np.isnan(corrected.rr_ms).any())
        # The routine may return NaN if too few truly adjacent differences remain;
        # it must never bridge the 250 ms rejected interval.
        rmssd, _ = hrv_from_corrected_rr(corrected)
        if np.isfinite(rmssd):
            valid_pairs = np.isfinite(corrected.rr_ms[:-1]) & np.isfinite(corrected.rr_ms[1:])
            expected = np.sqrt(np.mean(np.diff(corrected.rr_ms)[valid_pairs] ** 2))
            self.assertAlmostEqual(rmssd, expected)

    def test_group_folds_are_disjoint(self):
        folds = make_group_folds([f"P{x}" for x in range(1, 17)])
        for fold in folds:
            self.assertFalse(set(fold.train) & set(fold.val))
            self.assertFalse(set(fold.train) & set(fold.test))
            self.assertFalse(set(fold.val) & set(fold.test))

    def test_checkpoint_contract(self):
        config = self.tiny_config()
        model = RawContinuousHRVModel(config)
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = TargetScaler.fit(np.array([[20, 40], [30, 50]], dtype=float))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                epoch=2,
                best_validation_loss=1.0,
                config=config.to_dict(),
                fold={"name": "f", "train": [], "val": [], "test": []},
                target_scaler=scaler.state_dict(),
                history=[],
            )
            payload = load_checkpoint(path)
            self.assertEqual(payload["target_order"], ["rmssd_ms", "sdnn_ms"])
            self.assertEqual(payload["primary_input"], "ppg_rawslot_values")

    def test_window_prediction_artifacts(self):
        rows = []
        for participant_index, participant in enumerate(("P1", "P2")):
            for device_index, device in enumerate(("Earring", "Ring", "Watch")):
                target_rmssd = 30.0 + participant_index
                target_sdnn = 50.0 + participant_index
                rows.append(
                    {
                        "participant": participant,
                        "window_index": participant_index,
                        "device": device,
                        "coverage": 0.94 + device_index * 0.02,
                        "accel_motion_mean_mag": 0.1 + device_index,
                        "target_rmssd_ms": target_rmssd,
                        "target_sdnn_ms": target_sdnn,
                        "direct_rmssd_ms": target_rmssd + device_index,
                        "direct_sdnn_ms": target_sdnn + device_index,
                        "direct_rmssd_std_ms": 4.0 + device_index,
                        "direct_sdnn_std_ms": 5.0 + device_index,
                        "beat_rmssd_ms": float("nan"),
                        "beat_sdnn_ms": float("nan"),
                        "prediction_rmssd_ms": target_rmssd + device_index,
                        "prediction_sdnn_ms": target_sdnn + device_index,
                        "confidence": 0.9 - device_index * 0.1,
                        "accepted": True,
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_run_artifacts(directory, rows, {"kind": "test"})
            self.assertTrue(paths["predictions"].is_file())
            self.assertTrue(paths["metrics"].is_file())
            self.assertTrue(paths["summary"].is_file())

    def test_real_dataset_schema_when_present(self):
        root = Path("src/heuristic_baselines/outputs/synced_3device_rawaligned_training_v1_stride30_rawslots")
        files = sorted(root.glob("*_P5.npz"))
        if not files:
            self.skipTest("local HRV data not present")
        info = validate_npz_schema(files[0])
        self.assertEqual(info["primary_input"], "ppg_rawslot_values")
        self.assertEqual(info["n_samples"], 30000)


if __name__ == "__main__":
    unittest.main()
