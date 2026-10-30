"""Run a synthetic Kazemi forward/backward check, without fitting a model."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    import torch

    config = json.loads((REPO / "experiments/hrv_v4_3/model_comparison.json").read_text(encoding="utf-8"))
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and the CUDA-enabled PyTorch wheel")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    torch.set_num_threads(1)
    source = REPO / "src/model_baselines/supervised/hrv_ext_group4_kazemi_decoder_score.py"
    spec = importlib.util.spec_from_file_location("hrv_workstation_smoke_module", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.KazemiPeakNet(cin=2).to(device)
    batch = config["cli_args"]["batch_size"]
    length = int(300 * config["cli_args"]["resample_hz"])
    x = torch.randn(batch, 2, length, device=device)
    target = torch.zeros(batch, length, device=device)
    target[:, 25::50] = 1
    logits, _ = model(x)
    if tuple(logits.shape) != (batch, 1, length):
        raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")
    loss = module.PeakLoss(fs=50)(logits.squeeze(1), target)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite synthetic loss")
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("Non-finite gradient")
    if device.type == "cuda":
        torch.cuda.synchronize()
    versions = {}
    for name in ("torch", "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "einops"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {"status": "synthetic_forward_backward_passed", "device": str(device),
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "cuda_runtime": torch.version.cuda, "python": platform.python_version(),
              "platform": platform.platform(), "packages": versions,
              "batch_size": batch, "channels": 2, "samples": length,
              "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "synthetic_loss": float(loss.detach().cpu()),
              "optimizer_step_performed": False, "real_data_training_validated": False}
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
