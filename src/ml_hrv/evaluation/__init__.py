"""Window-level prediction export and leakage-safe aggregate reporting."""

from .evaluator import evaluate_model
from .metrics import summarize_predictions
from .reporting import write_run_artifacts

__all__ = ["evaluate_model", "summarize_predictions", "write_run_artifacts"]
