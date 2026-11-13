"""Load the unchanged colleague implementation; no copied or patched model."""
from pathlib import Path
import importlib.util

def load():
    source=Path(__file__).resolve().parents[2]/"model_baselines/supervised/hrv_ext_group4_kazemi_decoder_score.py"
    spec=importlib.util.spec_from_file_location("hrv_colleague_model",source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
