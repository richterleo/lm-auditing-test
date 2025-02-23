from pathlib import Path
import sys

# Add submodule to path once, in a central location
project_root = Path(__file__).resolve().parents[2]
submodule_path = project_root / "deep-anytime-testing"
if str(submodule_path) not in sys.path:
    sys.path.append(str(submodule_path))

# Import and re-export the needed components
from trainer.trainer import Trainer
from models.earlystopping import EarlyStopper
from models.mlp import MLP

__all__ = ["Trainer", "EarlyStopper", "MLP"]
