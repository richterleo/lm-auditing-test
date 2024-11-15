import hydra
import logging
import os
import sys

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Optional, Dict, List

# Add paths to sys.path if not already present
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# imports from other scripts
from arguments import TrainCfg

# from logging_config import setup_logging
from src.utils.utils import load_config

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

SCRIPT_DIR = SCRIPT_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # If debug mode is enabled
    if cfg.get("debug_mode", False):
        import debugpy

        debugpy.listen(("0.0.0.0", 5678))
        logger.info("waiting for debugger attach...")
        debugpy.wait_for_client()
        logger.info("Debugger attached")

    # Convert configuration to a dictionary if needed
    # cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Determine which experiment to run based on cfg.exp
    if cfg.exp == "generation":
        # Instantiate and run the GenerationExperiment
        from src.test.experiments import GenerationExperiment

        experiment = GenerationExperiment(cfg)
        experiment.run()

    elif cfg.exp == "test":
        # Instantiate and run the appropriate TestExperiment
        from src.test.experiments import TestExperiment

        train_cfg = TrainCfg()
        experiment = TestExperiment(cfg, train_cfg)
        experiment.run()


if __name__ == "__main__":
    main()
