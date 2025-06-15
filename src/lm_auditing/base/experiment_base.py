from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, List, Any
from omegaconf import OmegaConf
import logging
import wandb
import sys
from configs.experiment_config import ExperimentConfig
from lm_auditing.utils.utils import create_run_string
from lm_auditing.utils.logging_config import setup_logging


class ExperimentBase(ABC):
    SCRIPT_DIR = Path(__file__).resolve().parents[3]

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._calculate_dependent_attributes()
        self.initialize_wandb()

    @abstractmethod
    def _calculate_dependent_attributes(self):
        """Calculate and set dependent attributes from config"""
        pass

    def initialize_wandb(self, tags: List[str] = None):
        """Initialize wandb if enabled"""
        if self.logging_cfg.use_wandb:
            config_dict = self.config.to_dict()
            if not tags:
                tags = [f"{type(self).__name__}"]
            wandb.init(
                project=self.logging_cfg.wandb_project_name,
                entity=self.logging_cfg.entity,
                name=create_run_string(),
                config=config_dict,
                tags=tags,
            )

    def setup_logger(self, *args: Any, tag: Optional[str] = None):
        """ """

        if tag is None:
            tag = f"{type(self).__name__}"

        log_file_path = setup_logging(*args, tag=tag, quiet=self.logging_cfg.quiet)
        self.logger = logging.getLogger(__name__)
        
        # If quiet logging is enabled, print to terminal where logs are being saved
        if self.logging_cfg.quiet and log_file_path:
            print(f"Logging to file: {log_file_path}")
        # Only log to file if not in quiet mode
        elif log_file_path:
            self.logger.info(f"Logging to file: {log_file_path}.")

    def update_wandb(self):
        """Updates wandb config with current configuration values"""
        if not self.logging_cfg.use_wandb:
            return

        updated_config = OmegaConf.to_container(self.config, resolve=True)
        wandb.config.update(updated_config, allow_val_change=True)

    def _apply_overrides(self, overrides: Dict):
        """Apply configuration overrides"""

        # for ModelGenerator
        if "model_name" in overrides:
            self.config.model1.model_id = overrides["model_name"]
        if "hf_prefix" in overrides:
            self.config.model1.hf_prefix = overrides["hf_prefix"]
        if "seed" in overrides:
            self.config.model1.gen_seed = overrides["seed"]
        if "num_samples" in overrides:
            self.config.eval.num_samples = overrides["num_samples"]
        if "batch_size" in overrides:
            self.config.eval.batch_size = overrides["batch_size"]

        # for Test
        if "model_name1" in overrides:
            self.config.model1.model_id = overrides["model_name1"]
        if "seed1" in overrides:
            self.config.model1.gen_seed = overrides["seed1"]
        if "model_name2" in overrides:
            self.config.model2.model_id = overrides["model_name2"]
        if "seed2" in overrides:
            self.config.model2.gen_seed = overrides["seed2"]
        if "analyze_distance" in overrides:
            self.config.test_params.analyze_distance = overrides["analyze_distance"]
        if "run_davvt" in overrides:
            self.config.test_params.run_davtt = overrides["run_davtt"]
        if "fold_size" in overrides:
            self.config.test_params.fold_size = overrides["fold_size"]

        self._calculate_dependent_attributes()
        self.update_wandb()
