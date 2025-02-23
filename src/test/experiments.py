import os
import sys

from abc import ABC, abstractmethod
from dataclasses import dataclass
from omegaconf import OmegaConf
from pathlib import Path
from typing import Optional, Dict

# Add paths to sys.path if not already present
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# imports from other modules
from configs.experiment_config import (
    LoggingConfig,
    MetricConfig,
    NetworkConfig,
    ModelConfig,
    TestConfig,
    EvalConfig,
    TrainingConfig,
    GenerationExperimentConfig,
    TestExperimentConfig,
)
from src.evaluation.generate import ModelGenerator
from src.test.test import (
    AuditingTest,
    CalibratedAuditingTest,
)
from src.test.calibration_strategies import (
    DefaultStrategy,
    StdStrategy,
    IntervalStrategy,
)
from src.analysis.nn_distance import (
    CMLP,
    OptCMLP,
    HighCapacityCMLP,
)

SCRIPT_DIR = SCRIPT_DIR = Path(__file__).resolve().parent


class Experiment(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def run(self):
        pass


class GenerationExperiment(Experiment):
    def __init__(self, cfg: GenerationExperimentConfig):
        super().__init__(cfg)

    def run(self):
        """
        Runs the generation experiment.
        """
        generator = ModelGenerator(self.cfg)
        generator.generate()


class TestExperiment(Experiment):
    def __init__(self, cfg: TestExperimentConfig):
        super().__init__(cfg)

    def run(self):
        """Runs the test experiment"""

        if self.cfg.test_params.calibrate:
            eps = self._get_calibration_strategy()

            exp = CalibratedAuditingTest(self.cfg, eps)
        else:
            exp = AuditingTest(self.cfg)

        exp.run()
        #     model_name1=self.model1_config.model_id,
        #     seed1=self.model1_config.gen_seed,
        #     model_name2=self.model2_config.model_id,
        #     seed2=self.model2_config.gen_seed,
        #     fold_size=self.test_config.fold_size,
        #     analyze_distance=self.test_config.analyze_distance,
        #     calibrate_only=self.test_config.calibrate_only,
        # )

    def _get_calibration_strategy(self):
        calibration_cfg = self.cfg.calibration_params
        epsilon_strategy = (
            self.cfg.calibration_params.epsilon_strategy
        )
        overwrite = self.cfg.test_params.overwrite

        strategy_map = {
            "default": DefaultStrategy,
            "std": StdStrategy,
            "interval": IntervalStrategy,
        }
        strategy_class = strategy_map.get(epsilon_strategy)
        if not strategy_class:
            raise ValueError(
                f"Unknown calibration strategy: {epsilon_strategy}"
            )
        return strategy_class(
            calibration_cfg, overwrite=overwrite
        )
