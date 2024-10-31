import os
import sys

from abc import ABC, abstractmethod
from omegaconf import OmegaConf
from pathlib import Path
from typing import Optional, Dict

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep-anytime-testing"))

# imports from other modules
from auditing_test.eval import eval_model
from auditing_test.test import AuditingTest, CalibratedAuditingTest
from auditing_test.epsilon_strategies import (
    DefaultEpsilonStrategy,
    StandardDeviationEpsilonStrategy,
    IntervalEpsilonStrategy,
)

SCRIPT_DIR = SCRIPT_DIR = Path(__file__).resolve().parent


class Experiment(ABC):
    def __init__(self, cfg, train_cfg: Optional[Dict] = None):
        self.cfg = cfg
        self.train_cfg = train_cfg

    @abstractmethod
    def run(self):
        pass


class GenerationExperiment(Experiment):
    def run(
        self,
        model_id: Optional[str] = None,
        hf_prefix: Optional[str] = None,
        num_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        use_wandb: Optional[bool] = None,
    ):
        """
        Runs the generation experiment, allowing optional overrides for eval_model.

        Args:
            model_id: Optional; overrides the model ID from config if provided.
            hf_prefix: Optional; overrides the HF prefix from config if provided.
            num_samples: Optional; overrides the number of samples from config if provided.
            batch_size: Optional; overrides the batch size from config if provided.
            use_wandb: Optional; overrides the use_wandb flag from config if provided.
            eval_on_task: Optional; evaluates on the task if True.
            **kwargs: Additional keyword arguments.
        """
        # Call eval_model with the config and any overrides

        eval_model(
            OmegaConf.to_container(self.cfg, resolve=True),
            model_id=model_id,
            hf_prefix=hf_prefix,
            num_samples=num_samples,
            batch_size=batch_size,
            use_wandb=use_wandb,
        )


class TestExperiment(Experiment):
    def run(
        self,
        model_name1: Optional[str] = None,
        seed1: Optional[str] = None,
        model_name2: Optional[str] = None,
        seed2: Optional[str] = None,
        fold_size: Optional[int] = None,
        analyze_distance: bool = True,
        run_davtt: bool = True,
        calibrate: Optional[bool] = None,
        use_wandb: Optional[bool] = None,
    ):
        """ """
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)

        calibrate = calibrate if calibrate is not None else self.cfg.test_params.get("calibrate", False)
        use_wandb = use_wandb if use_wandb is not None else self.cfg.logging.use_wandb
        overwrite = self.cfg.test_params.get("overwrite", False)
        dir_prefix = self.cfg.dir_prefix

        if calibrate:
            calibration_strategy = self.cfg.calibration_params.get("calibration_strategy", "default")
            calibration_cfg = cfg_dict["calibration_params"]
            overwrite = self.cfg.test_params.get("overwrite", False)
            dir_prefix = self.cfg.dir_prefix

            if calibration_strategy == "default" or calibration_strategy.lower() == "defaultepsilonstrategy":
                eps = DefaultEpsilonStrategy(calibration_cfg, overwrite=overwrite, dir_prefix=dir_prefix)
            elif calibration_strategy == "std" or calibration_strategy.lower() == "standarddeviationepsilonstrategy":
                eps = StandardDeviationEpsilonStrategy(
                    epsilon_ticks=self.cfg.calibration_params["epsilon_ticks"],
                    bias=self.cfg.calibration_params["bias"],
                    num_runs=self.cfg.calibration_params["num_runs"],
                )
            elif calibration_strategy == "interval" or calibration_strategy.lower() == "intervalEpsilonStrategy":
                eps = IntervalEpsilonStrategy(
                    self.cfg.calibration_params["lower_model"],
                    self.cfg.calibration_params["lower_model_seed"],
                    self.cfg.calibration_params["upper_model"],
                    self.cfg.calibration_params["upper_model_seed"],
                    overwrite=self.cfg.calibration_params["overwrite"],
                    epsilon_ticks=self.cfg.calibration_params["epsilon_ticks"],
                    num_runs=self.cfg.calibration_params["num_runs"],
                )

            exp = CalibratedAuditingTest(
                cfg_dict,
                self.train_cfg,
                self.cfg.dir_prefix,
                eps,
                use_wandb=use_wandb,
                only_continuations=self.cfg.test_params.only_continuations,
            )
        else:
            exp = AuditingTest(
                cfg_dict,
                self.train_cfg,
                self.cfg.dir_prefix,
                use_wandb=use_wandb,
                only_continuations=self.cfg.test_params.only_continuations,
            )
        exp.run(
            model_name1=model_name1,
            seed1=seed1,
            model_name2=model_name2,
            seed2=seed2,
            fold_size=fold_size,
            analyze_distance=analyze_distance,
            run_davtt=run_davtt,
        )
