import hydra
import logging
import os
import torch

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Optional, Dict, List
from configs.experiment_config import (
    LoggingConfig,
    MetricConfig,
    NetworkConfig,
    ModelConfig,
    EvalConfig,
    TrainingConfig,
    TestParams,
    StoringConfig,
    GenerationConfig,
    TestConfig,
)
from src.evaluation.generate import ModelGenerator
from src.auditing.test import (
    AuditingTest,
    CalibratedAuditingTest,
)
from src.auditing.calibration_strategies import (
    DefaultStrategy,
    StdStrategy,
    IntervalStrategy,
)

# from logging_config import setup_logging

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

logger = logging.getLogger(__name__)


def run_generation(cfg: GenerationConfig):
    generator = ModelGenerator(cfg)
    generator.generate()


def run_test(cfg: TestConfig):
    if cfg.test_params.calibrate:
        eps = get_calibration_strategy(cfg)
        test = CalibratedAuditingTest(cfg, eps)
    else:
        test = AuditingTest(cfg)
    test.run()


def get_calibration_strategy(cfg: TestConfig):
    strategy_map = {
        "default": DefaultStrategy,
        "std": StdStrategy,
        "interval": IntervalStrategy,
    }
    strategy_class = strategy_map.get(cfg.calibration_params.epsilon_strategy)
    if not strategy_class:
        raise ValueError(
            f"Unknown calibration strategy: {cfg.calibration_params.epsilon_strategy}"
        )
    return strategy_class(
        cfg.calibration_params,
        overwrite=cfg.test_params.overwrite,
    )


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        logger.info(f"GPU is available: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        logger.warning("No GPU available, running on CPU")

    # If debug mode is enabled
    if cfg.get("debug_mode", False):
        import debugpy

        debugpy.listen(("0.0.0.0", 5678))
        logger.info("waiting for debugger attach...")
        debugpy.wait_for_client()
        logger.info("Debugger attached")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Determine which experiment to run based on cfg.exp
    if cfg.exp == "generation":
        model_cfg = ModelConfig.from_dict(cfg_dict["tau1"])
        generation_cfg = GenerationConfig(
            logging=LoggingConfig(**cfg_dict["logging"]),
            metric=MetricConfig(**cfg_dict["metric"]),
            storing=StoringConfig(**cfg_dict["storing"]),
            model=model_cfg,
            eval=EvalConfig(**cfg_dict["eval"]),
            exp="generation",
        )
        run_generation(generation_cfg)

    elif cfg.exp == "test":
        # Instantiate and run the appropriate TestExperiment
        model1_cfg = ModelConfig.from_dict(cfg_dict["tau1"])
        model2_cfg = ModelConfig.from_dict(cfg_dict["tau2"])
        test_params = TestParams.from_dict(cfg_dict["test_params"])
        train_cfg = TrainingConfig.from_dict(cfg_dict["training"])

        test_cfg = TestConfig(
            logging=LoggingConfig(**cfg_dict["logging"]),
            metric=MetricConfig(**cfg_dict["metric"]),
            storing=StoringConfig(**cfg_dict["storing"]),
            test_params=test_params,
            net=NetworkConfig(**cfg_dict["net"]),
            model1=model1_cfg,
            model2=model2_cfg,
            training=train_cfg,
            exp="test",
        )
        run_test(test_cfg)


if __name__ == "__main__":
    main()
