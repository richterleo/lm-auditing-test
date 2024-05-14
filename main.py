import argparse
import importlib
import os
import torch
import sys
import wandb

from typing import Optional, Dict

# imports from other scripts
from arguments import TrainCfg
from utils.utils import (
    load_config,
    create_run_string,
    initialize_from_config,
    time_block,
)
from utils.generate_and_evaluate import generate_and_evaluate

# Add the submodule to the path for eval_trainer
submodule_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "deep-anytime-testing")
)
if submodule_path not in sys.path:
    sys.path.append(submodule_path)

# Assuming the 'models' module is inside the 'deep-anytime-testing' directory
models_path = os.path.join(
    submodule_path, "models"
)  # Adjust this path if the location is different
if models_path not in sys.path:
    sys.path.append(models_path)

from dah_testing.eval_trainer import EvalTrainer

# Dynamically import the module
deep_anytime_testing = importlib.import_module("deep-anytime-testing")


def test_daht(train_cfg, config_path="config.yml", tau2_cfg: Optional[Dict] = None):
    """ """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(config_path)

    if config["logging"]["use_wandb"]:
        wandb.init(
            project=f"{config['metric']['behavior']}_test",
            entity=config["logging"]["entity"],
            name=create_run_string(),
            config=config,
        )

    models = importlib.import_module(
        "deep-anytime-testing.models.mlp", package="deep-anytime-testing"
    )
    MMDEMLP = getattr(models, "MMDEMLP")
    net = initialize_from_config(config["net"], MMDEMLP)

    with time_block("Instantiating EvalTrainer"):
        if tau2_cfg:
            trainer = EvalTrainer(
                train_cfg,
                net,
                config["tau1"],
                config["metric"]["dataset_name"],
                device,
                config["metric"]["behavior"],
                config["metric"]["metric"],
                config["logging"]["use_wandb"],
                tau2_cfg,
            )
        else:
            trainer = EvalTrainer(
                train_cfg,
                net,
                config["tau1"],
                config["metric"]["dataset_name"],
                device,
                config["metric"]["behavior"],
                config["metric"]["metric"],
                config["logging"]["use_wandb"],
                config["tau2"],
            )
        print("Now starting training")

    trainer.train()

    wandb.finish()


def eval_model(
    config_path="config.yml",
    num_samples=None,
    num_epochs=None,
    batch_size=None,
    evaluate=True,
):
    """ """
    config = load_config(config_path)

    project_name = (
        f"{config['metric']['behavior']}_evaluation" if evaluate else "continuations"
    )

    if config["logging"]["use_wandb"]:
        wandb.init(
            project=project_name,
            entity=config["logging"]["entity"],
            name=create_run_string(),
            config=config,
        )

    generate_and_evaluate(
        config["metric"]["dataset_name"],
        config["tau1"],
        config["eval"]["num_samples"] if not num_samples else num_samples,
        num_epochs=config["eval"]["epochs"] if not num_epochs else num_epochs,
        batch_size=config["eval"]["batch_size"] if not batch_size else batch_size,
        use_wandb=config["logging"]["use_wandb"],
        evaluate=evaluate,
        metric=config["metric"]["metric"],
    )

    if config["logging"]["use_wandb"]:
        wandb.finish()


def main():
    """ """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--exp",
        type=str,
        choices=["generation", "test_daht"],
        required=True,
        help="Select the experiment to run: evalution or testing the dat-test",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether to evaluate the model on the metric",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Determine which experiment to run based on the argument
    if args.exp == "generation":
        eval_model(evaluate=args.evaluate)
    elif args.exp == "test_daht":
        train_cfg = TrainCfg()
        test_daht(train_cfg)


if __name__ == "__main__":
    main()
