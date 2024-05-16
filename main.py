import argparse
import importlib
import os
import re
import torch
import sys
import wandb

from typing import Optional, Dict, List

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

from dah_testing.eval_trainer import OnlineTrainer, OfflineTrainer
from dah_testing.preprocessing import create_folds_from_generations

# Dynamically import the module
# deep_anytime_testing = importlib.import_module("deep-anytime-testing")


def test_daht(
    config,
    train_cfg,
    tau2_cfg: Optional[Dict] = None,
    train_online: bool = False,
    fold_num: int = 0,
    run_id1: Optional[str] = None,
    run_id2: Optional[str] = None,
):
    """ """
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

    if train_online:
        if tau2_cfg:
            trainer = OnlineTrainer(
                train_cfg,
                net,
                config["tau1"],
                config["metric"]["dataset_name"],
                config["metric"]["behavior"],
                config["metric"]["metric"],
                config["logging"]["use_wandb"],
                tau2_cfg,
            )
        else:
            trainer = OnlineTrainer(
                train_cfg,
                net,
                config["tau1"],
                config["metric"]["dataset_name"],
                config["metric"]["behavior"],
                config["metric"]["metric"],
                config["logging"]["use_wandb"],
                config["tau2"],
            )

    else:
        trainer = OfflineTrainer(
            train_cfg,
            net,
            run_id1,
            run_id2,
            metric=config["metric"]["metric"],
            use_wandb=config["logging"]["use_wandb"],
            fold_num=fold_num,
        )

    trainer.train()

    wandb.finish()


def run_test_with_wandb(
    config,
    train_cfg,
    tau2_cfg: Optional[Dict] = None,
    train_online: bool = False,
    fold_num: int = 0,
    run_id1: Optional[str] = None,
    run_id2: Optional[str] = None,
):
    """Wrapper function to handle wandb logging and call the core logic."""

    if config["logging"]["use_wandb"]:
        wandb.init(
            project=f"{config['metric']['behavior']}_test",
            entity=config["logging"]["entity"],
            name=create_run_string(),
            config=config,
        )
        wandb.config.update({"train_online": train_online})
        if train_online:
            run_id1 = run_id1 if run_id1 else config["run_id1"]
            run_id2 = run_id2 if run_id2 else config["run_id2"]
            wandb.config.update({"run_id1": run_id1, "run_id2": run_id2})

    test_daht(
        config,
        train_cfg,
        tau2_cfg=tau2_cfg,
        train_online=train_online,
        fold_num=fold_num,
        run_id1=run_id1,
        run_id2=run_id2,
    )

    if config["logging"]["use_wandb"]:
        wandb.finish()


def eval_model(
    config,
    num_samples=None,
    num_epochs=None,
    batch_size=None,
    evaluate=True,
):
    """ """
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
        wandb.config.update({"evaluate": evaluate})

    generate_and_evaluate(
        config["metric"]["dataset_name"],
        config["tau1"],
        config["eval"]["num_samples"] if not num_samples else num_samples,
        num_epochs=config["eval"]["epochs"] if not num_epochs else num_epochs,
        batch_size=config["eval"]["batch_size"] if not batch_size else batch_size,
        use_wandb=config["logging"]["use_wandb"],
        evaluate=evaluate,
        seed=config["tau1"]["gen_seed"],
        metric=config["metric"]["metric"],
    )

    if config["logging"]["use_wandb"]:
        wandb.finish()


def kfold_train(
    config,
    train_cfg,
    run_id1: Optional[str] = None,
    run_id2: Optional[str] = None,
    overwrite: Optional[bool] = True,
):
    """Do repeats on"""
    # Initialize wandb if logging is enabled
    if config["logging"]["use_wandb"]:
        wandb.init(
            project=f"{config['metric']['behavior']}_test",
            entity=config["logging"]["entity"],
            name=create_run_string(),
            config=config,
            tags=["kfold"],
        )

    run_id1 = run_id1 if run_id1 else config["run_id1"]
    run_id2 = run_id2 if run_id2 else config["run_id2"]
    if config["logging"]["use_wandb"]:
        wandb.config.update({"run_id1": run_id1, "run_id2": run_id2})

    # Check all folds available for the two runs
    directory = f"{run_id1}_{run_id2}"
    pattern = re.compile(rf"{config['metric']['metric']}_scores\.json_(\d+)\.json")
    folds = []
    if len(folds) == 0:
        create_folds_from_generations(
            run_id1, run_id2, config["metric"]["metric"], overwrite=overwrite
        )

    # Check the directory for matching files and append to the list
    for file_name in os.listdir(directory):
        match = pattern.match(file_name)
        if match:
            fold_number = int(match.group(1))
            folds.append(fold_number)

    if config["logging"]["use_wandb"]:
        wandb.config.update({"num_folds": folds})

    # Iterate over the folds and call test_daht
    for fold_num in folds:
        test_daht(
            config,
            train_cfg,
            train_online=False,
            fold_num=fold_num,
            run_id1=run_id1,
            run_id2=run_id2,
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

    parser.add_argument(
        "--online",
        action="store_true",
        help="Whether to use the OnlineTrainer instead of the OfflineTrainer",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yml",
        help="Path to config file",
    )

    parser.add_argument(
        "--fold_num",
        type=int,
        default=0,
        help="Fold number for repeated test runs",
    )

    parser.add_argument(
        "--run_id1",
        type=str,
        default=None,
        help="Run ID for the first model",
    )

    parser.add_argument(
        "--run_id2",
        type=str,
        default=None,
        help="Run ID for the second model",
    )

    parser.add_argument("--kfold", action="store_true", help="Run kfold training")

    args = parser.parse_args()
    config = load_config(args.config_path)

    # Determine which experiment to run based on the argument
    if args.exp == "generation":
        eval_model(config, evaluate=args.evaluate)

    elif args.exp == "test_daht":
        train_cfg = TrainCfg()
        if args.kfold:
            kfold_train(
                config,
                train_cfg,
                run_id1=args.run_id1,
                run_id2=args.run_id2,
            )
        else:
            run_test_with_wandb(
                config,
                train_cfg,
                train_online=args.online,
                fold_num=args.fold_num,
                run_id1=args.run_id1,
                run_id2=args.run_id2,
            )


if __name__ == "__main__":
    main()
