import argparse
import os
import re
import time
import sys
import wandb
import pandas as pd

from pathlib import Path
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

# Add the submodule and models to the path for eval_trainer
submodule_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "deep-anytime-testing"))
models_path = os.path.join(submodule_path, "models")

for path in [submodule_path, models_path]:
    if path not in sys.path:
        sys.path.append(path)

from dah_testing.eval_trainer import OnlineTrainer, OfflineTrainer
from dah_testing.preprocessing import create_folds_from_evaluations, cleanup_files

def davtt(
    config,
    train_cfg,
    fold_num: int = 0,
    model_name1: Optional[str] = None,
    seed1: Optional[str] = None,
    model_name2: Optional[str] = None,
    seed2: Optional[str] = None,
    use_wandb: Optional[str] = None,
):
    """ 
    Deep anytime-valid tolerance test 
    
    Args:
    config: Dict
        Configuration dictionary
    train_cfg: TrainCfg
        Training configuration
    """
    # Whether to use logging
    use_wandb = (
        use_wandb if use_wandb is not None else config["logging"]["use_wandb"]
    )
    
    model_name1 = model_name1 if model_name1 else config["tau1"]["model_id"]
    seed1 = seed1 if seed1 else config["tau1"]["gen_seed"]
    model_name2 = model_name2 if model_name2 else config["tau2"]["model_id"]
    seed2 = seed2 if seed2 else config["tau2"]["gen_seed"]
    
    # Define network for betting score
    betting_net = initialize_from_config(config["net"])
    
    trainer = OfflineTrainer(
        train_cfg,
        betting_net,
        model_name1,
        seed1,
        model_name2,
        seed2,
        metric=config["metric"]["metric"],
        use_wandb=use_wandb,
        fold_num=fold_num,
        epsilon=config["epsilon"]
    )

    data = trainer.train()
    return data


def run_test_with_wandb(
    config,
    train_cfg,
    tau2_cfg: Optional[Dict] = None,
    fold_num: int = 0,
    model_name1: Optional[str] = None,
    seed1: Optional[str] = None,
    model_name2: Optional[str] = None,
    seed2: Optional[str] = None,
):
    """Wrapper function to handle wandb logging and call the core logic."""

    if config["logging"]["use_wandb"]:
        wandb.init(
            project=f"{config['metric']['behavior']}_test",
            entity=config["logging"]["entity"],
            name=create_run_string(),
            config=config,
        )

    davtt(
        config,
        train_cfg,
        tau2_cfg=tau2_cfg,
        fold_num=fold_num,
        model_name1=model_name1,
        seed1=seed1,
        model_name2=model_name2,
        seed2=seed2,
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
    model_name1: Optional[str] = None,
    seed1: Optional[str] = None,
    model_name2: Optional[str] = None,
    seed2: Optional[str] = None,
    overwrite: Optional[bool] = False,
    use_wandb: Optional[bool] = None,
    fold_size: int = 4000,
    pattern: str = r"_fold_(\d+)\.json$",
    metric: Optional[bool] = None,
    output_dir: str = "test_outputs",
):
    """Do repeats on"""
    # Initialize wandb if logging is enabled

    use_wandb = use_wandb if use_wandb is not None else config["logging"]["use_wandb"]
    metric = metric if metric else config["metric"]["metric"]
    if use_wandb:
        wandb.init(
            project=f"{config['metric']['behavior']}_test",
            entity=config["logging"]["entity"],
            name=create_run_string(),
            config=config,
            tags=["kfold"],
        )

    start = time.time()

    model_name1 = model_name1 if model_name1 else config["tau1"]["model_id"]
    seed1 = seed1 if seed1 else config["tau1"]["gen_seed"]
    model_name2 = model_name2 if model_name2 else config["tau2"]["model_id"]
    seed2 = seed2 if seed2 else config["tau2"]["gen_seed"]

    if use_wandb:
        wandb.config.update(
            {
                "model_name1": model_name1,
                "seed1": seed1,
                "model_name2": model_name2,
                "seed2": seed2,
                "fold_size": fold_size,
            }
        )
    else:
        print(
            f"Testing {model_name1} with seed {seed1} against {model_name2} with seed {seed2}. Fold size: {fold_size}"
        )

    # Check all folds available for the two runs
    directory = f"{output_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}"
    folds = []

    create_folds_from_evaluations(
        model_name1,
        seed1,
        model_name2,
        seed2,
        config["metric"]["metric"],
        overwrite=overwrite,
        fold_size=fold_size,
    )

    for file_name in os.listdir(directory):
        match = re.search(pattern, file_name)
        if match:
            fold_number = int(match.group(1))
            folds.append(fold_number)

    folds.sort()

    end = time.time()
    print(
        f"We have {len(folds)} folds. The whole initialization took {end-start} seconds."
    )

    if use_wandb:
        wandb.config.update({"total_num_folds": folds})

    # Iterate over the folds and call test_daht
    all_folds_data = pd.DataFrame()

    for fold_num in folds:
        print(f"Now starting experiment for fold {fold_num}")
        data = davtt(
            config,
            train_cfg,
            fold_num=fold_num,
            model_name1=model_name1,
            seed1=seed1,
            model_name2=model_name2,
            seed2=seed2,
            use_wandb=use_wandb,
        )
        all_folds_data = pd.concat([all_folds_data, data], ignore_index=True)

    print(f"Calculating neural net distance.")
    
    

    file_path = Path(directory) / f"kfold_test_results_{fold_size}_epsilon_{config['epsilon']}.csv"
    all_folds_data.to_csv(file_path, index=False)

    cleanup_files(directory, f"{metric}_scores_fold_*.json")

    if use_wandb:
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
        help="Fold number for repeated test runs. If this is given, only a single fold will be tested.",
    )

    parser.add_argument(
        "--fold_size",
        type=int,
        default=4000,
        help="Fold size when running kfold tests. Default is 4000.",
    )

    parser.add_argument(
        "--model_name1",
        type=str,
        default=None,
        help="Name of first model as it appears in the folder name",
    )

    parser.add_argument(
        "--model_name2",
        type=str,
        default=None,
        help="Name of second model as it appears in the folder name",
    )

    parser.add_argument(
        "--seed1",
        type=str,
        default=None,
        help="Generation seed of first model as it appears in the folder name",
    )

    parser.add_argument(
        "--seed2",
        type=str,
        default=None,
        help="Generation seed of second model as it appears in the folder name",
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="If this is set to true, then no tracking on wandb",
    )

    args = parser.parse_args()
    config = load_config(args.config_path)

    # Determine which experiment to run based on the argument
    if args.exp == "generation":
        eval_model(config, evaluate=args.evaluate)

    elif args.exp == "davtt":
        train_cfg = TrainCfg()
        if not args.fold_num:
            kfold_train(
                config,
                train_cfg,
                model_name1=args.model_name1,
                seed1=args.seed1,
                model_name2=args.model_name2,
                seed2=args.seed2,
                use_wandb=not args.no_wandb,
                fold_size=args.fold_size,
            )
        else:
            run_test_with_wandb(
                config,
                train_cfg,
                train_online=args.online,
                fold_num=args.fold_num,
                model_name1=args.model_name1,
                seed1=args.seed1,
                model_name2=args.model_name2,
                seed2=args.seed2,
                use_wandb=not args.no_wandb,
            )


if __name__ == "__main__":
    main()


