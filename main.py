import argparse
import os
import logging
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
submodule_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "deep-anytime-testing")
)
models_path = os.path.join(submodule_path, "models")

for path in [submodule_path, models_path]:
    if path not in sys.path:
        sys.path.append(path)

# logging
from logging_config import setup_logging

from auditing_test.eval_trainer import OnlineTrainer, OfflineTrainer
from auditing_test.preprocessing import create_folds_from_evaluations, cleanup_files

from evaluation.nn_for_nn_distance import CMLP
from evaluation.analyze import get_distance_scores
from evaluation.plot import distance_box_plot


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
    use_wandb = use_wandb if use_wandb is not None else config["logging"]["use_wandb"]

    model_name1 = model_name1 if model_name1 else config["tau1"]["model_id"]
    seed1 = seed1 if seed1 else config["tau1"]["gen_seed"]
    model_name2 = model_name2 if model_name2 else config["tau2"]["model_id"]
    seed2 = seed2 if seed2 else config["tau2"]["gen_seed"]

    # Define network for betting score

    # TODO: change this betting_net = initialize_from_config(config["net"])
    betting_net = CMLP(
        config["net"]["input_size"],
        config["net"]["hidden_layer_size"],
        1,
        config["net"]["layer_norm"],
        False,
        0.4,
        config["net"]["bias"],
    )

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
        epsilon=config["epsilon"],
    )

    return trainer.train()


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


def kfold_test(
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
    verbose: bool = True,
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

    setup_logging(
        model_name1,
        seed1,
        model_name2=model_name2,
        seed2=seed2,
        fold_size=fold_size,
        epsilon=config["epsilon"],
        tag="test_results",
    )
    logger = logging.getLogger(__name__)

    # for fast analysis
    sum_positive = 0

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
        logger.info(
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
    logger.info(
        f"We have {len(folds)} folds. The whole initialization took {round(end-start, 3)} seconds."
    )

    if use_wandb:
        wandb.config.update({"total_num_folds": folds})

    # Iterate over the folds and call test_daht
    all_folds_data = pd.DataFrame()

    for fold_num in folds:
        logger.info(f"Now starting experiment for fold {fold_num}")
        data, test_positive = davtt(
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
        if test_positive:
            sum_positive += 1

    file_path = (
        Path(directory)
        / f"kfold_test_results_{fold_size}_epsilon_{config['epsilon']}.csv"
    )
    all_folds_data.to_csv(file_path, index=False)

    cleanup_files(directory, f"{metric}_scores_fold_*.json")

    logger.info(f"Positive tests: {sum_positive}/{len(folds)}")

    if config["analysis"]["calculate_distance"]:
        logger.info(
            f"Calculating actual distance over {config['analysis']['num_runs']} runs."
        )
        dist_df = analyze_and_plot_distance(
            config, model_name1, seed1, model_name2, seed2
        )
        dist_df.to_csv(Path(directory) / "distance_scores.csv", index=False)
        logger.info(
            f"Average nn distance: {dist_df['distance'].mean()}, std: {dist_df['distance'].std()}"
        )
        logger.info(f"Wasserstein distance: {dist_df['wasserstein'].mean()}")
        wandb.log(
            {
                "average_nn_distance": dist_df["distance"].mean(),
                "std_nn_distance": dist_df["distance"].std(),
                "wasserstein_distance": dist_df["wasserstein"].mean(),
            }
        )

    if use_wandb:
        wandb.finish()


def analyze_and_plot_distance(config, model_name1, seed1, model_name2, seed2):
    """ """
    # Load the data

    distance_df = get_distance_scores(
        model_name1,
        seed1,
        seed2,
        model_name2=model_name2,
        metric=config["metric"]["metric"],
        num_runs=config["analysis"]["num_runs"],
    )

    # Plot the results
    distance_box_plot(
        distance_df,
        model_name1,
        seed1,
        seed2,
        model_name2,
        config["metric"]["metric"],
    )

    return distance_df


def main():
    """ """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--exp",
        type=str,
        choices=["generation", "test"],
        required=True,
        help="Select the experiment to run: generating model outputs or auditing test.",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether to evaluate the model on the metric.",
    )

    # TODO: change this when OnlineTrainer is no longer deprecated.
    parser.add_argument(
        "--online",
        action="store_true",
        help="Whether to use the OnlineTrainer instead of the OfflineTrainer. Warning: OnlineTrainer is currently deprecated.",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yml",
        help="Path to config file",
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
        help="Name of first model as it appears in the folder name.",
    )

    parser.add_argument(
        "--model_name2",
        type=str,
        default=None,
        help="Name of second model as it appears in the folder name.",
    )

    parser.add_argument(
        "--seed1",
        type=str,
        default=None,
        help="Generation seed of first model as it appears in the folder name.",
    )

    parser.add_argument(
        "--seed2",
        type=str,
        default=None,
        help="Generation seed of second model as it appears in the folder name.",
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="If this is set to true, then no tracking on wandb.",
    )

    args = parser.parse_args()
    config = load_config(args.config_path)

    # Determine which experiment to run based on the argument
    if args.exp == "generation":
        eval_model(config, evaluate=args.evaluate)

    elif args.exp == "test":
        train_cfg = TrainCfg()
        kfold_test(
            config,
            train_cfg,
            model_name1=args.model_name1,
            seed1=args.seed1,
            model_name2=args.model_name2,
            seed2=args.seed2,
            use_wandb=not args.no_wandb,
            fold_size=args.fold_size,
        )


if __name__ == "__main__":
    config = load_config("config.yml")
    train_cfg = TrainCfg()
    model_name1 = "Meta-Llama-3-8B-Instruct"
    model_name2 = "Llama-3-8B-ckpt4"
    seed1 = "seed1000"
    seed2 = "seed1000"

    kfold_test(
        config,
        train_cfg,
        model_name1=model_name1,
        seed1=seed1,
        model_name2=model_name2,
        seed2=seed2,
        use_wandb=False,
        fold_size=4000,
    )

    # main()
