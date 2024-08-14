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


class Experiment:
    """ """

    def __init__(
        self,
        config: Dict,
        train_cfg: TrainCfg,
        overwrite: Optional[bool] = False,
        use_wandb: Optional[bool] = None,
        pattern: str = r"_fold_(\d+)\.json$",  # TODO: maybe class attribute?
        metric: Optional[bool] = None,
        output_dir: str = "test_outputs",
    ):
        self.config = config
        self.train_cfg = train_cfg
        self.overwrite = overwrite
        self.pattern = pattern
        self.output_dir = output_dir

        self.use_wandb = (
            use_wandb if use_wandb is not None else config["logging"]["use_wandb"]
        )
        self.metric = metric if metric else config["metric"]["metric"]

        # initialize instance parameters to None
        self.model_name1 = None
        self.seed1 = None
        self.model_name2 = None
        self.seed2 = None
        self.fold_size = None
        self.directory = None

        # logger setup
        self.logger = None

    def initialize_wandb(self, tags: List[str] = ["kfold"]):
        """ """
        wandb.init(
            project=f"{self.config['metric']['behavior']}_test",
            entity=self.config["logging"]["entity"],
            name=create_run_string(),
            config=self.config,
            tags=tags,
        )

    def update_wandb(self):
        """ """
        wandb.config.update(
            {
                "model_name1": self.model_name1,
                "seed1": self.seed1,
                "model_name2": self.model_name2,
                "seed2": self.seed2,
                "fold_size": self.fold_size,
            }
        )

    def setup_logger(self, tag: str = "test_results"):
        """ """
        setup_logging(
            self.model_name1,
            self.seed1,
            model_name2=self.model_name2,
            seed2=self.seed2,
            fold_size=self.fold_size,
            epsilon=self.config["epsilon"],
            tag=tag,
        )
        self.logger = logging.getLogger(__name__)

    def kfold_test(self):
        """ """

        file_path = (
            Path(self.directory)
            / f"kfold_test_results_{self.fold_size}_epsilon_{config['epsilon']}.csv"
        )

        if Path(file_path).exists() and not self.overwrite:
            self.logger.info(
                f"Skipping test as results file {file_path} already exists."
            )

        else:
            self.logger.info(
                f"Running test for {self.model_name1}_{self.seed1} and {self.model_name2}_{self.seed2}."
            )
            self.logger.info(f"Saving results in folder: {self.directory}.")

            # for fast analysis
            sum_positive = 0
            start = time.time()
            folds = []

            create_folds_from_evaluations(
                self.model_name1,
                self.seed1,
                self.model_name2,
                self.seed2,
                metric=config["metric"]["metric"],
                fold_size=self.fold_size,
                overwrite=self.overwrite,
            )

            for file_name in os.listdir(self.directory):
                match = re.search(self.pattern, file_name)
                if match:
                    fold_number = int(match.group(1))
                    folds.append(fold_number)

            folds.sort()

            end = time.time()
            self.logger.info(
                f"We have {len(folds)} folds. The whole initialization took {round(end-start, 3)} seconds."
            )

            if self.use_wandb:
                wandb.config.update({"total_num_folds": folds})

            # Iterate over the folds and call test
            all_folds_data = pd.DataFrame()

            for fold_num in folds:
                self.logger.info(f"Now starting experiment for fold {fold_num}.")
                data, test_positive = self.davtt(fold_num)
                all_folds_data = pd.concat([all_folds_data, data], ignore_index=True)
                if test_positive:
                    sum_positive += 1

            all_folds_data.to_csv(file_path, index=False)
            self.logger.info(
                f"Positive tests: {sum_positive}/{len(folds)}, {round(sum_positive/len(folds)*100, 2)}%."
            )

        cleanup_files(self.directory, f"{self.metric}_scores_fold_*.json")

    def davtt(self, fold_num: int):
        """
        Deep anytime-valid tolerance test

        Args:
            fold_num: int
                The fold number to run the test on.
        """

        # Define network for betting score
        # TODO: change this betting_net = initialize_from_config(config["net"])
        betting_net = CMLP(
            self.config["net"]["input_size"],
            self.config["net"]["hidden_layer_size"],
            1,
            self.config["net"]["layer_norm"],
            False,
            0.4,
            self.config["net"]["bias"],
        )

        trainer = OfflineTrainer(
            self.train_cfg,
            betting_net,
            self.model_name1,
            self.seed1,
            self.model_name2,
            self.seed2,
            metric=self.config["metric"]["metric"],
            use_wandb=self.use_wandb,
            fold_num=fold_num,
            epsilon=self.config["epsilon"],
        )

        return trainer.train()

    def analyze_and_plot_distance(self):
        """ """

        if config["analysis"]["num_samples"] == 0:
            train_num_samples = (
                self.fold_size // self.train_cfg.batch_size
            ) * self.train_cfg.batch_size - self.train_cfg.batch_size

        else:
            train_num_samples = config["analysis"]["num_samples"]

        num_runs = self.config["analysis"]["num_runs"]

        dist_path = (
            Path(self.directory) / f"distance_scores_{train_num_samples}_{num_runs}.csv"
        )
        if dist_path.exists():
            self.logger.info(
                f"Skipping distance analysis as results file {dist_path} already exists."
            )
            distance_df = pd.read_csv(dist_path)
        else:
            self.logger.info(
                f"Training neural net distance on {train_num_samples} samples for {num_runs} runs."
            )
            distance_df = get_distance_scores(
                self.model_name1,
                self.seed1,
                self.seed2,
                model_name2=self.model_name2,
                metric=self.metric,
                num_runs=num_runs,
                net_cfg=self.config["net"],
                train_cfg=self.train_cfg,
                num_samples=train_num_samples,
            )

            distance_df.to_csv(dist_path, index=False)
            self.logger.info(f"Distance analysis results saved to {dist_path}.")

        self.logger.info(
            f"Average nn distance: {distance_df['NeuralNet'].mean()}, std: {distance_df['NeuralNet'].std()}"
        )
        self.logger.info(f"Wasserstein distance: {distance_df['Wasserstein'].mean()}")

        if self.use_wandb:
            wandb.log(
                {
                    "average_nn_distance": distance_df["NeuralNet"].mean(),
                    "std_nn_distance": distance_df["NeuralNet"].std(),
                    "wasserstein_distance": distance_df["Wasserstein"].mean(),
                }
            )

        # Plot the results
        distance_box_plot(
            distance_df,
            self.model_name1,
            self.seed1,
            self.seed2,
            self.model_name2,
            metric=self.metric,
            num_samples=train_num_samples,
        )

    def run(
        self,
        model_name1=None,
        seed1=None,
        model_name2=None,
        seed2=None,
        fold_size=4000,
        analyze_distance=True,
    ):
        """ """
        self.model_name1 = (
            model_name1 if model_name1 else self.config["tau1"]["model_id"]
        )
        self.seed1 = seed1 if seed1 else self.config["tau1"]["gen_seed"]
        self.model_name2 = (
            model_name2 if model_name2 else self.config["tau2"]["model_id"]
        )
        self.seed2 = seed2 if seed2 else self.config["tau2"]["gen_seed"]
        self.fold_size = fold_size

        if self.use_wandb:
            self.initialize_wandb()
            self.update_wandb()

        self.setup_logger(
            tag="test_results_and_analyze" if analyze_distance else "test_results"
        )

        self.directory = f"{self.output_dir}/{self.model_name1}_{self.seed1}_{self.model_name2}_{self.seed2}"

        self.kfold_test()

        if analyze_distance:
            self.analyze_and_plot_distance()

        if self.use_wandb:
            wandb.finish()


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

    parser.add_argument(
        "--no_analysis",
        action="store_true",
        help="If this is set to true, then no analysis after runnning the test.",
    )

    args = parser.parse_args()
    config = load_config(args.config_path)

    # Determine which experiment to run based on the argument
    if args.exp == "generation":
        eval_model(config, evaluate=args.evaluate)

    elif args.exp == "test":
        train_cfg = TrainCfg()
        exp = Experiment(
            config,
            train_cfg,
            use_wandb=not args.no_wandb,
        )
        exp.run(
            model_name1=args.model_name1,
            seed1=args.seed1,
            model_name2=args.model_name2,
            seed2=args.seed2,
            fold_size=args.fold_size,
            analyze_distance=not args.no_analysis,
        )


if __name__ == "__main__":
    config = load_config("config.yml")
    train_cfg = TrainCfg()
    model_name1 = "Meta-Llama-3-8B-Instruct"
    model_name2 = "Llama-3-8B-ckpt5"
    seed1 = "seed1000"
    seed2 = "seed1000"

    exp = Experiment(
        config,
        train_cfg,
        use_wandb=False,
    )
    exp.run(
        model_name1=model_name1,
        seed1=seed1,
        model_name2=model_name2,
        seed2=seed2,
        fold_size=4000,
        analyze_distance=True,
    )

    # main()
