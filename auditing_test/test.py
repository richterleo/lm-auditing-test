import os
import importlib
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
from logging_config import setup_logging

from auditing_test.eval_trainer import OnlineTrainer, OfflineTrainer
from auditing_test.preprocessing import create_folds_from_evaluations, cleanup_files

from evaluation.nn_for_nn_distance import CMLP
from evaluation.analyze import get_distance_scores
from evaluation.plot import distance_box_plot

from utils.generate_and_evaluate import generate_and_evaluate
from utils.utils import (
    create_run_string,
)

# Add the parent directory of utils to sys.path
deep_anytime_testing = importlib.import_module("deep-anytime-testing")
train = importlib.import_module("deep-anytime-testing.trainer.trainer")
Trainer = getattr(train, "Trainer")


class Experiment:
    """ """

    def __init__(
        self,
        config: Dict,
        train_cfg: TrainCfg,
        output_dir: str,
        overwrite: bool = False,
        use_wandb: Optional[bool] = None,
        metric: Optional[bool] = None,
    ):
        self.config = config
        self.train_cfg = train_cfg
        self.output_dir = output_dir
        self.overwrite = overwrite

        self.use_wandb = (
            use_wandb if use_wandb is not None else config["logging"]["use_wandb"]
        )
        self.metric = metric if metric else config["metric"]["metric"]

        # initialize instance parameters to None
        self.model_name1 = None
        self.seed1 = None
        self.directory = None

        # logger setup
        self.logger = None

    def initialize_wandb(self, project: str, tags: List[str]):
        """ """
        wandb.init(
            project=project,
            entity=self.config["logging"]["entity"],
            name=create_run_string(),
            config=self.config,
            tags=tags,
        )

    def update_wandb(self, info_dict: Dict):
        """ """
        wandb.config.update(info_dict)

    def setup_logger(self, tag: str):
        """ """
        setup_logging(
            self.model_name1,
            self.seed1,
            tag=tag,
        )
        self.logger = logging.getLogger(__name__)

    def run(self):
        """ """
        raise NotImplementedError


class AuditingTest(Experiment):
    """ """

    def __init__(
        self,
        config: Dict,
        train_cfg: TrainCfg,
        overwrite: bool = False,
        use_wandb: Optional[bool] = None,
        metric: Optional[bool] = None,
        output_dir: str = "test_outputs",
        fold_pattern: str = r"_fold_(\d+)\.json$",  # TODO: maybe class attribute?
    ):
        super().__init__(
            config,
            train_cfg,
            output_dir,
            overwrite=overwrite,
            use_wandb=use_wandb,
            metric=metric,
        )
        self.fold_pattern = fold_pattern

        self.epsilon = self.config["epsilon"]

        # initialize instance parameters to None
        self.model_name2 = None
        self.seed2 = None
        self.fold_size = None

    def initialize_wandb(self, tags: List[str] = ["kfold"]):
        """ """
        project_name = f"{self.config['metric']['behavior']}_test"
        super().initialize_wandb(project=project_name, tags=tags)

    def update_wandb(self):
        """ """
        info_dict = {
            "model_name1": self.model_name1,
            "seed1": self.seed1,
            "model_name2": self.model_name2,
            "seed2": self.seed2,
            "fold_size": self.fold_size,
        }
        super().update_wandb(info_dict)

    def setup_logger(self, tag: str = "test_results"):
        """ """
        setup_logging(
            self.model_name1,
            self.seed1,
            self.model_name2,
            self.seed2,
            self.fold_size,
            self.epsilon,
            tag=tag,
        )
        self.logger = logging.getLogger(__name__)

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
            metric=self.metric,
            use_wandb=self.use_wandb,
            fold_num=fold_num,
            epsilon=self.epsilon,
        )

        return trainer.train()

    def kfold_davtt(self):
        """ """

        file_path = (
            Path(self.directory)
            / f"kfold_test_results_{self.fold_size}_epsilon_{self.epsilon}.csv"
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
                metric=self.config["metric"]["metric"],
                fold_size=self.fold_size,
                overwrite=self.overwrite,
            )

            for file_name in os.listdir(self.directory):
                match = re.search(self.fold_pattern, file_name)
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

        return sum_positive / len(folds)

    def analyze_and_plot_distance(self):
        """ """

        if self.config["analysis"]["num_samples"] == 0:
            num_train_samples = (
                self.fold_size // self.train_cfg.batch_size
            ) * self.train_cfg.batch_size - self.train_cfg.batch_size

        else:
            num_train_samples = self.config["analysis"]["num_samples"]

        num_runs = self.config["analysis"]["num_runs"]

        dist_path = (
            Path(self.directory) / f"distance_scores_{num_train_samples}_{num_runs}.csv"
        )
        if dist_path.exists():
            self.logger.info(
                f"Skipping distance analysis as results file {dist_path} already exists."
            )
            distance_df = pd.read_csv(dist_path)
        else:
            self.logger.info(
                f"Training neural net distance on {num_train_samples} samples for {num_runs} runs."
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
                num_samples=num_train_samples,
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
            num_samples=num_train_samples,
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

        if not self.logger:
            self.setup_logger(
                tag="test_results_and_analyze" if analyze_distance else "test_results"
            )
        if not self.directory:
            self.directory = f"{self.output_dir}/{self.model_name1}_{self.seed1}_{self.model_name2}_{self.seed2}"

        power = self.kfold_davtt()

        if analyze_distance:
            self.analyze_and_plot_distance()

        if self.use_wandb:
            wandb.finish()

        return power


class CalibratedAuditingTest(AuditingTest):
    def __init__(
        self,
        config: Dict,
        train_cfg: TrainCfg,
        overwrite: bool = False,
        use_wandb: Optional[bool] = None,
        metric: Optional[bool] = None,
        output_dir: str = "test_outputs",
        fold_pattern: str = r"_fold_(\d+)\.json$",  # TODO: maybe class attribute?
        num_samples: Optional[int] = 0,
        multiples_of_epsilon: int = 3,
        bias: float = 0,
    ):
        super().__init__(
            config,
            train_cfg,
            overwrite=overwrite,
            use_wandb=use_wandb,
            metric=metric,
            output_dir=output_dir,
            fold_pattern=fold_pattern,
        )

        self.num_samples = (
            num_samples if num_samples else config["analysis"]["num_samples"]
        )
        self.power_dict = {}
        self.multiples_of_epsilon = multiples_of_epsilon
        self.bias = bias

        self.num_train_samples = None
        self.num_runs = self.config["analysis"]["num_runs"]

    def setup_logger(self, tag: str = "test_results"):
        """ """
        setup_logging(
            self.model_name1,
            self.seed1,
            self.model_name2,
            self.seed2,
            self.fold_size,  # removed epsilon
            tag=tag,
        )
        self.logger = logging.getLogger(__name__)

    def calibrate_before_run(self):
        """ """

        dist_path = (
            Path(self.directory)
            / f"distance_scores_{self.num_train_samples}_{self.num_runs}.csv"
        )
        if dist_path.exists():
            self.logger.info(
                f"Skipping distance analysis as results file {dist_path} already exists."
            )
            distance_df = pd.read_csv(dist_path)
        else:
            self.logger.info(
                f"Training neural net distance on {self.num_train_samples} samples for {self.num_runs} runs."
            )
            distance_df = get_distance_scores(
                self.model_name1,
                self.seed1,
                self.seed2,
                model_name2=self.model_name2,
                metric=self.metric,
                num_runs=self.num_runs,
                net_cfg=self.config["net"],
                train_cfg=self.train_cfg,
                num_samples=self.num_train_samples,
            )

            distance_df.to_csv(dist_path, index=False)
            self.logger.info(f"Distance analysis results saved to {dist_path}.")

        nn_mean = distance_df["NeuralNet"].mean()
        nn_std = distance_df["NeuralNet"].std()

        if not self.bias == 0:
            self.logger.info(
                f"Subtracting bias of {self.bias} to the neural net distance epsilon."
            )
        return [
            nn_mean + nn_std * i - self.bias
            for i in range(-self.multiples_of_epsilon, self.multiples_of_epsilon + 1)
        ]

    def run(
        self, model_name1=None, seed1=None, model_name2=None, seed2=None, fold_size=4000
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

        self.directory = f"{self.output_dir}/{self.model_name1}_{self.seed1}_{self.model_name2}_{self.seed2}"

        if self.config["analysis"]["num_samples"] == 0:
            self.num_train_samples = (
                fold_size // self.train_cfg.batch_size
            ) * self.train_cfg.batch_size - self.train_cfg.batch_size

        else:
            self.num_train_samples = self.config["analysis"]["num_samples"]

        # set up logger
        self.setup_logger(tag="calibrated_test_results")

        epsilon_path = (
            Path(self.directory)
            / f"power_over_epsilon_{self.num_train_samples}_{self.num_runs}_{self.multiples_of_epsilon}.csv"
        )

        if epsilon_path.exists():
            self.logger.info(
                f"Calibrated testing results already exist in {epsilon_path}."
            )

        else:
            epsilons = self.calibrate_before_run()
            power_dict = {}

            for epsilon in epsilons:
                self.epsilon = epsilon
                self.logger.info(f"Running test for epsilon: {epsilon}.")
                power_dict[epsilon] = super().run(
                    model_name1=model_name1,
                    seed1=seed1,
                    model_name2=model_name2,
                    seed2=seed2,
                    fold_size=fold_size,
                    analyze_distance=False,
                )

            power_df = pd.DataFrame(power_dict.items(), columns=["epsilon", "power"])
            power_df.to_csv(epsilon_path, index=False)
            self.logger.info(f"Calibrated testing results saved to {epsilon_path}.")


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
