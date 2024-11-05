import numpy as np
import os
import importlib
import logging
import re
import time
import sys
import wandb
import pandas as pd

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple

# imports from other scripts
from arguments import TrainCfg
from logging_config import setup_logging

from auditing_test.calibration_strategies import CalibrationStrategy
from auditing_test.evaltrainer import OfflineTrainer
from auditing_test.preprocessing import create_folds_from_evaluations, cleanup_files
# from auditing_test.preprocessing_SuperNI import process_translation

from analysis.nn_for_nn_distance import CMLP
from analysis.analyze import get_distance_scores, get_mean_and_std_for_nn_distance
from analysis.plot import distance_box_plot, plot_calibrated_detection_rate

from utils.utils import (
    create_run_string,
)

# Add the parent directory of utils to sys.path
deep_anytime_testing = importlib.import_module("deep-anytime-testing")
train = importlib.import_module("deep-anytime-testing.trainer.trainer")
Trainer = getattr(train, "Trainer")

SCRIPT_DIR = Path(__file__).resolve().parent


class Test:
    """ """

    def __init__(
        self,
        config: Dict,
        train_cfg: TrainCfg,
        dir_prefix: str,
        test_dir: Optional[str] = "test_outputs",
        score_dir: Optional[str] = "model_scores",
        gen_dir: Optional[str] = "model_outputs",
        plot_dir: Optional[str] = "plots",
        overwrite: bool = False,
        use_wandb: Optional[bool] = None,
        metric: Optional[bool] = None,
        only_continuations: bool = True,
    ):
        self.config = config
        self.train_cfg = train_cfg

        self.dir_prefix = dir_prefix

        self.test_dir = SCRIPT_DIR.parent / dir_prefix / test_dir
        self.score_dir = SCRIPT_DIR.parent / dir_prefix / score_dir
        self.gen_dir = SCRIPT_DIR.parent / dir_prefix / gen_dir
        self.plot_dir = SCRIPT_DIR.parent / dir_prefix / plot_dir

        self.overwrite = overwrite

        self.use_wandb = use_wandb if use_wandb is not None else config["logging"]["use_wandb"]
        self.metric = metric if metric else config["metric"]["metric"]

        # initialize instance parameters to None
        self.model_name1 = None
        self.seed1 = None
        # self.directory = None

        # logger setup
        self.logger = None

        self.only_continuations = only_continuations

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


class AuditingTest(Test):
    """ """

    FOLD_PATTERN = r"_fold_(\d+)\.json$"

    def __init__(
        self,
        config: Dict,
        train_cfg: TrainCfg,
        dir_prefix: str,
        overwrite: bool = False,
        use_wandb: Optional[bool] = None,
        metric: Optional[bool] = None,
        only_continuations: bool = True,
    ):
        super().__init__(
            config,
            train_cfg,
            dir_prefix,
            overwrite=overwrite,
            use_wandb=use_wandb,
            metric=metric,
            only_continuations=only_continuations,
        )
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
            test_dir=self.test_dir,
            score_dir=self.score_dir,
            gen_dir=self.gen_dir,
            only_continuations=self.only_continuations,
        )

        return trainer.train()

    def kfold_davtt(self):
        """ """

        cont_string = "_continuations" if self.only_continuations else ""

        file_path = (
            Path(self.directory) / f"kfold_test_results{cont_string}_{self.fold_size}_epsilon_{self.epsilon}.csv"
        )

        if Path(file_path).exists() and not self.overwrite:
            self.logger.info(f"Skipping test as results file {file_path} already exists.")

            df = pd.read_csv(file_path)

            # calculate positive test rate
            test_positive_per_fold = df.groupby("fold_number")["test_positive"].max()
            positive_rate = test_positive_per_fold.sum() / len(test_positive_per_fold)

        else:
            self.logger.info(f"Running test for {self.model_name1}_{self.seed1} and {self.model_name2}_{self.seed2}.")
            self.logger.info(f"Saving results in folder: {self.directory}.")

            # for fast analysis
            sum_positive = int(0)
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
                score_dir=self.score_dir,
                gen_dir=self.gen_dir,
                test_dir=self.test_dir,
                only_continuations=self.only_continuations,
            )

            for file_name in os.listdir(self.directory):
                match = re.search(self.FOLD_PATTERN, file_name)
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
            positive_rate = sum_positive / len(folds)

        cleanup_files(self.directory, f"*scores_fold_*.json")

        self.logger.info(f"Positive tests: {positive_rate}, {round(positive_rate*100, 2)}%.")

        return positive_rate

    def analyze_and_plot_distance(self):
        """ """

        if self.config["analysis"]["num_samples"] == 0:
            num_train_samples = (
                self.fold_size // self.train_cfg.batch_size
            ) * self.train_cfg.batch_size - self.train_cfg.batch_size

        else:
            num_train_samples = self.config["analysis"]["num_samples"]

        num_runs = self.config["analysis"]["num_runs"]

        dist_path = Path(self.directory) / f"distance_scores_{num_train_samples}_{num_runs}.csv"
        if dist_path.exists():
            self.logger.info(f"Skipping distance analysis as results file {dist_path} already exists.")
            distance_df = pd.read_csv(dist_path)
        else:
            distance_df = get_distance_scores(
                self.model_name1,
                self.seed1,
                self.seed2,
                model_name2=self.model_name2,
                metric=self.metric,
                num_runs=num_runs,
                net_cfg=self.config["net"],
                train_cfg=self.train_cfg,
                num_samples=[self.train_cfg.batch_size, num_train_samples],
                num_test_samples=self.train_cfg.batch_size,
                only_continuations=self.only_continuations,
                score_dir=self.score_dir,
            )

            distance_df.to_csv(dist_path, index=False)
            self.logger.info(f"Distance analysis results saved to {dist_path}.")

        mean_nn_distance, std_nn_distance = get_mean_and_std_for_nn_distance(distance_df)
        self.logger.info(f"Average nn distance: {mean_nn_distance}, std: {std_nn_distance}")
        self.logger.info(f"Wasserstein distance: {distance_df['Wasserstein_comparison'].mean()}")

        if self.use_wandb:
            wandb.log(
                {
                    "average_nn_distance": distance_df["NeuralNet"].mean(),
                    "std_nn_distance": distance_df["NeuralNet"].std(),
                    "wasserstein_distance": distance_df["Wasserstein_comparison"].mean(),
                }
            )

        try:
            # Plot the results
            distance_box_plot(
                distance_df,
                self.model_name1,
                self.seed1,
                self.seed2,
                self.model_name2,
                metric=self.metric,
            )

        except FileNotFoundError as e:
            self.logger.error(f"Error plotting distance box plot: {e}")

    def run(
        self,
        model_name1=None,
        seed1=None,
        model_name2=None,
        seed2=None,
        fold_size=2000,
        analyze_distance=True,
        run_davtt=True,
        **kwargs,
    ):
        """ """
        self.model_name1 = model_name1 if model_name1 else self.config["tau1"]["model_id"]
        self.seed1 = seed1 if seed1 else self.config["tau1"]["gen_seed"]
        self.model_name2 = model_name2 if model_name2 else self.config["tau2"]["model_id"]
        self.seed2 = seed2 if seed2 else self.config["tau2"]["gen_seed"]
        self.fold_size = fold_size if fold_size else self.config["test_params"]["fold_size"]

        if self.use_wandb:
            self.initialize_wandb()
            self.update_wandb()

        self.setup_logger(tag="test_results_and_analyze" if analyze_distance else "test_results")
        self.directory = f"{self.test_dir}/{self.model_name1}_{self.seed1}_{self.model_name2}_{self.seed2}"

        if run_davtt:
            power = self.kfold_davtt()

        if analyze_distance:
            self.analyze_and_plot_distance()

        if self.use_wandb:
            wandb.finish()

        if run_davtt:
            return power


class CalibratedAuditingTest(AuditingTest):
    def __init__(
        self,
        config: Dict,
        train_cfg: TrainCfg,
        dir_prefix: str,
        calibration_strategy: CalibrationStrategy,
        overwrite: bool = False,
        use_wandb: Optional[bool] = None,
        # calibration_strategy: Optional[CalibrationStrategy] = None,
        metric: Optional[bool] = None,
        # num_samples: Optional[int] = 0,
        # use_full_ds_for_nn_distance: bool = False,
        only_continuations: bool = True,
    ):
        super().__init__(
            config,
            train_cfg,
            dir_prefix,
            overwrite=overwrite,
            use_wandb=use_wandb,
            metric=metric,
            # use_full_ds_for_nn_distance=use_full_ds_for_nn_distance,
            only_continuations=only_continuations,
        )

        # self.num_samples = num_samples if num_samples else config["analysis"]["num_samples"]
        self.power_dict = {}

        self.calibration_strategy = calibration_strategy

        self.num_runs = self.calibration_strategy.num_runs

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

    def run(
        self,
        model_name1=None,
        seed1=None,
        model_name2=None,
        seed2=None,
        fold_size=2000,
        calibrate_only=False,
        **kwargs,
    ):
        """ """

        self.model_name1 = model_name1 if model_name1 else self.config["tau1"]["model_id"]
        self.seed1 = seed1 if seed1 else self.config["tau1"]["gen_seed"]
        self.model_name2 = model_name2 if model_name2 else self.config["tau2"]["model_id"]
        self.seed2 = seed2 if seed2 else self.config["tau2"]["gen_seed"]
        self.fold_size = fold_size if fold_size else self.config["test_params"]["fold_size"]

        self.directory = self.test_dir / f"{self.model_name1}_{self.seed1}_{self.model_name2}_{self.seed2}"
        self.directory.mkdir(parents=True, exist_ok=True)

        cont_string = "_continuations" if self.only_continuations else ""

        self.setup_logger(tag="calibrated_test_results" if not calibrate_only else "calbrations")

        num_train_samples = (
            self.fold_size // self.train_cfg.batch_size
        ) * self.train_cfg.batch_size - self.train_cfg.batch_size

        epsilon_path = self.directory / f"power_over_epsilon{cont_string}_{num_train_samples}_{self.num_runs}.csv"

        if epsilon_path.exists() and not self.overwrite:
            self.logger.info(f"Calibrated testing results already exist in {epsilon_path}.")
            dist_path = Path(self.directory) / f"distance_scores_{self.num_train_samples}_{self.num_runs}.csv"
            try:
                distance_df = pd.read_csv(dist_path)
                true_epsilon, std_epsilon = get_mean_and_std_for_nn_distance(distance_df)
            except FileNotFoundError:
                self.logger.error(f"Distance analysis results file {dist_path} not found.")

        else:
            self.calibration_strategy.attach_logger(self.logger)
            dist_cfg = {
                "metric": self.metric,
                "net_cfg": self.config["net"],
                "train_cfg": self.train_cfg,
                "num_test_samples": self.train_cfg.batch_size,
                "only_continuations": self.only_continuations,
            }

            epsilons, true_epsilon, std_epsilon = self.calibration_strategy.calculate_epsilons(
                self.model_name1,
                self.seed1,
                self.model_name2,
                self.seed2,
                [self.train_cfg.batch_size, num_train_samples],
                self.dir_prefix,
                dist_cfg,
            )

            self.logger.info(f"Calibrated epsilons: {epsilons}.")
            self.logger.info(f"True distance: {true_epsilon}")

            epsilons.append(true_epsilon)

            if not calibrate_only:
                power_dict = {}

                for epsilon in epsilons:
                    self.epsilon = epsilon
                    self.logger.info(f"Running test for epsilon: {epsilon}.")
                    power_dict[epsilon] = super().run(
                        model_name1=self.model_name1,
                        seed1=self.seed1,
                        model_name2=self.model_name2,
                        seed2=self.seed2,
                        fold_size=self.fold_size,
                        analyze_distance=False,
                    )

                power_df = pd.DataFrame(power_dict.items(), columns=["epsilon", "power"])
                power_df.to_csv(epsilon_path, index=False)
                self.logger.info(f"Calibrated testing results saved to {epsilon_path}.")

        plot_calibrated_detection_rate(
            self.model_name1,
            self.seed1,
            self.model_name2,
            self.seed2,
            true_epsilon=true_epsilon,
            std_epsilon=std_epsilon,
            result_file=epsilon_path,
            draw_in_std=True,
            draw_in_first_checkpoint=False,
            draw_in_lowest_and_highest=True,
            fold_size=self.fold_size,
            # overwrite=True,
        )
