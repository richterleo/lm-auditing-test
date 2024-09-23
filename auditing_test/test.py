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
from typing import Optional, Dict, List, Union

# imports from other scripts
from arguments import TrainCfg
from logging_config import setup_logging

from auditing_test.eval_trainer import OnlineTrainer, OfflineTrainer
from auditing_test.preprocessing import create_folds_from_evaluations, cleanup_files

from evaluation.nn_for_nn_distance import CMLP
from evaluation.analyze import get_distance_scores, get_mean_and_std_for_nn_distance
from evaluation.plot import distance_box_plot, plot_calibrated_detection_rate

from utils.generate_and_evaluate import (
    generate_on_dataset,
    generate_on_dataset_with_model,
)
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

        self.use_wandb = use_wandb if use_wandb is not None else config["logging"]["use_wandb"]
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

    FOLD_PATTERN = r"_fold_(\d+)\.json$"

    def __init__(
        self,
        config: Dict,
        train_cfg: TrainCfg,
        overwrite: bool = False,
        use_wandb: Optional[bool] = None,
        metric: Optional[bool] = None,
        output_dir: str = "test_outputs",
        use_full_ds_for_nn_distance: bool = False,
    ):
        super().__init__(
            config,
            train_cfg,
            output_dir,
            overwrite=overwrite,
            use_wandb=use_wandb,
            metric=metric,
        )
        self.epsilon = self.config["epsilon"]

        # initialize instance parameters to None
        self.model_name2 = None
        self.seed2 = None
        self.fold_size = None

        # for neural net distance evaluation
        self.use_full_ds_for_nn_distance = use_full_ds_for_nn_distance

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

        file_path = Path(self.directory) / f"kfold_test_results_{self.fold_size}_epsilon_{self.epsilon}.csv"

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

        cleanup_files(self.directory, f"{self.metric}_scores_fold_*.json")

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
            )

            distance_df.to_csv(dist_path, index=False)
            self.logger.info(f"Distance analysis results saved to {dist_path}.")

        mean_nn_distance, std_nn_distance = get_mean_and_std_for_nn_distance(distance_df)
        self.logger.info(f"Average nn distance: {mean_nn_distance}, std: {std_nn_distance}")
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
        )

    def run(
        self,
        model_name1=None,
        seed1=None,
        model_name2=None,
        seed2=None,
        fold_size=4000,
        analyze_distance=True,
        run_davtt=True,
    ):
        """ """
        self.model_name1 = model_name1 if model_name1 else self.config["tau1"]["model_id"]
        self.seed1 = seed1 if seed1 else self.config["tau1"]["gen_seed"]
        self.model_name2 = model_name2 if model_name2 else self.config["tau2"]["model_id"]
        self.seed2 = seed2 if seed2 else self.config["tau2"]["gen_seed"]
        self.fold_size = fold_size

        if self.use_wandb:
            self.initialize_wandb()
            self.update_wandb()

        if not self.logger:
            self.setup_logger(tag="test_results_and_analyze" if analyze_distance else "test_results")
        if not self.directory:
            self.directory = f"{self.output_dir}/{self.model_name1}_{self.seed1}_{self.model_name2}_{self.seed2}"

        if run_davtt:
            power = self.kfold_davtt()

        if analyze_distance:
            self.analyze_and_plot_distance()

        if self.use_wandb:
            wandb.finish()

        if run_davtt:
            return power


class EpsilonStrategy(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the epsilon calculation strategy with any necessary parameters.

        :param kwargs: Dictionary of configuration parameters
        """
        pass

    def attach_logger(self, logger: logging.Logger):
        self.logger = logger

    @abstractmethod
    def calculate_epsilons(self, **kwargs) -> list:
        """
        Calculate a list of epsilons based on the distance dataframe.

        :param distance_df: DataFrame containing distance scores
        :return: List of calculated epsilons
        """
        pass


class DefaultEpsilonStrategy(EpsilonStrategy):
    def __init__(
        self,
        overwrite: bool = True,
        multiples_of_epsilon: Optional[int] = None,
        bias: Optional[float] = None,
        use_full_ds_for_nn_distance: Optional[bool] = None,
        num_runs: Optional[int] = None,
        config: Optional[Dict] = None,
    ):
        self.overwrite = overwrite
        self.multiples_of_epsilon = (
            multiples_of_epsilon if multiples_of_epsilon is not None else config["analysis"]["multiples_of_epsilon"]
        )
        self.bias = bias if bias else config["analysis"]["bias"]
        self.use_full_ds_for_nn_distance = (
            use_full_ds_for_nn_distance
            if use_full_ds_for_nn_distance is not None
            else config["analysis"]["use_full_ds_for_nn_distance"]
        )

        self.use_full_ds_for_nn_distance = (
            use_full_ds_for_nn_distance
            if use_full_ds_for_nn_distance is not None
            else config["analysis"]["use_full_ds_for_nn_distance"]
        )
        self.num_runs = num_runs if num_runs is not None else config["analysis"]["num_runs"]

    def calculate_epsilons(self, test_dir, num_train_samples, **distance_score_kwargs) -> list:
        dist_path = Path(test_dir) / f"distance_scores_{num_train_samples}_{self.num_runs}.csv"
        if dist_path.exists():
            self.logger.info(f"Skipping distance analysis as results file {dist_path} already exists.")
            distance_df = pd.read_csv(dist_path)
        else:
            self.logger.info(f"Training neural net distance on {num_train_samples} samples for {self.num_runs} runs.")
            # TODO: refactor this
            distance_df = get_distance_scores(
                distance_score_kwargs["model_name1"],
                distance_score_kwargs["seed1"],
                distance_score_kwargs["seed2"],
                model_name2=distance_score_kwargs["model_name2"],
                metric=distance_score_kwargs["metric"],
                num_runs=self.num_runs,
                net_cfg=distance_score_kwargs["net_cfg"],
                train_cfg=distance_score_kwargs["train_cfg"],
                num_samples=distance_score_kwargs["num_samples"],
                num_test_samples=distance_score_kwargs["num_test_samples"],
            )

            distance_df.to_csv(dist_path, index=False)
            self.logger.info(f"Distance analysis results saved to {dist_path}.")

        mean_nn_distance, std_nn_distance = get_mean_and_std_for_nn_distance(distance_df)
        self.logger.info(f"Average nn distance: {mean_nn_distance}, std: {std_nn_distance}.")

        if not self.bias == 0:
            self.logger.info(f"Subtracting bias of {self.bias} to the neural net distance epsilon.")
        return sorted(
            list(
                set(
                    max(mean_nn_distance + std_nn_distance * i - self.bias, 0)
                    for i in range(-self.multiples_of_epsilon, self.multiples_of_epsilon + 1)
                )
            )
        ), mean_nn_distance


class CrossValEpsilonStrategy(EpsilonStrategy):
    def __init__(
        self,
        models_and_seeds: List[Dict[str, Union[str, int]]],
        overwrite: bool = True,
        multiples_of_epsilon: Optional[int] = None,
        bias: Optional[float] = None,
        use_full_ds_for_nn_distance: Optional[bool] = None,
        num_runs: Optional[int] = None,
        config: Optional[Dict] = None,
        autocorrelate: bool = False,
    ):
        self.models_and_seeds = models_and_seeds
        self.overwrite = overwrite
        self.multiples_of_epsilon = (
            multiples_of_epsilon if multiples_of_epsilon is not None else config["analysis"]["multiples_of_epsilon"]
        )
        self.bias = bias if bias else config["analysis"]["bias"]
        self.use_full_ds_for_nn_distance = (
            use_full_ds_for_nn_distance
            if use_full_ds_for_nn_distance is not None
            else config["analysis"]["use_full_ds_for_nn_distance"]
        )
        self.num_runs = num_runs if num_runs is not None else config["analysis"]["num_runs"]

        self.autocorrelate = autocorrelate

    def calculate_epsilons(self, test_dir, num_train_samples, **distance_score_kwargs) -> list:
        all_distances = []

        # Extract the base directory
        base_dir = Path(test_dir).parent

        for model in self.models_and_seeds:
            model_name = model["model_name"]
            seed = model["seed"]

            base_model = distance_score_kwargs["model_name1"]
            base_seed = distance_score_kwargs["seed1"]

            test_model = distance_score_kwargs["model_name2"]
            test_seed = distance_score_kwargs["seed2"]

            if not self.autocorrelate:
                if (model_name == base_model and seed == base_seed) or (model_name == test_model and seed == test_seed):
                    self.logger.info(f"Skipped model {model_name}_{seed}.")
                    continue

            # Construct the new directory path
            new_dir = (
                base_dir
                / f"{distance_score_kwargs['model_name1']}_{distance_score_kwargs['seed1']}_{model_name}_{seed}"
            )

            # Ensure the directory exists
            new_dir.mkdir(parents=True, exist_ok=True)

            dist_path = new_dir / f"distance_scores_{num_train_samples}_{self.num_runs}.csv"

            if dist_path.exists() and not self.overwrite:
                self.logger.info(f"Loading existing distance analysis for {model_name}_{seed} from {dist_path}.")
                distance_df = pd.read_csv(dist_path)
            else:
                self.logger.info(
                    f"Training neural net distance for {model_name}_{seed} on {num_train_samples} samples for {self.num_runs} runs."
                )

                distance_df = get_distance_scores(
                    distance_score_kwargs["model_name1"],
                    distance_score_kwargs["seed1"],
                    seed,
                    model_name2=model_name,
                    metric=distance_score_kwargs["metric"],
                    num_runs=self.num_runs,
                    net_cfg=distance_score_kwargs["net_cfg"],
                    train_cfg=distance_score_kwargs["train_cfg"],
                    num_samples=distance_score_kwargs["num_samples"],
                    num_test_samples=distance_score_kwargs["num_test_samples"],
                )

                distance_df.to_csv(dist_path, index=False)
                self.logger.info(f"Distance analysis results for {model_name}_{seed} saved to {dist_path}.")

            mean_nn_distance, _ = get_mean_and_std_for_nn_distance(distance_df)
            all_distances.extend(distance_df["NeuralNet"].tolist())
            self.logger.info(f"Mean nn distance for {model_name}_{seed}: {mean_nn_distance}.")

        overall_mean = np.mean(all_distances)
        overall_std = np.std(all_distances)

        self.logger.info(f"Overall average nn distance: {overall_mean}, overall std: {overall_std}.")

        if not self.bias == 0:
            self.logger.info(f"Subtracting bias of {self.bias} to the neural net distance epsilon.")

        return sorted(
            list(
                set(
                    max(overall_mean + overall_std * i - self.bias, 0)
                    for i in range(-self.multiples_of_epsilon, self.multiples_of_epsilon + 1)
                )
            )
        ), overall_mean


class IntervalEpsilonStrategy(EpsilonStrategy):
    def __init__(
        self,
        lower_model: str,
        lower_seed: int,
        upper_model: str,
        upper_seed: int,
        base_model: Optional[str] = None,
        base_seed: Optional[int] = None,
        overwrite: bool = True,
        epsilon_ticks: Optional[int] = None,
        epsilon_interval: Optional[float] = None,
        use_full_ds_for_nn_distance: Optional[bool] = None,
        num_runs: Optional[int] = None,
        config: Optional[Dict] = None,
    ):
        self.lower_model = lower_model
        self.lower_seed = lower_seed
        self.upper_model = upper_model
        self.upper_seed = upper_seed

        self.base_model = base_model
        self.base_seed = base_seed

        self.overwrite = overwrite
        self.use_full_ds_for_nn_distance = (
            use_full_ds_for_nn_distance
            if use_full_ds_for_nn_distance is not None
            else config["analysis"]["use_full_ds_for_nn_distance"]
        )
        self.num_runs = num_runs if num_runs is not None else config["analysis"]["num_runs"]

        if epsilon_ticks:
            self.epsilon_ticks = epsilon_ticks
            self.epsilon_interval = None
        else:
            self.epsilon_ticks = None
            self.epsilon_interval = epsilon_interval

        if self.epsilon_ticks is None and self.epsilon_interval is None:
            # default is interval
            self.epsilon_ticks = config["analysis"]["epsilon_ticks"]

    def calculate_epsilons(self, test_dir, num_train_samples, **distance_score_kwargs) -> list:
        distances = {}
        stds = {}

        base_model = self.base_model if self.base_model else distance_score_kwargs["base_model"]
        base_seed = self.base_seed if self.base_seed else distance_score_kwargs["base_seed"]
        self.logger.info(f"Base model: {base_model}")

        test_model = distance_score_kwargs["test_model"]
        test_seed = distance_score_kwargs["test_seed"]

        base_dir = Path(test_dir).parent

        models = [
            ("test", test_model, test_seed),
            ("lower", self.lower_model, self.lower_seed),
            ("upper", self.upper_model, self.upper_seed),
        ]

        for model_type, model, seed in models:
            distances[model_type], stds[model_type] = self.process_model(
                base_dir, base_model, base_seed, model, seed, num_train_samples, distance_score_kwargs
            )

        if self.epsilon_interval:
            epsilons = np.arange(distances["lower"], distances["upper"], self.epsilon_interval)
        else:
            epsilons = np.linspace(distances["lower"], distances["upper"], self.epsilon_ticks)

        return (
            epsilons,
            distances["test"],
            stds["test"],
        )

    def process_model(self, base_dir, base_model, base_seed, model, seed, num_train_samples, distance_score_kwargs):
        model_dir = base_dir / f"{base_model}_{base_seed}_{model}_{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)

        dist_path = model_dir / f"distance_scores_{num_train_samples}_{self.num_runs}.csv"

        if dist_path.exists() and not self.overwrite:
            self.logger.info(f"Loading existing distance analysis for {model}_{seed} from {dist_path}.")
            distance_df = pd.read_csv(dist_path)
        else:
            self.logger.info(
                f"Training neural net distance for {model}_{seed} on {num_train_samples} samples for {self.num_runs} runs."
            )

            distance_df = get_distance_scores(
                base_model, base_seed, seed, model_name2=model, num_runs=self.num_runs, **distance_score_kwargs
            )

            distance_df.to_csv(dist_path, index=False)
            self.logger.info(f"Distance analysis results for {model}_{seed} saved to {dist_path}.")

        dist, std = get_mean_and_std_for_nn_distance(distance_df)
        self.logger.info(f"Mean nn distance for {model}_{seed}: {dist}.")
        self.logger.info(f"Std of nn distance for {model}_{seed}: {std}.")

        return dist, std


class CalibratedAuditingTest(AuditingTest):
    def __init__(
        self,
        config: Dict,
        train_cfg: TrainCfg,
        calibration_strategy: EpsilonStrategy,
        overwrite: bool = False,
        use_wandb: Optional[bool] = None,
        metric: Optional[bool] = None,
        output_dir: str = "test_outputs",
        num_samples: Optional[int] = 0,
        use_full_ds_for_nn_distance: bool = False,
        only_continuations: bool = True,
    ):
        super().__init__(
            config,
            train_cfg,
            overwrite=overwrite,
            use_wandb=use_wandb,
            metric=metric,
            output_dir=output_dir,
            use_full_ds_for_nn_distance=use_full_ds_for_nn_distance,
        )

        self.num_samples = num_samples if num_samples else config["analysis"]["num_samples"]
        self.power_dict = {}

        self.num_train_samples = None

        self.calibration_strategy = calibration_strategy

        self.num_runs = self.calibration_strategy.num_runs
        self.only_continuations = only_continuations

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
        fold_size=4000,
        calibrate_only=False,
    ):
        """ """

        self.model_name1 = model_name1 if model_name1 else self.config["tau1"]["model_id"]
        self.seed1 = seed1 if seed1 else self.config["tau1"]["gen_seed"]
        self.model_name2 = model_name2 if model_name2 else self.config["tau2"]["model_id"]
        self.seed2 = seed2 if seed2 else self.config["tau2"]["gen_seed"]
        self.fold_size = fold_size

        self.directory = f"{self.output_dir}/{self.model_name1}_{self.seed1}_{self.model_name2}_{self.seed2}"
        if not Path(self.directory).exists():
            Path(self.directory).mkdir(parents=True)

        if self.config["analysis"]["num_samples"] == 0:
            self.num_train_samples = (
                fold_size // self.train_cfg.batch_size
            ) * self.train_cfg.batch_size - self.train_cfg.batch_size

        else:
            self.num_train_samples = self.config["analysis"]["num_samples"]

        # set up logger
        self.setup_logger(tag="calibrated_test_results")

        epsilon_path = (
            Path(self.directory) / f"power_over_epsilon_{self.num_train_samples}_{self.num_runs}.csv"
            if not self.only_continuations
            else Path(self.directory) / f"power_over_epsilon_continuations_{self.num_train_samples}_{self.num_runs}.csv"
        )

        if epsilon_path.exists():
            self.logger.info(f"Calibrated testing results already exist in {epsilon_path}.")
            dist_path = Path(self.directory) / f"distance_scores_{self.num_train_samples}_{self.num_runs}.csv"
            try:
                distance_df = pd.read_csv(dist_path)
                true_epsilon, std_epsilon = get_mean_and_std_for_nn_distance(distance_df)
            except FileNotFoundError:
                self.logger.error(f"Distance analysis results file {dist_path} not found.")

        else:
            self.calibration_strategy.attach_logger(self.logger)
            calibration_cfg = {
                "base_model": self.model_name1,
                "base_seed": self.seed1,
                "test_model": self.model_name2,
                "test_seed": self.seed2,
                "metric": self.metric,
                "net_cfg": self.config["net"],
                "train_cfg": self.train_cfg,
                "num_samples": [self.train_cfg.batch_size, self.num_train_samples],
                "num_test_samples": self.train_cfg.batch_size,
                "only_continuations": self.only_continuations,
            }
            epsilons, true_epsilon, std_epsilon = self.calibration_strategy.calculate_epsilons(
                self.directory, self.num_train_samples, **calibration_cfg
            )

            self.logger.info(f"Calibrated epsilons: {epsilons}.")

            if not calibrate_only:
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

        plot_calibrated_detection_rate(
            true_epsilon,
            std_epsilon,
            self.model_name1,
            self.seed1,
            self.model_name2,
            self.seed2,
            result_file=epsilon_path,
            draw_in_first_checkpoint=False,
            draw_in_lowest_and_highest=True,
        )


def eval_model(
    config,
    model_id: Optional[int] = None,
    hf_prefix: Optional[str] = None,
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    use_wandb: Optional[int] = None,
):
    """ """
    use_wandb = use_wandb if use_wandb is not None else config["logging"]["use_wandb"]

    project_name = "continuations"

    if use_wandb:
        wandb.init(
            project=project_name,
            entity=config["logging"]["entity"],
            name=create_run_string(),
            config=config,
        )

    if model_id:
        config["tau1"]["model_id"] = model_id

    if hf_prefix:
        config["tau1"]["hf_prefix"] = hf_prefix

    generate_on_dataset(
        config["metric"]["dataset_name"],
        config["tau1"],
        config["eval"]["num_samples"] if not num_samples else num_samples,
        batch_size=config["eval"]["batch_size"] if not batch_size else batch_size,
        use_wandb=use_wandb,
        seed=config["tau1"]["gen_seed"],
        metric=config["metric"]["metric"],
    )

    if use_wandb:
        wandb.finish()
