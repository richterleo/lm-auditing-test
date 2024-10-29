import numpy as np
import importlib
import logging
import pandas as pd

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple

# imports from other scripts
from arguments import TrainCfg
from logging_config import setup_logging

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


class EpsilonStrategy(ABC):
    def __init__(self, **kwargs):
        """
        Initialize the epsilon calculation strategy with any necessary parameters.

        :param kwargs: Dictionary of configuration parameters
        """
        pass

    def attach_logger(self, logger: logging.Logger):
        self.logger = logger

    @abstractmethod
    def calculate_epsilons(self, **kwargs) -> Tuple[List[float], float, float]:
        """
        Calculate a list of epsilons based on the distance dataframe.

        :param distance_df: DataFrame containing distance scores
        :return: List of calculated epsilons and the mean and standard deviation of the neural net distance
        """
        pass

    def compute_distance(
        self,
        test_result_dir: Union[str, Path],
        num_train_samples: int,
        base_model: str,
        base_seed: str,
        test_model: str,
        test_seed: str,
        dist_kwargs: Dict,
    ) -> Tuple[float, float]:
        """
        Compute the neural net distance and its standard deviation between the base model and the test model.
        """
        dist_path = Path(test_result_dir) / f"distance_scores_{num_train_samples}_{dist_kwargs['num_runs']}.csv"
        if dist_path.exists() and not dist_kwargs.get("overwrite", False):
            self.logger.info(f"Loading existing distance analysis from {dist_path}.")
            distance_df = pd.read_csv(dist_path)
        else:
            self.logger.info(
                f"Calculating neural net distance between {base_model}_{base_seed} "
                f"and {test_model}_{test_seed} "
                f"on {dist_kwargs['num_samples']} samples for {dist_kwargs['num_runs']} runs."
            )
            distance_df = get_distance_scores(
                base_model,
                base_seed,
                test_seed,
                model_name2=test_model,
                **dist_kwargs,
            )
            distance_df.to_csv(dist_path, index=False)
            self.logger.info(f"Distance analysis results saved to {dist_path}.")

        mean_nn_distance, std_nn_distance = get_mean_and_std_for_nn_distance(distance_df)
        self.logger.info(
            f"Mean neural net distance between {base_model}_{base_seed} and {test_model}_{test_seed}: "
            f"{mean_nn_distance}, Std: {std_nn_distance}"
        )

        return mean_nn_distance, std_nn_distance


class DefaultEpsilonStrategy(EpsilonStrategy):
    def __init__(
        self,
        lower_interval_end: float = 0,
        upper_interval_end: float = 0.2,
        epsilon_ticks: int = 20,
    ):
        """
        Initialize the DefaultEpsilonStrategy.

        :param lower_interval_end: Lower bound of the epsilon interval.
        :param upper_interval_end: Upper bound of the epsilon interval.
        :param epsilon_ticks: Number of epsilon values to generate within the interval.
        :param config: Configuration dictionary.
        """
        super().__init__()

        try:
            assert lower_interval_end > 0
            assert upper_interval_end > lower_interval_end
        except AssertionError as e:
            self.logger.error("Invalid epsilon interval bounds.")
            raise e

        self.lower_interval_end = lower_interval_end
        self.upper_interval_end = upper_interval_end
        self.epsilon_ticks = epsilon_ticks

    def calculate_epsilons(
        self,
        test_result_dir: Union[str, Path],
        num_train_samples: int,
        base_model: str,
        base_seed: str,
        test_model: str,
        test_seed: str,
        dist_kwargs: Dict,
        **kwargs,
    ) -> Tuple[List[float], float, float]:
        """
        Calculate a list of epsilons based on a predefined interval and compute
        the neural net distance and its standard deviation between the base model
        and the test model.

        :return: A tuple containing the list of calculated epsilons,
                 mean neural net distance, and standard deviation.
        """
        # Compute the neural net distance between the base model and the test model
        mean_nn_distance, std_nn_distance = self.compute_distance(
            test_result_dir,
            num_train_samples,
            base_model,
            base_seed,
            test_model,
            test_seed,
            dist_kwargs,
        )

        # Generate epsilons using the interval
        epsilons = np.linspace(self.lower_interval_end, self.upper_interval_end, self.epsilon_ticks).tolist()

        return epsilons, mean_nn_distance, std_nn_distance


class StandardDeviationEpsilonStrategy(EpsilonStrategy):
    def __init__(
        self,
        epsilon_ticks: int = 20,
        bias: float = 0,
        num_runs: int = 20,
    ):
        self.multiples_of_epsilon = int(epsilon_ticks / 2)
        self.bias = bias
        self.num_runs = num_runs

    def calculate_epsilons(
        self,
        test_result_dir: str,
        num_train_samples: int,
        base_model: str,
        base_seed: str,
        test_model: str,
        test_seed: str,
        dist_kwargs,
        **kwargs,
    ) -> list:
        """ """
        mean_nn_distance, std_nn_distance = self.compute_distance(
            test_result_dir,
            num_train_samples,
            base_model,
            base_seed,
            test_model,
            test_seed,
            dist_kwargs,
        )

        # Generate epsilons based on multiples of the standard deviation
        epsilons = []
        for i in range(-self.multiples_of_epsilon, self.multiples_of_epsilon + 1):
            epsilon = max(mean_nn_distance + std_nn_distance * i - self.bias, 0)
            epsilons.append(epsilon)
        epsilons = sorted(set(epsilons))

        return epsilons, mean_nn_distance, std_nn_distance


class IntervalEpsilonStrategy(EpsilonStrategy):
    def __init__(
        self,
        lower_model: str,
        lower_seed: int,
        upper_model: str,
        upper_seed: int,
        base_model: Optional[str] = None,
        base_seed: Optional[int] = None,
        overwrite: bool = False,
        epsilon_ticks: Optional[int] = None,
        use_full_ds_for_nn_distance: Optional[bool] = None,
        num_runs: Optional[int] = None,
        config: Optional[Dict] = None,
    ):
        # lower epsilon bound
        self.lower_model = lower_model
        self.lower_seed = lower_seed
        # upper epsilon bound
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

        # number of epsilon values in the interval
        self.epsilon_ticks = epsilon_ticks if epsilon_ticks is not None else config["analysis"]["epsilon_ticks"]

    def calculate_epsilons(self, test_dir, num_train_samples, **distance_score_kwargs) -> list:
        distances = {}
        stds = {}

        base_model = self.base_model if self.base_model else distance_score_kwargs["base_model"]
        base_seed = self.base_seed if self.base_seed else distance_score_kwargs["base_seed"]
        self.logger.info(f"Base model: {base_model}")

        test_model = distance_score_kwargs["test_model"]
        test_seed = distance_score_kwargs["test_seed"]
        self.logger.info(f"Test model: {test_model}")

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
            list(epsilons),
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
