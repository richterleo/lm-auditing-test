import logging
import numpy as np
import pandas as pd
import sys

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple

# # Add paths to sys.path if not already present
# project_root = Path(__file__).resolve().parents[2]
# if str(project_root) not in sys.path:
#     sys.path.append(str(project_root))

# from src.analysis.analyze import get_distance_scores, get_mean_and_std_for_nn_distance

from lm_auditing.analysis.analyze import get_distance_scores, get_mean_and_std_for_nn_distance

SCRIPT_DIR = Path(__file__).resolve().parents[3]


class CalibrationStrategy(ABC):
    def __init__(self, calibration_params: Dict, overwrite: bool = False, test_dir: str = "test_outputs"):
        """
        Initialize the epsilon calculation strategy with any necessary parameters.

        :param kwargs: Dictionary of configuration parameters
        """
        self.epsilon_ticks = calibration_params.get("epsilon_ticks", 10)
        self.num_runs = calibration_params.get("num_runs", 20)
        self.test_dir = test_dir
        self.overwrite = overwrite

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
        model_name1: str,
        seed1: str,
        model_name2: str,
        seed2: str,
        num_train_samples: Union[int, List],
        dir_prefix: str,
        dist_kwargs: Dict,
        # **kwargs,
    ) -> Tuple[float, float]:
        """
        Compute the neural net distance and its standard deviation between the base model and the test model.
        """
        test_result_dir = SCRIPT_DIR / dir_prefix / self.test_dir / f"{model_name1}_{seed1}_{model_name2}_{seed2}"
        test_result_dir.mkdir(parents=True, exist_ok=True)

        noise_string = f"_noise_{dist_kwargs.get('noise', 0)}" if dist_kwargs.get("noise", 0) > 0 else ""

        dist_path = test_result_dir / f"distance_scores_{num_train_samples}_{self.num_runs}{noise_string}.csv"
        if dist_path.exists() and not self.overwrite:
            self.logger.info(f"Loading existing distance analysis from {dist_path}.")
            distance_df = pd.read_csv(dist_path)
        else:
            self.logger.info(
                f"Calculating neural net distance between {model_name1}_{seed1} "
                f"and {model_name2}_{seed2} "
                f"on {num_train_samples} samples for {self.num_runs} runs."
            )
            distance_df = get_distance_scores(
                model_name1,
                seed1,
                seed2,
                model_name2=model_name2,
                num_samples=num_train_samples,
                overwrite=self.overwrite,
                num_runs=self.num_runs,
                dir_prefix=dir_prefix,
                test_dir=self.test_dir,
                **dist_kwargs,
                # **kwargs,
            )
            distance_df.to_csv(dist_path, index=False)
            self.logger.info(f"Distance analysis results saved to {dist_path}.")

        mean_nn_distance, std_nn_distance = get_mean_and_std_for_nn_distance(distance_df)
        self.logger.info(
            f"Mean neural net distance between {model_name1}_{seed1} and {model_name2}_{seed2}: "
            f"{mean_nn_distance}, Std: {std_nn_distance}"
        )

        return mean_nn_distance, std_nn_distance


class DefaultStrategy(CalibrationStrategy):
    def __init__(self, calibration_params: Dict, overwrite: bool = False, test_dir="test_outputs"):
        """
        Initialize the DefaultEpsilonStrategy.

        :param lower_interval_end: Lower bound of the epsilon interval.
        :param upper_interval_end: Upper bound of the epsilon interval.
        :param epsilon_ticks: Number of epsilon values to generate within the interval.
        :param config: Configuration dictionary.
        """
        super().__init__(calibration_params, overwrite=overwrite, test_dir=test_dir)

        self.lower_interval_end = calibration_params.get("lower_interval_end", 0)
        self.upper_interval_end = calibration_params.get("upper_interval_end", 0.2)

    def calculate_epsilons(
        self,
        base_model: str,
        base_seed: str,
        test_model: str,
        test_seed: str,
        num_train_samples: Union[int, List],
        dir_prefix: str,
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
            base_model,
            base_seed,
            test_model,
            test_seed,
            num_train_samples,
            dir_prefix,
            dist_kwargs,
        )

        # Generate epsilons using the interval
        epsilons = np.linspace(self.lower_interval_end, self.upper_interval_end, self.epsilon_ticks).tolist()

        return epsilons, mean_nn_distance, std_nn_distance


class StdStrategy(CalibrationStrategy):
    def __init__(
        self,
        calibration_params: Dict,
        overwrite: bool = False,
        test_dir: str = "test_outputs",
    ):
        super().__init__(calibration_params, overwrite=overwrite, test_dir=test_dir)

        self.multiples_of_epsilon = int(self.epsilon_ticks / 2)
        self.bias = self.calibration_params.get("bias", 0)

    def calculate_epsilons(
        self,
        base_model: str,
        base_seed: str,
        test_model: str,
        test_seed: str,
        num_train_samples: Union[List, int],
        dir_prefix: str,
        dist_kwargs: Dict,
        **kwargs,
    ) -> list:
        """ """
        mean_nn_distance, std_nn_distance = self.compute_distance(
            base_model,
            base_seed,
            test_model,
            test_seed,
            num_train_samples,
            dir_prefix,
            dist_kwargs,
        )

        # Generate epsilons based on multiples of the standard deviation
        epsilons = []
        for i in range(-self.multiples_of_epsilon, self.multiples_of_epsilon + 1):
            epsilon = max(mean_nn_distance + std_nn_distance * i - self.bias, 0)
            epsilons.append(epsilon)
        epsilons = sorted(set(epsilons))

        return epsilons, mean_nn_distance, std_nn_distance


class IntervalStrategy(CalibrationStrategy):
    def __init__(
        self,
        calibration_params: Dict,
        overwrite: bool = False,
        test_dir: str = "test_outputs",
    ):
        super().__init__(calibration_params, overwrite=overwrite, test_dir=test_dir)

        # TODO: add checks that this exists
        # lower epsilon bound
        self.lower_model = calibration_params["lower_model_name"]
        self.lower_seed = calibration_params["lower_seed"]

        # upper epsilon bound
        self.upper_model = calibration_params["upper_model_name"]
        self.upper_seed = calibration_params["upper_seed"]

    def calculate_epsilons(
        self,
        base_model: str,
        base_seed: str,
        test_model: str,
        test_seed: str,
        num_train_samples: Union[int, List],
        dir_prefix: str,
        dist_kwargs: Dict,
        **kwargs,
    ) -> Tuple[List[float], float, float]:
        distances = {}
        stds = {}

        models = [
            ("test", test_model, test_seed),
            ("lower", self.lower_model, self.lower_seed),
            ("upper", self.upper_model, self.upper_seed),
        ]

        for model_type, model, seed in models:
            distances[model_type], stds[model_type] = self.compute_distance(
                base_model, base_seed, model, seed, num_train_samples, dir_prefix, dist_kwargs
            )

        epsilons = np.linspace(distances["lower"], distances["upper"], self.epsilon_ticks).to_list()

        return epsilons, distances["test"], stds["test"]
