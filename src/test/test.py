import os
import importlib
import logging
import pandas as pd
import re
import sys
import time
import wandb

from abc import ABC, abstractmethod
from omegaconf import OmegaConf
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple

# Add paths to sys.path if not already present
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# imports from other scripts
from logging_config import setup_logging

from configs.experiment_config import (
    TestConfig,
)

from src.test.calibration_strategies import (
    CalibrationStrategy,
)
from src.test.evaltrainer import (
    OfflineTrainer,
    OfflineTrainerCombined,
)
from src.test.preprocessing import (
    create_folds_from_evaluations,
)

from src.analysis.nn_distance import (
    CMLP,
    OptCMLP,
    HighCapacityCMLP,
)
from src.analysis.analyze import (
    get_distance_scores,
    get_mean_and_std_for_nn_distance,
)
from src.analysis.plot import (
    distance_box_plot,
    plot_calibrated_detection_rate,
)

from src.utils.utils import create_run_string, cleanup_files

from src.base.experiment_base import ExperimentBase

orig_models = importlib.import_module(
    "deep-anytime-testing.models.mlp",
    package="deep-anytime-testing",
)
MLP = getattr(orig_models, "MLP")


class Test(ExperimentBase):
    """ """

    def __init__(self, config: TestConfig):
        super().__init__(config)

    def _calculate_dependent_attributes(self):
        """ """
        self.logging_cfg = self.config.logging
        self.model1_cfg = self.config.model1
        self.model2_cfg = self.config.model2
        self.metric_cfg = self.config.metric
        self.test_params_cfg = self.config.test_params
        self.train_cfg = self.config.training
        self.net_cfg = self.config.net
        self.storing_cfg = self.config.storing

        dir_prefix = (
            self.config.storing.dir_prefix
            if self.config.storing.dir_prefix is not None
            else str(self.metric_cfg.metric)
        )
        self.test_dir = self.SCRIPT_DIR / dir_prefix / self.storing_cfg.test_dir
        self.score_dir = (
            self.SCRIPT_DIR / dir_prefix / self.storing_cfg.score_dir
        )
        self.gen_dir = (
            self.SCRIPT_DIR / dir_prefix / self.storing_cfg.output_dir
        )
        self.plot_dir = self.SCRIPT_DIR / dir_prefix / self.storing_cfg.plot_dir

        # # initialize instance parameters to None
        self.model_name1 = self.model1_cfg.model_id
        self.model_name2 = self.model2_cfg.model_id
        self.seed1 = self.model1_cfg.gen_seed
        self.seed2 = self.model2_cfg.gen_seed
        # # self.directory = None

        self.only_continuations = self.test_params_cfg.only_continuations

        self.fold_size = self.test_params_cfg.fold_size
        self.epsilon = self.test_params_cfg.epsilon

        # logger setup
        if not hasattr(self, "logger"):
            self.setup_logger(
                self.model_name1,
                self.seed1,
                self.model_name2,
                self.seed2,
                self.fold_size,
                self.epsilon)

class AuditingTest(Test):
    """ """

    FOLD_PATTERN = r"_fold_(\d+)\.json$"

    def __init__(
        self,
        config: TestConfig):
        
        super().__init__(config)
        self.track_c2st = self.config.test_params.track_c2st
        self.betting_network_type = self.config.net.model_type
        self.noise = self.test_params_cfg.noise 
        

    def initialize_wandb(self, tags: List[str] = ["Kfold_Auditing_Test"]):
        """ """
        super().initialize_wandb(tags=tags)

    def _initialize_network(self, for_c2st=False):
        """Initialize network with appropriate output size and input size

        Args:
            for_c2st (bool): If True, initialize network for C2ST (input_size=2, output_size=2)
                            If False, initialize network for DAVT (input_size=1, output_size=1)
        """
        
        # TODO: make this dynamic
        input_size = 2 if for_c2st else self.net_cfg.input_size
        output_size = 2 if for_c2st else 1

        if for_c2st:
            return MLP(
                input_size,
                self.net_cfg.hidden_layer_size,
                output_size,
                self.net_cfg.layer_norm,
                self.net_cfg.drop_out,
                self.net_cfg.drop_out_p,  
                self.net_cfg.bias,
            )

        else:
            if self.betting_network_type.lower() == "highcapacitycmlp":
                return HighCapacityCMLP(
                    input_size,  # 2 for C2ST, 1 for DAVT
                    self.net_cfg.hidden_layer_size,
                    output_size,  # 2 for C2ST, 1 for DAVT
                    self.net_cfg.layer_norm,
                    self.net_cfg.drop_out,
                    self.net_cfg.drop_out_p,
                    self.net_cfg.bias,
                    self.net_cfg.num_layers,
                    self.net_cfg.batch_size,
                )
            elif self.betting_network_type.lower() == "cmlp":
                return CMLP(
                    input_size,  # 2 for C2ST, 1 for DAVT
                    self.net_cfg.hidden_layer_size,
                    output_size,  # 2 for C2ST, 1 for DAVT
                    self.net_cfg.layer_norm,
                    self.net_cfg.drop_out,
                    self.net_cfg.drop_out_p,
                    self.net_cfg.bias,
                )

    def davtt(self, fold_num: int):
        """
        Deep anytime-valid tolerance test

        Args:
            fold_num: int
                The fold number to run the test on.
        """
        # Define network(s) for betting score
        betting_net_davt = self._initialize_network(for_c2st=False)

        if self.track_c2st:
            # Initialize second network for C2ST if needed
            betting_net_c2st = self._initialize_network(
                for_c2st=True
            )  # Use output_size=2 for C2ST
            trainer = OfflineTrainerCombined(
                self.train_cfg,
                betting_net_davt,
                betting_net_c2st,
                self.model_name1,
                self.seed1,
                self.model_name2,
                self.seed2,
                metric=self.metric_cfg.metric,
                use_wandb=self.logging_cfg.use_wandb,
                fold_num=fold_num,
                epsilon=self.epsilon,
                test_dir=self.test_dir,
                score_dir=self.score_dir,
                gen_dir=self.gen_dir,
                only_continuations=self.only_continuations,
                noise=self.noise,
            )
        else:
            trainer = OfflineTrainer(
                self.train_cfg,
                betting_net_davt,
                self.model_name1,
                self.seed1,
                self.model_name2,
                self.seed2,
                metric=self.metric_cfg.metric,
                use_wandb=self.logging_cfg.use_wandb,
                fold_num=fold_num,
                epsilon=self.epsilon,
                test_dir=self.test_dir,
                score_dir=self.score_dir,
                gen_dir=self.gen_dir,
                only_continuations=self.only_continuations,
                noise=self.noise,
            )

        return trainer.train()

    def kfold_davtt(self):
        """ """
        cont_string = "_continuations" if self.only_continuations else ""
        noise_string = f"_noise_{self.noise}" if self.noise > 0 else ""
        c2st_string = "_c2st" if self.track_c2st else ""

        file_path = (
            Path(self.directory)
            / f"kfold_test_results{cont_string}_{self.fold_size}_epsilon_{self.epsilon}{noise_string}{c2st_string}.csv"
        )
        stat_file_path = (
            Path(self.directory)
            / f"kfold_test_stats{cont_string}_{self.fold_size}_epsilon_{self.epsilon}{noise_string}{c2st_string}.csv"
        )

        if Path(file_path).exists() and not self.test_params_cfg.overwrite:
            self.logger.info(
                f"Skipping test as results file {file_path} already exists."
            )

            df = pd.read_csv(file_path)

            # calculate positive test rate
            test_positive_per_fold = df.groupby("fold_number")[
                "test_positive"
            ].max()
            positive_rate = test_positive_per_fold.sum() / len(
                test_positive_per_fold
            )

        else:
            self.logger.info(
                f"Running test for {self.model_name1}_{self.seed1} and {self.model_name2}_{self.seed2}."
            )
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
                metric=self.metric_cfg.metric,
                fold_size=self.fold_size,
                overwrite=self.test_params_cfg.overwrite,
                score_dir=self.score_dir,
                gen_dir=self.gen_dir,
                test_dir=self.test_dir,
                only_continuations=self.only_continuations,
                noise=self.noise,
            )

            for file_name in os.listdir(self.directory):
                match = re.search(self.FOLD_PATTERN, file_name)
                if match:
                    fold_number = int(match.group(1))
                    folds.append(fold_number)

            folds.sort()

            end = time.time()
            self.logger.info(
                f"We have {len(folds)} folds. The whole initialization took {round(end - start, 3)} seconds."
            )

            if self.logging_cfg.use_wandb:
                wandb.config.update({"total_num_folds": folds})

            # Iterate over the folds and call test
            all_folds_data = pd.DataFrame()
            all_folds_stats = pd.DataFrame()

            for fold_num in folds:
                self.logger.info(
                    f"Now starting experiment for fold {fold_num}."
                )
                data, test_positive, stat_df = self.davtt(fold_num)
                all_folds_data = pd.concat(
                    [all_folds_data, data],
                    ignore_index=True,
                )
                if stat_df is not None:
                    all_folds_stats = pd.concat(
                        [all_folds_stats, stat_df],
                        ignore_index=True,
                    )
                if test_positive:
                    sum_positive += 1

            all_folds_data.to_csv(file_path, index=False)
            positive_rate = sum_positive / len(folds)

            if all_folds_stats.shape[0] > 0:
                all_folds_stats.to_csv(stat_file_path, index=False)

        cleanup_files(
            self.directory,
            f"*scores{noise_string}_fold*.json",
        )

        self.logger.info(
            f"Positive tests: {positive_rate}, {round(positive_rate * 100, 2)}%."
        )

        return positive_rate

    def analyze_and_plot_distance(self):
        """ """

        if self.test_params_cfg.analysis_params.num_samples == -1:
            num_train_samples = (
                self.fold_size // self.train_cfg.batch_size
            ) * self.train_cfg.batch_size - self.train_cfg.batch_size

        else:
            num_train_samples = self.test_params_cfg.analysis_params.num_samples

        num_runs = self.test_params_cfg.analysis_params.num_runs

        noise_string = f"_noise_{self.noise}" if self.noise > 0 else ""

        dist_path = (
            Path(self.directory)
            / f"distance_scores_{num_train_samples}_{num_runs}{noise_string}.csv"
        )
        if dist_path.exists():
            self.logger.info(
                f"Skipping distance analysis as results file {dist_path} already exists."
            )
            distance_df = pd.read_csv(dist_path)
        else:
            distance_df = get_distance_scores(
                model_name1=self.model_name1,
                seed1=self.seed1,
                seed2=self.seed2,
                model_name2=self.model_name2,
                metric=self.metric_cfg.metric,
                num_runs=num_runs,
                net_cfg=self.net_cfg.to_dict(),
                train_cfg=self.train_cfg,
                num_samples=[
                    self.train_cfg.batch_size,
                    num_train_samples,
                ],
                num_test_samples=self.train_cfg.batch_size,
                only_continuations=self.only_continuations,
                score_dir=self.score_dir,
                noise=self.noise,
            )

            distance_df.to_csv(dist_path, index=False)
            self.logger.info(f"Distance analysis results saved to {dist_path}.")

        mean_nn_distance, std_nn_distance = get_mean_and_std_for_nn_distance(
            distance_df
        )
        self.logger.info(
            f"Average nn distance: {mean_nn_distance}, std: {std_nn_distance}"
        )
        self.logger.info(
            f"Wasserstein distance: {distance_df['Wasserstein_comparison'].mean()}"
        )

        if self.logging_cfg.use_wandb:
            wandb.log(
                {
                    "average_nn_distance": distance_df["NeuralNet"].mean(),
                    "std_nn_distance": distance_df["NeuralNet"].std(),
                    "wasserstein_distance": distance_df[
                        "Wasserstein_comparison"
                    ].mean(),
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
                metric=self.metric_cfg.metric,
            )

        except FileNotFoundError as e:
            self.logger.error(f"Error plotting distance box plot: {e}")

    def run(
        self,
        overrides: Optional[Dict] = None
    ):
        """ """
        if overrides:
            self._apply_overrides(overrides)

        self.directory = f"{self.test_dir}/{self.model_name1}_{self.seed1}_{self.model_name2}_{self.seed2}"

        # Set logger tag based on whether analysis will be performed
        logger_tag = "auditing_test_and_analyze" if self.test_params_cfg.analysis_params else "auditing_test"
        self.setup_logger(
            self.model_name1,
            self.seed1,
            self.model_name2,
            self.seed2,
            self.fold_size,
            self.epsilon,
            tag=logger_tag
        )

        if self.test_params_cfg.run_davtt:
            power = self.kfold_davtt()

        if self.test_params_cfg.analyze_distance:
            self.analyze_and_plot_distance()

        if self.logging_cfg.use_wandb:
            wandb.finish()

        if self.test_params_cfg.run_davtt:
            return power


class CalibratedAuditingTest(AuditingTest):
    def __init__(
        self,
        config: Dict,
        train_cfg: Dict,
        dir_prefix: str,
        eps_strategy: CalibrationStrategy,
        betting_network=CMLP,
        overwrite: bool = False,
        use_wandb: Optional[bool] = None,
        only_continuations: bool = True,
        noise: float = 0,
        track_c2st: bool = False,
        betting_network_type: str = "CMLP",
    ):
        super().__init__(
            config,
            train_cfg,
            dir_prefix,
            betting_network=betting_network,
            overwrite=overwrite,
            use_wandb=use_wandb,
            only_continuations=only_continuations,
            noise=noise,
            track_c2st=track_c2st,
            betting_network_type=betting_network_type,
        )
        self.eps_strategy = eps_strategy
        self.power_dict = {}
        self.num_runs = self.eps_strategy.num_runs

    def setup_logger(self, tag: str = "test_results"):
        """ """
        setup_logging(
            self.model_name1,
            self.seed1,
            self.model_name2,
            self.seed2,
            self.fold_size,
            tag=tag,
            quiet=self.config["logging"].get("quiet", True),
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

        self.model_name1 = (
            model_name1 if model_name1 else self.config["tau1"]["model_id"]
        )
        self.seed1 = seed1 if seed1 else self.config["tau1"]["gen_seed"]
        self.model_name2 = (
            model_name2 if model_name2 else self.config["tau2"]["model_id"]
        )
        self.seed2 = seed2 if seed2 else self.config["tau2"]["gen_seed"]
        self.fold_size = (
            fold_size if fold_size else self.config["test_params"]["fold_size"]
        )

        self.directory = (
            self.test_dir
            / f"{self.model_name1}_{self.seed1}_{self.model_name2}_{self.seed2}"
        )
        self.directory.mkdir(parents=True, exist_ok=True)

        cont_string = "_continuations" if self.only_continuations else ""
        noise_string = f"_noise_{self.noise}" if self.noise > 0 else ""
        c2st_string = "_c2st" if self.track_c2st else ""

        self.setup_logger(
            tag="calibrated_test_results"
            if not calibrate_only
            else "calbrations"
        )

        num_train_samples = (
            self.fold_size // self.train_cfg.batch_size
        ) * self.train_cfg.batch_size - self.train_cfg.batch_size

        epsilon_path = (
            self.directory
            / f"power_over_epsilon{cont_string}_{num_train_samples}_{self.num_runs}{noise_string}{c2st_string}.csv"
        )

        if epsilon_path.exists() and not self.overwrite:
            self.logger.info(
                f"Calibrated testing results already exist in {epsilon_path}."
            )
            dist_path = (
                Path(self.directory)
                / f"distance_scores_{self.num_train_samples}_{self.num_runs}{noise_string}{c2st_string}.csv"
            )
            try:
                distance_df = pd.read_csv(dist_path)
                true_epsilon, std_epsilon = get_mean_and_std_for_nn_distance(
                    distance_df
                )
            except FileNotFoundError:
                self.logger.error(
                    f"Distance analysis results file {dist_path} not found."
                )

        else:
            self.eps_strategy.attach_logger(self.logger)
            dist_cfg = {
                "metric": self.metric,
                "net_cfg": self.config["net"],
                "train_cfg": self.train_cfg,
                "num_test_samples": self.train_cfg.batch_size,
                "only_continuations": self.only_continuations,
                "noise": self.noise,
            }

            epsilons, true_epsilon, std_epsilon = (
                self.eps_strategy.calculate_epsilons(
                    self.model_name1,
                    self.seed1,
                    self.model_name2,
                    self.seed2,
                    [
                        self.train_cfg.batch_size,
                        num_train_samples,
                    ],
                    self.dir_prefix,
                    dist_cfg,
                )
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

                power_df = pd.DataFrame(
                    power_dict.items(),
                    columns=["epsilon", "power"],
                )
                power_df.to_csv(epsilon_path, index=False)
                self.logger.info(
                    f"Calibrated testing results saved to {epsilon_path}."
                )

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
            noise=self.noise,
            # overwrite=True,
        )
