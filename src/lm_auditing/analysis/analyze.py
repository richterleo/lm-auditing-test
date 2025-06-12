import json
import logging
import numpy as np
import pandas as pd
import random
import sys

from copy import deepcopy
from omegaconf import DictConfig
from pathlib import Path
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from typing import Union, List, Optional, Tuple, Dict, Any

from lm_auditing.utils.utils import load_config
from lm_auditing.analysis.distance import empirical_wasserstein_distance_p1, NeuralNetDistance

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parents[3]


def extract_data_for_models(
    model_name1: str,
    seed1: str,
    seed2: str,
    model_name2: Optional[str] = None,
    checkpoint: Optional[str] = None,
    checkpoint_base_name: Optional[str] = None,
    fold_size: int = 2000,
    test_dir: str = "test_outputs",
    epsilon: float = 0,
    only_continuations: bool = True,
    metric: str = "perspective",
    dir_prefix: Optional[str] = None,
    noise: float = 0,
    extract_stats: bool = False,
    c2st: bool = False,
    temp1: Optional[float] = None,
    tp1: Optional[float] = None,
    mnt1: Optional[float] = None,
    temp2: Optional[float] = None,
    tp2: Optional[float] = None,
    mnt2: Optional[float] = None,
):
    """ """
    assert model_name2 or (checkpoint and checkpoint_base_name), (
        "Either model_name2 or checkpoint and checkpoint_base_name must be provided"
    )

    if dir_prefix is None:
        dir_prefix = metric
    test_dir = SCRIPT_DIR / dir_prefix / test_dir

    continuation_str = "_continuations" if only_continuations else ""
    noise_string = f"_noise_{noise}" if noise > 0 else ""
    c2st_string = "_c2st" if c2st else ""

    sampling_string1 = ""
    if temp1:
        sampling_string1 += f"_temp{temp1}"
    if tp1:
        sampling_string1 += f"_tp{tp1}"
    if mnt1:
        sampling_string1 += f"_mnt{mnt1}"

    sampling_string2 = ""
    if temp2:
        sampling_string2 += f"_temp{temp2}"
    if tp2:
        sampling_string2 += f"_tp{tp2}"
    if mnt2:
        sampling_string2 += f"_mnt{mnt2}"

    if model_name2:
        base_path = f"{test_dir}/{model_name1}_{seed1}{sampling_string1}_{model_name2}_{seed2}{sampling_string2}"
    else:
        base_path = f"{test_dir}/{model_name1}_{seed1}{sampling_string1}_{checkpoint_base_name}{checkpoint}_{seed2}{sampling_string2}"

    # next two lines are legacy code
    if fold_size == 4000:
        file_path = f"{base_path}/kfold_test_results{continuation_str}_epsilon_{epsilon}{noise_string}{c2st_string}.csv"
        if not Path(file_path).exists():
            file_path = f"{base_path}/kfold_test_results{continuation_str}_{fold_size}_epsilon_{epsilon}{noise_string}{c2st_string}.csv"
    else:
        file_path = f"{base_path}/kfold_test_results{continuation_str}_{fold_size}_epsilon_{epsilon}{noise_string}{c2st_string}.csv"

    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"File path we are checking: {file_path}")

    data = pd.read_csv(file_path)
    if c2st:
        # Group by sequence and calculate means, handling NaN values for time and epochs
        test_data = data.groupby("sequence").agg(
            {
                "samples": lambda x: int(x[~np.isnan(x)].mean()),
                "test_positive_c2st": "mean",
                "test_positive_davt": "mean",
                "epochs_davt": lambda x: x[~np.isnan(x)].mean(),
                "epochs_c2st": lambda x: x[~np.isnan(x)].mean(),
                "time_davt": lambda x: x[~np.isnan(x)].mean(),
                "time_c2st": lambda x: x[~np.isnan(x)].mean(),
            }
        )

    else:
        # Group by sequence and calculate means, handling NaN values for time and epochs
        test_data = data.groupby("sequence").agg(
            {
                "samples": lambda x: int(x[~np.isnan(x)].mean()),
                "test_positive": "mean",
                "epochs": lambda x: x[~np.isnan(x)].mean(),
                "time": lambda x: x[~np.isnan(x)].mean(),
            }
        )

    logger.info(f"Number of folds: {data['fold_number'].max()}")

    if extract_stats:
        file_path = f"{base_path}/kfold_test_stats{continuation_str}_{fold_size}_epsilon_{epsilon}{noise_string}{c2st_string}.csv"
        df = pd.read_csv(file_path)

        # Initialize ks_test_positive column
        df["ks_test_positive"] = 0

        # Process each fold separately
        for fold in df["fold_number"].unique():
            fold_mask = df["fold_number"] == fold
            fold_data = df[fold_mask].copy()

            # Sort by sequence to ensure proper order
            fold_data = fold_data.sort_values("sequence")

            # Track if test has been triggered
            test_triggered = False

            # Process each sequence in order
            for idx, row in fold_data.iterrows():
                if test_triggered or row["ks_p-value"] < 0.05:
                    test_triggered = True
                    df.loc[idx, "ks_test_positive"] = 1

        # Calculate mean statistics per sequence across all folds
        sequence_stats = (
            df.groupby("sequence")
            .agg(
                {
                    "mean1": "mean",
                    "mean2": "mean",
                    "ws": "mean",
                    "std1": "mean",
                    "std2": "mean",
                    "ks_positive_tests": "mean",  # Now averaging the binary test results
                }
            )
            .reset_index()
        )

        # Add these statistics to the main dataframe
        test_data = test_data.merge(sequence_stats, on="sequence", how="left")

        # Count false positives (folds where test was triggered)
        false_positives = df.groupby("fold_number")["ks_test_positive"].max().sum()
        logger.info(
            f"Number of false positives for Kolmogorov-Smirnov test: {false_positives} after {df['sequence'].max()} sequences."
        )

    return test_data


def get_power_over_sequences(
    base_model_name: str,
    base_model_seed: str,
    seeds: Union[str, List[str]],
    checkpoints: Optional[Union[str, List[str]]] = None,
    checkpoint_base_name: Optional[str] = "Llama-3-8B-ckpt",
    model_names: Optional[Union[str, List[str]]] = None,
    fold_size: int = 2000,
    only_continuations: bool = True,
    epsilon: float = 0,
    test_dir: str = "test_outputs",
    dir_prefix: Optional[str] = None,
    metric: str = "perspective",
    noise: float = 0,
    extract_stats: bool = False,
    c2st: bool = False,
    calc_distance: bool = False,
    rank: bool = False,
    distance_measure: Optional[str] = "Wasserstein",
    num_runs_distance: Optional[int] = 1,
    base_model_temp: Optional[float] = None,
    base_model_tp: Optional[float] = None,
    base_model_mnt: Optional[float] = None,
    model_temps: Optional[List[float]] = None,
    model_tps: Optional[List[float]] = None,
    model_mnts: Optional[List[float]] = None,
):
    """
    Get power over sequences for either model comparisons or checkpoint comparisons.
    Can optionally rank results by distance measure.

    Args:
        base_model_name: Name of the base model
        base_model_seed: Seed for base model
        seeds: Seeds for comparison models/checkpoints
        model_names2: Optional list of model names to compare against
        checkpoints: Optional list of checkpoints to compare against
        checkpoint_base_name: Base name for checkpoint models
        fold_size: Size of each fold
        only_continuations: Whether to use only continuations
        epsilon: Epsilon value(s) for testing
        test_dir: Directory containing test outputs
        dir_prefix: Optional prefix for directories
        metric: Metric used for testing
        noise: Amount of noise added
        rank: Whether to rank results by distance measure
        distance_measure: Distance measure to use for ranking
        num_runs_distance: Number of runs for distance calculation

    Returns:
        DataFrame containing results
    """
    if model_names is not None and checkpoints is not None:
        raise ValueError("Cannot specify both model_names2 and checkpoints")
    if model_names is None and checkpoints is None:
        raise ValueError("Must specify either model_names2 or checkpoints")

    # Convert single values to lists
    if not isinstance(seeds, list):
        seeds = [seeds]

    if model_names is not None and not isinstance(model_names, list):
        model_names = [model_names]

    if checkpoints is not None and not isinstance(checkpoints, list):
        checkpoints = [checkpoints]

    if not isinstance(epsilon, list):
        epsilons = [epsilon] * len(seeds)
    else:
        epsilons = epsilon

    if not isinstance(model_temps, list):
        model_temps = [model_temps] * len(seeds)
    if not isinstance(model_tps, list):
        model_tps = [model_tps] * len(seeds)
    if not isinstance(model_mnts, list):
        model_mnts = [model_mnts] * len(seeds)

    result_dfs = []

    # Process either checkpoints or model names
    items_to_process = checkpoints if checkpoints is not None else model_names

    for item, seed, epsilon, temp2, tp2, mnt2 in zip(
        items_to_process,
        seeds,
        epsilons,
        model_temps,
        model_tps,
        model_mnts,
    ):
        logger.info(
            f"Base_model: {base_model_name}, base_model_seed: {base_model_seed}, "
            f"{'checkpoint: ' + checkpoint_base_name + str(item) if checkpoints else 'model_name2: ' + item}, "
            f"seed: {seed}"
        )

        try:
            if calc_distance:
                # Calculate distance if ranking is requested
                dist_df = get_distance_scores(
                    base_model_name,
                    base_model_seed,
                    seed,
                    checkpoint=item if checkpoints else None,
                    checkpoint_base_name=checkpoint_base_name if checkpoints else None,
                    model_name2=item if not checkpoints else None,
                    metric=metric,
                    test_dir=test_dir,
                    dir_prefix=dir_prefix,
                    distance_measures=[distance_measure],
                    num_runs=num_runs_distance,
                    evaluate_wasserstein_on_full=True,
                    only_continuations=only_continuations,
                    noise=noise,
                    temp1=base_model_temp,
                    tp1=base_model_tp,
                    mnt1=base_model_mnt,
                    temp2=temp2,
                    tp2=tp2,
                    mnt2=mnt2,
                )
                dist = dist_df[f"{distance_measure}_full"].mean()

            # Extract power data
            result_df = extract_data_for_models(
                base_model_name,
                base_model_seed,
                seed,
                checkpoint=item if checkpoints else None,
                checkpoint_base_name=checkpoint_base_name if checkpoints else None,
                model_name2=item if not checkpoints else None,
                fold_size=fold_size,
                test_dir=test_dir,
                epsilon=epsilon,
                only_continuations=only_continuations,
                metric=metric,
                dir_prefix=dir_prefix,
                noise=noise,
                extract_stats=extract_stats,
                c2st=c2st,
                temp1=base_model_temp,
                tp1=base_model_tp,
                mnt1=base_model_mnt,
                temp2=temp2,
                tp2=tp2,
                mnt2=mnt2,
            )

            # Add metadata columns
            result_df["model_name1"] = base_model_name
            result_df["seed1"] = base_model_seed
            if checkpoints:
                result_df["checkpoint_base_name"] = checkpoint_base_name
                result_df["checkpoint"] = item
            else:
                result_df["model_name2"] = item
            result_df["seed2"] = seed
            result_df["epsilon"] = epsilon

            if calc_distance:
                result_df[f"{distance_measure.lower()}_distance"] = dist

            result_dfs.append(result_df)

        except FileNotFoundError:
            if checkpoints:
                logger.error(f"File for checkpoint {item} does not exist yet")
            else:
                logger.error(f"File for model {item} does not exist yet")

    final_df = pd.concat(result_dfs, ignore_index=True)

    if rank:
        final_df[f"rank_by_{distance_measure.lower()}_distance"] = (
            final_df[f"{distance_measure.lower()}_distance"].rank(method="dense", ascending=True).astype(int)
        )

    return final_df


def get_distance_scores(
    model_name1: str,
    seed1: int,
    seed2: int,
    checkpoint: Optional[str] = None,
    checkpoint_base_name: Optional[str] = None,
    model_name2: Optional[str] = None,
    metric: str = "perspective",
    distance_measures: List[str] = ["NeuralNet", "Wasserstein"],
    net_cfg: Optional[Dict[str, Any]] = None,
    train_cfg: Optional[DictConfig] = None,
    pre_shuffle: bool = False,
    test_dir: str = "test_outputs",
    dir_prefix: Optional[str] = None,
    random_seed: int = 0,
    num_samples: Union[int, list[int]] = 100000,
    num_test_samples: int = 1000,
    test_split: float = 0.3,
    evaluate_wasserstein_on_full: bool = True,
    evaluate_nn_on_full: bool = False,
    compare_wasserstein: bool = False,
    num_runs: int = 1,
    use_scipy_wasserstein: bool = True,
    only_continuations: bool = True,
    save: bool = False,
    overwrite: bool = False,
    noise: float = 0,
    temp1: Optional[float] = None,
    tp1: Optional[float] = None,
    mnt1: Optional[float] = None,
    temp2: Optional[float] = None,
    tp2: Optional[float] = None,
    mnt2: Optional[float] = None,
    quiet: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """ """
    np.random.seed(random_seed)
    random.seed(random_seed)
    if not (checkpoint and checkpoint_base_name) and not model_name2:
        raise ValueError("Either checkpoint and checkpoint_base_name or model_name2 must be provided")

    cont_string = "continuation_" if only_continuations else ""
    noise_string = f"_noise_{noise}" if noise > 0 else ""

    if not dir_prefix:
        dir_prefix = metric
    score_dir = SCRIPT_DIR / dir_prefix / test_dir

    model_name2 = f"{checkpoint_base_name}{checkpoint}" if checkpoint else model_name2

    sampling_string1 = ""
    sampling_string2 = ""
    if temp1:
        sampling_string1 += f"_temp{temp1}"
    if tp1:
        sampling_string1 += f"_tp{tp1}"
    if mnt1:
        sampling_string1 += f"_mnt{mnt1}"

    if temp2:
        sampling_string2 += f"_temp{temp2}"
    if tp2:
        sampling_string2 += f"_tp{tp2}"
    if mnt2:
        sampling_string2 += f"_mnt{mnt2}"

    score_dir = Path(score_dir) / f"{model_name1}_{seed1}{sampling_string1}_{model_name2}_{seed2}{sampling_string2}"
    score_path = score_dir / f"{cont_string}scores{noise_string}.json"

    if not isinstance(num_samples, list):
        num_samples = [num_samples]

    try:
        with open(score_path, "r") as f:
            data = json.load(f)

        scores1 = data[f"{metric}_scores1"]
        scores2 = data[f"{metric}_scores2"]

        num_samples = [min(num_sample, len(scores1)) for num_sample in num_samples]

        dist_data = []

        if evaluate_wasserstein_on_full:
            if "Wasserstein" in distance_measures:
                wasserstein_dist_dict = {"num_samples": len(scores1)}
                if use_scipy_wasserstein:
                    wasserstein_dist_dict["Wasserstein_full"] = wasserstein_distance(scores1, scores2)
                else:
                    wasserstein_dist_dict["Wasserstein_full"] = empirical_wasserstein_distance_p1(scores1, scores2)
        if evaluate_nn_on_full:
            if "NeuralNet" in distance_measures:
                assert net_cfg, "net_dict must be provided for neuralnet distance"
                assert train_cfg, "train_cfg must be provided for neuralnet distance"

                shuffled_scores1, shuffled_scores2 = shuffle(
                    scores1,
                    scores2,
                    random_state=random_seed,
                )

                kf = KFold(n_splits=num_runs, shuffle=False)

                for train_index, _ in tqdm(kf.split(shuffled_scores1), desc="Training neural net distance", disable=quiet):
                    fold_scores1 = [shuffled_scores1[i] for i in train_index]
                    fold_scores2 = [shuffled_scores2[i] for i in train_index]

                    # Only keep folds that are of equal length
                    if len(fold_scores1) == len(scores1) // num_runs:
                        # Randomly split the fold into test and train data
                        fold_num_test_samples = int(len(fold_scores1) * test_split)
                        dist_dict = {"num_train_samples": len(fold_scores1) - fold_num_test_samples}
                        dist_dict["num_test_samples"] = fold_num_test_samples

                        indices = np.arange(len(fold_scores1))
                        np.random.shuffle(indices)

                        test_indices = indices[:fold_num_test_samples]
                        train_indices = indices[fold_num_test_samples:]

                        train_scores1 = [fold_scores1[i] for i in train_indices]
                        train_scores2 = [fold_scores2[i] for i in train_indices]
                        test_scores1 = [fold_scores1[i] for i in test_indices]
                        test_scores2 = [fold_scores2[i] for i in test_indices]

                        logger.info(f"Training neural net distance on {len(train_scores1)} samples.")

                        # Rest of the code for each fold...
                        if pre_shuffle:
                            neural_net_distance_shuffled = NeuralNetDistance(
                                net_cfg,
                                deepcopy(train_scores1),
                                deepcopy(train_scores2),
                                deepcopy(test_scores1),
                                deepcopy(test_scores2),
                                train_cfg,
                                pre_shuffle=pre_shuffle,
                                random_seed=random_seed,
                                quiet=quiet,
                            )
                            dist_dict["NeuralNet_unpaired"] = neural_net_distance_shuffled.train().item()
                        neural_net_distance = NeuralNetDistance(
                            net_cfg,
                            train_scores1,
                            train_scores2,
                            test_scores1,
                            test_scores2,
                            train_cfg,
                            pre_shuffle=False,
                            random_seed=random_seed,
                            quiet=quiet,
                        )
                        dist_dict["NeuralNet"] = neural_net_distance.train().item()

                        if "Wasserstein" in distance_measures and compare_wasserstein:
                            if use_scipy_wasserstein:
                                dist_dict["Wasserstein_comparison"] = wasserstein_distance(
                                    test_scores1,
                                    test_scores2,
                                )
                            else:
                                dist_dict["Wasserstein_comparison"] = empirical_wasserstein_distance_p1(
                                    test_scores1,
                                    test_scores2,
                                )

                        dist_data.append(dist_dict)

        else:
            if "NeuralNet" in distance_measures:
                num_train_samples_list = []
                if isinstance(num_samples, int):
                    num_samples = [num_samples]

                for num_train_samples in num_samples:
                    logger.info(f"Training neural net distance on {num_train_samples} samples.")
                    for run in tqdm(range(num_runs), desc="Num Runs", disable=quiet):
                        logger.info(f"Run: {run}/{num_runs}")
                        np.random.seed(random_seed + run)
                        random_test_indices = np.random.choice(
                            len(scores1),
                            num_test_samples,
                            replace=False,
                        )

                        dist_dict = {
                            "num_train_samples": num_train_samples,
                            "num_test_samples": num_test_samples,
                        }

                        test_scores1 = [scores1[i] for i in random_test_indices]
                        test_scores2 = [scores2[i] for i in random_test_indices]

                        logger.info(f"Testing neural net distance on {len(test_scores1)} samples.")

                        if "Wasserstein" in distance_measures:
                            if use_scipy_wasserstein:
                                dist_dict["Wasserstein_comparison"] = wasserstein_distance(
                                    test_scores1,
                                    test_scores2,
                                )
                            else:
                                dist_dict["Wasserstein_comparison"] = empirical_wasserstein_distance_p1(
                                    test_scores1,
                                    test_scores2,
                                )

                        train_scores1 = [scores1[i] for i in range(len(scores1)) if i not in random_test_indices]
                        train_scores2 = [scores2[i] for i in range(len(scores2)) if i not in random_test_indices]

                        if num_train_samples > len(train_scores1):
                            logger.warning(
                                f"Number of train samples {num_train_samples} is greater than the number of available samples {len(train_scores1)}. We are training on all available samples."
                            )
                            num_train_samples = len(train_scores1)

                        num_train_samples_list.append(num_train_samples)
                        dist_dict["num_train_samples"] = int(num_train_samples)

                        random_train_indices = np.random.choice(
                            len(train_scores1),
                            num_train_samples,
                            replace=False,
                        )
                        current_train_scores1 = [train_scores1[i] for i in random_train_indices]
                        current_train_scores2 = [train_scores2[i] for i in random_train_indices]

                        logger.info(f"Training neural net distance on {len(current_train_scores1)} samples.")

                        if pre_shuffle:
                            neural_net_distance_shuffled = NeuralNetDistance(
                                net_cfg,
                                deepcopy(current_train_scores1),
                                deepcopy(current_train_scores2),
                                deepcopy(test_scores1),
                                deepcopy(test_scores2),
                                train_cfg,
                                pre_shuffle=pre_shuffle,
                                random_seed=random_seed,
                                quiet=quiet,
                            )
                            dist_dict["NeuralNet_unpaired"] = neural_net_distance_shuffled.train().item()
                        neural_net_distance = NeuralNetDistance(
                            net_cfg,
                            current_train_scores1,
                            current_train_scores2,
                            test_scores1,
                            test_scores2,
                            train_cfg,
                            pre_shuffle=False,
                            random_seed=random_seed,
                            quiet=quiet,
                        )
                        dist_dict["NeuralNet"] = neural_net_distance.train().item()

                        dist_data.append(dist_dict)

        if dist_data:
            dist_df = pd.DataFrame(dist_data)
            if evaluate_wasserstein_on_full:
                dist_df["Wasserstein_full"] = wasserstein_dist_dict["Wasserstein_full"]
                dist_df["total_samples"] = wasserstein_dist_dict["num_samples"]
        else:
            if evaluate_wasserstein_on_full:
                dist_df = pd.DataFrame([wasserstein_dist_dict])

        if save:
            if not score_dir.exists():
                score_dir.mkdir(parents=True, exist_ok=True)

            dist_file = "distance_scores"
            nts_list = list(set(num_train_samples_list))
            for nts in nts_list:
                dist_file += f"_{nts}"
            dist_file += noise_string
            dist_file += f"_{num_runs}.csv"

            dist_path = score_dir / dist_file

            if not dist_path.exists() or overwrite:
                dist_df.to_csv(dist_path, index=False)
            else:
                logger.info(f"File {dist_path} already exists. Use overwrite=True to overwrite it.")

        return dist_df

    except FileNotFoundError:
        if checkpoint:
            logger.error(f"File for checkpoint {checkpoint} does not exist yet")
        else:
            logger.error(f"File for model {model_name2} does not exist yet")


def get_mean_and_std_for_nn_distance(df: pd.DataFrame) -> Tuple[float, float]:
    """"""

    unique_sample_sizes = df["num_train_samples"].unique()

    assert len(unique_sample_sizes) == 2, "Number of unique sample sizes must be 2"

    # Splitting the data into two groups
    group1 = df[df["num_train_samples"] == unique_sample_sizes[0]]["NeuralNet"]
    group2 = df[df["num_train_samples"] == unique_sample_sizes[1]]["NeuralNet"]

    # Calculate the mean for each group
    mean1 = group1.mean()
    mean2 = group2.mean()
    logger.info(f"Mean of group1 with {unique_sample_sizes[0]} samples: {mean1}")
    logger.info(f"Mean of group2 with {unique_sample_sizes[1]} samples: {mean2}")

    # Calculate the variance for each group
    var1 = group1.var()
    var2 = group2.var()

    # Calculate the average mean and std
    # TODO: check if there is a more principled way of calculating this
    avg_mean = (mean1 + mean2) / 2
    avg_variance = (var1 + var2) / 2
    avg_std_via_variance = avg_variance**0.5

    return avg_mean, avg_std_via_variance


def get_power_over_sequences_for_checkpoints_and_folds(
    base_model_name: str,
    base_model_seed: str,
    checkpoints: List[str],
    seeds: List[str],
    checkpoint_base_name: str = "LLama-3-8B-ckpt",
    fold_sizes: List[int] = [1000, 2000, 3000, 4000],
    metric: str = "perspective",
    distance_measure: str = "Wasserstein",
    only_continuations: bool = True,
    epsilon: float = 0,
    num_runs_distance: int = 1,
    test_dir: str = "test_outputs",
    dir_prefix: Optional[str] = None,
    noise: float = 0,
    base_model_temp: Optional[float] = None,
    base_model_tp: Optional[float] = None,
    base_model_mnt: Optional[float] = None,
    model_temps: Optional[List[float]] = None,
    model_tps: Optional[List[float]] = None,
    model_mnts: Optional[List[float]] = None,
):
    """
    This is a wrapper for get_power_over_sequences_for_ranked_checkpoints to use to multiple fold sizes and returns a concatenated dataframe.
    """
    result_dfs = []

    for fold_size in fold_sizes:
        result_dfs.append(
            get_power_over_sequences(
                base_model_name,
                base_model_seed,
                seeds,
                checkpoints,
                checkpoint_base_name=checkpoint_base_name,
                fold_size=fold_size,
                only_continuations=only_continuations,
                epsilon=epsilon,
                test_dir=test_dir,
                dir_prefix=dir_prefix,
                metric=metric,
                noise=noise,
                calc_distance=True,
                distance_measure=distance_measure,
                num_runs_distance=num_runs_distance,
                base_model_temp=base_model_temp,
                base_model_tp=base_model_tp,
                base_model_mnt=base_model_mnt,
                model_temps=model_temps,
                model_tps=model_tps,
                model_mnts=model_mnts,
            )
        )

    result_df = pd.concat(result_dfs)
    return result_df


def extract_power_from_sequence_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """ """

    cols_to_remove = [
        "epochs",
        "time",
        "epochs_c2st",
        "time_c2st",
        "epochs_davt",
        "time_davt",
    ]

    if "checkpoint" in df.columns:
        # Group by checkpoint and take the last row in each group
        grouped = df.groupby("checkpoint", as_index=False).last()
        cols_to_keep = [c for c in grouped.columns if c not in cols_to_remove]
        return grouped[cols_to_keep].reset_index(drop=True)

    # 2) If "checkpoint" is not a column but "model_name2" is
    elif "model_name2" in df.columns:
        # Group by model_name2 and take the last row in each group
        grouped = df.groupby("model_name2", as_index=False).last()
        cols_to_keep = [c for c in grouped.columns if c not in cols_to_remove]
        return grouped[cols_to_keep].reset_index(drop=True)

    # 3) If neither checkpoint nor model_name2 is present
    else:
        cols_to_keep = [c for c in df.columns if c not in cols_to_remove]
        last_line = df.iloc[[-1]][cols_to_keep].copy()
        return last_line.reset_index(drop=True)


def get_alpha_wrapper(
    model_names: Union[str, List[str]],
    seeds1: Union[str, List[str]],
    seeds2: Union[str, List[str]],
    fold_size: int = 4000,
    epsilon: float = 0,
    only_continuations: bool = True,
    test_dir: str = "test_outputs",
    dir_prefix: Optional[str] = None,
    metric: str = "perspective",
    noise: float = 0,
    temps: Optional[List[float]] = None,
    tps: Optional[List[float]] = None,
    mnts: Optional[List[float]] = None,
    extract_stats: bool = False,
    c2st: bool = False,
):
    if not isinstance(model_names, list):
        result_df = get_power_over_sequences(
            model_names,
            seeds1,
            seeds2,
            model_name2=model_names,
            fold_size=fold_size,
            only_continuations=only_continuations,
            epsilon=epsilon,
            test_dir=test_dir,
            dir_prefix=dir_prefix,
            metric=metric,
            noise=noise,
            extract_stats=extract_stats,
            c2st=c2st,
            calc_distance=False,
            base_model_temp=temps,
            base_model_tp=tps,
            base_model_mnt=mnts,
            model_temps=temps,
            model_tps=tps,
            model_mnts=mnts,
        )

    else:
        result_dfs = []
        for model_name, seed1, seed2, temp, tp, mnt in zip(model_names, seeds1, seeds2, temps, tps, mnts):
            result_df = get_power_over_sequences(
                model_name,
                seed1,
                seed2,
                model_name2=model_name,
                fold_size=fold_size,
                only_continuations=only_continuations,
                epsilon=epsilon,
                test_dir=test_dir,
                dir_prefix=dir_prefix,
                metric=metric,
                noise=noise,
                extract_stats=extract_stats,
                c2st=c2st,
                calc_distance=False,
                base_model_temp=temp,
                base_model_tp=tp,
                base_model_mnt=mnt,
                model_temps=temp,
                model_tps=tp,
                model_mnts=mnt,
            )
            result_dfs.append(result_df)

    final_df = pd.concat(result_dfs, ignore_index=True)
    # final_df["model_id"] = final_df["model_name1"]

    return final_df


def get_mean_tox_scores(
    model_names: Optional[None] = None,
    seeds: Optional[None] = None,
    score_dir="model_scores",
    dir_prefix: Optional[str] = None,
    metric="perspective",
    only_on_toxic_prompts=False,
    high_tox_file="high_toxicity_indices.json",
    only_continuations=False,
    diff=False,
    noise: float = 0,
    base_model_temp: Optional[float] = None,
    base_model_tp: Optional[float] = None,
    base_model_mnt: Optional[float] = None,
    model_temps: Optional[List[float]] = None,
    model_tps: Optional[List[float]] = None,
    model_mnts: Optional[List[float]] = None,
):
    if dir_prefix is None:
        dir_prefix = metric

    cont_string = "continuation_" if only_continuations else ""
    noise_string = f"_noise_{noise}" if noise > 0 else ""

    score_dir = SCRIPT_DIR / dir_prefix / score_dir

    if model_names:
        if not isinstance(model_names, list):
            model_names = [model_names]
            seeds = [seeds]
        model_files = [f"{model_name}_{seed}" for model_name, seed in zip(model_names, seeds)]
    else:
        model_files = [f.name for f in score_dir.iterdir() if f.is_dir()]
    all_scores = []

    for model_file in model_files:
        try:
            if not diff:
                score_path = score_dir / model_file / f"{cont_string}scores{noise_string}.json"
                with open(score_path, "r") as f:
                    scores = json.load(f)

                toxic_scores = scores[f"{metric}_scores"]

            else:
                score_path = score_dir / model_file / f"scores{noise_string}.json"
                with open(score_path, "r") as f:
                    scores = json.load(f)

                cont_score_path = score_dir / model_file / f"continuation_scores{noise_string}.json"
                with open(cont_score_path, "r") as f:
                    cont_scores = json.load(f)

                toxic_scores = [
                    scores[f"{metric}_scores"][i] - cont_scores[f"{metric}_scores"][i]
                    for i in range(len(scores[f"{metric}_scores"]))
                ]

            if only_on_toxic_prompts:
                with open(high_tox_file, "r") as f:
                    high_tox_indices = json.load(f)

                toxic_scores = [toxic_scores[i] for i in high_tox_indices]

            mean, std, median = (
                np.nanmean(toxic_scores),
                np.nanstd(toxic_scores),
                np.nanmedian(toxic_scores),
            )
            all_scores.append(
                {
                    "model": model_file,
                    "mean": mean,
                    "std": std,
                    "median": median,
                }
            )

        except FileNotFoundError:
            logger.warning(f"File for model {model_file} does not exist yet")
            continue

    if all_scores:
        if diff:
            file_name = (
                f"mean_{metric}_diff_scores{noise_string}.json"
                if not only_on_toxic_prompts
                else f"mean_{metric}_diff_scores_on_toxic_prompts{noise_string}.json"
            )
        else:
            if only_on_toxic_prompts:
                file_name = f"mean_{metric}_{cont_string}scores_on_toxic_prompts{noise_string}.json"
            else:
                file_name = f"mean_{metric}_{cont_string}scores{noise_string}.json"

        with open(score_dir / file_name, "w") as f:
            json.dump(all_scores, f, indent=4)


def get_neural_net_distance_stats(
    base_model_name: str,
    base_model_seed: str,
    seed2: str,
    model_name2: Optional[str] = None,
    checkpoint: Optional[str] = None,
    checkpoint_base_name: Optional[str] = None,
    num_train_samples: int = 1900,
    num_runs: int = 20,
    metric: str = "perspective",
    dir_prefix: Optional[str] = None,
    folder_addendum: Optional[str] = None,
    noise: float = 0,
) -> Tuple[float, float]:
    """
    Calculate mean and standard deviation of neural net distance from distance scores file.

    Args:
        base_model_name: Name of the base model
        base_model_seed: Seed used for base model
        seed2: Seed for second model
        model_name2: Optional name of second model (if comparing two different models)
        checkpoint: Optional checkpoint number (if comparing checkpoints)
        checkpoint_base_name: Base name for checkpoint model
        num_train_samples: Number of training samples used
        num_test_samples: Number of test samples used
        metric: Metric used (e.g. "perspective")
        dir_prefix: Optional prefix for directory path
        folder_addendum: Optional string to append to folder name

    Returns:
        Tuple[float, float]: Mean and standard deviation of neural net distance
    """
    if dir_prefix is None:
        dir_prefix = metric

    noise_string = f"_noise_{noise}" if noise > 0 else ""
    folder_addendum = f"_{folder_addendum}" if folder_addendum else ""

    if model_name2 is not None:
        model2_name = model_name2
    elif checkpoint is not None:
        model2_name = f"{checkpoint_base_name}{checkpoint}"
    else:
        raise ValueError("Either model_name2 or checkpoint must be provided")

    test_dir = SCRIPT_DIR / dir_prefix / "test_outputs"
    base_path = test_dir / f"{base_model_name}_{base_model_seed}_{model2_name}_{seed2}{folder_addendum}"
    file_path = base_path / f"distance_scores_{num_train_samples}_{num_runs}{noise_string}.csv"

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Calculate mean and std of neural net distance
    mean_distance = df["NeuralNet"].mean()
    std_distance = df["NeuralNet"].std()

    return mean_distance, std_distance


if __name__ == "__main__":
    # get_mean_tox_scores(only_continuations=True, diff=False)

    base_model_name = "Meta-Llama-3-8B-Instruct"
    base_seed = "seed1000"

    model_name1 = "Llama-3-8B-ckpt1"
    model_name2 = "Llama-3-8B-ckpt5"
    model_name3 = "Llama-3-8B-ckpt10"
    model_name4 = "Meta-Llama-3-8B-Instruct-hightemp"
    model_name5 = "Meta-Llama-3-8B-Instruct"
    model_names = [
        model_name1,
        model_name2,
        model_name3,
        model_name4,
        model_name5,
    ]
    seeds = [
        "seed1000",
        "seed1000",
        "seed1000",
        "seed1000",
        "seed2000",
    ]

    num_train_samples = [
        100,
        300,
        1000,
        3000,
        10000,
        30000,
        100000,
    ]

    train_cfg = TrainCfg()
    net_config = {
        "input_size": 1,
        "hidden_layer_size": [32, 32],
        "layer_norm": True,
        "bias": True,
    }

    seeds = [
        "seed2000",
        "seed2000",
        "seed2000",
        "seed2000",
        "seed1000",
        "seed1000",
        "seed1000",
        "seed1000",
        "seed1000",
        "seed1000",
    ]

    checkpoints = [i for i in range(1, int(10))]

    df = extract_data_for_models(
        "Meta-Llama-3-8B-Instruct",
        "seed1000",
        "seed1000",
        model_name2="Llama-3-8B-ckpt5",
        fold_size=2000,
        test_dir="test_outputs",
        epsilon=0,
        only_continuations=True,
        metric="toxicity",
    )

    small_df = extract_power_from_sequence_df(df)
    print(small_df)
