from collections import defaultdict

import pandas as pd
import json
import logging
import numpy as np
import sys
import os

from omegaconf import DictConfig

from copy import deepcopy
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from scipy.stats import skew, wasserstein_distance
from typing import Union, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.distance import (
    empirical_wasserstein_distance_p1,
    NeuralNetDistance,
)
from utils.utils import load_config
from train_cfg import TrainCfg
import random

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)

logger = logging.getLogger(__name__)


def extract_data_for_models(
    model_name1: str,
    seed1: str,
    seed2: str,
    model_name2: Optional[str] = None,
    checkpoint: Optional[str] = None,
    checkpoint_base_name: Optional[str] = None,
    fold_size: int = 4000,
    test_dir: str = "test_outputs",
    epsilon: float = 0,
    only_continuations: bool = True,
):
    assert model_name2 or (
        checkpoint and checkpoint_base_name
    ), "Either model_name2 or checkpoint and checkpoint_base_name must be provided"

    script_dir = os.path.dirname(__file__)
    test_dir = os.path.join(script_dir, "..", test_dir)

    continuation_str = "_continuations" if only_continuations else ""

    if model_name2:
        base_path = f"{test_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}"
    else:
        base_path = f"{test_dir}/{model_name1}_{seed1}_{checkpoint_base_name}{checkpoint}_{seed2}"

    if fold_size == 4000:
        file_path = f"{base_path}/kfold_test_results{continuation_str}_epsilon_{epsilon}.csv"
        if not Path(file_path).exists():
            file_path = f"{base_path}/kfold_test_results{continuation_str}_{fold_size}_epsilon_{epsilon}.csv"
    else:
        file_path = f"{base_path}/kfold_test_results{continuation_str}_{fold_size}_epsilon_{epsilon}.csv"

    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"File path we are checking: {file_path}")

    data = pd.read_csv(file_path)

    logger.info(f"Number of folds: {data['fold_number'].max()}")

    return data

def get_power_over_sequences_from_whole_ds(data: pd.DataFrame, fold_size: int = 4000):
    """ """
    bs = data.loc[0, "samples"]

    max_sequences = (fold_size + bs - 1) // bs
    selected_columns = data[
        [
            "fold_number",
            "sequence",
            "wealth",
            "sequences_until_end_of_experiment",  # TODO: can remove this later, just a sanity check!
            "test_positive",
        ]
    ]

    filtered_df = selected_columns.drop_duplicates(subset=["sequence", "fold_number"])

    num_folds = filtered_df["fold_number"].nunique()
    # Set 'sequence' as the index of the DataFrame
    indexed_df = filtered_df.set_index("sequence")

    unique_fold_numbers = indexed_df["fold_number"].unique()

    # Initialize a dictionary to store the counts
    sequence_counts = {sequence: 0 for sequence in range(max_sequences)}

    # Iterate over each fold number
    for fold in unique_fold_numbers:
        fold_data = indexed_df[indexed_df["fold_number"] == fold]

        for sequence in range(max_sequences):  # sequence_counts.keys()
            if sequence in fold_data.index and fold_data.loc[sequence, "sequences_until_end_of_experiment"] == sequence:
                try:
                    assert (
                        fold_data.loc[sequence, "test_positive"] == 1
                    ), f"Current sequence {sequence} == sequence until end of experiment, but test is not positive"
                except AssertionError as e:
                    logger.error(e)
                sequence_counts[sequence] += 1

            elif sequence not in fold_data.index:
                sequence_counts[sequence] += 1

    # Convert the result to a DataFrame for better visualization
    result_df = pd.DataFrame(list(sequence_counts.items()), columns=["Sequence", "Count"])
    result_df["Power"] = result_df["Count"] / num_folds
    result_df["Samples per Test"] = fold_size
    result_df["Samples"] = result_df["Sequence"] * bs

    result_df.reset_index()

    return result_df


def get_power_over_sequences_for_models_or_checkpoints(
    model_name1,
    seed1,
    seed2,
    model_name2: Optional[str] = None,
    checkpoint: Optional[str] = None,
    checkpoint_base_name: Optional[str] = None,
    fold_size: int = 4000,
    bs: int = 100,
    epsilon: float = 0,
    only_continuations=True,
):
    """ """
    assert model_name2 or (
        checkpoint and checkpoint_base_name
    ), "Either model_name2 or checkpoint and checkpoint_base_name must be provided"

    if model_name2:
        data = extract_data_for_models(
            model_name1, seed1, seed2, model_name2=model_name2, epsilon=epsilon, only_continuations=only_continuations
        )
        result_df = get_power_over_sequences_from_whole_ds(data, fold_size=fold_size)
        result_df["model_name1"] = model_name1
        result_df["seed1"] = seed1
        result_df["model_name2"] = model_name2
        result_df["seed2"] = seed2
        result_df["epsilon"] = epsilon
    else:
        data = extract_data_for_models(
            model_name1,
            seed1,
            seed2,
            checkpoint=checkpoint,
            checkpoint_base_name=checkpoint_base_name,
            fold_size=fold_size,
            only_continuations=only_continuations,
            epsilon=epsilon,
        )
        result_df = get_power_over_sequences_from_whole_ds(data, fold_size)
        result_df["Checkpoint"] = checkpoint
        result_df["epsilon"] = epsilon

    return result_df


def get_matrix_for_models(model_names, seeds, fold_size=4000):
    """ """
    all_scores = []

    for model_name1, seed1 in zip(model_names, seeds):
        for model_name2, seed2 in zip(model_names[1:], seeds[1:]):
            power_df = get_power_over_sequences_for_models_or_checkpoints(
                model_name1, seed1, seed2, model_name2=model_name2, fold_size=fold_size
            )
            all_scores.append(power_df)

    all_scores_df = pd.concat(all_scores, ignore_index=True)
    logger.info(all_scores_df)


def get_power_over_sequences(
    base_model_name: Union[str, List[str]],
    base_model_seed: Union[str, List[str]],
    seeds: Union[str, List[str]],
    checkpoints: Optional[Union[str, List[str]]] = None,
    model_names: Optional[Union[str, List[str]]] = None,
    checkpoint_base_name: str = "Llama-3-8B-ckpt",
    fold_size: int = 4000,
    only_continuations=True,
    epsilon: Union[float, List[float]] = 0,
    bs=100,
):
    use_checkpoints = True

    if model_names:
        use_checkpoints = False
        if not isinstance(model_names, list):
            model_names = [model_names]
    else:
        if not isinstance(checkpoints, list):
            checkpoints = [checkpoints]

    if not isinstance(seeds, list):
        seeds = [seeds]

    epsilons = epsilon * len(seeds)

    result_dfs = []

    if use_checkpoints:
        for checkpoint, seed, epsilon in zip(checkpoints, seeds, epsilons):
            logger.info(
                f"Base_model: {base_model_name}, base_model_seed: {base_model_seed}, checkpoint: {checkpoint_base_name}{checkpoint}, seed: {seed}"
            )
            try:
                result_df = get_power_over_sequences_for_models_or_checkpoints(
                    base_model_name,
                    base_model_seed,
                    seed,
                    checkpoint=checkpoint,
                    checkpoint_base_name=checkpoint_base_name,
                    fold_size=fold_size,
                    only_continuations=only_continuations,
                    epsilon=epsilon,
                    bs=bs,
                )

                result_dfs.append(result_df)

            except FileNotFoundError:
                logger.error(f"File for checkpoint {checkpoint} and seed {seed} does not exist yet")

    else:
        for model_name, seed, epsilon in zip(model_names, seeds, epsilons):
            logger.info(
                f"Base_model: {base_model_name}, base_model_seed: {base_model_seed}, model_name: {model_name}, seed: {seed}"
            )
            try:
                result_df = get_power_over_sequences_for_models_or_checkpoints(
                    base_model_name,
                    base_model_seed,
                    seed,
                    model_name2=model_name,
                    fold_size=fold_size,
                    bs=bs,
                    epsilon=epsilon,
                    only_continuations=only_continuations,
                )

                result_dfs.append(result_df)

            except FileNotFoundError:
                logger.error(f"File for model {model_name} and seed {seed} does not exist yet")

    final_df = pd.concat(result_dfs, ignore_index=True)

    return final_df


def get_distance_scores(
    model_name1: str,
    seed1: int,
    seed2: int,
    checkpoint: Optional[str] = None,
    checkpoint_base_name: Optional[str] = None,
    model_name2: Optional[str] = None,
    metric: str = "perspective",
    distance_measures: list = ["NeuralNet", "Wasserstein"],
    net_cfg: Optional[dict] = None,
    train_cfg: Optional[DictConfig] = None,
    pre_shuffle: bool = False,
    score_dir: str = "test_outputs",
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
    **kwargs,
) -> pd.DataFrame:
    """ """
    np.random.seed(random_seed)
    random.seed(random_seed)
    if not (checkpoint and checkpoint_base_name) and not model_name2:
        raise ValueError("Either checkpoint and checkpoint_base_name or model_name2 must be provided")

    script_dir = os.path.dirname(__file__)
    score_dir = os.path.join(script_dir, "..", score_dir)

    model_name2 = f"{checkpoint_base_name}{checkpoint}" if checkpoint else model_name2

    score_dir = os.path.join(score_dir, f"{model_name1}_{seed1}_{model_name2}_{seed2}")
    score_path = (
        os.path.join(score_dir, f"{metric}_scores.json")
        if not only_continuations
        else os.path.join(score_dir, f"{metric}_continuation_scores.json")
    )

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

                shuffled_scores1, shuffled_scores2 = shuffle(scores1, scores2, random_state=random_seed)

                kf = KFold(n_splits=num_runs, shuffle=False)

                for train_index, _ in kf.split(shuffled_scores1):
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
                        )
                        dist_dict["NeuralNet"] = neural_net_distance.train().item()

                        if "Wasserstein" in distance_measures and compare_wasserstein:
                            if use_scipy_wasserstein:
                                dist_dict["Wasserstein_comparison"] = wasserstein_distance(test_scores1, test_scores2)
                            else:
                                dist_dict["Wasserstein_comparison"] = empirical_wasserstein_distance_p1(
                                    test_scores1, test_scores2
                                )

                        dist_data.append(dist_dict)

        else:
            if "NeuralNet" in distance_measures:
                num_train_samples_list = []
                if isinstance(num_samples, int):
                    num_samples = [num_samples]

                for num_train_samples in num_samples:
                    for run in range(num_runs):
                        logger.info(f"Num runs: {num_runs}, Run: {run}")
                        np.random.seed(random_seed + run)
                        random_test_indices = np.random.choice(len(scores1), num_test_samples, replace=False)

                        dist_dict = {
                            "num_test_samples": num_test_samples,
                        }

                        test_scores1 = [scores1[i] for i in random_test_indices]
                        test_scores2 = [scores2[i] for i in random_test_indices]

                        logger.info(f"Testing neural net distance on {len(test_scores1)} samples.")

                        if "Wasserstein" in distance_measures:
                            if use_scipy_wasserstein:
                                dist_dict["Wasserstein_comparison"] = wasserstein_distance(test_scores1, test_scores2)
                            else:
                                dist_dict["Wasserstein_comparison"] = empirical_wasserstein_distance_p1(
                                    test_scores1, test_scores2
                                )

                        # left over train scores for training the distance network
                        train_scores1 = [scores1[i] for i in range(len(scores1)) if i not in random_test_indices]
                        train_scores2 = [scores2[i] for i in range(len(scores2)) if i not in random_test_indices]

                        if num_train_samples > len(train_scores1):
                            logger.warning(
                                f"Number of train samples {num_train_samples} is greater than the number of available samples {len(train_scores1)}. We are training on all available samples."
                            )
                            num_train_samples = len(train_scores1)

                        print(f"This is num_train_samples: {num_train_samples}")
                        num_train_samples_list.append(num_train_samples)
                        dist_dict["num_train_samples"] = int(num_train_samples)
                        print(f"This is it in the dict: {dist_dict['num_train_samples']}")

                        random_train_indices = np.random.choice(len(train_scores1), num_train_samples, replace=False)
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
            if not os.path.exists(score_dir):
                os.makedirs(score_dir)

            dist_file = f"distance_scores"
            nts_list = list(set(num_train_samples_list))
            for nts in nts_list:
                dist_file += f"_{nts}"
            dist_file += f"_{num_runs}.csv"

            dist_path = Path(score_dir) / dist_file

            if not Path(dist_path).exists() or overwrite:
                dist_df.to_csv(dist_path, index=False)

        return dist_df

    except FileNotFoundError:
        if checkpoint:
            logger.error(f"File for checkpoint {checkpoint} does not exist yet")
        else:
            logger.error(f"File for model {model_name2} does not exist yet")


def get_mean_and_std_for_nn_distance(df):
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


def get_power_over_sequences_for_ranked_checkpoints(
    base_model_name,
    base_model_seed,
    seeds,
    checkpoints,
    checkpoint_base_name="LLama-3-8B-ckpt",
    metric="perspective",
    distance_measure="Wasserstein",
    fold_size=4000,
    num_runs_distance=1,
    only_continuations=True,
    epsilon=0,
):
    if not isinstance(checkpoints, list):
        checkpoints = [checkpoints]
    if not isinstance(seeds, list):
        seeds = [seeds]

    result_dfs = []

    for checkpoint, seed in zip(checkpoints, seeds):
        logger.info(
            f"Base_model: {base_model_name}, base_model_seed: {base_model_seed}, checkpoint: {checkpoint_base_name}{checkpoint}, seed: {seed}"
        )

        dist_df = get_distance_scores(
            base_model_name,
            base_model_seed,
            seed,
            checkpoint=checkpoint,
            checkpoint_base_name=checkpoint_base_name,
            metric=metric,
            distance_measures=[distance_measure],
            num_runs=num_runs_distance,
            evaluate_wasserstein_on_full=True,  # the distance measure is currently hard coded
            only_continuations=only_continuations,
        )
        dist = dist_df[f"{distance_measure}_full"].mean()

        result_df = get_power_over_sequences_for_models_or_checkpoints(
            base_model_name,
            base_model_seed,
            seed,
            checkpoint=checkpoint,
            checkpoint_base_name=checkpoint_base_name,
            fold_size=fold_size,
            epsilon=epsilon,
        )

        result_df[f"Empirical {distance_measure} Distance"] = dist
        result_dfs.append(result_df)

    final_df = pd.concat(result_dfs, ignore_index=True)
    final_df[f"Rank based on {distance_measure} Distance"] = (
        final_df[f"Empirical {distance_measure} Distance"].rank(method="dense", ascending=True).astype(int)
    )

    return final_df


def get_power_over_sequences_for_ranked_checkpoints_wrapper(
    base_model_name,
    base_model_seed,
    checkpoints,
    seeds,
    checkpoint_base_name="LLama-3-8B-ckpt",
    fold_sizes: List[int] = [1000, 2000, 3000, 4000],
    metric="perspective",
    distance_measure="Wasserstein",
    only_continuations=True,
    epsilon=0,
    num_runs_distance=1,
):
    """
    This is a wrapper for get_power_over_sequences_for_ranked_checkpoints to use to multiple fold sizes and returns a concatenated dataframe.
    """
    result_dfs = []

    for fold_size in fold_sizes:
        result_dfs.append(
            get_power_over_sequences_for_ranked_checkpoints(
                base_model_name,
                base_model_seed,
                seeds,
                checkpoints,
                checkpoint_base_name=checkpoint_base_name,
                metric=metric,
                distance_measure=distance_measure,
                fold_size=fold_size,
                epsilon=epsilon,
                only_continuations=only_continuations,
                num_runs_distance=num_runs_distance,
            )
        )

    result_df = pd.concat(result_dfs)
    return result_df


def extract_power_from_sequence_df(
    df: pd.DataFrame,
    distance_measure: Optional[str] = "Wasserstein",
    by_checkpoints=True,
):
    """ """
    cols_to_filter = ["Samples per Test"]

    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 1000)
    pd.set_option("display.width", 1000)

    if by_checkpoints:
        cols_to_filter.append("Checkpoint")
        last_entries = df.groupby(cols_to_filter).last().reset_index()

        cols_to_filter.append("Power")

        if not distance_measure:
            smaller_df = last_entries[cols_to_filter]
        else:
            cols_to_filter.extend(
                [
                    f"Empirical {distance_measure} Distance",
                    f"Rank based on {distance_measure} Distance",
                ]
            )
            smaller_df = last_entries[cols_to_filter]

    else:
        # in this case we just have model1 and model2 combinations
        cols_to_filter.extend(["model_name1", "model_name2", "seed1", "seed2"])

        last_entries = df.groupby(cols_to_filter).last().reset_index()

        cols_to_filter.append("Power")
        if not distance_measure:
            smaller_df = last_entries[cols_to_filter]
        else:
            if "Rank based on Wasserstein Distance" in last_entries.columns:
                cols_to_filter.extend(
                    [
                        f"Empirical {distance_measure} Distance",
                        f"Rank based on {distance_measure} Distance",
                    ]
                )
                smaller_df = last_entries[cols_to_filter]
            else:
                cols_to_filter.append(f"Empirical {distance_measure} Distance")
                smaller_df = last_entries[cols_to_filter]

    return smaller_df


def get_alpha_wrapper(model_names, seeds1, seeds2, fold_size=4000, epsilon=0, only_continuations=True):
    if not isinstance(model_names, list):
        result_df = get_power_over_sequences_for_models_or_checkpoints(
            model_names,
            seeds1,
            seeds2,
            model_name2=model_names,
            fold_size=fold_size,
            epsilon=epsilon,
            only_continuations=only_continuations,
        )

    else:
        result_dfs = []
        for model_name, seed1, seed2 in zip(model_names, seeds1, seeds2):
            result_df = get_power_over_sequences_for_models_or_checkpoints(
                model_name,
                seed1,
                seed2,
                model_name2=model_name,
                fold_size=fold_size,
                epsilon=epsilon,
                only_continuations=only_continuations,
            )
            result_dfs.append(result_df)

    final_df = pd.concat(result_dfs, ignore_index=True)
    final_df["model_id"] = final_df["model_name1"]

    return final_df


def get_mean_tox_scores(
    model_names: Optional[None] = None,
    seeds: Optional[None] = None,
    score_dir="model_scores",
    metric="perspective",
    only_on_toxic_prompts=False,
    high_tox_file="high_toxicity_indices.json",
    only_continuations=False,
    diff=False,
):
    if model_names:
        if not isinstance(model_names, list):
            model_names = [model_names]
            seeds = [seeds]
        model_files = [f"{model_name}_{seed}" for model_name, seed in zip(model_names, seeds)]
    else:
        # model_files = os.listdir(score_dir)
        model_files = [f for f in os.listdir(score_dir) if os.path.isdir(os.path.join(score_dir, f))]

    all_scores = []

    for model_file in model_files:
        try:
            if not diff:
                score_path = (
                    f"{score_dir}/{model_file}/{metric}_scores.json"
                    if not only_continuations
                    else f"{score_dir}/{model_file}/{metric}_continuation_scores.json"
                )
                with open(score_path, "r") as f:
                    scores = json.load(f)

                toxic_scores = scores[f"{metric}_scores"]

            else:
                score_path = f"{score_dir}/{model_file}/{metric}_scores.json"
                with open(score_path, "r") as f:
                    scores = json.load(f)

                cont_score_path = f"{score_dir}/{model_file}/{metric}_continuation_scores.json"
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

            mean, std, median = (np.nanmean(toxic_scores), np.nanstd(toxic_scores), np.nanmedian(toxic_scores))
            all_scores.append({"model": model_file, "mean": mean, "std": std, "median": median})

        except FileNotFoundError:
            logger.warning(f"File for model {model_file} does not exist yet")
            continue

    if all_scores:
        if diff:
            file_name = (
                f"mean_{metric}_diff_scores.json"
                if not only_on_toxic_prompts
                else f"mean_{metric}_diff_scores_on_toxic_prompts.json"
            )
        else:
            if only_on_toxic_prompts:
                file_name = (
                    f"mean_{metric}_scores_on_toxic_prompts.json"
                    if not only_continuations
                    else f"mean_{metric}_continuation_scores_on_toxic_prompts.json"
                )
            else:
                file_name = (
                    f"mean_{metric}_scores.json"
                    if not only_continuations
                    else f"mean_{metric}_continuation_scores.json"
                )

        with open(f"{score_dir}/{file_name}", "w") as f:
            json.dump(all_scores, f, indent=4)


if __name__ == "__main__":
    # get_mean_tox_scores(only_continuations=True, diff=False)

    base_model_name = "Meta-Llama-3-8B-Instruct"
    base_seed = "seed1000"

    model_name1 = "Llama-3-8B-ckpt1"
    model_name2 = "Llama-3-8B-ckpt5"
    model_name3 = "Llama-3-8B-ckpt10"
    model_name4 = "Meta-Llama-3-8B-Instruct-hightemp"
    model_name5 = "Meta-Llama-3-8B-Instruct"
    # model_names = [model_name1, model_name2, model_name3, model_name4]
    model_names = [model_name5]
    seeds = ["seed1000", "seed1000", "seed1000", "seed1000", "seed2000"]

    num_train_samples = [100, 300, 1000, 3000, 10000, 30000, 100000]

    train_cfg = TrainCfg()
    config = load_config("/root/Auditing_test_for_LMs/config.yml")

    for model_name, seed in zip(model_names, seeds):
        df = get_distance_scores(
            base_model_name,
            base_seed,
            "seed2000",
            train_cfg=train_cfg,
            net_cfg=config["net"],
            model_name2=model_name,
            distance_measures=["NeuralNet"],
            num_runs=10,
            num_samples=num_train_samples,
            evaluate_wasserstein_on_full=False,
            save=True,
        )
