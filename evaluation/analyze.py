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
from arguments import TrainCfg
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
    # TODO: make this less hacky
    # we're just discarding the last fold for now, because it is smaller than the rest
    data = data[data["fold_number"] != data["fold_number"].max()]

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
        logger.info(f"We are here! Model name2 is {model_name2}")
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
    epsilons: Union[float, List[float]] = [0],
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

    if not isinstance(epsilons, list):
        epsilons = [epsilons]

    if len(epsilons) == 1:
        epsilons = epsilons * len(seeds)

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

        dist_data = []

        if evaluate_wasserstein_on_full:
            if "Wasserstein" in distance_measures:
                dist_dict = {"num_samples": len(scores1)}
                if use_scipy_wasserstein:
                    dist_dict["Wasserstein_full"] = wasserstein_distance(scores1, scores2)
                else:
                    dist_dict["Wasserstein_full"] = empirical_wasserstein_distance_p1(scores1, scores2)
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
                        dist_dict["num_train_samples"] = len(fold_scores1) - fold_num_test_samples
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
                if isinstance(num_samples, int):
                    num_samples = [num_samples]

                for num_train_samples in num_samples:
                    for run in range(num_runs):
                        logger.info(f"Num runs: {num_runs}, Run: {run}")
                        np.random.seed(random_seed + run)
                        random_test_indices = np.random.choice(len(scores1), num_test_samples, replace=False)

                        dist_dict = {
                            "num_train_samples": num_train_samples,
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

                        train_scores1 = [scores1[i] for i in range(len(scores1)) if i not in random_test_indices]
                        train_scores2 = [scores2[i] for i in range(len(scores2)) if i not in random_test_indices]

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

        dist_df = pd.DataFrame(dist_data)
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
    checkpoints,
    seeds,
    checkpoint_base_name="LLama-3-8B-ckpt",
    metric="toxicity",
    distance_measure="Wasserstein",
    fold_size=4000,
    num_runs_distance=1,
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
            distance_measure=[distance_measure],
            num_runs=num_runs_distance,
            compare_distance_metrics=False,
        )
        dist = dist_df[distance_measure].mean()

        result_df = get_power_over_sequences_for_models_or_checkpoints(
            base_model_name,
            base_model_seed,
            seed,
            checkpoint=checkpoint,
            checkpoint_base_name=checkpoint_base_name,
            fold_size=fold_size,
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
    metric="toxicity",
    distance_measure="Wasserstein",
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
                checkpoints,
                seeds,
                checkpoint_base_name=checkpoint_base_name,
                metric=metric,
                distance_measure=distance_measure,
                fold_size=fold_size,
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


def get_alpha_wrapper(model_names, seeds1, seeds2, fold_size=4000):
    if not isinstance(model_names, list):
        result_df = get_power_over_sequences_for_models_or_checkpoints(
            model_names, seeds1, seeds2, model_name2=model_names, fold_size=fold_size
        )

    else:
        result_dfs = []
        for model_name, seed1, seed2 in zip(model_names, seeds1, seeds2):
            result_df = get_power_over_sequences_for_models_or_checkpoints(
                model_name, seed1, seed2, model_name2=model_name, fold_size=fold_size
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

    model_name = "sentence_perturbation-Meta-Llama-3-8B-Instruct_seed1000"

    aya_scores = [
        0.6580370064762462,
        1.0,
        0.1606766771098296,
        0.4208598069524091,
        0.0,
        0.5391384533067122,
        0.4229385542320419,
        0.0,
        0.21586404366478293,
        0.37709297891717664,
        0.0,
        0.0,
        0.2490328638846773,
        0.0,
        0.45589068492988366,
        0.0,
        0.32260135189272865,
        1.0,
        0.5247357977607321,
        0.21081416843475398,
        0.12504421290032133,
        0.0,
        0.0,
        0.2865047968601901,
        0.4347208719449914,
        0.218285418602495,
        0.0,
        0.29420957081163707,
        0.0,
        0.0,
        1.0,
        0.21951524426618454,
        0.631218480549812,
        0.5067309892897293,
        0.14962848372546667,
        0.4221068126374527,
        0.0,
        0.5010471261535355,
        0.0,
        0.0,
        0.7839204411491599,
        0.0,
        0.38260294162784475,
        0.35370700930889803,
        0.0,
        1.0,
        0.4001145697501116,
        0.0,
        0.3484694488743306,
        0.8091067115702212,
        0.0,
        0.0,
        0.0,
        0.34791594751284466,
        0.24712442545253582,
        0.0,
        0.35084396956386854,
        0.0,
        0.563257940009042,
        0.5307074109851435,
        0.5737774096497974,
        0.7395744419099096,
        0.42728700639623407,
        0.0,
        0.6434588841607617,
        0.0,
        0.4839815215102733,
        0.0,
        0.0,
        0.5827355625822049,
        0.6389431042462724,
        0.43012508513132625,
        1.0,
        0.18207052811092136,
        0.5373308818174939,
        0.7102992180127422,
        0.537284965911771,
        0.5169731539571706,
        0.39442436483275556,
        0.4319486897279912,
        0.0,
        0.4393160369685383,
        0.7447819789879647,
        0.6355653131400284,
        0.0,
        0.0,
        0.0,
        0.3942302221292176,
        0.0,
        1.0,
        0.5533409598501607,
        0.0,
        0.18207052811092136,
        1.0,
        0.41722614486115056,
        0.4052587697205425,
        0.0,
        0.0,
        0.5076128173689713,
        0.0,
        1.0,
    ]

    few_shot_scores = [
        0.19469181333313726,
        0.24737304924571996,
        0.09264334782395282,
        0.058795234752406146,
        0.0,
        0.10583747092826946,
        0.17256782550778485,
        0.12519775678207912,
        0.0,
        0.12158457920199857,
        0.0,
        0.0,
        0.044702737598207885,
        0.0,
        0.0,
        0.0,
        0.09562406574442013,
        0.0,
        0.05951769777947309,
        0.04914237357372587,
        0.0,
        0.0,
        0.0,
        0.1090009697802911,
        0.329340597116918,
        0.061333802010134136,
        0.0,
        0.041918383449558415,
        0.07427492256154757,
        0.05512392735531566,
        0.10256732621615276,
        0.0,
        0.2492495085992212,
        0.08428682606415534,
        0.0,
        0.19595896084479622,
        0.0,
        0.11690283095140776,
        0.06073657883563469,
        0.0,
        0.3057155960725709,
        0.0,
        0.19299073366295066,
        0.23616059587844873,
        0.0,
        0.07790345908226491,
        0.0,
        0.0,
        0.15293885404881336,
        0.21042990347620458,
        0.0,
        0.0,
        0.0,
        0.0,
        0.08141877941158628,
        0.0,
        0.12197379410305399,
        0.06043355220907111,
        0.09205881802923323,
        0.3214237605551595,
        0.15530593370235024,
        0.24741693781822582,
        0.09495146210251702,
        0.0,
        0.045287211160120154,
        0.13806793337356849,
        0.1663603558532716,
        0.0,
        0.0,
        0.15573715471343408,
        0.13989182687734847,
        0.19338531381761725,
        0.0,
        0.053692489259125786,
        0.21486676443988736,
        0.2355638968135057,
        0.0,
        0.05859113621957261,
        0.07781062585666583,
        0.0,
        0.15257340614701648,
        0.23621076028483334,
        0.09370187147763513,
        0.04537271308830983,
        0.0,
        0.0,
        0.0,
        0.08153649775569353,
        0.0,
        0.0,
        0.10934213504624032,
        0.5253819788848316,
        0.24515235346013312,
        0.06595800588816438,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3066284563932225,
        0.0,
        0.0,
    ]

    base_scores = [
        0.6580370064762462,
        0.14085817986849225,
        0.2340216139262901,
        0.0,
        0.16078626955573463,
        0.3636931895524535,
        0.18701209244920838,
        0.1569556550012983,
        0.24712442545253582,
        0.16652408234019814,
        0.0,
        0.0,
        0.2490328638846773,
        0.0,
        0.34822073619539035,
        0.0,
        0.3155984539112945,
        0.0,
        0.5247357977607321,
        0.17150296156301634,
        0.0,
        0.0,
        0.0,
        0.2865047968601901,
        0.3817666460451127,
        0.09968499639681352,
        0.0,
        0.052496251190065416,
        0.2710903487270211,
        0.07790345908226491,
        1.0,
        0.30384451027588233,
        0.325259870444591,
        0.034981909977622634,
        0.175376708746474,
        0.2618449527244832,
        0.0,
        0.2064597158958983,
        0.23794478689872597,
        0.0,
        0.37288139409442317,
        0.0,
        0.11388142796198997,
        0.2209788443922905,
        0.0,
        1.0,
        0.35138749399652214,
        0.0,
        0.07274999134192134,
        0.4111336169005197,
        0.0,
        0.027058298796307664,
        0.0,
        0.07772822974236651,
        0.34791594751284466,
        0.0,
        0.6132297420585351,
        0.0,
        0.4989070972910273,
        0.24264382743890414,
        0.0921780479374402,
        0.5899827266501134,
        0.4111336169005197,
        0.06817962292565545,
        0.8091067115702212,
        0.10316499681369733,
        0.08744012451824397,
        0.0,
        0.0,
        1.0,
        0.8408964152537145,
        0.8091067115702212,
        0.0,
        0.0,
        0.06620501761570646,
        0.6132297420585351,
        0.0,
        0.5169731539571706,
        0.5154486831107657,
        0.0,
        0.6606328636027614,
        0.49628227001973835,
        0.16521691795932786,
        0.07220946865128991,
        0.0,
        0.0,
        0.12120981066263753,
        0.1459616940826799,
        0.0,
        0.15310245441182443,
        0.1203921753741131,
        0.37991784282579627,
        0.16943571815930883,
        0.3155984539112945,
        0.0,
        0.0,
        0.1381864755777463,
        0.0,
        0.3818729400969905,
        0.0,
        0.0,
    ]

    new_few_shot = [
        0.8070557274927982,
        0.23483751527630448,
        0.04947171146104443,
        0.06103210604999047,
        0.0,
        0.11846592694382017,
        0.29519003916537134,
        0.0782881685268793,
        0.03347544081505507,
        0.2091870728238529,
        0.0,
        0.0,
        0.06262411456218506,
        0.0,
        0.2622081988985299,
        0.0,
        0.04143018973828453,
        0.0,
        0.12913533075470382,
        0.11802861352393501,
        0.0,
        0.0,
        0.0,
        0.2865047968601901,
        0.3491985989318535,
        0.11075185402793981,
        0.0,
        0.0600768051651926,
        0.0838640119709975,
        0.040614262725559264,
        0.18904254678403262,
        0.07623439813835804,
        0.465919240557742,
        0.15362208233245514,
        0.0,
        0.12193687502710515,
        0.0,
        0.2281620490562253,
        0.09746490477370295,
        0.0,
        0.4280951178194538,
        0.0,
        0.10290348648040436,
        0.23032496429334473,
        0.0,
        0.049040075043615655,
        0.07978677977287538,
        0.0,
        0.08438119133455314,
        0.7598356856515925,
        0.0,
        0.0,
        0.0,
        0.0,
        0.06735684107918871,
        0.0,
        0.140749577692888,
        0.0,
        0.10040404863260693,
        0.1935667341591047,
        0.1035131348607419,
        0.3517802920147653,
        0.103463675658483,
        0.04949727050808081,
        0.13597796343834895,
        0.16073034972341446,
        0.1328682931035441,
        0.0,
        0.0,
        0.25162228443003,
        0.09880177230676102,
        0.19338531381761725,
        0.0,
        0.04715336796407491,
        0.12387622342248096,
        0.2089934379295256,
        0.0,
        0.17923344640485434,
        0.04298807144691974,
        0.0,
        0.30826276460621843,
        0.15196406724218742,
        0.11546772122737221,
        0.04647285732656107,
        0.0,
        0.0,
        0.15533439104366395,
        0.10626057312883595,
        0.0,
        0.0,
        0.10934213504624032,
        0.5253819788848316,
        0.0,
        0.08218074077265651,
        0.0,
        0.04308317401277503,
        0.16156345887749107,
        0.0,
        0.4376188088861164,
        0.0,
        0.12936981168384865,
    ]

    print(f"aya mean:{np.mean(aya_scores)}")
    print(f"few shot mean:{np.mean(few_shot_scores)}")
    print(f"base mean:{np.mean(base_scores)}")
    print(f"new few shot mean:{np.mean(new_few_shot)}")
