import pandas as pd
import json
import numpy as np
import sys
import os

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from copy import deepcopy
from pathlib import Path
from scipy.stats import skew, wasserstein_distance
from typing import Union, List, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from behavior_evaluation.distance import (
    empirical_wasserstein_distance_p1,
    kolmogorov_variation,
    NeuralNetDistance,
    calc_tot_discrete_variation,
)
from utils.utils import load_config
from arguments import TrainCfg
import random


pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)


def extract_data_for_models(
    model_name1,
    seed1,
    seed2,
    model_name2: Optional[str] = None,
    checkpoint: Optional[str] = None,
    checkpoint_base_name: Optional[str] = None,
    fold_size=4000,
    test_dir="test_outputs",
):
    """ """

    assert model_name2 or (
        checkpoint and checkpoint_base_name
    ), "Either model_name2 or checkpoint and checkpoint_base_name must be provided"

    script_dir = os.path.dirname(__file__)

    # Construct the absolute path to "test_outputs"
    test_dir = os.path.join(script_dir, "..", test_dir)

    print(f"model_name2: {model_name2}")
    if model_name2:
        if fold_size == 4000:
            file_path = f"{test_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}/kfold_test_results.csv"
            if not Path(file_path).exists():
                file_path = f"{test_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}/kfold_test_results_{fold_size}.csv"
        else:
            file_path = f"{test_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}/kfold_test_results_{fold_size}.csv"
    else:
        if fold_size == 4000:
            file_path = f"{test_dir}/{model_name1}_{seed1}_{checkpoint_base_name}{checkpoint}_{seed2}/kfold_test_results.csv"
            if not Path(file_path).exists():
                file_path = f"{test_dir}/{model_name1}_{seed1}_{checkpoint_base_name}{checkpoint}_{seed2}/kfold_test_results_{fold_size}.csv"
        else:
            file_path = f"{test_dir}/{model_name1}_{seed1}_{checkpoint_base_name}{checkpoint}_{seed2}/kfold_test_results_{fold_size}.csv"

    print(f"This is the file path: {file_path} and this is the fold_size: {fold_size}")
    data = pd.read_csv(file_path)

    # TODO: make this less hacky
    # we're just discarding the last fold for now, because it is smaller than the rest
    data = data[data["fold_number"] != data["fold_number"].max()]

    return data


def get_power_over_sequences_from_whole_ds(
    data: pd.DataFrame, fold_size: int = 4000, bs: int = 96
):
    """ """
    max_sequences = (fold_size + bs - 1) // bs
    selected_columns = data[
        [
            "fold_number",
            "sequence",
            "aggregated_davt",
            "sequences_until_end_of_experiment",
        ]
    ]

    filtered_df = selected_columns.drop_duplicates(subset=["sequence", "fold_number"])

    num_folds = filtered_df["fold_number"].nunique()
    # Set 'sequence' as the index of the DataFrame
    indexed_df = filtered_df.set_index("sequence")

    unique_fold_numbers = indexed_df["fold_number"].unique()

    # Initialize a dictionary to store the counts
    sequence_counts = {sequence: 0 for sequence in range(max_sequences)}

    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 1000)
    pd.set_option("display.width", 1000)

    # Iterate over each fold number
    for fold in unique_fold_numbers:
        fold_data = indexed_df[indexed_df["fold_number"] == fold]
        for sequence in range(max_sequences):  # sequence_counts.keys()
            if (
                sequence in fold_data.index
                and fold_data.loc[sequence, "sequences_until_end_of_experiment"]
                == sequence
            ):
                sequence_counts[sequence] += 1
            elif sequence not in fold_data.index:
                sequence_counts[sequence] += 1

    # Convert the result to a DataFrame for better visualization
    result_df = pd.DataFrame(
        list(sequence_counts.items()), columns=["Sequence", "Count"]
    )
    result_df["Power"] = result_df["Count"] / num_folds
    result_df["Samples per Test"] = fold_size
    result_df["Samples"] = result_df["Sequence"] * bs

    result_df.reset_index()

    print(f"This is the result df: {result_df}")

    return result_df


def get_power_over_sequences_for_models_or_checkpoints(
    model_name1,
    seed1,
    seed2,
    model_name2: Optional[str] = None,
    checkpoint: Optional[str] = None,
    checkpoint_base_name: Optional[str] = None,
    fold_size: int = 4000,
    bs: int = 96,
):
    """ """
    assert model_name2 or (
        checkpoint and checkpoint_base_name
    ), "Either model_name2 or checkpoint and checkpoint_base_name must be provided"

    if model_name2:
        data = extract_data_for_models(
            model_name1, seed1, seed2, model_name2=model_name2
        )
        result_df = get_power_over_sequences_from_whole_ds(
            data, fold_size=fold_size, bs=bs
        )
        result_df["model_name1"] = model_name1
        result_df["seed1"] = seed1
        result_df["model_name2"] = model_name2
        result_df["seed2"] = seed2
    else:
        data = extract_data_for_models(
            model_name1,
            seed1,
            seed2,
            checkpoint=checkpoint,
            checkpoint_base_name=checkpoint_base_name,
            fold_size=fold_size,
        )
        result_df = get_power_over_sequences_from_whole_ds(data, fold_size, bs)
        result_df["Checkpoint"] = checkpoint

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
    print(all_scores_df)


def get_power_over_sequences_for_checkpoints(
    base_model_name: Union[str, List[str]],
    base_model_seed: Union[str, List[str]],
    checkpoints: Union[str, List[str]],
    seeds: Union[str, List[str]],
    checkpoint_base_name: str = "Llama-3-8B-ckpt",
):
    if not isinstance(checkpoints, list):
        checkpoints = [checkpoints]
    if not isinstance(seeds, list):
        seeds = [seeds]

    result_dfs = []

    for checkpoint, seed in zip(checkpoints, seeds):
        print(
            f"Base_model: {base_model_name}, base_model_seed: {base_model_seed}, checkpoint: {checkpoint_base_name}{checkpoint}, seed: {seed}"
        )
        try:
            result_df = get_power_over_sequences_for_models_or_checkpoints(
                base_model_name,
                base_model_seed,
                seed,
                checkpoint=checkpoint,
                checkpoint_base_name=checkpoint_base_name,
            )

            result_dfs.append(result_df)

        except FileNotFoundError:
            print(
                f"File for checkpoint {checkpoint} and seed {seed} does not exist yet"
            )

    final_df = pd.concat(result_dfs, ignore_index=True)

    return final_df


def get_distance_scores(
    model_name1: str,
    seed1: int,
    seed2: int,
    checkpoint: Optional[str] = None,
    checkpoint_base_name: Optional[str] = None,
    model_name2: Optional[str] = None,
    metric: str = "toxicity",
    epoch1: int = 0,
    epoch2: int = 0,
    distance_measures: list = ["NeuralNet", "Wasserstein"],
    net_cfg: Optional[dict] = None,
    train_cfg: Optional[DictConfig] = None,
    pre_shuffle: bool = False,
    score_dir: str = "model_scores",
    random_seed: int = 0,
    test_random_seed: int = 0,
    num_samples: int = 100000,
    test_split: float = 0.3,
    compare_metrics: bool = True,
) -> dict:
    """ """
    random.seed(random_seed)
    np.random.seed(random_seed)

    if not (checkpoint and checkpoint_base_name) and not model_name2:
        raise ValueError(
            "Either checkpoint and checkpoint_base_name or model_name2 must be provided"
        )

    script_dir = os.path.dirname(__file__)
    score_dir = os.path.join(script_dir, "..", score_dir)

    score_path1 = os.path.join(
        score_dir, f"{model_name1}_{seed1}", f"{metric}_scores.json"
    )
    score_path2 = os.path.join(
        score_dir,
        f"{checkpoint_base_name}{checkpoint}_{seed2}"
        if checkpoint
        else f"{model_name2}_{seed2}",
        f"{metric}_scores.json",
    )

    try:
        with open(score_path1, "r") as f:
            scores1 = json.load(f)[str(epoch1)][f"{metric}_scores"]
        with open(score_path2, "r") as f:
            scores2 = json.load(f)[str(epoch2)][f"{metric}_scores"]

        # test_samples is always a fixed percentage of whole ds
        num_test_samples = int(test_split * len(scores1))
        # set separate random seed for test samples
        np.random.seed(test_random_seed)
        random_test_indices = np.random.randint(0, len(scores1), num_test_samples)
        print(f"These are the random test indices: {random_test_indices}")
        print(f"This is the number of random test indices: {len(random_test_indices)}")

        test_scores1 = [scores1[i] for i in random_test_indices]
        test_scores2 = [scores2[i] for i in random_test_indices]

        train_scores1 = [
            scores1[i] for i in range(len(scores1)) if i not in random_test_indices
        ]
        train_scores2 = [
            scores2[i] for i in range(len(scores2)) if i not in random_test_indices
        ]

        np.random.seed(random_seed)
        # TODO: Think if this is the best way to do this
        if num_samples < len(scores1) - num_test_samples:
            # num_samples -= num_test_samples
            random_train_indices = np.random.randint(0, len(train_scores1), num_samples)

            # Grab random subset of train scores
            train_scores1 = [train_scores1[i] for i in random_train_indices]
            train_scores2 = [train_scores2[i] for i in random_train_indices]

        dist_dict = {}
        if "Wasserstein" in distance_measures:
            if compare_metrics:
                dist_dict["Wasserstein"] = empirical_wasserstein_distance_p1(
                    test_scores1, test_scores2
                )
                dist_dict["Wasserstein_scipy"] = wasserstein_distance(
                    test_scores1, test_scores2
                )
            else:
                dist_dict["Wasserstein"] = empirical_wasserstein_distance_p1(
                    train_scores1 + test_scores1, train_scores2 + test_scores2
                )
                dist_dict["Wasserstein_scipy"] = wasserstein_distance(
                    train_scores1 + test_scores1, train_scores1 + test_scores2
                )
        # TODO: update this
        if "Kolmogorov" in distance_measures:
            dist_dict["Kolmogorov"] = kolmogorov_variation(scores1, scores2)

        if "NeuralNet" in distance_measures:
            assert net_cfg, "net_dict must be provided for neuralnet distance"
            assert train_cfg, "train_cfg must be provided for neuralnet distance"

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
                dist_dict["NeuralNet_shuffled"] = (
                    neural_net_distance_shuffled.train().item()
                )
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

        return dist_dict

    except FileNotFoundError:
        if checkpoint:
            print(f"File for checkpoint {checkpoint} does not exist yet")
        else:
            print(f"File for model {model_name2} does not exist yet")


def get_power_over_sequences_for_ranked_checkpoints(
    base_model_name,
    base_model_seed,
    checkpoints,
    seeds,
    checkpoint_base_name="LLama-3-8B-ckpt",
    epoch1=0,
    epoch2=0,
    metric="toxicity",
    distance_measure="Wasserstein",
    fold_size=4000,
):
    if not isinstance(checkpoints, list):
        checkpoints = [checkpoints]
    if not isinstance(seeds, list):
        seeds = [seeds]

    result_dfs = []

    for checkpoint, seed in zip(checkpoints, seeds):
        print(
            f"Base_model: {base_model_name}, base_model_seed: {base_model_seed}, checkpoint: {checkpoint_base_name}{checkpoint}, seed: {seed}"
        )

        dist = get_distance_scores(
            base_model_name,
            base_model_seed,
            seed,
            checkpoint=checkpoint,
            checkpoint_base_name=checkpoint_base_name,
            metric=metric,
            distance_measure=distance_measure,
            epoch1=epoch1,
            epoch2=epoch2,
        )

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
        final_df[f"Empirical {distance_measure} Distance"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )

    return final_df


def get_power_over_sequences_for_ranked_checkpoints_wrapper(
    base_model_name,
    base_model_seed,
    checkpoints,
    seeds,
    checkpoint_base_name="LLama-3-8B-ckpt",
    fold_sizes: List[int] = [1000, 2000, 3000, 4000],
    epoch1=0,
    epoch2=0,
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
                epoch1=epoch1,
                epoch2=epoch2,
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
