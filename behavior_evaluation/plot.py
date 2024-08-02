import pandas as pd
import json
import numpy as np
import sys
import os

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

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

    print(f"This is the indexed df: {indexed_df}")

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
    distance_measures: list = ["neuralnet", "Wasserstein"],
    net_cfg: Optional[dict] = None,
    train_cfg: Optional[DictConfig] = None,
    score_dir: str = "model_scores"
) -> dict:
    """ """
    if not (checkpoint and checkpoint_base_name) and not model_name2:
        raise ValueError("Either checkpoint and checkpoint_base_name or model_name2 must be provided")

    script_dir = os.path.dirname(__file__)
    score_dir = os.path.join(script_dir, "..", score_dir)

    score_path1 = os.path.join(score_dir, f"{model_name1}_{seed1}", f"{metric}_scores.json")
    score_path2 = os.path.join(score_dir, f"{checkpoint_base_name}{checkpoint}_{seed2}" if checkpoint else f"{model_name2}_{seed2}", f"{metric}_scores.json")

    try:
        with open(score_path1, "r") as f:
            scores1 = json.load(f)[str(epoch1)][f"{metric}_scores"]
        with open(score_path2, "r") as f:
            scores2 = json.load(f)[str(epoch2)][f"{metric}_scores"]

        dist_dict = {}
        if "Wasserstein" in distance_measures:
            dist_dict["Wasserstein"] = empirical_wasserstein_distance_p1(scores1, scores2)
            dist_dict["Wasserstein_scipy"] = wasserstein_distance(scores1, scores2)
        if "Kolmogorov" in distance_measures:
            dist_dict["Kolmogorov"] = kolmogorov_variation(scores1, scores2)
        if "neuralnet" in distance_measures:
            assert net_cfg, "net_dict must be provided for neuralnet distance"
            assert train_cfg, "train_cfg must be provided for neuralnet distance"
            neural_net_distance = NeuralNetDistance(net_cfg, scores1, scores2, train_cfg)
            dist_dict["neuralnet"] = neural_net_distance.train().item()

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


def plot_power_over_number_of_sequences(
    base_model_name: str,
    base_model_seed: str,
    checkpoints: List[str],
    seeds: List[str],
    checkpoint_base_name: str = "LLama-3-8B-ckpt",
    save: bool = True,
    group_by: str = "Checkpoint",
    marker: str = "X",
    save_as_pdf: bool = True,
    test_dir: str = "test_outputs",
    plot_dir: str = "plots",
    metric: str = "perspective",
):
    script_dir = os.path.dirname(__file__)

    # Construct the absolute path to "test_outputs"
    test_dir = os.path.join(script_dir, "..", test_dir)
    plot_dir = os.path.join(script_dir, "..", plot_dir)

    if group_by == "Checkpoint":
        result_df = get_power_over_sequences_for_checkpoints(
            base_model_name,
            base_model_seed,
            checkpoints,
            seeds,
            checkpoint_base_name=checkpoint_base_name,
        )
    elif (
        group_by == "Rank based on Wasserstein Distance"
        or group_by == "Empirical Wasserstein Distance"
    ):
        result_df = get_power_over_sequences_for_ranked_checkpoints(
            base_model_name,
            base_model_seed,
            checkpoints,
            seeds,
            checkpoint_base_name=checkpoint_base_name,
            metric="perspective",
        )

        result_df["Empirical Wasserstein Distance"] = result_df[
            "Empirical Wasserstein Distance"
        ].round(3)

    # Create the plot
    plt.figure(figsize=(12, 6))
    unique_groups = result_df[group_by].unique()
    palette = sns.color_palette("viridis", len(unique_groups))
    palette = palette[::-1]

    sns.lineplot(
        data=result_df,
        x="Samples",
        y="Power",
        hue=group_by,
        marker=marker,
        markersize=10,
        palette=palette,
    )

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Customize the plot
    plt.xlabel("samples", fontsize=16)
    plt.ylabel("detection frequency", fontsize=16)
    if group_by == "Checkpoint":
        title = "checkpoints"
    elif group_by == "Rank based on Wasserstein Distance":
        title = "rank"
    elif group_by == "Empirical Wasserstein Distance":
        title = "distance"
    plt.legend(
        title=title,
        loc="lower right",
        fontsize=14,
        title_fontsize=16,
        # bbox_to_anchor=(1, 1),
    )
    plt.grid(True, linewidth=0.5, color="#ddddee")

    # Making the box around the plot thicker
    plt.gca().spines["top"].set_linewidth(1.5)
    plt.gca().spines["right"].set_linewidth(1.5)
    plt.gca().spines["bottom"].set_linewidth(1.5)
    plt.gca().spines["left"].set_linewidth(1.5)

    if save:
        directory = f"{plot_dir}/{base_model_name}_{base_model_seed}_{checkpoint_base_name}_checkpoints"
        if not Path(directory).exists():
            Path(directory).mkdir(parents=True, exist_ok=True)
        else:
            for seed in seeds:
                directory += f"_{seed}"
            if not Path(directory).exists():
                Path(directory).mkdir(parents=True, exist_ok=True)
        if save_as_pdf:
            plt.savefig(
                f"{directory}/power_over_number_of_sequences_grouped_by_{group_by}_{base_model_name}_{base_model_seed}.pdf",
                bbox_inches="tight",
                format="pdf",
            )
        else:
            plt.savefig(
                f"{directory}/power_over_number_of_sequences_grouped_by_{group_by}_{base_model_name}_{base_model_seed}.png",
                dpi=300,
                bbox_inches="tight",
            )
    plt.show()


def plot_power_over_epsilon(
    base_model_name,
    base_model_seed,
    checkpoints,
    seeds,
    checkpoint_base_name="LLama-3-8B-ckpt",
    epoch1=0,
    epoch2=0,
    metric="toxicity",
    save=True,
    distance_measure="Wasserstein",
    fold_sizes: Union[int, List[int]] = [1000, 2000, 3000, 4000],
    marker="X",
    palette=["#E49B0F", "#C46210", "#B7410E", "#A81C07"],
    save_as_pdf=True,
    plot_dir: str = "plots",
):
    """
    This plots power over distance measure, potentially for different fold_sizes and models.
    """

    script_dir = os.path.dirname(__file__)

    # Construct the absolute path to "test_outputs"
    plot_dir = os.path.join(script_dir, "..", plot_dir)

    if isinstance(fold_sizes, list):
        result_df = get_power_over_sequences_for_ranked_checkpoints_wrapper(
            base_model_name,
            base_model_seed,
            checkpoints,
            seeds,
            checkpoint_base_name=checkpoint_base_name,
            epoch1=epoch1,
            epoch2=epoch2,
            metric=metric,
            distance_measure=distance_measure,
            fold_sizes=fold_sizes,
        )
    else:
        result_df = get_power_over_sequences_for_ranked_checkpoints(
            base_model_name,
            base_model_seed,
            checkpoints,
            seeds,
            checkpoint_base_name=checkpoint_base_name,
            epoch1=epoch1,
            epoch2=epoch2,
            metric=metric,
            distance_measure=distance_measure,
        )

    smaller_df = extract_power_from_sequence_df(
        result_df, distance_measure=distance_measure, by_checkpoints=True
    )

    # in case we have less folds
    palette = palette[-len(fold_sizes) :]

    plt.figure(figsize=(10, 6))

    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 1000)
    pd.set_option("display.width", 1000)

    print(
        f"This is the smaller df inside plot_power_over_epsilon: {smaller_df} for fold_sizes {fold_sizes}"
    )

    sns.lineplot(
        x=f"Empirical {distance_measure} Distance",
        y="Power",
        hue="Samples per Test" if "Samples per Test" in smaller_df.columns else None,
        # style="Samples per Test" if "Samples per Test" in smaller_df.columns else None,
        marker=marker,
        data=smaller_df,
        markersize=10,
        palette=palette,
    )

    # plt.xlabel(f"{distance_measure.lower()} distance", fontsize=14)
    plt.xlabel(f"distance to aligned model", fontsize=16)
    plt.ylabel("detection frequency", fontsize=16)
    plt.grid(True, linewidth=0.5, color="#ddddee")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(
        title="samples per test",
        loc="lower right",
        fontsize=14,
        title_fontsize=16,
        # bbox_to_anchor=(
        #     1.05,
        #     1,
        # ),  # Adjusted position to ensure the legend is outside the plot area
    )

    # Make the surrounding box thicker
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    if save:
        directory = f"{plot_dir}/{base_model_name}_{base_model_seed}_{checkpoint_base_name}_checkpoints"
        if not Path(directory).exists():
            Path(directory).mkdir(parents=True, exist_ok=True)
        else:
            for seed in seeds:
                directory += f"_{seed}"
            if not Path(directory).exists():
                Path(directory).mkdir(parents=True, exist_ok=True)

        if "Samples per Test" in smaller_df.columns:
            if save_as_pdf:
                plt.savefig(
                    f"{directory}/power_over_{distance_measure.lower()}_distance_grouped_by_fold_size_{base_model_name}_{base_model_seed}.pdf",
                    bbox_inches="tight",
                    format="pdf",
                )

            else:
                plt.savefig(
                    f"{directory}/power_over_{distance_measure.lower()}_distance_grouped_by_fold_size_{base_model_name}_{base_model_seed}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
        else:
            if save_as_pdf:
                plt.savefig(
                    f"{directory}/power_over_{distance_measure.lower()}_distance_{base_model_name}_{base_model_seed}.pdf",
                    bbox_inches="tight",
                    format="pdf",
                )
            else:
                plt.savefig(
                    f"{directory}/power_over_{distance_measure.lower()}_distance_{base_model_name}_{base_model_seed}.png",
                    dpi=300,
                    bbox_inches="tight",
                )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x=f"Rank based on {distance_measure} Distance",
        y="Power",
        hue="Samples per Test" if "Samples per Test" in smaller_df.columns else None,
        # style="Samples per Test" if "Samples per Test" in smaller_df.columns else None,
        marker=marker,
        data=smaller_df,
        markersize=10,
        palette=palette,
    )
    # plt.xlabel(f"rank based on {distance_measure.lower()} distance", fontsize=14)
    plt.xlabel(f"rank based on distance to aligned model", fontsize=16)
    plt.ylabel("detection frequency", fontsize=16)
    plt.grid(True, linewidth=0.5, color="#ddddee")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(
        title="samples per test",
        loc="lower right",
        fontsize=14,
        title_fontsize=16,
        # bbox_to_anchor=(
        #     1.05,
        #     1,
        # ),  # Adjusted position to ensure the legend is outside the plot area
    )

    # Make the surrounding box thicker
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    if save:
        directory = f"{plot_dir}/{base_model_name}_{base_model_seed}_{checkpoint_base_name}_checkpoints"
        if not Path(directory).exists():
            Path(directory).mkdir(parents=True, exist_ok=True)
        else:
            for seed in seeds:
                directory += f"_{seed}"
            if not Path(directory).exists():
                Path(directory).mkdir(parents=True, exist_ok=True)
        if "Samples per Test" in smaller_df.columns:
            if save_as_pdf:
                plt.savefig(
                    f"{directory}/power_over_{distance_measure.lower()}_rank_grouped_by_fold_size_{base_model_name}_{base_model_seed}.pdf",
                    bbox_inches="tight",
                    format="pdf",
                )
            else:
                plt.savefig(
                    f"{directory}/power_over_{distance_measure.lower()}_rank_grouped_by_fold_size_{base_model_name}_{base_model_seed}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
        else:
            if save_as_pdf:
                plt.savefig(
                    f"{directory}/power_over_{distance_measure.lower()}_rank_{base_model_name}_{base_model_seed}.pdf",
                    bbox_inches="tight",
                    format="pdf",
                )
            else:
                plt.savefig(
                    f"{directory}/power_over_{distance_measure.lower()}_rank_{base_model_name}_{base_model_seed}.png",
                    dpi=300,
                    bbox_inches="tight",
                )


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


def plot_alpha_over_sequences(
    model_names,
    seeds1,
    seeds2,
    save=True,
    save_as_pdf=True,
    markers=["X", "o", "s"],
    palette=["#94D2BD", "#EE9B00", "#BB3E03"],
    fold_size=4000,
    plot_dir: str = "plots",
):
    script_dir = os.path.dirname(__file__)

    # Construct the absolute path to "test_outputs"
    plot_dir = os.path.join(script_dir, "..", plot_dir)

    result_df = get_alpha_wrapper(model_names, seeds1, seeds2, fold_size=fold_size)
    group_by_model = "model_id" in result_df.columns

    # Create the plot
    plt.figure(figsize=(12, 6))

    if group_by_model:
        unique_models = result_df["model_id"].unique()
        for i, model in enumerate(unique_models):
            sns.lineplot(
                data=result_df[result_df["model_id"] == model],
                x="Samples",
                y="Power",
                marker=markers[i % len(markers)],
                dashes=False,  # No dashes, solid lines
                color=palette[i % len(palette)],
                label=model,
            )
    else:
        sns.lineplot(
            data=result_df,
            x="Samples",
            y="Power",
            marker="o",
            dashes=False,  # No dashes, solid lines
            color="black",
        )

    # Customize the plot
    plt.xlabel("samples", fontsize=16)
    plt.ylabel("false positive rate", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Adjust the spines (box) thickness
    ax = plt.gca()
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)

    if group_by_model:
        plt.legend(
            title="models",
            # loc="lower right",
            loc="lower right",
            fontsize=14,
            title_fontsize=16,
            # bbox_to_anchor=(
            #     1.05,
            #     1,
            # ),  # Adjusted position to ensure the legend is outside the plot area
        )
    plt.grid(True, linewidth=0.5, color="#ddddee")

    if save:
        directory = f"{plot_dir}/alpha_plots"
        if not Path(directory).exists():
            Path(directory).mkdir(parents=True, exist_ok=True)
        fig_path = f"{directory}/alpha_error_over_number_of_sequences"
        if isinstance(model_names, str):
            fig_path += f"_{model_names}"
        elif isinstance(model_names, list):
            for model_name in model_names:
                fig_path += f"_{model_name}"

        if save_as_pdf:
            fig_path += ".pdf"
            plt.savefig(fig_path, bbox_inches="tight", format="pdf")
        else:
            fig_path += ".png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")


def plot_rejection_rate_matrix(
    model_names1,
    seeds1,
    model_names2: Optional[List[str]] = None,
    seeds2: Optional[List[str]] = None,
    fold_size=4000,
    distance_measure: Optional[str] = "Wasserstein",
    metric: Optional[str] = "toxicity",
    epoch1: Optional[int] = 0,
    epoch2: Optional[int] = 0,
    save: bool = True,
    save_as_pdf: bool = True,
    plot_dir: str = "plots",
):
    """ """
    assert (model_names2 is None and seeds2 is None) or (
        model_names2 is not None and seeds2 is not None
    ), "Either give full list of test models or expect to iterate over all combinations"

    script_dir = os.path.dirname(__file__)

    # Construct the absolute path to "test_outputs"
    plot_dir = os.path.join(script_dir, "..", plot_dir)

    results_df = []
    if not model_names2:
        for i, (model_name1, seed1) in enumerate(zip(model_names1[:-1], seeds1[:-1])):
            for model_name2, seed2 in zip(model_names1[i + 1 :], seeds1[i + 1 :]):
                print(
                    f"Checking model {model_name1}, {seed1} against {model_name2}, {seed2}"
                )
                result_df = get_power_over_sequences_for_models_or_checkpoints(
                    model_name1,
                    seed1,
                    seed2,
                    model_name2=model_name2,
                    fold_size=fold_size,
                )
                if distance_measure:
                    dist = get_distance_scores(
                        model_name1,
                        seed1,
                        seed2,
                        model_name2=model_name2,
                        metric=metric,
                        distance_measure=distance_measure,
                        epoch1=epoch1,
                        epoch2=epoch2,
                    )
                    result_df[f"Empirical {distance_measure} Distance"] = dist
                small_df = extract_power_from_sequence_df(
                    result_df, distance_measure=distance_measure, by_checkpoints=False
                )

                results_df.append(small_df)

        results_df = pd.concat(results_df, ignore_index=True)

    print(results_df)
    pivot_table = results_df.pivot_table(values="Power", index="seed1", columns="seed2")

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(
        pivot_table,
        annot=True,
        cmap="viridis",
        cbar_kws={"label": "Frequency of Positive Test Result"},
    )
    heatmap.set_title(f"Positive Test Rates for model {model_names1[0]}")

    if save:
        directory = f"{plot_dir}/power_heatmaps"
        if not Path(directory).exists():
            Path(directory).mkdir(parents=True, exist_ok=True)
        file_name = "power_heatmap"
        for model_name, seed in zip(model_names1, seeds1):
            file_name += f"_{model_name}_{seed}"

        if save_as_pdf:
            file_name += ".pdf"
            output_path = os.path.join(directory, file_name)
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
        else:
            file_name += ".png"
            output_path = os.path.join(directory, file_name)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

    else:
        plt.show()

    plt.close()

    if distance_measure:
        distance_pivot_table = results_df.pivot_table(
            values=f"Empirical {distance_measure} Distance",
            index="seed1",
            columns="seed2",
        )

        # Create the heatmap
        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(
            distance_pivot_table,
            annot=True,
            cmap="viridis",
            cbar_kws={"label": "Distance"},
        )
        heatmap.set_title(f"Distance Heatmap for model {model_names1[0]}")

        if save:
            directory = f"{plot_dir}/power_heatmaps"
            if not Path(directory).exists():
                Path(directory).mkdir(parents=True, exist_ok=True)

            file_name = "distance_heatmap"
            for model_name, seed in zip(model_names1, seeds1):
                file_name += f"_{model_name}_{seed}"

            if save_as_pdf:
                file_name += ".pdf"
                output_path = os.path.join(directory, file_name)
                plt.savefig(output_path, format="pdf", bbox_inches="tight")
            else:
                file_name += ".png"
                output_path = os.path.join(directory, file_name)
                plt.savefig(output_path, dpi=300, bbox_inches="tight")

        else:
            plt.show()

        plt.close()


def plot_scores(
    model_name,
    seed,
    metric="toxicity",
    save=True,
    epoch=0,
    use_log_scale=True,
    color="blue",
    save_as_pdf=True,
    plot_dir: str = "plots",
):
    """ """
    script_dir = os.path.dirname(__file__)

    # Construct the absolute path to "test_outputs"
    plot_dir = os.path.join(script_dir, "..", plot_dir)

    directory = f"{plot_dir}/{model_name}_{seed}"
    file_path = f"{directory}/{metric}_scores.json"
    with open(file_path, "r") as f:
        data = json.load(f)

    scores = data[str(epoch)][f"{metric}_scores"]
    # Calculate statistics
    mean_score = np.mean(scores)
    array_scores = np.array(scores)

    # Calculate the skewness using scipy.stats.skew
    skewness = skew(array_scores)
    print(f"skewness: {skewness:.3f}")

    plt.figure(figsize=(14, 7))

    # Plot histogram with adjusted bins and density plot
    sns.histplot(
        scores,
        bins=50,
        # kde=True,
        color=color,
        edgecolor=None,
        alpha=0.7,
    )

    # # Add mean and std lines
    plt.axvline(
        mean_score,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_score:.3f}",
        # label_size=16,
    )
    # plt.axvline(
    #     mean_score + std_score,
    #     color="green",
    #     linestyle="--",
    #     linewidth=1.5,
    #     label=f"+1 Std Dev: {mean_score + std_score:.2f}",
    # )
    # plt.axvline(
    #     mean_score - std_score,
    #     color="green",
    #     linestyle="--",
    #     linewidth=1.5,
    #     label=f"-1 Std Dev: {mean_score - std_score:.2f}",
    # )

    # Set plot limits
    plt.xlim(0, 1)
    if use_log_scale:
        plt.yscale("log")

    # plt.title(
    #     f"Distribution of {metric.capitalize()} Scores for {model_name} (Seed: {seed})",
    #     fontsize=16,
    # )
    plt.xlabel(f"{metric.lower()} score", fontsize=16)
    plt.ylabel("log frequency" if use_log_scale else "frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)

    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.05))
    plt.grid(True, "minor", color="#ddddee")

    if save:
        if use_log_scale:
            if save_as_pdf:
                output_path = os.path.join(
                    directory, f"{metric}_scores_{model_name}_{seed}_log.pdf"
                )
                plt.savefig(output_path, format="pdf", bbox_inches="tight")
            else:
                output_path = os.path.join(
                    directory, f"{metric}_scores_{model_name}_{seed}_log.png"
                )
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            if save_as_pdf:
                output_path = os.path.join(
                    directory, f"{metric}_scores_{model_name}_{seed}.pdf"
                )
                plt.savefig(output_path, format="pdf", bbox_inches="tight")
            else:
                output_path = os.path.join(
                    directory, f"{metric}_{model_name}_{seed}_scores.png"
                )
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_scores_base_most_extreme(
    base_model_name,
    base_model_seed,
    checkpoints,
    checkpoint_seeds,
    checkpoint_base_name,
    save=True,
    use_log_scale=True,
    metric="toxicity",
    base_model_epoch=0,
    epochs=None,
    color="blue",
    darker_color="blue",
    dark=False,
    corrupted_color="red",
    darker_corrupted_color="red",
    save_as_pdf=True,
    plot_dir: str = "plots",
):
    if not epochs:
        epochs = [0 for i in checkpoints]

    script_dir = os.path.dirname(__file__)

    # Construct the absolute path to "test_outputs"
    plot_dir = os.path.join(script_dir, "..", plot_dir)

    directory = f"{plot_dir}/{base_model_name}_{base_model_seed}"
    file_path = f"{directory}/{metric}_scores.json"
    print(f"This is the original model: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)

    scores = data[str(base_model_epoch)][f"{metric}_scores"]
    scores_dict = {}
    wasserstein_distances = {}

    for ckpt, seed, epoch in zip(checkpoints, checkpoint_seeds, epochs):
        checkpoint_directory = f"{plot_dir}/{checkpoint_base_name}{ckpt}_{seed}"
        file_path = f"{checkpoint_directory}/{metric}_scores.json"
        with open(file_path, "r") as f:
            checkpoint_data = json.load(f)
            scores_ckpt = checkpoint_data[str(epoch)][f"{metric}_scores"]
            scores_dict[(ckpt, seed, epoch)] = scores_ckpt
            wasserstein_distances[(ckpt, seed, epoch)] = (
                empirical_wasserstein_distance_p1(scores, scores_ckpt)
            )

    max_distance_ckpt, max_distance_seed, max_distance_epoch = max(
        wasserstein_distances, key=wasserstein_distances.get
    )
    print(
        f"This is the max distance checkpoint: {max_distance_ckpt} with seed: {max_distance_seed}"
    )
    max_distance = wasserstein_distances[
        (max_distance_ckpt, max_distance_seed, max_distance_epoch)
    ]
    print(f"This is the max distance: {max_distance:.4f}")

    ckpt_scores = scores_dict[
        (max_distance_ckpt, max_distance_seed, max_distance_epoch)
    ]

    array_ckpt_scores = np.array(ckpt_scores)
    skewness = skew(array_ckpt_scores)
    print(
        f"skewness for model {checkpoint_base_name}{max_distance_ckpt}: {skewness:.3f}"
    )

    print(
        f"This is the max score of the base model {base_model_name}: {max(scores)} and this is the max score of the corrupted model {max(ckpt_scores)}"
    )

    df = pd.DataFrame(
        {
            "scores": scores + ckpt_scores,
            "model": [base_model_name] * len(scores)
            + [f"Checkpoint {max_distance_ckpt}"] * len(ckpt_scores),
            "seed": [base_model_seed] * len(scores)
            + [max_distance_seed] * len(ckpt_scores),
        }
    )

    mean_score = np.mean(scores)
    mean_ckpt_score = np.mean(ckpt_scores)

    plt.figure(figsize=(14, 7))

    sns.histplot(
        data=df,
        x="scores",
        hue="model",
        bins=50,
        edgecolor=None,
        alpha=0.7,
        # palette=[color, "#CD5C5C"],
        palette=[color, corrupted_color],
    )

    plt.axvline(
        mean_score,
        color=darker_color,
        linestyle="--",
        linewidth=1.5,
        label=f"aligned model mean {metric.lower()}: {mean_score:.3f}",
        # label_size=16,
    )

    plt.axvline(
        mean_ckpt_score,
        color=darker_corrupted_color,
        linestyle="--",
        linewidth=1.5,
        label=f"corrupted model mean {metric.lower()}: {mean_ckpt_score:.3f}",
        # label_size=16,
    )

    plt.xlim(0, 1)
    if use_log_scale:
        plt.yscale("log")

    plt.xlabel(f"{metric.lower()} score", fontsize=16)
    plt.ylabel("log frequency" if use_log_scale else "frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.05))
    plt.grid(True, "minor", color="#ddddee")

    if save:
        if use_log_scale:
            if save_as_pdf:
                output_path = os.path.join(
                    directory,
                    f"{metric}_scores_{base_model_name}_{base_model_seed}_checkpoint{max_distance_ckpt}_{max_distance:.3f}_log.pdf",
                )
                plt.savefig(output_path, bbox_inches="tight", format="pdf")
            else:
                output_path = os.path.join(
                    directory,
                    f"{metric}_scores_{base_model_name}_{base_model_seed}_checkpoint{max_distance_ckpt}_{max_distance:.3f}_log.png",
                )
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            if save_as_pdf:
                output_path = os.path.join(
                    directory,
                    f"{metric}_scores_{base_model_name}_{base_model_seed}_checkpoint{max_distance_ckpt}_{max_distance:.3f}.pdf",
                )
                plt.savefig(output_path, bbox_inches="tight", format="pdf")
            else:
                output_path = os.path.join(
                    directory,
                    f"{metric}_scores_{base_model_name}_{base_model_seed}_checkpoint{max_distance_ckpt}_{max_distance:.3f}.png",
                )
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_scores_two_models(
    model_name1,
    seed1,
    model_name2,
    seed2,
    save=True,
    use_log_scale=True,
    metric="toxicity",
    epoch1=0,
    epoch2=0,
    color="blue",
    darker_color="blue",
    dark=False,
    corrupted_color="red",
    darker_corrupted_color="red",
    save_as_pdf=True,
    score_dir: str = "model_scores",
    plot_dir: str = "plots",
):
    script_dir = os.path.dirname(__file__)

    # Construct the absolute path to "test_outputs"
    score_dir = os.path.join(script_dir, "..", score_dir)
    plot_dir = os.path.join(script_dir, "..", plot_dir)

    score_dir1 = f"{score_dir}/{model_name1}_{seed1}"
    score_path1 = f"{score_dir1}/{metric}_scores.json"
    score_dir2 = f"{score_dir}/{model_name2}_{seed2}"
    score_path2 = f"{score_dir2}/{metric}_scores.json"
    save_dir1 = f"{plot_dir}/{model_name1}_{seed1}"
    save_dir2 = f"{plot_dir}/{model_name2}_{seed2}"

    with open(score_path1, "r") as f:
        data1 = json.load(f)

    with open(score_path2, "r") as f:
        data2 = json.load(f)

    scores1 = data1[str(epoch1)][f"{metric}_scores"]
    scores2 = data2[str(epoch2)][f"{metric}_scores"]

    dist = empirical_wasserstein_distance_p1(scores1, scores2)

    print(
        f"This is the distance: {dist} between {model_name1}, {seed1} and {model_name2}, {seed2}"
    )
    skewness1 = skew(scores1)
    skewness2 = skew(scores2)
    print(f"skewness for model {model_name1}, {seed1}: {skewness1:.3f}")
    print(f"skewness for model {model_name2}, {seed2}: {skewness2:.3f}")

    df = pd.DataFrame(
        {
            "scores": scores1 + scores2,
            "model 1": [f"{model_name1}_{seed1}"] * len(scores1)
            + [f"{model_name2}_{seed2}"] * len(scores2),
        }
    )

    mean_score1 = np.mean(scores1)
    mean_score2 = np.mean(scores2)

    plt.figure(figsize=(14, 7))

    sns.histplot(
        data=df,
        x="scores",
        hue="model 1",
        bins=50,
        edgecolor=None,
        alpha=0.7,
        # palette=[color, "#CD5C5C"],
        palette=[color, corrupted_color],
    )

    plt.axvline(
        mean_score1,
        color=darker_color,
        linestyle="--",
        linewidth=1.5,
        label=f"model 1 mean {metric.lower()}: {mean_score1:.3f}",
        # label_size=16,
    )

    plt.axvline(
        mean_score2,
        color=darker_corrupted_color,
        linestyle="--",
        linewidth=1.5,
        label=f"model 2 mean {metric.lower()}: {mean_score2:.3f}",
        # label_size=16,
    )

    plt.xlim(0, 1)
    if use_log_scale:
        plt.yscale("log")

    plt.xlabel(f"{metric.lower()} score", fontsize=16)
    plt.ylabel("log frequency" if use_log_scale else "frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.05))
    plt.grid(True, "minor", color="#ddddee")

    if save:
        if use_log_scale:
            if save_as_pdf:
                output_path = os.path.join(
                    save_dir1,
                    f"{metric}_scores_{model_name1}_{seed1}_{model_name2}_{seed2}_log.pdf",
                )
                plt.savefig(output_path, bbox_inches="tight", format="pdf")
            else:
                output_path = os.path.join(
                    save_dir1,
                    f"{metric}_scores_{model_name1}_{seed1}_{model_name2}_{seed2}_log.png",
                )
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            if save_as_pdf:
                output_path = os.path.join(
                    save_dir1,
                    f"{metric}_scores_{model_name1}_{seed1}_{model_name2}_{seed2}.pdf",
                )
                plt.savefig(output_path, bbox_inches="tight", format="pdf")
            else:
                output_path = os.path.join(
                    save_dir1,
                    f"{metric}_scores_{model_name1}_{seed1}_{model_name2}_{seed2}.png",
                )
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


@hydra.main(
    config_path="/root/DistanceSimulation/behavior_evaluation",
    config_name="plotting_config.yml",
)
def plot_all(
    cfg: DictConfig, use_alternative_seeds: bool = False
):  # TODO: add alternative seeds
    # Loop over all base models
    for bm in cfg.models:
        checkpoints = [i for i in range(1, int(bm.checkpoint_range))]
        seeds = ["seed1000" for i in range(1, int(bm.checkpoint_range))]

        # Create power plot over sequences:
        plot_power_over_number_of_sequences(
            bm.name,
            bm.seed,
            checkpoints,
            seeds,
            checkpoint_base_name=bm.checkpoint_base_name,
            save=True,
            group_by="Empirical Wasserstein Distance",
            marker=bm.marker,
        )

        # Create power plot over distance:
        plot_power_over_epsilon(
            bm.name,
            "seed1000",
            checkpoints,
            seeds,
            checkpoint_base_name=bm.checkpoint_base_name,
            fold_sizes=list(cfg.fold_sizes),
            marker=bm.marker,
            metric=cfg.metric,
        )

    # for (
    #     bm_name,
    #     checkpoints,
    #     seeds,
    #     checkpoint_base_name,
    #     color,
    #     darker_color,
    #     corrupted_color,
    #     darker_corrupted_color,
    # ) in zip(
    #     base_model_name_list,
    #     custom_colors,
    #     checkpoints_list,
    #     seeds_list,
    #     checkpoint_base_name_list,
    #     custom_colors,
    #     darker_custom_colors,
    #     corrupted_model_custom_colors,
    #     darker_corrupted_model_custom_colors,
    # ):
    #     plot_scores_base_most_extreme(
    #         bm_name,
    #         "seed1000",
    #         checkpoints,
    #         seeds,
    #         checkpoint_base_name,
    #         metric="toxicity",
    #         color=color,
    #         darker_color=darker_color,
    #         corrupted_color=corrupted_color,
    #         darker_corrupted_color=darker_corrupted_color,
    #         save=True,
    #         use_log_scale=False,
    #         save_as_pdf=False,
    #     )


if __name__ == "__main__":
    # Definitions for plotting
    base_model_name_list = [
        "Meta-Llama-3-8B-Instruct",
        "Mistral-7B-Instruct-v0.2",
        "gemma-1.1-7b-it",
    ]
    checkpoint_base_name_list = [
        "Llama-3-8B-ckpt",
        "Mistral-7B-Instruct-ckpt",
        "gemma-1.1-7b-it-ckpt",
    ]
    base_model_seed_list = ["seed1000" for i in base_model_name_list]
    checkpoints_list = [[i for i in range(1, 11)] for j in range(3)]
    seeds_list = [["seed1000" for i in ckpts] for ckpts in checkpoints_list]

    alternative_llama_seeds = [
        "seed4000",
        "seed4000",
        "seed7000",
        "seed7000",
        "seed6000",
        "seed7000",
        "seed6000",
        "seed7000",
        "seed7000",
        "seed5000",
    ]

    alternative_seeds_list = [alternative_llama_seeds, seeds_list[1], seeds_list[2]]
    alternative_llama_palette = sns.color_palette("viridis", 10)
    alternative_llama_palette = alternative_llama_palette[::-1]

    base_model_markers = ["X", "o", "s"]

    # for power_over_epsilon
    fold_sizes = [1000, 2000, 3000, 4000]

    # for alpha_over_sequences
    custom_colors = ["#94D2BD", "#EE9B00", "#BB3E03"]
    darker_custom_colors = ["#85BDAA", "#D28B00", "#A63703"]
    corrupted_model_custom_colors = ["#25453a", "#4f3300", "#3e1401"]
    darker_corrupted_model_custom_colors = ["#101e19", "#221600", "#1b0900"]

    checkpoint1_palette = ["red", "green", "blue"]
    checkpoint1_markers = ["X", "X", "X"]
    checkpoint1_models = ["Llama-3-8B-ckpt1" for i in range(3)]
    checkpoint1_seeds1 = ["seed1000" for i in range(3)]
    checkpoint1_seeds2 = ["seed4000", "seed5000", "seed7000"]

    # checkpoint1_list = [1, 1, 1, 1]
    # checkpoint1_seeds_list = ["seed1000", "seed4000", "seed5000", "seed7000"]
    # df = get_alpha_wrapper(
    #     "Meta-Llama-3-8B-Instruct",
    #     "seed1000",
    #     checkpoint1_list,
    #     checkpoint1_seeds_list,
    #     checkpoint_base_name="Llama-3-8B-ckpt",
    #     fold_sizes=fold_sizes,
    #     marker="X",
    # )

    # for bm_name, bm_seed, ckpt_list, seed_list, ckpt_bm, marker in zip(
    #     base_model_name_list,
    #     base_model_seed_list,
    #     checkpoints_list,
    #     alternative_seeds_list,
    #     checkpoint_base_name_list,
    #     base_model_markers,
    # ):
    #     plot_power_over_number_of_sequences(
    #         bm_name,
    #         bm_seed,
    #         ckpt_list,
    #         seed_list,
    #         checkpoint_base_name=ckpt_bm,
    #         group_by="Empirical Wasserstein Distance",
    #         marker=marker,
    #     )

    #     plot_power_over_epsilon(
    #         bm_name,
    #         bm_seed,
    #         ckpt_list,
    #         seed_list,
    #         checkpoint_base_name=ckpt_bm,
    #         fold_sizes=[1000, 2000, 3000, 4000],
    #         marker=marker,
    #     )

    # plot_alpha_over_sequences(
    #     base_model_name_list, base_model_seed_list, ["seed2000", "seed2000", "seed2000"]
    # )

    # plot_rejection_rate_matrix(
    #     ["Llama-3-8B-ckpt1" for i in range(4)],
    #     ["seed1000", "seed4000", "seed5000", "seed7000"],
    # )

    # df = get_power_over_sequences_for_models_or_checkpoints(
    #     base_model_name_list[2],
    #     "seed1000",
    #     "seed2000",
    #     model_name2=base_model_name_list[2],
    # )

    # for i, seed in enumerate(alternative_llama_seeds):
    #     dist = get_distance_scores(
    #         "Meta-Llama-3-8B-Instruct",
    #         "seed1000",
    #         seed,
    #         model_name2=f"Llama-3-8B-ckpt{i+1}",
    #     )
    #     print(f"Distance for model Llama-3-8B-ckpt{i+1}_{seed}: {dist:.5f}")

    # for i in range(10):
    #     dist = get_distance_scores(
    #         base_model_name_list[1],
    #         "seed1000",
    #         "seed1000",
    #         checkpoint=i + 1,
    #         checkpoint_base_name=checkpoint_base_name_list[1],
    #     )
    #     print(
    #         f"Distance for model {checkpoint_base_name_list[1]}{i+1}_seed1000: {dist:.5f}"
    #     )

    # for i in range(10):
    #     dist = get_distance_scores(
    #         base_model_name_list[2],
    #         "seed1000",
    #         "seed1000",
    #         checkpoint=i + 1,
    #         checkpoint_base_name=checkpoint_base_name_list[2],
    #     )
    #     print(
    #         f"Distance for model {checkpoint_base_name_list[2]}{i+1}_seed1000: {dist:.5f}"
    #     )

    # model_name1 = "Mistral-7B-Instruct-v0.2"
    # seed1 = "seed1000"
    # seed2 = "seed2000"

    # plot_scores_two_models(model_name1, seed1, model_name1, seed2)

    # Prepare the data
    # data = {
    #     "Model": [
    #         "Llama-3-8B",
    #         "Llama-3-8B",
    #         "Llama-3-8B",
    #         "Llama-3-8B",
    #         "Llama-3-8B",
    #         "Llama-3-8B",
    #         "Llama-3-8B",
    #         "Llama-3-8B",
    #         "Llama-3-8B",
    #         "Llama-3-8B",
    #         "Mistral-7B-Instruct",
    #         "Mistral-7B-Instruct",
    #         "Mistral-7B-Instruct",
    #         "Mistral-7B-Instruct",
    #         "Mistral-7B-Instruct",
    #         "Mistral-7B-Instruct",
    #         "Mistral-7B-Instruct",
    #         "Mistral-7B-Instruct",
    #         "Mistral-7B-Instruct",
    #         "Mistral-7B-Instruct",
    #         "gemma-1.1-7b-it",
    #         "gemma-1.1-7b-it",
    #         "gemma-1.1-7b-it",
    #         "gemma-1.1-7b-it",
    #         "gemma-1.1-7b-it",
    #         "gemma-1.1-7b-it",
    #         "gemma-1.1-7b-it",
    #         "gemma-1.1-7b-it",
    #         "gemma-1.1-7b-it",
    #         "gemma-1.1-7b-it",
    #     ],
    #     "Checkpoint": [
    #         1,
    #         2,
    #         3,
    #         4,
    #         5,
    #         6,
    #         7,
    #         8,
    #         9,
    #         10,
    #         1,
    #         2,
    #         3,
    #         4,
    #         5,
    #         6,
    #         7,
    #         8,
    #         9,
    #         10,
    #         1,
    #         2,
    #         3,
    #         4,
    #         5,
    #         6,
    #         7,
    #         8,
    #         9,
    #         10,
    #     ],
    #     "Distance": [
    #         0.00013,
    #         0.00219,
    #         0.00543,
    #         0.00382,
    #         0.00439,
    #         0.00444,
    #         0.00420,
    #         0.00498,
    #         0.00497,
    #         0.00494,
    #         0.00024,
    #         0.00658,
    #         0.02112,
    #         0.00746,
    #         0.00704,
    #         0.01111,
    #         0.00709,
    #         0.00709,
    #         0.00899,
    #         0.00750,
    #         0.00027,
    #         0.00474,
    #         0.01683,
    #         0.01333,
    #         0.00742,
    #         0.00658,
    #         0.00841,
    #         0.00903,
    #         0.01378,
    #         0.01325,
    #     ],
    # }

    # df = pd.DataFrame(data)

    # # Create the plot
    # plt.figure(figsize=(14, 7))
    # sns.lineplot(data=df, x="Checkpoint", y="Distance", hue="Model", marker="o")
    # plt.title("Distance over Checkpoints for Different Model Families")
    # plt.xlabel("Checkpoint")
    # plt.ylabel("Distance")
    # plt.legend(title="Model")
    # plt.grid(True)
    # plt.savefig(
    #     "model_outputs/distance_over_checkpoints.pdf", bbox_inches="tight", format="pdf"
    # )

    #plot_all()
    
    model_name1 = "Meta-Llama-3-8B-Instruct"
    model_name2 = "Llama-3-8B-ckpt3"
    seed1 = "seed1000"
    seed2 = "seed1000"
    metric = "perspective"
    
    net_cfg = load_config("config.yml")
    train_cfg = TrainCfg()
    
    dist_dict = get_distance_scores(model_name1, seed1, seed2, model_name2=model_name2, metric=metric, net_cfg=net_cfg, train_cfg=train_cfg)
    print(dist_dict)
