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
from evaluation.distance import (
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

from analyze import (
    extract_data_for_models,
    get_power_over_sequences_from_whole_ds,
    get_power_over_sequences_for_models_or_checkpoints,
    get_power_over_sequences_for_checkpoints,
    get_distance_scores,
    get_matrix_for_models,
    get_power_over_sequences_for_ranked_checkpoints,
    get_power_over_sequences_for_ranked_checkpoints_wrapper,
    extract_power_from_sequence_df,
    get_alpha_wrapper,
)


def distance_box_plot(
    df,
    model_name1,
    seed1,
    seed2,
    model_name2,
    num_samples=100000,
    pre_shuffled=False,
    metric="perspective",
    plot_dir: str = "test_outputs",
):
    """ """

    # Create a box plot
    plt.figure(figsize=(10, 6))
    df.boxplot()
    plt.title("Box Plot of Different Methods to calculate Distance")
    plt.ylabel("Distance")
    plt.xticks(rotation=45)
    plt.grid(True)

    if pre_shuffled:
        file_path = f"{plot_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}/{metric}_distance_box_plot_{num_samples}_preshuffled.pdf"
    else:
        file_path = f"{plot_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}/{metric}_distance_box_plot_{num_samples}.pdf"

    plt.savefig(
        file_path,
        bbox_inches="tight",
        format="pdf",
    )


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
    model_name1 = "Meta-Llama-3-8B-Instruct"
    model_name2 = "Llama-3-8B-ckpt3"
    seed1 = "seed1000"
    seed2 = "seed1000"
    metric = "perspective"

    net_cfg = load_config("config.yml")
    train_cfg = TrainCfg()

    num_samples = 10000
    pre_shuffle = True

    # data = []

    # full_dict = []
    # for rand_val in range(10):
    #     dist_dict = get_distance_scores(
    #         model_name1,
    #         seed1,
    #         seed2,
    #         model_name2=model_name2,
    #         metric=metric,
    #         net_cfg=net_cfg,
    #         train_cfg=train_cfg,
    #         pre_shuffle=pre_shuffle,
    #         random_seed=rand_val,
    #         num_samples=num_samples,
    #     )
    #     full_dict.append(dist_dict)
    #     print(dist_dict)

    # # averages = {}
    # # for key in dist_dict.keys():
    # #     values = [d[key] for d in full_dict]
    # #     average = sum(values) / len(values)
    # #     averages[key] = average
    # # print(averages)

    # df = pd.DataFrame(full_dict)

    # # Create the plot
    # distance_box_plot(
    #     df,
    #     model_name1,
    #     seed1,
    #     seed2,
    #     model_name2,
    #     num_samples,
    #     metric="perspective",
    #     pre_shuffled=pre_shuffle,
    # )

    # for i in range(5):
    #     dist_dict = get_distance_scores(
    #         model_name1,
    #         seed1,
    #         seed2,
    #         model_name2=model_name2,
    #         metric=metric,
    #         net_cfg=net_cfg,
    #         train_cfg=train_cfg,
    #         pre_shuffle=pre_shuffle,
    #         random_seed=i,
    #         num_samples=num_samples,
    #     )
    #     data.append(dist_dict)

    # # Convert the list of dictionaries to a DataFrame
    # df = pd.DataFrame(data)

    # # Create the plot
    # distance_box_plot(
    #     df,
    #     model_name1,
    #     seed1,
    #     seed2,
    #     model_name2,
    #     num_samples,
    #     metric="perspective",
    #     pre_shuffled=pre_shuffle,
    # )

    # ns_data = []

    # nn = 0
    # ws = 0
    # nn_shuffled = 0
    # ws_scipy = 0

    # for i in range(1, 11):
    #     dist_dict = get_distance_scores(
    #         model_name1,
    #         seed1,
    #         seed2,
    #         model_name2=model_name2,
    #         metric=metric,
    #         net_cfg=net_cfg,
    #         train_cfg=train_cfg,
    #         distance_measures=["NeuralNet", "Wasserstein"],
    #         pre_shuffle=pre_shuffle,
    #         random_seed=i,
    #         num_samples=num_samples,
    #     )

    #     nn += dist_dict["NeuralNet"]
    #     nn_shuffled += dist_dict["NeuralNet_shuffled"]
    #     ws += dist_dict["Wasserstein"]
    #     ws_scipy += dist_dict["Wasserstein_scipy"]

    # ns_data.append(
    #     {
    #         # "num_samples": ns * 10000,
    #         "NeuralNet": nn / 10,
    #         "NeuralNet_shuffled": nn_shuffled / 10,
    #         "Wasserstein": ws / 10,
    #         "Wasserstein_scipy": ws_scipy / 10,
    #     }
    # )

    # ns_df = pd.DataFrame(ns_data)

    # # # Create the plot
    # distance_box_plot(
    #     ns_df,
    #     model_name1,
    #     seed1,
    #     seed2,
    #     model_name2,
    #     num_samples,
    #     metric="perspective",
    #     pre_shuffled=pre_shuffle,
    # )

    # Create the plot

    # Sample data generation for demonstration purposes (replace with your actual data collection code)
    # Melt the dataframe for easier plotting with seaborn
    # ns_df_melted = ns_df.melt(
    #     id_vars="num_samples",
    #     value_vars=[
    #         "NeuralNet",
    #         "NeuralNet_shuffled",
    #         "Wasserstein",
    #         "Wasserstein_scipy",
    #     ],
    #     var_name="Distance Type",
    #     value_name="Distance",
    # )

    # # Plotting with seaborn
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(
    #     data=ns_df_melted,
    #     x="num_samples",
    #     y="Distance",
    #     hue="Distance Type",
    #     marker="o",
    # )

    # plt.xlabel("Number of Samples")
    # plt.ylabel("Distance")
    # plt.title("Distances over Number of Samples")
    # plt.grid(True)
    # plt.savefig(
    #     "Wasserstein_neural_net_distance_vs_num_samples.pdf",
    #     bbox_inches="tight",
    #     format="pdf",
    # )

    # ws_data = []

    # for ns in range(1, 101):
    #     wasserstein = 0
    #     wasserstein_scipy = 0
    #     for i in range(5, 10):
    #         dist_dict = get_distance_scores(
    #             model_name1,
    #             seed1,
    #             seed2,
    #             model_name2=model_name2,
    #             metric=metric,
    #             net_cfg=net_cfg,
    #             train_cfg=train_cfg,
    #             distance_measures=["Wasserstein"],
    #             random_seed=i,
    #             num_samples=ns * 1000,
    #         )

    #         wasserstein += dist_dict["Wasserstein"]
    #         wasserstein_scipy += dist_dict["Wasserstein_scipy"]

    #     ws_data.append(
    #         {
    #             "num_samples": ns * 1000,
    #             "Wasserstein": wasserstein / 5,
    #             "Wasserstein_scipy": wasserstein_scipy / 5,
    #         }
    #     )

    # ws_df = pd.DataFrame(ws_data)
    # ws_df.to_json("wasserstein_distance_vs_num_samples.json", orient="records")

    # # Create the plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(ws_df["num_samples"], ws_df["Wasserstein"], marker="o")
    # plt.xlabel("Number of Samples")
    # plt.ylabel("Wasserstein Distance")
    # plt.title("Wasserstein Distance vs Number of Samples")
    # plt.grid(True)
    # plt.savefig(
    #     "wasserstein_distance_vs_num_samples.pdf", format="pdf", bbox_inches="tight"
    # )

    res_df = get_power_over_sequences_for_models_or_checkpoints(
        model_name1, seed1, seed2, model_name2=model_name2, epsilon=0.02
    )

    power_df = extract_power_from_sequence_df(
        res_df, distance_measure=None, by_checkpoints=False
    )
    print(power_df)
