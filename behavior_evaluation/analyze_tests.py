import pandas as pd
import json
import numpy as np
import sys
import os

from pathlib import Path
from scipy.stats import skew
from typing import Union, List

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from behavior_evaluation.distance_and_variation import (
    empirical_wasserstein_distance_p1,
    kolmogorov_variation,
    calc_tot_discrete_variation,
)


def get_average_time_until_end_of_experiment(model_name1, seed1, model_name2, seed2):
    file_path = f"model_outputs/{model_name1}_{seed1}_{model_name2}_{seed2}/kfold_test_results.csv"
    data = pd.read_csv(file_path)

    column_name = "sequences_until_end_of_experiment"
    column_values = data[column_name]

    range_values = column_values.max() - column_values.min()
    average_value = column_values.mean()

    print(f"Range of {column_name}: {range_values}")
    print(f"Average of {column_name}: {average_value}")


def get_df_for_checkpoint(
    base_model_name,
    base_model_seed,
    ckpt,
    seed,
    checkpoint_base_name,
    column_name="sequences_until_end_of_experiment",
    max_sequences=41,
    fold_size=None,
    bs=96,
):
    """ """
    if not fold_size:
        fold_size = 4000
    if fold_size == 4000:
        file_path = f"model_outputs/{base_model_name}_{base_model_seed}_{checkpoint_base_name}{ckpt}_{seed}/kfold_test_results.csv"
        if not Path(file_path).exists():
            file_path = f"model_outputs/{base_model_name}_{base_model_seed}_{checkpoint_base_name}{ckpt}_{seed}/kfold_test_results_{fold_size}.csv"
    else:
        file_path = f"model_outputs/{base_model_name}_{base_model_seed}_{checkpoint_base_name}{ckpt}_{seed}/kfold_test_results_{fold_size}.csv"

    max_sequences = (fold_size + bs - 1) // bs
    data = pd.read_csv(file_path)
    column_values = data[column_name]

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

    result_df["Checkpoint"] = ckpt
    result_df["Samples per Test"] = fold_size

    return result_df


def get_power_over_number_of_sequences(
    base_model_name: Union[str, List[str]],
    base_model_seed: Union[str, List[str]],
    checkpoints: Union[str, List[str]],
    seeds: Union[str, List[str]],
    checkpoint_base_name: str = "LLama-3-8B-ckpt",
    max_sequences: int = 41,
    column_name="sequences_until_end_of_experiment",
    reorder=False,
):
    if not isinstance(checkpoints, list):
        checkpoints = [checkpoints]
    if not isinstance(seeds, list):
        seeds = [seeds]

    result_dfs = []

    for ckpt, seed in zip(checkpoints, seeds):
        print(
            f"Base_model: {base_model_name}, base_model_seed: {base_model_seed}, checkpoint: {checkpoint_base_name}{ckpt}, seed: {seed}"
        )
        try:
            result_df = get_df_for_checkpoint(
                base_model_name,
                base_model_seed,
                ckpt,
                seed,
                checkpoint_base_name,
                column_name=column_name,
                max_sequences=max_sequences,
            )

            result_dfs.append(result_df)

        except FileNotFoundError:
            print(f"File for checkpoint {ckpt} does not exist yet")

    final_df = pd.concat(result_dfs, ignore_index=True)
    indexed_final_df = final_df.set_index("Checkpoint")

    return indexed_final_df


def plot_power_over_number_of_sequences(
    base_model_name,
    base_model_seed,
    checkpoints,
    seeds,
    checkpoint_base_name="LLama-3-8B-ckpt",
    save=True,
    print_df=False,
    group_by="Checkpoint",
    marker=None,
    save_as_pdf=True,
):
    if group_by == "Checkpoint":
        result_df = get_power_over_number_of_sequences(
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
        result_df = get_power_over_epsilon(
            base_model_name,
            base_model_seed,
            checkpoints,
            seeds,
            checkpoint_base_name=checkpoint_base_name,
        )

        result_df = result_df.reset_index()
        result_df["Empirical Wasserstein Distance"] = result_df[
            "Empirical Wasserstein Distance"
        ].round(3)

    if print_df:
        pd.set_option("display.max_rows", None)
        print(result_df)

    result_df["Samples"] = result_df["Sequence"] * 96

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
        marker=marker if marker else "X",
        markersize=10,
        palette=palette,
    )

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Customize the plot
    plt.xlabel("samples", fontsize=16)
    plt.ylabel("detection frequency", fontsize=16)
    if group_by == "Checkpoints":
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
        directory = f"model_outputs/{base_model_name}_{base_model_seed}_{checkpoint_base_name}_checkpoints"
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


def get_power_over_epsilon(
    base_model_name,
    base_model_seed,
    checkpoints,
    seeds,
    checkpoint_base_name="LLama-3-8B-ckpt",
    epoch1=0,
    epoch2=0,
    metric="toxicity",
    column_name="sequences_until_end_of_experiment",
    max_sequences=41,
    distance_measure="Wasserstein",
    fold_size=None,
):
    if not isinstance(checkpoints, list):
        checkpoints = [checkpoints]
    if not isinstance(seeds, list):
        seeds = [seeds]

    result_dfs = []

    for ckpt, seed in zip(checkpoints, seeds):
        print(
            f"Base_model: {base_model_name}, base_model_seed: {base_model_seed}, checkpoint: {checkpoint_base_name}{ckpt}, seed: {seed}"
        )
        try:
            score_path_base_model = f"model_outputs/{base_model_name}_{base_model_seed}/{metric}_scores.json"
            score_path_checkpoint = f"model_outputs/{checkpoint_base_name}{ckpt}_{seed}/{metric}_scores.json"
            with open(score_path_base_model, "r") as f:
                scores_base_model = json.load(f)
            with open(score_path_checkpoint, "r") as f:
                scores_checkpoint = json.load(f)

            scores_base = scores_base_model[str(epoch1)][f"{metric}_scores"]
            scores_ckpt = scores_checkpoint[str(epoch2)][f"{metric}_scores"]

            if distance_measure == "Wasserstein":
                dist = empirical_wasserstein_distance_p1(scores_base, scores_ckpt)

            elif distance_measure == "Kolmogorov":
                dist = kolmogorov_variation(scores_base, scores_ckpt)
                # print(dist)

            result_df = get_df_for_checkpoint(
                base_model_name,
                base_model_seed,
                ckpt,
                seed,
                checkpoint_base_name,
                column_name=column_name,
                max_sequences=max_sequences,
                fold_size=fold_size,
            )

            result_df[f"Empirical {distance_measure} Distance"] = dist
            result_dfs.append(result_df)

        except FileNotFoundError:
            print(f"File for checkpoint {ckpt} does not exist yet")

    final_df = pd.concat(result_dfs, ignore_index=True)
    final_df[f"Rank based on {distance_measure} Distance"] = (
        final_df[f"Empirical {distance_measure} Distance"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    # indexed_final_df = final_df.set_index(f"Rank based on {distance_measure} Distance")

    return final_df


def get_power_over_epsilon_wrapper(
    base_model_name,
    base_model_seed,
    checkpoints,
    seeds,
    checkpoint_base_name="LLama-3-8B-ckpt",
    epoch1=0,
    epoch2=0,
    metric="toxicity",
    column_name="sequences_until_end_of_experiment",
    max_sequences=41,
    distance_measure="Wasserstein",
    fold_sizes=None,
):
    result_dfs = []

    for fold_size in fold_sizes:
        result_dfs.append(
            get_power_over_epsilon(
                base_model_name,
                base_model_seed,
                checkpoints,
                seeds,
                checkpoint_base_name=checkpoint_base_name,
                epoch1=epoch1,
                epoch2=epoch2,
                metric=metric,
                column_name=column_name,
                max_sequences=max_sequences,
                distance_measure=distance_measure,
                fold_size=fold_size,
            )
        )

    result_df = pd.concat(result_dfs)
    return result_df


def plot_power_over_epsilon(
    base_model_name,
    base_model_seed,
    checkpoints,
    seeds,
    checkpoint_base_name="LLama-3-8B-ckpt",
    epoch1=0,
    epoch2=0,
    metric="toxicity",
    column_name="sequences_until_end_of_experiment",
    max_sequences=41,
    save=True,
    distance_measure="Wasserstein",
    fold_sizes=None,
    marker=None,
    save_as_pdf=True,
):
    if fold_sizes:
        result_df = get_power_over_epsilon_wrapper(
            base_model_name,
            base_model_seed,
            checkpoints,
            seeds,
            checkpoint_base_name=checkpoint_base_name,
            epoch1=epoch1,
            epoch2=epoch2,
            metric=metric,
            column_name=column_name,
            max_sequences=max_sequences,
            distance_measure=distance_measure,
            fold_sizes=fold_sizes,
        )
    else:
        result_df = get_power_over_epsilon(
            base_model_name,
            base_model_seed,
            checkpoints,
            seeds,
            checkpoint_base_name=checkpoint_base_name,
            epoch1=epoch1,
            epoch2=epoch2,
            metric=metric,
            column_name=column_name,
            max_sequences=max_sequences,
            distance_measure=distance_measure,
        )

    if "Samples per Test" in result_df.columns:
        print("Samples is in the columns")
        last_entries = (
            result_df.groupby(["Samples per Test", "Checkpoint"]).last().reset_index()
        )
        smaller_df = last_entries.set_index("Samples per Test")[
            [
                "Checkpoint",
                "Power",
                f"Empirical {distance_measure} Distance",
                f"Rank based on {distance_measure} Distance",
            ]
        ].reset_index()
    else:
        last_entries = result_df.groupby("Checkpoint").last().reset_index()
        smaller_df = last_entries[
            [
                "Checkpoint",
                "Power",
                f"Empirical {distance_measure} Distance",
                f"Rank based on {distance_measure} Distance",
            ]
        ].reset_index()

    # custom_palette = ["midnightblue", "#94D2BD", "#EE9B00", "#BB3E03"]
    # custom_palette = ["Gamboge", "Alloy orange", "Rust", "Rufus"]
    custom_palette = ["#E49B0F", "#C46210", "#B7410E", "#A81C07"]

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        x=f"Empirical {distance_measure} Distance",
        y="Power",
        hue="Samples per Test" if "Samples per Test" in smaller_df.columns else None,
        # style="Samples per Test" if "Samples per Test" in smaller_df.columns else None,
        marker=marker if marker else "X",
        data=smaller_df,
        markersize=10,
        palette=custom_palette,
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
        directory = f"model_outputs/{base_model_name}_{base_model_seed}_{checkpoint_base_name}_checkpoints"
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
        marker=marker if marker else "X",
        data=smaller_df,
        markersize=10,
        palette=custom_palette,
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
        directory = f"model_outputs/{base_model_name}_{base_model_seed}_{checkpoint_base_name}_checkpoints"
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


def get_alpha(model_name, seed1, seed2, max_sequences=41, fold_size=4000, bs=96):
    """ """
    max_sequences = (fold_size + bs - 1) // bs
    if fold_size == 4000:
        file_path = f"model_outputs/{model_name}_{seed1}_{model_name}_{seed2}/kfold_test_results.csv"
        if not Path(file_path).exists():
            file_path = f"model_outputs/{model_name}_{seed1}_{model_name}_{seed2}/kfold_test_results_{fold_size}.csv"
    else:
        file_path = f"model_outputs/{model_name}_{seed1}_{model_name}_{seed2}/kfold_test_results_{fold_size}.csv"
    data = pd.read_csv(file_path)

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

    return result_df


def get_alpha_wrapper(model_names, seeds1, seeds2, max_sequences=41, fold_size=4000):
    if not isinstance(model_names, list):
        return get_alpha(model_names, seeds1, seeds2, max_sequences)

    else:
        result_dfs = []
        for model_name, seed1, seed2 in zip(model_names, seeds1, seeds2):
            result_df = get_alpha(
                model_name, seed1, seed2, max_sequences, fold_size=fold_size
            )
            result_df["model_id"] = model_name
            result_dfs.append(result_df)

    final_df = pd.concat(result_dfs, ignore_index=True)
    return final_df


# def plot_alpha_over_sequences(model_names, seeds1, seeds2, save=True, print_df=False):
#     result_df = get_alpha_wrapper(model_names, seeds1, seeds2)
#     result_df = result_df.reset_index()
#     result_df["Samples"] = result_df["Sequence"] * 96

#     print(f"Hello from inside the fct")

#     group_by_model = "model_id" in result_df.columns

#     if print_df:
#         pd.set_option("display.max_rows", None)

#     markers = ["o", "X", "s"]  # Different markers: circle, X, square

#     # Create the plot
#     plt.figure(figsize=(12, 6))
#     sns.lineplot(
#         data=result_df,
#         x="Samples",
#         y="Power",
#         hue="model_id" if group_by_model else None,
#         style="model_id" if group_by_model else None,
#         markers=markers,
#         dashes=False,  # No dashes, solid lines
#         color="black",
#     )

#     # Customize the plot
#     plt.xlabel("samples", fontsize=14)
#     plt.ylabel("false positive rate", fontsize=14)
#     # plt.tight_layout(rect=[0, 0, 0.85, 1])

#     # Adjust the spines (box) thickness
#     ax = plt.gca()
#     ax.spines["top"].set_linewidth(1.5)
#     ax.spines["right"].set_linewidth(1.5)
#     ax.spines["bottom"].set_linewidth(1.5)
#     ax.spines["left"].set_linewidth(1.5)

#     if group_by_model:
#         plt.legend(
#             title="models",
#             loc="upper left",
#             bbox_to_anchor=(
#                 1.05,
#                 1,
#             ),  # Adjusted position to ensure the legend is outside the plot area
#         )
#     plt.grid(True, linewidth=0.5)

#     if save:
#         directory = "model_outputs/alpha_plots"
#         if not Path(directory).exists():
#             Path(directory).mkdir(parents=True, exist_ok=True)
#         fig_path = f"{directory}/alpha_error_over_number_of_sequences"
#         if isinstance(model_names, str):
#             fig_path += f"_{model_names}"
#         elif isinstance(model_names, list):
#             for model_name in model_names:
#                 fig_path += f"_{model_name}"

#         fig_path += ".png"
#         plt.savefig(fig_path, dpi=300, bbox_inches="tight")


def plot_alpha_over_sequences(
    model_names,
    seeds1,
    seeds2,
    save=True,
    print_df=False,
    save_as_pdf=True,
    markers=["X", "o", "s"],
    palette=["#94D2BD", "#EE9B00", "#BB3E03"],
    fold_size=4000,
):
    result_df = get_alpha_wrapper(model_names, seeds1, seeds2, fold_size=fold_size)
    result_df = result_df.reset_index()
    result_df["Samples"] = result_df["Sequence"] * 96

    group_by_model = "model_id" in result_df.columns

    if print_df:
        pd.set_option("display.max_rows", None)

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
            loc="upper left",
            fontsize=14,
            title_fontsize=16,
            # bbox_to_anchor=(
            #     1.05,
            #     1,
            # ),  # Adjusted position to ensure the legend is outside the plot area
        )
    plt.grid(True, linewidth=0.5, color="#ddddee")

    if save:
        directory = "model_outputs/alpha_plots"
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


def plot_scores(
    model_name,
    seed,
    metric="toxicity",
    save=True,
    epoch=0,
    use_log_scale=True,
    color="blue",
    save_as_pdf=True,
):
    """ """
    directory = f"model_outputs/{model_name}_{seed}"
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


def plot_scores_multiple_models(
    model_names, seeds, metric="toxicity", save=True, epoch=0
):
    """ """
    all_scores = []

    for m_name, seed in zip(model_names, seeds):
        directory = f"model_outputs/{m_name}_{seed}"
        file_path = f"{directory}/{metric}_scores.json"
        with open(file_path, "r") as f:
            data = json.load(f)

        scores = data[str(epoch)][f"{metric}_scores"]

        # Append scores to a list with model name
        for score in scores:
            all_scores.append({"score": score, "model_name": m_name})

    # Convert to DataFrame
    df = pd.DataFrame(all_scores)

    plt.figure(figsize=(14, 7))

    palette = sns.color_palette("viridis", len(model_names) - 1)
    palette.append("red")
    palette = palette[::-1]

    # Plot histogram with seaborn using hue for model name, without kde and with a nice color palette
    hist_plot = sns.histplot(
        df,
        x="score",
        hue="model_name",
        bins=50,
        element="step",
        palette=palette,
        alpha=0.5,
    )

    # Set plot limits
    plt.xlim(0, 1)
    plt.yscale("log")

    # plt.title(
    #     f"Distribution of {metric.capitalize()} Scores for Multiple Models",
    #     fontsize=16,
    # )
    plt.xlabel(f"{metric.lower()} score", fontsize=16)
    plt.ylabel("frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linewidth=0.5, color="#ddddee")

    # handles, labels = hist_plot.get_legend_handles_labels()
    # print(handles, labels)

    # # Manually create the legend
    # # plt.legend(title="Models", loc="upper right", fontsize="small")
    # plt.legend(
    #     title="Models",
    #     loc="center left",
    #     bbox_to_anchor=(1, 0.5),
    #     fontsize="xx-small",
    # )

    ax = plt.gca()
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)

    if save:
        output_path = os.path.join("model_outputs", f"{metric}_scores_comparison.png")
        # plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
):
    if not epochs:
        epochs = [0 for i in checkpoints]

    directory = f"model_outputs/{base_model_name}_{base_model_seed}"
    file_path = f"{directory}/{metric}_scores.json"
    print(f"This is the original model: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)

    scores = data[str(base_model_epoch)][f"{metric}_scores"]
    scores_dict = {}
    wasserstein_distances = {}

    for ckpt, seed, epoch in zip(checkpoints, checkpoint_seeds, epochs):
        checkpoint_directory = f"model_outputs/{checkpoint_base_name}{ckpt}_{seed}"
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
    print(f"This is the max distance: {max_distance}")

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


if __name__ == "__main__":
    base_model_name = "Meta-Llama-3-8B-Instruct"
    base_model_seed = "seed1000"
    checkpoint_base_name = "Llama-3-8B-ckpt"
    checkpoints = [i for i in range(1, 11)]
    seeds = ["seed1000" for i in checkpoints]

    checkpoint_base_names = [
        "Llama-3-8B-ckpt",
        "Mistral-7B-Instruct-ckpt",
        "gemma-1.1-7b-it-ckpt",
    ]
    checkpoints_list = [[i for i in range(1, 11)], [2, 4, 6, 8, 10], [2, 4, 6, 8, 10]]
    seeds_list = [["seed1000" for i in ckpts] for ckpts in checkpoints_list]

    model_name = "Meta-Llama-3-8B-Instruct"
    seed1 = "seed1000"
    seed2 = "seed2000"

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    model_names = [
        "Meta-Llama-3-8B-Instruct",
        "Mistral-7B-Instruct-v0.2",
        "gemma-1.1-7b-it",
    ]
    seed1s = ["seed1000" for i in model_names]
    seed2s = ["seed2000" for i in model_names]

    markers = ["X", "o", "s"]

    model_names_for_dist_plot = ["Meta-Llama-3-8B-Instruct"]
    checkpoint_list = [f"Llama-3-8B-ckpt{i}" for i in range(1, 11)]
    model_names_for_dist_plot.extend(checkpoint_list)
    seeds_for_dist_plot = ["seed1000" for i in model_names_for_dist_plot]

    fold_sizes = [1000, 2000, 3000, 4000]
    custom_colors = ["#94D2BD", "#EE9B00", "#BB3E03"]
    darker_custom_colors = ["#85BDAA", "#D28B00", "#A63703"]
    corrupted_model_custom_colors = ["#25453a", "#4f3300", "#3e1401"]
    darker_corrupted_model_custom_colors = ["#101e19", "#221600", "#1b0900"]

    # plot_power_over_epsilon(
    #     base_model_name,
    #     base_model_seed,
    #     checkpoints,
    #     seeds,
    #     checkpoint_base_name=checkpoint_base_name,
    #     distance_measure="Kolmogorov",
    # )

    # plot_alpha_over_sequences(model_name, seed1, seed2)

    # df = get_power_over_epsilon(base_model_name, base_model_seed, checkpoints, seeds)
    # print(df)

    # plot_alpha_over_sequences(model_names, seed1s, seed2s, print_df=True)
    # for bm_name, color in zip(model_names, custom_colors):
    #     plot_scores(bm_name, "seed1000", metric="toxicity", save=True, color=color)
    #     plot_scores(
    #         bm_name,
    #         "seed1000",
    #         metric="toxicity",
    #         save=True,
    #         use_log_scale=False,
    #         color=color,
    #     )
    # # plot_scores_multiple_models(model_names_for_dist_plot, seeds_for_dist_plot)

    # for bm_name, ckpts, seeds, ckpt_name, marker in zip(
    #     model_names, checkpoints_list, seeds_list, checkpoint_base_names, markers
    # ):
    #     plot_power_over_epsilon(
    #         bm_name,
    #         "seed1000",
    #         ckpts,
    #         seeds,
    #         checkpoint_base_name=ckpt_name,
    #         fold_sizes=fold_sizes,
    #         marker=marker,
    #     )
    #     plot_power_over_number_of_sequences(
    #         bm_name,
    #         "seed1000",
    #         ckpts,
    #         seeds,
    #         checkpoint_base_name=ckpt_name,
    #         save=True,
    #         group_by="Empirical Wasserstein Distance",  # "Rank based on Wasserstein Distance",
    #         marker=marker,
    #     )

    # for (
    #     bm_name,
    #     bm_seed,
    #     checkpoints,
    #     seeds,
    #     checkpoint_base_name,
    #     color,
    #     darker_color,
    #     corrupted_color,
    #     darker_corrupted_color,
    # ) in zip(
    #     model_names,
    #     custom_colors,
    #     checkpoints_list,
    #     seeds_list,
    #     checkpoint_base_names,
    #     custom_colors,
    #     darker_custom_colors,
    #     corrupted_model_custom_colors,
    #     darker_corrupted_model_custom_colors,
    # ):
    #     plot_scores_base_most_extreme(
    #         bm_name,
    #         base_model_seed,
    #         checkpoints,
    #         seeds,
    #         checkpoint_base_name,
    #         metric="toxicity",
    #         color=color,
    #         darker_color=darker_color,
    #         corrupted_color=corrupted_color,
    #         darker_corrupted_color=darker_corrupted_color,
    #         save=True,
    #     )

    alpha_palette = sns.color_palette("viridis", 10)
    alpha_palette = alpha_palette[::-1]
    alpha_seeds = [
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

    alpha_markers = ["X" for i in range(10)]

    plot_alpha_over_sequences(
        checkpoint_list,
        seeds_for_dist_plot,
        alpha_seeds,
        palette=alpha_palette,
        markers=alpha_markers,
    )
