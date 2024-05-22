import pandas as pd
import json
import numpy as np
import sys
import os

from pathlib import Path
from typing import Union, List

import matplotlib.pyplot as plt
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
        marker="X",
        markersize=10,
        palette=palette,
    )

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Customize the plot
    plt.xlabel("samples", fontsize=14)
    plt.ylabel("power", fontsize=14)
    if group_by == "Checkpoints":
        title = "checkpoints"
    elif group_by == "Rank based on Wasserstein Distance":
        title = "rank"
    elif group_by == "Empirical Wasserstein Distance":
        title = "distance"
    plt.legend(
        title=title,
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )
    plt.grid(True, linewidth=0.5)

    # Making the box around the plot thicker
    plt.gca().spines["top"].set_linewidth(1.5)
    plt.gca().spines["right"].set_linewidth(1.5)
    plt.gca().spines["bottom"].set_linewidth(1.5)
    plt.gca().spines["left"].set_linewidth(1.5)

    if save:
        directory = f"model_outputs/{base_model_name}_{base_model_seed}_{checkpoint_base_name}_checkpoints"
        if not Path(directory).exists():
            Path(directory).mkdir(parents=True, exist_ok=True)
        plt.savefig(
            f"{directory}/power_over_number_of_sequences_grouped_by_{group_by}.png",
            dpi=300,
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

    custom_palette = ["midnightblue", "#94D2BD", "#EE9B00", "#BB3E03"]

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        x=f"Empirical {distance_measure} Distance",
        y="Power",
        hue="Samples per Test" if "Samples per Test" in smaller_df.columns else None,
        # style="Samples per Test" if "Samples per Test" in smaller_df.columns else None,
        marker="X",
        data=smaller_df,
        markersize=10,
        palette=custom_palette,
    )

    plt.xlabel(f"{distance_measure.lower()} distance", fontsize=14)
    plt.ylabel("power", fontsize=14)
    plt.grid(True, linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Make the surrounding box thicker
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    if save:
        directory = f"model_outputs/{base_model_name}_{base_model_seed}_{checkpoint_base_name}_checkpoints"
        if not Path(directory).exists():
            Path(directory).mkdir(parents=True, exist_ok=True)
        if "Samples per Test" in smaller_df.columns:
            plt.savefig(
                f"{directory}/power_over_{distance_measure.lower()}_distance_grouped_by_fold_size.png",
                dpi=300,
            )
        plt.savefig(
            f"{directory}/power_over_{distance_measure.lower()}_distance.png",
            dpi=300,
        )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x=f"Rank based on {distance_measure} Distance",
        y="Power",
        hue="Samples per Test" if "Samples per Test" in smaller_df.columns else None,
        # style="Samples per Test" if "Samples per Test" in smaller_df.columns else None,
        marker="o",
        data=smaller_df,
        markersize=10,
        palette=custom_palette,
    )
    plt.xlabel(f"rank based on {distance_measure.lower()} distance", fontsize=14)
    plt.ylabel("power", fontsize=14)
    plt.grid(True, linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Make the surrounding box thicker
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    if save:
        directory = f"model_outputs/{base_model_name}_{base_model_seed}_{checkpoint_base_name}_checkpoints"
        if not Path(directory).exists():
            Path(directory).mkdir(parents=True, exist_ok=True)
        if "Samples per Test" in smaller_df.columns:
            plt.savefig(
                f"{directory}/power_over_{distance_measure.lower()}_rank_grouped_by_fold_size.png",
                dpi=300,
            )
        plt.savefig(
            f"{directory}/power_over_{distance_measure.lower()}_rank.png",
            dpi=300,
        )


def get_alpha(model_name, seed1, seed2, max_sequences=41, fold_size=None, bs=96):
    """ """
    if not fold_size:
        fold_size = 4000

    max_sequences = (fold_size + bs - 1) // bs
    if fold_size == 4000:
        file_path = f"model_outputs/{model_name}_{seed1}_{model_name}_{seed2}/kfold_test_results.csv"
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


def get_alpha_wrapper(model_names, seeds1, seeds2, max_sequences=41):
    if not isinstance(model_names, list):
        return get_alpha(model_names, seeds1, seeds2, max_sequences)

    else:
        result_dfs = []
        for model_name, seed1, seed2 in zip(model_names, seeds1, seeds2):
            result_df = get_alpha(model_name, seed1, seed2, max_sequences)
            result_df["model_id"] = model_name
            result_dfs.append(result_df)

    final_df = pd.concat(result_dfs, ignore_index=True)
    return final_df


def plot_alpha_over_sequences(model_names, seeds1, seeds2, save=True, print_df=False):
    result_df = get_alpha_wrapper(model_names, seeds1, seeds2)
    result_df = result_df.reset_index()
    result_df["Samples"] = result_df["Sequence"] * 96

    group_by_model = "model_id" in result_df.columns

    if print_df:
        pd.set_option("display.max_rows", None)

    custom_palette = ["#94D2BD", "#EE9B00", "#BB3E03"]

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=result_df,
        x="Samples",
        y="Power",
        hue="model_id" if group_by_model else None,
        marker="X",
        markersize=10,
        palette=custom_palette,
    )

    # Customize the plot
    plt.xlabel("samples", fontsize=14)
    plt.ylabel("alpha error", fontsize=14)
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
            bbox_to_anchor=(
                1.05,
                1,
            ),  # Adjusted position to ensure the legend is outside the plot area
        )
    plt.grid(True, linewidth=0.5)

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

        fig_path += ".png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")


def plot_scores(model_name, seed, metric="toxicity", save=True, epoch=0):
    """ """
    directory = f"model_outputs/{model_name}_{seed}"
    file_path = f"{directory}/{metric}_scores.json"
    with open(file_path, "r") as f:
        data = json.load(f)

    scores = data[str(epoch)][f"{metric}_scores"]

    # Calculate statistics
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    plt.figure(figsize=(14, 7))

    # Plot histogram with adjusted bins and density plot
    sns.histplot(scores, bins=50, kde=True, color="blue", edgecolor="black", alpha=0.7)

    # Add mean and std lines
    plt.axvline(
        mean_score,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_score:.2f}",
    )
    plt.axvline(
        mean_score + std_score,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"+1 Std Dev: {mean_score + std_score:.2f}",
    )
    plt.axvline(
        mean_score - std_score,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"-1 Std Dev: {mean_score - std_score:.2f}",
    )

    # Set plot limits
    plt.xlim(0, 1)

    plt.title(
        f"Distribution of {metric.capitalize()} Scores for {model_name} (Seed: {seed})",
        fontsize=16,
    )
    plt.xlabel(f"{metric.capitalize()} Score", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend()
    plt.grid(True, linewidth=0.5)

    if save:
        output_path = os.path.join(directory, f"{metric}_scores.png")
        plt.savefig(output_path, dpi=300)
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
    plt.xlabel(f"{metric.lower()} score", fontsize=14)
    plt.ylabel("frequency", fontsize=14)
    plt.grid(True, linewidth=0.25)

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
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    base_model_name = "Meta-Llama-3-8B-Instruct"
    base_model_seed = "seed1000"
    checkpoint_base_name = "LLama-3-8B-ckpt"
    checkpoints = [i for i in range(1, 11)]
    seeds = ["seed1000" for i in checkpoints]

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

    model_names_for_dist_plot = ["Meta-Llama-3-8B-Instruct"]
    checkpoint_list = [f"LLama-3-8B-ckpt{i}" for i in range(1, 11)]
    model_names_for_dist_plot.extend(checkpoint_list)
    seeds_for_dist_plot = ["seed1000" for i in model_names_for_dist_plot]

    fold_sizes = [1000, 2000, 3000, 4000]

    plot_power_over_number_of_sequences(
        base_model_name,
        base_model_seed,
        checkpoints,
        seeds,
        checkpoint_base_name=checkpoint_base_name,
        save=True,
        group_by="Empirical Wasserstein Distance",  # "Rank based on Wasserstein Distance",
    )

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

    plot_alpha_over_sequences(model_names, seed1s, seed2s, print_df=True)
    # plot_scores(model_name, seed1, metric="toxicity", save=True)
    plot_scores_multiple_models(model_names_for_dist_plot, seeds_for_dist_plot)

    plot_power_over_epsilon(
        base_model_name, base_model_seed, checkpoints, seeds, fold_sizes=fold_sizes
    )
