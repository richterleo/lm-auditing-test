import pandas as pd
import json
import sys
import os

from pathlib import Path
from typing import Union, List

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from behavior_evaluation.distance_and_variation import empirical_wasserstein_distance_p1


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
):
    """ """
    file_path = f"model_outputs/{base_model_name}_{base_model_seed}_{checkpoint_base_name}{ckpt}_{seed}/kfold_test_results.csv"
    data = pd.read_csv(file_path)
    column_values = data[column_name]

    range_values = column_values.max() - column_values.min()
    average_value = column_values.mean()

    print(
        f"Range of {column_name}: {range_values}, with the min: {column_values.min()} and max: {column_values.max()}"
    )
    print(f"Average of {column_name}: {average_value}")

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
    sequence_counts = {sequence: 0 for sequence in range(max_sequences + 1)}

    # Iterate over each fold number
    for fold in unique_fold_numbers:
        fold_data = indexed_df[indexed_df["fold_number"] == fold]
        for sequence in range(max_sequences + 1):  # sequence_counts.keys()
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
    elif group_by == "Rank based on Wasserstein Distance":
        result_df = get_power_over_epsilon(
            base_model_name,
            base_model_seed,
            checkpoints,
            seeds,
            checkpoint_base_name=checkpoint_base_name,
        )

    result_df = result_df.reset_index()

    if print_df:
        pd.set_option("display.max_rows", None)
        print(result_df)

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=result_df,
        x="Sequence",
        y="Power",
        hue=group_by,
        marker="o",
        palette="tab10",
    )

    # Customize the plot
    plt.title(
        f"Power vs. Sequence grouped by {group_by} for {base_model_name} and checkpoints"
    )
    plt.xlabel("Sequence")
    plt.ylabel("Power")
    plt.legend(
        title="Checkpoints" if group_by == "Checkpoints" else "Rank",
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )
    plt.grid(True)

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
            emp_wasserstein_dist = empirical_wasserstein_distance_p1(
                scores_base, scores_ckpt
            )

            result_df = get_df_for_checkpoint(
                base_model_name,
                base_model_seed,
                ckpt,
                seed,
                checkpoint_base_name,
                column_name=column_name,
                max_sequences=max_sequences,
            )

            result_df["Empirical Wasserstein Distance"] = emp_wasserstein_dist
            result_dfs.append(result_df)

        except FileNotFoundError:
            print(f"File for checkpoint {ckpt} does not exist yet")

    final_df = pd.concat(result_dfs, ignore_index=True)
    final_df["Rank based on Wasserstein Distance"] = (
        final_df["Empirical Wasserstein Distance"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    indexed_final_df = final_df.set_index("Rank based on Wasserstein Distance")

    return indexed_final_df


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
):
    pass


if __name__ == "__main__":
    base_model_name = "Meta-Llama-3-8B-Instruct"
    base_model_seed = "seed1000"
    checkpoint_base_name = "LLama-3-8B-ckpt"
    checkpoints = [i for i in range(1, 11)]
    seeds = ["seed1000" for i in checkpoints]

    plot_power_over_number_of_sequences(
        base_model_name,
        base_model_seed,
        checkpoints,
        seeds,
        checkpoint_base_name=checkpoint_base_name,
        save=True,
        group_by="Rank based on Wasserstein Distance",
    )
