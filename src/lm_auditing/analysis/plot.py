import pandas as pd
import json
import logging
import numpy as np
import sys
import os

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from pathlib import Path
from scipy.stats import skew, wasserstein_distance
from typing import Union, List, Optional, Dict, Tuple, Literal

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import textwrap
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import to_rgb
import seaborn as sns
import colorsys
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase

# Add paths to sys.path if not already present
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lm_auditing.analysis.distance import (
    empirical_wasserstein_distance_p1,
    kolmogorov_variation,
    NeuralNetDistance,
    calc_tot_discrete_variation,
)
from lm_auditing.utils.utils import load_config
# from train_cfg import TrainCfg


from lm_auditing.analysis.analyze import (
    extract_data_for_models,
    get_power_over_sequences,
    get_distance_scores,
    get_power_over_sequences_for_checkpoints_and_folds,
    extract_power_from_sequence_df,
    get_alpha_wrapper,
    get_mean_and_std_for_nn_distance,
    get_mean_tox_scores,
)

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[3]

TASK_CLUSTER = [
    ["Program Execution", "Pos Tagging", "Mathematics"],
    [
        "Gender Classification",
        "Commonsense Classification",
        "Translation",
    ],
    [
        "Code to Text",
        "Stereotype Detection",
        "Sentence Perturbation",
    ],
    [
        "Text to Code",
        "Linguistic Probing",
        "Language Identification",
    ],
    [
        "Data to Text",
        "Word Semantics",
        "Question Rewriting",
    ],
]


def plot_power_over_number_of_sequences(
    base_model_name: str,
    base_model_seed: str,
    seeds: List[str],
    checkpoints: Optional[List[str]] = None,
    model_names: Optional[Union[str, List[str]]] = None,
    checkpoint_base_name: str = "Llama-3-8B-ckpt",
    group_by: str = "checkpoint",
    marker: str = "X",
    save_as_pdf: bool = True,
    test_dir: str = "test_outputs",
    plot_dir: str = "plots",
    metric: str = "perspective",
    only_continuations: bool=True,
    fold_size: int = 2000,
    epsilon: Union[float, List[float]] = 0,
    overwrite: bool = False,
    dir_prefix: Optional[str] = None,
    noise: float = 0,
    calc_distance: bool = False,
    distance_measure: str = "Wasserstein",
    rank: bool = False,
):
    if dir_prefix is None:
        dir_prefix = metric

    plot_dir = ROOT_DIR / dir_prefix / plot_dir
    test_dir = ROOT_DIR / dir_prefix / test_dir

    use_models = model_names is not None

    noise_string = f"_noise_{noise}" if noise > 0 else ""

    if group_by == "checkpoint" or use_models:
        if use_models:
            result_df = get_power_over_sequences(
                base_model_name,
                base_model_seed,
                seeds,
                model_names=model_names,
                only_continuations=only_continuations,
                fold_size=fold_size,
                epsilon=epsilon,
                test_dir=test_dir,
                metric=metric,
                dir_prefix=dir_prefix,
                noise=noise,
            )
        else:
            result_df = get_power_over_sequences(
                base_model_name,
                base_model_seed,
                seeds=seeds,
                checkpoints=checkpoints,
                checkpoint_base_name=checkpoint_base_name,
                only_continuations=only_continuations,
                fold_size=fold_size,
                epsilon=epsilon,
                test_dir=test_dir,
                metric=metric,
                dir_prefix=dir_prefix,
                noise=noise,
                calc_distance=calc_distance,
                distance_measure=distance_measure,
                rank=rank
            )

    # Create the plot
    plt.figure(figsize=(12, 6))
    if not use_models:
        unique_groups = result_df[group_by].unique()
        num_groups = len(unique_groups)
    else:
        num_groups = len(model_names)

    palette = sns.color_palette("viridis", num_groups)
    palette = palette[::-1]
    # palette = ["#8A2BE2"]  # for aya

    if group_by == "checkpoint":
        if calc_distance:
            hue = f"rank_by_{distance_measure.lower()}_distance"
        else:
            hue = "checkpoint"
    elif group_by == "model":
        hue = "model_name2"

    sns.lineplot(
        data=result_df,
        x="samples",
        y="test_positive",
        hue=hue,
        marker=marker,
        markersize=10,
        palette=palette,
    )

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)

    # Customize the plot
    plt.xlabel("samples", fontsize=18)
    plt.ylabel("proportion of triggered tests", fontsize=18)
    if group_by == "checkpoint":
        if calc_distance:
            title = "distance"
        else:
            title = "checkpoints"
    elif group_by == "model":
        title = "models"
    else:
        title = group_by

    # Modify legend labels if showing Wasserstein distances
    if calc_distance and distance_measure == "Wasserstein" and rank:
        # Get the current legend handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # Create new labels that only show the distance
        new_labels = []
        for label in labels:
            # Extract the rank number
            rank_num = int(label)
            # Get the corresponding distance value
            distance = result_df[result_df[f"rank_by_{distance_measure.lower()}_distance"] == rank_num][f"{distance_measure.lower()}_distance"].iloc[0]
            # Round the distance to 3 decimal places
            rounded_distance = round(distance, 3)
            # Create new label with just the distance
            new_labels.append(f"{rounded_distance}")
        
        # Create the legend with modified labels
        plt.legend(
            handles=handles,
            labels=new_labels,
            title=title,
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=14,
            title_fontsize=16,
        )
    else:
        plt.legend(
            title=title,
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=14,
            title_fontsize=16,
        )

    plt.grid(True, linewidth=0.5, color="#ddddee")

    # Making the box around the plot thicker
    plt.gca().spines["top"].set_linewidth(1.5)
    plt.gca().spines["right"].set_linewidth(1.5)
    plt.gca().spines["bottom"].set_linewidth(1.5)
    plt.gca().spines["left"].set_linewidth(1.5)

    if use_models:
        directory = f"{plot_dir}/{base_model_name}_{base_model_seed}_models"
        if not Path(directory).exists():
            Path(directory).mkdir(parents=True, exist_ok=True)
        else:
            for model_name in model_names:
                directory += f"_{model_name}"
            if not Path(directory).exists():
                Path(directory).mkdir(parents=True, exist_ok=True)

    else:
        directory = f"{plot_dir}/{base_model_name}_{base_model_seed}_{checkpoint_base_name}_checkpoints"
        if not Path(directory).exists():
            Path(directory).mkdir(parents=True, exist_ok=True)
        else:
            for seed in seeds:
                directory += f"_{seed}"
            if not Path(directory).exists():
                Path(directory).mkdir(parents=True, exist_ok=True)

    if save_as_pdf:
        if use_models:
            file_name = f"power_over_number_of_sequences{noise_string}.pdf"
        else:
            file_name = f"power_over_number_of_sequences_grouped_by_{group_by}_{base_model_name}_{base_model_seed}{noise_string}.pdf"
        plt.savefig(
            f"{directory}/{file_name}",
            bbox_inches="tight",
            format="pdf",
        )
    else:
        if use_models:
            file_name = f"power_over_number_of_sequences{noise_string}.png"
        else:
            file_name = f"power_over_number_of_sequences_grouped_by_{group_by}_{base_model_name}_{base_model_seed}{noise_string}.png"

        plt.savefig(
            f"{directory}/{file_name}",
            dpi=300,
            bbox_inches="tight",
        )

    logger.info(f"File saved at {directory}/{file_name}")

def plot_power_over_number_of_sequences_for_finetuned_models(
    base_model_name: str,
    base_model_seed: str,
    seeds: List[str],
    checkpoints: Optional[List[str]] = None,
    model_names: Optional[Union[str, List[str]]] = None,
    checkpoint_base_name: str = "Llama-3-8B-ckpt",
    group_by: str = "checkpoint",
    marker: Union[str, List[str], Dict[str, str]] = "X",
    save_as_pdf: bool = True,
    test_dir: str = "test_outputs",
    plot_dir: str = "plots",
    metric: str = "perspective",
    only_continuations: bool = True,
    fold_size: int = 2000,
    epsilon: Union[float, List[float]] = 0,
    palette: Optional[Union[List[str], Dict[str, str]]] = None,
    sizes: Optional[Dict[str, int]] = None,
    line_styles: Optional[Dict[str, Union[str, Tuple[int, int]]]] = None,
    model_labels: Optional[List[str]] = None,
    model_name_to_label: Optional[Dict[str, str]] = None,
    dir_prefix: Optional[str] = None,
    noise: float = 0,
    calc_distance: bool = False,
    distance_measure: str = "Wasserstein",
    rank: bool = False,
):
    if dir_prefix is None:
        dir_prefix = metric

    plot_dir = ROOT_DIR / dir_prefix / plot_dir
    test_dir = ROOT_DIR / dir_prefix / test_dir

    use_models = model_names is not None

    noise_string = f"_noise_{noise}" if noise > 0 else ""

    # Retrieve the result DataFrame based on the parameters
    if group_by == "checkpoint" or use_models:
        if use_models:
            result_df = get_power_over_sequences(
                base_model_name,
                base_model_seed,
                seeds,
                model_names=model_names,
                only_continuations=only_continuations,
                fold_size=fold_size,
                test_dir=test_dir,
                dir_prefix=dir_prefix,
                metric=metric,
                epsilon=epsilon,
                noise=noise,
            )
        else:
            result_df = get_power_over_sequences(
                base_model_name,
                base_model_seed,
                seeds=seeds,
                checkpoints=checkpoints,
                checkpoint_base_name=checkpoint_base_name,
                only_continuations=only_continuations,
                fold_size=fold_size,
                test_dir=test_dir,
                dir_prefix=dir_prefix,
                metric=metric,
                epsilon=epsilon,
                noise=noise,
                calc_distance=calc_distance,
                distance_measure=distance_measure,
                rank=rank,
            )
    # Map model names to custom labels
    if use_models:
        if model_name_to_label is None:
            model_name_to_label = dict(zip(model_names, model_names))
        result_df["label"] = result_df["model_name2"].map(model_name_to_label)
        if model_labels is None:
            model_labels = [model_name_to_label[name] for name in model_names]
    else:
        result_df["label"] = result_df[group_by]

    # Prepare plotting parameters
    if palette:
        if isinstance(palette, dict):
            custom_palette = palette
        else:
            custom_palette = dict(zip(model_labels, palette))
    else:
        palette = sns.color_palette("viridis", len(model_labels))
        custom_palette = dict(zip(model_labels, palette))

    if isinstance(marker, dict):
        markers = marker
    elif isinstance(marker, list):
        markers = dict(zip(model_labels, marker))
    else:
        markers = {label: marker for label in model_labels}

    if line_styles and isinstance(line_styles, dict):
        dashes = line_styles
    else:
        dashes = None

    if sizes and isinstance(sizes, dict):
        sizes = sizes
    else:
        sizes = None

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=result_df,
        x="samples",
        y="test_positive",
        hue="label",
        hue_order=model_labels,
        style="label",
        style_order=model_labels,
        dashes=dashes,
        markers=markers,
        size="label",
        sizes=sizes,
        markersize=10,
        palette=custom_palette,
    )

    plt.xticks(fontsize=14)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)

    # Customize the plot labels
    plt.xlabel("samples", fontsize=18)
    plt.ylabel("proportion of triggered tests", fontsize=18)

    # Remove the default legend
    ax = plt.gca()
    ax.legend_.remove()

    # Retrieve the legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    # Define the groups and handles
    fine_tuned_labels = model_labels[:5]
    fine_tuned_handles = [label_to_handle[label] for label in fine_tuned_labels]

    uncensored_label = model_labels[5]
    uncensored_handle = label_to_handle[uncensored_label]

    # Create a divider line
    divider_line = Line2D([0], [0], color="black", linewidth=2)

    # Assemble the legend entries
    legend_handles = fine_tuned_handles + [divider_line] + [uncensored_handle]
    legend_labels = fine_tuned_labels + [""] + [uncensored_label]

    # Create custom handler for the divider line
    class HorizontalLineHandler(HandlerBase):
        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            x = xdescent
            # y = ydescent + height / 2
            y = ydescent + height / 2
            line = Line2D(
                # [x, x + width],
                [x, x + 446],
                [y, y],
                transform=trans,
                color="black",
                linewidth=0.5,
            )
            return [line]

    # Create a helper function to wrap labels
    def wrap_label(label, width):
        return "\n".join(textwrap.wrap(label, width=12))  # Adjust width as needed

    # Assemble the legend entries
    legend_handles = fine_tuned_handles + [divider_line] + [uncensored_handle]
    legend_labels = fine_tuned_labels + [""] + [uncensored_label]

    # Wrap the legend labels
    wrapped_legend_labels = []
    for label in legend_labels:
        if label != "":
            wrapped_label = wrap_label(label, width=4)  # Adjust the width parameter as needed
            wrapped_legend_labels.append(wrapped_label)
        else:
            wrapped_legend_labels.append(label)  # Keep the divider line as is

    # Create the legend with wrapped labels
    leg = ax.legend(
        legend_handles,
        wrapped_legend_labels,
        loc="lower right",
        fontsize=13,
        frameon=True,
        handler_map={divider_line: HorizontalLineHandler()},
    )

    # Create the legend
    leg = ax.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        fontsize=13,
        frameon=True,
        handler_map={divider_line: HorizontalLineHandler()},
    )

    plt.grid(True, linewidth=0.5, color="#ddddee")

    # Enhance plot spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Directory handling and saving the plot
    if use_models:
        directory = f"{plot_dir}/{base_model_name}_{base_model_seed}_models"
        for model_name in model_names:
            directory += f"_{model_name}"
        Path(directory).mkdir(parents=True, exist_ok=True)
    else:
        directory = f"{plot_dir}/{base_model_name}_{base_model_seed}_{checkpoint_base_name}_checkpoints"
        for seed in seeds:
            directory += f"_{seed}"
        Path(directory).mkdir(parents=True, exist_ok=True)

    file_extension = "pdf" if save_as_pdf else "png"
    file_name = f"power_over_number_of_sequences_grouped_by_{group_by}_{base_model_name}_{base_model_seed}{noise_string}.{file_extension}"
    plt.savefig(
        f"{directory}/{file_name}",
        bbox_inches="tight",
        format=file_extension,
        dpi=300 if not save_as_pdf else None,
    )


def plot_power_over_epsilon(
    base_model_name: str,
    base_model_seed: str,
    checkpoints: List[str],
    seeds: List[str],
    checkpoint_base_name: str = "Llama-3-8B-ckpt",
    metric: str = "perspective",
    distance_measure: str = "Wasserstein",
    fold_sizes: Union[int, List[int]] = [
        1000,
        2000,
        3000,
        4000,
    ],
    marker: str = "X",
    palette: List[str] = ["#E49B0F", "#C46210", "#B7410E", "#A81C07"],
    save_as_pdf: bool = True,
    plot_dir: str = "plots",
    epsilon: Union[float, List[float]] = 0,
    only_continuations: bool = True,
    dir_prefix: Optional[str] = None,
    noise: float = 0, 
):
    """
    This plots power over distance measure, potentially for different fold_sizes and models.
    """

    if dir_prefix is None:
        dir_prefix = metric

    plot_dir = ROOT_DIR / dir_prefix / plot_dir

    noise_string = f"_noise_{noise}" if noise > 0 else ""

    if isinstance(fold_sizes, list):
        result_df = get_power_over_sequences(
            base_model_name,
            base_model_seed,
            checkpoints,
            seeds,
            checkpoint_base_name=checkpoint_base_name,
            metric=metric,
            distance_measure=distance_measure,
            fold_sizes=fold_sizes,
            epsilon=epsilon,
            only_continuations=only_continuations,
            noise=noise,
        )
    else:
        result_df = get_power_over_sequences(
            base_model_name,
            base_model_seed,
            checkpoints,
            seeds,
            checkpoint_base_name=checkpoint_base_name,
            metric=metric,
            distance_measure=distance_measure,
            epsilon=epsilon,
            only_continuatinos=only_continuations,
            noise=noise,
        )

    smaller_df = extract_power_from_sequence_df(
        result_df,
    )

    # in case we have less folds
    palette = palette[-len(fold_sizes) :]

    plt.figure(figsize=(10, 6))

    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 1000)
    pd.set_option("display.width", 1000)

    sns.lineplot(
        x=f"Empirical {distance_measure} Distance",
        y="test_positive",
        hue="Samples per Test" if "Samples per Test" in smaller_df.columns else None,
        # style="Samples per Test" if "Samples per Test" in smaller_df.columns else None,
        marker=marker,
        data=smaller_df,
        markersize=10,
        palette=palette,
    )

    # plt.xlabel(f"{distance_measure.lower()} distance", fontsize=14)
    plt.xlabel(f"Distance to aligned model", fontsize=18)
    plt.ylabel("proportion of triggered tests", fontsize=18)
    plt.grid(True, linewidth=0.5, color="#ddddee")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)

    plt.legend(
        title="Samples per test",
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
                f"{directory}/power_over_{distance_measure.lower()}_distance_grouped_by_fold_size_{base_model_name}_{base_model_seed}{noise_string}.pdf",
                bbox_inches="tight",
                format="pdf",
            )

        else:
            plt.savefig(
                f"{directory}/power_over_{distance_measure.lower()}_distance_grouped_by_fold_size_{base_model_name}_{base_model_seed}{noise_string}.png",
                dpi=300,
                bbox_inches="tight",
            )
    else:
        if save_as_pdf:
            plt.savefig(
                f"{directory}/power_over_{distance_measure.lower()}_distance_{base_model_name}_{base_model_seed}{noise_string}.pdf",
                bbox_inches="tight",
                format="pdf",
            )
        else:
            plt.savefig(
                f"{directory}/power_over_{distance_measure.lower()}_distance_{base_model_name}_{base_model_seed}{noise_string}.png",
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
    plt.xlabel(
        f"rank based on distance to aligned model",
        fontsize=16,
    )
    plt.ylabel("proportion of triggered tests", fontsize=16)
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
                f"{directory}/power_over_{distance_measure.lower()}_rank_grouped_by_fold_size_{base_model_name}_{base_model_seed}{noise_string}.pdf",
                bbox_inches="tight",
                format="pdf",
            )
        else:
            plt.savefig(
                f"{directory}/power_over_{distance_measure.lower()}_rank_grouped_by_fold_size_{base_model_name}_{base_model_seed}{noise_string}.png",
                dpi=300,
                bbox_inches="tight",
            )
    else:
        if save_as_pdf:
            plt.savefig(
                f"{directory}/power_over_{distance_measure.lower()}_rank_{base_model_name}_{base_model_seed}{noise_string}.pdf",
                bbox_inches="tight",
                format="pdf",
            )
        else:
            plt.savefig(
                f"{directory}/power_over_{distance_measure.lower()}_rank_{base_model_name}_{base_model_seed}{noise_string}.png",
                dpi=300,
                bbox_inches="tight",
            )


def plot_alpha_over_sequences(
    model_names: List[str],
    seeds1: List[str],
    seeds2: List[str],
    save_as_pdf: bool = True,
    markers: List[str] = ["X", "o", "s"],
    palette: List[str] = ["#94D2BD", "#EE9B00", "#BB3E03"],
    fold_size: int = 4000,
    plot_dir: str = "plots",
    epsilon: float = 0,
    only_continuations: bool = True,
    dir_prefix: Optional[str] = None,
    metric: str = "perspective",
    test_dir: str = "test_outputs",
    noise: float = 0,
):
    # ROOT_DIR = os.path.dirname(__file__)
    if dir_prefix is None:
        dir_prefix = metric

    # Construct the absolute path to "test_outputs"
    plot_dir = ROOT_DIR / dir_prefix / plot_dir

    noise_string = f"_noise_{noise}" if noise > 0 else ""

    result_df = get_alpha_wrapper(
        model_names,
        seeds1,
        seeds2,
        fold_size=fold_size,
        epsilon=epsilon,
        only_continuations=only_continuations,
        dir_prefix=dir_prefix,
        test_dir=test_dir,
        metric=metric,
        noise=noise,
    )
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
                markersize=10,
            )
    else:
        sns.lineplot(
            data=result_df,
            x="samples",
            y="test_positive",
            marker="o",
            dashes=False,  # No dashes, solid lines
            color="black",
            markersize=10,
        )

    # Customize the plot
    plt.xlabel("samples", fontsize=18)
    plt.ylabel("proportion of triggered tests", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)
    plt.yticks(np.arange(0, 0.06, 0.01), fontsize=14)

    # plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Adjust the spines (box) thickness
    ax = plt.gca()
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)

    if group_by_model:
        plt.legend(
            # title="Models",
            loc="center right",
            # loc="lower right",
            fontsize=14,
            bbox_to_anchor=(1.0, 0.2),
            # title_fontsize=16,
            # bbox_to_anchor=(
            #     1.05,
            #     1,
            # ),  # Adjusted position to ensure the legend is outside the plot area
        )
    plt.grid(True, linewidth=0.5, color="#ddddee")

    directory = f"{plot_dir}/alpha_plots"
    if not Path(directory).exists():
        Path(directory).mkdir(parents=True, exist_ok=True)
    fig_path = f"{directory}/alpha_error_over_number_of_sequences{noise_string}"
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
    model_names1: List[str],
    seeds1: List[str],
    model_names2: Optional[List[str]] = None,
    seeds2: Optional[List[str]] = None,
    fold_size: int = 4000,
    distance_measure: Optional[str] = "Wasserstein",
    metric: Optional[str] = "perspective",
    save_as_pdf: bool = True,
    plot_dir: str = "plots",
    dir_prefix: Optional[str] = None,
    noise: float = 0,
):
    """ """
    assert (model_names2 is None and seeds2 is None) or (model_names2 is not None and seeds2 is not None), (
        "Either give full list of test models or expect to iterate over all combinations"
    )

    if dir_prefix is None:
        dir_prefix = metric

    noise_string = f"_noise_{noise}" if noise else ""

    plot_dir = ROOT_DIR / dir_prefix / plot_dir

    results_df = []
    if not model_names2:
        for i, (model_name1, seed1) in enumerate(zip(model_names1[:-1], seeds1[:-1])):
            for model_name2, seed2 in zip(model_names1[i + 1 :], seeds1[i + 1 :]):
                print(f"Checking model {model_name1}, {seed1} against {model_name2}, {seed2}")
                result_df = get_power_over_sequences(
                    model_name1,
                    seed1,
                    seed2,
                    model_name2=model_name2,
                    fold_size=fold_size,
                    noise=noise,
                )
                if distance_measure:
                    dist = get_distance_scores(
                        model_name1,
                        seed1,
                        seed2,
                        model_name2=model_name2,
                        metric=metric,
                        distance_measure=distance_measure,
                        noise=noise,
                    )
                    result_df[f"Empirical {distance_measure} Distance"] = dist
                small_df = extract_power_from_sequence_df(
                    result_df,
                    distance_measure=distance_measure,
                    by_checkpoints=False,
                )

                results_df.append(small_df)

        results_df = pd.concat(results_df, ignore_index=True)

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

    directory = f"{plot_dir}/power_heatmaps"
    if not Path(directory).exists():
        Path(directory).mkdir(parents=True, exist_ok=True)
    file_name = "power_heatmap"
    for model_name, seed in zip(model_names1, seeds1):
        file_name += f"_{model_name}_{seed}"
        file_name += noise_string

    if save_as_pdf:
        file_name += ".pdf"
        output_path = os.path.join(directory, file_name)
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
    else:
        file_name += ".png"
        output_path = os.path.join(directory, file_name)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

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

        directory = f"{plot_dir}/power_heatmaps"
        if not Path(directory).exists():
            Path(directory).mkdir(parents=True, exist_ok=True)

        file_name = "distance_heatmap"
        for model_name, seed in zip(model_names1, seeds1):
            file_name += f"_{model_name}_{seed}"
            file_name += noise_string

        if save_as_pdf:
            file_name += ".pdf"
            output_path = os.path.join(directory, file_name)
            plt.savefig(
                output_path,
                format="pdf",
                bbox_inches="tight",
            )
        else:
            file_name += ".png"
            output_path = os.path.join(directory, file_name)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")



def plot_calibrated_detection_rate(
    model_name1: str,
    seed1: str,
    model_name2: str,
    seed2: str,
    true_epsilon: Optional[float] = None,
    std_epsilon: Optional[float] = None,
    result_file: Optional[Union[str, Path]] = None,
    num_runs: int = 20,
    multiples_of_epsilon: Optional[int] = None,
    test_dir: str = "test_outputs",
    save_as_pdf: bool = True,
    overwrite: bool = False,
    draw_in_std: bool = False,
    fold_size: int = 4000,
    bs: int = 100,
    only_continuations: bool = True,
    dir_prefix: Optional[str] = None,
    metric: str = "perspective",
    noise: float = 0,
):
    """Plot calibrated detection rate without legend and with a text label for the vertical line."""
    if dir_prefix is None:
        dir_prefix = metric

    test_dir = ROOT_DIR / dir_prefix / test_dir
    plot_dir = ROOT_DIR / dir_prefix / "plots"

    noise_string = f"_noise_{noise}" if noise else ""

    if result_file is not None:
        result_file_path = Path(result_file)
    else:
        multiples_str = f"_{multiples_of_epsilon}" if multiples_of_epsilon else ""
        continuations_str = "_continuations" if only_continuations else ""
        result_file_path = os.path.join(
            plot_dir,
            f"power_over_epsilon{continuations_str}_{fold_size - bs}_{num_runs}{multiples_str}{noise_string}.csv",
        )

    if not true_epsilon:
        distance_path = os.path.join(
            plot_dir,
            f"distance_scores_{fold_size - bs}_{num_runs}{noise_string}.csv",
        )
        try:
            distance_df = pd.read_csv(distance_path)
            true_epsilon, std_epsilon = get_mean_and_std_for_nn_distance(distance_df)
            logger.info(
                f"True epsilon for {model_name1}_{seed1} and {model_name2}_{seed2}: {true_epsilon}, std epsilon: {std_epsilon}"
            )
        except FileNotFoundError:
            logger.error(f"Could not find file at {distance_path}")
            sys.exit(1)

    df = pd.read_csv(result_file_path)
    df_sorted = df.sort_values(by="epsilon")

    # Plotting
    plt.figure(figsize=(12, 8))

    ax = sns.lineplot(
        data=df_sorted,
        x="epsilon",
        y="power",
        marker="X",
        markersize=10,
        linewidth=2.5,
        color="#1f77b4",
        legend=False,  # Ensure the legend is not displayed
    )

    # Ensure markers are visible
    for line in ax.lines:
        line.set_markerfacecolor(line.get_color())
        line.set_markeredgecolor(line.get_color())

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    line_color = ax.get_lines()[0].get_color()

    # Add vertical line for true epsilon
    plt.axvline(
        x=true_epsilon,
        linestyle="--",
        color=line_color,
        alpha=0.7,
        linewidth=2,
    )

    # # Add text label next to the vertical line
    # plt.text(
    #     x=true_epsilon + (plt.xlim()[1] - plt.xlim()[0]) * 0.01,  # Slightly to the right of the line
    #     y=0.95 * plt.ylim()[1],  # Near the top of the plot
    #     s="Estimated distance to checkpoint",
    #     rotation=90,  # Rotate the text vertically
    #     verticalalignment="top",
    #     fontsize=14,
    #     color=line_color,
    # )

    # Adding the standard deviation range
    if draw_in_std:
        line_color_rgb = to_rgb(line_color)
        h, l, s = colorsys.rgb_to_hls(*line_color_rgb)
        lighter_l = min(1, l + (1 - l) * 0.3)
        lighter_line_color = colorsys.hls_to_rgb(h, lighter_l, s)

        plt.axvspan(
            true_epsilon - std_epsilon,
            true_epsilon + std_epsilon,
            alpha=0.2,
            color=lighter_line_color,
            label="Std Dev Range",
        )

    # Adjust labels
    plt.xlabel("Test epsilon", fontsize=22)
    plt.ylabel("Proportion of triggered tests", fontsize=22)

    # Remove the legend by commenting out or removing the plt.legend() call
    # legend = plt.legend(
    #     loc="lower left",
    #     fontsize=16,
    #     frameon=True,  # Enable the frame around the legend
    #     fancybox=False,  # Disable rounded corners
    #     shadow=False,  # Disable shadow to remove 3D effect
    # )

    # Adjust grid
    plt.grid(True, color="#ddddee", linewidth=0.5)

    # Add a box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    plot_path = os.path.join(
        plot_dir,
        f"calibrated_detection_rate_{fold_size}{noise_string}.pdf",
    )

    if not overwrite and Path(plot_path).exists():
        logger.info(f"File already exists at {plot_path}. Skipping...")
    else:
        logger.info(f"Saving plot to {plot_path}")
        plt.tight_layout()
        plt.savefig(plot_path, format="pdf", bbox_inches="tight")


def plot_multiple_calibrated_detection_rates(
    model_names: List[str],
    seeds: List[str],
    true_epsilon: Optional[List[float]] = None,
    base_model: str = "Meta-Llama-3-8B-Instruct",
    base_seed: str = "seed1000",
    num_runs: int = 20,
    multiples_of_epsilon: Optional[int] = None,
    test_dir: str = "test_outputs",
    plot_dir: str = "plots",
    save_as_pdf: bool = True,
    overwrite: bool = False,
    draw_in_std: bool = False,
    fold_size: int = 4000,
    bs: int = 100,
    only_continuations: bool = True,
    metric: str = "perspective",
    dir_prefix: Optional[str] = None,
    noise: float = 0,
):
    """
    Plot calibrated detection rates for multiple models on the same graph using seaborn lineplot.
    """

    if dir_prefix is None:
        dir_prefix = metric

    test_dir = ROOT_DIR / dir_prefix / test_dir
    plot_dir = ROOT_DIR / dir_prefix / plot_dir

    noise_string = f"_noise_{noise}" if noise else ""

    plt.figure(figsize=(18, 7))

    custom_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ]

    # Prepare data for seaborn plotting
    all_data = []

    for i, (model_name, seed) in enumerate(zip(model_names, seeds)):
        result_dir = os.path.join(
            test_dir,
            f"{base_model}_{base_seed}_{model_name}_{seed}",
        )
        multiples_str = f"_{multiples_of_epsilon}" if multiples_of_epsilon else ""
        continuations_str = "_continuations" if only_continuations else ""
        result_file_path = os.path.join(
            result_dir,
            f"power_over_epsilon{continuations_str}_{fold_size - bs}_{num_runs}{multiples_str}{noise_string}.csv",
        )
        distance_path = os.path.join(
            result_dir,
            f"distance_scores_{fold_size - bs}_{num_runs}{noise_string}.csv",
        )

        try:
            distance_df = pd.read_csv(distance_path)
            true_epsilon_i, std_epsilon_i = get_mean_and_std_for_nn_distance(distance_df)
            logger.info(f"True epsilon for {model_name}_{seed}: {true_epsilon_i}, std epsilon: {std_epsilon_i}")
        except FileNotFoundError:
            logger.error(f"Could not find file at {distance_path}")
            continue

        try:
            df = pd.read_csv(result_file_path)
            df_sorted = df.sort_values(by="epsilon")

            # Use task cluster name for the legend
            task_cluster_name = ", ".join(TASK_CLUSTER[i]) if i < len(TASK_CLUSTER) else f"Model {i + 1}"

            df_sorted["Model"] = task_cluster_name
            all_data.append(df_sorted)
        except FileNotFoundError:
            logger.error(f"Could not find file for {model_name}_{seed}")

    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)

    # Plot using seaborn
    ax = sns.lineplot(
        data=combined_data,
        x="epsilon",
        y="power",
        hue="Model",
        marker="X",
        markersize=10,
        linewidth=2.5,
        palette=custom_palette,
    )

    # Ensure markers are visible
    for line in ax.lines:
        line.set_markerfacecolor(line.get_color())
        line.set_markeredgecolor(line.get_color())

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    # Add vertical lines for true epsilon
    for i, (model_name, seed) in enumerate(zip(model_names, seeds)):
        true_epsilon_i, _ = get_mean_and_std_for_nn_distance(
            pd.read_csv(
                os.path.join(
                    test_dir,
                    f"{base_model}_{base_seed}_{model_name}_{seed}",
                    f"distance_scores_{fold_size - bs}_{num_runs}{noise_string}.csv",
                )
            )
        )
        plt.axvline(
            x=true_epsilon_i,
            linestyle="--",
            color=sns.color_palette()[i],
            alpha=0.7,
            linewidth=2,
        )

    # Titles and labels
    plt.xlabel("test epsilon", fontsize=22)
    plt.ylabel("proportion of triggered tests", fontsize=22)

    handles, labels = ax.get_legend_handles_labels()
    wrapped_labels = [textwrap.fill(label, width=36) for label in labels]  # Adjust width as needed

    # Adjust legend
    legend = plt.legend(
        handles,
        wrapped_labels,
        # title="Models Fine-Tuned On ...",
        loc="upper right",
        fontsize=16,
        # title_fontsize=18,
        # bbox_to_anchor=(0.02, 0.02),  # Adjust these values to fine-tune position
        # bbox_to_anchor=(0.98, 0.98),  # Adjust these values to fine-tune position
        ncol=1,
        frameon=True,  # Enable the frame around the legend
        fancybox=True,  # Disable rounded corners
        shadow=False,  # Disable shadow to remove 3D effect
        # borderaxespad=0.0,
    )

    # Set legend background to be opaque white and border color to black
    # legend.get_frame().set_facecolor("white")  # Solid white background
    # legend.get_frame().set_edgecolor("black")  # Black border
    # legend.get_frame().set_linewidth(0.5)  # Border line width

    # Remove any transparency settings
    # legend.get_frame().set_alpha(1.0)  # Optional, as default alpha is 1.0 (fully opaque)

    # Make the grid less noticeable
    plt.grid(True, color="#ddddee", linewidth=0.5)

    # Add a box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    # Save the plot
    plot_path = os.path.join(
        plot_dir,
        f"multi_model_calibrated_detection_rate_{fold_size}{noise_string}.pdf",
    )
    if not overwrite and Path(plot_path).exists():
        logger.info(f"File already exists at {plot_path}. Skipping...")
    else:
        logger.info(f"Saving plot to {plot_path}")
        plt.tight_layout()
        plt.savefig(plot_path, format="pdf", bbox_inches="tight")


def darken_color(color, factor=0.7):
    """
    Darken the given color by multiplying RGB values by the factor.
    """
    h, l, s = colorsys.rgb_to_hls(*color)
    return colorsys.hls_to_rgb(h, max(0, min(1, l * factor)), s)


def plot_scores(
    model_name: str,
    seed: str,
    metric: str = "perspective",
    save: bool = True,
    epoch: int = 0,
    use_log_scale: bool = True,
    color: str = "blue",
    save_as_pdf: bool = True,
    plot_dir: str = "plots",
    dir_prefix: Optional[str] = None,
    noise: float = 0,
    only_continuations: bool = True,
):
    """ """

    if dir_prefix is None:
        dir_prefix = metric

    # Construct the absolute path to "test_outputs"
    plot_dir = ROOT_DIR / dir_prefix / plot_dir

    noise_string = f"_noise_{noise}" if noise > 0 else ""
    cont_string = "continuations_" if only_continuations else ""

    directory = f"{plot_dir}/{model_name}_{seed}"
    file_path = f"{directory}/{cont_string}scores{noise_string}.json"
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
    plt.ylabel(
        "log frequency" if use_log_scale else "frequency",
        fontsize=16,
    )
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
                    directory,
                    f"{metric}_scores_{model_name}_{seed}{noise_string}_log.pdf",
                )
                plt.savefig(
                    output_path,
                    format="pdf",
                    bbox_inches="tight",
                )
            else:
                output_path = os.path.join(
                    directory,
                    f"{metric}_scores_{model_name}_{seed}{noise_string}_log.png",
                )
                plt.savefig(
                    output_path,
                    dpi=300,
                    bbox_inches="tight",
                )
        else:
            if save_as_pdf:
                output_path = os.path.join(
                    directory,
                    f"{metric}_scores_{model_name}_{seed}{noise_string}.pdf",
                )
                plt.savefig(
                    output_path,
                    format="pdf",
                    bbox_inches="tight",
                )
            else:
                output_path = os.path.join(
                    directory,
                    f"{metric}_{model_name}_{seed}_scores{noise_string}.png",
                )
                plt.savefig(
                    output_path,
                    dpi=300,
                    bbox_inches="tight",
                )
    else:
        plt.show()

    plt.close()



def plot_power_over_number_of_sequences_for_different_levels_of_noise(
    base_model_name: str,
    base_model_seed: str,
    seeds: List[str],
    checkpoints: Optional[List[str]] = None,
    checkpoint_base_name: str = "LLama-3-8B-ckpt",
    marker: Union[str, List[str], Dict[str, str]] = "X",
    save_as_pdf: bool = True,
    test_dir: str = "test_outputs",
    plot_dir: str = "plots",
    metric: str = "perspective",
    only_continuations: bool = True,
    fold_size: int = 2000,
    epsilon: Union[float, List[float]] = 0,
    palette: Optional[Union[List[str], Dict[str, str]]] = [
        "#E49B0F",
        "#C46210",
        "#B7410E",
        "#A81C07",
    ],
    sizes: Optional[Dict[str, int]] = None,
    line_styles: Optional[Dict[str, Union[str, Tuple[int, int]]]] = None,
    dir_prefix: Optional[str] = None,
    noise_levels: List[float] = [0, 0.01, 0.05, 0.1],
):
    if dir_prefix is None:
        dir_prefix = metric
    plot_dir = ROOT_DIR / dir_prefix / plot_dir
    test_dir = ROOT_DIR / dir_prefix / test_dir
    result_dfs = []
    for noise_level in noise_levels:
        noise_string = f"_noise_{noise_level}" if noise_level > 0 else ""
        result_df = get_power_over_sequences(
            base_model_name,
            base_model_seed,
            seeds=seeds,
            checkpoints=checkpoints,
            checkpoint_base_name=checkpoint_base_name,
            only_continuations=only_continuations,
            fold_size=fold_size,
            epsilon=epsilon,
            noise=noise_level,
        )
        result_df["noise"] = noise_level
        result_dfs.append(result_df)

    final_df = pd.concat(result_dfs, ignore_index=True)
    noise_df = (
        final_df.groupby(
            [
                "Sequence",
                "Samples per Test",
                "Samples",
                "epsilon",
                "noise",
            ]
        )["Power"]
        .mean()
        .reset_index()
    )

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Get unique noise levels
    noise_levels = sorted(noise_df["noise"].unique())

    # Create custom palette that uses black for 0 noise and original colors for others
    if palette is None:
        palette = sns.color_palette("viridis", len(noise_levels) - 1)  # -1 because we handle 0 separately
    custom_palette = {0: "black"}  # Set 0 noise to black
    for i, noise in enumerate([n for n in noise_levels if n != 0]):
        custom_palette[noise] = palette[i] if isinstance(palette, list) else palette

    # Create custom line styles
    # custom_styles = {0: '--'}  # Set 0 noise to dashed
    custom_styles = {0: "-"}
    for noise in [n for n in noise_levels if n != 0]:
        custom_styles[noise] = "-"  # Set other noise levels to solid

    # Create plot for each noise level separately to control styles
    for noise in noise_levels:
        data = noise_df[noise_df["noise"] == noise]
        # Create custom label: "no noise" for 0, and formatted float for others
        label = "no noise" if noise == 0 else f"N(0, {noise:.2f})"
        sns.lineplot(
            data=data,
            x="samples",
            y="test_positive",
            label=label,
            marker=marker,
            markersize=10,
            color=custom_palette[noise],
            linestyle=custom_styles[noise],
        )

    # Customize the plot
    plt.xlabel("samples", fontsize=18)
    plt.ylabel("proportion of triggered tests", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)

    # Legend
    plt.legend(
        title="noise level",
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=14,
        title_fontsize=16,
    )

    # Grid
    plt.grid(True, linewidth=0.5, color="#ddddee")

    # Make plot borders thicker
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)

    # Save plot
    if save_as_pdf:
        plt.savefig(
            f"{plot_dir}/power_over_sequences_noise_comparison.pdf",
            bbox_inches="tight",
            format="pdf",
        )
    else:
        plt.savefig(
            f"{plot_dir}/power_over_sequences_noise_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()



def plot_mean_scores_for_checkpoints(
    base_model_name: str = "Meta-Llama-3-8B-Instruct",
    base_seed: str = "seed1000",
    seeds: List[str] = [
        "seed1000",
        "seed1000",
        "seed1000",
        "seed1000",
        "seed1000",
        "seed1000",
    ],
    checkpoints: List[int] = [1, 2, 3, 4, 5, 6],
    checkpoint_base_name: str = "Llama-3-8B-ckpt",
    metric: str = "perspective",
    plot_dir: str = "plots",
    dir_prefix: Optional[str] = None,
    noise: float = 0,
    only_continuations: bool = True,
):
    """
    Plot the mean scores for the base model and the most extreme checkpoint.
    """
    if dir_prefix is None:
        dir_prefix = metric

    # Construct the absolute path to "test_outputs"
    plot_dir = ROOT_DIR / dir_prefix / plot_dir

    cont_string = "continuations_" if only_continuations else ""
    noise_string = f"_noise_{noise}" if noise > 0 else ""

    directory = f"{plot_dir}/{base_model_name}_{base_seed}"
    file_path = f"{directory}/{cont_string}scores{noise_string}.json"
    with open(file_path, "r") as f:
        data = json.load(f)

    scores = data[str(0)][f"{metric}_scores"]
    scores_dict = {}
    wasserstein_distances = {}

    for ckpt, seed in zip(checkpoints, seeds):
        checkpoint_directory = f"{plot_dir}/{checkpoint_base_name}{ckpt}_{seed}"
        file_path = f"{checkpoint_directory}/{cont_string}scores{noise_string}.json"
        with open(file_path, "r") as f:
            checkpoint_data = json.load(f)
            scores_ckpt = checkpoint_data[str(0)][f"{metric}_scores"]
            scores_dict[(ckpt, seed)] = scores_ckpt
            wasserstein_distances[(ckpt, seed)] = empirical_wasserstein_distance_p1(scores, scores_ckpt)

    max_distance_ckpt, max_distance_seed = max(wasserstein_distances, key=wasserstein_distances.get)
    print(f"This is the max distance checkpoint: {max_distance_ckpt} with seed: {max_distance_seed}")
    max_distance = wasserstein_distances[(max_distance_ckpt, max_distance_seed)]
    print(f"This is the max distance: {max_distance:.4f}")

    ckpt_scores = scores_dict[(max_distance_ckpt, max_distance_seed)]

    array_ckpt_scores = np.array(ckpt_scores)
    skewness = skew(array_ckpt_scores)
    # print(f


@hydra.main(
    config_path=".",
    config_name="tox_plotting_config.yml",
)
def plot_toxicity(cfg: DictConfig):
    """ """
    base_models = []
    base_seeds = []
    base_seeds2 = []

    # Loop over all base models
    for bm in cfg.models:
        base_models.append(bm.name)
        base_seeds.append(bm.seed)
        base_seeds2.append(bm.seed2)

        checkpoints = [i for i in range(1, int(bm.checkpoint_range))]
        if "llama" in bm.name.lower():
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
        else:
            seeds = ["seed1000" for i in range(1, int(bm.checkpoint_range))]

        # Create power plot over sequences:
        plot_power_over_number_of_sequences(
            bm.name,
            bm.seed,
            seeds,
            checkpoints,
            checkpoint_base_name=bm.checkpoint_base_name,
            fold_size=list(cfg.fold_sizes)[-1],
            group_by="Wasserstein",
            only_continuations=cfg.only_continuations,
            marker=bm.marker,
            noise=cfg.noise,
        )

    #create false positive plot
    plot_alpha_over_sequences(
        base_models,
        base_seeds,
        base_seeds2,
        noise=cfg.noise,
        only_continuations=cfg.only_continuations,
    )


def plot_davt_vs_c2st_power(
    csv_path: str,
    save_as_pdf: bool = True,
    plot_dir: str = "plots",
    metric: str = "toxicity",
    dir_prefix: Optional[str] = None,
):
    """Plot power over sequences for both DAVT and C2ST tests from CSV data.

    Args:
        csv_path: Path to the CSV file containing test results
        save_as_pdf: Whether to save as PDF
        plot_dir: Directory to save plots
        metric: Metric used for testing
        dir_prefix: Optional prefix for directories
    """
    # Construct path to results
    if dir_prefix is None:
        dir_prefix = metric

    plot_dir = ROOT_DIR / dir_prefix / plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Read and process data
    df = pd.read_csv(csv_path)

    # Group by samples and calculate mean test rates across folds
    grouped_df = (
        df.groupby("samples")
        .agg(
            {
                "test_positive_davt": "mean",
                "test_positive_c2st": "mean",
            }
        )
        .reset_index()
    )

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot DAVT test rate
    sns.lineplot(
        data=grouped_df,
        x="samples",
        y="test_positive_davt",
        label="DAVT",
        marker="o",
        markersize=8,
        color="#E49B0F",
    )

    # Plot C2ST test rate
    sns.lineplot(
        data=grouped_df,
        x="samples",
        y="test_positive_c2st",
        label="C2ST",
        marker="s",
        markersize=8,
        color="#BB3E03",
    )

    # Customize plot
    plt.xlabel("samples", fontsize=18)
    plt.ylabel("proportion of triggered tests", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)

    # Legend
    plt.legend(
        fontsize=14,
        title_fontsize=16,
        bbox_to_anchor=(1.0, 0.5),
        loc="center left",
    )

    # Grid
    plt.grid(True, linewidth=0.5, color="#ddddee")

    # Make plot borders thicker
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)

    # Extract model names from csv path for filename
    path_parts = csv_path.split("/")
    results_dir = path_parts[-2] if len(path_parts) > 1 else "results"

    # Save plot
    if save_as_pdf:
        plt.savefig(
            f"{plot_dir}/davt_vs_c2st_power_{results_dir}.pdf",
            bbox_inches="tight",
            format="pdf",
        )
    else:
        plt.savefig(
            f"{plot_dir}/davt_vs_c2st_power_{results_dir}.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()



