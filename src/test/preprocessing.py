import json
import logging
import os

import random
import sys
import numpy as np

from collections import defaultdict
from datasets import load_dataset
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Add paths to sys.path if not already present
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.evaluation.evaluate import evaluate_single_model
from src.evaluation.score import eval_on_metric
from src.utils.legacy_utils import remove_zero_key_and_flatten
from src.utils.utils import (
    time_block,
    create_run_string,
    load_config,
    cleanup_files,
    load_entire_json,
)
from logging_config import setup_logging

# setup_logging()
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]


def create_common_json(
    model_name1,
    seed1,
    model_name2,
    seed2,
    metric="perspective",
    overwrite=True,
    score_dir="model_scores",
    test_dir="test_outputs",
    only_continuations=True,
    noise=0,
):
    """ """

    file_path1 = f"{score_dir}/{model_name1}_{seed1}"
    file_path2 = f"{score_dir}/{model_name2}_{seed2}"
    new_folder_path = Path(test_dir) / f"{model_name1}_{seed1}_{model_name2}_{seed2}"
    new_folder_path.mkdir(parents=True, exist_ok=True)

    cont_string = "continuation_" if only_continuations else ""
    noise_string = f"_noise_{noise}" if noise > 0 else ""

    common_scores_file_path = new_folder_path / f"{cont_string}scores{noise_string}.json"
    if overwrite or not common_scores_file_path.exists():
        file_name1 = f"{file_path1}/{cont_string}scores{noise_string}.json"
        file_name2 = f"{file_path2}/{cont_string}scores{noise_string}.json"

        with open(file_name1, "r", encoding="utf-8") as file1, open(file_name2, "r", encoding="utf-8") as file2:
            data1 = json.load(file1)
            data2 = json.load(file2)

        data = defaultdict(list)
        data["metadata1"] = data1["metadata"]
        data["metadata2"] = data2["metadata"]
        unfiltered_scores1 = data1[f"{metric}_scores"]
        unfiltered_scores2 = data2[f"{metric}_scores"]

        try:
            assert len(unfiltered_scores1) == len(unfiltered_scores2), "Scores are not the same length."
        except AssertionError as e:
            logger.error(f"Assertion failed: {str(e)}")
            raise

        for score1, score2 in zip(unfiltered_scores1, unfiltered_scores2):
            if not (np.isnan(score1) or np.isnan(score2)):
                data[f"{metric}_scores1"].append(score1)
                data[f"{metric}_scores2"].append(score2)

        logger.warning(f"Discarding {len(unfiltered_scores1) - len(data[f'{metric}_scores1'])} NaN scores.")

        with open(common_scores_file_path, "w") as file:
            json.dump(data, file, indent=4)


def create_folds(
    model_name1,
    seed1,
    model_name2,
    seed2,
    metric="perspective",
    fold_size=2000,
    overwrite=True,
    test_dir="test_outputs",
    only_continuations=True,
    noise=0,
):
    """ """

    # Fix random seed to be different for each fold_size, such that the folds always have different samples.
    # random.seed(fold_size)
    random.seed(10)

    cont_string = "continuation_" if only_continuations else ""
    noise_string = f"_noise_{noise}" if noise > 0 else ""

    directory = f"{test_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}"
    file_pattern = f"{cont_string}scores{noise_string}_fold_*.json"

    # Cleanup existing fold files
    cleanup_files(directory, file_pattern, verbose=False)

    file_name = f"{test_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}/{cont_string}scores{noise_string}.json"
    data = load_entire_json(file_name)

    # Extract metadata and other lists
    metadata1 = data["metadata1"]
    metadata2 = data["metadata2"]

    # Create batches
    total_num_samples = len(data[f"{metric}_scores1"])
    logger.info(f"Total number of samples: {total_num_samples}")
    indices = list(range(total_num_samples))
    random.shuffle(indices)
    index_batches = [indices[i : i + fold_size] for i in range(0, total_num_samples, fold_size)]

    # The last batch might contain fewer samples
    if len(index_batches[-1]) < fold_size:
        logger.warning(
            f"Last fold contains fewer samples and is discarded, resulting in {len(index_batches[-1])} samples being discarded."
        )
        index_batches = index_batches[:-1]

    for i, batch in tqdm(enumerate(index_batches)):  # The last batch is not used because it
        fold_file_path = (
            f"{test_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}/{cont_string}scores{noise_string}_fold_{i}.json"
        )
        if overwrite or not os.path.exists(fold_file_path):
            fold_data = defaultdict(list)
            fold_data["metadata1"] = metadata1
            fold_data["metadata2"] = metadata2

            for key, value in data.items():
                # legacy: in future, only scores should be saved
                if key in ["prompts", "continuations1", "continuations2"]:
                    fold_data[key] = [value[j] for j in batch]
                elif key in [f"{metric}_scores1", f"{metric}_scores2"]:
                    fold_data[key] = [value[j] for j in batch]

            with open(fold_file_path, "w") as file:
                json.dump(fold_data, file, indent=4)


def create_folds_from_evaluations(
    model_name1,
    seed1,
    model_name2,
    seed2,
    metric="perspective",
    fold_size=2000,
    overwrite=True,
    only_continuations=True,
    test_dir="test_outputs",
    score_dir="model_scores",
    gen_dir="model_outputs",
    noise=0,
):
    try:
        create_common_json(
            model_name1,
            seed1,
            model_name2,
            seed2,
            metric,
            overwrite=overwrite,
            only_continuations=only_continuations,
            score_dir=score_dir,
            test_dir=test_dir,
            noise=noise,
        )
    except FileNotFoundError as e:
        logger.info(f"File not found: {e}. Trying to create the folds from generations.")
        evaluate_single_model(
            model_name=model_name1,
            seed=seed1,
            metric=metric,
            overwrite=False,
            only_continuation=only_continuations,
            gen_dir=gen_dir,
            score_dir=score_dir,
            noise=noise,
        )
        evaluate_single_model(
            model_name=model_name2,
            seed=seed2,
            metric=metric,
            overwrite=False,
            only_continuation=only_continuations,
            gen_dir=gen_dir,
            score_dir=score_dir,
            noise=noise,
        )

        create_common_json(
            model_name1,
            seed1,
            model_name2,
            seed2,
            metric,
            overwrite=overwrite,
            only_continuations=only_continuations,
            score_dir=score_dir,
            test_dir=test_dir,
            noise=noise,
        )

    create_folds(
        model_name1,
        seed1,
        model_name2,
        seed2,
        metric,
        fold_size=fold_size,
        overwrite=overwrite,
        only_continuations=only_continuations,
        test_dir=test_dir,
        noise=noise,
    )


def create_toxic_prompt_mask(dataset_name="allenai/real-toxicity-prompts", impute_with_continuation=False):
    """ """
    dataset = load_dataset(dataset_name, split="train")

    high_toxicity_indices = []
    no_prompt = 0
    toxic_cont = 0
    skipped = 0

    for idx, item in enumerate(dataset):  # Assuming you're using the 'train' split
        # print(f"These are the keys in the dataset: {item['prompt'].keys()}")
        try:
            if float(item["prompt"]["toxicity"]) >= 0.5:
                high_toxicity_indices.append(idx)
        except TypeError:
            # logger.info(f"Missing prompt tox score for item {idx}")
            no_prompt += 1

            if impute_with_continuation:
                # logger.info("Using tox score from continuation.")
                try:
                    if float(item["continuation"]["toxicity"]) >= 0.5:
                        high_toxicity_indices.append(idx)
                        toxic_cont += 1
                except TypeError:
                    logger.info(f"Missing continuation tox score for item {idx}. Skipping the item.")
                    skipped += 1

    if impute_with_continuation:
        logger.info(f"We had to use the continuation score from {no_prompt} items.")
        logger.info(f"This leads to {toxic_cont} items being added to the high toxicity list.")
        logger.info(f"{skipped} items were skipped due to missing tox scores in both prompt and continuation.")

    logger.info(f"Number of prompts with toxicity >= 0.5: {len(high_toxicity_indices)}")

    # Save the list of high toxicity indices

    if impute_with_continuation:
        file_name = "high_toxicity_indices_imputed.json"
    else:
        file_name = "high_toxicity_indices.json"
    with open(file_name, "w") as f:
        json.dump(high_toxicity_indices, f)


if __name__ == "__main__":
    model_name1 = "Meta-Llama-3-8B-Instruct"
    seed1 = "seed1000"
    model_names2 = [f"Llama-3-8B-ckpt{i}" for i in range(1, 10)]
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

    score_dir = ROOT_DIR / "perspective" / "model_scores"
    test_dir = ROOT_DIR / "perspective" / "test_outputs"

    for mname2, seed2 in zip(model_names2, seeds):
        create_common_json(
            model_name1,
            seed1,
            mname2,
            seed2,
            metric="perspective",
            overwrite=True,
            score_dir=score_dir,
            test_dir=test_dir,
            only_continuations=True,
            noise=0,
        )
