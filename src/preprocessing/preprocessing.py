import json
import logging
import os
import glob
import evaluate
import typing
import random
import sys
import numpy as np
import time

from collections import defaultdict
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from transformers import pipeline
from datasets import load_dataset

# Add the parent directory of utils to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import download_file_from_wandb, time_block, create_run_string, load_config
from utils.score import eval_on_metric
from utils.remove_unused_epoch_key import remove_zero_key_and_flatten
from logging_config import setup_logging

# setup_logging()
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def load_entire_json(filepath, encoding="utf-8"):
    try:
        with open(filepath, "r", encoding=encoding) as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {e}")
        line, column = e.lineno, e.colno
        logger.error(f"Error at line {line}, column {column}")
        # Print a few lines around the error to help debug
        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()
            start = max(0, line - 3)
            end = min(len(lines), line + 2)
            for i in range(start, end):
                logger.debug(f"{i + 1}: {lines[i].strip()}")
        # Re-raise the error to avoid further processing
        raise


def evaluate_single_model(
    model_name: Optional[str] = None,
    seed: Optional[str] = None,
    model_gen_dir: Optional[str] = None,
    metric: str = "perspective",
    overwrite=True,
    asynchronously=True,
    save_intermittently=True,
    ds_batch_size=1000,
    model_batch_size=8,
    remove_intermediate_files=True,
    gen_dir="model_outputs",
    score_dir="model_scores",
    verbose=True,
    only_continuation=False,
    short=False,
):
    """
    Evaluate a single model and save the scores.

    Args:
        model_name (str): The name of the model to evaluate.
        seed (str): The seed value for reproducibility.
        model_dir (str): alternatively, give the directory where the generations are stored.
        metric: The evaluation metric to use.
        overwrite (bool, optional): Whether to overwrite existing scores file. Defaults to True.
        asynchronously (bool, optional): Whether to evaluate the generations asynchronously. Defaults to True.
        save_intermittently (bool, optional): Whether to save scores intermittently during evaluation. Defaults to True.
        ds_batch_size (int, optional): The batch size for querying the API. Defaults to 1000.
        model_batch_size (int, optional): The batch size for evaluating the generations. Defaults to 8.
        remove_intermediate_files (bool, optional): Whether to remove intermediate files. Defaults to True.
        output_dir (str, optional): The directory to save the scores file. Defaults to "model_scores".

    Raises:
        FileNotFoundError: If the data file is not found.

    Returns:
        None
    """
    data = None
    if (model_name is None or seed is None) and model_gen_dir is None:
        raise ValueError("Either model_name and seed or dir must be provided.")

    if not model_name:
        split = model_gen_dir.split("_seed")
        model_name = split[0]
        seed = f"seed{split[1]}"

    if not model_gen_dir:
        model_gen_dir = Path(gen_dir) / f"{model_name}_{seed}"

    model_score_dir = Path(score_dir) / f"{model_name}_{seed}"
    model_score_dir.mkdir(parents=True, exist_ok=True)

    short_string = "_short" if short else ""
    cont_string = "continuation_" if only_continuation else ""

    model_score_path = model_score_dir / f"{cont_string}scores{short_string}.json"

    # get continuation file
    if overwrite or not model_score_path.exists():
        data = load_entire_json(model_gen_dir / "continuations.json", return_data=True)

        metadata = data["metadata"]
        generations = data["continuations"]

        # only for translation evaluation (bleu/rouge)
        ground_truths = data.get("ground_truths", None)
        # only for prompt + continuation evaluation, e.g. toxicity
        prompts = data.get("prompts", None)

        if not only_continuation:
            if prompts is None:
                logger.error("Prompts must be provided for prompt + continuation evaluation.")
                sys.exit(1)

            generations = [f"{prompt} {continuation}" for prompt, continuation in zip(prompts, generations)]

        # Filter out entries where ground_truths are empty strings
        if ground_truths is not None:
            # Identify indices where ground_truths are not empty
            valid_indices = [index for index, gt in enumerate(ground_truths) if gt.strip() != ""]
            if not valid_indices:
                logger.error("No valid ground_truth data found.")
                sys.exit(1)

            # Filter generations, ground_truths, and prompts based on valid indices
            generations = [generations[i] for i in valid_indices]
            ground_truths = [ground_truths[i] for i in valid_indices]
            if prompts is not None:
                prompts = [prompts[i] for i in valid_indices]

        if verbose:
            count = sum(len(cont) < 10 for cont in generations)
            logger.info(f"Number of continuations with less than 10 tokens: {count}")
            count_empty = sum(len(cont) == 0 for cont in generations)
            logger.info(f"Number of empty continuations: {count_empty}")

        scores = []
        num_samples = len(generations)
        logger.info(f"Evaluating {num_samples} samples.")

        for i in tqdm(
            range(0, num_samples, ds_batch_size),
            disable=not verbose,
        ):
            start = time.time()

            batch_generations = generations[i : i + ds_batch_size]

            if ground_truths is not None:
                batch_ground_truths = ground_truths[i : i + ds_batch_size]
            else:
                batch_ground_truths = None

            new_scores = eval_on_metric(
                metric,
                batch_generations,
                ground_truths=batch_ground_truths,
                asynchronously=asynchronously,
                batch_size=model_batch_size,
            )

            scores.extend(new_scores)

            if i > 0 and i % 10000 == 0 and save_intermittently:
                _save_intermittently(
                    scores, model_score_dir, metric, i, metadata, ds_batch_size, only_continuation=only_continuation
                )

            end = time.time()
            if verbose:
                logger.info(f"Processing batch {i} to {i+ds_batch_size} took {round(end-start, 3)} seconds")

        output_data = {"metadata": metadata, f"{metric}_scores": scores}
        with open(model_score_path, "w") as file:
            json.dump(output_data, file, indent=4)

        logger.info(f"Evaluation completed. File stored in {model_score_path} ")

        if remove_intermediate_files:
            pattern = f"{cont_string}scores_*.json"
            cleanup_files(model_score_dir, pattern)


def evaluate_all_models(
    metric: str = "perspective",
    overwrite=True,
    asynchronously=True,
    save_intermittently=True,
    ds_batch_size=1000,
    model_batch_size=8,
    remove_intermediate_files=True,
    gen_dir="model_outputs",
    score_dir="model_scores",
    only_continuations=True,
):
    for model_gen_dir in tqdm(os.listdir(gen_dir)):
        start = time.time()

        logger.info(f"Start evaluating model {model_gen_dir} using metric {metric}.")

        try:
            evaluate_single_model(
                model_gen_dir=model_gen_dir,
                metric=metric,
                overwrite=overwrite,
                asynchronously=asynchronously,
                save_intermittently=save_intermittently,
                ds_batch_size=ds_batch_size,
                model_batch_size=model_batch_size,
                remove_intermediate_files=remove_intermediate_files,
                score_dir=score_dir,
                verbose=False,
                only_continuation=only_continuations,
            )
            end = time.time()
            logger.info(f"Model {model_gen_dir} took {round(end-start, 3)} seconds to evaluate.")

        except IndexError:
            logger.error(f"{model_gen_dir} does not contain the correct files. Skipping...")
            continue


def _save_intermittently(scores, model_score_dir, metric, i, metadata, ds_batch_size, only_continuation=False):
    cont_str = "continuation_" if only_continuation else ""
    current_scores_path = os.path.join(model_score_dir, f"{cont_str}scores_{i}.json")
    temp_data = {
        "metadata": metadata,
        f"{metric}_scores": scores,
    }
    # Assert that the length of scores is as expected
    expected_length = i + ds_batch_size
    assert (
        len(scores) == expected_length
    ), f"The current number of scores is not as expected: {len(scores)} vs {expected_length}"
    with open(current_scores_path, "w") as file:
        json.dump(temp_data, file, indent=4)


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
):
    """ """

    file_path1 = f"{score_dir}/{model_name1}_{seed1}"
    file_path2 = f"{score_dir}/{model_name2}_{seed2}"
    new_folder_path = Path(test_dir) / f"{model_name1}_{seed1}_{model_name2}_{seed2}"
    new_folder_path.mkdir(parents=True, exist_ok=True)

    cont_string = "continuation_" if only_continuations else ""

    common_scores_file_path = new_folder_path / f"{cont_string}scores.json"
    if overwrite or not common_scores_file_path.exists():
        file_name1 = f"{file_path1}/{cont_string}scores.json"
        file_name2 = f"{file_path2}/{cont_string}scores.json"

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


def cleanup_files(directory, pattern, verbose=True):
    files_to_delete = glob.glob(os.path.join(directory, pattern))
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            if verbose:
                logger.info(f"Deleted file: {file_path}")
        except OSError as e:
            logger.error(f"Error deleting file {file_path}: {e.strerror}")


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
):
    """ """

    # Fix random seed to be different for each fold_size, such that the folds always have different samples.
    random.seed(fold_size)

    cont_string = "continuation_" if only_continuations else ""

    directory = f"{test_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}"
    file_pattern = f"{cont_string}scores_fold_*.json"

    # Cleanup existing fold files
    cleanup_files(directory, file_pattern, verbose=False)

    file_name = f"{test_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}/{cont_string}scores.json"
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
        fold_file_path = f"{test_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}/{cont_string}scores_fold_{i}.json"
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
        )
        evaluate_single_model(
            model_name=model_name2,
            seed=seed2,
            metric=metric,
            overwrite=False,
            only_continuation=only_continuations,
            gen_dir=gen_dir,
            score_dir=score_dir,
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
    evaluate_all_models()
