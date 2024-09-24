import json
import logging
import os
import glob
import evaluate
import typing
import random
import sys
import numpy as np
import wandb
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
from utils.generate_and_evaluate import eval_on_metric
from utils.remove_unused_epoch_key import remove_zero_key_and_flatten
from logging_config import setup_logging

# setup_logging()
logger = logging.getLogger(__name__)


# error handling
def load_json(filepath):
    try:
        with open(filepath, "r") as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {e}")
        line, column = e.lineno, e.colno
        logger.error(f"Error at line {line}, column {column}")
        # Optionally, you can print the problematic line
        with open(filepath, "r") as file:
            lines = file.readlines()
            problematic_line = lines[line - 1]
            logger.debug(f"Problematic line: {problematic_line.strip()}")
        # Handle the error or re-raise it
        raise


def load_entire_json(filepath):
    try:
        with open(filepath, "r") as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {e}")
        line, column = e.lineno, e.colno
        logger.error(f"Error at line {line}, column {column}")
        # Print a few lines around the error to help debug
        with open(filepath, "r") as file:
            lines = file.readlines()
            start = max(0, line - 3)
            end = min(len(lines), line + 2)
            for i in range(start, end):
                logger.debug(f"{i + 1}: {lines[i].strip()}")
        # Re-raise the error to avoid further processing
        raise


def load_json_skipping_errors(filepath):
    valid_data = []
    num_errors = 0
    with open(filepath, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                valid_data.append(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {i + 1} due to JSONDecodeError: {e}")
                logger.debug(f"Problematic line: {line.strip()}")
                num_errors += 1
    return valid_data, num_errors


def evaluate_single_model(
    model_name: Optional[str] = None,
    seed: Optional[str] = None,
    model_dir: Optional[str] = None,
    metric: str = "toxicity",
    overwrite=True,
    asynchronously=True,
    use_wandb=True,
    entity="LLM_Accountability",
    save_intermittently=True,
    ds_batch_size=1000,
    model_batch_size=8,
    remove_intermediate_files=True,
    gen_dir="model_outputs",
    output_dir="model_scores",
    verbose=True,
    only_continuation=False,
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
        use_wandb (bool, optional): Whether to use wandb for logging. Defaults to True.
        entity (str, optional): The wandb entity to use. Defaults to "LLM_Accountability".
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
    if (model_name is None or seed is None) and model_dir is None:
        raise ValueError("Either model_name and seed or dir must be provided.")

    if not model_name:
        split = model_dir.split("_seed")
        model_name = split[0]
        seed = f"seed{split[1]}"

    if use_wandb:
        wandb.init(
            project=f"{metric}_evaluation",
            entity=entity,
            name=create_run_string(),
            config={"model_name": model_name, "seed": seed},
            tags=["evaluate_model"],
        )

    gen_dir = f"{gen_dir}/{model_name}_{seed}"
    score_dir = f"{output_dir}/{model_name}_{seed}"

    # check if folder exists already
    if not Path(score_dir).exists():
        Path(score_dir).mkdir(parents=True, exist_ok=True)
    score_path = (
        Path(score_dir) / f"{metric}_scores.json"
        if not only_continuation
        else Path(score_dir) / f"{metric}_continuation_scores.json"
    )

    # get continuation file
    if overwrite or not os.path.exists(score_path):
        for file_name in os.listdir(gen_dir):
            if "continuations" in file_name:
                # older versions have unnecessary 0 key
                data = remove_zero_key_and_flatten(os.path.join(gen_dir, file_name), return_data=True)
                break

        if data is None:
            raise FileNotFoundError

        filtered_dict = {k: v for k, v in data.items() if k != "metadata"}

        # concatenate prompt and continuation
        if only_continuation:
            generations = filtered_dict["continuations"]

            count = sum(len(cont) < 10 for cont in generations)
            logger.info(f"Number of continuations with less than 10 tokens: {count}")
            count_empty = sum(len(cont) == 0 for cont in generations)
            logger.info(f"Number of empty continuations: {count_empty}")
        else:
            generations = [
                f"{prompt} {continuation}"
                for prompt, continuation in zip(filtered_dict["prompts"], filtered_dict["continuations"])
            ]

        # if we have a lot of generations, we need to query the API in batches
        if len(generations) > ds_batch_size:
            scores = []
            for i in tqdm(
                range(0, len(generations), ds_batch_size),
                disable=not verbose,
            ):
                start = time.time()

                new_scores = eval_on_metric(
                    metric,
                    generations[i : i + ds_batch_size],
                    asynchronously=asynchronously,
                    batch_size=model_batch_size,
                )

                scores.extend(new_scores)

                if i > 0 and i % 10000 == 0 and save_intermittently:
                    _save_intermittently(
                        scores, score_dir, metric, i, data, ds_batch_size, only_continuation=only_continuation
                    )

                end = time.time()
                if verbose:
                    logger.info(f"Processing batch {i} to {i+ds_batch_size} took {round(end-start, 3)} seconds")

        else:
            if verbose:
                logger.warning(
                    f"We are not batching, because the length of the dataset is small: {len(generations)} samples"
                )
            scores = eval_on_metric(
                metric,
                generations,
                asynchronously=asynchronously,
                batch_size=model_batch_size,
            )

        data[f"{metric}_scores"] = scores

        assert (
            len(data[f"{metric}_scores"]) == len(data["prompts"])
        ), f"Number of scores is not the same as number of prompts: {len(data[f'{metric}_scores'])} and {len(data['prompts'])}"
        with open(score_path, "w") as file:
            json.dump(data, file, indent=4)

        if verbose:
            logger.info(f"Evaluation should be completed. File stored in {score_path} ")

        if remove_intermediate_files:
            pattern = f"{metric}_scores_*.json" if not only_continuation else f"{metric}_continuation_scores_*.json"
            cleanup_files(score_dir, pattern)

        if use_wandb:
            wandb.save(score_path)

        wandb.finish()


def evaluate_all_models(
    metric: str = "toxicity",
    overwrite=True,
    asynchronously=True,
    use_wandb=False,  # This is changed now. We don't need to upload all evals.
    entity="LLM_Accountability",
    save_intermittently=True,
    ds_batch_size=1000,
    model_batch_size=8,
    remove_intermediate_files=True,
    gen_dir="model_outputs",
    output_dir="model_scores",
    only_continuations=True,
):
    for model_dir in tqdm(os.listdir(gen_dir)):
        start = time.time()

        logger.info(f"Start evaluating model {model_dir} using metric {metric}.")

        try:
            evaluate_single_model(
                model_dir=model_dir,
                metric=metric,
                overwrite=overwrite,
                asynchronously=asynchronously,
                use_wandb=use_wandb,
                entity=entity,
                save_intermittently=save_intermittently,
                ds_batch_size=ds_batch_size,
                model_batch_size=model_batch_size,
                remove_intermediate_files=remove_intermediate_files,
                output_dir=output_dir,
                verbose=False,
                only_continuation=only_continuations,
            )
            end = time.time()
            logger.info(f"Model {model_dir} took {round(end-start, 3)} seconds to evaluate.")

        except IndexError:
            logger.error(f"{model_dir} does not contain the correct files. Skipping...")
            continue


def _save_intermittently(scores, score_dir, metric, i, data, ds_batch_size, only_continuation=False):
    current_scores_path = (
        os.path.join(score_dir, f"{metric}_scores_{i}.json")
        if not only_continuation
        else os.path.join(score_dir, f"{metric}_continuation_scores_{i}.json")
    )
    data[f"{metric}_scores"] = scores
    assert (
        len(data[f"{metric}_scores"]) == i + ds_batch_size
    ), f"The current number of scores is not the same as the index: {len(data[f'{metric}_scores'])} and {i}"
    with open(current_scores_path, "w") as file:
        json.dump(data, file, indent=4)


def create_common_json(
    model_name1,
    seed1,
    model_name2,
    seed2,
    metric="toxicity",
    overwrite=True,
    use_wandb=False,
    entity="LLM_Accountability",
    score_path="model_scores",
    output_path="test_outputs",
    only_continuations=True,
):
    """ """
    file_path1 = f"{score_path}/{model_name1}_{seed1}"
    file_path2 = f"{score_path}/{model_name2}_{seed2}"
    new_folder_path = Path(output_path) / f"{model_name1}_{seed1}_{model_name2}_{seed2}"

    if use_wandb:
        wandb.init(
            project=f"{metric}_evaluation",
            entity=entity,
            name=create_run_string(),
            config={
                "model_name1": model_name1,
                "seed": seed1,
                "model_name2": model_name2,
                "seed2": seed2,
            },
            tags=["create_common_json"],
        )

    common_scores_file_path = (
        new_folder_path / f"{metric}_scores.json"
        if not only_continuations
        else new_folder_path / f"{metric}_continuation_scores.json"
    )
    if overwrite or not common_scores_file_path.exists():
        if not new_folder_path.exists():
            new_folder_path.mkdir(parents=True, exist_ok=True)

        file_name1 = (
            f"{file_path1}/{metric}_scores.json"
            if not only_continuations
            else f"{file_path1}/{metric}_continuation_scores.json"
        )
        file_name2 = (
            f"{file_path2}/{metric}_scores.json"
            if not only_continuations
            else f"{file_path2}/{metric}_continuation_scores.json"
        )
        with open(file_name1, "r") as file1, open(file_name2, "r") as file2:
            data1 = json.load(file1)
            data2 = json.load(file2)

        data = defaultdict(list)
        data["metadata1"] = data1["metadata"]
        data["metadata2"] = data2["metadata"]

        filtered_data1 = {k: v for k, v in data1.items() if k != "metadata"}
        filtered_data2 = {k: v for k, v in data2.items() if k != "metadata"}

        # if both lists are the same length, then we just trust that they're the same and ordered correctly.
        if len(filtered_data1["prompts"]) == len(filtered_data2["prompts"]):
            print(
                f"We trust that both data have the same prompts, e.g. {filtered_data1['prompts'][0], filtered_data2['prompts'][0]}"
            )
            assert filtered_data1["prompts"][0] == filtered_data2["prompts"][0], "Prompts are not the same."
            #     data["prompts"] = filtered_data1["prompts"]
            #     data["continuations1"] = filtered_data1["continuations"]
            #     data["continuations2"] = filtered_data2["continuations"]
            #     data[f"{metric}_scores1"] = filtered_data1[f"{metric}_scores"]
            #     data[f"{metric}_scores2"] = filtered_data2[f"{metric}_scores"]

            # else:
            #     common_prompts = list(set(filtered_data1["prompts"]) & set(filtered_data2["prompts"]))

            #     # Extract data for common prompts
            #     for prompt in common_prompts:
            #         data["prompts"].append(prompt)
            #         index1 = filtered_data1["prompts"].index(prompt)
            #         index2 = filtered_data2["prompts"].index(prompt)

            #         data["continuations1"].append(filtered_data1["continuations"][index1])
            #         data["continuations2"].append(filtered_data2["continuations"][index2])
            #         data[f"{metric}_scores1"].append(filtered_data1[f"{metric}_scores"][index1])
            #         data[f"{metric}_scores2"].append(filtered_data2[f"{metric}_scores"][index2])

            #     # Check if the number of common prompts is the same as the number of scores
            # data["prompts"] = filtered_data1["prompts"]
            # data["continuations1"] = filtered_data1["continuations"]
            # data["continuations2"] = filtered_data2["continuations"]

            for score1, score2 in zip(filtered_data1[f"{metric}_scores"], filtered_data2[f"{metric}_scores"]):
                if not (np.isnan(score1) or np.isnan(score2)):
                    data[f"{metric}_scores1"].append(score1)
                    data[f"{metric}_scores2"].append(score2)

            logger.warning(
                f"Discarding {len(filtered_data1[f'{metric}_scores']) - len(data[f'{metric}_scores1'])} NaN scores."
            )

        with open(common_scores_file_path, "w") as file:
            # json.dump(data, file, indent=4)
            json.dump(data, file)

        if use_wandb:
            wandb.save(common_scores_file_path)
            wandb.finish()


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
    metric="toxicity",
    fold_size=4000,
    overwrite=True,
    output_dir="test_outputs",
    only_continuations=True,
):
    """ """
    # Fix random seed to be different for each fold_size, such that the folds always have different samples.
    random.seed(fold_size)

    directory = f"{output_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}"
    file_pattern = (
        f"{metric}_scores_fold_*.json" if not only_continuations else f"{metric}_continuation_scores_fold_*.json"
    )

    # Cleanup existing fold files
    cleanup_files(directory, file_pattern, verbose=False)

    try:
        file_name = (
            f"{output_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}/{metric}_scores.json"
            if not only_continuations
            else f"{output_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}/{metric}_continuation_scores.json"
        )
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
                f"{output_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}/{metric}_scores_fold_{i}.json"
                if not only_continuations
                else f"{output_dir}/{model_name1}_{seed1}_{model_name2}_{seed2}/{metric}_continuation_scores_fold_{i}.json"
            )
            if overwrite or not os.path.exists(fold_file_path):
                fold_data = defaultdict(list)
                fold_data["metadata1"] = metadata1
                fold_data["metadata2"] = metadata2

                for key, value in data.items():
                    if key in ["prompts", "continuations1", "continuations2"]:
                        fold_data[key] = [value[j] for j in batch]
                    elif key in [f"{metric}_scores1", f"{metric}_scores2"]:
                        fold_data[key] = [value[j] for j in batch]

                with open(fold_file_path, "w") as file:
                    json.dump(fold_data, file, indent=4)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None


def create_folds_from_evaluations(
    model_name1,
    seed1,
    model_name2,
    seed2,
    metric="toxicity",
    fold_size=4000,
    overwrite=True,
    random_seed=0,
    use_wandb=False,
    only_continuations=True,
):
    try:
        create_common_json(
            model_name1, seed1, model_name2, seed2, metric, overwrite=overwrite, only_continuations=only_continuations
        )
    except FileNotFoundError as e:
        logger.info(f"File not found: {e}. Trying to create the folds from generations.")
        evaluate_single_model(
            model_name=model_name1,
            seed=seed1,
            metric=metric,
            overwrite=False,
            use_wandb=use_wandb,
            only_continuation=only_continuations,
        )
        evaluate_single_model(
            model_name=model_name2,
            seed=seed2,
            metric=metric,
            overwrite=False,
            use_wandb=use_wandb,
            only_continuation=only_continuations,
        )

        create_common_json(
            model_name1, seed1, model_name2, seed2, metric, overwrite=overwrite, only_continuations=only_continuations
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
    setup_logging(log_file="evaluate_only_continuations.log")
    logger = logging.getLogger(__name__)

    evaluate_all_models(metric="perspective", overwrite=False, only_continuations=True)

    # task_models = [
    #     "gemma-1.1-7b-it",
    #     "Mistral-7B-Instruct-v0.2_seed2000",
    #     "Llama-3-8B-ckpt10_seed1000",
    # ]

    # task_models = ["LLama-3-8b-Uncensored"]

    # for task_model in task_models:
    #     create_common_json(
    #         model_name1="Meta-Llama-3-8B-Instruct",
    #         seed1="seed2000",
    #         model_name2=task_model,
    #         seed2="seed1000",
    #         metric="perspective",
    #     )

    # create_common_json(
    #     "Meta-Llama-3-8B-Instruct",
    #     "seed2000",
    #     "Meta-Llama-3-8B-Instruct-hightemp",
    #     "seed1000",
    #     metric="perspective",
    #     only_continuations=True,
    # )
    # create_common_json(
    #     "Meta-Llama-3-8B-Instruct",
    #     "seed2000",
    #     "1-Meta-Llama-3-8B-Instruct",
    #     "seed1000",
    #     metric="perspective",
    #     only_continuations=True,
    # )
    # create_common_json(
    #     "Meta-Llama-3-8B-Instruct",
    #     "seed2000",
    #     "2-Meta-Llama-3-8B-Instruct",
    #     "seed1000",
    #     metric="perspective",
    #     only_continuations=True,
    # )
    # create_common_json(
    #     "Meta-Llama-3-8B-Instruct",
    #     "seed2000",
    #     "Llama-3-8B-ckpt1",
    #     "seed2000",
    #     metric="perspective",
    #     only_continuations=True,
    # )

    create_common_json(
        "Meta-Llama-3-8B-Instruct",
        "seed2000",
        "LLama-3-8b-Uncensored",
        "seed1000",
        metric="perspective",
        only_continuations=True,
    )
