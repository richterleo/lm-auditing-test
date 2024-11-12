import json
import logging
import sys
import time

from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Add paths to sys.path if not already present
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.utils import (
    download_file_from_wandb,
    time_block,
    create_run_string,
    load_config,
    load_entire_json,
    cleanup_files,
)
from src.evaluation.score import eval_on_metric
from src.utils.legacy_utils import remove_zero_key_and_flatten
from logging_config import setup_logging

# setup_logging()
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


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
    gen_path = Path(gen_dir)
    for model_gen_dir in tqdm(list(gen_path.iterdir())):
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
    current_scores_path = Path(model_score_dir) / f"{cont_str}scores_{i}.json"
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


if __name__ == "__main__":
    evaluate_all_models()
