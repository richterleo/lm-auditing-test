import json
import os
import evaluate
import typing
import random
import sys
import numpy as np
import wandb
import orjson  # Using orjson for faster JSON operations
import time


from collections import defaultdict
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from transformers import pipeline

# Add the parent directory of utils to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import download_file_from_wandb, time_block, create_run_string
from utils.generate_and_evaluate import eval_on_metric


# error handling
def load_json(filepath):
    try:
        with open(filepath, "r") as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        line, column = e.lineno, e.colno
        print(f"Error at line {line}, column {column}")
        # Optionally, you can print the problematic line
        with open(filepath, "r") as file:
            lines = file.readlines()
            problematic_line = lines[line - 1]
            print(f"Problematic line: {problematic_line.strip()}")
        # Handle the error or re-raise it
        raise


def load_entire_json(filepath):
    try:
        with open(filepath, "r") as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        line, column = e.lineno, e.colno
        print(f"Error at line {line}, column {column}")
        # Print a few lines around the error to help debug
        with open(filepath, "r") as file:
            lines = file.readlines()
            start = max(0, line - 3)
            end = min(len(lines), line + 2)
            for i in range(start, end):
                print(f"{i + 1}: {lines[i].strip()}")
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
                print(f"Skipping line {i + 1} due to JSONDecodeError: {e}")
                print(f"Problematic line: {line.strip()}")
                num_errors += 1
    return valid_data, num_errors


# def evaluate_single_model(
#     metric: str, run_id: Optional[str] = None, overwrite=True, asynchronously=True
# ):
#     """ """
#     file_path = f"outputs/{run_id}"
#     data = None

#     try:
#         for file_name in os.listdir(file_path):
#             if "continuations.json" in file_name:
#                 with open(os.path.join(file_path, file_name), "r") as file:
#                     data = json.load(file)
#                 break

#         if data is None:
#             raise FileNotFoundError

#     except FileNotFoundError:
#         file_path = download_file_from_wandb(
#             run_id=run_id,
#             project_name="continuations",
#             pattern="continuations",
#             return_file_path=True,
#         )
#         try:
#             with open(os.path.join(file_path), "r") as file:
#                 data = json.load(file)
#         except TypeError as e:
#             raise FileNotFoundError(f"No file found on wandb, error: {e}")

#     filtered_dict = {k: v for k, v in data.items() if k != "metadata"}

#     for epoch, d in filtered_dict.items():
#         concatenated_generations = [
#             f"{prompt} {continuation}"
#             for prompt, continuation in zip(d["prompts"], d["continuations"])
#         ]

#         # if we have a lot of generations, we need to query the API in batches
#         if len(concatenated_generations) > 100:
#             scores = []
#             for i in range(0, len(concatenated_generations), 100):
#                 print(f"Processing batch {i} to {i+100}")
#                 new_scores = eval_on_metric(
#                     metric,
#                     concatenated_generations[i : i + 100],
#                     asynchronously=asynchronously,
#                 )
#                 scores.extend(new_scores)

#             assert (
#                 len(scores) == len(concatenated_generations)
#             ), f"Did not get all scores: only {len(scores)} scores, but {len(concatenated_generations)} generations"
#         else:
#             scores = eval_on_metric(
#                 metric,
#                 concatenated_generations,
#                 asynchronously=asynchronously,
#             )

#         data[epoch][f"{metric}_scores"] = scores

#     file_path = os.path.join(file_path, f"{metric}_scores.json")
#     if overwrite or not os.path.exists(file_path):
#         with open(file_path, "w") as file:
#             json.dump(data, file, indent=4)


def evaluate_single_model(
    model_name: str,
    seed: str,
    metric,
    overwrite=True,
    asynchronously=True,
    use_wandb=True,
    entity="LLM_Accountability",
    save_intermittently=True,
    ds_batch_size=1000,
    model_batch_size=8,
):
    """
    Evaluate a single model and save the scores in the same directory as the generations.
    """

    if use_wandb:
        wandb.init(
            project=f"{metric}_evaluation",
            entity=entity,
            name=create_run_string(),
            config={"model_name": model_name, "seed": seed},
            tags=["evaluate_single_model"],
        )

    gen_file_path = f"model_outputs/{model_name}_{seed}"
    print(gen_file_path)
    scores_file_path = os.path.join(gen_file_path, f"{metric}_scores.json")
    if overwrite or not os.path.exists(scores_file_path):
        for file_name in os.listdir(gen_file_path):
            if "continuations" in file_name:
                with open(os.path.join(gen_file_path, file_name), "r") as file:
                    data = json.load(file)
                break

        if data is None:
            raise FileNotFoundError

        filtered_dict = {k: v for k, v in data.items() if k != "metadata"}

        for epoch, d in filtered_dict.items():
            concatenated_generations = [
                f"{prompt} {continuation}"
                for prompt, continuation in zip(d["prompts"], d["continuations"])
            ]

            data[epoch][f"{metric}_scores"] = []

            # if we have a lot of generations, we need to query the API in batches
            if len(concatenated_generations) > 100 and metric == "perspective":
                scores = []
                for i in range(0, len(concatenated_generations), 100):
                    print(f"Processing batch {i} to {i+100}")
                    new_scores = eval_on_metric(
                        metric,
                        concatenated_generations[i : i + 100],
                        asynchronously=asynchronously,
                    )
                    scores.extend(new_scores)

                assert (
                    len(scores) == len(concatenated_generations)
                ), f"Did not get all scores: only {len(scores)} scores, but {len(concatenated_generations)} generations"

            # tracking more to see why evaluations are so slow
            elif len(concatenated_generations) > 1000:
                scores = []
                # metric_model = evaluate.load(metric) not on GPU

                # TODO: this is hardcoded now, should not be in future
                model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
                toxic_classifier = pipeline(
                    "text-classification",
                    model=model_name,
                    top_k=99999,
                    truncation=True,
                    device_map="auto",
                )
                print(f"Evaluation metric using device: {toxic_classifier.device}")

                # TODO remove this
                start = time.time()

                for i in tqdm(range(0, len(concatenated_generations), ds_batch_size)):
                    end = time.time()
                    print(
                        f"Processing batch {i} to {i+ds_batch_size}. {i}th batch took {end-start} seconds"
                    )
                    start = time.time()

                    toxic_or_not_toxic_scores = toxic_classifier(
                        concatenated_generations[i : i + ds_batch_size],
                        batch_size=model_batch_size,
                    )
                    new_toxic_scores = [
                        score[1]["score"] for score in toxic_or_not_toxic_scores
                    ]  # r["score"] for r in pred_toxic if r["label"] == "hate"

                    scores.extend(new_toxic_scores)

                    if i > 0 and i % 10000 == 0 and save_intermittently:
                        current_scores_path = os.path.join(
                            gen_file_path, f"{metric}_scores_{i}.json"
                        )
                        data[epoch][f"{metric}_scores"] = scores
                        assert (
                            len(data[epoch][f"{metric}_scores"]) == i + ds_batch_size
                        ), f"The current number of scores is not the same as the index: {len(data[epoch][f'{metric}_scores'])} and {i}"
                        with open(current_scores_path, "w") as file:
                            json.dump(data, file, indent=4)

                        if use_wandb:
                            wandb.save(current_scores_path)

            else:
                print(
                    f"We are not batching, because the length of the dataset is small: {len(concatenated_generations)} samples"
                )
                scores = eval_on_metric(
                    metric, concatenated_generations, asynchronously=asynchronously
                )

            data[epoch][f"{metric}_scores"] = scores

            assert (
                len(data[epoch][f"{metric}_scores"]) == len(data[epoch]["prompts"])
            ), f"Number of scores is not the same as number of prompts: {len(data[epoch][f'{metric}_scores'])} and {len(data[epoch]['prompts'])}"
            with open(scores_file_path, "w") as file:
                json.dump(data, file, indent=4)

            print(f"Evaluation should be completed. File stored in {scores_file_path} ")

            if use_wandb:
                wandb.save(scores_file_path)
            wandb.finish()


def create_common_json(
    model_name1,
    seed1,
    model_name2,
    seed2,
    metric="toxicity",
    epoch1=0,
    epoch2=0,
    overwrite=True,
    use_wandb=False,
    entity="LLM_Accountability",
):
    """ """
    file_path1 = f"model_outputs/{model_name1}_{seed1}"
    file_path2 = f"model_outputs/{model_name2}_{seed2}"
    new_folder_path = (
        Path("model_outputs") / f"{model_name1}_{seed1}_{model_name2}_{seed2}"
    )

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

    common_scores_file_path = new_folder_path / f"{metric}_scores.json"
    if overwrite or not common_scores_file_path.exists():
        if not new_folder_path.exists():
            new_folder_path.mkdir(parents=True, exist_ok=True)

        with open(
            os.path.join(file_path1, f"{metric}_scores.json"), "r"
        ) as file1, open(
            os.path.join(file_path2, f"{metric}_scores.json"), "r"
        ) as file2:
            data1 = json.load(file1)
            data2 = json.load(file2)

        data = defaultdict(list)
        data["metadata1"] = data1["metadata"]
        data["metadata2"] = data2["metadata"]

        filtered_data1 = {k: v for k, v in data1.items() if k != "metadata"}[
            str(epoch1)
        ]
        filtered_data2 = {k: v for k, v in data2.items() if k != "metadata"}[
            str(epoch2)
        ]

        # if both lists are the same length, then we just trust that they're the same and ordered correctly.
        if len(filtered_data1["prompts"]) == len(filtered_data2["prompts"]):
            print(
                f"We trust that both data have the same prompts, e.g. {filtered_data1['prompts'][0], filtered_data2['prompts'][0]}"
            )
            data["prompts"] = filtered_data1["prompts"]
            data["continuations1"] = filtered_data1["continuations"]
            data["continuations2"] = filtered_data2["continuations"]
            data[f"{metric}_scores1"] = filtered_data1[f"{metric}_scores"]
            data[f"{metric}_scores2"] = filtered_data2[f"{metric}_scores"]

        else:
            common_prompts = list(
                set(filtered_data1["prompts"]) & set(filtered_data2["prompts"])
            )

            # Extract data for common prompts
            for prompt in common_prompts:
                data["prompts"].append(prompt)
                index1 = filtered_data1["prompts"].index(prompt)
                index2 = filtered_data2["prompts"].index(prompt)

                data["continuations1"].append(filtered_data1["continuations"][index1])
                data["continuations2"].append(filtered_data2["continuations"][index2])
                data[f"{metric}_scores1"].append(
                    filtered_data1[f"{metric}_scores"][index1]
                )
                data[f"{metric}_scores2"].append(
                    filtered_data2[f"{metric}_scores"][index2]
                )

        with open(common_scores_file_path, "w") as file:
            # json.dump(data, file, indent=4)
            json.dump(data, file)

        if use_wandb:
            wandb.save(common_scores_file_path)
            wandb.finish()


def create_common_json_fast(
    model_name1, seed1, model_name2, seed2, metric, epoch=0, overwrite=True
):
    file_path1 = f"model_outputs/{model_name1}_{seed1}"
    file_path2 = f"model_outputs/{model_name2}_{seed2}"
    new_folder_path = (
        Path("model_outputs") / f"{model_name1}_{seed1}_{model_name2}_{seed2}"
    )

    new_folder_path.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(file_path1, f"{metric}_scores.json"), "rb") as file1, open(
        os.path.join(file_path2, f"{metric}_scores.json"), "rb"
    ) as file2:
        data1 = orjson.loads(file1.read())
        data2 = orjson.loads(file2.read())

    data = defaultdict(list)
    data["metadata1"] = data1["metadata"]
    data["metadata2"] = data2["metadata"]

    filtered_data1 = data1.get(str(epoch), {})
    filtered_data2 = data2.get(str(epoch), {})

    common_prompts = set(filtered_data1.get("prompts", [])) & set(
        filtered_data2.get("prompts", [])
    )

    prompt_indices1 = {
        prompt: i for i, prompt in enumerate(filtered_data1.get("prompts", []))
    }
    prompt_indices2 = {
        prompt: i for i, prompt in enumerate(filtered_data2.get("prompts", []))
    }

    for prompt in common_prompts:
        data["prompts"].append(prompt)
        index1 = prompt_indices1[prompt]
        index2 = prompt_indices2[prompt]

        data["continuations1"].append(filtered_data1["continuations"][index1])
        data["continuations2"].append(filtered_data2["continuations"][index2])
        data[f"{metric}_scores1"].append(filtered_data1[f"{metric}_scores"][index1])
        data[f"{metric}_scores2"].append(filtered_data2[f"{metric}_scores"][index2])

    file_path = new_folder_path / f"{metric}_scores.json"
    if overwrite or not file_path.exists():
        with open(file_path, "wb") as file:
            file.write(orjson.dumps(data))


def create_folds(
    model_name1,
    seed1,
    model_name2,
    seed2,
    metric="toxicity",
    fold_size=4000,
    overwrite=True,
):
    """ """
    try:
        file_name = f"model_outputs/{model_name1}_{seed1}_{model_name2}_{seed2}/{metric}_scores.json"
        data = load_entire_json(file_name)

        # Extract metadata and other lists
        metadata1 = data["metadata1"]
        metadata2 = data["metadata2"]

        # Create batches
        total_num_samples = len(data["prompts"])
        indices = list(range(total_num_samples))
        random.shuffle(indices)
        index_batches = [
            indices[i : i + fold_size] for i in range(0, total_num_samples, fold_size)
        ]

        for i, batch in tqdm(enumerate(index_batches)):
            fold_file_path = f"model_outputs/{model_name1}_{seed1}_{model_name2}_{seed2}/{metric}_scores_fold_{i}.json"
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
        print(f"File not found: {e}")
        return None


def create_folds_from_generations(
    model_name1,
    seed1,
    model_name2,
    seed2,
    metric="toxicity",
    fold_size=4000,
    overwrite=True,
):
    evaluate_single_model(model_name1, seed1, metric, overwrite=overwrite)
    evaluate_single_model(model_name2, seed2, metric, overwrite=overwrite)

    create_common_json(
        model_name1, seed1, model_name2, seed2, metric, overwrite=overwrite
    )
    create_folds(
        model_name1,
        seed1,
        model_name2,
        seed2,
        metric,
        fold_size=fold_size,
        overwrite=overwrite,
    )


def create_folds_from_evaluations(
    model_name1,
    seed1,
    model_name2,
    seed2,
    metric="toxicity",
    fold_size=4000,
    overwrite=True,
):
    start = time.time()
    create_common_json(
        model_name1, seed1, model_name2, seed2, metric, overwrite=overwrite
    )

    end = time.time()
    print(f"Creating the common json takes {end-start} seconds")
    start = time.time()
    create_folds(
        model_name1,
        seed1,
        model_name2,
        seed2,
        metric,
        fold_size=fold_size,
        overwrite=overwrite,
    )
    end = time.time()
    print(f"Creating the folds takes {end-start} seconds.")


if __name__ == "__main__":
    # Put json file with generations in folder model_outputs/{model_name}_{seed}

    model_name = "gemma-1.1-7b-it"
    seed = "seed2000"

    model_name1 = "Llama-3-8B-ckpt4"  # change this to the checkpoint to evaluate
    # checkpoints still to evaluate: 6,7,8,9,10, all gemma models, base instruct model

    seed1 = "seed7000"  # change this to the current seed

    model_name2 = "Llama-3-8B-ckpt7"
    seed2 = "seed1000"

    evaluate_single_model(
        model_name1, seed1, "toxicity", overwrite=True, use_wandb=True
    )
    # create_common_json(model_name1, seed1, model_name2, seed2)
    # create_folds(model_name1, seed1, model_name2, seed2)

    # create_folds_from_evaluations(model_name1, seed1, model_name2, seed2)
