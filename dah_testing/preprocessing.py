import json
import os
import typing
import random
import sys

from collections import defaultdict
from pathlib import Path

# Add the parent directory of utils to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import download_file_from_wandb
from dah_testing.dataloader import ScoresDataset
from utils.generate_and_evaluate import eval_on_metric


def evaluate_single_model(run_id: str, metric: str, overwrite=True):
    """ """
    file_path = f"outputs/{run_id}"
    file_found = False

    for file_name in os.listdir(file_path):
        if "continuations.json" in file_name:
            file_found = True
            with open(os.path.join(file_path, file_name), "r") as file:
                data = json.load(file)
            break

    if not file_found:
        file_path = download_file_from_wandb(
            run_id=run_id, pattern="continuations.json", return_file_path=True
        )
        with open(os.path.join(file_path), "r") as file:
            data = json.load(file)

    filtered_dict = {k: v for k, v in data.items() if k != "metadata"}

    for epoch, d in filtered_dict.items():
        concatenated_generations = [
            f"{prompt} {continuation}"
            for prompt, continuation in zip(d["prompts"], d["continuations"])
        ]

        scores = eval_on_metric(
            metric,
            concatenated_generations,
        )

        data[epoch][f"{metric}_scores"] = scores

    file_path = os.path.join(file_path, f"{metric}_scores.json")
    if overwrite or not os.path.exists(file_path):
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)


def create_common_json(run_id1, run_id2, metric, epoch=0, overwrite=True):
    """ """
    file_path1 = f"outputs/{run_id1}"
    file_path2 = f"outputs/{run_id2}"
    new_folder_path = Path("outputs") / f"{run_id1}_{run_id2}"

    if not new_folder_path.exists():
        new_folder_path.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(file_path1, f"{metric}_scores.json"), "r") as file1, open(
        os.path.join(file_path2, f"{metric}_scores.json"), "r"
    ) as file2:
        data1 = json.load(file1)
        data2 = json.load(file2)

    data = defaultdict(lambda: defaultdict(list))
    data["metadata1"] = data1["metadata"]
    data["metadata2"] = data2["metadata"]

    filtered_data1 = {k: v for k, v in data1.items() if k != "metadata"}[str(epoch)]
    filtered_data2 = {k: v for k, v in data2.items() if k != "metadata"}[str(epoch)]

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
        data[f"{metric}_scores1"].append(filtered_data1[f"{metric}_scores"][index1])
        data[f"{metric}_scores2"].append(filtered_data2[f"{metric}_scores"][index2])

    file_path = new_folder_path / f"{metric}_scores.json"
    if overwrite or not file_path.exists():
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)


def create_folds(run_id1, run_id2, metric, fold_size=4000, overwrite=True):
    """ """
    try:
        with open(f"outputs/{run_id1}_{run_id2}/{metric}_scores.json", "r") as file:
            data = json.load(file)

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

        for i, batch in enumerate(index_batches):
            fold_data = defaultdict(list)
            fold_data["metadata1"] = metadata1
            fold_data["metadata2"] = metadata2

            for key, value in data.items():
                if key in ["prompts", "continuations1", "continuations2"]:
                    fold_data[key] = [value[j] for j in batch]
                elif key in [f"{metric}_scores1", f"{metric}_scores2"]:
                    fold_data[key] = [value[j] for j in batch]

            file_path = f"outputs/{run_id1}_{run_id2}/{metric}_scores_fold_{i}.json"
            if overwrite or not os.path.exists(file_path):
                with open(file_path, "w") as file:
                    json.dump(fold_data, file, indent=4)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None


def create_folds_from_generations(
    run_id1, run_id2, metric, fold_size=4000, overwrite=True
):
    evaluate_single_model(run_id1, metric, overwrite=overwrite)
    evaluate_single_model(run_id2, metric, overwrite=overwrite)

    create_common_json(run_id1, run_id2, metric, overwrite=overwrite)
    create_folds(run_id1, run_id2, metric, fold_size=fold_size, overwrite=overwrite)


if __name__ == "__main__":
    run_id1 = "c5wloj6r"
    run_id2 = "zxd1hh7d"
    # evaluate_single_model(run_id2, "perspective")
    create_common_json(run_id1, run_id2, "perspective")
