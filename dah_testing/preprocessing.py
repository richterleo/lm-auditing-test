import json
import os
import typing

from utils.utils import download_file_from_wandb
from dah_testing.dataloader import ScoresDataset
from utils.generate_and_evaluate import eval_on_metric


def evaluate_single_model(run_id: str, metric: str, epoch: int = 0):
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

    concatenated_generations = [
        f"{prompt} {continuation}"
        for prompt, continuation in zip(
            data[epoch]["prompts"], data[epoch]["continuations"]
        )
    ]

    scores = eval_on_metric(
        metric,
        concatenated_generations,
    )

    data[f"{metric}_scores"] = scores

    with open(os.path.join(file_path, f"{metric}_scores.json"), "w") as file:
        json.dump(data, file, indent=4)
