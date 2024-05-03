import json
import random
import time
import wandb

from contextlib import contextmanager
from pathlib import Path
from datetime import datetime


def create_run_string():
    """
    Create unique string for wandb run name
    """
    current_datetime = datetime.now()
    datetime_str = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    return f"run_{datetime_str}"


def get_random_prompts(prompt_dataset, num_examples=500):
    """ """
    assert num_examples <= len(
        prompt_dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(prompt_dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(prompt_dataset) - 1)
        picks.append(pick)
    return prompt_dataset[picks]


def log_scores(scores, prefix="tox"):
    # Create a dictionary with the tox_scores
    data = {f"{prefix}_scores": scores}

    # Define the filename with the current epoch number
    filename = f"{prefix}_scores.json"

    # Save tox_scores to a JSON file
    with open(filename, "w") as f:
        json.dump(data, f)

    # Log the JSON file to WandB
    wandb.save(filename)


def get_scores_from_wandb(
    run_id,
    project_name="toxicity_evaluation",
    prefix="tox",
    user_name="richter-leo94",
    return_file_path=True,
):
    # Initialize W&B API
    api = wandb.Api()

    # Path to the file you want to download
    file_path = f"{prefix}_scores.json"

    run_name = f"{user_name}/{project_name}/{run_id}"

    # Access the run
    run = api.run(run_name)

    # Define the path to the folder you want to check and create
    folder_path = Path(f"outputs/{run_id}")

    # Check if the folder exists
    if not folder_path.exists():
        # Create the folder if it does not exist
        folder_path.mkdir(parents=True, exist_ok=True)

    # Download the file
    run.file(file_path).download(root=folder_path, replace=True)

    if return_file_path:
        return folder_path / file_path


@contextmanager
def time_block(label):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{label}: {end - start} seconds")
