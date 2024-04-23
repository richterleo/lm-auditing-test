import json
import random
import wandb

from datetime import datetime

def create_run_string():
    # Get current date and time
    current_datetime = datetime.now()
    # Format the datetime to a string
    datetime_str = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    # Create and return the string with "run" appended with the current date and time
    return f"run_{datetime_str}"


def get_random_prompts(dataset, num_examples=500):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    return(dataset[picks])


def log_scores(scores, prefix="tox"):
    # Create a dictionary with the tox_scores
    data = {f"{prefix}_scores": scores}
    
    # Define the filename with the current epoch number
    filename = f"{prefix}_scores.json"
    
    # Save tox_scores to a JSON file
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    # Log the JSON file to WandB
    wandb.save(filename)