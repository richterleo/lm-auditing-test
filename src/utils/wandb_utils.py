import os
import wandb
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
src_path = os.path.join(project_root, "src")
if project_root not in sys.path:
    sys.path.append(project_root)
if src_path not in sys.path:
    sys.path.append(src_path)

from utils.utils import create_run_string

wandb_api_key = os.getenv("WANDB_API_KEY")


def download_from_wandb(run_name: str, file_name: str = "test_outputs.zip"):
    """ """
    wandb.login(key=wandb_api_key)
    api = wandb.Api()
    run = api.run(run_name)

    files = [file for file in run.files() if file_name in file.name]
    file = files[0]
    file.download()

    # Finish the run
    wandb.finish()


def push_to_wandb(file_name="perspective_data.zip", project="toxicity_test", entity="LLM_Accountability"):
    """ """
    wandb.init(
        project=project,
        entity=entity,
        name=create_run_string(),
        tags=["all_data"],
    )

    wandb.save(file_name)
    wandb.finish()


if __name__ == "__main__":
    # push_to_wandb()
    pass
