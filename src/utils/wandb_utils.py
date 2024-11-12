import sys
from pathlib import Path
import wandb

from os import getenv

# Get project paths using pathlib
project_root = Path(__file__).resolve().parents[2]  # Go up two levels
src_path = project_root / "src"

# Add paths to sys.path if not already present
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from utils.utils import create_run_string

wandb_api_key = getenv("WANDB_API_KEY")


def download_from_wandb(run_name: str, file_name: str = "test_outputs.zip") -> None:
    """
    Downloads a file from a WandB run.

    Args:
        run_name: The name of the run to download from
        file_name: The name of the file to download (default: 'test_outputs.zip')
    """
    wandb.login(key=wandb_api_key)
    api = wandb.Api()
    run = api.run(run_name)
    files = [file for file in run.files() if file_name in file.name]
    file = files[0]
    file.download()
    wandb.finish()


def push_to_wandb(
    file_name: str = "perspective_data.zip", project: str = "toxicity_test", entity: str = "LLM_Accountability"
) -> None:
    """
    Pushes a file to WandB.

    Args:
        file_name: Name of the file to push (default: 'perspective_data.zip')
        project: WandB project name (default: 'toxicity_test')
        entity: WandB entity/organization (default: 'LLM_Accountability')
    """
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
