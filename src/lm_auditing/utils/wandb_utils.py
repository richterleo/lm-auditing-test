import logging
import os
import shutil
import sys
import wandb

from pathlib import Path
from typing import Optional, Callable

from lm_auditing.utils.utils import create_run_string
from lm_auditing.utils.env_loader import get_required_env_var

# Get wandb API key with proper error handling
try:
    wandb_api_key = get_required_env_var("WANDB_API_KEY", "Required for experiment tracking")
except ValueError as e:
    wandb_api_key = None
    logging.warning(f"WANDB_API_KEY not found: {e}")

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parents[2]


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


def get_scores_from_wandb(
    run_id: str,
    project_name="toxicity_evaluation",
    prefix="toxicity",
    user_name="richter-leo94",
    return_file_path=True,
) -> Optional[Path]:
    """
    Helper function for downloading the scores file from a W&B run.

    Args:
        run_id: The ID of the W&B run (not identical to the run name)
        project_name: The name of the W&B project.
        prefix: The prefix of the file to download.
        user_name: The name of the W&B user.
        return_file_path: Whether to return the file path.

    Returns:
        (Optional) The path to the downloaded file.

    """
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


def download_file_from_wandb(
    run_path: Optional[str] = None,
    run_id: Optional[str] = None,
    project_name: Optional[str] = None,
    file_name: Optional[str] = None,
    pattern: Optional[str] = None,
    entity: str = "LLM_Accountability",
    return_file_path: bool = True,
    get_save_path: Optional[Callable] = None,
    metric: str = "perspective",
) -> Optional[Path]:
    """
    Helper function for downloading the scores file from a W&B run.

    Args:
        run_path: The path to the W&B run
        run_id: The ID of the W&B run (not identical to the run name)
        project_name: The name of the W&B project.
        file_name: The name of the file to download.
        pattern: Alternatively, download file with specific pattern
        user_name: The name of the W&B user.
        return_file_path: Whether to return the file path.

    Returns:
        (Optional) The path to the downloaded file.

    """
    assert file_name or pattern, "Either file_name or pattern must be provided"
    assert run_path or (run_id and project_name and entity), (
        "Either run_path or run_id, project_name and entity must be provided"
    )
    # Initialize W&B API
    api = wandb.Api()

    # Path to the file you want to download
    run_name = run_path if run_path else f"{entity}/{project_name}/{run_id}"
    run = api.run(run_name)

    if file_name:
        files = [file for file in run.files() if file_name in file.name]
    else:
        files = [file for file in run.files() if pattern in file.name]

    if not files:
        logger.error(f"No file found matching {'file_name' if file_name else 'pattern'}: {file_name or pattern}")
        return None

    file = files[0]

    try:
        # Define the path to the folder where the file will be saved
        if get_save_path:
            file_path = get_save_path(file.name, metric=metric)
        else:
            if not run_id:
                run_id = Path(run_path).name
            file_path = Path("outputs") / run_id / Path(file.name).name

        file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = file_path.parent / "temp_download"
        temp_dir.mkdir(exist_ok=True)
        file.download(root=temp_dir, replace=True)

        downloaded_file = next(temp_dir.rglob(file.name))
        downloaded_file.rename(file_path)

        # Delete temp folder
        shutil.rmtree(temp_dir)

        if return_file_path:
            return file_path

    except Exception as e:
        logger.error(f"Error downloading file: {file.name}: {e}")
        return None


def folder_from_model_and_seed(
    file_name, save_path: str = "model_outputs", dir_prefix: Optional[str] = None, metric: str = "perspective"
):
    """ """
    if dir_prefix is None:
        dir_prefix = metric

    save_path = SCRIPT_DIR / dir_prefix / save_path
    file_path = Path(file_name)

    folder_name = file_path.parent.stem.replace("_continuations", "")
    folder_path = save_path / folder_name
    new_file_path = folder_path / file_path.name

    return new_file_path


if __name__ == "__main__":
    # run_paths = ["LLM_Accountability/continuations/3yflpcqd"]
    # pattern = "continuations"

    # download_file_from_wandb(
    #     run_path=run_paths[0], pattern=pattern, get_save_path=folder_from_model_and_seed, metric="bleu"
    # )

    run_path = "LLM_Accountability/toxicity_test/ac5k8ha3"
    file_name = "perspective_data.zip"
    download_from_wandb(run_name=run_path, file_name=file_name)
