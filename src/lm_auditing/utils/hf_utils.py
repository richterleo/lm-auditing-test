from huggingface_hub import Repository, hf_hub_download, snapshot_download
import os
import subprocess
import shutil
from huggingface_hub import HfApi


def push_to_huggingface(local_repo_path: str = "data/behavior_data", repo_id: str = None) -> None:
    """
    Push data to Hugging Face Hub repository.

    Args:
        local_repo_path (str): Path to local repository
        repo_id (str): Hugging Face repository ID (e.g., 'username/repo-name')
    """
    if repo_id is None:
        raise ValueError("repo_id must be specified (e.g., 'username/repo-name')")

    api = HfApi()

    # Create the repo if it doesn't exist (will not affect existing repos)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    # Upload all files in the directory
    local_path = Path(local_repo_path)
    for filepath in local_path.rglob("*"):
        if filepath.is_file():
            # Get relative path for upload
            relative_path = str(filepath.relative_to(local_path))

            # Upload the file
            api.upload_file(
                path_or_fileobj=str(filepath), path_in_repo=relative_path, repo_id=repo_id, repo_type="dataset"
            )

    print(f"Successfully pushed data from {local_repo_path} to {repo_id}")


def pull_from_huggingface(local_repo_path: str = "data/behavior_data", repo_id: str = None) -> None:
    """
    Pull data from Hugging Face Hub repository, handling large files properly.

    Args:
        local_repo_path (str): Path to local repository
        repo_id (str): Hugging Face repository ID (e.g., 'username/repo-name')
    """
    if repo_id is None:
        raise ValueError("repo_id must be specified (e.g., 'username/repo-name')")

    # Use snapshot_download as the preferred method
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_repo_path,
        repo_type="dataset",
        ignore_patterns=[".*"],
    )

    print(f"Repository {repo_id} synchronized to {local_repo_path}")


if __name__ == "__main__":
    repo_id = "LLMAccountability/behavior_data"
    pull_from_huggingface(repo_id=repo_id)
