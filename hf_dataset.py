from huggingface_hub import Repository
import os
import subprocess
import shutil

# Define local path to dataset
local_repo_path = "data/behavior_data"  # local folder name for the cloned repo

# Instantiate repository object
repo = Repository(local_dir=local_repo_path)  # update with your details

# Check if there are any changes (new, modified, or deleted files)
result = subprocess.run(["git", "status", "--porcelain"], cwd=local_repo_path, capture_output=True, text=True)
if result.stdout.strip():
    print("Changes detected. Pushing to the Hub...")
    repo.push_to_hub(commit_message="Update experiment data with new changes")
else:
    print("No changes detected. Nothing to push.")
