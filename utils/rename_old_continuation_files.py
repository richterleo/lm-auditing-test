import os


def rename_json_files(root_dir):
    """
    Rename all JSON files in the given directory and its subdirectories to 'continuations.json'

    Args:
        root_dir (str): Root directory path to start the search from
    """
    # Walk through all directories and files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter for JSON files
        json_files = [f for f in filenames if f.endswith(".json")]

        # Rename each JSON file
        for json_file in json_files:
            old_path = os.path.join(dirpath, json_file)
            new_path = os.path.join(dirpath, "continuations.json")

            try:
                # Check if destination file already exists
                if os.path.exists(new_path):
                    print(f"Warning: {new_path} already exists, skipping {old_path}")
                    continue

                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")
            except Exception as e:
                print(f"Error renaming {old_path}: {str(e)}")


def rename_specific_json_files(root_dir):
    """
    Rename specific JSON files in the given directory and its subdirectories:
    - 'perspective_continuation_scores.json' -> 'continuation_scores.json'
    - 'perspective_scores.json' -> 'scores.json'

    Args:
        root_dir (str): Root directory path to start the search from
    """
    # Define the renaming mappings
    rename_map = {
        "perspective_continuation_scores.json": "continuation_scores.json",
        "perspective_scores.json": "scores.json",
    }

    # Walk through all directories and files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check each file
        for filename in filenames:
            if filename in rename_map:
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, rename_map[filename])

                try:
                    # Check if destination file already exists
                    if os.path.exists(new_path):
                        print(f"Warning: {new_path} already exists, skipping {old_path}")
                        continue

                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"Error renaming {old_path}: {str(e)}")


if __name__ == "__main__":
    # # Usage
    # model_outputs_path = "/root/Auditing_test_for_LMs/Auditing_test_for_LMs/Auditing_test_for_LMs/perspective/model_outputs"  # Replace with your actual path
    # rename_json_files(model_outputs_path)
    model_scores_path = "/root/Auditing_test_for_LMs/Auditing_test_for_LMs/Auditing_test_for_LMs/perspective/model_scores"  # Replace with your actual path
    rename_specific_json_files(model_scores_path)
