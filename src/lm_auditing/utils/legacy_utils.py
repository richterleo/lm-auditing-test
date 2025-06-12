import os
import json


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


def remove_zero_key_and_flatten(file_path, return_data=False, save_file=True):
    """
    Remove the "0" key from JSON data and flatten the structure.
    
    Args:
        file_path (str): Path to the JSON file
        return_data (bool): Whether to return the processed data
        save_file (bool): Whether to save changes back to file
        
    Returns:
        dict: Processed data if return_data is True, None otherwise
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        # Check if flattening is needed
        if "0" in data:
            print(f"Found legacy format in {file_path}, flattening data...")
            new_data = data["0"]
            if "metadata" in data:
                new_data["metadata"] = data["metadata"]

            # Save the modified data back to the file if requested
            if save_file:
                with open(file_path, "w") as file:
                    json.dump(new_data, file, indent=4)
                print(f"Successfully flattened and saved: {file_path}")

            return new_data if return_data else None
        else:
            return data if return_data else None

    except IOError as e:
        print(f"Error reading or writing file '{file_path}': {str(e)}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{file_path}': {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error processing '{file_path}': {str(e)}")
        raise


def rename_continuation_files(metric_dir):
    """
    Rename JSON files containing 'continuations' in their name to 'continuations.json',
    except for files matching pattern 'continuations_<number>.json'
    
    Args:
        metric_dir (str): Base directory path (should be the <metric> folder)
    """
    model_outputs_dir = os.path.join(metric_dir, "model_outputs")
    if not os.path.exists(model_outputs_dir):
        print(f"Directory not found: {model_outputs_dir}")
        return

    # Walk through all directories under model_outputs
    for dirpath, dirnames, filenames in os.walk(model_outputs_dir):
        # Filter for JSON files containing 'continuations'
        continuation_files = [f for f in filenames if f.endswith('.json') and 'continuations' in f.lower()]
        
        for file_name in continuation_files:
            # Skip files matching pattern 'continuations_<number>.json'
            if file_name.lower().replace('.json', '').replace('continuations_', '').isdigit():
                continue
                
            old_path = os.path.join(dirpath, file_name)
            new_path = os.path.join(dirpath, "continuations.json")
            
            try:
                # Check if destination file already exists
                if os.path.exists(new_path) and old_path != new_path:
                    print(f"Warning: {new_path} already exists, skipping {old_path}")
                    continue

                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
            except Exception as e:
                print(f"Error renaming {old_path}: {str(e)}")


def rename_perspective_continuation_scores(root_dir, old_name="perspective_continuation_scores.json", new_name="continuation_scores.json"):
    """
    Rename all 'perspective_continuation_scores.json' files to 'continuation_scores.json'
    in the given directory and its subdirectories.

    Args:
        root_dir (str): Root directory path to start the search from
    """

    # Walk through all directories and files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if old_name in filenames:
            old_path = os.path.join(dirpath, old_name)
            new_path = os.path.join(dirpath, new_name)

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
    # Define the folder path
    folder_path = "/workspace/lm-auditing-test/data/behavior_data/perspective/model_scores"  # Replace with the actual path

    #Walk through the directory and process each JSON file
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".json"):
                json_file_path = os.path.join(root, file_name)
                print(f"Processing: {json_file_path}")
                #remove_zero_key_and_flatten(json_file_path)

    model_scores_path = "/root/Auditing_test_for_LMs/Auditing_test_for_LMs/Auditing_test_for_LMs/perspective/model_scores"  # Replace with your actual path
    #rename_specific_json_files(model_scores_path)

    # # Add this to rename continuation files
    # metric_path = "/root/Auditing_Test_for_LMs/toxicity"  # Replace with your actual metric path
    #rename_continuation_files(folder_path)

    # # Example usage of the new function
    # perspective_scores_path = "/root/Auditing_Test_for_LMs/perspective/model_scores"  # Replace with actual path
    
    rename_perspective_continuation_scores(folder_path, old_name="perspective_scores.json")
