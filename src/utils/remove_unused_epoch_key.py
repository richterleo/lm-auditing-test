import os
import json


# Assuming remove_zero_key_and_flatten is defined like this:
def remove_zero_key_and_flatten(file_path, return_data=False):
    try:
        with open(file_path, "r") as file:
            file_content = file.read()

        if not file_content.strip():
            print(f"The file '{file_path}' is empty.")
            return

        try:
            data = json.loads(file_content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in '{file_path}': {str(e)}")
            print(f"File content (first 100 characters): {file_content[:100]}")
            return

        # Flattening logic and removing zero key
        if "0" in data:
            print("Removing unnecessary 0 key.")
            new_data = data["0"]
            if "metadata" in data:
                new_data["metadata"] = data["metadata"]

            # Save the modified data back to the file
            with open(file_path, "w") as file:
                json.dump(new_data, file, indent=4)

            print(f"Successfully processed '{file_path}'")

        else:
            print(f"Key '0' not found in {file_path}. No changes made.")
            new_data = data

        if return_data:
            return new_data

    except IOError as e:
        print(f"Error reading or writing file '{file_path}': {str(e)}")
    except Exception as e:
        print(f"Unexpected error processing '{file_path}': {str(e)}")


if __name__ == "__main__":
    # Define the folder path
    folder_path = "/root/Auditing_Test_for_LMs/model_scores"  # Replace with the actual path

    # Walk through the directory and process each JSON file
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".json"):
                json_file_path = os.path.join(root, file_name)
                print(f"Processing: {json_file_path}")
                remove_zero_key_and_flatten(json_file_path)
