import os
import json


# Assuming remove_zero_key_and_flatten is defined like this:
def remove_zero_key_and_flatten(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    # Flattening logic and removing zero key
    if "0" in data:
        del data["0"]

    # Save the modified data back to the file
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


# Define the folder path
folder_path = "/root/Auditing_Test_for_LMs/model_scores"  # Replace with the actual path

# Walk through the directory and process each JSON file
for root, dirs, files in os.walk(folder_path):
    for file_name in files:
        if file_name.endswith(".json"):
            json_file_path = os.path.join(root, file_name)
            print(f"Processing: {json_file_path}")
            remove_zero_key_and_flatten(json_file_path)
