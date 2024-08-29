#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

folder_path = 'tasks'
output_folder = 'extracted'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Dictionary to map categories to their respective instances
category_to_instances = {}

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):  # Check if the file is a JSON file
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)

            categories = data.get("Categories", None)
            if categories is None or len(categories) == 0:
                print(f"Skipping file {filename} due to missing categories.")
                continue

            category = categories[0].replace(" ", "_").lower()

            instances = data.get("Instances", [])

            if category not in category_to_instances:
                category_to_instances[category] = []
            category_to_instances[category].extend(instances)

# Save the instances to separate JSON files for each category
for category, instances in category_to_instances.items():
    output_file_path = os.path.join(output_folder, f"{category}.json")
    with open(output_file_path, 'w') as out_file:
        json.dump({"Instances": instances}, out_file, indent=4)

    print(f"Saved {len(instances)} instances to {output_file_path}")