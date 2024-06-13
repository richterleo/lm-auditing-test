import json

# Load the JSON file
file_path = "model_outputs/Meta-Llama-3-8B-Instruct_seed1000/Meta-Llama-3-8B-Instruct_continuations_seed1000.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Display the first 20 entries
first_20_entries = data[:20]

# Print the first 20 entries
for entry in first_20_entries:
    print(entry)
