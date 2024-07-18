# %%
import json
import pandas as pd
import os
from pathlib import Path


# %%
# Define the directory path
directory = "model_continuations_1406"
# Iterate through the subfolders
file_paths = []
for folder in os.listdir(directory):
    folder_path = os.path.join(directory, folder)
    if os.path.isdir(folder_path):
        # Define the file path
        for file in os.listdir(folder_path):
            if file.endswith(".json"):
                file_path = os.path.join(folder_path, file)
                break

        file_paths.append(file_path)

file_paths.sort()
file_paths_iter = iter(file_paths)

# %%
fp = next(file_paths_iter)
model_name = Path(fp).stem
print(f"Current model: {model_name}")
with open(fp, "r") as file:
    data = json.load(file)

# Only check first epoch
entries = data["0"]
prompts = entries["prompts"]
continuations = entries["continuations"]

# Create a DataFrame
df = pd.DataFrame({"prompts": prompts, "continuations": continuations})
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
df_head = df.head(20)
df_head

# %%
