import wandb

from keys import WANDB_API_KEY

# Login to Wandb using API key
wandb.login(key=WANDB_API_KEY)

# Initialize Wandb
# run = wandb.init(project="toxicity_test", entity="LLM_Accountability")

# # Use the Wandb API to access the artifact
# api = wandb.Api()

api = wandb.Api()

# Path to the file you want to download
run_name = "LLM_Accountability/toxicity_test/97flfqwa"
run = api.run(run_name)

# Replace "artifact_name:version" with your specific artifact name and version
# artifact = api.artifact(
#     "richter-leo94/Model_Continuations/dataset:latest", type="dataset"
# )

# # Download the artifact
# artifact_dir = artifact.download()

# print(f"Artifact downloaded to: {artifact_dir}")

files = [file for file in run.files() if "test_outputs.zip" in file.name]
file = files[0]
file.download()

# Finish the run
wandb.finish()
