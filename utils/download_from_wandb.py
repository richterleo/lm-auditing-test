import wandb

from keys import WANDB_API_KEY

# Login to Wandb using API key
wandb.login(key=WANDB_API_KEY)

# Initialize Wandb
run = wandb.init(project="Model_Continuations", entity="richter-leo94")

# Use the Wandb API to access the artifact
api = wandb.Api()

# Replace "artifact_name:version" with your specific artifact name and version
artifact = api.artifact(
    "richter-leo94/Model_Continuations/dataset:latest", type="dataset"
)

# Download the artifact
artifact_dir = artifact.download()

print(f"Artifact downloaded to: {artifact_dir}")

# Finish the run
wandb.finish()
