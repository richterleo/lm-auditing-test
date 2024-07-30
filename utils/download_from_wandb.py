import wandb

# Login to Wandb using API key
wandb.login(key="1c84a4abed1d390fbe37478c7cb82a84e4650881")

# Initialize Wandb
run = wandb.init(project="Model_Continuations", entity="richter-leo94")

# Use the Wandb API to access the artifact
api = wandb.Api()

# Replace "artifact_name:version" with your specific artifact name and version
artifact = api.artifact('richter-leo94/Model_Continuations/dataset:latest', type='dataset')

# Download the artifact
artifact_dir = artifact.download()

print(f"Artifact downloaded to: {artifact_dir}")

# Finish the run
wandb.finish()
