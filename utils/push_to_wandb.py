import wandb

# from utils.utils import create_run_string
from utils import create_run_string

wandb.init(
    project="toxicity_test",
    entity="LLM_Accountability",
    name=create_run_string(),
    tags=["all_data"],
)

file_name = "perspective_data.zip"
wandb.save(file_name)
wandb.finish()
