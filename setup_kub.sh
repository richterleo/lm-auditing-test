#!/bin/bash

# If os is mac, need md5sha1sum, e.g. by 
# brew install md5sha1sum

# Add wandb api key 
export WANDB_LOG_LEVEL=debug
export WANDB_API_KEY= # Add your key here
export HF_TOKEN= # Add your key here

# install flash attention
pip install torch transformers datasets wandb numpy evaluate tqdm


