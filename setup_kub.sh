#!/bin/bash

# If os is mac, need md5sha1sum, e.g. by 
# brew install md5sha1sum

# Add wandb api key 
export WANDB_API_KEY=1c84a4abed1d390fbe37478c7cb82a84e4650881 # Add your key here
export WANDB_LOG_LEVEL=debug

# install flash attention
pip install torch transformers datasets wandb numpy evaluate tqdm


