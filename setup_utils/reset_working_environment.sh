#!/bin/bash

# Deactivate the Conda environment
conda init
conda deactivate

# Remove the Conda environment activation command from bashrc
sed -i '/conda activate/d' ~/.bashrc

# Remove the Conda environment
conda env remove --name distancevenv

# Uninitialize Conda for Bash shell
~/miniconda3/bin/conda init --reverse bash

# Remove the Conda installation
rm -rf ~/miniconda3

# Remove path to miniconda from path variable
NEW_PATH=$(echo $ORIGINAL_PATH | sed 's|/root/miniconda3/condabin:||g' | sed 's|/opt/conda/bin:||g')

# Export the new PATH
export PATH=$NEW_PATH

# Unset the wandb api key and log level
unset WANDB_API_KEY
unset WANDB_LOG_LEVEL

# Assuming ~/.bashrc was sourced previously and might be restored to previous state if needed
# source ~/.bashrc  # Uncomment this line if you need to refresh the environment immediately
