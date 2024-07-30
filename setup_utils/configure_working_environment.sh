#!/bin/bash

# Create directory for Miniconda and download the installer
# mkdir -p ~/miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

# # Install Miniconda
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# # Remove the installer script
# rm -rf ~/miniconda3/miniconda.sh

# # Initialize Conda for Bash shell
# ~/miniconda3/bin/conda init bash

# # Export the new Conda path explicitly
# export PATH=~/miniconda3/bin:$PATH

# # Check whether the right version is now being used
# echo "Conda version used: $(conda info --base)"

# Add wandb api key 
# export WANDB_API_KEY=1c84a4abed1d390fbe37478c7cb82a84e4650881 # Add your key here
# export WANDB_LOG_LEVEL=debug

# Create a Conda environment
echo "Creating conda environment"
conda env create -f setup_utils/environment.yml

# Add conda activation command to bashrc
echo "conda activate '/root/miniconda3/envs/auditenv'" >> ~/.bashrc

# Activate the Conda environment
conda activate "/opt/conda/envs/auditenv"

# Try installing flash attention
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation


