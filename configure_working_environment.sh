#!/bin/bash

# If os is mac, need md5sha1sum, e.g. by 
# brew install md5sha1sum

# Create directory for Miniconda and download the installer
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

# Install Miniconda
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# Remove the installer script
rm -rf ~/miniconda3/miniconda.sh

# Initialize Conda for Bash shell
~/miniconda3/bin/conda init bash

# Add wandb api key 
export WANDB_API_KEY=1c84a4abed1d390fbe37478c7cb82a84e4650881 # Add your key here
export WANDB_LOG_LEVEL=debug

# Source the bashrc to refresh the environment
source ~/.bashrc

# Create a Conda environment
echo "Creating conda environment"
conda env create -f environment.yml

# Add conda activation command to bashrc
echo "conda activate '/opt/conda/envs/distancevenv'" >> ~/.bashrc

# Activate the Conda environment
conda activate "/opt/conda/envs/distancevenv"

# install flash attention
pip install ninja
pip install flash-attn --no-build-isolation


