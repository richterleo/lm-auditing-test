#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Variables
MINICONDA_INSTALL_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="auditenv"
REQUIREMENTS_FILE="requirements.txt"

# Function to print messages
echo_msg() {
    echo -e "\n[INFO] $1\n"
}

# 1. Install Miniconda
if [ ! -d "$MINICONDA_INSTALL_DIR" ]; then
    echo_msg "Installing Miniconda..."

    # Create a temporary directory for the installer
    TMP_DIR=$(mktemp -d)
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$TMP_DIR/miniconda.sh"

    # Install Miniconda
    bash "$TMP_DIR/miniconda.sh" -b -p "$MINICONDA_INSTALL_DIR"

    # Remove the installer script
    rm -rf "$TMP_DIR"

    # Export PATH
    export PATH="$MINICONDA_INSTALL_DIR/bin:$PATH"

    # Initialize conda for bash
    source "$MINICONDA_INSTALL_DIR/etc/profile.d/conda.sh"

    # Update Conda
    conda update -y conda
else
    echo_msg "Miniconda already installed at $MINICONDA_INSTALL_DIR"
    export PATH="$MINICONDA_INSTALL_DIR/bin:$PATH"
    source "$MINICONDA_INSTALL_DIR/etc/profile.d/conda.sh"
fi

# 2. Create Conda Environment
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo_msg "Conda environment '$CONDA_ENV_NAME' already exists."
else
    echo_msg "Creating conda environment '$CONDA_ENV_NAME'..."
    conda create -y -n "$CONDA_ENV_NAME" python=3.11
fi

# 3. Activate Conda Environment
echo_msg "Activating conda environment '$CONDA_ENV_NAME'..."
conda activate "$CONDA_ENV_NAME"

# 4. Install Dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo_msg "Installing Python dependencies from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo_msg "Requirements file '$REQUIREMENTS_FILE' not found. Exiting."
    exit 1
fi

# 5. Install Additional Packages
echo_msg "Installing additional packages..."
pip install packaging ninja
pip install flash-attn --no-build-isolation

# 6. Setup Environment Variables
echo_msg "Setting up environment variables..."

# Check if WANDB_API_KEY is set, if not, prompt the user
if [ -z "$WANDB_API_KEY" ]; then
    read -sp "Enter your WANDB_API_KEY: " WANDB_API_KEY_INPUT
    export WANDB_API_KEY="$WANDB_API_KEY_INPUT"
    echo
fi

# Check if HF_TOKEN is set, if not, prompt the user
if [ -z "$HF_TOKEN" ]; then
    read -sp "Enter your Hugging Face token (HF_TOKEN): " HF_TOKEN_INPUT
    export HF_TOKEN="$HF_TOKEN_INPUT"
    echo
fi

export WANDB_LOG_LEVEL=debug

# 7. Summary
echo_msg "Setup completed successfully!"
echo "Conda version: $(conda --version)"
echo "Conda environment: $CONDA_ENV_NAME"
echo "Python version: $(python --version)"
echo "Installed packages:"
pip list

# End of script
