#!/bin/bash

# Constants
ENV_NAME="cellvit_env"
ENV_FILE="environment_verbose.yaml"
REQ_FILE="requirements.txt"

# Step 1: Set CONDA_ENVS_PATH to create the environment locally (in the current working directory)
export CONDA_ENVS_PATH="$(pwd)"

# Step 2: Create the conda environment auto-confirming prompts. Suppress broken pipe errors.
echo "Creating conda environment from $ENV_FILE in $(pwd)/$ENV_NAME"
yes 2>/dev/null | conda env create -f "$ENV_FILE"

# Step 3: Initialize conda for the current shell session
echo "Initializing conda shell integration..."
eval "$(conda shell.bash hook)"

# Step 4: Activate the environment
echo "Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Step 5: Install pip packages from requirements.txt if the file exists
if [ -f "$REQ_FILE" ]; then
    echo "Installing pip packages from $REQ_FILE"
    pip install -r "$REQ_FILE"
else
    echo "No $REQ_FILE found. Skipping pip requirements."
fi

# Step 6: Install specific PyTorch packages without pip cache
echo "Installing specific PyTorch packages (CUDA 12.1)"
pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

echo "✅ Environment setup complete at: $(pwd)/$ENV_NAME"
