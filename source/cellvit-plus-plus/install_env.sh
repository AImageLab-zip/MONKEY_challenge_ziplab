#!/bin/bash

# Exit immediately if a command fails
set -e

# Step 1: Create the environment from the YAML file in the current directory
echo "Creating conda environment 'cellvit_env' from environment_verbose.yaml..."
conda env create --prefix /work/grana_urologia/MONKEY_challenge/source/sota_architectures/CellViT-plus-plus/cellvit_env -f environment_verbose.yaml || { echo "Failed to create the environment. Exiting."; exit 1; }

# Step 2: Activate the environment
echo "Activating the conda environment 'cellvit_env'..."
#source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure conda is properly initialized
conda activate /work/grana_urologia/MONKEY_challenge/source/sota_architectures/CellViT-plus-plus/cellvit_env || { echo "Failed to activate the environment. Exiting."; exit 1; }

# Step 3: Install pip packages from requirements.txt
echo "Installing pip packages from requirements.txt..."
pip install -r requirements.txt || { echo "Failed to install pip packages. Exiting."; exit 1; }

#python -m cupyx.tools.install_library --cuda 12.1 --library cutensor

# Step 4: Install PyTorch packages with pip
echo "Installing PyTorch (torch, torchvision, torchaudio) with CUDA 12.1 support..."
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121 || { echo "Failed to install PyTorch packages. Exiting."; exit 1; }

echo "Environment setup complete!"
