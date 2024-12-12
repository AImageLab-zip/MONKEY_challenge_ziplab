#!/bin/bash

Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "conda could not be found. Make sure it is installed and accessible."
    exit 1
fi

# Environment name and paths
ENV_NAME="monkey_env"
ENV_FILE="env.yml"
PROJECT_ROOT="$(dirname "$0")"
ENV_PATH="$PROJECT_ROOT/$ENV_NAME"

echo "Creating the conda environment in the project root directory: $ENV_PATH..."

# Check if env.yml exists
if [ ! -f "$ENV_FILE" ]; then
    echo "$ENV_FILE not found. Please provide a valid environment.yml file."
    exit 1
fi

# Create conda environment from env.yml in the specified directory
conda env create -f "$ENV_FILE" --prefix "$ENV_PATH"

if [ $? -ne 0 ]; then
    echo "Failed to create the conda environment."
    exit 1
fi

echo "Deactivating any active conda environments..."
conda deactivate

echo "Activating the new environment at: $ENV_PATH..."
conda activate "$ENV_PATH"

# Ensure pip is installed in the new environment
if ! command -v pip &> /dev/null; then
    echo "pip is not available in the environment. Installing pip..."
    conda install -y pip
fi

echo "Upgrading pip to the latest version..."
pip install --upgrade pip

echo "Starting pip installations..."

# List of pip packages to install
pip install --no-cache-dir torch==1.10.1+cu111 \
    torchvision==0.11.2+cu111 \
    torchaudio==0.10.1 \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip install --no-cache-dir detectron2 \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

pip install --no-cache-dir openslide-bin openslide-python

pip install --no-cache-dir git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main

# Check if requirements.txt exists and install if present
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install --no-cache-dir -r requirements.txt
else
    echo "requirements.txt not found. Skipping."
fi

echo "All installations completed successfully in the environment at: $ENV_PATH!"
