#!/bin/bash


###NOTE: It will install the conda env in your HOME directory!!!! It will weight at least 5/6 GB of space
# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "conda could not be found. Make sure it is installed and accessible."
    exit 1
fi

# Environment name
ENV_NAME="monkey_env"
ENV_FILE="env.yml"

echo "Creating the conda environment from $ENV_FILE..."

# Check if env.yml exists
if [ ! -f "$ENV_FILE" ]; then
    echo "$ENV_FILE not found. Please provide a valid environment.yml file."
    exit 1
fi

# Create conda environment from env.yml
conda env create -f "$ENV_FILE" --name "$ENV_NAME"

if [ $? -ne 0 ]; then
    echo "Failed to create the conda environment."
    exit 1
fi

echo "Deactivating any active conda environments..."
conda deactivate

echo "Activating the new environment: $ENV_NAME..."
conda activate "$ENV_NAME"

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

echo "All installations completed successfully in the environment: $ENV_NAME!"
