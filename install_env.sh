#!/bin/bash

# Constants
ENV_NAME="cellvit_env"
ENV_PREFIX="$(pwd)/$ENV_NAME"
ENV_FILE="environment_verbose.yaml"
REQ_FILE="requirements.txt"

# Maximum number of attempts for creating the conda environment
max_attempts=3
attempt=1

# Step 1: Create the conda environment automatically with auto-confirmation and retry loop
echo "Creating conda environment at $ENV_PREFIX"

while [ $attempt -le $max_attempts ]; do
    echo "Attempt $attempt of $max_attempts..."
    if yes | conda env create --prefix "$ENV_PREFIX" --file "$ENV_FILE"; then
        echo "Conda environment created successfully."
        break
    else
        echo "Attempt $attempt failed. Retrying in 5 seconds..."
        attempt=$((attempt+1))
        sleep 5
    fi
done

if [ $attempt -gt $max_attempts ]; then
    echo "Failed to create the conda environment after $max_attempts attempts. Exiting."
    exit 1
fi

# Step 2: Initialize conda for the current shell session
echo "Initializing conda for the current shell session"
eval "$(conda shell.bash hook)"

# Step 3: Activate the environment
echo "Activating conda environment at $ENV_PREFIX"
conda activate "$ENV_PREFIX"

# Step 4: Install additional Python requirements if available
if [ -f "$REQ_FILE" ]; then
    echo "Installing requirements from $REQ_FILE"
    pip install -r "$REQ_FILE"
else
    echo "Requirements file $REQ_FILE not found. Skipping."
fi

# Step 5: Install specific versions of PyTorch packages without using cache
echo "Installing specific versions of PyTorch packages"
pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

echo "Environment setup complete!"
