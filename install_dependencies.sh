#!/bin/bash

# Check if pip is installed
if ! command -v pip &> /dev/null
then
    echo "pip could not be found. Make sure it is installed and accessible."
    exit 1
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

echo "All pip installations completed successfully!"
