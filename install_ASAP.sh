#!/bin/bash

# Define the installation folder for ASAP
PROJECT_ROOT=$(pwd)
ASAP_INSTALL_FOLDER="$PROJECT_ROOT/asap"
CONDA_ENV="$PROJECT_ROOT/monkey_env"

# Step 1: Create the directory for ASAP installation
echo "Setting up ASAP in the folder: $ASAP_INSTALL_FOLDER"
mkdir -p "$ASAP_INSTALL_FOLDER"

# Step 2: Download and extract the ASAP package
echo "Downloading and extracting ASAP..."
curl -sL "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb" -o ASAP.deb
dpkg-deb -x ASAP.deb "$ASAP_INSTALL_FOLDER"
rm ASAP.deb

# Step 3: Install missing libpugixml.so.1 dependency
echo "Installing libpugixml.so.1 dependency..."
wget http://ftp.de.debian.org/debian/pool/main/p/pugixml/libpugixml1v5_1.11.4-1_amd64.deb -O libpugixml1v5_1.11.4-1_amd64.deb
dpkg-deb -x libpugixml1v5_1.11.4-1_amd64.deb "$ASAP_INSTALL_FOLDER"
rm libpugixml1v5_1.11.4-1_amd64.deb

# Step 4: Link the ASAP binary path to Python
echo "Linking ASAP's binary path to the Python environment..."
echo "$ASAP_INSTALL_FOLDER/opt/ASAP/bin" > "$CONDA_ENV/lib/python3.8/site-packages/asap.pth"

# Step 5: Verify installation and dependencies
echo "Verifying ASAP installation and dependencies..."
ldd "$ASAP_INSTALL_FOLDER/opt/ASAP/bin/_multiresolutionimageinterface.so" | grep "not found" && {
    echo "Missing dependencies detected. Ensure all required libraries are installed locally."
    exit 1
}

# Completion message
echo "ASAP setup complete!"
echo "ASAP is installed in $ASAP_INSTALL_FOLDER and linked to your Conda environment at $CONDA_ENV."
