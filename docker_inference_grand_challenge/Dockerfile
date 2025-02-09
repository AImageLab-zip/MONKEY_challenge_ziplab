# Use CUDA 12.1.0 with cuDNN 8 for Ubuntu 22.04
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set timezone
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install required system packages in one RUN step
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    software-properties-common \
    openslide-tools libopenslide0 \
    build-essential \
    libffi-dev \
    libxml2-dev \
    libjpeg-turbo8-dev \
    zlib1g-dev \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda into /opt/conda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -a -y

# Add Conda to PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Set working directory
WORKDIR /opt/app

# Copy environment files
COPY environment_verbose.yaml requirements.txt /opt/app/

# Upgrade Python in base (if needed) and initialize Conda for bash
RUN conda install -n base python=3.10.14 -y && conda init bash

# Create the Conda environment at /venv
RUN conda env create -p /venv -f /opt/app/environment_verbose.yaml -y

# Use the new environment for subsequent RUN commands
SHELL ["conda", "run", "-p", "/venv", "/bin/bash", "-c"]

# Install additional pip packages
RUN pip install --no-cache-dir -r /opt/app/requirements.txt && \
    pip install git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main && \
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install ASAP for Whole Slide Imaging
RUN wget --quiet 'https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb' && \
    dpkg --install ASAP-2.2-Ubuntu2204.deb || true && \
    apt-get update && apt-get install -f -y && \
    ldconfig && \
    echo "/opt/ASAP/bin" > /venv/lib/python3.10/site-packages/asap.pth && \
    rm ASAP-2.2-Ubuntu2204.deb && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install OpenSlide Python bindings
RUN pip install --no-cache-dir openslide-python

# Create non-root user and adjust permissions
RUN groupadd -r user && useradd -m -r -g user user && \
    chown -R user:user /venv /opt/app

USER user

# Copy project files with correct ownership
COPY --chown=user:user cellvit /opt/app/cellvit
COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user inference.py /opt/app/inference.py

# Ensure Python output is unbuffered
ENV PYTHONUNBUFFERED=1

# Use inference.py as the container entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-p", "/venv", "python", "inference.py"]
