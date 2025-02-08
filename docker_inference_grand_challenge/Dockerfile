# Use CUDA 12.0.1 with cuDNN 8 for Ubuntu 20.04
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu20.04

# Set timezone
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install required system packages
RUN apt-get update && apt-get install -y \
    git \
    curl \
    software-properties-common \
    openslide-tools libopenslide0 \
    build-essential libffi-dev libxml2-dev libjpeg-turbo8-dev zlib1g-dev \
 && apt-get clean

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

# Copy Conda environment and requirements
COPY environment_verbose.yaml /opt/app/environment_verbose.yaml
COPY requirements.txt /opt/app/requirements.txt

# Upgrade Python and initialize Conda for bash
RUN conda install -n base python=3.10.14 -y && conda init bash

# Create the Conda environment at /venv (mimicking the old venv setup)
RUN conda env create -p /venv -f /opt/app/environment_verbose.yaml -y

# Activate Conda environment before using Python
SHELL ["conda", "run", "-p", "/venv", "/bin/bash", "-c"]

# Install additional pip packages inside Conda
RUN pip install --no-cache-dir -r /opt/app/requirements.txt && \
    pip install git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main && \
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install ASAP for Whole Slide Imaging
RUN wget --quiet https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1-(Nightly)/ASAP-2.1-Ubuntu2004.deb && \
    dpkg --install ASAP-2.1-Ubuntu2004.deb || true && \
    apt-get -f install --fix-missing --fix-broken --assume-yes && \
    ldconfig -v && \
    apt-get clean && \
    echo "/opt/ASAP/bin" > /venv/lib/python3.10/site-packages/asap.pth && \
    rm ASAP-2.1-Ubuntu2004.deb

# Install OpenSlide Python bindings
RUN pip install --no-cache-dir openslide-python

# Create a non-root user and update permissions
RUN groupadd -r user && useradd -m --no-log-init -r -g user user && \
    chown -R user:user /venv /opt/app

USER user

# Copy project files
COPY --chown=user:user cellvit /opt/app/cellvit
COPY --chown=user:user model_and_info /opt/app/model_and_info
COPY --chown=user:user backbone /opt/app/backbone
COPY --chown=user:user resources /opt/app/resources

# Ensure Python output is unbuffered
ENV PYTHONUNBUFFERED=1

# Expose necessary directories for input/output
VOLUME ["/input", "/output", "/opt/ml/model"]

# Copy external model from /opt/ml/model to /opt/app/model_and_info/checkpoints at runtime
RUN mkdir -p /opt/app/model_and_info/checkpoints && \
    cp /opt/ml/model/*.pth /opt/app/model_and_info/checkpoints/ 2>/dev/null || true

# Activate Conda environment and run the inference script
ENTRYPOINT ["conda", "run", "-p", "/venv", "python", "/opt/app/cellvit/training/evaluate/inference_cellvit_wsi_single.py"]
