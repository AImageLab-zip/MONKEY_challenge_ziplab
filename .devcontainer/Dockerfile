FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# Set timezone (optional, can adjust or remove)
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Python 3.8 and necessary dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    software-properties-common \
    build-essential \
    libffi-dev \
    libxml2-dev \
    libjpeg-turbo8-dev \
    zlib1g-dev \
    python3.8 \
    python3-pip \
    python3.8-venv \
    libpython3.8-dev \
    libopenslide0 \
    openslide-tools && \
    add-apt-repository -y ppa:deadsnakes && \
    apt-get clean

# Set Python 3.8 as the default Python
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    python -m pip install --no-cache-dir --upgrade pip

# # Install ASAP
# RUN curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1-(Nightly)/ASAP-2.1-Ubuntu2004.deb" && \
#     dpkg --install ASAP-2.1-Ubuntu2004.deb || true && \
#     apt-get -f install --fix-missing --fix-broken --assume-yes && \
#     ldconfig -v && \
#     echo "/opt/ASAP/bin" > /usr/local/lib/python3.8/site-packages/asap.pth && \
#     rm ASAP-2.1-Ubuntu2004.deb && \
#     apt-get clean

# Install PyTorch, torchvision, torchaudio, and detectron2
RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html && \
    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

# Install Python OpenSlide bindings
RUN pip install --no-cache-dir openslide-python

# Copy requirements.txt from the build context
COPY ../requirements.txt /workspace/requirements.txt

# Clean up
RUN apt-get autoremove -y && apt-get clean

# Add the vscode user for non-root access
RUN groupadd -r vscode && useradd -m --no-log-init -r -g vscode vscode

# Set the working directory
WORKDIR /workspace

# Change ownership of the workspace folder
RUN chown -R vscode:vscode /workspace
