FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# Set timezone (optional)
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies and Python 3.8
RUN apt-get update && \
    apt-get install -y \
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

# Set up Python virtual environment and update PATH
RUN python3.8 -m venv /workspace/venv && \
    /workspace/venv/bin/python -m pip install --upgrade pip
ENV PATH="/workspace/venv/bin:$PATH"

# Install ASAP (Advanced Slide Analysis Platform)
RUN curl -sL "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1-(Nightly)/ASAP-2.1-Ubuntu2004.deb" -o ASAP.deb && \
    dpkg --install ASAP.deb || true && \
    apt-get -f install --fix-missing --fix-broken --assume-yes && \
    ldconfig && \
    echo "/opt/ASAP/bin" > /workspace/venv/lib/python3.8/site-packages/asap.pth && \
    rm ASAP.deb && \
    apt-get clean

# Install PyTorch, torchvision, and torchaudio
RUN /workspace/venv/bin/python -m pip install torch==1.10.1+cu111 \
    torchvision==0.11.2+cu111 torchaudio==0.10.1 \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Install Detectron2
RUN /workspace/venv/bin/python -m pip install detectron2 \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

# Install OpenSlide Python bindings
RUN /workspace/venv/bin/python -m pip install --no-cache-dir openslide-bin openslide-python

# Install Whole Slide Data library
RUN /workspace/venv/bin/python -m pip install \
    'git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main'

# Copy requirements.txt and install additional Python libraries
COPY requirements.txt /workspace/requirements.txt
RUN /workspace/venv/bin/python -m pip install --no-cache-dir -r /workspace/requirements.txt

# Clean up unused packages
RUN apt-get autoremove -y && apt-get clean

# Set up dynamic user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    apt-get update && \
    apt-get install -y sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Set working directory and permissions
RUN mkdir -p /workspace && \
    chown -R $USERNAME:$USERNAME /workspace/venv /workspace
WORKDIR /workspace

# Switch to the user
USER vscode
