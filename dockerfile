FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    tzdata \
    python3 python3-pip git git-lfs \
    libgl1 \
    libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Enable Git LFS
RUN git lfs install

# Set working directory
WORKDIR /workspace

# Clone the pixel-nerf repository
RUN git clone https://github.com/etiiiR/pixel-nerf.git
WORKDIR /workspace/pixel-nerf

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision
RUN pip3 install -r requirements.txt

# Clone Hugging Face dataset using Git LFS
RUN mkdir -p pollen && \
    cd pollen && \
    git lfs clone https://huggingface.co/datasets/Etiiir/Pollen .

# Optional: expose a port or entrypoint
#CMD ["python3", "./train/train.py", "-n", "pollen", "-c", "conf/exp/pollen.conf", "-D", "pollen", "--gpu_id=0", "--resume"]