# Dockerfile for running seq-CIFAR-10 experiments on Beaker
# Base: NVIDIA PyTorch image with CUDA 12.x + Python 3.11
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Copy the full repo
COPY . /app

# Install the fla package + experiment dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir torchvision wandb

# Default: show help
CMD ["python", "experiments/seq_cifar.py", "--help"]
