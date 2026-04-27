# Dockerfile for APN experiments on Beaker (H200)
#
# Build & push (from your LOCAL machine with Docker + Beaker CLI):
#   cd flash-linear-attention-based-APN
#   docker build --platform linux/amd64 -t apn-cifar10 .
#   beaker image create apn-cifar10 --name apn-cifar10-v2 --workspace ai1/aihub-nd-scalempn
#
# Then on HPC:
#   export BEAKER_IMAGE=zhixin-lu/apn-cifar10-v2
#   ./experiments/beaker/launch.sh --model apn ...
#
# Base: NGC PyTorch 24.12 = Python 3.12, CUDA 12.x, torchvision pre-installed
FROM nvcr.io/nvidia/pytorch:24.12-py3

ENV DEBIAN_FRONTEND=noninteractive

# Upgrade torch + triton to match HPC env (torch 2.6 in NGC is too old for fla/transformers).
# Using --index-url to get CUDA 12.8 wheels.
RUN pip install --no-cache-dir \
    'torch>=2.7' \
    'triton>=3.3.0' \
    'torchvision>=0.22' \
    --index-url https://download.pytorch.org/whl/cu128

# Install fla (no-deps to avoid re-downloading torch) + runtime deps
COPY pyproject.toml setup.py /tmp/fla/
COPY fla/ /tmp/fla/fla/
RUN pip install --no-cache-dir --no-deps -e /tmp/fla && \
    pip install --no-cache-dir transformers einops wandb

# Gantry will clone the repo to /gantry-runtime at run time,
# so we don't bake the experiments/ code into the image.
WORKDIR /gantry-runtime
