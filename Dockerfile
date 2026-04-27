# Dockerfile for APN experiments on Beaker (H200)
#
# Build & push (from your LOCAL machine with Docker + Beaker CLI):
#   cd flash-linear-attention-based-APN
#   docker build -t apn-cifar10 .
#   beaker image create apn-cifar10 --name apn-cifar10 --workspace ai1/aihub-nd-scalempn
#
# Then on HPC, use:  gantry run --beaker-image ai1/aihub-nd-scalempn/apn-cifar10 ...
#
# Base: NGC PyTorch 24.12 = Python 3.12, CUDA 12.x, torch 2.5, torchvision, triton
FROM nvcr.io/nvidia/pytorch:24.12-py3

ENV DEBIAN_FRONTEND=noninteractive

# Install only the extra deps not in the base image.
# torch/torchvision/triton are already in the NGC image — don't reinstall them.
# We install fla's deps (transformers, einops) + wandb.
COPY pyproject.toml setup.py /tmp/fla/
COPY fla/ /tmp/fla/fla/
RUN pip install --no-cache-dir --no-deps -e /tmp/fla && \
    pip install --no-cache-dir transformers einops wandb

# Gantry will clone the repo to /gantry-runtime at run time,
# so we don't bake the experiments/ code into the image.
WORKDIR /gantry-runtime
