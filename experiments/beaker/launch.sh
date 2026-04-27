#!/usr/bin/env bash
# Launch an APN experiment on Beaker.
#
# Usage (from repo root, with fla_apn conda env active):
#
#   ./experiments/beaker/launch.sh [seq_cifar.py args...]
#
# Examples:
#   ./experiments/beaker/launch.sh --model apn --n-layers 10 --d-hidden 173 --epochs 200
#   ./experiments/beaker/launch.sh --model deltanet --n-layers 10 --d-hidden 58 --epochs 200
#
# To use a pre-built Beaker image (zero install overhead), set:
#   export BEAKER_IMAGE=ai1/aihub-nd-scalempn/apn-cifar10
#   ./experiments/beaker/launch.sh ...
#
# To rebuild the image (run on a machine with Docker + Beaker CLI):
#   docker build --platform linux/amd64 -t apn-cifar10 .
#   beaker image create apn-cifar10 --name apn-cifar10 --workspace ai1/aihub-nd-scalempn

set -euo pipefail

WORKSPACE="ai1/aihub-nd-scalempn"
CLUSTER="ai1/octo.hub-gcp-h200"

# Default to NGC image; override with BEAKER_IMAGE env var for pre-built image.
if [[ -n "${BEAKER_IMAGE:-}" ]]; then
  IMAGE_FLAG="--beaker-image $BEAKER_IMAGE"
else
  IMAGE_FLAG="--docker-image nvcr.io/nvidia/pytorch:24.12-py3"
fi

echo "=== Launching on Beaker (${CLUSTER}) ==="
echo "Image: ${BEAKER_IMAGE:-nvcr.io/nvidia/pytorch:24.12-py3 (NGC default)}"
echo "Make sure you have committed and pushed your changes!"
echo ""

gantry run \
  --yes \
  --workspace "$WORKSPACE" \
  --cluster "$CLUSTER" \
  --gpus 1 \
  --priority low \
  $IMAGE_FLAG \
  --no-python \
  --install "pip install --no-deps -e . && pip install 'triton>=3.3.0' 'transformers==4.46.3' einops wandb" \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --dataset "zhixin-lu/cifar10:/data" \
  --show-logs \
  -- python experiments/seq_cifar.py --data-dir /data "$@"
