#!/usr/bin/env bash
# Launch an APN experiment on Beaker via Gantry.
#
# Usage (from repo root, with fla_apn conda env active):
#
#   ./experiments/beaker/launch.sh [seq_cifar.py args...]
#
# Examples:
#   ./experiments/beaker/launch.sh --model apn --n-layers 10 --d-hidden 173 --epochs 200
#   ./experiments/beaker/launch.sh --model deltanet --n-layers 10 --d-hidden 58 --epochs 200

set -euo pipefail

WORKSPACE="ai1/aihub-nd-scalempn"
CLUSTER="ai1/octo.hub-gcp-h200"
IMAGE="nvcr.io/nvidia/pytorch:24.12-py3"

echo "=== Launching on Beaker (${CLUSTER}) ==="
echo "Image: ${IMAGE}"
echo "Make sure you have committed and pushed your changes!"
echo ""

gantry run \
  --yes \
  --workspace "$WORKSPACE" \
  --cluster "$CLUSTER" \
  --gpus 1 \
  --priority low \
  --docker-image "$IMAGE" \
  --no-python \
  --install "pip install --no-deps -e . && pip install 'triton>=3.3.0' 'transformers==4.46.3' einops wandb" \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --dataset "zhixin-lu/cifar10:/data" \
  --show-logs \
  -- python experiments/seq_cifar.py --data-dir /data "$@"
