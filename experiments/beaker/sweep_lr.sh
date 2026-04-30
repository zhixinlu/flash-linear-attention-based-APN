#!/bin/bash
# Learning Rate Sweep: ~103K param-matched group (no FFN)
# Models: APN D=100, DeltaNet D=58, Transformer D=29
# LR values: 1e-4, 3e-4, 1e-3, 3e-3, 1e-2
# W&B project: seq-cifar10-apn-lr-sweep
#
# USAGE:
#   bash sweep_lr.sh          # Submit next batch (up to MAX_CONCURRENT total on cluster)
#   bash sweep_lr.sh --all    # Submit ALL remaining jobs (use only if cluster queues properly)
#
# The script tracks which jobs have been submitted in a state file.
# Re-run it after a batch finishes to submit the next batch.

set -e

WORKSPACE="ai1/aihub-nd-scalempn"
CLUSTER="ai1/octo.hub-gcp-h200"
IMAGE="nvcr.io/nvidia/pytorch:24.12-py3"
INSTALL="pip install --no-deps -e . && pip install 'triton>=3.3.0,<3.4.0' 'transformers==4.46.3' einops wandb"
DATASET="zhixin-lu/cifar10:/data"
WANDB_PROJECT="seq-cifar10-apn-lr-sweep"

MAX_CONCURRENT=4  # Max jobs that can run simultaneously on the cluster

COMMON="--data-dir /data --n-layers 10 --epochs 200 --batch-size 64 --warmup-epochs 20 --wandb-project ${WANDB_PROJECT}"

STATE_FILE="$(dirname "$0")/.sweep_lr_state"

# All 15 jobs defined as: "model d_hidden lr wandb_name"
JOBS=(
    "apn 100 1e-4 apn_D100_lr1e-4"
    "apn 100 3e-4 apn_D100_lr3e-4"
    "apn 100 1e-3 apn_D100_lr1e-3"
    "apn 100 3e-3 apn_D100_lr3e-3"
    "apn 100 1e-2 apn_D100_lr1e-2"
    "deltanet 58 1e-4 deltanet_D58_lr1e-4"
    "deltanet 58 3e-4 deltanet_D58_lr3e-4"
    "deltanet 58 1e-3 deltanet_D58_lr1e-3"
    "deltanet 58 3e-3 deltanet_D58_lr3e-3"
    "deltanet 58 1e-2 deltanet_D58_lr1e-2"
    "transformer 29 1e-4 transformer_D29_lr1e-4"
    "transformer 29 3e-4 transformer_D29_lr3e-4"
    "transformer 29 1e-3 transformer_D29_lr1e-3"
    "transformer 29 3e-3 transformer_D29_lr3e-3"
    "transformer 29 1e-2 transformer_D29_lr1e-2"
)

submit_job() {
    local model="$1" d_hidden="$2" lr="$3" name="$4"
    echo "  -> Submitting: ${name}"
    conda run -n fla_apn gantry run --yes \
        --workspace "${WORKSPACE}" --cluster "${CLUSTER}" --gpus 1 --priority low \
        --docker-image "${IMAGE}" --no-python \
        --install "${INSTALL}" \
        --env-secret WANDB_API_KEY=WANDB_API_KEY \
        --dataset "${DATASET}" \
        -- python experiments/seq_cifar.py ${COMMON} \
            --model "${model}" --d-hidden "${d_hidden}" --lr "${lr}" \
            --wandb-name "${name}"
    echo "${name}" >> "${STATE_FILE}"
}

# Initialize state file if missing
touch "${STATE_FILE}"

# Count already-submitted jobs
submitted=$(wc -l < "${STATE_FILE}" | tr -d ' ')
total=${#JOBS[@]}

if [[ "${submitted}" -ge "${total}" ]]; then
    echo "All ${total} jobs have already been submitted. Nothing to do."
    echo "Delete ${STATE_FILE} to reset."
    exit 0
fi

# Determine how many to submit this batch
if [[ "$1" == "--all" ]]; then
    batch_size=$(( total - submitted ))
    echo "=== Submitting ALL remaining ${batch_size} jobs ==="
else
    batch_size="${MAX_CONCURRENT}"
    echo "=== Submitting up to ${batch_size} jobs (batch mode) ==="
    echo "    (Already submitted: ${submitted}/${total})"
    echo "    Re-run this script after the batch finishes for the next batch."
    echo ""
fi

count=0
for i in $(seq "${submitted}" $(( total - 1 ))); do
    if [[ "${count}" -ge "${batch_size}" ]]; then
        break
    fi
    read -r model d_hidden lr name <<< "${JOBS[$i]}"
    submit_job "${model}" "${d_hidden}" "${lr}" "${name}"
    count=$(( count + 1 ))
done

new_submitted=$(wc -l < "${STATE_FILE}" | tr -d ' ')
echo ""
echo "=== Submitted ${count} jobs this batch (${new_submitted}/${total} total) ==="
if [[ "${new_submitted}" -lt "${total}" ]]; then
    echo "Run again after this batch completes to submit the next $(( total - new_submitted )) jobs."
fi
