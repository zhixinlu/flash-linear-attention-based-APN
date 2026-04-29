# Seq-CIFAR-10: DeltaNet vs APN vs Transformer Benchmark

## Overview

This experiment compares three sequence models on sequential CIFAR-10 classification:

- **DeltaNet** — the delta-rule linear attention model from the `fla` library
- **APN** (Associative Plasticity Network) — a Hebbian plasticity model, mapped onto the **gated delta-rule** Triton kernel for fast GPU training
- **Transformer** — standard softmax attention baseline using `F.scaled_dot_product_attention` (auto-dispatches to FlashAttention-2 on CUDA)

The task treats each CIFAR-10 image as a sequence of 1024 RGB pixels (32×32 pixels, 3 features each). The model reads the sequence and classifies from the last time-step into one of 10 classes.

All models use the same outer architecture: `input_proj → tanh → n_layers (with residual connections) → LayerNorm → linear classifier`.

---

## Mathematical Background

### APN Recurrence

The Associative Plasticity Network maintains a plastic (Hebbian) memory matrix $M_t$ that evolves over time:

$$
\mathbf{x}_{\text{act}} = \tanh(\mathbf{x}_t)
$$

$$
\mathbf{h}_t = W \mathbf{x}_{\text{act}} + M_{t-1} \mathbf{x}_{\text{act}}
$$

$$
M_t = \lambda M_{t-1} + \eta \, \mathbf{h}_t \mathbf{x}_{\text{act}}^\top
$$

where:
- $W$ is a static (learned) weight matrix
- $M_t$ is the plastic memory matrix, updated at every time-step
- $\lambda \in (0,1)$ is a decay factor (controls forgetting)
- $\eta$ is a learning rate (controls how fast the memory adapts)

### Gated Delta Rule

The gated delta-rule kernel (from `fla`) computes:

$$
S_t = e^{g} S_{t-1} + \beta \, \mathbf{k}_t \otimes (\mathbf{v}_t - S_{t-1} \mathbf{k}_t)
$$

$$
\mathbf{o}_t = \mathbf{q}_t^\top S_t
$$

where $g$ is a scalar gate (decay in log-space), $\beta$ is a scalar step-size, and $S_t$ is the recurrent state matrix.

### Mapping APN onto the Gated Delta-Rule Kernel

We can run the APN recurrence using the gated delta-rule kernel by setting:

| APN concept | Kernel variable | Value |
|---|---|---|
| key | $\mathbf{k}$ | $\tanh(\mathbf{x}_t)$ |
| value | $\mathbf{v}$ | $W \tanh(\mathbf{x}_t)$ |
| query | $\mathbf{q}$ | $\tanh(\mathbf{x}_t)$ |
| decay | $g$ | $\log(\lambda)$ |
| step-size | $\beta$ | $\eta \cdot (1 - \lambda) / D$ |
| residual | — | $+ W \tanh(\mathbf{x}_t)$ (static feedforward path) |

This gives the update:

$$
S_t = \lambda \, S_{t-1} + \frac{\eta(1-\lambda)}{D} \left( \mathbf{k}_t \otimes (\mathbf{v}_t - S_{t-1} \mathbf{k}_t) \right)
$$

### Beta Normalization: $(1-\lambda)/D$

The raw delta-rule update $S_t = \lambda S_{t-1} + \beta (\mathbf{k} \otimes (\mathbf{v} - S\mathbf{k}))$ is unstable when $|\beta|$ is too large relative to the decay $(1-\lambda)$ and dimension $D$. With 10 stacked layers, even $\beta = 0.05$ causes NaN for $\lambda = 0.99$.

The normalization $\beta = \eta \cdot (1-\lambda)/D$ solves this by:
1. **Scaling with decay**: when $\lambda \approx 1$ (slow forgetting), the update step shrinks proportionally
2. **Scaling with dimension**: larger state matrices need smaller per-element updates
3. **Freeing $\eta$**: the optimizer can move $\eta$ at O(1) scale (e.g., $\eta = \pm 10$) without blowing up the recurrence

For example, with $\lambda = 0.999$ and $D = 100$: effective $\beta = \eta \times 10^{-5}$, so even $\eta = 10$ gives $\beta = 10^{-4}$.

**Note on the approximation.** The original APN update $M_t = \lambda M_{t-1} + \eta \, \mathbf{h}_t \mathbf{x}_{\text{act}}^\top$ expands to $M_t = (\lambda I + \eta \, \mathbf{x}_{\text{act}} \mathbf{x}_{\text{act}}^\top) M_{t-1} + \eta \, W \mathbf{x}_{\text{act}} \mathbf{x}_{\text{act}}^\top$. The delta-rule kernel's erase term produces $(\lambda I - \eta \, \mathbf{k}\mathbf{k}^\top) S_{t-1}$ instead — a sign difference on the self-outer-product term. Despite this, the kernel gives a well-behaved recurrence that can be trained end-to-end.

### Why Use the Gated Delta-Rule Kernel?

A naive implementation of APN requires materializing the $D \times D$ matrix $M_t$ at every time-step, for every layer and every batch element. For 10 layers, 1024 steps, and $D=100$, this would need ~26 GB of GPU memory for backpropagation.

The `chunk_gated_delta_rule` Triton kernel from `fla` avoids this by:
1. **Chunked computation**: processes the sequence in chunks, only materializing intermediate states at chunk boundaries
2. **Fused operations**: all matrix updates and queries happen in a single fused kernel
3. **Memory efficiency**: the 10-layer APN uses only ~2 GB of GPU memory

---

## Transformer Baseline

The transformer baseline is a standard pre-norm causal transformer. Each `TransformerLayer` contains:

```
LayerNorm → Multi-Head Self-Attention → residual → LayerNorm → FFN → residual
```

### Architecture Details

| Component | Details |
|---|---|
| Attention | `F.scaled_dot_product_attention` with `is_causal=True` — auto-dispatches to FlashAttention-2 on CUDA |
| QKV projection | Single fused `Linear(D, 3D)`, split into Q, K, V |
| Output projection | `Linear(D, D)` |
| FFN | `Linear(D, 4D)` → GELU → `Linear(4D, D)` |
| Normalization | Pre-norm (LayerNorm before attention and FFN) |
| Heads | Configurable via `--n-heads` (default: 1 for fair comparison with single-head DeltaNet/APN) |

### Parameter Count Comparison

The transformer has significantly more parameters per layer due to the FFN (4x expansion):

| Component | DeltaNet (per layer) | APN (per layer) | APN + FFN (per layer) | Transformer (per layer) |
|---|---|---|---|---|
| Q/K/V projections | $3D^2$ | $D^2$ (W only) | $D^2$ (W only) | $3D^2$ (fused QKV) |
| Output projection | — | — | — | $D^2$ |
| FFN | — | — | $8D^2$ (up + down) | $8D^2$ (up + down) |
| Beta projection | $D$ | — | — | — |
| LayerNorm | $2D$ | $2D$ | $4D$ (two norms) | $4D$ (two norms) |
| Scalars | — | 2 ($\eta$, $\lambda$) | 2 ($\eta$, $\lambda$) | — |
| **Total** | **$\approx 3D^2$** | **$\approx D^2$** | **$\approx 9D^2$** | **$\approx 12D^2$** |

At $D = 100$, 10 layers: DeltaNet ≈ 300K, APN ≈ 100K, APN+FFN ≈ 900K, Transformer ≈ 1.2M params (in attention/FFN only).

### Complexity

| Model | Time per layer | Memory |
|---|---|---|
| Transformer | $O(T^2 D + T D^2)$ | $O(T^2)$ per head (but FlashAttention reduces to $O(T)$ HBM) |
| DeltaNet / APN | $O(T C D + T D^2 / C)$ | $O(D^2)$ per head + $O(C^2)$ intra-chunk |

For $T = 1024$, $D = 100$: DeltaNet/APN have fewer FLOPs in the attention computation. The transformer's advantage is in mature FlashAttention kernel optimization.

---

## Trainable Parameters

Both $\eta$ and $\lambda$ are **trainable scalars** (one per layer):

| Parameter | Parameterization | Constraint | Gradient |
|---|---|---|---|
| $\eta$ | Direct `nn.Parameter` | Unconstrained (can be positive or negative) | Flows through $\beta = \eta(1-\lambda)/D$ |
| $\lambda$ | Logit-space: `sigmoid(lam_logit)` | Always in $(0, 1)$ | Flows through both $g$ and $\beta$ |

$\eta$ being unconstrained means the optimizer can freely learn whether the plasticity update should be Hebbian ($\eta > 0$) or anti-Hebbian ($\eta < 0$). The $(1-\lambda)/D$ normalization ensures this is safe at any scale.

Both parameters can be frozen via CLI flags (`--freeze-eta`, `--freeze-lam`).

---

## Usage

```bash
conda activate fla_apn

# Train APN (10 layers, 100 hidden, 50 epochs)
python experiments/seq_cifar.py --model apn --n-layers 10 --d-hidden 100 \
    --apn-lam 0.99 --apn-eta 1.0 --epochs 50 --batch-size 64

# Train DeltaNet
python experiments/seq_cifar.py --model deltanet --n-layers 10 --d-hidden 100 \
    --epochs 50 --batch-size 64

# Train Transformer baseline
python experiments/seq_cifar.py --model transformer --n-layers 10 --d-hidden 100 \
    --n-heads 4 --epochs 200 --warmup-epochs 20 --batch-size 64

# Compare DeltaNet and APN
python experiments/seq_cifar.py --model both --n-layers 10 --d-hidden 100 \
    --apn-lam 0.99 --apn-eta 1.0 --epochs 50 --batch-size 64

# With learning rate warmup
python experiments/seq_cifar.py --model apn --n-layers 10 --d-hidden 100 \
    --apn-lam 0.999 --apn-eta 1.0 --epochs 200 --warmup-epochs 20 --batch-size 64

# APN with FFN (adds pre-norm FFN block after each layer, like a transformer)
python experiments/seq_cifar.py --model apn --use-ffn --n-layers 10 --d-hidden 100 \
    --apn-lam 0.999 --apn-eta 1.0 --epochs 200 --warmup-epochs 20 --batch-size 64

# APN with FFN and multi-head attention
python experiments/seq_cifar.py --model apn --use-ffn --n-heads 4 --n-layers 10 --d-hidden 100 \
    --apn-lam 0.999 --apn-eta 1.0 --epochs 200 --warmup-epochs 20 --batch-size 64

# DeltaNet with FFN
python experiments/seq_cifar.py --model deltanet --use-ffn --n-layers 10 --d-hidden 100 \
    --epochs 200 --warmup-epochs 20 --batch-size 64

# Custom FFN expansion factor (2x instead of default 4x)
python experiments/seq_cifar.py --model apn --use-ffn --ffn-mult 2 --n-layers 10 --d-hidden 100 \
    --apn-lam 0.999 --apn-eta 1.0 --epochs 200 --batch-size 64

# Freeze lambda (fixed decay, only W and eta train)
python experiments/seq_cifar.py --model apn --n-layers 10 --d-hidden 100 \
    --apn-lam 0.99 --apn-eta 1.0 --freeze-lam --epochs 50 --batch-size 64
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `both` | `deltanet`, `apn`, `transformer`, or `both` (DeltaNet + APN) |
| `--d-hidden` | `100` | Hidden dimension per layer |
| `--n-layers` | `10` | Number of layers |
| `--n-heads` | `1` | Number of attention heads (transformer and APN; `d-hidden` must be divisible) |
| `--use-ffn` | off | Add a pre-norm FFN block after each APN/DeltaNet layer |
| `--ffn-mult` | `4` | FFN expansion factor (e.g. 4 means `D → 4D → D`) |
| `--apn-lam` | `0.99` | Initial $\lambda$ for APN |
| `--apn-eta` | `0.01` | Initial $\eta$ for APN (O(1) scale is fine due to normalization) |
| `--apn-activation` | `tanh` | Activation for APN keys/queries (`none`, `tanh`, `sigmoid`, `softsign`, `silu`, `gelu`) |
| `--epochs` | `50` | Training epochs |
| `--batch-size` | `64` | Batch size |
| `--lr` | `1e-3` | Learning rate (AdamW) |
| `--warmup-epochs` | `0` | Linear warmup epochs before cosine decay |
| `--freeze-lam` | off | Freeze $\lambda$ (not trainable) |
| `--freeze-eta` | off | Freeze $\eta$ (not trainable) |

---

## Preliminary Results

From an earlier 30-epoch run of the old APN (non-trainable $\lambda$, different $\eta$ parameterization):

| Model | Layers | Hidden | Best Test Acc | Peak GPU Mem | ~Time/epoch |
|---|---|---|---|---|---|
| APN (old) | 10 | 100 | 45.88% | 2083 MB | ~40s |
| DeltaNet | 4 | 100 | 57.12% (10 epochs) | — | ~5s |

*These are not directly comparable (different configs / epochs). A fair head-to-head with the current code is pending.*

---

## File Structure

```
experiments/
├── README.md          # This file
├── seq_cifar.py       # Benchmark script (DeltaNet + APN + Transformer)
└── beaker/
    └── launch.sh      # One-command Gantry launcher
```

---

## Running on Beaker (H200 GPUs)

Beaker provides H200 GPUs (143 GB VRAM, 228 KB shared memory per SM), which are required for large hidden dimensions (D > 128) that exceed the A100's shared memory limit (163 KB).

### Prerequisites

1. **Beaker CLI + Gantry** installed on HPC (in `fla_apn` conda env):
   ```bash
   pip install beaker-py beaker-gantry
   ```

2. **Beaker auth** configured:
   ```bash
   beaker account login
   # Config at ~/.beaker/config.yml: default_workspace=ai1/aihub-nd-scalempn, default_org=ai1
   ```

3. **WANDB_API_KEY** stored as a Beaker secret:
   ```bash
   beaker secret write WANDB_API_KEY <your-key> --workspace ai1/aihub-nd-scalempn
   ```

4. **CIFAR-10 dataset** uploaded to Beaker (one-time):
   ```bash
   python -c "
   from beaker import Beaker
   with Beaker.from_env() as b:
       ds = b.dataset.create('cifar10', 'data/')
       print(f'Dataset: {ds.name} (id={ds.id})')
   "
   ```
   The dataset is referenced as `zhixin-lu/cifar10` (account/name format).

5. **Git repo** pushed to GitHub (`zhixinlu/flash-linear-attention-based-APN`) — Gantry clones from the latest commit.

### Quick Launch

Uses the public NGC PyTorch image and installs deps at runtime (~15s overhead):

```bash
./experiments/beaker/launch.sh \
  --model apn --n-layers 10 --d-hidden 173 \
  --apn-lam 0.999 --apn-eta 1.0 \
  --epochs 200 --batch-size 64 \
  --wandb-project seq-cifar10-apn --wandb-name apn_L10_D173_H200
```

The launch script uses `gantry run` under the hood with:
- `--docker-image nvcr.io/nvidia/pytorch:24.12-py3` — public NGC image (torch, torchvision, CUDA pre-installed)
- `--no-python` — use the image's system Python (skip venv creation)
- `--install` — installs fla (no-deps), triton 3.3.x (pinned <3.4 to avoid Hopper bug), transformers==4.46.3, einops, wandb
- `--dataset "zhixin-lu/cifar10:/data"` — mounts CIFAR-10 dataset at `/data`

### Example Experiment Commands

```bash
# APN with large hidden dim (needs H200)
./experiments/beaker/launch.sh \
  --model apn --n-layers 10 --d-hidden 173 \
  --apn-lam 0.999 --apn-eta 1.0 \
  --epochs 200 --batch-size 64 \
  --wandb-project seq-cifar10-apn --wandb-name apn_L10_D173_H200

# DeltaNet (D=58 matches APN D=100 in param count)
./experiments/beaker/launch.sh \
  --model deltanet --n-layers 10 --d-hidden 58 \
  --epochs 200 --batch-size 64 \
  --wandb-project seq-cifar10-apn --wandb-name deltanet_L10_D58_H200

# Transformer baseline (with warmup)
./experiments/beaker/launch.sh \
  --model transformer --n-layers 10 --d-hidden 100 \
  --n-heads 1 --epochs 200 --warmup-epochs 20 --batch-size 64 \
  --wandb-project seq-cifar10-apn --wandb-name transformer_L10_D100_H200_warmup20
```

### Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `BeakerDatasetNotFound: cifar10` | Dataset name must be `account/name` | Use `zhixin-lu/cifar10` |
| `triton ... ValueError: 'BT' is not in list` | NGC image has Triton 3.0, fla needs ≥ 3.3 | Add `'triton>=3.3.0'` to `--install` |
| `ModuleNotFoundError: transformers` | NGC image doesn't include transformers | Add `transformers einops` to `--install` |
| `priority ... exceeds max priority` | Workspace max is "low" | Use `--priority low` |
| Shared memory overflow (A100) | D > 128 needs > 163 KB shared mem | Use H200 cluster (228 KB limit) |
| Chunk backward crash (Hopper) | Triton ≥ 3.4 produces wrong gradients on H200/H100 (fla #640) | Pin `triton>=3.3.0,<3.4.0` |
| `TransformGetItemToIndex` import error | transformers ≥ 4.47 requires torch ≥ 2.7 | Pin `transformers==4.46.3` |

---

## Dependencies

- `fla` (flash-linear-attention) — installed in editable mode from this repo
- PyTorch ≥ 2.0, Triton ≥ 3.3.0 and < 3.4.0 (Hopper compatibility), torchvision
- transformers, einops (required by fla layer imports)
- wandb (experiment tracking)
- Conda environment: `fla_apn`
