# Seq-CIFAR-10: DeltaNet vs APN Benchmark

## Overview

This experiment compares two recurrent sequence models on sequential CIFAR-10 classification:

- **DeltaNet** — the delta-rule linear attention model from the `fla` library
- **APN** (Associative Plasticity Network) — a Hebbian plasticity model, mapped onto the **gated delta-rule** Triton kernel for fast GPU training

The task treats each CIFAR-10 image as a sequence of 1024 RGB pixels (32×32 pixels, 3 features each). The model reads the sequence and classifies from the last time-step into one of 10 classes.

Both models use the same outer architecture: `input_proj → tanh → n_layers (with residual connections) → LayerNorm → linear classifier`.

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

# Compare both
python experiments/seq_cifar.py --model both --n-layers 10 --d-hidden 100 \
    --apn-lam 0.99 --apn-eta 1.0 --epochs 50 --batch-size 64

# Freeze lambda (fixed decay, only W and eta train)
python experiments/seq_cifar.py --model apn --n-layers 10 --d-hidden 100 \
    --apn-lam 0.99 --apn-eta 1.0 --freeze-lam --epochs 50 --batch-size 64
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `both` | `deltanet`, `apn`, or `both` |
| `--d-hidden` | `100` | Hidden dimension per layer |
| `--n-layers` | `10` | Number of recurrent layers |
| `--apn-lam` | `0.99` | Initial $\lambda$ for APN |
| `--apn-eta` | `0.01` | Initial $\eta$ for APN (O(1) scale is fine due to normalization) |
| `--epochs` | `50` | Training epochs |
| `--batch-size` | `64` | Batch size |
| `--lr` | `1e-3` | Learning rate (AdamW + cosine schedule) |
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
└── seq_cifar.py       # Benchmark script (DeltaNet + APN)
```

## Dependencies

- `fla` (flash-linear-attention) — installed in editable mode from this repo
- PyTorch ≥ 2.0, Triton, torchvision
- Conda environment: `fla_apn`
