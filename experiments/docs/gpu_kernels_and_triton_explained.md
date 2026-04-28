# GPU Kernels, Triton, and the Hopper Problem: A Detailed Explanation

This document explains the full stack — from how GPUs execute math, to what Triton does, to why your APN runs slower on H200 with Triton 3.3.

---

## Table of Contents

1. [What is a GPU Kernel?](#1-what-is-a-gpu-kernel)
2. [Why GPUs Are Fast (and When They're Not)](#2-why-gpus-are-fast-and-when-theyre-not)
3. [Shared Memory: The Fast Scratchpad](#3-shared-memory-the-fast-scratchpad)
4. [What is Triton?](#4-what-is-triton)
5. [Kernel Compilation and Caching](#5-kernel-compilation-and-caching)
6. [The Naive APN vs. Kernel-Based APN](#6-the-naive-apn-vs-kernel-based-apn)
7. [Chunk Method: Why It's Fast](#7-chunk-method-why-its-fast)
8. [What is Hopper (H100/H200)?](#8-what-is-hopper-h100h200)
9. [The Triton ≥ 3.4 Hopper Bug](#9-the-triton--34-hopper-bug)
10. [Why Triton 3.3 is Slow on Hopper](#10-why-triton-33-is-slow-on-hopper)
11. [Summary of Your Situation](#11-summary-of-your-situation)

---

## 1. What is a GPU Kernel?

A **kernel** is a small program that runs on the GPU. When you write PyTorch code like:

```python
y = torch.matmul(A, B)
```

PyTorch doesn't do the multiplication itself. It calls a **kernel** — a pre-compiled GPU program that knows how to multiply matrices efficiently using thousands of GPU cores in parallel.

Think of it like this:
- **CPU code** (Python/PyTorch): the manager that decides *what* to compute
- **GPU kernel**: the factory floor that actually does the computation, massively in parallel

Every operation on tensors — matmul, softmax, LayerNorm, the delta-rule recurrence — is executed by a kernel. Some are built into PyTorch (like matmul, which calls NVIDIA's cuBLAS library). Others are custom-written for specific algorithms — like the `chunk_gated_delta_rule` kernel in fla.

### Why custom kernels matter

A naive implementation of the delta rule in PyTorch would use many separate kernels:

```python
# Each line = a separate kernel launch on the GPU
for t in range(T):
    v_new = v[t] - S @ k[t]         # kernel 1: matmul
    v_new = beta[t] * v_new          # kernel 2: elementwise multiply
    S = lam * S + outer(k[t], v_new) # kernel 3: scale, kernel 4: outer product, kernel 5: add
    o[t] = q[t] @ S                  # kernel 6: matmul
```

Each kernel launch has overhead (~5-20 microseconds), and data must travel between GPU global memory and compute cores for each kernel. With 1024 timesteps × 6 kernels = 6144 kernel launches, this overhead dominates.

A **fused kernel** combines all these operations into a single GPU program that keeps data in fast on-chip memory and avoids the launch overhead entirely.

---

## 2. Why GPUs Are Fast (and When They're Not)

### The memory hierarchy

A GPU has a layered memory system. Each level is faster but smaller:

```
┌──────────────────────────────────┐
│     Global Memory (HBM)         │  ← 80-143 GB, ~2-3 TB/s bandwidth
│     Slow, but huge              │     Data lives here between kernels
├──────────────────────────────────┤
│     L2 Cache                    │  ← 40-60 MB, ~6 TB/s
│     Automatic, transparent      │
├──────────────────────────────────┤
│     Shared Memory (SRAM)        │  ← 100-228 KB per SM, ~20 TB/s
│     Fast scratchpad, programmer │     This is where the magic happens
│     controlled                  │
├──────────────────────────────────┤
│     Registers                   │  ← ~256 KB per SM, instantaneous
│     Per-thread private storage  │     Fastest possible
└──────────────────────────────────┘
```

**HBM** (High Bandwidth Memory) is the main GPU memory — the 80 GB on an A100 or 143 GB on an H200. It's fast compared to CPU RAM, but it's the bottleneck for most operations. Reading a D×D matrix from HBM takes real time.

**Shared memory** is a small, fast scratchpad on each Streaming Multiprocessor (SM). A kernel can explicitly load data from HBM into shared memory, work on it there at ~10× the speed, and write results back.

### The key insight

Most GPU computations are **memory-bound**, not compute-bound. The GPU's arithmetic units can do trillions of operations per second, but they spend most of their time waiting for data to arrive from HBM.

The art of writing fast kernels is **keeping data in shared memory/registers as long as possible** to minimize HBM round-trips.

---

## 3. Shared Memory: The Fast Scratchpad

Each SM (Streaming Multiprocessor) on the GPU has a fixed amount of shared memory:

| GPU | Architecture | Shared Memory per SM |
|-----|-------------|---------------------|
| A100 | Ampere (sm_80) | **164 KB** |
| H100 | Hopper (sm_90) | **228 KB** |
| H200 | Hopper (sm_90) | **228 KB** |

This is why **D=173 can't run on A100 but works on H200**: the chunk kernel needs to hold a D×D tile in shared memory during the intra-chunk computation. For D=173 in float32:

$$173 \times 173 \times 4 \text{ bytes} = 119{,}716 \text{ bytes} \approx 117 \text{ KB}$$

With additional workspace (indices, accumulators, etc.), this exceeds A100's 164 KB limit but fits within H200's 228 KB.

For D=100:

$$100 \times 100 \times 4 \text{ bytes} = 40{,}000 \text{ bytes} \approx 39 \text{ KB}$$

This comfortably fits on both A100 and H200.

---

## 4. What is Triton?

**Triton** is a programming language and compiler created by OpenAI for writing GPU kernels. It sits between two worlds:

- **CUDA** (NVIDIA's native GPU language): Maximum control, maximum complexity. Writing a fast CUDA kernel requires managing threads, warps, memory coalescing, bank conflicts, etc.
- **PyTorch** (Python): Easy to write, but no control over how the GPU executes it.

Triton lets you write kernels in a Python-like syntax while the **Triton compiler** handles the hard GPU optimization work:

```python
# Simplified Triton kernel example
@triton.jit
def fused_recurrence_kernel(q, k, v, beta, o, ...):
    # Load a tile of data from HBM into registers
    b_k = tl.load(p_k, mask=mask_k)
    b_v = tl.load(p_v, mask=mask_v)
    
    # Do the delta-rule update entirely in registers (fast!)
    b_v_minus = tl.sum(b_h * b_k[None, :], axis=1)
    b_v -= b_v_minus
    b_v *= b_beta
    b_h += b_k[None, :] * b_v[:, None]
    
    # Write result back to HBM
    tl.store(p_o, b_o)
```

### How Triton compiles

When you first call a Triton kernel, the compiler:

1. Takes your Python-like Triton code
2. Converts it to an intermediate representation (IR)
3. Optimizes it for the specific GPU architecture (Ampere, Hopper, etc.)
4. Generates actual GPU machine code (PTX → CUBIN)
5. Launches the compiled kernel on the GPU

This compilation step is what makes the **first epoch slow** (kernel JIT compilation) and subsequent epochs fast (compiled code is cached).

### Architecture-specific optimization

This is crucial: **the same Triton source code produces different GPU machine code for different architectures**. The Triton compiler knows about each GPU's specific features and generates code to exploit them. Newer Triton versions add support for newer GPU features.

---

## 5. Kernel Compilation and Caching

### JIT (Just-In-Time) compilation

Triton kernels are compiled **the first time they're called** with a specific set of parameters (tensor shapes, data types, etc.). This is called JIT compilation.

For the `chunk_gated_delta_rule` kernel, the first backward pass triggers compilation of several sub-kernels. This can take 30-120 seconds, which is why the first epoch is much slower.

### Caching

After compilation, the generated GPU machine code is cached on disk (typically in `~/.triton/cache/`). On subsequent runs:

1. Triton checks: "Have I seen this exact kernel + parameters before?"
2. If yes: load the cached machine code (milliseconds)
3. If no: recompile (seconds to minutes)

This is why:
- **First run after a Triton version change**: slow (cache miss, full recompile)
- **Subsequent runs**: fast (cache hit)
- **Different tensor shapes**: may trigger recompilation (new cache entry)

### On Beaker

Each Beaker job starts fresh — no persistent cache. So every job pays the JIT compilation cost on the first epoch. This is an unavoidable ~30-60s overhead per job.

---

## 6. The Naive APN vs. Kernel-Based APN

### Your original naive PyTorch APN

Before using the fla kernel, a naive APN implementation would look like:

```python
M = torch.zeros(B, D, D)  # Plastic memory matrix — stored in HBM

for t in range(1024):      # Sequential loop in Python
    x_act = torch.tanh(x[:, t])           # Kernel launch 1: tanh
    h = W @ x_act + M @ x_act             # Kernel launch 2-3: two matmuls
    M = lam * M + eta * outer(h, x_act)   # Kernel launch 4-6: scale, outer, add
    output[:, t] = h                       # Kernel launch 7: copy
```

**Problems:**
1. **1024 × 7 = 7168 kernel launches** — each has ~10μs overhead = 72ms just in launch overhead
2. **M (D×D) is read/written to HBM every timestep** — 1024 × 2 × D² × 4 bytes of HBM traffic
3. **Python loop** — the CPU sends one kernel at a time, GPU idles between launches
4. **Backpropagation** — PyTorch must save all 1024 intermediate M matrices (1024 × D × D × 4 bytes) for the backward pass. For D=173, that's **1024 × 173 × 173 × 4 = 123 MB per layer**, and with 10 layers = **1.2 GB** just for storing intermediate states.

### The fused kernel APN

The `chunk_gated_delta_rule` kernel does the **entire recurrence** in a single kernel launch:

```
One kernel launch:
  - Load q, k, v, g, beta from HBM
  - Compute the full 1024-step recurrence on-chip
  - Write output o back to HBM
```

**Improvements:**
1. **1 kernel launch** instead of 7168
2. **State matrix stays in shared memory / registers** — never touches HBM between timesteps
3. **No Python loop** — the GPU does the entire computation autonomously
4. **Smart backward pass** — the chunk method avoids storing all intermediate states (see next section)

This is why the kernel-based APN is **10-100× faster** than the naive version.

---

## 7. Chunk Method: Why It's Fast

### The sequential bottleneck

The delta-rule recurrence is inherently sequential: $S_t$ depends on $S_{t-1}$. You can't compute timestep 500 without first computing timesteps 1-499.

A `fused_recurrent` kernel respects this: it loops through all T timesteps one by one. The GPU's massive parallelism (thousands of cores) can only parallelize across the batch and the D×D state dimensions — **not across time**.

### How chunking helps

The `chunk` method splits the sequence into chunks of C=64 tokens and uses a mathematical trick (the **WY representation**) to restructure the computation:

```
Sequence: [t=0 ... t=63] [t=64 ... t=127] ... [t=960 ... t=1023]
           chunk 0         chunk 1              chunk 15

Step 1 (PARALLEL across chunks): 
  Within each chunk, solve the recurrence using a matrix formulation.
  This converts the sequential loop into a triangular matrix solve,
  which the GPU can parallelize.

Step 2 (SEQUENTIAL across chunks):
  Propagate the state from chunk 0 → chunk 1 → ... → chunk 15.
  Only 16 sequential steps instead of 1024.
```

### The math (simplified)

Within a chunk of C timesteps, the recurrence can be written as a triangular linear system:

$$\begin{bmatrix} 1 & 0 & 0 & \cdots \\ -\beta_1 k_1 k_0^T & 1 & 0 & \cdots \\ -\beta_2 k_2 k_0^T g_{2,1} & -\beta_2 k_2 k_1^T & 1 & \cdots \\ \vdots & & & \ddots \end{bmatrix} \begin{bmatrix} \tilde{v}_0 \\ \tilde{v}_1 \\ \tilde{v}_2 \\ \vdots \end{bmatrix} = \begin{bmatrix} v_0 - S_{\text{prev}} k_0 \\ v_1 - S_{\text{prev}} k_1 g_{1,0} \\ v_2 - S_{\text{prev}} k_2 g_{2,0} \\ \vdots \end{bmatrix}$$

This triangular system can be solved in parallel using a forward substitution that maps well to GPU matrix operations. The key insight is:

1. **Intra-chunk** (parallel): Solve C=64 timesteps simultaneously using this matrix formulation. The GPU can use its matrix units for this.
2. **Inter-chunk** (sequential): Update the state S across the 16 chunk boundaries. This is just 16 sequential state updates instead of 1024.

### Complexity comparison

| Method | Sequential depth | Parallel work | HBM reads of S |
|--------|-----------------|---------------|----------------|
| Naive Python loop | T = 1024 | Batch × D² | T × D² = 1024 × D² |
| Fused recurrent | T = 1024 | Batch × D² | 1 (load once) |
| Chunk (C=64) | T/C = 16 | Batch × D² × C | T/C × D² = 16 × D² |

The chunk method's advantage grows with sequence length:
- T=1024: 16 sequential steps (64× reduction)
- T=8192: 128 sequential steps (64× reduction)
- T=65536: 1024 sequential steps (64× reduction)

For LLM training with T=2048 or longer, the chunk method is significantly faster.

### Memory advantage for backpropagation

During the backward pass, you need the intermediate state $S_t$ to compute gradients. The naive approach stores all T states. The chunk method only stores states at **chunk boundaries** (T/C states), reducing memory by a factor of C=64:

- Naive backward: store 1024 states × D² × 4 bytes = 123 MB (for D=173)
- Chunk backward: store 16 states × D² × 4 bytes ≈ 2 MB (for D=173)

This is why you saw 2 GB GPU memory for 10-layer APN instead of the 26 GB a naive implementation would need.

---

## 8. What is Hopper (H100/H200)?

**Hopper** is NVIDIA's GPU architecture (2022-2024), used in the H100 and H200 GPUs. Each GPU architecture generation introduces new hardware features. The relevant ones for kernel performance:

### Architecture generations

| Generation | GPUs | Year | Key features |
|-----------|------|------|-------------|
| Ampere (sm_80) | A100, A6000 | 2020 | Async memory copy, large shared memory |
| Hopper (sm_90) | H100, H200 | 2022 | TMA, warp specialization, larger shared memory |

### Hopper's new features

#### 1. Tensor Memory Accelerator (TMA)

On Ampere, loading data from HBM to shared memory requires explicit programming:
- The kernel must compute memory addresses
- Issue load instructions for each element
- Handle alignment and coalescing manually

Hopper adds **TMA** — a dedicated hardware unit that handles bulk memory transfers:
- The kernel just says "load this 64×64 tile from HBM to shared memory"
- TMA handles the addresses, alignment, and transfer autonomously
- The compute cores can do useful work while TMA loads data in the background

This is like the difference between manually carrying boxes one by one vs. having a conveyor belt that loads them automatically.

#### 2. Warp Specialization

On Ampere, all GPU threads (grouped into "warps" of 32 threads) do the same thing: load data, compute, store results. They alternate between memory operations and compute operations.

On Hopper, different warps can **specialize**:
- "Producer" warps: focus on loading data from HBM via TMA
- "Consumer" warps: focus on computation using tensor cores

This is like a factory where some workers only move materials and others only assemble — instead of everyone doing both jobs and idling while waiting.

#### 3. Distributed Shared Memory

Hopper allows SMs to directly access each other's shared memory (within a cluster of SMs). This enables cross-SM cooperation without going through slow HBM.

#### 4. Larger Shared Memory

228 KB per SM (vs. A100's 164 KB). This allows larger tile sizes, reducing the number of tiles and HBM round-trips.

### Why these features matter for delta-rule kernels

The chunk kernel's inner loop:
1. **Load** q, k, v tiles from HBM → shared memory (TMA accelerates this)
2. **Compute** the intra-chunk triangular solve (warp specialization: compute while loading next tile)
3. **Store** results back to HBM

On Hopper with proper compiler support, steps 1 and 2 can **overlap** — the GPU loads the next chunk's data while still computing the current chunk. This pipelining can nearly double throughput.

---

## 9. The Triton ≥ 3.4 Hopper Bug

### What happened

Triton 3.4.0 added Hopper-specific optimizations — TMA instructions, warp specialization, new register allocation strategies. These optimizations are complex and interact with each other in subtle ways.

For the specific kernel `chunk_bwd_dqkwg` (the backward pass of the gated chunk kernel), Triton 3.4+ generates code that **produces numerically incorrect results** on Hopper GPUs. The issue is tracked as [fla issue #640](https://github.com/fla-org/flash-linear-attention/issues/640).

The likely cause is a bug in Triton's code generation for one of:
- Register spilling (when there aren't enough registers, data is temporarily saved to local memory — if the spill/reload order is wrong, data gets corrupted)
- Memory barrier placement (on Hopper, TMA operations are asynchronous — if the compiler doesn't insert proper synchronization barriers, a compute warp might read data before TMA finishes writing it)
- Instruction scheduling (reordering operations in a way that's valid on Ampere but not on Hopper due to the different memory model)

### Why it only affects the backward pass

The forward pass is simpler — it reads inputs and writes outputs in a straightforward pattern. The backward pass of the chunk kernel is much more complex:
- It must recompute intermediate values (to save memory)
- It has more data dependencies (gradients flow backward through the triangular solve)
- It uses more shared memory (holding both forward and backward quantities)

This complexity exposes more compiler edge cases.

### The tilelang alternative

`tilelang` is an alternative kernel compiler (built on TVM) that generates GPU code differently from Triton. The fla library checks: if Triton ≥ 3.4 on Hopper, try tilelang instead.

But tilelang has its own limitation: it requires tensor dimensions to be **tile-aligned** (multiples of the tile size, typically 32 or 64). D=173 is not tile-aligned, so tilelang crashes with a layout error.

---

## 10. Why Triton 3.3 on Hopper: Not Slow, But Necessary

### The surprising result

Initial hypothesis: Triton 3.3 would be slow on Hopper because it lacks Hopper-specific optimizations (TMA, warp specialization). **This turned out to be wrong.**

Benchmarks show D=100 runs **faster** on H200 with Triton 3.3 (28s) than on A100 with Triton 3.6 (33s). Hopper's raw hardware advantages (more SMs, higher bandwidth, larger shared memory) more than compensate for missing compiler optimizations at this scale.

### Why we pin to Triton 3.3 anyway

We pin `triton>=3.3.0,<3.4.0` not for performance but for **correctness**: Triton ≥ 3.4 produces wrong gradients in the gated chunk backward kernel on Hopper (fla issue #640). Triton 3.3 generates correct code.

### The real performance bottleneck: non-tile-aligned dimensions

The 10× slowdown from D=100 to D=173 comes from **tile alignment**, not from the Triton version. See [Section 11](#11-summary-of-your-situation) for details.

---

## 11. Summary of Your Situation

### Actual benchmark results

| Config | D | GPU | Triton | Time/epoch |
|--------|---|-----|--------|-----------|
| A100 | 100 | Ampere | 3.6 | 33s |
| H200 | 100 | Hopper | 3.3 | **28s** |
| H200 | 173 | Hopper | 3.3 | 281s |

**Key finding**: Triton 3.3 on Hopper is NOT slow — D=100 runs *faster* on H200 than A100. The bottleneck is **D=173 being non-tile-aligned**.

### Why D=173 is 10× slower than D=100 (not 3×)

The chunk kernel processes the D×D state in tiles (typically 64×64 blocks):

- **D=100**: ceil(100/64) = 2 tiles per dimension → 2×2 = **4 tile blocks**
- **D=173**: ceil(173/64) = 3 tiles per dimension → 3×3 = **9 tile blocks** (with 19 elements of padding waste per dimension)

So the actual scaling is:
- D² arithmetic scaling: $(173/100)^2 = 3\times$
- Tile block scaling: $9/4 = 2.25\times$
- Padding waste + reduced occupancy: ~$1.3\text{-}1.5\times$
- **Combined**: $3 \times 2.25 \times 1.3 \approx 8\text{-}10\times$ (matches the observed 10×)

### Recommended dimensions for efficient tiling

| D | Tiles per dim | Tile blocks | Padding waste | Relative efficiency |
|---|--------------|-------------|---------------|-------------------|
| 64 | 1 | 1 | 0% | Best |
| 100 | 2 | 4 | 22% | Good |
| 128 | 2 | 4 | 0% | Best |
| 173 | 3 | 9 | 10% | **Poor** (tile count jump) |
| 192 | 3 | 9 | 0% | OK |
| 256 | 4 | 16 | 0% | Good (standard LLM head dim) |

**For future experiments**: prefer D=128 or D=192 over D=173. D=128 gives 1.6× the parameters of D=100 with the same number of tile blocks (4).

### Future resolution paths

1. **fla fixes #640** → unpin Triton → get full Hopper speed
2. **tilelang fixes non-aligned dimensions** → works as fallback for D=173
3. **fla implements fused_recurrent backward for gated variant** → avoids chunk kernel entirely
4. **Use tile-aligned dimensions** (D=128 or D=192) → tilelang works now, chunk method works with Triton 3.6

For LLM training later, you'll likely use D=128 or D=256 (standard head dimensions), which sidesteps the tilelang alignment issue and lets you use Triton 3.6 with tilelang on Hopper.
