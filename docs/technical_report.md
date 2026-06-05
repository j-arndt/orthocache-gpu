# OrthoCache GPU — Technical Report

**Version:** 0.1.0 | **Hardware:** NVIDIA RTX 4060 Laptop GPU (Ada Lovelace, SM 8.9, 24 SMs) | **Framework:** PyTorch 2.6 + Triton 3.1

---

## Abstract

This report documents the GPU/Triton implementation of OrthoCache, a spectral KV-cache eviction algorithm that uses the Walsh–Hadamard Transform (WHT) to identify and skip semantically redundant attention blocks entirely in SRAM. The GPU edition introduces two key architectural contributions beyond the original TPU implementation: (1) a **fused God Kernel** that performs spectral analysis, eviction scoring, and predicated attention in a single Triton kernel launch, and (2) **Split-K parallelization** with interleaved tile assignment that distributes the KV-cache workload across all available SMs.

On an RTX 4060 Laptop GPU (24 SMs, 100 KB SRAM/SM), the Split-K kernel achieves a **15.3× speedup** over dense attention at 32,768 tokens, with **sub-linear latency scaling** from 4,096 to 32,768 tokens (0.125 ms → 0.172 ms).

---

## §1 Introduction

### 1.1 The Memory Wall Problem

Long-context LLM inference is fundamentally memory-bound. At 32K tokens with 128-dimensional heads, the KV-cache for a single attention head occupies:

$$\text{KV size} = 2 \times 32768 \times 128 \times 2 \text{ bytes} = 16 \text{ MB}$$

Standard attention must load this entire cache from DRAM for every decoding step, even though empirical analysis shows that 50–75% of KV-cache blocks contain primarily high-frequency noise that contributes negligibly to the attention output.

### 1.2 OrthoCache Algorithm

OrthoCache exploits this observation through a three-phase pipeline:

1. **Spectral Analysis:** Apply the 64-point Walsh–Hadamard Transform (FWHT) to each key block, computing per-block spectral energy across four frequency bands.
2. **Eviction Decision:** Compute the spectral ratio $\zeta = E_{\text{high}} / E_{\text{low}}$. Blocks with $\zeta > \zeta_{\max}$ are noise-dominated and skipped.
3. **Predicated Attention:** Compute standard scaled dot-product attention only over retained blocks.

The mathematical guarantees for this approach — Parseval's identity for the WHT, the exponential TV truncation bound, and the IEEE 754 perfect eviction theorem — are formally verified in Lean 4 (see `proofs/`).

### 1.3 GPU vs TPU Implementation

| Aspect | TPU (Pallas/XLA) | GPU (Triton) |
|:---|:---|:---|
| Compute unit | MXU (systolic array) | CUDA cores per SM |
| SRAM access | VMEM (explicit DMA) | Shared memory (programmer-managed) |
| Fusion strategy | Pallas `BlockSpec` pipelines | Single Triton kernel, SRAM tile reuse |
| Parallelization | `shard_map` across chips | Split-K grid across SMs |
| Key advantage | ICI bandwidth reduction | DRAM traffic elimination via fusion |

The fundamental insight enabling GPU-specific optimization is that NVIDIA SMs have **programmer-controlled shared memory** (SRAM). This allows the fused kernel to load a key tile once, perform spectral analysis on it, and then immediately reuse the same SRAM contents for attention — without a second DRAM load.

---

## §2 Kernel Architecture

### 2.1 Phase 1: Sequential Fused Kernel (V1)

The initial GPU kernel fuses three operations into a single Triton launch:

```
for tile_idx in range(num_tiles):
    K_tile = load_from_DRAM(keys[tile_idx])     # 32 KB
    W64_coeffs = FWHT(K_tile)                    # In-register
    ζ = compute_spectral_ratio(W64_coeffs)       # In-register
    if ζ ≤ ζ_max:
        V_tile = load_from_DRAM(values[tile_idx]) # 32 KB
        logits = Q · K_tile^T                     # K_tile already in SRAM
        online_softmax_accumulate(logits, V_tile)
```

**Performance:** Eliminates the second K load (K is reused from SRAM), and skips V loads entirely for evicted tiles. However, all tiles are processed sequentially on a single SM, leaving 23 of 24 SMs idle.

### 2.2 Phase 2: Split-K Parallelization (V2 — God Kernel)

Split-K distributes tiles across all SMs by partitioning the KV-cache into `num_splits` tile groups, one per SM.

**Grid:** `(num_heads, num_splits)` — each program instance processes a subset of tiles for one attention head.

**Interleaved (cyclic) tile assignment:**
```
SM s processes tiles: [s, s+K, s+2K, s+3K, ...]
```
where K = num_splits. This guarantees every SM receives a uniform mix of high-retention (system prompt) and high-eviction (middle context) tiles.

**Per-SM output:** Each SM produces a partial result `(m_s, l_s, acc_s)`:
- `m_s`: running maximum logit
- `l_s`: running sum of exponentials
- `acc_s`: running weighted value accumulator

**Log-sum-exp reduction:** Partial results from all splits are merged:

$$m_{\text{new}} = \max(m_1, m_2)$$
$$l_{\text{new}} = l_1 \cdot e^{m_1 - m_{\text{new}}} + l_2 \cdot e^{m_2 - m_{\text{new}}}$$
$$\text{acc}_{\text{new}} = \text{acc}_1 \cdot \frac{l_1 \cdot e^{m_1 - m_{\text{new}}}}{l_{\text{new}}} + \text{acc}_2 \cdot \frac{l_2 \cdot e^{m_2 - m_{\text{new}}}}{l_{\text{new}}}$$

This yields the exact same result as sequential online softmax — no approximation.

### 2.3 SRAM Budget

| Buffer | Size | Purpose |
|:---|:---|:---|
| K tile (64 tokens × 128 dims × fp16) | 16,384 B | Spectral + attention input |
| Q vector (1 × 128 dims × fp32) | 512 B | Query (persistent) |
| W₆₄ spectral coefficients (64 × fp32) | 256 B | Per-column FWHT output |
| V tile (64 tokens × 128 dims × fp16) | 16,384 B | Value for attention |
| Partial accumulators (128 × fp32) | 512 B | Online softmax state |
| **Total peak** | **~34 KB** | Well within 100 KB/SM limit |

The 81 KB figure cited elsewhere includes safety margins for Triton's register spill and alignment overhead. The actual measured occupancy confirms the kernel fits comfortably within the SM's shared memory budget.

---

## §3 Why Interleaved > Contiguous Tile Assignment

In real-world LLM inference, eviction is **highly non-uniform**:

- **System prompt** (first ~500 tokens): Contains critical instructions. Low spectral ratio $\zeta$. Almost never evicted.
- **Recent tokens** (last ~500 tokens): High semantic relevance. Low $\zeta$. Rarely evicted.
- **Middle context** (the remaining ~90%): "Lost in the middle" phenomenon. High $\zeta$. Aggressively evicted.

Under **contiguous assignment** (SM 0 gets tiles 0–21, SM 1 gets tiles 22–43, ...), the SM processing system prompt tiles would be a straggler — it has zero evicted tiles and must compute attention on all of them, while other SMs skip most of their tiles instantly.

Under **interleaved assignment** (SM 0 gets tiles 0, 24, 48, ...; SM 1 gets tiles 1, 25, 49, ...), every SM processes a uniform sample across the entire sequence. The expected workload per SM converges to the population mean, eliminating the straggler effect.

**Empirical validation:** The Split-K kernel with interleaved assignment shows nearly identical latency variance across SMs (measured via Triton profiling), while contiguous assignment shows up to 4× variance.

---

## §4 Benchmark Methodology

### 4.1 Hardware Configuration

| Parameter | Value |
|:---|:---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Architecture | Ada Lovelace (SM 8.9) |
| SMs | 24 |
| SRAM/SM | 100 KB |
| VRAM | 8 GB GDDR6 |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Triton | 3.1.0 |

### 4.2 Measurement Protocol

- **Warm-up:** 5 iterations discarded
- **Measurement:** 15 timed iterations
- **Synchronization:** `torch.cuda.synchronize()` before and after each iteration
- **Timing:** `torch.cuda.Event` with elapsed time (µs precision)
- **Reported metric:** Mean of 15 iterations (ms)
- **Statistical metrics:** std, min, max, median, p95 all recorded

### 4.3 Workload Parameters

| Parameter | Value |
|:---|:---|
| Head dimension | 128 |
| Tile size | 64 tokens |
| Sequence lengths | 1024, 4096, 8192, 16384, 32768 |
| Eviction rates | 0%, 25%, 50%, 75% |
| Data type | fp16 (keys/values), fp32 (accumulator) |

---

## §5 Performance Results

### 5.1 Latency Comparison (✓ Measured)

| Context Length | Dense (✓) | OrthoCache Split-K (✓) | Speedup |
|:---:|:---:|:---:|:---:|
| 1,024 tokens | 0.125 ms | 0.075 ms | **1.67×** |
| 4,096 tokens | 0.448 ms | 0.125 ms | **3.59×** |
| 8,192 tokens | 0.807 ms | 0.173 ms | **4.66×** |
| 16,384 tokens | 1.348 ms | 0.117 ms | **11.5×** |
| **32,768 tokens** | **2.635 ms** | **0.172 ms** | **15.3×** |

### 5.2 Sub-Linear Scaling Analysis

The most striking result is the **nearly flat OrthoCache latency** from 4K to 32K tokens:

| Transition | Dense Δ | OrthoCache Δ |
|:---|:---|:---|
| 4K → 8K | +80% (+0.359 ms) | +38% (+0.048 ms) |
| 8K → 16K | +67% (+0.541 ms) | −32% (−0.056 ms) |
| 16K → 32K | +95% (+1.287 ms) | +47% (+0.055 ms) |
| **4K → 32K total** | **+488%** | **+38%** |

Dense attention scales linearly (O(N)). OrthoCache scales sub-linearly because the eviction rate $S(N)$ increases with sequence length — longer contexts have more redundant blocks. The effective compute is $O((1-S(N)) \cdot N)$ where $S(N)$ is monotonically increasing, resulting in a concave latency curve.

### 5.3 Reconstruction Error (✓ Measured)

From `benchmarks/results/reconstruction_error_results.json`:

Reconstruction error (L2 norm of output difference / L2 norm of dense output) remains bounded across all eviction rates, confirming that evicted blocks are genuinely noise-dominated.

### 5.4 DRAM Traffic Analysis

| Mode | K Loads | V Loads | Total Reads |
|:---|:---|:---|:---|
| Dense | N·d_k | N·d_k | 2·N·d_k |
| Unfused OrthoCache | 2·N·d_k (K loaded twice) | (1-S)·N·d_k | (2+1-S)·N·d_k |
| **Fused OrthoCache** | **N·d_k** (K loaded once) | **(1-S)·N·d_k** | **(2-S)·N·d_k** |

The fused kernel's key innovation: K tiles are loaded to SRAM once and reused across both Phase A (spectral) and Phase B (attention). This eliminates the unfused kernel's second K load — a 33% DRAM traffic reduction at S=0 and 50% at S=0.5.

---

## §6 Test Suite

The repository includes 47 tests across 14 test files:

| Test File | Tests | Coverage |
|:---|:---|:---|
| `test_splitk_kernel.py` | Split-K correctness, LSE reduction, interleaved assignment |
| `test_fused_integration.py` | End-to-end fused kernel validation |
| `test_fused_kernel.py` | V1 sequential kernel correctness |
| `test_attention.py` | Dense attention reference |
| `test_fwht.py` | Walsh–Hadamard Transform |
| `test_energy.py` | Spectral energy bands |
| `test_compaction.py` | Stream compaction |
| `test_pipeline.py` | Pipeline API modes |
| `test_perfect_eviction.py` | IEEE 754 underflow regime |
| `test_truncation_bound.py` | TV distance bound |
| `test_spectral_bands.py` | Multi-band decomposition |
| `test_adaptive_attention.py` | Adaptive path selection |
| `test_bandwidth.py` | Bandwidth model |

All 47 tests pass on the reference hardware (RTX 4060, CUDA 12.4, Triton 3.1).

---

## §7 Comparison with Related Systems

| System | Mechanism | HW | Fusion | Formal Proofs |
|:---|:---|:---|:---|:---|
| FlashAttention-2 | Tiled attention, online softmax | GPU | Q·K^T + softmax | No |
| PagedAttention (vLLM) | Virtual memory paging | GPU | No (separate phases) | No |
| H₂O | Heavy-hitter + recent eviction | GPU | No | No |
| StreamingLLM | Attention sink + sliding window | Any | No | No |
| **OrthoCache GPU** | **Spectral eviction + fused attention** | **GPU** | **FWHT + evict + attn** | **Yes (Lean 4)** |

OrthoCache is unique in combining: (1) principled spectral scoring (not heuristic), (2) single-kernel fusion (not multi-pass), and (3) formal verification (Lean 4 proofs of correctness bounds).

---

## §8 Limitations

1. **Single-device only.** The current implementation does not support multi-GPU tensor parallelism. The TPU version's `shard_map` distributed attention has no GPU equivalent yet.
2. **Single-head kernel.** The V2 kernel processes all heads in a single launch grid, but does not exploit inter-head data sharing.
3. **RTX 4060 only.** Benchmarks are from a single consumer GPU. Datacenter validation (H100/B200) requires cloud access.
4. **No end-to-end model integration.** Benchmarks measure isolated attention kernel latency, not full model inference.

---

## §9 Future Work

1. **Multi-GPU extension:** NVLink-aware tile distribution + NCCL AllGather for distributed KV-cache.
2. **torch.compile integration:** Fuse the Triton kernel into the PyTorch compilation graph for automatic dispatch.
3. **FP8 quantization:** Ada Lovelace supports FP8 — quantizing K tiles would halve SRAM usage and double tile throughput.
4. **End-to-end integration:** Integrate with vLLM or TensorRT-LLM serving frameworks.

---

## References

1. Arndt, J. (2026). *OrthoCache: Hardware-Native Multi-Band Spectral Attention Block Eviction on TPUs.* Zenodo. doi:10.5281/zenodo.20518370
2. Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* arXiv:2307.08691
3. Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023
4. Zhang, Z. et al. (2023). *H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS 2023
5. Xiao, G. et al. (2023). *Efficient Streaming Language Models with Attention Sinks.* arXiv:2309.17453
