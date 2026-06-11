<p align="center">
  <img src="benchmarks/plots/hero_multihead.png" alt="OrthoCache: Fused Spectral Eviction + Attention — crossover at 4K tokens" width="720" />
</p>

<h1 align="center">OrthoCache GPU</h1>

<p align="center">
  <strong>Spectral KV-Cache Eviction for NVIDIA GPUs — Fused Walsh–Hadamard Attention with Split-K Parallelization and GQA Cauchy-Schwarz Consensus</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" /></a>
  <a href="https://pytorch.org/"><img alt="PyTorch 2.5+" src="https://img.shields.io/badge/PyTorch-%E2%89%A52.5-ee4c2c?logo=pytorch&logoColor=white" /></a>
  <a href="https://triton-lang.org/"><img alt="Triton 3.0+" src="https://img.shields.io/badge/Triton-%E2%89%A53.0-7B68EE" /></a>
  <a href="LICENSE"><img alt="License: AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-green" /></a>
</p>

---

## What Is OrthoCache?

OrthoCache is a **KV-cache eviction algorithm** that uses spectral analysis (Walsh–Hadamard Transform) to identify and skip semantically redundant attention blocks — entirely in SRAM, with zero CPU round-trips. Instead of scoring blocks with attention itself (circular), OrthoCache analyzes the **frequency-domain energy distribution** of each key block: blocks dominated by high-frequency noise get evicted before attention is ever computed.

### Key Results (RTX 4060 Laptop GPU, 32 heads, 50% eviction)

| Context Length | Dense Attention | Split-K OrthoCache | Speedup | KV Memory Saved |
|:---:|:---:|:---:|:---:|:---:|
| 1,024 tokens | 0.106 ms | 0.207 ms | 0.51× | 50% |
| 2,048 tokens | 0.332 ms | 0.367 ms | 0.91× | 50% |
| **4,096 tokens** | **0.668 ms** | **0.614 ms** | **1.09×** | **50%** |
| 8,192 tokens | 1.279 ms | 1.020 ms | **1.25×** | 50% |
| 16,384 tokens | 2.536 ms | 2.042 ms | **1.24×** | 50% |
| **32,768 tokens** | **4.862 ms** | **3.789 ms** | **1.28×** | **50%** |

> **Crossover at ~4K tokens.** Below 4K the spectral analysis overhead exceeds the eviction savings. Above 4K, OrthoCache is both faster and uses half the KV-cache memory — meaning you can serve **2× more concurrent users** on the same GPU. Mathematical guarantees are [formally verified in Lean 4](#lean-4-formal-verification).

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/j-arndt/orthocache-gpu.git && cd orthocache-gpu
pip install -e ".[dev]"

# Run the test suite (150 tests)
pytest

# Run benchmarks (requires CUDA GPU)
python benchmarks/profiling.py
```

### Requirements

| Dependency | Version |
|:---|:---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.5.0 (with CUDA) |
| Triton | ≥ 3.0.0 |
| NVIDIA GPU | Ampere (SM 8.0) or newer |
| CUDA Toolkit | ≥ 12.0 |

---

## Usage

### Multi-Head Split-K Attention (Recommended)

```python
from orthocache_gpu import fused_orthocache_attention_v2

# All heads processed in a single kernel launch
# Interleaved tile assignment across SMs for balanced workload
output, metadata = fused_orthocache_attention_v2(
    q,          # (num_heads, head_dim) — queries
    keys,       # (num_heads, seq_len, head_dim) — key cache
    values,     # (num_heads, seq_len, head_dim) — value cache
    zeta_max=5.0,
)
# metadata contains: num_splits, tile_assignment, latency_ms
```

### Pipeline API

```python
from orthocache_gpu import orthocache_forward

output, metadata = orthocache_forward(
    q, keys, values,
    mode='triton_fused',  # Uses Split-K God Kernel
    zeta_max=5.0,
)
```

### Single-Head V1 (for debugging/comparison)

```python
from orthocache_gpu import fused_orthocache_attention

output, metadata = fused_orthocache_attention(
    q,          # (1, head_dim) — single query
    keys,       # (seq_len, head_dim) — key cache
    values,     # (seq_len, head_dim) — value cache
    zeta_max=5.0,
)
```

### GQA / MQA Attention (Phase 7c — Cauchy-Schwarz Gate)

For models using Grouped-Query Attention (LLaMA-3, Mistral, Gemma), the V3 kernel replaces the blind ζ threshold with a **query-aware Cauchy-Schwarz spectral gate**:

```python
from orthocache_gpu.triton_kernels.gqa_eviction import fused_orthocache_attention_v3_gqa

# G query heads share each KV head (e.g., LLaMA-3: G=4, Mistral: G=4)
output, metadata = fused_orthocache_attention_v3_gqa(
    q,          # (num_query_heads, head_dim) — all query heads
    keys,       # (num_kv_heads, seq_len, head_dim) — key cache
    values,     # (num_kv_heads, seq_len, head_dim) — value cache
    tau=1.0,    # Cauchy-Schwarz threshold
    num_query_groups=4,  # G — queries per KV head
)
# Eviction decision: max_g(‖Q_g‖₂ · ‖K_high‖_F) ≤ τ
```

**Why Cauchy-Schwarz instead of naive consensus?** A passive consensus protocol ("evict only if ALL query heads agree") cripples eviction rates from 50% to ~20%. The Cauchy-Schwarz gate evaluates the **actual spectral alignment** between each query and the K tile's high-frequency band. Query heads with no high-frequency energy have a near-zero Cauchy-Schwarz multiplier, neutralizing their veto.

**Mathematical guarantee** ([formally verified in Lean 4](#lean-4-formal-verification)):

$$\max_{g \in [1, G]} \left( \|Q_{g, \text{high}}\|_2 \cdot \|K_{\text{high}}\|_2 \right) \le \tau \implies \text{tile eviction is safe for ALL } G \text{ heads}$$

---

## Evaluation Harness

The `eval/` directory contains a complete evaluation suite for validating OrthoCache correctness and quality.

### Perplexity Evaluation

```bash
# Full tau sweep (256 tokens, CPU)
python eval/perplexity_eval.py --model-path /path/to/tinyllama1.1b --device cpu
```

### Autoregressive Generation (Needle-In-A-Haystack)

```bash
# NIAH test — verifies perfect retrieval under eviction
python eval/generate.py --model-path /path/to/tinyllama1.1b \
    --needle --tau 1.06 --alpha 0.0 --device cpu

# With Spectral Auto-Clutch (Platinum 2)
python eval/generate.py --model-path /path/to/tinyllama1.1b \
    --needle --tau 1.06 --auto-clutch --device cpu

# Hallucination suppression (Platinum 3)
python eval/generate.py --model-path /path/to/tinyllama1.1b \
    --empty-haystack --tau 1.06 --amputate --device cpu
```

---

## End-to-End Validation (TinyLlama 1.1B)

| Test | Command | Result |
|:---|:---|:---|
| Perplexity (256-tok) | `perplexity_eval.py` | PPL 7.74 at tau=0.677 (dense baseline: 6.91) |
| Perplexity (2048-tok) | `perplexity_eval.py` | Denoising valley at tau=1.06: PPL **improves** under eviction |
| NIAH (ungoverned) | `generate.py --needle` | Perfect retrieval at 73-93% decode skip |
| NIAH (auto-clutch) | `generate.py --needle --auto-clutch` | Perfect retrieval, autonomous alpha modulation |
| Hallucination (empty haystack) | `generate.py --empty-haystack --amputate` | Fabrication suppressed: `"3425"` → hedged refusal |
| Gradient flow (regularizer) | `SpectralRegularizer.validate_gradient_flow()` | Finite, nonzero gradients through Walsh projection |

---

## Platinum Paradigms

Four "Platinum" upgrades extend the core spectral eviction engine:

1. **Walsh Subspace Projection** — Exact spectral Q-norm via block averaging (72 FLOPs). Tightens the Cauchy-Schwarz bound, boosting decode skip rate from 57-79% to 73-93%.
2. **Spectral Auto-Clutch** — Autonomously switches between generative (α=0.3) and retrieval (α=0.0) modes using per-token Q_high energy.
3. **Active Hallucination Amputation** — Wires H_score into softmax temperature to suppress fabrication when the model searches for non-existent information.
4. **Spectral Pre-Training Regularizer** — Differentiable Walsh-domain penalty that forces models to learn natively sparse KV representations.

---

## Architecture

### Split-K Fused Kernel (Phase 7b)

The capstone optimization fuses three operations — FWHT spectral analysis, ζ eviction decision, and predicated attention — into a **single Triton kernel launch** with Split-K parallelization across all SMs.

```mermaid
flowchart TB
    classDef kernel fill:#1a1b26,stroke:#58a6ff,stroke-width:2px,color:#e6edf3
    classDef sm fill:#161b22,stroke:#8b949e,stroke-width:1px,color:#c9d1d9
    classDef phaseA fill:#1c2333,stroke:#d2a8ff,stroke-width:2px,color:#d2a8ff
    classDef phaseB fill:#1c2333,stroke:#58a6ff,stroke-width:2px,color:#58a6ff
    classDef evict fill:#2d1b1b,stroke:#f85149,stroke-width:2px,color:#f85149
    classDef keep fill:#1b2d1b,stroke:#3fb950,stroke-width:2px,color:#3fb950
    classDef reduce fill:#1a1b26,stroke:#f0883e,stroke-width:2px,color:#f0883e
    classDef output fill:#0d1117,stroke:#3fb950,stroke-width:3px,color:#3fb950
    classDef sram fill:#1c2333,stroke:#8b949e,stroke-dasharray:5 5,color:#8b949e

    LAUNCH["🚀 Single Kernel Launch<br/><b>Grid: (num_heads, num_splits)</b><br/>32 heads × auto splits → 24 SMs saturated"]:::kernel
    LAUNCH --> TILE_ASSIGN

    TILE_ASSIGN["<b>Interleaved (Cyclic) Tile Assignment</b><br/>Each SM gets tiles [s, s+K, s+2K, ...]<br/>Uniform mix of dense + sparse regions"]:::kernel
    TILE_ASSIGN --> SM0 & SM1 & SM2 & SMN

    SM0["<b>SM 0</b><br/>Tiles 0, K, 2K, ..."]:::sm
    SM1["<b>SM 1</b><br/>Tiles 1, K+1, 2K+1, ..."]:::sm
    SM2["<b>SM 2</b><br/>Tiles 2, K+2, 2K+2, ..."]:::sm
    SMN["<b>SM 23</b><br/>Tiles 23, K+23, ..."]:::sm

    SM0 --> LOAD_K
    SM1 --> LOAD_K
    SM2 --> LOAD_K
    SMN --> LOAD_K

    subgraph PIPELINE ["Per-SM Pipeline (× 24 SMs)"]
        direction TB
        LOAD_K["📥 <b>Load K_tile to SRAM</b><br/>32 KB · 64 tokens × 128 dims × fp32"]:::sram

        subgraph PHASE_A ["Phase A: Spectral Eviction"]
            direction TB
            FWHT["<b>FWHT Transform</b><br/>W₆₄ · K_tile → spectral coefficients<br/><i>K_tile STAYS in SRAM</i>"]:::phaseA
            ENERGY["<b>Band Energy Scoring</b><br/>E_low = Σ|c[0:8]|² — E_high = Σ|c[56:64]|²<br/>ζ = E_high / E_low"]:::phaseA
            DECISION{{"ζ > ζ_max ?"}}:::phaseA
            FWHT --> ENERGY --> DECISION
        end

        LOAD_K --> FWHT
        DECISION -- "YES: noise tile" --> SKIP["⏭️ <b>SKIP</b><br/>No V load · no attention<br/>DRAM saved: 32 KB"]:::evict
        DECISION -- "NO: semantic tile" --> ATTN

        subgraph PHASE_B ["Phase B: Predicated Attention"]
            direction TB
            ATTN["<b>Q · K_tileᵀ → logits</b><br/>K_tile REUSED from SRAM · zero reload"]:::phaseB
            LOAD_V["📥 <b>Load V_tile from HBM</b><br/>32 KB"]:::sram
            SOFTMAX["<b>Online Softmax Accumulate</b><br/>Update running m, l, acc"]:::phaseB
            ATTN --> LOAD_V --> SOFTMAX
        end

        SKIP --> PARTIAL
        SOFTMAX --> PARTIAL
        PARTIAL["📤 <b>Partial State</b><br/>m_partial, l_partial, acc_partial"]:::sm
    end

    PARTIAL --> REDUCE
    REDUCE["<b>Log-Sum-Exp Reduction Kernel</b><br/>Grid: (num_heads,)<br/>Merge all partial states → numerically exact output"]:::reduce
    REDUCE --> OUTPUT
    OUTPUT["✅ <b>Attention Output</b><br/>(num_heads, head_dim)<br/><br/>SRAM: 81 KB peak < 100 KB/SM ✓<br/>DRAM: K loaded once · V skipped for evicted tiles"]:::output
```

### Why Interleaved (Cyclic) Tile Assignment?

In real LLM inference, eviction is **non-uniform**: the system prompt (first ~500 tokens) and recent tokens are almost never evicted, while the middle 90% gets aggressively pruned. **Contiguous** tile assignment would create straggler SMs — one SM gets all the dense system-prompt tiles while another gets only evicted tiles and finishes instantly.

**Interleaved assignment** (`tile_ids = [s, s+K, s+2K, ...]`) guarantees every SM gets a uniform mix of high-retention and high-eviction tiles, preventing any single SM from becoming a bottleneck.

---

## Relationship to TPU Version

| Aspect | TPU ([orthocache](https://github.com/j-arndt/orthocache)) | GPU (this repo) |
|:---|:---|:---|
| Algorithm | Identical | Identical |
| Formal proofs | Lean 4 (shared) | Lean 4 (shared) |
| Kernel language | Pallas | Triton |
| Parallelization | `shard_map` | Split-K grid |
| Compilation | XLA/HLO | `torch.compile` |
| Framework | JAX | PyTorch |

The mathematical guarantees (Parseval identity, exponential TV bound, Cauchy-Schwarz spectral gate) are properties of the algorithm, not the hardware.

---

## Lean 4 Formal Verification

The mathematical guarantees are formally verified in [Lean 4](https://leanprover.github.io/) with Mathlib:

| Proof Module | Theorem | Description |
|:---|:---|:---|
| [`ParsevalWHT.lean`](proofs/OrthoCacheMath/ParsevalWHT.lean) | `WHT_orthogonal` | H_nᵀ · H_n = 2ⁿ · I (orthogonality) |
| [`ParsevalWHT.lean`](proofs/OrthoCacheMath/ParsevalWHT.lean) | `parseval_WHT` | ‖H_n · x‖² = 2ⁿ · ‖x‖² (energy preservation) |
| [`TruncationBound.lean`](proofs/OrthoCacheMath/TruncationBound.lean) | `orthocache_truncation_bound` | TV(α, α̂) ≤ \|S^c\| · exp(β − z_max) |
| [`QuantizedTruncation.lean`](proofs/OrthoCacheMath/QuantizedTruncation.lean) | `perfect_eviction_tv_zero` | When z_max − β ≥ 88.72, TV = 0 exactly |
| [`CauchySchwarzGate.lean`](proofs/OrthoCacheMath/CauchySchwarzGate.lean) | `inner_eq_spectral_inner` | ⟨q, k⟩ = ⟨Hq, Hk⟩ / 2ⁿ (spectral inner product) |
| [`CauchySchwarzGate.lean`](proofs/OrthoCacheMath/CauchySchwarzGate.lean) | `subband_decomposition` | ⟨v, w⟩ = ⟨v_low, w_low⟩ + ⟨v_high, w_high⟩ |
| [`CauchySchwarzGate.lean`](proofs/OrthoCacheMath/CauchySchwarzGate.lean) | `spectral_gate_criterion` | \|⟨Q̂_high, K̂_high⟩\| ≤ ‖Q̂_high‖₂ · τ |
| [`GQAMonotonicity.lean`](proofs/OrthoCacheMath/GQAMonotonicity.lean) | `gqa_eviction_safe` | sup_g \|⟨Q̂_g, K̂⟩_high\| ≤ τ (group safety) |
| [`GQAMonotonicity.lean`](proofs/OrthoCacheMath/GQAMonotonicity.lean) | `gqa_spectral_gate` | ‖Q̂_g_high‖ ≤ τ_q ∧ ‖K̂_high‖ ≤ τ_k → sup ≤ τ_q·τ_k |

These proofs are **algorithm-generic** — they hold over ℝ and general matrices, with no GPU or TPU specifics. The IEEE 754 underflow threshold (88.72) applies identically to all float32 hardware.

```bash
# Verify proofs (requires Lean 4 + Mathlib)
cd proofs && lake build
```

---

## Documentation

| Document | Description |
|:---|:---|
| [`docs/mathematical_framework.md`](docs/mathematical_framework.md) | Rigorous mathematical reference: spectral energy, truncation bounds, Split-K correctness |
| [`docs/technical_report.md`](docs/technical_report.md) | GPU kernel architecture, benchmark methodology, performance analysis |
| [`docs/cost_benefit_analysis.md`](docs/cost_benefit_analysis.md) | NVIDIA fleet economics, consumer GPU analysis, cloud cost impact |
| [`paper/orthocache_gpu.tex`](paper/orthocache_gpu.tex) | GPU-specific paper (IEEE format) |
| [`eval/`](eval/) | Perplexity sweeps, NIAH generation, hallucination eval, entropy calibration |

---

## Repository Structure

```
orthocache-gpu/
├── src/orthocache_gpu/
│   ├── __init__.py                   # Public API surface
│   ├── pipeline.py                   # End-to-end forward pass (all modes)
│   ├── fwht.py                       # Fast Walsh–Hadamard Transform
│   ├── spectral_energy.py            # Multi-band spectral decomposition
│   ├── compaction.py                 # Stream compaction (sort + gather)
│   ├── adaptive_attention.py         # Adaptive path dispatcher
│   ├── lean_attention.py             # Pure PyTorch fallback
│   ├── bandwidth_model.py            # Multi-GPU bandwidth model
│   ├── perfect_eviction.py           # Eviction regime classifier
│   ├── eviction_governor.py          # Residual Governor (Pareto perplexity control)
│   ├── norm_cache.py                 # SpectralNormCache (O(1) decode gate)
│   ├── reference.py                  # PyTorch reference implementation
│   ├── spectral_clutch.py            # Platinum 2: Spectral Auto-Clutch
│   ├── hallucination_gate.py         # Platinum 3: Hallucination Amputator
│   ├── spectral_regularizer.py       # Platinum 4: Spectral Pre-Training Regularizer
│   ├── bucketed_attention.py         # Bucketed attention variant
│   ├── cuda_bridge.py                # CUDA bridge utilities
│   ├── dynamic_attention.py          # Dynamic attention routing
│   └── triton_kernels/
│       ├── __init__.py               # Kernel exports
│       ├── fused_eviction.py         # V2 Split-K God Kernel + V1 sequential
│       ├── gqa_eviction.py           # V3 GQA Cauchy-Schwarz spectral gate
│       ├── decode_gate.py            # O(1) decode gate + Platinum 1 Walsh Subspace Projection
│       ├── sparse_attention.py       # Block-sparse attention kernel
│       ├── indirect_attention.py     # Indirect indexing kernel
│       └── fwht_fused_prototype.py   # FWHT spectral eviction (TILE=64)
├── eval/
│   ├── perplexity_eval.py            # Full perplexity evaluation harness (PPL sweep + FWHT)
│   ├── generate.py                   # Autoregressive generation (NIAH, hallucination, Platinum)
│   ├── hallucination_eval.py         # Gold 3: Hallucination Exhaust baseline
│   ├── pyramid_eval.py              # Pyramid VRAM scaling evaluation
│   ├── calibrate_entropy.py          # Entropy governor calibration
│   └── results/                      # JSON results from all evaluation runs (14 files)
├── proofs/                           # Lean 4 formal verification (5 modules)
│   ├── OrthoCacheMath/
│   │   ├── ParsevalWHT.lean          # WHT orthogonality + Parseval's identity
│   │   ├── TruncationBound.lean      # Exponential TV bound
│   │   ├── QuantizedTruncation.lean  # IEEE 754 perfect eviction
│   │   ├── CauchySchwarzGate.lean    # Walsh-domain Cauchy-Schwarz bound
│   │   └── GQAMonotonicity.lean      # GQA group safety theorem
│   ├── lakefile.lean                 # Lean 4 build config (Mathlib dep)
│   └── lean-toolchain                # leanprover/lean4:v4.8.0
├── paper/
│   └── orthocache_gpu.tex            # GPU paper (IEEE format)
├── docs/
│   ├── mathematical_framework.md     # Formal math reference
│   ├── technical_report.md           # GPU architecture + benchmarks
│   └── cost_benefit_analysis.md      # Fleet economics + consumer analysis
├── tests/                            # 150 tests (15 test files)
├── benchmarks/
│   ├── profiling.py                  # Latency sweep benchmarks
│   ├── profile_fusion.py             # Fused kernel profiling (single-head)
│   ├── profile_multihead.py          # Multi-head benchmark (hero figure data)
│   ├── generate_figures.py           # Publication-quality dark-theme plots
│   ├── generate_hero_figure.py       # Hero figure generator (multihead data)
│   └── plots/                        # Pre-generated SVG + PNG figures
├── COMMERCIAL_LICENSING.md           # Dual-license terms (Patent Pending)
├── CITATION.cff                      # Machine-readable citation metadata
├── pyproject.toml                    # Build configuration
└── LICENSE                           # AGPL-3.0-only
```

---

## Benchmark Figures

<p align="center">
  <img src="benchmarks/plots/hero_speedup_bars.png" alt="Split-K OrthoCache speedup vs dense attention" width="600" />
</p>

<p align="center"><em>Speedup vs dense attention at 50% eviction rate. OrthoCache breaks even at ~4K tokens and provides 1.28× speedup at 32K — while saving 50% KV-cache memory.</em></p>

<p align="center">
  <img src="benchmarks/plots/fusion_sram_utilization.png" alt="SRAM budget breakdown" width="600" />
</p>

<p align="center"><em>SRAM budget: the fused kernel fits within the 100 KB/SM limit of the RTX 4060, keeping K and W₆₄ resident across both phases.</em></p>

---

## Citation

```bibtex
@software{orthocache_gpu_2026,
  title     = {OrthoCache GPU: Hardware-Native Multi-Band Spectral
               Attention Block Eviction with Split-K Parallelization},
  author    = {Arndt, Justin},
  year      = {2026},
  url       = {https://github.com/j-arndt/orthocache-gpu},
  license   = {AGPL-3.0-only}
}
```

---

## License

**[GNU Affero General Public License v3.0 only (AGPL-3.0-only)](LICENSE)**

Free for academic research, personal projects, and AGPL-compatible open-source use. Network service deployment requires source code disclosure under the same license.

**Commercial use** — including production API endpoints, cloud inference, and SaaS integration — requires a separate enterprise license. See [COMMERCIAL_LICENSING.md](COMMERCIAL_LICENSING.md) for details.

📧 **Commercial licensing:** [justinarndt05@gmail.com](mailto:justinarndt05@gmail.com)

**Patent Pending** — the OrthoCache algorithm is patent pending.
