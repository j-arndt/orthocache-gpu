"""OrthoCache GPU — PyTorch/Triton Edition.

Hardware-Native Multi-Band Spectral Attention Block Eviction for NVIDIA GPUs.

This package provides the GPU port of OrthoCache, originally developed for
Google TPU v5e using JAX/Pallas. The core algorithm (FWHT, spectral decay
ratio, TV distance bound) is identical; the runtime targets PyTorch/Triton/CUDA.
"""

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # FWHT
    "fwht_512",
    # Spectral Energy
    "compute_block_energy",
    "compute_spectral_bands",
    "compute_spectral_decay_ratio",
    "compute_query_aware_bounds",
    "compute_query_aware_mask",
    "compute_multiband_mask",
    "generate_threshold_mask",
    # Lean Attention
    "lean_bucketed_attention",
    # Stream Compaction
    "stream_compact",
    "stream_decompact",
    "compact_and_attend",
    # Adaptive Attention
    "orthocache_attention",
    "orthocache_attention_batched",
    # Pipeline
    "orthocache_forward",
    "CROSSOVER_SEQ_LEN",
    # Bandwidth Model
    "ici_bytes_per_step",
    "ici_bandwidth_table",
    "model_configs",
    # Perfect Eviction
    "classify_eviction",
    "perfect_eviction_check",
    "compute_block_beta",
    "EvictionRegime",
    "EvictionMetadata",
    # Triton Kernels
    "triton_fwht_eviction",
    "generate_walsh_matrix",
    "fused_orthocache_attention",
    "fused_orthocache_attention_v2",
]

# ── Core: Walsh–Hadamard Transform ──────────────────────────────────
from orthocache_gpu.fwht import fwht_512

# ── Core: Spectral Energy & Masking ─────────────────────────────────
from orthocache_gpu.spectral_energy import (
    compute_block_energy,
    compute_spectral_bands,
    compute_spectral_decay_ratio,
    compute_query_aware_bounds,
    compute_query_aware_mask,
    compute_multiband_mask,
    generate_threshold_mask,
)

# ── Attention: Lean (pure PyTorch, no Triton) ───────────────────────
from orthocache_gpu.lean_attention import lean_bucketed_attention

# ── Attention: Stream Compaction ────────────────────────────────────
from orthocache_gpu.compaction import (
    stream_compact,
    stream_decompact,
    compact_and_attend,
)

# ── Attention: Adaptive Dispatcher ──────────────────────────────────
from orthocache_gpu.adaptive_attention import (
    orthocache_attention,
    orthocache_attention_batched,
)

# ── Pipeline: End-to-End Forward Pass ───────────────────────────────
from orthocache_gpu.pipeline import orthocache_forward, CROSSOVER_SEQ_LEN

# ── Infrastructure: Bandwidth Model ─────────────────────────────────
from orthocache_gpu.bandwidth_model import (
    ici_bytes_per_step,
    ici_bandwidth_table,
    model_configs,
)

# ── Eviction: Perfect Eviction Governor ─────────────────────────────
from orthocache_gpu.perfect_eviction import (
    classify_eviction,
    perfect_eviction_check,
    compute_block_beta,
    EvictionRegime,
    EvictionMetadata,
    FLOAT32_UNDERFLOW_THRESHOLD,
    BFLOAT16_UNDERFLOW_THRESHOLD,
)

# ── Triton: Fused God Kernel (Phase 7) ──────────────────────────────
from orthocache_gpu.triton_kernels.fwht_fused_prototype import (
    triton_fwht_eviction,
    generate_walsh_matrix,
)
from orthocache_gpu.triton_kernels.fused_eviction import (
    fused_orthocache_attention,
    fused_orthocache_attention_v2,
)
