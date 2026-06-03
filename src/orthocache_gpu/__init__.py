"""OrthoCache GPU — PyTorch/Triton Edition.

Hardware-Native Multi-Band Spectral Attention Block Eviction for NVIDIA GPUs.

This package provides the GPU port of OrthoCache, originally developed for
Google TPU v5e using JAX/Pallas. The core algorithm (FWHT, spectral decay
ratio, TV distance bound) is identical; the runtime targets PyTorch/Triton/CUDA.
"""

__version__ = "0.1.0"

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
from orthocache_gpu.pipeline import orthocache_forward

# ── Infrastructure: Bandwidth Model ─────────────────────────────────
from orthocache_gpu.bandwidth_model import (
    ici_bytes_per_step,
    ici_bandwidth_table,
    model_configs,
)
