"""Triton kernels for OrthoCache GPU Edition.

This package contains fused Triton kernels for block-sparse attention,
indirect-indexing attention, FWHT, spectral energy computation, and the
fused "God Kernel" on NVIDIA GPUs.

Implemented kernels:
    - sparse_attention: Block-sparse attention with boolean mask gating
      and FlashAttention-style online softmax.
    - indirect_attention: Indirect-indexing attention using an explicit
      index list for scatter-gather KV access (zero data copy).
    - fwht_fused_prototype: 64-tile Fast Walsh–Hadamard Transform with
      spectral eviction (ζ filter) for RTX 4060 (SM 8.9).
    - fused_eviction: God Kernel — FWHT + ζ + predicated attention in a
      single kernel launch. Eliminates redundant K reloads from HBM.

Target hardware: NVIDIA RTX 4060 (SM 8.9), H100 (SM 9.0), B200 (SM 10.0+).
"""

from orthocache_gpu.triton_kernels.sparse_attention import (
    triton_block_sparse_attention,
)
from orthocache_gpu.triton_kernels.indirect_attention import (
    triton_indirect_attention,
)
from orthocache_gpu.triton_kernels.fwht_fused_prototype import (
    triton_fwht_eviction,
    generate_walsh_matrix,
)
from orthocache_gpu.triton_kernels.fused_eviction import (
    fused_orthocache_attention,
    fused_orthocache_attention_v2,
)

__all__ = [
    "triton_block_sparse_attention",
    "triton_indirect_attention",
    "triton_fwht_eviction",
    "generate_walsh_matrix",
    "fused_orthocache_attention",
    "fused_orthocache_attention_v2",
]
