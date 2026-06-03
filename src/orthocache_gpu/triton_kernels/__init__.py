"""Triton kernels for OrthoCache GPU Edition.

This package contains fused Triton kernels for block-sparse attention,
indirect-indexing attention, FWHT, and spectral energy computation on
NVIDIA GPUs.

Implemented kernels:
    - sparse_attention: Block-sparse attention with boolean mask gating
      and FlashAttention-style online softmax.
    - indirect_attention: Indirect-indexing attention using an explicit
      index list for scatter-gather KV access (zero data copy).

Planned kernels:
    - fwht_512_kernel: Triton implementation of the 512-row Fast Walsh-Hadamard
      Transform with warp-level butterfly operations.
    - spectral_band_energy: Fused FWHT + per-band energy accumulation.

Target hardware: NVIDIA H100 (SM 9.0) and B200 (SM 10.0+).
"""

from orthocache_gpu.triton_kernels.sparse_attention import (
    triton_block_sparse_attention,
)
from orthocache_gpu.triton_kernels.indirect_attention import (
    triton_indirect_attention,
)

__all__ = [
    "triton_block_sparse_attention",
    "triton_indirect_attention",
]
