"""Triton kernels for OrthoCache GPU Edition.

This package will contain fused Triton kernels for block-sparse attention,
FWHT, and spectral energy computation on NVIDIA GPUs.

Planned kernels:
    - fused_block_sparse_attention: Fused gather + QK^T + softmax + V multiply
      with shared memory tiling for block-sparse access patterns.
    - fwht_512_kernel: Triton implementation of the 512-row Fast Walsh-Hadamard
      Transform with warp-level butterfly operations.
    - spectral_band_energy: Fused FWHT + per-band energy accumulation.

Target hardware: NVIDIA H100 (SM 9.0) and B200 (SM 10.0+).
"""
