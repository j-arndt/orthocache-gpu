# OrthoCache GPU — PyTorch/Triton Edition

<p align="center">
  <strong>Hardware-Native Multi-Band Spectral Attention Block Eviction for NVIDIA GPUs</strong>
</p>
<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" /></a>
  <a href="https://pytorch.org/"><img alt="PyTorch 2.5+" src="https://img.shields.io/badge/PyTorch-%E2%89%A52.5-ee4c2c?logo=pytorch&logoColor=white" /></a>
  <a href="https://triton-lang.org/"><img alt="Triton 3.0+" src="https://img.shields.io/badge/Triton-%E2%89%A53.0-7B68EE?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PHRleHQgeD0iMCIgeT0iMTQiIGZvbnQtc2l6ZT0iMTQiPuKWsiA8L3RleHQ+PC9zdmc+" /></a>
  <a href="https://doi.org/10.5281/zenodo.20518370"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.20518370.svg" /></a>
</p>

───────────────────────────────────────────────────────────────────────

## Overview

This is the **GPU/Triton port** of [OrthoCache](https://github.com/j-arndt/orthocache), originally developed and validated on Google TPU v5e. The core algorithm — Multi-Band Sequency Filtering via Walsh–Hadamard Transform with formal TV distance bounds — is identical. The runtime has been rewritten for PyTorch and Triton to target NVIDIA GPUs (H100, B200, A100).

### Why GPU?

The TPU implementation required elaborate workarounds (bucketed compaction, loop indirection, `shard_map` stratified routing) because the MXU systolic array **cannot skip work** — it processes zeroed blocks at full cost. GPUs don't have this limitation:

- **Warp-level divergence** lets threads skip blocks natively
- **Triton `tl.where`** provides actual branch elimination
- **H100 sparse tensor cores** process 2:4 structured sparsity at 2× throughput
- **Native pointer arithmetic** eliminates the "gather tax" entirely

───────────────────────────────────────────────────────────────────────

## Quick Start

```bash
# Clone and install
git clone https://github.com/j-arndt/orthocache-gpu.git && cd orthocache-gpu
pip install -e ".[dev]"

# Run the test suite
pytest

# Run benchmarks (requires CUDA GPU)
python benchmarks/profiling.py
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.5.0 (with CUDA support)
- Triton ≥ 3.0.0
- NVIDIA GPU (Ampere or newer recommended)
- CUDA Toolkit ≥ 12.0

───────────────────────────────────────────────────────────────────────

## Repository Structure

```
orthocache-gpu/
├── src/
│   └── orthocache_gpu/
│       ├── __init__.py              # Public API surface
│       ├── fwht.py                  # Fast Walsh–Hadamard Transform (512-tile)
│       ├── spectral_energy.py       # Multi-band spectral decomposition & ζ filter
│       ├── reference.py             # NumPy reference implementations
│       ├── compaction.py            # Stream compaction (sort + gather)
│       ├── bucketed_attention.py    # Bucketed dense attention on compacted operands
│       ├── lean_attention.py        # Lean attention (pure PyTorch, no Triton)
│       ├── adaptive_attention.py    # Adaptive dispatcher (path A/B selection)
│       ├── dynamic_attention.py     # Dynamic block-sparse attention
│       ├── pipeline.py              # End-to-end OrthoCache forward pass
│       ├── cuda_bridge.py           # CUDA reference implementations
│       ├── bandwidth_model.py       # NVLink/ICI bandwidth model (H100, B200, TPU)
│       └── triton_kernels/
│           ├── __init__.py
│           ├── sparse_attention.py  # Triton block-sparse attention kernel
│           └── indirect_attention.py # Triton indirect indexing kernel
├── tests/                           # PyTest test suite
├── benchmarks/                      # GPU benchmarks
├── pyproject.toml                   # Build configuration
└── README.md                        # ← You are here
```

───────────────────────────────────────────────────────────────────────

## Relationship to TPU Version

| Aspect | TPU (`orthocache`) | GPU (`orthocache-gpu`) |
|:-------|:-------------------|:----------------------|
| Algorithm | Identical | Identical |
| Formal proofs | Lean 4 (shared) | Lean 4 (shared) |
| Kernel language | Pallas | Triton |
| Collective comms | ICI AllGather | NCCL (planned) |
| Compilation | XLA/HLO | torch.compile |
| Framework | JAX | PyTorch |

The mathematical guarantees (Parseval identity, exponential TV bound) apply to both implementations — they are properties of the algorithm, not the hardware.

───────────────────────────────────────────────────────────────────────

## Citation

```bibtex
@software{orthocache2026,
  title     = {OrthoCache: Hardware-Native Multi-Band Spectral Attention
               Block Eviction on TPUs},
  author    = {Arndt, Justin},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20518370},
  url       = {https://doi.org/10.5281/zenodo.20518370}
}
```

───────────────────────────────────────────────────────────────────────

## License

**[PolyForm Noncommercial License 1.0.0](../LICENSE)**

📧 **Commercial licensing:** [justinarndt05@gmail.com](mailto:justinarndt05@gmail.com)
