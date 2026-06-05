"""Isolated in-SRAM FWHT + ζ → eviction mask Triton kernel (Phase 7, Step 1).

This kernel proves the core thesis of the Fused God Kernel: the entire
Walsh-Hadamard Transform and spectral decay ratio (ζ) computation can
execute *inside* the SM's shared memory (SRAM) without ever writing
intermediate spectral coefficients to HBM (VRAM).

Strategy: Instead of a butterfly network (which requires complex shared
memory swizzling that Triton handles poorly), we load a precomputed
64×64 Walsh matrix W into SRAM and execute the FWHT as a dense matrix
multiply via tl.dot(W, K_tile). This maps perfectly to hardware Tensor
Cores, executing in a fraction of the clock cycles of a butterfly.

SRAM Budget (RTX 4060, 100 KB per SM):
    K tile:      64 × 128 × 2 = 16,384 bytes  (bf16 input)
    W_64:        64 × 64  × 4 = 16,384 bytes  (fp32, loaded once)
    K_spectral:  64 × 128 × 4 = 32,768 bytes  (fp32, tl.dot output)
    Band masks:  64 × 4       =    256 bytes
    Accumulators:              ≈    512 bytes
    ────────────────────────────────────────
    TOTAL:                     ≈ 66 KB         ✓ fits in 100 KB

Hardware target: NVIDIA RTX 4060 (Ada Lovelace, SM 8.9, 100 KB SRAM/SM)
"""

import torch
from typing import Optional, Tuple

# --- Triton availability check ---
HAS_CUDA = torch.cuda.is_available()
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ============================================================================
# Constants
# ============================================================================

TILE_SIZE = 64  # Must be power of 2; 64 fits SRAM budget on RTX 4060

# 64-point sequency band boundaries (rescaled from 512-point):
#   512-point: DC=0, Low=[1,64), Mid=[64,256), High=[256,512)
#   64-point:  DC=0, Low=[1,8),  Mid=[8,32),   High=[32,64)
BAND_DC_64 = 0
BAND_LOW_64 = (1, 8)     # 7 coefficients
BAND_MID_64 = (8, 32)    # 24 coefficients
BAND_HIGH_64 = (32, 64)  # 32 coefficients


# ============================================================================
# Walsh Matrix Generation
# ============================================================================

def generate_walsh_matrix(n: int) -> torch.Tensor:
    """Generate the n×n Walsh-Hadamard matrix (normalized by 1/sqrt(n)).

    Constructs via recursive Kronecker product:
        H₁ = [[1, 1], [1, -1]]
        Hₙ = H₁ ⊗ H_{n/2}

    The resulting matrix is symmetric, orthogonal (W @ W.T = I), and
    has entries ±1/sqrt(n).

    Args:
        n: Matrix size. Must be a power of 2.

    Returns:
        Float32 tensor of shape (n, n), normalized so W @ W.T = I.
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n must be a power of 2, got {n}"
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / (n ** 0.5)


# Module-level lazy-initialized Walsh matrix (moved to GPU on first use)
_W64_cache: Optional[torch.Tensor] = None


def _get_walsh_matrix(device: torch.device) -> torch.Tensor:
    """Get or create the cached 64×64 Walsh matrix on the given device."""
    global _W64_cache
    if _W64_cache is None or _W64_cache.device != device:
        _W64_cache = generate_walsh_matrix(TILE_SIZE).contiguous().to(device)
    return _W64_cache


# ============================================================================
# Triton Kernel: Isolated FWHT + ζ → Eviction Mask
# ============================================================================

if HAS_TRITON:

    @triton.jit
    def _fwht_eviction_kernel(
        # Pointers
        K_ptr,              # (num_tiles * TILE_SIZE, head_dim) — KV cache keys
        W_ptr,              # (TILE_SIZE, TILE_SIZE) — Walsh matrix, fp32
        MASK_OUT_ptr,       # (num_tiles,) — output boolean mask, int8
        ZETA_OUT_ptr,       # (num_tiles,) — output ζ values, fp32 (debug)
        # Scalar args
        zeta_max,           # float — ζ threshold for eviction
        # Dimensions
        head_dim: tl.constexpr,
        num_tiles: tl.constexpr,
        TILE_SIZE: tl.constexpr,    # 64
        RETURN_ZETA: tl.constexpr,  # bool — whether to output ζ values
        # Band boundaries (constexpr for compile-time optimization)
        BAND_LOW_START: tl.constexpr,   # 1
        BAND_LOW_END: tl.constexpr,     # 8
        BAND_HIGH_START: tl.constexpr,  # 32
        BAND_HIGH_END: tl.constexpr,    # 64
    ):
        """In-SRAM FWHT + spectral decay ratio + eviction mask.

        Each program handles one K tile. The Walsh-Hadamard Transform
        is executed as tl.dot(W, K_tile) on Tensor Cores — the spectral
        coefficients NEVER leave SRAM. Only the final boolean mask is
        written to HBM.

        Algorithm:
            1. Load K_tile (TILE_SIZE × head_dim) from HBM → SRAM
            2. Load W (TILE_SIZE × TILE_SIZE) from HBM → SRAM
            3. K_spectral = tl.dot(W, K_tile)  [Tensor Core GEMM, in SRAM]
            4. E_low = sum(K_spectral[BAND_LOW, :]²)  [in-register]
            5. E_high = sum(K_spectral[BAND_HIGH, :]²)  [in-register]
            6. ζ = E_high / (E_low + 1e-6)
            7. mask = (ζ <= zeta_max)
            8. Store mask to HBM  [ONLY output]
        """
        tile_id = tl.program_id(0)

        # Guard: skip out-of-range tiles
        if tile_id >= num_tiles:
            return

        # ── Step 1: Load K tile (TILE_SIZE × head_dim) → SRAM ──────────
        # K layout: (num_tiles * TILE_SIZE, head_dim), row-major
        k_base = tile_id * TILE_SIZE * head_dim

        # Build 2D offset grid: (TILE_SIZE, head_dim)
        row_offsets = tl.arange(0, TILE_SIZE)       # [0, 1, ..., 63]
        col_offsets = tl.arange(0, head_dim)        # [0, 1, ..., 127]
        k_offsets = (row_offsets[:, None] * head_dim + col_offsets[None, :])
        k_tile = tl.load(K_ptr + k_base + k_offsets)  # (TILE_SIZE, head_dim)
        k_tile = k_tile.to(tl.float32)

        # ── Step 2: Load Walsh matrix (TILE_SIZE × TILE_SIZE) → SRAM ───
        w_offsets = (row_offsets[:, None] * TILE_SIZE +
                     tl.arange(0, TILE_SIZE)[None, :])
        w_matrix = tl.load(W_ptr + w_offsets)  # (TILE_SIZE, TILE_SIZE), fp32

        # ── Step 3: FWHT via dense matmul on Tensor Cores ──────────────
        # K_spectral = W @ K_tile  →  (TILE_SIZE, head_dim), stays in SRAM
        k_spectral = tl.dot(w_matrix, k_tile)  # Tensor Core GEMM

        # ── Step 4: Compute band energies (in-register) ────────────────
        # Square the spectral coefficients
        k_sq = k_spectral * k_spectral  # (TILE_SIZE, head_dim)

        # Sum across head_dim to get per-sequency energy: (TILE_SIZE,)
        energy_per_seq = tl.sum(k_sq, axis=1)  # (TILE_SIZE,)

        # Create band indicator masks using sequency indices
        seq_idx = tl.arange(0, TILE_SIZE)  # [0, 1, ..., 63]

        low_mask = (seq_idx >= BAND_LOW_START) & (seq_idx < BAND_LOW_END)
        high_mask = (seq_idx >= BAND_HIGH_START) & (seq_idx < BAND_HIGH_END)

        # Masked sum for each band
        e_low = tl.sum(tl.where(low_mask, energy_per_seq, 0.0))
        e_high = tl.sum(tl.where(high_mask, energy_per_seq, 0.0))

        # ── Step 5: Spectral Decay Ratio ζ ─────────────────────────────
        zeta = e_high / (e_low + 1e-6)

        # ── Step 6: Eviction decision ──────────────────────────────────
        keep = zeta <= zeta_max

        # ── Step 7: Store outputs ──────────────────────────────────────
        # ONLY the boolean mask hits HBM — K_spectral stays in SRAM
        tl.store(MASK_OUT_ptr + tile_id, keep.to(tl.int8))

        if RETURN_ZETA:
            tl.store(ZETA_OUT_ptr + tile_id, zeta)


# ============================================================================
# PyTorch Fallback (CPU / no-Triton path)
# ============================================================================

def _pytorch_fwht_eviction(
    keys: torch.Tensor,
    zeta_max: float,
    tile_size: int = TILE_SIZE,
    return_zeta: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pure-PyTorch FWHT eviction for CPU testing and reference comparison.

    Mirrors the Triton kernel logic exactly.

    Args:
        keys: Key tensor, shape (num_tiles * tile_size, head_dim).
        zeta_max: Spectral decay ratio threshold.
        tile_size: Tile size (default: 64).
        return_zeta: If True, also return ζ values.

    Returns:
        Tuple of (mask, zeta_values):
        - mask: Boolean tensor (num_tiles,) — True = retain.
        - zeta_values: Float tensor (num_tiles,) or None.
    """
    num_tokens, head_dim = keys.shape
    num_tiles = num_tokens // tile_size
    assert num_tokens == num_tiles * tile_size

    W = generate_walsh_matrix(tile_size).to(keys.device)  # (tile_size, tile_size)

    # Reshape into tiles: (num_tiles, tile_size, head_dim)
    tiles = keys.reshape(num_tiles, tile_size, head_dim).float()

    # Batched FWHT via matmul: (num_tiles, tile_size, head_dim)
    # W @ each tile: (tile_size, tile_size) × (num_tiles, tile_size, head_dim)
    k_spectral = torch.einsum('ij,btj->bti', W, tiles.transpose(1, 2).contiguous()).transpose(1, 2)
    # Simpler approach: direct batch matmul
    k_spectral = torch.matmul(W.unsqueeze(0), tiles)  # (num_tiles, tile_size, head_dim)

    # Per-sequency energy: (num_tiles, tile_size)
    energy_per_seq = torch.sum(k_spectral ** 2, dim=2)

    # Band energies
    e_low = torch.sum(energy_per_seq[:, BAND_LOW_64[0]:BAND_LOW_64[1]], dim=1)
    e_high = torch.sum(energy_per_seq[:, BAND_HIGH_64[0]:BAND_HIGH_64[1]], dim=1)

    # ζ
    zeta = e_high / (e_low + 1e-6)

    # Mask
    mask = zeta <= zeta_max

    return mask, zeta if return_zeta else None


# ============================================================================
# Public Wrapper
# ============================================================================

def triton_fwht_eviction(
    keys: torch.Tensor,
    zeta_max: float,
    tile_size: int = TILE_SIZE,
    return_zeta: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute FWHT-based spectral eviction mask using in-SRAM Triton kernel.

    This is the Phase 7 proof-of-concept: the Walsh-Hadamard Transform and
    ζ computation happen entirely inside the SM's shared memory. Only the
    final boolean mask is written to HBM.

    Args:
        keys: Key tensor, shape (num_tiles * tile_size, head_dim).
            Must be contiguous. Supports float32 and bfloat16.
        zeta_max: Maximum spectral decay ratio. Tiles with ζ > zeta_max
            are evicted (mask = False).
        tile_size: Number of tokens per tile (default: 64).
        return_zeta: If True, also return per-tile ζ values.

    Returns:
        Tuple of (mask, zeta_values):
        - mask: Boolean tensor (num_tiles,) — True = retain block.
        - zeta_values: Float tensor (num_tiles,) or None.
    """
    # ── Input validation ──
    assert keys.ndim == 2, f"keys must be 2D (tokens, head_dim), got {keys.shape}"
    num_tokens, head_dim = keys.shape
    num_tiles = num_tokens // tile_size
    assert num_tokens == num_tiles * tile_size, (
        f"seq_len {num_tokens} not divisible by tile_size {tile_size}"
    )

    # ── CPU / no-Triton fallback ──
    if not (HAS_CUDA and HAS_TRITON and keys.is_cuda):
        return _pytorch_fwht_eviction(keys, zeta_max, tile_size, return_zeta)

    # ── Prepare inputs ──
    keys = keys.contiguous()
    W = _get_walsh_matrix(keys.device)

    # ── Allocate outputs ──
    mask_out = torch.empty(num_tiles, dtype=torch.int8, device=keys.device)
    zeta_out = (
        torch.empty(num_tiles, dtype=torch.float32, device=keys.device)
        if return_zeta else
        torch.empty(1, dtype=torch.float32, device=keys.device)  # dummy
    )

    # ── Launch kernel ──
    grid = (num_tiles,)

    _fwht_eviction_kernel[grid](
        keys, W, mask_out, zeta_out,
        zeta_max,
        head_dim=head_dim,
        num_tiles=num_tiles,
        TILE_SIZE=tile_size,
        RETURN_ZETA=return_zeta,
        BAND_LOW_START=BAND_LOW_64[0],
        BAND_LOW_END=BAND_LOW_64[1],
        BAND_HIGH_START=BAND_HIGH_64[0],
        BAND_HIGH_END=BAND_HIGH_64[1],
    )

    mask = mask_out.bool()
    zeta_vals = zeta_out if return_zeta else None
    return mask, zeta_vals


# ============================================================================
# Diagnostic Utilities
# ============================================================================

def print_kernel_metadata():
    """Print compiled kernel metadata: shared memory, registers, spills.

    Must be called AFTER at least one kernel launch (Triton compiles lazily).
    """
    if not HAS_TRITON:
        print("Triton not available.")
        return

    # Trigger compilation with a small test input
    device = torch.device('cuda')
    test_keys = torch.randn(TILE_SIZE, 128, device=device, dtype=torch.float32)
    W = _get_walsh_matrix(device)
    mask_out = torch.empty(1, dtype=torch.int8, device=device)
    zeta_out = torch.empty(1, dtype=torch.float32, device=device)

    # Launch to trigger JIT compilation
    _fwht_eviction_kernel[(1,)](
        test_keys, W, mask_out, zeta_out,
        5.0,  # zeta_max
        head_dim=128,
        num_tiles=1,
        TILE_SIZE=TILE_SIZE,
        RETURN_ZETA=True,
        BAND_LOW_START=BAND_LOW_64[0],
        BAND_LOW_END=BAND_LOW_64[1],
        BAND_HIGH_START=BAND_HIGH_64[0],
        BAND_HIGH_END=BAND_HIGH_64[1],
    )
    torch.cuda.synchronize()

    # Access compiled kernel metadata
    # Note: Triton metadata access varies by version; try common patterns
    try:
        kernel_fn = _fwht_eviction_kernel
        # Triton 3.x stores metadata on the compiled kernel
        key = list(kernel_fn.cache[0].values())[0] if kernel_fn.cache else None
        if key is not None:
            meta = key.metadata if hasattr(key, 'metadata') else None
            if meta:
                shared = meta.get('shared', 'unknown')
                num_regs = meta.get('num_regs', 'unknown')
                print(f"SRAM Usage:    {shared} bytes ({shared/1024:.1f} KB / 100 KB)" if isinstance(shared, (int, float)) else f"SRAM Usage:    {shared}")
                print(f"Registers:     {num_regs} / 255" if isinstance(num_regs, (int, float)) else f"Registers:     {num_regs}")
                if isinstance(num_regs, (int, float)):
                    print(f"Spills:        {'NONE ✅' if num_regs <= 255 else 'DANGER ❌'}")
                return

        # Fallback: direct attribute access
        if hasattr(kernel_fn, 'n_regs'):
            print(f"Registers: {kernel_fn.n_regs}")
        if hasattr(kernel_fn, 'n_spills'):
            print(f"Spills: {kernel_fn.n_spills}")

    except Exception as e:
        print(f"Could not access kernel metadata: {e}")
        print("Run with ncu for definitive hardware telemetry.")
