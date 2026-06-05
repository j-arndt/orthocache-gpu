"""Fused FWHT + ζ Eviction + FlashAttention Triton Kernel (Phase 7, Step 3).

THE GOD KERNEL: Fuses the entire OrthoCache pipeline into one kernel launch:
    1. Load K tile from HBM → SRAM
    2. FWHT via Walsh matrix tl.dot() on Tensor Cores (stays in SRAM)
    3. Compute ζ (spectral decay ratio) in-register
    4. If ζ > threshold: skip this tile entirely (true branch elimination)
    5. If ζ ≤ threshold: load V tile, compute Q×K^T, online softmax accumulate

This was PHYSICALLY IMPOSSIBLE on TPU due to Pallas scratchpad and control
flow limitations. The GPU's explicit branch elimination via `if/continue`
in Triton gives us a capability the TPU architecture cannot match.

Key invariant: Intermediate tensors (K_spectral, ζ, band energies) NEVER
leave the SM's L1 cache / shared memory. Only the final attention output O
is written to HBM.

SRAM Budget (RTX 4060, 100 KB per SM):
    Phase A (eviction):
        K_tile:      64 × 128 × 2 = 16 KB  (bf16 → fp32 cast in-register)
        W_64:        64 × 64  × 4 = 16 KB
        K_spectral:  Intermediate of tl.dot, reuses K_tile SRAM after cast
        Band energies: scalars, in-register
        ≈ 32-48 KB

    Phase B (attention, only if retained):
        K_tile:      reused from Phase A
        V_tile:      64 × 128 × 2 = 16 KB
        Q row/tile:   1 × 128 × 4 =  0.5 KB
        Logits:       1 × 64  × 4 =  0.25 KB
        Accumulators: 128 × 4     =  0.5 KB
        ≈ 33 KB

    Peak (overlap between phases): ≈ 65-81 KB ✓

Hardware target: NVIDIA RTX 4060 (Ada Lovelace, SM 8.9, 100 KB SRAM/SM)
"""

import torch
from typing import Optional, Tuple, Dict, Any

from orthocache_gpu.triton_kernels.fwht_fused_prototype import (
    generate_walsh_matrix,
    TILE_SIZE,
    BAND_LOW_64,
    BAND_HIGH_64,
)

# --- Triton availability check ---
HAS_CUDA = torch.cuda.is_available()
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# Module-level Walsh matrix cache
_W64_cache: Optional[torch.Tensor] = None


def _get_walsh_matrix(device: torch.device) -> torch.Tensor:
    """Get or create the cached 64×64 Walsh matrix on the given device."""
    global _W64_cache
    if _W64_cache is None or _W64_cache.device != device:
        _W64_cache = generate_walsh_matrix(TILE_SIZE).contiguous().to(device)
    return _W64_cache


# ============================================================================
# Triton Kernel: Fused FWHT + ζ + FlashAttention
# ============================================================================

if HAS_TRITON:

    @triton.jit
    def _fused_orthocache_kernel(
        # ── Tensor pointers ──
        Q_ptr,              # (1, head_dim) — single query (decode mode)
        K_ptr,              # (num_k_tiles * TILE_SIZE, head_dim) — key cache
        V_ptr,              # (num_k_tiles * TILE_SIZE, head_dim) — value cache
        W_ptr,              # (TILE_SIZE, TILE_SIZE) — Walsh matrix, fp32
        O_ptr,              # (1, head_dim) — output
        MASK_DEBUG_ptr,     # (num_k_tiles,) — debug: eviction mask, int8
        # ── Scalar args ──
        zeta_max,           # float — ζ threshold
        # ── Dimensions (constexpr) ──
        head_dim: tl.constexpr,
        num_k_tiles: tl.constexpr,
        TILE_SIZE: tl.constexpr,       # 64
        RETURN_MASK: tl.constexpr,     # bool — write debug mask
        # ── Band boundaries (constexpr) ──
        BAND_LOW_START: tl.constexpr,   # 1
        BAND_LOW_END: tl.constexpr,     # 8
        BAND_HIGH_START: tl.constexpr,  # 32
        BAND_HIGH_END: tl.constexpr,    # 64
    ):
        """Fused in-SRAM FWHT spectral eviction + FlashAttention.

        Single-query decode mode: one program handles the entire KV cache
        for one head, iterating over K tiles with fused eviction.

        The key insight: K_tile loaded for spectral analysis in Phase A is
        REUSED for the Q×K^T dot product in Phase B. Zero redundant HBM loads.
        V_tile is only loaded if the block passes eviction — evicted blocks
        save both the V load AND the Q×K^T compute.
        """
        pid = tl.program_id(0)  # head index (for multi-head extension)

        # ── Load query vector (1, head_dim) → fp32 ────────────────────
        q_offsets = tl.arange(0, head_dim)
        q = tl.load(Q_ptr + q_offsets)
        q = q.to(tl.float32)
        # head_dim is constexpr, so compute scale at compile time
        inv_sqrt_d: tl.constexpr = 1.0 / (head_dim ** 0.5)

        # ── Load Walsh matrix (TILE_SIZE × TILE_SIZE) → SRAM ─────────
        w_row = tl.arange(0, TILE_SIZE)
        w_col = tl.arange(0, TILE_SIZE)
        w_offsets = w_row[:, None] * TILE_SIZE + w_col[None, :]
        w_matrix = tl.load(W_ptr + w_offsets)  # (TILE_SIZE, TILE_SIZE), fp32

        # ── Online softmax accumulators ───────────────────────────────
        m_i = tl.full([1], value=-1e9, dtype=tl.float32)   # running max
        l_i = tl.full([1], value=0.0, dtype=tl.float32)    # running exp-sum
        acc = tl.zeros([head_dim], dtype=tl.float32)        # running output

        # ── Sequency index for band masking ───────────────────────────
        seq_idx = tl.arange(0, TILE_SIZE)
        low_mask = (seq_idx >= BAND_LOW_START) & (seq_idx < BAND_LOW_END)
        high_mask = (seq_idx >= BAND_HIGH_START) & (seq_idx < BAND_HIGH_END)

        # ── Iterate over K tiles ──────────────────────────────────────
        for tile_id in range(num_k_tiles):
            kv_base = tile_id * TILE_SIZE * head_dim

            # ── PHASE A: In-SRAM spectral eviction ────────────────────
            # Load K tile: (TILE_SIZE, head_dim) from HBM → SRAM
            row_offsets = tl.arange(0, TILE_SIZE)
            col_offsets = tl.arange(0, head_dim)
            k_offsets = kv_base + row_offsets[:, None] * head_dim + col_offsets[None, :]
            k_tile = tl.load(K_ptr + k_offsets)
            k_tile = k_tile.to(tl.float32)

            # FWHT via Tensor Core matmul: K_spectral = W @ K_tile
            # This stays entirely in SRAM — no HBM write
            k_spectral = tl.dot(w_matrix, k_tile)  # (TILE_SIZE, head_dim)

            # Per-sequency energy: sum(coeff²) across head_dim
            k_sq = k_spectral * k_spectral
            energy_per_seq = tl.sum(k_sq, axis=1)  # (TILE_SIZE,)

            # Band energies via masked sum
            e_low = tl.sum(tl.where(low_mask, energy_per_seq, 0.0))
            e_high = tl.sum(tl.where(high_mask, energy_per_seq, 0.0))

            # Spectral decay ratio
            zeta = e_high / (e_low + 1e-6)

            # ── Eviction decision ─────────────────────────────────────
            keep = zeta <= zeta_max

            if RETURN_MASK:
                tl.store(MASK_DEBUG_ptr + tile_id, keep.to(tl.int8))

            if keep:
                # ── PHASE B: FlashAttention (retained tiles only) ─────

                # Q × K^T: (1, head_dim) × (head_dim, TILE_SIZE) → (1, TILE_SIZE)
                # We reuse k_tile from Phase A — zero redundant HBM loads!
                # Compute logits: q · each row of k_tile
                # logits[j] = sum(q * k_tile[j, :]) * scale
                # Use: (TILE_SIZE, head_dim) × (head_dim,) → (TILE_SIZE,)
                logits = tl.sum(k_tile * q[None, :], axis=1) * inv_sqrt_d  # (TILE_SIZE,)

                # Online softmax: update running max + rescale
                tile_max = tl.max(logits)
                new_max = tl.maximum(m_i, tile_max)

                # Rescale old accumulators
                alpha = tl.exp(m_i - new_max)
                # Attention weights for this tile
                p = tl.exp(logits - new_max)  # (TILE_SIZE,)
                p_sum = tl.sum(p)

                # Update running sum
                l_i = l_i * alpha + p_sum

                # Load V tile (ONLY if retained — evicted tiles skip this!)
                v_tile = tl.load(V_ptr + k_offsets)
                v_tile = v_tile.to(tl.float32)  # (TILE_SIZE, head_dim)

                # Weighted V accumulation: acc += p @ V_tile
                # p: (TILE_SIZE,), V: (TILE_SIZE, head_dim) → (head_dim,)
                weighted_v = tl.sum(p[:, None] * v_tile, axis=0)  # (head_dim,)

                acc = acc * alpha + weighted_v
                m_i = new_max

        # ── Normalize and store ───────────────────────────────────────
        safe_l = tl.maximum(l_i, 1e-9)
        out = acc / safe_l

        tl.store(O_ptr + q_offsets, out, mask=q_offsets < head_dim)


# ============================================================================
# PyTorch Fallback (CPU / no-Triton)
# ============================================================================

def _pytorch_fused_orthocache_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    zeta_max: float,
    tile_size: int = TILE_SIZE,
    return_mask: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Pure-PyTorch fused FWHT+ζ+attention for CPU testing and reference.

    Args:
        q: Query, shape (1, head_dim).
        keys: Keys, shape (num_tiles * tile_size, head_dim).
        values: Values, shape (num_tiles * tile_size, head_dim).
        zeta_max: ζ threshold for eviction.
        tile_size: Tokens per tile (default: 64).
        return_mask: If True, include eviction mask in metadata.

    Returns:
        Tuple of (output, metadata).
    """
    head_dim = q.shape[-1]
    num_tokens = keys.shape[0]
    num_tiles = num_tokens // tile_size
    scale = 1.0 / (head_dim ** 0.5)

    W = generate_walsh_matrix(tile_size).to(keys.device)

    # Accumulators (online softmax)
    q_f32 = q.float()
    m_i = torch.full((1,), -1e9, dtype=torch.float32, device=q.device)
    l_i = torch.zeros((1,), dtype=torch.float32, device=q.device)
    acc = torch.zeros((1, head_dim), dtype=torch.float32, device=q.device)

    eviction_mask = torch.zeros(num_tiles, dtype=torch.bool, device=q.device)
    tiles_retained = 0
    tiles_evicted = 0

    for t in range(num_tiles):
        start = t * tile_size
        end = start + tile_size

        k_tile = keys[start:end].float()  # (tile_size, head_dim)

        # ── PHASE A: FWHT + ζ ──
        k_spectral = W @ k_tile  # (tile_size, head_dim)
        energy_per_seq = torch.sum(k_spectral ** 2, dim=1)  # (tile_size,)

        e_low = torch.sum(energy_per_seq[BAND_LOW_64[0]:BAND_LOW_64[1]])
        e_high = torch.sum(energy_per_seq[BAND_HIGH_64[0]:BAND_HIGH_64[1]])
        zeta = (e_high / (e_low + 1e-6)).item()

        if zeta > zeta_max:
            tiles_evicted += 1
            continue

        # ── PHASE B: Attention ──
        eviction_mask[t] = True
        tiles_retained += 1

        v_tile = values[start:end].float()  # (tile_size, head_dim)

        # logits: (1, tile_size)
        logits = (q_f32 @ k_tile.T) * scale

        # Online softmax
        local_max = logits.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(m_i.unsqueeze(0), local_max)

        exp_logits = torch.exp(logits - new_max)
        sum_exp = exp_logits.sum(dim=-1, keepdim=True)

        alpha = torch.exp(m_i.unsqueeze(0) - new_max)

        l_i = (l_i.unsqueeze(0) * alpha + sum_exp).squeeze(0)
        acc = acc * alpha + exp_logits @ v_tile
        m_i = new_max.squeeze(0)

    # Normalize
    out = acc / torch.clamp(l_i.unsqueeze(-1), min=1e-9)

    metadata = {
        'tiles_retained': tiles_retained,
        'tiles_evicted': tiles_evicted,
        'eviction_rate': tiles_evicted / max(1, num_tiles),
    }
    if return_mask:
        metadata['eviction_mask'] = eviction_mask

    return out, metadata


# ============================================================================
# Public Wrapper
# ============================================================================

def fused_orthocache_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    zeta_max: float,
    tile_size: int = TILE_SIZE,
    return_mask: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Fused FWHT+ζ+attention: the OrthoCache God Kernel.

    Executes the entire OrthoCache pipeline in a single Triton kernel launch:
    FWHT → ζ → eviction → sparse attention. Intermediate spectral coefficients
    never leave the SM's shared memory (SRAM).

    This was physically impossible on TPU due to Pallas scratchpad and control
    flow limitations.

    Args:
        q: Query tensor, shape (1, head_dim). Single-query decode mode.
        keys: Key cache, shape (num_tiles * tile_size, head_dim).
        values: Value cache, shape (num_tiles * tile_size, head_dim).
        zeta_max: Spectral decay ratio threshold. Tiles with ζ > zeta_max
            are evicted (skipped entirely — both V load and Q×K^T compute).
        tile_size: Tokens per tile (default: 64).
        return_mask: If True, return the eviction mask in metadata.

    Returns:
        Tuple of (output, metadata):
        - output: Attention output, shape (1, head_dim).
        - metadata: Dict with eviction stats and optional mask.
    """
    # ── Validation ──
    assert q.ndim == 2 and q.shape[0] == 1, f"q must be (1, head_dim), got {q.shape}"
    head_dim = q.shape[-1]
    num_tokens = keys.shape[0]
    num_tiles = num_tokens // tile_size
    assert num_tokens == num_tiles * tile_size, (
        f"seq_len {num_tokens} not divisible by tile_size {tile_size}"
    )

    input_dtype = q.dtype

    # ── CPU / no-Triton fallback ──
    if not (HAS_CUDA and HAS_TRITON and q.is_cuda):
        out, meta = _pytorch_fused_orthocache_attention(
            q, keys, values, zeta_max, tile_size, return_mask
        )
        return out.to(input_dtype), meta

    # ── Prepare inputs ──
    q = q.contiguous()
    keys = keys.contiguous()
    values = values.contiguous()
    W = _get_walsh_matrix(keys.device)

    # ── Allocate outputs ──
    out = torch.empty((1, head_dim), dtype=torch.float32, device=q.device)
    mask_debug = (
        torch.empty(num_tiles, dtype=torch.int8, device=q.device)
        if return_mask else
        torch.empty(1, dtype=torch.int8, device=q.device)  # dummy
    )

    # ── Launch the God Kernel ──
    grid = (1,)  # Single program for single-query decode

    _fused_orthocache_kernel[grid](
        q, keys, values, W, out, mask_debug,
        zeta_max,
        head_dim=head_dim,
        num_k_tiles=num_tiles,
        TILE_SIZE=tile_size,
        RETURN_MASK=return_mask,
        BAND_LOW_START=BAND_LOW_64[0],
        BAND_LOW_END=BAND_LOW_64[1],
        BAND_HIGH_START=BAND_HIGH_64[0],
        BAND_HIGH_END=BAND_HIGH_64[1],
    )

    torch.cuda.synchronize()

    # ── Build metadata ──
    metadata: Dict[str, Any] = {}
    if return_mask:
        eviction_mask = mask_debug.bool()
        metadata['eviction_mask'] = eviction_mask
        retained = int(eviction_mask.sum().item())
        metadata['tiles_retained'] = retained
        metadata['tiles_evicted'] = num_tiles - retained
        metadata['eviction_rate'] = (num_tiles - retained) / max(1, num_tiles)

    return out.to(input_dtype), metadata
