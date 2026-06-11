"""Fused FWHT + ζ Eviction + FlashAttention Triton Kernel (Phase 7/7b).

THE GOD KERNEL: Fuses the entire OrthoCache pipeline into one kernel launch:
    1. Load K tile from HBM → SRAM
    2. FWHT via Walsh matrix tl.dot() on Tensor Cores (stays in SRAM)
    3. Compute ζ (spectral decay ratio) in-register
    4. If ζ > threshold: skip this tile entirely (true branch elimination)
    5. If ζ ≤ threshold: load V tile, compute Q×K^T, online softmax accumulate

Phase 7b adds Split-K parallelization:
    - Grid: (num_heads, num_splits) — one launch for ALL heads
    - Interleaved (cyclic) tile assignment for perfect load balancing
    - Each CTA writes partial online-softmax state (m, l, acc)
    - Lightweight reduction kernel merges partials via log-sum-exp

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
# V1 Kernel: Single-CTA Sequential (Phase 7 original, kept for reference)
# ============================================================================

if HAS_TRITON:

    @triton.jit
    def _fused_orthocache_kernel_v1(
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
        """V1: Single-CTA sequential kernel (Phase 7 original).

        Kept for correctness comparison against Split-K. Do not use in
        production — does not scale beyond ~4K tokens.
        # Cache invalidation comment
        """
        pid = tl.program_id(0)  # head index (for multi-head extension)

        # ── Load query vector (1, head_dim) → fp32 ────────────────────
        q_offsets = tl.arange(0, head_dim)
        q = tl.load(Q_ptr + q_offsets)
        q = q.to(tl.float32)
        inv_sqrt_d: tl.constexpr = 1.0 / (head_dim ** 0.5)

        # ── Load Walsh matrix (TILE_SIZE × TILE_SIZE) → SRAM ─────────
        w_row = tl.arange(0, TILE_SIZE)
        w_col = tl.arange(0, TILE_SIZE)
        w_offsets = w_row[:, None] * TILE_SIZE + w_col[None, :]
        w_matrix = tl.load(W_ptr + w_offsets)

        # ── Online softmax accumulators ───────────────────────────────
        m_i = tl.full([1], value=-1e9, dtype=tl.float32)
        l_i = tl.full([1], value=0.0, dtype=tl.float32)
        acc = tl.zeros([head_dim], dtype=tl.float32)

        # ── Sequency index for band masking ───────────────────────────
        seq_idx = tl.arange(0, TILE_SIZE)
        low_mask = (seq_idx >= BAND_LOW_START) & (seq_idx < BAND_LOW_END)
        high_mask = (seq_idx >= BAND_HIGH_START) & (seq_idx < BAND_HIGH_END)

        # ── Iterate over K tiles ──────────────────────────────────────
        for tile_id in range(num_k_tiles):
            kv_base = tile_id * TILE_SIZE * head_dim

            row_offsets = tl.arange(0, TILE_SIZE)
            col_offsets = tl.arange(0, head_dim)
            k_offsets = kv_base + row_offsets[:, None] * head_dim + col_offsets[None, :]
            k_tile = tl.load(K_ptr + k_offsets)
            k_tile = k_tile.to(tl.float32)

            k_spectral = tl.dot(w_matrix, k_tile)
            k_sq = k_spectral * k_spectral
            energy_per_seq = tl.sum(k_sq, axis=1)

            e_low = tl.sum(tl.where(low_mask, energy_per_seq, 0.0))
            e_high = tl.sum(tl.where(high_mask, energy_per_seq, 0.0))
            zeta = e_high / (e_low + 1e-6)

            keep = zeta <= zeta_max

            if RETURN_MASK:
                tl.store(MASK_DEBUG_ptr + tile_id, keep.to(tl.int8))

            if keep:
                logits = tl.sum(k_tile * q[None, :], axis=1) * inv_sqrt_d
                tile_max = tl.max(logits)
                new_max = tl.maximum(m_i, tile_max)
                alpha = tl.exp(m_i - new_max)
                p = tl.exp(logits - new_max)
                p_sum = tl.sum(p)
                l_i = l_i * alpha + p_sum
                v_tile = tl.load(V_ptr + k_offsets)
                v_tile = v_tile.to(tl.float32)
                weighted_v = tl.sum(p[:, None] * v_tile, axis=0)
                acc = acc * alpha + weighted_v
                m_i = new_max

        safe_l = tl.maximum(l_i, 1e-9)
        out = acc / safe_l
        tl.store(O_ptr + q_offsets, out, mask=q_offsets < head_dim)


# ============================================================================
# V2 Kernel: Split-K with Interleaved Tile Assignment (Phase 7b)
# ============================================================================

if HAS_TRITON:

    @triton.jit
    def _fused_orthocache_splitk_kernel(
        # ── Tensor pointers ──
        Q_ptr,              # (num_heads, head_dim) — queries, all heads
        K_ptr,              # (num_heads, seq_len, head_dim) — key cache
        V_ptr,              # (num_heads, seq_len, head_dim) — value cache
        W_ptr,              # (TILE_SIZE, TILE_SIZE) — Walsh matrix, fp32
        # ── Partial output staging buffers ──
        M_ptr,              # (num_heads, num_splits) — partial max
        L_ptr,              # (num_heads, num_splits) — partial exp-sum
        ACC_ptr,            # (num_heads, num_splits, head_dim) — partial acc
        # ── Scalar args ──
        zeta_max,           # float — ζ threshold
        seq_len,            # int — total KV sequence length
        # ── Dimensions (constexpr) ──
        head_dim: tl.constexpr,
        num_k_tiles: tl.constexpr,
        num_splits: tl.constexpr,
        TILE_SIZE: tl.constexpr,       # 64
        # ── Band boundaries (constexpr) ──
        BAND_LOW_START: tl.constexpr,
        BAND_LOW_END: tl.constexpr,
        BAND_HIGH_START: tl.constexpr,
        BAND_HIGH_END: tl.constexpr,
    ):
        """Split-K God Kernel with interleaved (cyclic) tile assignment.

        Grid: (num_heads, num_splits)
          - program_id(0) = head index
          - program_id(1) = split index (which CTA chunk)

        Each CTA processes tiles: split_id, split_id + num_splits,
        split_id + 2*num_splits, ... (interleaved / cyclic). This ensures
        every CTA gets a uniform mix of system-prompt tiles (low eviction)
        and middle-context tiles (high eviction), preventing straggler SMs.

        Writes partial online-softmax state (m, l, acc) to staging buffers.
        A separate reduction kernel merges these into the final output.
        """
        head_id = tl.program_id(0)
        split_id = tl.program_id(1)

        # ── Compute base offsets for this head ────────────────────────
        q_base = head_id * head_dim
        kv_head_base = head_id * seq_len * head_dim

        # ── Load query vector for this head → fp32 ───────────────────
        q_offsets = tl.arange(0, head_dim)
        q = tl.load(Q_ptr + q_base + q_offsets)
        q = q.to(tl.float32)
        inv_sqrt_d: tl.constexpr = 1.0 / (head_dim ** 0.5)

        # ── Load Walsh matrix (shared across all CTAs) ───────────────
        w_row = tl.arange(0, TILE_SIZE)
        w_col = tl.arange(0, TILE_SIZE)
        w_offsets = w_row[:, None] * TILE_SIZE + w_col[None, :]
        w_matrix = tl.load(W_ptr + w_offsets)

        # ── Online softmax accumulators (per-CTA partial) ────────────
        # Use float scalars via [1]-shaped tensors for m_i, l_i
        m_i = -1e9
        l_i = 0.0
        acc = tl.zeros([head_dim], dtype=tl.float32)

        # ── Band masks for ζ computation ─────────────────────────────
        seq_idx = tl.arange(0, TILE_SIZE)
        low_mask = (seq_idx >= BAND_LOW_START) & (seq_idx < BAND_LOW_END)
        high_mask = (seq_idx >= BAND_HIGH_START) & (seq_idx < BAND_HIGH_END)

        # ── Row/col offset templates (reused each iteration) ─────────
        row_offsets = tl.arange(0, TILE_SIZE)
        col_offsets = tl.arange(0, head_dim)

        # ── INTERLEAVED TILE LOOP ────────────────────────────────────
        # CTA with split_id=s processes tiles: s, s+K, s+2K, ...
        # where K = num_splits. This guarantees every CTA gets a
        # uniform mix of high-retention and high-eviction tiles.
        for tile_id in range(split_id, num_k_tiles, num_splits):
            kv_base = kv_head_base + tile_id * TILE_SIZE * head_dim

            # ── PHASE A: In-SRAM spectral eviction ────────────────────
            k_offsets = kv_base + row_offsets[:, None] * head_dim + col_offsets[None, :]
            k_tile = tl.load(K_ptr + k_offsets)
            k_tile = k_tile.to(tl.float32)

            # FWHT via Tensor Core matmul (stays in SRAM)
            k_spectral = tl.dot(w_matrix, k_tile)

            # Per-sequency energy → band energies → ζ
            k_sq = k_spectral * k_spectral
            energy_per_seq = tl.sum(k_sq, axis=1)
            e_low = tl.sum(tl.where(low_mask, energy_per_seq, 0.0))
            e_high = tl.sum(tl.where(high_mask, energy_per_seq, 0.0))
            zeta = e_high / (e_low + 1e-6)

            # ── Eviction decision ─────────────────────────────────────
            keep = zeta <= zeta_max

            if keep:
                # ── PHASE B: Predicated Attention (K reused from SRAM) ─
                logits = tl.sum(k_tile * q[None, :], axis=1) * inv_sqrt_d

                # Online softmax update
                tile_max = tl.max(logits)
                new_max = tl.where(m_i > tile_max, m_i, tile_max)
                alpha = tl.exp(m_i - new_max)
                p = tl.exp(logits - new_max)
                p_sum = tl.sum(p)

                l_i = l_i * alpha + p_sum

                # Load V tile (ONLY for retained tiles — skip HBM read)
                v_tile = tl.load(V_ptr + k_offsets)
                v_tile = v_tile.to(tl.float32)
                weighted_v = tl.sum(p[:, None] * v_tile, axis=0)

                acc = acc * alpha + weighted_v
                m_i = new_max

        # ── Write partial state to staging buffers ────────────────────
        # M_ptr layout: (num_heads, num_splits)
        partial_idx = head_id * num_splits + split_id
        tl.store(M_ptr + partial_idx, m_i)
        tl.store(L_ptr + partial_idx, l_i)

        # ACC_ptr layout: (num_heads, num_splits, head_dim)
        acc_base = (head_id * num_splits + split_id) * head_dim
        tl.store(ACC_ptr + acc_base + q_offsets, acc)


    # ========================================================================
    # Reduction Kernel: Merge Split-K Partials via Log-Sum-Exp
    # ========================================================================

    @triton.jit
    def _splitk_reduce_kernel(
        # ── Partial input buffers ──
        M_ptr,              # (num_heads, num_splits) — partial max
        L_ptr,              # (num_heads, num_splits) — partial exp-sum
        ACC_ptr,            # (num_heads, num_splits, head_dim) — partial acc
        # ── Final output ──
        O_ptr,              # (num_heads, head_dim) — final output
        # ── Dimensions (constexpr) ──
        head_dim: tl.constexpr,
        num_splits: tl.constexpr,
    ):
        """Merge Split-K partial online-softmax states into final output.

        Grid: (num_heads,)

        For each head, loads all num_splits partial states (m_s, l_s, acc_s)
        and performs exact log-sum-exp correction:

            m_new  = max(m_1, m_2)
            l_new  = l_1·exp(m_1 - m_new) + l_2·exp(m_2 - m_new)
            acc_new = acc_1·exp(m_1 - m_new) + acc_2·exp(m_2 - m_new)
            O = acc_final / l_final

        This is exact — no approximation. Same correction as FlashAttention.
        """
        head_id = tl.program_id(0)
        d_offsets = tl.arange(0, head_dim)

        # ── Load first partial as initial state ───────────────────────
        base_idx = head_id * num_splits
        m_run = tl.load(M_ptr + base_idx)
        l_run = tl.load(L_ptr + base_idx)
        acc_base = (head_id * num_splits) * head_dim
        acc_run = tl.load(ACC_ptr + acc_base + d_offsets)

        # ── Merge remaining partials ──────────────────────────────────
        for s in range(1, num_splits):
            m_s = tl.load(M_ptr + base_idx + s)
            l_s = tl.load(L_ptr + base_idx + s)
            acc_s_offset = (head_id * num_splits + s) * head_dim
            acc_s = tl.load(ACC_ptr + acc_s_offset + d_offsets)

            # Log-sum-exp correction
            new_max = tl.maximum(m_run, m_s)
            alpha_run = tl.exp(m_run - new_max)
            alpha_s = tl.exp(m_s - new_max)

            l_run = l_run * alpha_run + l_s * alpha_s
            acc_run = acc_run * alpha_run + acc_s * alpha_s
            m_run = new_max

        # ── Normalize and write final output ──────────────────────────
        safe_l = tl.maximum(l_run, 1e-9)
        out = acc_run / safe_l

        out_base = head_id * head_dim
        tl.store(O_ptr + out_base + d_offsets, out)


# ============================================================================
# PyTorch Fallback: Multi-Head Fused Attention (CPU / no-Triton)
# ============================================================================

def _pytorch_fused_orthocache_multihead(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    zeta_max: float,
    tile_size: int = TILE_SIZE,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Pure-PyTorch fused FWHT+ζ+attention for multi-head input.

    Args:
        q: Query, shape (num_heads, head_dim).
        keys: Keys, shape (num_heads, seq_len, head_dim).
        values: Values, shape (num_heads, seq_len, head_dim).
        zeta_max: ζ threshold for eviction.
        tile_size: Tokens per tile (default: 64).

    Returns:
        Tuple of (output, metadata).
    """
    num_heads, seq_len, head_dim = keys.shape
    num_tiles = seq_len // tile_size
    scale = 1.0 / (head_dim ** 0.5)
    W = generate_walsh_matrix(tile_size).to(keys.device)

    all_outputs = []
    total_retained = 0
    total_evicted = 0

    for h in range(num_heads):
        q_h = q[h].float().unsqueeze(0)  # (1, head_dim)
        k_h = keys[h].float()             # (seq_len, head_dim)
        v_h = values[h].float()            # (seq_len, head_dim)

        m_i = torch.full((1,), -1e9, dtype=torch.float32, device=q.device)
        l_i = torch.zeros((1,), dtype=torch.float32, device=q.device)
        acc = torch.zeros((1, head_dim), dtype=torch.float32, device=q.device)

        for t in range(num_tiles):
            start = t * tile_size
            end = start + tile_size

            k_tile = k_h[start:end]  # (tile_size, head_dim)

            # PHASE A: FWHT + ζ
            k_spectral = W @ k_tile
            energy_per_seq = torch.sum(k_spectral ** 2, dim=1)
            e_low = torch.sum(energy_per_seq[BAND_LOW_64[0]:BAND_LOW_64[1]])
            e_high = torch.sum(energy_per_seq[BAND_HIGH_64[0]:BAND_HIGH_64[1]])
            zeta = (e_high / (e_low + 1e-6)).item()

            if zeta > zeta_max:
                total_evicted += 1
                continue

            # PHASE B: Attention
            total_retained += 1
            v_tile = v_h[start:end]
            logits = (q_h @ k_tile.T) * scale
            local_max = logits.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(m_i.unsqueeze(0), local_max)
            exp_logits = torch.exp(logits - new_max)
            sum_exp = exp_logits.sum(dim=-1, keepdim=True)
            alpha = torch.exp(m_i.unsqueeze(0) - new_max)
            l_i = (l_i.unsqueeze(0) * alpha + sum_exp).squeeze(0)
            acc = acc * alpha + exp_logits @ v_tile
            m_i = new_max.squeeze(0)

        out = acc / torch.clamp(l_i.unsqueeze(-1), min=1e-9)
        all_outputs.append(out.squeeze(0))  # (head_dim,)

    output = torch.stack(all_outputs, dim=0)  # (num_heads, head_dim)

    metadata = {
        'tiles_retained': total_retained,
        'tiles_evicted': total_evicted,
        'eviction_rate': total_evicted / max(1, num_tiles * num_heads),
    }
    return output, metadata


def _pytorch_fused_orthocache_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    zeta_max: float,
    tile_size: int = TILE_SIZE,
    return_mask: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """V1-compatible pure-PyTorch fallback (single-head).

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
# Public API: V2 Multi-Head Split-K (Phase 7b)
# ============================================================================

def _auto_num_splits(num_tiles: int, device: torch.device) -> int:
    """Auto-select num_splits based on SM count and tile count.

    Strategy:
    - Use all SMs, but ensure each CTA gets at least 4 tiles to
      amortize kernel launch overhead.
    - Cap at num_tiles (no point having more splits than tiles).
    - Minimum 1 split.
    """
    if device.type != 'cuda':
        return 1
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    # Each split should handle at least 4 tiles
    max_splits = max(1, num_tiles // 4)
    return max(1, min(num_sms, max_splits, num_tiles))


def fused_orthocache_attention_v2(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    zeta_max: float,
    num_splits: Optional[int] = None,
    tile_size: int = TILE_SIZE,
    k_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Split-K God Kernel: multi-head fused FWHT+ζ+attention (Phase 7b).

    Single-launch processing of ALL attention heads with grid-parallel
    Split-K tiling. Interleaved (cyclic) tile assignment ensures perfect
    load balancing across SMs regardless of non-uniform eviction patterns.

    Args:
        q: Query tensor, shape (num_heads, head_dim).
        keys: Key cache, shape (num_heads, seq_len, head_dim).
        values: Value cache, shape (num_heads, seq_len, head_dim).
        zeta_max: Spectral decay ratio threshold.
        num_splits: Number of Split-K partitions per head. If None,
            auto-selected based on SM count (RTX 4060: up to 24).
        tile_size: Tokens per tile (default: 64).
        k_scale: Scaling factor for FP8 key dequantization. If not 1.0, dequantizes query-side.

    Returns:
        Tuple of (output, metadata):
        - output: Attention output, shape (num_heads, head_dim).
        - metadata: Dict with eviction stats, split info.
    """
    num_heads = q.shape[0]
    head_dim = q.shape[-1]
    seq_len = keys.shape[1]
    num_tiles = seq_len // tile_size

    # Robustness assertions
    assert (head_dim & (head_dim - 1)) == 0 and head_dim > 0, f"head_dim ({head_dim}) must be a power of 2"
    assert (tile_size & (tile_size - 1)) == 0 and tile_size > 0, f"tile_size ({tile_size}) must be a power of 2"
    assert seq_len == num_tiles * tile_size, (
        f"seq_len {seq_len} not divisible by tile_size {tile_size}"
    )
    if num_splits is not None:
        assert num_splits > 0, f"num_splits ({num_splits}) must be positive"

    input_dtype = q.dtype

    # Dequantize keys if they are in FP8 to avoid Triton compiler crashes on Windows
    if keys.dtype == torch.float8_e4m3fn:
        keys = keys.to(q.dtype) * k_scale
        k_scale = 1.0

    # Dequantize using PyTorch-side scaling: q_scaled = q * k_scale
    if k_scale != 1.0:
        q = q * k_scale

    # ── CPU / no-Triton fallback ──
    if not (HAS_CUDA and HAS_TRITON and q.is_cuda):
        out, meta = _pytorch_fused_orthocache_multihead(
            q, keys, values, zeta_max, tile_size
        )
        return out.to(input_dtype), meta



    # ── Auto-select num_splits ──
    if num_splits is None:
        num_splits = _auto_num_splits(num_tiles, q.device)

    # ── Prepare contiguous inputs ──
    q = q.contiguous()
    keys = keys.contiguous()
    values = values.contiguous()
    W = _get_walsh_matrix(keys.device)

    # ── Allocate staging buffers for partial online-softmax state ──
    M_partial = torch.empty(
        (num_heads, num_splits), dtype=torch.float32, device=q.device
    )
    L_partial = torch.empty(
        (num_heads, num_splits), dtype=torch.float32, device=q.device
    )
    ACC_partial = torch.empty(
        (num_heads, num_splits, head_dim), dtype=torch.float32, device=q.device
    )

    # ── Allocate final output ──
    out = torch.empty(
        (num_heads, head_dim), dtype=torch.float32, device=q.device
    )

    # ── Launch Split-K kernel: (num_heads, num_splits) ──
    grid_main = (num_heads, num_splits)
    _fused_orthocache_splitk_kernel[grid_main](
        q, keys, values, W,
        M_partial, L_partial, ACC_partial,
        zeta_max, seq_len,
        head_dim=head_dim,
        num_k_tiles=num_tiles,
        num_splits=num_splits,
        TILE_SIZE=tile_size,
        BAND_LOW_START=BAND_LOW_64[0],
        BAND_LOW_END=BAND_LOW_64[1],
        BAND_HIGH_START=BAND_HIGH_64[0],
        BAND_HIGH_END=BAND_HIGH_64[1],
    )

    # ── Launch reduction kernel: (num_heads,) ──
    grid_reduce = (num_heads,)
    _splitk_reduce_kernel[grid_reduce](
        M_partial, L_partial, ACC_partial,
        out,
        head_dim=head_dim,
        num_splits=num_splits,
    )

    torch.cuda.synchronize()

    metadata: Dict[str, Any] = {
        'num_splits': num_splits,
        'num_tiles': num_tiles,
        'num_heads': num_heads,
        'tile_assignment': 'interleaved',
    }

    return out.to(input_dtype), metadata


# ============================================================================
# Public API: V1-Compatible Single-Head Wrapper
# ============================================================================

def fused_orthocache_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    zeta_max: float,
    tile_size: int = TILE_SIZE,
    return_mask: bool = False,
    k_scale: float = 1.0,
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
        k_scale: Scaling factor for FP8 key dequantization. If not 1.0, dequantizes query-side.

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

    # Robustness assertions
    assert (head_dim & (head_dim - 1)) == 0 and head_dim > 0, f"head_dim ({head_dim}) must be a power of 2"
    assert (tile_size & (tile_size - 1)) == 0 and tile_size > 0, f"tile_size ({tile_size}) must be a power of 2"
    assert num_tokens == num_tiles * tile_size, (
        f"seq_len {num_tokens} not divisible by tile_size {tile_size}"
    )

    input_dtype = q.dtype

    # Dequantize keys if they are in FP8 to avoid Triton compiler crashes on Windows
    if keys.dtype == torch.float8_e4m3fn:
        keys = keys.to(q.dtype) * k_scale
        k_scale = 1.0

    # Dequantize using PyTorch-side scaling: q_scaled = q * k_scale
    if k_scale != 1.0:
        q = q * k_scale

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

    # ── Launch V1 kernel (single-CTA, for backward compat + mask) ──
    grid = (1,)

    _fused_orthocache_kernel_v1[grid](
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
