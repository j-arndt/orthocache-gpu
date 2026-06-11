"""Paged FWHT + ζ Eviction + FlashAttention Triton Kernel (Phase 3).

Implements PagedAttention support: keys/values are loaded from non-contiguous
physical blocks via block_tables.
"""

import math
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
_W64_cache_paged: Optional[torch.Tensor] = None


def _get_walsh_matrix_paged(device: torch.device) -> torch.Tensor:
    """Get or create the cached 64×64 Walsh matrix on the given device."""
    global _W64_cache_paged
    if _W64_cache_paged is None or _W64_cache_paged.device != device:
        _W64_cache_paged = generate_walsh_matrix(TILE_SIZE).contiguous().to(device)
    return _W64_cache_paged


def _auto_num_splits(num_tiles: int, device: torch.device) -> int:
    """Auto-select num_splits based on SM count and tile count."""
    if device.type != 'cuda':
        return 1
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    max_splits = max(1, num_tiles // 4)
    return max(1, min(num_sms, max_splits, num_tiles))


# ============================================================================
# Triton Kernel: Paged Split-K Fused Eviction & Attention
# ============================================================================

if HAS_TRITON:

    @triton.jit
    def _fused_orthocache_paged_splitk_kernel(
        # ── Tensor pointers ──
        Q_ptr,              # (num_seqs, num_heads, head_dim)
        K_ptr,              # (num_physical_blocks, num_kv_heads, block_size, head_dim)
        V_ptr,              # (num_physical_blocks, num_kv_heads, block_size, head_dim)
        W_ptr,              # (TILE_SIZE, TILE_SIZE) — Walsh matrix, fp32
        block_tables_ptr,   # (num_seqs, max_blocks_per_seq)
        # ── Staging buffers ──
        M_ptr,              # (num_seqs, num_heads, num_splits)
        L_ptr,              # (num_seqs, num_heads, num_splits)
        ACC_ptr,            # (num_seqs, num_heads, num_splits, head_dim)
        # ── Scalar args ──
        zeta_max,           # float — ζ threshold
        max_blocks_per_seq, # int
        block_size,         # int — tokens per block
        # ── Strides ──
        stride_q_seq, stride_q_head, stride_q_dim,
        stride_k_block, stride_k_head, stride_k_token, stride_k_dim,
        stride_v_block, stride_v_head, stride_v_token, stride_v_dim,
        stride_bt_seq, stride_bt_block,
        # ── Dimensions (constexpr) ──
        num_heads: tl.constexpr,
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
        """Split-K Paged Attention God Kernel loading physical blocks dynamically."""
        global_head_id = tl.program_id(0)
        split_id = tl.program_id(1)

        seq_id = global_head_id // num_heads
        head_id = global_head_id % num_heads

        # ── Compute base offsets for this head ──
        q_base = seq_id * stride_q_seq + head_id * stride_q_head

        # ── Load query vector → fp32 ──
        col_offsets = tl.arange(0, head_dim)
        q = tl.load(Q_ptr + q_base + col_offsets * stride_q_dim)
        q = q.to(tl.float32)
        inv_sqrt_d: tl.constexpr = 1.0 / (head_dim ** 0.5)

        # ── Load Walsh matrix (shared across all CTAs) ──
        w_row = tl.arange(0, TILE_SIZE)
        w_col = tl.arange(0, TILE_SIZE)
        w_offsets = w_row[:, None] * TILE_SIZE + w_col[None, :]
        w_matrix = tl.load(W_ptr + w_offsets)

        # ── Online softmax accumulators ──
        m_i = -1e9
        l_i = 0.0
        acc = tl.zeros([head_dim], dtype=tl.float32)

        # ── Band masks for ζ computation ──
        seq_idx = tl.arange(0, TILE_SIZE)
        low_mask = (seq_idx >= BAND_LOW_START) & (seq_idx < BAND_LOW_END)
        high_mask = (seq_idx >= BAND_HIGH_START) & (seq_idx < BAND_HIGH_END)

        row_offsets = tl.arange(0, TILE_SIZE)

        # ── Tile loop (interleaved) ──
        for tile_id in range(split_id, num_k_tiles, num_splits):
            # Map logical token index to physical block and token offset
            logical_token_idx = tile_id * TILE_SIZE + row_offsets[:, None]  # (TILE_SIZE, 1)
            block_idx_in_seq = logical_token_idx // block_size
            token_idx_in_block = logical_token_idx % block_size

            # Load physical block IDs from block table
            table_offsets = seq_id * stride_bt_seq + block_idx_in_seq * stride_bt_block
            physical_block_id = tl.load(block_tables_ptr + table_offsets)  # (TILE_SIZE, 1)

            # Address calculation for keys
            k_offsets = (
                physical_block_id * stride_k_block +
                head_id * stride_k_head +
                token_idx_in_block * stride_k_token +
                col_offsets[None, :] * stride_k_dim
            )
            k_tile = tl.load(K_ptr + k_offsets)
            k_tile = k_tile.to(tl.float32)

            # FWHT via Tensor Core matmul
            k_spectral = tl.dot(w_matrix, k_tile)

            # Per-sequency energy → band energies → ζ
            k_sq = k_spectral * k_spectral
            energy_per_seq = tl.sum(k_sq, axis=1)
            e_low = tl.sum(tl.where(low_mask, energy_per_seq, 0.0))
            e_high = tl.sum(tl.where(high_mask, energy_per_seq, 0.0))
            zeta = e_high / (e_low + 1e-6)

            keep = zeta <= zeta_max

            if keep:
                logits = tl.sum(k_tile * q[None, :], axis=1) * inv_sqrt_d

                # Online softmax update
                tile_max = tl.max(logits)
                new_max = tl.where(m_i > tile_max, m_i, tile_max)
                alpha = tl.exp(m_i - new_max)
                p = tl.exp(logits - new_max)
                p_sum = tl.sum(p)

                l_i = l_i * alpha + p_sum

                # Load V tile
                v_offsets = (
                    physical_block_id * stride_v_block +
                    head_id * stride_v_head +
                    token_idx_in_block * stride_v_token +
                    col_offsets[None, :] * stride_v_dim
                )
                v_tile = tl.load(V_ptr + v_offsets)
                v_tile = v_tile.to(tl.float32)

                weighted_v = tl.sum(p[:, None] * v_tile, axis=0)
                acc = acc * alpha + weighted_v
                m_i = new_max

        # Write partial state to staging buffers
        num_grids = num_heads * num_splits
        partial_idx = seq_id * num_grids + head_id * num_splits + split_id
        tl.store(M_ptr + partial_idx, m_i)
        tl.store(L_ptr + partial_idx, l_i)

        acc_base = (seq_id * num_grids + head_id * num_splits + split_id) * head_dim
        tl.store(ACC_ptr + acc_base + col_offsets, acc)


    # ========================================================================
    # Reduction Kernel: Merge Split-K Partials
    # ========================================================================

    @triton.jit
    def _paged_splitk_reduce_kernel(
        M_ptr,              # (num_seqs, num_heads, num_splits)
        L_ptr,              # (num_seqs, num_heads, num_splits)
        ACC_ptr,            # (num_seqs, num_heads, num_splits, head_dim)
        O_ptr,              # (num_seqs, num_heads, head_dim)
        head_dim: tl.constexpr,
        num_splits: tl.constexpr,
        num_heads: tl.constexpr,
    ):
        """Merge Split-K partials for batched paged attention."""
        global_head_id = tl.program_id(0)
        seq_id = global_head_id // num_heads
        head_id = global_head_id % num_heads

        d_offsets = tl.arange(0, head_dim)

        base_idx = (seq_id * num_heads + head_id) * num_splits
        m_run = tl.load(M_ptr + base_idx)
        l_run = tl.load(L_ptr + base_idx)

        acc_base = base_idx * head_dim
        acc_run = tl.load(ACC_ptr + acc_base + d_offsets)

        for s in range(1, num_splits):
            m_s = tl.load(M_ptr + base_idx + s)
            l_s = tl.load(L_ptr + base_idx + s)
            acc_s_offset = (base_idx + s) * head_dim
            acc_s = tl.load(ACC_ptr + acc_s_offset + d_offsets)

            new_max = tl.maximum(m_run, m_s)
            alpha_run = tl.exp(m_run - new_max)
            alpha_s = tl.exp(m_s - new_max)

            l_run = l_run * alpha_run + l_s * alpha_s
            acc_run = acc_run * alpha_run + acc_s * alpha_s
            m_run = new_max

        safe_l = tl.maximum(l_run, 1e-9)
        out = acc_run / safe_l

        out_base = (seq_id * num_heads + head_id) * head_dim
        tl.store(O_ptr + out_base + d_offsets, out)


# ============================================================================
# PyTorch Fallback: Paged Eviction & Attention
# ============================================================================

def _pytorch_paged_orthocache_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    zeta_max: float,
    block_size: int,
    tile_size: int = TILE_SIZE,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Pure-PyTorch paged attention fallback for multi-head, multi-sequence inputs."""
    num_seqs, num_heads, head_dim = q.shape
    max_blocks_per_seq = block_tables.shape[1]
    num_tiles = (max_blocks_per_seq * block_size) // tile_size
    scale = 1.0 / (head_dim ** 0.5)
    W = generate_walsh_matrix(tile_size).to(q.device)

    all_outputs = []

    for seq_idx in range(num_seqs):
        seq_outputs = []
        for h in range(num_heads):
            q_h = q[seq_idx, h].float().unsqueeze(0)  # (1, head_dim)

            # Reconstruct contiguous keys/values from blocks
            k_list = []
            v_list = []
            for b in range(max_blocks_per_seq):
                physical_block_id = block_tables[seq_idx, b].item()
                k_block = k_cache[physical_block_id, h].float()
                v_block = v_cache[physical_block_id, h].float()
                k_list.append(k_block)
                v_list.append(v_block)

            k_h = torch.cat(k_list, dim=0)
            v_h = torch.cat(v_list, dim=0)

            # Standard online softmax + eviction
            m_i = torch.full((1,), -1e9, dtype=torch.float32, device=q.device)
            l_i = torch.zeros((1,), dtype=torch.float32, device=q.device)
            acc = torch.zeros((1, head_dim), dtype=torch.float32, device=q.device)

            for t in range(num_tiles):
                start = t * tile_size
                end = start + tile_size

                k_tile = k_h[start:end]
                k_spectral = W @ k_tile
                energy_per_seq = torch.sum(k_spectral ** 2, dim=1)
                e_low = torch.sum(energy_per_seq[BAND_LOW_64[0]:BAND_LOW_64[1]])
                e_high = torch.sum(energy_per_seq[BAND_HIGH_64[0]:BAND_HIGH_64[1]])
                zeta = (e_high / (e_low + 1e-6)).item()

                if zeta > zeta_max:
                    continue

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
            seq_outputs.append(out.squeeze(0))

        all_outputs.append(torch.stack(seq_outputs, dim=0))

    output = torch.stack(all_outputs, dim=0)
    return output, {}


# ============================================================================
# Public API: Paged OrthoCache Attention
# ============================================================================

def fused_orthocache_attention_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    zeta_max: float,
    num_splits: Optional[int] = None,
    tile_size: int = TILE_SIZE,
    k_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Paged Split-K: batched multi-head fused FWHT+ζ+attention with block table layout."""
    num_seqs, num_heads, head_dim = q.shape
    num_physical_blocks = k_cache.shape[0]

    # Stride & block dimensions check
    # Assumes layout: (num_physical_blocks, num_kv_heads, block_size, head_dim)
    block_size = k_cache.shape[2]
    max_blocks_per_seq = block_tables.shape[1]
    num_tiles = (max_blocks_per_seq * block_size) // tile_size
    seq_len = num_tiles * tile_size

    # Robustness assertions
    assert (head_dim & (head_dim - 1)) == 0 and head_dim > 0, f"head_dim ({head_dim}) must be a power of 2"
    assert (tile_size & (tile_size - 1)) == 0 and tile_size > 0, f"tile_size ({tile_size}) must be a power of 2"
    assert (block_size & (block_size - 1)) == 0 and block_size > 0, f"block_size ({block_size}) must be a power of 2"
    assert (seq_len % tile_size) == 0, f"seq_len {seq_len} not divisible by tile_size {tile_size}"
    if num_splits is not None:
        assert num_splits > 0, f"num_splits ({num_splits}) must be positive"

    input_dtype = q.dtype

    # Dequantize keys if they are in FP8 to avoid Triton compiler crashes on Windows
    if k_cache.dtype == torch.float8_e4m3fn:
        k_cache = k_cache.to(q.dtype) * k_scale
        k_scale = 1.0

    # Dequantize using PyTorch-side scaling: q_scaled = q * k_scale
    if k_scale != 1.0:
        q = q * k_scale

    # ── CPU / no-Triton fallback ──
    if not (HAS_CUDA and HAS_TRITON and q.is_cuda):
        out, meta = _pytorch_paged_orthocache_attention(
            q, k_cache, v_cache, block_tables, zeta_max, block_size, tile_size
        )
        return out.to(input_dtype), meta

    # ── Auto-select num_splits ──
    if num_splits is None:
        num_splits = _auto_num_splits(num_tiles, q.device)

    # ── Prepare inputs ──
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_tables = block_tables.contiguous()

    W = _get_walsh_matrix_paged(q.device)

    # ── Allocate staging buffers for partial online-softmax state ──
    M_partial = torch.empty(
        (num_seqs, num_heads, num_splits), dtype=torch.float32, device=q.device
    )
    L_partial = torch.empty(
        (num_seqs, num_heads, num_splits), dtype=torch.float32, device=q.device
    )
    ACC_partial = torch.empty(
        (num_seqs, num_heads, num_splits, head_dim), dtype=torch.float32, device=q.device
    )

    # ── Allocate final output ──
    out = torch.empty(
        (num_seqs, num_heads, head_dim), dtype=torch.float32, device=q.device
    )

    # Get strides
    stride_q_seq, stride_q_head, stride_q_dim = q.stride()
    stride_k_block, stride_k_head, stride_k_token, stride_k_dim = k_cache.stride()
    stride_v_block, stride_v_head, stride_v_token, stride_v_dim = v_cache.stride()
    stride_bt_seq, stride_bt_block = block_tables.stride()

    # ── Launch Split-K kernel: (num_seqs * num_heads, num_splits) ──
    grid_main = (num_seqs * num_heads, num_splits)
    _fused_orthocache_paged_splitk_kernel[grid_main](
        q, k_cache, v_cache, W, block_tables,
        M_partial, L_partial, ACC_partial,
        zeta_max, max_blocks_per_seq, block_size,
        stride_q_seq, stride_q_head, stride_q_dim,
        stride_k_block, stride_k_head, stride_k_token, stride_k_dim,
        stride_v_block, stride_v_head, stride_v_token, stride_v_dim,
        stride_bt_seq, stride_bt_block,
        num_heads=num_heads,
        head_dim=head_dim,
        num_k_tiles=num_tiles,
        num_splits=num_splits,
        TILE_SIZE=tile_size,
        BAND_LOW_START=BAND_LOW_64[0],
        BAND_LOW_END=BAND_LOW_64[1],
        BAND_HIGH_START=BAND_HIGH_64[0],
        BAND_HIGH_END=BAND_HIGH_64[1],
    )

    # ── Launch reduction kernel: (num_seqs * num_heads,) ──
    grid_reduce = (num_seqs * num_heads,)
    _paged_splitk_reduce_kernel[grid_reduce](
        M_partial, L_partial, ACC_partial,
        out,
        head_dim=head_dim,
        num_splits=num_splits,
        num_heads=num_heads,
    )

    torch.cuda.synchronize()

    metadata: Dict[str, Any] = {
        'num_splits': num_splits,
        'num_tiles': num_tiles,
        'num_heads': num_heads,
        'num_seqs': num_seqs,
        'layout': 'paged',
    }

    return out.to(input_dtype), metadata
