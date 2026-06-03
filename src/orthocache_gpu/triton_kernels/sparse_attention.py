"""Triton block-sparse attention kernel for OrthoCache GPU.

Ported from the JAX/Pallas implementation in orthocache/sparse_attention.py.
This kernel uses a boolean block_mask to skip evicted KV-cache blocks entirely,
achieving true FLOP savings (unlike the TPU version which zeros operands but
still fires the MXU).

Algorithm: Online softmax (FlashAttention-style numerically stable accumulation)
  - Maintains running max (m_i), running exp-sum (l_i), running weighted output (acc)
  - For each active block: update max, rescale old accumulators, accumulate new
  - Final output = acc / l_i

Grid: one program per query position (trivially 1 for single-query decode).
Each program iterates over num_blocks, loading only where block_mask[b] == True.

Hardware target: NVIDIA H100 (SM 9.0), B200 (SM 10.0+).
"""

import torch
from typing import Optional

# --- Triton availability check ---
HAS_CUDA = torch.cuda.is_available()
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

BLOCK_SIZE: int = 512


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def _block_sparse_attention_kernel(
        # Pointers
        Q_ptr,           # (1, head_dim) — single query row
        K_ptr,           # (num_blocks * BLOCK_SIZE, head_dim)
        V_ptr,           # (num_blocks * BLOCK_SIZE, head_dim)
        MASK_ptr,        # (num_blocks,) — boolean (stored as int8/bool)
        OUT_ptr,         # (1, head_dim)
        # Dimensions (passed as tl.constexpr where possible)
        head_dim: tl.constexpr,
        num_blocks: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel: block-sparse attention with online softmax.

        One program handles the full query. It iterates over KV blocks,
        skipping any block whose mask entry is False, and accumulates
        the attention output using the online softmax algorithm.

        All accumulation is done in float32 for numerical stability,
        regardless of the input dtype (bfloat16 supported).
        """
        # Program index (trivially 0 for single-query decode)
        pid = tl.program_id(0)  # noqa: F841 — reserved for multi-query extension

        # ------- Load query vector (1, head_dim) → float32 -------
        q_offsets = tl.arange(0, head_dim)  # [0, 1, ..., head_dim-1]
        q = tl.load(Q_ptr + q_offsets, mask=q_offsets < head_dim, other=0.0)
        q = q.to(tl.float32)

        scale = 1.0 / tl.sqrt(head_dim.to(tl.float32))

        # ------- Online softmax accumulators -------
        m_i = tl.full([1], value=-1e9, dtype=tl.float32)     # running max
        l_i = tl.full([1], value=0.0, dtype=tl.float32)      # running exp-sum
        acc = tl.zeros([head_dim], dtype=tl.float32)          # running output

        # ------- Iterate over KV blocks -------
        for b in range(num_blocks):
            # Load mask for this block (scalar boolean)
            mask_val = tl.load(MASK_ptr + b)

            # Skip evicted blocks — true FLOP savings on GPU
            if mask_val:
                # Pointer base for this block
                kv_base = b * BLOCK_SIZE * head_dim

                # Accumulate attention over sub-rows within the block.
                # We process one key row at a time to stay within Triton's
                # register budget for large head_dim values.

                # --- Block-level online softmax ---
                # First pass: compute logits for all rows in this block,
                # find block max, compute exp-sum, and accumulate weighted V.
                block_max = tl.full([1], value=-1e9, dtype=tl.float32)
                block_sum = tl.full([1], value=0.0, dtype=tl.float32)
                block_acc = tl.zeros([head_dim], dtype=tl.float32)

                for r in range(BLOCK_SIZE):
                    # Load one key row
                    row_offset = kv_base + r * head_dim
                    k_offsets = row_offset + q_offsets
                    k_row = tl.load(K_ptr + k_offsets, mask=q_offsets < head_dim, other=0.0)
                    k_row = k_row.to(tl.float32)

                    # Dot product: q · k_row (scalar)
                    logit = tl.sum(q * k_row, axis=0) * scale

                    # Online softmax within this block
                    new_block_max = tl.maximum(block_max, logit)
                    # Rescale old accumulator
                    scale_old = tl.exp(block_max - new_block_max)
                    exp_logit = tl.exp(logit - new_block_max)

                    block_sum = block_sum * scale_old + exp_logit

                    # Load value row and accumulate
                    v_row = tl.load(V_ptr + k_offsets, mask=q_offsets < head_dim, other=0.0)
                    v_row = v_row.to(tl.float32)
                    block_acc = block_acc * scale_old + exp_logit * v_row

                    block_max = new_block_max

                # --- Merge block accumulator into global accumulator ---
                # Online merge: combine (m_i, l_i, acc) with (block_max, block_sum, block_acc)
                new_max = tl.maximum(m_i, block_max)
                scale_global = tl.exp(m_i - new_max)
                scale_block = tl.exp(block_max - new_max)

                l_i = l_i * scale_global + block_sum * scale_block
                acc = acc * scale_global + block_acc * scale_block
                m_i = new_max

        # ------- Normalize and store -------
        # Guard against division by zero (all blocks masked out)
        safe_l = tl.maximum(l_i, 1e-9)
        out = acc / safe_l

        # Store result (cast back to input dtype if needed — handled by wrapper)
        tl.store(OUT_ptr + q_offsets, out, mask=q_offsets < head_dim)


# ---------------------------------------------------------------------------
# PyTorch fallback (CPU / no-Triton path)
# ---------------------------------------------------------------------------
def _pytorch_block_sparse_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Pure-PyTorch block-sparse attention with online softmax.

    Mirrors the Triton kernel logic for correctness testing and CPU fallback.

    Args:
        q: Query tensor of shape (1, head_dim), float32 or bfloat16.
        keys: Key tensor of shape (num_blocks * block_size, head_dim).
        values: Value tensor of shape (num_blocks * block_size, head_dim).
        block_mask: Boolean tensor of shape (num_blocks,).
        block_size: Tokens per KV block.

    Returns:
        Output tensor of shape (1, head_dim).
    """
    head_dim = q.shape[-1]
    num_blocks = keys.shape[0] // block_size
    scale = 1.0 / (head_dim ** 0.5)

    # Accumulate in float32
    q_f32 = q.float()
    m_i = torch.full((1,), -1e9, dtype=torch.float32, device=q.device)
    l_i = torch.zeros((1,), dtype=torch.float32, device=q.device)
    acc = torch.zeros((1, head_dim), dtype=torch.float32, device=q.device)

    for b in range(num_blocks):
        if not block_mask[b].item():
            continue

        start = b * block_size
        end = start + block_size
        k_block = keys[start:end].float()    # (block_size, head_dim)
        v_block = values[start:end].float()  # (block_size, head_dim)

        # logits: (1, block_size)
        logits = (q_f32 @ k_block.T) * scale

        # Online softmax
        local_max = logits.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(m_i.unsqueeze(0), local_max)

        exp_logits = torch.exp(logits - new_max)
        sum_exp = exp_logits.sum(dim=-1, keepdim=True)

        scale_old = torch.exp(m_i.unsqueeze(0) - new_max)

        l_i = (l_i.unsqueeze(0) * scale_old + sum_exp).squeeze(0)
        acc = acc * scale_old + exp_logits @ v_block
        m_i = new_max.squeeze(0)

    # Normalize
    out = acc / torch.clamp(l_i.unsqueeze(-1), min=1e-9)
    return out


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------
def triton_block_sparse_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Block-sparse attention using a Triton kernel (GPU) or PyTorch fallback (CPU).

    Implements the OrthoCache sparse-attention primitive: only KV blocks whose
    corresponding ``block_mask`` entry is True are loaded and processed,
    yielding true computational savings proportional to the eviction rate.

    The algorithm uses FlashAttention-style online softmax for numerical
    stability without materialising the full attention matrix.

    Args:
        q: Query tensor of shape ``(1, head_dim)``.
        keys: Key tensor of shape ``(num_blocks * block_size, head_dim)``.
        values: Value tensor of shape ``(num_blocks * block_size, head_dim)``.
        block_mask: Boolean tensor of shape ``(num_blocks,)`` — True keeps a block.
        block_size: Number of tokens per KV block (default: 512).

    Returns:
        Output tensor of shape ``(1, head_dim)``.

    Notes:
        - Inputs may be bfloat16; accumulation is always float32.
        - Output dtype matches input ``q`` dtype.
        - Falls back to pure PyTorch when CUDA/Triton is unavailable.
    """
    # ---- Input validation ----
    assert q.ndim == 2 and q.shape[0] == 1, f"q must be (1, head_dim), got {q.shape}"
    head_dim = q.shape[-1]
    num_tokens = keys.shape[0]
    num_blocks = num_tokens // block_size
    assert num_tokens == num_blocks * block_size, (
        f"keys length {num_tokens} not divisible by block_size {block_size}"
    )
    assert block_mask.shape == (num_blocks,), (
        f"block_mask shape {block_mask.shape} != ({num_blocks},)"
    )

    input_dtype = q.dtype

    # ---- Dispatch ----
    if not (HAS_CUDA and HAS_TRITON and q.is_cuda):
        out = _pytorch_block_sparse_attention(q, keys, values, block_mask, block_size)
        return out.to(input_dtype)

    # Ensure contiguous layout for pointer arithmetic
    q = q.contiguous()
    keys = keys.contiguous()
    values = values.contiguous()

    # Mask must be on same device, stored as int8 for Triton loads
    block_mask_int = block_mask.to(dtype=torch.int8, device=q.device).contiguous()

    # Allocate output (float32 for accumulation, cast after)
    out = torch.empty((1, head_dim), dtype=torch.float32, device=q.device)

    # Grid: 1 program (single query)
    grid = (1,)

    _block_sparse_attention_kernel[grid](
        q, keys, values, block_mask_int, out,
        head_dim=head_dim,
        num_blocks=num_blocks,
        BLOCK_SIZE=block_size,
    )

    return out.to(input_dtype)
