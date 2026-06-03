"""Triton indirect-indexing attention kernel for OrthoCache GPU.

Ported from the JAX/Pallas implementation in orthocache/indirect_attention.py.
Instead of a boolean mask, this kernel receives an explicit index list of active
blocks (``active_indices``). The kernel uses pointer arithmetic to jump directly
to the relevant KV blocks — no data copy, no boolean branching per block.

This is the GPU analogue of the Pallas indirect kernel's ``dynamic_slice`` approach.
On GPU, we compute ``base_ptr + active_indices[i] * block_size * head_dim`` to
achieve scatter-gather style access into the original KV-cache layout.

Algorithm: Online softmax (FlashAttention-style numerically stable accumulation)
  - Maintains running max (m_i), running exp-sum (l_i), running weighted output (acc)
  - For each indexed block: update max, rescale old accumulators, accumulate new
  - Final output = acc / l_i

Grid: one program per query position (trivially 1 for single-query decode).
Each program iterates over num_active indices only.

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
    def _indirect_attention_kernel(
        # Pointers
        Q_ptr,             # (1, head_dim)
        K_ptr,             # (num_blocks * BLOCK_SIZE, head_dim)
        V_ptr,             # (num_blocks * BLOCK_SIZE, head_dim)
        IDX_ptr,           # (num_active,) int32 — active block indices
        OUT_ptr,           # (1, head_dim)
        # Dimensions
        head_dim: tl.constexpr,
        num_active: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel: indirect-indexing attention with online softmax.

        Iterates over ``num_active`` entries from the indirection table.
        Each entry specifies the original block index to load via pointer
        arithmetic, achieving true proportional FLOP savings.

        All accumulation is done in float32 for numerical stability,
        regardless of the input dtype (bfloat16 supported).
        """
        pid = tl.program_id(0)  # noqa: F841 — reserved for multi-query extension

        # ------- Load query vector (1, head_dim) → float32 -------
        q_offsets = tl.arange(0, head_dim)
        q = tl.load(Q_ptr + q_offsets, mask=q_offsets < head_dim, other=0.0)
        q = q.to(tl.float32)

        scale = 1.0 / tl.sqrt(head_dim.to(tl.float32))

        # ------- Online softmax accumulators -------
        m_i = tl.full([1], value=-1e9, dtype=tl.float32)     # running max
        l_i = tl.full([1], value=0.0, dtype=tl.float32)      # running exp-sum
        acc = tl.zeros([head_dim], dtype=tl.float32)          # running output

        # ------- Iterate over active block indices -------
        for i in range(num_active):
            # Look up original block index from the indirection table
            orig_b = tl.load(IDX_ptr + i).to(tl.int32)

            # Compute base pointer for this block in the KV-cache
            # Layout: (total_tokens, head_dim), row-major
            kv_base = orig_b * BLOCK_SIZE * head_dim

            # --- Block-level online softmax ---
            block_max = tl.full([1], value=-1e9, dtype=tl.float32)
            block_sum = tl.full([1], value=0.0, dtype=tl.float32)
            block_acc = tl.zeros([head_dim], dtype=tl.float32)

            for r in range(BLOCK_SIZE):
                # Load one key row via pointer arithmetic
                row_offset = kv_base + r * head_dim
                k_offsets = row_offset + q_offsets
                k_row = tl.load(K_ptr + k_offsets, mask=q_offsets < head_dim, other=0.0)
                k_row = k_row.to(tl.float32)

                # Dot product: q · k_row (scalar)
                logit = tl.sum(q * k_row, axis=0) * scale

                # Online softmax within this block
                new_block_max = tl.maximum(block_max, logit)
                scale_old = tl.exp(block_max - new_block_max)
                exp_logit = tl.exp(logit - new_block_max)

                block_sum = block_sum * scale_old + exp_logit

                # Load value row and accumulate
                v_row = tl.load(V_ptr + k_offsets, mask=q_offsets < head_dim, other=0.0)
                v_row = v_row.to(tl.float32)
                block_acc = block_acc * scale_old + exp_logit * v_row

                block_max = new_block_max

            # --- Merge block accumulator into global accumulator ---
            new_max = tl.maximum(m_i, block_max)
            scale_global = tl.exp(m_i - new_max)
            scale_block = tl.exp(block_max - new_max)

            l_i = l_i * scale_global + block_sum * scale_block
            acc = acc * scale_global + block_acc * scale_block
            m_i = new_max

        # ------- Normalize and store -------
        safe_l = tl.maximum(l_i, 1e-9)
        out = acc / safe_l

        tl.store(OUT_ptr + q_offsets, out, mask=q_offsets < head_dim)


# ---------------------------------------------------------------------------
# PyTorch fallback (CPU / no-Triton path)
# ---------------------------------------------------------------------------
def _pytorch_indirect_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    active_indices: torch.Tensor,
    num_active: int,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Pure-PyTorch indirect-indexing attention with online softmax.

    Mirrors the Triton kernel logic for correctness testing and CPU fallback.

    Args:
        q: Query tensor of shape (1, head_dim), float32 or bfloat16.
        keys: Key tensor of shape (num_blocks * block_size, head_dim).
        values: Value tensor of shape (num_blocks * block_size, head_dim).
        active_indices: Int32 tensor of shape (num_active,) with block indices.
        num_active: Number of active blocks to process.
        block_size: Tokens per KV block.

    Returns:
        Output tensor of shape (1, head_dim).
    """
    head_dim = q.shape[-1]
    scale = 1.0 / (head_dim ** 0.5)

    # Accumulate in float32
    q_f32 = q.float()
    m_i = torch.full((1,), -1e9, dtype=torch.float32, device=q.device)
    l_i = torch.zeros((1,), dtype=torch.float32, device=q.device)
    acc = torch.zeros((1, head_dim), dtype=torch.float32, device=q.device)

    for i in range(num_active):
        orig_b = active_indices[i].item()
        start = orig_b * block_size
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
def triton_indirect_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    active_indices: torch.Tensor,
    num_active: int,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Indirect-indexing attention using a Triton kernel (GPU) or PyTorch fallback (CPU).

    Implements the OrthoCache indirect-attention primitive: only KV blocks at
    positions specified by ``active_indices`` are loaded and processed. The
    KV-cache stays in place — no data copy, just pointer arithmetic.

    This achieves proportional FLOP savings: if 70% of blocks are evicted,
    only 30% of the work is performed.

    The algorithm uses FlashAttention-style online softmax for numerical
    stability without materialising the full attention matrix.

    Args:
        q: Query tensor of shape ``(1, head_dim)``.
        keys: Key tensor of shape ``(num_blocks * block_size, head_dim)`` —
              the **original**, non-compacted KV-cache.
        values: Value tensor of shape ``(num_blocks * block_size, head_dim)``.
        active_indices: Int32 tensor of shape ``(num_active,)`` containing
                        the block indices to attend to.
        num_active: Number of active blocks (length of ``active_indices``).
        block_size: Number of tokens per KV block (default: 512).

    Returns:
        Output tensor of shape ``(1, head_dim)``.

    Notes:
        - Inputs may be bfloat16; accumulation is always float32.
        - Output dtype matches input ``q`` dtype.
        - Falls back to pure PyTorch when CUDA/Triton is unavailable.
        - ``active_indices`` values must be valid block indices into the
          KV-cache (0-indexed, no bounds checking in the kernel).
    """
    # ---- Input validation ----
    assert q.ndim == 2 and q.shape[0] == 1, f"q must be (1, head_dim), got {q.shape}"
    head_dim = q.shape[-1]
    assert active_indices.ndim == 1, (
        f"active_indices must be 1-D, got shape {active_indices.shape}"
    )
    assert num_active <= active_indices.shape[0], (
        f"num_active ({num_active}) > len(active_indices) ({active_indices.shape[0]})"
    )

    input_dtype = q.dtype

    # Early exit: no active blocks
    if num_active == 0:
        return torch.zeros_like(q)

    # ---- Dispatch ----
    if not (HAS_CUDA and HAS_TRITON and q.is_cuda):
        out = _pytorch_indirect_attention(
            q, keys, values, active_indices, num_active, block_size
        )
        return out.to(input_dtype)

    # Ensure contiguous layout for pointer arithmetic
    q = q.contiguous()
    keys = keys.contiguous()
    values = values.contiguous()

    # Indices must be int32 on the same device
    active_indices = active_indices.to(dtype=torch.int32, device=q.device).contiguous()
    # Trim to num_active (kernel uses num_active as tl.constexpr loop bound)
    active_indices = active_indices[:num_active].contiguous()

    # Allocate output (float32 for accumulation, cast after)
    out = torch.empty((1, head_dim), dtype=torch.float32, device=q.device)

    # Grid: 1 program (single query)
    grid = (1,)

    _indirect_attention_kernel[grid](
        q, keys, values, active_indices, out,
        head_dim=head_dim,
        num_active=num_active,
        BLOCK_SIZE=block_size,
    )

    return out.to(input_dtype)
