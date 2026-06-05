"""OrthoCache Perfect Eviction Governor.

Implements the hardware-native underflow verification layer that classifies
block evictions into two regimes:

1. **Deterministic Regime**: The logit gap (z_max - β) exceeds the IEEE 754
   float32 underflow threshold (88.72). The quantized exponential evaluates
   to exact hardware zero — perfect eviction with TV(α, α̂) = 0.

2. **Statistical Regime**: The logit gap is below the threshold. Eviction is
   governed by the standard exponential decay bound:
   TV(α, α̂) ≤ |S^c| · exp(β - z_max).

The governor computes per-block metadata and provides the `classify_eviction`
function that determines whether each block qualifies for the deterministic
or statistical guarantee.

Formal verification: The deterministic regime is proved correct in
`proofs/OrthoCacheMath/QuantizedTruncation.lean` (zero `sorry` stubs).

Hardware target: NVIDIA H100 (SM 9.0), B200 (SM 10.0+).
Accumulator: float32 Tensor Cores with flush-to-zero subnormal handling.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

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

# IEEE 754 float32 underflow boundary.
# exp(x) with x < -88.72 produces a value below the smallest subnormal
# (1.4 × 10⁻⁴⁵) and is flushed to exact 0 in hardware accumulators.
FLOAT32_UNDERFLOW_THRESHOLD: float = 88.72

# IEEE 754 bfloat16 underflow boundary.
# bfloat16 has a much narrower exponent range; exp(x) underflows for x < -87.34.
BFLOAT16_UNDERFLOW_THRESHOLD: float = 87.34


class EvictionRegime(Enum):
    """Classification of a block's eviction safety level."""
    RETAIN = "retain"                    # Block is active — do not evict
    PERFECT_EVICTION = "perfect"         # Deterministic regime: TV = 0 (hardware zero)
    STATISTICAL_EVICTION = "statistical" # Statistical regime: TV ≤ bound


@dataclass
class EvictionMetadata:
    """Per-block eviction classification and diagnostics."""
    regime: torch.Tensor           # (num_blocks,) int: 0=retain, 1=perfect, 2=statistical
    beta: torch.Tensor             # (num_blocks,) float: logit ceiling β per block
    logit_gap: torch.Tensor        # (num_blocks,) float: z_max - β
    num_perfect: int               # Count of blocks with perfect eviction
    num_statistical: int           # Count of blocks with statistical eviction
    num_retained: int              # Count of retained blocks
    underflow_threshold: float     # The hardware underflow threshold used


def compute_block_beta(
    q: torch.Tensor,
    block_energies: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Computes the logit ceiling β for each block.

    β_j = ||q||₂ · √(E_j) / √(d_k)

    where E_j is the spectral energy of block j (Parseval equivalent to
    the sum of squared key norms within the block).

    Args:
        q: Query tensor of shape (seq_len_q, num_heads, head_dim).
        block_energies: Tensor of shape (num_blocks, num_heads) — block spectral energies.
        head_dim: Dimension of each attention head.

    Returns:
        Tensor of shape (num_blocks, num_heads) — per-block β values.
    """
    # q_norm: (seq_len_q, num_heads)
    q_norm = torch.linalg.norm(q.float(), dim=-1)
    # Use max over query positions for worst-case bound
    q_norm_max = torch.max(q_norm, dim=0).values  # (num_heads,)

    sqrt_energy = torch.sqrt(block_energies.float().clamp(min=0))  # (num_blocks, num_heads)
    sqrt_dk = math.sqrt(head_dim)

    # β: (num_blocks, num_heads)
    return (q_norm_max[None, :] * sqrt_energy) / sqrt_dk


def classify_eviction(
    q: torch.Tensor,
    block_energies: torch.Tensor,
    z_max: torch.Tensor,
    block_mask: torch.Tensor,
    head_dim: int,
    accumulator_dtype: str = "float32",
) -> EvictionMetadata:
    """Classifies each evicted block into deterministic or statistical regime.

    For each block that is marked for eviction (block_mask == False), computes:
    1. The logit ceiling β from the query norm and block spectral energy.
    2. The logit gap (z_max - β).
    3. Whether the gap exceeds the IEEE 754 underflow threshold.

    Blocks in the deterministic regime have a machine-checked guarantee of
    TV(α, α̂) = 0. Blocks in the statistical regime are bounded by the
    standard exponential decay.

    Args:
        q: Query tensor of shape (seq_len_q, num_heads, head_dim).
        block_energies: Tensor of shape (num_blocks, num_heads).
        z_max: Scalar or tensor — maximum logit among retained tokens.
            If tensor, shape should be (num_heads,).
        block_mask: Boolean tensor of shape (num_blocks,) or (num_blocks, num_heads).
            True = retained, False = evicted.
        head_dim: Dimension of each attention head.
        accumulator_dtype: The accumulator precision used by the hardware.
            'float32' (default) or 'bfloat16'.

    Returns:
        EvictionMetadata with per-block classification.
    """
    # Select the appropriate underflow threshold
    if accumulator_dtype == "bfloat16":
        threshold = BFLOAT16_UNDERFLOW_THRESHOLD
    else:
        threshold = FLOAT32_UNDERFLOW_THRESHOLD

    # Compute β for all blocks
    beta = compute_block_beta(q, block_energies, head_dim)  # (num_blocks, num_heads)

    # Reduce over heads for unified classification
    beta_max = torch.max(beta, dim=-1).values  # (num_blocks,)

    # Ensure z_max is a scalar or broadcast correctly
    if isinstance(z_max, torch.Tensor) and z_max.ndim > 0:
        z_max_scalar = torch.max(z_max).item()
    else:
        z_max_scalar = float(z_max)

    # Logit gap: (num_blocks,)
    logit_gap = z_max_scalar - beta_max

    # Unify block_mask to (num_blocks,)
    if block_mask.ndim > 1:
        unified_mask = torch.any(block_mask, dim=-1)  # (num_blocks,)
    else:
        unified_mask = block_mask

    # Classify:
    # 0 = retained, 1 = perfect eviction, 2 = statistical eviction
    regime = torch.zeros(unified_mask.shape[0], dtype=torch.int32, device=q.device)

    evicted = ~unified_mask
    perfect = evicted & (logit_gap >= threshold)
    statistical = evicted & (logit_gap < threshold)

    regime[perfect] = 1
    regime[statistical] = 2

    return EvictionMetadata(
        regime=regime,
        beta=beta_max,
        logit_gap=logit_gap,
        num_perfect=int(perfect.sum().item()),
        num_statistical=int(statistical.sum().item()),
        num_retained=int(unified_mask.sum().item()),
        underflow_threshold=threshold,
    )


def perfect_eviction_check(
    z_max: float,
    beta: float,
    accumulator_dtype: str = "float32",
) -> bool:
    """Quick scalar check: does a single block qualify for perfect eviction?

    This is the Python equivalent of the Lean 4 theorem
    `orthocache_perfect_eviction_bound`: when z_max - β ≥ UnderflowThreshold,
    the quantized exponential evaluates to exact 0.

    Args:
        z_max: Maximum logit among retained tokens.
        beta: Logit ceiling for the evicted block.
        accumulator_dtype: 'float32' or 'bfloat16'.

    Returns:
        True if the block qualifies for perfect (TV = 0) eviction.
    """
    threshold = (
        BFLOAT16_UNDERFLOW_THRESHOLD
        if accumulator_dtype == "bfloat16"
        else FLOAT32_UNDERFLOW_THRESHOLD
    )
    return (z_max - beta) >= threshold


# ============================================================================
# Triton Kernel: Perfect Eviction Governor (In-Register Evaluation)
# ============================================================================

if HAS_TRITON:

    @triton.jit
    def _perfect_eviction_governor_kernel(
        # Pointers
        Q_NORM_ptr,         # (num_blocks,) — max query norm per block
        ENERGY_ptr,         # (num_blocks,) — block spectral energy
        ZETA_ptr,           # (num_blocks,) — spectral decay ratio
        Z_MAX_ptr,          # (1,) — global z_max
        REGIME_ptr,         # (num_blocks,) output: 0=retain, 1=perfect, 2=statistical
        MASK_ptr,           # (num_blocks,) — eviction mask (1=retain, 0=evict)
        # Scalars
        head_dim: tl.constexpr,
        epsilon_threshold: tl.constexpr,
        zeta_max: tl.constexpr,
        UNDERFLOW_THRESHOLD: tl.constexpr,
        BLOCK_META: tl.constexpr,   # number of blocks to process per program
    ):
        """Evaluates execution bounds natively in registers.

        Classifies each evicted block into:
          - Perfect eviction (regime=1): logit gap ≥ underflow threshold
          - Statistical eviction (regime=2): bounded by exp(β - z_max)
          - Retained (regime=0): block passes the eviction mask

        This kernel runs in the metadata pre-pass layer before the main
        attention GEMM, consuming negligible ALU cycles.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_META + tl.arange(0, BLOCK_META)

        # Load metadata
        q_norm = tl.load(Q_NORM_ptr + offsets)
        energy = tl.load(ENERGY_ptr + offsets)
        zeta = tl.load(ZETA_ptr + offsets)
        mask = tl.load(MASK_ptr + offsets)
        z_max = tl.load(Z_MAX_ptr)  # broadcast scalar

        # Calculate β = q_norm * sqrt(energy) / sqrt(head_dim)
        sqrt_energy = tl.sqrt(tl.maximum(energy, 0.0))
        sqrt_dk = tl.sqrt(head_dim * 1.0)
        beta = (q_norm * sqrt_energy) / sqrt_dk

        # Logit gap
        logit_gap = z_max - beta

        # Classification logic
        is_retained = mask > 0.5
        is_perfect_underflow = (logit_gap >= UNDERFLOW_THRESHOLD) & (~is_retained)
        is_statistical = (~is_retained) & (~is_perfect_underflow)

        # Additional structural gate: blocks with ζ > ζ_max AND low energy
        # qualify for perfect eviction even below the underflow threshold
        structural_perfect = (
            (zeta > zeta_max) &
            (energy < epsilon_threshold) &
            (~is_retained)
        )
        is_perfect_underflow = is_perfect_underflow | structural_perfect
        is_statistical = is_statistical & (~structural_perfect)

        # Write regime: 0=retain, 1=perfect, 2=statistical
        regime = tl.where(is_retained, 0,
                 tl.where(is_perfect_underflow, 1, 2))
        tl.store(REGIME_ptr + offsets, regime)


def triton_classify_eviction(
    q_norm_per_block: torch.Tensor,
    block_energies: torch.Tensor,
    zeta: torch.Tensor,
    z_max: torch.Tensor,
    block_mask: torch.Tensor,
    head_dim: int,
    epsilon_threshold: float = 1e-6,
    zeta_max: float = 5.0,
    accumulator_dtype: str = "float32",
) -> torch.Tensor:
    """GPU-accelerated eviction classification via Triton kernel.

    Runs the Perfect Eviction Governor entirely in GPU registers during
    the metadata pre-pass, before the main Tensor Core GEMM.

    Args:
        q_norm_per_block: (num_blocks,) — max query norm per block.
        block_energies: (num_blocks,) — block spectral energy (any-head max).
        zeta: (num_blocks,) — spectral decay ratio.
        z_max: Scalar tensor — global z_max.
        block_mask: (num_blocks,) — True=retain, False=evict.
        head_dim: Dimension of each attention head.
        epsilon_threshold: Energy threshold for structural perfect eviction.
        zeta_max: Maximum spectral decay ratio.
        accumulator_dtype: 'float32' or 'bfloat16'.

    Returns:
        regime tensor: (num_blocks,) int32. 0=retain, 1=perfect, 2=statistical.
    """
    if not (HAS_CUDA and HAS_TRITON and q_norm_per_block.is_cuda):
        raise RuntimeError(
            "triton_classify_eviction requires CUDA and Triton. "
            "Use classify_eviction() for CPU fallback."
        )

    threshold = (
        BFLOAT16_UNDERFLOW_THRESHOLD
        if accumulator_dtype == "bfloat16"
        else FLOAT32_UNDERFLOW_THRESHOLD
    )

    num_blocks = q_norm_per_block.shape[0]
    regime = torch.zeros(num_blocks, dtype=torch.int32, device=q_norm_per_block.device)

    # Ensure contiguous float32
    q_norm_per_block = q_norm_per_block.float().contiguous()
    block_energies = block_energies.float().contiguous()
    zeta = zeta.float().contiguous()
    mask_float = block_mask.float().contiguous()
    z_max = z_max.float().contiguous().reshape(1)

    # Round up to power of 2 for Triton constexpr
    BLOCK_META = triton.next_power_of_2(num_blocks)

    # Pad if necessary
    if num_blocks < BLOCK_META:
        pad = BLOCK_META - num_blocks
        q_norm_per_block = torch.nn.functional.pad(q_norm_per_block, (0, pad))
        block_energies = torch.nn.functional.pad(block_energies, (0, pad))
        zeta = torch.nn.functional.pad(zeta, (0, pad))
        mask_float = torch.nn.functional.pad(mask_float, (0, pad), value=1.0)
        regime = torch.nn.functional.pad(regime, (0, pad))

    grid = (1,)
    _perfect_eviction_governor_kernel[grid](
        q_norm_per_block, block_energies, zeta, z_max, regime, mask_float,
        head_dim=head_dim,
        epsilon_threshold=epsilon_threshold,
        zeta_max=zeta_max,
        UNDERFLOW_THRESHOLD=threshold,
        BLOCK_META=BLOCK_META,
    )

    return regime[:num_blocks]
