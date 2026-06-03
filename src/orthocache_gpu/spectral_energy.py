import torch
from .fwht import fwht_512

# ============================================================================
# Sequency Band Boundaries
# ============================================================================
# The 512 Walsh-Hadamard coefficients partition into discrete sequency bands:
#   - DC (index 0): block mean / macro-semantic pivot
#   - Low-sequency (1–63): smooth semantic trends across the block
#   - Mid-sequency (64–255): syntactic/token-relational context
#   - High-sequency (256–511): rapid oscillations / formatting noise
#
# These are configurable parameters, not hardcoded magic numbers.
BAND_DC = 0
BAND_LOW = (1, 64)      # indices [1, 64) — 63 coefficients
BAND_MID = (64, 256)     # indices [64, 256) — 192 coefficients
BAND_HIGH = (256, 512)   # indices [256, 512) — 256 coefficients


def compute_block_energy(keys: torch.Tensor, block_size: int = 512) -> torch.Tensor:
    """Computes the spatial energy of keys per block (Parseval equivalent).
    
    Args:
        keys: Tensor of shape (seq_len, num_heads, head_dim). seq_len must be a multiple of block_size.
        block_size: Size of the blocks (default: 512).
        
    Returns:
        Tensor of shape (num_blocks, num_heads) containing the spatial energy per block.
    """
    seq_len, num_heads, head_dim = keys.shape
    num_blocks = seq_len // block_size
    blocks = keys.reshape(num_blocks, block_size, num_heads, head_dim)
    # Spatial energy is sum of squared key norms per block
    return torch.sum(blocks ** 2, dim=(1, 3))


def _run_fwht_on_blocks(keys: torch.Tensor, block_size: int = 512) -> torch.Tensor:
    """Internal helper: runs FWHT on blocked keys and returns spectral tensor.
    
    Args:
        keys: (seq_len, num_heads, head_dim)
    
    Returns:
        spectral: (num_blocks, block_size, num_heads, head_dim) — WHT coefficients
    """
    seq_len, num_heads, head_dim = keys.shape
    num_blocks = seq_len // block_size
    
    # Reshape keys to (num_blocks, block_size, num_heads, head_dim)
    blocks = keys.reshape(num_blocks, block_size, num_heads, head_dim)
    
    # Transpose to (block_size, num_blocks, num_heads, head_dim) for fwht_512
    blocks_t = blocks.permute(1, 0, 2, 3)
    
    # Flatten non-sequence dims for batched FWHT: (block_size, num_blocks*num_heads*head_dim)
    flat = blocks_t.reshape(block_size, num_blocks * num_heads * head_dim)
    
    # Run the 9-stage unrolled FWHT
    spectral_flat = fwht_512(flat)
    
    # Reshape back: (block_size, num_blocks, num_heads, head_dim)
    spectral = spectral_flat.reshape(block_size, num_blocks, num_heads, head_dim)
    
    # Transpose back: (num_blocks, block_size, num_heads, head_dim)
    return spectral.permute(1, 0, 2, 3)


def compute_spectral_bands(
    keys: torch.Tensor,
    block_size: int = 512,
    band_low: tuple = BAND_LOW,
    band_mid: tuple = BAND_MID,
    band_high: tuple = BAND_HIGH,
) -> tuple:
    """Decomposes FWHT coefficients into discrete sequency bands.
    
    This is the core function that makes the FWHT genuinely load-bearing.
    Per-band energy decomposition requires individual spectral coefficients
    and CANNOT be computed from spatial statistics alone.
    
    Args:
        keys: (seq_len, num_heads, head_dim)
        block_size: Block size for partitioning (default: 512).
        band_low: (start, end) indices for the low-sequency band.
        band_mid: (start, end) indices for the mid-sequency band.
        band_high: (start, end) indices for the high-sequency band.
        
    Returns:
        Tuple of:
            dc_component: (num_blocks, num_heads, head_dim) — the 0th coefficient
            low_energy: (num_blocks, num_heads) — energy in low-sequency band
            mid_energy: (num_blocks, num_heads) — energy in mid-sequency band
            high_energy: (num_blocks, num_heads) — energy in high-sequency band
    """
    spectral = _run_fwht_on_blocks(keys, block_size)
    
    # DC component: index 0 → (num_blocks, num_heads, head_dim)
    dc_component = spectral[:, BAND_DC, :, :]
    
    # Per-band energy: sum of squared coefficients across sequence and head_dim axes
    low_energy = torch.sum(spectral[:, band_low[0]:band_low[1], :, :] ** 2, dim=(1, 3))
    mid_energy = torch.sum(spectral[:, band_mid[0]:band_mid[1], :, :] ** 2, dim=(1, 3))
    high_energy = torch.sum(spectral[:, band_high[0]:band_high[1], :, :] ** 2, dim=(1, 3))
    
    return dc_component, low_energy, mid_energy, high_energy


def compute_spectral_decay_ratio(
    keys: torch.Tensor,
    block_size: int = 512,
    band_low: tuple = BAND_LOW,
    band_high: tuple = BAND_HIGH,
) -> torch.Tensor:
    """Computes the Spectral Decay Ratio ζ per block.
    
    ζ_j = high-frequency energy / low-frequency energy
    
    This ratio is THE load-bearing metric that justifies the FWHT:
    - High ζ (>> 1): block dominated by high-frequency noise (formatting, punctuation)
    - Low ζ (<< 1): block dominated by coherent low-frequency semantic structure
    
    Two blocks with IDENTICAL spatial variance can have completely different ζ values.
    This is provable by construction and tested in test_spectral_bands.py.
    
    Args:
        keys: (seq_len, num_heads, head_dim)
        block_size: Block size (default: 512).
        band_low: Low-sequency band indices.
        band_high: High-sequency band indices.
        
    Returns:
        ζ: (num_blocks, num_heads) — spectral decay ratio per block per head
    """
    dc, low_energy, mid_energy, high_energy = compute_spectral_bands(
        keys, block_size, band_low, (band_low[1], band_high[0]), band_high
    )
    return high_energy / (low_energy + 1e-6)


def compute_query_aware_bounds(
    q: torch.Tensor, keys: torch.Tensor, block_size: int = 512
) -> torch.Tensor:
    """Computes query-aware attention bounds per block using Walsh-Hadamard DC/AC decomposition.
    
    Args:
        q: Query tensor of shape (seq_len_q, num_heads, head_dim).
        keys: Key tensor of shape (seq_len_k, num_heads, head_dim).
        block_size: Size of the blocks (default: 512).
        
    Returns:
        Tensor of shape (seq_len_q, num_blocks, num_heads) containing the logit upper bounds.
    """
    seq_len_k, num_heads, head_dim = keys.shape
    
    spectral = _run_fwht_on_blocks(keys, block_size)
    
    # DC component: (num_blocks, num_heads, head_dim)
    dc_component = spectral[:, 0, :, :]
    
    # AC energy: sum of squared coefficients for indices 1..511
    ac_components = spectral[:, 1:, :, :]
    ac_energy = torch.sum(ac_components ** 2, dim=(1, 3))  # (num_blocks, num_heads)
    
    # Block mean = DC / sqrt(block_size)
    block_mean = dc_component / torch.sqrt(torch.tensor(float(block_size)))
    
    # Query-mean alignment: (seq_len_q, num_blocks, num_heads)
    alignment = torch.einsum("qhd,bhd->qbh", q, block_mean) / torch.sqrt(torch.tensor(float(head_dim)))
    
    # Residual bound via Cauchy-Schwarz
    q_norm = torch.linalg.norm(q, dim=-1)  # (seq_len_q, num_heads)
    residual_bound = (
        q_norm[:, None, :] * torch.sqrt(ac_energy)[None, :, :]
    ) / torch.sqrt(torch.tensor(float(head_dim)))
    
    return alignment + residual_bound


def compute_query_aware_mask(
    q: torch.Tensor, keys: torch.Tensor, tau: float, block_size: int = 512
) -> torch.Tensor:
    """Generates a query-aware boolean mask for block eviction (single-gate).
    
    Args:
        q: Query tensor of shape (seq_len_q, num_heads, head_dim).
        keys: Key tensor of shape (seq_len_k, num_heads, head_dim).
        tau: The threshold value. Blocks with bound >= tau are retained (True).
        block_size: Size of the blocks (default: 512).
        
    Returns:
        A boolean tensor of shape (num_blocks, num_heads) indicating retained blocks.
    """
    bounds = compute_query_aware_bounds(q, keys, block_size)
    max_bounds = torch.max(bounds, dim=0).values  # (num_blocks, num_heads)
    return max_bounds >= tau


def compute_multiband_mask(
    q: torch.Tensor,
    keys: torch.Tensor,
    tau: float,
    zeta_max: float,
    block_size: int = 512,
) -> torch.Tensor:
    """Two-gate block eviction mask using query-aware bounds AND spectral decay ratio.
    
    A block is retained ONLY if it passes BOTH gates:
      1. Query-aware logit bound >= tau  (the block could matter to some query)
      2. Spectral decay ratio zeta <= zeta_max  (the block is not pure noise)
    
    This is the core eviction function that makes the FWHT genuinely load-bearing.
    Gate 2 (ζ) requires per-band frequency decomposition that is impossible to compute
    from spatial statistics alone.
    
    Args:
        q: Query tensor of shape (seq_len_q, num_heads, head_dim).
        keys: Key tensor of shape (seq_len_k, num_heads, head_dim).
        tau: Logit bound threshold. Blocks below this are evicted.
        zeta_max: Maximum spectral decay ratio. Blocks above this are evicted.
        block_size: Block size (default: 512).
        
    Returns:
        A boolean tensor of shape (num_blocks, num_heads) indicating retained blocks.
    """
    # Gate 1: Query-aware logit bound
    logit_mask = compute_query_aware_mask(q, keys, tau, block_size)
    
    # Gate 2: Spectral decay ratio
    zeta = compute_spectral_decay_ratio(keys, block_size)
    noise_mask = zeta <= zeta_max
    
    # Both gates must pass
    return logit_mask & noise_mask


def generate_threshold_mask(energies: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Generates a boolean mask indicating whether blocks are retained (backward compatibility)."""
    return energies >= epsilon
