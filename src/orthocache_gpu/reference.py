"""Reference (NumPy) implementations for OrthoCache GPU Edition.

Pure NumPy implementations used for correctness verification of the
PyTorch/Triton GPU kernels. These are NOT performance-critical — they exist
solely to provide ground-truth outputs for testing.
"""

import numpy as np

def numpy_fwht_1d(a: np.ndarray) -> np.ndarray:
    """Computes the 1D Fast Walsh-Hadamard Transform using the butterfly algorithm.
    
    Args:
        a: 1D array of length power-of-2 (e.g. 512).
        
    Returns:
        The transformed 1D array, normalized by 1 / sqrt(len(a)).
    """
    x = a.copy().astype(np.float64)
    h = 1
    n = len(x)
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = x[j]
                v = x[j + h]
                x[j] = u + v
                x[j + h] = u - v
        h *= 2
    return x / np.sqrt(n)

def numpy_fwht(tile: np.ndarray) -> np.ndarray:
    """Computes the FWHT of a 2D tile along the first (row/sequence) axis.
    
    Args:
        tile: 2D array of shape (num_tokens, head_dim) where num_tokens is a power-of-2.
        
    Returns:
        The row-wise transformed 2D array.
    """
    transformed = np.zeros_like(tile, dtype=np.float64)
    for d in range(tile.shape[1]):
        transformed[:, d] = numpy_fwht_1d(tile[:, d])
    return transformed.astype(tile.dtype)

def compute_block_energy_reference(keys: np.ndarray, block_size: int = 512) -> np.ndarray:
    """Computes the reference spectral energy per block using the numpy FWHT.
    
    Args:
        keys: 3D array of shape (seq_len, num_heads, head_dim).
        block_size: Size of block to segment sequence into (e.g. 512).
        
    Returns:
        An array of shape (num_blocks, num_heads) containing spectral energy per block.
    """
    seq_len, num_heads, head_dim = keys.shape
    num_blocks = seq_len // block_size
    energies = np.zeros((num_blocks, num_heads), dtype=np.float64)
    
    for h in range(num_heads):
        for b in range(num_blocks):
            block_keys = keys[b * block_size : (b + 1) * block_size, h, :]
            spectral = numpy_fwht(block_keys)
            energies[b, h] = np.sum(spectral ** 2)
            
    return energies


def compute_spectral_bands_reference(
    keys: np.ndarray,
    block_size: int = 512,
    band_low: tuple = (1, 64),
    band_mid: tuple = (64, 256),
    band_high: tuple = (256, 512),
) -> tuple:
    """Reference implementation of multi-band spectral decomposition.
    
    Args:
        keys: (seq_len, num_heads, head_dim)
        block_size: Block size.
        band_low, band_mid, band_high: Band boundary tuples.
        
    Returns:
        (dc, low_energy, mid_energy, high_energy):
            dc: (num_blocks, num_heads, head_dim)
            low_energy: (num_blocks, num_heads)
            mid_energy: (num_blocks, num_heads)
            high_energy: (num_blocks, num_heads)
    """
    seq_len, num_heads, head_dim = keys.shape
    num_blocks = seq_len // block_size
    
    dc = np.zeros((num_blocks, num_heads, head_dim), dtype=np.float64)
    low_energy = np.zeros((num_blocks, num_heads), dtype=np.float64)
    mid_energy = np.zeros((num_blocks, num_heads), dtype=np.float64)
    high_energy = np.zeros((num_blocks, num_heads), dtype=np.float64)
    
    for h in range(num_heads):
        for b in range(num_blocks):
            block_keys = keys[b * block_size : (b + 1) * block_size, h, :]
            spectral = numpy_fwht(block_keys)
            
            dc[b, h, :] = spectral[0, :]
            low_energy[b, h] = np.sum(spectral[band_low[0]:band_low[1], :] ** 2)
            mid_energy[b, h] = np.sum(spectral[band_mid[0]:band_mid[1], :] ** 2)
            high_energy[b, h] = np.sum(spectral[band_high[0]:band_high[1], :] ** 2)
    
    return dc, low_energy, mid_energy, high_energy


def compute_spectral_decay_ratio_reference(
    keys: np.ndarray,
    block_size: int = 512,
    band_low: tuple = (1, 64),
    band_high: tuple = (256, 512),
) -> np.ndarray:
    """Reference implementation of the Spectral Decay Ratio ζ.
    
    ζ_j = high-frequency energy / (low-frequency energy + ε)
    
    For 2D input (single block, no head dim), treats as (block_size, head_dim) directly.
    
    Args:
        keys: Either (seq_len, num_heads, head_dim) or (block_size, head_dim) for single-block.
        block_size: Block size.
        band_low: Low-sequency band indices.
        band_high: High-sequency band indices.
        
    Returns:
        ζ: (num_blocks, num_heads) or scalar for single-block input.
    """
    if keys.ndim == 2:
        # Single block, no head dimension: (block_size, head_dim)
        spectral = numpy_fwht(keys)
        low_e = np.sum(spectral[band_low[0]:band_low[1], :] ** 2)
        high_e = np.sum(spectral[band_high[0]:band_high[1], :] ** 2)
        return high_e / (low_e + 1e-6)
    
    dc, low_energy, mid_energy, high_energy = compute_spectral_bands_reference(
        keys, block_size, band_low, (band_low[1], band_high[0]), band_high
    )
    return high_energy / (low_energy + 1e-6)


def compute_query_aware_bounds_reference(q: np.ndarray, keys: np.ndarray, block_size: int = 512) -> np.ndarray:
    """Computes the reference query-aware bounds per block using numpy FWHT.
    
    Args:
        q: Query array of shape (seq_len_q, num_heads, head_dim).
        keys: Key array of shape (seq_len_k, num_heads, head_dim).
        block_size: Size of block to segment sequence into (e.g. 512).
        
    Returns:
        An array of shape (seq_len_q, num_blocks, num_heads) containing logit bounds.
    """
    seq_len_k, num_heads, head_dim = keys.shape
    seq_len_q = q.shape[0]
    num_blocks = seq_len_k // block_size
    bounds = np.zeros((seq_len_q, num_blocks, num_heads), dtype=np.float64)
    
    for h in range(num_heads):
        for b in range(num_blocks):
            block_keys = keys[b * block_size : (b + 1) * block_size, h, :]
            spectral = numpy_fwht(block_keys)
            
            dc = spectral[0]
            ac = spectral[1:]
            ac_energy = np.sum(ac ** 2)
            
            block_mean = dc / np.sqrt(block_size)
            
            for qi in range(seq_len_q):
                q_vec = q[qi, h, :]
                alignment = np.dot(q_vec, block_mean) / np.sqrt(head_dim)
                q_norm = np.linalg.norm(q_vec)
                residual = (q_norm * np.sqrt(ac_energy)) / np.sqrt(head_dim)
                bounds[qi, b, h] = alignment + residual
                
    return bounds


def compute_multiband_mask_reference(
    q: np.ndarray,
    keys: np.ndarray,
    tau: float,
    zeta_max: float,
    block_size: int = 512,
) -> np.ndarray:
    """Reference two-gate eviction mask using query-aware bounds AND spectral decay ratio.
    
    Args:
        q: (seq_len_q, num_heads, head_dim)
        keys: (seq_len_k, num_heads, head_dim)
        tau: Logit bound threshold.
        zeta_max: Maximum spectral decay ratio.
        block_size: Block size.
        
    Returns:
        Boolean array of shape (num_blocks, num_heads) indicating retained blocks.
    """
    # Gate 1: Query-aware logit bound
    bounds = compute_query_aware_bounds_reference(q, keys, block_size)
    max_bounds = np.max(bounds, axis=0)  # (num_blocks, num_heads)
    logit_mask = max_bounds >= tau
    
    # Gate 2: Spectral decay ratio
    zeta = compute_spectral_decay_ratio_reference(keys, block_size)
    noise_mask = zeta <= zeta_max
    
    return logit_mask & noise_mask


def compute_tv_distance(alpha: np.ndarray, alpha_hat: np.ndarray) -> float:
    """Computes the Total Variation (TV) distance between two attention distributions.
    
    Args:
        alpha: Full attention probability distribution (1D or 2D).
        alpha_hat: Truncated attention probability distribution (same shape as alpha).
        
    Returns:
        The Total Variation distance.
    """
    return 0.5 * np.sum(np.abs(alpha - alpha_hat))
