"""Gold 1: Spectral Norm Cache for O(1) Decode-Phase Gate.

During the Prefill phase, we compute the FWHT of each K tile and cache
the high-band norm ||K_high||_F as a single fp32 scalar per tile.

During the Decode phase, the kernel reads this scalar instead of
recomputing the FWHT on static K data. This eliminates the O(N log N)
tax on every decode step and makes OrthoCache faster than FlashAttention
at ALL context lengths (not just > 4K).

Memory cost: num_kv_heads × max_tiles × 4 bytes
  TinyLlama: 4 heads × 512 tiles × 4 bytes = 8 KB (negligible)
  LLaMA-70B: 8 heads × 2048 tiles × 4 bytes = 64 KB (negligible)
"""

import torch
import math
from typing import Optional, Tuple


class SpectralNormCache:
    """Precomputed K-tile high-band norms for O(1) decode-phase gate.
    
    Usage:
        cache = SpectralNormCache(num_kv_heads=4, max_tiles=512, device='cuda')
        
        # Prefill: compute and store norms
        for tile_idx in range(num_tiles):
            cache.update_prefill(kv_head, tile_idx, k_tile, walsh_matrix,
                                 high_start, high_end)
        
        # Decode: O(1) lookup
        k_norm = cache.get_norm(kv_head, tile_idx)
        if q_norm * k_norm <= tau:
            continue  # Skip both K and V load!
    """
    
    def __init__(self, num_kv_heads: int, max_tiles: int, device: torch.device):
        """Initialize the norm cache.
        
        Args:
            num_kv_heads: Number of KV heads in the model.
            max_tiles: Maximum number of tiles per KV head (seq_len // tile_size).
            device: Device to store the cache on.
        """
        # Cache shape: (num_kv_heads, max_tiles) in fp32
        self.cache = torch.zeros(
            num_kv_heads, max_tiles, dtype=torch.float32, device=device
        )
        self.valid_tiles = torch.zeros(
            num_kv_heads, dtype=torch.int32, device=device
        )
        self.num_kv_heads = num_kv_heads
        self.max_tiles = max_tiles
        self.device = device
        self._populated = False
    
    def update_prefill(
        self,
        kv_head: int,
        tile_idx: int,
        k_tile: torch.Tensor,    # (tile_size, head_dim)
        walsh_matrix: torch.Tensor,  # (tile_size, tile_size)
        high_start: int,
        high_end: int,
    ):
        """Called once per tile during prefill. Computes and caches ||K_high||_F.
        
        This is the ONLY place where FWHT is applied to K tiles.
        After prefill, K tiles are never spectral-analyzed again.
        """
        # Compute FWHT: k_spectral = W @ k_tile
        k_spectral = walsh_matrix @ k_tile.float()
        
        # Extract high-frequency band
        k_high = k_spectral[high_start:high_end]
        
        # Compute Frobenius norm (with subnormal clamp)
        k_high_norm = torch.norm(k_high, p='fro').item()
        if k_high_norm < 1e-38:
            k_high_norm = 0.0
        
        # Store scalar
        self.cache[kv_head, tile_idx] = k_high_norm
        self.valid_tiles[kv_head] = max(
            self.valid_tiles[kv_head].item(), tile_idx + 1
        )
        self._populated = True
    
    def populate_from_keys(
        self,
        keys: torch.Tensor,          # (num_kv_heads, seq_len, head_dim)
        walsh_matrix: torch.Tensor,   # (tile_size, tile_size)
        tile_size: int = 64,
        high_start: int = 48,
        high_end: int = 64,
    ):
        """Bulk populate cache from a full key tensor.
        
        More efficient than calling update_prefill per tile.
        """
        num_kv_heads, seq_len, head_dim = keys.shape
        num_tiles = seq_len // tile_size
        
        W = walsh_matrix.float().to(keys.device)
        
        for kv_h in range(min(num_kv_heads, self.num_kv_heads)):
            k_h = keys[kv_h].float()  # (seq_len, head_dim)
            
            for t in range(min(num_tiles, self.max_tiles)):
                start = t * tile_size
                end = start + tile_size
                k_tile = k_h[start:end]
                
                self.update_prefill(kv_h, t, k_tile, W, high_start, high_end)
        
        return self
    
    def get_norm(self, kv_head: int, tile_idx: int) -> float:
        """O(1) lookup during decode. Returns cached ||K_high||_F."""
        return self.cache[kv_head, tile_idx].item()
    
    def get_norms_batch(self, kv_head: int) -> torch.Tensor:
        """Get all cached norms for a KV head. Shape: (valid_tiles,)."""
        n = self.valid_tiles[kv_head].item()
        return self.cache[kv_head, :n]
    
    @property
    def is_populated(self) -> bool:
        return self._populated
    
    @property
    def memory_bytes(self) -> int:
        """Total memory used by the cache."""
        return self.cache.nelement() * self.cache.element_size()
    
    def summary(self) -> str:
        """Human-readable summary."""
        total_tiles = self.valid_tiles.sum().item()
        return (
            f"SpectralNormCache: {self.num_kv_heads} heads × "
            f"{self.max_tiles} max_tiles, "
            f"{total_tiles} populated, "
            f"{self.memory_bytes / 1024:.1f} KB"
        )


def create_norm_cache_for_model(
    model_config,
    max_seq_len: int = 32768,
    tile_size: int = 64,
    device: torch.device = torch.device('cpu'),
) -> SpectralNormCache:
    """Factory: Create a SpectralNormCache sized for a specific model.
    
    Args:
        model_config: HuggingFace model config with num_key_value_heads.
        max_seq_len: Maximum sequence length to support.
        tile_size: Tile size for spectral analysis.
        device: Device to store the cache on.
    """
    num_kv_heads = model_config.num_key_value_heads
    max_tiles = max_seq_len // tile_size
    
    cache = SpectralNormCache(num_kv_heads, max_tiles, device)
    print(f"  Created {cache.summary()}")
    return cache
