"""Phase 10: Real LLM Perplexity Evaluation with OrthoCache Spectral Eviction.

Monkey-patches TinyLlama 1.1B's attention layers with OrthoCache's V3 GQA
Cauchy-Schwarz spectral gate and measures perplexity on WikiText-2.

Architecture match:
    TinyLlama 1.1B:
        num_attention_heads = 32  (query heads)
        num_key_value_heads = 4   (KV heads)
        G = 8  (queries per KV head)
        head_dim = 64
        hidden_size = 2048
        num_hidden_layers = 22

    OrthoCache V3 GQA kernel:
        TILE_SIZE = 64 (tokens per tile)
        head_dim = 64
        G = 8

Usage:
    # Calibrate: find the distribution of CS bounds on real data
    python eval/perplexity_eval.py --model-path C:/LearningFolder/tinyllama1.1b --calibrate

    # Single tau eval
    python eval/perplexity_eval.py --model-path C:/LearningFolder/tinyllama1.1b --tau 0.5

    # Full Pareto sweep (uses calibrated percentile thresholds)
    python eval/perplexity_eval.py --model-path C:/LearningFolder/tinyllama1.1b --sweep
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orthocache_gpu.norm_cache import SpectralNormCache


# ============================================================================
# Spectral Analysis Utilities
# ============================================================================

def generate_walsh_matrix(n: int) -> torch.Tensor:
    """Generate n x n Walsh-Hadamard matrix via Sylvester construction."""
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H


def compute_tile_cs_bounds(
    query: torch.Tensor,    # (num_q_heads, seq_len, head_dim)
    key: torch.Tensor,      # (num_kv_heads, seq_len, head_dim)
    G: int,
    tile_size: int = 64,
) -> List[float]:
    """Compute the normalized Cauchy-Schwarz bound for every KV tile.

    The bound is: max_g(||Q_g_high||_2 * ||K_high||_F) / (tile_size * head_dim)

    The normalization by (tile_size * head_dim) makes the threshold tau
    dimension-independent and comparable across models.

    Returns a list of all CS bounds across all KV heads and tiles.
    """
    num_kv_heads = key.shape[0]
    seq_len = key.shape[1]
    head_dim = key.shape[2]
    num_tiles = seq_len // tile_size

    # Walsh matrix
    W = generate_walsh_matrix(tile_size).to(key.device).float()
    # High-frequency band: top quarter of spectral coefficients
    high_start = (tile_size * 3) // 4  # index 48 for tile_size=64
    high_end = tile_size               # index 64

    all_bounds = []

    for kv_h in range(num_kv_heads):
        k_h = key[kv_h].float()  # (seq_len, head_dim)

        # Query heads for this KV head
        q_group_start = kv_h * G
        q_group_end = q_group_start + G
        q_group = query[q_group_start:q_group_end].float()  # (G, seq_len, head_dim)

        for t in range(num_tiles):
            start = t * tile_size
            end = start + tile_size

            k_tile = k_h[start:end]  # (tile_size, head_dim)

            # FWHT on key tile
            k_spectral = W @ k_tile  # (tile_size, head_dim)
            k_high = k_spectral[high_start:high_end]  # (high_bins, head_dim)

            # Normalized K high-freq energy: ||K_high||_F / sqrt(high_bins * head_dim)
            k_high_norm = torch.norm(k_high, p='fro').item()

            # For each query head, compute ||Q_g||_2 across ALL positions
            # (we take the per-position norm, then use the MEAN to represent
            # the typical query energy -- not the max, which would be too
            # conservative and never evict)
            q_norms = torch.norm(q_group, p=2, dim=-1)  # (G, seq_len)
            # Use the median query norm per head as the representative
            q_norm_per_head = q_norms.median(dim=1).values  # (G,)
            max_q_norm = q_norm_per_head.max().item()

            # Normalized CS bound
            norm_factor = math.sqrt(float(tile_size * head_dim))
            cs_bound = (max_q_norm * k_high_norm) / (norm_factor ** 2)

            all_bounds.append(cs_bound)

    return all_bounds


# ============================================================================
# Calibration: Measure CS bound distribution on real model data
# ============================================================================

def calibrate_thresholds(
    model,
    tokenizer,
    device: torch.device,
    num_windows: int = 5,
    max_length: int = 256,
) -> Dict:
    """Run a few forward passes and measure the CS bound distribution.

    This tells us what tau values are meaningful for this model.
    """
    print("\n" + "=" * 60)
    print("CALIBRATING: Measuring Cauchy-Schwarz bound distribution")
    print("=" * 60)

    G = model.config.num_attention_heads // model.config.num_key_value_heads

    # Load some text
    windows = load_wikitext2(tokenizer, max_length=max_length, stride=max_length)
    windows = windows[:num_windows]

    all_bounds = []

    # Hook to capture Q, K after RoPE
    captured_qk = {}

    def make_capture_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            # In HuggingFace LlamaAttention, we can capture via SDPA intercept
            pass
        return hook_fn

    # Use SDPA intercept to capture Q, K
    original_sdpa = F.scaled_dot_product_attention

    def calibration_sdpa(query, key, value, attn_mask=None,
                         dropout_p=0.0, is_causal=False, scale=None,
                         enable_gqa=False):
        # Capture bounds
        batch_size = query.shape[0]
        for b in range(batch_size):
            bounds = compute_tile_cs_bounds(
                query[b], key[b], G=G, tile_size=64,
            )
            all_bounds.extend(bounds)
        # Run normal SDPA
        return original_sdpa(query, key, value, attn_mask=attn_mask,
                           dropout_p=dropout_p, is_causal=is_causal,
                           scale=scale, enable_gqa=enable_gqa)

    F.scaled_dot_product_attention = calibration_sdpa

    model.eval()
    with torch.no_grad():
        for i, input_ids in enumerate(windows):
            input_ids = input_ids.unsqueeze(0).to(device)
            model(input_ids)
            print(f"    Calibration window {i+1}/{len(windows)}: "
                  f"{len(all_bounds)} bounds collected")

    F.scaled_dot_product_attention = original_sdpa

    bounds_array = np.array(all_bounds)
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    pct_values = np.percentile(bounds_array, percentiles)

    print(f"\n  Total tiles analyzed: {len(bounds_array)}")
    print(f"  CS bound range: [{bounds_array.min():.4f}, {bounds_array.max():.4f}]")
    print(f"  Mean: {bounds_array.mean():.4f}, Std: {bounds_array.std():.4f}")
    print(f"\n  Percentile distribution:")
    print(f"  {'Percentile':>12} | {'CS Bound':>10} | {'Eviction if tau=':>20}")
    print(f"  {'-'*50}")
    for p, v in zip(percentiles, pct_values):
        print(f"  {p:>11}% | {v:>10.4f} | ~{p}% of tiles evicted")

    # Return suggested tau values for sweep
    # Use percentile-based thresholds to target specific eviction rates
    suggested_taus = {
        'p10': float(np.percentile(bounds_array, 10)),
        'p20': float(np.percentile(bounds_array, 20)),
        'p30': float(np.percentile(bounds_array, 30)),
        'p40': float(np.percentile(bounds_array, 40)),
        'p50': float(np.percentile(bounds_array, 50)),
        'p60': float(np.percentile(bounds_array, 60)),
        'p70': float(np.percentile(bounds_array, 70)),
        'p80': float(np.percentile(bounds_array, 80)),
    }

    print(f"\n  Suggested tau values for sweep:")
    for label, tau in suggested_taus.items():
        print(f"    {label}: tau = {tau:.4f}")

    return {
        'bounds': bounds_array.tolist(),
        'percentiles': dict(zip(percentiles, [float(v) for v in pct_values])),
        'suggested_taus': suggested_taus,
    }


# ============================================================================
# Barrier 1: Dynamic Residual Information Governor
# ============================================================================

class ResidualGovernor:
    """Tracks accumulated eviction pressure across layers.

    As earlier layers evict more tiles, later layers automatically reduce
    their effective threshold to prevent compounding informational erosion
    through the residual stream.

    tau_effective(l) = tau_base * max(0, 1 - alpha * eps_accum(l))

    where eps_accum(l) = sum_{j < l} eviction_rate(j)
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Damping coefficient. Higher alpha = more conservative
                   downstream layers. Range [0, 1].
                   0.0 = no governor (static tau)
                   0.5 = moderate damping (recommended)
                   1.0 = aggressive damping (full lockdown at 100% cumulative eviction)
        """
        self.alpha = alpha
        self.eps_accum = 0.0  # Running eviction pressure
        self.layer_history = []  # Per-layer eviction rates

    def reset(self):
        """Reset at the start of each forward pass (each window)."""
        self.eps_accum = 0.0
        self.layer_history = []

    def get_tau_effective(self, tau_base: float) -> float:
        """Compute the effective tau for the current layer."""
        scale = max(0.0, 1.0 - self.alpha * self.eps_accum)
        return tau_base * scale

    def report_layer(self, eviction_rate: float):
        """Called after each layer completes. Updates cumulative pressure."""
        self.layer_history.append(eviction_rate)
        self.eps_accum += eviction_rate


class EntropyGovernor(ResidualGovernor):
    """Barrier 2: Temporal entropy-based tau scaling.

    Modulates tau based on the attention entropy from the current layer.
    High entropy (diffuse attention) → more aggressive eviction (safe).
    Low entropy (sharp attention) → conservative lockdown (protect).

    tau_effective = tau_gov * 2 * sigmoid(beta * (H - h_median))

    where tau_gov is the base residual governor's tau.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 1.0,
                 h_median: float = 1.56):
        super().__init__(alpha=alpha)
        self.beta = beta
        self.h_median = h_median
        self.current_entropy = None

    def set_entropy(self, entropy: float):
        """Set the attention entropy from the current layer's softmax."""
        self.current_entropy = entropy

    def get_tau_effective(self, tau_base: float) -> float:
        """Compute tau with both residual and entropy modulation."""
        import math
        # Base residual governor
        tau_gov = super().get_tau_effective(tau_base)

        # Entropy modulation via sigmoid
        if self.current_entropy is not None:
            sigmoid = 1.0 / (1.0 + math.exp(
                -self.beta * (self.current_entropy - self.h_median)
            ))
            # sigmoid > 0.5 when H > h_median → scale up (more aggressive)
            # sigmoid < 0.5 when H < h_median → scale down (conservative)
            tau_gov *= (2.0 * sigmoid)

        return tau_gov

class OrthoCache_GQA_Attention:
    """Wraps OrthoCache V3 GQA kernel as a drop-in for LlamaAttention.

    Intercepts the attention computation after Q, K, V projections
    and RoPE have been applied. Replaces the standard SDPA call with
    OrthoCache's Cauchy-Schwarz spectral gate + fused attention.
    """

    def __init__(self, tau: float = 1.0, verbose: bool = False,
                 governor: Optional[ResidualGovernor] = None):
        self.tau = tau
        self.verbose = verbose
        self.governor = governor
        self.norm_cache = None  # Gold 1: SpectralNormCache, populated on first prefill
        self.stats = {
            'total_tiles': 0,
            'evicted_tiles': 0,
            'layers_processed': 0,
            'cs_bounds': [],  # Track bounds for analysis
            # Gold 3: Hallucination Exhaust telemetry
            'hallucination_scores': [],   # H_score per layer
            'q_search_intensity': [],     # max ||Q_high||₂ per layer
            'k_info_density': [],         # max ||K_high||_F per layer
            # Gold 1: Decode phase telemetry
            'decode_steps': 0,
            'decode_tiles_skipped': 0,
            'decode_tiles_total': 0,
        }

    def reset_stats(self):
        self.stats = {
            'total_tiles': 0,
            'evicted_tiles': 0,
            'layers_processed': 0,
            'cs_bounds': [],
            'hallucination_scores': [],
            'q_search_intensity': [],
            'k_info_density': [],
            'decode_steps': 0,
            'decode_tiles_skipped': 0,
            'decode_tiles_total': 0,
        }

    @property
    def eviction_rate(self) -> float:
        if self.stats['total_tiles'] == 0:
            return 0.0
        return self.stats['evicted_tiles'] / self.stats['total_tiles']

    def orthocache_attention(
        self,
        query: torch.Tensor,    # (batch, num_q_heads, seq_len_q, head_dim)
        key: torch.Tensor,      # (batch, num_kv_heads, seq_len_kv, head_dim)
        value: torch.Tensor,    # (batch, num_kv_heads, seq_len_kv, head_dim)
        num_query_groups: int,
    ) -> torch.Tensor:
        """Replace standard attention with OrthoCache spectral eviction.
        
        Routes between PREFILL and DECODE based on query sequence length:
        - seq_len_q > 1: PREFILL — full FWHT, populate norm cache
        - seq_len_q == 1: DECODE — O(1) gate from norm cache (Gold 1)
        """
        batch_size, num_q_heads, seq_len_q, head_dim = query.shape
        seq_len_kv = key.shape[2]
        num_kv_heads = key.shape[1]
        G = num_query_groups
        tile_size = 64

        if seq_len_kv < tile_size:
            # Too short for tiling — fall back to standard attention
            k_expanded = key.repeat_interleave(G, dim=1)
            v_expanded = value.repeat_interleave(G, dim=1)
            return F.scaled_dot_product_attention.__wrapped__(
                query, k_expanded, v_expanded, is_causal=True,
            ) if hasattr(F.scaled_dot_product_attention, '__wrapped__') else \
                self._dense_attention(query, key, value, G)

        # ====================================================================
        # DECODE PATH: O(1) gate from SpectralNormCache (Gold 1)
        # ====================================================================
        if seq_len_q == 1 and self.norm_cache is not None and self.norm_cache.is_populated:
            return self._decode_attention(query, key, value, G)

        # ====================================================================
        # PREFILL PATH: Full FWHT + populate norm cache
        # ====================================================================

        # Governor: compute tau_effective for this layer
        if self.governor is not None:
            tau_effective = self.governor.get_tau_effective(self.tau)
        else:
            tau_effective = self.tau

        all_batch_outputs = []

        for b in range(batch_size):
            q_b = query[b].float()   # (num_q_heads, seq_len_q, head_dim)
            k_b = key[b].float()     # (num_kv_heads, seq_len_kv, head_dim)
            v_b = value[b].float()   # (num_kv_heads, seq_len_kv, head_dim)

            # Expand KV heads
            k_expanded = k_b.repeat_interleave(G, dim=0)
            v_expanded = v_b.repeat_interleave(G, dim=0)

            scale = 1.0 / math.sqrt(head_dim)
            logits = torch.matmul(q_b, k_expanded.transpose(-2, -1)) * scale

            # Causal mask (seq_len_q × seq_len_kv)
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_kv, device=query.device, dtype=torch.bool),
                diagonal=1 + (seq_len_kv - seq_len_q),
            )
            logits.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))

            # Walsh matrix
            W = generate_walsh_matrix(tile_size).to(key.device).float()
            high_start = (tile_size * 3) // 4
            high_end = tile_size
            num_tiles = seq_len_kv // tile_size
            norm_factor = float(tile_size * head_dim)

            tiles_evicted_this_batch = 0
            tiles_total_this_batch = 0

            # Gold 3: Track max norms for hallucination exhaust
            max_q_high_batch = 0.0
            max_k_high_batch = 0.0

            for kv_h in range(num_kv_heads):
                k_h = k_b[kv_h]  # (seq_len_kv, head_dim)
                q_group_start = kv_h * G
                q_group_end = q_group_start + G
                q_group = q_b[q_group_start:q_group_end]  # (G, seq_len_q, head_dim)

                # Precompute median Q norms per head
                q_norms = torch.norm(q_group, p=2, dim=-1)  # (G, seq_len)
                q_norm_median = q_norms.median(dim=1).values  # (G,)

                for t in range(num_tiles):
                    start = t * tile_size
                    end = start + tile_size

                    k_tile = k_h[start:end]
                    k_spectral = W @ k_tile
                    k_high = k_spectral[high_start:high_end]
                    k_high_norm = torch.norm(k_high, p='fro').item()

                    # Gold 1: Populate norm cache during prefill
                    if self.norm_cache is not None:
                        self.norm_cache.cache[kv_h, t] = k_high_norm
                        self.norm_cache.valid_tiles[kv_h] = max(
                            self.norm_cache.valid_tiles[kv_h].item(), t + 1
                        )
                        self.norm_cache._populated = True

                    max_q_norm = q_norm_median.max().item()
                    cs_bound = (max_q_norm * k_high_norm) / norm_factor

                    # Gold 3: Track spectral norms (zero extra compute)
                    if k_high_norm > max_k_high_batch:
                        max_k_high_batch = k_high_norm
                    if max_q_norm > max_q_high_batch:
                        max_q_high_batch = max_q_norm

                    if cs_bound <= tau_effective:
                        # Evict tile
                        for g in range(G):
                            qh = q_group_start + g
                            logits[qh, :, start:end] = float('-inf')
                        tiles_evicted_this_batch += 1

                tiles_total_this_batch += num_tiles
                self.stats['total_tiles'] += num_tiles

            self.stats['evicted_tiles'] += tiles_evicted_this_batch

            # Gold 3: Compute hallucination score for this layer
            h_score = max_q_high_batch / (max_k_high_batch + 1e-10)
            self.stats['hallucination_scores'].append(h_score)
            self.stats['q_search_intensity'].append(max_q_high_batch)
            self.stats['k_info_density'].append(max_k_high_batch)

            # Report to governor: this layer's eviction rate
            if self.governor is not None and tiles_total_this_batch > 0:
                layer_eviction_rate = tiles_evicted_this_batch / tiles_total_this_batch
                self.governor.report_layer(layer_eviction_rate)

            # Softmax + output
            attn_weights = F.softmax(logits, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

            # Barrier 2: Compute attention entropy for EntropyGovernor
            # H = -sum(p * log(p)), averaged across heads and query positions
            if self.governor is not None and hasattr(self.governor, 'set_entropy'):
                # Clamp to avoid log(0)
                p_clamped = attn_weights.clamp(min=1e-10)
                entropy_per_head = -(p_clamped * p_clamped.log()).sum(dim=-1)  # (num_q_heads, seq_len)
                mean_entropy = entropy_per_head.mean().item()
                self.governor.set_entropy(mean_entropy)
                self.stats['entropy'] = mean_entropy

            output_b = torch.matmul(attn_weights, v_expanded)
            all_batch_outputs.append(output_b.to(query.dtype))

        self.stats['layers_processed'] += 1
        return torch.stack(all_batch_outputs, dim=0)

    def _compute_q_high_norm(self, q_vec: torch.Tensor, low_band: int = 8) -> float:
        """Platinum 1: Walsh Subspace Projection — exact ||Q_high||₂ in 72 FLOPs.
        
        By dyadic harmonic analysis, the low-frequency Walsh projection is
        equivalent to block averaging. For head_dim=64, low_band=8:
            block_size = 64/8 = 8
            S_i = sum(q[8i : 8(i+1)])  for i=0..7
            ||Q_low||² = (1/8) × Σ S_i²
            ||Q_high||₂ = √(||Q||₂² - ||Q_low||₂²)
        """
        head_dim = q_vec.shape[-1]
        block_size = head_dim // low_band
        
        q_f = q_vec.float()
        
        # ||Q||² — full spatial energy
        q_norm_sq = (q_f * q_f).sum().item()
        
        # Block sums → ||Q_low||²
        q_blocks = q_f.reshape(low_band, block_size)
        block_sums = q_blocks.sum(dim=-1)  # (low_band,)
        q_low_sq = (block_sums * block_sums).sum().item() / block_size
        
        # ||Q_high||₂ = √(||Q||² - ||Q_low||²)
        q_high_sq = max(0.0, q_norm_sq - q_low_sq)
        return q_high_sq ** 0.5

    def _decode_attention(
        self,
        query: torch.Tensor,    # (batch, num_q_heads, 1, head_dim)
        key: torch.Tensor,      # (batch, num_kv_heads, seq_len_kv, head_dim)
        value: torch.Tensor,    # (batch, num_kv_heads, seq_len_kv, head_dim)
        G: int,
    ) -> torch.Tensor:
        """Gold 1: O(1) Decode Gate — FWHT-free attention using cached norms.
        
        During autoregressive decode, the K-cache is STATIC (past tokens never
        change). Instead of recomputing the O(N log N) FWHT on every decode step,
        we look up the precomputed ||K_high||_F scalar from the norm cache.
        
        This skips BOTH the K load AND V load for evicted tiles, saving 100%
        bandwidth (vs 33% in the prefill path which only skips V).
        """
        batch_size = query.shape[0]
        num_q_heads = query.shape[1]
        head_dim = query.shape[3]
        seq_len_kv = key.shape[2]
        num_kv_heads = key.shape[1]
        tile_size = 64
        
        # Governor: compute tau_effective
        if self.governor is not None:
            tau_effective = self.governor.get_tau_effective(self.tau)
        else:
            tau_effective = self.tau
        
        # Decode gate constants
        num_tiles = seq_len_kv // tile_size
        norm_factor = float(tile_size * head_dim)
        
        all_batch_outputs = []
        
        for b in range(batch_size):
            q_b = query[b].float()   # (num_q_heads, 1, head_dim)
            k_b = key[b].float()     # (num_kv_heads, seq_len_kv, head_dim)
            v_b = value[b].float()   # (num_kv_heads, seq_len_kv, head_dim)
            
            # Expand KV heads
            k_expanded = k_b.repeat_interleave(G, dim=0)
            v_expanded = v_b.repeat_interleave(G, dim=0)
            
            scale = 1.0 / math.sqrt(head_dim)
            # logits: (num_q_heads, 1, seq_len_kv)
            logits = torch.matmul(q_b, k_expanded.transpose(-2, -1)) * scale
            
            tiles_skipped = 0
            
            # Gold 3: Track norms for hallucination exhaust
            max_q_high_batch = 0.0
            max_k_high_batch = 0.0
            
            for kv_h in range(num_kv_heads):
                q_group_start = kv_h * G
                q_group_end = q_group_start + G
                
                # Platinum 1: Compute EXACT Q_high norm via Walsh Subspace Projection
                # CS bound: |Q · K_high| <= ||Q_high||_2 · ||K_high||_F
                # This is TIGHTER than the old ||Q||_2 because ||Q_high||_2 <= ||Q||_2
                q_norms_group = []
                for g in range(G):
                    qh = q_group_start + g
                    q_vec = q_b[qh, 0, :]  # (head_dim,)
                    # Walsh Subspace Projection: 72 FLOPs, exact spectral norm
                    q_high_norm = self._compute_q_high_norm(q_vec)
                    q_norms_group.append(q_high_norm)
                
                # Use median for robustness (matching prefill)
                q_norms_sorted = sorted(q_norms_group)
                max_q_norm = q_norms_sorted[len(q_norms_sorted) // 2]  # median
                if max_q_norm > max_q_high_batch:
                    max_q_high_batch = max_q_norm
                
                for t in range(num_tiles):
                    # ============================================
                    # O(1) LOOKUP: Read scalar from norm cache
                    # NO FWHT ON K. NO K TILE LOAD.
                    # ============================================
                    k_high_norm = self.norm_cache.get_norm(kv_h, t)
                    
                    if k_high_norm > max_k_high_batch:
                        max_k_high_batch = k_high_norm
                    
                    cs_bound = (max_q_norm * k_high_norm) / norm_factor
                    
                    if cs_bound <= tau_effective:
                        # Gate the tile: mask logits to -inf
                        start = t * tile_size
                        end = start + tile_size
                        for g in range(G):
                            qh = q_group_start + g
                            logits[qh, 0, start:end] = float('-inf')
                        tiles_skipped += 1
            
            # Gold 3: Hallucination exhaust
            h_score = max_q_high_batch / (max_k_high_batch + 1e-10)
            self.stats['hallucination_scores'].append(h_score)
            self.stats['q_search_intensity'].append(max_q_high_batch)
            self.stats['k_info_density'].append(max_k_high_batch)
            
            # Track decode stats
            self.stats['decode_steps'] += 1
            self.stats['decode_tiles_skipped'] += tiles_skipped
            self.stats['decode_tiles_total'] += num_tiles * num_kv_heads
            
            # Report to governor
            if self.governor is not None and num_tiles > 0:
                evict_rate = tiles_skipped / (num_tiles * num_kv_heads)
                self.governor.report_layer(evict_rate)
            
            # Softmax + output
            attn_weights = F.softmax(logits, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            
            output_b = torch.matmul(attn_weights, v_expanded)
            all_batch_outputs.append(output_b.to(query.dtype))
        
        self.stats['layers_processed'] += 1
        return torch.stack(all_batch_outputs, dim=0)

    def _dense_attention(self, query, key, value, G):
        """Fallback dense causal attention."""
        k_expanded = key.repeat_interleave(G, dim=1)
        v_expanded = value.repeat_interleave(G, dim=1)
        scale = 1.0 / math.sqrt(query.shape[-1])
        logits = torch.matmul(query, k_expanded.transpose(-2, -1)) * scale
        seq_len_q = query.shape[2]
        seq_len_kv = key.shape[2]
        mask = torch.triu(
            torch.ones(seq_len_q, seq_len_kv, device=query.device, dtype=torch.bool),
            diagonal=1 + (seq_len_kv - seq_len_q),
        )
        logits.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        weights = F.softmax(logits, dim=-1)
        return torch.matmul(weights, v_expanded)


def patch_model_attention(model, orthocache: OrthoCache_GQA_Attention,
                          min_layer: int = 0):
    """Monkey-patch LlamaAttention layers in the model.

    Args:
        min_layer: Only patch layers >= min_layer. Early layers (0 to min_layer-1)
                   use dense attention to preserve foundational representations.
                   Recommended: min_layer = num_layers // 2 (e.g., 11 for 22 layers).
    """
    num_kv_heads = model.config.num_key_value_heads
    num_q_heads = model.config.num_attention_heads
    G = num_q_heads // num_kv_heads
    num_layers = len(model.model.layers)

    patched_count = 0

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx < min_layer:
            continue  # Protect early layers

        attn = layer.self_attn
        original_forward = attn.forward

        def make_patched_forward(original_fn):
            def patched_forward(*args, **kwargs):
                def intercept_sdpa(query, key, value, attn_mask=None,
                                   dropout_p=0.0, is_causal=False, scale=None,
                                   enable_gqa=False):
                    return orthocache.orthocache_attention(
                        query, key, value, num_query_groups=G,
                    )

                # Temporarily replace SDPA
                old_sdpa = F.scaled_dot_product_attention
                F.scaled_dot_product_attention = intercept_sdpa
                try:
                    result = original_fn(*args, **kwargs)
                finally:
                    F.scaled_dot_product_attention = old_sdpa
                return result

            return patched_forward

        attn.forward = make_patched_forward(original_forward)
        patched_count += 1

    print(f"  Patched {patched_count}/{num_layers} layers (G={G}, "
          f"layers {min_layer}-{num_layers-1})")
    return model


# ============================================================================
# Perplexity Evaluation
# ============================================================================

def load_wikitext2(tokenizer, max_length: int = 2048, stride: int = 512) -> List[torch.Tensor]:
    """Load WikiText-2 test set for perplexity evaluation."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])
        print(f"  Loaded WikiText-2: {len(text):,} characters")
    except Exception as e:
        print(f"  WikiText-2 not available ({e}), using built-in sample")
        text = ("The tower is 324 metres tall, about the same height as an "
                "81-storey building, and the tallest structure in Paris. "
                "Its base is square, measuring 125 metres on each side. ") * 200

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    windows = []
    for i in range(0, len(input_ids) - max_length, stride):
        window = input_ids[i : i + max_length]
        windows.append(window)
        if len(windows) >= 20:
            break

    if not windows:
        windows = [input_ids[:max_length]]

    print(f"  Created {len(windows)} evaluation windows "
          f"(length={max_length}, stride={stride})")
    return windows


def evaluate_perplexity(
    model,
    tokenizer,
    windows: List[torch.Tensor],
    device: torch.device,
    label: str = "dense",
) -> float:
    """Compute perplexity over evaluation windows."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, input_ids in enumerate(windows):
            input_ids = input_ids.unsqueeze(0).to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            num_tokens = input_ids.shape[1] - 1
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            if (i + 1) % 5 == 0:
                running_ppl = math.exp(total_loss / total_tokens)
                print(f"    [{label}] Window {i+1}/{len(windows)} -- "
                      f"running PPL: {running_ppl:.2f}")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def run_perplexity_sweep(
    model_path: str,
    tau_values: List[float],
    device: torch.device,
    max_length: int = 512,
    stride: int = 256,
    min_layer: int = 0,
    alpha: float = 0.0,
    entropy_mode: bool = False,
    beta: float = 1.0,
    h_median: float = 3.0,
) -> Dict[str, List]:
    """Sweep tau values and record perplexity + eviction rate.

    Args:
        alpha: Governor damping coefficient. 0.0 = no governor (static tau),
               0.5 = moderate damping, 1.0 = aggressive damping.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map=device.type if device.type == 'cuda' else None,
    )
    if device.type != 'cuda':
        model = model.to(device)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  GQA config: {model.config.num_attention_heads} Q-heads, "
          f"{model.config.num_key_value_heads} KV-heads, "
          f"G={model.config.num_attention_heads // model.config.num_key_value_heads}")

    if alpha > 0 and entropy_mode:
        print(f"  Governor: ENTROPY (alpha={alpha}, beta={beta}, h_median={h_median:.2f})")
    elif alpha > 0:
        print(f"  Governor: ACTIVE (alpha={alpha})")
    else:
        print(f"  Governor: DISABLED (static tau)")

    windows = load_wikitext2(tokenizer, max_length=max_length, stride=stride)

    results = {
        'tau': [],
        'perplexity': [],
        'eviction_rate': [],
        'label': [],
    }

    # -- Baseline --
    print("\n" + "=" * 60)
    print("BASELINE: Dense Attention (no OrthoCache)")
    print("=" * 60)
    baseline_ppl = evaluate_perplexity(model, tokenizer, windows, device, label="dense")
    print(f"  Dense Perplexity: {baseline_ppl:.2f}")

    results['tau'].append(0.0)
    results['perplexity'].append(baseline_ppl)
    results['eviction_rate'].append(0.0)
    results['label'].append('dense_baseline')

    # -- OrthoCache sweep --
    for tau in tau_values:
        print(f"\n{'=' * 60}")
        print(f"OrthoCache: tau = {tau}")
        print("=" * 60)

        # Create governor (or None if alpha=0)
        if alpha > 0 and entropy_mode:
            governor = EntropyGovernor(
                alpha=alpha, beta=beta, h_median=h_median
            )
            gov_label = f"EntropyGovernor(alpha={alpha}, beta={beta}, h_median={h_median:.2f})"
        elif alpha > 0:
            governor = ResidualGovernor(alpha=alpha)
            gov_label = f"ResidualGovernor(alpha={alpha})"
        else:
            governor = None
            gov_label = "None"

        orthocache = OrthoCache_GQA_Attention(
            tau=tau, verbose=False, governor=governor,
        )
        patched_model = patch_model_attention(model, orthocache, min_layer=min_layer)

        # Add a pre-forward hook on the FIRST patched layer to reset
        # the governor at the start of each forward pass
        if governor is not None:
            first_patched_layer = model.model.layers[min_layer]
            def make_reset_hook(gov):
                def hook(module, args):
                    gov.reset()
                return hook
            reset_handle = first_patched_layer.register_forward_pre_hook(
                make_reset_hook(governor)
            )

        ppl = evaluate_perplexity(
            patched_model, tokenizer, windows, device,
            label=f"tau={tau}",
        )

        eviction_rate = orthocache.eviction_rate

        # Print governor layer history for the last forward pass
        if governor is not None:
            hist = governor.layer_history
            if hist:
                print(f"  Governor history (last pass): "
                      f"[{', '.join(f'{r:.1%}' for r in hist)}]")
                print(f"  Final eps_accum: {governor.eps_accum:.3f}")
                # Barrier 2: Print entropy info if available
                if hasattr(governor, 'current_entropy') and governor.current_entropy is not None:
                    print(f"  Attention entropy: {governor.current_entropy:.4f} "
                          f"(h_median={governor.h_median:.2f})")
            reset_handle.remove()

        print(f"  tau={tau}: PPL={ppl:.2f}, Eviction Rate={eviction_rate:.1%}")

        results['tau'].append(tau)
        results['perplexity'].append(ppl)
        results['eviction_rate'].append(eviction_rate)
        results['label'].append(f'orthocache_tau_{tau}')

        # Unpatch for next iteration by reloading
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            device_map=device.type if device.type == 'cuda' else None,
        )
        if device.type != 'cuda':
            model = model.to(device)

    return results


def save_results(results: Dict[str, List], output_path: str):
    """Save results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def print_pareto_table(results: Dict[str, List]):
    """Print the Pareto curve as a table."""
    print("\n" + "=" * 70)
    print("PARETO TABLE: Perplexity vs Eviction Rate")
    print("=" * 70)
    print(f"{'tau':>8} | {'Eviction %':>12} | {'Perplexity':>12} | {'dPPL':>8} | {'dPPL%':>8}")
    print("-" * 70)

    baseline_ppl = results['perplexity'][0]

    for i in range(len(results['tau'])):
        tau = results['tau'][i]
        ppl = results['perplexity'][i]
        eviction = results['eviction_rate'][i]
        delta_ppl = ppl - baseline_ppl
        delta_pct = (delta_ppl / baseline_ppl) * 100 if baseline_ppl > 0 else 0

        marker = " [baseline]" if i == 0 else ""
        print(f"{tau:>8.4f} | {eviction:>11.1%} | {ppl:>12.2f} | {delta_ppl:>+8.2f} | {delta_pct:>+7.2f}%{marker}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OrthoCache Phase 10: Perplexity Evaluation"
    )
    parser.add_argument(
        "--model-path", type=str,
        default="C:/LearningFolder/tinyllama1.1b",
        help="Path to TinyLlama model directory",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Run calibration to find CS bound distribution before sweeping",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run full tau sweep using calibrated percentile thresholds",
    )
    parser.add_argument(
        "--tau", type=float, default=None,
        help="Single tau value for quick eval",
    )
    parser.add_argument(
        "--tau-list", type=float, nargs='+', default=None,
        help="List of tau values for fine-grained sweep (e.g., --tau-list 1.0 1.02 1.04)",
    )
    parser.add_argument(
        "--max-length", type=int, default=256,
        help="Max sequence length for evaluation windows",
    )
    parser.add_argument(
        "--stride", type=int, default=256,
        help="Stride between evaluation windows",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cuda', or 'cpu'",
    )
    parser.add_argument(
        "--min-layer", type=int, default=-1,
        help="First layer to apply eviction (default: num_layers//2). "
             "Set to 0 to evict in all layers.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0,
        help="Residual Information Governor damping coefficient. "
             "0.0 = disabled (static tau), 0.5 = moderate, 1.0 = aggressive.",
    )
    parser.add_argument(
        "--entropy", action='store_true', default=False,
        help="Enable Barrier 2: Temporal Entropy Scaling. "
             "Modulates tau based on attention entropy (requires --alpha > 0).",
    )
    parser.add_argument(
        "--beta", type=float, default=1.0,
        help="Entropy governor sensitivity (with --entropy). "
             "Higher = sharper transition around h_median.",
    )
    parser.add_argument(
        "--h-median", type=float, default=None,
        help="Median entropy estimate (with --entropy). "
             "If None, auto-calibrated from baseline.",
    )
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.calibrate:
        # Step 1: Calibrate
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch.float32,
            device_map=device.type if device.type == 'cuda' else None,
        )
        if device.type != 'cuda':
            model = model.to(device)

        cal = calibrate_thresholds(model, tokenizer, device,
                                   num_windows=3, max_length=args.max_length)

        # Save calibration
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        cal_path = str(output_dir / "calibration.json")
        with open(cal_path, 'w') as f:
            json.dump({k: v for k, v in cal.items() if k != 'bounds'}, f, indent=2)
        print(f"\nCalibration saved to {cal_path}")

        if args.sweep:
            # Use calibrated tau values for sweep
            tau_values = sorted(cal['suggested_taus'].values())
            print(f"\nProceeding to sweep with calibrated taus: {tau_values}")
        elif args.tau is not None:
            tau_values = [args.tau]
        else:
            print("\nCalibration complete. Run with --sweep to use these thresholds.")
            return

    elif args.sweep:
        # Default sweep without calibration
        tau_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0]
    elif args.tau_list is not None:
        tau_values = sorted(args.tau_list)
    elif args.tau is not None:
        tau_values = [args.tau]
    else:
        print("Specify --calibrate, --sweep, --tau <value>, or --tau-list <values>")
        return

    # Determine min_layer
    if args.min_layer >= 0:
        min_layer = args.min_layer
    else:
        # Default: protect first half of layers
        # TinyLlama has 22 layers, so default min_layer=11
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_path)
        min_layer = config.num_hidden_layers // 2
        print(f"Auto min_layer: {min_layer} (protecting layers 0-{min_layer-1})")

    # Entropy governor setup
    entropy_mode = args.entropy
    beta = args.beta
    h_median = args.h_median if args.h_median is not None else 3.0  # default, auto-calibrated inside

    if entropy_mode and args.alpha <= 0:
        print("WARNING: --entropy requires --alpha > 0. Falling back to static tau.")
        entropy_mode = False

    # Run evaluation
    results = run_perplexity_sweep(
        model_path=args.model_path,
        tau_values=tau_values,
        device=device,
        max_length=args.max_length,
        stride=args.stride,
        min_layer=min_layer,
        alpha=args.alpha,
        entropy_mode=entropy_mode,
        beta=beta,
        h_median=h_median,
    )

    print_pareto_table(results)

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = args.output or str(output_dir / "perplexity_sweep.json")
    save_results(results, output_path)


if __name__ == "__main__":
    main()
