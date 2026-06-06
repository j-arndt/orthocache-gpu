"""Platinum 4: Spectral Pre-Training Regularizer.

OrthoCache currently works DESPITE the model being trained densely. The
eviction profiles (L11=57%, L12=33%, L13+=<5%) are emergent, not optimized.

This module injects spectral awareness directly into the training loop by
adding a regularization term that penalizes high-frequency energy in the
KV-cache:

    L_total = L_CE + lambda * sum_l ||K_high^(l)||_F^2

This forces the optimizer to minimize high-frequency K energy in routing
layers, making the KV-cache natively sparse-friendly. The result: a
Transformer that LEARNS to be sparse from the start.

Usage:
    regularizer = SpectralRegularizer(
        lambda_reg=0.01,
        target_layers=[11, 12, 13, 14, 15],
        low_band=8,
    )
    
    # During training forward pass:
    for layer_idx, K in enumerate(key_states):
        reg_loss += regularizer.compute_penalty(K, layer_idx)
    
    total_loss = ce_loss + reg_loss
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional


class SpectralRegularizer(nn.Module):
    """Spectral Pre-Training Regularizer for Walsh-Hadamard Sparsity.
    
    Adds a differentiable penalty on the high-frequency Frobenius norm of
    K-states in designated layers. The penalty is computed entirely via
    block averaging (Walsh Subspace Projection), so it requires NO FWHT
    and only ~72 FLOPs per head_dim vector.
    
    The regularizer encourages the model to concentrate semantic content
    in the low-frequency Walsh bands, making OrthoCache's eviction
    mathematically lossless by construction.
    """
    
    def __init__(
        self,
        lambda_reg: float = 0.01,
        target_layers: Optional[List[int]] = None,
        low_band: int = 8,
        per_layer_weights: Optional[Dict[int, float]] = None,
    ):
        """Initialize the Spectral Regularizer.
        
        Args:
            lambda_reg: Global regularization strength.
            target_layers: Which layers to regularize. Default: [11-15] (routing layers).
            low_band: Number of low-frequency Walsh coefficients.
            per_layer_weights: Optional per-layer scaling. E.g. {11: 2.0, 12: 1.5}
                               for stronger regularization on routing chokepoints.
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.target_layers = target_layers or list(range(11, 16))
        self.low_band = low_band
        self.per_layer_weights = per_layer_weights or {}
        
        # Telemetry
        self.penalty_history: Dict[int, List[float]] = {l: [] for l in self.target_layers}
    
    def compute_k_high_energy(
        self,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ||K_high||_F^2 for a batch of K vectors using Walsh Subspace Projection.
        
        Args:
            k: Key tensor, shape (batch, num_kv_heads, seq_len, head_dim)
        
        Returns:
            k_high_sq: Scalar tensor — total high-frequency energy (differentiable).
        """
        head_dim = k.shape[-1]
        block_size = head_dim // self.low_band
        
        assert head_dim % self.low_band == 0, (
            f"head_dim ({head_dim}) must be divisible by low_band ({self.low_band})"
        )
        
        # ||K||_F^2 — total spatial energy
        k_norm_sq = (k * k).sum()
        
        # Block sums for low-frequency projection
        k_blocks = k.reshape(*k.shape[:-1], self.low_band, block_size)
        block_sums = k_blocks.sum(dim=-1)  # (..., low_band)
        
        # ||K_low||_F^2 = (1/block_size) * sum(S_i^2)
        k_low_sq = (block_sums * block_sums).sum() / block_size
        
        # ||K_high||_F^2 = ||K||_F^2 - ||K_low||_F^2
        k_high_sq = k_norm_sq - k_low_sq
        
        return k_high_sq
    
    def compute_penalty(
        self,
        k: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute regularization penalty for one layer.
        
        Args:
            k: Key tensor, shape (batch, num_kv_heads, seq_len, head_dim)
            layer_idx: Which layer this is (for per-layer weighting).
        
        Returns:
            penalty: Scalar tensor (differentiable, for backprop).
        """
        if layer_idx not in self.target_layers:
            return torch.tensor(0.0, device=k.device, requires_grad=False)
        
        k_high_sq = self.compute_k_high_energy(k)
        
        # Per-layer weight
        layer_weight = self.per_layer_weights.get(layer_idx, 1.0)
        
        # Normalize by total elements for scale-invariance
        num_elements = k.numel()
        penalty = self.lambda_reg * layer_weight * k_high_sq / num_elements
        
        # Telemetry
        self.penalty_history[layer_idx].append(penalty.item())
        
        return penalty
    
    def forward(
        self,
        key_states: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute total spectral regularization penalty across all layers.
        
        Args:
            key_states: Dict mapping layer_idx -> K tensor.
        
        Returns:
            total_penalty: Scalar tensor to add to cross-entropy loss.
        """
        total = torch.tensor(0.0, device=next(iter(key_states.values())).device)
        
        for layer_idx, k in key_states.items():
            total = total + self.compute_penalty(k, layer_idx)
        
        return total
    
    def get_telemetry(self) -> dict:
        """Return telemetry for logging."""
        return {
            'lambda_reg': self.lambda_reg,
            'target_layers': self.target_layers,
            'low_band': self.low_band,
            'per_layer_weights': self.per_layer_weights,
            'penalty_history': {
                k: v[-10:] for k, v in self.penalty_history.items()
            },
        }
    
    @staticmethod
    def validate_gradient_flow(head_dim: int = 64, low_band: int = 8):
        """Validate that gradients flow correctly through the Walsh projection.
        
        This is a unit test: creates a small K tensor with requires_grad=True,
        computes the spectral penalty, calls backward(), and verifies that
        gradients are non-zero and finite.
        
        Returns:
            dict with validation results.
        """
        # Small test K
        k = torch.randn(1, 4, 16, head_dim, requires_grad=True)
        
        reg = SpectralRegularizer(lambda_reg=0.1, target_layers=[0], low_band=low_band)
        penalty = reg.compute_penalty(k, layer_idx=0)
        
        # Backward
        penalty.backward()
        
        grad = k.grad
        
        results = {
            'penalty_value': penalty.item(),
            'grad_shape': list(grad.shape),
            'grad_norm': grad.norm().item(),
            'grad_finite': torch.isfinite(grad).all().item(),
            'grad_nonzero': (grad != 0).any().item(),
            'passed': torch.isfinite(grad).all().item() and (grad != 0).any().item(),
        }
        
        return results
