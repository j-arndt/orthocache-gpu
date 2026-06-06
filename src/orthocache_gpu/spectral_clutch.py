"""Platinum 2: Spectral Auto-Clutch — Autonomous Cognitive Transmission.

The Dual-Regime discovery proved that:
  - Generative mode needs alpha=0.3 (governor engaged, protect perplexity)
  - RAG/Hunter mode needs alpha=0.0 (governor off, maximize recall)

The model already KNOWS which mode it's in. When it searches for a fact,
||Q_high||_2 spikes (measured in Gold 3). The Spectral Auto-Clutch uses
this signal to autonomously modulate alpha:

    alpha_t = alpha_base * exp(-gamma * ||Q_t,high||_2)

  - Low Q_high (generative prose): alpha stays at ~0.3, governor engaged
  - High Q_high (fact retrieval): alpha drops toward 0, governor disengages
  - Transition is smooth, instantaneous, and mathematically continuous

The LLM is now driving itself.

Usage:
    clutch = SpectralAutoClutch(alpha_base=0.3, gamma=100.0)
    
    # During decode step:
    q_high_norm = compute_q_high_norm_exact(q_vec)  # Platinum 1
    alpha_t = clutch.compute_alpha(q_high_norm)
    governor.alpha = alpha_t  # Dynamic alpha for this step
"""

import math
from typing import List, Optional, Tuple


class SpectralAutoClutch:
    """Autonomous Spectral Transmission for Dual-Regime switching.
    
    Continuously modulates the Residual Governor's alpha parameter based on
    the per-token high-frequency query energy (||Q_high||_2).
    
    When the model writes prose (low Q_high), alpha stays at alpha_base
    to protect perplexity. The instant the model hunts for a fact
    (high Q_high spike), alpha drops exponentially toward 0, disengaging
    the governor and maximizing eviction/skip rate.
    """
    
    def __init__(
        self,
        alpha_base: float = 0.3,
        gamma: float = 150.0,
        q_high_baseline: float = 0.0,
        ema_decay: float = 0.9,
    ):
        """Initialize the Auto-Clutch.
        
        Args:
            alpha_base: Maximum alpha (generative mode). Default: 0.3.
            gamma: Exponential decay rate. Controls how quickly alpha drops
                   when Q_high energy rises. Higher = sharper transition.
            q_high_baseline: Baseline Q_high norm (subtracted before scaling).
                             Set to the mean Q_high from calibration data.
            ema_decay: Exponential moving average decay for smoothing.
        """
        self.alpha_base = alpha_base
        self.gamma = gamma
        self.q_high_baseline = q_high_baseline
        self.ema_decay = ema_decay
        
        # State
        self.q_high_ema = 0.0       # Smoothed Q_high norm
        self.alpha_history: List[float] = []
        self.q_high_history: List[float] = []
    
    def reset(self):
        """Reset state for a new generation."""
        self.q_high_ema = 0.0
        self.alpha_history = []
        self.q_high_history = []
    
    def compute_alpha(self, q_high_norm: float) -> float:
        """Compute dynamic alpha from current Q_high energy.
        
        alpha_t = alpha_base * exp(-gamma * max(0, q_high - baseline))
        
        Args:
            q_high_norm: The exact ||Q_high||_2 from Walsh Subspace Projection.
        
        Returns:
            alpha_t: Dynamic alpha for this decode step.
        """
        # Update EMA
        if self.q_high_ema == 0.0:
            self.q_high_ema = q_high_norm
        else:
            self.q_high_ema = (
                self.ema_decay * self.q_high_ema + 
                (1 - self.ema_decay) * q_high_norm
            )
        
        # Compute deviation from baseline
        delta = max(0.0, self.q_high_ema - self.q_high_baseline)
        
        # Exponential decay: high Q_high -> alpha drops toward 0
        alpha_t = self.alpha_base * math.exp(-self.gamma * delta)
        
        # Clamp
        alpha_t = max(0.0, min(self.alpha_base, alpha_t))
        
        # Record telemetry
        self.alpha_history.append(alpha_t)
        self.q_high_history.append(q_high_norm)
        
        return alpha_t
    
    @classmethod
    def from_calibration_data(
        cls,
        h_scores: List[float],
        alpha_base: float = 0.3,
    ) -> 'SpectralAutoClutch':
        """Create auto-clutch calibrated from observed H_score data.
        
        Sets q_high_baseline to the mean, and gamma so that alpha drops
        to ~0.01 at the max observed H_score.
        """
        if not h_scores:
            return cls(alpha_base=alpha_base)
        
        q_baseline = sum(h_scores) / len(h_scores)
        q_max = max(h_scores)
        delta_max = q_max - q_baseline
        
        if delta_max > 0:
            # Solve: alpha_base * exp(-gamma * delta_max) = 0.01
            # gamma = -ln(0.01/alpha_base) / delta_max
            gamma = -math.log(0.01 / alpha_base) / delta_max
        else:
            gamma = 150.0  # Default
        
        return cls(
            alpha_base=alpha_base,
            gamma=gamma,
            q_high_baseline=q_baseline,
        )
    
    def get_telemetry(self) -> dict:
        """Return telemetry for logging/JSON output."""
        return {
            'alpha_base': self.alpha_base,
            'gamma': self.gamma,
            'q_high_baseline': self.q_high_baseline,
            'alpha_history': self.alpha_history,
            'q_high_history': self.q_high_history,
            'alpha_mean': sum(self.alpha_history) / len(self.alpha_history) if self.alpha_history else 0,
            'alpha_min': min(self.alpha_history) if self.alpha_history else 0,
            'alpha_max': max(self.alpha_history) if self.alpha_history else 0,
        }
