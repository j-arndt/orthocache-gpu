"""Dynamic Residual Information Governor for OrthoCache.

Tracks accumulated eviction pressure across transformer layers
and dynamically adjusts per-layer thresholds to prevent compounding
informational erosion through the residual stream.

Mathematical Foundation:
    tau_effective(l) = tau_base * max(0, 1 - alpha * eps_accum(l))

    where eps_accum(l) = sum_{j < l} eviction_rate(j)

    This ensures that:
    1. Early layers with cheap evictions do the heavy lifting
    2. Later layers automatically scale back as information loss accumulates
    3. Total eviction is front-loaded where it's cheapest
    4. The compounding error cascade is absorbed by the governor

Proven properties (informally):
    - Monotonicity: tau_effective(l) >= tau_effective(l+1) for all l
    - Boundedness: 0 <= tau_effective(l) <= tau_base for all l
    - Conservation: total eviction is bounded by design
"""

from typing import List, Optional


class ResidualGovernor:
    """Tracks accumulated eviction pressure across layers.

    As earlier layers evict more tiles, later layers automatically reduce
    their effective threshold to prevent compounding informational erosion
    through the residual stream.

    Usage:
        governor = ResidualGovernor(alpha=0.5)

        for layer in transformer_layers:
            governor.reset()  # Reset at start of each forward pass

            tau_eff = governor.get_tau_effective(tau_base)
            eviction_rate = layer.process(tau_eff)
            governor.report_layer(eviction_rate)
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Damping coefficient. Higher alpha = more conservative
                   downstream layers. Range [0, 1].
                   0.0 = no governor (static tau)
                   0.5 = moderate damping (recommended for 22-layer models)
                   1.0 = aggressive damping (full lockdown at 100% cumulative eviction)
        """
        if not 0.0 <= alpha <= 2.0:
            raise ValueError(f"alpha must be in [0, 2], got {alpha}")
        self.alpha = alpha
        self.eps_accum = 0.0
        self.layer_history: List[float] = []

    def reset(self):
        """Reset at the start of each forward pass."""
        self.eps_accum = 0.0
        self.layer_history = []

    def get_tau_effective(self, tau_base: float) -> float:
        """Compute the effective tau for the current layer.

        Returns:
            tau_effective = tau_base * max(0, 1 - alpha * eps_accum)
        """
        scale = max(0.0, 1.0 - self.alpha * self.eps_accum)
        return tau_base * scale

    def report_layer(self, eviction_rate: float):
        """Called after each layer completes. Updates cumulative pressure.

        Args:
            eviction_rate: Fraction of tiles evicted in this layer [0, 1].
        """
        self.layer_history.append(eviction_rate)
        self.eps_accum += eviction_rate

    @property
    def total_eviction_pressure(self) -> float:
        """Total accumulated eviction pressure across all reported layers."""
        return self.eps_accum

    @property
    def num_layers_reported(self) -> int:
        """Number of layers that have reported eviction rates."""
        return len(self.layer_history)

    def summary(self) -> str:
        """Human-readable summary of governor state."""
        if not self.layer_history:
            return "Governor: no layers reported yet"
        rates = ", ".join(f"{r:.1%}" for r in self.layer_history)
        return (
            f"Governor(alpha={self.alpha}): "
            f"{self.num_layers_reported} layers, "
            f"eps_accum={self.eps_accum:.3f}, "
            f"history=[{rates}]"
        )


class EntropyGovernor(ResidualGovernor):
    """Extension: Temporal entropy-based tau scaling (Barrier 2).

    Modulates tau based on the attention entropy from the previous decode step.
    High entropy (diffuse attention) → more aggressive eviction.
    Low entropy (sharp attention) → conservative lockdown.

    This is a Phase 2 optimization built on top of the base ResidualGovernor.

    NOTE: Not yet integrated into the eval harness. Placeholder for Phase 11.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 1.0,
                 h_median: float = 3.0):
        """
        Args:
            alpha: Residual governor damping coefficient.
            beta: Entropy scaling sensitivity. Higher = sharper transition.
            h_median: Median entropy estimate for the model.
        """
        super().__init__(alpha=alpha)
        self.beta = beta
        self.h_median = h_median
        self.current_entropy: Optional[float] = None

    def set_entropy(self, entropy: float):
        """Set the attention entropy from the previous decode step."""
        self.current_entropy = entropy

    def get_tau_effective(self, tau_base: float) -> float:
        """Compute tau with both residual and entropy modulation."""
        # Base governor
        tau_gov = super().get_tau_effective(tau_base)

        # Entropy modulation
        if self.current_entropy is not None:
            import math
            sigmoid = 1.0 / (1.0 + math.exp(-self.beta * (self.current_entropy - self.h_median)))
            # sigmoid > 0.5 when entropy > median → scale up tau (more aggressive)
            # sigmoid < 0.5 when entropy < median → scale down tau (conservative)
            tau_gov *= (2.0 * sigmoid)  # Range: [0, 2*tau_gov]

        return tau_gov
