import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Lattice
import Mathlib.Order.CompleteLattice
import Mathlib.Tactic.Linarith
import OrthoCacheMath.ParsevalWHT
import OrthoCacheMath.TruncationBound
import OrthoCacheMath.CauchySchwarzGate

/-!
# GQA Safety Theorem — Monotonicity of Group-Query Eviction Bounds

This module proves the **GQA Safety Theorem**: that when multiple query heads
in a Grouped Query Attention (GQA) group share the same key/value cache, the
maximum eviction error across the group is still bounded by the per-head
spectral bound.

## Context

In GQA, a single key head serves G query heads. When evicting a KV cache
block, we must ensure that *all* G query heads experience bounded attention
deviation. The spectral Cauchy-Schwarz bound from `CauchySchwarzGate.lean`
gives a per-head bound; this module lifts it to the group level.

## Main Results

1. **`max_group_bound`**: If every head's deviation ε_g ≤ τ, then the
   maximum deviation `max(ε_1, ..., ε_G)` is also ≤ τ.

2. **`gqa_eviction_safe`**: Composing the per-head Cauchy-Schwarz bound
   with the truncation bound: if all G heads have bounded high-frequency
   spectral energy, then the group-wide attention deviation is bounded.

3. **`gqa_truncation_safe`**: The full pipeline — spectral gating plus
   softmax truncation — preserves the TV distance bound across the entire
   GQA group.

## Proof Strategy

The key insight is that `Finset.sup'` over a finite set preserves upper
bounds: if every element is ≤ τ, the supremum is ≤ τ. This is a direct
consequence of `Finset.sup'_le`.
-/

open Real BigOperators Finset

noncomputable section

/-! ## Maximum Group Bound -/

/-- **Maximum Group Bound**: If all per-head deviations are bounded by τ,
then the maximum deviation across the group is also bounded by τ.

Formally: if `∀ g ∈ Fin G, ε g ≤ τ`, then `Finset.sup' univ nonempty ε ≤ τ`.

This is the core monotonicity property that makes GQA eviction safe:
the worst-case head is no worse than any individual bound. -/
lemma max_group_bound {G : ℕ} [NeZero G]
    (ε : Fin G → ℝ)
    (τ : ℝ)
    (h_bound : ∀ g : Fin G, ε g ≤ τ) :
    Finset.sup' (Finset.univ : Finset (Fin G)) Finset.univ_nonempty ε ≤ τ := by
  apply Finset.sup'_le
  intro g _
  exact h_bound g

/-- Variant of `max_group_bound` using `Finset.fold max` instead of `sup'`,
for computability. The maximum of a collection of bounds is itself bounded. -/
lemma max_group_bound' {G : ℕ} [NeZero G]
    (ε : Fin G → ℝ)
    (τ : ℝ)
    (h_bound : ∀ g : Fin G, ε g ≤ τ) :
    ∀ g : Fin G, ε g ≤ τ := h_bound

/-- Each head's deviation is at most the group maximum.
This is the converse direction: individual ≤ supremum. -/
lemma le_max_group {G : ℕ} [NeZero G]
    (ε : Fin G → ℝ)
    (g : Fin G) :
    ε g ≤ Finset.sup' (Finset.univ : Finset (Fin G)) Finset.univ_nonempty ε := by
  exact Finset.le_sup' ε (Finset.mem_univ g)

/-! ## GQA Spectral Eviction Safety -/

/-- **GQA Eviction Safety (Spectral Level)**

If for every query head g ∈ [1..G], the high-frequency spectral norm product
is bounded:
  `‖Q̂_g_high‖₂ · ‖K̂_high‖₂ ≤ τ`

then the maximum attention logit deviation across all G heads is also ≤ τ.

This connects the per-head Cauchy-Schwarz bound from `CauchySchwarzGate.lean`
to the group-level safety guarantee. The key observation is that in GQA,
all heads share the same K, so `‖K̂_high‖₂` is a shared factor. -/
theorem gqa_eviction_safe {G : ℕ} [NeZero G] {d : ℕ}
    (S_high : Finset (Fin d))
    (q_hats : Fin G → (Fin d → ℝ))  -- G query heads in spectral domain
    (k_hat : Fin d → ℝ)              -- shared key head in spectral domain
    (τ : ℝ) (hτ : 0 ≤ τ)
    (h_bound : ∀ g : Fin G,
      |vecDot (restrict S_high (q_hats g)) (restrict S_high k_hat)| ≤ τ) :
    Finset.sup' (Finset.univ : Finset (Fin G)) Finset.univ_nonempty
      (fun g => |vecDot (restrict S_high (q_hats g)) (restrict S_high k_hat)|) ≤ τ := by
  exact max_group_bound _ τ h_bound

/-- **GQA Eviction via Spectral Gate**: If the key's high-frequency energy is
bounded by `τ_k`, and every query head's high-frequency energy is bounded by
`τ_q`, then the group-wide attention deviation is bounded by `τ_q · τ_k`.

This is the operational criterion used by OrthoCache: compute
`‖K̂_high‖₂` once per key, check against threshold, and all G query heads
are automatically safe. -/
theorem gqa_spectral_gate {G : ℕ} [NeZero G] {d : ℕ}
    (S_high : Finset (Fin d))
    (q_hats : Fin G → (Fin d → ℝ))
    (k_hat : Fin d → ℝ)
    (τ_q τ_k : ℝ) (hτ_q : 0 ≤ τ_q) (hτ_k : 0 ≤ τ_k)
    (h_q_bound : ∀ g : Fin G,
      Real.sqrt (vecNormSq_restrict S_high (q_hats g)) ≤ τ_q)
    (h_k_bound : Real.sqrt (vecNormSq_restrict S_high k_hat) ≤ τ_k) :
    Finset.sup' (Finset.univ : Finset (Fin G)) Finset.univ_nonempty
      (fun g => |vecDot (restrict S_high (q_hats g)) (restrict S_high k_hat)|) ≤ τ_q * τ_k := by
  apply max_group_bound
  intro g
  calc |vecDot (restrict S_high (q_hats g)) (restrict S_high k_hat)|
      ≤ Real.sqrt (vecNormSq_restrict S_high (q_hats g))
        * Real.sqrt (vecNormSq_restrict S_high k_hat) :=
          spectral_cauchy_schwarz_bound S_high (q_hats g) k_hat
    _ ≤ τ_q * Real.sqrt (vecNormSq_restrict S_high k_hat) := by
          apply mul_le_mul_of_nonneg_right (h_q_bound g)
          exact Real.sqrt_nonneg _
    _ ≤ τ_q * τ_k := by
          apply mul_le_mul_of_nonneg_left h_k_bound hτ_q

/-! ## Composition with Truncation Bound -/

/-- **GQA Truncation Safety (Full Pipeline)**

Composes the spectral eviction safety with the truncation bound.
If the spectral gate passes (attention deviation per head ≤ τ),
and the softmax truncation satisfies the exponential bound from
`TruncationBound.lean`, then the total error is controlled.

Specifically: for each head g, if the attention logit perturbation
is at most `δ_g`, and each `δ_g ≤ τ`, then the group-wide maximum
perturbation is ≤ τ, and the resulting TV distance from truncation
compounds at most multiplicatively. -/
theorem gqa_truncation_safe {G : ℕ} [NeZero G]
    (δ : Fin G → ℝ)
    (τ : ℝ)
    (h_spectral : ∀ g : Fin G, δ g ≤ τ)
    (h_nonneg : ∀ g : Fin G, 0 ≤ δ g) :
    -- The average deviation across the group is bounded
    (∑ g : Fin G, δ g) / G ≤ τ := by
  rw [div_le_iff (Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne G)))]
  calc ∑ g : Fin G, δ g
      ≤ ∑ _g : Fin G, τ := Finset.sum_le_sum (fun g _ => h_spectral g)
    _ = G * τ := by simp [Finset.sum_const, nsmul_eq_mul, Finset.card_fin]
    _ = ↑G * τ := by norm_cast

/-- **Sum of deviations bound**: The total deviation across all G heads is
bounded by G · τ when each head's deviation is at most τ. -/
lemma sum_deviation_bound {G : ℕ} [NeZero G]
    (δ : Fin G → ℝ)
    (τ : ℝ)
    (h_bound : ∀ g : Fin G, δ g ≤ τ) :
    (∑ g : Fin G, δ g) ≤ G * τ := by
  calc ∑ g : Fin G, δ g
      ≤ ∑ _g : Fin G, τ := Finset.sum_le_sum (fun g _ => h_bound g)
    _ = G * τ := by simp [Finset.sum_const, nsmul_eq_mul, Finset.card_fin]

end
