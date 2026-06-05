import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Topology.Algebra.Order.LiminfLimsup
import Mathlib.Order.Filter.Basic

/-!
# OrthoCache Truncation Bound

We prove the Total Variation distance bound for softmax truncation:

Given logits z : Fin N → ℝ, a retained set S, and an evicted set Sᶜ where:
- All evicted logits satisfy z_i < β
- The maximum retained logit is z_max

Then:
  TV(α, α̂) ≤ |Sᶜ| · exp(β - z_max)

The proof is structured in two parts:
1. Each evicted softmax probability is bounded by exp(β - z_max)
2. Summing over all evicted tokens gives the bound
-/

open Real BigOperators Finset

noncomputable section

/-- The softmax partition function is positive when N ≥ 1. -/
lemma partition_pos {N : ℕ} [NeZero N] (z : Fin N → ℝ) :
    0 < ∑ j : Fin N, Real.exp (z j) := by
  apply Finset.sum_pos
  · intro j _; exact Real.exp_pos (z j)
  · exact Finset.univ_nonempty

/-- A single term exp(z_{j₀}) is at most the full partition function. -/
lemma single_exp_le_partition {N : ℕ} (z : Fin N → ℝ) (j₀ : Fin N) :
    Real.exp (z j₀) ≤ ∑ j : Fin N, Real.exp (z j) :=
  Finset.single_le_sum (fun k _ => le_of_lt (Real.exp_pos (z k))) (Finset.mem_univ j₀)

/-- Each softmax probability for an evicted token with z_i < β is bounded by
    exp(β - z_max). -/
lemma softmax_evicted_le {N : ℕ} [NeZero N]
    (z : Fin N → ℝ) (i j₀ : Fin N)
    (z_max beta : ℝ)
    (hj₀ : z j₀ = z_max)
    (hi : z i < beta) :
    Real.exp (z i) / (∑ j : Fin N, Real.exp (z j)) ≤ Real.exp (beta - z_max) := by
  have hZ := partition_pos z
  rw [div_le_iff hZ]
  -- Goal: exp(z_i) ≤ exp(β - z_max) * Z
  -- Since Z ≥ exp(z_max): exp(β - z_max) * Z ≥ exp(β - z_max) * exp(z_max) = exp(β)
  -- And exp(z_i) < exp(β) since z_i < β
  calc Real.exp (z i)
      ≤ Real.exp beta := le_of_lt (Real.exp_lt_exp_of_lt hi)
    _ = Real.exp (beta - z_max) * Real.exp z_max := by
        rw [← Real.exp_add]; ring_nf
    _ ≤ Real.exp (beta - z_max) * (∑ j : Fin N, Real.exp (z j)) := by
        apply mul_le_mul_of_nonneg_left _ (le_of_lt (Real.exp_pos _))
        rw [← hj₀]
        exact single_exp_le_partition z j₀

/-- **OrthoCache Truncation Bound (Main Theorem)**

The sum of softmax probabilities over evicted tokens (= the TV distance
between full and truncated attention, by the TV-δ lemma) is bounded by:

  ∑_{i ∈ Sᶜ} α_i ≤ |Sᶜ| · exp(β - z_max)

This is the core mathematical guarantee of OrthoCache: the attention
distribution shift from block eviction decays exponentially in the gap
(z_max - β). -/
theorem orthocache_truncation_bound {N : ℕ} [NeZero N]
    (z : Fin N → ℝ)
    (S_c : Finset (Fin N))
    (z_max beta : ℝ)
    (hz_max : ∃ j : Fin N, z j = z_max)
    (hbeta : ∀ i ∈ S_c, z i < beta) :
    (∑ i ∈ S_c, Real.exp (z i) / (∑ j : Fin N, Real.exp (z j)))
      ≤ (S_c.card : ℝ) * Real.exp (beta - z_max) := by
  obtain ⟨j₀, hj₀⟩ := hz_max
  -- Convert nsmul to mul for Finset.sum_le_card_nsmul
  have h : ∀ i ∈ S_c,
      Real.exp (z i) / (∑ j : Fin N, Real.exp (z j)) ≤ Real.exp (beta - z_max) := by
    intro i hi
    exact softmax_evicted_le z i j₀ z_max beta hj₀ (hbeta i hi)
  calc (∑ i ∈ S_c, Real.exp (z i) / (∑ j : Fin N, Real.exp (z j)))
      ≤ ∑ _i ∈ S_c, Real.exp (beta - z_max) :=
        Finset.sum_le_sum h
    _ = (S_c.card : ℝ) * Real.exp (beta - z_max) := by
        simp [Finset.sum_const, nsmul_eq_mul]

/-- Corollary: When β ≤ z_max, the exponential factor is ≤ 1. -/
theorem evicted_mass_le_card {N : ℕ} [NeZero N]
    (z : Fin N → ℝ)
    (S_c : Finset (Fin N))
    (z_max beta : ℝ)
    (hz_max : ∃ j : Fin N, z j = z_max)
    (hbeta : ∀ i ∈ S_c, z i < beta)
    (hbeta_le : beta ≤ z_max) :
    (∑ i ∈ S_c, Real.exp (z i) / (∑ j : Fin N, Real.exp (z j)))
      ≤ (S_c.card : ℝ) := by
  calc (∑ i ∈ S_c, Real.exp (z i) / (∑ j : Fin N, Real.exp (z j)))
      ≤ (S_c.card : ℝ) * Real.exp (beta - z_max) :=
        orthocache_truncation_bound z S_c z_max beta hz_max hbeta
    _ ≤ (S_c.card : ℝ) * 1 := by
        apply mul_le_mul_of_nonneg_left _ (Nat.cast_nonneg _)
        rw [← Real.exp_zero]
        apply Real.exp_le_exp_of_le
        linarith
    _ = (S_c.card : ℝ) := mul_one _

end
