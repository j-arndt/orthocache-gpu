import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Kronecker

/-!
# Parseval's Identity for Walsh-Hadamard Transform

We define the Walsh-Hadamard matrix H_n of dimension 2^n × 2^n using the
Kronecker product recurrence:

  H_0 = [[1]]
  H_{n+1} = H_n ⊗ₖ [[1, 1], [1, -1]]

reindexed via `finProdFinEquiv` to stay on `Fin (2^n)`.

We prove that H_n is orthogonal up to scaling:

  H_nᵀ * H_n = 2^n • I

which implies Parseval's identity:

  ‖H_n · x‖² = 2^n · ‖x‖²

## Proof Strategy

The orthogonality proof proceeds by induction on n:
- Base case: direct computation on the 1×1 identity.
- Inductive step: uses three Mathlib lemmas:
  1. `transpose_submatrix` to pull transpose through reindexing
  2. `submatrix_mul` to combine the product under the same reindexing
  3. `kroneckerMap_transpose` + `mul_kronecker_mul` (mixed product)
     to factor (H_n ⊗ H₂)ᵀ * (H_n ⊗ H₂) = (H_nᵀ*H_n) ⊗ (H₂ᵀ*H₂)
  4. Inductive hypothesis + base case + `smul_kronecker`/`one_kronecker_one`
     to simplify to 2^(n+1) • I
  5. `submatrix_one_equiv` + `submatrix_smul` to push reindexing through.
-/

open Matrix BigOperators Finset

noncomputable section

open Kronecker in

/-! ## Definitions -/

/-- The 2×2 Hadamard base matrix [[1, 1], [1, -1]]. -/
def H₂ : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1; 1, -1]

/-- Index equivalence: Fin (2^n) × Fin 2 ≃ Fin (2^(n+1)).
    Composes the product-to-Fin equivalence with a cast for `2^n * 2 = 2^(n+1)`. -/
def whtIdx (n : ℕ) : Fin (2 ^ n) × Fin 2 ≃ Fin (2 ^ (n + 1)) :=
  finProdFinEquiv.trans (finCongr (by ring))

/-- The Walsh-Hadamard matrix of dimension 2^n × 2^n, defined by Kronecker recurrence.
    Uses `submatrix` with `whtIdx` to reindex from the product type to `Fin (2^(n+1))`. -/
def WHT : (n : ℕ) → Matrix (Fin (2 ^ n)) (Fin (2 ^ n)) ℝ
  | 0 => 1
  | n + 1 =>
    ((WHT n).kroneckerMap (· * ·) H₂).submatrix (whtIdx n).symm (whtIdx n).symm

/-! ## Base Case -/

/-- H₂ᵀ * H₂ = 2 • I. Proved by exhaustive computation over Fin 2. -/
lemma H₂_mul_H₂ :
    H₂ᵀ * H₂ = (2 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp (config := { decide := true }) [H₂, Matrix.mul_apply, Matrix.transpose_apply,
          Fin.sum_univ_two, Matrix.smul_apply, Matrix.one_apply,
          vecHead, vecTail, Fin.isValue] <;>
    norm_num

/-! ## Orthogonality Theorem -/

/-- **WHT Orthogonality**: H_nᵀ * H_n = 2^n • I.

This is the key structural property: the unnormalized Walsh-Hadamard matrix
is orthogonal up to a factor of 2^n. Normalizing by 1/√(2^n) yields a true
orthogonal matrix. -/
theorem WHT_orthogonal (n : ℕ) :
    (WHT n)ᵀ * WHT n = ((2 : ℝ) ^ n) • (1 : Matrix (Fin (2 ^ n)) (Fin (2 ^ n)) ℝ) := by
  induction n with
  | zero =>
    -- WHT 0 = 1 (the 1×1 identity), so 1ᵀ * 1 = 1 = 2^0 • 1
    simp [WHT]
  | succ n ih =>
    -- WHT (n+1) = K.submatrix e e  where K = WHT n ⊗ₖ H₂, e = (whtIdx n).symm
    simp only [WHT]
    set K := (WHT n).kroneckerMap (· * ·) H₂ with hK_def
    set e := (whtIdx n).symm
    -- Step 1: Pull transpose through submatrix
    --   (K.submatrix e e)ᵀ = Kᵀ.submatrix e e
    rw [transpose_submatrix]
    -- Step 2: Combine multiplication under shared bijective reindexing
    --   Kᵀ.submatrix e e * K.submatrix e e = (Kᵀ * K).submatrix e e
    --   Using submatrix_mul with e bijective (it's an equiv)
    rw [← submatrix_mul Kᵀ K e (whtIdx n).symm e (whtIdx n).symm.bijective]
    -- Step 3: Factor Kᵀ * K using Kronecker properties
    -- Kᵀ = (WHT n)ᵀ ⊗ₖ H₂ᵀ   [kroneckerMap_transpose]
    -- Kᵀ * K = ((WHT n)ᵀ * WHT n) ⊗ₖ (H₂ᵀ * H₂)  [mul_kronecker_mul]
    have hKtK : Kᵀ * K =
        ((WHT n)ᵀ * WHT n).kroneckerMap (· * ·) (H₂ᵀ * H₂) := by
      rw [hK_def]
      rw [← kroneckerMap_transpose (· * ·) (WHT n) H₂]
      rw [mul_kronecker_mul]
    rw [hKtK]
    -- Step 4: Apply inductive hypothesis and base case
    rw [ih, H₂_mul_H₂]
    -- Now goal is: ((2^n • 1) ⊗ₖ (2 • 1)).submatrix e e = 2^(n+1) • 1
    -- Step 5: Pull scalars through Kronecker product and simplify
    simp only [smul_kronecker, kronecker_smul, one_kronecker_one, smul_smul]
    -- Goal: ((2^n * 2) • 1).submatrix ⇑e ⇑e = 2^(n+1) • 1
    -- Push submatrix through smul, then apply submatrix_one_equiv
    conv_lhs => rw [show ((2 * (2 : ℝ) ^ n) • (1 : Matrix _ _ ℝ)).submatrix (⇑e) (⇑e) =
        (2 * (2 : ℝ) ^ n) • ((1 : Matrix _ _ ℝ).submatrix (⇑e) (⇑e)) from rfl]
    rw [submatrix_one_equiv]
    ring_nf

/-! ## Parseval's Identity -/

/-- **Parseval's Identity for WHT**: ‖H_n · x‖² = 2^n · ‖x‖².

The squared L2 norm of the WHT output equals 2^n times the squared L2 norm
of the input. When the WHT is normalized by 1/√(2^n), this becomes exact
energy preservation: spectral energy ≡ spatial energy. -/
theorem parseval_WHT (n : ℕ) (x : Fin (2 ^ n) → ℝ) :
    let y := (WHT n).mulVec x
    (∑ i, y i ^ 2) = (2 : ℝ) ^ n * (∑ i, x i ^ 2) := by
  simp only
  -- Strategy: ∑ᵢ (Hx)ᵢ² = xᵀ (Hᵀ H) x = xᵀ (2^n I) x = 2^n · ∑ᵢ xᵢ²
  -- Step 1: Express ‖y‖² as the dot product yᵀ · y = (Hx)ᵀ · (Hx)
  have key : ∀ v : Fin (2 ^ n) → ℝ, ∑ i, v i ^ 2 = dotProduct v v := by
    intro v; simp [dotProduct, sq]
  rw [key]
  -- Step 2: Express dotProduct (H *ᵥ x) (H *ᵥ x) = x ⬝ᵥ ((Hᵀ * H) *ᵥ x)
  change dotProduct ((WHT n).mulVec x) ((WHT n).mulVec x) = _
  rw [dotProduct_mulVec, vecMul_mulVec]
  -- Goal: x ᵥ* ((WHT n)ᵀ * WHT n) ⬝ᵥ x = ...
  -- Step 3: Substitute Hᵀ H = 2^n • I
  rw [WHT_orthogonal]
  -- Goal: x ᵥ* (2^n • 1) ⬝ᵥ x = ...
  -- Step 4: Unfold vecMul and dotProduct to Finset.sum, then use algebra
  simp only [vecMul, dotProduct, Matrix.smul_apply, Matrix.one_apply,
             smul_eq_mul, mul_boole, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  -- Goal is now: ∑ x₁, (∑ x₂, x x₂ * if x₂ = x₁ then 2^n else 0) * x x₁ = 2^n * ∑ x i ^ 2
  -- Push multiplication inside the conditional: a * if P then b else 0 = if P then a*b else 0
  simp only [mul_ite, mul_zero]
  -- Now: ∑ x₁, (∑ x₂, if x₂ = x₁ then x x₂ * 2^n else 0) * x x₁ = ...
  -- Collapse inner sum with Finset.sum_ite_eq'
  simp only [Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  -- Now: ∑ x₁, (x x₁ * 2^n) * x x₁ = 2^n * ∑ i, x i ^ 2
  -- Rewrite x i ^ 2 as x i * x i on the RHS
  simp_rw [sq]
  -- Both sides: ∑ x₁, (x x₁ * 2^n) * x x₁ = 2^n * ∑ x₁, x x₁ * x x₁
  -- Normalize each term: (x * 2^n) * x = 2^n * (x * x)
  simp_rw [show ∀ a : Fin (2 ^ n), (x a * (2 : ℝ) ^ n) * x a = (2 : ℝ) ^ n * (x a * x a)
    from fun a => by ring]
  -- Factor out 2^n from the sum
  rw [← Finset.mul_sum]

end
