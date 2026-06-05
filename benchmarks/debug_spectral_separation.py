"""Debug script: Verify Walsh basis signal construction for spectral separation.

Constructs a semantically coherent signal using ONLY low-sequency Walsh basis
functions, ensuring E_high ≈ 0 by construction. This is the technically correct
way to generate test signals with known spectral properties.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
from orthocache_gpu.triton_kernels.fwht_fused_prototype import (
    generate_walsh_matrix, BAND_LOW_64, BAND_HIGH_64
)

torch.manual_seed(42)
W = generate_walsh_matrix(64)

print("=" * 70)
print("WALSH BASIS SIGNAL CONSTRUCTION — SPECTRAL SEPARATION ANALYSIS")
print("=" * 70)

# Strategy 1: Direct Walsh basis construction
# Walsh rows 0-7 ARE the DC + low-sequency basis functions
# A signal constructed from ONLY these basis vectors will have
# E_high = 0 exactly (by Parseval's theorem + orthogonality)
print("\n--- Strategy: Walsh Basis Construction ---")
coeffs = torch.zeros(64, 128)
for i in range(8):  # DC + low band indices 0-7
    coeffs[i] = torch.randn(128) * (5.0 / (i + 1))

# Since W is symmetric and orthogonal (W = W^T, W @ W^T = I),
# the inverse WHT is W^T = W itself
semantic = W @ coeffs  # This is actually the INVERSE transform

# Verify by forward-transforming
k_sp = W @ semantic  # Should recover coeffs exactly (W @ W @ coeffs = coeffs)
energy = torch.sum(k_sp ** 2, dim=1)  # per-sequency energy

e_dc = energy[0].item()
e_low = energy[BAND_LOW_64[0]:BAND_LOW_64[1]].sum().item()
e_mid = energy[BAND_LOW_64[1]:BAND_HIGH_64[0]].sum().item()
e_high = energy[BAND_HIGH_64[0]:BAND_HIGH_64[1]].sum().item()
zeta_sem = e_high / (e_low + 1e-6)

print(f"  E_DC   = {e_dc:.2f}")
print(f"  E_low  = {e_low:.2f}  (band {BAND_LOW_64})")
print(f"  E_mid  = {e_mid:.6f}  (should be ~0)")
print(f"  E_high = {e_high:.6f}  (should be ~0)")
print(f"  zeta   = {zeta_sem:.6f}")

# Noise block
print("\n--- Noise Block (IID Gaussian) ---")
noise = torch.randn(64, 128)
k_sp2 = W @ noise
en2 = torch.sum(k_sp2 ** 2, dim=1)

e_dc2 = en2[0].item()
e_low2 = en2[BAND_LOW_64[0]:BAND_LOW_64[1]].sum().item()
e_mid2 = en2[BAND_LOW_64[1]:BAND_HIGH_64[0]].sum().item()
e_high2 = en2[BAND_HIGH_64[0]:BAND_HIGH_64[1]].sum().item()
zeta_noise = e_high2 / (e_low2 + 1e-6)

print(f"  E_DC   = {e_dc2:.2f}")
print(f"  E_low  = {e_low2:.2f}  (band {BAND_LOW_64})")
print(f"  E_mid  = {e_mid2:.2f}")
print(f"  E_high = {e_high2:.2f}")
print(f"  zeta   = {zeta_noise:.4f}")

print(f"\n--- Separation ---")
print(f"  zeta_semantic = {zeta_sem:.6f}")
print(f"  zeta_noise    = {zeta_noise:.4f}")
ratio = zeta_noise / max(zeta_sem, 1e-10)
print(f"  Ratio: {ratio:.0f}x")
print(f"  PASS: {'YES' if zeta_noise > zeta_sem else 'NO'}")

# Strategy 2: Simpler — use block repetition (realistic KV cache scenario)
# In real transformer KV caches, semantic tokens have similar key vectors
# (from the same concept/entity), while noise tokens are random
print("\n\n--- Strategy 2: Repeated Key Vector (Realistic Semantic Block) ---")
template = torch.randn(1, 128) * 3.0  # one semantic "concept"
semantic2 = template.expand(64, 128) + 0.1 * torch.randn(64, 128)  # small variations

k_sp3 = W @ semantic2
en3 = torch.sum(k_sp3 ** 2, dim=1)
e_low3 = en3[BAND_LOW_64[0]:BAND_LOW_64[1]].sum().item()
e_high3 = en3[BAND_HIGH_64[0]:BAND_HIGH_64[1]].sum().item()
zeta3 = e_high3 / (e_low3 + 1e-6)
print(f"  E_low = {e_low3:.2f}, E_high = {e_high3:.2f}, zeta = {zeta3:.4f}")
print(f"  Separation vs noise: {zeta_noise / max(zeta3, 1e-10):.1f}x")
print(f"  PASS: {'YES' if zeta_noise > zeta3 else 'NO'}")
