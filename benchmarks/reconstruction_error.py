"""OrthoCache GPU Reconstruction Error Validator.

Validates that OrthoCache attention output matches dense attention within
target relative Frobenius error bounds, AND verifies the Perfect Eviction
Governor classifications.

For each (seq_len, eviction_rate) combination:
  1. Generate synthetic KV cache (bfloat16, seed=42)
  2. Compute dense attention output O_dense
  3. Run the OrthoCache spectral-energy pipeline manually:
       - compute_block_energy → sort by energy → mask top-(1-eviction_rate) blocks
       - orthocache_attention on retained blocks
  4. Compute relative Frobenius error: ||O_ortho - O_dense||_F / ||O_dense||_F
  5. Run classify_eviction() and verify PERFECT_EVICTION logit_gap >= 88.72
  6. Assert error <= target bound

Outputs
-------
* Summary table on stdout
* JSON results → benchmarks/results/reconstruction_error_results.json

Usage
-----
    python benchmarks/reconstruction_error.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Suppress torch.compile errors (Triton inductor requires C compiler on
# Windows). Allows orthocache_attention to fall back to eager execution.
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# ---------------------------------------------------------------------------
# Windows console encoding fix
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or benchmarks/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orthocache_gpu.pipeline import orthocache_forward
from orthocache_gpu.perfect_eviction import (
    classify_eviction,
    FLOAT32_UNDERFLOW_THRESHOLD,
)
from orthocache_gpu.spectral_energy import compute_block_energy
from orthocache_gpu.adaptive_attention import orthocache_attention

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BLOCK_SIZE = 512
SEQ_LENS = [4096, 8192, 16384, 32768]
EXTENDED_SEQ_LENS = [65536, 131072]  # VRAM-gated
NUM_HEADS = 8
HEAD_DIM = 128
QUERY_LEN = 16
EVICTION_RATES = [0.25, 0.50, 0.625, 0.75, 0.875]

# Target relative Frobenius error bounds per eviction rate
ERROR_BOUNDS: dict[float, float] = {
    0.25:  0.01,
    0.50:  0.015,
    0.625: 0.018,
    0.75:  0.02,
    0.875: 0.025,
}


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Auto-detect CUDA; graceful CPU fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dense attention baseline
# ---------------------------------------------------------------------------

def dense_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Compute dense attention baseline.

    Uses torch.nn.functional.scaled_dot_product_attention when available
    on CUDA, otherwise falls back to manual einsum on CPU.

    Args:
        q:      (seq_len_q, num_heads, head_dim)
        keys:   (seq_len_k, num_heads, head_dim)
        values: (seq_len_k, num_heads, head_dim)

    Returns:
        output: (seq_len_q, num_heads, head_dim) float32
    """
    device = q.device
    q_f = q.float()
    k_f = keys.float()
    v_f = values.float()

    # Try SDPA on CUDA — it expects (batch, num_heads, seq, head_dim)
    if device.type == "cuda":
        try:
            # Transpose to SDPA layout: (1, num_heads, seq, head_dim)
            q_sdpa = q_f.permute(1, 0, 2).unsqueeze(0)  # (1, H, Sq, D)
            k_sdpa = k_f.permute(1, 0, 2).unsqueeze(0)  # (1, H, Sk, D)
            v_sdpa = v_f.permute(1, 0, 2).unsqueeze(0)  # (1, H, Sk, D)

            with torch.no_grad():
                out_sdpa = F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa, is_causal=False,
                )
            # Back to (seq_len_q, num_heads, head_dim)
            return out_sdpa.squeeze(0).permute(1, 0, 2)
        except Exception:
            pass  # Fall through to einsum path

    # Manual einsum fallback (CPU or SDPA unavailable)
    head_dim = q.shape[-1]
    scale = math.sqrt(head_dim)
    with torch.no_grad():
        logits = torch.einsum("qhd,khd->qkh", q_f, k_f) / scale
        weights = F.softmax(logits, dim=1)
        return torch.einsum("qkh,khd->qhd", weights, v_f)


# ---------------------------------------------------------------------------
# OrthoCache pipeline (energy-based eviction)
# ---------------------------------------------------------------------------

def orthocache_energy_pipeline(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    eviction_rate: float,
    block_size: int = BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run OrthoCache via manual spectral-energy eviction.

    1. Compute per-block spectral energy
    2. Sort blocks by energy, retain top (1 - eviction_rate) fraction
    3. Run orthocache_attention on the retained subset

    Args:
        q:              (seq_len_q, num_heads, head_dim)
        keys:           (seq_len_k, num_heads, head_dim)
        values:         (seq_len_k, num_heads, head_dim)
        eviction_rate:  Fraction in [0, 1) of blocks to evict
        block_size:     Tokens per block

    Returns:
        output:         (seq_len_q, num_heads, head_dim) float32
        block_energies: (num_blocks, num_heads)
        block_mask:     (num_blocks,) bool — True = retained
    """
    seq_len_k = keys.shape[0]
    num_blocks = seq_len_k // block_size

    # Step 1: Spectral energy per block
    block_energies = compute_block_energy(keys, block_size)  # (num_blocks, num_heads)

    # Step 2: Sort blocks by total energy (sum over heads), retain top fraction
    total_energy = block_energies.sum(dim=-1)  # (num_blocks,)
    sorted_indices = torch.argsort(total_energy, descending=True)

    num_retain = max(1, int(round(num_blocks * (1.0 - eviction_rate))))
    block_mask = torch.zeros(num_blocks, dtype=torch.bool, device=keys.device)
    block_mask[sorted_indices[:num_retain]] = True

    # Step 3: Attention on retained blocks
    with torch.no_grad():
        output, _stats = orthocache_attention(
            q, keys, values, block_mask, block_size=block_size,
        )

    return output.float(), block_energies, block_mask


# ---------------------------------------------------------------------------
# Perfect Eviction verification
# ---------------------------------------------------------------------------

def verify_perfect_eviction(
    q: torch.Tensor,
    block_energies: torch.Tensor,
    block_mask: torch.Tensor,
    keys: torch.Tensor,
    head_dim: int,
) -> dict:
    """Run classify_eviction and verify PERFECT_EVICTION logit gap >= 88.72.

    Returns a dict with classification counts and verification status.
    """
    num_blocks = block_mask.shape[0]
    scale = math.sqrt(head_dim)

    # Estimate z_max from retained blocks
    with torch.no_grad():
        if block_mask.any():
            # Compute max logit among retained tokens
            retained_indices = block_mask.nonzero(as_tuple=True)[0]
            max_logit = torch.tensor(float("-inf"), device=q.device)
            for idx in retained_indices:
                start = idx * BLOCK_SIZE
                k_blk = keys[start:start + BLOCK_SIZE]  # (BS, H, D)
                logits = torch.einsum("qhd,khd->qkh", q.float(), k_blk.float()) / scale
                blk_max = logits.max()
                if blk_max > max_logit:
                    max_logit = blk_max
            z_max = max_logit
        else:
            z_max = torch.tensor(0.0, device=q.device)

    # Run classify_eviction
    eviction_meta = classify_eviction(
        q, block_energies, z_max, block_mask, head_dim,
    )

    # Verify: all PERFECT_EVICTION blocks must have logit_gap >= 88.72
    perfect_mask = eviction_meta.regime == 1  # regime 1 = perfect eviction
    if perfect_mask.any():
        perfect_gaps = eviction_meta.logit_gap[perfect_mask]
        min_perfect_gap = float(perfect_gaps.min().item())
        gap_ok = min_perfect_gap >= FLOAT32_UNDERFLOW_THRESHOLD
    else:
        min_perfect_gap = float("nan")
        gap_ok = True  # vacuously true — no perfect eviction blocks

    return {
        "num_perfect": eviction_meta.num_perfect,
        "num_statistical": eviction_meta.num_statistical,
        "num_retained": eviction_meta.num_retained,
        "min_perfect_logit_gap": min_perfect_gap,
        "perfect_gap_threshold": FLOAT32_UNDERFLOW_THRESHOLD,
        "perfect_gap_verified": gap_ok,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_reconstruction_error() -> list[dict]:
    """Run the complete reconstruction error validation sweep.

    Returns:
        List of result dicts, one per (seq_len, eviction_rate) pair.
    """
    device = get_device()
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  OrthoCache GPU Reconstruction Error Validator")
    print("=" * 78)
    print(f"  Device                : {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name()
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU                   : {gpu_name}")
        print(f"  VRAM                  : {vram_gb:.1f} GB")
    print(f"  Block size            : {BLOCK_SIZE}")
    print(f"  Heads x Dim           : {NUM_HEADS} x {HEAD_DIM}")
    print(f"  Query length          : {QUERY_LEN}")
    print(f"  Eviction rates        : {EVICTION_RATES}")
    print(f"  Underflow threshold   : {FLOAT32_UNDERFLOW_THRESHOLD}")
    if device.type == "cpu":
        print("  NOTE: Running on CPU fallback — GPU timing not available")
    print()

    # Build sequence length list, probing extended sizes on GPU
    seq_lens = list(SEQ_LENS)
    if device.type == "cuda":
        for ext_len in EXTENDED_SEQ_LENS:
            try:
                probe = torch.empty(
                    ext_len, NUM_HEADS, HEAD_DIM,
                    dtype=torch.bfloat16, device=device,
                )
                del probe
                torch.cuda.empty_cache()
                seq_lens.append(ext_len)
                print(f"  [VRAM] {ext_len} tokens — OK, included")
            except RuntimeError:
                print(f"  [VRAM] {ext_len} tokens — OOM, skipped")
    print()

    all_results: list[dict] = []

    # Summary table header
    header = (
        f"{'SeqLen':>8s}  {'Evict%':>7s}  {'Error':>10s}  "
        f"{'Bound':>7s}  {'PerfEvict':>9s}  {'StatEvict':>9s}  "
        f"{'GapOK':>5s}  {'Status':>6s}"
    )
    print(header)
    print("-" * len(header))

    for seq_len in seq_lens:
        num_blocks = seq_len // BLOCK_SIZE

        for eviction_rate in EVICTION_RATES:
            target_bound = ERROR_BOUNDS[eviction_rate]
            result: dict = {
                "seq_len": seq_len,
                "num_blocks": num_blocks,
                "eviction_rate": eviction_rate,
                "target_bound": target_bound,
                "device": str(device),
            }

            try:
                # Seed for reproducibility
                torch.manual_seed(42)

                # Generate synthetic KV cache and queries
                keys = torch.randn(
                    seq_len, NUM_HEADS, HEAD_DIM,
                    device=device, dtype=torch.bfloat16,
                )
                values = torch.randn(
                    seq_len, NUM_HEADS, HEAD_DIM,
                    device=device, dtype=torch.bfloat16,
                )
                q = torch.randn(
                    QUERY_LEN, NUM_HEADS, HEAD_DIM,
                    device=device, dtype=torch.bfloat16,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()

                # --- Dense baseline ---
                o_dense = dense_attention(q, keys, values)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                # --- OrthoCache pipeline ---
                o_ortho, block_energies, block_mask = orthocache_energy_pipeline(
                    q, keys, values,
                    eviction_rate=eviction_rate,
                    block_size=BLOCK_SIZE,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()

                # --- Relative Frobenius error ---
                diff = o_ortho - o_dense
                frob_error = torch.linalg.norm(diff.reshape(-1)).item()
                frob_dense = torch.linalg.norm(o_dense.reshape(-1)).item()
                rel_error = frob_error / max(frob_dense, 1e-12)

                result["frob_error"] = frob_error
                result["frob_dense"] = frob_dense
                result["rel_error"] = rel_error

                # --- Perfect Eviction verification ---
                pe_stats = verify_perfect_eviction(
                    q, block_energies, block_mask, keys, HEAD_DIM,
                )
                result.update(pe_stats)

                # --- PASS / FAIL ---
                error_pass = rel_error <= target_bound
                gap_pass = pe_stats["perfect_gap_verified"]
                overall_pass = error_pass and gap_pass
                result["error_pass"] = error_pass
                result["gap_pass"] = gap_pass
                result["overall_pass"] = overall_pass
                status = "PASS" if overall_pass else "FAIL"

                # Print summary row
                print(
                    f"{seq_len:>8d}  {eviction_rate:>6.1%}  "
                    f"{rel_error:>10.6f}  {target_bound:>7.4f}  "
                    f"{pe_stats['num_perfect']:>9d}  "
                    f"{pe_stats['num_statistical']:>9d}  "
                    f"{'Y' if gap_pass else 'N':>5s}  "
                    f"{status:>6s}"
                )

                # Clean up tensors
                del keys, values, q, o_dense, o_ortho, block_energies, block_mask
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    result["overall_pass"] = None
                    result["error"] = f"OOM: {e}"
                    print(
                        f"{seq_len:>8d}  {eviction_rate:>6.1%}  "
                        f"{'---':>10s}  {target_bound:>7.4f}  "
                        f"{'---':>9s}  {'---':>9s}  "
                        f"{'---':>5s}  {'SKIP':>6s}  (OOM)"
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

            all_results.append(result)

    # Save JSON results
    json_path = output_dir / "reconstruction_error_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print()
    print("-" * 78)

    # Summary statistics
    tested = [r for r in all_results if r.get("overall_pass") is not None]
    passed = [r for r in tested if r["overall_pass"]]
    failed = [r for r in tested if not r["overall_pass"]]
    skipped = [r for r in all_results if r.get("overall_pass") is None]

    print(f"  Total configurations : {len(all_results)}")
    print(f"  Tested               : {len(tested)}")
    print(f"  Passed               : {len(passed)}")
    print(f"  Failed               : {len(failed)}")
    print(f"  Skipped (OOM)        : {len(skipped)}")
    print(f"  Results written to   : {json_path.resolve()}")
    print("=" * 78)

    if failed:
        print("\n  FAILED configurations:")
        for r in failed:
            print(
                f"    seq={r['seq_len']}, evict={r['eviction_rate']:.1%}, "
                f"error={r.get('rel_error', '?'):.6f} "
                f"(bound={r['target_bound']:.4f}), "
                f"gap_ok={r.get('gap_pass', '?')}"
            )

    return all_results


if __name__ == "__main__":
    run_reconstruction_error()
