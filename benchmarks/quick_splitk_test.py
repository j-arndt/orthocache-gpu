"""Quick latency comparison: V1 (single-CTA) vs V2 (Split-K) God Kernel."""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')

import torch
import time

from orthocache_gpu.triton_kernels.fused_eviction import (
    fused_orthocache_attention,
    fused_orthocache_attention_v2,
)

device = torch.device('cuda')
head_dim = 128
zeta_max = 5.0

print("Split-K vs V1 Latency Comparison (single head)")
print("=" * 65)
print(f"{'seq_len':>10} {'V1 (ms)':>12} {'SplitK (ms)':>14} {'Speedup':>10}")
print("-" * 65)

for seq_len in [1024, 4096, 8192, 16384, 32768]:
    torch.manual_seed(42)
    q1 = torch.randn(1, head_dim, device=device)
    k1 = torch.randn(seq_len, head_dim, device=device)
    v1 = torch.randn(seq_len, head_dim, device=device)
    q2 = q1.squeeze(0).unsqueeze(0)  # (1, head_dim)
    k2 = k1.unsqueeze(0)             # (1, seq, dim)
    v2 = v1.unsqueeze(0)             # (1, seq, dim)

    # Warmup
    fused_orthocache_attention(q1, k1, v1, zeta_max=zeta_max)
    fused_orthocache_attention_v2(q2, k2, v2, zeta_max=zeta_max)
    torch.cuda.synchronize()

    # V1 timing
    times_v1 = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fused_orthocache_attention(q1, k1, v1, zeta_max=zeta_max)
        torch.cuda.synchronize()
        times_v1.append((time.perf_counter() - t0) * 1000)
    v1_ms = sum(sorted(times_v1)[2:8]) / 6

    # V2 Split-K timing
    times_v2 = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fused_orthocache_attention_v2(q2, k2, v2, zeta_max=zeta_max)
        torch.cuda.synchronize()
        times_v2.append((time.perf_counter() - t0) * 1000)
    v2_ms = sum(sorted(times_v2)[2:8]) / 6

    speedup = v1_ms / v2_ms if v2_ms > 0 else float('inf')
    print(f"{seq_len:>10} {v1_ms:>12.3f} {v2_ms:>14.3f} {speedup:>10.2f}x")

print()
print("V1 = grid=(1,) single-CTA sequential")
print("V2 = grid=(1, auto) interleaved Split-K")
num_sms = torch.cuda.get_device_properties(device).multi_processor_count
print(f"GPU: {torch.cuda.get_device_name()} ({num_sms} SMs)")
