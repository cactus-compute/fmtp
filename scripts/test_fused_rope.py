"""Test fused RoPE kernel correctness."""

import mlx.core as mx
from nanochat.mlx.model import (
    apply_rope_with_positions_kernel,
    apply_rope_fused_qk_kernel,
)

def test_fused_rope():
    # Create test tensors
    B, L = 1, 3
    n_heads_q, n_heads_k = 8, 4
    head_dim = 64

    q = mx.random.normal((B, n_heads_q, L, head_dim))
    k = mx.random.normal((B, n_heads_k, L, head_dim))
    positions = mx.array([10, 11, 12])

    base = 10000.0
    scale = 1.0

    # Reference: two separate kernel calls
    q_ref = apply_rope_with_positions_kernel(q, positions, base=base, scale=scale)
    k_ref = apply_rope_with_positions_kernel(k, positions, base=base, scale=scale)
    mx.eval(q_ref, k_ref)

    # Fused kernel
    q_fused, k_fused = apply_rope_fused_qk_kernel(q, k, positions, base=base, scale=scale)
    mx.eval(q_fused, k_fused)

    # Compare
    q_diff = mx.abs(q_ref - q_fused).max().item()
    k_diff = mx.abs(k_ref - k_fused).max().item()

    print(f"Q max diff: {q_diff:.6e}")
    print(f"K max diff: {k_diff:.6e}")

    if q_diff < 1e-5 and k_diff < 1e-5:
        print("✓ Fused kernel matches reference!")
    else:
        print("✗ Fused kernel has errors!")
        print(f"Q ref[0,0,0,:4]: {q_ref[0,0,0,:4]}")
        print(f"Q fused[0,0,0,:4]: {q_fused[0,0,0,:4]}")

    # Benchmark
    import time

    n_iters = 100

    # Warm up
    for _ in range(10):
        _ = apply_rope_with_positions_kernel(q, positions, base=base, scale=scale)
        _ = apply_rope_with_positions_kernel(k, positions, base=base, scale=scale)
    mx.eval()

    start = time.perf_counter()
    for _ in range(n_iters):
        q_out = apply_rope_with_positions_kernel(q, positions, base=base, scale=scale)
        k_out = apply_rope_with_positions_kernel(k, positions, base=base, scale=scale)
        mx.eval(q_out, k_out)
    separate_time = (time.perf_counter() - start) / n_iters * 1000

    # Warm up fused
    for _ in range(10):
        _ = apply_rope_fused_qk_kernel(q, k, positions, base=base, scale=scale)
    mx.eval()

    start = time.perf_counter()
    for _ in range(n_iters):
        q_out, k_out = apply_rope_fused_qk_kernel(q, k, positions, base=base, scale=scale)
        mx.eval(q_out, k_out)
    fused_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"\nBenchmark ({n_iters} iterations):")
    print(f"  Separate kernels: {separate_time:.3f} ms")
    print(f"  Fused kernel:     {fused_time:.3f} ms")
    print(f"  Speedup:          {separate_time/fused_time:.2f}x")


if __name__ == "__main__":
    test_fused_rope()
