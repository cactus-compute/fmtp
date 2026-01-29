"""
Benchmark cache operations and tree attention overhead.

Measures:
1. Cache trim time (simple slicing)
2. Cache compact time (gather operation)
3. Tree attention mask build time
4. Forward pass time with/without tree mask
"""

import time
import mlx.core as mx
from nanochat.mlx.model import (
    GemmaMedusaModel,
    build_tree_attention_mask_mlx,
    generate_tree_buffers,
    DEFAULT_TREES,
)


def benchmark_cache_trim(cache, n_trim=1, n_iters=100):
    """Benchmark simple cache trim (slice off last n tokens)."""
    # Warm up
    for layer_cache in cache:
        _ = layer_cache.keys[:, :, :-n_trim, :]
    mx.eval()

    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        for layer_cache in cache:
            layer_cache.keys = layer_cache.keys[:, :, :-n_trim, :]
            layer_cache.values = layer_cache.values[:, :, :-n_trim, :]
        mx.eval()
        times.append(time.perf_counter() - start)

        # Restore cache size by padding (so we can repeat)
        for layer_cache in cache:
            pad_shape = list(layer_cache.keys.shape)
            pad_shape[2] = n_trim
            layer_cache.keys = mx.concatenate([
                layer_cache.keys,
                mx.zeros(pad_shape, dtype=layer_cache.keys.dtype)
            ], axis=2)
            layer_cache.values = mx.concatenate([
                layer_cache.values,
                mx.zeros(pad_shape, dtype=layer_cache.values.dtype)
            ], axis=2)
        mx.eval()

    return sum(times) / len(times) * 1000  # ms


def benchmark_cache_gather(cache, cache_len, tree_len, accepted_positions, n_iters=100):
    """Benchmark cache compaction with gather."""
    # Build gather indices
    cache_indices = list(range(cache_len))
    tree_indices_list = [cache_len + pos for pos in accepted_positions]
    keep_indices = mx.array(cache_indices + tree_indices_list)

    # Warm up
    for layer_cache in cache:
        _ = layer_cache.keys[:, :, keep_indices, :]
    mx.eval()

    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        for layer_cache in cache:
            layer_cache.keys = layer_cache.keys[:, :, keep_indices, :]
            layer_cache.values = layer_cache.values[:, :, keep_indices, :]
        mx.eval()
        times.append(time.perf_counter() - start)

        # Restore by padding tree positions back
        n_restore = tree_len - len(accepted_positions)
        if n_restore > 0:
            for layer_cache in cache:
                pad_shape = list(layer_cache.keys.shape)
                pad_shape[2] = n_restore
                layer_cache.keys = mx.concatenate([
                    layer_cache.keys,
                    mx.zeros(pad_shape, dtype=layer_cache.keys.dtype)
                ], axis=2)
                layer_cache.values = mx.concatenate([
                    layer_cache.values,
                    mx.zeros(pad_shape, dtype=layer_cache.values.dtype)
                ], axis=2)
            mx.eval()

    return sum(times) / len(times) * 1000  # ms


def benchmark_tree_mask_build(tree_attn_mask, cache_len, n_iters=100):
    """Benchmark tree attention mask construction."""
    # Warm up
    _ = build_tree_attention_mask_mlx(tree_attn_mask, cache_len)
    mx.eval()

    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        mask = build_tree_attention_mask_mlx(tree_attn_mask, cache_len)
        mx.eval(mask)
        times.append(time.perf_counter() - start)

    return sum(times) / len(times) * 1000  # ms


def benchmark_forward_with_mask(model, input_ids, cache, tree_attn_mask, tree_position_offsets, cache_len, n_iters=20):
    """Benchmark forward pass with tree attention mask."""
    tree_len = tree_attn_mask.shape[-1]

    # Create dummy tree input
    tree_input = mx.zeros((1, tree_len), dtype=mx.int32)

    # Build full mask
    full_mask = build_tree_attention_mask_mlx(tree_attn_mask, cache_len)

    # Warm up
    h = model._get_hidden_states(tree_input, cache=cache, tree_attn_mask=full_mask, tree_position_offsets=tree_position_offsets)
    mx.eval(h)

    times = []
    for _ in range(n_iters):
        # Trim cache back to cache_len before each iteration
        for layer_cache in cache:
            layer_cache.keys = layer_cache.keys[:, :, :cache_len, :]
            layer_cache.values = layer_cache.values[:, :, :cache_len, :]
            layer_cache.offset = cache_len
        mx.eval()

        start = time.perf_counter()
        h = model._get_hidden_states(tree_input, cache=cache, tree_attn_mask=full_mask, tree_position_offsets=tree_position_offsets)
        mx.eval(h)
        times.append(time.perf_counter() - start)

    return sum(times) / len(times) * 1000  # ms


def benchmark_forward_without_mask(model, input_ids, cache, n_tokens, cache_len, n_iters=20):
    """Benchmark forward pass without tree attention mask (standard causal)."""
    # Create dummy input
    token_input = mx.zeros((1, n_tokens), dtype=mx.int32)

    # Warm up - trim cache first
    for layer_cache in cache:
        layer_cache.keys = layer_cache.keys[:, :, :cache_len, :]
        layer_cache.values = layer_cache.values[:, :, :cache_len, :]
        layer_cache.offset = cache_len
    mx.eval()

    h = model._get_hidden_states(token_input, cache=cache)
    mx.eval(h)

    times = []
    for _ in range(n_iters):
        # Trim cache back to cache_len before each iteration
        for layer_cache in cache:
            layer_cache.keys = layer_cache.keys[:, :, :cache_len, :]
            layer_cache.values = layer_cache.values[:, :, :cache_len, :]
            layer_cache.offset = cache_len
        mx.eval()

        start = time.perf_counter()
        h = model._get_hidden_states(token_input, cache=cache)
        mx.eval(h)
        times.append(time.perf_counter() - start)

    return sum(times) / len(times) * 1000  # ms


def main():
    import os

    checkpoint_path = os.path.expanduser("~/.cache/nanochat/gemma_medusa_270m_wildchat_100k_nolora")

    print("Loading model...")
    model = GemmaMedusaModel.from_checkpoint(
        checkpoint_path=checkpoint_path,
        mlx_model_name="mlx-community/gemma-3-270m-it-bf16",
    )
    print("Model loaded!")

    # Create cache and prefill with some tokens
    cache = model.base_model.make_cache()
    prompt = "Hello, how are you today?"
    input_ids = model.tokenizer.encode(prompt)
    input_array = mx.array([input_ids], dtype=mx.int32)

    # Prefill
    h = model._get_hidden_states(input_array, cache=cache)
    mx.eval(h)

    cache_len = len(input_ids)
    print(f"\nCache length after prefill: {cache_len}")

    # Get tree buffers for 1 head
    tree_choices = DEFAULT_TREES[1]
    buffers = generate_tree_buffers(tree_choices, topk=10)
    tree_attn_mask = buffers["tree_attn_mask"]
    tree_position_offsets = buffers["tree_position_ids"]
    tree_len = tree_attn_mask.shape[-1]
    print(f"Tree length: {tree_len}")

    # Add tree tokens to cache (simulate forward pass with tree)
    for layer_cache in cache:
        pad_shape = list(layer_cache.keys.shape)
        pad_shape[2] = tree_len
        layer_cache.keys = mx.concatenate([
            layer_cache.keys,
            mx.zeros(pad_shape, dtype=layer_cache.keys.dtype)
        ], axis=2)
        layer_cache.values = mx.concatenate([
            layer_cache.values,
            mx.zeros(pad_shape, dtype=layer_cache.values.dtype)
        ], axis=2)
        layer_cache.offset = cache_len + tree_len
    mx.eval()

    print("\n" + "=" * 60)
    print("CACHE OPERATION BENCHMARKS")
    print("=" * 60)

    # Benchmark simple trim
    trim_time = benchmark_cache_trim(cache, n_trim=1, n_iters=100)
    print(f"Cache trim (1 token):      {trim_time:.3f} ms")

    trim_time_2 = benchmark_cache_trim(cache, n_trim=2, n_iters=100)
    print(f"Cache trim (2 tokens):     {trim_time_2:.3f} ms")

    # Benchmark gather/compact
    accepted_positions = [0, 1]  # Accept 2 tokens from tree
    gather_time = benchmark_cache_gather(cache, cache_len, tree_len, accepted_positions, n_iters=100)
    print(f"Cache gather (keep 2):     {gather_time:.3f} ms")

    accepted_positions_3 = [0, 1, 5]  # Accept 3 tokens from tree
    gather_time_3 = benchmark_cache_gather(cache, cache_len, tree_len, accepted_positions_3, n_iters=100)
    print(f"Cache gather (keep 3):     {gather_time_3:.3f} ms")

    print("\n" + "=" * 60)
    print("TREE ATTENTION MASK BENCHMARKS")
    print("=" * 60)

    # Benchmark mask building
    mask_time = benchmark_tree_mask_build(tree_attn_mask, cache_len, n_iters=100)
    print(f"Build tree mask:           {mask_time:.3f} ms")

    # Reset cache for forward benchmarks
    for layer_cache in cache:
        layer_cache.keys = layer_cache.keys[:, :, :cache_len, :]
        layer_cache.values = layer_cache.values[:, :, :cache_len, :]
        layer_cache.offset = cache_len
    mx.eval()

    print("\n" + "=" * 60)
    print("FORWARD PASS BENCHMARKS")
    print("=" * 60)

    # Benchmark forward with tree mask
    fwd_tree_time = benchmark_forward_with_mask(
        model, input_ids, cache, tree_attn_mask, tree_position_offsets, cache_len, n_iters=20
    )
    print(f"Forward (tree mask, {tree_len} tokens): {fwd_tree_time:.3f} ms")

    # Benchmark forward without mask (standard causal, same number of tokens)
    fwd_causal_time = benchmark_forward_without_mask(
        model, input_ids, cache, n_tokens=tree_len, cache_len=cache_len, n_iters=20
    )
    print(f"Forward (causal, {tree_len} tokens):    {fwd_causal_time:.3f} ms")

    # Benchmark forward for 2-token speculation
    fwd_2tok_time = benchmark_forward_without_mask(
        model, input_ids, cache, n_tokens=2, cache_len=cache_len, n_iters=20
    )
    print(f"Forward (causal, 2 tokens):   {fwd_2tok_time:.3f} ms")

    # Benchmark forward for 3-token speculation
    fwd_3tok_time = benchmark_forward_without_mask(
        model, input_ids, cache, n_tokens=3, cache_len=cache_len, n_iters=20
    )
    print(f"Forward (causal, 3 tokens):   {fwd_3tok_time:.3f} ms")

    print("\n" + "=" * 60)
    print("OVERHEAD ANALYSIS")
    print("=" * 60)

    tree_overhead = fwd_tree_time - fwd_causal_time
    print(f"Tree attention overhead:   {tree_overhead:.3f} ms ({tree_overhead/fwd_causal_time*100:.1f}%)")

    total_simple_2tok = fwd_2tok_time + trim_time
    total_simple_3tok = fwd_3tok_time + trim_time_2
    total_tree = fwd_tree_time + mask_time + gather_time

    print(f"\nPer-iteration cost estimates:")
    print(f"  Simple 2-tok: forward={fwd_2tok_time:.2f} + trim={trim_time:.2f} = {total_simple_2tok:.2f} ms")
    print(f"  Simple 3-tok: forward={fwd_3tok_time:.2f} + trim={trim_time_2:.2f} = {total_simple_3tok:.2f} ms")
    print(f"  Tree ({tree_len}):    forward={fwd_tree_time:.2f} + mask={mask_time:.2f} + gather={gather_time:.2f} = {total_tree:.2f} ms")


if __name__ == "__main__":
    main()
