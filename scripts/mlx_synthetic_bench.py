"""
Synthetic benchmark for MLX model forward pass timing.

Tests different combinations of:
- Context sizes (pre-filled KV cache): 1, 2, 4, 8, 16, 32, 64, 128, 256
- Input/verification sizes: 1, 2, 3, 4, 8, 16
- Each combination run 10 times

Usage:
    python -m scripts.mlx_synthetic_bench
    python -m scripts.mlx_synthetic_bench --checkpoint /path/to/checkpoint
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any

import mlx.core as mx

from nanochat.mlx.model import GemmaMedusaModel


def run_benchmark(
    model: GemmaMedusaModel,
    context_sizes: List[int],
    input_sizes: List[int],
    num_runs: int = 10,
) -> List[Dict[str, Any]]:
    """
    Run synthetic benchmark across all context/input size combinations.

    Args:
        model: Loaded GemmaMedusaModel
        context_sizes: List of pre-filled context sizes to test
        input_sizes: List of input/verification sizes to test
        num_runs: Number of runs per combination

    Returns:
        List of result dictionaries
    """
    results = []
    vocab_size = model.vocab_size

    # Use a fixed set of random tokens for consistency
    max_context = max(context_sizes)
    max_input = max(input_sizes)
    # Generate random tokens (using common token IDs to avoid special tokens)
    all_tokens = [100 + (i * 7) % 1000 for i in range(max_context + max_input)]

    # Global warmup - run a few forward passes to trigger JIT/shader compilation
    print("Warming up model (JIT compilation)...")
    warmup_cache = model.base_model.make_cache()
    warmup_context = mx.array([[100, 101, 102, 103]], dtype=mx.int32)
    warmup_input = mx.array([[104, 105, 106, 107]], dtype=mx.int32)

    # Warmup prefill
    h = model._get_hidden_states(warmup_context, cache=warmup_cache)
    main_logits, medusa_logits = model._compute_logits(h, return_medusa=True)
    mx.eval(main_logits, medusa_logits)

    # Warmup decode
    h = model._get_hidden_states(warmup_input, cache=warmup_cache)
    main_logits, medusa_logits = model._compute_logits(h, return_medusa=True)
    mx.eval(main_logits, medusa_logits)

    # Run a few more times to fully warm up
    for _ in range(3):
        warmup_cache = model.base_model.make_cache()
        h = model._get_hidden_states(warmup_context, cache=warmup_cache)
        main_logits, medusa_logits = model._compute_logits(h, return_medusa=True)
        mx.eval(main_logits, medusa_logits)

    print("Warmup complete.\n")

    total_combos = len(context_sizes) * len(input_sizes)
    combo_idx = 0

    for context_size in context_sizes:
        for input_size in input_sizes:
            combo_idx += 1
            print(f"\n[{combo_idx}/{total_combos}] Context={context_size}, Input={input_size}")

            times = []
            tokens_per_sec_list = []

            for run in range(num_runs):
                # Create fresh cache for each run
                cache = model.base_model.make_cache()

                # Prefill: process context tokens to populate KV cache
                context_tokens = all_tokens[:context_size]
                context_array = mx.array([context_tokens], dtype=mx.int32)

                # Prefill forward pass
                _ = model._get_hidden_states(context_array, cache=cache)
                mx.eval(cache[0].keys)  # Force computation

                # Now benchmark the verification/generation forward pass
                input_tokens = all_tokens[context_size:context_size + input_size]
                input_array = mx.array([input_tokens], dtype=mx.int32)

                # Timed forward pass
                mx.synchronize()
                start = time.perf_counter()

                hidden_states = model._get_hidden_states(input_array, cache=cache)
                main_logits, medusa_logits = model._compute_logits(
                    hidden_states, return_medusa=True, last_only=False
                )
                mx.eval(main_logits, medusa_logits)
                mx.synchronize()

                elapsed = time.perf_counter() - start
                times.append(elapsed)
                tokens_per_sec = input_size / elapsed if elapsed > 0 else 0
                tokens_per_sec_list.append(tokens_per_sec)

                print(f"  Run {run+1}/{num_runs}: {elapsed*1000:.2f}ms, {tokens_per_sec:.1f} tok/s")

            # Compute statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            avg_tok_s = sum(tokens_per_sec_list) / len(tokens_per_sec_list)

            result = {
                "context_size": context_size,
                "input_size": input_size,
                "num_runs": num_runs,
                "avg_time_ms": avg_time * 1000,
                "min_time_ms": min_time * 1000,
                "max_time_ms": max_time * 1000,
                "avg_tokens_per_sec": avg_tok_s,
                "all_times_ms": [t * 1000 for t in times],
            }
            results.append(result)

            print(f"  Avg: {avg_time*1000:.2f}ms, Min: {min_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
            print(f"  Avg tok/s: {avg_tok_s:.1f}")

    return results


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """Print a summary table of results."""
    # Get unique context and input sizes
    context_sizes = sorted(set(r["context_size"] for r in results))
    input_sizes = sorted(set(r["input_size"] for r in results))

    # Build lookup
    lookup = {(r["context_size"], r["input_size"]): r for r in results}

    # Print header
    print("\n" + "=" * 80)
    print("SUMMARY: Average Time (ms)")
    print("=" * 80)

    header = "Context\\Input |" + " | ".join(f"{s:>6}" for s in input_sizes)
    print(header)
    print("-" * len(header))

    for ctx in context_sizes:
        row = f"{ctx:>12} |"
        for inp in input_sizes:
            r = lookup.get((ctx, inp))
            if r:
                row += f" {r['avg_time_ms']:>6.2f}"
            else:
                row += f" {'N/A':>6}"
            row += " |"
        print(row)

    # Print tokens/sec table
    print("\n" + "=" * 80)
    print("SUMMARY: Tokens/sec")
    print("=" * 80)

    header = "Context\\Input |" + " | ".join(f"{s:>6}" for s in input_sizes)
    print(header)
    print("-" * len(header))

    for ctx in context_sizes:
        row = f"{ctx:>12} |"
        for inp in input_sizes:
            r = lookup.get((ctx, inp))
            if r:
                row += f" {r['avg_tokens_per_sec']:>6.1f}"
            else:
                row += f" {'N/A':>6}"
            row += " |"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Synthetic MLX model benchmark")
    parser.add_argument("--checkpoint", type=str,
                        default="~/.cache/nanochat/gemma_medusa_270m_wildchat_100k_nolora",
                        help="Medusa checkpoint path")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="Number of runs per combination")
    args = parser.parse_args()

    # Define benchmark parameters
    context_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    input_sizes = [1, 2, 3, 4, 8, 16]

    # Expand checkpoint path
    checkpoint_path = os.path.expanduser(args.checkpoint)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = GemmaMedusaModel.from_checkpoint(
        checkpoint_path=checkpoint_path,
        mlx_model_name="mlx-community/gemma-3-270m-it-bf16",
    )
    print("Model loaded!")

    print(f"\nRunning synthetic benchmark:")
    print(f"  Context sizes: {context_sizes}")
    print(f"  Input sizes: {input_sizes}")
    print(f"  Runs per combo: {args.num_runs}")
    print(f"  Total combinations: {len(context_sizes) * len(input_sizes)}")

    # Run benchmark
    results = run_benchmark(
        model=model,
        context_sizes=context_sizes,
        input_sizes=input_sizes,
        num_runs=args.num_runs,
    )

    # Print summary
    print_summary_table(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"mlx_synthetic_bench_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "context_sizes": context_sizes,
            "input_sizes": input_sizes,
            "num_runs": args.num_runs,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
