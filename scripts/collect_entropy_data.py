"""
Collect entropy data for adaptive speculation threshold analysis.

Uses wildchat data (separate from eval benchmarks) to avoid optimizing on test set.
Saves results in the checkpoint directory for proper association with the MTP setup.

Usage:
    conda activate fmtp-mlx
    python -m scripts.collect_entropy_data --mode 1h1t --n 50
    python -m scripts.collect_entropy_data --mode depth2 --n 50
    python -m scripts.collect_entropy_data --mode both --n 50
"""

import argparse
import json
import os
import time
from typing import List, Optional

import mlx.core as mx

from nanochat.mlx.model import GemmaMedusaModel, EntropyRecord


def load_wildchat_prompts(data_path: str, n_samples: int) -> List[str]:
    """Load and format prompts from wildchat data."""
    prompts = []

    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break

            try:
                data = json.loads(line)
                messages = data.get("messages", [])

                # Format as Gemma chat template
                prompt_parts = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")

                    if role == "user":
                        prompt_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
                    elif role == "assistant":
                        # Only include assistant content up to a point for generation
                        # We want to generate from user prompt, not replay assistant
                        break

                if prompt_parts:
                    # Add model turn start
                    prompt = "\n".join(prompt_parts) + "\n<start_of_turn>model\n"
                    prompts.append(prompt)

            except json.JSONDecodeError:
                continue

    return prompts


def collect_entropy_1h1t(
    model: GemmaMedusaModel,
    prompts: List[str],
    max_tokens: int = 64,
) -> List[EntropyRecord]:
    """Collect entropy data using 1 head, 1 token speculation."""
    all_records = []

    print(f"\nCollecting 1H×1T entropy data: {len(prompts)} prompts, max_tokens={max_tokens}")
    print("-" * 60)

    for i, prompt in enumerate(prompts):
        input_ids = model.tokenizer.encode(prompt)

        start = time.perf_counter()
        output_tokens, stats = model.generate_simple_speculation(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            stop_token_ids=model.tokenizer.eos_token_ids,
            collect_entropy=True,
        )
        elapsed = time.perf_counter() - start

        n_tokens = len(output_tokens) - len(input_ids)
        tok_s = n_tokens / elapsed if elapsed > 0 else 0

        if stats.entropy_log:
            all_records.extend(stats.entropy_log)

        print(f"  [{i+1}/{len(prompts)}] tokens={n_tokens}, tok/s={tok_s:.1f}, "
              f"iterations={len(stats.entropy_log) if stats.entropy_log else 0}")

    return all_records


def collect_entropy_depth2(
    model: GemmaMedusaModel,
    prompts: List[str],
    max_tokens: int = 64,
) -> List[EntropyRecord]:
    """Collect entropy data using 2 heads, 1 token per head speculation."""
    all_records = []

    print(f"\nCollecting 2H×1T (depth2) entropy data: {len(prompts)} prompts, max_tokens={max_tokens}")
    print("-" * 60)

    for i, prompt in enumerate(prompts):
        input_ids = model.tokenizer.encode(prompt)

        start = time.perf_counter()
        output_tokens, stats = model.generate_depth2_speculation(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            stop_token_ids=model.tokenizer.eos_token_ids,
            collect_entropy=True,
        )
        elapsed = time.perf_counter() - start

        n_tokens = len(output_tokens) - len(input_ids)
        tok_s = n_tokens / elapsed if elapsed > 0 else 0

        if stats.entropy_log:
            all_records.extend(stats.entropy_log)

        print(f"  [{i+1}/{len(prompts)}] tokens={n_tokens}, tok/s={tok_s:.1f}, "
              f"iterations={len(stats.entropy_log) if stats.entropy_log else 0}")

    return all_records


def analyze_entropy_data(records: List[EntropyRecord], mode: str) -> dict:
    """Analyze entropy data and compute statistics."""
    if not records:
        return {}

    entropies = [r.entropy for r in records]
    accept_lens = [r.accept_length for r in records]

    # Compute bucket statistics
    buckets = [(0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, float('inf'))]
    bucket_stats = []

    for low, high in buckets:
        bucket_records = [r for r in records if low <= r.entropy < high]
        if bucket_records:
            count = len(bucket_records)
            mean_accept = sum(r.accept_length for r in bucket_records) / count
            # Additional tokens = accept_length - 1 (the extra tokens beyond the base token)
            mean_additional = sum(r.accept_length - 1 for r in bucket_records) / count
            accept_rate = sum(1 for r in bucket_records if r.accept_length > 1) / count
            bucket_stats.append({
                "range": f"[{low}, {high})" if high != float('inf') else f"[{low}, inf)",
                "count": count,
                "mean_accept_length": mean_accept,
                "mean_additional_tokens": mean_additional,
                "speculation_success_rate": accept_rate,
            })

    return {
        "mode": mode,
        "total_iterations": len(records),
        "mean_entropy": sum(entropies) / len(entropies),
        "mean_accept_length": sum(accept_lens) / len(accept_lens),
        "entropy_percentiles": {
            "p25": sorted(entropies)[len(entropies) // 4],
            "p50": sorted(entropies)[len(entropies) // 2],
            "p75": sorted(entropies)[3 * len(entropies) // 4],
            "p90": sorted(entropies)[int(0.9 * len(entropies))],
            "p95": sorted(entropies)[int(0.95 * len(entropies))],
        },
        "bucket_analysis": bucket_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect entropy data for adaptive speculation")
    parser.add_argument("--mode", type=str, required=True, choices=["1h1t", "depth2", "both"],
                        help="Speculation mode to collect data for")
    parser.add_argument("--n", type=int, default=50, help="Number of wildchat prompts to process")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per generation")
    parser.add_argument("--checkpoint", type=str,
                        default="~/.cache/nanochat/gemma_medusa_270m_wildchat_100k_nolora",
                        help="Medusa checkpoint path")
    parser.add_argument("--data-path", type=str, default="data/wildchat_100k.jsonl",
                        help="Path to wildchat data")
    args = parser.parse_args()

    # Expand paths
    checkpoint_path = os.path.expanduser(args.checkpoint)

    # Verify data exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        return

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = GemmaMedusaModel.from_checkpoint(
        checkpoint_path=checkpoint_path,
        mlx_model_name="mlx-community/gemma-3-270m-it-bf16",
    )
    print("Model loaded!")

    # Load prompts
    print(f"\nLoading {args.n} prompts from {args.data_path}...")
    prompts = load_wildchat_prompts(args.data_path, args.n)
    print(f"Loaded {len(prompts)} prompts")

    results = {}

    # Collect entropy data
    if args.mode in ["1h1t", "both"]:
        records_1h1t = collect_entropy_1h1t(model, prompts, args.max_tokens)
        results["1h1t"] = {
            "records": [{"entropy": r.entropy, "accept_length": r.accept_length} for r in records_1h1t],
            "analysis": analyze_entropy_data(records_1h1t, "1h1t"),
        }

        print(f"\n1H×1T Results:")
        print(f"  Total iterations: {len(records_1h1t)}")
        print(f"  Mean entropy: {results['1h1t']['analysis']['mean_entropy']:.4f}")
        print(f"  Mean accept length: {results['1h1t']['analysis']['mean_accept_length']:.3f}")

    if args.mode in ["depth2", "both"]:
        records_depth2 = collect_entropy_depth2(model, prompts, args.max_tokens)
        results["depth2"] = {
            "records": [{"entropy": r.entropy, "accept_length": r.accept_length} for r in records_depth2],
            "analysis": analyze_entropy_data(records_depth2, "depth2"),
        }

        print(f"\n2H×1T (depth2) Results:")
        print(f"  Total iterations: {len(records_depth2)}")
        print(f"  Mean entropy: {results['depth2']['analysis']['mean_entropy']:.4f}")
        print(f"  Mean accept length: {results['depth2']['analysis']['mean_accept_length']:.3f}")

    # Save to checkpoint directory
    output_path = os.path.join(checkpoint_path, f"entropy_data_{args.mode}.json")
    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "data_source": args.data_path,
                "n_prompts": len(prompts),
                "max_tokens": args.max_tokens,
                "mode": args.mode,
            },
            **results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Entropy data saved to: {output_path}")
    print(f"{'='*60}")

    # Print bucket analysis
    for mode_key in ["1h1t", "depth2"]:
        if mode_key in results:
            print(f"\n{mode_key.upper()} Bucket Analysis:")
            print(f"{'Entropy Range':<15} {'Count':>8} {'Mean Accept':>12} {'Mean Extra':>11} {'Success':>9}")
            print("-" * 58)
            for bucket in results[mode_key]["analysis"]["bucket_analysis"]:
                print(f"{bucket['range']:<15} {bucket['count']:>8} "
                      f"{bucket['mean_accept_length']:>12.2f} "
                      f"{bucket['mean_additional_tokens']:>11.2f} "
                      f"{bucket['speculation_success_rate']*100:>8.1f}%")


if __name__ == "__main__":
    main()
