"""
Synthetic benchmark for MLX tree attention forward pass timing.

Tests different combinations of:
- Context sizes (pre-filled KV cache): 1, 2, 4, 8, 16, 32, 64, 128, 256
- Tree sizes: 1-21 (using optimal tree layout based on head accuracies)
- Each combination run 10 times

Usage:
    python -m scripts.mlx_tree_bench
    python -m scripts.mlx_tree_bench --checkpoint /path/to/checkpoint
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import mlx.core as mx

from nanochat.mlx.model import (
    GemmaMedusaModel,
    generate_tree_buffers,
    build_tree_attention_mask_mlx,
)


# =============================================================================
# Optimal tree building (copied from nanochat/gemma_medusa/model.py to keep
# MLX module separate from PyTorch dependencies)
# =============================================================================

def _get_node_expectation(
    accuracies: Dict[int, Dict[int, float]],
    node: Tuple[int, ...],
    topk: int,
) -> float:
    """
    Calculate expected acceptance for a tree node.

    The expectation is the product of acceptance probabilities along the path.
    For node (i, j, k), this is: P(head_0 accepts rank i) * P(head_1 accepts rank j) * ...

    Args:
        accuracies: head_idx -> k -> recall rate (cumulative)
        node: Tuple of ranks at each depth
        topk: Max k value

    Returns:
        Expected acceptance probability for this node
    """
    expectation = 1.0
    for depth, rank in enumerate(node):
        if depth not in accuracies:
            return 0.0

        head_acc = accuracies[depth]
        # Convert cumulative recall to per-rank probability
        # P(rank = k) = recall[k+1] - recall[k]
        # Keys are ints: {1: 0.37, 2: 0.48, ...}
        k_cur = rank + 1
        k_prev = rank if rank > 0 else None

        if k_cur in head_acc:
            if k_prev and k_prev in head_acc:
                prob = head_acc[k_cur] - head_acc[k_prev]
            else:
                prob = head_acc[k_cur]
        else:
            prob = 0.0

        expectation *= prob

    return expectation


def _explore_tree_greedy(
    accuracies: Dict[int, Dict[int, float]],
    max_depth: int,
    max_child: List[int],
    num_iterations: int,
    topk: int,
) -> List[Tuple[int, ...]]:
    """
    Greedy tree exploration algorithm from Medusa paper.

    Args:
        accuracies: head_idx -> k -> recall rate
        max_depth: maximum tree depth (= num_heads)
        max_child: max children per depth level
        num_iterations: number of nodes to add (tree_size - 1)
        topk: max k value to consider

    Returns:
        List of accepted node tuples representing the tree
    """
    explored_nodes = {}
    accept_nodes = [tuple([0])]  # Start with root: top-1 from head 0
    explored_nodes[tuple([0])] = _get_node_expectation(accuracies, (0,), topk)

    for _ in range(num_iterations):
        # Find all neighbor nodes
        neighbors = []
        for node in accept_nodes:
            # Option 1: Increment last element (try next top-k at same depth)
            if node[-1] < max_child[len(node) - 1] - 1:
                neighbor = list(node)
                neighbor[-1] = neighbor[-1] + 1
                neighbors.append(tuple(neighbor))

            # Option 2: Extend to next depth (add child from next head)
            if len(node) < max_depth:
                neighbor = list(node)
                neighbor.append(0)
                neighbors.append(tuple(neighbor))

        # Find best neighbor not already accepted
        best_neighbor = None
        best_expectation = 0

        for neighbor in neighbors:
            if neighbor in accept_nodes:
                continue

            if neighbor in explored_nodes:
                expectation = explored_nodes[neighbor]
            else:
                expectation = _get_node_expectation(accuracies, neighbor, topk)
                explored_nodes[neighbor] = expectation

            if expectation > best_expectation:
                best_neighbor = neighbor
                best_expectation = expectation

        if best_neighbor is None:
            break

        accept_nodes.append(best_neighbor)

    return accept_nodes


def generate_optimal_tree_from_head_acc(
    head_acc_path: str,
    num_heads: int,
    tree_size: int = 79,
    topk: int = 64,
) -> Optional[List[Tuple[int, ...]]]:
    """
    Generate optimal tree choices from a head_acc.json file.

    Args:
        head_acc_path: Path to head_acc.json file from checkpoint
        num_heads: Number of Medusa heads to use
        tree_size: Target tree size (default 79)
        topk: Max top-k to consider per head (default 64)

    Returns:
        List of tree node tuples, or None if file doesn't exist/is invalid
    """
    if not os.path.exists(head_acc_path):
        return None

    try:
        with open(head_acc_path) as f:
            data = json.load(f)

        recall = data.get("recall", {})

        # Convert recall data to format expected by tree generation
        # recall format: {"head_0": {"1": 0.65, "2": 0.72, ...}, ...}
        accuracies: Dict[int, Dict[int, float]] = {}
        for h in range(num_heads):
            head_key = f"head_{h}"
            if head_key in recall:
                accuracies[h] = {int(k): v for k, v in recall[head_key].items()}
            else:
                accuracies[h] = {}

        # Generate tree using greedy algorithm
        max_child = [topk] * num_heads
        tree_choices = _explore_tree_greedy(
            accuracies=accuracies,
            max_depth=num_heads,
            max_child=max_child,
            num_iterations=tree_size - 1,  # -1 because we start with root
            topk=topk,
        )

        return tree_choices

    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def load_head_accuracies(head_acc_path: str, num_heads: int) -> Dict[int, Dict[int, float]]:
    """
    Load head accuracies from head_acc.json.

    Returns:
        Dict mapping head_idx -> k -> cumulative recall
    """
    with open(head_acc_path) as f:
        data = json.load(f)

    recall = data.get("recall", {})
    accuracies: Dict[int, Dict[int, float]] = {}
    for h in range(num_heads):
        head_key = f"head_{h}"
        if head_key in recall:
            accuracies[h] = {int(k): v for k, v in recall[head_key].items()}
        else:
            accuracies[h] = {}
    return accuracies


def compute_tree_mean_acceptance(
    tree_choices: List[Tuple[int, ...]],
    accuracies: Dict[int, Dict[int, float]],
    topk: int = 64,
) -> float:
    """
    Compute the expected mean number of tokens accepted per forward pass.

    Mean acceptance = 1 (root always accepted) + sum of P(node) for each tree node.

    Args:
        tree_choices: List of tree node tuples (excluding root)
        accuracies: head_idx -> k -> cumulative recall
        topk: Max k value

    Returns:
        Expected mean tokens accepted per forward pass
    """
    # Root is always accepted (1 token)
    mean_acceptance = 1.0

    # Add probability of each node being accepted
    for node in tree_choices:
        prob = _get_node_expectation(accuracies, node, topk)
        mean_acceptance += prob

    return mean_acceptance


# =============================================================================
# Benchmark code
# =============================================================================

def get_tree_max_depth(tree_choices: List[Tuple[int, ...]]) -> int:
    """Get the maximum depth of the tree (= number of heads needed).

    Returns 0 for empty tree (standard decoding - no Medusa heads needed).
    """
    if not tree_choices:
        return 0  # No Medusa heads needed for standard decoding
    return max(len(choice) for choice in tree_choices)


def run_benchmark(
    model: GemmaMedusaModel,
    context_sizes: List[int],
    tree_sizes: List[int],
    optimal_trees: Dict[int, List[Tuple[int, ...]]],
    accuracies: Dict[int, Dict[int, float]],
    num_runs: int = 10,
    topk: int = 10,
) -> List[Dict[str, Any]]:
    """
    Run synthetic benchmark across all context/tree size combinations.
    """
    results = []

    # Pre-generate tree buffers and compute mean acceptance for each tree size
    tree_buffers = {}
    tree_depths = {}  # Track max depth for each tree size
    tree_mean_acceptance = {}  # Track expected mean acceptance for each tree
    print("\nTree structures:")
    for tree_size in tree_sizes:
        choices = optimal_trees[tree_size]
        mean_acc = compute_tree_mean_acceptance(choices, accuracies, topk)
        tree_mean_acceptance[tree_size] = mean_acc
        max_depth = get_tree_max_depth(choices)
        tree_depths[tree_size] = max_depth
        if len(choices) == 0:
            tree_buffers[tree_size] = None
            print(f"  Tree size {tree_size}: [root only], depth={max_depth} (standard decoding), E[accept]={mean_acc:.2f}")
        else:
            tree_buffers[tree_size] = generate_tree_buffers(choices, topk=topk)
            actual_size = tree_buffers[tree_size]["tree_attn_mask"].shape[-1]
            print(f"  Tree size {tree_size}: {len(choices)} choices, positions={actual_size}, depth={max_depth}, E[accept]={mean_acc:.2f}")
            # Show first few choices
            if len(choices) <= 5:
                print(f"    Choices: {choices}")
            else:
                print(f"    First 5: {choices[:5]}...")

    # Generate random tokens
    max_context = max(context_sizes)
    max_tree = max(tree_sizes)
    all_tokens = [100 + (i * 7) % 1000 for i in range(max_context + max_tree + 10)]

    # Global warmup - warmup with different head counts (including 0 for standard decoding)
    print("\nWarming up model (JIT compilation)...")
    for num_heads in [0, 1, 2, 3, 4]:
        warmup_cache = model.base_model.make_cache()
        warmup_context = mx.array([[100, 101, 102, 103]], dtype=mx.int32)

        # Prefill
        h = model._get_hidden_states(warmup_context, cache=warmup_cache)
        main_logits, medusa_logits = model._compute_logits(
            h, return_medusa=True, num_active_heads=num_heads
        )
        mx.eval(main_logits, medusa_logits)

        # Decode
        warmup_input = mx.array([[104, 105, 106, 107, 108, 109, 110, 111]], dtype=mx.int32)
        h = model._get_hidden_states(warmup_input, cache=warmup_cache)
        main_logits, medusa_logits = model._compute_logits(
            h, return_medusa=True, num_active_heads=num_heads
        )
        mx.eval(main_logits, medusa_logits)

    # Warmup tree attention for a few sizes
    for tree_size in [1, 5, 10, 15, 21]:
        if tree_size not in tree_sizes:
            continue
        warmup_cache = model.base_model.make_cache()
        warmup_context = mx.array([[100, 101, 102, 103]], dtype=mx.int32)

        h = model._get_hidden_states(warmup_context, cache=warmup_cache)
        mx.eval(h)

        buffers = tree_buffers.get(tree_size)
        num_heads = tree_depths[tree_size]
        if buffers is None:
            input_array = mx.array([[104]], dtype=mx.int32)
            h = model._get_hidden_states(input_array, cache=warmup_cache)
        else:
            tree_len = buffers["tree_attn_mask"].shape[-1]
            tree_tokens = mx.array([[104 + i for i in range(tree_len)]], dtype=mx.int32)
            cache_len = 4
            full_mask = build_tree_attention_mask_mlx(buffers["tree_attn_mask"], cache_len)
            h = model._get_hidden_states(
                tree_tokens,
                cache=warmup_cache,
                tree_attn_mask=full_mask,
                tree_position_offsets=buffers["tree_position_ids"]
            )
        main_logits, medusa_logits = model._compute_logits(
            h, return_medusa=True, num_active_heads=num_heads
        )
        mx.eval(main_logits, medusa_logits)

    print("Warmup complete.\n")

    total_combos = len(context_sizes) * len(tree_sizes)
    combo_idx = 0

    for context_size in context_sizes:
        for tree_size in tree_sizes:
            combo_idx += 1
            print(f"\n[{combo_idx}/{total_combos}] Context={context_size}, Tree={tree_size}")

            times = []
            tokens_per_sec_list = []
            buffers = tree_buffers.get(tree_size)
            num_active_heads = tree_depths[tree_size]

            for run in range(num_runs):
                # Create fresh cache
                cache = model.base_model.make_cache()

                # Prefill context
                context_tokens = all_tokens[:context_size]
                context_array = mx.array([context_tokens], dtype=mx.int32)
                _ = model._get_hidden_states(context_array, cache=cache)
                mx.eval(cache[0].keys)

                # Prepare tree input
                if buffers is None:
                    tree_len = 1
                    input_tokens = [all_tokens[context_size]]
                    input_array = mx.array([input_tokens], dtype=mx.int32)
                    tree_attn_mask = None
                    tree_position_offsets = None
                else:
                    tree_len = buffers["tree_attn_mask"].shape[-1]
                    input_tokens = all_tokens[context_size:context_size + tree_len]
                    input_array = mx.array([input_tokens], dtype=mx.int32)
                    tree_attn_mask = build_tree_attention_mask_mlx(
                        buffers["tree_attn_mask"], context_size
                    )
                    tree_position_offsets = buffers["tree_position_ids"]

                # Timed forward pass
                mx.synchronize()
                start = time.perf_counter()

                if tree_attn_mask is None:
                    hidden_states = model._get_hidden_states(input_array, cache=cache)
                else:
                    hidden_states = model._get_hidden_states(
                        input_array,
                        cache=cache,
                        tree_attn_mask=tree_attn_mask,
                        tree_position_offsets=tree_position_offsets,
                    )

                # Only compute as many heads as needed for this tree depth
                main_logits, medusa_logits = model._compute_logits(
                    hidden_states,
                    return_medusa=True,
                    last_only=False,
                    num_active_heads=num_active_heads,
                )
                mx.eval(main_logits, medusa_logits)
                mx.synchronize()

                elapsed = time.perf_counter() - start
                times.append(elapsed)
                tokens_per_sec = tree_len / elapsed if elapsed > 0 else 0
                tokens_per_sec_list.append(tokens_per_sec)

                print(f"  Run {run+1}/{num_runs}: {elapsed*1000:.2f}ms, {tokens_per_sec:.1f} tok/s")

            # Compute statistics (skip first 5 runs as warmup)
            warmup = 5
            valid_times = times[warmup:] if len(times) > warmup else times
            valid_tok_s = tokens_per_sec_list[warmup:] if len(tokens_per_sec_list) > warmup else tokens_per_sec_list
            avg_time = sum(valid_times) / len(valid_times)
            min_time = min(valid_times)
            max_time = max(valid_times)
            avg_tok_s = sum(valid_tok_s) / len(valid_tok_s)

            # Compute estimated effective tokens/sec based on mean acceptance
            mean_acc = tree_mean_acceptance[tree_size]
            # Forward passes per second = 1 / avg_time
            # Estimated effective tok/s = forward_passes_per_sec * mean_acceptance
            fwd_per_sec = 1.0 / avg_time if avg_time > 0 else 0
            estimated_tok_s = fwd_per_sec * mean_acc

            result = {
                "context_size": context_size,
                "tree_size": tree_size,
                "tree_depth": num_active_heads,
                "mean_acceptance": mean_acc,
                "num_runs": num_runs,
                "avg_time_ms": avg_time * 1000,
                "min_time_ms": min_time * 1000,
                "max_time_ms": max_time * 1000,
                "avg_tokens_per_sec": avg_tok_s,
                "estimated_effective_tok_s": estimated_tok_s,
                "all_times_ms": [t * 1000 for t in times],
            }
            results.append(result)

            print(f"  Avg: {avg_time*1000:.2f}ms, Min: {min_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
            print(f"  Avg tok/s: {avg_tok_s:.1f}, Est. effective tok/s: {estimated_tok_s:.1f} (E[accept]={mean_acc:.2f})")

    return results


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """Print summary tables."""
    context_sizes = sorted(set(r["context_size"] for r in results))
    tree_sizes = sorted(set(r["tree_size"] for r in results))
    lookup = {(r["context_size"], r["tree_size"]): r for r in results}

    # Time table
    print("\n" + "=" * 130)
    print("AVERAGE TIME (ms)")
    print("=" * 130)
    header = "Ctx\\Tree|" + "|".join(f"{s:>5}" for s in tree_sizes)
    print(header)
    print("-" * len(header))
    for ctx in context_sizes:
        row = f"{ctx:>7} |"
        for tree in tree_sizes:
            r = lookup.get((ctx, tree))
            row += f"{r['avg_time_ms']:>5.1f}|" if r else "  N/A|"
        print(row)

    # Tokens/sec table (raw throughput)
    print("\n" + "=" * 130)
    print("RAW TOKENS/SEC (tree_size / time)")
    print("=" * 130)
    print(header)
    print("-" * len(header))
    for ctx in context_sizes:
        row = f"{ctx:>7} |"
        for tree in tree_sizes:
            r = lookup.get((ctx, tree))
            row += f"{r['avg_tokens_per_sec']:>5.0f}|" if r else "  N/A|"
        print(row)

    # Estimated effective tokens/sec table
    print("\n" + "=" * 130)
    print("ESTIMATED EFFECTIVE TOK/SEC (mean_acceptance / time)")
    print("=" * 130)
    print(header)
    print("-" * len(header))
    for ctx in context_sizes:
        row = f"{ctx:>7} |"
        for tree in tree_sizes:
            r = lookup.get((ctx, tree))
            row += f"{r['estimated_effective_tok_s']:>5.0f}|" if r else "  N/A|"
        print(row)

    # Min time table
    print("\n" + "=" * 130)
    print("MIN TIME (ms)")
    print("=" * 130)
    print(header)
    print("-" * len(header))
    for ctx in context_sizes:
        row = f"{ctx:>7} |"
        for tree in tree_sizes:
            r = lookup.get((ctx, tree))
            row += f"{r['min_time_ms']:>5.1f}|" if r else "  N/A|"
        print(row)

    # Print tree depth and mean acceptance info
    print("\n" + "=" * 130)
    print("TREE DEPTH (num heads computed)")
    print("=" * 130)
    depths = {r["tree_size"]: r["tree_depth"] for r in results}
    depth_row = "Depth:  |" + "|".join(f"{depths.get(s, 0):>5}" for s in tree_sizes)
    print(depth_row)

    print("\n" + "=" * 130)
    print("MEAN ACCEPTANCE (expected tokens per forward pass)")
    print("=" * 130)
    mean_accs = {r["tree_size"]: r["mean_acceptance"] for r in results}
    acc_row = "E[acc]: |" + "|".join(f"{mean_accs.get(s, 0):>5.2f}" for s in tree_sizes)
    print(acc_row)


def main():
    parser = argparse.ArgumentParser(description="Synthetic MLX tree attention benchmark")
    parser.add_argument("--checkpoint", type=str,
                        default="~/.cache/nanochat/gemma_medusa_270m_wildchat_100k_nolora",
                        help="Medusa checkpoint path")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="Number of runs per combination")
    parser.add_argument("--topk", type=int, default=10,
                        help="Top-k for tree construction")
    args = parser.parse_args()

    checkpoint_path = os.path.expanduser(args.checkpoint)

    # Load head accuracies and generate optimal trees
    head_acc_path = os.path.join(checkpoint_path, "head_acc.json")
    print(f"Loading head accuracies from {head_acc_path}...")

    if not os.path.exists(head_acc_path):
        raise FileNotFoundError(f"head_acc.json not found at {head_acc_path}")

    # Define benchmark parameters
    context_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    tree_sizes = list(range(1, 22)) + [40, 80]  # 1 to 21, plus 40 and 80

    # Build optimal trees for all sizes at once (use max size)
    max_tree_size = max(tree_sizes)
    print(f"\nBuilding optimal tree for max size {max_tree_size} with topk={args.topk}...")

    full_tree = generate_optimal_tree_from_head_acc(
        head_acc_path=head_acc_path,
        num_heads=4,  # 4 Medusa heads
        tree_size=max_tree_size,
        topk=args.topk,
    )

    if full_tree is None:
        raise ValueError(f"Failed to generate tree from {head_acc_path}")

    print(f"Generated tree with {len(full_tree)} nodes")

    # Load head accuracies for mean acceptance calculation
    accuracies = load_head_accuracies(head_acc_path, num_heads=4)

    # Create trees for each size by taking prefix
    optimal_trees = {}
    for size in tree_sizes:
        if size == 1:
            optimal_trees[size] = []
        else:
            optimal_trees[size] = full_tree[:size - 1]

    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = GemmaMedusaModel.from_checkpoint(
        checkpoint_path=checkpoint_path,
        mlx_model_name="mlx-community/gemma-3-270m-it-bf16",
    )
    print("Model loaded!")

    print(f"\nRunning tree attention benchmark:")
    print(f"  Context sizes: {context_sizes}")
    print(f"  Tree sizes: {tree_sizes}")
    print(f"  Runs per combo: {args.num_runs}")
    print(f"  Total combinations: {len(context_sizes) * len(tree_sizes)}")

    results = run_benchmark(
        model=model,
        context_sizes=context_sizes,
        tree_sizes=tree_sizes,
        optimal_trees=optimal_trees,
        accuracies=accuracies,
        num_runs=args.num_runs,
        topk=args.topk,
    )

    print_summary_table(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"mlx_tree_bench_{timestamp}.json"

    # Convert optimal_trees tuples to lists for JSON serialization
    trees_for_json = {k: [list(t) for t in v] for k, v in optimal_trees.items()}

    with open(output_file, "w") as f:
        json.dump({
            "context_sizes": context_sizes,
            "tree_sizes": tree_sizes,
            "optimal_trees": trees_for_json,
            "num_runs": args.num_runs,
            "topk": args.topk,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
