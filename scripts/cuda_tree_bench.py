"""
Synthetic benchmark for CUDA tree attention forward pass timing.

Tests different combinations of:
- Context sizes (pre-filled KV cache): 1, 2, 4, 8, 16, 32, 64, 128, 256
- Tree sizes: 1-21 (using optimal tree layout based on head accuracies)
- Each combination run 5 times

Usage:
    python -m scripts.cuda_tree_bench
    python -m scripts.cuda_tree_bench --checkpoint /path/to/checkpoint
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import torch

from nanochat.gemma_medusa.model import (
    GemmaMedusaModel,
    generate_tree_buffers,
)
from nanochat.gemma_common.speculative import build_tree_attention_mask


def _get_node_expectation(
    accuracies: Dict[int, Dict[int, float]],
    node: Tuple[int, ...],
    topk: int,
) -> float:
    """Calculate expected acceptance for a tree node."""
    expectation = 1.0
    for depth, rank in enumerate(node):
        if depth not in accuracies:
            return 0.0

        head_acc = accuracies[depth]
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
    """Greedy tree exploration algorithm from Medusa paper."""
    explored_nodes = {}
    accept_nodes = [tuple([0])]
    explored_nodes[tuple([0])] = _get_node_expectation(accuracies, (0,), topk)

    for _ in range(num_iterations):
        neighbors = []
        for node in accept_nodes:
            if node[-1] < max_child[len(node) - 1] - 1:
                neighbor = list(node)
                neighbor[-1] = neighbor[-1] + 1
                neighbors.append(tuple(neighbor))

            if len(node) < max_depth:
                neighbor = list(node)
                neighbor.append(0)
                neighbors.append(tuple(neighbor))

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
    """Generate optimal tree choices from a head_acc.json file."""
    if not os.path.exists(head_acc_path):
        return None

    try:
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

        max_child = [topk] * num_heads
        tree_choices = _explore_tree_greedy(
            accuracies=accuracies,
            max_depth=num_heads,
            max_child=max_child,
            num_iterations=tree_size - 1,
            topk=topk,
        )

        return tree_choices

    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def load_head_accuracies(head_acc_path: str, num_heads: int) -> Dict[int, Dict[int, float]]:
    """Load head accuracies from head_acc.json."""
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
    """Compute the expected mean number of tokens accepted per forward pass."""
    mean_acceptance = 1.0
    for node in tree_choices:
        prob = _get_node_expectation(accuracies, node, topk)
        mean_acceptance += prob
    return mean_acceptance


def get_tree_max_depth(tree_choices: List[Tuple[int, ...]]) -> int:
    """Get the maximum depth of the tree."""
    if not tree_choices:
        return 0
    return max(len(choice) for choice in tree_choices)


def run_benchmark(
    model: GemmaMedusaModel,
    context_sizes: List[int],
    tree_sizes: List[int],
    optimal_trees: Dict[int, List[Tuple[int, ...]]],
    accuracies: Dict[int, Dict[int, float]],
    num_runs: int = 5,
    topk: int = 10,
) -> List[Dict[str, Any]]:
    """Run synthetic benchmark across all context/tree size combinations."""
    results = []
    device = model.get_device()

    # Pre-generate tree buffers and compute mean acceptance for each tree size
    tree_buffers = {}
    tree_depths = {}
    tree_mean_acceptance = {}
    print("\nTree structures:")
    for tree_size in tree_sizes:
        choices = optimal_trees[tree_size]
        mean_acc = compute_tree_mean_acceptance(choices, accuracies, topk)
        tree_mean_acceptance[tree_size] = mean_acc
        max_depth = get_tree_max_depth(choices)
        tree_depths[tree_size] = max_depth
        if len(choices) == 0:
            tree_buffers[tree_size] = None
            print(f"  Tree size {tree_size}: [root only], depth={max_depth}, E[accept]={mean_acc:.2f}")
        else:
            tree_buffers[tree_size] = generate_tree_buffers(choices, device, topk=topk)
            actual_size = tree_buffers[tree_size]["tree_attn_mask"].shape[-1]
            print(f"  Tree size {tree_size}: {len(choices)} choices, positions={actual_size}, depth={max_depth}, E[accept]={mean_acc:.2f}")

    # Generate random tokens
    max_context = max(context_sizes)
    max_tree = max(tree_sizes)
    all_tokens = [100 + (i * 7) % 1000 for i in range(max_context + max_tree + 10)]

    # Warmup
    print("\nWarming up model...")
    for _ in range(3):
        warmup_input = torch.tensor([[100, 101, 102, 103]], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model.base_model.model(
                input_ids=warmup_input,
                use_cache=True,
                return_dict=True,
            )
            _ = model.base_model.lm_head(outputs.last_hidden_state)
        torch.cuda.synchronize()

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

            for run in range(num_runs):
                # Prefill context to create KV cache
                context_tokens = all_tokens[:context_size]
                context_array = torch.tensor([context_tokens], dtype=torch.long, device=device)

                with torch.no_grad():
                    outputs = model.base_model.model(
                        input_ids=context_array,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = outputs.past_key_values

                # Prepare tree input
                if buffers is None:
                    tree_len = 1
                    input_tokens = [all_tokens[context_size]]
                    input_array = torch.tensor([input_tokens], dtype=torch.long, device=device)
                    attention_mask = None
                    position_ids = torch.tensor([[context_size]], dtype=torch.long, device=device)
                else:
                    tree_len = buffers["tree_attn_mask"].shape[-1]
                    input_tokens = all_tokens[context_size:context_size + tree_len]
                    input_array = torch.tensor([input_tokens], dtype=torch.long, device=device)

                    # Build full attention mask (must match model dtype - bfloat16)
                    tree_mask = buffers["tree_attn_mask"].to(device)  # (1, 1, tree_len, tree_len)
                    attention_mask, _ = build_tree_attention_mask(tree_mask, context_size, dtype=torch.bfloat16)

                    # Position IDs - need (batch, seq_len) shape
                    position_offsets = buffers["tree_position_ids"]
                    position_ids = (torch.tensor([context_size], dtype=torch.long, device=device) + position_offsets.to(device)).unsqueeze(0)

                # Timed forward pass
                torch.cuda.synchronize()
                start = time.perf_counter()

                with torch.no_grad():
                    if attention_mask is None:
                        outputs = model.base_model.model(
                            input_ids=input_array,
                            past_key_values=past_key_values,
                            position_ids=position_ids,
                            use_cache=True,
                            return_dict=True,
                        )
                    else:
                        outputs = model.base_model.model(
                            input_ids=input_array,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=True,
                            return_dict=True,
                        )

                    hidden_states = outputs.last_hidden_state
                    main_logits = model.base_model.lm_head(hidden_states)

                    # Compute medusa logits
                    medusa_logits = model._compute_logits(
                        hidden_states,
                        return_medusa=True,
                    )

                torch.cuda.synchronize()
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

            mean_acc = tree_mean_acceptance[tree_size]
            fwd_per_sec = 1.0 / avg_time if avg_time > 0 else 0
            estimated_tok_s = fwd_per_sec * mean_acc

            result = {
                "context_size": context_size,
                "tree_size": tree_size,
                "tree_depth": tree_depths[tree_size],
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
    parser = argparse.ArgumentParser(description="Synthetic CUDA tree attention benchmark")
    parser.add_argument("--checkpoint", type=str,
                        default="~/.cache/nanochat/gemma_medusa_270m_wildchat_100k_kl",
                        help="Medusa checkpoint path")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Number of runs per combination")
    parser.add_argument("--topk", type=int, default=10,
                        help="Top-k for tree construction")
    args = parser.parse_args()

    checkpoint_path = os.path.expanduser(args.checkpoint)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    # Load head accuracies and generate optimal trees
    head_acc_path = os.path.join(checkpoint_path, "head_acc.json")
    print(f"Loading head accuracies from {head_acc_path}...")

    if not os.path.exists(head_acc_path):
        raise FileNotFoundError(f"head_acc.json not found at {head_acc_path}")

    # Define benchmark parameters
    context_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    tree_sizes = list(range(1, 22)) + [40, 80]

    max_tree_size = max(tree_sizes)
    print(f"\nBuilding optimal tree for max size {max_tree_size} with topk={args.topk}...")

    full_tree = generate_optimal_tree_from_head_acc(
        head_acc_path=head_acc_path,
        num_heads=4,
        tree_size=max_tree_size,
        topk=args.topk,
    )

    if full_tree is None:
        raise ValueError(f"Failed to generate tree from {head_acc_path}")

    print(f"Generated tree with {len(full_tree)} nodes")

    accuracies = load_head_accuracies(head_acc_path, num_heads=4)

    optimal_trees = {}
    for size in tree_sizes:
        if size == 1:
            optimal_trees[size] = []
        else:
            optimal_trees[size] = full_tree[:size - 1]

    # Load model
    print(f"\nLoading model from {checkpoint_path}...")

    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    model = GemmaMedusaModel(
        model_name=config.get('model_name', 'google/gemma-3-270m-it'),
        medusa_num_heads=config.get('medusa_num_heads', 4),
        medusa_num_layers=config.get('medusa_num_layers', 2),
        lora_rank=config.get('lora_rank', 256),
        lora_alpha=config.get('lora_alpha', 512),
        device=device,
        dtype=torch.bfloat16,
        freeze_base=True,
        zero_init_mlp=config.get('zero_init_mlp', True),
        use_head_mixer=config.get('use_head_mixer', False),
        mixer_type=config.get('mixer_type', 'mlp'),
        attn_num_layers=config.get('attn_num_layers', 0),
    )

    checkpoint_file = os.path.join(checkpoint_path, "final", "medusa_heads.pt")
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
    model.load_medusa_state_dict(checkpoint, strict=False)
    model.eval()

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
    output_file = f"cuda_tree_bench_{timestamp}.json"

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
