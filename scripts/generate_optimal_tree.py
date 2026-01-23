"""
Generate optimal Medusa tree structure from head recall data.

Uses greedy algorithm from Medusa paper:
- Start with root node (0,)
- Iteratively add the neighbor with highest expected acceptance probability
- Expected acceptance = product of recall@k values along the path

Usage:
    python -m scripts.generate_optimal_tree \
        --recall-files head_acc_gsm8k.json head_acc_mbpp.json \
        --num-heads 4 \
        --tree-size 79 \
        --topk 10

Output: Tree choices in format compatible with nanochat's tree generation.
"""

import argparse
import json
import copy
from typing import List, Dict, Tuple


def load_recall_data(files: List[str], num_heads: int) -> Dict[int, Dict[int, float]]:
    """
    Load and average recall data from multiple JSON files.

    Returns:
        accuracies: dict mapping head_idx -> k -> recall rate
    """
    # Accumulate recall values
    accumulated = {h: {} for h in range(num_heads)}
    num_files = 0

    for filepath in files:
        with open(filepath) as f:
            data = json.load(f)

        recall = data.get("recall", {})
        total_preds = data.get("total_predictions", {})

        # Skip files where heads have no predictions (like ARC/MMLU for heads 1-3)
        # Only use files that have meaningful data for all heads
        min_preds = min(total_preds.get(str(h), 0) for h in range(num_heads))
        if min_preds < 100:
            print(f"Skipping {filepath} - insufficient predictions for some heads")
            continue

        for h in range(num_heads):
            head_key = f"head_{h}"
            if head_key in recall:
                for k_str, val in recall[head_key].items():
                    k = int(k_str)
                    if k not in accumulated[h]:
                        accumulated[h][k] = []
                    accumulated[h][k].append(val)

        num_files += 1

    if num_files == 0:
        raise ValueError("No valid recall files found!")

    # Average the accumulated values
    averaged = {h: {} for h in range(num_heads)}
    for h in range(num_heads):
        for k, vals in accumulated[h].items():
            averaged[h][k] = sum(vals) / len(vals)

    print(f"Loaded recall data from {num_files} files")
    return averaged


def get_node_expectation(accuracies: Dict[int, Dict[int, float]], node: Tuple[int, ...], topk: int) -> float:
    """
    Compute expected acceptance probability for a node path.

    Node is a tuple like (0,), (0, 1), (2, 0, 3) representing:
    - First element: which top-k prediction from head 0 (depth 0)
    - Second element: which top-k prediction from head 1 (depth 1)
    - etc.

    Expected acceptance = product of recall@(rank+1) for each position.
    """
    expectation = 1.0
    for depth, rank in enumerate(node):
        # rank is 0-indexed, so rank=0 means top-1, rank=1 means top-2, etc.
        # We use recall@(rank+1) as the probability this candidate is correct
        k = min(rank + 1, topk)
        recall_at_k = accuracies[depth].get(k, 0.0)
        expectation *= recall_at_k
    return expectation


def explore_graph(
    accuracies: Dict[int, Dict[int, float]],
    max_depth: int,
    max_child: List[int],
    num_iterations: int,
    topk: int,
) -> List[Tuple[int, ...]]:
    """
    Greedy tree exploration algorithm from Medusa.

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
    explored_nodes[tuple([0])] = get_node_expectation(accuracies, (0,), topk)

    for iteration in range(num_iterations):
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
                expectation = get_node_expectation(accuracies, neighbor, topk)
                explored_nodes[neighbor] = expectation

            if expectation > best_expectation:
                best_neighbor = neighbor
                best_expectation = expectation

        if best_neighbor is None:
            print(f"No more valid neighbors at iteration {iteration}")
            break

        accept_nodes.append(best_neighbor)

    return accept_nodes


def format_tree_choices(accept_nodes: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """
    Format the tree for nanochat's tree generation.
    The accept_nodes already represent the tree structure.
    """
    # Sort by length (depth) then by values
    sorted_nodes = sorted(accept_nodes, key=lambda x: (len(x), x))
    return sorted_nodes


def analyze_tree(accept_nodes: List[Tuple[int, ...]], num_heads: int):
    """Print analysis of the generated tree."""
    print(f"\nTree Analysis:")
    print(f"  Total nodes: {len(accept_nodes)}")

    # Count nodes at each depth
    depth_counts = {}
    for node in accept_nodes:
        d = len(node) - 1  # depth is 0-indexed
        depth_counts[d] = depth_counts.get(d, 0) + 1

    print(f"  Nodes per depth:")
    for d in range(num_heads):
        count = depth_counts.get(d, 0)
        print(f"    Depth {d} (head {d}): {count} nodes")

    # Find max k used at each depth
    max_k_per_depth = {}
    for node in accept_nodes:
        for depth, rank in enumerate(node):
            max_k_per_depth[depth] = max(max_k_per_depth.get(depth, 0), rank + 1)

    print(f"  Max top-k used per depth:")
    for d in range(num_heads):
        max_k = max_k_per_depth.get(d, 0)
        print(f"    Depth {d}: top-{max_k}")


def main():
    parser = argparse.ArgumentParser(description="Generate optimal Medusa tree from recall data")
    parser.add_argument("--recall-files", nargs="+", required=True,
                        help="Paths to recall JSON files")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="Number of Medusa heads")
    parser.add_argument("--tree-size", type=int, default=79,
                        help="Target tree size (number of nodes)")
    parser.add_argument("--topk", type=int, default=10,
                        help="Max top-k to consider per head")
    parser.add_argument("--max-child", nargs="+", type=int, default=None,
                        help="Max children per depth (default: topk for all)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output JSON file for tree structure")

    args = parser.parse_args()

    # Load recall data
    accuracies = load_recall_data(args.recall_files, args.num_heads)

    # Print recall summary
    print("\nRecall Summary (top-1 / top-10):")
    for h in range(args.num_heads):
        r1 = accuracies[h].get(1, 0)
        r10 = accuracies[h].get(10, 0)
        print(f"  Head {h}: {r1:.3f} / {r10:.3f}")

    # Set max_child
    if args.max_child:
        max_child = args.max_child
    else:
        max_child = [args.topk] * args.num_heads

    # Generate tree
    print(f"\nGenerating tree with {args.tree_size} nodes...")
    accept_nodes = explore_graph(
        accuracies=accuracies,
        max_depth=args.num_heads,
        max_child=max_child,
        num_iterations=args.tree_size - 1,  # -1 because we start with root
        topk=args.topk,
    )

    # Analyze and format
    analyze_tree(accept_nodes, args.num_heads)

    # Format for output
    tree_choices = format_tree_choices(accept_nodes)

    print(f"\nTree choices (for nanochat):")
    print(f"  {tree_choices}")

    # Save to file
    if args.output:
        output_data = {
            "num_heads": args.num_heads,
            "tree_size": len(accept_nodes),
            "topk": args.topk,
            "max_child": max_child,
            "tree_choices": [list(node) for node in tree_choices],
            "recall_files": args.recall_files,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
