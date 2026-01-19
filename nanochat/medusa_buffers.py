"""
Medusa tree attention buffer generation for speculative decoding.

This module generates the data structures needed for tree-based speculative verification:
- Tree attention mask: allows each node to attend to its ancestors
- Tree indices: maps flat candidate array to tree structure
- Position IDs: depth-based position offsets for RoPE
- Retrieve indices: maps tree paths back to candidate sequences
"""

import torch
from typing import List, Tuple, Dict, Optional


def generate_medusa_buffers(
    medusa_choices: List[Tuple[int, ...]],
    device: str = "cuda",
    topk: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Generate buffers for tree attention in Medusa speculative decoding.

    Args:
        medusa_choices: List of tuples defining the tree structure.
            Each tuple (i, j, k, ...) represents a path through the tree:
            - Position 1: i-th top prediction from head 0
            - Position 2: j-th top prediction from head 1
            - etc.
        device: Device to place tensors on
        topk: Number of top predictions from each Medusa head

    Returns:
        Dictionary containing:
        - medusa_attn_mask: (1, 1, tree_len, tree_len) Tree attention mask
        - tree_indices: (tree_len,) Maps positions to candidate token indices
        - medusa_position_ids: (tree_len,) Position offsets for each tree node
        - retrieve_indices: (num_candidates, max_depth) Maps paths to node indices

    Example:
        >>> choices = [(0,), (0, 0), (1,), (0, 1)]
        >>> buffers = generate_medusa_buffers(choices, device="cpu", topk=10)
        >>> buffers["medusa_attn_mask"].shape
        torch.Size([1, 1, 5, 5])  # 4 choices + 1 root
    """
    # Sort choices by (depth, lexicographic order) for consistent indexing
    sorted_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    tree_len = len(sorted_choices) + 1  # +1 for root token

    # 1. Create tree attention mask
    # Each node can attend to: itself, root, and all ancestors
    attn_mask = torch.zeros(tree_len, tree_len, device=device)

    # Diagonal: each node attends to itself
    attn_mask.fill_diagonal_(1.0)

    # All nodes can see the root (position 0)
    attn_mask[:, 0] = 1.0

    # For each non-root node, allow attention to ancestors
    for idx, choice in enumerate(sorted_choices):
        node_idx = idx + 1  # +1 because root is at position 0

        # Find all ancestors and allow attention to them
        for depth in range(len(choice) - 1):
            ancestor_choice = choice[: depth + 1]
            ancestor_idx = sorted_choices.index(ancestor_choice) + 1
            attn_mask[node_idx, ancestor_idx] = 1.0

    # 2. Tree indices: map tree positions to candidate token indices
    # Format: index = token_rank + topk * depth + 1 (0 is reserved for root)
    tree_indices = torch.zeros(tree_len, dtype=torch.long, device=device)
    tree_indices[0] = 0  # Root maps to position 0 (the base model prediction)

    for idx, choice in enumerate(sorted_choices):
        node_idx = idx + 1
        depth = len(choice) - 1  # 0-indexed depth (first head = depth 0)
        token_rank = choice[-1]  # Which top-k prediction to use
        # Map to flat candidate array: [root, head0_top0..topk-1, head1_top0..topk-1, ...]
        tree_indices[node_idx] = token_rank + topk * depth + 1

    # 3. Position IDs: depth in tree determines position offset for RoPE
    # Root is at position 0, depth-1 nodes at position 1, etc.
    position_ids = torch.zeros(tree_len, dtype=torch.long, device=device)
    for idx, choice in enumerate(sorted_choices):
        position_ids[idx + 1] = len(choice)  # Depth = length of choice tuple

    # 4. Retrieve indices: for extracting complete candidate paths
    # Each row is a path from root to a leaf (or internal node)
    max_depth = max(len(c) for c in sorted_choices) + 1  # +1 for root
    num_candidates = len(sorted_choices) + 1  # Include root as a candidate (accept 0 tokens)

    # Initialize with -1 (padding for shorter paths)
    retrieve_indices = torch.full(
        (num_candidates, max_depth), -1, dtype=torch.long, device=device
    )

    # Root candidate (path of length 1: just the root)
    retrieve_indices[0, 0] = 0

    # Fill in paths for each choice
    for idx, choice in enumerate(sorted_choices):
        candidate_idx = idx + 1
        retrieve_indices[candidate_idx, 0] = 0  # Start with root

        # Add each node in the path
        for depth in range(len(choice)):
            partial_choice = choice[: depth + 1]
            node_idx = sorted_choices.index(partial_choice) + 1
            retrieve_indices[candidate_idx, depth + 1] = node_idx

    return {
        "medusa_attn_mask": attn_mask.unsqueeze(0).unsqueeze(0),  # (1, 1, tree_len, tree_len)
        "tree_indices": tree_indices,
        "medusa_position_ids": position_ids,
        "retrieve_indices": retrieve_indices,
    }


def get_default_medusa_choices(num_heads: int, topk: int = 10) -> List[Tuple[int, ...]]:
    """
    Generate default tree configuration based on number of Medusa heads.

    Creates a balanced tree that trades off between:
    - Coverage: more candidates = higher chance of accepting longer sequences
    - Overhead: more candidates = larger tree attention computation

    Args:
        num_heads: Number of Medusa prediction heads
        topk: Maximum top-k predictions to consider from each head

    Returns:
        List of choice tuples defining the tree structure

    Example:
        >>> choices = get_default_medusa_choices(4, topk=10)
        >>> len(choices)  # Number of tree nodes (excluding root)
        63
    """
    choices = []

    if num_heads >= 1:
        # Single-level: top predictions from head 0
        for i in range(min(topk, 10)):
            choices.append((i,))

    if num_heads >= 2:
        # Two-level paths
        for i in range(min(topk, 5)):
            for j in range(min(topk, 5)):
                choices.append((i, j))

    if num_heads >= 3:
        # Three-level paths (more selective to limit tree size)
        for i in range(min(topk, 3)):
            for j in range(min(topk, 3)):
                for k in range(min(topk, 3)):
                    choices.append((i, j, k))

    if num_heads >= 4:
        # Four-level paths (very selective)
        for i in range(min(topk, 2)):
            for j in range(min(topk, 2)):
                for k in range(min(topk, 2)):
                    for l in range(min(topk, 2)):
                        choices.append((i, j, k, l))

    return choices


def get_sparse_medusa_choices(num_heads: int) -> List[Tuple[int, ...]]:
    """
    Generate a sparse tree configuration optimized for speed.

    Uses fewer candidates than the default, suitable for:
    - Lower-accuracy Medusa heads
    - Latency-sensitive applications
    - Initial testing

    Args:
        num_heads: Number of Medusa prediction heads

    Returns:
        List of choice tuples (smaller tree)
    """
    choices = []

    # Top-3 from first head
    for i in range(min(3, num_heads and 3)):
        choices.append((i,))

    if num_heads >= 2:
        # Limited two-level paths
        for i in range(2):
            for j in range(2):
                choices.append((i, j))

    if num_heads >= 3:
        # Very limited three-level
        choices.append((0, 0, 0))
        choices.append((0, 0, 1))
        choices.append((0, 1, 0))
        choices.append((1, 0, 0))

    if num_heads >= 4:
        # Single four-level path
        choices.append((0, 0, 0, 0))

    return choices


def validate_medusa_choices(
    medusa_choices: List[Tuple[int, ...]], num_heads: int, topk: int = 10
) -> None:
    """
    Validate that medusa_choices are well-formed.

    Checks:
    1. All choices are non-empty tuples of integers
    2. No choice exceeds num_heads in depth
    3. All indices are within [0, topk)
    4. Parent paths exist for all non-root choices

    Args:
        medusa_choices: List of choice tuples to validate
        num_heads: Number of Medusa heads available
        topk: Maximum top-k value

    Raises:
        ValueError: If choices are invalid
    """
    if not medusa_choices:
        raise ValueError("medusa_choices cannot be empty")

    choice_set = set(medusa_choices)

    for choice in medusa_choices:
        if not isinstance(choice, tuple):
            raise ValueError(f"Each choice must be a tuple, got {type(choice)}")

        if len(choice) == 0:
            raise ValueError("Choice tuples cannot be empty")

        if len(choice) > num_heads:
            raise ValueError(
                f"Choice {choice} has depth {len(choice)} but only {num_heads} heads available"
            )

        for idx in choice:
            if not isinstance(idx, int):
                raise ValueError(f"Choice indices must be integers, got {type(idx)} in {choice}")
            if idx < 0 or idx >= topk:
                raise ValueError(f"Choice index {idx} out of range [0, {topk}) in {choice}")

        # Check that parent path exists (except for depth-1 choices)
        if len(choice) > 1:
            parent = choice[:-1]
            if parent not in choice_set:
                raise ValueError(f"Parent path {parent} not found for choice {choice}")
