"""
Tree attention utilities for HST speculation verification.

This module handles the conversion of HST trees into formats suitable for
single-pass verification via tree attention:

1. Tree flattening: Topological sort into sequence
2. Attention mask generation: 2D mask where token i attends to ancestors
3. Position ID computation: Depth-based positions for RoPE

These utilities integrate with existing Medusa buffers or can be used
standalone for custom tree structures.
"""

import torch
from typing import Optional
from dataclasses import dataclass

from nanochat.hst.tree_builder import HSTNode


@dataclass
class HSTBuffers:
    """
    Buffers for tree attention in HST speculation.

    Similar to Medusa buffers but generated dynamically from HST trees.
    """
    # Attention mask: [1, 1, tree_len, tree_len]
    # 1.0 where token i can attend to token j (j is ancestor of i)
    attn_mask: torch.Tensor

    # Tree tokens: [tree_len] flattened token IDs
    tree_tokens: torch.Tensor

    # Position IDs: [tree_len] depth-based positions for RoPE
    position_ids: torch.Tensor

    # Parent indices: [tree_len] index of parent node (-1 for root)
    parent_indices: torch.Tensor

    # Depth of each node: [tree_len]
    depths: torch.Tensor

    # Mapping from tree index to candidate path index
    # retrieve_indices: [num_paths, max_depth] indices into tree_tokens
    retrieve_indices: torch.Tensor

    @property
    def tree_len(self) -> int:
        return len(self.tree_tokens)


def flatten_tree(
    tree: list[HSTNode],
    sort_by: str = "depth_first",
) -> tuple[list[int], list[int], list[int]]:
    """
    Flatten HST tree into sequence via topological sort.

    Args:
        tree: List of HSTNode objects (tree[0] is root)
        sort_by: Sorting strategy:
            - "depth_first": Sort by depth, then by score within depth
            - "score": Sort by cumulative score (best-first)
            - "insertion": Keep original insertion order

    Returns:
        token_ids: Flattened token IDs
        parent_indices: Index of parent in flattened order (-1 for root)
        depths: Depth of each node
    """
    if not tree:
        return [], [], []

    # Create mapping from original index to new index
    if sort_by == "depth_first":
        # Sort by (depth, -score) to get depth-first with high scores first
        sorted_indices = sorted(
            range(len(tree)),
            key=lambda i: (tree[i].depth, -tree[i].score)
        )
    elif sort_by == "score":
        # Sort by score descending
        sorted_indices = sorted(
            range(len(tree)),
            key=lambda i: -tree[i].score
        )
    else:  # "insertion"
        sorted_indices = list(range(len(tree)))

    # Create old->new index mapping
    old_to_new = {old: new for new, old in enumerate(sorted_indices)}

    # Build flattened arrays
    token_ids = []
    parent_indices = []
    depths = []

    for new_idx, old_idx in enumerate(sorted_indices):
        node = tree[old_idx]
        token_ids.append(node.token_id)
        depths.append(node.depth)

        if node.parent_idx is None:
            parent_indices.append(-1)
        else:
            parent_indices.append(old_to_new[node.parent_idx])

    return token_ids, parent_indices, depths


def build_tree_mask(
    parent_indices: list[int],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build 2D attention mask for tree structure.

    Each node can attend to:
    - Itself
    - All ancestors (following parent pointers to root)

    Args:
        parent_indices: List of parent indices (-1 for root)
        device: Device for output tensor

    Returns:
        attn_mask: [tree_len, tree_len] where mask[i,j]=1 if i can attend to j
    """
    tree_len = len(parent_indices)
    mask = torch.zeros(tree_len, tree_len, device=device)

    for i in range(tree_len):
        # Each node attends to itself
        mask[i, i] = 1.0

        # Follow parent chain to root
        current = parent_indices[i]
        while current >= 0:
            mask[i, current] = 1.0
            current = parent_indices[current]

    return mask


def build_retrieve_indices(
    tree: list[HSTNode],
    parent_indices: list[int],
    old_to_new: dict[int, int],
    max_depth: int,
) -> torch.Tensor:
    """
    Build indices for extracting candidate paths from flattened tree.

    Each row represents a path from root to a node, padded with -1.

    Args:
        tree: Original HSTNode list
        parent_indices: Parent indices in flattened order
        old_to_new: Mapping from original to flattened indices
        max_depth: Maximum path length

    Returns:
        retrieve_indices: [num_nodes, max_depth] indices into flattened tree
    """
    tree_len = len(tree)
    retrieve_indices = torch.full((tree_len, max_depth), -1, dtype=torch.long)

    for old_idx, node in enumerate(tree):
        new_idx = old_to_new[old_idx]

        # Build path from root to this node
        path = []
        current_old = old_idx
        while current_old is not None and current_old >= 0:
            path.append(old_to_new[current_old])
            current_old = tree[current_old].parent_idx

        # Reverse to get root-first order
        path = list(reversed(path))

        # Fill in retrieve indices
        for j, idx in enumerate(path):
            if j < max_depth:
                retrieve_indices[new_idx, j] = idx

    return retrieve_indices


def generate_hst_buffers(
    tree: list[HSTNode],
    device: str = "cuda",
) -> HSTBuffers:
    """
    Generate all buffers needed for tree attention from HST tree.

    Args:
        tree: List of HSTNode objects from HSTTreeBuilder
        device: Device for tensors

    Returns:
        HSTBuffers containing all necessary tensors
    """
    if not tree:
        # Empty tree - return minimal buffers
        return HSTBuffers(
            attn_mask=torch.zeros(1, 1, 1, 1, device=device),
            tree_tokens=torch.zeros(1, dtype=torch.long, device=device),
            position_ids=torch.zeros(1, dtype=torch.long, device=device),
            parent_indices=torch.tensor([-1], dtype=torch.long, device=device),
            depths=torch.zeros(1, dtype=torch.long, device=device),
            retrieve_indices=torch.zeros(1, 1, dtype=torch.long, device=device),
        )

    # Flatten tree with depth-first ordering
    token_ids, parent_indices, depths = flatten_tree(tree, sort_by="depth_first")

    # Create old->new mapping
    sorted_indices = sorted(
        range(len(tree)),
        key=lambda i: (tree[i].depth, -tree[i].score)
    )
    old_to_new = {old: new for new, old in enumerate(sorted_indices)}

    # Build attention mask
    attn_mask = build_tree_mask(parent_indices, device=device)

    # Build retrieve indices
    max_depth = max(depths) + 1 if depths else 1
    retrieve_indices = build_retrieve_indices(tree, parent_indices, old_to_new, max_depth)

    return HSTBuffers(
        attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),  # [1, 1, L, L]
        tree_tokens=torch.tensor(token_ids, dtype=torch.long, device=device),
        position_ids=torch.tensor(depths, dtype=torch.long, device=device),
        parent_indices=torch.tensor(parent_indices, dtype=torch.long, device=device),
        depths=torch.tensor(depths, dtype=torch.long, device=device),
        retrieve_indices=retrieve_indices.to(device),
    )


def extract_candidate_paths(
    buffers: HSTBuffers,
) -> list[list[int]]:
    """
    Extract all unique candidate paths from HST buffers.

    Returns list of token sequences, each representing a speculation path
    from root (exclusive) to a leaf.

    Args:
        buffers: HSTBuffers generated from tree

    Returns:
        List of token ID lists
    """
    paths = []
    tree_tokens = buffers.tree_tokens
    retrieve_indices = buffers.retrieve_indices
    depths = buffers.depths

    for i in range(len(tree_tokens)):
        if depths[i] == 0:
            continue  # Skip root

        # Get path for this node
        path_indices = retrieve_indices[i]
        path = []

        for idx in path_indices:
            if idx < 0:
                break
            # Skip root (depth 0)
            if buffers.depths[idx] > 0:
                path.append(tree_tokens[idx].item())

        if path:
            paths.append(path)

    # Remove duplicate prefixes
    unique_paths = []
    seen = set()
    for path in sorted(paths, key=len, reverse=True):
        path_tuple = tuple(path)
        if path_tuple not in seen:
            unique_paths.append(path)
            seen.add(path_tuple)

    return unique_paths


def merge_with_context(
    context_ids: torch.Tensor,
    buffers: HSTBuffers,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merge context sequence with HST tree for verification forward pass.

    Creates the extended input and attention mask needed for tree verification.

    Args:
        context_ids: [B, T] context token IDs
        buffers: HSTBuffers from HST tree

    Returns:
        extended_ids: [B, T + tree_len] token IDs
        extended_mask: [1, 1, T + tree_len, T + tree_len] attention mask
        position_ids: [T + tree_len] position IDs
    """
    B, T = context_ids.shape
    tree_len = buffers.tree_len
    device = context_ids.device

    # Extend input with tree tokens
    tree_tokens = buffers.tree_tokens.unsqueeze(0).expand(B, -1)
    extended_ids = torch.cat([context_ids, tree_tokens], dim=1)

    # Build extended attention mask
    full_len = T + tree_len
    extended_mask = torch.zeros(1, 1, full_len, full_len, device=device)

    # Context: standard causal attention
    extended_mask[:, :, :T, :T] = torch.tril(torch.ones(T, T, device=device))

    # Tree tokens can attend to full context
    extended_mask[:, :, T:, :T] = 1.0

    # Tree tokens: use tree attention mask
    extended_mask[:, :, T:, T:] = buffers.attn_mask

    # Position IDs: context uses 0..T-1, tree uses T + depth
    context_positions = torch.arange(T, device=device)
    tree_positions = T + buffers.position_ids
    position_ids = torch.cat([context_positions, tree_positions])

    return extended_ids, extended_mask, position_ids


def verify_tree_greedy(
    tree_logits: torch.Tensor,
    buffers: HSTBuffers,
) -> tuple[int, list[int]]:
    """
    Greedy verification of HST tree candidates.

    Finds the longest path where argmax predictions match the tree tokens.

    Args:
        tree_logits: [tree_len, vocab_size] logits at each tree position
        buffers: HSTBuffers

    Returns:
        accept_length: Number of tokens accepted (0 = only base token)
        accepted_tokens: List of accepted token IDs
    """
    tree_tokens = buffers.tree_tokens
    retrieve_indices = buffers.retrieve_indices
    depths = buffers.depths

    # Get predictions at each position
    predictions = tree_logits.argmax(dim=-1)  # [tree_len]

    # Find longest matching path
    best_length = 0
    best_path = []

    for i in range(len(tree_tokens)):
        if depths[i] == 0:
            continue  # Skip root

        # Get path to this node
        path_indices = retrieve_indices[i]

        # Check if entire path matches
        matched = True
        path_tokens = []

        for j, idx in enumerate(path_indices):
            if idx < 0:
                break

            if depths[idx] == 0:
                continue  # Skip root

            token = tree_tokens[idx].item()
            path_tokens.append(token)

            # Check: prediction at parent position should equal this token
            parent_idx = buffers.parent_indices[idx]
            if parent_idx >= 0:
                if predictions[parent_idx] != token:
                    matched = False
                    break

        if matched and len(path_tokens) > best_length:
            best_length = len(path_tokens)
            best_path = path_tokens

    return best_length, best_path


def verify_tree_typical(
    tree_logits: torch.Tensor,
    buffers: HSTBuffers,
    temperature: float = 1.0,
    posterior_threshold: float = 0.09,
    posterior_alpha: float = 0.3,
) -> tuple[int, list[int]]:
    """
    Typical acceptance verification for stochastic sampling.

    Accepts token if: p_original(token) > min(epsilon, delta * exp(-H))
    where H is the entropy of the distribution.

    Args:
        tree_logits: [tree_len, vocab_size] logits
        buffers: HSTBuffers
        temperature: Sampling temperature
        posterior_threshold: Hard acceptance threshold (epsilon)
        posterior_alpha: Entropy-adaptive factor (delta)

    Returns:
        accept_length: Number of accepted tokens
        accepted_tokens: List of accepted token IDs
    """
    import torch.nn.functional as F

    tree_tokens = buffers.tree_tokens
    retrieve_indices = buffers.retrieve_indices
    depths = buffers.depths

    # Compute probabilities
    probs = F.softmax(tree_logits / temperature, dim=-1)

    # Compute entropy at each position
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

    # Adaptive threshold
    threshold = torch.minimum(
        torch.full_like(entropy, posterior_threshold),
        torch.exp(-entropy) * posterior_alpha,
    )

    # Find longest accepted path
    best_length = 0
    best_path = []

    for i in range(len(tree_tokens)):
        if depths[i] == 0:
            continue

        path_indices = retrieve_indices[i]
        accepted = True
        path_tokens = []

        for j, idx in enumerate(path_indices):
            if idx < 0:
                break

            if depths[idx] == 0:
                continue

            token = tree_tokens[idx].item()
            path_tokens.append(token)

            # Check acceptance at parent position
            parent_idx = buffers.parent_indices[idx]
            if parent_idx >= 0:
                token_prob = probs[parent_idx, token]
                if token_prob <= threshold[parent_idx]:
                    accepted = False
                    break

        if accepted and len(path_tokens) > best_length:
            best_length = len(path_tokens)
            best_path = path_tokens

    return best_length, best_path
