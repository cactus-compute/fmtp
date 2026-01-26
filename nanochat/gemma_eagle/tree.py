"""
Tree utilities for EAGLE speculative decoding.

Handles tree structure generation, attention masks, and candidate evaluation.
"""

import copy
from typing import List, Tuple, Dict, Optional
import torch


# Default top-k for tree construction
TOPK = 10


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Args:
        path: The original list that needs padding
        length: The desired length of the padded list
        pad_value: The value to use for padding (default: -2)

    Returns:
        A new list based on the original path but padded to the desired length

    Example:
        >>> pad_path([1, 2, 3], 5)
        [1, 2, 3, -2, -2]
    """
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(
    tree_choices: List[List[int]],
    device: torch.device = torch.device("cuda"),
    topk: int = TOPK,
) -> Dict[str, torch.Tensor]:
    """
    Generate buffers for tree attention in speculative decoding.

    Args:
        tree_choices: List of paths defining the tree structure.
            Each path is a list of indices representing token selections.
            E.g., [[0], [1], [0,0], [0,1]] represents:
                - Root's top-1 child
                - Root's top-2 child
                - Root's top-1's top-1 grandchild
                - Root's top-1's top-2 grandchild
        device: Device to place tensors on
        topk: Number of top predictions per node

    Returns:
        Dictionary containing:
        - tree_attn_mask: (1, 1, tree_len, tree_len) Tree attention mask
        - tree_indices: (tree_len,) Maps positions to candidate token indices
        - tree_position_ids: (tree_len,) Position offsets for each tree node
        - retrieve_indices: (num_paths, max_depth) Maps paths to node indices
    """
    sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
    tree_len = len(sorted_tree_choices) + 1  # +1 for root

    # Count nodes at each depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    # Build attention mask - each node attends to itself and ancestors
    tree_attn_mask = torch.eye(tree_len, tree_len, device=device)
    tree_attn_mask[:, 0] = 1  # All nodes attend to root

    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            if len(cur_tree_choice) == 1:
                continue
            # Find ancestor positions
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
            tree_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Build tree indices - map tree positions to candidate token indices
    tree_indices = torch.zeros(tree_len, dtype=torch.long, device=device)
    p_indices = [0 for _ in range(tree_len - 1)]
    b_indices = [[] for _ in range(tree_len - 1)]
    tree_indices[0] = 0  # Root maps to position 0

    start = 0
    bias = 0
    for i in range(len(depth_counts)):
        inlayer_bias = 0
        b = []
        parent = None
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            cur_parent = cur_tree_choice[:-1]
            if j != 0:
                if cur_parent != parent:
                    bias += 1
                    inlayer_bias += 1
                    parent = cur_parent
                    b = []
            else:
                parent = cur_parent
            tree_indices[start + j + 1] = cur_tree_choice[-1] + topk * (i + bias) + 1
            p_indices[start + j] = inlayer_bias
            if len(b) > 0:
                b_indices[start + j] = copy.deepcopy(b)
            else:
                b_indices[start + j] = []
            b.append(cur_tree_choice[-1] + topk * (i + bias) + 1)
        start += depth_counts[i]

    p_indices = [-1] + p_indices

    # Build position IDs - depth of each node
    tree_position_ids = torch.zeros(tree_len, dtype=torch.long, device=device)
    start = 0
    for i in range(len(depth_counts)):
        tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Build retrieve indices - for extracting candidate paths
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_tree_choices)):
        cur_tree_choice = sorted_tree_choices[-i - 1]
        retrieve_indice = []
        if cur_tree_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_tree_choice)):
                retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                retrieve_paths.append(cur_tree_choice[:c + 1])
        retrieve_indices_nest.append(retrieve_indice)

    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long, device=device)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat(
        [torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long, device=device), retrieve_indices],
        dim=1
    )

    maxitem = retrieve_indices.max().item() + 5

    def custom_sort(lst):
        return [x if x >= 0 else maxitem for x in lst.tolist()]

    retrieve_indices = retrieve_indices.tolist()
    retrieve_indices = sorted(retrieve_indices, key=custom_sort)
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long, device=device)

    return {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }


def generate_candidates(
    tree_logits: torch.Tensor,
    tree_indices: torch.Tensor,
    retrieve_indices: torch.Tensor,
    sample_token: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate candidate token sequences from tree logits.

    Args:
        tree_logits: Logits from draft model for each tree position
        tree_indices: Mapping from tree positions to token indices
        retrieve_indices: Indices for extracting candidate paths
        sample_token: The initially sampled token

    Returns:
        cart_candidates: Candidate sequences for each path (num_paths, max_depth)
        tree_candidates: All candidates in tree order (1, tree_len)
    """
    sample_token = sample_token.to(tree_indices.device)
    candidates_logit = sample_token[0]
    candidates_tree_logits = tree_logits
    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]
    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1,), dtype=torch.long, device=tree_candidates.device) - 1],
        dim=0
    )
    cart_candidates = tree_candidates_ext[retrieve_indices]
    tree_candidates = tree_candidates.unsqueeze(0)

    return cart_candidates, tree_candidates


def evaluate_posterior_greedy(
    logits: torch.Tensor,
    candidates: torch.Tensor,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """
    Evaluate candidates using greedy decoding (temperature=0).

    Args:
        logits: Base model logits for each candidate (num_paths, max_depth, vocab)
        candidates: Candidate token sequences (num_paths, max_depth)

    Returns:
        best_candidate: Index of best candidate path
        accept_length: Number of tokens accepted
        next_token_logits: Logits for sampling next token
    """
    # Find matches between candidates and argmax of logits
    posterior_mask = (
        candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
    ).int()
    candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
    accept_length = candidates_accept_length.max()

    if accept_length == 0:
        best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
    else:
        best_candidate = torch.argmax(candidates_accept_length).to(torch.long)

    return best_candidate, accept_length.item(), logits[best_candidate, accept_length]


def evaluate_posterior_sampling(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """
    Evaluate candidates using sampling with rejection.

    Uses speculative sampling: accept draft token if it matches what
    the target model would have sampled.

    Args:
        logits: Base model logits for each candidate (num_paths, max_depth, vocab)
        candidates: Candidate token sequences (num_paths, max_depth)
        temperature: Sampling temperature
        top_p: Nucleus sampling probability

    Returns:
        best_candidate: Index of best candidate path
        accept_length: Number of tokens accepted
        sample_p: Probability distribution for next token
    """
    import random

    accept_length = 1
    accept_cand = candidates[0][:1]
    best_candidate = 0

    for i in range(1, candidates.shape[1]):
        if i != accept_length:
            break

        adjustflag = False
        is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
        fi = torch.nonzero(is_eq, as_tuple=True)[0][0]

        gt_logits = logits[fi, i - 1][None] / temperature
        gtp = torch.softmax(gt_logits, dim=-1)[0]

        candidates_set = []
        for j in range(candidates.shape[0]):
            if is_eq[j]:
                x = candidates[j, i]
                xi = x.item()
                if xi in candidates_set or xi == -1:
                    continue
                candidates_set.append(xi)

                r = random.random()
                px = gtp[xi].item()
                qx = 1.0  # Draft proposal probability (simplified)
                acp = min(1.0, px / qx)

                if r <= acp:
                    accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                    accept_length += 1
                    best_candidate = j
                    break
                else:
                    gtp[xi] = 0
                    gtp = gtp / gtp.sum()
                    adjustflag = True

    if adjustflag and accept_length != candidates.shape[1]:
        sample_p = gtp
    else:
        gt_logits = logits[best_candidate, accept_length - 1][None] / temperature
        sample_p = torch.softmax(gt_logits, dim=-1)[0]

    return torch.tensor(best_candidate), accept_length - 1, sample_p


def evaluate_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """
    Evaluate candidates and find best accepted sequence.

    Args:
        logits: Base model logits for each candidate
        candidates: Candidate token sequences
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling probability

    Returns:
        best_candidate: Index of best candidate path
        accept_length: Number of tokens accepted
        next_token_dist: Distribution for next token
    """
    if temperature < 1e-5:
        return evaluate_posterior_greedy(logits, candidates)
    else:
        return evaluate_posterior_sampling(logits, candidates, temperature, top_p)


# Predefined tree structures for different configurations
# Based on empirical speedup measurements

MC_SIM_7B_63 = [
    [0], [1], [2], [3],
    [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0],
    [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 2, 0],
    [0, 2, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [2, 0, 0],
    [2, 0, 1], [2, 1, 0], [3, 0, 0],
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 0], [0, 0, 2, 0],
    [0, 1, 0, 0], [0, 1, 1, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 0, 1, 0],
    [1, 1, 0, 0], [2, 0, 0, 0],
    [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 2, 0],
    [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0],
]

MC_SIM_1B_32 = [
    [0], [1], [2],
    [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 0],
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 2, 0], [1, 0, 0], [1, 1, 0],
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0],
    [0, 0, 0, 0, 0], [0, 0, 0, 0, 1],
]


def get_default_tree_choices(total_tokens: int = 63) -> List[List[int]]:
    """
    Get default tree structure based on token budget.

    Args:
        total_tokens: Maximum number of draft tokens

    Returns:
        List of tree paths
    """
    if total_tokens >= 63:
        return MC_SIM_7B_63
    elif total_tokens >= 32:
        return MC_SIM_1B_32
    else:
        # Simple linear tree for small budgets
        return [[i] for i in range(min(total_tokens, 10))]
