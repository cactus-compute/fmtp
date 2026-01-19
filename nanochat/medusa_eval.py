"""
Efficient Medusa speculative decoding evaluation.

This module provides an optimized 2-pass Medusa evaluation that:
1. Does ONE forward pass to get Medusa speculations at ALL positions (amortized)
2. For each speculation point, does a verification forward pass with tree candidates

Complexity: O(1) for speculation + O(seq_len / accept_length) for verification
This is faster than the naive O(seq_len) approach when accept_length > 1.

The 2-pass algorithm at each speculation point:
1. Get Medusa speculations from pre-computed logits (first pass is amortized)
2. Build candidate tree from top-k predictions
3. Forward prefix + tree candidates to verify with main model
4. Find best candidate path (longest matching prefix via greedy acceptance)
5. Record accepted predictions, advance position
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from nanochat.medusa_buffers import generate_medusa_buffers, get_sparse_medusa_choices


@torch.no_grad()
def forward_model_medusa(model, input_ids: torch.Tensor,
                         num_heads_to_use: Optional[int] = None,
                         topk: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Efficient Medusa speculative decoding evaluation with amortized speculation.

    This is the main entry point for Medusa evaluation. It implements true 2-pass
    verification with an optimization: the first pass (speculation) is done once
    for the entire sequence, then verification passes are done incrementally.

    Algorithm:
    1. Forward full sequence once to get Medusa speculations at ALL positions
    2. At each position, build candidate tree from pre-computed top-k
    3. Forward prefix + tree candidates for verification (2nd pass)
    4. Greedy acceptance: find longest prefix where verification matches speculation
    5. Advance by accept_length, repeat until end of sequence

    This achieves O(1) speculation + O(seq_len / avg_accept_length) verification,
    which is faster than naive O(seq_len) when Medusa heads are accurate.

    Args:
        model: GPT model with Medusa heads
        input_ids: Input token IDs (B, T) - batch processing supported
        num_heads_to_use: Number of Medusa heads to use (default: all)
        topk: Number of top predictions per head for tree construction

    Returns:
        losses: (B, T) cross-entropy losses at each position (NaN for last pos)
        predictions: (B, T) predicted tokens at each position
    """
    batch_size, seq_len = input_ids.size()
    device = input_ids.device

    config = model.config
    total_medusa_heads = config.medusa_num_heads
    num_medusa_heads = num_heads_to_use if num_heads_to_use is not None else total_medusa_heads
    num_medusa_heads = min(num_medusa_heads, total_medusa_heads)

    # Build tree structure (shared across all positions)
    medusa_choices = get_sparse_medusa_choices(num_medusa_heads)
    buffers = generate_medusa_buffers(medusa_choices, device=device, topk=topk)
    tree_indices = buffers["tree_indices"]
    retrieve_indices = buffers["retrieve_indices"]
    num_candidates, max_depth = retrieve_indices.shape

    # Output tensors
    all_predictions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    all_losses = torch.full((batch_size, seq_len), float('nan'), device=device)

    # Ground truth targets (for loss computation)
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)

    # === First pass (amortized): Get Medusa speculations at ALL positions ===
    logits, medusa_logits = model(input_ids, return_medusa=True)
    # logits: (B, T, V), medusa_logits: (num_heads, B, T, V)

    # Pre-compute top-k for all positions (avoids repeated topk calls)
    main_topk = torch.topk(logits, topk, dim=-1).indices  # (B, T, topk)
    medusa_topk = torch.topk(medusa_logits[:num_medusa_heads], topk, dim=-1).indices  # (H, B, T, topk)

    # Process each batch item
    for b in range(batch_size):
        input_single = input_ids[b:b+1, :]  # (1, seq_len)

        pos = 0
        while pos < seq_len - 1:
            # Build candidates from pre-computed Medusa speculations at this position
            flat_candidates = torch.cat([
                main_topk[b, pos, 0:1],  # Top-1 from main head as root
                medusa_topk[:, b, pos, :].reshape(-1),  # Top-k from each Medusa head
            ])
            tree_candidates = flat_candidates[tree_indices]  # (tree_len,)

            # Extract all candidate paths using retrieve_indices
            candidates = torch.zeros(num_candidates, max_depth, dtype=torch.long, device=device)
            for i in range(num_candidates):
                for j in range(max_depth):
                    idx = retrieve_indices[i, j].item()
                    if idx >= 0:
                        candidates[i, j] = tree_candidates[idx]

            # How many tokens can we predict ahead?
            tokens_remaining = seq_len - pos - 1
            effective_depth = min(max_depth, tokens_remaining)

            if effective_depth == 0:
                break

            # === Second pass: Verify with tree candidates ===
            # This is the key 2-pass step: we forward prefix + speculated tree
            prefix = input_single[:, :pos + 1]
            verify_input = torch.cat([prefix, tree_candidates.unsqueeze(0)], dim=1)
            verify_logits = model(verify_input)  # (1, pos+1+tree_len, vocab)

            # Extract candidate-aligned verification logits
            # verify_logits[0, pos + idx] predicts token at position pos + idx + 1
            candidate_logits = torch.zeros(num_candidates, max_depth, verify_logits.shape[-1], device=device)
            for i in range(num_candidates):
                for j in range(max_depth):
                    idx = retrieve_indices[i, j].item()
                    if idx >= 0:
                        candidate_logits[i, j] = verify_logits[0, pos + idx, :]

            # Greedy acceptance: find candidate with longest matching prefix
            predictions_from_logits = candidate_logits.argmax(dim=-1)  # (num_candidates, max_depth)
            matches = (candidates[:, :effective_depth] == predictions_from_logits[:, :effective_depth]).int()
            cumulative_matches = torch.cumprod(matches, dim=1)
            accept_lengths = cumulative_matches.sum(dim=1)

            # Select best candidate
            best_candidate_idx = accept_lengths.argmax().item()
            accept_length = accept_lengths[best_candidate_idx].item()

            # Record accepted predictions and losses
            if accept_length > 0:
                for k in range(accept_length):
                    pred_pos = pos + k + 1
                    if pred_pos < seq_len:
                        all_predictions[b, pred_pos] = candidates[best_candidate_idx, k]
                        idx = retrieve_indices[best_candidate_idx, k].item()
                        if idx >= 0:
                            all_losses[b, pred_pos] = F.cross_entropy(
                                verify_logits[0, pos + idx, :].unsqueeze(0),
                                target_ids[b, pred_pos].unsqueeze(0),
                                reduction='none'
                            )
                pos += accept_length
            else:
                # No speculation accepted - use main model prediction
                pred_pos = pos + 1
                if pred_pos < seq_len:
                    main_pred = verify_logits[0, pos, :].argmax(dim=-1)
                    all_predictions[b, pred_pos] = main_pred
                    all_losses[b, pred_pos] = F.cross_entropy(
                        verify_logits[0, pos, :].unsqueeze(0),
                        target_ids[b, pred_pos].unsqueeze(0),
                        reduction='none'
                    )
                pos += 1

    # Last position has no target
    all_losses[:, -1] = float('nan')
    return all_losses, all_predictions
