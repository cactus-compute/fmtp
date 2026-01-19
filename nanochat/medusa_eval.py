"""
Efficient Medusa speculative decoding evaluation.

This module evaluates Medusa MTP heads using single-pass tree-based speculation
with the same O(N²) complexity as baseline evaluation.

The evaluation simulates speculative decoding:
1. At each position, build a tree of speculative candidates from Medusa heads
2. Use the pre-computed logits to verify which candidates match
3. Accept the longest matching path and advance

Note: This is a "single-pass approximation" - the verification uses the same
logits that were computed for speculation (position P's logit predicts P+1).
True 2-pass verification (where speculated tokens are in context) would require
O(N³/accept_length) complexity which is much slower.
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
    Medusa speculative decoding evaluation with single forward pass.

    Algorithm:
    1. Forward full sequence ONCE with return_medusa=True to get:
       - Main logits at all positions (logits[pos] predicts token at pos+1)
       - Medusa logits at all positions (medusa_logits[k][pos] predicts token at pos+k+2)
    2. For each position P, simulate speculative decoding:
       - Build tree candidates from Medusa top-k predictions
       - Verify using pre-computed logits (single-pass approximation)
       - Accept longest matching path

    Complexity: O(N²) - same as baseline evaluation

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

    # Process each batch item
    for b in range(batch_size):
        input_single = input_ids[b:b+1, :]  # (1, seq_len)

        # === Single forward pass to get main + Medusa logits ===
        logits, medusa_logits = model(input_single, return_medusa=True)
        # logits: (1, T, V) - logits[0, pos] predicts token at pos+1
        # medusa_logits: (num_heads, 1, T, V) - medusa_logits[k, 0, pos] predicts token at pos+k+2

        # Pre-compute top-k for all positions
        main_topk = torch.topk(logits[0], topk, dim=-1).indices  # (T, topk)
        medusa_topk = torch.topk(medusa_logits[:num_medusa_heads, 0], topk, dim=-1).indices  # (H, T, topk)

        # Build stacked logits for verification:
        # For verifying a candidate path [t1, t2, t3, ...] starting at position P:
        # - t1 (pos P+1) is verified by logits[0, P] (main head at P)
        # - t2 (pos P+2) is verified by medusa_logits[0, 0, P] (Medusa head 0 at P)
        # - t3 (pos P+3) is verified by medusa_logits[1, 0, P] (Medusa head 1 at P)
        # Stack them: verify_logits[pos, depth] gives logits for verifying depth-th token
        verify_logits = torch.cat([
            logits.unsqueeze(2),  # (1, T, 1, V) - main head
            medusa_logits[:num_medusa_heads, :, :, :].permute(1, 2, 0, 3),  # (1, T, H, V) - Medusa heads
        ], dim=2)  # (1, T, 1+H, V)
        # Now verify_logits[0, pos, d] gives logits for predicting token at pos+d+1

        pos = 0
        while pos < seq_len - 1:
            # Build candidates from pre-computed Medusa speculations at this position
            flat_candidates = torch.cat([
                main_topk[pos, 0:1],  # Top-1 from main head as root
                medusa_topk[:, pos, :].reshape(-1),  # Top-k from each Medusa head
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

            # === Verify using pre-computed logits ===
            # For each candidate path, check if main/Medusa predictions match
            # verify_logits[0, pos, d] predicts token at position pos+d+1

            # Get verification logits for this position
            pos_verify_logits = verify_logits[0, pos, :effective_depth, :]  # (effective_depth, V)

            # Get argmax predictions from verification logits
            verify_predictions = pos_verify_logits.argmax(dim=-1)  # (effective_depth,)

            # For each candidate, check how many positions match the verification predictions
            # The tree structure means different candidates may share prefixes
            # We need to map from tree indices back to depth indices

            # Actually, we need to be careful here. The candidates are structured as paths through
            # the tree, but verification needs to check each depth position independently.
            # For depth d, verify_predictions[d] is the argmax of verify_logits at depth d.
            # A candidate matches at depth d if candidates[i, d] == verify_predictions[d].

            # But wait - the tree structure means position 0 uses main head, position 1 uses Medusa[0], etc.
            # So we should verify:
            # - candidates[i, 0] against logits[0, pos].argmax() = verify_predictions[0]
            # - candidates[i, 1] against medusa_logits[0, 0, pos].argmax() = verify_predictions[1]
            # - candidates[i, 2] against medusa_logits[1, 0, pos].argmax() = verify_predictions[2]

            # Find matches for each candidate
            matches = (candidates[:, :effective_depth] == verify_predictions[:effective_depth].unsqueeze(0)).int()
            cumulative_matches = torch.cumprod(matches, dim=1)
            accept_lengths = cumulative_matches.sum(dim=1)

            # Select best candidate (longest matching prefix)
            best_candidate_idx = accept_lengths.argmax().item()
            accept_length = accept_lengths[best_candidate_idx].item()

            # Record accepted predictions and losses
            if accept_length > 0:
                for k in range(accept_length):
                    pred_pos = pos + k + 1
                    if pred_pos < seq_len:
                        all_predictions[b, pred_pos] = candidates[best_candidate_idx, k]
                        # Loss uses verification logits at this depth
                        all_losses[b, pred_pos] = F.cross_entropy(
                            pos_verify_logits[k:k+1, :],
                            target_ids[b, pred_pos].unsqueeze(0),
                            reduction='none'
                        )
                pos += accept_length
            else:
                # No speculation accepted - use main model prediction
                pred_pos = pos + 1
                if pred_pos < seq_len:
                    main_pred = verify_predictions[0]
                    all_predictions[b, pred_pos] = main_pred
                    all_losses[b, pred_pos] = F.cross_entropy(
                        pos_verify_logits[0:1, :],
                        target_ids[b, pred_pos].unsqueeze(0),
                        reduction='none'
                    )
                pos += 1

    # Last position has no target
    all_losses[:, -1] = float('nan')
    return all_losses, all_predictions
