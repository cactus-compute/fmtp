"""
Efficient Medusa speculative decoding evaluation with KV caching.

This module provides O(N²) complexity evaluation (same as baseline) by:
1. First pass: Forward full sequence with KV cache to get Medusa speculations
2. Verification passes: Reuse KV cache, only forward tree candidates (O(k) tokens)

The key insight: the first pass populates KV cache for the full sequence.
For verification at position P, we can truncate the KV cache to length P+1
and only forward the tree candidates, making each verification O(k*P) instead of O(P²).

Total complexity: O(N²) for first pass + O(k * N² / accept_length) for verification
                = O(N²) when k << N
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from nanochat.medusa_buffers import generate_medusa_buffers, get_sparse_medusa_choices
from nanochat.engine import KVCache


@torch.no_grad()
def forward_model_medusa(model, input_ids: torch.Tensor,
                         num_heads_to_use: Optional[int] = None,
                         topk: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Efficient Medusa speculative decoding evaluation with KV cache reuse.

    Algorithm:
    1. Forward full sequence WITH KV cache to get Medusa speculations at ALL positions
       - This populates KV cache with keys/values for all N positions
       - Cost: O(N²)
    2. For each speculation point P:
       - Truncate KV cache to position P+1 (just update the length counter)
       - Forward only tree candidates (k tokens) using the cached prefix
       - Cost: O(k * P) per verification
    3. Total: O(N²) + O(k * sum(P_i)) where P_i are verification positions
            ≈ O(N²) when k is small

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
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

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
    tree_len = tree_indices.shape[0]

    # Output tensors
    all_predictions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    all_losses = torch.full((batch_size, seq_len), float('nan'), device=device)

    # Ground truth targets (for loss computation)
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)

    # KV cache config
    kv_kwargs = {
        "num_heads": config.n_kv_head,
        "head_dim": config.n_embd // config.n_head,
        "num_layers": config.n_layer,
    }

    # Process each batch item
    for b in range(batch_size):
        input_single = input_ids[b:b+1, :]  # (1, seq_len)

        # === First pass: Forward full sequence WITH KV cache ===
        # This populates the KV cache with all N positions
        kv_cache = KVCache(
            batch_size=1,
            seq_len=seq_len + tree_len + 10,  # Extra space for tree candidates
            device=device,
            dtype=dtype,
            **kv_kwargs
        )

        # Forward with KV cache - this fills the cache
        # Note: return_medusa doesn't work with KV cache, so we do it separately
        logits_with_cache = model(input_single, kv_cache=kv_cache)
        # KV cache now has all N positions cached

        # Get Medusa speculations (without KV cache since return_medusa needs it)
        logits, medusa_logits = model(input_single, return_medusa=True)
        # logits: (1, T, V), medusa_logits: (num_heads, 1, T, V)

        # Pre-compute top-k for all positions
        main_topk = torch.topk(logits[0], topk, dim=-1).indices  # (T, topk)
        medusa_topk = torch.topk(medusa_logits[:num_medusa_heads, 0], topk, dim=-1).indices  # (H, T, topk)

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

            # === Second pass: Verify using KV cache ===
            # Truncate KV cache to position pos+1 (rollback to prefix length)
            kv_cache.cache_seqlens.fill_(pos + 1)

            # Forward only the tree candidates (k tokens) - O(k * (pos+1))
            tree_input = tree_candidates.unsqueeze(0)  # (1, tree_len)
            verify_logits = model(tree_input, kv_cache=kv_cache)  # (1, tree_len, vocab)

            # verify_logits[0, i] is the logit after seeing prefix + tree_candidates[:i+1]
            # This predicts token at position (pos+1) + i + 1 = pos + i + 2
            #
            # But we need:
            # - Logit predicting pos+1: this is logits[0, pos] from first pass
            # - Logit predicting pos+2: verify_logits[0, 0] (after seeing prefix + tree[0])
            # - Logit predicting pos+3: verify_logits[0, 1] (after seeing prefix + tree[0:2])
            # etc.

            # Combine: first position uses logits from first pass, rest from verification
            full_verify_logits = torch.cat([
                logits[:, pos:pos+1, :],  # Logit at pos predicts pos+1
                verify_logits,  # Logits for pos+2, pos+3, ...
            ], dim=1)  # (1, 1 + tree_len, vocab)

            # Extract candidate-aligned verification logits
            candidate_logits = torch.zeros(num_candidates, max_depth, full_verify_logits.shape[-1], device=device)
            for i in range(num_candidates):
                for j in range(max_depth):
                    idx = retrieve_indices[i, j].item()
                    if idx >= 0:
                        # idx maps to position in the tree
                        # full_verify_logits[0, idx] gives the logit that predicts candidate[i, j]
                        candidate_logits[i, j] = full_verify_logits[0, idx, :]

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
                                full_verify_logits[0, idx, :].unsqueeze(0),
                                target_ids[b, pred_pos].unsqueeze(0),
                                reduction='none'
                            )
                pos += accept_length
            else:
                # No speculation accepted - use main model prediction
                pred_pos = pos + 1
                if pred_pos < seq_len:
                    main_pred = full_verify_logits[0, 0, :].argmax(dim=-1)
                    all_predictions[b, pred_pos] = main_pred
                    all_losses[b, pred_pos] = F.cross_entropy(
                        full_verify_logits[0, 0, :].unsqueeze(0),
                        target_ids[b, pred_pos].unsqueeze(0),
                        reduction='none'
                    )
                pos += 1

    # Last position has no target
    all_losses[:, -1] = float('nan')
    return all_losses, all_predictions
