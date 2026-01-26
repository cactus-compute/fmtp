"""
Shared speculative decoding helpers for Gemma Medusa and EAGLE.

Includes tree attention mask construction and KV cache compaction after
accepting a speculative path.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple
import torch


def build_tree_attention_mask(
    tree_attn_mask: torch.Tensor,
    base_seq_len: int,
    hf_tree_mask: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build HF-style attention mask for tree verification with KV cache.

    Args:
        tree_attn_mask: (1, 1, tree_len, tree_len) with 1 = attend, 0 = block
        base_seq_len: Length of cached prefix (tree tokens attend to all cache tokens)
        hf_tree_mask: Optional precomputed HF-format tree mask (0 or -inf)

    Returns:
        hf_attn_mask: (1, 1, tree_len, base_seq_len + tree_len)
        hf_tree_mask: Cached HF-format tree mask (same shape as tree_attn_mask)
    """
    if dtype is None:
        dtype = tree_attn_mask.dtype if tree_attn_mask.dtype != torch.bool else torch.float32

    tree_attn_mask = tree_attn_mask.to(dtype)

    if hf_tree_mask is None or hf_tree_mask.dtype != dtype:
        hf_tree_mask = torch.where(
            tree_attn_mask > 0.5,
            torch.zeros_like(tree_attn_mask, dtype=dtype),
            torch.full_like(tree_attn_mask, torch.finfo(dtype).min, dtype=dtype),
        )

    tree_len = tree_attn_mask.shape[-1]
    hf_attn_mask = torch.zeros(
        1, 1, tree_len, base_seq_len + tree_len,
        device=tree_attn_mask.device, dtype=dtype,
    )
    hf_attn_mask[:, :, :, base_seq_len:] = hf_tree_mask
    return hf_attn_mask, hf_tree_mask


def update_kv_cache_from_tree(
    past_key_values,
    accepted_tree_positions: Iterable[int],
    base_seq_len: int,
    tree_len: int,
):
    """
    Compact KV cache after tree verification to keep only the accepted path.

    Args:
        past_key_values: HF DynamicCache or legacy tuple of (k, v)
        accepted_tree_positions: Tree indices to keep (including root)
        base_seq_len: Length of cached prefix before tree tokens
        tree_len: Number of tree tokens appended during verification

    Returns:
        updated_past_key_values, need_fallback
    """
    accepted_tree_positions = [int(p) for p in accepted_tree_positions if int(p) >= 0]
    num_accepted = len(accepted_tree_positions)
    need_fallback = False

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if not hasattr(layer, "keys") or layer.keys is None:
                continue

            if getattr(layer, "sliding_window", None) is not None:
                sw = layer.sliding_window
                cache_len_before_tree = min(base_seq_len, sw)
                expected_tree_end = cache_len_before_tree + tree_len
                cache_size = layer.keys.shape[2]

                if expected_tree_end <= cache_size:
                    if num_accepted > 0:
                        for i, tree_pos in enumerate(accepted_tree_positions):
                            src_idx = cache_len_before_tree + tree_pos
                            dst_idx = cache_len_before_tree + i
                            if src_idx != dst_idx and src_idx < cache_size:
                                layer.keys[:, :, dst_idx, :] = layer.keys[:, :, src_idx, :]
                                layer.values[:, :, dst_idx, :] = layer.values[:, :, src_idx, :]
                    final_len = cache_len_before_tree + num_accepted
                    layer.keys = layer.keys[:, :, :final_len, :]
                    layer.values = layer.values[:, :, :final_len, :]
                else:
                    layer.keys = layer.keys[:, :, :cache_len_before_tree, :]
                    layer.values = layer.values[:, :, :cache_len_before_tree, :]
                    need_fallback = True

                layer.cumulative_length = base_seq_len + num_accepted
            else:
                if num_accepted > 0:
                    for i, tree_pos in enumerate(accepted_tree_positions):
                        src_idx = base_seq_len + tree_pos
                        dst_idx = base_seq_len + i
                        if src_idx != dst_idx:
                            layer.keys[:, :, dst_idx, :] = layer.keys[:, :, src_idx, :]
                            layer.values[:, :, dst_idx, :] = layer.values[:, :, src_idx, :]
                final_len = base_seq_len + num_accepted
                layer.keys = layer.keys[:, :, :final_len, :]
                layer.values = layer.values[:, :, :final_len, :]

        return past_key_values, need_fallback

    # Legacy tuple cache (k, v) per layer
    new_past = []
    for layer in past_key_values:
        keys, values = layer
        if num_accepted > 0:
            idx = torch.tensor(accepted_tree_positions, device=keys.device, dtype=torch.long)
            gathered_keys = keys.index_select(2, base_seq_len + idx)
            gathered_values = values.index_select(2, base_seq_len + idx)
            new_keys = torch.cat([keys[:, :, :base_seq_len, :], gathered_keys], dim=2)
            new_values = torch.cat([values[:, :, :base_seq_len, :], gathered_values], dim=2)
        else:
            new_keys = keys[:, :, :base_seq_len, :]
            new_values = values[:, :, :base_seq_len, :]
        new_past.append((new_keys, new_values))

    return tuple(new_past), need_fallback
