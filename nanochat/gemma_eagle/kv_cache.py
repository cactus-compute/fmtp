"""
Pre-allocated KV cache for EAGLE speculative decoding.

Provides efficient memory management by pre-allocating buffers
and avoiding dynamic memory allocation during generation.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn


class EagleKVCache:
    """
    Pre-allocated KV cache for efficient generation.

    Pre-allocates key and value tensors for a maximum sequence length,
    then uses views to efficiently update and read from the cache.
    """

    def __init__(
        self,
        batch_size: int,
        max_length: int,
        num_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize pre-allocated KV cache.

        Args:
            batch_size: Maximum batch size
            max_length: Maximum sequence length
            num_layers: Number of transformer layers
            num_key_value_heads: Number of KV heads (for GQA)
            head_dim: Dimension per attention head
            device: Device to allocate on
            dtype: Data type for tensors
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Pre-allocate KV buffers for each layer
        # Shape: (batch_size, num_kv_heads, max_length, head_dim)
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        for _ in range(num_layers):
            self.key_cache.append(
                torch.zeros(
                    batch_size, num_key_value_heads, max_length, head_dim,
                    device=device, dtype=dtype
                )
            )
            self.value_cache.append(
                torch.zeros(
                    batch_size, num_key_value_heads, max_length, head_dim,
                    device=device, dtype=dtype
                )
            )

        # Current sequence length for each item in batch
        self.current_length = torch.zeros(batch_size, dtype=torch.long, device=device)

    def reset(self):
        """Reset cache for new generation."""
        self.current_length.fill_(0)

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value states.

        Args:
            layer_idx: Which layer to update
            key_states: New key states (batch, num_kv_heads, seq_len, head_dim)
            value_states: New value states (batch, num_kv_heads, seq_len, head_dim)
            cache_position: Explicit positions to write (optional)

        Returns:
            Updated key and value states including cached values
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape

        if cache_position is not None:
            # Explicit position update (for tree attention)
            for i, pos in enumerate(cache_position):
                self.key_cache[layer_idx][:, :, pos:pos+1] = key_states[:, :, i:i+1]
                self.value_cache[layer_idx][:, :, pos:pos+1] = value_states[:, :, i:i+1]
            max_pos = cache_position.max().item() + 1
        else:
            # Sequential update
            start_pos = self.current_length[0].item()
            end_pos = start_pos + seq_len
            self.key_cache[layer_idx][:, :, start_pos:end_pos] = key_states
            self.value_cache[layer_idx][:, :, start_pos:end_pos] = value_states
            max_pos = end_pos

        # Return full cached values up to current position
        return (
            self.key_cache[layer_idx][:, :, :max_pos],
            self.value_cache[layer_idx][:, :, :max_pos]
        )

    def get(self, layer_idx: int, length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached key/value states for a layer.

        Args:
            layer_idx: Which layer to get
            length: Optional length to retrieve (default: current length)

        Returns:
            Cached key and value states
        """
        if length is None:
            length = self.current_length[0].item()
        return (
            self.key_cache[layer_idx][:, :, :length],
            self.value_cache[layer_idx][:, :, :length]
        )

    def set_length(self, length: int):
        """Set current sequence length."""
        self.current_length.fill_(length)

    def copy_from_indices(
        self,
        layer_idx: int,
        source_indices: torch.Tensor,
        target_start: int,
    ):
        """
        Copy cache entries from specific indices to contiguous positions.

        Used for tree verification: after accepting tokens, we need to
        consolidate the accepted path into the cache.

        Args:
            layer_idx: Which layer to update
            source_indices: Indices of entries to copy
            target_start: Where to start writing in the target
        """
        source_keys = self.key_cache[layer_idx][:, :, source_indices]
        source_values = self.value_cache[layer_idx][:, :, source_indices]

        num_entries = len(source_indices)
        target_end = target_start + num_entries

        self.key_cache[layer_idx][:, :, target_start:target_end] = source_keys
        self.value_cache[layer_idx][:, :, target_start:target_end] = source_values


class StaticKVCache:
    """
    Simpler KV cache that stores tensors directly.

    Used for compatibility with HuggingFace's cache interface.
    """

    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new states.

        Args:
            key_states: New key states
            value_states: New value states
            layer_idx: Layer index
            cache_kwargs: Additional arguments (ignored)

        Returns:
            Concatenated key and value states
        """
        # Ensure we have storage for this layer
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get current sequence length."""
        if len(self.key_cache) == 0 or self.key_cache[0] is None:
            return 0
        return self.key_cache[0].shape[2]

    def get_max_length(self) -> Optional[int]:
        """Return None - no max length for dynamic cache."""
        return None

    def reset(self):
        """Clear the cache."""
        self.key_cache = []
        self.value_cache = []


class DualKVCache:
    """
    Dual KV cache for speculative decoding.

    Maintains separate caches for:
    1. Base model - stores verified tokens
    2. Draft model - stores speculative tokens (may be discarded)

    After verification, accepted tokens are copied from draft to base cache.
    """

    def __init__(
        self,
        batch_size: int,
        max_length: int,
        base_num_layers: int,
        draft_num_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize dual KV cache.

        Args:
            batch_size: Maximum batch size
            max_length: Maximum sequence length
            base_num_layers: Number of base model layers
            draft_num_layers: Number of draft model layers
            num_key_value_heads: Number of KV heads
            head_dim: Head dimension
            device: Device
            dtype: Data type
        """
        self.base_cache = EagleKVCache(
            batch_size, max_length, base_num_layers,
            num_key_value_heads, head_dim, device, dtype
        )
        self.draft_cache = EagleKVCache(
            batch_size, max_length, draft_num_layers,
            num_key_value_heads, head_dim, device, dtype
        )

        self.verified_length = 0  # Length of verified tokens

    def reset(self):
        """Reset both caches."""
        self.base_cache.reset()
        self.draft_cache.reset()
        self.verified_length = 0

    def accept_tokens(self, num_accepted: int):
        """
        Accept tokens and update verified length.

        Args:
            num_accepted: Number of tokens to accept
        """
        self.verified_length += num_accepted
        self.base_cache.set_length(self.verified_length)

    def rollback_draft(self):
        """Rollback draft cache to verified length."""
        self.draft_cache.set_length(self.verified_length)
