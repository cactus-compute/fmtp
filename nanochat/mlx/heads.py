"""
MLX implementations of Medusa heads for multi-token prediction.

Mirrors the PyTorch implementations in nanochat/gemma_medusa/heads.py:
- MedusaResBlock: Residual block with SiLU activation
- MedusaHead: Full head with ResBlocks + optional LoRA
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List


class MedusaResBlock(nn.Module):
    """
    Residual block for Medusa heads with SiLU activation.

    Architecture: x + SiLU(linear(x))

    Optional zero-init for identity initialization at start of training.
    Uses bfloat16 weights to match Gemma model for optimal performance.
    """

    def __init__(self, hidden_size: int, zero_init: bool = False):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        # Convert to bfloat16 to match model weights - critical for performance!
        # Mixed dtype (float32 weights + bfloat16 input) causes 5x slowdown in lm_head
        self.linear.weight = self.linear.weight.astype(mx.bfloat16)

        if zero_init:
            # Zero-init so block starts as identity
            self.linear.weight = mx.zeros((hidden_size, hidden_size), dtype=mx.bfloat16)

    def __call__(self, x: mx.array) -> mx.array:
        return x + nn.silu(self.linear(x))


class MedusaHead(nn.Module):
    """
    Single Medusa prediction head.

    Architecture when lora_rank == 0 (no LoRA):
        ResBlocks(x) -> hidden_states
        Caller applies lm_head to get logits

    Architecture when lora_rank > 0:
        ResBlocks(x) -> lora_A -> lora_B -> vocab_delta
        Final logits = lm_head(x) + scaling * vocab_delta

    Args:
        hidden_size: Model hidden dimension
        vocab_size: Vocabulary size (only used if lora_rank > 0)
        num_layers: Number of ResBlock layers
        lora_rank: LoRA rank (0 = no LoRA)
        lora_alpha: LoRA alpha scaling (defaults to lora_rank)
        zero_init_mlp: Whether to zero-init ResBlock weights
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int,
        lora_rank: int = 0,
        lora_alpha: Optional[int] = None,
        zero_init_mlp: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank

        # Compute scaling factor
        self.scaling = self.lora_alpha / self.lora_rank if self.lora_rank > 0 else 0.0

        # Create ResBlock layers
        self.blocks = [
            MedusaResBlock(hidden_size, zero_init=zero_init_mlp)
            for _ in range(num_layers)
        ]

        # Create LoRA projections if lora_rank > 0
        if self.lora_rank > 0:
            self.lora_A = nn.Linear(hidden_size, lora_rank, bias=False)
            self.lora_B = nn.Linear(lora_rank, vocab_size, bias=False)
            # Convert to bfloat16 to match model weights
            self.lora_A.weight = self.lora_A.weight.astype(mx.bfloat16)
            # Zero-init lora_B for stable training (standard LoRA init)
            self.lora_B.weight = mx.zeros((vocab_size, lora_rank), dtype=mx.bfloat16)
        else:
            self.lora_A = None
            self.lora_B = None

    @property
    def has_lora(self) -> bool:
        """Returns True if this head has LoRA projections."""
        return self.lora_rank > 0

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply the head to hidden states.

        Args:
            x: Hidden states (B, T, hidden_size)

        Returns:
            If lora_rank > 0: LoRA delta to add to lm_head(x), shape (B, T, vocab_size)
            If lora_rank == 0: Transformed hidden states for lm_head, shape (B, T, hidden_size)
        """
        # Apply ResBlocks
        for block in self.blocks:
            x = block(x)

        if self.has_lora:
            # LoRA mode: return delta to add to lm_head output
            return self.scaling * self.lora_B(self.lora_A(x))
        else:
            # No LoRA: return transformed hidden states (caller applies lm_head)
            return x


def load_medusa_heads_from_torch(
    state_dict: dict,
    hidden_size: int,
    vocab_size: int,
    num_heads: int,
    num_layers: int,
    lora_rank: int = 0,
    lora_alpha: Optional[int] = None,
    zero_init_mlp: bool = False,
) -> List[MedusaHead]:
    """
    Load Medusa heads from a PyTorch checkpoint state dict (torch tensors).

    Args:
        state_dict: The 'medusa_heads' dict from torch checkpoint
        hidden_size: Model hidden dimension
        vocab_size: Vocabulary size
        num_heads: Number of Medusa heads
        num_layers: Number of ResBlock layers per head
        lora_rank: LoRA rank (0 = no LoRA)
        lora_alpha: LoRA alpha scaling
        zero_init_mlp: Whether to zero-init ResBlock weights (ignored, weights from checkpoint)

    Returns:
        List of MedusaHead modules with loaded weights
    """
    heads = []

    for head_idx in range(num_heads):
        head = MedusaHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            zero_init_mlp=False,  # We'll load weights anyway
        )

        # Load ResBlock weights (convert to bfloat16 to match model)
        for layer_idx in range(num_layers):
            key = f"{head_idx}.blocks.{layer_idx}.linear.weight"
            if key in state_dict:
                tensor = state_dict[key]
                # Handle torch bfloat16 -> need to convert to float32 first for numpy
                if hasattr(tensor, 'dtype') and 'bfloat16' in str(tensor.dtype):
                    weight = tensor.float().numpy()
                else:
                    weight = tensor.numpy()
                head.blocks[layer_idx].linear.weight = mx.array(weight).astype(mx.bfloat16)

        # Load LoRA weights if present (convert to bfloat16)
        if lora_rank > 0:
            lora_a_key = f"{head_idx}.lora_A.weight"
            lora_b_key = f"{head_idx}.lora_B.weight"
            if lora_a_key in state_dict:
                tensor = state_dict[lora_a_key]
                if hasattr(tensor, 'dtype') and 'bfloat16' in str(tensor.dtype):
                    weight = tensor.float().numpy()
                else:
                    weight = tensor.numpy()
                head.lora_A.weight = mx.array(weight).astype(mx.bfloat16)
            if lora_b_key in state_dict:
                tensor = state_dict[lora_b_key]
                if hasattr(tensor, 'dtype') and 'bfloat16' in str(tensor.dtype):
                    weight = tensor.float().numpy()
                else:
                    weight = tensor.numpy()
                head.lora_B.weight = mx.array(weight).astype(mx.bfloat16)

        heads.append(head)

    return heads


def load_medusa_heads_from_numpy(
    state_dict: dict,
    hidden_size: int,
    vocab_size: int,
    num_heads: int,
    num_layers: int,
    lora_rank: int = 0,
    lora_alpha: Optional[int] = None,
    zero_init_mlp: bool = False,
) -> List[MedusaHead]:
    """
    Load Medusa heads from a state dict with numpy arrays.

    Args:
        state_dict: Dict mapping weight names to numpy arrays
        hidden_size: Model hidden dimension
        vocab_size: Vocabulary size
        num_heads: Number of Medusa heads
        num_layers: Number of ResBlock layers per head
        lora_rank: LoRA rank (0 = no LoRA)
        lora_alpha: LoRA alpha scaling
        zero_init_mlp: Whether to zero-init ResBlock weights (ignored)

    Returns:
        List of MedusaHead modules with loaded weights
    """
    heads = []

    for head_idx in range(num_heads):
        head = MedusaHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            zero_init_mlp=False,
        )

        # Load ResBlock weights (convert to bfloat16 to match model)
        for layer_idx in range(num_layers):
            key = f"{head_idx}.blocks.{layer_idx}.linear.weight"
            if key in state_dict:
                head.blocks[layer_idx].linear.weight = mx.array(state_dict[key]).astype(mx.bfloat16)

        # Load LoRA weights if present (convert to bfloat16)
        if lora_rank > 0:
            lora_a_key = f"{head_idx}.lora_A.weight"
            lora_b_key = f"{head_idx}.lora_B.weight"
            if lora_a_key in state_dict:
                head.lora_A.weight = mx.array(state_dict[lora_a_key]).astype(mx.bfloat16)
            if lora_b_key in state_dict:
                head.lora_B.weight = mx.array(state_dict[lora_b_key]).astype(mx.bfloat16)

        heads.append(head)

    return heads
