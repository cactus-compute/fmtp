"""
Medusa head implementations for multi-token prediction.

Contains all Medusa head variants:
- MedusaResBlock: Residual block with SiLU activation
- MedusaHead: Full projection head (hidden -> vocab)
- MedusaLoRAHead: Low-rank adapter head (hidden -> rank -> vocab)
- MedusaDeltaHead: Hidden-space delta head (reuses lm_head)
- IndependentMedusaHead: Deprecated, use MedusaDeltaHead instead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedusaResBlock(nn.Module):
    """Residual block for Medusa heads. Optional zero-init for identity at start."""
    def __init__(self, hidden_size, zero_init=False):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        if zero_init:
            nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return x + F.silu(self.linear(x))


class MedusaHead(nn.Module):
    """Single Medusa prediction head: ResBlocks + linear projection."""
    def __init__(self, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList([MedusaResBlock(hidden_size) for _ in range(num_layers)])
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.proj(x)


class MedusaLoRAHead(nn.Module):
    """
    Medusa head using LoRA adapter over the shared lm_head.

    Instead of a full (vocab_size, n_embd) projection, uses:
    - lora_A: (hidden_size -> rank) down-projection
    - lora_B: (rank -> vocab_size) up-projection

    Output = lm_head(x) + (alpha / rank) * lora_B(lora_A(x))

    The alpha/rank scaling keeps the adapter contribution stable across different ranks.
    """
    def __init__(self, hidden_size, vocab_size, num_layers, lora_rank, lora_alpha=None, zero_init_mlp=False):
        super().__init__()
        self.blocks = nn.ModuleList([MedusaResBlock(hidden_size, zero_init=zero_init_mlp) for _ in range(num_layers)])
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank  # default: alpha = rank (scaling = 1)
        self.scaling = self.lora_alpha / self.lora_rank
        self.lora_A = nn.Linear(hidden_size, lora_rank, bias=False)
        self.lora_B = nn.Linear(lora_rank, vocab_size, bias=False)
        # Standard LoRA init: A gets default Kaiming, B gets zeros
        # This ensures LoRA contribution starts at zero
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        """
        Compute LoRA delta for inference.
        Returns the delta to be added to lm_head(x).
        """
        for block in self.blocks:
            x = block(x)
        return self.scaling * self.lora_B(self.lora_A(x))

    def get_merged_weight(self, lm_head_weight):
        """Merge LoRA weights for fused CE: W_merged = W_base + scaling * B @ A"""
        return lm_head_weight + self.scaling * (self.lora_B.weight @ self.lora_A.weight)


class IndependentMedusaHead(nn.Module):
    """
    DEPRECATED: Use MedusaDeltaHead instead for better performance.

    Low-rank Medusa head that adds a delta to lm_head output.
    This implementation is 6-7x slower than baseline due to chunked cross-entropy.

    Architecture: output = lm_head(h) + W_b(W_a(ResBlocks(h)))
    - ResBlocks contain the SiLU nonlinearity
    - W_a: down-projection (hidden -> rank)
    - W_b: up-projection (rank -> vocab)
    """
    def __init__(self, hidden_size, vocab_size, num_layers, rank=32):
        super().__init__()
        import warnings
        warnings.warn(
            "IndependentMedusaHead is deprecated and 6-7x slower than baseline. "
            "Use MedusaDeltaHead (--medusa-delta) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.blocks = nn.ModuleList([MedusaResBlock(hidden_size) for _ in range(num_layers)])
        self.rank = rank

        # Low-rank projection: hidden -> rank -> vocab
        self.W_a = nn.Linear(hidden_size, rank, bias=False)
        self.W_b = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.W_b(self.W_a(x))


class MedusaDeltaHead(nn.Module):
    """
    Medusa head that computes a delta in hidden space, then uses shared lm_head.

    Architecture: h_modified = h + ResBlocks(h)
    Output (computed by caller): lm_head(h_modified)

    This is fast because:
    - No vocab-sized projections in the head itself
    - Reuses lm_head with Liger fused CE (no weight composition)
    - All operations are in hidden-dim space, not vocab-dim space
    """
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList([MedusaResBlock(hidden_size) for _ in range(num_layers)])

    def forward(self, x):
        """Returns the delta to add to hidden states. Caller computes lm_head(x + delta)."""
        delta = x
        for block in self.blocks:
            delta = block(delta)
        return delta
