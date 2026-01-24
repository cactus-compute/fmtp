"""
Medusa head implementations for multi-token prediction.

Contains all Medusa head variants:
- MedusaResBlock: Residual block with SiLU activation
- MedusaHead: Full projection head (hidden -> vocab)
- MedusaLoRAHead: Low-rank adapter head (hidden -> rank -> vocab)
- MedusaLoRAHeadWithMixer: MedusaLoRAHead with cross-head MLP mixing
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


class MedusaLoRAHeadWithMixer(MedusaLoRAHead):
    """
    MedusaLoRAHead with cross-head MLP mixing for improved multi-token prediction.

    This extends MedusaLoRAHead to support mixing information across multiple heads.
    The mixer is a small MLP that operates on the head dimension, allowing head k
    to leverage information from heads 0..k-1 for better sequential token prediction.

    The mixer is applied to the stacked ResBlock outputs before LoRA projection.
    Since num_heads is typically 4-5, this adds negligible compute overhead
    (~128 parameters for 4 heads with mixer_hidden=16).

    Architecture:
        1. Each head computes ResBlock(hidden_states) independently
        2. Stack outputs: (num_heads, B, T, hidden_size)
        3. Mix across heads: MLP(num_heads -> mixer_hidden -> num_heads) + residual
        4. Each head applies its LoRA projection to its mixed output

    The residual connection ensures the mixer starts as identity (when zero-initialized)
    and can learn to add cross-head information gradually.

    Note: This class is designed to work with GemmaMedusaModel which handles the
    stacking and mixing coordination. Individual forward() calls work like the
    parent class for backward compatibility.
    """
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int,
        lora_rank: int,
        lora_alpha: int | None = None,
        zero_init_mlp: bool = False,
        num_heads: int = 4,
        mixer_hidden: int = 16,
    ):
        super().__init__(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            zero_init_mlp=zero_init_mlp,
        )
        self.num_heads = num_heads
        self.mixer_hidden = mixer_hidden

        # Cross-head mixer MLP: num_heads -> mixer_hidden -> num_heads
        # This is shared across all heads and applied after ResBlocks
        self.mixer_fc1 = nn.Linear(num_heads, mixer_hidden, bias=False)
        self.mixer_fc2 = nn.Linear(mixer_hidden, num_heads, bias=False)

        # Zero-init fc2 so mixer starts as identity (residual only)
        nn.init.zeros_(self.mixer_fc2.weight)

    @staticmethod
    def apply_mixer(
        stacked_resblock: torch.Tensor,
        mixer_fc1: nn.Linear,
        mixer_fc2: nn.Linear,
        channel_fc: nn.Linear | None = None,
    ) -> torch.Tensor:
        """
        Apply cross-head mixing to stacked ResBlock outputs, followed by channel mixing.

        This follows the MLP-Mixer pattern:
        1. Head mixing: MLP across the num_heads dimension (token-mixing analog)
        2. Channel mixing: MLP across the hidden dimension (channel-mixing analog)

        Both use residual connections for stable training.

        Args:
            stacked_resblock: (num_heads, B, T, hidden_size) - stacked head outputs
            mixer_fc1: First layer of head mixer MLP
            mixer_fc2: Second layer of head mixer MLP
            channel_fc: Channel mixing linear (hidden -> hidden), optional

        Returns:
            (num_heads, B, T, hidden_size) - mixed head outputs
        """
        # Step 1: Head mixing (mix across num_heads dimension)
        # Permute to (B, T, hidden_size, num_heads) for mixing
        x_perm = stacked_resblock.permute(1, 2, 3, 0)

        # MLP mixing across heads with residual
        mixed = mixer_fc1(x_perm)
        mixed = F.gelu(mixed)
        mixed = mixer_fc2(mixed)

        # Permute back and add residual
        mixed = mixed.permute(3, 0, 1, 2)
        x = stacked_resblock + mixed  # (num_heads, B, T, hidden_size)

        # Step 2: Channel mixing (mix across hidden dimension) - ResBlock style
        if channel_fc is not None:
            x = x + F.silu(channel_fc(x))

        return x


class MedusaHeadAttention(nn.Module):
    """
    Cross-head attention block(s) for Medusa heads.

    Wraps nanochat/gpt.py Block to operate across heads:
    - Input: (num_heads, B, T, hidden_size)
    - Reshape to (B*T, num_heads, hidden_size) treating heads as sequence
    - Apply attention block(s) with QK norm, ReLU² MLP
    - Reshape back to (num_heads, B, T, hidden_size)

    Supports both causal and bidirectional attention via the causal parameter.
    Supports stacking multiple attention blocks via num_layers parameter.
    """
    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        n_attn_head: int | None = None,
        num_layers: int = 1,
        causal: bool = False,
    ):
        # Default: use 4 attention heads with head_dim=hidden_size/4 (matches Gemma 270M's ratio)
        if n_attn_head is None:
            n_attn_head = 4
        super().__init__()
        from dataclasses import dataclass
        from nanochat.gpt import Block

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create a minimal config for the Block
        @dataclass
        class MixerConfig:
            n_head: int = n_attn_head
            n_kv_head: int = n_attn_head
            n_embd: int = hidden_size

        # Create stacked attention blocks
        self.blocks = nn.ModuleList([
            Block(MixerConfig(), layer_idx=i, causal=causal)
            for i in range(num_layers)
        ])

        # Zero-init output projections so blocks start as identity
        for block in self.blocks:
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.zeros_(block.mlp.c_proj.weight)

        # Learnable position embeddings for head indices
        self.pos_emb = nn.Parameter(torch.zeros(num_heads, hidden_size))

        # RoPE disabled - using learnable position embeddings instead
        # Keeping RoPE code commented out in case we want to experiment with it later
        # head_dim = hidden_size // n_attn_head
        # pos = torch.arange(num_heads)
        # dim = torch.arange(0, head_dim, 2)
        # freqs = 1.0 / (100 ** (dim / head_dim))  # base=100 for short sequences
        # angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
        # cos = torch.cos(angles).repeat(1, 2).unsqueeze(0).unsqueeze(2)  # (1, num_heads, 1, head_dim)
        # sin = torch.sin(angles).repeat(1, 2).unsqueeze(0).unsqueeze(2)

        # Identity RoPE (no rotation) since we use learnable position embeddings
        # Note: apply_rotary_emb expects cos/sin with half the head_dim (rotates pairs)
        head_dim = hidden_size // n_attn_head
        self.register_buffer('cos', torch.ones(1, num_heads, 1, head_dim // 2), persistent=False)
        self.register_buffer('sin', torch.zeros(1, num_heads, 1, head_dim // 2), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-head attention block(s).

        Args:
            x: (num_heads, B, T, hidden_size)

        Returns:
            (num_heads, B, T, hidden_size)
        """
        num_heads, B, T, hidden_size = x.shape

        # Reshape: (num_heads, B, T, hidden) -> (B*T, num_heads, hidden)
        x_flat = x.permute(1, 2, 0, 3).reshape(B * T, num_heads, hidden_size)

        # Add learnable position embeddings
        x_flat = x_flat + self.pos_emb  # (B*T, num_heads, hidden) + (num_heads, hidden)

        # Apply all blocks (cos/sin are identity since we use learnable position embeddings)
        for block in self.blocks:
            x_flat = block(x_flat, (self.cos, self.sin), window_size=(-1, -1), kv_cache=None)

        # Reshape back: (B*T, num_heads, hidden) -> (num_heads, B, T, hidden)
        return x_flat.reshape(B, T, num_heads, hidden_size).permute(2, 0, 1, 3)


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


def compute_multi_layer_indices(n_layers: int) -> list[int]:
    """
    Compute evenly-spaced intermediate layer indices for multi-layer fusion.

    Given N total layers (0 to N-1), we always use the final layer (N-1) plus
    2 intermediate layers that divide the remaining (N-1) layers into thirds.

    For N layers, we pick layers at positions that cut (N-1) into thirds:
    - Layer at round((N-1) * 1/3)
    - Layer at round((N-1) * 2/3)
    - Layer N-1 (final layer, always included)

    Examples:
        N=26: layers [9, 17, 25] (round(25/3)=8, round(50/3)=17, final=25)
        N=18: layers [6, 11, 17]
        N=12: layers [4, 7, 11]

    Args:
        n_layers: Total number of transformer layers

    Returns:
        List of 3 layer indices in ascending order
    """
    # N-1 is the index of the last layer (0-indexed)
    last_idx = n_layers - 1

    # Compute the two intermediate positions that cut N-1 into thirds
    first_third = round(last_idx * 1 / 3)
    second_third = round(last_idx * 2 / 3)

    assert first_third < second_third < last_idx
    return [first_third, second_third, last_idx]


class MultiLayerFusion(nn.Module):
    """
    Preprocessor that fuses hidden states from multiple transformer layers.

    This module takes concatenated hidden states from multiple layers and
    down-projects them back to the original hidden dimension. It is designed
    to be used as a preprocessing step before any Medusa head (MedusaLoRAHead,
    MedusaDeltaHead, etc.).

    Architecture:
        Input: (B, T, num_layers * hidden_size) - concatenated layer outputs
               (B, T, hidden_size) - final layer hidden states (for residual)
        -> Linear down-projection
        -> SiLU activation
        -> Add residual from final hidden states
        Output: (B, T, hidden_size) - fused representation

    The residual connection ensures the model can always fall back to the
    original final layer representation while learning to incorporate
    information from intermediate layers.

    Args:
        hidden_size: Hidden dimension of the model
        num_fused_layers: Number of layers being fused (default: 3)
    """
    def __init__(self, hidden_size: int, num_fused_layers: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_fused_layers = num_fused_layers

        # Down-projection from concatenated multi-layer hidden states to original size
        # Input: (B, T, num_fused_layers * hidden_size)
        # Output: (B, T, hidden_size)
        self.down_proj = nn.Linear(num_fused_layers * hidden_size, hidden_size, bias=False)

    def forward(
        self,
        multi_layer_hidden: torch.Tensor,
        final_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse multi-layer hidden states into single hidden representation.

        Args:
            multi_layer_hidden: (B, T, num_fused_layers * hidden_size) concatenated hidden states
            final_hidden: (B, T, hidden_size) final layer hidden states for residual

        Returns:
            (B, T, hidden_size) fused hidden states with residual from final layer
        """
        return final_hidden + F.silu(self.down_proj(multi_layer_hidden))
