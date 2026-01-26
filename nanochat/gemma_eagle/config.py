"""
Configuration classes for Gemma EAGLE models.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GemmaEagleConfig:
    """
    Configuration for GemmaEagleModel.

    Attributes:
        base_model_name: HuggingFace model name for the base Gemma model
        draft_num_layers: Number of draft decoder layers (default: 1, like EAGLE-3)
        draft_depth: Tree depth for speculation (default: 5)
        draft_top_k: Top-k candidates per tree node (default: 8)
        draft_main_k: Number of candidates to expand per depth (default: None -> draft_top_k)
        total_tokens: Maximum draft tokens per speculation (default: 63)
        threshold: Log-probability threshold for acceptance (default: 1.0)
        use_draft_vocab: Whether to use reduced draft vocabulary (default: False)
        draft_vocab_size: Size of draft vocabulary if use_draft_vocab=True
        freeze_base: Whether to freeze base model parameters during training
        multi_layer_indices: Which layers to use for feature fusion (default: auto-computed)
    """
    # Base model
    base_model_name: str = "google/gemma-3-1b-it"

    # Draft model architecture
    draft_num_layers: int = 1  # EAGLE-3 uses 1 decoder layer
    draft_depth: int = 5  # Tree depth
    draft_top_k: int = 8  # Top-k per node
    draft_main_k: Optional[int] = None  # Number of nodes to expand per depth
    total_tokens: int = 63  # Max draft tokens

    # Acceptance threshold
    threshold: float = 1.0

    # Draft vocabulary (optional reduction for efficiency)
    use_draft_vocab: bool = False
    draft_vocab_size: Optional[int] = None

    # Training options
    freeze_base: bool = True

    # Multi-layer fusion indices (None = auto-compute)
    multi_layer_indices: Optional[list[int]] = None

    # These are populated from the base model after loading
    hidden_size: int = field(default=1536)
    intermediate_size: int = field(default=6144)
    num_hidden_layers: int = field(default=26)
    num_attention_heads: int = field(default=8)
    num_key_value_heads: int = field(default=4)
    vocab_size: int = field(default=262144)
    max_position_embeddings: int = field(default=8192)
    rms_norm_eps: float = field(default=1e-6)
    rope_theta: float = field(default=10000.0)
    head_dim: Optional[int] = None  # Gemma3 can have explicit head_dim

    def __post_init__(self):
        """Compute derived values."""
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.draft_vocab_size is None:
            self.draft_vocab_size = self.vocab_size

    @classmethod
    def from_hf_config(cls, hf_config, **kwargs):
        """
        Create a GemmaEagleConfig from a HuggingFace config.

        Args:
            hf_config: HuggingFace model config
            **kwargs: Override any config values

        Returns:
            GemmaEagleConfig instance
        """
        return cls(
            hidden_size=getattr(hf_config, 'hidden_size', 1536),
            intermediate_size=getattr(hf_config, 'intermediate_size', 6144),
            num_hidden_layers=getattr(hf_config, 'num_hidden_layers', 26),
            num_attention_heads=getattr(hf_config, 'num_attention_heads', 8),
            num_key_value_heads=getattr(hf_config, 'num_key_value_heads', 4),
            vocab_size=getattr(hf_config, 'vocab_size', 262144),
            max_position_embeddings=getattr(hf_config, 'max_position_embeddings', 8192),
            rms_norm_eps=getattr(hf_config, 'rms_norm_eps', 1e-6),
            rope_theta=getattr(hf_config, 'rope_theta', 10000.0),
            head_dim=getattr(hf_config, 'head_dim', None),
            **kwargs
        )

    def compute_multi_layer_indices(self) -> list[int]:
        """
        Compute evenly-spaced intermediate layer indices for multi-layer fusion.

        Returns layers at positions that divide the network into thirds:
        - Early layer (layer 2)
        - Middle layer (layer N/2)
        - Final layer (layer N-1)

        Returns:
            List of 3 layer indices in ascending order
        """
        if self.multi_layer_indices is not None:
            return self.multi_layer_indices

        n_layers = self.num_hidden_layers
        assert n_layers >= 6, f"Need at least 6 layers, got {n_layers}"

        first_idx = 2
        middle_idx = n_layers // 2
        last_idx = n_layers - 1

        return [first_idx, middle_idx, last_idx]
