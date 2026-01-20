"""
Configuration classes for Gemma Medusa models.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GemmaConfigWrapper:
    """Wrapper to make Gemma config match nanochat's expected interface."""
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    medusa_num_heads: int = 0


@dataclass
class GemmaMedusaConfig(GemmaConfigWrapper):
    """
    Configuration for GemmaMedusaModel.

    Extends GemmaConfigWrapper with Medusa-specific options.

    Attributes:
        base_model_name: HuggingFace model name for the base Gemma model
        medusa_num_heads: Number of Medusa prediction heads (default: 4)
        medusa_num_layers: Number of ResBlock layers per head (default: 1)
        medusa_head_type: Type of head architecture ("lora", "delta", "full")
        lora_rank: Rank for LoRA projections (only used if head_type="lora")
        lora_alpha: Alpha scaling for LoRA (defaults to lora_rank)
        freeze_base: Whether to freeze base model parameters during training
    """
    # Base model
    base_model_name: str = "google/gemma-3-1b-it"

    # Medusa head configuration
    medusa_num_heads: int = 4
    medusa_num_layers: int = 1
    medusa_head_type: Literal["lora", "delta", "full"] = "lora"

    # LoRA-specific options (only used if head_type="lora")
    lora_rank: int = 64
    lora_alpha: int | None = None  # Defaults to lora_rank if None

    # Training options
    freeze_base: bool = True

    # These are populated from the base model after loading
    sequence_len: int = field(default=8192)
    vocab_size: int = field(default=262144)
    n_layer: int = field(default=26)
    n_head: int = field(default=8)
    n_kv_head: int = field(default=4)
    n_embd: int = field(default=1536)

    def __post_init__(self):
        """Set lora_alpha to lora_rank if not specified."""
        if self.lora_alpha is None:
            self.lora_alpha = self.lora_rank

    @classmethod
    def from_hf_config(cls, hf_config, **kwargs):
        """
        Create a GemmaMedusaConfig from a HuggingFace config.

        Args:
            hf_config: HuggingFace model config
            **kwargs: Override any config values

        Returns:
            GemmaMedusaConfig instance
        """
        return cls(
            sequence_len=getattr(hf_config, 'max_position_embeddings', 8192),
            vocab_size=getattr(hf_config, 'vocab_size', 262144),
            n_layer=getattr(hf_config, 'num_hidden_layers', 26),
            n_head=getattr(hf_config, 'num_attention_heads', 8),
            n_kv_head=getattr(hf_config, 'num_key_value_heads',
                             getattr(hf_config, 'num_attention_heads', 8)),
            n_embd=getattr(hf_config, 'hidden_size', 1536),
            **kwargs
        )
