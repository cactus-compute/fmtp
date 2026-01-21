"""
Gemma Medusa module for nanochat.

Provides Gemma 3 model integration with nanochat's evaluation and training infrastructure,
with Medusa multi-token prediction heads for speculative decoding.
"""

from .config import GemmaConfigWrapper, GemmaMedusaConfig
from .tokenizer import GemmaTokenizerWrapper
from .model import (
    GemmaModelWrapper,
    GemmaModelWrapperWithKVCache,
    GemmaMedusaModel,
    load_gemma_model,
    load_gemma_medusa_model,
)
from .heads import MedusaLoRAHead, MedusaLoRAHeadWithMixer, MedusaResBlock, MedusaHead, MedusaDeltaHead, IndependentMedusaHead

__all__ = [
    # Config
    "GemmaConfigWrapper",
    "GemmaMedusaConfig",
    # Tokenizer
    "GemmaTokenizerWrapper",
    # Models
    "GemmaModelWrapper",
    "GemmaModelWrapperWithKVCache",
    "GemmaMedusaModel",
    "load_gemma_model",
    "load_gemma_medusa_model",
    # Heads
    "MedusaLoRAHead",
    "MedusaLoRAHeadWithMixer",
    "MedusaResBlock",
    "MedusaHead",
    "MedusaDeltaHead",
    "IndependentMedusaHead",
]
