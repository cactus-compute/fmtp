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
from .heads import (
    MedusaLoRAHead,
    MedusaLoRAHeadWithMixer,
    MedusaHeadAttention,
    MedusaResBlock,
    MedusaHead,
    MedusaDeltaHead,
    IndependentMedusaHead,
    MultiLayerFusion,
    compute_multi_layer_indices,
)
from .hst_scorer import HSTScorer

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
    "MedusaHeadAttention",
    "MedusaResBlock",
    "MedusaHead",
    "MedusaDeltaHead",
    "IndependentMedusaHead",
    # Multi-layer fusion
    "MultiLayerFusion",
    "compute_multi_layer_indices",
    # HST scoring
    "HSTScorer",
]
