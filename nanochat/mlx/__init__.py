"""
MLX implementations for Gemma models with Medusa speculative decoding.

This module provides Apple Silicon-optimized inference using MLX,
matching the PyTorch implementations in nanochat/gemma_medusa/.

Components:
- heads: Medusa head implementations (ResBlock, MLP, LoRA)
- model: GemmaMedusaModel with KV caching and tree attention
- eval: Unified evaluation script for benchmarking
"""

from .heads import MedusaResBlock, MedusaHead
from .model import GemmaMedusaModel, MTPStats

__all__ = [
    "MedusaResBlock",
    "MedusaHead",
    "GemmaMedusaModel",
    "MTPStats",
]
