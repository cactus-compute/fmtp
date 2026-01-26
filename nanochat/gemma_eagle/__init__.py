"""
EAGLE-3 speculative decoding for Gemma3 models.

This module implements EAGLE-3 (Extrapolation Algorithm for Greater Language-model
Efficiency) adapted for Gemma3 architecture. EAGLE achieves significant speedups
(up to 6x) over vanilla autoregressive decoding by using a lightweight draft model
to predict multiple tokens that are verified in parallel.

Key Components:
    - GemmaEagleConfig: Configuration for EAGLE draft model
    - GemmaEagleModel: Draft model with multi-layer fusion and tree generation
    - EagleGenerator: High-level generation interface with speculative decoding
    - EagleKVCache: Pre-allocated KV cache for efficient generation

Usage:
    from nanochat.gemma_eagle import GemmaEagleConfig, GemmaEagleModel, EagleGenerator

    # Create model
    config = GemmaEagleConfig(base_model_name="google/gemma-3-1b-it")
    model = GemmaEagleModel(config)

    # Load trained draft weights
    model.load_draft_state_dict(torch.load("eagle_draft.pt"))

    # Generate with speculative decoding
    generator = EagleGenerator(model)
    output, stats = generator.generate_simple("Hello, how are you?", max_new_tokens=256)

    print(f"Output: {output}")
    print(f"Speedup: {stats.mean_accepted_length:.2f}x")
    print(f"Tokens/sec: {stats.tokens_per_second:.1f}")

Training:
    See scripts/eagle_train.py for training the draft model.

    uv run python -m scripts.eagle_train \\
        --base-model google/gemma-3-1b-it \\
        --data-path data/train.json \\
        --output-dir checkpoints/eagle
"""

from .config import GemmaEagleConfig
from .model import GemmaEagleModel, MultiLayerFusion, EagleDecoderLayer
from .inference import EagleGenerator, EagleGenerationStats, benchmark_eagle
from .kv_cache import EagleKVCache, StaticKVCache, DualKVCache
from .tree import (
    generate_tree_buffers,
    evaluate_posterior,
    generate_candidates,
    get_default_tree_choices,
)

__all__ = [
    # Config
    "GemmaEagleConfig",
    # Model
    "GemmaEagleModel",
    "MultiLayerFusion",
    "EagleDecoderLayer",
    # Inference
    "EagleGenerator",
    "EagleGenerationStats",
    "benchmark_eagle",
    # KV Cache
    "EagleKVCache",
    "StaticKVCache",
    "DualKVCache",
    # Tree utilities
    "generate_tree_buffers",
    "evaluate_posterior",
    "generate_candidates",
    "get_default_tree_choices",
]
