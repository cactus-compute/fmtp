"""
Hybrid Smoothed Tree (HST) Speculation module.

HST is a novel multi-token prediction algorithm that fuses:
1. MTP head drafter (leveraging cross-head attention architecture)
2. Learned retrieval module (3-layer MLP + SVD-compressed output)
3. Context suffix matching (zero-parameter lookup mechanism)

These components feed into a unified tree builder using Bayesian smoothed scoring
and priority-queue based expansion.

Key components:
- RetrievalMLP: Learned retrieval with SVD-compressed vocab projection
- SuffixMatcher: Context-aware suffix matching for repetition patterns
- HybridScorer: Combines MTP + retrieval + suffix scores
- HSTTreeBuilder: Priority-queue based adaptive tree construction

Usage:
    from nanochat.hst import RetrievalMLP, SuffixMatcher, HSTTreeBuilder
    from nanochat.hst_engine import HSTEngine

    # For full HST inference:
    engine = HSTEngine(model, tokenizer, retrieval_checkpoint="path/to/ckpt.pt")
    for tokens, masks in engine.generate(prompt_tokens):
        print(tokenizer.decode(tokens))
"""

from nanochat.hst.retrieval import (
    RetrievalMLP,
    RetrievalModuleTiny,
    load_svd_basis,
    compute_svd_basis,
    save_svd_basis,
)
from nanochat.hst.suffix_match import (
    SuffixMatcher,
    ContextBuffer,
)
from nanochat.hst.tree_builder import (
    HSTNode,
    HybridScorer,
    HSTTreeBuilder,
    build_hst_tree,
)
from nanochat.hst.tree_attention import (
    flatten_tree,
    build_tree_mask,
    HSTBuffers,
    generate_hst_buffers,
    extract_candidate_paths,
    merge_with_context,
    verify_tree_greedy,
    verify_tree_typical,
)

__all__ = [
    # Retrieval
    "RetrievalMLP",
    "RetrievalModuleTiny",
    "load_svd_basis",
    "compute_svd_basis",
    "save_svd_basis",
    # Suffix matching
    "SuffixMatcher",
    "ContextBuffer",
    # Tree building
    "HSTNode",
    "HybridScorer",
    "HSTTreeBuilder",
    "build_hst_tree",
    # Tree attention
    "flatten_tree",
    "build_tree_mask",
    "HSTBuffers",
    "generate_hst_buffers",
    "extract_candidate_paths",
    "merge_with_context",
    "verify_tree_greedy",
    "verify_tree_typical",
]
