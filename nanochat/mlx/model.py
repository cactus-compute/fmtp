"""
MLX implementation of GemmaMedusaModel for speculative decoding.

Provides Apple Silicon-optimized inference with:
- KV caching for efficient autoregressive generation
- Tree attention for speculative decoding verification
- Compatible with PyTorch checkpoints from nanochat/gemma_medusa

Key differences from PyTorch version:
- Uses mlx_lm for base model loading
- MLX-native KV cache management
- Simplified tree attention using MLX primitives
"""

import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

from .heads import MedusaHead, load_medusa_heads_from_torch, load_medusa_heads_from_numpy


# Default tree structures by head count (same as PyTorch version)
DEFAULT_TREES: Dict[int, List[Tuple[int, ...]]] = {
    1: [(i,) for i in range(10)],
    2: (
        [(i,) for i in range(10)] +
        [(i, j) for i in range(5) for j in range(5)]
    ),
    3: (
        [(i,) for i in range(10)] +
        [(i, j) for i in range(5) for j in range(5)] +
        [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]
    ),
    4: (
        [(i,) for i in range(10)] +
        [(i, j) for i in range(5) for j in range(5)] +
        [(i, j, k) for i in range(3) for j in range(3) for k in range(3)] +
        [(i, j, k, m) for i in range(2) for j in range(2) for k in range(2) for m in range(2)]
    ),
}

SPARSE_TREES: Dict[int, List[Tuple[int, ...]]] = {
    1: [(0,), (1,), (2,)],
    2: [(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (1, 1)],
    3: (
        [(0,), (1,), (2,)] +
        [(0, 0), (0, 1), (1, 0), (1, 1)] +
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)]
    ),
    4: (
        [(0,), (1,), (2,)] +
        [(0, 0), (0, 1), (1, 0), (1, 1)] +
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)] +
        [(0, 0, 0, 0)]
    ),
}

# Small trees optimized for low overhead (~12 positions total)
# Tree length = 1 (root) + len(tree_choices)
SMALL_TREES: Dict[int, List[Tuple[int, ...]]] = {
    1: [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)],  # 12 total
    2: (
        [(0,), (1,), (2,), (3,), (4,)] +  # 5 depth-1
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]  # 6 depth-2
    ),  # 12 total (1 root + 11 choices)
}


@dataclass
class EntropyRecord:
    """Record of entropy and acceptance for a single speculation iteration."""
    entropy: float  # Entropy of main model logits before speculation
    accept_length: int  # Number of tokens accepted (1 = only main, 2+ = speculation succeeded)


@dataclass
class MTPStats:
    """Statistics from MTP generation for benchmarking."""
    tokens_generated: int
    forward_passes: int
    total_proposed: int
    total_accepted: int
    timing: Optional[Dict[str, float]] = None
    entropy_log: Optional[List[EntropyRecord]] = None  # Per-iteration entropy data
    speculation_skipped: int = 0  # Iterations where speculation was skipped due to entropy gating

    @property
    def mean_accepted_length(self) -> float:
        """Average number of tokens accepted per forward pass."""
        return self.tokens_generated / max(1, self.forward_passes)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed tokens that were accepted."""
        return self.total_accepted / max(1, self.total_proposed)

    @property
    def skip_rate(self) -> float:
        """Fraction of iterations where speculation was skipped."""
        return self.speculation_skipped / max(1, self.forward_passes)


# ============================================================================
# Entropy Computation for Adaptive Speculation
# ============================================================================

def compute_entropy(logits: mx.array) -> float:
    """
    Compute entropy of softmax distribution from logits.

    Args:
        logits: (vocab_size,) logits for single position

    Returns:
        Scalar entropy value (higher = more uncertain)
    """
    probs = mx.softmax(logits, axis=-1)
    # Avoid log(0) by adding small epsilon
    log_probs = mx.log(probs + 1e-10)
    entropy = -mx.sum(probs * log_probs)
    return float(entropy.item())


# ============================================================================
# Custom Metal Kernel for Per-Position RoPE
# ============================================================================

_ROPE_KERNEL_CACHE: Dict[str, Any] = {}
_ROPE_FREQ_CACHE: Dict[Tuple[int, float], mx.array] = {}


def _get_rope_frequencies(half_dims: int, base: float) -> mx.array:
    """Get cached RoPE frequencies to avoid recomputation."""
    key = (half_dims, base)
    if key not in _ROPE_FREQ_CACHE:
        freq_exps = mx.arange(0, half_dims, dtype=mx.float32) / half_dims
        _ROPE_FREQ_CACHE[key] = 1.0 / (base ** freq_exps)
    return _ROPE_FREQ_CACHE[key]


def _create_rope_positions_kernel():
    """
    Create a Metal kernel for RoPE with per-position offsets.

    This is ~2x faster than the pure Python implementation for tree-sized inputs.
    """
    source = '''
        uint idx = thread_position_in_grid.x;

        uint B = dims[0];
        uint n_heads = dims[1];
        uint L = dims[2];
        uint head_dim = dims[3];
        uint half_dims = dims[4];
        float base = params[0];
        float scale_val = params[1];

        uint total = B * n_heads * L * half_dims;
        if (idx >= total) return;

        uint d = idx % half_dims;
        uint l = (idx / half_dims) % L;
        uint h = (idx / (half_dims * L)) % n_heads;
        uint b = idx / (half_dims * L * n_heads);

        float pos = float(positions[l]) * scale_val;

        float freq_exp = float(d) / float(half_dims);
        float freq = 1.0f / metal::pow(base, freq_exp);
        float theta = pos * freq;
        float cos_theta = metal::cos(theta);
        float sin_theta = metal::sin(theta);

        uint base_idx = b * (n_heads * L * head_dim) + h * (L * head_dim) + l * head_dim;
        uint idx1 = base_idx + d;
        uint idx2 = base_idx + d + half_dims;

        float x1 = float(x[idx1]);
        float x2 = float(x[idx2]);

        float rx1 = x1 * cos_theta - x2 * sin_theta;
        float rx2 = x1 * sin_theta + x2 * cos_theta;

        out[idx1] = T(rx1);
        out[idx2] = T(rx2);
    '''

    return mx.fast.metal_kernel(
        name="rope_with_positions",
        input_names=["x", "positions", "dims", "params"],
        output_names=["out"],
        source=source,
    )


def _create_fused_rope_qk_kernel():
    """
    Create a Metal kernel that applies RoPE to both Q and K in a single pass.

    This reduces kernel launch overhead by ~50% compared to two separate calls.
    """
    source = '''
        uint idx = thread_position_in_grid.x;

        uint B = dims[0];
        uint n_heads_q = dims[1];
        uint n_heads_k = dims[2];
        uint L = dims[3];
        uint head_dim = dims[4];
        uint half_dims = dims[5];
        float base = params[0];
        float scale_val = params[1];

        // Total work items for Q
        uint total_q = B * n_heads_q * L * half_dims;
        // Total work items for K
        uint total_k = B * n_heads_k * L * half_dims;
        uint total = total_q + total_k;

        if (idx >= total) return;

        // Determine if we're processing Q or K
        bool is_q = idx < total_q;
        uint local_idx = is_q ? idx : (idx - total_q);
        uint n_heads = is_q ? n_heads_q : n_heads_k;

        uint d = local_idx % half_dims;
        uint l = (local_idx / half_dims) % L;
        uint h = (local_idx / (half_dims * L)) % n_heads;
        uint b = local_idx / (half_dims * L * n_heads);

        float pos = float(positions[l]) * scale_val;

        float freq_exp = float(d) / float(half_dims);
        float freq = 1.0f / metal::pow(base, freq_exp);
        float theta = pos * freq;
        float cos_theta = metal::cos(theta);
        float sin_theta = metal::sin(theta);

        uint base_idx = b * (n_heads * L * head_dim) + h * (L * head_dim) + l * head_dim;
        uint idx1 = base_idx + d;
        uint idx2 = base_idx + d + half_dims;

        if (is_q) {
            float x1 = float(q[idx1]);
            float x2 = float(q[idx2]);
            q_out[idx1] = T(x1 * cos_theta - x2 * sin_theta);
            q_out[idx2] = T(x1 * sin_theta + x2 * cos_theta);
        } else {
            float x1 = float(k[idx1]);
            float x2 = float(k[idx2]);
            k_out[idx1] = T(x1 * cos_theta - x2 * sin_theta);
            k_out[idx2] = T(x1 * sin_theta + x2 * cos_theta);
        }
    '''

    return mx.fast.metal_kernel(
        name="fused_rope_qk",
        input_names=["q", "k", "positions", "dims", "params"],
        output_names=["q_out", "k_out"],
        source=source,
    )


def apply_rope_fused_qk_kernel(
    q: mx.array,
    k: mx.array,
    positions: mx.array,
    base: float = 10000.0,
    scale: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """
    Apply RoPE to both Q and K in a single fused kernel call.

    This is ~30% faster than two separate kernel calls due to reduced
    kernel launch overhead and better memory access patterns.

    Args:
        q: (B, n_heads_q, L, head_dim) queries
        k: (B, n_heads_k, L, head_dim) keys
        positions: (L,) position offsets
        base: RoPE base frequency
        scale: Position scale factor

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    B, n_heads_q, L, head_dim = q.shape
    _, n_heads_k, _, _ = k.shape
    half_dims = head_dim // 2

    if "fused_rope_qk" not in _ROPE_KERNEL_CACHE:
        _ROPE_KERNEL_CACHE["fused_rope_qk"] = _create_fused_rope_qk_kernel()
    kernel = _ROPE_KERNEL_CACHE["fused_rope_qk"]

    dims_arr = mx.array([B, n_heads_q, n_heads_k, L, head_dim, half_dims], dtype=mx.uint32)
    params_arr = mx.array([base, scale], dtype=mx.float32)

    # Total grid size covers both Q and K
    total_q = B * n_heads_q * L * half_dims
    total_k = B * n_heads_k * L * half_dims
    grid_size = total_q + total_k

    outputs = kernel(
        inputs=[q, k, positions, dims_arr, params_arr],
        template=[("T", q.dtype)],
        grid=(grid_size, 1, 1),
        threadgroup=(min(256, grid_size), 1, 1),
        output_shapes=[q.shape, k.shape],
        output_dtypes=[q.dtype, k.dtype],
    )

    return outputs[0], outputs[1]


# ============================================================================
# Compiled Pure Functions for MTP
# ============================================================================

def trim_cache(cache: List, n_trim: int) -> None:
    """
    Efficiently trim the last n_trim tokens from all cache layers.

    This is faster than the naive loop because it batches the slice operations.
    """
    for layer_cache in cache:
        if layer_cache.keys is not None:
            layer_cache.keys = layer_cache.keys[:, :, :-n_trim, :]
            layer_cache.values = layer_cache.values[:, :, :-n_trim, :]
            layer_cache.offset -= n_trim
            if hasattr(layer_cache, '_idx'):
                layer_cache._idx -= n_trim


@mx.compile
def _compiled_get_top2(logits: mx.array) -> Tuple[mx.array, mx.array]:
    """
    Compiled function to get top-2 tokens using two argmax calls.

    Faster than argsort for getting just top-2 from large vocab.
    """
    token1 = mx.argmax(logits)
    # Mask out top-1 and find top-2
    masked = mx.where(
        mx.arange(logits.shape[0]) == token1,
        mx.array(-float('inf')),
        logits
    )
    token2 = mx.argmax(masked)
    return token1, token2

@mx.compile
def _compiled_generate_candidates(
    main_logits: mx.array,
    medusa_logits: mx.array,
    tree_indices: mx.array,
    retrieve_indices: mx.array,
    topk: int,
) -> Tuple[mx.array, mx.array]:
    """
    Compiled candidate generation - pure function version.

    Args:
        main_logits: (1, vocab_size) Main model logits
        medusa_logits: (num_heads, 1, vocab_size) Medusa logits
        tree_indices: (tree_len,) Mapping from flat to tree positions
        retrieve_indices: (num_candidates, max_depth) Candidate path indices
        topk: Number of top-k predictions per head

    Returns:
        candidates: (num_candidates, max_depth) Candidate sequences
        tree_candidates: (tree_len,) Tokens in tree structure
    """
    # Greedy selection
    base_token = mx.argmax(main_logits[0])

    # Get top-k from each Medusa head
    sorted_indices = mx.argsort(-medusa_logits[:, 0, :], axis=-1)
    medusa_topk = sorted_indices[:, :topk]

    # Build flat candidate array
    flat_candidates = mx.concatenate([base_token[None], medusa_topk.reshape(-1)])

    # Map to tree structure
    tree_candidates = flat_candidates[tree_indices]

    # Extract candidate paths
    safe_indices = mx.clip(retrieve_indices, 0, None)
    candidates = tree_candidates[safe_indices]
    candidates = candidates * (retrieve_indices >= 0).astype(mx.int32)

    return candidates, tree_candidates


@mx.compile
def _compiled_evaluate_candidates(
    tree_logits: mx.array,
    candidates: mx.array,
    retrieve_indices: mx.array,
) -> Tuple[mx.array, mx.array]:
    """
    Compiled candidate evaluation - pure function version.

    Returns arrays instead of ints to stay in compiled graph.

    Args:
        tree_logits: (tree_len, vocab_size) Logits at tree positions
        candidates: (num_candidates, max_depth) Candidate sequences
        retrieve_indices: (num_candidates, max_depth) Indices into tree

    Returns:
        best_candidate: Scalar array with best candidate index
        accept_length: Scalar array with number of tokens to accept
    """
    safe_indices = mx.clip(retrieve_indices, 0, None)
    valid_mask = retrieve_indices >= 0

    # Compute argmax for all tree positions
    tree_predictions = mx.argmax(tree_logits, axis=-1)

    # Get predictions at needed positions
    candidate_predictions = tree_predictions[safe_indices]

    # Check matches
    matches = (candidates[:, 1:] == candidate_predictions[:, :-1])
    matches = matches & valid_mask[:, 1:]

    # Find longest matching prefix
    cumulative_matches = mx.cumprod(matches.astype(mx.int32), axis=1)
    accept_lengths = mx.sum(cumulative_matches, axis=1)

    best_candidate = mx.argmax(accept_lengths)
    accept_length = accept_lengths[best_candidate]

    return best_candidate, accept_length


def apply_rope_with_positions_kernel(
    x: mx.array,
    positions: mx.array,
    base: float = 10000.0,
    scale: float = 1.0,
) -> mx.array:
    """
    Apply RoPE with per-position offsets using custom Metal kernel.

    Args:
        x: (B, n_heads, L, head_dim) tensor
        positions: (L,) position offsets for each sequence position
        base: RoPE base frequency
        scale: Position scale factor

    Returns:
        Rotated tensor of same shape as x
    """
    B, n_heads, L, head_dim = x.shape
    half_dims = head_dim // 2

    if "rope" not in _ROPE_KERNEL_CACHE:
        _ROPE_KERNEL_CACHE["rope"] = _create_rope_positions_kernel()
    kernel = _ROPE_KERNEL_CACHE["rope"]

    dims_arr = mx.array([B, n_heads, L, head_dim, half_dims], dtype=mx.uint32)
    params_arr = mx.array([base, scale], dtype=mx.float32)

    grid_size = B * n_heads * L * half_dims

    outputs = kernel(
        inputs=[x, positions, dims_arr, params_arr],
        template=[("T", x.dtype)],
        grid=(grid_size, 1, 1),
        threadgroup=(min(256, grid_size), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )

    return outputs[0]


_TREE_MASK_CACHE: Dict[Tuple[int, int], mx.array] = {}


def build_tree_attention_mask_mlx(
    tree_attn_mask_bool: mx.array,
    cache_len: int,
) -> mx.array:
    """
    Build full attention mask for tree verification with KV cache.

    Uses caching to avoid repeated allocations for common cache lengths.

    The mask allows:
    - All tree positions to attend to all cached positions (causal wrt cache)
    - Tree positions to attend to ancestors according to tree_attn_mask

    Args:
        tree_attn_mask_bool: (1, 1, tree_len, tree_len) boolean mask (pre-converted)
        cache_len: Length of cached prefix

    Returns:
        full_mask: (1, 1, tree_len, cache_len + tree_len) boolean mask
    """
    tree_len = tree_attn_mask_bool.shape[-1]

    # Check cache for pre-allocated ones mask of this size
    cache_key = (tree_len, cache_len)
    if cache_key not in _TREE_MASK_CACHE:
        # Only cache for common sizes to avoid memory bloat
        if cache_len <= 512:
            _TREE_MASK_CACHE[cache_key] = mx.ones((1, 1, tree_len, cache_len), dtype=mx.bool_)
            cache_mask = _TREE_MASK_CACHE[cache_key]
        else:
            cache_mask = mx.ones((1, 1, tree_len, cache_len), dtype=mx.bool_)
    else:
        cache_mask = _TREE_MASK_CACHE[cache_key]

    # Concatenate: [cache_mask | tree_mask]
    full_mask = mx.concatenate([cache_mask, tree_attn_mask_bool], axis=-1)

    return full_mask


def generate_tree_buffers(
    medusa_choices: List[Tuple[int, ...]],
    topk: int = 10,
) -> Dict[str, mx.array]:
    """
    Generate buffers for tree attention in MTP speculative decoding.

    Args:
        medusa_choices: List of tuples defining the tree structure
        topk: Number of top predictions from each Medusa head

    Returns:
        Dictionary containing tree attention buffers
    """
    import numpy as np

    sorted_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    tree_len = len(sorted_choices) + 1  # +1 for root token

    # 1. Create tree attention mask (using numpy, then convert)
    attn_mask = np.zeros((tree_len, tree_len), dtype=np.float32)
    np.fill_diagonal(attn_mask, 1.0)  # Each node attends to itself
    attn_mask[:, 0] = 1.0  # All nodes attend to root

    for idx, choice in enumerate(sorted_choices):
        node_idx = idx + 1
        for depth in range(len(choice) - 1):
            ancestor_choice = choice[: depth + 1]
            ancestor_idx = sorted_choices.index(ancestor_choice) + 1
            attn_mask[node_idx, ancestor_idx] = 1.0

    # 2. Tree indices: map tree positions to candidate token indices
    tree_indices = np.zeros(tree_len, dtype=np.int32)
    tree_indices[0] = 0  # Root maps to position 0

    for idx, choice in enumerate(sorted_choices):
        node_idx = idx + 1
        depth = len(choice) - 1
        token_rank = choice[-1]
        tree_indices[node_idx] = token_rank + topk * depth + 1

    # 3. Position IDs for RoPE
    position_ids = np.zeros(tree_len, dtype=np.int32)
    for idx, choice in enumerate(sorted_choices):
        position_ids[idx + 1] = len(choice)

    # 4. Retrieve indices for extracting candidate paths
    max_depth = max(len(c) for c in sorted_choices) + 1
    num_candidates = len(sorted_choices) + 1

    retrieve_indices = np.full((num_candidates, max_depth), -1, dtype=np.int32)
    retrieve_indices[0, 0] = 0  # Root candidate points to root

    for idx, choice in enumerate(sorted_choices):
        candidate_idx = idx + 1
        retrieve_indices[candidate_idx, 0] = 0  # Root
        for depth in range(len(choice)):
            partial_choice = choice[: depth + 1]
            node_idx = sorted_choices.index(partial_choice) + 1
            retrieve_indices[candidate_idx, depth + 1] = node_idx

    # Pre-convert to boolean for faster mask building
    tree_attn_mask_bool = mx.array(attn_mask[None, None, :, :] > 0.5)

    return {
        "tree_attn_mask": tree_attn_mask_bool,  # (1, 1, tree_len, tree_len) boolean
        "tree_indices": mx.array(tree_indices),
        "tree_position_ids": mx.array(position_ids),
        "retrieve_indices": mx.array(retrieve_indices),
    }


class GemmaMedusaModel:
    """
    MLX wrapper for Gemma model with Medusa heads for speculative decoding.

    This class:
    1. Loads the base Gemma model using mlx_lm
    2. Loads Medusa heads from a PyTorch checkpoint
    3. Provides generation methods with KV caching and tree attention

    Usage:
        model = GemmaMedusaModel.from_checkpoint(
            checkpoint_path="~/.cache/nanochat/gemma_medusa_270m_wildchat_100k_nolora",
            mlx_model_name="mlx-community/gemma-3-270m-it-bf16",
        )
        output_ids, stats = model.generate_mtp(input_ids, max_new_tokens=128)
    """

    def __init__(
        self,
        base_model,
        tokenizer,
        medusa_heads: List[MedusaHead],
        hidden_size: int,
        vocab_size: int,
        medusa_num_heads: int,
        medusa_num_layers: int,
        lora_rank: int = 0,
    ):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.medusa_heads = medusa_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.lora_rank = lora_rank

        # Cache for tree buffers
        self._tree_buffers_cache: Optional[Dict[str, mx.array]] = None
        self._tree_buffers_config: Optional[Tuple] = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        mlx_model_name: Optional[str] = None,
    ) -> "GemmaMedusaModel":
        """
        Load GemmaMedusaModel from a PyTorch checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory containing config.json and medusa_heads.pt
            mlx_model_name: MLX model name (e.g., "mlx-community/gemma-3-270m-it-bf16").
                           If None, auto-detects from checkpoint config.

        Returns:
            GemmaMedusaModel instance
        """
        checkpoint_path = os.path.expanduser(checkpoint_path)

        # Load config
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        base_model_name = config.get("base_model", "google/gemma-3-270m-it")
        medusa_num_heads = config.get("medusa_num_heads", 4)
        medusa_num_layers = config.get("medusa_num_layers", 1)
        lora_rank = config.get("lora_rank", 0)
        lora_alpha = config.get("lora_alpha", lora_rank)

        # Auto-detect MLX model name if not provided
        if mlx_model_name is None:
            # Convert HuggingFace name to MLX community name
            # google/gemma-3-270m-it -> mlx-community/gemma-3-270m-it-bf16
            model_suffix = base_model_name.split("/")[-1]
            mlx_model_name = f"mlx-community/{model_suffix}-bf16"

        print(f"Loading MLX base model: {mlx_model_name}")
        base_model, tokenizer = mlx_load(mlx_model_name)

        # Get model dimensions from base model
        # MLX models have different attribute access patterns
        if hasattr(base_model, "model"):
            inner_model = base_model.model
        else:
            inner_model = base_model

        # Try to get hidden_size from model config
        if hasattr(inner_model, "args"):
            hidden_size = inner_model.args.hidden_size
            vocab_size = inner_model.args.vocab_size
        elif hasattr(base_model, "config"):
            hidden_size = base_model.config.get("hidden_size", 640)
            vocab_size = base_model.config.get("vocab_size", 262144)
        else:
            # Fallback for Gemma 3 270M
            hidden_size = 640
            vocab_size = 262144

        print(f"Model dimensions: hidden_size={hidden_size}, vocab_size={vocab_size}")

        # Load Medusa heads from checkpoint
        checkpoint_file = os.path.join(checkpoint_path, "final", "medusa_heads.pt")
        if not os.path.exists(checkpoint_file):
            checkpoint_file = os.path.join(checkpoint_path, "medusa_heads.pt")

        print(f"Loading Medusa heads from: {checkpoint_file}")

        # Try to load with torch if available, otherwise use safetensors/pickle
        try:
            import torch
            state = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
            heads_state = state.get("medusa_heads", state)
            # Convert torch tensors to numpy for MLX
            # Handle bfloat16 by converting to float32 first
            heads_state_np = {}
            for k, v in heads_state.items():
                if v.dtype == torch.bfloat16:
                    v = v.float()  # Convert bfloat16 to float32
                heads_state_np[k] = v.numpy()
        except ImportError:
            # Fallback: install torch in mlx env or use alternative loading
            raise ImportError(
                "torch is required to load PyTorch checkpoints. "
                "Install with: pip install torch"
            )

        medusa_heads = load_medusa_heads_from_numpy(
            state_dict=heads_state_np,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_heads=medusa_num_heads,
            num_layers=medusa_num_layers,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        print(f"Loaded {len(medusa_heads)} Medusa heads with {medusa_num_layers} layers each")

        return cls(
            base_model=base_model,
            tokenizer=tokenizer,
            medusa_heads=medusa_heads,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            medusa_num_heads=medusa_num_heads,
            medusa_num_layers=medusa_num_layers,
            lora_rank=lora_rank,
        )

    def _get_tree_buffers(
        self,
        tree_choices: Optional[List[Tuple[int, ...]]] = None,
        topk: int = 10,
    ) -> Dict[str, mx.array]:
        """Get tree attention buffers, using cache if config matches."""
        if tree_choices is None:
            tree_choices = DEFAULT_TREES.get(self.medusa_num_heads, DEFAULT_TREES[4])

        config = (tuple(tree_choices), topk)
        if self._tree_buffers_cache is None or self._tree_buffers_config != config:
            self._tree_buffers_cache = generate_tree_buffers(tree_choices, topk)
            self._tree_buffers_config = config
        return self._tree_buffers_cache

    def generate_standard(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> Tuple[str, int, float]:
        """
        Standard generation using mlx_lm (no speculative decoding).

        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (generated_text, num_tokens, time_seconds)
        """
        sampler = make_sampler(temp=temperature)

        t0 = time.perf_counter()
        response = mlx_generate(
            self.base_model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
            verbose=False,
        )
        t1 = time.perf_counter()

        # Count generated tokens
        prompt_tokens = self.tokenizer.encode(prompt)
        full_tokens = self.tokenizer.encode(prompt + response)
        gen_tokens = len(full_tokens) - len(prompt_tokens)

        return response, gen_tokens, t1 - t0

    def _get_hidden_states(
        self,
        input_ids: mx.array,
        cache: Optional[List] = None,
        tree_attn_mask: Optional[mx.array] = None,
        tree_position_offsets: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Get hidden states from the base model (before lm_head).

        The MLX gemma3_text model structure is:
        - model.model: The transformer (embed_tokens -> layers -> norm -> hidden_states)
        - model.lm_head: The final projection (hidden_states -> logits)

        Args:
            input_ids: (B, T) input token IDs
            cache: Optional KV cache list (modified in-place by MLX)
            tree_attn_mask: Optional custom attention mask for tree verification.
                           Shape: (1, 1, tree_len, cache_len + tree_len)
                           Values: True/1 = attend, False/0 = don't attend
            tree_position_offsets: Optional depth-based position offsets for tree tokens.
                                   Shape: (tree_len,) with values like [0, 1, 1, 1, 2, 2, ...]

        Returns:
            hidden_states: (B, T, hidden_size)
        """
        # Access the inner transformer model (before lm_head)
        inner_model = self.base_model.model

        if tree_attn_mask is not None:
            # Custom forward with tree attention mask and position offsets
            hidden_states = self._forward_with_tree_mask(
                input_ids, cache, tree_attn_mask, tree_position_offsets
            )
        elif cache is not None:
            hidden_states = inner_model(input_ids, cache=cache)
        else:
            hidden_states = inner_model(input_ids)

        return hidden_states

    def _forward_with_tree_mask(
        self,
        input_ids: mx.array,
        cache: List,
        tree_attn_mask: mx.array,
        tree_position_offsets: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Custom forward pass with tree attention mask for speculative decoding.

        This implements the same forward logic as Gemma3Model but allows passing
        a custom attention mask for tree verification, enabling proper tree
        attention where each position only attends to its ancestors.

        Uses depth-based RoPE positions to ensure accepted tokens have correct
        position encodings after cache compaction.

        Args:
            input_ids: (B, T) input token IDs
            cache: KV cache list (one per layer)
            tree_attn_mask: (1, 1, tree_len, cache_len + tree_len) attention mask
            tree_position_offsets: (tree_len,) depth-based offsets [0, 1, 1, 1, 2, ...]
                                   If None, uses sequential positions.

        Returns:
            hidden_states: (B, T, hidden_size)
        """
        inner_model = self.base_model.model

        # Get embeddings
        h = inner_model.embed_tokens(input_ids)
        h *= mx.array(inner_model.args.hidden_size ** 0.5, mx.bfloat16).astype(h.dtype)

        # Convert mask to MLX format (True = attend, False = don't)
        if tree_attn_mask.dtype != mx.bool_:
            mask = tree_attn_mask > 0.5
        else:
            mask = tree_attn_mask

        # Forward through all layers with the tree mask
        # Use depth-based RoPE positions if provided
        for i, (layer, c) in enumerate(zip(inner_model.layers, cache)):
            if tree_position_offsets is not None:
                # Custom RoPE application with depth-based positions
                h = self._forward_layer_with_positions(
                    layer, h, mask, c, tree_position_offsets
                )
            else:
                h = layer(h, mask=mask, cache=c)

        return inner_model.norm(h)

    def _apply_rope_with_positions(
        self,
        x: mx.array,
        positions: mx.array,
        rope_module,
    ) -> mx.array:
        """
        Apply RoPE with per-position offsets.

        Args:
            x: (B, n_heads, L, head_dim) tensor
            positions: (L,) position offsets for each token
            rope_module: The RoPE module from attention layer

        Returns:
            x with RoPE applied based on per-position offsets
        """
        # Manual RoPE implementation with per-position offsets
        # RoPE formula: (x * cos(theta)) + (rotate_half(x) * sin(theta))
        # where theta depends on position and dimension

        B, n_heads, L, head_dim = x.shape
        dims = rope_module.dims
        base = rope_module.base
        scale = rope_module.scale
        traditional = rope_module.traditional

        # Compute frequencies: base^(-2i/dims) for i in [0, dims/2)
        half_dims = dims // 2
        freq_exps = mx.arange(0, half_dims, dtype=mx.float32) / half_dims
        freqs = 1.0 / (base ** freq_exps)  # (half_dims,)

        # positions: (L,) -> (L, 1) for broadcasting with (half_dims,)
        # theta = position * freq -> (L, half_dims)
        positions = positions.astype(mx.float32) * scale
        theta = positions[:, None] * freqs[None, :]  # (L, half_dims)

        # cos and sin: (L, half_dims)
        cos_theta = mx.cos(theta)
        sin_theta = mx.sin(theta)

        # Expand for broadcasting with x: (1, 1, L, half_dims)
        cos_theta = cos_theta[None, None, :, :]
        sin_theta = sin_theta[None, None, :, :]

        # Apply rotation
        # x has shape (B, n_heads, L, head_dim)
        x_rope = x[:, :, :, :dims]
        x_pass = x[:, :, :, dims:] if dims < head_dim else None

        if traditional:
            # Traditional: rotate consecutive pairs
            x1 = x_rope[:, :, :, 0::2]
            x2 = x_rope[:, :, :, 1::2]
            rx1 = x1 * cos_theta - x2 * sin_theta
            rx2 = x1 * sin_theta + x2 * cos_theta
            # Interleave back
            x_rotated = mx.stack([rx1, rx2], axis=-1).reshape(B, n_heads, L, dims)
        else:
            # Default: rotate first half with second half
            x1 = x_rope[:, :, :, :half_dims]
            x2 = x_rope[:, :, :, half_dims:dims]
            rx1 = x1 * cos_theta - x2 * sin_theta
            rx2 = x1 * sin_theta + x2 * cos_theta
            x_rotated = mx.concatenate([rx1, rx2], axis=-1)

        if x_pass is not None:
            x_rotated = mx.concatenate([x_rotated, x_pass], axis=-1)

        return x_rotated.astype(x.dtype)

    def _forward_layer_with_positions(
        self,
        layer,
        h: mx.array,
        mask: mx.array,
        cache,
        tree_position_offsets: mx.array,
        use_kernel: bool = True,
    ) -> mx.array:
        """
        Forward through a single transformer layer with custom position offsets.

        This replicates the layer forward but uses depth-based positions for RoPE.
        Uses custom Metal kernel for RoPE when available (2x faster than Python).

        Args:
            layer: Transformer layer
            h: Hidden states (B, L, hidden_size)
            mask: Attention mask
            cache: KV cache for this layer
            tree_position_offsets: (L,) depth-based position offsets
            use_kernel: If True, use custom Metal kernel for RoPE (faster)
        """
        attn = layer.self_attn

        # Layer norm
        x = layer.input_layernorm(h)

        B, L, _ = x.shape

        # QKV projections
        queries = attn.q_proj(x)
        keys = attn.k_proj(x)
        values = attn.v_proj(x)

        # Reshape for multi-head attention
        queries = queries.reshape(B, L, attn.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # QK norms
        queries = attn.q_norm(queries)
        keys = attn.k_norm(keys)

        # Apply RoPE with depth-based positions
        # positions = base_offset + tree_position_offsets
        base_offset = cache.offset
        positions = base_offset + tree_position_offsets.astype(mx.int32)

        if use_kernel:
            # Use fused Metal kernel for Q and K (30% faster than two separate calls)
            base = attn.rope.base
            scale = attn.rope.scale
            queries, keys = apply_rope_fused_qk_kernel(queries, keys, positions, base=base, scale=scale)
        else:
            # Python fallback
            queries = self._apply_rope_with_positions(queries, positions, attn.rope)
            keys = self._apply_rope_with_positions(keys, positions, attn.rope)

        # Update cache with positioned keys/values
        keys, values = cache.update_and_fetch(keys, values)

        # Attention with mask
        from mlx_lm.models.base import scaled_dot_product_attention
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=attn.scale, mask=mask
        )

        # Reshape and project
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        r = attn.o_proj(output)

        # Post attention layernorm and residual
        from mlx_lm.models.gemma3_text import clip_residual
        out = clip_residual(h, layer.post_attention_layernorm(r))

        # MLP
        r = layer.mlp(layer.pre_feedforward_layernorm(out))
        out = clip_residual(out, layer.post_feedforward_layernorm(r))

        return out

    def _compute_logits(
        self,
        hidden_states: mx.array,
        return_medusa: bool = True,
        last_only: bool = False,
        num_active_heads: Optional[int] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Compute main logits and Medusa logits.

        Args:
            hidden_states: (B, T, hidden_size) from transformer
            return_medusa: Whether to compute Medusa logits
            last_only: If True, only compute logits for last position
            num_active_heads: If set, only use first N Medusa heads (default: all)

        Returns:
            main_logits: (B, T, vocab_size) or (B, 1, vocab_size)
            medusa_logits: (num_active_heads, B, T, vocab_size) or None
        """
        if last_only:
            hidden_states = hidden_states[:, -1:, :]

        # Get lm_head from base model
        lm_head = self.base_model.lm_head

        if not return_medusa or len(self.medusa_heads) == 0 or num_active_heads == 0:
            # Just compute main logits (no Medusa heads)
            main_logits = lm_head(hidden_states)
            return main_logits, None

        # Determine which heads to use
        active_heads = self.medusa_heads
        if num_active_heads is not None:
            active_heads = self.medusa_heads[:num_active_heads]

        # Compute Medusa logits - batch all heads together for efficiency
        # First, apply ResBlocks to get transformed hidden states for each head
        head_hiddens = [head(hidden_states) for head in active_heads]

        # Check if any head has LoRA (determines computation path)
        has_lora = any(head.has_lora for head in active_heads)

        if has_lora:
            # LoRA mode: each head returns a delta to add to base logits
            # Need separate main_logits computation
            main_logits = lm_head(hidden_states)
            medusa_logits = mx.stack([main_logits + h for h in head_hiddens], axis=0)
        else:
            # No LoRA: batch main + all head hiddens and apply lm_head once
            # Stack: [main, head0, head1, ...] -> (1 + num_heads, B, T, hidden_size)
            all_hiddens = [hidden_states] + head_hiddens
            stacked_hiddens = mx.stack(all_hiddens, axis=0)
            num_total, B, T, H = stacked_hiddens.shape

            # Reshape to ((1 + num_heads) * B * T, hidden_size) for batched matmul
            flat_hiddens = stacked_hiddens.reshape(-1, H)

            # Single batched lm_head call for main + all Medusa heads
            flat_logits = lm_head(flat_hiddens)

            # Reshape back: (1 + num_heads, B, T, vocab_size)
            all_logits = flat_logits.reshape(num_total, B, T, -1)

            # Split: main_logits is first, medusa_logits is rest
            main_logits = all_logits[0]  # (B, T, vocab_size)
            medusa_logits = all_logits[1:]  # (num_heads, B, T, vocab_size)

        return main_logits, medusa_logits

    def _generate_candidates(
        self,
        main_logits: mx.array,
        medusa_logits: mx.array,
        buffers: Dict[str, mx.array],
        topk: int = 10,
        temperature: float = 0.0,
    ) -> Tuple[mx.array, mx.array]:
        """
        Generate candidate token sequences from model predictions.

        Args:
            main_logits: (B, vocab_size) Main model logits for last position
            medusa_logits: (num_heads, B, vocab_size) Medusa logits
            buffers: Tree attention buffers
            topk: Number of top-k predictions per head
            temperature: Sampling temperature

        Returns:
            candidates: (num_candidates, max_depth) Candidate token sequences
            tree_candidates: (tree_len,) Tokens arranged in tree structure
        """
        tree_indices = buffers["tree_indices"]
        retrieve_indices = buffers["retrieve_indices"]

        # Get main model prediction
        if temperature == 0.0:
            base_token = mx.argmax(main_logits[0])
        else:
            probs = mx.softmax(main_logits[0] / temperature, axis=-1)
            base_token = mx.random.categorical(mx.log(probs + 1e-10))

        # Get required topk from tree structure
        max_tree_idx = int(mx.max(tree_indices).item())
        num_heads = medusa_logits.shape[0]
        required_topk = max(topk, (max_tree_idx + num_heads - 1) // num_heads)

        # Get top-k from each Medusa head using argsort (simpler than argpartition + sort)
        # medusa_logits: (num_heads, B, vocab_size) -> sort along vocab dimension
        sorted_indices = mx.argsort(-medusa_logits[:, 0, :], axis=-1)  # (num_heads, vocab_size)
        medusa_topk = sorted_indices[:, :required_topk]  # (num_heads, required_topk)

        # Build flat candidate array: [base_token, head0_topk, head1_topk, ...]
        flat_candidates = mx.concatenate([base_token[None], medusa_topk.reshape(-1)])

        # Map to tree structure
        tree_candidates = flat_candidates[tree_indices]

        # Extract candidate paths
        safe_indices = mx.clip(retrieve_indices, 0, None)
        candidates = tree_candidates[safe_indices]
        # Zero out invalid positions
        candidates = candidates * (retrieve_indices >= 0).astype(mx.int32)

        return candidates, tree_candidates

    def _evaluate_candidates_greedy(
        self,
        tree_logits: mx.array,
        candidates: mx.array,
        retrieve_indices: mx.array,
        valid_mask: mx.array,
    ) -> Tuple[int, int]:
        """
        Evaluate candidates with greedy acceptance.

        Args:
            tree_logits: (tree_len, vocab_size) Logits at tree positions
            candidates: (num_candidates, max_depth) Candidate sequences
            retrieve_indices: (num_candidates, max_depth) Indices into tree
            valid_mask: (num_candidates, max_depth) Valid positions

        Returns:
            best_candidate: Index of best candidate path
            accept_length: Number of tokens to accept
        """
        safe_indices = mx.clip(retrieve_indices, 0, None)

        # Compute argmax for all tree positions
        tree_predictions = mx.argmax(tree_logits, axis=-1)

        # Get predictions at needed positions
        candidate_predictions = tree_predictions[safe_indices]

        # Check matches: prediction at position j should equal candidate at j+1
        matches = (candidates[:, 1:] == candidate_predictions[:, :-1])
        matches = matches & valid_mask[:, 1:]

        # Find longest matching prefix using cumulative product
        cumulative_matches = mx.cumprod(matches.astype(mx.int32), axis=1)
        accept_lengths = mx.sum(cumulative_matches, axis=1)

        best_candidate = int(mx.argmax(accept_lengths).item())
        accept_length = int(accept_lengths[best_candidate].item())

        return best_candidate, accept_length

    def generate_mtp(
        self,
        input_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        topk: int = 10,
        tree_choices: Optional[List[Tuple[int, ...]]] = None,
        eos_token_id: Optional[int] = None,
        stop_token_ids: Optional[set] = None,
        use_tree_attention: bool = False,
        use_compiled: bool = True,
        num_active_heads: Optional[int] = None,
        use_small_tree: bool = False,
    ) -> Tuple[List[int], MTPStats]:
        """
        Generate tokens using MTP (Multi-Token Prediction) speculative decoding.

        This uses Medusa heads to speculate multiple tokens ahead, then verifies
        them in parallel, providing speedup over standard generation.

        Args:
            input_ids: Initial prompt as list of token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            topk: Number of top-k predictions per Medusa head
            tree_choices: Tree structure for speculation (None = default)
            eos_token_id: EOS token ID for stopping (deprecated, use stop_token_ids)
            stop_token_ids: Set of token IDs that should stop generation
            use_tree_attention: If True, use tree attention masking and cache compaction
                to avoid re-processing. If False (default), use simpler re-processing
                approach.
            use_compiled: If True (default), use mx.compile for candidate generation
                and evaluation to reduce Python overhead.
            num_active_heads: If set, only use first N Medusa heads (default: all)
            use_small_tree: If True, use a smaller tree (~12 positions) for lower overhead

        Returns:
            Tuple of (generated_token_ids, stats)
        """
        # Build stop token set from both parameters for backwards compatibility
        stop_tokens = set()
        if stop_token_ids is not None:
            stop_tokens.update(stop_token_ids)
        if eos_token_id is not None:
            stop_tokens.add(eos_token_id)

        if self.medusa_num_heads == 0:
            raise ValueError("Model has no Medusa heads - cannot use MTP generation")

        # Check for sliding window attention - tree attention is incompatible
        inner_model = self.base_model.model if hasattr(self.base_model, 'model') else self.base_model
        has_sliding_window = False
        if hasattr(inner_model, 'args'):
            sliding_window = getattr(inner_model.args, 'sliding_window', None)
            if sliding_window is not None and sliding_window > 0:
                has_sliding_window = True

        if use_tree_attention and has_sliding_window:
            # Fall back to re-processing mode for models with sliding window attention
            # because different layers have different cache sizes
            use_tree_attention = False

        # Determine number of active heads
        active_heads = num_active_heads if num_active_heads is not None else self.medusa_num_heads

        # Get tree buffers - use small tree if requested
        if tree_choices is None:
            if use_small_tree:
                tree_choices = SMALL_TREES.get(active_heads, SMALL_TREES[2])
            else:
                tree_choices = DEFAULT_TREES.get(active_heads, DEFAULT_TREES[4])

        buffers = self._get_tree_buffers(tree_choices, topk)
        retrieve_indices = buffers["retrieve_indices"]
        tree_attn_mask = buffers["tree_attn_mask"]  # (1, 1, tree_len, tree_len)
        tree_position_offsets = buffers["tree_position_ids"]  # Depth-based offsets
        tree_len = tree_attn_mask.shape[-1]
        max_speculation = retrieve_indices.shape[1] - 1

        # Initialize stats
        stats = MTPStats(
            tokens_generated=0,
            forward_passes=0,
            total_proposed=0,
            total_accepted=0,
        )

        # Create KV cache
        cache = self.base_model.make_cache()

        output_tokens = list(input_ids)
        num_generated = 0

        # ===== PREFILL PHASE =====
        # Process entire prompt to populate KV cache
        input_array = mx.array([input_ids], dtype=mx.int32)
        hidden_states = self._get_hidden_states(input_array, cache=cache)
        main_logits, medusa_logits = self._compute_logits(
            hidden_states, return_medusa=True, last_only=True, num_active_heads=active_heads
        )
        mx.eval(main_logits, medusa_logits)  # Force computation

        # Squeeze to (B, vocab) and (num_heads, B, vocab)
        last_main = main_logits[:, 0, :]
        last_medusa = medusa_logits[:, :, 0, :]

        stats.forward_passes += 1  # Count prefill

        # ===== DECODE PHASE =====
        # Extract buffers for compiled functions
        tree_indices = buffers["tree_indices"]

        while num_generated < max_new_tokens:
            # Get cache length from layer 0 (used for mask building)
            if cache[0].keys is not None:
                mx.eval(cache[0].keys)
                cache_len = cache[0].keys.shape[2]
            else:
                cache_len = 0

            # Generate candidate tree from current predictions
            if use_compiled and temperature == 0.0:
                # Use compiled version (greedy only)
                candidates, tree_candidates = _compiled_generate_candidates(
                    last_main, last_medusa, tree_indices, retrieve_indices, topk
                )
            else:
                # Use regular version (supports temperature)
                candidates, tree_candidates = self._generate_candidates(
                    last_main, last_medusa, buffers, topk, temperature
                )

            if use_tree_attention:
                # TREE ATTENTION MODE: Use custom attention mask and depth-based RoPE
                # Build full attention mask: [attend to cache | tree attention]
                full_attn_mask = build_tree_attention_mask_mlx(tree_attn_mask, cache_len)

                # Forward pass with tree candidates using tree attention mask and depth-based positions
                tree_input = tree_candidates[None, :]  # (1, tree_len)
                tree_hidden_states = self._get_hidden_states(
                    tree_input, cache=cache, tree_attn_mask=full_attn_mask,
                    tree_position_offsets=tree_position_offsets
                )
                # OPTIMIZATION: Only compute main logits for verification (not medusa)
                # We'll compute medusa logits only for the last accepted position later
                tree_logits, _ = self._compute_logits(
                    tree_hidden_states, return_medusa=False
                )
                tree_logits = tree_logits[0]  # (tree_len, vocab)
                mx.eval(tree_logits)  # Force computation
                # Keep hidden states for later medusa computation
                tree_hidden_states_for_medusa = tree_hidden_states
            else:
                # RE-PROCESSING MODE: Standard causal forward, then trim and re-process
                # Only compute main logits for verification
                tree_input = tree_candidates[None, :]
                tree_hidden_states = self._get_hidden_states(tree_input, cache=cache)
                tree_logits, _ = self._compute_logits(
                    tree_hidden_states, return_medusa=False
                )
                tree_logits = tree_logits[0]
                mx.eval(tree_logits)
                tree_hidden_states_for_medusa = tree_hidden_states

            stats.forward_passes += 1
            stats.total_proposed += max_speculation

            # Evaluate candidates
            if use_compiled:
                # Use compiled version (returns arrays)
                best_candidate_arr, accept_length_arr = _compiled_evaluate_candidates(
                    tree_logits, candidates, retrieve_indices
                )
                mx.eval(best_candidate_arr, accept_length_arr)
                best_candidate = int(best_candidate_arr.item())
                accept_length = int(accept_length_arr.item())
            else:
                valid_mask = retrieve_indices >= 0
                best_candidate, accept_length = self._evaluate_candidates_greedy(
                    tree_logits, candidates, retrieve_indices, valid_mask
                )

            # Get accepted token indices and tree positions
            accepted_tokens = [int(candidates[best_candidate, i].item())
                              for i in range(accept_length + 1)]
            num_accepted = len(accepted_tokens)
            stats.total_accepted += num_accepted

            if use_tree_attention:
                # Get accepted tree positions for cache compaction
                accepted_tree_positions = [int(retrieve_indices[best_candidate, i].item())
                                           for i in range(num_accepted)]

                # Compact cache: keep only accepted tree positions
                self._compact_cache(cache, accepted_tree_positions, cache_len, tree_len)
            else:
                # Trim cache: remove ALL tree tokens
                for layer_cache in cache:
                    layer_cache.trim(tree_len)

            # Add accepted tokens to output and check for EOS
            should_stop = False
            tokens_added = 0
            for token in accepted_tokens:
                if stop_tokens and token in stop_tokens:
                    should_stop = True
                    break

                output_tokens.append(token)
                num_generated += 1
                tokens_added += 1

                if num_generated >= max_new_tokens:
                    should_stop = True
                    break

            if should_stop:
                break

            if use_tree_attention:
                # TREE ATTENTION: Compute medusa logits only for the last accepted position
                last_accepted_tree_idx = accepted_tree_positions[-1]

                # Extract main logits for last accepted position
                last_main = tree_logits[last_accepted_tree_idx:last_accepted_tree_idx+1, :]  # (1, vocab)
                last_main = last_main[None, :, :]  # (1, 1, vocab)

                # Extract hidden states for just the last accepted position and compute medusa logits
                last_hidden = tree_hidden_states_for_medusa[:, last_accepted_tree_idx:last_accepted_tree_idx+1, :]
                _, last_medusa = self._compute_logits(
                    last_hidden, return_medusa=True, num_active_heads=active_heads
                )
                mx.eval(last_medusa)
            else:
                # RE-PROCESSING: Need to re-forward accepted tokens to update cache
                accepted_array = mx.array([accepted_tokens], dtype=mx.int32)
                hidden_states = self._get_hidden_states(accepted_array, cache=cache)
                main_logits, medusa_logits = self._compute_logits(
                    hidden_states, return_medusa=True, last_only=True, num_active_heads=active_heads
                )
                mx.eval(main_logits, medusa_logits)
                last_main = main_logits
                last_medusa = medusa_logits

            # Reshape for next iteration
            last_main = last_main[:, 0, :]  # (1, vocab)
            last_medusa = last_medusa[:, :, 0, :]  # (num_heads, 1, vocab)

        stats.tokens_generated = num_generated
        return output_tokens, stats

    def generate_simple_speculation(
        self,
        input_ids: List[int],
        max_new_tokens: int = 100,
        eos_token_id: Optional[int] = None,
        stop_token_ids: Optional[set] = None,
        collect_entropy: bool = False,
        entropy_threshold: Optional[float] = None,
    ) -> Tuple[List[int], MTPStats]:
        """
        Simple 2-token speculation: predict main + medusa[0], verify both, accept 1 or 2.

        This is a simpler and often faster alternative to full tree-based MTP,
        especially on memory-bandwidth-limited hardware where the marginal cost
        of processing extra tokens is very low.

        Args:
            input_ids: Initial prompt as list of token IDs
            max_new_tokens: Maximum new tokens to generate
            eos_token_id: EOS token ID for stopping (deprecated, use stop_token_ids)
            stop_token_ids: Set of token IDs that should stop generation
            collect_entropy: If True, collect per-iteration entropy data for analysis
            entropy_threshold: If set, skip speculation when entropy exceeds this value

        Returns:
            Tuple of (generated_token_ids, stats)
        """
        # Build stop token set from both parameters for backwards compatibility
        stop_tokens = set()
        if stop_token_ids is not None:
            stop_tokens.update(stop_token_ids)
        if eos_token_id is not None:
            stop_tokens.add(eos_token_id)

        if self.medusa_num_heads == 0:
            raise ValueError("Model has no Medusa heads - cannot use speculation")

        cache = self.base_model.make_cache()
        entropy_log: List[EntropyRecord] = [] if collect_entropy else None

        # Prefill
        input_array = mx.array([input_ids], dtype=mx.int32)
        h = self._get_hidden_states(input_array, cache=cache)
        main_logits, medusa_logits = self._compute_logits(
            h, return_medusa=True, last_only=True, num_active_heads=1
        )
        mx.eval(main_logits, medusa_logits)

        output_tokens = list(input_ids)
        num_generated = 0
        num_iterations = 0
        total_accepted = 0

        num_skipped = 0  # Track how many times we skipped speculation due to high entropy

        while num_generated < max_new_tokens:
            # Compute entropy before speculation (if collecting or gating)
            if collect_entropy or entropy_threshold is not None:
                entropy = compute_entropy(main_logits[0, 0])

            # Get base token prediction
            token1 = int(mx.argmax(main_logits[0, 0]).item())

            # Check stop tokens
            if stop_tokens and token1 in stop_tokens:
                break

            # Entropy gating: skip speculation if entropy is too high
            if entropy_threshold is not None and entropy > entropy_threshold:
                # High entropy - just forward the single base token (no speculation)
                num_skipped += 1
                if collect_entropy:
                    entropy_log.append(EntropyRecord(entropy=entropy, accept_length=1))

                spec_input = mx.array([[token1]], dtype=mx.int32)
                h = self._get_hidden_states(spec_input, cache=cache)
                main_logits, medusa_logits = self._compute_logits(
                    h, return_medusa=True, last_only=True, num_active_heads=1
                )
                mx.eval(main_logits, medusa_logits)

                output_tokens.append(token1)
                num_generated += 1
                total_accepted += 1
                num_iterations += 1
                continue

            # Get medusa prediction for speculation
            token2 = int(mx.argmax(medusa_logits[0, 0, 0]).item())

            # Forward both tokens with standard causal attention
            spec_input = mx.array([[token1, token2]], dtype=mx.int32)
            h = self._get_hidden_states(spec_input, cache=cache)
            verify_logits, new_medusa = self._compute_logits(
                h, return_medusa=True, last_only=False, num_active_heads=1
            )
            mx.eval(verify_logits, new_medusa)

            # Check if token2 was correct
            # verify_logits[0, 0] predicts what should come after token1
            verified_token2 = int(mx.argmax(verify_logits[0, 0]).item())

            if verified_token2 == token2:
                # Log entropy with accept_length=2
                if collect_entropy:
                    entropy_log.append(EntropyRecord(entropy=entropy, accept_length=2))

                # Check stop tokens before adding
                if stop_tokens and token2 in stop_tokens:
                    # token2 is stop - add both and stop
                    output_tokens.extend([token1, token2])
                    num_generated += 2
                    total_accepted += 2
                    break

                # Both accepted, no stop token
                output_tokens.extend([token1, token2])
                num_generated += 2
                total_accepted += 2

                # Use logits from position 1 for next iteration
                main_logits = verify_logits[:, 1:2, :]
                medusa_logits = new_medusa[:, :, 1:2, :]
            else:
                # Log entropy with accept_length=1
                if collect_entropy:
                    entropy_log.append(EntropyRecord(entropy=entropy, accept_length=1))

                # Only token1 accepted - trim the second token from cache
                trim_cache(cache, 1)

                output_tokens.append(token1)
                num_generated += 1
                total_accepted += 1

                # Check if the model wanted to generate a stop token after token1
                if stop_tokens and verified_token2 in stop_tokens:
                    break

                # Use logits from position 0 for next iteration
                main_logits = verify_logits[:, 0:1, :]
                medusa_logits = new_medusa[:, :, 0:1, :]

            num_iterations += 1

        stats = MTPStats(
            tokens_generated=num_generated,
            forward_passes=num_iterations + 1,  # +1 for prefill
            total_proposed=(num_iterations - num_skipped) * 2 + num_skipped,  # 2 when speculating, 1 when skipped
            total_accepted=total_accepted,
            entropy_log=entropy_log,
            speculation_skipped=num_skipped,
        )
        return output_tokens, stats

    def generate_simple_speculation_3tok(
        self,
        input_ids: List[int],
        max_new_tokens: int = 100,
        eos_token_id: Optional[int] = None,
        stop_token_ids: Optional[set] = None,
    ) -> Tuple[List[int], MTPStats]:
        """
        Simple 3-token speculation using 1 head: predict main + top-2 from medusa[0].

        Uses only 1 Medusa head but takes its top-2 predictions to speculate
        positions +1 and +2. This tests whether a single head can predict
        multiple future positions.

        Token prediction:
        - token1: argmax of main model logits (position +0)
        - token2: top-1 from medusa head 0 (position +1)
        - token3: top-2 from medusa head 0 (position +2, using same head)

        Args:
            input_ids: Initial prompt as list of token IDs
            max_new_tokens: Maximum new tokens to generate
            eos_token_id: EOS token ID for stopping (deprecated, use stop_token_ids)
            stop_token_ids: Set of token IDs that should stop generation

        Returns:
            Tuple of (generated_token_ids, stats)
        """
        # Build stop token set from both parameters for backwards compatibility
        stop_tokens = set()
        if stop_token_ids is not None:
            stop_tokens.update(stop_token_ids)
        if eos_token_id is not None:
            stop_tokens.add(eos_token_id)

        if self.medusa_num_heads == 0:
            raise ValueError("Model has no Medusa heads - cannot use speculation")

        cache = self.base_model.make_cache()

        # Prefill - only need 1 head
        input_array = mx.array([input_ids], dtype=mx.int32)
        h = self._get_hidden_states(input_array, cache=cache)
        main_logits, medusa_logits = self._compute_logits(
            h, return_medusa=True, last_only=True, num_active_heads=1
        )
        mx.eval(main_logits, medusa_logits)

        output_tokens = list(input_ids)
        num_generated = 0
        num_iterations = 0
        total_accepted = 0
        total_proposed = 0

        while num_generated < max_new_tokens:
            # Get predictions: main model + top-2 from medusa head 0
            token1 = int(mx.argmax(main_logits[0, 0]).item())

            # Get top-2 from medusa head 0 using compiled two-argmax (faster than argsort)
            medusa_head0_logits = medusa_logits[0, 0, 0]  # (vocab_size,)
            token2_arr, token3_arr = _compiled_get_top2(medusa_head0_logits)
            mx.eval(token2_arr, token3_arr)
            token2 = int(token2_arr.item())
            token3 = int(token3_arr.item())

            # Check stop tokens
            if stop_tokens and token1 in stop_tokens:
                break

            # Forward all 3 tokens with standard causal attention
            spec_input = mx.array([[token1, token2, token3]], dtype=mx.int32)
            h = self._get_hidden_states(spec_input, cache=cache)
            verify_logits, new_medusa = self._compute_logits(
                h, return_medusa=True, last_only=False, num_active_heads=1
            )
            mx.eval(verify_logits, new_medusa)

            # Verify tokens
            # verify_logits[0, 0] predicts what should come after token1 -> should be token2
            # verify_logits[0, 1] predicts what should come after token2 -> should be token3
            verified_token2 = int(mx.argmax(verify_logits[0, 0]).item())
            verified_token3 = int(mx.argmax(verify_logits[0, 1]).item())

            num_iterations += 1
            total_proposed += 3

            if verified_token2 == token2 and verified_token3 == token3:
                # Check stop tokens before adding - don't add tokens after stop
                if stop_tokens and token2 in stop_tokens:
                    # token2 is stop - only add token1 and token2, discard token3
                    output_tokens.extend([token1, token2])
                    num_generated += 2
                    total_accepted += 2
                    trim_cache(cache, 1)  # Remove token3 from cache
                    break

                if stop_tokens and token3 in stop_tokens:
                    # token3 is stop - add all 3 and stop
                    output_tokens.extend([token1, token2, token3])
                    num_generated += 3
                    total_accepted += 3
                    break

                # All 3 accepted, no stop token
                output_tokens.extend([token1, token2, token3])
                num_generated += 3
                total_accepted += 3

                # Use logits from position 2 for next iteration
                main_logits = verify_logits[:, 2:3, :]
                medusa_logits = new_medusa[:, :, 2:3, :]

            elif verified_token2 == token2:
                # token1 and token2 accepted, token3 rejected - trim 1 from cache
                trim_cache(cache, 1)

                output_tokens.extend([token1, token2])
                num_generated += 2
                total_accepted += 2

                # Check stop - either token2 is stop, or model wants to generate stop after token2
                if stop_tokens and (token2 in stop_tokens or verified_token3 in stop_tokens):
                    break

                # Use logits from position 1 for next iteration
                main_logits = verify_logits[:, 1:2, :]
                medusa_logits = new_medusa[:, :, 1:2, :]

            else:
                # Only token1 accepted - trim 2 from cache
                trim_cache(cache, 2)

                output_tokens.append(token1)
                num_generated += 1
                total_accepted += 1

                # Check if the model wanted to generate a stop token after token1
                # (verified_token2 is what model actually predicts after token1)
                if stop_tokens and verified_token2 in stop_tokens:
                    break

                # Use logits from position 0 for next iteration
                main_logits = verify_logits[:, 0:1, :]
                medusa_logits = new_medusa[:, :, 0:1, :]

        stats = MTPStats(
            tokens_generated=num_generated,
            forward_passes=num_iterations + 1,  # +1 for prefill
            total_proposed=total_proposed,
            total_accepted=total_accepted,
        )
        return output_tokens, stats

    def generate_depth2_speculation(
        self,
        input_ids: List[int],
        max_new_tokens: int = 100,
        eos_token_id: Optional[int] = None,
        stop_token_ids: Optional[set] = None,
        collect_entropy: bool = False,
        entropy_threshold: Optional[float] = None,
    ) -> Tuple[List[int], MTPStats]:
        """
        Optimized 3-token speculation using 2 Medusa heads.

        Uses head0 for position +1 and head1 for position +2.
        This is more accurate than generate_simple_speculation_3tok which
        uses only head0 for both positions.

        Equivalent to tree structure [(0,), (0, 0)] but without tree overhead.

        Args:
            input_ids: Initial prompt as list of token IDs
            max_new_tokens: Maximum new tokens to generate
            eos_token_id: EOS token ID for stopping (deprecated, use stop_token_ids)
            stop_token_ids: Set of token IDs that should stop generation
            collect_entropy: If True, collect per-iteration entropy data for analysis
            entropy_threshold: If set, skip speculation when entropy exceeds this value

        Returns:
            Tuple of (generated_token_ids, stats)
        """
        # Build stop token set
        stop_tokens = set()
        if stop_token_ids is not None:
            stop_tokens.update(stop_token_ids)
        if eos_token_id is not None:
            stop_tokens.add(eos_token_id)

        if self.medusa_num_heads < 2:
            raise ValueError("Need at least 2 Medusa heads for depth-2 speculation")

        cache = self.base_model.make_cache()
        entropy_log: List[EntropyRecord] = [] if collect_entropy else None

        # Prefill - need 2 heads
        input_array = mx.array([input_ids], dtype=mx.int32)
        h = self._get_hidden_states(input_array, cache=cache)
        main_logits, medusa_logits = self._compute_logits(
            h, return_medusa=True, last_only=True, num_active_heads=2
        )
        mx.eval(main_logits, medusa_logits)

        output_tokens = list(input_ids)
        num_generated = 0
        num_iterations = 0
        total_accepted = 0
        total_proposed = 0
        num_skipped = 0  # Track how many times we skipped speculation due to high entropy

        while num_generated < max_new_tokens:
            # Compute entropy before speculation (if collecting or gating)
            if collect_entropy or entropy_threshold is not None:
                entropy = compute_entropy(main_logits[0, 0])

            # Get base token prediction
            token1 = int(mx.argmax(main_logits[0, 0]).item())

            # Check stop tokens
            if stop_tokens and token1 in stop_tokens:
                break

            # Entropy gating: skip speculation if entropy is too high
            if entropy_threshold is not None and entropy > entropy_threshold:
                # High entropy - just forward the single base token (no speculation)
                num_skipped += 1
                total_proposed += 1
                if collect_entropy:
                    entropy_log.append(EntropyRecord(entropy=entropy, accept_length=1))

                spec_input = mx.array([[token1]], dtype=mx.int32)
                h = self._get_hidden_states(spec_input, cache=cache)
                main_logits, medusa_logits = self._compute_logits(
                    h, return_medusa=True, last_only=True, num_active_heads=2
                )
                mx.eval(main_logits, medusa_logits)

                output_tokens.append(token1)
                num_generated += 1
                total_accepted += 1
                num_iterations += 1
                continue

            # Get medusa predictions for speculation
            token2 = int(mx.argmax(medusa_logits[0, 0, 0]).item())  # head0
            token3 = int(mx.argmax(medusa_logits[1, 0, 0]).item())  # head1

            # Forward all 3 tokens
            spec_input = mx.array([[token1, token2, token3]], dtype=mx.int32)
            h = self._get_hidden_states(spec_input, cache=cache)
            verify_logits, new_medusa = self._compute_logits(
                h, return_medusa=True, last_only=False, num_active_heads=2
            )
            mx.eval(verify_logits, new_medusa)

            # Verify:
            # verify_logits[0, 0] predicts what should come after token1 -> should be token2
            # verify_logits[0, 1] predicts what should come after token2 -> should be token3
            verified_token2 = int(mx.argmax(verify_logits[0, 0]).item())
            verified_token3 = int(mx.argmax(verify_logits[0, 1]).item())

            num_iterations += 1
            total_proposed += 3

            if verified_token2 == token2 and verified_token3 == token3:
                # Log entropy with accept_length=3
                if collect_entropy:
                    entropy_log.append(EntropyRecord(entropy=entropy, accept_length=3))

                # Check stop tokens before adding
                if stop_tokens and token2 in stop_tokens:
                    output_tokens.extend([token1, token2])
                    num_generated += 2
                    total_accepted += 2
                    trim_cache(cache, 1)
                    break

                if stop_tokens and token3 in stop_tokens:
                    output_tokens.extend([token1, token2, token3])
                    num_generated += 3
                    total_accepted += 3
                    break

                # All 3 accepted
                output_tokens.extend([token1, token2, token3])
                num_generated += 3
                total_accepted += 3

                # Use logits from position 2 for next iteration
                main_logits = verify_logits[:, 2:3, :]
                medusa_logits = new_medusa[:, :, 2:3, :]

            elif verified_token2 == token2:
                # Log entropy with accept_length=2
                if collect_entropy:
                    entropy_log.append(EntropyRecord(entropy=entropy, accept_length=2))

                # token1 and token2 accepted, token3 rejected
                trim_cache(cache, 1)

                output_tokens.extend([token1, token2])
                num_generated += 2
                total_accepted += 2

                if stop_tokens and (token2 in stop_tokens or verified_token3 in stop_tokens):
                    break

                # Use logits from position 1
                main_logits = verify_logits[:, 1:2, :]
                medusa_logits = new_medusa[:, :, 1:2, :]

            else:
                # Log entropy with accept_length=1
                if collect_entropy:
                    entropy_log.append(EntropyRecord(entropy=entropy, accept_length=1))

                # Only token1 accepted
                trim_cache(cache, 2)

                output_tokens.append(token1)
                num_generated += 1
                total_accepted += 1

                if stop_tokens and verified_token2 in stop_tokens:
                    break

                # Use logits from position 0
                main_logits = verify_logits[:, 0:1, :]
                medusa_logits = new_medusa[:, :, 0:1, :]

        stats = MTPStats(
            tokens_generated=num_generated,
            forward_passes=num_iterations + 1,
            total_proposed=total_proposed,
            total_accepted=total_accepted,
            entropy_log=entropy_log,
            speculation_skipped=num_skipped,
        )
        return output_tokens, stats

    def _compact_cache(
        self,
        cache: List,
        accepted_tree_positions: List[int],
        cache_len: int,
        tree_len: int,
    ) -> None:
        """
        Compact KV cache after tree verification to keep only accepted path.

        Uses concatenation to build the compacted cache (more reliable than gather
        with MLX's lazy evaluation).

        Note: Different layers may have different cache lengths (e.g., global attention
        layers in Gemma). We use each layer's actual size to find where tree tokens are.

        Args:
            cache: List of layer caches
            accepted_tree_positions: Tree indices of accepted tokens (e.g., [0, 1, 11])
            cache_len: Current cache length before tree tokens (for layer 0)
            tree_len: Number of tree tokens added during verification
        """
        num_accepted = len(accepted_tree_positions)

        for layer_cache in cache:
            keys = layer_cache.keys
            values = layer_cache.values

            if keys is None:
                continue

            # Get this layer's actual cache length (may differ from cache_len for global attention layers)
            layer_total_len = keys.shape[2]
            layer_cache_len = layer_total_len - tree_len  # Tree tokens were appended at the end

            # Extract prefix (original cache before tree tokens)
            prefix_keys = keys[:, :, :layer_cache_len, :]
            prefix_values = values[:, :, :layer_cache_len, :]

            # Extract accepted tree positions and concatenate
            layer_final_len = layer_cache_len + num_accepted
            if num_accepted > 0:
                accepted_keys = mx.concatenate([
                    keys[:, :, layer_cache_len + pos:layer_cache_len + pos + 1, :]
                    for pos in accepted_tree_positions
                ], axis=2)
                accepted_values = mx.concatenate([
                    values[:, :, layer_cache_len + pos:layer_cache_len + pos + 1, :]
                    for pos in accepted_tree_positions
                ], axis=2)
                layer_cache.keys = mx.concatenate([prefix_keys, accepted_keys], axis=2)
                layer_cache.values = mx.concatenate([prefix_values, accepted_values], axis=2)
            else:
                layer_cache.keys = prefix_keys
                layer_cache.values = prefix_values

            # Update cache internal state (use layer-specific length)
            layer_cache.offset = layer_final_len
            # For RotatingKVCache, also update _idx
            if hasattr(layer_cache, '_idx'):
                layer_cache._idx = layer_final_len

        # Force evaluation of ALL layers to ensure cache state is fully updated
        for layer_cache in cache:
            if layer_cache.keys is not None:
                mx.eval(layer_cache.keys, layer_cache.values)
