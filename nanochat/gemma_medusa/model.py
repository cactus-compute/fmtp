"""
Model wrapper for Gemma 3 models.

Adapts HuggingFace Gemma model to match nanochat's model interface.
Includes Medusa MTP support with batched LM head for maximum speed.

Tree attention generation:
- Supports speculative decoding with tree-structured candidate verification
- forward_mtp: Single forward pass with tree attention for candidate verification
- generate_mtp: Full generation loop with Medusa speculation

Memory optimization:
- use_chunked_loss: Critical for fitting larger batches - processes loss in chunks
- Gradient checkpointing: Optional, trades compute for memory
"""

from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass
import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig

from .config import GemmaConfigWrapper, GemmaMedusaConfig
from .heads import MedusaLoRAHead, MedusaLoRAHeadWithMixer, MedusaHeadAttention, MedusaResBlock


# Cache for loaded optimal tree choices (keyed by checkpoint path or num_heads)
_OPTIMAL_TREE_CACHE: Dict[str, List[Tuple[int, ...]]] = {}


# =============================================================================
# Pre-defined tree constants for manual override
# =============================================================================

# Default trees by head count (original heuristic structure)
# These can be passed to tree_choices parameter for manual override
DEFAULT_TREES: Dict[int, List[Tuple[int, ...]]] = {  # type: ignore[assignment]
    1: [(i,) for i in range(10)],  # 10 depth-1 candidates
    2: (
        [(i,) for i in range(10)] +  # 10 depth-1
        [(i, j) for i in range(5) for j in range(5)]  # 25 depth-2
    ),  # Total: 35 nodes
    3: (
        [(i,) for i in range(10)] +  # 10 depth-1
        [(i, j) for i in range(5) for j in range(5)] +  # 25 depth-2
        [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]  # 27 depth-3
    ),  # Total: 62 nodes
    4: (
        [(i,) for i in range(10)] +  # 10 depth-1
        [(i, j) for i in range(5) for j in range(5)] +  # 25 depth-2
        [(i, j, k) for i in range(3) for j in range(3) for k in range(3)] +  # 27 depth-3
        [(i, j, k, m) for i in range(2) for j in range(2) for k in range(2) for m in range(2)]  # 16 depth-4
    ),  # Total: 78 nodes
}

# Sparse trees for fast testing (minimal candidates per depth)
SPARSE_TREES: Dict[int, List[Tuple[int, ...]]] = {  # type: ignore[assignment]
    1: [(0,), (1,), (2,)],  # 3 nodes
    2: [(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (1, 1)],  # 7 nodes
    3: (
        [(0,), (1,), (2,)] +  # 3 depth-1
        [(0, 0), (0, 1), (1, 0), (1, 1)] +  # 4 depth-2
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)]  # 4 depth-3
    ),  # Total: 11 nodes
    4: (
        [(0,), (1,), (2,)] +  # 3 depth-1
        [(0, 0), (0, 1), (1, 0), (1, 1)] +  # 4 depth-2
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)] +  # 4 depth-3
        [(0, 0, 0, 0)]  # 1 depth-4
    ),  # Total: 12 nodes
}


def _get_node_expectation(
    accuracies: Dict[int, Dict[int, float]],
    node: Tuple[int, ...],
    topk: int
) -> float:
    """
    Compute expected acceptance probability for a node path.

    Node is a tuple like (0,), (0, 1), (2, 0, 3) representing:
    - First element: which top-k prediction from head 0 (depth 0)
    - Second element: which top-k prediction from head 1 (depth 1)
    - etc.

    Expected acceptance = product of marginal probabilities P(rank k is correct).
    For rank k, this is Recall@k - Recall@(k-1), i.e. the probability that
    the correct token is exactly at rank k (not just somewhere in top-k).
    """
    expectation = 1.0
    for depth, rank in enumerate(node):
        # rank is 0-indexed, so rank=0 means top-1, rank=1 means top-2, etc.
        k = min(rank + 1, topk)
        recall_at_k = accuracies[depth].get(k, 0.0)
        # Marginal probability: P(correct is exactly at rank k)
        # = Recall@k - Recall@(k-1)
        if rank == 0:
            marginal_prob = recall_at_k
        else:
            recall_at_k_minus_1 = accuracies[depth].get(k - 1, 0.0)
            marginal_prob = recall_at_k - recall_at_k_minus_1
        expectation *= marginal_prob
    return expectation


def _explore_tree_greedy(
    accuracies: Dict[int, Dict[int, float]],
    max_depth: int,
    max_child: List[int],
    num_iterations: int,
    topk: int,
) -> List[Tuple[int, ...]]:
    """
    Greedy tree exploration algorithm from Medusa paper.

    Args:
        accuracies: head_idx -> k -> recall rate
        max_depth: maximum tree depth (= num_heads)
        max_child: max children per depth level
        num_iterations: number of nodes to add (tree_size - 1)
        topk: max k value to consider

    Returns:
        List of accepted node tuples representing the tree
    """
    explored_nodes = {}
    accept_nodes = [tuple([0])]  # Start with root: top-1 from head 0
    explored_nodes[tuple([0])] = _get_node_expectation(accuracies, (0,), topk)

    for _ in range(num_iterations):
        # Find all neighbor nodes
        neighbors = []
        for node in accept_nodes:
            # Option 1: Increment last element (try next top-k at same depth)
            if node[-1] < max_child[len(node) - 1] - 1:
                neighbor = list(node)
                neighbor[-1] = neighbor[-1] + 1
                neighbors.append(tuple(neighbor))

            # Option 2: Extend to next depth (add child from next head)
            if len(node) < max_depth:
                neighbor = list(node)
                neighbor.append(0)
                neighbors.append(tuple(neighbor))

        # Find best neighbor not already accepted
        best_neighbor = None
        best_expectation = 0

        for neighbor in neighbors:
            if neighbor in accept_nodes:
                continue

            if neighbor in explored_nodes:
                expectation = explored_nodes[neighbor]
            else:
                expectation = _get_node_expectation(accuracies, neighbor, topk)
                explored_nodes[neighbor] = expectation

            if expectation > best_expectation:
                best_neighbor = neighbor
                best_expectation = expectation

        if best_neighbor is None:
            break

        accept_nodes.append(best_neighbor)

    # Sort by length (depth) then by values
    return sorted(accept_nodes, key=lambda x: (len(x), x))


def generate_optimal_tree_from_head_acc(
    head_acc_path: str,
    num_heads: int,
    tree_size: int = 79,
    topk: int = 64,
) -> Optional[List[Tuple[int, ...]]]:
    """
    Generate optimal tree choices from a head_acc.json file.

    Args:
        head_acc_path: Path to head_acc.json file from checkpoint
        num_heads: Number of Medusa heads to use
        tree_size: Target tree size (default 79)
        topk: Max top-k to consider per head (default 64)

    Returns:
        List of tree node tuples, or None if file doesn't exist/is invalid
    """
    if not os.path.exists(head_acc_path):
        return None

    try:
        with open(head_acc_path) as f:
            data = json.load(f)

        recall = data.get("recall", {})

        # Convert recall data to format expected by tree generation
        # recall format: {"head_0": {"1": 0.65, "2": 0.72, ...}, ...}
        accuracies: Dict[int, Dict[int, float]] = {}
        for h in range(num_heads):
            head_key = f"head_{h}"
            if head_key in recall:
                accuracies[h] = {int(k): v for k, v in recall[head_key].items()}
            else:
                accuracies[h] = {}

        # Generate tree using greedy algorithm
        max_child = [topk] * num_heads
        tree_choices = _explore_tree_greedy(
            accuracies=accuracies,
            max_depth=num_heads,
            max_child=max_child,
            num_iterations=tree_size - 1,  # -1 because we start with root
            topk=topk,
        )

        return tree_choices

    except (json.JSONDecodeError, KeyError, TypeError):
        return None


@dataclass
class MTPStats:
    """Statistics from MTP generation for benchmarking."""
    tokens_generated: int
    forward_passes: int
    total_proposed: int
    total_accepted: int
    timing: Dict[str, float | int] | None = None

    @property
    def mean_accepted_length(self) -> float:
        """Average number of tokens accepted per forward pass."""
        return self.tokens_generated / max(1, self.forward_passes)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed tokens that were accepted."""
        return self.total_accepted / max(1, self.total_proposed)

    @property
    def speedup(self) -> float:
        """Theoretical max speedup (= mean_accepted), assuming zero tree overhead.

        Actual speedup is: mean_accepted / tree_overhead_factor
        where tree_overhead_factor is typically 1.2-1.5x for small models.
        """
        return self.mean_accepted_length


def generate_tree_buffers(
    medusa_choices: List[Tuple[int, ...]],
    device: torch.device,
    topk: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Generate buffers for tree attention in MTP speculative decoding.

    Args:
        medusa_choices: List of tuples defining the tree structure.
            Each tuple (i, j, k, ...) represents a path through the tree:
            - Position 1: i-th top prediction from head 0
            - Position 2: j-th top prediction from head 1
            - etc.
        device: Device to place tensors on
        topk: Number of top predictions from each Medusa head

    Returns:
        Dictionary containing:
        - tree_attn_mask: (1, 1, tree_len, tree_len) Tree attention mask
        - tree_indices: (tree_len,) Maps positions to candidate token indices
        - tree_position_ids: (tree_len,) Position offsets for each tree node
        - retrieve_indices: (num_candidates, max_depth) Maps paths to node indices
    """
    sorted_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    tree_len = len(sorted_choices) + 1  # +1 for root token

    # 1. Create tree attention mask
    attn_mask = torch.zeros(tree_len, tree_len, device=device)
    attn_mask.fill_diagonal_(1.0)
    attn_mask[:, 0] = 1.0  # All nodes see root

    for idx, choice in enumerate(sorted_choices):
        node_idx = idx + 1
        for depth in range(len(choice) - 1):
            ancestor_choice = choice[: depth + 1]
            ancestor_idx = sorted_choices.index(ancestor_choice) + 1
            attn_mask[node_idx, ancestor_idx] = 1.0

    # 2. Tree indices: map tree positions to candidate token indices
    tree_indices = torch.zeros(tree_len, dtype=torch.long, device=device)
    tree_indices[0] = 0  # Root maps to position 0

    for idx, choice in enumerate(sorted_choices):
        node_idx = idx + 1
        depth = len(choice) - 1
        token_rank = choice[-1]
        tree_indices[node_idx] = token_rank + topk * depth + 1

    # 3. Position IDs for RoPE
    position_ids = torch.zeros(tree_len, dtype=torch.long, device=device)
    for idx, choice in enumerate(sorted_choices):
        position_ids[idx + 1] = len(choice)

    # 4. Retrieve indices for extracting candidate paths
    max_depth = max(len(c) for c in sorted_choices) + 1
    num_candidates = len(sorted_choices) + 1

    retrieve_indices = torch.full(
        (num_candidates, max_depth), -1, dtype=torch.long, device=device
    )
    retrieve_indices[0, 0] = 0

    for idx, choice in enumerate(sorted_choices):
        candidate_idx = idx + 1
        retrieve_indices[candidate_idx, 0] = 0
        for depth in range(len(choice)):
            partial_choice = choice[: depth + 1]
            node_idx = sorted_choices.index(partial_choice) + 1
            retrieve_indices[candidate_idx, depth + 1] = node_idx

    return {
        "tree_attn_mask": attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": position_ids,
        "retrieve_indices": retrieve_indices,
    }


def get_tree_choices(
    num_heads: int,
    checkpoint_path: str,
    tree_size: int = 79,
    topk: int = 64,
) -> List[Tuple[int, ...]]:
    """
    Generate optimal tree choices from calibrated head accuracies.

    This is the single entry point for tree generation. It requires a checkpoint
    with head_acc.json containing calibrated head accuracy statistics.

    For manual override with pre-defined trees, use DEFAULT_TREES or SPARSE_TREES
    constants directly instead of this function.

    Args:
        num_heads: Number of Medusa heads
        checkpoint_path: Path to checkpoint directory containing head_acc.json
        tree_size: Target tree size (default 79)
        topk: Max top-k to consider per head (default 64)

    Returns:
        List of tree node tuples representing the optimal tree structure

    Raises:
        FileNotFoundError: If head_acc.json not found at checkpoint_path
        ValueError: If head_acc.json is invalid or cannot generate tree
    """
    # Normalize checkpoint path (handle final/ subdirectory)
    if checkpoint_path.endswith("/final"):
        base_checkpoint = os.path.dirname(checkpoint_path)
    else:
        base_checkpoint = checkpoint_path

    head_acc_path = os.path.join(base_checkpoint, "head_acc.json")
    cache_key = f"{head_acc_path}:{num_heads}:{tree_size}:{topk}"

    if cache_key in _OPTIMAL_TREE_CACHE:
        return _OPTIMAL_TREE_CACHE[cache_key]

    if not os.path.exists(head_acc_path):
        raise FileNotFoundError(
            f"head_acc.json not found at {head_acc_path}. "
            f"Run calibration first or use DEFAULT_TREES/SPARSE_TREES for manual override."
        )

    tree_choices = generate_optimal_tree_from_head_acc(head_acc_path, num_heads, tree_size, topk)
    if tree_choices is None:
        raise ValueError(
            f"Failed to generate tree from {head_acc_path}. "
            f"File may be corrupted or missing required fields."
        )

    _OPTIMAL_TREE_CACHE[cache_key] = tree_choices
    return tree_choices


# Legacy function aliases for backward compatibility (deprecated)
def get_default_tree_choices(num_heads: int, topk: int = 10) -> List[Tuple[int, ...]]:
    """Deprecated: Use DEFAULT_TREES[num_heads] instead."""
    return list(DEFAULT_TREES.get(num_heads, DEFAULT_TREES[4]))


def get_sparse_tree_choices(num_heads: int) -> List[Tuple[int, ...]]:
    """Deprecated: Use SPARSE_TREES[num_heads] instead."""
    return list(SPARSE_TREES.get(num_heads, SPARSE_TREES[4]))


class GemmaModelWrapper(nn.Module):
    """Wraps HuggingFace Gemma to match nanochat model interface."""

    def __init__(self, model_name="google/gemma-3-1b-it", device=None, dtype=None):
        super().__init__()
        self.model_name = model_name

        # Determine device and dtype
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self._device = device
        self._dtype = dtype

        # Load HuggingFace model
        # Don't use device_map on CPU to avoid accelerate requirement
        if device.type == "cuda":
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=dtype,
                    device_map=device,
                    trust_remote_code=True,
                )
            except ValueError:
                # Fallback if accelerate is not available
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=dtype,
                    trust_remote_code=True,
                ).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=True,
            ).to(device)
        self.model.eval()

        # Build nanochat-compatible config
        hf_config = self.model.config
        self._config = self._build_config(hf_config)

    def _build_config(self, hf_config):
        """Build nanochat-compatible config from HuggingFace config."""
        # Gemma 3 config fields (may vary by model version)
        # Try different attribute names for compatibility
        n_head = getattr(hf_config, 'num_attention_heads', None)
        n_kv_head = getattr(hf_config, 'num_key_value_heads', n_head)
        n_embd = getattr(hf_config, 'hidden_size', None)
        n_layer = getattr(hf_config, 'num_hidden_layers', None)
        vocab_size = getattr(hf_config, 'vocab_size', None)
        sequence_len = getattr(hf_config, 'max_position_embeddings', 8192)

        return GemmaConfigWrapper(
            sequence_len=sequence_len,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_kv_head,
            n_embd=n_embd,
            medusa_num_heads=0,
        )

    @property
    def config(self):
        """Return nanochat-compatible config object."""
        return self._config

    def get_device(self) -> torch.device:
        """Return model device."""
        return self._device

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', return_medusa=False):
        """
        Forward pass matching nanochat interface.

        Args:
            idx: Input token IDs, shape (B, T)
            targets: Target token IDs for training, shape (B, T)
            kv_cache: KV cache for efficient inference (not yet supported for Gemma)
            loss_reduction: 'mean' or 'none' for loss computation
            return_medusa: Whether to return Medusa head outputs (not supported yet)

        Returns:
            If targets is None: logits (B, T, vocab_size)
            If targets provided: loss scalar or (loss, medusa_losses) if return_medusa
        """
        if kv_cache is not None:
            # KV cache inference - use HuggingFace's cache mechanism
            # For now, we'll implement basic inference without nanochat's KV cache
            # This is a simplification for Phase 0
            raise NotImplementedError("KV cache not yet supported for Gemma wrapper. Use without cache for now.")

        # Forward pass through HuggingFace model
        if targets is not None:
            # Training mode
            outputs = self.model(
                input_ids=idx,
                labels=targets,
                return_dict=True,
            )
            loss = outputs.loss
            if return_medusa:
                return loss, []  # No Medusa heads yet
            return loss
        else:
            # Inference mode
            outputs = self.model(
                input_ids=idx,
                return_dict=True,
            )
            logits = outputs.logits

            if return_medusa:
                return logits, None  # No Medusa heads yet
            return logits

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self

    def train(self, mode=True):
        """Set model to training mode."""
        self.model.train(mode)
        return self

    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()

    def named_parameters(self):
        """Return named model parameters."""
        return self.model.named_parameters()

    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        self._device = device
        return self


class GemmaModelWrapperWithKVCache(GemmaModelWrapper):
    """
    Extended Gemma wrapper with KV cache support for efficient inference.
    Uses HuggingFace's native cache mechanism.
    """

    def __init__(self, model_name="google/gemma-3-1b-it", device=None, dtype=None):
        super().__init__(model_name, device, dtype)
        self._past_key_values = None

    def reset_cache(self):
        """Reset the KV cache."""
        self._past_key_values = None

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', return_medusa=False):
        """
        Forward pass with HuggingFace KV cache support.
        """
        if targets is not None:
            # Training mode - no cache
            return super().forward(idx, targets, None, loss_reduction, return_medusa)

        # Inference mode with HF cache
        use_cache = kv_cache is not None

        outputs = self.model(
            input_ids=idx,
            past_key_values=self._past_key_values if use_cache else None,
            use_cache=use_cache,
            return_dict=True,
        )

        if use_cache:
            self._past_key_values = outputs.past_key_values

        logits = outputs.logits

        if return_medusa:
            return logits, None
        return logits


def load_gemma_model(model_name="google/gemma-3-1b-it", device=None, dtype=None, use_kv_cache=False):
    """
    Load a Gemma model with nanochat-compatible interface.

    Args:
        model_name: HuggingFace model name
        device: Device to load model on
        dtype: Data type for model weights
        use_kv_cache: Whether to use KV cache wrapper

    Returns:
        GemmaModelWrapper instance
    """
    if use_kv_cache:
        return GemmaModelWrapperWithKVCache(model_name, device, dtype)
    return GemmaModelWrapper(model_name, device, dtype)


class GemmaMedusaModel(nn.Module):
    """
    Gemma model with Medusa LoRA heads for speculative decoding.

    Architecture:
        input_ids -> Gemma base -> hidden_states
                                        |
                                        +-> lm_head -> base_logits
                                        |
                                        +-> head[0](h) -> lora_delta_0 -> base_logits + lora_delta_0
                                        +-> head[1](h) -> lora_delta_1 -> base_logits + lora_delta_1
                                        ...

    Optimization: Only ONE lm_head matmul needed. LoRA heads output vocab-sized deltas
    that are added to the base logits. The LoRA computation is much smaller than lm_head.

    Multi-layer mode:
        When use_multi_layer=True, heads receive concatenated hidden states from multiple
        transformer layers (final layer + 2 intermediate layers evenly spaced), allowing
        them to leverage both low-level and high-level features for better prediction.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-1b-it",
        medusa_num_heads: int = 4,
        medusa_num_layers: int = 1,
        lora_rank: int = 64,
        lora_alpha: int | None = None,
        device=None,
        dtype=None,
        freeze_base: bool = True,
        zero_init_mlp: bool = False,
        use_head_mixer: bool = False,
        mixer_hidden: int = 16,
        mixer_num_layers: int = 1,  # Number of mixer layers to stack
        mixer_type: str = "mlp",  # "mlp" or "attention"
        attention_head_dim: int | None = None,  # head_dim for attention mixer (default: min(64, hidden_size))
        attn_num_layers: int = 1,  # Number of attention blocks for attention mixer
        causal_attn: bool = False,  # Use causal attention for attention mixer (default: bidirectional)
        use_multi_layer: bool = False,  # Use multi-layer hidden state fusion
    ):
        super().__init__()
        self.model_name = model_name
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.zero_init_mlp = zero_init_mlp
        self.use_head_mixer = use_head_mixer
        self.mixer_hidden = mixer_hidden
        self.mixer_num_layers = mixer_num_layers
        self.mixer_type = mixer_type
        self.attention_head_dim = attention_head_dim
        self.attn_num_layers = attn_num_layers
        self.causal_attn = causal_attn
        self.use_multi_layer = use_multi_layer

        # Determine device and dtype
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self._device = device
        self._dtype = dtype
        self._checkpoint_path: Optional[str] = None  # Set when loading checkpoint

        # Load base model
        if device.type == "cuda":
            try:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=dtype,
                    device_map=device,
                    trust_remote_code=True,
                )
            except ValueError:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=dtype,
                    trust_remote_code=True,
                ).to(device)
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=True,
            ).to(device)

        # Freeze base model if requested
        if freeze_base:
            self.freeze_base_model()

        # Build config
        hf_config = self.base_model.config
        hidden_size = hf_config.hidden_size
        vocab_size = hf_config.vocab_size
        n_layers = hf_config.num_hidden_layers
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._n_layers = n_layers

        # Compute multi-layer indices and create fusion module if using multi-layer fusion
        if use_multi_layer:
            from .heads import compute_multi_layer_indices, MultiLayerFusion
            self._multi_layer_indices = compute_multi_layer_indices(n_layers)
            num_fused_layers = len(self._multi_layer_indices)  # Always 3

            # Create multi-layer fusion preprocessor (shared across all heads)
            self.multi_layer_fusion = MultiLayerFusion(
                hidden_size=hidden_size,
                num_fused_layers=num_fused_layers,
            ).to(device=device, dtype=dtype)
        else:
            self._multi_layer_indices = None
            self.multi_layer_fusion = None

        # Create standard Medusa LoRA heads (same heads used with or without multi-layer)
        self.medusa_heads = nn.ModuleList([
            MedusaLoRAHead(hidden_size, vocab_size, medusa_num_layers, lora_rank,
                          lora_alpha=self.lora_alpha, zero_init_mlp=zero_init_mlp)
            for _ in range(medusa_num_heads)
        ])
        # Move heads to device and dtype
        self.medusa_heads = self.medusa_heads.to(device=device, dtype=dtype)

        # Create cross-head mixer if enabled (supports multiple stacked layers)
        if use_head_mixer:
            if mixer_type == "mlp":
                # Create multiple mixer layers
                head_mixer_fc1_list = []
                head_mixer_fc2_list = []
                channel_mixer_fc_list = []
                for _ in range(mixer_num_layers):
                    # Head mixing MLP: num_heads -> mixer_hidden -> num_heads
                    fc1 = nn.Linear(medusa_num_heads, mixer_hidden, bias=False)
                    fc2 = nn.Linear(mixer_hidden, medusa_num_heads, bias=False)
                    # Channel mixing MLP: hidden_size -> hidden_size (ResBlock style)
                    channel_fc = nn.Linear(hidden_size, hidden_size, bias=False)
                    # Zero-init output layers so mixer starts as identity
                    nn.init.zeros_(fc2.weight)
                    nn.init.zeros_(channel_fc.weight)
                    head_mixer_fc1_list.append(fc1)
                    head_mixer_fc2_list.append(fc2)
                    channel_mixer_fc_list.append(channel_fc)
                self.head_mixer_fc1 = nn.ModuleList(head_mixer_fc1_list).to(device=device, dtype=dtype)
                self.head_mixer_fc2 = nn.ModuleList(head_mixer_fc2_list).to(device=device, dtype=dtype)
                self.channel_mixer_fc = nn.ModuleList(channel_mixer_fc_list).to(device=device, dtype=dtype)
                self.head_attention = None
            elif mixer_type == "attention":
                # Cross-head attention mixer with stacked blocks
                self.head_attention = MedusaHeadAttention(
                    num_heads=medusa_num_heads,
                    hidden_size=hidden_size,
                    num_layers=attn_num_layers,
                    causal=causal_attn,
                ).to(device=device, dtype=dtype)
                # Channel mixing MLP after attention (ResBlock style)
                self.channel_mixer_fc = nn.Linear(hidden_size, hidden_size, bias=False)
                nn.init.zeros_(self.channel_mixer_fc.weight)
                self.channel_mixer_fc = self.channel_mixer_fc.to(device=device, dtype=dtype)
                # Not used for attention mixer
                self.head_mixer_fc1 = None
                self.head_mixer_fc2 = None
            else:
                raise ValueError(f"Unknown mixer_type: {mixer_type}. Must be 'mlp' or 'attention'.")
        else:
            self.head_mixer_fc1 = None
            self.head_mixer_fc2 = None
            self.channel_mixer_fc = None
            self.head_attention = None

        # Pre-compute stacked weights and scalings for efficient batched forward
        self._cache_stacked_weights()

        # Build nanochat-compatible config
        self._config = GemmaConfigWrapper(
            sequence_len=getattr(hf_config, 'max_position_embeddings', 8192),
            vocab_size=vocab_size,
            n_layer=hf_config.num_hidden_layers,
            n_head=hf_config.num_attention_heads,
            n_kv_head=getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads),
            n_embd=hidden_size,
            medusa_num_heads=medusa_num_heads,
        )

        # Candidate scorer callback (optional, for custom tree scoring/pruning)
        # Any callable with signature: (main_logits, medusa_logits, tree_candidates, context) -> scores
        self._candidate_scorer = None
        self._scorer_context: List[int] = []

    @property
    def config(self):
        return self._config

    def get_device(self) -> torch.device:
        return self._device

    @property
    def candidate_scorer(self):
        """Return the attached candidate scorer callback, if any."""
        return self._candidate_scorer

    def set_candidate_scorer(self, scorer) -> None:
        """
        Set a callback for custom candidate scoring during speculative decoding.

        The scorer should be a callable (or object with __call__) with signature:
            scorer(main_logits, medusa_logits, tree_candidates, context_tokens) -> scores

        Where:
            main_logits: [vocab_size] Base model logits
            medusa_logits: [num_heads, vocab_size] Medusa head logits
            tree_candidates: [tree_len] Candidate tokens in tree structure
            context_tokens: List[int] Current context token IDs

        Returns:
            scores: [tree_len] Scores for each tree position (higher = better)

        The scorer can also implement optional methods:
            - reset(): Called at start of new generation
            - on_tokens_accepted(token_ids: List[int]): Called when tokens are accepted

        Args:
            scorer: Callable scorer or None to disable custom scoring
        """
        self._candidate_scorer = scorer
        self._scorer_context = []

    def _reset_scorer(self) -> None:
        """Reset scorer state for new generation."""
        self._scorer_context = []
        if self._candidate_scorer is not None and hasattr(self._candidate_scorer, 'reset'):
            self._candidate_scorer.reset()

    def _update_scorer_context(self, token_ids: List[int]) -> None:
        """Update scorer with accepted tokens."""
        self._scorer_context.extend(token_ids)
        if self._candidate_scorer is not None and hasattr(self._candidate_scorer, 'on_tokens_accepted'):
            self._candidate_scorer.on_tokens_accepted(token_ids)

    def freeze_base_model(self):
        """Freeze all base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        """Unfreeze all base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = True

    def get_medusa_param_count(self) -> int:
        """Return number of trainable Medusa parameters (including fusion module if present)."""
        count = sum(p.numel() for p in self.medusa_heads.parameters())
        if self.multi_layer_fusion is not None:
            count += sum(p.numel() for p in self.multi_layer_fusion.parameters())
        return count

    def get_medusa_state_dict(self) -> dict:
        """Get state dict for all trainable Medusa components.

        This is the recommended way to save checkpoints - it automatically
        includes all Medusa-related weights regardless of mixer type.
        """
        state = {
            'medusa_heads': self.medusa_heads.state_dict(),
        }
        if self.head_attention is not None:
            state['head_attention'] = self.head_attention.state_dict()
        if self.head_mixer_fc1 is not None:
            state['head_mixer_fc1'] = self.head_mixer_fc1.state_dict()
        if self.head_mixer_fc2 is not None:
            state['head_mixer_fc2'] = self.head_mixer_fc2.state_dict()
        if self.channel_mixer_fc is not None:
            state['channel_mixer_fc'] = self.channel_mixer_fc.state_dict()
        if self.multi_layer_fusion is not None:
            state['multi_layer_fusion'] = self.multi_layer_fusion.state_dict()
        return state

    def load_medusa_state_dict(self, state: dict, strict: bool = True) -> list[str]:
        """Load state dict for all trainable Medusa components.

        Args:
            state: State dict from get_medusa_state_dict()
            strict: If True, raise error on missing/unexpected keys

        Returns:
            List of warning messages (empty if all loaded successfully)
        """
        warnings = []

        # Always load medusa_heads
        self.medusa_heads.load_state_dict(state['medusa_heads'])

        # Load optional components, warn if mismatch
        if 'head_attention' in state:
            if self.head_attention is not None:
                self.head_attention.load_state_dict(state['head_attention'])
            elif strict:
                warnings.append("Checkpoint has head_attention but model doesn't (pass attn_num_layers > 0)")
        elif self.head_attention is not None and strict:
            warnings.append("Model has head_attention but checkpoint doesn't")

        if 'head_mixer_fc1' in state:
            if self.head_mixer_fc1 is not None:
                self.head_mixer_fc1.load_state_dict(state['head_mixer_fc1'])
                self.head_mixer_fc2.load_state_dict(state['head_mixer_fc2'])
            elif strict:
                warnings.append("Checkpoint has MLP mixer but model doesn't (pass use_head_mixer=True)")
        elif self.head_mixer_fc1 is not None and strict:
            warnings.append("Model has MLP mixer but checkpoint doesn't")

        if 'channel_mixer_fc' in state:
            if self.channel_mixer_fc is not None:
                self.channel_mixer_fc.load_state_dict(state['channel_mixer_fc'])
            elif strict:
                warnings.append("Checkpoint has channel_mixer_fc but model doesn't")
        elif self.channel_mixer_fc is not None and strict:
            warnings.append("Model has channel_mixer_fc but checkpoint doesn't - using zero-init")

        if 'multi_layer_fusion' in state:
            if self.multi_layer_fusion is not None:
                self.multi_layer_fusion.load_state_dict(state['multi_layer_fusion'])
            elif strict:
                warnings.append("Checkpoint has multi_layer_fusion but model doesn't (pass use_multi_layer=True)")
        elif self.multi_layer_fusion is not None and strict:
            warnings.append("Model has multi_layer_fusion but checkpoint doesn't")

        return warnings

    def _cache_stacked_weights(self):
        """Pre-stack LoRA weights and scalings for efficient batched forward pass."""
        if len(self.medusa_heads) == 0:
            self._stacked_lora_a = None
            self._stacked_lora_b = None
            self._scalings = None
            return

        # Stack lora_A weights: (num_heads, rank, hidden)
        self._stacked_lora_a = torch.stack([head.lora_A.weight for head in self.medusa_heads], dim=0)
        # Stack lora_B weights: (num_heads, vocab, rank)
        self._stacked_lora_b = torch.stack([head.lora_B.weight for head in self.medusa_heads], dim=0)
        # Pre-compute scalings tensor: (num_heads,)
        self._scalings = torch.tensor([head.scaling for head in self.medusa_heads],
                                      device=self._device, dtype=self._dtype)

    def _get_hidden_states(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Get hidden states from the transformer (before lm_head).

        Args:
            input_ids: (B, T) input token IDs

        Returns:
            If use_multi_layer=False:
                hidden_states: (B, T, hidden_size) - final layer only
            If use_multi_layer=True:
                Tuple of:
                - hidden_states: (B, T, hidden_size) - final layer for lm_head
                - multi_layer_hidden: (B, T, 3 * hidden_size) - concatenated layers for MTP heads
        """
        # Determine if we need intermediate hidden states
        need_all_hidden = self.use_multi_layer and self._multi_layer_indices is not None

        outputs = self.base_model.model(
            input_ids=input_ids,
            output_hidden_states=need_all_hidden,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        # NaN/Inf detection for debugging
        if self.training:
            if torch.isnan(hidden_states).any():
                print(f"[NaN DEBUG] NaN in hidden_states from base model!", flush=True)
                print(f"[NaN DEBUG]   hidden_states shape: {hidden_states.shape}", flush=True)
                print(f"[NaN DEBUG]   hidden_states has {torch.isnan(hidden_states).sum().item()} NaN values", flush=True)
            if torch.isinf(hidden_states).any():
                print(f"[NaN DEBUG] Inf in hidden_states from base model!", flush=True)
                print(f"[NaN DEBUG]   hidden_states max: {hidden_states.max().item()}", flush=True)
                print(f"[NaN DEBUG]   hidden_states min: {hidden_states.min().item()}", flush=True)

        if need_all_hidden:
            # outputs.hidden_states is a tuple of (n_layers + 1) tensors:
            # [embedding_output, layer_0_output, layer_1_output, ..., layer_(n-1)_output]
            # We want specific layer outputs (0-indexed from the first transformer layer)
            all_hidden = outputs.hidden_states
            # +1 because index 0 is embedding output, layer i output is at index i+1
            selected_hidden = [all_hidden[idx + 1] for idx in self._multi_layer_indices]
            # Concatenate along hidden dimension: (B, T, 3 * hidden_size)
            multi_layer_hidden = torch.cat(selected_hidden, dim=-1)
            return hidden_states, multi_layer_hidden

        return hidden_states

    def _get_hidden_states_with_cache(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Get hidden states with KV cache support.

        Args:
            input_ids: (B, T) input token IDs (or just new tokens if using cache)
            past_key_values: Cached key-value states from previous forward pass
            position_ids: Optional position IDs for RoPE (required when using cache)
            attention_mask: Optional attention mask for custom attention patterns
                           Shape: (B, 1, T, T+cache_len) for tree attention

        Returns:
            hidden_states: (B, T, hidden_size) or (B, new_tokens, hidden_size)
            new_past_key_values: Updated KV cache
        """
        outputs = self.base_model.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.last_hidden_state, outputs.past_key_values

    def _compute_logits(
        self,
        hidden_states: torch.Tensor,
        return_medusa: bool = True,
        last_only: bool = False,
        multi_layer_hidden: torch.Tensor | None = None,
    ):
        """
        Compute main logits and Medusa logits efficiently with batched matmuls.

        Architecture for LoRA heads:
        - Main: lm_head(h) -> (B, T, vocab)
        - Each head: lm_head(h) + scaling * lora_B(lora_A(ResBlocks(h)))

        For multi-layer heads:
        - Main: lm_head(h_final) -> (B, T, vocab)
        - Each head: lm_head(h_final) + scaling * lora_B(lora_A(ResBlocks(down_proj(h_multi))))

        Optimization: We batch the lora_B projections (rank -> vocab) into a single matmul.
        This is the expensive operation since vocab_size >> rank.

        Args:
            hidden_states: (B, T, hidden_size) from transformer (final layer)
            return_medusa: Whether to compute Medusa logits
            last_only: If True, only compute logits for the last token position.
                       This is much faster for generation where we only need [:, -1, :].
            multi_layer_hidden: (B, T, 3*hidden_size) concatenated multi-layer hidden states.
                               Only used when use_multi_layer=True.

        Returns:
            main_logits: (B, T, vocab_size) or (B, 1, vocab_size) if last_only
            medusa_logits: (num_heads, B, T, vocab_size) or (num_heads, B, 1, vocab_size) or None
        """
        # For generation, we only need the last position's logits
        if last_only:
            hidden_states = hidden_states[:, -1:, :]  # (B, 1, hidden_size)
            if multi_layer_hidden is not None:
                multi_layer_hidden = multi_layer_hidden[:, -1:, :]  # (B, 1, 3*hidden_size)

        if not return_medusa or len(self.medusa_heads) == 0:
            return self.base_model.lm_head(hidden_states), None  # (B, T, vocab) or (B, 1, vocab)

        num_heads = len(self.medusa_heads)

        # Apply multi-layer fusion if enabled (shared preprocessing for all heads)
        if self.use_multi_layer and multi_layer_hidden is not None and self.multi_layer_fusion is not None:
            head_input = self.multi_layer_fusion(multi_layer_hidden, hidden_states)  # (B, T, hidden_size)
        else:
            head_input = hidden_states

        # Step 1: Compute ResBlocks for each head (sequential to preserve gradients during training)
        resblock_outputs = []
        for head_idx, head in enumerate(self.medusa_heads):
            x = head_input
            for block in head.blocks:
                x = block(x)
            # NaN/Inf detection after ResBlocks
            if self.training:
                if torch.isnan(x).any():
                    print(f"[NaN DEBUG] NaN after ResBlock for head {head_idx}!", flush=True)
                if torch.isinf(x).any():
                    print(f"[NaN DEBUG] Inf after ResBlock for head {head_idx}! max={x.max().item()}, min={x.min().item()}", flush=True)
            resblock_outputs.append(x)  # (B, T, hidden_size)

        # Step 2: Stack ResBlock outputs and do batched lora_A projection
        stacked_resblock = torch.stack(resblock_outputs, dim=0)  # (num_heads, B, T, hidden)

        # Step 2.5: Apply cross-head mixer if enabled (supports multiple stacked layers)
        if self.use_head_mixer:
            if self.mixer_type == "mlp" and self.head_mixer_fc1 is not None and self.head_mixer_fc2 is not None:
                for layer_idx in range(len(self.head_mixer_fc1)):
                    stacked_resblock = MedusaLoRAHeadWithMixer.apply_mixer(
                        stacked_resblock,
                        self.head_mixer_fc1[layer_idx],
                        self.head_mixer_fc2[layer_idx],
                        self.channel_mixer_fc[layer_idx],
                    )
            elif self.mixer_type == "attention" and self.head_attention is not None:
                # Apply cross-head attention (stacked blocks inside MedusaHeadAttention)
                stacked_resblock = self.head_attention(stacked_resblock)
                # Apply channel mixing (ResBlock style) after attention
                if self.channel_mixer_fc is not None:
                    stacked_resblock = stacked_resblock + F.silu(self.channel_mixer_fc(stacked_resblock))

        # During training, stack weights fresh each forward to capture gradient updates
        # During inference, use pre-cached weights for efficiency
        if self.training:
            stacked_lora_a = torch.stack([head.lora_A.weight for head in self.medusa_heads], dim=0)
            stacked_lora_b = torch.stack([head.lora_B.weight for head in self.medusa_heads], dim=0)
            scalings = torch.tensor([head.scaling for head in self.medusa_heads],
                                    device=self._device, dtype=self._dtype)
        else:
            stacked_lora_a = self._stacked_lora_a
            stacked_lora_b = self._stacked_lora_b
            scalings = self._scalings

        # lora_A projection: (num_heads, rank, hidden)
        lora_a_out = torch.einsum('hbti,hri->hbtr', stacked_resblock, stacked_lora_a)

        # NaN/Inf detection after lora_A
        if self.training:
            if torch.isnan(lora_a_out).any():
                print(f"[NaN DEBUG] NaN after lora_A projection!", flush=True)
                print(f"[NaN DEBUG]   stacked_resblock has NaN: {torch.isnan(stacked_resblock).any()}", flush=True)
                print(f"[NaN DEBUG]   stacked_lora_a has NaN: {torch.isnan(stacked_lora_a).any()}", flush=True)
            if torch.isinf(lora_a_out).any():
                print(f"[NaN DEBUG] Inf after lora_A projection! max={lora_a_out.max().item()}, min={lora_a_out.min().item()}", flush=True)
            for i, head in enumerate(self.medusa_heads):
                w_max = head.lora_A.weight.abs().max().item()
                if w_max > 1e4:
                    print(f"[NaN DEBUG]   head{i} lora_A weight max: {w_max:.4f} (LARGE!)", flush=True)

        # Step 3: Batched lora_B projection
        # (num_heads, B, T, rank) @ (num_heads, vocab, rank).T -> (num_heads, B, T, vocab)
        lora_deltas = torch.einsum('hbtr,hvr->hbtv', lora_a_out, stacked_lora_b)

        # NaN/Inf detection after lora_B
        if self.training:
            if torch.isnan(lora_deltas).any():
                print(f"[NaN DEBUG] NaN after lora_B projection!", flush=True)
                print(f"[NaN DEBUG]   lora_a_out has NaN: {torch.isnan(lora_a_out).any()}", flush=True)
                print(f"[NaN DEBUG]   stacked_lora_b has NaN: {torch.isnan(stacked_lora_b).any()}", flush=True)
            if torch.isinf(lora_deltas).any():
                print(f"[NaN DEBUG] Inf after lora_B projection! max={lora_deltas.max().item()}, min={lora_deltas.min().item()}", flush=True)
            for i, head in enumerate(self.medusa_heads):
                w_max = head.lora_B.weight.abs().max().item()
                if w_max > 1e4:
                    print(f"[NaN DEBUG]   head{i} lora_B weight max: {w_max:.4f} (LARGE!)", flush=True)

        # Step 4: Apply per-head scaling
        lora_deltas = lora_deltas * scalings.view(num_heads, 1, 1, 1)

        # Step 5: Batched lm_head projection for main + all heads
        all_hiddens = torch.cat([hidden_states.unsqueeze(0), stacked_resblock], dim=0)  # (num_heads+1, B, T, hidden)
        base_logits = self.base_model.lm_head(all_hiddens)  # (num_heads+1, B, T, vocab)

        # NaN/Inf detection after lm_head
        if self.training:
            if torch.isnan(base_logits).any():
                print(f"[NaN DEBUG] NaN after lm_head projection!", flush=True)
                print(f"[NaN DEBUG]   all_hiddens has NaN: {torch.isnan(all_hiddens).any()}", flush=True)
            if torch.isinf(base_logits).any():
                print(f"[NaN DEBUG] Inf after lm_head projection! max={base_logits.max().item()}, min={base_logits.min().item()}", flush=True)
                print(f"[NaN DEBUG]   all_hiddens has Inf: {torch.isinf(all_hiddens).any()}", flush=True)
                print(f"[NaN DEBUG]   lm_head weight max: {self.base_model.lm_head.weight.abs().max().item():.4f}", flush=True)

        medusa_logits = base_logits[1:] + lora_deltas  # (num_heads, B, T, vocab)

        # For ablation testing: slice to only use first N heads during inference
        effective_heads = self.medusa_num_heads
        if effective_heads < num_heads:
            medusa_logits = medusa_logits[:effective_heads]

        return base_logits[0], medusa_logits

    @torch.compiler.disable  # Variable seq lengths cause excessive recompilations
    def _compute_losses_chunked(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
        loss_reduction: str = 'mean',
        chunk_size: int = 128,
        multi_layer_hidden: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute losses with chunked cross-entropy to reduce peak memory.

        Instead of materializing the full (num_heads+1, B, T, vocab) logits tensor,
        we process the sequence in chunks, computing lm_head projection and CE loss
        for each chunk. This reduces peak memory from O(B*T*vocab) to O(B*chunk*vocab).

        The ResBlock and LoRA projections are still computed in full (they're small),
        only the vocab-sized tensors are chunked.

        Args:
            hidden_states: (B, T, hidden_size) from transformer (final layer)
            targets: (B, T) target token IDs
            loss_reduction: 'mean' or 'none'
            chunk_size: Number of sequence positions to process at once
            multi_layer_hidden: (B, T, 3*hidden_size) concatenated multi-layer hidden states.
                               Only used when use_multi_layer=True.

        Returns:
            main_loss: scalar loss for main head
            medusa_losses: list of scalar losses for each Medusa head
        """
        B, T, hidden_size = hidden_states.shape
        num_heads = len(self.medusa_heads)
        lm_head = self.base_model.lm_head

        # Apply multi-layer fusion if enabled (shared preprocessing for all heads)
        if self.use_multi_layer and multi_layer_hidden is not None and self.multi_layer_fusion is not None:
            head_input = self.multi_layer_fusion(multi_layer_hidden, hidden_states)  # (B, T, hidden_size)
        else:
            head_input = hidden_states

        # Step 1: Compute ResBlocks for all heads (small memory: num_heads * B * T * hidden)
        resblock_outputs = []
        for head in self.medusa_heads:
            x = head_input
            for block in head.blocks:
                x = block(x)
            resblock_outputs.append(x)
        stacked_resblock = torch.stack(resblock_outputs, dim=0)  # (num_heads, B, T, hidden)

        # Step 1.5: Apply cross-head mixer if enabled (supports multiple stacked layers)
        if self.use_head_mixer:
            if self.mixer_type == "mlp" and self.head_mixer_fc1 is not None and self.head_mixer_fc2 is not None:
                for layer_idx in range(len(self.head_mixer_fc1)):
                    stacked_resblock = MedusaLoRAHeadWithMixer.apply_mixer(
                        stacked_resblock,
                        self.head_mixer_fc1[layer_idx],
                        self.head_mixer_fc2[layer_idx],
                        self.channel_mixer_fc[layer_idx],
                    )
            elif self.mixer_type == "attention" and self.head_attention is not None:
                # Apply cross-head attention (stacked blocks inside MedusaHeadAttention)
                stacked_resblock = self.head_attention(stacked_resblock)
                # Apply channel mixing (ResBlock style) after attention
                if self.channel_mixer_fc is not None:
                    stacked_resblock = stacked_resblock + F.silu(self.channel_mixer_fc(stacked_resblock))

        # Step 2: Compute LoRA projections (small memory: num_heads * B * T * rank)
        # Stack weights for batched computation
        stacked_lora_a = torch.stack([h.lora_A.weight for h in self.medusa_heads], dim=0)
        stacked_lora_b = torch.stack([h.lora_B.weight for h in self.medusa_heads], dim=0)
        scalings = torch.tensor([h.scaling for h in self.medusa_heads], device=self._device, dtype=self._dtype)

        # lora_A: (num_heads, B, T, hidden) @ (num_heads, rank, hidden).T -> (num_heads, B, T, rank)
        lora_a_out = torch.einsum('hbti,hri->hbtr', stacked_resblock, stacked_lora_a)

        # Step 3: Chunked loss computation
        # For main loss, we need full sequence. For medusa, we need shifted sequences.
        # We accumulate chunk losses and then compute mean. This ensures proper gradient flow.

        # Process main head in chunks - collect losses for proper gradient flow
        # NOTE: We always compute CE for every chunk (no `if chunk_valid > 0` conditional)
        # to ensure all DDP ranks have identical computation graphs. CE with ignore_index=-1
        # handles chunks with no valid targets correctly (returns 0).
        main_chunk_losses = []
        total_valid = (targets != -1).sum()  # Keep as tensor to avoid graph break

        for t_start in range(0, T, chunk_size):
            t_end = min(t_start + chunk_size, T)

            # Main head: lm_head(hidden_states)
            chunk_hidden = hidden_states[:, t_start:t_end, :]  # (B, chunk, hidden)
            chunk_logits = lm_head(chunk_hidden)  # (B, chunk, vocab)
            chunk_targets = targets[:, t_start:t_end]  # (B, chunk)

            chunk_loss = F.cross_entropy(
                chunk_logits.reshape(-1, chunk_logits.shape[-1]),
                chunk_targets.reshape(-1),
                ignore_index=-1,
                reduction='sum',
            )
            main_chunk_losses.append(chunk_loss)

        # Compute mean main loss
        if total_valid > 0:
            main_loss = sum(main_chunk_losses) / total_valid
        else:
            main_loss = sum(main_chunk_losses)  # Will be 0, but keeps graph consistent

        # Process each Medusa head in chunks
        # Same principle: always compute CE for every chunk to keep DDP graphs identical
        medusa_losses = []
        for k in range(num_heads):
            shift = 2 + k
            if shift >= T:
                # Still need a tensor in the graph for DDP consistency
                medusa_losses.append(hidden_states.new_zeros((), requires_grad=True) * 0)
                continue

            # Effective sequence length for this head
            T_eff = T - shift
            head_chunk_losses = []

            # Count valid targets for this head (shifted targets)
            head_targets = targets[:, shift:]
            head_valid = (head_targets != -1).sum()  # Keep as tensor to avoid graph break

            for t_start in range(0, T_eff, chunk_size):
                t_end = min(t_start + chunk_size, T_eff)

                # Get chunk of ResBlock output and LoRA-A output
                chunk_resblock = stacked_resblock[k, :, t_start:t_end, :]  # (B, chunk, hidden)
                chunk_lora_a = lora_a_out[k, :, t_start:t_end, :]  # (B, chunk, rank)

                # Compute lm_head(resblock) for this chunk
                chunk_base_logits = lm_head(chunk_resblock)  # (B, chunk, vocab)

                # Compute LoRA delta: lora_B(lora_a) * scaling
                # (B, chunk, rank) @ (vocab, rank).T -> (B, chunk, vocab)
                chunk_lora_delta = torch.einsum('btr,vr->btv', chunk_lora_a, stacked_lora_b[k]) * scalings[k]

                # Full logits for this head
                chunk_logits = chunk_base_logits + chunk_lora_delta  # (B, chunk, vocab)

                # Targets are shifted
                chunk_targets = targets[:, t_start + shift:t_end + shift]  # (B, chunk)

                chunk_loss = F.cross_entropy(
                    chunk_logits.reshape(-1, chunk_logits.shape[-1]),
                    chunk_targets.reshape(-1),
                    ignore_index=-1,
                    reduction='sum',
                )
                head_chunk_losses.append(chunk_loss)

            # Compute mean loss for this head
            if head_valid > 0:
                medusa_losses.append(sum(head_chunk_losses) / head_valid)
            else:
                medusa_losses.append(sum(head_chunk_losses))  # Will be 0, but keeps graph consistent

        return main_loss, medusa_losses

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache=None,
        loss_reduction: str = 'mean',
        return_medusa: bool = False,
        last_only: bool = False,
        use_chunked_loss: bool = False,
        chunk_size: int = 128,
    ):
        """
        Forward pass with optional Medusa head computation.

        Args:
            input_ids: (B, T) input token IDs
            targets: (B, T) target token IDs for training
            kv_cache: KV cache (not yet supported)
            loss_reduction: 'mean' or 'none'
            return_medusa: Whether to return Medusa outputs
            last_only: If True, only compute logits for the last token position.
                       Use this during generation for efficiency.
            use_chunked_loss: If True, compute losses in chunks to reduce memory.
                              Reduces peak memory from O(B*T*vocab) to O(B*chunk*vocab).
            chunk_size: Sequence chunk size when use_chunked_loss=True.

        Returns:
            If targets is None:
                logits or (logits, medusa_logits)
            If targets provided:
                loss or (loss, medusa_losses)
        """
        if kv_cache is not None:
            raise NotImplementedError("KV cache not yet supported for GemmaMedusaModel")

        # Get hidden states from transformer
        hidden_result = self._get_hidden_states(input_ids)

        # Handle multi-layer vs single-layer case
        if self.use_multi_layer and isinstance(hidden_result, tuple):
            hidden_states, multi_layer_hidden = hidden_result
        else:
            hidden_states = hidden_result
            multi_layer_hidden = None

        # Use chunked loss computation for memory efficiency during training
        if use_chunked_loss and targets is not None and return_medusa:
            return self._compute_losses_chunked(hidden_states, targets, loss_reduction, chunk_size, multi_layer_hidden)

        # Standard path: compute logits then losses
        main_logits, medusa_logits = self._compute_logits(hidden_states, return_medusa, last_only, multi_layer_hidden)

        if targets is not None:
            # Training mode - compute losses
            vocab_size = main_logits.shape[-1]

            # NaN/Inf detection before loss computation
            if self.training:
                if torch.isnan(main_logits).any():
                    print(f"[NaN DEBUG] NaN in main_logits before CE loss!", flush=True)
                if torch.isinf(main_logits).any():
                    print(f"[NaN DEBUG] Inf in main_logits before CE loss!", flush=True)
                    print(f"[NaN DEBUG]   main_logits max: {main_logits.max().item()}", flush=True)
                    print(f"[NaN DEBUG]   main_logits min: {main_logits.min().item()}", flush=True)
                # Check targets for invalid values
                valid_targets = targets[targets != -1]
                if valid_targets.numel() > 0:
                    if (valid_targets < 0).any() or (valid_targets >= vocab_size).any():
                        print(f"[NaN DEBUG] Invalid target IDs! min={valid_targets.min().item()}, max={valid_targets.max().item()}, vocab_size={vocab_size}", flush=True)

            # Main loss
            main_loss = F.cross_entropy(
                main_logits.view(-1, vocab_size),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )

            # NaN detection after main loss
            if self.training and torch.isnan(main_loss):
                print(f"[NaN DEBUG] NaN in main_loss after CE!", flush=True)
                # Check if all targets are ignored
                valid_count = (targets != -1).sum().item()
                total_count = targets.numel()
                print(f"[NaN DEBUG]   valid targets: {valid_count}/{total_count}", flush=True)
                print(f"[NaN DEBUG]   targets unique values: {torch.unique(targets).tolist()[:20]}", flush=True)
                print(f"[NaN DEBUG]   main_logits has NaN: {torch.isnan(main_logits).any()}", flush=True)
                print(f"[NaN DEBUG]   main_logits has Inf: {torch.isinf(main_logits).any()}", flush=True)
                if not torch.isnan(main_logits).any() and not torch.isinf(main_logits).any():
                    print(f"[NaN DEBUG]   main_logits max: {main_logits.max().item()}", flush=True)
                    print(f"[NaN DEBUG]   main_logits min: {main_logits.min().item()}", flush=True)

            if return_medusa and medusa_logits is not None:
                # NaN/Inf detection for medusa logits
                if self.training:
                    if torch.isnan(medusa_logits).any():
                        print(f"[NaN DEBUG] NaN in medusa_logits before CE loss!", flush=True)
                    if torch.isinf(medusa_logits).any():
                        print(f"[NaN DEBUG] Inf in medusa_logits before CE loss!", flush=True)
                        print(f"[NaN DEBUG]   medusa_logits max: {medusa_logits.max().item()}", flush=True)
                        print(f"[NaN DEBUG]   medusa_logits min: {medusa_logits.min().item()}", flush=True)

                # Compute per-head losses
                # Head k predicts token at position t+k+2
                medusa_losses = []
                for k in range(len(self.medusa_heads)):
                    shift = 2 + k
                    if shift >= hidden_states.shape[1]:
                        medusa_losses.append(torch.tensor(0.0, device=self._device))
                        continue

                    head_logits = medusa_logits[k, :, :-shift, :].contiguous()
                    head_targets = targets[:, shift:].contiguous()

                    head_loss = F.cross_entropy(
                        head_logits.view(-1, vocab_size),
                        head_targets.view(-1),
                        ignore_index=-1,
                        reduction=loss_reduction,
                    )

                    # NaN detection per head
                    if self.training and torch.isnan(head_loss):
                        print(f"[NaN DEBUG] NaN in head{k}_loss after CE!", flush=True)
                        valid_count = (head_targets != -1).sum().item()
                        total_count = head_targets.numel()
                        print(f"[NaN DEBUG]   head{k} valid targets: {valid_count}/{total_count}", flush=True)
                        print(f"[NaN DEBUG]   head_logits has NaN: {torch.isnan(head_logits).any()}", flush=True)
                        print(f"[NaN DEBUG]   head_logits has Inf: {torch.isinf(head_logits).any()}", flush=True)
                        if not torch.isnan(head_logits).any() and not torch.isinf(head_logits).any():
                            print(f"[NaN DEBUG]   head_logits max: {head_logits.max().item()}", flush=True)
                            print(f"[NaN DEBUG]   head_logits min: {head_logits.min().item()}", flush=True)

                    medusa_losses.append(head_loss)

                return main_loss, medusa_losses

            return main_loss

        else:
            # Inference mode
            if return_medusa:
                return main_logits, medusa_logits
            return main_logits

    # =========================================================================
    # MTP (Multi-Token Prediction) / Speculative Decoding Methods
    # =========================================================================

    _tree_buffers_cache: Optional[Dict[str, torch.Tensor]] = None
    _tree_buffers_config: Optional[Tuple[Tuple[Tuple[int, ...], ...], int]] = None

    def _get_tree_buffers(
        self,
        tree_choices: Optional[List[Tuple[int, ...]]] = None,
        topk: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Get tree attention buffers, using cache if config matches.

        Args:
            tree_choices: Tree structure to use. If None, generates optimal tree from
                         checkpoint's head_acc.json. For manual override, pass
                         DEFAULT_TREES[num_heads] or SPARSE_TREES[num_heads].
            topk: Number of top predictions from each head (used for buffer sizing)

        Returns:
            Dictionary of tree buffers

        Raises:
            FileNotFoundError: If tree_choices is None and head_acc.json not found
            ValueError: If tree_choices is None and head_acc.json is invalid
        """
        # Generate choices if not provided
        if tree_choices is None:
            if self._checkpoint_path is None:
                raise ValueError(
                    "Cannot generate optimal tree: checkpoint_path not set. "
                    "Either set checkpoint_path or pass tree_choices explicitly "
                    "(e.g., tree_choices=DEFAULT_TREES[num_heads])."
                )
            tree_choices = get_tree_choices(
                self.medusa_num_heads,
                self._checkpoint_path,
            )

        # Use tuple for hashable cache key
        config = (tuple(tree_choices), topk)
        if self._tree_buffers_cache is None or self._tree_buffers_config != config:
            # Ensure topk is large enough for the tree structure
            effective_topk = max(topk, len(tree_choices))
            self._tree_buffers_cache = generate_tree_buffers(tree_choices, self._device, effective_topk)
            self._tree_buffers_config = config
        return self._tree_buffers_cache

    @torch.inference_mode()
    def _generate_candidates(
        self,
        main_logits: torch.Tensor,
        medusa_logits: torch.Tensor,
        buffers: Dict[str, torch.Tensor],
        topk: int = 10,
        temperature: float = 0.0,
        retrieval_logits: Optional[torch.Tensor] = None,
        retrieval_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate candidate token sequences from model predictions.
        Optimized with vectorized indexing.

        Args:
            main_logits: (B, vocab_size) Main model logits for last position
            medusa_logits: (num_heads, B, vocab_size) Medusa logits for last position
            buffers: Tree attention buffers
            topk: Number of top-k predictions per head
            temperature: Sampling temperature (0.0 = greedy)
            retrieval_logits: Optional (vocab_size,) retrieval model logits for blending
            retrieval_weight: Weight for retrieval (alpha in: (1-α)*medusa + α*retrieval)

        Returns:
            candidates: (num_candidates, max_depth) Candidate token sequences
            tree_candidates: (tree_len,) Tokens arranged in tree structure
        """
        tree_indices = buffers["tree_indices"]
        retrieve_indices = buffers["retrieve_indices"]

        # Get main model prediction (greedy or sampled)
        if temperature == 0.0:
            base_token = main_logits[0].argmax()
        else:
            probs = F.softmax(main_logits[0] / temperature, dim=-1)
            base_token = torch.multinomial(probs, num_samples=1)[0]

        # Determine required topk from tree_indices (for fixed-size tree ablations)
        # tree_indices maps to positions in flat_candidates = [base, head0_k0..k(topk-1), head1_k0..., ...]
        # Position i maps to: 0 = base, 1..topk = head0, topk+1..2*topk = head1, etc.
        max_tree_idx = tree_indices.max().item()
        num_heads = medusa_logits.shape[0]
        required_topk = max(topk, (max_tree_idx + num_heads - 1) // num_heads)  # ceiling division

        # Get top-k from each Medusa head: (num_heads, required_topk)
        # Blend with retrieval logits if provided: (1-α)*medusa + α*retrieval
        medusa_logits_for_topk = medusa_logits[:, 0, :]
        if retrieval_logits is not None and retrieval_weight > 0:
            # Blend: new_logits = (1 - α) * medusa_logits + α * retrieval_logits
            medusa_logits_for_topk = (
                (1 - retrieval_weight) * medusa_logits_for_topk +
                retrieval_weight * retrieval_logits.unsqueeze(0)
            )
        medusa_topk = torch.topk(medusa_logits_for_topk, required_topk, dim=-1).indices

        # Build flat candidate array: [base_token, head0_topk, head1_topk, ...]
        flat_candidates = torch.cat([base_token.unsqueeze(0), medusa_topk.view(-1)])

        # Map to tree structure using tree_indices (vectorized)
        tree_candidates = flat_candidates[tree_indices]

        # Extract candidate paths - VECTORIZED (27x faster than loop)
        safe_indices = retrieve_indices.clamp(min=0)
        candidates = tree_candidates[safe_indices]
        # Zero out invalid positions (where original index was -1)
        candidates = candidates * (retrieve_indices >= 0).long()

        return candidates, tree_candidates

    @torch.inference_mode()
    def forward_mtp(
        self,
        input_ids: torch.Tensor,
        tree_candidates: torch.Tensor,
        buffers: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with tree attention for verifying MTP candidates.
        Returns tree-indexed logits to avoid expensive extraction.

        Args:
            input_ids: (B, T) Current token sequence
            tree_candidates: (tree_len,) Candidate tokens in tree structure
            buffers: Tree attention buffers

        Returns:
            tree_logits: (tree_len, vocab_size) Logits at tree positions
            retrieve_indices: (num_candidates, max_depth) For candidate extraction
            valid_mask: (num_candidates, max_depth) Which positions are valid
        """
        B, T = input_ids.shape
        retrieve_indices = buffers["retrieve_indices"]

        # Concatenate input with tree candidates: (B, T + tree_len)
        extended_input = torch.cat([
            input_ids,
            tree_candidates.unsqueeze(0).expand(B, -1),
        ], dim=1)

        # Full forward pass (no KV cache)
        hidden_states = self._get_hidden_states(extended_input)
        logits, _ = self._compute_logits(hidden_states, return_medusa=False)

        # Return tree logits directly (avoid expensive candidate extraction)
        tree_logits = logits[0, T:, :]  # (tree_len, vocab_size)
        valid_mask = retrieve_indices >= 0

        return tree_logits, retrieve_indices, valid_mask

    @torch.inference_mode()
    def forward_mtp_with_cache(
        self,
        tree_candidates: torch.Tensor,
        buffers: Dict[str, torch.Tensor],
        past_key_values: Tuple,
        base_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple, torch.Tensor]:
        """
        Forward pass with tree attention for verifying MTP candidates, using KV cache.

        This is much faster than forward_mtp because it only processes the tree tokens,
        not the entire input sequence. The KV cache contains the states for the prompt.

        Args:
            tree_candidates: (tree_len,) Candidate tokens in tree structure
            buffers: Tree attention buffers
            past_key_values: Cached KV states from prompt processing
            base_seq_len: Length of the cached sequence (for position IDs)

        Returns:
            tree_logits: (tree_len, vocab_size) Logits at tree positions
            retrieve_indices: (num_candidates, max_depth) For candidate extraction
            valid_mask: (num_candidates, max_depth) Which positions are valid
            new_past_key_values: Updated KV cache (for potential reuse)
            tree_hidden_states: (tree_len, hidden_size) Hidden states at tree positions
        """
        retrieve_indices = buffers["retrieve_indices"]
        tree_position_ids = buffers["tree_position_ids"]
        tree_attn_mask = buffers["tree_attn_mask"]  # (1, 1, tree_len, tree_len)

        # Prepare inputs - only process tree tokens (KV cache has prompt)
        tree_input = tree_candidates.unsqueeze(0)  # (1, tree_len)
        tree_len = tree_candidates.shape[0]

        # Position IDs: base_seq_len + tree position offsets
        # tree_position_ids contains depth-based offsets (0, 1, 1, 1, 2, 2, ...)
        position_ids = base_seq_len + tree_position_ids.unsqueeze(0)  # (1, tree_len)

        # Build attention mask efficiently (HF format: 0 = attend, -inf = don't)
        # All tree tokens attend to all cached tokens (zeros), plus tree attention pattern
        # Use pre-computed HF-format tree mask from buffers
        hf_tree_mask = buffers.get("hf_tree_attn_mask")
        if hf_tree_mask is None:
            # Convert tree mask to HF format once and cache it
            hf_tree_mask = torch.where(
                tree_attn_mask > 0.5,
                torch.zeros_like(tree_attn_mask),
                torch.full_like(tree_attn_mask, float('-inf'))
            )
            buffers["hf_tree_attn_mask"] = hf_tree_mask

        # Create full mask: [zeros for cache attention, tree pattern]
        # Shape: (1, 1, tree_len, base_seq_len + tree_len)
        hf_attn_mask = torch.zeros(1, 1, tree_len, base_seq_len + tree_len,
                                   device=self._device, dtype=tree_attn_mask.dtype)
        hf_attn_mask[:, :, :, base_seq_len:] = hf_tree_mask

        # Forward pass with cache and tree attention mask
        hidden_states, new_past_key_values = self._get_hidden_states_with_cache(
            tree_input,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=hf_attn_mask,
        )

        # Compute logits (only for tree positions)
        logits, _ = self._compute_logits(hidden_states, return_medusa=False)

        # tree_logits is (1, tree_len, vocab) -> (tree_len, vocab)
        tree_logits = logits[0]
        valid_mask = retrieve_indices >= 0

        # Return hidden states for reuse in Medusa head computation
        tree_hidden_states = hidden_states[0]  # (tree_len, hidden_size)

        return tree_logits, retrieve_indices, valid_mask, new_past_key_values, tree_hidden_states

    @torch.inference_mode()
    def _evaluate_candidates_greedy_fast(
        self,
        tree_logits: torch.Tensor,
        candidates: torch.Tensor,
        retrieve_indices: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[int, int]:
        """
        Optimized greedy acceptance working directly on tree logits.
        Avoids extracting full vocab-sized tensors per candidate.

        Args:
            tree_logits: (tree_len, vocab_size) Logits at tree positions
            candidates: (num_candidates, max_depth) Candidate token sequences
            retrieve_indices: (num_candidates, max_depth) Indices into tree
            valid_mask: (num_candidates, max_depth) Valid positions

        Returns:
            best_candidate: Index of best candidate path
            accept_length: Number of tokens to accept
        """
        # Get predictions at each tree position
        tree_predictions = tree_logits.argmax(dim=-1)  # (tree_len,)

        # Get predictions for each candidate position (vectorized)
        safe_indices = retrieve_indices.clamp(min=0)
        candidate_predictions = tree_predictions[safe_indices]  # (num_candidates, max_depth)

        # Check matches: prediction at position j should equal candidate at j+1
        # candidates[:, 1:] are the speculated tokens
        # candidate_predictions[:, :-1] are the model's predictions for those positions
        matches = (candidates[:, 1:] == candidate_predictions[:, :-1])
        matches = matches & valid_mask[:, 1:]  # Only count valid positions

        # Find longest matching prefix using cumprod
        cumulative_matches = torch.cumprod(matches.int(), dim=1)
        accept_lengths = cumulative_matches.sum(dim=1)

        best_candidate = int(accept_lengths.argmax().item())
        accept_length = int(accept_lengths[best_candidate].item())

        return best_candidate, accept_length

    @torch.inference_mode()
    def _evaluate_candidates_typical_fast(
        self,
        tree_logits: torch.Tensor,
        candidates: torch.Tensor,
        retrieve_indices: torch.Tensor,
        valid_mask: torch.Tensor,
        temperature: float,
        posterior_threshold: float = 0.09,
        posterior_alpha: float = 0.3,
    ) -> Tuple[int, int]:
        """
        Optimized typical acceptance working directly on tree logits.
        CUDA-optimized: uses vectorized gather instead of loops.

        Args:
            tree_logits: (tree_len, vocab_size) Logits at tree positions
            candidates: (num_candidates, max_depth) Candidate token sequences
            retrieve_indices: (num_candidates, max_depth) Indices into tree
            valid_mask: (num_candidates, max_depth) Valid positions
            temperature: Sampling temperature
            posterior_threshold: Hard acceptance threshold
            posterior_alpha: Entropy-adaptive factor

        Returns:
            best_candidate: Index of best candidate
            accept_length: Number of tokens to accept
        """
        num_candidates, max_depth = candidates.shape
        safe_indices = retrieve_indices.clamp(min=0)

        # Compute probabilities for tree positions
        tree_probs = F.softmax(tree_logits / temperature, dim=-1)  # (tree_len, vocab)

        # CUDA-optimized: vectorized probability gathering
        # Get tree positions for prediction (positions 0 to max_depth-2)
        pred_tree_pos = safe_indices[:, :-1]  # (num_candidates, max_depth-1)
        # Get tokens to evaluate (positions 1 to max_depth-1)
        eval_tokens = candidates[:, 1:]  # (num_candidates, max_depth-1)

        # Gather probabilities: tree_probs[pred_tree_pos[i,j], eval_tokens[i,j]]
        # Flatten for gather, then reshape
        flat_tree_pos = pred_tree_pos.flatten()  # (num_candidates * (max_depth-1),)
        flat_tokens = eval_tokens.flatten()  # (num_candidates * (max_depth-1),)
        flat_probs = tree_probs[flat_tree_pos, flat_tokens]
        candidate_probs = flat_probs.view(num_candidates, max_depth - 1)

        # Compute entropy at each tree position (vectorized)
        tree_entropy = -torch.sum(tree_probs * torch.log(tree_probs + 1e-10), dim=-1)
        candidate_entropy = tree_entropy[pred_tree_pos]

        # Adaptive threshold
        threshold = torch.minimum(
            torch.full_like(candidate_entropy, posterior_threshold),
            torch.exp(-candidate_entropy) * posterior_alpha,
        )

        # Accept where probability exceeds threshold
        accepts = candidate_probs > threshold
        accepts = accepts & valid_mask[:, 1:]

        # Find longest accepting prefix
        cumulative_accepts = torch.cumprod(accepts.int(), dim=1)
        accept_lengths = cumulative_accepts.sum(dim=1)

        best_candidate = int(accept_lengths.argmax().item())
        accept_length = int(accept_lengths[best_candidate].item())

        return best_candidate, accept_length

    # Keep old methods for backward compatibility but mark deprecated
    @torch.inference_mode()
    def _evaluate_candidates_greedy(
        self,
        verify_logits: torch.Tensor,
        candidates: torch.Tensor,
    ) -> Tuple[int, int]:
        """DEPRECATED: Use _evaluate_candidates_greedy_fast instead."""
        predictions = verify_logits.argmax(dim=-1)
        matches = (candidates[:, 1:] == predictions[:, :-1]).int()
        cumulative_matches = torch.cumprod(matches, dim=1)
        accept_lengths = cumulative_matches.sum(dim=1)
        best_candidate = int(accept_lengths.argmax().item())
        accept_length = int(accept_lengths[best_candidate].item())
        return best_candidate, accept_length

    @torch.inference_mode()
    def _evaluate_candidates_typical(
        self,
        verify_logits: torch.Tensor,
        candidates: torch.Tensor,
        temperature: float,
        posterior_threshold: float = 0.09,
        posterior_alpha: float = 0.3,
    ) -> Tuple[int, int]:
        """DEPRECATED: Use _evaluate_candidates_typical_fast instead."""
        probs = F.softmax(verify_logits[:, :-1] / temperature, dim=-1)
        candidate_tokens = candidates[:, 1:].unsqueeze(-1)
        candidate_probs = torch.gather(probs, dim=-1, index=candidate_tokens).squeeze(-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        threshold = torch.minimum(
            torch.full_like(entropy, posterior_threshold),
            torch.exp(-entropy) * posterior_alpha,
        )
        accepts = candidate_probs > threshold
        cumulative_accepts = torch.cumprod(accepts.int(), dim=1)
        accept_lengths = cumulative_accepts.sum(dim=1)
        best_candidate = int(accept_lengths.argmax().item())
        accept_length = int(accept_lengths[best_candidate].item())
        return best_candidate, accept_length

    @torch.inference_mode()
    def generate_mtp(
        self,
        input_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        topk: int = 10,
        tree_choices: Optional[List[Tuple[int, ...]]] = None,
        posterior_threshold: float = 0.09,
        posterior_alpha: float = 0.3,
        eos_token_id: Optional[int] = None,
        candidate_scorer: Optional[Callable] = None,
    ) -> Tuple[List[int], MTPStats]:
        """
        Generate tokens using MTP (Multi-Token Prediction) speculative decoding.

        This uses Medusa heads to speculate multiple tokens ahead, then verifies
        them in parallel with tree attention, providing ~2-3x speedup.

        Args:
            input_ids: Initial prompt as list of token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            topk: Number of top-k predictions per Medusa head
            tree_choices: Tree structure for speculation. If None, uses optimal tree
                         from checkpoint's head_acc.json. For manual override, pass
                         DEFAULT_TREES[num_heads] or SPARSE_TREES[num_heads].
            posterior_threshold: Typical acceptance hard threshold
            posterior_alpha: Typical acceptance entropy factor
            eos_token_id: EOS token ID for stopping (None = no early stop)
            candidate_scorer: Optional callback for custom candidate scoring.
                Signature: (main_logits, medusa_logits, tree_candidates, context) -> scores
                Where scores is [tree_len] tensor. If provided, candidates are re-ranked
                by these scores before verification. The scorer can also implement:
                - reset(): Called at start of generation
                - on_tokens_accepted(tokens: List[int]): Called when tokens accepted

        Returns:
            Tuple of (generated_token_ids, stats)
        """
        if self.medusa_num_heads == 0:
            raise ValueError("Model has no Medusa heads - cannot use MTP generation")

        self.eval()
        dtype = self._dtype
        is_cuda = self._device.type == "cuda"

        # Get cached tree buffers (avoids regeneration each call)
        buffers = self._get_tree_buffers(tree_choices, topk)
        retrieve_indices = buffers["retrieve_indices"]
        max_speculation = retrieve_indices.shape[1] - 1

        # Initialize stats
        stats = MTPStats(
            tokens_generated=0,
            forward_passes=0,
            total_proposed=0,
            total_accepted=0,
        )

        # Initialize tokens
        current_tokens = list(input_ids)
        num_generated = 0

        # Use instance scorer if no explicit scorer provided
        scorer = candidate_scorer or self._candidate_scorer

        # Reset scorer state for new generation
        if scorer is not None and hasattr(scorer, 'reset'):
            scorer.reset()

        # Helper to run forward with autocast (CUDA optimized)
        def run_forward(tokens: List[int], return_medusa: bool = True, last_only: bool = True):
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=self._device)
            if is_cuda:
                with torch.amp.autocast('cuda', dtype=dtype):
                    return self.forward(input_tensor, return_medusa=return_medusa, last_only=last_only)
            else:
                return self.forward(input_tensor, return_medusa=return_medusa, last_only=last_only)

        def run_forward_mtp(tokens: List[int], tree_cands: torch.Tensor, bufs: Dict[str, torch.Tensor]):
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=self._device)
            if is_cuda:
                with torch.amp.autocast('cuda', dtype=dtype):
                    return self.forward_mtp(input_tensor, tree_cands, bufs)
            else:
                return self.forward_mtp(input_tensor, tree_cands, bufs)

        # Initial forward pass to get logits
        # With last_only=True: main_logits is (B, 1, vocab), medusa_logits is (num_heads, B, 1, vocab)
        main_logits, medusa_logits = run_forward(current_tokens)
        assert medusa_logits is not None, "Medusa logits should not be None"

        # Squeeze to get: (B, vocab) and (num_heads, B, vocab)
        last_main = main_logits[:, 0, :]
        last_medusa = medusa_logits[:, :, 0, :]

        while num_generated < max_new_tokens:
            # Get retrieval logits for blending if scorer provides them
            retrieval_logits = None
            retrieval_weight = 0.0
            if scorer is not None:
                get_retrieval_fn = getattr(scorer, 'get_retrieval_logits', None)
                if get_retrieval_fn is not None:
                    retrieval_logits = get_retrieval_fn(current_tokens)
                    retrieval_weight = getattr(scorer, 'retrieval_blend_weight', 0.0)

            # Generate candidates (vectorized, fast)
            candidates, tree_candidates = self._generate_candidates(
                last_main, last_medusa, buffers, topk, temperature,
                retrieval_logits, retrieval_weight
            )

            # Apply custom scoring if scorer is provided
            if scorer is not None:
                # Call scorer: (main_logits, medusa_logits, tree_candidates, context) -> scores
                custom_scores = scorer(
                    last_main[0],  # [vocab_size]
                    last_medusa[:, 0, :],  # [num_heads, vocab_size]
                    tree_candidates,  # [tree_len]
                    current_tokens,  # List[int]
                )
                # Store scores for use in candidate evaluation
                buffers["_custom_scores"] = custom_scores

                # Use scores to re-rank candidates - sort by score and take top candidates
                # This influences which path we explore first in verification
                threshold = getattr(scorer, 'score_threshold', None)
                if threshold is not None:
                    # Mask out low-scoring tree positions (set to padding token 0)
                    # This effectively prunes the tree without changing its structure
                    low_score_mask = custom_scores < threshold
                    tree_candidates = tree_candidates.clone()
                    tree_candidates[low_score_mask] = 0  # Padding token

            # Verify candidates with tree attention forward pass
            # Returns tree_logits directly to avoid expensive extraction
            tree_logits, ret_indices, valid_mask = run_forward_mtp(
                current_tokens, tree_candidates, buffers
            )

            stats.forward_passes += 1
            stats.total_proposed += max_speculation

            # Evaluate which candidates to accept (optimized, works on tree_logits directly)
            if temperature == 0.0:
                best_candidate, accept_length = self._evaluate_candidates_greedy_fast(
                    tree_logits, candidates, ret_indices, valid_mask
                )
            else:
                best_candidate, accept_length = self._evaluate_candidates_typical_fast(
                    tree_logits, candidates, ret_indices, valid_mask, temperature,
                    posterior_threshold, posterior_alpha
                )

            # Accept tokens from the best candidate
            accepted_tokens = candidates[best_candidate, : accept_length + 1].tolist()
            stats.total_accepted += len(accepted_tokens)

            # Notify scorer of accepted tokens
            if scorer is not None and hasattr(scorer, 'on_tokens_accepted'):
                scorer.on_tokens_accepted(accepted_tokens)

            # Add accepted tokens
            for token in accepted_tokens:
                if eos_token_id is not None and token == eos_token_id:
                    stats.tokens_generated = num_generated
                    return current_tokens, stats

                current_tokens.append(int(token))
                num_generated += 1

                if num_generated >= max_new_tokens:
                    break

            if num_generated >= max_new_tokens:
                break

            # Update logits for next iteration
            main_logits, medusa_logits = run_forward(current_tokens)
            assert medusa_logits is not None, "Medusa logits should not be None"
            last_main = main_logits[:, 0, :]
            last_medusa = medusa_logits[:, :, 0, :]

        stats.tokens_generated = num_generated
        return current_tokens, stats

    @torch.inference_mode()
    def generate_mtp_with_cache(
        self,
        input_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        topk: int = 10,
        tree_choices: Optional[List[Tuple[int, ...]]] = None,
        posterior_threshold: float = 0.09,
        posterior_alpha: float = 0.3,
        eos_token_id: Optional[int] = None,
        collect_timing: bool = False,
        candidate_scorer: Optional[Callable] = None,
    ) -> Tuple[List[int], MTPStats]:
        """
        Generate tokens using MTP with KV caching for maximum speed.

        This is significantly faster than generate_mtp because:
        1. Initial prompt is processed once and cached
        2. Tree verification only processes tree tokens (not full sequence)
        3. After accepting tokens, only new tokens need forward pass

        Args:
            input_ids: Initial prompt as list of token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            topk: Number of top-k predictions per Medusa head
            tree_choices: Tree structure for speculation. If None, uses optimal tree
                         from checkpoint's head_acc.json. For manual override, pass
                         DEFAULT_TREES[num_heads] or SPARSE_TREES[num_heads].
            posterior_threshold: Typical acceptance hard threshold
            posterior_alpha: Typical acceptance entropy factor
            eos_token_id: EOS token ID for stopping (None = no early stop)
            collect_timing: Whether to collect detailed timing metrics
            candidate_scorer: Optional callback for custom candidate scoring.
                Signature: (main_logits, medusa_logits, tree_candidates, context) -> scores

        Returns:
            Tuple of (generated_token_ids, stats)
        """
        if self.medusa_num_heads == 0:
            raise ValueError("Model has no Medusa heads - cannot use MTP generation")

        self.eval()
        dtype = self._dtype
        is_cuda = self._device.type == "cuda"

        # Get cached tree buffers
        buffers = self._get_tree_buffers(tree_choices, topk)
        retrieve_indices = buffers["retrieve_indices"]
        max_speculation = retrieve_indices.shape[1] - 1

        timing = None
        if collect_timing:
            timing = {
                "prefill_s": 0.0,
                "candidate_s": 0.0,
                "tree_verify_s": 0.0,
                "eval_s": 0.0,
                "kv_update_s": 0.0,
                "compute_logits_s": 0.0,
                "iterations": 0,
                "tree_len": int(buffers["tree_indices"].shape[0]),
                "max_speculation": int(max_speculation),
                "topk": int(topk),
            }

        # Initialize stats
        stats = MTPStats(
            tokens_generated=0,
            forward_passes=0,
            total_proposed=0,
            total_accepted=0,
            timing=timing,
        )

        # Initialize tokens
        current_tokens = list(input_ids)
        num_generated = 0

        # Use instance scorer if no explicit scorer provided
        scorer = candidate_scorer or self._candidate_scorer

        # Reset scorer state for new generation
        if scorer is not None and hasattr(scorer, 'reset'):
            scorer.reset()

        # Process initial prompt and get KV cache
        input_tensor = torch.tensor([current_tokens], dtype=torch.long, device=self._device)

        if collect_timing and is_cuda:
            torch.cuda.synchronize()
        if collect_timing:
            t0 = time.perf_counter()
        if is_cuda:
            with torch.amp.autocast('cuda', dtype=dtype):
                hidden_states, past_key_values = self._get_hidden_states_with_cache(input_tensor)
                main_logits, medusa_logits = self._compute_logits(hidden_states, return_medusa=True, last_only=True)
        else:
            hidden_states, past_key_values = self._get_hidden_states_with_cache(input_tensor)
            main_logits, medusa_logits = self._compute_logits(hidden_states, return_medusa=True, last_only=True)
        if collect_timing:
            if is_cuda:
                torch.cuda.synchronize()
            timing["prefill_s"] += time.perf_counter() - t0

        assert medusa_logits is not None
        # Note: Don't count initial prompt processing as a forward pass for mean_accepted_length
        # We only count speculation verification passes so the metric reflects speculation quality

        # Track current sequence length for position IDs
        current_seq_len = len(current_tokens)

        # Squeeze to get: (B, vocab) and (num_heads, B, vocab)
        last_main = main_logits[:, 0, :]
        last_medusa = medusa_logits[:, :, 0, :]

        while num_generated < max_new_tokens:
            # Get retrieval logits for blending if scorer provides them
            retrieval_logits = None
            retrieval_weight = 0.0
            if scorer is not None:
                get_retrieval_fn = getattr(scorer, 'get_retrieval_logits', None)
                if get_retrieval_fn is not None:
                    retrieval_logits = get_retrieval_fn(current_tokens)
                    retrieval_weight = getattr(scorer, 'retrieval_blend_weight', 0.0)

            # Generate candidates (vectorized, fast)
            if collect_timing and is_cuda:
                torch.cuda.synchronize()
            if collect_timing:
                t0 = time.perf_counter()
            candidates, tree_candidates = self._generate_candidates(
                last_main, last_medusa, buffers, topk, temperature,
                retrieval_logits, retrieval_weight
            )

            # Apply custom scoring if scorer is provided
            if scorer is not None:
                custom_scores = scorer(
                    last_main[0],  # [vocab_size]
                    last_medusa[:, 0, :],  # [num_heads, vocab_size]
                    tree_candidates,  # [tree_len]
                    current_tokens,  # List[int]
                )
                buffers["_custom_scores"] = custom_scores

                # Use scores to prune low-confidence candidates
                threshold = getattr(scorer, 'score_threshold', None)
                if threshold is not None:
                    low_score_mask = custom_scores < threshold
                    tree_candidates = tree_candidates.clone()
                    tree_candidates[low_score_mask] = 0  # Padding token

            if collect_timing:
                if is_cuda:
                    torch.cuda.synchronize()
                timing["candidate_s"] += time.perf_counter() - t0

            # Verify candidates with tree attention using KV cache
            # The tree verification extends the KV cache with tree tokens.
            # After acceptance, we keep only the accepted path's KV entries (no extra forward pass!)
            if collect_timing and is_cuda:
                torch.cuda.synchronize()
            if collect_timing:
                t0 = time.perf_counter()
            if is_cuda:
                with torch.amp.autocast('cuda', dtype=dtype):
                    tree_logits, ret_indices, valid_mask, _, tree_hidden_states = self.forward_mtp_with_cache(
                        tree_candidates, buffers, past_key_values, current_seq_len
                    )
            else:
                tree_logits, ret_indices, valid_mask, _, tree_hidden_states = self.forward_mtp_with_cache(
                    tree_candidates, buffers, past_key_values, current_seq_len
                )
            if collect_timing:
                if is_cuda:
                    torch.cuda.synchronize()
                timing["tree_verify_s"] += time.perf_counter() - t0

            stats.forward_passes += 1
            stats.total_proposed += max_speculation

            # Evaluate which candidates to accept
            if collect_timing and is_cuda:
                torch.cuda.synchronize()
            if collect_timing:
                t0 = time.perf_counter()
            if temperature == 0.0:
                best_candidate, accept_length = self._evaluate_candidates_greedy_fast(
                    tree_logits, candidates, ret_indices, valid_mask
                )
            else:
                best_candidate, accept_length = self._evaluate_candidates_typical_fast(
                    tree_logits, candidates, ret_indices, valid_mask, temperature,
                    posterior_threshold, posterior_alpha
                )
            if collect_timing:
                if is_cuda:
                    torch.cuda.synchronize()
                timing["eval_s"] += time.perf_counter() - t0

            # Accept tokens from the best candidate
            accepted_tokens = candidates[best_candidate, : accept_length + 1].tolist()
            stats.total_accepted += len(accepted_tokens)

            # Notify scorer of accepted tokens
            if scorer is not None and hasattr(scorer, 'on_tokens_accepted'):
                scorer.on_tokens_accepted(accepted_tokens)

            # Check for EOS and add accepted tokens
            should_stop = False
            tokens_to_add = []
            for token in accepted_tokens:
                if eos_token_id is not None and token == eos_token_id:
                    should_stop = True
                    break
                tokens_to_add.append(int(token))
                if num_generated + len(tokens_to_add) >= max_new_tokens:
                    break

            # Get the tree positions for the accepted path (needed for KV cache extraction)
            # ret_indices[best_candidate, 0:num_accepted] gives tree positions for accepted tokens
            num_accepted = len(tokens_to_add)
            accepted_tree_positions = ret_indices[best_candidate, :num_accepted].tolist()

            # CRITICAL OPTIMIZATION: Extract only accepted path's KV entries from tree cache
            # The cache currently has: [original_cache | tree_tokens]
            # We need: [original_cache | accepted_tree_positions]
            # This avoids a redundant forward pass!
            tree_len = tree_candidates.shape[0]
            need_fallback = False  # Track if any layer needs fallback

            if collect_timing and is_cuda:
                torch.cuda.synchronize()
            if collect_timing:
                t0 = time.perf_counter()
            for layer in past_key_values.layers:
                if hasattr(layer, 'keys') and layer.keys is not None:
                    if hasattr(layer, 'sliding_window') and layer.sliding_window is not None:
                        # Sliding window layer
                        sw = layer.sliding_window
                        cache_len_before_tree = min(current_seq_len, sw)

                        # For sliding window, we need to be careful about wrapping
                        # The KV tensor has shape (B, heads, min(total_len, sw), head_dim)
                        total_len_with_tree = current_seq_len + tree_len

                        # Check actual cache size to determine if tree tokens fit
                        cache_size = layer.keys.shape[2]
                        expected_tree_end = cache_len_before_tree + tree_len

                        if expected_tree_end <= cache_size:
                            # No wrapping - use in-place optimization
                            if num_accepted > 0:
                                # Copy accepted positions to their final locations
                                for i, tree_pos in enumerate(accepted_tree_positions):
                                    src_idx = cache_len_before_tree + tree_pos
                                    dst_idx = cache_len_before_tree + i
                                    if src_idx != dst_idx and src_idx < cache_size:
                                        layer.keys[:, :, dst_idx, :] = layer.keys[:, :, src_idx, :]
                                        layer.values[:, :, dst_idx, :] = layer.values[:, :, src_idx, :]
                            # Trim to final size
                            final_len = cache_len_before_tree + num_accepted
                            layer.keys = layer.keys[:, :, :final_len, :]
                            layer.values = layer.values[:, :, :final_len, :]
                        else:
                            # Wrapping case - need fallback for this layer
                            # Truncate back to original and mark for fallback
                            layer.keys = layer.keys[:, :, :cache_len_before_tree, :]
                            layer.values = layer.values[:, :, :cache_len_before_tree, :]
                            need_fallback = True

                        layer.cumulative_length = current_seq_len + num_accepted
                    else:
                        # Regular (non-sliding window) layer - simple extraction
                        # Tree tokens are at positions current_seq_len to current_seq_len + tree_len - 1
                        # Optimization: copy accepted KV entries in-place, then slice to final size
                        if num_accepted > 0:
                            # Copy accepted positions to their final locations
                            for i, tree_pos in enumerate(accepted_tree_positions):
                                src_idx = current_seq_len + tree_pos
                                dst_idx = current_seq_len + i
                                if src_idx != dst_idx:  # Only copy if needed
                                    layer.keys[:, :, dst_idx, :] = layer.keys[:, :, src_idx, :]
                                    layer.values[:, :, dst_idx, :] = layer.values[:, :, src_idx, :]
                        # Trim to final size (this is a view, no allocation)
                        final_len = current_seq_len + num_accepted
                        layer.keys = layer.keys[:, :, :final_len, :]
                        layer.values = layer.values[:, :, :final_len, :]
            if collect_timing:
                if is_cuda:
                    torch.cuda.synchronize()
                timing["kv_update_s"] += time.perf_counter() - t0

            # Update sequence with accepted tokens
            current_tokens.extend(tokens_to_add)
            num_generated += len(tokens_to_add)
            current_seq_len += len(tokens_to_add)

            if should_stop or num_generated >= max_new_tokens:
                break

            # OPTIMIZATION: Reuse hidden state from tree verification for Medusa heads
            # With proper tree attention masking, the hidden state at the last accepted
            # position is computed with the correct causal context (only ancestors).
            # We use this directly for Medusa heads - NO transformer forward pass needed!
            last_accepted_tree_idx = int(ret_indices[best_candidate, accept_length].item())

            # Extract hidden state at the last accepted position
            # tree_hidden_states is (tree_len, hidden_size)
            last_hidden = tree_hidden_states[last_accepted_tree_idx].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden)

            # Compute main logits and Medusa heads from this hidden state
            if collect_timing and is_cuda:
                torch.cuda.synchronize()
            if collect_timing:
                t0 = time.perf_counter()
            if is_cuda:
                with torch.amp.autocast('cuda', dtype=dtype):
                    main_logits, medusa_logits = self._compute_logits(
                        last_hidden, return_medusa=True
                    )
            else:
                main_logits, medusa_logits = self._compute_logits(
                    last_hidden, return_medusa=True
                )
            if collect_timing:
                if is_cuda:
                    torch.cuda.synchronize()
                timing["compute_logits_s"] += time.perf_counter() - t0

            assert medusa_logits is not None
            # main_logits is (1, 1, vocab), medusa_logits is (num_heads, 1, 1, vocab)
            last_main = main_logits[0]  # (1, vocab)
            last_medusa = medusa_logits[:, 0, :, :]  # (num_heads, 1, vocab)

        stats.tokens_generated = num_generated
        return current_tokens, stats

    @torch.inference_mode()
    def generate_standard(
        self,
        input_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[List[int], int]:
        """
        Standard autoregressive generation (no speculation) for comparison.

        Args:
            input_ids: Initial prompt as list of token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            eos_token_id: EOS token ID for stopping

        Returns:
            Tuple of (generated_token_ids, num_forward_passes)
        """
        self.eval()
        dtype = self._dtype
        current_tokens = list(input_ids)
        forward_passes = 0

        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([current_tokens], dtype=torch.long, device=self._device)
            if self._device.type == "cuda":
                with torch.amp.autocast('cuda', dtype=dtype):
                    result = self.forward(input_tensor, return_medusa=False, last_only=True)
            else:
                result = self.forward(input_tensor, return_medusa=False, last_only=True)

            # forward returns logits when return_medusa=False
            logits: torch.Tensor = result  # type: ignore

            forward_passes += 1

            # Get next token (last_only=True gives shape (B, 1, vocab))
            last_logits = logits[0, 0, :]
            if temperature == 0.0:
                next_token = int(last_logits.argmax().item())
            else:
                probs = F.softmax(last_logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

            if eos_token_id is not None and next_token == eos_token_id:
                break

            current_tokens.append(next_token)

        return current_tokens, forward_passes

    @torch.inference_mode()
    def generate_standard_with_cache(
        self,
        input_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[List[int], int]:
        """
        Standard autoregressive generation with KV caching for fair comparison.

        This provides a baseline for comparing MTP speedup, using the same
        KV caching strategy as generate_mtp_with_cache.

        Args:
            input_ids: Initial prompt as list of token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            eos_token_id: EOS token ID for stopping

        Returns:
            Tuple of (generated_token_ids, num_forward_passes)
        """
        self.eval()
        dtype = self._dtype
        is_cuda = self._device.type == "cuda"
        current_tokens = list(input_ids)
        forward_passes = 0

        # Process initial prompt and get KV cache
        input_tensor = torch.tensor([current_tokens], dtype=torch.long, device=self._device)

        if is_cuda:
            with torch.amp.autocast('cuda', dtype=dtype):
                hidden_states, past_key_values = self._get_hidden_states_with_cache(input_tensor)
                logits, _ = self._compute_logits(hidden_states, return_medusa=False, last_only=True)
        else:
            hidden_states, past_key_values = self._get_hidden_states_with_cache(input_tensor)
            logits, _ = self._compute_logits(hidden_states, return_medusa=False, last_only=True)

        forward_passes += 1
        current_seq_len = len(current_tokens)

        # Get first token
        last_logits = logits[0, 0, :]
        if temperature == 0.0:
            next_token = int(last_logits.argmax().item())
        else:
            probs = F.softmax(last_logits / temperature, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())

        if eos_token_id is not None and next_token == eos_token_id:
            return current_tokens, forward_passes

        current_tokens.append(next_token)
        current_seq_len += 1

        # Generate remaining tokens with KV cache
        for _ in range(max_new_tokens - 1):
            # Only process the last token
            new_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self._device)
            new_position_ids = torch.tensor([[current_seq_len - 1]], dtype=torch.long, device=self._device)

            if is_cuda:
                with torch.amp.autocast('cuda', dtype=dtype):
                    hidden_states, past_key_values = self._get_hidden_states_with_cache(
                        new_token_tensor,
                        past_key_values=past_key_values,
                        position_ids=new_position_ids,
                    )
                    logits, _ = self._compute_logits(hidden_states, return_medusa=False)
            else:
                hidden_states, past_key_values = self._get_hidden_states_with_cache(
                    new_token_tensor,
                    past_key_values=past_key_values,
                    position_ids=new_position_ids,
                )
                logits, _ = self._compute_logits(hidden_states, return_medusa=False)

            forward_passes += 1

            # Get next token
            last_logits = logits[0, 0, :]
            if temperature == 0.0:
                next_token = int(last_logits.argmax().item())
            else:
                probs = F.softmax(last_logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

            if eos_token_id is not None and next_token == eos_token_id:
                break

            current_tokens.append(next_token)
            current_seq_len += 1

        return current_tokens, forward_passes

    def eval(self):
        """Set model to evaluation mode and refresh cached weights."""
        self.base_model.eval()
        self.medusa_heads.eval()
        # Refresh cached LoRA weights for inference (they may have changed during training)
        self._cache_stacked_weights()
        return self

    def train(self, mode=True):
        """Set model to training mode."""
        if not any(p.requires_grad for p in self.base_model.parameters()):
            self.base_model.eval()
        else:
            self.base_model.train(mode)
        self.medusa_heads.train(mode)
        return self

    def medusa_parameters(self):
        """Return only Medusa head parameters (for training)."""
        return self.medusa_heads.parameters()

    def setup_optimizers(self, matrix_lr=0.02, proj_lr=0.004, weight_decay=0.2, adam_weight_decay=0.0, adam_betas=(0.8, 0.95)):
        """
        Setup optimizers following nanochat's pattern:
        - Muon for matrix params (ResBlock linears, LoRA A weights)
        - AdamW for projection params (LoRA B weights)

        Args:
            matrix_lr: Learning rate for Muon (matrix parameters)
            proj_lr: Learning rate for AdamW (projection parameters)
            weight_decay: Weight decay for Muon
            adam_betas: Betas for AdamW

        Returns:
            List of [adamw_optimizer, muon_optimizer]
        """
        from functools import partial
        from nanochat.common import get_dist_info, print0
        from nanochat.muon import Muon, DistMuon
        from nanochat.adamw import DistAdamW

        ddp, rank, local_rank, world_size = get_dist_info()
        hidden_size = self._hidden_size

        # Separate Medusa params into matrix (Muon) and projection (AdamW)
        # Muon: Only ResBlock linears (2D matrix params)
        # AdamW: LoRA A, LoRA B (all LoRA params)
        medusa_matrix_params = []
        medusa_proj_params = []

        for head in self.medusa_heads:
            # ResBlock linear weights -> Muon
            for block in head.blocks:
                medusa_matrix_params.append(block.linear.weight)
            # LoRA A and B -> AdamW (user requested LoRA uses AdamW, not Muon)
            if hasattr(head, 'lora_A'):
                medusa_proj_params.append(head.lora_A.weight)
            if hasattr(head, 'lora_B'):
                medusa_proj_params.append(head.lora_B.weight)

        # Add mixer params (supports multiple stacked layers as ModuleLists)
        if self.use_head_mixer:
            if self.mixer_type == "mlp" and self.head_mixer_fc1 is not None:
                # Head mixer params go to AdamW (small projection-like params)
                for fc1 in self.head_mixer_fc1:
                    medusa_proj_params.append(fc1.weight)
                if self.head_mixer_fc2 is not None:
                    for fc2 in self.head_mixer_fc2:
                        medusa_proj_params.append(fc2.weight)
            elif self.mixer_type == "attention" and self.head_attention is not None:
                # Attention mixer params go to AdamW (iterate over stacked blocks)
                for block in self.head_attention.blocks:
                    # Each block has attention and MLP
                    attn = block.attn
                    medusa_proj_params.append(attn.c_q.weight)
                    medusa_proj_params.append(attn.c_k.weight)
                    medusa_proj_params.append(attn.c_v.weight)
                    medusa_proj_params.append(attn.c_proj.weight)
                    # Also add MLP params from the block
                    mlp = block.mlp
                    medusa_proj_params.append(mlp.c_fc.weight)
                    medusa_proj_params.append(mlp.c_proj.weight)
            # Channel mixer is a matrix param -> Muon (used by both MLP and attention)
            if self.channel_mixer_fc is not None:
                if isinstance(self.channel_mixer_fc, nn.ModuleList):
                    for channel_fc in self.channel_mixer_fc:
                        medusa_matrix_params.append(channel_fc.weight)
                else:
                    medusa_matrix_params.append(self.channel_mixer_fc.weight)

        # Collect small params that can't be sharded in DDP (first dim < world_size)
        # These go to a separate non-distributed AdamW
        small_params = []
        if self.use_head_mixer and self.mixer_type == "attention" and self.head_attention is not None:
            # pos_emb has shape (num_heads, hidden_size) which is too small to shard
            small_params.append(self.head_attention.pos_emb)

        # Scale LR by 1/sqrt(hidden_size/768) like nanochat does
        dmodel_lr_scale = (hidden_size / 768) ** -0.5
        print0(f"Scaling AdamW LR by 1/sqrt({hidden_size}/768) = {dmodel_lr_scale:.4f}")

        # AdamW for projection params (distributed if DDP)
        adam_groups = []
        if medusa_proj_params:
            adam_groups.append(dict(params=medusa_proj_params, lr=proj_lr * dmodel_lr_scale))

        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=adam_weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)

        if adam_groups:
            adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        else:
            # Dummy optimizer if no AdamW params
            adamw_optimizer = torch.optim.AdamW([torch.zeros(1, device=self._device)], lr=1e-10)

        # Separate non-distributed AdamW for small params that can't be sharded
        if small_params:
            small_adamw_optimizer = torch.optim.AdamW(
                [dict(params=small_params, lr=proj_lr * dmodel_lr_scale)],
                **adamw_kwargs
            )
        else:
            small_adamw_optimizer = None

        # Muon for matrix params
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon

        if medusa_matrix_params:
            muon_optimizer = MuonFactory(medusa_matrix_params, **muon_kwargs)
        else:
            # Dummy optimizer if no Muon params
            muon_optimizer = torch.optim.SGD([torch.zeros(1, device=self._device)], lr=1e-10)

        optimizers = [adamw_optimizer, muon_optimizer]
        if small_adamw_optimizer is not None:
            optimizers.append(small_adamw_optimizer)

        # Store initial LR for scheduling
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        print0(f"Medusa matrix params (Muon): {sum(p.numel() for p in medusa_matrix_params):,}")
        print0(f"Medusa proj params (AdamW): {sum(p.numel() for p in medusa_proj_params):,}")
        if small_params:
            print0(f"Small params (non-distributed AdamW): {sum(p.numel() for p in small_params):,}")

        return optimizers


def load_gemma_medusa_model(
    model_name: str = "google/gemma-3-1b-it",
    medusa_num_heads: int = 4,
    medusa_num_layers: int = 1,
    lora_rank: int = 64,
    lora_alpha: int | None = None,
    device=None,
    dtype=None,
    freeze_base: bool = True,
    zero_init_mlp: bool = False,
    use_head_mixer: bool = False,
    mixer_hidden: int = 16,
    mixer_num_layers: int = 1,
    mixer_type: str = "mlp",
    attention_head_dim: int | None = None,
    attn_num_layers: int = 1,
    causal_attn: bool = False,
    use_multi_layer: bool = False,
):
    """
    Load a Gemma model with Medusa LoRA heads.

    Args:
        model_name: HuggingFace model name
        medusa_num_heads: Number of Medusa prediction heads
        medusa_num_layers: Number of ResBlock layers per head
        lora_rank: Rank for LoRA projections
        lora_alpha: LoRA alpha scaling (defaults to lora_rank)
        device: Device to load model on
        dtype: Data type for model weights
        freeze_base: Whether to freeze base model parameters
        zero_init_mlp: Whether to zero-initialize ResBlock MLP weights
        use_head_mixer: Whether to use cross-head mixer
        mixer_hidden: Hidden dimension for the cross-head MLP mixer
        mixer_num_layers: Number of mixer layers to stack (default: 1)
        mixer_type: Type of mixer ("mlp" or "attention")
        attention_head_dim: Head dimension for attention mixer (default: min(64, hidden_size))
        attn_num_layers: Number of attention blocks for attention mixer (default: 1)
        causal_attn: Use causal attention for attention mixer (default: bidirectional)
        use_multi_layer: Use multi-layer hidden state fusion (3 layers: 2 evenly spaced + final)

    Returns:
        GemmaMedusaModel instance
    """
    return GemmaMedusaModel(
        model_name=model_name,
        medusa_num_heads=medusa_num_heads,
        medusa_num_layers=medusa_num_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=device,
        dtype=dtype,
        freeze_base=freeze_base,
        zero_init_mlp=zero_init_mlp,
        use_head_mixer=use_head_mixer,
        mixer_hidden=mixer_hidden,
        mixer_num_layers=mixer_num_layers,
        mixer_type=mixer_type,
        attention_head_dim=attention_head_dim,
        attn_num_layers=attn_num_layers,
        causal_attn=causal_attn,
        use_multi_layer=use_multi_layer,
    )
