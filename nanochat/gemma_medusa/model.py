"""
Model wrapper for Gemma 3 models.

Adapts HuggingFace Gemma model to match nanochat's model interface.
Includes Medusa MTP support with batched LM head for maximum speed.

Tree attention generation:
- Supports speculative decoding with tree-structured candidate verification
- forward_mtp: Single forward pass with tree attention for candidate verification
- generate_mtp: Full generation loop with Medusa speculation
"""

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig

from .config import GemmaConfigWrapper, GemmaMedusaConfig
from .heads import MedusaLoRAHead, MedusaResBlock


@dataclass
class MTPStats:
    """Statistics from MTP generation for benchmarking."""
    tokens_generated: int
    forward_passes: int
    total_proposed: int
    total_accepted: int

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
        """Approximate speedup vs standard autoregressive decoding."""
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


def get_default_tree_choices(num_heads: int, topk: int = 10) -> List[Tuple[int, ...]]:
    """Generate default tree configuration based on number of Medusa heads."""
    choices = []

    if num_heads >= 1:
        for i in range(min(topk, 10)):
            choices.append((i,))

    if num_heads >= 2:
        for i in range(min(topk, 5)):
            for j in range(min(topk, 5)):
                choices.append((i, j))

    if num_heads >= 3:
        for i in range(min(topk, 3)):
            for j in range(min(topk, 3)):
                for k in range(min(topk, 3)):
                    choices.append((i, j, k))

    if num_heads >= 4:
        for i in range(min(topk, 2)):
            for j in range(min(topk, 2)):
                for k in range(min(topk, 2)):
                    for m in range(min(topk, 2)):
                        choices.append((i, j, k, m))

    return choices


def get_sparse_tree_choices(num_heads: int) -> List[Tuple[int, ...]]:
    """Generate a sparse tree configuration optimized for speed."""
    choices = []

    for i in range(min(3, num_heads and 3)):
        choices.append((i,))

    if num_heads >= 2:
        for i in range(2):
            for j in range(2):
                choices.append((i, j))

    if num_heads >= 3:
        choices.extend([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)])

    if num_heads >= 4:
        choices.append((0, 0, 0, 0))

    return choices


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
    ):
        super().__init__()
        self.model_name = model_name
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.zero_init_mlp = zero_init_mlp

        # Determine device and dtype
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self._device = device
        self._dtype = dtype

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
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size

        # Create Medusa LoRA heads
        self.medusa_heads = nn.ModuleList([
            MedusaLoRAHead(hidden_size, vocab_size, medusa_num_layers, lora_rank,
                          lora_alpha=self.lora_alpha, zero_init_mlp=zero_init_mlp)
            for _ in range(medusa_num_heads)
        ])
        # Move heads to device and dtype
        self.medusa_heads = self.medusa_heads.to(device=device, dtype=dtype)

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

    @property
    def config(self):
        return self._config

    def get_device(self) -> torch.device:
        return self._device

    def freeze_base_model(self):
        """Freeze all base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        """Unfreeze all base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = True

    def get_medusa_param_count(self) -> int:
        """Return number of trainable Medusa parameters."""
        return sum(p.numel() for p in self.medusa_heads.parameters())

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

    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get final hidden states from the transformer (before lm_head).

        Args:
            input_ids: (B, T) input token IDs

        Returns:
            hidden_states: (B, T, hidden_size)
        """
        outputs = self.base_model.model(
            input_ids=input_ids,
            output_hidden_states=False,
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

        return hidden_states

    def _get_hidden_states_with_cache(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Get hidden states with KV cache support.

        Args:
            input_ids: (B, T) input token IDs (or just new tokens if using cache)
            past_key_values: Cached key-value states from previous forward pass
            position_ids: Optional position IDs for RoPE (required when using cache)

        Returns:
            hidden_states: (B, T, hidden_size) or (B, new_tokens, hidden_size)
            new_past_key_values: Updated KV cache
        """
        outputs = self.base_model.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
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
    ):
        """
        Compute main logits and Medusa logits efficiently with batched matmuls.

        Architecture for LoRA heads:
        - Main: lm_head(h) -> (B, T, vocab)
        - Each head: lm_head(h) + scaling * lora_B(lora_A(ResBlocks(h)))

        Optimization: We batch the lora_B projections (rank -> vocab) into a single matmul.
        This is the expensive operation since vocab_size >> rank.

        Args:
            hidden_states: (B, T, hidden_size) from transformer
            return_medusa: Whether to compute Medusa logits
            last_only: If True, only compute logits for the last token position.
                       This is much faster for generation where we only need [:, -1, :].

        Returns:
            main_logits: (B, T, vocab_size) or (B, 1, vocab_size) if last_only
            medusa_logits: (num_heads, B, T, vocab_size) or (num_heads, B, 1, vocab_size) or None
        """
        # For generation, we only need the last position's logits
        if last_only:
            hidden_states = hidden_states[:, -1:, :]  # (B, 1, hidden_size)

        if not return_medusa or len(self.medusa_heads) == 0:
            return self.base_model.lm_head(hidden_states), None  # (B, T, vocab) or (B, 1, vocab)

        num_heads = len(self.medusa_heads)

        # Step 1: Compute ResBlocks for each head (sequential to preserve gradients during training)
        resblock_outputs = []
        for head_idx, head in enumerate(self.medusa_heads):
            x = hidden_states
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

        return base_logits[0], medusa_logits

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache=None,
        loss_reduction: str = 'mean',
        return_medusa: bool = False,
        last_only: bool = False,
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

        Returns:
            If targets is None:
                logits or (logits, medusa_logits)
            If targets provided:
                loss or (loss, medusa_losses)
        """
        if kv_cache is not None:
            raise NotImplementedError("KV cache not yet supported for GemmaMedusaModel")

        # Get hidden states from transformer
        hidden_states = self._get_hidden_states(input_ids)

        # Compute logits (optionally only for last position during generation)
        main_logits, medusa_logits = self._compute_logits(hidden_states, return_medusa, last_only)

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
    _tree_buffers_config: Optional[Tuple[int, bool]] = None

    def _get_tree_buffers(
        self,
        topk: int = 10,
        use_sparse_tree: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Get tree attention buffers, using cache if config matches.

        Args:
            topk: Number of top predictions from each head
            use_sparse_tree: Use smaller tree for faster but less accurate speculation

        Returns:
            Dictionary of tree buffers
        """
        config = (topk, use_sparse_tree)
        if self._tree_buffers_cache is None or self._tree_buffers_config != config:
            if use_sparse_tree:
                choices = get_sparse_tree_choices(self.medusa_num_heads)
            else:
                choices = get_default_tree_choices(self.medusa_num_heads, topk)
            self._tree_buffers_cache = generate_tree_buffers(choices, self._device, topk)
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

        # Get top-k from each Medusa head: (num_heads, topk)
        medusa_topk = torch.topk(medusa_logits[:, 0, :], topk, dim=-1).indices

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
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
        """
        retrieve_indices = buffers["retrieve_indices"]
        tree_position_ids = buffers["tree_position_ids"]

        # Prepare inputs - only process tree tokens (KV cache has prompt)
        tree_input = tree_candidates.unsqueeze(0)  # (1, tree_len)

        # Position IDs: base_seq_len + tree position offsets
        # tree_position_ids contains depth-based offsets (0, 1, 1, 1, 2, 2, ...)
        position_ids = base_seq_len + tree_position_ids.unsqueeze(0)  # (1, tree_len)

        # Forward pass with cache - only processes tree tokens
        hidden_states, new_past_key_values = self._get_hidden_states_with_cache(
            tree_input,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Compute logits (only for tree positions)
        logits, _ = self._compute_logits(hidden_states, return_medusa=False)

        # tree_logits is (1, tree_len, vocab) -> (tree_len, vocab)
        tree_logits = logits[0]
        valid_mask = retrieve_indices >= 0

        return tree_logits, retrieve_indices, valid_mask, new_past_key_values

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
        use_sparse_tree: bool = False,
        posterior_threshold: float = 0.09,
        posterior_alpha: float = 0.3,
        eos_token_id: Optional[int] = None,
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
            use_sparse_tree: Use smaller tree for faster speculation
            posterior_threshold: Typical acceptance hard threshold
            posterior_alpha: Typical acceptance entropy factor
            eos_token_id: EOS token ID for stopping (None = no early stop)

        Returns:
            Tuple of (generated_token_ids, stats)
        """
        if self.medusa_num_heads == 0:
            raise ValueError("Model has no Medusa heads - cannot use MTP generation")

        self.eval()
        dtype = self._dtype
        is_cuda = self._device.type == "cuda"

        # Get cached tree buffers (avoids regeneration each call)
        buffers = self._get_tree_buffers(topk, use_sparse_tree)
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
            # Generate candidates (vectorized, fast)
            candidates, tree_candidates = self._generate_candidates(
                last_main, last_medusa, buffers, topk, temperature
            )

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
        use_sparse_tree: bool = False,
        posterior_threshold: float = 0.09,
        posterior_alpha: float = 0.3,
        eos_token_id: Optional[int] = None,
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
            use_sparse_tree: Use smaller tree for faster speculation
            posterior_threshold: Typical acceptance hard threshold
            posterior_alpha: Typical acceptance entropy factor
            eos_token_id: EOS token ID for stopping (None = no early stop)

        Returns:
            Tuple of (generated_token_ids, stats)
        """
        if self.medusa_num_heads == 0:
            raise ValueError("Model has no Medusa heads - cannot use MTP generation")

        self.eval()
        dtype = self._dtype
        is_cuda = self._device.type == "cuda"

        # Get cached tree buffers
        buffers = self._get_tree_buffers(topk, use_sparse_tree)
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

        # Process initial prompt and get KV cache
        input_tensor = torch.tensor([current_tokens], dtype=torch.long, device=self._device)

        if is_cuda:
            with torch.amp.autocast('cuda', dtype=dtype):
                hidden_states, past_key_values = self._get_hidden_states_with_cache(input_tensor)
                main_logits, medusa_logits = self._compute_logits(hidden_states, return_medusa=True, last_only=True)
        else:
            hidden_states, past_key_values = self._get_hidden_states_with_cache(input_tensor)
            main_logits, medusa_logits = self._compute_logits(hidden_states, return_medusa=True, last_only=True)

        assert medusa_logits is not None
        stats.forward_passes += 1

        # Track current sequence length for position IDs
        current_seq_len = len(current_tokens)

        # Squeeze to get: (B, vocab) and (num_heads, B, vocab)
        last_main = main_logits[:, 0, :]
        last_medusa = medusa_logits[:, :, 0, :]

        while num_generated < max_new_tokens:
            # Generate candidates (vectorized, fast)
            candidates, tree_candidates = self._generate_candidates(
                last_main, last_medusa, buffers, topk, temperature
            )

            # Verify candidates with tree attention using KV cache
            if is_cuda:
                with torch.amp.autocast('cuda', dtype=dtype):
                    tree_logits, ret_indices, valid_mask, _ = self.forward_mtp_with_cache(
                        tree_candidates, buffers, past_key_values, current_seq_len
                    )
            else:
                tree_logits, ret_indices, valid_mask, _ = self.forward_mtp_with_cache(
                    tree_candidates, buffers, past_key_values, current_seq_len
                )

            stats.forward_passes += 1
            stats.total_proposed += max_speculation

            # Evaluate which candidates to accept
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

            # Update sequence with accepted tokens
            current_tokens.extend(tokens_to_add)
            num_generated += len(tokens_to_add)
            current_seq_len += len(tokens_to_add)

            if should_stop or num_generated >= max_new_tokens:
                break

            # Update KV cache and get new logits for accepted tokens
            # Only process the newly accepted tokens
            new_tokens_tensor = torch.tensor([tokens_to_add], dtype=torch.long, device=self._device)
            # Position IDs for the new tokens
            new_position_ids = torch.arange(
                current_seq_len - len(tokens_to_add),
                current_seq_len,
                dtype=torch.long,
                device=self._device
            ).unsqueeze(0)

            if is_cuda:
                with torch.amp.autocast('cuda', dtype=dtype):
                    hidden_states, past_key_values = self._get_hidden_states_with_cache(
                        new_tokens_tensor,
                        past_key_values=past_key_values,
                        position_ids=new_position_ids,
                    )
                    main_logits, medusa_logits = self._compute_logits(
                        hidden_states, return_medusa=True, last_only=True
                    )
            else:
                hidden_states, past_key_values = self._get_hidden_states_with_cache(
                    new_tokens_tensor,
                    past_key_values=past_key_values,
                    position_ids=new_position_ids,
                )
                main_logits, medusa_logits = self._compute_logits(
                    hidden_states, return_medusa=True, last_only=True
                )

            assert medusa_logits is not None
            last_main = main_logits[:, 0, :]
            last_medusa = medusa_logits[:, :, 0, :]

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

        # Scale LR by 1/sqrt(hidden_size/768) like nanochat does
        dmodel_lr_scale = (hidden_size / 768) ** -0.5
        print0(f"Scaling AdamW LR by 1/sqrt({hidden_size}/768) = {dmodel_lr_scale:.4f}")

        # AdamW for projection params
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

        # Muon for matrix params
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon

        if medusa_matrix_params:
            muon_optimizer = MuonFactory(medusa_matrix_params, **muon_kwargs)
        else:
            # Dummy optimizer if no Muon params
            muon_optimizer = torch.optim.SGD([torch.zeros(1, device=self._device)], lr=1e-10)

        optimizers = [adamw_optimizer, muon_optimizer]

        # Store initial LR for scheduling
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        print0(f"Medusa matrix params (Muon): {sum(p.numel() for p in medusa_matrix_params):,}")
        print0(f"Medusa proj params (AdamW): {sum(p.numel() for p in medusa_proj_params):,}")

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
    )
