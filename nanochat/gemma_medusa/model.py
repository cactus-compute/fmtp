"""
Model wrapper for Gemma 3 models.

Adapts HuggingFace Gemma model to match nanochat's model interface.
Includes Medusa MTP support with batched LM head for maximum speed.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig

from .config import GemmaConfigWrapper, GemmaMedusaConfig
from .heads import MedusaLoRAHead, MedusaResBlock


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
                    torch_dtype=dtype,
                    device_map=device,
                    trust_remote_code=True,
                )
            except ValueError:
                # Fallback if accelerate is not available
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                ).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
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
        device=None,
        dtype=None,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.lora_rank = lora_rank

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
                    torch_dtype=dtype,
                    device_map=device,
                    trust_remote_code=True,
                )
            except ValueError:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                ).to(device)
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
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
            MedusaLoRAHead(hidden_size, vocab_size, medusa_num_layers, lora_rank)
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
        return outputs.last_hidden_state

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

        # Step 1: Compute ResBlocks for each head (sequential due to data dependency)
        resblock_outputs = []
        for head in self.medusa_heads:
            x = hidden_states
            for block in head.blocks:
                x = block(x)
            resblock_outputs.append(x)  # (B, T, hidden_size)

        # Step 2: Stack ResBlock outputs and do batched lora_A projection
        stacked_resblock = torch.stack(resblock_outputs, dim=0)  # (num_heads, B, T, hidden)
        # Use pre-cached lora_A weights: (num_heads, rank, hidden)
        stacked_lora_a = torch.einsum('hbti,hri->hbtr', stacked_resblock, self._stacked_lora_a)

        # Step 3: Batched lora_B projection using pre-cached weights
        # (num_heads, B, T, rank) @ (num_heads, vocab, rank).T -> (num_heads, B, T, vocab)
        lora_deltas = torch.einsum('hbtr,hvr->hbtv', stacked_lora_a, self._stacked_lora_b)

        # Step 4: Apply pre-cached per-head scaling
        lora_deltas = lora_deltas * self._scalings.view(num_heads, 1, 1, 1)

        # Step 5: Batched lm_head projection for main + all heads
        all_hiddens = torch.cat([hidden_states.unsqueeze(0), stacked_resblock], dim=0)  # (num_heads+1, B, T, hidden)
        base_logits = self.base_model.lm_head(all_hiddens)  # (num_heads+1, B, T, vocab)
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

            # Main loss
            main_loss = F.cross_entropy(
                main_logits.view(-1, vocab_size),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )

            if return_medusa and medusa_logits is not None:
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
                    medusa_losses.append(head_loss)

                return main_loss, medusa_losses

            return main_loss

        else:
            # Inference mode
            if return_medusa:
                return main_logits, medusa_logits
            return main_logits

    def eval(self):
        """Set model to evaluation mode."""
        self.base_model.eval()
        self.medusa_heads.eval()
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


def load_gemma_medusa_model(
    model_name: str = "google/gemma-3-1b-it",
    medusa_num_heads: int = 4,
    medusa_num_layers: int = 1,
    lora_rank: int = 64,
    device=None,
    dtype=None,
    freeze_base: bool = True,
):
    """
    Load a Gemma model with Medusa LoRA heads.

    Args:
        model_name: HuggingFace model name
        medusa_num_heads: Number of Medusa prediction heads
        medusa_num_layers: Number of ResBlock layers per head
        lora_rank: Rank for LoRA projections
        device: Device to load model on
        dtype: Data type for model weights
        freeze_base: Whether to freeze base model parameters

    Returns:
        GemmaMedusaModel instance
    """
    return GemmaMedusaModel(
        model_name=model_name,
        medusa_num_heads=medusa_num_heads,
        medusa_num_layers=medusa_num_layers,
        lora_rank=lora_rank,
        device=device,
        dtype=dtype,
        freeze_base=freeze_base,
    )
