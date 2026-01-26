"""
EAGLE-3 Draft Model for Gemma3.

Implements the EAGLE-3 speculative decoding draft model adapted for Gemma3 architecture.
Key components:
- Multi-level feature fusion from transformer layers
- Modified decoder layer that takes both embeddings and hidden states
- Tree-based candidate generation
"""

import math
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import GemmaEagleConfig
from nanochat.gemma_common.speculative import build_tree_attention_mask


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embedding to query and key tensors."""
    # cos/sin shape: (batch, seq_len, head_dim) or (1, seq_len, head_dim)
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match query heads for GQA."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0
):
    """Create causal attention mask."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expand attention mask from [bsz, seq_len] to [bsz, 1, tgt_seq_len, src_seq_len]."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMS normalization with (1 + weight) scaling."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaRotaryEmbedding(nn.Module):
    """Rotary position embedding for Gemma."""

    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device)

    def _set_cos_sin_cache(self, seq_len: int, device=None):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32, device=device or self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, position_ids):
        """Return cos and sin for the given positions."""
        if position_ids.max() >= self.max_seq_len_cached:
            self._set_cos_sin_cache(position_ids.max().item() + 1, x.device)

        cos = self.cos_cached[position_ids]  # (batch, seq_len, head_dim)
        sin = self.sin_cached[position_ids]
        return cos, sin


class EagleAttention(nn.Module):
    """
    Multi-headed attention for EAGLE draft model.

    Takes 2x hidden_size input (concatenated embeddings + hidden states).
    """

    def __init__(self, config: GemmaEagleConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Input is 2x hidden_size (concatenated embeddings + hidden states)
        input_size = self.hidden_size * 2
        self.q_proj = nn.Linear(input_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(input_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(input_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK normalization (Gemma3 style)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # QK normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # RoPE
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # GQA: repeat k/v heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class EagleMLP(nn.Module):
    """nanochat-style MLP with ReLU² activation."""

    def __init__(self, config: GemmaEagleConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.c_fc = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.c_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class EagleDecoderLayer(nn.Module):
    """
    EAGLE draft decoder layer for Gemma3.

    Takes both input embeddings and hidden states, concatenates them,
    and processes through attention and MLP.
    """

    def __init__(self, config: GemmaEagleConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = EagleAttention(config)
        self.mlp = EagleMLP(config)

        # Separate normalizations for embeddings and hidden states
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            input_emb: Token embeddings (B, T, hidden_size)
            hidden_states: Fused hidden states from base model (B, T, hidden_size)
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            past_key_value: Cached KV states
            use_cache: Whether to return KV cache

        Returns:
            hidden_states: Output hidden states (B, T, hidden_size)
            past_key_value: Updated KV cache if use_cache=True
        """
        residual = hidden_states

        # Normalize both inputs
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        # Concatenate: (B, T, 2 * hidden_size)
        combined = torch.cat((input_emb, hidden_states), dim=-1)

        # Self-attention
        hidden_states, _, past_key_value = self.self_attn(
            hidden_states=combined,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value


class MultiLayerFusion(nn.Module):
    """
    Fuses hidden states from multiple transformer layers.

    Takes concatenated hidden states from 3 layers and projects
    back to hidden_size with residual connection.
    """

    def __init__(self, hidden_size: int, num_layers: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc = nn.Linear(hidden_size * num_layers, hidden_size, bias=False)

    def forward(self, multi_layer_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multi_layer_hidden: (B, T, num_layers * hidden_size)

        Returns:
            fused: (B, T, hidden_size)
        """
        return self.fc(multi_layer_hidden)


class GemmaEagleModel(nn.Module):
    """
    EAGLE-3 Draft Model for Gemma3.

    Architecture:
        Base Model (frozen):
            input_ids -> Embeddings -> [Layer 0] -> ... -> [Layer N]
                                           |                    |
                                     hidden_0              hidden_N

        Draft Model:
            [hidden_0, hidden_N/2, hidden_N] -> Concat -> FC -> fused_hidden

            For each prediction step:
                fused_hidden + embed(prev_token) -> EagleDecoderLayer -> hidden_out
                hidden_out -> LM Head -> logits -> top-k tokens
    """

    def __init__(
        self,
        config: GemmaEagleConfig,
        base_model_path: Optional[str] = None,
        load_base: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.config = config

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        if dtype is None:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self._device = device
        self._dtype = dtype

        # Load base model
        if load_base:
            base_path = base_model_path or config.base_model_name
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_path,
                dtype=dtype,
                device_map=device if device.type == "cuda" else None,
                trust_remote_code=True,
            )
            if device.type != "cuda":
                self.base_model = self.base_model.to(device)

            # Update config from base model
            hf_config = self.base_model.config
            if hasattr(hf_config, 'text_config'):
                hf_config = hf_config.text_config
            config.hidden_size = hf_config.hidden_size
            config.intermediate_size = hf_config.intermediate_size
            config.num_hidden_layers = hf_config.num_hidden_layers
            config.num_attention_heads = hf_config.num_attention_heads
            config.num_key_value_heads = getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads)
            config.vocab_size = hf_config.vocab_size
            config.head_dim = getattr(hf_config, 'head_dim', config.hidden_size // config.num_attention_heads)

            # Freeze base model
            if config.freeze_base:
                self.freeze_base_model()

            # Share embeddings
            if hasattr(self.base_model, 'model'):
                self.embed_tokens = self.base_model.model.embed_tokens
            else:
                self.embed_tokens = self.base_model.embed_tokens
            for param in self.embed_tokens.parameters():
                param.requires_grad = False

            self.tokenizer = AutoTokenizer.from_pretrained(base_path)
        else:
            self.base_model = None
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.tokenizer = None

        # Multi-layer fusion
        self.multi_layer_indices = config.compute_multi_layer_indices()
        self.fusion = MultiLayerFusion(config.hidden_size, num_layers=len(self.multi_layer_indices))

        # Draft decoder layer
        self.draft_layer = EagleDecoderLayer(config)

        # Output normalization and LM head
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.use_draft_vocab and config.draft_vocab_size < config.vocab_size:
            self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
            # Token mapping buffers (to be populated during training data scan)
            self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long))
            self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool))
        else:
            # Share lm_head with base model
            if self.base_model is not None:
                self.lm_head = self.base_model.lm_head
            else:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Move trainable components to device/dtype
        self.fusion = self.fusion.to(device=device, dtype=dtype)
        self.draft_layer = self.draft_layer.to(device=device, dtype=dtype)
        self.norm = self.norm.to(device=device, dtype=dtype)

        # Tree generation state
        self.tree_mask = None
        self.stable_kv = None

        # Precompute tree buffers for inference
        self._init_tree_buffers()

    def _init_tree_buffers(self):
        """Initialize buffers for tree-based generation."""
        top_k = self.config.draft_top_k
        main_k = self.config.draft_main_k or top_k
        if main_k > top_k:
            main_k = top_k
        device = self._device

        self.tree_mask_init = torch.eye(main_k, device=device)[None, None]
        self.position_ids_buffer = torch.zeros(main_k, device=device, dtype=torch.long)

    def freeze_base_model(self):
        """Freeze all base model parameters."""
        if self.base_model is not None:
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.eval()

    def unfreeze_base_model(self):
        """Unfreeze all base model parameters."""
        if self.base_model is not None:
            for param in self.base_model.parameters():
                param.requires_grad = True

    def get_trainable_parameters(self):
        """Return only trainable parameters (draft model components)."""
        params = []
        params.extend(self.fusion.parameters())
        params.extend(self.draft_layer.parameters())
        params.extend(self.norm.parameters())
        if self.config.use_draft_vocab:
            params.extend(self.lm_head.parameters())
        return params

    def setup_optimizers(self, matrix_lr=0.02, proj_lr=0.004, weight_decay=0.2, adam_weight_decay=0.0, adam_betas=(0.8, 0.95)):
        """
        Setup optimizers following nanochat's pattern:
        - Muon for matrix params (attention projections, MLP weights)
        - AdamW for small/projection params (fusion FC, norm, lm_head if separate)

        Args:
            matrix_lr: Learning rate for Muon (matrix parameters)
            proj_lr: Learning rate for AdamW (projection parameters)
            weight_decay: Weight decay for Muon
            adam_weight_decay: Weight decay for AdamW
            adam_betas: Betas for AdamW

        Returns:
            List of [adamw_optimizer, muon_optimizer]
        """
        from functools import partial
        from nanochat.common import get_dist_info, print0
        from nanochat.muon import Muon, DistMuon
        from nanochat.adamw import DistAdamW

        ddp, rank, local_rank, world_size = get_dist_info()
        hidden_size = self.config.hidden_size

        # Separate params into matrix (Muon) and projection (AdamW)
        # Muon: 2D matrix params in attention and MLP (Q/K/V/O projections, MLP c_fc/c_proj)
        # AdamW: fusion FC, norm weights, lm_head (if separate)
        matrix_params = []
        proj_params = []

        # Draft layer attention - Q/K/V/O projections go to Muon
        attn = self.draft_layer.self_attn
        matrix_params.extend([
            attn.q_proj.weight,
            attn.k_proj.weight,
            attn.v_proj.weight,
            attn.o_proj.weight,
        ])

        # Draft layer MLP - c_fc and c_proj go to Muon
        mlp = self.draft_layer.mlp
        matrix_params.extend([
            mlp.c_fc.weight,
            mlp.c_proj.weight,
        ])

        # Fusion FC goes to Muon (it's a matrix projection)
        matrix_params.append(self.fusion.fc.weight)

        # Norm weights go to AdamW (small params)
        # Draft layer norms
        proj_params.extend([
            self.draft_layer.input_layernorm.weight,
            self.draft_layer.hidden_norm.weight,
            self.draft_layer.post_attention_layernorm.weight,
        ])
        # QK norm weights in attention
        proj_params.extend([
            attn.q_norm.weight,
            attn.k_norm.weight,
        ])
        # Output norm
        proj_params.append(self.norm.weight)

        # lm_head goes to AdamW if we have a separate draft vocab
        if self.config.use_draft_vocab:
            proj_params.append(self.lm_head.weight)

        # Scale LR by 1/sqrt(hidden_size/768) like nanochat does
        dmodel_lr_scale = (hidden_size / 768) ** -0.5
        print0(f"Scaling AdamW LR by 1/sqrt({hidden_size}/768) = {dmodel_lr_scale:.4f}")

        # AdamW for projection params
        adam_groups = []
        if proj_params:
            adam_groups.append(dict(params=proj_params, lr=proj_lr * dmodel_lr_scale))

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

        if matrix_params:
            muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        else:
            # Dummy optimizer if no Muon params
            muon_optimizer = torch.optim.SGD([torch.zeros(1, device=self._device)], lr=1e-10)

        optimizers = [adamw_optimizer, muon_optimizer]

        # Store initial LR for scheduling
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        print0(f"EAGLE matrix params (Muon): {sum(p.numel() for p in matrix_params):,}")
        print0(f"EAGLE proj params (AdamW): {sum(p.numel() for p in proj_params):,}")

        return optimizers

    def get_draft_state_dict(self) -> dict:
        """Get state dict for draft model components only."""
        state = {
            'fusion': self.fusion.state_dict(),
            'draft_layer': self.draft_layer.state_dict(),
            'norm': self.norm.state_dict(),
        }
        if self.config.use_draft_vocab:
            state['lm_head'] = self.lm_head.state_dict()
            state['d2t'] = self.d2t
            state['t2d'] = self.t2d
        return state

    def load_draft_state_dict(self, state: dict):
        """Load state dict for draft model components."""
        self.fusion.load_state_dict(state['fusion'])
        self.draft_layer.load_state_dict(state['draft_layer'])
        self.norm.load_state_dict(state['norm'])
        if 'lm_head' in state and self.config.use_draft_vocab:
            self.lm_head.load_state_dict(state['lm_head'])
        if 'd2t' in state:
            self.d2t.copy_(state['d2t'])
        if 't2d' in state:
            self.t2d.copy_(state['t2d'])

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
    ) -> torch.Tensor:
        """Prepare causal attention mask with optional tree mask."""
        combined_attention_mask = None
        bsz, seq_len = input_shape

        if seq_len > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                torch.float32,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=seq_len)
            expanded_attn_mask = expanded_attn_mask.to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        # Add tree mask for speculative decoding
        if self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
            ] = torch.finfo(torch.float32).min

        return combined_attention_mask

    def _fuse_selected_hidden(
        self,
        all_hidden: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Fuse selected base-model layers into a single hidden representation."""
        selected_hidden = [all_hidden[idx + 1] for idx in self.multi_layer_indices]
        multi_layer_hidden = torch.cat(selected_hidden, dim=-1)
        return self.fusion(multi_layer_hidden)

    def get_base_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get hidden states from base model for multi-layer fusion.

        Returns:
            fused_hidden: (B, T, hidden_size) - fused multi-layer hidden states
            target_logits: (B, T, vocab_size) - base model logits for training
        """
        # Base model forward is no_grad (frozen), but fusion needs gradients
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            # Detach hidden states so fusion layer can compute gradients
            all_hidden = tuple(h.detach() for h in outputs.hidden_states)
            target_logits = outputs.logits.detach()

        # Fusion layer IS trainable - do NOT wrap in no_grad
        fused_hidden = self._fuse_selected_hidden(all_hidden)

        return fused_hidden, target_logits

    @torch.no_grad()
    def get_base_hidden_states_with_cache(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Tuple, Optional[Tuple[torch.Tensor, ...]]]:
        """
        Get base model hidden states with KV cache support.

        Args:
            input_ids: (B, T) input token IDs (or just new tokens when using cache)
            past_key_values: Cached KV states from previous forward pass
            position_ids: Optional position IDs for RoPE (required with cache)
            attention_mask: Optional attention mask (can be 4D for tree attention)
            output_hidden_states: Whether to return all hidden states

        Returns:
            hidden_states: (B, T, hidden_size) for the provided input_ids
            past_key_values: Updated KV cache
            all_hidden_states: Optional tuple of hidden states (n_layers + 1)
        """
        outputs = self.base_model.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        all_hidden = outputs.hidden_states if output_hidden_states else None
        return outputs.last_hidden_state, outputs.past_key_values, all_hidden

    @torch.no_grad()
    def forward_base_with_tree_cache(
        self,
        tree_candidates: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        past_key_values: Tuple,
        base_seq_len: int,
    ) -> Tuple[torch.Tensor, Tuple, torch.Tensor]:
        """
        Forward pass on base model with tree attention using KV cache.

        Args:
            tree_candidates: (1, tree_len) candidate tokens in tree order
            tree_mask: (1, 1, tree_len, tree_len) tree attention mask (0/1)
            tree_position_ids: (tree_len,) depth-based position offsets
            past_key_values: Cached KV states for the prefix (excluding tree tokens)
            base_seq_len: Length of cached prefix (for position ID offset)

        Returns:
            tree_logits: (tree_len, vocab_size) logits for tree positions
            new_past_key_values: Updated KV cache including tree tokens
            tree_fused_hidden: (tree_len, hidden_size) fused hidden states
        """
        tree_len = tree_candidates.shape[1]
        position_ids = base_seq_len + tree_position_ids.unsqueeze(0)

        # Build HF-style attention mask: 0 = attend, -inf = mask
        hf_attn_mask, _ = build_tree_attention_mask(
            tree_mask,
            base_seq_len,
            dtype=self._dtype,
        )

        hidden_states, new_past_key_values, all_hidden_states = self.get_base_hidden_states_with_cache(
            tree_candidates,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=hf_attn_mask,
            output_hidden_states=True,
        )
        tree_logits = self.base_model.lm_head(hidden_states)
        tree_fused_hidden = self._fuse_selected_hidden(all_hidden_states)[0]
        return tree_logits[0], new_past_key_values, tree_fused_hidden

    def forward_draft(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Forward pass through draft model.

        Args:
            hidden_states: Fused hidden states from base model (B, T, hidden_size)
            input_ids: Token IDs for embedding lookup (B, T)
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: KV cache
            use_cache: Whether to return updated cache

        Returns:
            hidden_states: Output hidden states (B, T, hidden_size)
            past_key_values: Updated KV cache if use_cache=True
        """
        batch_size, seq_length, _ = hidden_states.shape
        past_key_values_length = 0

        # Get input embeddings
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length,
                dtype=torch.long, device=hidden_states.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length),
                dtype=torch.bool, device=hidden_states.device
            )

        attention_mask = self._prepare_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        # Draft decoder layer
        past_kv = past_key_values[0] if past_key_values is not None else None
        hidden_states, new_kv = self.draft_layer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_kv,
            use_cache=use_cache,
        )

        if use_cache:
            return hidden_states, (new_kv,)
        return hidden_states, None

    def _compute_chunked_kl_loss(
        self,
        hidden_out: torch.Tensor,
        target_logits: torch.Tensor,
        loss_mask: torch.Tensor,
        chunk_size: int = 128,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute KL divergence loss in chunks to save memory.

        Args:
            hidden_out: Draft model output hidden states (B, T, hidden_size)
            target_logits: Target logits from base model (B, T, vocab_size)
            loss_mask: Mask for which positions to compute loss (B, T)
            chunk_size: Number of positions to process at once

        Returns:
            loss: Scalar loss value
            accuracy: Top-1 accuracy
        """
        batch_size, seq_len, _ = hidden_out.shape
        device = hidden_out.device

        total_loss = torch.tensor(0.0, device=device)
        total_correct = 0
        total_count = loss_mask.sum().item()

        if total_count == 0:
            return total_loss, 0.0

        for t_start in range(0, seq_len, chunk_size):
            t_end = min(t_start + chunk_size, seq_len)

            # Get chunk
            chunk_hidden = hidden_out[:, t_start:t_end, :]
            chunk_target_logits = target_logits[:, t_start:t_end, :]
            chunk_mask = loss_mask[:, t_start:t_end]

            # Compute draft logits for this chunk
            chunk_logits = self.lm_head(self.norm(chunk_hidden))

            # Compute softmax/log_softmax for this chunk only
            with torch.no_grad():
                chunk_target_p = F.softmax(chunk_target_logits, dim=-1)
                chunk_target_max = chunk_target_logits.argmax(dim=-1)

            chunk_log_p = F.log_softmax(chunk_logits.float(), dim=-1)

            # KL loss for this chunk: -sum(p * log(q))
            # Weight by mask and sum
            chunk_kl = -torch.sum(chunk_target_p * chunk_log_p, dim=-1)  # (B, chunk)
            chunk_loss = (chunk_kl * chunk_mask).sum()
            total_loss = total_loss + chunk_loss

            # Accuracy for this chunk
            with torch.no_grad():
                chunk_pred = chunk_logits.argmax(dim=-1)
                chunk_correct = ((chunk_pred == chunk_target_max) * chunk_mask).sum().item()
                total_correct += chunk_correct

        # Average loss over all valid positions
        mean_loss = total_loss / total_count
        accuracy = total_correct / total_count

        return mean_loss, accuracy

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        num_steps: int = 7,
        use_chunked_loss: bool = True,
        chunk_size: int = 128,
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Training forward pass with multi-step prediction.

        Args:
            input_ids: Input token IDs (B, T)
            attention_mask: Attention mask
            loss_mask: Mask for which positions to compute loss (B, T)
            num_steps: Number of prediction steps (default: 7 like EAGLE-3)
            use_chunked_loss: If True, compute KL loss in chunks to save memory
            chunk_size: Chunk size for chunked loss computation

        Returns:
            losses: List of losses for each prediction step
            accuracies: List of top-1 accuracies for each step
        """
        device = input_ids.device
        batch_size, seq_length = input_ids.shape

        # Get base model hidden states and targets
        fused_hidden, target_logits = self.get_base_hidden_states(input_ids, attention_mask)

        # Shift targets for next-token prediction
        def shift_left(tensor):
            """Shift tensor left by 1, padding with zeros on right."""
            zero_pad = torch.zeros_like(tensor[:, -1:])
            return torch.cat((tensor[:, 1:], zero_pad), dim=1)

        target_logits = shift_left(target_logits)
        shifted_ids = shift_left(input_ids)

        # Prepare loss mask
        if loss_mask is None:
            loss_mask = torch.ones(batch_size, seq_length, device=device)
        loss_mask = shift_left(loss_mask.float())

        losses = []
        accuracies = []

        hidden_states = fused_hidden
        current_ids = shifted_ids

        for step in range(num_steps):
            # Draft model forward
            hidden_out, _ = self.forward_draft(
                hidden_states=hidden_states,
                input_ids=current_ids,
                attention_mask=attention_mask,
            )

            if use_chunked_loss:
                # Memory-efficient chunked loss computation
                loss, acc = self._compute_chunked_kl_loss(
                    hidden_out, target_logits, loss_mask, chunk_size
                )
            else:
                # Original full computation (more memory intensive)
                logits = self.lm_head(self.norm(hidden_out))

                with torch.no_grad():
                    target_p = F.softmax(target_logits, dim=-1)
                    target_max = target_logits.argmax(dim=-1)

                log_p = F.log_softmax(logits.float(), dim=-1)
                loss = -torch.sum(loss_mask.unsqueeze(-1) * target_p * log_p, dim=-1).mean()

                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    correct = (pred == target_max) * loss_mask
                    acc = correct.sum().item() / (loss_mask.sum().item() + 1e-6)

            losses.append(loss)
            accuracies.append(acc)

            # Prepare for next step (shift everything left)
            if step < num_steps - 1:
                hidden_states = hidden_out
                current_ids = shift_left(current_ids)
                target_logits = shift_left(target_logits)
                loss_mask = shift_left(loss_mask)

        return losses, accuracies

    def reset_kv(self):
        """Reset KV cache for new generation."""
        self.stable_kv = None
        self.tree_mask = None

    @torch.no_grad()
    def topk_generate(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        logits_processor=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate tree of candidate tokens using top-k beam search.

        Args:
            hidden_states: Fused hidden states (B, T, hidden_size)
            input_ids: Current token IDs including the sampled token (B, T+1)

        Returns:
            draft_tokens: Candidate tokens (1, total_tokens+1)
            retrieve_indices: Indices for extracting candidate paths
            tree_mask: Attention mask for tree structure
            tree_position_ids: Position IDs for tree nodes
        """
        total_tokens = self.config.total_tokens
        depth = self.config.draft_depth
        top_k = self.config.draft_top_k
        main_k = self.config.draft_main_k or top_k
        if main_k > top_k:
            main_k = top_k
        extra_k = top_k - main_k
        device = hidden_states.device

        sample_token = input_ids[:, -1]
        # input_ids: [BOS, t1, ..., tn, sample] has length T+1
        # hidden_states: might have length T (initial) or T+1 (after re-fetch of full input_ids)
        #
        # Training alignment:
        #   - shifted_ids = [t1, t2, ..., tn, 0]
        #   - fused_hidden = [h0, h1, ..., hn-1] where h0 is hidden after BOS
        #   - Position i: hidden[i] is used to predict token at shifted_ids[i]
        #
        # For inference, we want the same alignment:
        #   - input_ids for draft: [t1, t2, ..., tn, sample] (tokens after BOS, including new sample)
        #   - hidden_states: [h0, h1, ..., hn] (keep aligned with input positions before sample)
        #
        # This ensures position i has hidden[i] (context up to position i) paired with
        # input_ids[i] (the next token), matching training.
        draft_input_ids = input_ids[:, 1:]  # Remove BOS: [t1, ..., tn, sample]

        # Ensure hidden_states and draft_input_ids have matching lengths
        # We want to keep them aligned: hidden[i] predicts input[i]
        min_len = min(draft_input_ids.shape[1], hidden_states.shape[1])
        draft_input_ids = draft_input_ids[:, :min_len]
        hidden_states = hidden_states[:, :min_len]

        input_ids = draft_input_ids

        len_posi = input_ids.shape[1]
        self.tree_mask = None

        # Initial forward through draft model
        if self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            print(f"DEBUG: kv_len={kv_len}, slicing hidden_states[:, {kv_len}:] and input_ids[:, {kv_len}:]")
            out_hidden, past_key_values = self.forward_draft(
                hidden_states[:, kv_len:], input_ids[:, kv_len:],
                past_key_values=self.stable_kv, use_cache=True
            )
        else:
            out_hidden, past_key_values = self.forward_draft(
                hidden_states, input_ids, use_cache=True
            )
        self.stable_kv = past_key_values

        last_hidden = out_hidden[:, -1:]
        last_logits = self.lm_head(self.norm(last_hidden))
        last_p = F.log_softmax(last_logits, dim=-1)

        top = torch.topk(last_p[0, 0], top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        cand_tokens: List[torch.Tensor] = []
        cand_scores: List[torch.Tensor] = []
        cand_parents: List[int] = []

        main_tokens = topk_index[:main_k]
        main_scores = topk_p[:main_k]
        for j in range(main_k):
            cand_tokens.append(main_tokens[j])
            cand_scores.append(main_scores[j])
            cand_parents.append(0)

        if extra_k > 0:
            extra_tokens = topk_index[main_k:main_k + extra_k]
            extra_scores = topk_p[main_k:main_k + extra_k]
            for j in range(extra_tokens.numel()):
                cand_tokens.append(extra_tokens[j])
                cand_scores.append(extra_scores[j])
                cand_parents.append(0)

        beam_tokens = main_tokens
        beam_scores = main_scores
        beam_node_indices = torch.arange(1, main_k + 1, device=device, dtype=torch.long)

        input_ids = beam_tokens[None]
        input_hidden = last_hidden.repeat(1, main_k, 1)
        tree_mask = self.tree_mask_init.clone()

        # Expand tree through depth levels
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = (len_posi + self.position_ids_buffer).unsqueeze(0)

            out_hidden, past_key_values = self.forward_draft(
                input_hidden, input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True
            )
            len_posi += 1

            # Get logits and top-k
            last_logits = self.lm_head(self.norm(out_hidden))
            last_p = F.log_softmax(last_logits, dim=-1)
            top = torch.topk(last_p[0], top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            # Cumulative scores
            cu_scores = topk_p + beam_scores[:, None]

            # Main candidates per beam (expandable)
            main_tokens = topk_index[:, :main_k]
            main_scores = cu_scores[:, :main_k]
            main_node_indices = torch.empty((main_k, main_k), device=device, dtype=torch.long)
            for b in range(main_k):
                parent_idx = int(beam_node_indices[b].item())
                for j in range(main_k):
                    cand_tokens.append(main_tokens[b, j])
                    cand_scores.append(main_scores[b, j])
                    cand_parents.append(parent_idx)
                    main_node_indices[b, j] = len(cand_tokens)

            # Extra leaf-only candidates from global top-(top_k - main_k)
            if extra_k > 0:
                extra_scores = cu_scores[:, main_k:]
                extra_tokens = topk_index[:, main_k:]
                flat_extra = extra_scores.reshape(-1)
                if flat_extra.numel() > 0:
                    extra_top = torch.topk(flat_extra, min(extra_k, flat_extra.numel()))
                    extra_flat_idx = extra_top.indices
                    extra_parents = extra_flat_idx // extra_k
                    extra_children = extra_flat_idx % extra_k
                    for idx in range(extra_flat_idx.numel()):
                        parent_idx = int(beam_node_indices[extra_parents[idx]].item())
                        token = extra_tokens[extra_parents[idx], extra_children[idx]]
                        score = extra_scores[extra_parents[idx], extra_children[idx]]
                        cand_tokens.append(token)
                        cand_scores.append(score)
                        cand_parents.append(parent_idx)

            # Select next beams from main candidates only
            flat_main = main_scores.reshape(-1)
            next_top = torch.topk(flat_main, main_k, dim=-1)
            next_flat_idx = next_top.indices
            out_ids = next_flat_idx // main_k
            child_ids = next_flat_idx % main_k
            beam_scores = flat_main[next_flat_idx]
            beam_tokens = main_tokens[out_ids, child_ids]
            beam_node_indices = main_node_indices[out_ids, child_ids]

            # Update for next iteration
            input_hidden = out_hidden[:, out_ids]
            input_ids = beam_tokens.unsqueeze(0)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

        # Build final tree structure
        cand_tokens_tensor = torch.stack(cand_tokens)
        cand_scores_tensor = torch.stack(cand_scores)
        cand_parents_tensor = torch.tensor(cand_parents, device=device, dtype=torch.long)

        num_select = min(total_tokens, cand_tokens_tensor.numel())
        top_scores = torch.topk(cand_scores_tensor, num_select, dim=-1)
        top_scores_index = torch.sort(top_scores.indices).values

        draft_tokens = cand_tokens_tensor[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        # Build parent/mask indices
        draft_parents = cand_parents_tensor[top_scores_index]
        selected_pre_indices = top_scores_index + 1
        mask_index = torch.searchsorted(selected_pre_indices, draft_parents, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()

        # Build tree attention mask
        tree_mask = torch.eye(num_select + 1, dtype=torch.bool, device=device)
        tree_mask[:, 0] = True
        for i in range(num_select):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask.int(), dim=1) - 1

        # Build retrieve indices
        max_depth = tree_position_ids.max().item() + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = num_select - noleaf_num

        retrieve_indices = torch.full(
            (leaf_num, max_depth), -1, dtype=torch.long, device=device
        )
        position_ids_list = tree_position_ids.tolist()

        rid = 0
        for i in range(num_select + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1] if cid > 0 else 0
                rid += 1

        # Sort retrieve indices
        maxitem = num_select + 5
        def custom_sort(lst):
            return [x if x >= 0 else maxitem for x in lst.tolist()]

        sorted_indices = sorted(range(len(retrieve_indices)),
                               key=lambda i: custom_sort(retrieve_indices[i]))
        retrieve_indices = retrieve_indices[sorted_indices]

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids
