"""
Learned Retrieval Module for HST speculation.

This module predicts likely next tokens from TOKEN IDs (not hidden states).
This design allows it to work directly with MTP head outputs during tree construction.

Key components:
- RetrievalMixer: MLP-Mixer with tied SVD embeddings (~17M params with svd_rank=64)
- RetrievalModuleTiny: Ultra-light alternative for minimal overhead
- SVD utilities: Compute and load compressed vocabulary matrices

Architecture (with tied SVD embeddings):
    Input: Last K token IDs [B, K]
      ↓
    SVD Embedding Lookup: svd_embedding[token_ids]  # TRAINABLE, tied with output
      → [B, K, svd_rank]
      ↓
    Input Projection: Linear(svd_rank → embed_dim)
      → [B, K, embed_dim]
      ↓
    MLP-Mixer Blocks (1-2 layers):
      - Token mixing: Linear(K → 2K → K) across positions
      - Channel mixing: Linear(embed_dim → embed_dim//4 → embed_dim)
      - GELU activation, LayerNorm between blocks
      ↓
    Mean pooling: [B, K, embed_dim] → [B, embed_dim]
      ↓
    Down projection: Linear(embed_dim → svd_rank)
      → [B, svd_rank]
      ↓
    Output: h @ svd_embedding.T → [B, vocab_size]  # TIED with input embedding

Key Design Decisions:
    1. Token IDs as input: MTP heads output token IDs, not hidden states
    2. Tied SVD embeddings: Same embedding for input lookup and output projection
    3. Trainable: SVD embedding is initialized from precomputed SVD, then fine-tuned
    4. Compact: vocab_size × svd_rank params instead of vocab_size × embed_dim

Parameter count (Gemma 262k vocab, svd_rank=64, embed_dim=256):
    - SVD embedding: 262k × 64 = 16.8M params (trainable)
    - Input projection: 64 × 256 = 16k params
    - MLP-Mixer: ~70k params
    - Down projection: 256 × 64 = 16k params
    - Total: ~17M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Literal

# Default SVD artifacts directory
SVD_CACHE_DIR = Path.home() / ".cache" / "nanochat" / "svd"


class MLPMixerBlock(nn.Module):
    """
    Single MLP-Mixer block with token mixing and channel mixing.

    Token mixing: exchanges information across the sequence dimension (K positions)
    Channel mixing: exchanges information across the embedding dimension

    Both use residual connections for stable training.
    """

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        token_hidden_mult: int = 2,
        channel_hidden_div: int = 4,
    ):
        """
        Args:
            seq_len: Context window size K (number of input tokens)
            embed_dim: Embedding dimension
            token_hidden_mult: Multiplier for token mixing hidden dim (default 2x)
            channel_hidden_div: Divisor for channel mixing hidden dim (default /4)
        """
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Token mixing: K → 2K → K
        token_hidden = seq_len * token_hidden_mult
        self.token_norm = nn.LayerNorm(embed_dim)
        self.token_fc1 = nn.Linear(seq_len, token_hidden)
        self.token_fc2 = nn.Linear(token_hidden, seq_len)

        # Channel mixing: embed_dim → embed_dim//4 → embed_dim
        channel_hidden = embed_dim // channel_hidden_div
        self.channel_norm = nn.LayerNorm(embed_dim)
        self.channel_fc1 = nn.Linear(embed_dim, channel_hidden)
        self.channel_fc2 = nn.Linear(channel_hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, K, embed_dim] input embeddings

        Returns:
            [B, K, embed_dim] mixed embeddings
        """
        # Token mixing (across positions)
        # Transpose: [B, K, D] -> [B, D, K] for mixing across K
        y = self.token_norm(x)
        y = y.transpose(1, 2)  # [B, D, K]
        y = self.token_fc1(y)
        y = F.gelu(y)
        y = self.token_fc2(y)
        y = y.transpose(1, 2)  # [B, K, D]
        x = x + y  # Residual

        # Channel mixing (across embedding dimension)
        y = self.channel_norm(x)
        y = self.channel_fc1(y)
        y = F.gelu(y)
        y = self.channel_fc2(y)
        x = x + y  # Residual

        return x


class RetrievalMixer(nn.Module):
    """
    MLP-Mixer retrieval module with tied SVD embeddings.

    This module predicts likely next tokens from the last K TOKEN IDs.
    Uses a TRAINABLE SVD-compressed embedding that is TIED between input and output:
    - Input: token_ids -> svd_embedding[token_ids] -> [B, K, svd_rank]
    - Output: h_proj @ svd_embedding.T -> [B, vocab_size]

    Key design: Takes TOKEN IDs (not hidden states) because:
    - MTP heads output token IDs, not hidden states
    - Allows independent operation during tree construction

    Total parameters (~100k-200k):
    - SVD embedding: vocab_size × svd_rank (initialized from precomputed SVD, trainable)
    - Input projection: svd_rank × embed_dim
    - MLP-Mixer: ~50k-100k
    - Down projection: embed_dim × svd_rank

    The SVD embedding is initialized from precomputed SVD of the base model's
    embedding matrix, then fine-tuned during training.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        context_window: int = 4,
        num_layers: int = 2,
        svd_rank: int = 64,
        token_hidden_mult: int = 2,
        channel_hidden_div: int = 4,
        input_mode: Literal["last_k", "last_1", "avg_k", "weighted_k"] = "last_k",
    ):
        """
        Args:
            vocab_size: Vocabulary size (must match SVD)
            embed_dim: Internal embedding dimension for MLP-Mixer processing
            context_window: Number of recent tokens to use (K)
            num_layers: Number of MLP-Mixer blocks
            svd_rank: Rank for SVD compression (64 recommended)
            token_hidden_mult: Multiplier for token mixing hidden dim
            channel_hidden_div: Divisor for channel mixing hidden dim
            input_mode: How to process input tokens:
                - "last_k": Use all K tokens with mixer
                - "last_1": Use only the last token
                - "avg_k": Average embeddings of K tokens
                - "weighted_k": Learned weighted sum of K token embeddings
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_window = context_window
        self.svd_rank = svd_rank
        self.input_mode = input_mode

        # Tied SVD embedding - TRAINABLE, used for both input lookup and output projection
        # Shape: [vocab_size, svd_rank]
        # Initialized from precomputed SVD, then fine-tuned
        self.svd_embedding = nn.Parameter(torch.zeros(vocab_size, svd_rank))
        self._svd_initialized = False

        # Project from svd_rank to embed_dim for MLP-Mixer processing
        self.input_proj = nn.Linear(svd_rank, embed_dim, bias=False)

        # MLP-Mixer blocks
        if input_mode == "last_k":
            self.mixer_blocks = nn.ModuleList([
                MLPMixerBlock(
                    seq_len=context_window,
                    embed_dim=embed_dim,
                    token_hidden_mult=token_hidden_mult,
                    channel_hidden_div=channel_hidden_div,
                )
                for _ in range(num_layers)
            ])
        else:
            self.mixer_blocks = None

        # Learned weights for weighted_k mode
        if input_mode == "weighted_k":
            self.position_weights = nn.Parameter(torch.ones(context_window))
        else:
            self.position_weights = None

        # Final layer norm before projection
        self.final_norm = nn.LayerNorm(embed_dim)

        # Down projection back to SVD rank space for output
        self.down_proj = nn.Linear(embed_dim, svd_rank, bias=False)

    def load_svd(self, compressed_vocab: torch.Tensor) -> None:
        """
        Initialize SVD embedding from precomputed SVD-compressed vocabulary.

        Args:
            compressed_vocab: [vocab_size, svd_rank] matrix from SVD decomposition
        """
        if compressed_vocab.shape != (self.vocab_size, self.svd_rank):
            raise ValueError(
                f"Expected compressed_vocab shape ({self.vocab_size}, {self.svd_rank}), "
                f"got {compressed_vocab.shape}"
            )
        with torch.no_grad():
            self.svd_embedding.copy_(compressed_vocab)
        self._svd_initialized = True

    def forward(
        self,
        token_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next token logits from context token IDs.

        Args:
            token_ids: [B, K] last K token IDs (integers)
                       or [B] if input_mode is "last_1"
            return_hidden: If True, also return hidden state before projection

        Returns:
            logits: [B, vocab_size] next token logits
            hidden: [B, embed_dim] (only if return_hidden=True)
        """
        if not self._svd_initialized:
            raise RuntimeError(
                "SVD embedding not initialized. "
                "Call load_svd() before forward."
            )

        # Embed token IDs using tied SVD embedding
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(1)  # [B] -> [B, 1]

        # Lookup in SVD embedding and project up
        svd_embeds = self.svd_embedding[token_ids]  # [B, K, svd_rank]
        embeddings = self.input_proj(svd_embeds)  # [B, K, embed_dim]

        # Process based on mode
        if self.input_mode == "last_1":
            h = embeddings[:, -1, :]  # Take last token embedding

        elif self.input_mode == "avg_k":
            h = embeddings.mean(dim=1)  # [B, embed_dim]

        elif self.input_mode == "weighted_k":
            weights = F.softmax(self.position_weights, dim=0)  # [K]
            # Pad weights if needed
            if embeddings.shape[1] < len(weights):
                weights = weights[-embeddings.shape[1]:]
            elif embeddings.shape[1] > len(weights):
                embeddings = embeddings[:, -len(weights):, :]
            h = torch.einsum("bke,k->be", embeddings, weights)  # [B, embed_dim]

        else:  # "last_k" - full MLP-Mixer
            # Pad if needed to match context_window
            B, K, D = embeddings.shape
            if K < self.context_window:
                padding = torch.zeros(
                    B, self.context_window - K, D,
                    device=embeddings.device,
                    dtype=embeddings.dtype
                )
                embeddings = torch.cat([padding, embeddings], dim=1)
            elif K > self.context_window:
                embeddings = embeddings[:, -self.context_window:, :]

            # Apply MLP-Mixer blocks
            x = embeddings
            for block in self.mixer_blocks:
                x = block(x)

            # Mean pooling across sequence
            h = x.mean(dim=1)  # [B, embed_dim]

        # Final norm
        h = self.final_norm(h)

        # Project to SVD rank space
        h_proj = self.down_proj(h)  # [B, svd_rank]

        # Compute logits via TIED SVD embedding (same embedding used for input)
        # [B, svd_rank] @ [svd_rank, vocab_size] -> [B, vocab_size]
        logits = h_proj @ self.svd_embedding.T

        if return_hidden:
            return logits, h
        return logits


class RetrievalModuleTiny(nn.Module):
    """
    Ultra-light retrieval module for minimal overhead.

    Architecture: token_id → embed → down_proj → svd_output

    No MLP-Mixer, just embedding + projection. Useful for:
    - Testing/debugging
    - Extremely latency-sensitive applications
    - Baseline comparison

    Total parameters: vocab_size × embed_dim + embed_dim × svd_rank
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        svd_rank: int = 64,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.svd_rank = svd_rank

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.down_proj = nn.Linear(embed_dim, svd_rank, bias=False)

        self.register_buffer(
            "compressed_vocab",
            torch.zeros(vocab_size, svd_rank),
            persistent=False
        )
        self._svd_loaded = False

    def load_svd(self, compressed_vocab: torch.Tensor) -> None:
        """Load precomputed SVD-compressed vocabulary."""
        if compressed_vocab.shape != (self.vocab_size, self.svd_rank):
            raise ValueError(
                f"Expected shape ({self.vocab_size}, {self.svd_rank}), "
                f"got {compressed_vocab.shape}"
            )
        self.compressed_vocab.copy_(compressed_vocab)
        self._svd_loaded = True

    def forward(self, token_id: torch.Tensor) -> torch.Tensor:
        """
        Predict next token logits from single token ID.

        Args:
            token_id: [B] last token ID (integer)

        Returns:
            logits: [B, vocab_size] next token logits
        """
        if not self._svd_loaded:
            raise RuntimeError("SVD not loaded. Call load_svd() first.")

        h = self.embedding(token_id)  # [B, embed_dim]
        h = self.norm(h)
        h_proj = self.down_proj(h)
        return h_proj @ self.compressed_vocab.T


def compute_svd_basis(
    embedding_weight: torch.Tensor,
    rank: int = 64,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute SVD-compressed vocabulary matrix from embedding weights.

    Performs truncated SVD on the embedding matrix and returns the
    compressed vocabulary for efficient logit computation.

    Args:
        embedding_weight: [vocab_size, embed_dim] embedding matrix
        rank: SVD rank (64 or 128 recommended)
        device: Device for computation

    Returns:
        compressed_vocab: [vocab_size, rank] matrix where
                         compressed_vocab = U[:, :rank] @ diag(S[:rank])
    """
    embedding_weight = embedding_weight.to(device).float()

    # Compute full SVD (for large matrices, consider randomized SVD)
    # embedding_weight: [vocab_size, embed_dim]
    # U: [vocab_size, min(vocab_size, embed_dim)]
    # S: [min(vocab_size, embed_dim)]
    # Vt: [min(vocab_size, embed_dim), embed_dim]
    U, S, Vt = torch.linalg.svd(embedding_weight, full_matrices=False)

    # Truncate to rank
    U_k = U[:, :rank]  # [vocab_size, rank]
    S_k = S[:rank]     # [rank]

    # Compressed vocab: U[:, :k] @ diag(S[:k])
    # This is [vocab_size, rank]
    compressed_vocab = U_k * S_k.unsqueeze(0)  # Broadcasting: [V, k] * [1, k]

    return compressed_vocab


def load_svd_basis(
    rank: int = 64,
    model_name: str = "gemma",
    cache_dir: Optional[Path] = None,
) -> torch.Tensor:
    """
    Load precomputed SVD-compressed vocabulary from cache.

    Args:
        rank: SVD rank (must match precomputed file)
        model_name: Model identifier (e.g., "gemma")
        cache_dir: Directory containing SVD files (default: ~/.cache/nanochat/svd/)

    Returns:
        compressed_vocab: [vocab_size, rank] tensor

    Raises:
        FileNotFoundError: If SVD file doesn't exist
    """
    if cache_dir is None:
        cache_dir = SVD_CACHE_DIR

    cache_dir = Path(cache_dir)
    svd_path = cache_dir / f"{model_name}_svd_{rank}.pt"

    if not svd_path.exists():
        raise FileNotFoundError(
            f"SVD file not found at {svd_path}. "
            f"Run scripts/hst_compute_svd.py to generate it."
        )

    return torch.load(svd_path, weights_only=True)


def save_svd_basis(
    compressed_vocab: torch.Tensor,
    rank: int,
    model_name: str = "gemma",
    cache_dir: Optional[Path] = None,
) -> Path:
    """
    Save computed SVD-compressed vocabulary to cache.

    Args:
        compressed_vocab: [vocab_size, rank] tensor
        rank: SVD rank used
        model_name: Model identifier
        cache_dir: Directory for SVD files

    Returns:
        Path to saved file
    """
    if cache_dir is None:
        cache_dir = SVD_CACHE_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    svd_path = cache_dir / f"{model_name}_svd_{rank}.pt"
    torch.save(compressed_vocab, svd_path)

    return svd_path
