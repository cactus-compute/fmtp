"""
Learned Retrieval Module for HST speculation.

This module predicts likely next tokens from TOKEN IDs (not hidden states).
This design allows it to work directly with MTP head outputs during tree construction.

Key components:
- RetrievalMLP: 3-layer MLP with tied SVD embeddings (~17M params with svd_rank=64)
- RetrievalModuleTiny: Ultra-light alternative for minimal overhead
- SVD utilities: Compute and load compressed vocabulary matrices

Architecture (with tied SVD embeddings):
    Input: Last K token IDs [B, K]
      ↓
    SVD Embedding Lookup: svd_embedding[token_ids]  # TRAINABLE, tied with output
      → [B, K, svd_rank]
      ↓
    Flatten: [B, K * svd_rank]
      ↓
    3-Layer MLP:
      - Linear(K * svd_rank → hidden_dim) + GELU
      - Linear(hidden_dim → hidden_dim) + GELU
      - Linear(hidden_dim → svd_rank)
      ↓
    Output: h @ svd_embedding.T → [B, vocab_size]  # TIED with input embedding

Key Design Decisions:
    1. Token IDs as input: MTP heads output token IDs, not hidden states
    2. Tied SVD embeddings: Same embedding for input lookup and output projection
    3. Trainable: SVD embedding is initialized from precomputed SVD, then fine-tuned
    4. Compact: vocab_size × svd_rank params instead of vocab_size × embed_dim
    5. Simple MLP: 3-layer MLP is ~3x faster than MLP-Mixer on CPU (memory-bound)

Parameter count (Gemma 262k vocab, svd_rank=64, hidden_dim=128, K=4):
    - SVD embedding: 262k × 64 = 16.8M params (trainable)
    - Layer 1: (4 * 64) × 128 = 32k params
    - Layer 2: 128 × 128 = 16k params
    - Layer 3: 128 × 64 = 8k params
    - Total: ~17M params (~56k excluding embedding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

# Default SVD artifacts directory
SVD_CACHE_DIR = Path.home() / ".cache" / "nanochat" / "svd"


class RetrievalMLP(nn.Module):
    """
    3-layer MLP retrieval module with tied SVD embeddings.

    This module predicts likely next tokens from the last K TOKEN IDs.
    Uses a TRAINABLE SVD-compressed embedding that is TIED between input and output:
    - Input: token_ids -> svd_embedding[token_ids] -> [B, K, svd_rank] -> flatten
    - Output: h_proj @ svd_embedding.T -> [B, vocab_size]

    Key design: Takes TOKEN IDs (not hidden states) because:
    - MTP heads output token IDs, not hidden states
    - Allows independent operation during tree construction

    Architecture:
    - Flatten K token embeddings: [B, K, svd_rank] -> [B, K * svd_rank]
    - 3-layer MLP with GELU: K*svd_rank -> hidden_dim -> hidden_dim -> svd_rank
    - Output via tied embedding: [B, svd_rank] @ [svd_rank, vocab_size]

    This is ~3x faster than MLP-Mixer on CPU because:
    - Flattened input means contiguous memory access
    - No transpose operations
    - Larger matrix multiplications have better cache utilization
    - Since we're memory-bound on the output projection anyway, extra MLP
      compute is essentially free

    Total parameters:
    - SVD embedding: vocab_size × svd_rank (initialized from precomputed SVD, trainable)
    - MLP: ~56k params (for hidden_dim=128, K=4, svd_rank=64)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        context_window: int = 4,
        svd_rank: int = 64,
    ):
        """
        Args:
            vocab_size: Vocabulary size (must match SVD)
            hidden_dim: Hidden dimension for MLP layers (default 128)
            context_window: Number of recent tokens to use (K)
            svd_rank: Rank for SVD compression (64 recommended)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        self.svd_rank = svd_rank

        # Tied SVD embedding - TRAINABLE, used for both input lookup and output projection
        # Shape: [vocab_size, svd_rank]
        # Initialized from precomputed SVD, then fine-tuned
        self.svd_embedding = nn.Parameter(torch.zeros(vocab_size, svd_rank))
        self._svd_initialized = False

        # 3-layer MLP: flatten(K * svd_rank) -> hidden -> hidden -> svd_rank
        input_dim = context_window * svd_rank
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, svd_rank, bias=False)

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
                       Can be [B] or [B, <K] - will be padded to context_window
            return_hidden: If True, also return hidden state before projection

        Returns:
            logits: [B, vocab_size] next token logits
            hidden: [B, hidden_dim] (only if return_hidden=True)
        """
        if not self._svd_initialized:
            raise RuntimeError(
                "SVD embedding not initialized. "
                "Call load_svd() before forward."
            )

        # Embed token IDs using tied SVD embedding
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(1)  # [B] -> [B, 1]

        B, K = token_ids.shape

        # Pad or truncate to context_window
        if K < self.context_window:
            # Pad with zeros on the left (older positions)
            padding = torch.zeros(
                B, self.context_window - K,
                device=token_ids.device,
                dtype=token_ids.dtype
            )
            token_ids = torch.cat([padding, token_ids], dim=1)
        elif K > self.context_window:
            # Take most recent tokens
            token_ids = token_ids[:, -self.context_window:]

        # Lookup in SVD embedding: [B, K, svd_rank]
        svd_embeds = self.svd_embedding[token_ids]

        # Flatten: [B, K * svd_rank]
        h = svd_embeds.reshape(B, -1)

        # 3-layer MLP with GELU
        h = F.gelu(self.fc1(h))
        h = F.gelu(self.fc2(h))
        h_proj = self.fc3(h)  # [B, svd_rank]

        # Compute logits via TIED SVD embedding (same embedding used for input)
        # [B, svd_rank] @ [svd_rank, vocab_size] -> [B, vocab_size]
        logits = h_proj @ self.svd_embedding.T

        if return_hidden:
            return logits, h
        return logits

    def forward_topk(
        self,
        token_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fast inference: compute logits only for specific candidate tokens.

        Instead of projecting to full vocabulary (262k tokens), this method
        only computes logits for the specified candidates. This provides
        86-98x speedup when scoring MTP head candidates.

        Args:
            token_ids: [B, K] context token IDs (will be padded/truncated to context_window)
            candidate_ids: [num_candidates] or [B, num_candidates] candidate token IDs

        Returns:
            logits: [B, num_candidates] logits for candidate tokens only
        """
        if not self._svd_initialized:
            raise RuntimeError(
                "SVD embedding not initialized. "
                "Call load_svd() before forward_topk."
            )

        # Handle input dimensions
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)  # [K] -> [1, K]

        B, K = token_ids.shape

        # Pad or truncate to context_window
        if K < self.context_window:
            padding = torch.zeros(
                B, self.context_window - K,
                device=token_ids.device,
                dtype=token_ids.dtype
            )
            token_ids = torch.cat([padding, token_ids], dim=1)
        elif K > self.context_window:
            token_ids = token_ids[:, -self.context_window:]

        # Lookup in SVD embedding: [B, K, svd_rank]
        svd_embeds = self.svd_embedding[token_ids]

        # Flatten: [B, K * svd_rank]
        h = svd_embeds.reshape(B, -1)

        # 3-layer MLP with GELU
        h = F.gelu(self.fc1(h))
        h = F.gelu(self.fc2(h))
        h_proj = self.fc3(h)  # [B, svd_rank]

        # Project only to candidate embeddings (the key optimization)
        if candidate_ids.dim() == 1:
            # Same candidates for all batch items: [num_candidates]
            candidate_embeds = self.svd_embedding[candidate_ids]  # [num_candidates, svd_rank]
            # [B, svd_rank] @ [svd_rank, num_candidates] -> [B, num_candidates]
            logits = h_proj @ candidate_embeds.T
        else:
            # Different candidates per batch item: [B, num_candidates]
            candidate_embeds = self.svd_embedding[candidate_ids]  # [B, num_candidates, svd_rank]
            # Batched dot product: [B, svd_rank] x [B, num_candidates, svd_rank] -> [B, num_candidates]
            logits = torch.einsum('br,bnr->bn', h_proj, candidate_embeds)

        return logits


class RetrievalRNN(nn.Module):
    """
    RNN-based retrieval module with tied SVD embeddings.

    Uses a GRU to encode the token sequence, capturing sequential dependencies
    that the MLP (which just flattens) cannot model. The SVD embedding trick
    keeps the output projection efficient.

    Architecture:
        Input: Last K token IDs [B, K]
          ↓
        SVD Embedding Lookup: svd_embedding[token_ids]  # TRAINABLE, tied with output
          → [B, K, svd_rank]
          ↓
        GRU: processes sequence, outputs final hidden state
          → [B, hidden_dim]
          ↓
        Output projection: Linear → [B, svd_rank]
          ↓
        Output: h @ svd_embedding.T → [B, vocab_size]  # TIED with input embedding

    Why GRU over LSTM:
    - Faster (fewer gates: 3 vs 4)
    - Often similar or better performance for short sequences
    - K=8 context is quite short, don't need LSTM's longer-term memory

    Parameter count (Gemma 262k vocab, svd_rank=64, hidden_dim=128):
        - SVD embedding: 262k × 64 = 16.8M params (shared with MLP)
        - GRU: 3 * (svd_rank + hidden_dim) * hidden_dim = ~73k params (1-layer)
        - Output: hidden_dim × svd_rank = 8k params
        - Total: ~16.9M params
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        context_window: int = 8,
        svd_rank: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """
        Args:
            vocab_size: Vocabulary size (must match SVD)
            hidden_dim: GRU hidden dimension
            context_window: Number of recent tokens to use (K)
            svd_rank: Rank for SVD compression (64 recommended)
            num_layers: Number of GRU layers (1 is usually enough for K=8)
            dropout: Dropout between GRU layers (only if num_layers > 1)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        self.svd_rank = svd_rank
        self.num_layers = num_layers

        # Tied SVD embedding - TRAINABLE, used for both input lookup and output projection
        self.svd_embedding = nn.Parameter(torch.zeros(vocab_size, svd_rank))
        self._svd_initialized = False

        # GRU encoder - processes token sequence
        self.gru = nn.GRU(
            input_size=svd_rank,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection: hidden_dim -> svd_rank (for tied output)
        self.output_proj = nn.Linear(hidden_dim, svd_rank, bias=False)

    def load_svd(self, compressed_vocab: torch.Tensor) -> None:
        """Initialize SVD embedding from precomputed SVD-compressed vocabulary."""
        if compressed_vocab.shape[0] != self.vocab_size:
            raise ValueError(
                f"Vocab size mismatch: expected {self.vocab_size}, "
                f"got {compressed_vocab.shape[0]}"
            )
        if compressed_vocab.shape[1] != self.svd_rank:
            raise ValueError(
                f"SVD rank mismatch: expected {self.svd_rank}, "
                f"got {compressed_vocab.shape[1]}"
            )
        self.svd_embedding.data.copy_(compressed_vocab)
        self._svd_initialized = True

    def forward(
        self,
        token_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Predict next token logits from last K token IDs.

        Args:
            token_ids: [B, K] or [B,] last K token IDs
            return_hidden: If True, also return hidden state

        Returns:
            logits: [B, vocab_size] next token logits
            hidden: [B, hidden_dim] if return_hidden=True
        """
        if not self._svd_initialized:
            raise RuntimeError(
                "SVD embedding not initialized. "
                "Call load_svd() with precomputed vocabulary before forward."
            )

        # Handle input dimensions
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)  # [K] -> [1, K]

        B, K = token_ids.shape

        # Pad or truncate to context_window
        if K < self.context_window:
            padding = torch.zeros(
                B, self.context_window - K,
                device=token_ids.device,
                dtype=token_ids.dtype
            )
            token_ids = torch.cat([padding, token_ids], dim=1)
        elif K > self.context_window:
            token_ids = token_ids[:, -self.context_window:]

        # Lookup in SVD embedding: [B, K, svd_rank]
        svd_embeds = self.svd_embedding[token_ids]

        # GRU forward: get final hidden state
        # output: [B, K, hidden_dim], h_n: [num_layers, B, hidden_dim]
        _, h_n = self.gru(svd_embeds)

        # Use final layer's hidden state: [B, hidden_dim]
        h = h_n[-1]

        # Project to SVD rank: [B, svd_rank]
        h_proj = self.output_proj(h)

        # Compute logits via TIED SVD embedding
        logits = h_proj @ self.svd_embedding.T

        if return_hidden:
            return logits, h
        return logits

    def forward_topk(
        self,
        token_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fast inference: compute logits only for specific candidate tokens.

        Args:
            token_ids: [B, K] context token IDs
            candidate_ids: [num_candidates] or [B, num_candidates] candidate token IDs

        Returns:
            logits: [B, num_candidates] logits for candidate tokens only
        """
        if not self._svd_initialized:
            raise RuntimeError("SVD embedding not initialized.")

        # Handle input dimensions
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        B, K = token_ids.shape

        # Pad or truncate
        if K < self.context_window:
            padding = torch.zeros(
                B, self.context_window - K,
                device=token_ids.device,
                dtype=token_ids.dtype
            )
            token_ids = torch.cat([padding, token_ids], dim=1)
        elif K > self.context_window:
            token_ids = token_ids[:, -self.context_window:]

        # Lookup and GRU forward
        svd_embeds = self.svd_embedding[token_ids]
        _, h_n = self.gru(svd_embeds)
        h = h_n[-1]
        h_proj = self.output_proj(h)

        # Project only to candidate embeddings
        if candidate_ids.dim() == 1:
            candidate_embeds = self.svd_embedding[candidate_ids]
            logits = h_proj @ candidate_embeds.T
        else:
            candidate_embeds = self.svd_embedding[candidate_ids]
            logits = torch.einsum('br,bnr->bn', h_proj, candidate_embeds)

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
