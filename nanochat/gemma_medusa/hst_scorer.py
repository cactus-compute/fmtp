"""
HST (Hybrid Smoothed Tree) Scorer for GemmaMedusaModel.

This module provides HST-based candidate re-ranking and tree pruning for
speculative decoding. It can be attached to a GemmaMedusaModel to enhance
speculation quality using:

1. Learned retrieval (RetrievalMLP with K=8 context window)
2. Suffix matching (captures in-context repetition patterns)
3. Hybrid scoring (additive combination with tunable weights)

Usage:
    from nanochat.gemma_medusa import GemmaMedusaModel
    from nanochat.gemma_medusa.hst_scorer import HSTScorer

    model = GemmaMedusaModel(...)
    scorer = HSTScorer(
        vocab_size=model.config.vocab_size,
        retrieval_checkpoint="path/to/retrieval.pt",
    )
    model.set_hst_scorer(scorer)

    # Now generate_mtp will use HST scoring
    tokens, stats = model.generate_mtp(input_ids, ...)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from nanochat.hst.retrieval import RetrievalMLP, load_svd_basis
from nanochat.hst.suffix_match import SuffixMatcher


class HSTScorer:
    """
    HST-based candidate scorer for Medusa speculative decoding.

    Provides hybrid scoring that combines:
    - MTP head predictions (from Medusa heads)
    - Learned retrieval predictions (RetrievalMLP)
    - Suffix matching predictions (for in-context repetition)

    Scoring formula:
        S(token) = α × P_mtp + β × P_retrieval + γ × P_suffix

    With agreement bonus when multiple sources predict the same token.
    """

    def __init__(
        self,
        vocab_size: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
        # Retrieval configuration
        retrieval_checkpoint: Optional[str] = None,
        retrieval_context_window: int = 8,  # K=8 based on ablation
        retrieval_hidden_dim: int = 128,
        svd_rank: int = 64,
        # Suffix matching configuration
        suffix_buffer_size: int = 1024,
        suffix_max_len: int = 4,
        # Scoring weights (use hst_tune_weights.py to optimize)
        alpha: float = 0.6,   # MTP head weight
        beta: float = 0.3,    # Retrieval weight
        gamma: float = 0.1,   # Suffix match weight
        agreement_bonus: float = 1.5,  # Multiplier when sources agree
        # Pruning configuration
        score_threshold: float = 0.01,  # Minimum score to keep candidate
    ):
        """
        Initialize the HST scorer.

        Args:
            vocab_size: Vocabulary size (must match model)
            device: Device for computation
            dtype: Data type for computation
            retrieval_checkpoint: Path to trained RetrievalMLP checkpoint
            retrieval_context_window: Context window size (K=8 recommended)
            retrieval_hidden_dim: Hidden dimension for retrieval MLP
            svd_rank: SVD rank for compressed vocabulary projection
            suffix_buffer_size: Size of suffix matching context buffer
            suffix_max_len: Maximum suffix length to match
            alpha: Weight for MTP head predictions
            beta: Weight for retrieval predictions
            gamma: Weight for suffix match predictions
            agreement_bonus: Score multiplier when sources agree
            score_threshold: Minimum hybrid score to keep a candidate
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.device = device
        self.dtype = dtype
        self.vocab_size = vocab_size

        # Scoring weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.agreement_bonus = agreement_bonus
        self.score_threshold = score_threshold

        # Context window for retrieval
        self.retrieval_context_window = retrieval_context_window

        # Initialize retrieval module
        self.retrieval_module = RetrievalMLP(
            vocab_size=vocab_size,
            hidden_dim=retrieval_hidden_dim,
            context_window=retrieval_context_window,
            svd_rank=svd_rank,
        )

        # Load SVD basis
        try:
            compressed_vocab = load_svd_basis(rank=svd_rank, model_name="gemma")
            self.retrieval_module.load_svd(compressed_vocab)
        except (FileNotFoundError, ValueError):
            # Initialize randomly for testing if SVD not found or vocab size mismatch
            self.retrieval_module.svd_embedding.data.normal_(0, 0.01)
            self.retrieval_module._svd_initialized = True

        # Load checkpoint if provided
        if retrieval_checkpoint and Path(retrieval_checkpoint).exists():
            state_dict = torch.load(retrieval_checkpoint, weights_only=True)
            self.retrieval_module.load_state_dict(state_dict)

        self.retrieval_module = self.retrieval_module.to(device=device, dtype=dtype)
        self.retrieval_module.eval()

        # Initialize suffix matcher
        self.suffix_matcher = SuffixMatcher(
            buffer_size=suffix_buffer_size,
            max_suffix_len=suffix_max_len,
            device=str(device),
        )

        # Track whether we're enabled
        self._enabled = True

    def reset(self) -> None:
        """Reset state for new generation (clears suffix buffer)."""
        self.suffix_matcher.reset()

    def append_context(self, token_ids: List[int]) -> None:
        """
        Append tokens to the suffix matching context buffer.

        Should be called after tokens are accepted during generation.

        Args:
            token_ids: List of token IDs to append
        """
        self.suffix_matcher.append(token_ids)

    @torch.inference_mode()
    def score_candidates(
        self,
        mtp_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        context_tokens: List[int],
        speculative_path: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Score candidate tokens using hybrid HST scoring.

        This is the fast path that uses forward_topk() to only compute
        retrieval scores for the specific candidates, not full vocabulary.

        Args:
            mtp_logits: [vocab_size] MTP head logits for this position
            candidate_ids: [num_candidates] Candidate token IDs to score
            context_tokens: Full context (committed tokens)
            speculative_path: Speculative tokens already in path (for sliding window)

        Returns:
            scores: [num_candidates] Hybrid scores for each candidate
        """
        if not self._enabled:
            # If disabled, just return MTP probabilities
            mtp_probs = F.softmax(mtp_logits, dim=-1)
            return mtp_probs[candidate_ids]

        num_candidates = candidate_ids.shape[0]

        # 1. MTP probabilities for candidates
        mtp_probs = F.softmax(mtp_logits, dim=-1)
        mtp_scores = mtp_probs[candidate_ids]  # [num_candidates]

        # 2. Retrieval scores using sliding window context
        # Build context: committed tokens + speculative path
        if speculative_path:
            full_context = context_tokens + speculative_path
        else:
            full_context = context_tokens

        # Use last K tokens for retrieval
        K = self.retrieval_context_window
        retrieval_context = full_context[-K:] if len(full_context) >= K else full_context

        context_tensor = torch.tensor(
            [retrieval_context], device=self.device, dtype=torch.long
        )

        # Fast path: only compute logits for candidates
        retrieval_logits = self.retrieval_module.forward_topk(
            context_tensor, candidate_ids
        )  # [1, num_candidates]
        retrieval_probs = F.softmax(retrieval_logits[0], dim=-1)  # [num_candidates]

        # 3. Suffix probabilities for candidates
        suffix_context = full_context[-4:] if len(full_context) >= 4 else full_context
        suffix_probs_full = self.suffix_matcher.get_suffix_probabilities(
            torch.tensor(suffix_context, device=self.device),
            vocab_size=self.vocab_size,
        )
        suffix_scores = suffix_probs_full[candidate_ids]  # [num_candidates]

        # 4. Compute hybrid scores (additive)
        scores = (
            self.alpha * mtp_scores +
            self.beta * retrieval_probs +
            self.gamma * suffix_scores
        )

        # 5. Apply agreement bonus where multiple sources agree
        # Agreement = both MTP and retrieval have high confidence
        mtp_confident = mtp_scores > 0.1
        retrieval_confident = retrieval_probs > 0.1
        agreement_mask = mtp_confident & retrieval_confident
        scores = torch.where(agreement_mask, scores * self.agreement_bonus, scores)

        return scores

    @torch.inference_mode()
    def rerank_tree_candidates(
        self,
        main_logits: torch.Tensor,
        medusa_logits: torch.Tensor,
        tree_candidates: torch.Tensor,
        tree_indices: torch.Tensor,
        context_tokens: List[int],
        topk: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-rank tree candidates using HST scoring and prune low-scoring paths.

        This method takes the current tree structure and re-scores each node
        using hybrid scoring. Nodes below the score threshold are pruned.

        Args:
            main_logits: [vocab_size] Base model logits
            medusa_logits: [num_heads, vocab_size] Medusa head logits
            tree_candidates: [tree_len] Current tree candidate tokens
            tree_indices: [tree_len] Indices mapping tree positions to flat candidates
            context_tokens: Current context tokens
            topk: Number of top candidates per head

        Returns:
            scored_candidates: [tree_len] with low-scoring nodes zeroed out
            scores: [tree_len] HST scores for each tree position
        """
        if not self._enabled:
            # Return unchanged if disabled
            return tree_candidates, torch.ones(tree_candidates.shape[0], device=self.device)

        tree_len = tree_candidates.shape[0]
        num_heads = medusa_logits.shape[0]

        # Collect all unique candidate tokens from tree
        unique_candidates = tree_candidates.unique()

        # Score root position (depth 0) using main logits
        root_scores = self.score_candidates(
            main_logits,
            unique_candidates,
            context_tokens,
            speculative_path=None,
        )

        # Build score lookup
        score_lookup = torch.zeros(self.vocab_size, device=self.device)
        score_lookup[unique_candidates] = root_scores

        # Initialize tree scores
        tree_scores = torch.zeros(tree_len, device=self.device)
        tree_scores[0] = score_lookup[tree_candidates[0]]  # Root

        # Score each depth level
        # tree_indices format: position 0 = root (index 0)
        # positions 1 to topk = head 0 top-k
        # positions topk+1 to 2*topk = head 1 top-k, etc.

        for depth in range(num_heads):
            # Get candidates at this depth
            start_idx = 1 + depth * topk
            end_idx = min(start_idx + topk, tree_len)

            if start_idx >= tree_len:
                break

            # Get the speculative path up to this depth
            # For simplicity, use the highest-scoring path so far
            # In practice, you'd score each path individually
            depth_candidates = tree_candidates[start_idx:end_idx]

            # Build speculative path (simplified: just use first candidate from each depth)
            spec_path = []
            for d in range(depth):
                d_start = 1 + d * topk
                if d_start < tree_len:
                    spec_path.append(int(tree_candidates[d_start].item()))

            # Score candidates at this depth
            depth_scores = self.score_candidates(
                medusa_logits[depth],
                depth_candidates,
                context_tokens,
                speculative_path=spec_path if spec_path else None,
            )

            tree_scores[start_idx:end_idx] = depth_scores

        # Prune low-scoring candidates (zero them out)
        pruned_candidates = tree_candidates.clone()
        prune_mask = tree_scores < self.score_threshold
        pruned_candidates[prune_mask] = 0  # Zero out pruned candidates

        return pruned_candidates, tree_scores

    def enable(self) -> None:
        """Enable HST scoring."""
        self._enabled = True

    def disable(self) -> None:
        """Disable HST scoring (use pure MTP)."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if HST scoring is enabled."""
        return self._enabled

    def get_config(self) -> Dict:
        """Return current configuration for logging."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "agreement_bonus": self.agreement_bonus,
            "score_threshold": self.score_threshold,
            "retrieval_context_window": self.retrieval_context_window,
            "enabled": self._enabled,
        }

    def set_weights(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> None:
        """
        Update scoring weights.

        Args:
            alpha: New MTP weight (if provided)
            beta: New retrieval weight (if provided)
            gamma: New suffix weight (if provided)
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma

        # Normalize to sum to 1
        total = self.alpha + self.beta + self.gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total

    # ========================================================================
    # Callback interface for GemmaMedusaModel.generate_mtp()
    # ========================================================================

    def __call__(
        self,
        main_logits: torch.Tensor,
        medusa_logits: torch.Tensor,
        tree_candidates: torch.Tensor,
        context_tokens: List[int],
    ) -> torch.Tensor:
        """
        Score tree candidates using HST hybrid scoring.

        This is the callback interface used by GemmaMedusaModel.generate_mtp().

        Args:
            main_logits: [vocab_size] Base model logits
            medusa_logits: [num_heads, vocab_size] Medusa head logits
            tree_candidates: [tree_len] Candidate tokens in tree structure
            context_tokens: List[int] Current context token IDs

        Returns:
            scores: [tree_len] HST scores for each tree position
        """
        if not self._enabled:
            # Return uniform scores if disabled
            return torch.ones(tree_candidates.shape[0], device=self.device)

        tree_len = tree_candidates.shape[0]
        num_heads = medusa_logits.shape[0]

        # Get unique candidates for efficient scoring
        unique_candidates = tree_candidates.unique()

        # Score all unique candidates at root level (depth 0)
        root_scores = self.score_candidates(
            main_logits,
            unique_candidates,
            context_tokens,
            speculative_path=None,
        )

        # Build score lookup table
        score_lookup = torch.zeros(self.vocab_size, device=self.device)
        score_lookup[unique_candidates] = root_scores

        # Map scores to tree positions
        tree_scores = score_lookup[tree_candidates]

        return tree_scores

    def on_tokens_accepted(self, token_ids: List[int]) -> None:
        """
        Callback when tokens are accepted during generation.

        Updates the suffix matching context buffer.

        Args:
            token_ids: List of accepted token IDs
        """
        self.suffix_matcher.append(token_ids)
