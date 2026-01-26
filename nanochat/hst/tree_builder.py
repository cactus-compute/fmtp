"""
Hybrid Smoothed Tree (HST) construction for speculative decoding.

This module implements the unified tree builder that combines:
1. MTP head logits (from Medusa/cross-head attention)
2. Learned retrieval logits
3. Context suffix match probabilities

Into a single dynamically constructed candidate tree using priority-queue
based expansion with Bayesian smoothed scoring.

Key components:
- HSTNode: Tree node dataclass with scoring and lineage
- HybridScorer: Combines multiple signal sources
- HSTTreeBuilder: Priority-queue based adaptive tree construction

Algorithm:
1. Initialize PQ with root node (last committed token)
2. Expansion Loop:
   - Pop highest-scored node
   - Query candidates from MTP/Retrieval/Suffix sources
   - Deduplicate and merge (agreement bonus for consensus)
   - Prune low-score children
   - Push valid children to PQ
3. Terminate when tree budget reached or PQ empty
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
import heapq


@dataclass
class HSTNode:
    """
    Node in the Hybrid Smoothed Tree.

    Each node represents a candidate token at a specific position in the
    speculative tree, with lineage information and source attribution.
    """
    # Token information
    token_id: int
    depth: int  # 0 = root (last committed token)

    # Scoring
    score: float  # Cumulative smoothed score: Π_i S(t_i)
    log_score: float = 0.0  # Log of cumulative score for numerical stability

    # Source attribution (for debugging/analysis)
    sources: set = field(default_factory=set)  # {"mtp", "retrieval", "suffix"}
    agreement_bonus: float = 1.0  # Multiplier from source agreement

    # Tree structure
    parent_idx: Optional[int] = None  # Index in flattened tree
    node_idx: int = -1  # This node's index in flattened tree

    # Component scores (before mixing)
    mtp_prob: float = 0.0
    retrieval_prob: float = 0.0
    suffix_prob: float = 0.0

    def __lt__(self, other: "HSTNode") -> bool:
        """For max-heap (we negate scores in heapq)."""
        return self.score > other.score  # Higher score = higher priority


class HybridScorer:
    """
    Combines MTP head logits, retrieval logits, and suffix match probabilities
    into unified candidate scores using Bayesian smoothing.

    Scoring function:
        S(token) = α * P_mtp(token) + β * P_retrieval(token) + γ * P_suffix(token)

    Where α + β + γ = 1.0 and each P is a calibrated probability distribution.

    Current implementation uses fixed weights (default α=0.6, β=0.3, γ=0.1).
    Use scripts/hst_tune_weights.py to find optimal weights via grid search.

    TODO: Future improvements for adaptive weighting:
    1. Entropy-adaptive weighting: Adjust β based on base model entropy
       - High entropy → lower β (retrieval may be overconfident when model is uncertain)
       - Could compute: β_adaptive = β * sigmoid(-entropy + threshold)

    2. Entropy-normalized combination: Instead of fixed weights, normalize by entropy
       - H_mtp = entropy(P_mtp), H_ret = entropy(P_ret)
       - Confident sources (low entropy) should get higher weight
       - S(token) = P_mtp^(1/H_mtp) * P_ret^(1/H_ret) (product of experts)

    3. Learned weighting: Train a small MLP to predict (α, β, γ) from:
       - Base model entropy
       - Retrieval confidence (max probability)
       - Context features (length, domain indicators)
       - This would require additional training data with oracle acceptance labels

    4. Per-source temperature calibration: Learn τ_mtp, τ_ret, τ_suffix
       - P_calibrated = softmax(logits / τ)
       - Different sources may need different calibration
    """

    def __init__(
        self,
        alpha: float = 0.6,  # MTP head weight
        beta: float = 0.3,   # Retrieval weight
        gamma: float = 0.1,  # Suffix match weight
        retrieval_temperature: float = 1.0,  # Temperature for retrieval calibration
        agreement_bonus: float = 1.5,  # Score multiplier when sources agree
        min_score_threshold: float = 0.001,  # Minimum score to consider
    ):
        """
        Args:
            alpha: Weight for MTP head predictions
            beta: Weight for learned retrieval predictions
            gamma: Weight for suffix match predictions
            retrieval_temperature: Temperature for calibrating retrieval logits
            agreement_bonus: Score multiplier when multiple sources predict same token
            min_score_threshold: Minimum smoothed score to consider a candidate
        """
        # Normalize weights to sum to 1
        total = alpha + beta + gamma
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total

        self.retrieval_temperature = retrieval_temperature
        self.agreement_bonus = agreement_bonus
        self.min_score_threshold = min_score_threshold

    def score_candidates(
        self,
        mtp_logits: Optional[torch.Tensor],  # [vocab_size] or None
        retrieval_logits: Optional[torch.Tensor],  # [vocab_size] or None
        suffix_probs: Optional[torch.Tensor],  # [vocab_size] or None
        candidate_tokens: Optional[torch.Tensor] = None,  # [k] specific tokens to score
        top_k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute smoothed scores for candidate tokens.

        Args:
            mtp_logits: Logits from MTP head for this position
            retrieval_logits: Logits from retrieval module
            suffix_probs: Probability distribution from suffix matching
            candidate_tokens: Specific tokens to score (if None, use top-k from each source)
            top_k: Number of top candidates per source if candidate_tokens is None

        Returns:
            tokens: [k] candidate token IDs
            scores: [k] smoothed scores
            metadata: Dict with per-token source information
        """
        device = "cpu"
        vocab_size = 0

        # Determine device and vocab_size from available inputs
        if mtp_logits is not None:
            device = mtp_logits.device
            vocab_size = mtp_logits.shape[-1]
        elif retrieval_logits is not None:
            device = retrieval_logits.device
            vocab_size = retrieval_logits.shape[-1]
        elif suffix_probs is not None:
            device = suffix_probs.device
            vocab_size = suffix_probs.shape[-1]
        else:
            raise ValueError("At least one of mtp_logits, retrieval_logits, suffix_probs must be provided")

        # Convert logits to probabilities
        mtp_probs = None
        if mtp_logits is not None:
            mtp_probs = F.softmax(mtp_logits, dim=-1)

        retrieval_probs = None
        if retrieval_logits is not None:
            retrieval_probs = F.softmax(retrieval_logits / self.retrieval_temperature, dim=-1)

        # If no specific candidates, gather top-k from each source
        if candidate_tokens is None:
            candidate_set = set()

            if mtp_probs is not None:
                top_mtp = torch.topk(mtp_probs, min(top_k, vocab_size)).indices
                candidate_set.update(top_mtp.tolist())

            if retrieval_probs is not None:
                top_ret = torch.topk(retrieval_probs, min(top_k, vocab_size)).indices
                candidate_set.update(top_ret.tolist())

            if suffix_probs is not None:
                # Suffix probs might be sparse, take non-zero entries
                nonzero = (suffix_probs > 1e-8).nonzero(as_tuple=True)[0]
                if len(nonzero) > 0:
                    suffix_scores = suffix_probs[nonzero]
                    top_idx = torch.topk(suffix_scores, min(top_k, len(nonzero))).indices
                    candidate_set.update(nonzero[top_idx].tolist())

            if not candidate_set:
                return (
                    torch.tensor([], dtype=torch.long, device=device),
                    torch.tensor([], dtype=torch.float, device=device),
                    {}
                )

            candidate_tokens = torch.tensor(list(candidate_set), dtype=torch.long, device=device)

        # Compute smoothed scores for each candidate
        scores = torch.zeros(len(candidate_tokens), device=device)
        metadata = {"sources": [], "mtp_probs": [], "retrieval_probs": [], "suffix_probs": []}

        for i, token in enumerate(candidate_tokens):
            token_id = token.item()
            sources = set()

            # Get probability from each source
            p_mtp = 0.0
            if mtp_probs is not None:
                p_mtp = mtp_probs[token_id].item()
                if p_mtp > 0.01:  # Threshold for "significant" prediction
                    sources.add("mtp")

            p_ret = 0.0
            if retrieval_probs is not None:
                p_ret = retrieval_probs[token_id].item()
                if p_ret > 0.01:
                    sources.add("retrieval")

            p_suf = 0.0
            if suffix_probs is not None:
                p_suf = suffix_probs[token_id].item()
                if p_suf > 0.01:
                    sources.add("suffix")

            # Compute smoothed score
            smoothed_score = (
                self.alpha * p_mtp +
                self.beta * p_ret +
                self.gamma * p_suf
            )

            # Apply agreement bonus if multiple sources predict this token
            if len(sources) >= 2:
                smoothed_score *= self.agreement_bonus

            scores[i] = smoothed_score
            metadata["sources"].append(sources)
            metadata["mtp_probs"].append(p_mtp)
            metadata["retrieval_probs"].append(p_ret)
            metadata["suffix_probs"].append(p_suf)

        return candidate_tokens, scores, metadata

    def get_top_candidates(
        self,
        mtp_logits: Optional[torch.Tensor],
        retrieval_logits: Optional[torch.Tensor],
        suffix_probs: Optional[torch.Tensor],
        mtp_top_k: int = 1,
        retrieval_top_k: int = 2,
        suffix_top_k: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, list[set]]:
        """
        Get top candidates from each source per the specification.

        Per spec:
        - MTP Head Source: Top-1 candidate
        - Retrieval Source: Top-2 candidates
        - Suffix Source: Top-1 candidate (if match exists)

        Args:
            mtp_logits: [vocab_size] MTP head logits
            retrieval_logits: [vocab_size] Retrieval logits
            suffix_probs: [vocab_size] Suffix match probabilities
            mtp_top_k: Number of MTP candidates (default 1)
            retrieval_top_k: Number of retrieval candidates (default 2)
            suffix_top_k: Number of suffix candidates (default 1)

        Returns:
            tokens: [k] unique candidate token IDs (deduplicated)
            scores: [k] smoothed scores
            sources: List of source sets for each token
        """
        device = "cpu"
        vocab_size = 0

        if mtp_logits is not None:
            device = mtp_logits.device
            vocab_size = mtp_logits.shape[-1]
        elif retrieval_logits is not None:
            device = retrieval_logits.device
            vocab_size = retrieval_logits.shape[-1]
        elif suffix_probs is not None:
            device = suffix_probs.device
            vocab_size = suffix_probs.shape[-1]

        # Gather candidates from each source
        candidates: dict[int, set] = {}  # token_id -> sources

        if mtp_logits is not None:
            mtp_probs = F.softmax(mtp_logits, dim=-1)
            top_mtp = torch.topk(mtp_probs, mtp_top_k).indices
            for t in top_mtp.tolist():
                if t not in candidates:
                    candidates[t] = set()
                candidates[t].add("mtp")

        if retrieval_logits is not None:
            ret_probs = F.softmax(retrieval_logits / self.retrieval_temperature, dim=-1)
            top_ret = torch.topk(ret_probs, retrieval_top_k).indices
            for t in top_ret.tolist():
                if t not in candidates:
                    candidates[t] = set()
                candidates[t].add("retrieval")

        if suffix_probs is not None:
            # Get top suffix matches
            nonzero = (suffix_probs > 1e-8).nonzero(as_tuple=True)[0]
            if len(nonzero) > 0:
                suffix_scores = suffix_probs[nonzero]
                k = min(suffix_top_k, len(nonzero))
                top_idx = torch.topk(suffix_scores, k).indices
                for idx in top_idx:
                    t = nonzero[idx].item()
                    if t not in candidates:
                        candidates[t] = set()
                    candidates[t].add("suffix")

        if not candidates:
            return (
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.float, device=device),
                []
            )

        # Score all candidates
        candidate_tokens = torch.tensor(list(candidates.keys()), dtype=torch.long, device=device)
        tokens, scores, metadata = self.score_candidates(
            mtp_logits, retrieval_logits, suffix_probs,
            candidate_tokens=candidate_tokens
        )

        return tokens, scores, metadata["sources"]


class HSTTreeBuilder:
    """
    Priority-queue based adaptive tree construction for HST.

    Instead of fixed-width level-by-level expansion, uses Best-First search
    to build a tree that prioritizes the most promising branches.

    Algorithm:
    1. Initialize: PQ contains root node (last committed token)
    2. Expansion loop:
       - Pop node with highest score
       - Query MTP/Retrieval/Suffix for child candidates
       - Deduplicate and merge (agreement bonus)
       - Prune low-score children
       - Push valid children to PQ
    3. Stop when tree_budget reached or PQ empty
    """

    def __init__(
        self,
        scorer: HybridScorer,
        max_depth: int = 4,
        tree_budget: int = 64,
        min_score_threshold: float = 0.001,
        mtp_top_k: int = 1,
        retrieval_top_k: int = 2,
        suffix_top_k: int = 1,
    ):
        """
        Args:
            scorer: HybridScorer instance for computing candidate scores
            max_depth: Maximum tree depth (number of speculative tokens)
            tree_budget: Maximum total nodes in tree
            min_score_threshold: Minimum score to add a node
            mtp_top_k: Top-k from MTP head per expansion
            retrieval_top_k: Top-k from retrieval per expansion
            suffix_top_k: Top-k from suffix matching per expansion
        """
        self.scorer = scorer
        self.max_depth = max_depth
        self.tree_budget = tree_budget
        self.min_score_threshold = min_score_threshold
        self.mtp_top_k = mtp_top_k
        self.retrieval_top_k = retrieval_top_k
        self.suffix_top_k = suffix_top_k

    def build_tree(
        self,
        root_token: int,
        get_mtp_logits: callable,  # (depth: int, parent_tokens: list[int]) -> Tensor[vocab_size]
        get_retrieval_logits: callable,  # (context_tokens: list[int]) -> Tensor[vocab_size]
        get_suffix_probs: callable,  # (suffix: list[int]) -> Tensor[vocab_size]
        context_tokens: list[int],
        vocab_size: int,
        device: str = "cuda",
    ) -> list[HSTNode]:
        """
        Build HST tree using priority-queue expansion.

        Args:
            root_token: The last committed token (tree root)
            get_mtp_logits: Function to get MTP head logits for a given depth
            get_retrieval_logits: Function to get retrieval logits given context
            get_suffix_probs: Function to get suffix match probabilities
            context_tokens: Current context (for retrieval/suffix)
            vocab_size: Vocabulary size
            device: Device for tensors

        Returns:
            List of HSTNode objects representing the tree in expansion order
        """
        # Initialize tree with root node
        root = HSTNode(
            token_id=root_token,
            depth=0,
            score=1.0,
            log_score=0.0,
            node_idx=0,
            sources={"root"},
        )

        tree = [root]
        # Priority queue: (negative_score, insertion_order, node_idx)
        # Using negative score because heapq is min-heap
        pq = [(-root.score, 0, 0)]
        insertion_counter = 1

        while pq and len(tree) < self.tree_budget:
            # Pop highest-scored node
            _, _, current_idx = heapq.heappop(pq)
            current = tree[current_idx]

            # Don't expand beyond max depth
            if current.depth >= self.max_depth:
                continue

            # Build path from root to current node
            path_tokens = self._get_path_tokens(tree, current_idx)
            extended_context = context_tokens + path_tokens

            # Get logits/probs from each source
            mtp_logits = None
            if get_mtp_logits is not None:
                try:
                    mtp_logits = get_mtp_logits(current.depth, path_tokens)
                except Exception:
                    pass  # MTP might not be available for all depths

            retrieval_logits = None
            if get_retrieval_logits is not None:
                try:
                    retrieval_logits = get_retrieval_logits(extended_context)
                except Exception:
                    pass

            suffix_probs = None
            if get_suffix_probs is not None:
                try:
                    # Use last few tokens as suffix
                    suffix = extended_context[-4:] if len(extended_context) >= 4 else extended_context
                    suffix_probs = get_suffix_probs(suffix)
                except Exception:
                    pass

            # Skip if no sources available
            if mtp_logits is None and retrieval_logits is None and suffix_probs is None:
                continue

            # Get top candidates from each source
            child_tokens, child_scores, child_sources = self.scorer.get_top_candidates(
                mtp_logits=mtp_logits,
                retrieval_logits=retrieval_logits,
                suffix_probs=suffix_probs,
                mtp_top_k=self.mtp_top_k,
                retrieval_top_k=self.retrieval_top_k,
                suffix_top_k=self.suffix_top_k,
            )

            # Create child nodes
            for i, (token, score) in enumerate(zip(child_tokens, child_scores)):
                token_id = token.item()
                node_score = score.item()

                # Compute cumulative score
                cumulative_score = current.score * node_score

                # Prune low-score nodes
                if cumulative_score < self.min_score_threshold:
                    continue

                # Check tree budget
                if len(tree) >= self.tree_budget:
                    break

                # Check for agreement bonus
                sources = child_sources[i] if i < len(child_sources) else set()
                agreement = len(sources) >= 2

                child = HSTNode(
                    token_id=token_id,
                    depth=current.depth + 1,
                    score=cumulative_score,
                    log_score=current.log_score + torch.log(torch.tensor(node_score)).item(),
                    parent_idx=current_idx,
                    node_idx=len(tree),
                    sources=sources,
                    agreement_bonus=self.scorer.agreement_bonus if agreement else 1.0,
                )

                tree.append(child)

                # Add to priority queue
                heapq.heappush(pq, (-cumulative_score, insertion_counter, child.node_idx))
                insertion_counter += 1

        return tree

    def _get_path_tokens(self, tree: list[HSTNode], node_idx: int) -> list[int]:
        """Get token sequence from root to given node (excluding root)."""
        path = []
        current_idx = node_idx

        while current_idx is not None and current_idx > 0:
            node = tree[current_idx]
            path.append(node.token_id)
            current_idx = node.parent_idx

        return list(reversed(path))

    def get_candidate_paths(self, tree: list[HSTNode]) -> list[list[int]]:
        """
        Extract all candidate token paths from the tree.

        Each path represents a complete speculation from root to a leaf or internal node.

        Returns:
            List of token sequences, each starting from depth 1 (not including root)
        """
        paths = []

        for node in tree:
            if node.depth == 0:
                continue  # Skip root

            # Build path from root to this node
            path = self._get_path_tokens(tree, node.node_idx)
            if path:
                paths.append(path)

        # Remove duplicate prefixes, keep only unique paths
        unique_paths = []
        path_set = set()
        for path in sorted(paths, key=len, reverse=True):
            path_tuple = tuple(path)
            if path_tuple not in path_set:
                unique_paths.append(path)
                path_set.add(path_tuple)

        return unique_paths


def build_hst_tree(
    root_token: int,
    mtp_logits_fn: callable,
    retrieval_logits_fn: callable,
    suffix_probs_fn: callable,
    context_tokens: list[int],
    vocab_size: int,
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1,
    max_depth: int = 4,
    tree_budget: int = 64,
    device: str = "cuda",
) -> tuple[list[HSTNode], list[list[int]]]:
    """
    Convenience function to build HST tree with default configuration.

    Args:
        root_token: Last committed token
        mtp_logits_fn: Function (depth, path) -> logits
        retrieval_logits_fn: Function (context) -> logits
        suffix_probs_fn: Function (suffix) -> probs
        context_tokens: Current context tokens
        vocab_size: Vocabulary size
        alpha, beta, gamma: Source weights
        max_depth: Maximum tree depth
        tree_budget: Maximum nodes
        device: Device for tensors

    Returns:
        tree: List of HSTNode objects
        paths: List of candidate token paths
    """
    scorer = HybridScorer(alpha=alpha, beta=beta, gamma=gamma)
    builder = HSTTreeBuilder(
        scorer=scorer,
        max_depth=max_depth,
        tree_budget=tree_budget,
    )

    tree = builder.build_tree(
        root_token=root_token,
        get_mtp_logits=mtp_logits_fn,
        get_retrieval_logits=retrieval_logits_fn,
        get_suffix_probs=suffix_probs_fn,
        context_tokens=context_tokens,
        vocab_size=vocab_size,
        device=device,
    )

    paths = builder.get_candidate_paths(tree)

    return tree, paths
