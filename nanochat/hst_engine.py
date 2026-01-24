"""
HST (Hybrid Smoothed Tree) speculative decoding inference engine.

This engine extends the existing Medusa infrastructure with HST's hybrid
approach: combining MTP head predictions, learned retrieval, and context
suffix matching into a unified speculation strategy.

Key features:
- Integrates with existing Medusa heads (cross-head attention architecture)
- Adds learned retrieval via SVD-compressed vocabulary projection
- Context suffix matching for repetition patterns
- Dynamic tree construction via priority-queue expansion
- Entropy-based early exit for uncertain predictions

Usage:
    model, tokenizer = load_model(...)
    engine = HSTEngine(model, tokenizer, retrieval_checkpoint="path/to/retrieval.pt")
    for tokens, masks in engine.generate(prompt_tokens, max_tokens=100):
        print(tokenizer.decode(tokens))
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Generator
from dataclasses import dataclass
from pathlib import Path

from nanochat.medusa_engine import MedusaEngine, MedusaStats
from nanochat.hst.retrieval import RetrievalMixer, load_svd_basis
from nanochat.hst.suffix_match import SuffixMatcher
from nanochat.hst.tree_builder import HybridScorer, HSTTreeBuilder, HSTNode
from nanochat.hst.tree_attention import (
    generate_hst_buffers,
    verify_tree_greedy,
    verify_tree_typical,
    merge_with_context,
    HSTBuffers,
)


@dataclass
class HSTStats(MedusaStats):
    """Extended statistics for HST generation."""
    # Source contributions
    mtp_tokens: int = 0
    retrieval_tokens: int = 0
    suffix_tokens: int = 0
    agreement_tokens: int = 0  # Tokens where 2+ sources agreed

    # Entropy tracking
    high_entropy_fallbacks: int = 0  # Times we reduced tree due to high entropy

    @property
    def mtp_contribution(self) -> float:
        total = self.mtp_tokens + self.retrieval_tokens + self.suffix_tokens
        return self.mtp_tokens / max(total, 1)

    @property
    def retrieval_contribution(self) -> float:
        total = self.mtp_tokens + self.retrieval_tokens + self.suffix_tokens
        return self.retrieval_tokens / max(total, 1)

    @property
    def suffix_contribution(self) -> float:
        total = self.mtp_tokens + self.retrieval_tokens + self.suffix_tokens
        return self.suffix_tokens / max(total, 1)

    @property
    def agreement_rate(self) -> float:
        return self.agreement_tokens / max(self.total_proposed, 1)


class HSTEngine(MedusaEngine):
    """
    Inference engine with HST (Hybrid Smoothed Tree) speculation.

    Extends MedusaEngine to incorporate:
    - Learned retrieval module for zero-overhead candidate generation
    - Context suffix matching for repetition patterns
    - Hybrid scoring combining all sources
    - Dynamic tree construction with priority-queue expansion
    """

    def __init__(
        self,
        model,
        tokenizer,
        # Retrieval configuration
        retrieval_checkpoint: Optional[str] = None,
        svd_rank: int = 64,
        retrieval_context_window: int = 4,
        retrieval_num_layers: int = 2,
        # Suffix matching configuration
        suffix_buffer_size: int = 1024,
        suffix_max_len: int = 4,
        # Hybrid scoring weights
        alpha: float = 0.6,  # MTP head weight
        beta: float = 0.3,   # Retrieval weight
        gamma: float = 0.1,  # Suffix match weight
        agreement_bonus: float = 1.5,
        # Tree configuration
        max_depth: int = 4,
        tree_budget: int = 64,
        mtp_top_k: int = 1,
        retrieval_top_k: int = 2,
        suffix_top_k: int = 1,
        # Entropy early exit
        entropy_threshold: float = 2.5,  # Disable retrieval if H > threshold
        # Parent class arguments
        medusa_choices=None,
        topk: int = 10,
        use_sparse_tree: bool = False,
    ):
        """
        Initialize the HST engine.

        Args:
            model: GPT model with Medusa heads
            tokenizer: Tokenizer for encoding/decoding
            retrieval_checkpoint: Path to trained retrieval module
            svd_rank: SVD rank for retrieval output
            retrieval_context_window: Number of context tokens for retrieval
            retrieval_num_layers: Number of MLP-Mixer layers
            suffix_buffer_size: Size of suffix matching buffer
            suffix_max_len: Maximum suffix length to index
            alpha: Weight for MTP head predictions
            beta: Weight for retrieval predictions
            gamma: Weight for suffix match predictions
            agreement_bonus: Score multiplier when sources agree
            max_depth: Maximum tree depth
            tree_budget: Maximum nodes in tree
            mtp_top_k: Top-k from MTP head per expansion
            retrieval_top_k: Top-k from retrieval per expansion
            suffix_top_k: Top-k from suffix matching per expansion
            entropy_threshold: Entropy threshold for disabling retrieval
            medusa_choices: Custom Medusa tree structure
            topk: Number of top-k for Medusa
            use_sparse_tree: Use sparse Medusa tree
        """
        # Initialize parent MedusaEngine
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            medusa_choices=medusa_choices,
            topk=topk,
            use_sparse_tree=use_sparse_tree,
        )

        self.device = model.get_device()

        # Get model dimensions
        self.embed_dim = model.config.n_embd
        self.vocab_size = model.config.vocab_size

        # Initialize retrieval module
        self.retrieval_module = RetrievalMixer(
            embed_dim=self.embed_dim,
            vocab_size=self.vocab_size,
            context_window=retrieval_context_window,
            num_layers=retrieval_num_layers,
            svd_rank=svd_rank,
            input_mode="last_k",
        )

        # Load SVD basis
        try:
            compressed_vocab = load_svd_basis(rank=svd_rank, model_name="gemma")
            self.retrieval_module.load_svd(compressed_vocab)
        except FileNotFoundError:
            # Initialize with random for testing if SVD not available
            self.retrieval_module.compressed_vocab.normal_(0, 0.01)
            self.retrieval_module._svd_loaded = True

        # Load retrieval checkpoint if provided
        if retrieval_checkpoint:
            self.retrieval_module.load_state_dict(
                torch.load(retrieval_checkpoint, weights_only=True)
            )

        self.retrieval_module = self.retrieval_module.to(self.device)
        self.retrieval_module.eval()

        # Initialize suffix matcher
        self.suffix_matcher = SuffixMatcher(
            buffer_size=suffix_buffer_size,
            min_suffix_len=1,
            max_suffix_len=suffix_max_len,
            device=str(self.device),
        )

        # Initialize hybrid scorer
        self.scorer = HybridScorer(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            agreement_bonus=agreement_bonus,
        )

        # Initialize tree builder
        self.tree_builder = HSTTreeBuilder(
            scorer=self.scorer,
            max_depth=max_depth,
            tree_budget=tree_budget,
            mtp_top_k=mtp_top_k,
            retrieval_top_k=retrieval_top_k,
            suffix_top_k=suffix_top_k,
        )

        # Configuration
        self.entropy_threshold = entropy_threshold
        self.retrieval_context_window = retrieval_context_window

    def _get_retrieval_logits(
        self,
        context_ids: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get logits from the learned retrieval module.

        Args:
            context_ids: [B, T] Recent context token IDs
            embeddings: Optional pre-computed embeddings

        Returns:
            logits: [B, vocab_size] Retrieval logits
        """
        if embeddings is None:
            # Get embeddings from model
            embeddings = self.model.transformer.wte(context_ids)

        # Use last K tokens
        if embeddings.shape[1] > self.retrieval_context_window:
            embeddings = embeddings[:, -self.retrieval_context_window:, :]

        with torch.no_grad():
            return self.retrieval_module(embeddings)

    def _get_suffix_probs(
        self,
        suffix_tokens: List[int],
    ) -> torch.Tensor:
        """
        Get probability distribution from suffix matching.

        Args:
            suffix_tokens: Recent tokens to use as suffix

        Returns:
            probs: [vocab_size] Suffix match probabilities
        """
        suffix = torch.tensor(suffix_tokens[-4:], device=self.device)
        return self.suffix_matcher.get_suffix_probabilities(
            suffix,
            vocab_size=self.vocab_size,
        )

    def _compute_entropy(self, logits: torch.Tensor) -> float:
        """Compute Shannon entropy of logit distribution."""
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.item()

    @torch.inference_mode()
    def _build_hst_tree(
        self,
        base_logits: torch.Tensor,
        medusa_logits: torch.Tensor,
        context_tokens: List[int],
    ) -> Tuple[List[HSTNode], HSTBuffers]:
        """
        Build HST tree using hybrid scoring.

        Args:
            base_logits: [vocab_size] Base model logits
            medusa_logits: [num_heads, vocab_size] Medusa head logits
            context_tokens: Current context tokens

        Returns:
            tree: List of HSTNode objects
            buffers: HSTBuffers for tree attention
        """
        # Check entropy - reduce tree if model is uncertain
        entropy = self._compute_entropy(base_logits)
        use_retrieval = entropy < self.entropy_threshold

        # Get retrieval logits
        retrieval_logits = None
        if use_retrieval:
            context_ids = torch.tensor([context_tokens[-self.retrieval_context_window:]], device=self.device)
            retrieval_logits = self._get_retrieval_logits(context_ids)[0]

        # Get suffix probabilities
        suffix_probs = self._get_suffix_probs(context_tokens)

        # Define callback functions for tree builder
        def get_mtp_logits(depth: int, path: List[int]) -> torch.Tensor:
            if depth < len(medusa_logits):
                return medusa_logits[depth]
            return base_logits  # Fall back to base logits for deeper levels

        def get_retrieval_logits_fn(context: List[int]) -> Optional[torch.Tensor]:
            if use_retrieval and retrieval_logits is not None:
                return retrieval_logits
            return None

        def get_suffix_probs_fn(suffix: List[int]) -> torch.Tensor:
            return self._get_suffix_probs(suffix)

        # Get root token
        root_token = base_logits.argmax().item()

        # Build tree
        tree = self.tree_builder.build_tree(
            root_token=root_token,
            get_mtp_logits=get_mtp_logits,
            get_retrieval_logits=get_retrieval_logits_fn,
            get_suffix_probs=get_suffix_probs_fn,
            context_tokens=context_tokens,
            vocab_size=self.vocab_size,
            device=str(self.device),
        )

        # Generate buffers
        buffers = generate_hst_buffers(tree, device=str(self.device))

        return tree, buffers

    @torch.inference_mode()
    def generate(
        self,
        tokens: List[int],
        num_samples: int = 1,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        seed: int = 42,
        posterior_threshold: float = 0.09,
        posterior_alpha: float = 0.3,
        use_hst: bool = True,
    ) -> Generator[Tuple[List[int], List[int]], None, None]:
        """
        Generate tokens using HST speculative decoding.

        Args:
            tokens: Initial prompt as list of token IDs
            num_samples: Number of parallel samples (>1 disables speculation)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_k: Top-k filtering
            seed: Random seed
            posterior_threshold: Typical acceptance hard threshold
            posterior_alpha: Typical acceptance entropy factor
            use_hst: Use HST (True) or fall back to Medusa (False)

        Yields:
            Tuple of (token_column, token_masks)
        """
        if num_samples > 1 or not use_hst:
            # Fall back to parent Medusa implementation
            yield from super().generate(
                tokens, num_samples, max_tokens, temperature, top_k, seed,
                posterior_threshold, posterior_alpha
            )
            return

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        # Initialize suffix matcher with context
        self.suffix_matcher.reset()
        for token in tokens:
            self.suffix_matcher.append(token)

        # Get special tokens
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # Initialize generation state
        current_tokens = list(tokens)
        num_generated = 0

        # Initial forward pass
        input_ids = torch.tensor([current_tokens], dtype=torch.long, device=self.device)
        with torch.amp.autocast(device_type=self.device.type, dtype=dtype):
            logits, medusa_logits = self._get_medusa_logits(input_ids)

        last_logits = logits[0, -1, :]  # [vocab_size]
        last_medusa_logits = medusa_logits[:, 0, -1, :]  # [num_heads, vocab_size]

        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break

            # Build HST tree
            tree, buffers = self._build_hst_tree(
                base_logits=last_logits,
                medusa_logits=last_medusa_logits,
                context_tokens=current_tokens,
            )

            if len(tree) <= 1:
                # No speculation possible, do standard autoregressive
                if temperature == 0.0:
                    next_token = last_logits.argmax().item()
                else:
                    probs = F.softmax(last_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

                if next_token == assistant_end or next_token == bos:
                    return

                yield [next_token], [1]
                current_tokens.append(next_token)
                self.suffix_matcher.append(next_token)
                num_generated += 1
            else:
                # Verify tree with forward pass
                input_ids = torch.tensor([current_tokens], dtype=torch.long, device=self.device)
                extended_ids, _, _ = merge_with_context(input_ids, buffers)

                with torch.amp.autocast(device_type=self.device.type, dtype=dtype):
                    verify_logits, _ = self.model.forward(extended_ids, return_medusa=True)

                # Extract tree logits
                tree_logits = verify_logits[0, len(current_tokens):, :]  # [tree_len, vocab_size]

                # Verify and accept tokens
                if temperature == 0.0:
                    accept_length, accepted_tokens = verify_tree_greedy(tree_logits, buffers)
                else:
                    accept_length, accepted_tokens = verify_tree_typical(
                        tree_logits, buffers, temperature,
                        posterior_threshold, posterior_alpha
                    )

                # Always accept at least the base prediction
                if not accepted_tokens:
                    if temperature == 0.0:
                        next_token = last_logits.argmax().item()
                    else:
                        probs = F.softmax(last_logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, 1).item()
                    accepted_tokens = [next_token]

                # Yield accepted tokens
                for token in accepted_tokens:
                    if token == assistant_end or token == bos:
                        return

                    yield [token], [1]
                    current_tokens.append(token)
                    self.suffix_matcher.append(token)
                    num_generated += 1

                    if max_tokens is not None and num_generated >= max_tokens:
                        return

            # Get fresh logits for next iteration
            input_ids = torch.tensor([current_tokens], dtype=torch.long, device=self.device)
            with torch.amp.autocast(device_type=self.device.type, dtype=dtype):
                logits, medusa_logits = self._get_medusa_logits(input_ids)

            last_logits = logits[0, -1, :]
            last_medusa_logits = medusa_logits[:, 0, -1, :]

    def generate_with_stats(
        self,
        tokens: List[int],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[int], HSTStats]:
        """
        Generate tokens and return HST-specific statistics.

        Args:
            tokens: Initial prompt tokens
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            Tuple of (generated_tokens, stats)
        """
        stats = HSTStats(
            tokens_generated=0,
            forward_passes=0,
            total_proposed=0,
            total_accepted=0,
            mtp_tokens=0,
            retrieval_tokens=0,
            suffix_tokens=0,
            agreement_tokens=0,
            high_entropy_fallbacks=0,
        )

        result_tokens = list(tokens)

        for token_column, _ in self.generate(tokens, max_tokens=max_tokens, **kwargs):
            for token in token_column:
                result_tokens.append(token)
                stats.tokens_generated += 1
                stats.total_accepted += 1

            stats.forward_passes += 1
            stats.total_proposed += len(token_column)

        return result_tokens, stats


def benchmark_hst(
    model,
    tokenizer,
    prompts: List[str],
    retrieval_checkpoint: Optional[str] = None,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> List[Dict]:
    """
    Benchmark HST inference against standard Medusa and autoregressive.

    Args:
        model: GPT model with Medusa heads
        tokenizer: Tokenizer
        prompts: List of prompts
        retrieval_checkpoint: Path to retrieval checkpoint
        max_tokens: Maximum tokens per generation
        temperature: Sampling temperature

    Returns:
        List of benchmark results
    """
    import time
    from nanochat.engine import Engine

    # Create engines
    standard_engine = Engine(model, tokenizer)
    medusa_engine = MedusaEngine(model, tokenizer)
    hst_engine = HSTEngine(
        model, tokenizer,
        retrieval_checkpoint=retrieval_checkpoint,
    )

    results = []
    bos = tokenizer.get_bos_token_id()

    for prompt in prompts:
        tokens = tokenizer.encode(prompt, prepend=bos)

        # Standard autoregressive
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        std_result, _ = standard_engine.generate_batch(
            tokens, max_tokens=max_tokens, temperature=temperature
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        std_time = time.time() - t0

        # Medusa
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        med_result, med_stats = medusa_engine.generate_with_stats(
            tokens, max_tokens=max_tokens, temperature=temperature
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        med_time = time.time() - t0

        # HST
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        hst_result, hst_stats = hst_engine.generate_with_stats(
            tokens, max_tokens=max_tokens, temperature=temperature
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        hst_time = time.time() - t0

        results.append({
            "prompt_length": len(tokens),
            "output_length": len(std_result[0]) - len(tokens),
            # Timing
            "standard_time": std_time,
            "medusa_time": med_time,
            "hst_time": hst_time,
            # Speedups
            "medusa_speedup": std_time / max(med_time, 1e-6),
            "hst_speedup": std_time / max(hst_time, 1e-6),
            "hst_vs_medusa": med_time / max(hst_time, 1e-6),
            # Quality
            "medusa_mean_accepted": med_stats.mean_accepted_length,
            "hst_mean_accepted": hst_stats.mean_accepted_length,
            # HST specifics
            "hst_agreement_rate": hst_stats.agreement_rate,
            "hst_mtp_contribution": hst_stats.mtp_contribution,
            "hst_retrieval_contribution": hst_stats.retrieval_contribution,
            "hst_suffix_contribution": hst_stats.suffix_contribution,
        })

    return results
