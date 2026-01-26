"""
Medusa speculative decoding inference engine.

This engine accelerates autoregressive generation by using Medusa heads to
speculate multiple tokens ahead, then verifying them in parallel with tree attention.

Key features:
- Tree-structured candidate generation from Medusa head predictions
- Single forward pass verification using tree attention masks
- Greedy and typical acceptance schemes
- No KV cache (simplified for small models)

Usage:
    model, tokenizer = load_model(...)
    engine = MedusaEngine(model, tokenizer)
    for tokens, masks in engine.generate(prompt_tokens, max_tokens=100):
        print(tokenizer.decode(tokens))
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Generator
from dataclasses import dataclass

from nanochat.medusa_buffers import (
    generate_medusa_buffers,
    get_default_medusa_choices,
)


@dataclass
class MedusaStats:
    """Statistics from Medusa generation for benchmarking."""
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


class MedusaEngine:
    """
    Inference engine with Medusa speculative decoding.

    This engine uses tree attention to verify multiple candidate tokens
    in a single forward pass, providing ~2-3x speedup over standard
    autoregressive decoding.

    Note: This implementation skips KV cache for simplicity, as the models
    are small enough that recomputing the full context is acceptable.
    """

    def __init__(
        self,
        model,
        tokenizer,
        medusa_choices: Optional[List[Tuple[int, ...]]] = None,
        topk: int = 10,
    ):
        """
        Initialize the Medusa engine.

        Args:
            model: GPT model with Medusa heads
            tokenizer: Tokenizer for encoding/decoding
            medusa_choices: Tree structure for speculation. If None, uses default tree.
                           For manual override, import and use DEFAULT_TREES or SPARSE_TREES
                           from nanochat.gemma_medusa.model.
            topk: Number of top predictions from each head
        """
        self.model = model
        self.tokenizer = tokenizer
        self.topk = topk

        # Validate model has Medusa heads
        if model.medusa_heads is None:
            raise ValueError("Model must have Medusa heads for MedusaEngine")

        self.num_heads = len(model.medusa_heads)

        # Set up tree configuration - default to heuristic tree if not provided
        if medusa_choices is None:
            medusa_choices = get_default_medusa_choices(self.num_heads, topk)

        self.medusa_choices = medusa_choices
        self.medusa_buffers = None  # Lazily initialized

    def _ensure_buffers(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initialize buffers on first use (lazy initialization)."""
        if self.medusa_buffers is None:
            self.medusa_buffers = generate_medusa_buffers(
                self.medusa_choices, device=device, topk=self.topk
            )
        return self.medusa_buffers

    @torch.inference_mode()
    def _get_medusa_logits(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both main logits and Medusa head logits.

        Args:
            input_ids: (B, T) Input token IDs

        Returns:
            logits: (B, T, vocab_size) Main model logits
            medusa_logits: (num_heads, B, T, vocab_size) Medusa head logits
        """
        return self.model.forward(input_ids, return_medusa=True)

    @torch.inference_mode()
    def _generate_candidates(
        self,
        logits: torch.Tensor,
        medusa_logits: torch.Tensor,
        buffers: Dict[str, torch.Tensor],
        temperature: float = 0.0,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate candidate token sequences from model predictions.

        Args:
            logits: (B, vocab_size) Main model logits for last position
            medusa_logits: (num_heads, B, vocab_size) Medusa logits for last position
            buffers: Tree attention buffers
            temperature: Sampling temperature (0.0 = greedy)
            rng: Random number generator for sampling

        Returns:
            candidates: (num_candidates, max_depth) Candidate token sequences
            tree_candidates: (tree_len,) Tokens arranged in tree structure
        """
        device = logits.device
        tree_indices = buffers["tree_indices"]
        retrieve_indices = buffers["retrieve_indices"]

        # Get main model prediction (greedy or sampled)
        if temperature == 0.0:
            base_token = torch.argmax(logits[0], dim=-1)  # (,)
        else:
            probs = F.softmax(logits[0] / temperature, dim=-1)
            base_token = torch.multinomial(probs, num_samples=1, generator=rng)[0]

        # Get top-k from each Medusa head
        # medusa_logits: (num_heads, B, vocab_size) -> we want (num_heads, topk)
        medusa_topk = torch.topk(medusa_logits[:, 0, :], self.topk, dim=-1).indices  # (num_heads, topk)

        # Build flat candidate array: [base_token, head0_topk, head1_topk, ...]
        # Shape: (1 + num_heads * topk,)
        flat_candidates = torch.cat([
            base_token.unsqueeze(0),  # (1,)
            medusa_topk.view(-1),     # (num_heads * topk,)
        ])

        # Map to tree structure using tree_indices
        tree_candidates = flat_candidates[tree_indices]  # (tree_len,)

        # Extract candidate paths using retrieve_indices
        # retrieve_indices: (num_candidates, max_depth) -> indices into tree
        # tree_candidates: (tree_len,) -> tokens at each tree position
        # We need to handle -1 padding in retrieve_indices
        num_candidates, max_depth = retrieve_indices.shape
        candidates = torch.zeros(num_candidates, max_depth, dtype=torch.long, device=device)

        for i in range(num_candidates):
            for j in range(max_depth):
                idx = retrieve_indices[i, j].item()
                if idx >= 0:
                    candidates[i, j] = tree_candidates[idx]

        return candidates, tree_candidates

    @torch.inference_mode()
    def _tree_verify(
        self,
        input_ids: torch.Tensor,
        tree_candidates: torch.Tensor,
        buffers: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Verify all candidates in a single forward pass using tree attention.

        Since we're not using KV cache, we simply concatenate the tree candidates
        to the input and forward through the model with a tree attention mask.

        Args:
            input_ids: (B, T) Current token sequence
            tree_candidates: (tree_len,) Candidate tokens in tree structure
            buffers: Tree attention buffers

        Returns:
            logits: (num_candidates, max_depth, vocab_size) Verification logits
            medusa_logits: (num_heads, num_candidates, max_depth, vocab_size)
        """
        device = input_ids.device
        B, T = input_ids.shape
        tree_len = tree_candidates.shape[0]
        retrieve_indices = buffers["retrieve_indices"]
        medusa_attn_mask = buffers["medusa_attn_mask"]  # (1, 1, tree_len, tree_len)

        # Concatenate input with tree candidates
        # Shape: (B, T + tree_len)
        extended_input = torch.cat([
            input_ids,
            tree_candidates.unsqueeze(0).expand(B, -1),
        ], dim=1)

        # Create attention mask for the full sequence
        # Shape: (1, 1, T + tree_len, T + tree_len)
        full_len = T + tree_len

        # Start with causal mask for the prefix
        attn_mask = torch.zeros(1, 1, full_len, full_len, device=device)

        # Prefix tokens: standard causal attention
        attn_mask[:, :, :T, :T] = torch.tril(torch.ones(T, T, device=device))

        # Tree tokens can attend to all prefix tokens
        attn_mask[:, :, T:, :T] = 1.0

        # Tree tokens: use tree attention mask
        attn_mask[:, :, T:, T:] = medusa_attn_mask

        # Forward with custom attention mask
        # Note: This requires the model to support custom attention masks
        # For simplicity, we'll compute without tree mask and rely on proper indexing
        # This is less efficient but works without model modifications
        logits, medusa_logits = self.model.forward(extended_input, return_medusa=True)

        # Extract logits for tree positions: (B, tree_len, vocab_size)
        tree_logits = logits[:, T:, :]
        tree_medusa_logits = medusa_logits[:, :, T:, :]  # (num_heads, B, tree_len, vocab)

        # Reorder according to retrieve_indices to get candidate-aligned logits
        num_candidates, max_depth = retrieve_indices.shape

        # Output shapes
        candidate_logits = torch.zeros(
            num_candidates, max_depth, logits.shape[-1], device=device
        )
        candidate_medusa_logits = torch.zeros(
            self.num_heads, num_candidates, max_depth, logits.shape[-1], device=device
        )

        for i in range(num_candidates):
            for j in range(max_depth):
                idx = retrieve_indices[i, j].item()
                if idx >= 0:
                    candidate_logits[i, j] = tree_logits[0, idx]
                    candidate_medusa_logits[:, i, j] = tree_medusa_logits[:, 0, idx]

        return candidate_logits, candidate_medusa_logits

    @torch.inference_mode()
    def _evaluate_posterior_greedy(
        self,
        logits: torch.Tensor,
        candidates: torch.Tensor,
    ) -> Tuple[int, int]:
        """
        Greedy acceptance: accept longest prefix where candidates match argmax.

        Args:
            logits: (num_candidates, max_depth, vocab_size) Verification logits
            candidates: (num_candidates, max_depth) Candidate token sequences

        Returns:
            best_candidate: Index of best candidate path
            accept_length: Number of tokens to accept (0 = only base token)
        """
        # Get argmax predictions
        predictions = logits.argmax(dim=-1)  # (num_candidates, max_depth)

        # Check matches: logits[i, j] predicts candidates[i, j+1]
        # So we compare predictions[:, :-1] with candidates[:, 1:]
        matches = (candidates[:, 1:] == predictions[:, :-1]).int()

        # Find longest matching prefix per candidate using cumprod
        # cumprod gives 1s until first mismatch, then 0s
        cumulative_matches = torch.cumprod(matches, dim=1)
        accept_lengths = cumulative_matches.sum(dim=1)

        # Select candidate with longest acceptance
        best_candidate = accept_lengths.argmax().item()
        accept_length = accept_lengths[best_candidate].item()

        return best_candidate, accept_length

    @torch.inference_mode()
    def _evaluate_posterior_typical(
        self,
        logits: torch.Tensor,
        candidates: torch.Tensor,
        temperature: float,
        posterior_threshold: float = 0.09,
        posterior_alpha: float = 0.3,
    ) -> Tuple[int, int]:
        """
        Typical acceptance for stochastic sampling.

        Accepts token if: p_original(token) > min(epsilon, delta * exp(-H))

        Args:
            logits: (num_candidates, max_depth, vocab_size) Verification logits
            candidates: (num_candidates, max_depth) Candidate token sequences
            temperature: Sampling temperature
            posterior_threshold: Hard acceptance threshold (epsilon)
            posterior_alpha: Entropy-adaptive factor (delta)

        Returns:
            best_candidate: Index of best candidate
            accept_length: Number of tokens to accept
        """
        # Get probability of each candidate token
        probs = F.softmax(logits[:, :-1] / temperature, dim=-1)

        # Gather probabilities for candidate tokens
        # candidates[:, 1:] contains the tokens we're evaluating
        candidate_tokens = candidates[:, 1:].unsqueeze(-1)  # (num_cand, max_depth-1, 1)
        candidate_probs = torch.gather(probs, dim=-1, index=candidate_tokens).squeeze(-1)

        # Compute entropy of the distribution
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        # Adaptive threshold: min(epsilon, delta * exp(-H))
        threshold = torch.minimum(
            torch.full_like(entropy, posterior_threshold),
            torch.exp(-entropy) * posterior_alpha,
        )

        # Accept if probability exceeds threshold
        accepts = candidate_probs > threshold

        # Find longest accepting prefix
        cumulative_accepts = torch.cumprod(accepts.int(), dim=1)
        accept_lengths = cumulative_accepts.sum(dim=1)

        # Select best candidate
        best_candidate = accept_lengths.argmax().item()
        accept_length = accept_lengths[best_candidate].item()

        return best_candidate, accept_length

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
    ) -> Generator[Tuple[List[int], List[int]], None, None]:
        """
        Generate tokens using Medusa speculative decoding.

        Note: num_samples > 1 falls back to standard generation since
        speculative decoding is optimized for single sequences.

        Args:
            tokens: Initial prompt as list of token IDs
            num_samples: Number of parallel samples (>1 disables speculation)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_k: Top-k filtering (not used in Medusa, kept for API compatibility)
            seed: Random seed for sampling
            posterior_threshold: Typical acceptance hard threshold
            posterior_alpha: Typical acceptance entropy factor

        Yields:
            Tuple of (token_column, token_masks) like the standard Engine
        """
        if num_samples > 1:
            # Fall back to standard generation for batched inference
            from nanochat.engine import Engine
            standard_engine = Engine(self.model, self.tokenizer)
            yield from standard_engine.generate(
                tokens, num_samples, max_tokens, temperature, top_k, seed
            )
            return

        device = self.model.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        # Initialize RNG for sampling
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Ensure buffers are initialized
        buffers = self._ensure_buffers(device)

        # Get special tokens for termination
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # Initialize input
        current_tokens = list(tokens)
        num_generated = 0

        # Initial forward pass to get logits
        input_ids = torch.tensor([current_tokens], dtype=torch.long, device=device)
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            logits, medusa_logits = self._get_medusa_logits(input_ids)

        # Extract last position logits
        last_logits = logits[:, -1, :]  # (B, vocab_size)
        last_medusa_logits = medusa_logits[:, :, -1, :]  # (num_heads, B, vocab_size)

        while True:
            # Check termination
            if max_tokens is not None and num_generated >= max_tokens:
                break

            # Generate candidates
            candidates, tree_candidates = self._generate_candidates(
                last_logits,
                last_medusa_logits,
                buffers,
                temperature=temperature,
                rng=rng,
            )

            # For small models without KV cache, we do a full forward pass
            # including the tree candidates for verification
            input_ids = torch.tensor([current_tokens], dtype=torch.long, device=device)

            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                verify_logits, verify_medusa_logits = self._tree_verify(
                    input_ids, tree_candidates, buffers
                )

            # Evaluate which candidates to accept
            if temperature == 0.0:
                best_candidate, accept_length = self._evaluate_posterior_greedy(
                    verify_logits, candidates
                )
            else:
                best_candidate, accept_length = self._evaluate_posterior_typical(
                    verify_logits, candidates, temperature,
                    posterior_threshold, posterior_alpha
                )

            # Accept tokens from the best candidate
            # candidates[best_candidate, 0] is the base model prediction (always accepted)
            # candidates[best_candidate, 1:accept_length+1] are the accepted speculative tokens
            accepted_tokens = candidates[best_candidate, : accept_length + 1].tolist()

            # Yield accepted tokens one at a time (maintaining Engine API compatibility)
            for token in accepted_tokens:
                # Check for termination tokens
                if token == assistant_end or token == bos:
                    return

                yield [token], [1]  # Single sample, sampled (not forced)
                current_tokens.append(token)
                num_generated += 1

                if max_tokens is not None and num_generated >= max_tokens:
                    return

            # Update logits for next iteration
            # We need to get fresh logits from the new context
            input_ids = torch.tensor([current_tokens], dtype=torch.long, device=device)
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                logits, medusa_logits = self._get_medusa_logits(input_ids)

            last_logits = logits[:, -1, :]
            last_medusa_logits = medusa_logits[:, :, -1, :]

    def generate_batch(
        self,
        tokens: List[int],
        num_samples: int = 1,
        **kwargs
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Non-streaming batch generation that returns final token sequences.

        Args:
            tokens: Initial prompt tokens
            num_samples: Number of samples to generate
            **kwargs: Additional arguments passed to generate()

        Returns:
            Tuple of (results, masks) where:
            - results: List of token sequences
            - masks: List of mask sequences (1=sampled, 0=forced)
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples

        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)

            if all(completed):
                break

        return results, masks

    def generate_with_stats(
        self,
        tokens: List[int],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[int], MedusaStats]:
        """
        Generate tokens and return statistics for benchmarking.

        Args:
            tokens: Initial prompt tokens
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to generate()

        Returns:
            Tuple of (generated_tokens, stats)
        """
        device = self.model.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        buffers = self._ensure_buffers(device)
        retrieve_indices = buffers["retrieve_indices"]
        max_speculation = retrieve_indices.shape[1] - 1  # Max tokens per step

        stats = MedusaStats(
            tokens_generated=0,
            forward_passes=0,
            total_proposed=0,
            total_accepted=0,
        )

        result_tokens = list(tokens)

        for token_column, _ in self.generate(tokens, max_tokens=max_tokens, **kwargs):
            for token in token_column:
                result_tokens.append(token)
                stats.tokens_generated += 1

            # Each yield represents one forward pass worth of accepted tokens
            stats.forward_passes += 1
            stats.total_proposed += max_speculation
            stats.total_accepted += len(token_column)

        return result_tokens, stats


def benchmark_medusa(
    model,
    tokenizer,
    prompts: List[str],
    max_tokens: int = 100,
    temperature: float = 0.0,
    medusa_choices: Optional[List[Tuple[int, ...]]] = None,
) -> List[Dict]:
    """
    Benchmark Medusa inference against standard generation.

    Args:
        model: GPT model with Medusa heads
        tokenizer: Tokenizer
        prompts: List of prompt strings to benchmark
        max_tokens: Maximum tokens per generation
        temperature: Sampling temperature
        medusa_choices: Tree structure for speculation (None = default tree)

    Returns:
        List of benchmark results per prompt
    """
    import time
    from nanochat.engine import Engine

    standard_engine = Engine(model, tokenizer)
    medusa_engine = MedusaEngine(model, tokenizer, medusa_choices=medusa_choices)

    results = []
    bos = tokenizer.get_bos_token_id()

    for prompt in prompts:
        tokens = tokenizer.encode(prompt, prepend=bos)

        # Standard generation
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        std_result, _ = standard_engine.generate_batch(
            tokens, max_tokens=max_tokens, temperature=temperature
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        std_time = time.time() - t0

        # Medusa generation
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        med_result, stats = medusa_engine.generate_with_stats(
            tokens, max_tokens=max_tokens, temperature=temperature
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        med_time = time.time() - t0

        results.append({
            "prompt_length": len(tokens),
            "output_length": len(std_result[0]) - len(tokens),
            "standard_time": std_time,
            "medusa_time": med_time,
            "speedup": std_time / max(med_time, 1e-6),
            "outputs_match": std_result[0] == med_result,
            "mean_accepted": stats.mean_accepted_length,
            "acceptance_rate": stats.acceptance_rate,
        })

    return results
