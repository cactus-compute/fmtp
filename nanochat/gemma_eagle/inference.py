"""
EAGLE speculative decoding inference for Gemma3.

Provides the main generation loop with draft-then-verify strategy.
"""

from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass
import time
import torch
import torch.nn.functional as F

from .model import GemmaEagleModel
from .tree import evaluate_posterior
from nanochat.gemma_common.speculative import update_kv_cache_from_tree


@dataclass
class EagleGenerationStats:
    """Statistics from EAGLE generation for benchmarking."""
    tokens_generated: int
    forward_passes: int  # Number of base model forward passes
    draft_passes: int  # Number of draft model forward passes
    total_proposed: int
    total_accepted: int
    time_elapsed: float
    timing: Optional[Dict[str, float]] = None

    @property
    def mean_accepted_length(self) -> float:
        """Average number of tokens accepted per speculation round."""
        return self.tokens_generated / max(1, self.forward_passes)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed tokens that were accepted."""
        return self.total_accepted / max(1, self.total_proposed)

    @property
    def tokens_per_second(self) -> float:
        """Generation throughput."""
        return self.tokens_generated / max(0.001, self.time_elapsed)

    @property
    def speedup_estimate(self) -> float:
        """Estimated speedup over vanilla autoregressive (assuming same forward cost)."""
        return self.mean_accepted_length

    @property
    def eagle_overhead_fraction(self) -> float:
        """Fraction of speculation-loop time spent outside base verification."""
        if not self.timing:
            return 0.0
        overhead = (
            self.timing.get("draft_s", 0.0)
            + self.timing.get("accept_s", 0.0)
            + self.timing.get("kv_update_s", 0.0)
            + self.timing.get("sample_s", 0.0)
        )
        verify = self.timing.get("verify_s", 0.0)
        total = overhead + verify
        return overhead / total if total > 0 else 0.0


@dataclass
class GenerationConfig:
    """Configuration for EAGLE generation."""
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None


def prepare_logits_processor(
    temperature: float = 0.0,
    repetition_penalty: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> Optional[Callable]:
    """
    Create a logits processor for sampling.

    Returns None for greedy decoding (temperature=0).
    """
    if temperature < 1e-5:
        return None

    def process(input_ids, logits):
        # Temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Repetition penalty
        if repetition_penalty != 1.0 and input_ids is not None:
            for token_id in set(input_ids.view(-1).tolist()):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits

    return process


class EagleGenerator:
    """
    EAGLE speculative decoding generator for Gemma3.

    Usage:
        model = GemmaEagleModel(config)
        generator = EagleGenerator(model)
        output_ids, stats = generator.generate(input_ids, max_new_tokens=256)
    """

    def __init__(self, model: GemmaEagleModel):
        """
        Initialize the generator.

        Args:
            model: GemmaEagleModel instance with loaded base and draft models
        """
        self.model = model
        self.config = model.config
        self.device = model._device
        self.dtype = model._dtype

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        return_stats: bool = True,
        collect_timing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[EagleGenerationStats]]:
        """
        Generate tokens using EAGLE speculative decoding.

        Args:
            input_ids: Input token IDs (batch_size=1 only, shape: (1, seq_len))
            attention_mask: Optional attention mask
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling probability
            top_k: Top-k filtering (0 = disabled)
            repetition_penalty: Repetition penalty
            eos_token_id: End-of-sequence token ID
            return_stats: Whether to return generation statistics

        Returns:
            output_ids: Generated token IDs (1, seq_len + new_tokens)
            stats: Generation statistics (if return_stats=True)
        """
        assert input_ids.shape[0] == 1, "EAGLE only supports batch_size=1"

        start_time = time.time()
        is_cuda = self.device.type == "cuda"
        timing = None
        if collect_timing:
            timing = {
                "prefill_s": 0.0,
                "draft_s": 0.0,
                "verify_s": 0.0,
                "accept_s": 0.0,
                "kv_update_s": 0.0,
                "sample_s": 0.0,
                "iterations": 0.0,
            }

        stats = EagleGenerationStats(
            tokens_generated=0,
            forward_passes=0,
            draft_passes=0,
            total_proposed=0,
            total_accepted=0,
            time_elapsed=0.0,
            timing=timing,
        )

        # Set up logits processor
        logits_processor = prepare_logits_processor(
            temperature, repetition_penalty, top_p, top_k
        )

        # Get EOS token
        if eos_token_id is None and self.model.tokenizer is not None:
            eos_token_id = self.model.tokenizer.eos_token_id

        # Reset KV caches
        self.model.reset_kv()

        # Initial prefill - get hidden states and first token
        input_ids = input_ids.to(self.device)
        if collect_timing:
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        fused_hidden, base_logits = self.model.get_base_hidden_states(input_ids, attention_mask)
        stats.forward_passes += 1

        # Prefill base KV cache with prefix only (exclude sample token)
        _, base_past_key_values, _ = self.model.get_base_hidden_states_with_cache(input_ids)
        base_seq_len = input_ids.shape[1]
        if collect_timing:
            if is_cuda:
                torch.cuda.synchronize()
            timing["prefill_s"] += time.perf_counter() - t0

        # Sample first token
        last_logits = base_logits[:, -1:]
        if logits_processor is not None:
            last_logits = logits_processor(input_ids, last_logits)

        if temperature < 1e-5:
            sample_token = last_logits.argmax(dim=-1)
        else:
            probs = F.softmax(last_logits, dim=-1)
            sample_token = torch.multinomial(probs.squeeze(1), num_samples=1)

        input_ids = torch.cat([input_ids, sample_token], dim=1)
        new_token_count = 1
        stats.tokens_generated += 1

        # Check for EOS
        if eos_token_id is not None and sample_token.item() == eos_token_id:
            stats.time_elapsed = time.time() - start_time
            return input_ids, stats if return_stats else None

        # Main generation loop
        while new_token_count < max_new_tokens:
            if collect_timing:
                timing["iterations"] += 1
            # === Draft Phase ===
            # Generate tree of candidate tokens
            if collect_timing:
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.model.topk_generate(
                fused_hidden, input_ids, logits_processor
            )
            if collect_timing:
                if is_cuda:
                    torch.cuda.synchronize()
                timing["draft_s"] += time.perf_counter() - t0
            stats.draft_passes += 1

            num_candidates = draft_tokens.shape[1] - 1  # Exclude root
            stats.total_proposed += num_candidates

            # === Verify Phase ===
            # Build candidate sequences for evaluation
            tree_candidates_ext = torch.cat([
                draft_tokens.squeeze(0),
                torch.tensor([-1], device=self.device)
            ], dim=0)
            cart_candidates = tree_candidates_ext[retrieve_indices]

            # Verify with tree attention + KV cache (prefix excludes sample token)
            if collect_timing:
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            tree_logits, base_past_key_values, tree_fused_hidden = self.model.forward_base_with_tree_cache(
                draft_tokens,
                tree_mask,
                tree_position_ids,
                base_past_key_values,
                base_seq_len,
            )
            stats.forward_passes += 1
            if collect_timing:
                if is_cuda:
                    torch.cuda.synchronize()
                timing["verify_s"] += time.perf_counter() - t0

            # Get verification logits aligned with candidate positions
            verify_logits = tree_logits[retrieve_indices]

            # === Accept Phase ===
            if collect_timing:
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            best_candidate, accept_length, next_logits = evaluate_posterior(
                verify_logits, cart_candidates, temperature, top_p
            )
            if collect_timing:
                if is_cuda:
                    torch.cuda.synchronize()
                timing["accept_s"] += time.perf_counter() - t0

            stats.total_accepted += accept_length

            # Get accepted tokens
            if accept_length > 0:
                accepted_tokens = cart_candidates[best_candidate, 1:accept_length + 1]
                input_ids = torch.cat([input_ids, accepted_tokens.unsqueeze(0)], dim=1)
                new_token_count += accept_length
                stats.tokens_generated += accept_length

            # Update base KV cache with root + accepted tokens from tree cache
            num_to_add = accept_length + 1
            accepted_tree_positions = retrieve_indices[best_candidate, :num_to_add].tolist()
            if collect_timing:
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            base_past_key_values, _ = update_kv_cache_from_tree(
                base_past_key_values,
                accepted_tree_positions,
                base_seq_len,
                draft_tokens.shape[1],
            )
            base_seq_len += num_to_add
            if accepted_tree_positions:
                new_fused = tree_fused_hidden[accepted_tree_positions]
                fused_hidden = torch.cat([fused_hidden, new_fused.unsqueeze(0)], dim=1)
            if collect_timing:
                if is_cuda:
                    torch.cuda.synchronize()
                timing["kv_update_s"] += time.perf_counter() - t0

            # Sample next token from corrected distribution
            if collect_timing:
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            if logits_processor is not None:
                next_logits = logits_processor(input_ids, next_logits.unsqueeze(0))

            if temperature < 1e-5:
                if isinstance(next_logits, torch.Tensor) and next_logits.dim() == 1:
                    sample_token = next_logits.argmax().unsqueeze(0).unsqueeze(0)
                else:
                    sample_token = next_logits.argmax(dim=-1).unsqueeze(0)
            else:
                if isinstance(next_logits, torch.Tensor) and next_logits.dim() == 1:
                    probs = F.softmax(next_logits, dim=-1)
                    sample_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
                else:
                    probs = F.softmax(next_logits, dim=-1)
                    sample_token = torch.multinomial(probs.squeeze(0), num_samples=1).unsqueeze(0)

            input_ids = torch.cat([input_ids, sample_token], dim=1)
            new_token_count += 1
            stats.tokens_generated += 1
            if collect_timing:
                if is_cuda:
                    torch.cuda.synchronize()
                timing["sample_s"] += time.perf_counter() - t0

            # Check for EOS
            if eos_token_id is not None and sample_token.item() == eos_token_id:
                break

            # Clear tree mask and reset draft KV cache for next round
            self.model.tree_mask = None
            self.model.stable_kv = None

        stats.time_elapsed = time.time() - start_time
        return input_ids, stats if return_stats else None

    @torch.no_grad()
    def generate_simple(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs,
    ) -> Tuple[str, Optional[EagleGenerationStats]]:
        """
        Generate text from a string prompt.

        Args:
            prompt: Input text
            max_new_tokens: Maximum new tokens
            temperature: Sampling temperature
            **kwargs: Additional arguments for generate()

        Returns:
            output_text: Generated text
            stats: Generation statistics
        """
        assert self.model.tokenizer is not None, "Tokenizer not loaded"

        input_ids = self.model.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)

        output_ids, stats = self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

        output_text = self.model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text, stats


@torch.no_grad()
def benchmark_eagle(
    model: GemmaEagleModel,
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    warmup_runs: int = 1,
) -> Dict:
    """
    Benchmark EAGLE generation on a set of prompts.

    Args:
        model: GemmaEagleModel instance
        prompts: List of prompts to benchmark
        max_new_tokens: Tokens to generate per prompt
        temperature: Sampling temperature
        warmup_runs: Number of warmup runs before timing

    Returns:
        Dictionary with benchmark results
    """
    generator = EagleGenerator(model)

    # Warmup
    for _ in range(warmup_runs):
        generator.generate_simple(prompts[0], max_new_tokens=32, temperature=temperature)

    # Benchmark
    all_stats = []
    for prompt in prompts:
        _, stats = generator.generate_simple(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature
        )
        all_stats.append(stats)

    # Aggregate results
    total_tokens = sum(s.tokens_generated for s in all_stats)
    total_time = sum(s.time_elapsed for s in all_stats)
    total_forward = sum(s.forward_passes for s in all_stats)
    total_proposed = sum(s.total_proposed for s in all_stats)
    total_accepted = sum(s.total_accepted for s in all_stats)

    return {
        "total_tokens": total_tokens,
        "total_time": total_time,
        "tokens_per_second": total_tokens / total_time,
        "mean_accepted_length": total_tokens / total_forward,
        "acceptance_rate": total_accepted / max(1, total_proposed),
        "forward_passes": total_forward,
        "num_prompts": len(prompts),
    }
