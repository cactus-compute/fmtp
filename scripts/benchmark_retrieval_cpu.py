"""
Benchmark CPU inference speed for HST retrieval module.

Compares different context window sizes (K=4, 6, 8) and measures:
- Latency per forward pass with optimized top-k projection
- Throughput (samples/second) at different batch sizes

The key optimization: instead of projecting to full vocab (262k), we only
compute logits for the top-k candidates from MTP heads (typically 1k tokens).

Usage:
    uv run python -m scripts.benchmark_retrieval_cpu
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import statistics


@dataclass
class BenchmarkResult:
    context_window: int
    batch_size: int
    mean_latency_ms: float
    std_latency_ms: float
    throughput_samples_per_sec: float
    mlp_params: int
    mode: str  # "full_vocab" or "top_k"


class RetrievalMLPBenchmark(nn.Module):
    """
    Benchmark version of RetrievalMLP that supports both full vocab and top-k projection.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        context_window: int = 4,
        svd_rank: int = 64,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        self.svd_rank = svd_rank

        # SVD embedding
        self.svd_embedding = nn.Parameter(torch.randn(vocab_size, svd_rank) * 0.02)

        # 3-layer MLP
        input_dim = context_window * svd_rank
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, svd_rank, bias=False)

    def forward_full(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Full vocab projection (original, slow)."""
        B = token_ids.shape[0]
        svd_embeds = self.svd_embedding[token_ids]
        h = svd_embeds.reshape(B, -1)
        h = F.gelu(self.fc1(h))
        h = F.gelu(self.fc2(h))
        h_proj = self.fc3(h)
        # Full vocab projection: [B, svd_rank] @ [svd_rank, vocab_size]
        logits = h_proj @ self.svd_embedding.T
        return logits

    def forward_topk(self, token_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """
        Optimized top-k projection - only compute logits for candidate tokens.

        Args:
            token_ids: [B, K] context token IDs
            candidate_ids: [B, num_candidates] or [num_candidates] candidate token IDs from MTP heads

        Returns:
            logits: [B, num_candidates] logits for candidate tokens only
        """
        B = token_ids.shape[0]
        svd_embeds = self.svd_embedding[token_ids]
        h = svd_embeds.reshape(B, -1)
        h = F.gelu(self.fc1(h))
        h = F.gelu(self.fc2(h))
        h_proj = self.fc3(h)  # [B, svd_rank]

        # Get embeddings for candidates only
        if candidate_ids.dim() == 1:
            # Same candidates for all batch items: [num_candidates]
            candidate_embeds = self.svd_embedding[candidate_ids]  # [num_candidates, svd_rank]
            # [B, svd_rank] @ [svd_rank, num_candidates] -> [B, num_candidates]
            logits = h_proj @ candidate_embeds.T
        else:
            # Different candidates per batch item: [B, num_candidates]
            candidate_embeds = self.svd_embedding[candidate_ids]  # [B, num_candidates, svd_rank]
            # Batched dot product
            logits = torch.einsum('br,bnr->bn', h_proj, candidate_embeds)

        return logits


def count_mlp_params(model: RetrievalMLPBenchmark) -> int:
    """Count parameters in MLP layers only."""
    mlp_params = 0
    for name, param in model.named_parameters():
        if 'fc' in name:
            mlp_params += param.numel()
    return mlp_params


def benchmark_model(
    model: RetrievalMLPBenchmark,
    batch_size: int,
    num_candidates: int,
    use_topk: bool,
    num_warmup: int = 50,
    num_iterations: int = 200,
    device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark a single model configuration."""
    model = model.to(device)
    model.eval()

    # Create random input
    token_ids = torch.randint(0, model.vocab_size, (batch_size, model.context_window), device=device)

    # Simulate MTP head candidates - same for all batch items (common case)
    candidate_ids = torch.randint(0, model.vocab_size, (num_candidates,), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if use_topk:
                _ = model.forward_topk(token_ids, candidate_ids)
            else:
                _ = model.forward_full(token_ids)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            if use_topk:
                _ = model.forward_topk(token_ids, candidate_ids)
            else:
                _ = model.forward_full(token_ids)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    mean_latency = statistics.mean(latencies)
    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
    throughput = (batch_size * 1000) / mean_latency  # samples/sec

    return BenchmarkResult(
        context_window=model.context_window,
        batch_size=batch_size,
        mean_latency_ms=mean_latency,
        std_latency_ms=std_latency,
        throughput_samples_per_sec=throughput,
        mlp_params=count_mlp_params(model),
        mode="top_k" if use_topk else "full_vocab",
    )


def main():
    print("=" * 70)
    print("HST Retrieval Module CPU Benchmark")
    print("=" * 70)

    # Configuration
    vocab_size = 262144  # Gemma vocab size
    hidden_dim = 128
    svd_rank = 64
    context_windows = [4, 6, 8]
    batch_sizes = [1, 1000]
    num_candidates = 1000  # Top-k from MTP heads

    print("\nConfiguration:")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  SVD rank: {svd_rank}")
    print(f"  Context windows: {context_windows}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  MTP candidates (top-k): {num_candidates:,}")

    # Benchmark full vocab projection (baseline)
    print("\n" + "=" * 70)
    print("BASELINE: Full Vocabulary Projection (262k tokens)")
    print("=" * 70)

    full_results: list[BenchmarkResult] = []
    for k in context_windows:
        model = RetrievalMLPBenchmark(vocab_size, hidden_dim, k, svd_rank)
        for batch_size in batch_sizes:
            result = benchmark_model(model, batch_size, num_candidates, use_topk=False)
            full_results.append(result)
            print(f"K={k}, Batch {batch_size:4d}: {result.mean_latency_ms:7.3f} ± {result.std_latency_ms:5.3f} ms | {result.throughput_samples_per_sec:10.1f} samples/sec")

    # Benchmark optimized top-k projection
    print("\n" + "=" * 70)
    print(f"OPTIMIZED: Top-{num_candidates} Candidate Projection")
    print("=" * 70)

    topk_results: list[BenchmarkResult] = []
    for k in context_windows:
        model = RetrievalMLPBenchmark(vocab_size, hidden_dim, k, svd_rank)
        for batch_size in batch_sizes:
            result = benchmark_model(model, batch_size, num_candidates, use_topk=True)
            topk_results.append(result)
            print(f"K={k}, Batch {batch_size:4d}: {result.mean_latency_ms:7.3f} ± {result.std_latency_ms:5.3f} ms | {result.throughput_samples_per_sec:10.1f} samples/sec")

    # Speedup comparison
    print("\n" + "=" * 70)
    print("SPEEDUP: Top-k vs Full Vocab")
    print("=" * 70)
    print(f"{'K':<5} {'Batch':<8} {'Full (ms)':<12} {'Top-k (ms)':<12} {'Speedup':<10}")
    print("-" * 50)

    for full_r, topk_r in zip(full_results, topk_results):
        speedup = full_r.mean_latency_ms / topk_r.mean_latency_ms
        print(f"K={full_r.context_window:<3} {full_r.batch_size:<8} {full_r.mean_latency_ms:<12.3f} {topk_r.mean_latency_ms:<12.3f} {speedup:.1f}x")

    # Memory analysis
    print("\n" + "=" * 70)
    print("Memory Analysis")
    print("=" * 70)
    full_proj_bytes = vocab_size * svd_rank * 4
    topk_proj_bytes = num_candidates * svd_rank * 4
    print(f"Full vocab embedding read:  {full_proj_bytes / 1e6:.1f} MB")
    print(f"Top-{num_candidates} embedding read: {topk_proj_bytes / 1e6:.3f} MB")
    print(f"Memory reduction: {full_proj_bytes / topk_proj_bytes:.0f}x")

    print("\n" + "=" * 70)
    print("Conclusion")
    print("=" * 70)
    print(f"Using top-{num_candidates} candidates from MTP heads eliminates the")
    print("memory-bound full vocab projection, making context window size")
    print("the dominant factor in latency.")


if __name__ == "__main__":
    main()
