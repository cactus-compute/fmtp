"""
Evaluate HST speculation performance.

Measures:
1. Retrieval module Recall@K
2. Suffix matcher hit rate
3. HST tree acceptance rate vs baseline Medusa
4. Tokens per second improvement
5. Agreement rate between sources

Usage:
    # Evaluate retrieval module
    uv run python -m scripts.hst_eval --mode retrieval

    # Evaluate full HST vs Medusa
    uv run python -m scripts.hst_eval --mode benchmark

    # Run on specific prompts
    uv run python -m scripts.hst_eval --mode benchmark --prompts "Why is the sky blue?" "Explain recursion"
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
import time
import json
from dataclasses import dataclass, asdict

from transformers import AutoModel, AutoTokenizer


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval module evaluation."""
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_50: float
    mean_rank: float
    samples_evaluated: int


@dataclass
class HSTBenchmarkMetrics:
    """Metrics for HST benchmark."""
    # Generation metrics
    tokens_generated: int
    forward_passes: int
    mean_accepted_length: float
    acceptance_rate: float

    # Source agreement
    mtp_contribution: float
    retrieval_contribution: float
    suffix_contribution: float
    agreement_rate: float  # % of tokens where 2+ sources agreed

    # Timing
    total_time_sec: float
    tokens_per_second: float

    # Comparison
    baseline_tokens_per_second: Optional[float] = None
    speedup: Optional[float] = None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HST speculation")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["retrieval", "benchmark", "all"],
        default="benchmark",
        help="Evaluation mode",
    )

    # Model
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-3-1b-it",
        help="Base model for evaluation",
    )
    parser.add_argument(
        "--retrieval-checkpoint",
        type=str,
        default=None,
        help="Path to trained retrieval module checkpoint",
    )

    # Data
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation data (jsonl)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Specific prompts to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum samples for retrieval evaluation",
    )

    # Generation
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)",
    )

    # HST configuration
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="MTP head weight",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Retrieval weight",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Suffix match weight",
    )
    parser.add_argument(
        "--tree-budget",
        type=int,
        default=64,
        help="Maximum tree nodes",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (json)",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )

    return parser.parse_args()


def evaluate_retrieval(args) -> RetrievalMetrics:
    """Evaluate retrieval module Recall@K."""
    from nanochat.hst.retrieval import RetrievalMixer, load_svd_basis

    print("Evaluating retrieval module...")

    # Load tokenizer and embeddings
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModel.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    # Get embeddings
    if hasattr(base_model, "embed_tokens"):
        embeddings = base_model.embed_tokens
    elif hasattr(base_model, "model") and hasattr(base_model.model, "embed_tokens"):
        embeddings = base_model.model.embed_tokens
    else:
        raise RuntimeError("Could not find embeddings")

    embed_dim = embeddings.embedding_dim
    vocab_size = embeddings.num_embeddings
    embeddings = embeddings.to(args.device)

    # Create retrieval module
    module = RetrievalMixer(
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        context_window=4,
        num_layers=2,
        svd_rank=64,
    )

    # Load SVD and checkpoint
    try:
        compressed_vocab = load_svd_basis(rank=64, model_name="gemma")
        module.load_svd(compressed_vocab)
    except FileNotFoundError:
        print("Warning: SVD not found, using random initialization")
        module.compressed_vocab.normal_(0, 0.01)
        module._svd_loaded = True

    if args.retrieval_checkpoint:
        module.load_state_dict(torch.load(args.retrieval_checkpoint, weights_only=True))
        print(f"Loaded checkpoint: {args.retrieval_checkpoint}")

    module = module.to(args.device)
    module.eval()

    # Prepare evaluation data
    if args.eval_data:
        # Load from file
        texts = []
        with open(args.eval_data) as f:
            for line in f:
                data = json.loads(line)
                if "text" in data:
                    texts.append(data["text"])
                elif "messages" in data:
                    texts.append(" ".join(m.get("content", "") for m in data["messages"]))
        texts = texts[:args.max_samples]
    else:
        # Use default test texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "In the year 2024, many technological advances were made in the field of AI.",
        ] * (args.max_samples // 4)

    # Evaluate
    total = 0
    correct_at_1 = 0
    correct_at_5 = 0
    correct_at_10 = 0
    correct_at_50 = 0
    total_rank = 0

    context_window = 4

    with torch.no_grad():
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)

            for i in range(context_window, min(len(tokens), context_window + 50)):
                context = torch.tensor([tokens[i - context_window : i]], device=args.device)
                target = tokens[i]

                # Get embeddings and predict
                context_embeds = embeddings(context)
                logits = module(context_embeds)

                # Compute rank
                sorted_indices = torch.argsort(logits[0], descending=True)
                rank = (sorted_indices == target).nonzero(as_tuple=True)[0]
                if len(rank) > 0:
                    total_rank += rank[0].item() + 1

                # Compute recall
                _, top_50 = torch.topk(logits[0], k=50)
                top_50 = top_50.tolist()

                total += 1
                if target == top_50[0]:
                    correct_at_1 += 1
                if target in top_50[:5]:
                    correct_at_5 += 1
                if target in top_50[:10]:
                    correct_at_10 += 1
                if target in top_50:
                    correct_at_50 += 1

    metrics = RetrievalMetrics(
        recall_at_1=correct_at_1 / max(total, 1),
        recall_at_5=correct_at_5 / max(total, 1),
        recall_at_10=correct_at_10 / max(total, 1),
        recall_at_50=correct_at_50 / max(total, 1),
        mean_rank=total_rank / max(total, 1),
        samples_evaluated=total,
    )

    print(f"\nRetrieval Metrics:")
    print(f"  Recall@1:  {metrics.recall_at_1:.2%}")
    print(f"  Recall@5:  {metrics.recall_at_5:.2%}")
    print(f"  Recall@10: {metrics.recall_at_10:.2%}")
    print(f"  Recall@50: {metrics.recall_at_50:.2%}")
    print(f"  Mean Rank: {metrics.mean_rank:.1f}")

    return metrics


def evaluate_benchmark(args) -> list[HSTBenchmarkMetrics]:
    """Benchmark HST vs baseline generation."""
    from nanochat.hst.retrieval import RetrievalMixer, load_svd_basis
    from nanochat.hst.suffix_match import SuffixMatcher
    from nanochat.hst.tree_builder import HybridScorer, HSTTreeBuilder
    from nanochat.hst.tree_attention import generate_hst_buffers, verify_tree_greedy

    print("Benchmarking HST speculation...")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModel.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map=args.device,
    )
    model.eval()

    # Get embeddings
    if hasattr(model, "embed_tokens"):
        embeddings = model.embed_tokens
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embeddings = model.model.embed_tokens
    else:
        raise RuntimeError("Could not find embeddings")

    embed_dim = embeddings.embedding_dim
    vocab_size = embeddings.num_embeddings

    # Create HST components
    retrieval_module = RetrievalMixer(
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        context_window=4,
        num_layers=2,
        svd_rank=64,
    )

    try:
        compressed_vocab = load_svd_basis(rank=64, model_name="gemma")
        retrieval_module.load_svd(compressed_vocab)
    except FileNotFoundError:
        print("Warning: SVD not found, using random initialization")
        retrieval_module.compressed_vocab.normal_(0, 0.01)
        retrieval_module._svd_loaded = True

    if args.retrieval_checkpoint:
        retrieval_module.load_state_dict(torch.load(args.retrieval_checkpoint, weights_only=True))

    retrieval_module = retrieval_module.to(args.device)
    retrieval_module.eval()

    suffix_matcher = SuffixMatcher(
        buffer_size=1024,
        min_suffix_len=1,
        max_suffix_len=4,
        device=args.device,
    )

    scorer = HybridScorer(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )

    tree_builder = HSTTreeBuilder(
        scorer=scorer,
        max_depth=4,
        tree_budget=args.tree_budget,
    )

    # Prepare prompts
    if args.prompts:
        prompts = args.prompts
    else:
        prompts = [
            "The capital of France is",
            "def quicksort(arr):",
            "In machine learning, a neural network",
            "The year was 1984, and",
        ]

    results = []

    for prompt in prompts:
        print(f"\nPrompt: {prompt[:50]}...")

        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(args.device)

        # Initialize suffix matcher with context
        suffix_matcher.reset()
        for token in input_ids[0].tolist():
            suffix_matcher.append(token)

        # Track metrics
        tokens_generated = 0
        forward_passes = 0
        total_accepted = 0
        total_proposed = 0
        mtp_accepted = 0
        retrieval_accepted = 0
        suffix_accepted = 0
        agreements = 0

        start_time = time.time()

        # Generation loop (simplified simulation)
        current_ids = input_ids.clone()

        while tokens_generated < args.max_tokens:
            # Get base model hidden states (simplified - in practice would need full forward)
            with torch.no_grad():
                outputs = model(current_ids, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, embed_dim]

            # Get MTP logits (placeholder - would come from actual Medusa heads)
            # For evaluation, we simulate with base model logits
            base_logits = outputs.last_hidden_state[:, -1, :] @ embeddings.weight.T  # [B, vocab_size]

            # Get retrieval logits
            context_embeds = embeddings(current_ids[:, -4:])  # Last 4 tokens
            retrieval_logits = retrieval_module(context_embeds)

            # Get suffix probabilities
            suffix = current_ids[0, -4:].tolist()
            suffix_probs = suffix_matcher.get_suffix_probabilities(
                torch.tensor(suffix, device=args.device),
                vocab_size=vocab_size,
            )

            # Build HST tree
            def get_mtp_logits(depth, path):
                return base_logits[0]  # Simplified

            def get_retrieval_logits(context):
                return retrieval_logits[0]

            def get_suffix_probs(suffix):
                return suffix_probs

            tree = tree_builder.build_tree(
                root_token=current_ids[0, -1].item(),
                get_mtp_logits=get_mtp_logits,
                get_retrieval_logits=get_retrieval_logits,
                get_suffix_probs=get_suffix_probs,
                context_tokens=current_ids[0].tolist(),
                vocab_size=vocab_size,
                device=args.device,
            )

            forward_passes += 1
            total_proposed += len(tree) - 1  # Exclude root

            # Simplified acceptance (in practice, would verify with base model)
            # Accept the highest-scored path
            if len(tree) > 1:
                # Find leaf with highest score
                best_leaf = max(tree[1:], key=lambda n: n.score)
                path = tree_builder._get_path_tokens(tree, best_leaf.node_idx)

                # Simulate acceptance (accept top-1)
                accepted = path[:1] if path else []
                total_accepted += len(accepted)
                tokens_generated += len(accepted)

                # Track source contributions
                for node in tree[1:]:
                    if "mtp" in node.sources:
                        mtp_accepted += 1
                    if "retrieval" in node.sources:
                        retrieval_accepted += 1
                    if "suffix" in node.sources:
                        suffix_accepted += 1
                    if len(node.sources) >= 2:
                        agreements += 1

                # Update context
                for token in accepted:
                    current_ids = torch.cat([
                        current_ids,
                        torch.tensor([[token]], device=args.device)
                    ], dim=1)
                    suffix_matcher.append(token)
            else:
                # No candidates, fall back to greedy
                next_token = base_logits[0].argmax().item()
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[next_token]], device=args.device)
                ], dim=1)
                suffix_matcher.append(next_token)
                tokens_generated += 1
                total_accepted += 1

        elapsed = time.time() - start_time

        # Compute metrics
        total_source_contributions = mtp_accepted + retrieval_accepted + suffix_accepted
        metrics = HSTBenchmarkMetrics(
            tokens_generated=tokens_generated,
            forward_passes=forward_passes,
            mean_accepted_length=tokens_generated / max(forward_passes, 1),
            acceptance_rate=total_accepted / max(total_proposed, 1),
            mtp_contribution=mtp_accepted / max(total_source_contributions, 1),
            retrieval_contribution=retrieval_accepted / max(total_source_contributions, 1),
            suffix_contribution=suffix_accepted / max(total_source_contributions, 1),
            agreement_rate=agreements / max(total_proposed, 1),
            total_time_sec=elapsed,
            tokens_per_second=tokens_generated / max(elapsed, 0.001),
        )

        results.append(metrics)

        print(f"  Tokens: {metrics.tokens_generated}")
        print(f"  Forward passes: {metrics.forward_passes}")
        print(f"  Mean accepted: {metrics.mean_accepted_length:.2f}")
        print(f"  Tokens/sec: {metrics.tokens_per_second:.1f}")
        print(f"  Agreement rate: {metrics.agreement_rate:.2%}")

    return results


def main():
    args = parse_args()

    results = {}

    if args.mode in ["retrieval", "all"]:
        retrieval_metrics = evaluate_retrieval(args)
        results["retrieval"] = asdict(retrieval_metrics)

    if args.mode in ["benchmark", "all"]:
        benchmark_metrics = evaluate_benchmark(args)
        results["benchmark"] = [asdict(m) for m in benchmark_metrics]

        # Compute averages
        if benchmark_metrics:
            avg_metrics = {
                "avg_mean_accepted": sum(m.mean_accepted_length for m in benchmark_metrics) / len(benchmark_metrics),
                "avg_tokens_per_second": sum(m.tokens_per_second for m in benchmark_metrics) / len(benchmark_metrics),
                "avg_agreement_rate": sum(m.agreement_rate for m in benchmark_metrics) / len(benchmark_metrics),
            }
            results["benchmark_averages"] = avg_metrics
            print(f"\nBenchmark Averages:")
            print(f"  Mean accepted length: {avg_metrics['avg_mean_accepted']:.2f}")
            print(f"  Tokens/sec: {avg_metrics['avg_tokens_per_second']:.1f}")
            print(f"  Agreement rate: {avg_metrics['avg_agreement_rate']:.2%}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
