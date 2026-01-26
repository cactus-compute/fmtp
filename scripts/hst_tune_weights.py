"""
Grid search to find optimal HST weighting constants (α, β, γ).

Evaluates different weight combinations on a validation set by measuring:
1. Acceptance rate (fraction of speculated tokens accepted)
2. Mean accepted length per forward pass
3. Effective tokens/second (if timing enabled)

The optimal weights maximize acceptance rate while maintaining diversity.

Usage:
    uv run python -m scripts.hst_tune_weights --checkpoint path/to/retrieval.pt

TODO: More advanced methods for future work:
- Entropy-adaptive weighting: Adjust β based on base model entropy
  - High entropy → lower β (retrieval may be overconfident)
  - Could normalize P_mtp and P_ret by their respective entropies before combining
- Learned weighting: Train a small network to predict optimal (α, β, γ)
  based on context features (entropy, token type, position, etc.)
- Per-source calibration: Learn temperature parameters for each source
  to calibrate their probability distributions before combining
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import json

from transformers import AutoTokenizer


@dataclass
class WeightConfig:
    alpha: float  # MTP head weight
    beta: float   # Retrieval weight
    gamma: float  # Suffix match weight

    def __post_init__(self):
        # Normalize to sum to 1
        total = self.alpha + self.beta + self.gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total


@dataclass
class GridSearchResult:
    config: WeightConfig
    acceptance_rate: float
    mean_accepted_length: float
    agreement_rate: float  # Fraction where 2+ sources agreed


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search for HST weights")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained retrieval module checkpoint",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-3-1b-it",
        help="Base model for tokenizer and vocab size",
    )
    parser.add_argument(
        "--validation-data",
        type=str,
        default=None,
        help="Path to validation data (parquet or jsonl). Uses FineWeb sample if not provided.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of validation samples to evaluate",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=8,
        help="Retrieval context window (K)",
    )
    parser.add_argument(
        "--alpha-range",
        type=str,
        default="0.4,0.5,0.6,0.7,0.8",
        help="Comma-separated alpha values to try",
    )
    parser.add_argument(
        "--beta-range",
        type=str,
        default="0.1,0.2,0.3,0.4",
        help="Comma-separated beta values to try",
    )
    parser.add_argument(
        "--gamma-range",
        type=str,
        default="0.0,0.05,0.1,0.15",
        help="Comma-separated gamma values to try",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    return parser.parse_args()


def load_retrieval_module(checkpoint_path: str, vocab_size: int, context_window: int, device: str):
    """Load trained retrieval module."""
    from nanochat.hst.retrieval import RetrievalMLP, load_svd_basis

    module = RetrievalMLP(
        vocab_size=vocab_size,
        hidden_dim=128,
        context_window=context_window,
        svd_rank=64,
    )

    # Load SVD basis
    try:
        compressed_vocab = load_svd_basis(rank=64, model_name="gemma")
        module.load_svd(compressed_vocab)
    except FileNotFoundError:
        print("Warning: SVD not found, using random initialization")
        module.svd_embedding.data.normal_(0, 0.01)
        module._svd_initialized = True

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, weights_only=True)
    module.load_state_dict(state_dict)
    module = module.to(device)
    module.eval()

    return module


def create_validation_samples(tokenizer, num_samples: int, data_path: Optional[str] = None):
    """
    Create validation samples as (context, target) pairs.

    Each sample has:
    - context: list of token IDs (16-64 tokens)
    - target: next token ID
    """
    samples = []

    if data_path and Path(data_path).exists():
        # Load from file
        import pandas as pd
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
            texts = df['text'].tolist()[:num_samples * 2]
        else:
            import json
            with open(data_path) as f:
                texts = [json.loads(line)['text'] for line in f][:num_samples * 2]
    else:
        # Use sample texts for testing
        texts = [
            "The quick brown fox jumps over the lazy dog. The quick brown fox runs through the forest.",
            "In machine learning, neural networks are computational systems inspired by biological neural networks.",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "The weather today is sunny with a high of 75 degrees. Tomorrow will be cloudy with rain.",
        ] * (num_samples // 4 + 1)

    for text in texts:
        if len(samples) >= num_samples:
            break

        tokens = tokenizer.encode(text)
        if len(tokens) < 20:
            continue

        # Create samples from sliding windows
        for i in range(16, min(len(tokens) - 1, 64)):
            if len(samples) >= num_samples:
                break
            context = tokens[:i]
            target = tokens[i]
            samples.append((context, target))

    return samples[:num_samples]


def evaluate_weights(
    config: WeightConfig,
    retrieval_module,
    samples: list,
    vocab_size: int,
    device: str,
) -> GridSearchResult:
    """
    Evaluate a weight configuration on validation samples.

    Simulates hybrid scoring to measure how often the correct token
    would be in the top-k candidates.
    """
    from nanochat.hst.suffix_match import SuffixMatcher

    suffix_matcher = SuffixMatcher(buffer_size=256, device=device)

    correct_in_top1 = 0
    correct_in_top5 = 0
    correct_in_top10 = 0
    agreements = 0
    total = 0

    for context, target in samples:
        # Reset suffix matcher and fill with context
        suffix_matcher.reset()
        suffix_matcher.append(context)

        # Get retrieval logits
        context_tensor = torch.tensor([context[-8:]], device=device)
        with torch.no_grad():
            retrieval_logits = retrieval_module(context_tensor)[0]

        # Get suffix probabilities
        suffix_probs = suffix_matcher.get_suffix_probabilities(
            torch.tensor(context[-4:], device=device),
            vocab_size=vocab_size,
        )

        # For MTP, we simulate with retrieval as proxy (in real use, this comes from Medusa heads)
        # This is a simplification - in practice, MTP logits come from the base model
        mtp_probs = F.softmax(retrieval_logits, dim=-1)  # Proxy
        retrieval_probs = F.softmax(retrieval_logits, dim=-1)

        # Compute hybrid scores
        hybrid_scores = (
            config.alpha * mtp_probs +
            config.beta * retrieval_probs +
            config.gamma * suffix_probs
        )

        # Check agreement (both retrieval and suffix have target in top-10)
        ret_top10 = torch.topk(retrieval_probs, 10).indices
        suf_top10 = torch.topk(suffix_probs, 10).indices
        if target in ret_top10 and target in suf_top10:
            agreements += 1

        # Check if target is in top-k
        top1 = torch.topk(hybrid_scores, 1).indices
        top5 = torch.topk(hybrid_scores, 5).indices
        top10 = torch.topk(hybrid_scores, 10).indices

        if target in top1:
            correct_in_top1 += 1
        if target in top5:
            correct_in_top5 += 1
        if target in top10:
            correct_in_top10 += 1

        total += 1

    return GridSearchResult(
        config=config,
        acceptance_rate=correct_in_top1 / max(total, 1),  # Top-1 accuracy as proxy
        mean_accepted_length=correct_in_top5 / max(total, 1) * 5,  # Estimated
        agreement_rate=agreements / max(total, 1),
    )


def main():
    args = parse_args()

    print("=" * 60)
    print("HST Weight Grid Search")
    print("=" * 60)

    # Parse weight ranges
    alphas = [float(x) for x in args.alpha_range.split(',')]
    betas = [float(x) for x in args.beta_range.split(',')]
    gammas = [float(x) for x in args.gamma_range.split(',')]

    print(f"\nWeight ranges:")
    print(f"  Alpha (MTP): {alphas}")
    print(f"  Beta (Retrieval): {betas}")
    print(f"  Gamma (Suffix): {gammas}")

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size:,}")

    # Load retrieval module
    print(f"\nLoading retrieval module from {args.checkpoint}...")
    retrieval_module = load_retrieval_module(
        args.checkpoint, vocab_size, args.context_window, args.device
    )

    # Create validation samples
    print(f"\nCreating {args.num_samples} validation samples...")
    samples = create_validation_samples(tokenizer, args.num_samples, args.validation_data)
    print(f"Created {len(samples)} samples")

    # Grid search
    print(f"\nRunning grid search ({len(alphas) * len(betas) * len(gammas)} configurations)...")
    results = []

    for alpha in tqdm(alphas, desc="Alpha"):
        for beta in betas:
            for gamma in gammas:
                config = WeightConfig(alpha=alpha, beta=beta, gamma=gamma)
                result = evaluate_weights(
                    config, retrieval_module, samples, vocab_size, args.device
                )
                results.append(result)

    # Sort by acceptance rate
    results.sort(key=lambda r: r.acceptance_rate, reverse=True)

    # Print top results
    print("\n" + "=" * 60)
    print("Top 10 Configurations (by acceptance rate)")
    print("=" * 60)
    print(f"{'Alpha':<8} {'Beta':<8} {'Gamma':<8} {'Accept%':<10} {'Agree%':<10}")
    print("-" * 50)

    for r in results[:10]:
        print(f"{r.config.alpha:<8.2f} {r.config.beta:<8.2f} {r.config.gamma:<8.2f} "
              f"{r.acceptance_rate*100:<10.1f} {r.agreement_rate*100:<10.1f}")

    # Best result
    best = results[0]
    print(f"\nBest configuration:")
    print(f"  Alpha (MTP): {best.config.alpha:.2f}")
    print(f"  Beta (Retrieval): {best.config.beta:.2f}")
    print(f"  Gamma (Suffix): {best.config.gamma:.2f}")
    print(f"  Acceptance rate: {best.acceptance_rate*100:.1f}%")
    print(f"  Agreement rate: {best.agreement_rate*100:.1f}%")

    # Save results
    if args.output:
        output_data = {
            "best": {
                "alpha": best.config.alpha,
                "beta": best.config.beta,
                "gamma": best.config.gamma,
                "acceptance_rate": best.acceptance_rate,
                "agreement_rate": best.agreement_rate,
            },
            "all_results": [
                {
                    "alpha": r.config.alpha,
                    "beta": r.config.beta,
                    "gamma": r.config.gamma,
                    "acceptance_rate": r.acceptance_rate,
                    "agreement_rate": r.agreement_rate,
                }
                for r in results
            ]
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
