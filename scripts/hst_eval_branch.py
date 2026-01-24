"""
Evaluate HST retrieval module branch prediction accuracy.

Measures recall for predicting the next 2-3 tokens (branch accuracy):
- For each position, predict top-k candidates for the next token
- For each candidate, predict top-k candidates for the following token
- Measure what fraction of actual 2-gram and 3-gram sequences appear in the candidate set

Usage:
    uv run python -m scripts.hst_eval_branch \
        --model-path ~/.cache/nanochat/hst_retrieval/retrieval_phase1_final.pt \
        --data-path ~/fmtp/data/wildchat_distill_v2_gemma3_1b.jsonl \
        --num-samples 10000
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HST branch accuracy")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained retrieval model checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to evaluation data (JSONL format)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-3-1b-it",
        help="Base model for tokenizer",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=4,
        help="Context window size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation",
    )
    return parser.parse_args()


def load_model(model_path: str, vocab_size: int, device: str):
    """Load trained retrieval model."""
    from nanochat.hst.retrieval import RetrievalMixer, load_svd_basis

    # Create model
    model = RetrievalMixer(
        vocab_size=vocab_size,
        embed_dim=256,
        context_window=4,
        num_layers=2,
        svd_rank=64,
    )

    # Load SVD basis to initialize embedding
    try:
        compressed_vocab = load_svd_basis(rank=64, model_name="gemma")
        model.load_svd(compressed_vocab)
    except FileNotFoundError:
        print("Warning: SVD basis not found, initializing randomly")
        model.svd_embedding.data.normal_(0, 0.01)
        model._svd_initialized = True

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def load_data(data_path: str, tokenizer, num_samples: int, context_window: int):
    """Load and tokenize evaluation data."""
    samples = []

    with open(data_path) as f:
        for i, line in enumerate(f):
            if len(samples) >= num_samples:
                break

            data = json.loads(line)

            # Concatenate messages
            if "messages" in data:
                text = " ".join(m.get("content", "") for m in data["messages"])
            elif "text" in data:
                text = data["text"]
            else:
                continue

            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)

            # Need at least context_window + 3 tokens for evaluation
            if len(tokens) >= context_window + 3:
                samples.append(tokens)

    return samples


def evaluate_branch_accuracy(
    model,
    samples: list,
    context_window: int,
    device: str,
    batch_size: int = 256,
    k_values: list = [1, 10, 50, 80, 100],
):
    """
    Evaluate branch prediction accuracy.

    For each position:
    - Get top-k predictions for next token (depth 1)
    - For each top-k candidate, get top-k predictions for following token (depth 2)
    - For depth 3, repeat

    Measures:
    - 2-gram recall: Does the actual (t+1, t+2) pair appear in our candidates?
    - 3-gram recall: Does the actual (t+1, t+2, t+3) triple appear in our candidates?
    """
    model.eval()

    # Metrics for different k values and depths
    metrics = {
        k: {
            "depth2_correct": 0,
            "depth3_correct": 0,
            "total": 0,
        }
        for k in k_values
    }

    total_positions = 0

    with torch.no_grad():
        for tokens in tqdm(samples, desc="Evaluating"):
            # Slide through the sequence
            for pos in range(context_window, len(tokens) - 3):
                context = tokens[pos - context_window:pos]
                target_1 = tokens[pos]      # Next token
                target_2 = tokens[pos + 1]  # Token after that
                target_3 = tokens[pos + 2]  # Third token

                # Get predictions for depth 1
                context_tensor = torch.tensor([context], device=device)
                logits_1 = model(context_tensor)  # [1, vocab_size]

                for k in k_values:
                    # Get top-k predictions for depth 1
                    top_k_1 = torch.topk(logits_1[0], k=k).indices.tolist()

                    # Check if target_1 is in top-k
                    if target_1 not in top_k_1:
                        # Can't get correct 2-gram or 3-gram if first token is wrong
                        metrics[k]["total"] += 1
                        continue

                    # For depth 2: check all k candidates from depth 1
                    # Build context for each candidate
                    depth2_found = False
                    depth3_found = False

                    # Check if we can find the 2-gram (target_1, target_2)
                    # We need target_1 in top-k (checked above) AND target_2 in top-k given target_1
                    new_context = context[1:] + [target_1]  # Shift context, add target_1
                    context_tensor_2 = torch.tensor([new_context], device=device)
                    logits_2 = model(context_tensor_2)
                    top_k_2 = torch.topk(logits_2[0], k=k).indices.tolist()

                    if target_2 in top_k_2:
                        depth2_found = True

                        # For depth 3: check if target_3 is reachable
                        new_context_3 = new_context[1:] + [target_2]
                        context_tensor_3 = torch.tensor([new_context_3], device=device)
                        logits_3 = model(context_tensor_3)
                        top_k_3 = torch.topk(logits_3[0], k=k).indices.tolist()

                        if target_3 in top_k_3:
                            depth3_found = True

                    if depth2_found:
                        metrics[k]["depth2_correct"] += 1
                    if depth3_found:
                        metrics[k]["depth3_correct"] += 1
                    metrics[k]["total"] += 1

                total_positions += 1

    # Compute final metrics
    results = {}
    for k in k_values:
        total = metrics[k]["total"]
        if total > 0:
            results[f"branch2_recall@{k}"] = metrics[k]["depth2_correct"] / total
            results[f"branch3_recall@{k}"] = metrics[k]["depth3_correct"] / total
        else:
            results[f"branch2_recall@{k}"] = 0.0
            results[f"branch3_recall@{k}"] = 0.0

    results["total_positions"] = total_positions
    return results


def main():
    args = parse_args()

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, vocab_size, args.device)

    print(f"Loading data from: {args.data_path}")
    samples = load_data(args.data_path, tokenizer, args.num_samples, args.context_window)
    print(f"Loaded {len(samples)} samples")

    print("\nEvaluating branch accuracy...")
    results = evaluate_branch_accuracy(
        model=model,
        samples=samples,
        context_window=args.context_window,
        device=args.device,
        batch_size=args.batch_size,
        k_values=[1, 10, 50, 80, 100],
    )

    print("\n" + "=" * 60)
    print("Branch Prediction Accuracy Results")
    print("=" * 60)

    print("\n2-Token Branch Recall (can we predict both t+1 and t+2?):")
    for k in [1, 10, 50, 80, 100]:
        key = f"branch2_recall@{k}"
        print(f"  Recall@{k}: {results[key]:.4f} ({results[key]*100:.2f}%)")

    print("\n3-Token Branch Recall (can we predict t+1, t+2, and t+3?):")
    for k in [1, 10, 50, 80, 100]:
        key = f"branch3_recall@{k}"
        print(f"  Recall@{k}: {results[key]:.4f} ({results[key]*100:.2f}%)")

    print(f"\nTotal positions evaluated: {results['total_positions']}")

    # Save results
    output_path = Path(args.model_path).parent / "branch_accuracy.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
