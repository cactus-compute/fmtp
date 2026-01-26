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
import random

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
        "--num-positions",
        type=int,
        default=50000,
        help="Number of positions to evaluate",
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
        default=512,
        help="Batch size for evaluation",
    )
    return parser.parse_args()


def load_model(model_path: str, vocab_size: int, device: str):
    """Load trained retrieval model."""
    from nanochat.hst.retrieval import RetrievalMLP, load_svd_basis

    # Create model
    model = RetrievalMLP(
        vocab_size=vocab_size,
        hidden_dim=128,
        context_window=4,
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


def extract_positions(data_path: str, tokenizer, num_positions: int, context_window: int):
    """Extract evaluation positions from data."""
    positions = []  # List of (context, target_1, target_2, target_3)

    with open(data_path) as f:
        for line in f:
            if len(positions) >= num_positions * 2:  # Get extra to sample from
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

            # Extract positions - sample every 10th position to avoid redundancy
            for pos in range(context_window, len(tokens) - 3, 10):
                context = tokens[pos - context_window:pos]
                target_1 = tokens[pos]
                target_2 = tokens[pos + 1]
                target_3 = tokens[pos + 2]
                positions.append((context, target_1, target_2, target_3))

    # Randomly sample
    random.shuffle(positions)
    return positions[:num_positions]


@torch.no_grad()
def evaluate_branch_accuracy_batched(
    model,
    positions: list,
    device: str,
    batch_size: int = 512,
    k_values: list = [1, 10, 50, 80, 100],
):
    """
    Evaluate branch prediction accuracy with batching.

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
            "depth1_correct": 0,
            "depth2_correct": 0,
            "depth3_correct": 0,
            "total": 0,
        }
        for k in k_values
    }

    max_k = max(k_values)

    # Process in batches
    num_batches = (len(positions) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(positions))
        batch_positions = positions[batch_start:batch_end]

        # Prepare batch tensors
        contexts = torch.tensor([p[0] for p in batch_positions], device=device)
        targets_1 = torch.tensor([p[1] for p in batch_positions], device=device)
        targets_2 = torch.tensor([p[2] for p in batch_positions], device=device)
        targets_3 = torch.tensor([p[3] for p in batch_positions], device=device)

        B = contexts.shape[0]

        # Depth 1: predict first token
        logits_1 = model(contexts)  # [B, vocab_size]
        top_k_1_all = torch.topk(logits_1, k=max_k, dim=-1).indices  # [B, max_k]

        for k in k_values:
            top_k_1 = top_k_1_all[:, :k]  # [B, k]

            # Check depth 1 accuracy
            depth1_correct = (top_k_1 == targets_1.unsqueeze(1)).any(dim=1)  # [B]
            metrics[k]["depth1_correct"] += depth1_correct.sum().item()
            metrics[k]["total"] += B

            # For depth 2: we need target_1 to be in top-k to continue
            # Create new contexts: shift by 1, append target_1
            new_contexts = torch.cat([contexts[:, 1:], targets_1.unsqueeze(1)], dim=1)  # [B, context_window]

            # Get predictions for depth 2
            logits_2 = model(new_contexts)  # [B, vocab_size]
            top_k_2 = torch.topk(logits_2, k=k, dim=-1).indices  # [B, k]

            # Depth 2 is correct if: target_1 in top_k_1 AND target_2 in top_k_2
            depth2_t2_correct = (top_k_2 == targets_2.unsqueeze(1)).any(dim=1)  # [B]
            depth2_correct = depth1_correct & depth2_t2_correct
            metrics[k]["depth2_correct"] += depth2_correct.sum().item()

            # For depth 3: shift again, append target_2
            new_contexts_3 = torch.cat([new_contexts[:, 1:], targets_2.unsqueeze(1)], dim=1)

            # Get predictions for depth 3
            logits_3 = model(new_contexts_3)  # [B, vocab_size]
            top_k_3 = torch.topk(logits_3, k=k, dim=-1).indices  # [B, k]

            # Depth 3 is correct if: depth_2 correct AND target_3 in top_k_3
            depth3_t3_correct = (top_k_3 == targets_3.unsqueeze(1)).any(dim=1)  # [B]
            depth3_correct = depth2_correct & depth3_t3_correct
            metrics[k]["depth3_correct"] += depth3_correct.sum().item()

    # Compute final metrics
    results = {}
    for k in k_values:
        total = metrics[k]["total"]
        if total > 0:
            results[f"token1_recall@{k}"] = metrics[k]["depth1_correct"] / total
            results[f"branch2_recall@{k}"] = metrics[k]["depth2_correct"] / total
            results[f"branch3_recall@{k}"] = metrics[k]["depth3_correct"] / total
        else:
            results[f"token1_recall@{k}"] = 0.0
            results[f"branch2_recall@{k}"] = 0.0
            results[f"branch3_recall@{k}"] = 0.0

    results["total_positions"] = len(positions)
    return results


def main():
    args = parse_args()

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, vocab_size, args.device)

    print(f"Extracting positions from: {args.data_path}")
    positions = extract_positions(args.data_path, tokenizer, args.num_positions, args.context_window)
    print(f"Extracted {len(positions)} positions for evaluation")

    print("\nEvaluating branch accuracy...")
    results = evaluate_branch_accuracy_batched(
        model=model,
        positions=positions,
        device=args.device,
        batch_size=args.batch_size,
        k_values=[1, 10, 50, 80, 100],
    )

    print("\n" + "=" * 60)
    print("Branch Prediction Accuracy Results")
    print("=" * 60)

    print("\nSingle Token Recall (can we predict t+1?):")
    for k in [1, 10, 50, 80, 100]:
        key = f"token1_recall@{k}"
        print(f"  Recall@{k}: {results[key]:.4f} ({results[key]*100:.2f}%)")

    print("\n2-Token Branch Recall (can we predict t+1 AND t+2?):")
    for k in [1, 10, 50, 80, 100]:
        key = f"branch2_recall@{k}"
        print(f"  Recall@{k}: {results[key]:.4f} ({results[key]*100:.2f}%)")

    print("\n3-Token Branch Recall (can we predict t+1, t+2, AND t+3?):")
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
