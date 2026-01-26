#!/usr/bin/env python
"""
Evaluate tree selection acceptance rates for speculative decoding.

This script measures acceptance rates with different tree configurations:
1. Current tree (baseline with existing budget, e.g., 64 nodes)
2. Enlarged tree (larger budget for theoretical comparison, e.g., 1k nodes)
3. HST-scored tree (optional: uses hybrid scoring with retrieval model)

The HST mode uses the HSTScorer callback to re-rank candidates using:
- MTP head predictions (α=0.6)
- Learned retrieval predictions from K=8 context window (β=0.3)
- Suffix matching for in-context repetition (γ=0.1)

Usage:
    # Basic evaluation with default settings
    uv run python -m scripts.hst_eval_tree_acceptance --checkpoint path/to/medusa_checkpoint

    # With HST scoring (requires retrieval checkpoint)
    uv run python -m scripts.hst_eval_tree_acceptance --checkpoint path/to/ckpt \
        --hst-checkpoint ~/.cache/nanochat/hst_retrieval_ablation/k8/retrieval_phase1_final.pt

    # Custom tree sizes
    uv run python -m scripts.hst_eval_tree_acceptance --checkpoint path/to/ckpt \
        --current-budget 64 --enlarged-budget 1000

    # Specify evaluation prompts
    uv run python -m scripts.hst_eval_tree_acceptance --checkpoint path/to/ckpt \
        --prompts "Why is the sky blue?" "Write a Python function"
"""

import argparse
import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import time


@dataclass
class TreeAcceptanceMetrics:
    """Metrics for tree acceptance evaluation."""
    # Current tree metrics
    current_budget: int
    current_mean_acceptance: float  # Mean accepted tokens per forward pass
    current_acceptance_rate: float  # Accepted / proposed
    current_total_accepted: int
    current_total_proposed: int

    # Enlarged tree metrics (without re-running model)
    enlarged_budget: int
    enlarged_mean_acceptance: float
    enlarged_acceptance_rate: float
    enlarged_total_accepted: int
    enlarged_total_proposed: int

    # HST-scored tree metrics (optional)
    hst_enabled: bool = False
    hst_mean_acceptance: float = 0.0
    hst_acceptance_rate: float = 0.0
    hst_total_accepted: int = 0
    hst_total_proposed: int = 0

    # Improvement metrics
    acceptance_improvement: float = 0.0  # enlarged_mean - current_mean
    acceptance_improvement_pct: float = 0.0  # (enlarged_mean - current_mean) / current_mean * 100
    hst_improvement: float = 0.0  # hst_mean - current_mean
    hst_improvement_pct: float = 0.0  # (hst_mean - current_mean) / current_mean * 100

    # Details
    num_prompts: int = 0
    tokens_generated: int = 0
    forward_passes: int = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate tree acceptance rates for speculative decoding"
    )

    # Model/checkpoint
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to Medusa checkpoint directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model name (auto-detected from checkpoint config)",
    )

    # Tree configuration
    parser.add_argument(
        "--current-budget",
        type=int,
        default=64,
        help="Current tree budget (number of nodes)",
    )
    parser.add_argument(
        "--enlarged-budget",
        type=int,
        default=1000,
        help="Enlarged tree budget for theoretical comparison",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum tree depth",
    )

    # Evaluation settings
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Prompts to evaluate (default: built-in set)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per prompt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )

    # HST configuration
    parser.add_argument(
        "--hst-checkpoint",
        type=str,
        default=None,
        help="Path to HST retrieval module checkpoint (enables HST scoring mode)",
    )
    parser.add_argument(
        "--hst-alpha",
        type=float,
        default=0.6,
        help="HST weight for MTP head predictions",
    )
    parser.add_argument(
        "--hst-beta",
        type=float,
        default=0.3,
        help="HST weight for retrieval predictions",
    )
    parser.add_argument(
        "--hst-gamma",
        type=float,
        default=0.1,
        help="HST weight for suffix match predictions",
    )
    parser.add_argument(
        "--hst-no-pruning",
        action="store_true",
        help="Disable threshold-based pruning (set score_threshold to 0)",
    )
    parser.add_argument(
        "--hst-blend-mode",
        type=str,
        choices=["convex", "agreement"],
        default="agreement",
        help="Blending mode: 'convex' for standard blend, 'agreement' for boost-when-agree",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def create_hst_scorer(
    vocab_size: int,
    device: str,
    checkpoint_path: Optional[str] = None,
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1,
    no_pruning: bool = False,
    blend_mode: str = "agreement",
):
    """Create HST scorer with retrieval module."""
    from nanochat.gemma_medusa.hst_scorer import HSTScorer

    device_obj = torch.device(device)
    dtype = torch.bfloat16 if device_obj.type == "cuda" else torch.float32

    # Set score_threshold to 0 to disable pruning if requested
    score_threshold = 0.0 if no_pruning else 0.01

    scorer = HSTScorer(
        vocab_size=vocab_size,
        device=device_obj,
        dtype=dtype,
        retrieval_checkpoint=checkpoint_path,
        retrieval_context_window=8,  # K=8 from ablation
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        score_threshold=score_threshold,
        blend_mode=blend_mode,
    )

    return scorer


def load_model(checkpoint_dir: str, device: str, model_name: Optional[str] = None):
    """Load Medusa model from checkpoint."""
    from nanochat.gemma_medusa.model import GemmaMedusaModel

    checkpoint_path = Path(checkpoint_dir)

    # Find the checkpoint file
    if (checkpoint_path / "final" / "medusa_heads.pt").exists():
        ckpt_file = checkpoint_path / "final" / "medusa_heads.pt"
    elif (checkpoint_path / "medusa_heads.pt").exists():
        ckpt_file = checkpoint_path / "medusa_heads.pt"
    else:
        raise FileNotFoundError(f"No medusa_heads.pt found in {checkpoint_dir}")

    # Load config
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Try to load from checkpoint
        checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=True)
        config = checkpoint.get('config', {})

    # Use provided model name or from config (handle both 'model_name' and 'base_model' keys)
    if model_name is None:
        model_name = config.get('model_name') or config.get('base_model', 'google/gemma-3-270m-it')

    device_obj = torch.device(device)
    dtype = torch.bfloat16 if device_obj.type == "cuda" else torch.float32

    model = GemmaMedusaModel(
        model_name=model_name,
        medusa_num_heads=config.get('medusa_num_heads', 4),
        medusa_num_layers=config.get('medusa_num_layers', 2),
        lora_rank=config.get('lora_rank', 256),
        lora_alpha=config.get('lora_alpha', 512),
        device=device_obj,
        dtype=dtype,
        freeze_base=True,
        zero_init_mlp=config.get('zero_init_mlp', True),
        use_head_mixer=config.get('use_head_mixer', True),
        mixer_type=config.get('mixer_type', 'attention'),
        attn_num_layers=config.get('attn_num_layers', 2),
    )

    checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=True)
    model.load_medusa_state_dict(checkpoint, strict=False)
    model.eval()

    return model, config


def get_tree_choices_for_budget(
    num_heads: int,
    budget: int,
    max_depth: int,
    checkpoint_path: Optional[str] = None,
) -> list:
    """
    Generate optimal tree choices for a given budget.

    Uses the actual optimal tree generation algorithm from the model code,
    which is based on measured head accuracies from training.
    """
    from nanochat.gemma_medusa.model import generate_optimal_tree_from_head_acc
    import os

    # Try to use head_acc.json from checkpoint for optimal tree
    if checkpoint_path:
        if checkpoint_path.endswith('.pt'):
            base_checkpoint = os.path.dirname(checkpoint_path)
        else:
            base_checkpoint = checkpoint_path

        head_acc_path = os.path.join(base_checkpoint, "head_acc.json")

        if os.path.exists(head_acc_path):
            tree_choices = generate_optimal_tree_from_head_acc(
                head_acc_path,
                num_heads,
                tree_size=budget,
                topk=64,  # Search up to rank 64
            )
            if tree_choices is not None:
                return tree_choices

    # Fallback to heuristic if no head_acc.json available
    return _get_heuristic_tree_choices(num_heads, budget, max_depth)


def _get_heuristic_tree_choices(
    num_heads: int,
    budget: int,
    max_depth: int,
) -> list:
    """
    Fallback heuristic tree generation when head_acc.json is not available.
    """
    import heapq

    # Default heuristic: accuracy decreases with depth
    head_accuracies = [0.6, 0.4, 0.3, 0.25][:num_heads]

    pq = []
    topk_per_depth = [
        min(budget, 20),
        min(budget // 2, 15),
        min(budget // 4, 10),
        min(budget // 8, 8),
    ]

    for i in range(topk_per_depth[0]):
        rank_discount = 1.0 / (1 + i * 0.1)
        score = head_accuracies[0] * rank_discount
        heapq.heappush(pq, (-score, 1, (i,)))

    choices = []
    visited = set()

    while pq and len(choices) < budget - 1:
        neg_score, depth, choice = heapq.heappop(pq)

        if choice in visited:
            continue
        visited.add(choice)
        choices.append(choice)

        if depth < min(max_depth, num_heads):
            parent_score = -neg_score
            next_topk = topk_per_depth[depth] if depth < len(topk_per_depth) else 5

            for j in range(next_topk):
                child_choice = choice + (j,)
                if child_choice not in visited:
                    rank_discount = 1.0 / (1 + j * 0.1)
                    if depth < len(head_accuracies):
                        child_score = parent_score * head_accuracies[depth] * rank_discount
                    else:
                        child_score = parent_score * 0.2 * rank_discount

                    if child_score > 0.001:
                        heapq.heappush(pq, (-child_score, depth + 1, child_choice))

    return choices


def build_enlarged_tree_choices(
    current_choices: list,
    num_heads: int,
    enlarged_budget: int,
    max_depth: int,
) -> list:
    """
    Build enlarged tree by extending current choices.

    Returns choices that would be in the enlarged tree but NOT in the current tree.
    """
    current_set = set(current_choices)

    # Generate all possible candidates up to enlarged_budget
    all_choices = get_tree_choices_for_budget(
        num_heads, enlarged_budget, max_depth
    )

    # Return only the new choices
    new_choices = [c for c in all_choices if c not in current_set]
    return new_choices


def compute_acceptance_for_tree(
    model,
    input_ids: torch.Tensor,
    tree_choices: list,
    topk: int = 10,
    verbose: bool = False,
) -> tuple[int, int, dict]:
    """
    Compute acceptance statistics for a given tree.

    Returns:
        accepted: Number of tokens accepted
        proposed: Number of tokens proposed (tree_size - 1)
        details: Dict with per-depth acceptance info
    """
    from nanochat.gemma_medusa.model import generate_tree_buffers

    device = model.get_device()

    # Generate buffers for tree
    buffers = generate_tree_buffers(tree_choices, device, topk)

    # Get initial hidden states and logits
    with torch.inference_mode():
        hidden = model._get_hidden_states(input_ids)
        main_logits, medusa_logits = model._compute_logits(
            hidden, return_medusa=True, last_only=True
        )

    # Generate candidates using tree structure
    last_main = main_logits[:, 0, :]
    last_medusa = medusa_logits[:, :, 0, :]

    candidates, tree_candidates = model._generate_candidates(
        last_main, last_medusa, buffers, topk, temperature=0.0
    )

    # Forward pass through tree
    with torch.inference_mode():
        tree_logits, ret_indices, valid_mask = model.forward_mtp(
            input_ids, tree_candidates, buffers
        )

    # Get predictions
    predictions = tree_logits.argmax(dim=-1)

    # Compute acceptance for each candidate path
    max_accept = 0
    best_path = None

    for cand_idx in range(candidates.shape[0]):
        cand_tokens = candidates[cand_idx]
        ret_idx = ret_indices[cand_idx]
        mask = valid_mask[cand_idx]

        # Check acceptance
        safe_indices = ret_idx.clamp(min=0)
        cand_preds = predictions[safe_indices]

        # Match: cand_tokens[1:] should equal cand_preds[:-1] where mask is valid
        matches = (cand_tokens[1:] == cand_preds[:-1]) & mask[1:]
        cumulative = torch.cumprod(matches.int(), dim=0)
        accept_len = cumulative.sum().item()

        if accept_len > max_accept:
            max_accept = accept_len
            best_path = cand_tokens[:accept_len + 1].tolist()

    proposed = len(tree_choices)  # Number of non-root nodes
    accepted = max_accept

    details = {
        "tree_size": len(tree_choices) + 1,
        "best_accept_length": max_accept,
        "best_path": best_path,
    }

    return accepted, proposed, details


def compute_theoretical_acceptance(
    predictions: torch.Tensor,
    enlarged_choices: list,
    all_candidates: torch.Tensor,
    topk: int = 10,
) -> tuple[int, int]:
    """
    Compute theoretical acceptance if enlarged tree had been used.

    This doesn't re-run the model - it uses the already-computed predictions
    to check if additional candidates would have been accepted.

    Args:
        predictions: [tree_len] argmax predictions from model
        enlarged_choices: Additional tree choices beyond current
        all_candidates: Full candidate tensor from enlarged tree
        topk: Top-k per head

    Returns:
        additional_accepted: Extra tokens that would have been accepted
        additional_proposed: Extra proposals in enlarged tree
    """
    # For now, return the new proposals count
    # The actual acceptance check would need the full tree buffers
    return 0, len(enlarged_choices)


def evaluate_tree_acceptance(args) -> TreeAcceptanceMetrics:
    """
    Main evaluation function.

    Runs generate_mtp twice with different tree sizes to get accurate acceptance rates.
    The model uses KV cache and proper tree attention masks for correct verification.
    """
    from transformers import AutoTokenizer
    from nanochat.gemma_medusa.model import (
        generate_tree_buffers,
        get_tree_choices,
        DEFAULT_TREES,
    )

    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.device, args.model_name)
    num_heads = config.get('medusa_num_heads', 4)

    tokenizer = AutoTokenizer.from_pretrained(
        config.get('model_name') or config.get('base_model', 'google/gemma-3-270m-it')
    )

    # Get prompts
    if args.prompts:
        prompts = args.prompts
    else:
        prompts = [
            "The capital of France is",
            "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    return",
            "In machine learning, gradient descent is an optimization algorithm that",
            "The year was 1984, and Winston Smith",
            "To implement a binary search tree in Python, first",
            "The mitochondria is the powerhouse of",
            "import torch\nimport torch.nn as nn\n\nclass Transformer(nn.Module):\n    def __init__(self",
            "Once upon a time in a land far away, there lived a",
        ]

    # Generate tree choices for current and enlarged budgets
    print(f"\nGenerating tree choices...")
    print(f"  Current budget: {args.current_budget}")
    print(f"  Enlarged budget: {args.enlarged_budget}")

    # Try to use calibrated tree for current
    try:
        current_choices = get_tree_choices(
            num_heads, args.checkpoint, tree_size=args.current_budget
        )
    except (FileNotFoundError, ValueError):
        # Fall back to default tree or budget-based generation
        if args.current_budget <= len(DEFAULT_TREES.get(num_heads, [])) + 1:
            current_choices = DEFAULT_TREES.get(num_heads, DEFAULT_TREES[4])
        else:
            current_choices = get_tree_choices_for_budget(
                num_heads, args.current_budget, args.max_depth,
                checkpoint_path=args.checkpoint
            )

    enlarged_choices = get_tree_choices_for_budget(
        num_heads, args.enlarged_budget, args.max_depth,
        checkpoint_path=args.checkpoint
    )

    print(f"  Current tree: {len(current_choices)} nodes (+ root)")
    print(f"  Enlarged tree: {len(enlarged_choices)} nodes (+ root)")

    # Create HST scorer if checkpoint provided
    hst_scorer = None
    hst_enabled = args.hst_checkpoint is not None
    if hst_enabled:
        print(f"\nLoading HST scorer from {args.hst_checkpoint}...")
        vocab_size = model.config.vocab_size
        hst_scorer = create_hst_scorer(
            vocab_size=vocab_size,
            device=args.device,
            checkpoint_path=args.hst_checkpoint,
            alpha=args.hst_alpha,
            beta=args.hst_beta,
            gamma=args.hst_gamma,
            no_pruning=args.hst_no_pruning,
            blend_mode=args.hst_blend_mode,
        )
        pruning_status = "disabled" if args.hst_no_pruning else "enabled (threshold=0.01)"
        print(f"  HST weights: α={args.hst_alpha}, β={args.hst_beta}, γ={args.hst_gamma}")
        print(f"  Pruning: {pruning_status}")
        print(f"  Blend mode: {args.hst_blend_mode}")

    # Tracking variables
    current_total_accepted = 0
    current_forward_passes = 0
    enlarged_total_accepted = 0
    enlarged_forward_passes = 0
    hst_total_accepted = 0
    hst_forward_passes = 0
    total_tokens_generated = 0

    topk = 10

    print(f"\nEvaluating {len(prompts)} prompts...")

    for prompt_idx, prompt in enumerate(prompts):
        if args.verbose:
            print(f"\n[{prompt_idx + 1}/{len(prompts)}] {prompt[:50]}...")

        input_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # Run with CURRENT tree using generate_mtp_with_cache
        with torch.inference_mode():
            _, current_stats = model.generate_mtp_with_cache(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=0.0,
                topk=topk,
                tree_choices=current_choices,
            )
        current_total_accepted += current_stats.total_accepted
        current_forward_passes += current_stats.forward_passes

        # Run with ENLARGED tree (same prompt for fair comparison)
        with torch.inference_mode():
            _, enlarged_stats = model.generate_mtp_with_cache(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=0.0,
                topk=topk,
                tree_choices=enlarged_choices,
            )
        enlarged_total_accepted += enlarged_stats.total_accepted
        enlarged_forward_passes += enlarged_stats.forward_passes
        total_tokens_generated += enlarged_stats.tokens_generated

        # Run with HST scoring if enabled (uses current tree + HST scorer)
        hst_stats = None
        if hst_scorer is not None:
            hst_scorer.reset()
            with torch.inference_mode():
                _, hst_stats = model.generate_mtp_with_cache(
                    input_ids,
                    max_new_tokens=args.max_tokens,
                    temperature=0.0,
                    topk=topk,
                    tree_choices=current_choices,
                    candidate_scorer=hst_scorer,
                )
            hst_total_accepted += hst_stats.total_accepted
            hst_forward_passes += hst_stats.forward_passes

        if args.verbose:
            print(f"    Current tree:  mean_accepted={current_stats.mean_accepted_length:.2f}")
            print(f"    Enlarged tree: mean_accepted={enlarged_stats.mean_accepted_length:.2f}")
            if hst_stats:
                print(f"    HST-scored:    mean_accepted={hst_stats.mean_accepted_length:.2f}")
        else:
            msg = f"  Prompt {prompt_idx + 1}: current={current_stats.mean_accepted_length:.2f}, enlarged={enlarged_stats.mean_accepted_length:.2f}"
            if hst_stats:
                msg += f", hst={hst_stats.mean_accepted_length:.2f}"
            print(msg)

    # Compute final metrics
    current_mean_acceptance = current_total_accepted / max(current_forward_passes, 1)
    current_acceptance_rate = current_total_accepted / max(current_forward_passes * len(current_choices), 1)

    enlarged_mean_acceptance = enlarged_total_accepted / max(enlarged_forward_passes, 1)
    enlarged_acceptance_rate = enlarged_total_accepted / max(enlarged_forward_passes * len(enlarged_choices), 1)

    improvement = enlarged_mean_acceptance - current_mean_acceptance
    improvement_pct = (improvement / max(current_mean_acceptance, 0.001)) * 100

    # HST metrics
    hst_mean_acceptance = 0.0
    hst_acceptance_rate = 0.0
    hst_improvement = 0.0
    hst_improvement_pct = 0.0
    if hst_enabled:
        hst_mean_acceptance = hst_total_accepted / max(hst_forward_passes, 1)
        hst_acceptance_rate = hst_total_accepted / max(hst_forward_passes * len(current_choices), 1)
        hst_improvement = hst_mean_acceptance - current_mean_acceptance
        hst_improvement_pct = (hst_improvement / max(current_mean_acceptance, 0.001)) * 100

    metrics = TreeAcceptanceMetrics(
        current_budget=args.current_budget,
        current_mean_acceptance=current_mean_acceptance,
        current_acceptance_rate=current_acceptance_rate,
        current_total_accepted=current_total_accepted,
        current_total_proposed=current_forward_passes * len(current_choices),
        enlarged_budget=args.enlarged_budget,
        enlarged_mean_acceptance=enlarged_mean_acceptance,
        enlarged_acceptance_rate=enlarged_acceptance_rate,
        enlarged_total_accepted=enlarged_total_accepted,
        enlarged_total_proposed=enlarged_forward_passes * len(enlarged_choices),
        hst_enabled=hst_enabled,
        hst_mean_acceptance=hst_mean_acceptance,
        hst_acceptance_rate=hst_acceptance_rate,
        hst_total_accepted=hst_total_accepted,
        hst_total_proposed=hst_forward_passes * len(current_choices),
        acceptance_improvement=improvement,
        acceptance_improvement_pct=improvement_pct,
        hst_improvement=hst_improvement,
        hst_improvement_pct=hst_improvement_pct,
        num_prompts=len(prompts),
        tokens_generated=total_tokens_generated,
        forward_passes=current_forward_passes + enlarged_forward_passes + hst_forward_passes,
    )

    return metrics


def main():
    args = parse_args()

    print("=" * 60)
    print("Tree Acceptance Rate Evaluation")
    print("=" * 60)

    metrics = evaluate_tree_acceptance(args)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n--- Current Tree (budget={metrics.current_budget}) ---")
    print(f"  Mean accepted tokens/forward: {metrics.current_mean_acceptance:.3f}")
    print(f"  Acceptance rate:              {metrics.current_acceptance_rate:.2%}")
    print(f"  Total accepted/proposed:      {metrics.current_total_accepted}/{metrics.current_total_proposed}")

    print(f"\n--- Enlarged Tree (budget={metrics.enlarged_budget}) ---")
    print(f"  Mean accepted tokens/forward: {metrics.enlarged_mean_acceptance:.3f}")
    print(f"  Acceptance rate:              {metrics.enlarged_acceptance_rate:.2%}")
    print(f"  Total accepted/proposed:      {metrics.enlarged_total_accepted}/{metrics.enlarged_total_proposed}")

    if metrics.hst_enabled:
        print(f"\n--- HST-Scored Tree (budget={metrics.current_budget}) ---")
        print(f"  Mean accepted tokens/forward: {metrics.hst_mean_acceptance:.3f}")
        print(f"  Acceptance rate:              {metrics.hst_acceptance_rate:.2%}")
        print(f"  Total accepted/proposed:      {metrics.hst_total_accepted}/{metrics.hst_total_proposed}")

    print(f"\n--- Improvement vs Current ---")
    print(f"  Enlarged tree:      +{metrics.acceptance_improvement:.3f} tokens/forward ({metrics.acceptance_improvement_pct:+.1f}%)")
    if metrics.hst_enabled:
        print(f"  HST scoring:        +{metrics.hst_improvement:.3f} tokens/forward ({metrics.hst_improvement_pct:+.1f}%)")

    print(f"\n--- Evaluation Stats ---")
    print(f"  Prompts evaluated:  {metrics.num_prompts}")
    print(f"  Tokens generated:   {metrics.tokens_generated}")
    print(f"  Forward passes:     {metrics.forward_passes}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(asdict(metrics), f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
