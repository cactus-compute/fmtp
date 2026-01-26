#!/usr/bin/env python
"""
Fast MTP acceptance rate evaluation.

Measures mean_acceptance rate without task evaluation or speedup measurement.
For quick iteration on acceptance metrics.

Usage:
    uv run python -m scripts.mtp_acceptance_eval --checkpoint path/to/ckpt
    uv run python -m scripts.mtp_acceptance_eval --checkpoint path/to/ckpt --task gsm8k -n 200
"""

import argparse
import json
import os

import torch

from nanochat.gemma_medusa import GemmaTokenizerWrapper, load_gemma_medusa_model
from nanochat.gemma_medusa.model import get_tree_choices

from tasks.gsm8k import GSM8K
from tasks.arc import ARC
from tasks.mmlu import MMLU


def load_checkpoint_config(checkpoint_path: str) -> dict:
    """Load config.json from checkpoint directory if it exists."""
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def load_task(task_name: str):
    """Load a task by name."""
    task_name = task_name.lower()
    if task_name == 'gsm8k':
        return GSM8K(subset="main", split="test")
    elif task_name == 'arc-easy':
        return ARC(subset="ARC-Easy", split="test")
    elif task_name == 'arc-challenge':
        return ARC(subset="ARC-Challenge", split="test")
    elif task_name == 'mmlu':
        return MMLU(subset="all", split="test")
    else:
        raise ValueError(f"Unknown task: {task_name}. Supported: gsm8k, arc-easy, arc-challenge, mmlu")


def main():
    parser = argparse.ArgumentParser(description="Fast MTP acceptance rate evaluation")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Path to Medusa checkpoint")
    parser.add_argument("--task", type=str, default="gsm8k", help="Task for prompts (default: gsm8k)")
    parser.add_argument("--num-samples", "-n", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per generation")
    parser.add_argument("--tree-budget", type=int, default=64, help="Tree budget (default: 64)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-sample stats")
    args = parser.parse_args()

    # Load checkpoint config
    ckpt_config = load_checkpoint_config(args.checkpoint)

    def get_cfg(config_key, default):
        return ckpt_config.get(config_key, default)

    model_name = get_cfg('base_model', 'google/gemma-3-270m-it')
    medusa_num_heads = get_cfg('medusa_num_heads', 4)
    medusa_num_layers = get_cfg('medusa_num_layers', 2)
    lora_rank = get_cfg('lora_rank', 64)
    lora_alpha = get_cfg('lora_alpha', lora_rank)
    zero_init_mlp = get_cfg('zero_init_mtp_mlp', True)
    mixer_hidden = get_cfg('mlp_mixer_hidden', 16)
    attn_num_layers = get_cfg('attn_num_layers', 0)
    use_multi_layer = get_cfg('use_multi_layer', False)
    use_head_mixer = get_cfg('use_mlp_mixer', False) or attn_num_layers > 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Loading model: {model_name}")
    mixer_type = "attention" if attn_num_layers > 0 else "mlp"

    model = load_gemma_medusa_model(
        model_name=model_name,
        medusa_num_heads=medusa_num_heads,
        medusa_num_layers=medusa_num_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=device,
        dtype=dtype,
        zero_init_mlp=zero_init_mlp,
        use_head_mixer=use_head_mixer,
        mixer_hidden=mixer_hidden,
        mixer_type=mixer_type,
        attn_num_layers=attn_num_layers,
        use_multi_layer=use_multi_layer,
    )

    # Load checkpoint weights
    checkpoint_path = os.path.join(args.checkpoint, "final", "medusa_heads.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(args.checkpoint, "medusa_heads.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_medusa_state_dict(checkpoint, strict=True)
    model._checkpoint_path = args.checkpoint
    model.eval()

    tokenizer = GemmaTokenizerWrapper(model_name)
    eos_token_id = tokenizer.hf_tokenizer.eos_token_id

    # Load task
    task = load_task(args.task)
    num_samples = min(args.num_samples, len(task))
    print(f"Task: {args.task} ({num_samples} samples)")

    # Get tree choices
    try:
        tree_choices = get_tree_choices(medusa_num_heads, args.checkpoint)
        print(f"Tree: optimal ({len(tree_choices)} nodes)")
    except FileNotFoundError:
        from nanochat.gemma_medusa.model import DEFAULT_TREES
        tree_choices = DEFAULT_TREES.get(medusa_num_heads, DEFAULT_TREES[4])
        print(f"Tree: heuristic ({len(tree_choices)} nodes)")

    # Evaluate
    total_accepted = 0
    total_forward_passes = 0
    total_tokens = 0

    print(f"\nEvaluating...")
    for i in range(num_samples):
        conversation = task[i]
        input_ids = tokenizer.render_for_completion(conversation)

        with torch.inference_mode():
            _, stats = model.generate_mtp_with_cache(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=0.0,
                eos_token_id=eos_token_id,
                tree_choices=tree_choices,
            )

        total_accepted += stats.total_accepted
        total_forward_passes += stats.forward_passes
        total_tokens += stats.tokens_generated

        mean_acc = total_accepted / total_forward_passes if total_forward_passes > 0 else 0
        if args.verbose:
            print(f"[{i+1}/{num_samples}] this={stats.mean_accepted_length:.2f}, running={mean_acc:.3f}")
        else:
            print(f"\r[{i+1}/{num_samples}] mean_acceptance={mean_acc:.3f}", end="", flush=True)

    print()
    mean_acceptance = total_accepted / total_forward_passes if total_forward_passes > 0 else 0
    print(f"\nResults:")
    print(f"  Mean acceptance: {mean_acceptance:.3f}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Forward passes: {total_forward_passes:,}")


if __name__ == "__main__":
    main()
