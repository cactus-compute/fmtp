#!/usr/bin/env python
"""
EAGLE acceptance rate evaluation on key benchmarks.

Measures mean acceptance rate and optional task accuracy for EAGLE speculative decoding.

Usage:
    # Single task
    uv run python -m scripts.eagle_acceptance_eval --checkpoint path/to/ckpt --task gsm8k -n 200

    # Run all benchmarks in parallel (use CUDA_VISIBLE_DEVICES to assign GPUs)
    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.eagle_acceptance_eval --checkpoint path/to/ckpt --task mmlu -n 500 &
    CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.eagle_acceptance_eval --checkpoint path/to/ckpt --task gsm8k -n 500 &
    CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.eagle_acceptance_eval --checkpoint path/to/ckpt --task arc -n 500 &
    CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.eagle_acceptance_eval --checkpoint path/to/ckpt --task humaneval -n 164 &
"""

import argparse
import json
import os
import time
from typing import Optional, List

import torch

from nanochat.gemma_eagle import GemmaEagleConfig, GemmaEagleModel, EagleGenerator
from nanochat.gemma_eagle.inference import EagleGenerationStats
from nanochat.gemma_medusa import GemmaTokenizerWrapper

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
    elif task_name == 'arc':
        # Combined ARC (challenge)
        return ARC(subset="ARC-Challenge", split="test")
    elif task_name == 'mmlu':
        return MMLU(subset="all", split="test")
    elif task_name == 'humaneval':
        try:
            from tasks.humaneval import HumanEval
            return HumanEval()
        except ImportError:
            print("HumanEval task not available, using GSM8K instead")
            return GSM8K(subset="main", split="test")
    else:
        raise ValueError(f"Unknown task: {task_name}. Supported: gsm8k, arc-easy, arc-challenge, arc, mmlu, humaneval")


def generate_with_eagle(
    generator: EagleGenerator,
    tokenizer: GemmaTokenizerWrapper,
    input_ids: List[int],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    collect_timing: bool = False,
) -> tuple:
    """
    Generate tokens using EAGLE speculative decoding.

    Returns:
        output_ids: Generated token IDs
        stats: EagleGenerationStats
    """
    device = generator.device
    input_tensor = torch.tensor([input_ids], device=device)
    eos_token_id = tokenizer.hf_tokenizer.eos_token_id

    # Use the EagleGenerator for proper tree-based speculative decoding
    output_ids, stats = generator.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_token_id=eos_token_id,
        return_stats=True,
        collect_timing=collect_timing,
    )

    return output_ids[0].tolist(), stats


def main():
    parser = argparse.ArgumentParser(description="EAGLE acceptance rate evaluation")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Path to EAGLE checkpoint")
    parser.add_argument("--base-model", type=str, default=None, help="Base model name (auto-detected from config)")
    parser.add_argument("--task", type=str, default="gsm8k", help="Task for prompts: gsm8k, arc, mmlu, humaneval")
    parser.add_argument("--num-samples", "-n", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-sample stats")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file for results")
    parser.add_argument("--timing", dest="timing", action="store_true", help="Collect per-stage timing metrics")
    parser.add_argument("--no-timing", dest="timing", action="store_false", help="Disable timing metrics")
    parser.set_defaults(timing=True)
    parser.add_argument("--draft-top-k", type=int, default=None, help="Override draft top-k")
    parser.add_argument("--draft-main-k", type=int, default=None, help="Override draft main-k (expanded beams)")
    parser.add_argument("--draft-depth", type=int, default=None, help="Override draft depth")
    parser.add_argument("--total-tokens", type=int, default=None, help="Override total draft tokens")
    args = parser.parse_args()

    # Load checkpoint config
    ckpt_config = load_checkpoint_config(args.checkpoint)

    # Get model config
    base_model = args.base_model or ckpt_config.get('base_model', 'google/gemma-3-270m-it')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Loading EAGLE model: {base_model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    # Create config and model
    config = GemmaEagleConfig(
        base_model_name=base_model,
        freeze_base=True,
    )
    if args.draft_top_k is not None:
        config.draft_top_k = args.draft_top_k
    if args.draft_main_k is not None:
        config.draft_main_k = args.draft_main_k
    if args.draft_depth is not None:
        config.draft_depth = args.draft_depth
    if args.total_tokens is not None:
        config.total_tokens = args.total_tokens

    model = GemmaEagleModel(
        config,
        device=device,
        dtype=dtype,
    )

    # Load checkpoint weights
    checkpoint_file = os.path.join(args.checkpoint, "final", "eagle_draft.pt")
    if not os.path.exists(checkpoint_file):
        checkpoint_file = os.path.join(args.checkpoint, "eagle_draft.pt")
    if not os.path.exists(checkpoint_file):
        # Try step checkpoints
        step_dirs = sorted([d for d in os.listdir(args.checkpoint) if d.startswith("step_")])
        if step_dirs:
            checkpoint_file = os.path.join(args.checkpoint, step_dirs[-1], "eagle_draft.pt")

    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_draft_state_dict(checkpoint)
    else:
        print(f"WARNING: No checkpoint found, using untrained draft model")

    model.eval()

    tokenizer = GemmaTokenizerWrapper(base_model)

    # Create EAGLE generator
    generator = EagleGenerator(model)

    # Load task
    task = load_task(args.task)
    num_samples = min(args.num_samples, len(task))
    print(f"Task: {args.task} ({num_samples} samples)")

    # Evaluate
    total_accepted = 0
    total_forward_passes = 0
    total_tokens = 0
    total_time = 0.0
    timing_totals = None

    print(f"\nEvaluating...")
    for i in range(num_samples):
        conversation = task[i]
        input_ids = tokenizer.render_for_completion(conversation)

        _, stats = generate_with_eagle(
            generator=generator,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            collect_timing=args.timing,
        )

        total_accepted += stats.total_accepted
        total_forward_passes += stats.forward_passes
        total_tokens += stats.tokens_generated
        total_time += stats.time_elapsed
        if stats.timing is not None:
            if timing_totals is None:
                timing_totals = {key: 0.0 for key in stats.timing}
            for key, value in stats.timing.items():
                timing_totals[key] = timing_totals.get(key, 0.0) + float(value)

        mean_acc = total_accepted / total_forward_passes if total_forward_passes > 0 else 0
        if args.verbose:
            print(f"[{i+1}/{num_samples}] tokens={stats.tokens_generated}, "
                  f"acc={stats.mean_accepted_length:.2f}, "
                  f"tok/s={stats.tokens_per_second:.1f}")
        else:
            tps = total_tokens / total_time if total_time > 0 else 0
            print(f"\r[{i+1}/{num_samples}] mean_acceptance={mean_acc:.3f}, tok/s={tps:.1f}", end="", flush=True)

    print()
    mean_acceptance = total_accepted / total_forward_passes if total_forward_passes > 0 else 0
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    overhead_pct = None
    overhead_ms = None
    verify_ms = None
    timing_breakdown = None
    prefill_s = None
    if timing_totals is not None:
        iter_count = timing_totals.get("iterations", 0.0)
        overhead_s = (
            timing_totals.get("draft_s", 0.0)
            + timing_totals.get("accept_s", 0.0)
            + timing_totals.get("kv_update_s", 0.0)
            + timing_totals.get("sample_s", 0.0)
        )
        verify_s = timing_totals.get("verify_s", 0.0)
        total_iter_s = overhead_s + verify_s
        overhead_pct = (100.0 * overhead_s / total_iter_s) if total_iter_s > 0 else 0.0
        if iter_count > 0:
            overhead_ms = 1000.0 * overhead_s / iter_count
            verify_ms = 1000.0 * verify_s / iter_count
            timing_breakdown = {
                "draft_ms": 1000.0 * timing_totals.get("draft_s", 0.0) / iter_count,
                "accept_ms": 1000.0 * timing_totals.get("accept_s", 0.0) / iter_count,
                "kv_update_ms": 1000.0 * timing_totals.get("kv_update_s", 0.0) / iter_count,
                "sample_ms": 1000.0 * timing_totals.get("sample_s", 0.0) / iter_count,
            }
        prefill_s = timing_totals.get("prefill_s", 0.0)

    results = {
        "task": args.task,
        "num_samples": num_samples,
        "checkpoint": args.checkpoint,
        "base_model": base_model,
        "draft_top_k": config.draft_top_k,
        "draft_main_k": config.draft_main_k,
        "draft_depth": config.draft_depth,
        "draft_total_tokens": config.total_tokens,
        "mean_acceptance": mean_acceptance,
        "total_tokens": total_tokens,
        "forward_passes": total_forward_passes,
        "tokens_per_second": tokens_per_second,
        "total_time_seconds": total_time,
        "eagle_overhead_percent": overhead_pct,
        "eagle_overhead_ms_per_iter": overhead_ms,
        "verify_ms_per_iter": verify_ms,
        "overhead_breakdown_ms_per_iter": timing_breakdown,
        "prefill_time_seconds": prefill_s,
    }

    print(f"\nResults for {args.task}:")
    print(f"  Mean acceptance: {mean_acceptance:.3f}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Forward passes: {total_forward_passes:,}")
    print(f"  Tokens/second: {tokens_per_second:.1f}")
    print(f"  Total time: {total_time:.1f}s")
    if timing_totals is not None and overhead_ms is not None and verify_ms is not None:
        print(f"  EAGLE overhead: {overhead_pct:.1f}% ({overhead_ms:.2f} ms/iter)")
        print(f"  Verify time: {verify_ms:.2f} ms/iter")
        if timing_breakdown is not None:
            print(
                "  Overhead breakdown (ms/iter): "
                f"draft={timing_breakdown['draft_ms']:.2f}, "
                f"accept={timing_breakdown['accept_ms']:.2f}, "
                f"kv_update={timing_breakdown['kv_update_ms']:.2f}, "
                f"sample={timing_breakdown['sample_ms']:.2f}"
            )
        if prefill_s is not None and prefill_s > 0:
            print(f"  Prefill time: {prefill_s:.2f}s total")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
