#!/usr/bin/env python
"""
MLX-based unified benchmarking/evaluation script for Gemma models.

Evaluates Gemma models using Apple's MLX framework on Apple Silicon.
Supports the same benchmarks as unified_eval.py:
- GSM8K (generative): Grade school math word problems
- HumanEval (generative): Python code completion

Designed to run on Apple Silicon (M1/M2/M3/M4) using Metal GPU acceleration.

Usage:
    # Basic evaluation with bf16
    conda activate fmtp-mlx
    python -m scripts.mlx_unified_eval --model mlx-community/gemma-3-270m-it-bf16 --task gsm8k -n 25

    # HumanEval benchmark
    python -m scripts.mlx_unified_eval --model mlx-community/gemma-3-270m-it-bf16 --task humaneval -n 25

    # All generative benchmarks
    python -m scripts.mlx_unified_eval --model mlx-community/gemma-3-1b-it-bf16 --task all
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

# Add project root to path for task imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval


@dataclass
class EvalResult:
    """Results from a single benchmark evaluation."""
    task: str
    num_samples: int
    num_correct: int
    accuracy: float
    tokens_per_second: Optional[float] = None
    total_tokens: Optional[int] = None
    total_time_seconds: Optional[float] = None
    model_name: Optional[str] = None


def load_task(task_name: str):
    """Load a task by name."""
    task_name = task_name.lower()
    if task_name == 'gsm8k':
        return GSM8K(subset="main", split="test")
    elif task_name == 'humaneval':
        return HumanEval()
    else:
        raise ValueError(f"Unknown task: {task_name}. Supported: gsm8k, humaneval")


def render_prompt_for_completion(conversation: Dict, tokenizer) -> str:
    """
    Render a conversation into a prompt string for completion.
    Adapts the nanochat conversation format to Gemma's chat template.
    """
    messages = conversation["messages"]

    # Build prompt using Gemma 3 chat format
    prompt_parts = ["<bos>"]

    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            prompt_parts.append(f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n")
        elif msg["role"] == "assistant":
            # For the last assistant message, we want the model to complete it
            if i == len(messages) - 1:
                # Don't include last assistant message - model will generate it
                pass
            else:
                content = msg["content"]
                if isinstance(content, str):
                    prompt_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>\n")
                elif isinstance(content, list):
                    # Handle structured content (text, python, etc.)
                    text_content = ""
                    for part in content:
                        if part["type"] == "text":
                            text_content += part["text"]
                        elif part["type"] == "python":
                            text_content += f"```python\n{part['text']}\n```"
                        elif part["type"] == "python_output":
                            text_content += f"\n**Output:**\n```\n{part['text']}\n```\n"
                    prompt_parts.append(f"<start_of_turn>model\n{text_content}<end_of_turn>\n")

    # Prime for assistant response
    prompt_parts.append("<start_of_turn>model\n")

    return "".join(prompt_parts)


def generate_mlx(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[str, int, float]:
    """
    Generate text using MLX.

    Returns:
        tuple: (generated_text, num_tokens, generation_time_seconds)
    """
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)

    # Create sampler with specified temperature
    sampler = make_sampler(temp=temperature)

    # Time only generation
    t0 = time.perf_counter()

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_new_tokens,
        sampler=sampler,
        verbose=False,
    )

    t1 = time.perf_counter()
    gen_time = t1 - t0

    # Count generated tokens
    full_tokens = tokenizer.encode(prompt + response)
    gen_tokens = len(full_tokens) - prompt_len

    return response, gen_tokens, gen_time


def run_generative_eval(
    task_object,
    model,
    tokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    max_problems: Optional[int] = None,
    model_name: str = "",
) -> EvalResult:
    """Run generative evaluation using MLX."""
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    total_tokens = 0
    total_gen_time = 0.0
    num_passed, total = 0, 0

    for i in range(num_problems):
        conversation = task_object[i]
        prompt = render_prompt_for_completion(conversation, tokenizer)

        # Generate
        completion, gen_tokens, gen_time = generate_mlx(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        total_tokens += gen_tokens
        total_gen_time += gen_time

        # Evaluate
        outcome = task_object.evaluate(conversation, completion)
        num_passed += int(outcome)
        total += 1

        tps = total_tokens / total_gen_time if total_gen_time > 0 else 0
        print(f"\r[{total}/{num_problems}] acc: {100*num_passed/total:.2f}%, tok/s: {tps:.1f}", end="", flush=True)

    print()

    return EvalResult(
        task=task_object.__class__.__name__,
        num_samples=total,
        num_correct=num_passed,
        accuracy=num_passed / total,
        tokens_per_second=total_tokens / total_gen_time if total_gen_time > 0 else None,
        total_tokens=total_tokens,
        total_time_seconds=total_gen_time,
        model_name=model_name,
    )


def main():
    parser = argparse.ArgumentParser(
        description="MLX-based benchmarking for Gemma models on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", "-m", type=str, required=True,
                       help="MLX model name (e.g., mlx-community/gemma-3-270m-it-bf16)")
    parser.add_argument("--task", "-t", type=str, default="all",
                       help="Task to evaluate: gsm8k, humaneval, or all")
    parser.add_argument("--num-samples", "-n", type=int, default=None,
                       help="Number of samples to evaluate (default: full dataset)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (0=greedy)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output JSON file for results")
    args = parser.parse_args()

    # Check MLX availability
    print(f"MLX version: {mx.__version__}")
    print(f"Metal available: {mx.metal.is_available()}")
    print(f"Model: {args.model}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load(args.model)
    print("Model loaded successfully!")

    # Determine tasks to run (only generative tasks supported in MLX version)
    all_tasks = ["gsm8k", "humaneval"]
    if args.task.lower() == "all":
        tasks = all_tasks
    else:
        tasks = [args.task.lower()]

    # Run evaluations
    results = []
    for task_name in tasks:
        print(f"\n{'='*60}")
        print(f"Evaluating {task_name.upper()}")
        print(f"{'='*60}")

        task_object = load_task(task_name)

        result = run_generative_eval(
            task_object,
            model, tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_problems=args.num_samples,
            model_name=args.model,
        )

        results.append(result)

        print(f"\nResults for {task_name.upper()}:")
        print(f"  Accuracy: {100*result.accuracy:.2f}% ({result.num_correct}/{result.num_samples})")
        if result.tokens_per_second is not None:
            print(f"  Tokens/second: {result.tokens_per_second:.1f}")
        if result.total_time_seconds is not None:
            print(f"  Total time: {result.total_time_seconds:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for result in results:
        tps_str = f", {result.tokens_per_second:.1f} tok/s" if result.tokens_per_second else ""
        print(f"  {result.task}: {100*result.accuracy:.2f}%{tps_str}")

    # Calculate aggregate metrics
    if len(results) > 1:
        avg_accuracy = sum(r.accuracy for r in results) / len(results)
        print(f"\n  Average accuracy: {100*avg_accuracy:.2f}%")

        tps_results = [r for r in results if r.tokens_per_second is not None]
        if tps_results:
            avg_tps = sum(r.tokens_per_second for r in tps_results) / len(tps_results)
            print(f"  Average tokens/second: {avg_tps:.1f}")

    # Save results
    if args.output:
        output_data = {
            "model": args.model,
            "mlx_version": mx.__version__,
            "metal_available": mx.metal.is_available(),
            "results": [asdict(r) for r in results],
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
