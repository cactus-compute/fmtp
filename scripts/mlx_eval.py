"""
Evaluate simple 2-token speculation on GSM8K and HumanEval.

Compares:
1. Baseline (standard autoregressive generation)
2. Simple speculation (predict 2 tokens, verify, accept 1 or 2)

Usage:
    conda activate fmtp-mlx
    python -m scripts.mlx_eval --task gsm8k --n 25 --max-tokens 128
    python -m scripts.mlx_eval --task humaneval --n 25 --max-tokens 128
"""

import argparse
import time
import json
import re
from typing import Optional

import mlx.core as mx

# Import task classes
from tasks.gsm8k import GSM8K, extract_answer as gsm8k_extract_answer
from tasks.humaneval import HumanEval, extract_program

# Import MLX model
from nanochat.mlx.model import GemmaMedusaModel


# Patterns to extract answer from various formats
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
FINAL_ANSWER_RE = re.compile(r"[Ff]inal [Aa]nswer[:\s]+(?:\$)?(-?[\d,]+(?:\.\d+)?)")
# "Therefore, ... is/are X" or "So, the answer is X" patterns
THEREFORE_RE = re.compile(r"[Tt]herefore[,\s]+.*?(?:is|are|=)\s+(-?[\d,]+(?:\.\d+)?)\s*(?:\.|$|[^\d])", re.MULTILINE)
ANSWER_IS_RE = re.compile(r"[Tt]he\s+(?:final\s+)?answer\s+is\s+(?:\$)?(-?[\d,]+(?:\.\d+)?)")


def extract_answer(completion: str) -> Optional[str]:
    """
    Extract numerical answer from model completion.

    Handles multiple formats:
    1. GSM8K format: #### <number>
    2. LaTeX boxed format: \\boxed{<number>}
    3. "Final Answer: <number>" format
    """
    # First try the standard GSM8K format
    answer = gsm8k_extract_answer(completion)
    if answer is not None:
        return answer

    # Try \boxed{...} format
    boxed_match = BOXED_RE.search(completion)
    if boxed_match:
        match_str = boxed_match.group(1).strip()
        # Remove commas from numbers
        match_str = match_str.replace(",", "")
        # Handle LaTeX formatting like \text{} or $
        match_str = re.sub(r"\\text\{([^}]*)\}", r"\1", match_str)
        match_str = match_str.replace("$", "").strip()
        return match_str

    # Try "Final Answer: X" format
    final_match = FINAL_ANSWER_RE.search(completion)
    if final_match:
        match_str = final_match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str

    # Try "The answer is X" format
    answer_is_match = ANSWER_IS_RE.search(completion)
    if answer_is_match:
        match_str = answer_is_match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str

    # Try "Therefore, ... is X" format (get the last occurrence)
    therefore_matches = list(THEREFORE_RE.finditer(completion))
    if therefore_matches:
        match_str = therefore_matches[-1].group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str

    return None


def format_gsm8k_prompt(question: str) -> str:
    """Format GSM8K question as chat prompt."""
    return f"""<start_of_turn>user
{question}

Solve this step by step and give your final answer after ####<end_of_turn>
<start_of_turn>model
"""


def format_humaneval_prompt(code_prompt: str) -> str:
    """Format HumanEval code prompt."""
    return f"""<start_of_turn>user
Complete this Python function:

{code_prompt}<end_of_turn>
<start_of_turn>model
```python
{code_prompt}"""


def run_gsm8k_eval(model: GemmaMedusaModel, n_samples: int, max_tokens: int, use_speculation: bool):
    """Run GSM8K evaluation."""
    task = GSM8K(subset="main", split="test")

    correct = 0
    total = 0
    total_tokens = 0
    total_time = 0.0
    total_accepted = 0
    total_proposed = 0

    n_samples = min(n_samples, task.num_examples())

    print(f"\nRunning GSM8K ({'speculation' if use_speculation else 'baseline'}): {n_samples} samples, max_tokens={max_tokens}")
    print("-" * 60)

    for i in range(n_samples):
        example = task.get_example(i)
        question = example['messages'][0]['content']

        # Format prompt
        prompt = format_gsm8k_prompt(question)

        # Generate
        if use_speculation:
            input_ids = model.tokenizer.encode(prompt)
            start = time.perf_counter()
            output_tokens, stats = model.generate_simple_speculation(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=model.tokenizer.eos_token_id,
            )
            elapsed = time.perf_counter() - start
            # Decode response (only new tokens)
            response = model.tokenizer.decode(output_tokens[len(input_ids):])
            n_tokens = len(output_tokens) - len(input_ids)
            total_accepted += stats.total_accepted
            total_proposed += stats.total_proposed
        else:
            start = time.perf_counter()
            response, n_tokens, _ = model.generate_standard(
                prompt=prompt,
                max_new_tokens=max_tokens,
            )
            elapsed = time.perf_counter() - start

        # Extract answer and evaluate
        pred_answer = extract_answer(response)
        # Get ground truth from example
        assistant_content = example['messages'][-1]['content']
        last_text = assistant_content[-1]['text']
        ref_answer = extract_answer(last_text)

        is_correct = pred_answer == ref_answer
        if is_correct:
            correct += 1
        total += 1

        total_tokens += n_tokens
        total_time += elapsed

        tok_s = n_tokens / elapsed if elapsed > 0 else 0
        print(f"  [{i+1}/{n_samples}] tokens={n_tokens}, time={elapsed:.2f}s, tok/s={tok_s:.1f}, "
              f"pred={pred_answer}, ref={ref_answer}, {'✓' if is_correct else '✗'}")

    accuracy = correct / total if total > 0 else 0
    avg_tok_s = total_tokens / total_time if total_time > 0 else 0

    print("-" * 60)
    print(f"GSM8K Results ({'speculation' if use_speculation else 'baseline'}):")
    print(f"  Accuracy: {correct}/{total} = {accuracy*100:.1f}%")
    print(f"  Avg tok/s: {avg_tok_s:.1f}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total time: {total_time:.2f}s")
    if use_speculation:
        accept_rate = total_accepted / total_proposed if total_proposed > 0 else 0
        avg_tok_iter = total_accepted / (total_proposed / 2) if total_proposed > 0 else 0
        print(f"  Accept rate: {total_accepted}/{total_proposed} = {accept_rate*100:.1f}%")
        print(f"  Avg tok/iter: {avg_tok_iter:.2f}")

    return {
        "task": "gsm8k",
        "mode": "speculation" if use_speculation else "baseline",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_tok_s": avg_tok_s,
        "total_tokens": total_tokens,
        "total_time": total_time,
    }


def run_humaneval_eval(model: GemmaMedusaModel, n_samples: int, max_tokens: int, use_speculation: bool):
    """Run HumanEval evaluation."""
    task = HumanEval()

    correct = 0
    total = 0
    total_tokens = 0
    total_time = 0.0
    total_accepted = 0
    total_proposed = 0

    n_samples = min(n_samples, task.num_examples())

    print(f"\nRunning HumanEval ({'speculation' if use_speculation else 'baseline'}): {n_samples} samples, max_tokens={max_tokens}")
    print("-" * 60)

    for i in range(n_samples):
        example = task.get_example(i)
        code_prompt = example['messages'][0]['content']

        # Format prompt
        prompt = format_humaneval_prompt(code_prompt)

        # Generate
        if use_speculation:
            input_ids = model.tokenizer.encode(prompt)
            start = time.perf_counter()
            output_tokens, stats = model.generate_simple_speculation(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=model.tokenizer.eos_token_id,
            )
            elapsed = time.perf_counter() - start
            # Decode response (only new tokens)
            response = model.tokenizer.decode(output_tokens[len(input_ids):])
            n_tokens = len(output_tokens) - len(input_ids)
            total_accepted += stats.total_accepted
            total_proposed += stats.total_proposed
        else:
            start = time.perf_counter()
            response, n_tokens, _ = model.generate_standard(
                prompt=prompt,
                max_new_tokens=max_tokens,
            )
            elapsed = time.perf_counter() - start

        # Try to evaluate (extract code and run tests)
        # The prompt includes code_prompt, so we need to prepend it to the response
        # to get the full function definition
        full_response = code_prompt + response
        try:
            is_correct = task.evaluate(example, full_response)
        except Exception as e:
            print(f"    Eval error: {e}")
            is_correct = False

        if is_correct:
            correct += 1
        total += 1

        total_tokens += n_tokens
        total_time += elapsed

        tok_s = n_tokens / elapsed if elapsed > 0 else 0
        print(f"  [{i+1}/{n_samples}] tokens={n_tokens}, time={elapsed:.2f}s, tok/s={tok_s:.1f}, "
              f"{'✓' if is_correct else '✗'}")

    accuracy = correct / total if total > 0 else 0
    avg_tok_s = total_tokens / total_time if total_time > 0 else 0

    print("-" * 60)
    print(f"HumanEval Results ({'speculation' if use_speculation else 'baseline'}):")
    print(f"  Accuracy: {correct}/{total} = {accuracy*100:.1f}%")
    print(f"  Avg tok/s: {avg_tok_s:.1f}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total time: {total_time:.2f}s")
    if use_speculation:
        accept_rate = total_accepted / total_proposed if total_proposed > 0 else 0
        avg_tok_iter = total_accepted / (total_proposed / 2) if total_proposed > 0 else 0
        print(f"  Accept rate: {total_accepted}/{total_proposed} = {accept_rate*100:.1f}%")
        print(f"  Avg tok/iter: {avg_tok_iter:.2f}")

    return {
        "task": "humaneval",
        "mode": "speculation" if use_speculation else "baseline",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_tok_s": avg_tok_s,
        "total_tokens": total_tokens,
        "total_time": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLX speculation on benchmarks")
    parser.add_argument("--task", type=str, required=True, choices=["gsm8k", "humaneval", "both"],
                        help="Task to evaluate")
    parser.add_argument("--n", type=int, default=25, help="Number of samples")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--checkpoint", type=str,
                        default="~/.cache/nanochat/gemma_medusa_270m_wildchat_100k_nolora",
                        help="Medusa checkpoint path")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline")
    parser.add_argument("--speculation-only", action="store_true", help="Only run speculation")
    args = parser.parse_args()

    # Expand checkpoint path
    import os
    checkpoint_path = os.path.expanduser(args.checkpoint)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = GemmaMedusaModel.from_checkpoint(
        checkpoint_path=checkpoint_path,
        mlx_model_name="mlx-community/gemma-3-270m-it-bf16",
    )
    print("Model loaded!")

    results = []

    run_baseline = not args.speculation_only
    run_speculation = not args.baseline_only

    if args.task in ["gsm8k", "both"]:
        if run_baseline:
            r = run_gsm8k_eval(model, args.n, args.max_tokens, use_speculation=False)
            results.append(r)
        if run_speculation:
            r = run_gsm8k_eval(model, args.n, args.max_tokens, use_speculation=True)
            results.append(r)

    if args.task in ["humaneval", "both"]:
        if run_baseline:
            r = run_humaneval_eval(model, args.n, args.max_tokens, use_speculation=False)
            results.append(r)
        if run_speculation:
            r = run_humaneval_eval(model, args.n, args.max_tokens, use_speculation=True)
            results.append(r)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"{r['task']} ({r['mode']}): accuracy={r['accuracy']*100:.1f}%, tok/s={r['avg_tok_s']:.1f}")

    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"mlx_eval_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
