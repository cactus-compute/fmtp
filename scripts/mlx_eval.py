"""
Evaluate speculative decoding on GSM8K and HumanEval.

Compares:
1. Baseline (standard autoregressive generation)
2. Simple speculation (predict 2 or 3 tokens, verify, accept 1-3)
3. Tree-based speculation (using optimal tree from head_acc.json)

Usage:
    conda activate fmtp-mlx
    python -m scripts.mlx_eval --task gsm8k --n 25 --max-tokens 128
    python -m scripts.mlx_eval --task humaneval --n 25 --max-tokens 128
    python -m scripts.mlx_eval --task gsm8k --n 25 --tree-size 4  # Tree-based

Precise Experiments:
1 head, 1 token speculation (tree size of 2):
    python -m scripts.mlx_eval --task gsm8k --n 25 --max-tokens 128 --skip-eval \
    && python -m scripts.mlx_eval --task humaneval --n 25 --max-tokens 128 --skip-eval

    gsm8k (baseline): accuracy=0.0%, tok/s=191.8
    gsm8k (spec-2tok): accuracy=0.0%, tok/s=215.0
    Mean accepted: 1.40 tokens/iter
    
    humaneval (baseline): accuracy=0.0%, tok/s=143.3
    humaneval (spec-2tok): accuracy=0.0%, tok/s=191.1
    Mean accepted: 1.38 tokens/iter

1 head, 2 token speculation (tree size of 3):
    python -m scripts.mlx_eval --task gsm8k --n 25 --max-tokens 128 --spec-tokens 3 --skip-eval \
    && python -m scripts.mlx_eval --task humaneval --n 25 --max-tokens 128 --spec-token 3 --skip-eval
    
    gsm8k (baseline): accuracy=0.0%, tok/s=192.9
    gsm8k (spec-3tok): accuracy=0.0%, tok/s=201.9
    Mean accepted: 1.42 tokens/iter

    humaneval (baseline): accuracy=0.0%, tok/s=143.3
    humaneval (spec-3tok): accuracy=0.0%, tok/s=187.5
    Mean accepted: 1.37 tokens/iter

2 head, 1 token speculation per head (tree size of 3):
    python -m scripts.mlx_eval --task gsm8k --n 25 --max-tokens 128 --depth2 --skip-eval \       
    && python -m scripts.mlx_eval --task humaneval --n 25 --max-tokens 128 --depth2 --skip-eval

    gsm8k (baseline): accuracy=0.0%, tok/s=199.1
    gsm8k (depth2): accuracy=0.0%, tok/s=208.1
    Mean accepted: 1.48 tokens/iter

    humaneval (baseline): accuracy=0.0%, tok/s=147.9
    humaneval (depth2): accuracy=0.0%, tok/s=186.9
    Mean accepted: 1.46 tokens/iter
"""

import argparse
import os
import time
import json
import re
from typing import Optional, List, Dict, Tuple

import mlx.core as mx

# Import task classes
from tasks.gsm8k import GSM8K, extract_answer as gsm8k_extract_answer
from tasks.humaneval import HumanEval, extract_program

# Import MLX model
from nanochat.mlx.model import GemmaMedusaModel


# =============================================================================
# Optimal tree building (from mlx_tree_bench.py)
# =============================================================================

def _get_node_expectation(
    accuracies: Dict[int, Dict[int, float]],
    node: Tuple[int, ...],
    topk: int,
) -> float:
    """Calculate expected acceptance for a tree node."""
    expectation = 1.0
    for depth, rank in enumerate(node):
        if depth not in accuracies:
            return 0.0
        head_acc = accuracies[depth]
        k_cur = rank + 1
        k_prev = rank if rank > 0 else None
        if k_cur in head_acc:
            if k_prev and k_prev in head_acc:
                prob = head_acc[k_cur] - head_acc[k_prev]
            else:
                prob = head_acc[k_cur]
        else:
            prob = 0.0
        expectation *= prob
    return expectation


def _explore_tree_greedy(
    accuracies: Dict[int, Dict[int, float]],
    max_depth: int,
    max_child: List[int],
    num_iterations: int,
    topk: int,
) -> List[Tuple[int, ...]]:
    """Greedy tree exploration algorithm from Medusa paper."""
    explored_nodes = {}
    accept_nodes = [tuple([0])]
    explored_nodes[tuple([0])] = _get_node_expectation(accuracies, (0,), topk)

    for _ in range(num_iterations):
        neighbors = []
        for node in accept_nodes:
            if node[-1] < max_child[len(node) - 1] - 1:
                neighbor = list(node)
                neighbor[-1] = neighbor[-1] + 1
                neighbors.append(tuple(neighbor))
            if len(node) < max_depth:
                neighbor = list(node)
                neighbor.append(0)
                neighbors.append(tuple(neighbor))

        best_neighbor = None
        best_expectation = 0

        for neighbor in neighbors:
            if neighbor in accept_nodes:
                continue
            if neighbor in explored_nodes:
                expectation = explored_nodes[neighbor]
            else:
                expectation = _get_node_expectation(accuracies, neighbor, topk)
                explored_nodes[neighbor] = expectation
            if expectation > best_expectation:
                best_neighbor = neighbor
                best_expectation = expectation

        if best_neighbor is None:
            break
        accept_nodes.append(best_neighbor)

    return accept_nodes


def generate_optimal_tree(
    head_acc_path: str,
    num_heads: int,
    tree_size: int,
    topk: int = 64,
) -> Optional[List[Tuple[int, ...]]]:
    """Generate optimal tree choices from head_acc.json."""
    if not os.path.exists(head_acc_path):
        return None

    try:
        with open(head_acc_path) as f:
            data = json.load(f)
        recall = data.get("recall", {})
        accuracies: Dict[int, Dict[int, float]] = {}
        for h in range(num_heads):
            head_key = f"head_{h}"
            if head_key in recall:
                accuracies[h] = {int(k): v for k, v in recall[head_key].items()}
            else:
                accuracies[h] = {}

        max_child = [topk] * num_heads
        tree_choices = _explore_tree_greedy(
            accuracies=accuracies,
            max_depth=num_heads,
            max_child=max_child,
            num_iterations=tree_size - 1,
            topk=topk,
        )
        return tree_choices
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def get_tree_max_depth(tree_choices: List[Tuple[int, ...]]) -> int:
    """Get the maximum depth of the tree (= number of heads needed)."""
    if not tree_choices:
        return 0
    return max(len(choice) for choice in tree_choices)


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


def run_gsm8k_eval(
    model: GemmaMedusaModel,
    n_samples: int,
    max_tokens: int,
    use_speculation: bool,
    spec_tokens: int = 2,
    tree_choices: Optional[List[Tuple[int, ...]]] = None,
    tree_size: Optional[int] = None,
    use_depth2: bool = False,
    skip_eval: bool = False,
    collect_entropy: bool = False,
    entropy_threshold: Optional[float] = None,
    profile: bool = False,
    use_fused_kernels: bool = True,
):
    """Run GSM8K evaluation."""
    task = GSM8K(subset="main", split="test")

    correct = 0
    total = 0
    total_tokens = 0
    total_time = 0.0
    total_forward_passes = 0
    total_skipped = 0  # Track total skipped speculations
    all_entropy_records = []  # Aggregate entropy data from all samples
    all_timing_data = {}  # Aggregate timing data across samples

    n_samples = min(n_samples, task.num_examples())

    # Determine mode string for output
    if use_depth2:
        mode_str = "depth2"
        depth = 2
    elif tree_choices is not None:
        mode_str = f"tree-{tree_size}"
        depth = get_tree_max_depth(tree_choices)
    elif use_speculation:
        mode_str = f"spec-{spec_tokens}tok"
        depth = spec_tokens - 1
    else:
        mode_str = "baseline"
        depth = 0

    print(f"\nRunning GSM8K ({mode_str}): {n_samples} samples, max_tokens={max_tokens}")
    if tree_choices:
        print(f"  Tree size: {len(tree_choices)}, depth: {depth}")
    print("-" * 60)

    for i in range(n_samples):
        example = task.get_example(i)
        question = example['messages'][0]['content']

        # Format prompt
        prompt = format_gsm8k_prompt(question)

        # Generate
        if use_depth2:
            # Optimized depth-2 speculation (2 heads, 3 tokens)
            input_ids = model.tokenizer.encode(prompt)
            start = time.perf_counter()
            output_tokens, stats = model.generate_depth2_speculation(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                stop_token_ids=model.tokenizer.eos_token_ids,
                collect_entropy=collect_entropy,
                entropy_threshold=entropy_threshold,
            )
            elapsed = time.perf_counter() - start
            response = model.tokenizer.decode(output_tokens[len(input_ids):])
            n_tokens = len(output_tokens) - len(input_ids)
            total_forward_passes += stats.forward_passes
            total_skipped += stats.speculation_skipped
            if collect_entropy and stats.entropy_log:
                all_entropy_records.extend(stats.entropy_log)
        elif tree_choices is not None:
            # Tree-based speculation using generate_mtp with tree attention
            input_ids = model.tokenizer.encode(prompt)
            start = time.perf_counter()
            output_tokens, stats = model.generate_mtp(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                stop_token_ids=model.tokenizer.eos_token_ids,
                tree_choices=tree_choices,
                num_active_heads=depth,
                use_tree_attention=True,
                profile=profile,
                use_fused_kernels=use_fused_kernels,
            )
            elapsed = time.perf_counter() - start
            response = model.tokenizer.decode(output_tokens[len(input_ids):])
            n_tokens = len(output_tokens) - len(input_ids)
            total_forward_passes += stats.forward_passes
            # Aggregate timing data if profiling
            if profile and stats.timing:
                for key, value in stats.timing.items():
                    if key not in all_timing_data:
                        all_timing_data[key] = value
                    elif key.endswith("_total_ms"):
                        all_timing_data[key] += value
                    elif key.endswith("_count"):
                        all_timing_data[key] += value
        elif use_speculation:
            input_ids = model.tokenizer.encode(prompt)
            start = time.perf_counter()
            if spec_tokens == 3:
                output_tokens, stats = model.generate_simple_speculation_3tok(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    stop_token_ids=model.tokenizer.eos_token_ids,
                )
            else:
                output_tokens, stats = model.generate_simple_speculation(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    stop_token_ids=model.tokenizer.eos_token_ids,
                    collect_entropy=collect_entropy,
                    entropy_threshold=entropy_threshold,
                )
            elapsed = time.perf_counter() - start
            # Decode response (only new tokens)
            response = model.tokenizer.decode(output_tokens[len(input_ids):])
            n_tokens = len(output_tokens) - len(input_ids)
            total_forward_passes += stats.forward_passes
            total_skipped += stats.speculation_skipped
            if collect_entropy and stats.entropy_log:
                all_entropy_records.extend(stats.entropy_log)
        else:
            start = time.perf_counter()
            response, n_tokens, _ = model.generate_standard(
                prompt=prompt,
                max_new_tokens=max_tokens,
            )
            elapsed = time.perf_counter() - start

        # Extract answer and evaluate
        if skip_eval:
            is_correct = False
            pred_answer = None
            ref_answer = None
        else:
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
        if skip_eval:
            print(f"  [{i+1}/{n_samples}] tokens={n_tokens}, time={elapsed:.2f}s, tok/s={tok_s:.1f}")
        else:
            print(f"  [{i+1}/{n_samples}] tokens={n_tokens}, time={elapsed:.2f}s, tok/s={tok_s:.1f}, "
                  f"pred={pred_answer}, ref={ref_answer}, {'✓' if is_correct else '✗'}")

    accuracy = correct / total if total > 0 else 0
    avg_tok_s = total_tokens / total_time if total_time > 0 else 0

    print("-" * 60)
    print(f"GSM8K Results ({mode_str}):")
    print(f"  Accuracy: {correct}/{total} = {accuracy*100:.1f}%")
    print(f"  Avg tok/s: {avg_tok_s:.1f}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total time: {total_time:.2f}s")
    if use_speculation or tree_choices or use_depth2:
        mean_accepted = total_tokens / total_forward_passes if total_forward_passes > 0 else 0
        print(f"  Mean accepted: {mean_accepted:.2f} tokens/iter")
        if entropy_threshold is not None:
            skip_rate = total_skipped / total_forward_passes if total_forward_passes > 0 else 0
            print(f"  Entropy gating: {total_skipped}/{total_forward_passes} skipped ({skip_rate*100:.1f}%)")

    # Print profiling data if collected
    if profile and all_timing_data:
        print("\n  Profiling Data (per-operation timing):")
        # Calculate mean times from totals and counts
        ops = set(k.replace("_total_ms", "").replace("_count", "").replace("_mean_ms", "") for k in all_timing_data.keys())
        for op in sorted(ops):
            total_key = f"{op}_total_ms"
            count_key = f"{op}_count"
            if total_key in all_timing_data and count_key in all_timing_data:
                total_ms = all_timing_data[total_key]
                count = all_timing_data[count_key]
                mean_ms = total_ms / count if count > 0 else 0
                print(f"    {op}: {mean_ms:.3f}ms mean, {total_ms:.1f}ms total ({int(count)} calls)")

    mean_accepted = total_tokens / total_forward_passes if total_forward_passes > 0 else 1.0

    return {
        "task": "gsm8k",
        "mode": mode_str,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_tok_s": avg_tok_s,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "mean_accepted": mean_accepted,
        "total_forward_passes": total_forward_passes,
        "tree_size": tree_size if tree_choices else None,
        "depth": depth,
        "entropy_threshold": entropy_threshold,
        "speculation_skipped": total_skipped if entropy_threshold else None,
        "entropy_records": [{"entropy": r.entropy, "accept_length": r.accept_length} for r in all_entropy_records] if collect_entropy else None,
        "timing": all_timing_data if profile else None,
    }


def run_humaneval_eval(
    model: GemmaMedusaModel,
    n_samples: int,
    max_tokens: int,
    use_speculation: bool,
    spec_tokens: int = 2,
    tree_choices: Optional[List[Tuple[int, ...]]] = None,
    tree_size: Optional[int] = None,
    use_depth2: bool = False,
    skip_eval: bool = False,
    collect_entropy: bool = False,
    entropy_threshold: Optional[float] = None,
    profile: bool = False,
    use_fused_kernels: bool = True,
):
    """Run HumanEval evaluation."""
    task = HumanEval()

    correct = 0
    total = 0
    total_tokens = 0
    total_time = 0.0
    total_forward_passes = 0
    total_skipped = 0  # Track total skipped speculations
    all_entropy_records = []  # Aggregate entropy data from all samples
    all_timing_data = {}  # Aggregate timing data across samples

    n_samples = min(n_samples, task.num_examples())

    # Determine mode string for output
    if use_depth2:
        mode_str = "depth2"
        depth = 2
    elif tree_choices is not None:
        mode_str = f"tree-{tree_size}"
        depth = get_tree_max_depth(tree_choices)
    elif use_speculation:
        mode_str = f"spec-{spec_tokens}tok"
        depth = spec_tokens - 1
    else:
        mode_str = "baseline"
        depth = 0

    print(f"\nRunning HumanEval ({mode_str}): {n_samples} samples, max_tokens={max_tokens}")
    if tree_choices:
        print(f"  Tree size: {len(tree_choices)}, depth: {depth}")
    print("-" * 60)

    for i in range(n_samples):
        example = task.get_example(i)
        code_prompt = example['messages'][0]['content']

        # Format prompt
        prompt = format_humaneval_prompt(code_prompt)

        # Generate
        if use_depth2:
            # Optimized depth-2 speculation (2 heads, 3 tokens)
            input_ids = model.tokenizer.encode(prompt)
            start = time.perf_counter()
            output_tokens, stats = model.generate_depth2_speculation(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                stop_token_ids=model.tokenizer.eos_token_ids,
                collect_entropy=collect_entropy,
                entropy_threshold=entropy_threshold,
            )
            elapsed = time.perf_counter() - start
            response = model.tokenizer.decode(output_tokens[len(input_ids):])
            n_tokens = len(output_tokens) - len(input_ids)
            total_forward_passes += stats.forward_passes
            total_skipped += stats.speculation_skipped
            if collect_entropy and stats.entropy_log:
                all_entropy_records.extend(stats.entropy_log)
        elif tree_choices is not None:
            # Tree-based speculation using generate_mtp with tree attention
            input_ids = model.tokenizer.encode(prompt)
            start = time.perf_counter()
            output_tokens, stats = model.generate_mtp(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                stop_token_ids=model.tokenizer.eos_token_ids,
                tree_choices=tree_choices,
                num_active_heads=depth,
                use_tree_attention=True,
                profile=profile,
                use_fused_kernels=use_fused_kernels,
            )
            elapsed = time.perf_counter() - start
            response = model.tokenizer.decode(output_tokens[len(input_ids):])
            n_tokens = len(output_tokens) - len(input_ids)
            total_forward_passes += stats.forward_passes
            # Aggregate timing data if profiling
            if profile and stats.timing:
                for key, value in stats.timing.items():
                    if key not in all_timing_data:
                        all_timing_data[key] = value
                    elif key.endswith("_total_ms"):
                        all_timing_data[key] += value
                    elif key.endswith("_count"):
                        all_timing_data[key] += value
        elif use_speculation:
            input_ids = model.tokenizer.encode(prompt)
            start = time.perf_counter()
            if spec_tokens == 3:
                output_tokens, stats = model.generate_simple_speculation_3tok(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    stop_token_ids=model.tokenizer.eos_token_ids,
                )
            else:
                output_tokens, stats = model.generate_simple_speculation(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    stop_token_ids=model.tokenizer.eos_token_ids,
                    collect_entropy=collect_entropy,
                    entropy_threshold=entropy_threshold,
                )
            elapsed = time.perf_counter() - start
            # Decode response (only new tokens)
            response = model.tokenizer.decode(output_tokens[len(input_ids):])
            n_tokens = len(output_tokens) - len(input_ids)
            total_forward_passes += stats.forward_passes
            total_skipped += stats.speculation_skipped
            if collect_entropy and stats.entropy_log:
                all_entropy_records.extend(stats.entropy_log)
        else:
            start = time.perf_counter()
            response, n_tokens, _ = model.generate_standard(
                prompt=prompt,
                max_new_tokens=max_tokens,
            )
            elapsed = time.perf_counter() - start

        # Try to evaluate (extract code and run tests)
        if skip_eval:
            is_correct = False
        else:
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
        if skip_eval:
            print(f"  [{i+1}/{n_samples}] tokens={n_tokens}, time={elapsed:.2f}s, tok/s={tok_s:.1f}")
        else:
            print(f"  [{i+1}/{n_samples}] tokens={n_tokens}, time={elapsed:.2f}s, tok/s={tok_s:.1f}, "
                  f"{'✓' if is_correct else '✗'}")

    accuracy = correct / total if total > 0 else 0
    avg_tok_s = total_tokens / total_time if total_time > 0 else 0

    print("-" * 60)
    print(f"HumanEval Results ({mode_str}):")
    print(f"  Accuracy: {correct}/{total} = {accuracy*100:.1f}%")
    print(f"  Avg tok/s: {avg_tok_s:.1f}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total time: {total_time:.2f}s")
    if use_speculation or tree_choices or use_depth2:
        mean_accepted = total_tokens / total_forward_passes if total_forward_passes > 0 else 0
        print(f"  Mean accepted: {mean_accepted:.2f} tokens/iter")
        if entropy_threshold is not None:
            skip_rate = total_skipped / total_forward_passes if total_forward_passes > 0 else 0
            print(f"  Entropy gating: {total_skipped}/{total_forward_passes} skipped ({skip_rate*100:.1f}%)")

    # Print profiling data if collected
    if profile and all_timing_data:
        print("\n  Profiling Data (per-operation timing):")
        # Calculate mean times from totals and counts
        ops = set(k.replace("_total_ms", "").replace("_count", "").replace("_mean_ms", "") for k in all_timing_data.keys())
        for op in sorted(ops):
            total_key = f"{op}_total_ms"
            count_key = f"{op}_count"
            if total_key in all_timing_data and count_key in all_timing_data:
                total_ms = all_timing_data[total_key]
                count = all_timing_data[count_key]
                mean_ms = total_ms / count if count > 0 else 0
                print(f"    {op}: {mean_ms:.3f}ms mean, {total_ms:.1f}ms total ({int(count)} calls)")

    mean_accepted = total_tokens / total_forward_passes if total_forward_passes > 0 else 1.0

    return {
        "task": "humaneval",
        "mode": mode_str,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_tok_s": avg_tok_s,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "mean_accepted": mean_accepted,
        "total_forward_passes": total_forward_passes,
        "tree_size": tree_size if tree_choices else None,
        "depth": depth,
        "entropy_threshold": entropy_threshold,
        "speculation_skipped": total_skipped if entropy_threshold else None,
        "entropy_records": [{"entropy": r.entropy, "accept_length": r.accept_length} for r in all_entropy_records] if collect_entropy else None,
        "timing": all_timing_data if profile else None,
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
    parser.add_argument("--spec-tokens", type=int, default=2, choices=[2, 3],
                        help="Number of tokens to speculate (2 or 3) - simple speculation")
    parser.add_argument("--depth2", action="store_true",
                        help="Use optimized depth-2 speculation (2 heads, 3 tokens)")
    parser.add_argument("--tree-size", type=int, default=None,
                        help="Speculation size: 1=baseline, 2=simple-2tok, 3=depth2, 4+=tree-mtp")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip correctness evaluation (for speed benchmarking only)")
    parser.add_argument("--collect-entropy-data", type=str, default=None,
                        help="Path to save entropy/acceptance data (JSON) for threshold analysis")
    parser.add_argument("--entropy-threshold", type=float, default=None,
                        help="Skip speculation when main model entropy exceeds this threshold")
    parser.add_argument("--profile", action="store_true",
                        help="Collect per-operation timing data for kernel optimization analysis")
    parser.add_argument("--use-fused-kernels", action="store_true", default=True,
                        help="Use fused Metal kernels for verification (default: True)")
    parser.add_argument("--no-fused-kernels", action="store_false", dest="use_fused_kernels",
                        help="Disable fused Metal kernels (use Python fallback)")
    parser.add_argument("--results-dir", type=str, default=".",
                        help="Directory to save results (default: current directory)")
    args = parser.parse_args()

    # Expand checkpoint path
    checkpoint_path = os.path.expanduser(args.checkpoint)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = GemmaMedusaModel.from_checkpoint(
        checkpoint_path=checkpoint_path,
        mlx_model_name="mlx-community/gemma-3-270m-it-bf16",
    )
    print("Model loaded!")

    # Determine speculation mode based on --tree-size
    # tree-size 1: baseline (no speculation)
    # tree-size 2: simple 2-token speculation
    # tree-size 3: depth-2 speculation (2 heads, 3 tokens)
    # tree-size 4+: tree-based speculation with generate_mtp
    tree_choices = None
    use_depth2 = args.depth2
    spec_tokens = args.spec_tokens

    if args.tree_size is not None:
        if args.tree_size == 1:
            # Size 1 = baseline (just main model prediction)
            print("Using tree-size 1: baseline (no speculation)")
            # Will only run baseline
        elif args.tree_size == 2:
            # Size 2 = simple 2-token speculation
            print("Using tree-size 2: simple 2-token speculation")
            spec_tokens = 2
        elif args.tree_size == 3:
            # Size 3 = depth-2 speculation (2 heads)
            print("Using tree-size 3: depth-2 speculation (2 heads, 3 tokens)")
            use_depth2 = True
        else:
            # Size 4+ = tree-based speculation
            head_acc_path = os.path.join(checkpoint_path, "head_acc.json")
            tree_choices = generate_optimal_tree(
                head_acc_path=head_acc_path,
                num_heads=model.medusa_num_heads,
                tree_size=args.tree_size,
            )
            if tree_choices is None:
                print(f"Warning: Could not load head_acc.json from {head_acc_path}")
                print("Falling back to simple speculation")
            else:
                depth = get_tree_max_depth(tree_choices)
                print(f"Using optimal tree: size={len(tree_choices)}, depth={depth}")
                print(f"Tree nodes: {tree_choices[:10]}..." if len(tree_choices) > 10 else f"Tree nodes: {tree_choices}")

    results = []
    collect_entropy = args.collect_entropy_data is not None

    run_baseline = not args.speculation_only
    # For tree-size 1, only run baseline
    run_speculation = not args.baseline_only and args.tree_size != 1

    if args.task in ["gsm8k", "both"]:
        if run_baseline:
            r = run_gsm8k_eval(model, args.n, args.max_tokens, use_speculation=False, skip_eval=args.skip_eval)
            results.append(r)
        if run_speculation:
            if use_depth2:
                r = run_gsm8k_eval(model, args.n, args.max_tokens, use_speculation=False,
                                   use_depth2=True, skip_eval=args.skip_eval, collect_entropy=collect_entropy,
                                   entropy_threshold=args.entropy_threshold)
            elif tree_choices is not None:
                r = run_gsm8k_eval(model, args.n, args.max_tokens, use_speculation=False,
                                   tree_choices=tree_choices, tree_size=args.tree_size, skip_eval=args.skip_eval,
                                   profile=args.profile, use_fused_kernels=args.use_fused_kernels)
            else:
                r = run_gsm8k_eval(model, args.n, args.max_tokens, use_speculation=True, spec_tokens=spec_tokens,
                                   skip_eval=args.skip_eval, collect_entropy=collect_entropy,
                                   entropy_threshold=args.entropy_threshold)
            results.append(r)

    if args.task in ["humaneval", "both"]:
        if run_baseline:
            r = run_humaneval_eval(model, args.n, args.max_tokens, use_speculation=False, skip_eval=args.skip_eval)
            results.append(r)
        if run_speculation:
            if use_depth2:
                r = run_humaneval_eval(model, args.n, args.max_tokens, use_speculation=False,
                                       use_depth2=True, skip_eval=args.skip_eval, collect_entropy=collect_entropy,
                                       entropy_threshold=args.entropy_threshold)
            elif tree_choices is not None:
                r = run_humaneval_eval(model, args.n, args.max_tokens, use_speculation=False,
                                       tree_choices=tree_choices, tree_size=args.tree_size, skip_eval=args.skip_eval,
                                       profile=args.profile, use_fused_kernels=args.use_fused_kernels)
            else:
                r = run_humaneval_eval(model, args.n, args.max_tokens, use_speculation=True, spec_tokens=spec_tokens,
                                       skip_eval=args.skip_eval, collect_entropy=collect_entropy,
                                       entropy_threshold=args.entropy_threshold)
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
    results_dir = os.path.expanduser(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"mlx_eval_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Save entropy data if collected
    if args.collect_entropy_data:
        all_entropy_records = []
        for r in results:
            if r.get("entropy_records"):
                for rec in r["entropy_records"]:
                    rec["task"] = r["task"]
                    rec["mode"] = r["mode"]
                    all_entropy_records.append(rec)

        if all_entropy_records:
            with open(args.collect_entropy_data, "w") as f:
                json.dump({
                    "records": all_entropy_records,
                    "total_iterations": len(all_entropy_records),
                    "summary": {
                        "mean_entropy": sum(r["entropy"] for r in all_entropy_records) / len(all_entropy_records),
                        "mean_accept_length": sum(r["accept_length"] for r in all_entropy_records) / len(all_entropy_records),
                    }
                }, f, indent=2)
            print(f"Entropy data saved to {args.collect_entropy_data} ({len(all_entropy_records)} iterations)")


if __name__ == "__main__":
    main()
