"""
Profile overhead sources in MLX speculation implementations.

Isolates:
1. Pure forward pass scaling with tree sizes
2. mx.eval() synchronization cost
3. Equivalent tree comparison (3tok simple vs MTP)

Usage:
    conda activate fmtp-mlx
    python -m scripts.mlx_profile_overhead
"""

import time
import argparse
import os
from typing import List, Dict, Tuple

import mlx.core as mx

from nanochat.mlx.model import GemmaMedusaModel, trim_cache


def profile_forward_scaling(model: GemmaMedusaModel, prompt_ids: List[int], n_iters: int = 50) -> Dict[int, float]:
    """
    Test 1: Pure forward pass scaling with tree sizes.

    Measures raw GPU time for forward passes at different sequence lengths,
    isolating GPU compute from Python overhead.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Forward Pass Scaling")
    print("=" * 60)

    results = {}

    # Prefill
    cache = model.base_model.make_cache()
    x = mx.array([prompt_ids], dtype=mx.int32)
    hidden = model._get_hidden_states(x, cache=cache)
    main_logits, medusa_logits = model._compute_logits(hidden, return_medusa=True)
    mx.eval(main_logits, medusa_logits)

    print(f"Prefill complete: {len(prompt_ids)} tokens cached")
    print(f"Running {n_iters} iterations per tree size...\n")

    baseline_time = None

    for tree_size in [1, 2, 3, 4, 8, 16, 32, 64, 80]:
        # Create dummy tree input (simulating tree tokens)
        dummy_tokens = mx.array([[0] * tree_size], dtype=mx.int32)

        times = []
        for i in range(n_iters):
            start = time.perf_counter()
            hidden = model._get_hidden_states(dummy_tokens, cache=cache)
            logits, med_logits = model._compute_logits(hidden, return_medusa=True, num_active_heads=2)
            mx.eval(logits, med_logits)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # ms

            # Trim cache back to restore state
            trim_cache(cache, tree_size)

        # Skip warmup iterations
        avg = sum(times[5:]) / len(times[5:])
        std = (sum((t - avg) ** 2 for t in times[5:]) / len(times[5:])) ** 0.5
        results[tree_size] = avg

        if tree_size == 1:
            baseline_time = avg
            print(f"  Tree size {tree_size:2d}: {avg:6.2f}ms (±{std:.2f}ms) [baseline]")
        else:
            overhead_pct = (avg - baseline_time) / baseline_time * 100
            print(f"  Tree size {tree_size:2d}: {avg:6.2f}ms (±{std:.2f}ms) [+{overhead_pct:.1f}%]")

    return results


def profile_eval_overhead(n_iters: int = 100) -> Dict[str, float]:
    """
    Test 2: mx.eval() synchronization cost.

    Measures the Python-to-Metal synchronization overhead by comparing
    lazy compute time vs actual evaluation time.
    """
    print("\n" + "=" * 60)
    print("TEST 2: mx.eval() Synchronization Cost")
    print("=" * 60)

    # Create operations of different sizes
    sizes = [(1, 1024), (1, 4096), (1, 16384)]

    results = {}

    for batch, dim in sizes:
        x = mx.ones((batch, dim))
        w = mx.ones((dim, dim))

        # Measure lazy compute time (just graph building)
        lazy_times = []
        for _ in range(n_iters):
            start = time.perf_counter()
            y = x @ w
            lazy_times.append((time.perf_counter() - start) * 1000)

        # Measure eval time (actual execution)
        eval_times = []
        for _ in range(n_iters):
            y = x @ w
            start = time.perf_counter()
            mx.eval(y)
            eval_times.append((time.perf_counter() - start) * 1000)

        # Combined (graph + eval)
        combined_times = []
        for _ in range(n_iters):
            start = time.perf_counter()
            y = x @ w
            mx.eval(y)
            combined_times.append((time.perf_counter() - start) * 1000)

        lazy_avg = sum(lazy_times[10:]) / len(lazy_times[10:])
        eval_avg = sum(eval_times[10:]) / len(eval_times[10:])
        combined_avg = sum(combined_times[10:]) / len(combined_times[10:])

        print(f"\n  Shape ({batch}, {dim}) x ({dim}, {dim}):")
        print(f"    Lazy (graph build):  {lazy_avg:.4f}ms")
        print(f"    Eval (sync+compute): {eval_avg:.4f}ms")
        print(f"    Combined:            {combined_avg:.4f}ms")

        results[f"{dim}"] = {
            "lazy_ms": lazy_avg,
            "eval_ms": eval_avg,
            "combined_ms": combined_avg,
        }

    return results


def profile_python_overhead(model: GemmaMedusaModel, prompt_ids: List[int], n_iters: int = 20) -> Dict[str, float]:
    """
    Test 3: Python overhead in speculation loops.

    Measures time spent in Python code vs GPU by comparing:
    - Total generation time
    - Number of mx.eval() calls
    - Estimated GPU time from forward pass scaling
    """
    print("\n" + "=" * 60)
    print("TEST 3: Python Overhead Analysis")
    print("=" * 60)

    n_tokens = 64

    # Run baseline (1 eval per token)
    baseline_times = []
    baseline_evals = []
    for _ in range(n_iters):
        start = time.perf_counter()
        response, n_tok, _ = model.generate_standard(
            prompt=model.tokenizer.decode(prompt_ids),
            max_new_tokens=n_tokens,
        )
        elapsed = time.perf_counter() - start
        baseline_times.append(elapsed)
        baseline_evals.append(n_tok)  # 1 eval per token

    baseline_avg = sum(baseline_times[3:]) / len(baseline_times[3:])
    baseline_tok_avg = sum(baseline_evals[3:]) / len(baseline_evals[3:])

    # Run simple 2-tok speculation
    spec2_times = []
    spec2_passes = []
    for _ in range(n_iters):
        start = time.perf_counter()
        out, stats = model.generate_simple_speculation(
            input_ids=prompt_ids,
            max_new_tokens=n_tokens,
            stop_token_ids=model.tokenizer.eos_token_ids,
        )
        elapsed = time.perf_counter() - start
        spec2_times.append(elapsed)
        spec2_passes.append(stats.forward_passes)

    spec2_avg = sum(spec2_times[3:]) / len(spec2_times[3:])
    spec2_passes_avg = sum(spec2_passes[3:]) / len(spec2_passes[3:])
    spec2_tok_avg = n_tokens  # Max tokens

    # Run simple 3-tok speculation
    spec3_times = []
    spec3_passes = []
    for _ in range(n_iters):
        start = time.perf_counter()
        out, stats = model.generate_simple_speculation_3tok(
            input_ids=prompt_ids,
            max_new_tokens=n_tokens,
            stop_token_ids=model.tokenizer.eos_token_ids,
        )
        elapsed = time.perf_counter() - start
        spec3_times.append(elapsed)
        spec3_passes.append(stats.forward_passes)

    spec3_avg = sum(spec3_times[3:]) / len(spec3_times[3:])
    spec3_passes_avg = sum(spec3_passes[3:]) / len(spec3_passes[3:])

    print(f"\nGenerating {n_tokens} tokens:")
    print(f"\n  Baseline (standard):")
    print(f"    Total time:     {baseline_avg*1000:.1f}ms")
    print(f"    Forward passes: {baseline_tok_avg:.0f}")
    print(f"    Time per pass:  {baseline_avg*1000/baseline_tok_avg:.2f}ms")

    print(f"\n  Simple 2-tok speculation:")
    print(f"    Total time:     {spec2_avg*1000:.1f}ms")
    print(f"    Forward passes: {spec2_passes_avg:.1f}")
    print(f"    Time per pass:  {spec2_avg*1000/spec2_passes_avg:.2f}ms")
    print(f"    Speedup:        {baseline_avg/spec2_avg:.2f}x")

    print(f"\n  Simple 3-tok speculation:")
    print(f"    Total time:     {spec3_avg*1000:.1f}ms")
    print(f"    Forward passes: {spec3_passes_avg:.1f}")
    print(f"    Time per pass:  {spec3_avg*1000/spec3_passes_avg:.2f}ms")
    print(f"    Speedup:        {baseline_avg/spec3_avg:.2f}x")

    return {
        "baseline": {"time_ms": baseline_avg * 1000, "passes": baseline_tok_avg},
        "spec2": {"time_ms": spec2_avg * 1000, "passes": spec2_passes_avg},
        "spec3": {"time_ms": spec3_avg * 1000, "passes": spec3_passes_avg},
    }


def profile_equivalent_comparison(model: GemmaMedusaModel, prompt_ids: List[int], n_iters: int = 10) -> Dict[str, dict]:
    """
    Test 4: Compare 3tok simple vs equivalent MTP tree.

    Both methods verify 3 tokens per iteration with depth-2 tree.
    This isolates implementation overhead from algorithmic differences.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Equivalent Tree Comparison")
    print("=" * 60)
    print("Comparing generate_simple_speculation_3tok vs generate_mtp")
    print("with equivalent 3-node tree: [(0,), (0,0)]")

    n_tokens = 64
    results = {}

    # Run simple 3-tok multiple times
    spec3_times = []
    spec3_tokens = []
    spec3_passes = []
    for _ in range(n_iters):
        start = time.perf_counter()
        out1, stats1 = model.generate_simple_speculation_3tok(
            input_ids=prompt_ids,
            max_new_tokens=n_tokens,
            stop_token_ids=model.tokenizer.eos_token_ids,
        )
        elapsed = time.perf_counter() - start
        spec3_times.append(elapsed)
        spec3_tokens.append(len(out1) - len(prompt_ids))
        spec3_passes.append(stats1.forward_passes)

    spec3_avg_time = sum(spec3_times[2:]) / len(spec3_times[2:])
    spec3_avg_tokens = sum(spec3_tokens[2:]) / len(spec3_tokens[2:])
    spec3_avg_passes = sum(spec3_passes[2:]) / len(spec3_passes[2:])

    # Run MTP with equivalent tree [(0,), (0,0)]
    tree_choices = [(0,), (0, 0)]
    mtp_times = []
    mtp_tokens = []
    mtp_passes = []
    for _ in range(n_iters):
        start = time.perf_counter()
        out2, stats2 = model.generate_mtp(
            input_ids=prompt_ids,
            max_new_tokens=n_tokens,
            tree_choices=tree_choices,
            num_active_heads=2,
            stop_token_ids=model.tokenizer.eos_token_ids,
        )
        elapsed = time.perf_counter() - start
        mtp_times.append(elapsed)
        mtp_tokens.append(len(out2) - len(prompt_ids))
        mtp_passes.append(stats2.forward_passes)

    mtp_avg_time = sum(mtp_times[2:]) / len(mtp_times[2:])
    mtp_avg_tokens = sum(mtp_tokens[2:]) / len(mtp_tokens[2:])
    mtp_avg_passes = sum(mtp_passes[2:]) / len(mtp_passes[2:])

    print(f"\n  generate_simple_speculation_3tok:")
    print(f"    Time:           {spec3_avg_time*1000:.1f}ms")
    print(f"    Tokens:         {spec3_avg_tokens:.1f}")
    print(f"    Forward passes: {spec3_avg_passes:.1f}")
    print(f"    Tok/s:          {spec3_avg_tokens/spec3_avg_time:.1f}")
    print(f"    Tok/pass:       {spec3_avg_tokens/spec3_avg_passes:.2f}")

    print(f"\n  generate_mtp (tree=3):")
    print(f"    Time:           {mtp_avg_time*1000:.1f}ms")
    print(f"    Tokens:         {mtp_avg_tokens:.1f}")
    print(f"    Forward passes: {mtp_avg_passes:.1f}")
    print(f"    Tok/s:          {mtp_avg_tokens/mtp_avg_time:.1f}")
    print(f"    Tok/pass:       {mtp_avg_tokens/mtp_avg_passes:.2f}")

    overhead_ratio = mtp_avg_time / spec3_avg_time
    print(f"\n  OVERHEAD RATIO: {overhead_ratio:.2f}x")
    print(f"  (MTP is {(overhead_ratio-1)*100:.0f}% slower for equivalent tree)")

    # Per-iteration breakdown
    spec3_time_per_pass = spec3_avg_time * 1000 / spec3_avg_passes
    mtp_time_per_pass = mtp_avg_time * 1000 / mtp_avg_passes
    overhead_per_pass = mtp_time_per_pass - spec3_time_per_pass

    print(f"\n  Per-iteration overhead:")
    print(f"    Simple 3tok: {spec3_time_per_pass:.2f}ms/iter")
    print(f"    MTP tree-3:  {mtp_time_per_pass:.2f}ms/iter")
    print(f"    Overhead:    {overhead_per_pass:.2f}ms/iter (+{overhead_per_pass/spec3_time_per_pass*100:.0f}%)")

    return {
        "spec3": {
            "time_s": spec3_avg_time,
            "tokens": spec3_avg_tokens,
            "tok_s": spec3_avg_tokens / spec3_avg_time,
            "forward_passes": spec3_avg_passes,
            "ms_per_pass": spec3_time_per_pass,
        },
        "mtp_tree3": {
            "time_s": mtp_avg_time,
            "tokens": mtp_avg_tokens,
            "tok_s": mtp_avg_tokens / mtp_avg_time,
            "forward_passes": mtp_avg_passes,
            "ms_per_pass": mtp_time_per_pass,
        },
        "overhead_ratio": overhead_ratio,
        "overhead_per_pass_ms": overhead_per_pass,
    }


def profile_component_breakdown(model: GemmaMedusaModel, prompt_ids: List[int], tree_size: int = 80, n_iters: int = 20) -> Dict[str, float]:
    """
    Test 5: Component breakdown for large tree sizes.

    Accurately profiles what generate_mtp actually does per iteration:
    1. Tree verification: backbone + LM head for tree_size tokens (no medusa)
    2. Medusa for next iter: 4 ResBlocks + 4 LM heads for 1 token only

    Total LM head calls per iter: tree_size + num_heads (e.g., 80 + 4 = 84)
    """
    print("\n" + "=" * 60)
    print(f"TEST 5: Component Breakdown (tree_size={tree_size})")
    print("=" * 60)

    num_heads = model.medusa_num_heads

    # Prefill
    cache = model.base_model.make_cache()
    x = mx.array([prompt_ids], dtype=mx.int32)
    hidden = model._get_hidden_states(x, cache=cache)
    main_logits, medusa_logits = model._compute_logits(hidden, return_medusa=True)
    mx.eval(main_logits, medusa_logits)

    print(f"Prefill complete: {len(prompt_ids)} tokens cached")
    print(f"Model has {num_heads} Medusa heads")
    print(f"Running {n_iters} iterations...\n")

    # Create dummy tree input
    dummy_tokens = mx.array([[0] * tree_size], dtype=mx.int32)
    single_token = mx.array([[0]], dtype=mx.int32)

    # =========================================================================
    # 1. Profile backbone only (transformer, no LM head)
    # =========================================================================
    backbone_only_times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        hidden = model._get_hidden_states(dummy_tokens, cache=cache)
        mx.eval(hidden)
        elapsed = time.perf_counter() - start
        backbone_only_times.append(elapsed * 1000)
        trim_cache(cache, tree_size)

    backbone_only_avg = sum(backbone_only_times[3:]) / len(backbone_only_times[3:])

    # =========================================================================
    # 2. Profile tree verification: backbone + LM head for tree_size tokens
    #    This is what generate_mtp does: _compute_logits(return_medusa=False)
    # =========================================================================
    tree_verify_times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        hidden = model._get_hidden_states(dummy_tokens, cache=cache)
        main_logits, _ = model._compute_logits(hidden, return_medusa=False)
        mx.eval(main_logits)
        elapsed = time.perf_counter() - start
        tree_verify_times.append(elapsed * 1000)
        trim_cache(cache, tree_size)

    tree_verify_avg = sum(tree_verify_times[3:]) / len(tree_verify_times[3:])
    lm_head_tree_avg = tree_verify_avg - backbone_only_avg  # LM head cost for tree_size tokens

    # =========================================================================
    # 3. Profile Medusa computation for 1 token only (what generate_mtp does)
    #    This is: 4 ResBlocks + 4 LM head projections for single position
    # =========================================================================
    # First get hidden states for 1 token
    hidden_1tok = model._get_hidden_states(single_token, cache=cache)
    mx.eval(hidden_1tok)
    trim_cache(cache, 1)

    medusa_1tok_times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        _, medusa_logits = model._compute_logits(hidden_1tok, return_medusa=True, num_active_heads=num_heads)
        mx.eval(medusa_logits)
        elapsed = time.perf_counter() - start
        medusa_1tok_times.append(elapsed * 1000)

    medusa_1tok_avg = sum(medusa_1tok_times[3:]) / len(medusa_1tok_times[3:])

    # =========================================================================
    # 4. Total MTP iteration cost (what generate_mtp actually does)
    # =========================================================================
    # Simulate one MTP iteration: tree verify + medusa for 1 token
    mtp_iter_times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        # Step 1: Tree verification (backbone + LM head for tree_size tokens)
        hidden = model._get_hidden_states(dummy_tokens, cache=cache)
        main_logits, _ = model._compute_logits(hidden, return_medusa=False)
        mx.eval(main_logits)

        # Step 2: Medusa for last accepted position (simulate with last hidden)
        last_hidden = hidden[:, -1:, :]  # Extract last position
        _, medusa_logits = model._compute_logits(last_hidden, return_medusa=True, num_active_heads=num_heads)
        mx.eval(medusa_logits)

        elapsed = time.perf_counter() - start
        mtp_iter_times.append(elapsed * 1000)
        trim_cache(cache, tree_size)

    mtp_iter_avg = sum(mtp_iter_times[3:]) / len(mtp_iter_times[3:])

    # =========================================================================
    # 5. Run actual generate_mtp to compare
    # =========================================================================
    n_tokens = 32

    # Simple tree for testing
    tree_choices = []
    for i in range(min(tree_size, num_heads)):
        tree_choices.append(tuple([0] * (i + 1)))
    for i in range(1, min(tree_size - num_heads + 1, 64)):
        tree_choices.append((i,))
    tree_choices = tree_choices[:tree_size]
    tree_depth = max(len(c) for c in tree_choices) if tree_choices else 0

    print(f"  Tree: {len(tree_choices)} nodes, depth {tree_depth}")
    print(f"  Expected LM head calls/iter: {tree_size} (verify) + {num_heads} (medusa) = {tree_size + num_heads}")

    mtp_times = []
    mtp_passes = []
    for _ in range(n_iters):
        start = time.perf_counter()
        out, stats = model.generate_mtp(
            input_ids=prompt_ids,
            max_new_tokens=n_tokens,
            tree_choices=tree_choices,
            num_active_heads=min(tree_depth, num_heads),
            stop_token_ids=model.tokenizer.eos_token_ids,
        )
        elapsed = time.perf_counter() - start
        mtp_times.append(elapsed * 1000)
        mtp_passes.append(stats.forward_passes)

    mtp_gen_avg = sum(mtp_times[3:]) / len(mtp_times[3:])
    mtp_passes_avg = sum(mtp_passes[3:]) / len(mtp_passes[3:])
    mtp_per_pass = mtp_gen_avg / mtp_passes_avg if mtp_passes_avg > 0 else 0

    # =========================================================================
    # Output
    # =========================================================================
    print(f"\n  === Component Costs (tree_size={tree_size}) ===")
    print(f"  Backbone only (no LM head):     {backbone_only_avg:6.2f}ms")
    print(f"  LM head for {tree_size} tokens:         {lm_head_tree_avg:6.2f}ms")
    print(f"  Tree verification total:        {tree_verify_avg:6.2f}ms")
    print(f"  Medusa for 1 token ({num_heads} heads):   {medusa_1tok_avg:6.2f}ms")
    print(f"  ----------------------------------------")
    print(f"  Simulated MTP iteration:        {mtp_iter_avg:6.2f}ms")

    python_overhead = mtp_per_pass - mtp_iter_avg
    print(f"\n  === Actual generate_mtp ===")
    print(f"  GPU work per iter:              {mtp_iter_avg:6.2f}ms")
    print(f"  Python overhead per iter:       {python_overhead:6.2f}ms")
    print(f"  Actual time per iter:           {mtp_per_pass:6.2f}ms")
    print(f"  Python overhead ratio:          {python_overhead/mtp_iter_avg*100:.1f}%")

    print(f"\n  === MTP Performance ===")
    print(f"  Forward passes:                 {mtp_passes_avg:.1f}")
    print(f"  Total time:                     {mtp_gen_avg:.1f}ms")
    print(f"  Tok/s:                          {n_tokens/(mtp_gen_avg/1000):.1f}")

    return {
        "tree_size": tree_size,
        "backbone_only_ms": backbone_only_avg,
        "lm_head_tree_ms": lm_head_tree_avg,
        "tree_verify_ms": tree_verify_avg,
        "medusa_1tok_ms": medusa_1tok_avg,
        "mtp_iter_simulated_ms": mtp_iter_avg,
        "mtp_per_pass_actual_ms": mtp_per_pass,
        "python_overhead_ms": python_overhead,
        "python_overhead_pct": python_overhead / mtp_iter_avg * 100 if mtp_iter_avg > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile MLX speculation overhead")
    parser.add_argument("--checkpoint", type=str,
                        default="~/.cache/nanochat/gemma_medusa_270m_wildchat_100k_nolora",
                        help="Medusa checkpoint path")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "forward", "eval", "python", "comparison", "breakdown"],
                        help="Which test to run")
    parser.add_argument("--tree-size", type=int, default=80,
                        help="Tree size for breakdown test")
    parser.add_argument("--n-iters", type=int, default=20,
                        help="Number of iterations for profiling")
    args = parser.parse_args()

    # Expand checkpoint path
    checkpoint_path = os.path.expanduser(args.checkpoint)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = GemmaMedusaModel.from_checkpoint(
        checkpoint_path=checkpoint_path,
        mlx_model_name="mlx-community/gemma-3-270m-it-bf16",
    )
    model.checkpoint_path = checkpoint_path  # Store for head_acc.json lookup
    print("Model loaded!")

    # Create a test prompt
    prompt = "<start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n"
    prompt_ids = model.tokenizer.encode(prompt)
    print(f"Prompt: {len(prompt_ids)} tokens")

    results = {}

    if args.test in ["all", "forward"]:
        results["forward_scaling"] = profile_forward_scaling(model, prompt_ids)

    if args.test in ["all", "eval"]:
        results["eval_overhead"] = profile_eval_overhead()

    if args.test in ["all", "python"]:
        results["python_overhead"] = profile_python_overhead(model, prompt_ids)

    if args.test in ["all", "comparison"]:
        results["equivalent_comparison"] = profile_equivalent_comparison(model, prompt_ids)

    if args.test == "breakdown":
        results["component_breakdown"] = profile_component_breakdown(model, prompt_ids, tree_size=args.tree_size, n_iters=args.n_iters)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if "forward_scaling" in results:
        fs = results["forward_scaling"]
        print(f"\nForward pass scaling (GPU compute):")
        print(f"  1 token:  {fs[1]:.2f}ms")
        print(f"  4 tokens: {fs[4]:.2f}ms (+{(fs[4]/fs[1]-1)*100:.0f}%)")
        if 8 in fs:
            print(f"  8 tokens: {fs[8]:.2f}ms (+{(fs[8]/fs[1]-1)*100:.0f}%)")

    if "equivalent_comparison" in results:
        ec = results["equivalent_comparison"]
        print(f"\nImplementation overhead (equivalent tree):")
        print(f"  Simple 3tok: {ec['spec3']['ms_per_pass']:.2f}ms/iter")
        print(f"  MTP tree-3:  {ec['mtp_tree3']['ms_per_pass']:.2f}ms/iter")
        print(f"  Overhead:    {ec['overhead_per_pass_ms']:.2f}ms/iter")
        print(f"  Ratio:       {ec['overhead_ratio']:.2f}x")


if __name__ == "__main__":
    main()
