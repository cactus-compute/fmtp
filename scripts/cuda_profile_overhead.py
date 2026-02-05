"""
Profile overhead sources in PyTorch/CUDA speculation implementations.

This is the CUDA equivalent of mlx_profile_overhead.py for comparing
M4 Pro (MLX) vs A100 (CUDA) performance characteristics.

Usage:
    python -m scripts.cuda_profile_overhead --checkpoint ~/.cache/nanochat/gemma_medusa_270m_wildchat_100k_kl
    python -m scripts.cuda_profile_overhead --test breakdown --tree-size 8 --n-iters 25
"""

import time
import argparse
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch

from nanochat.gemma_medusa.model import GemmaMedusaModel


def load_model_and_checkpoint(checkpoint_dir: str, device: str = "cuda"):
    """Load model with checkpoint on specified device."""
    device = torch.device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Load checkpoint and config
    checkpoint_path = Path(checkpoint_dir) / "final" / "medusa_heads.pt"
    config_path = Path(checkpoint_dir) / "config.json"

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = checkpoint.get('config', {})

    model = GemmaMedusaModel(
        model_name=config.get('model_name', 'google/gemma-3-270m-it'),
        medusa_num_heads=config.get('medusa_num_heads', 4),
        medusa_num_layers=config.get('medusa_num_layers', 2),
        lora_rank=config.get('lora_rank', 256),
        lora_alpha=config.get('lora_alpha', 512),
        device=device,
        dtype=dtype,
        freeze_base=True,
        zero_init_mlp=config.get('zero_init_mlp', True),
        use_head_mixer=config.get('use_head_mixer', False),
        mixer_type=config.get('mixer_type', 'mlp'),
        attn_num_layers=config.get('attn_num_layers', 0),
    )

    warnings = model.load_medusa_state_dict(checkpoint, strict=False)
    for w in warnings:
        print(f"  Warning: {w}")

    model.eval()
    return model, config


def profile_component_breakdown(
    model: GemmaMedusaModel,
    prompt_ids: List[int],
    tree_size: int = 80,
    n_iters: int = 20,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Profile component breakdown for tree-based speculation.

    Accurately profiles what generate_mtp actually does per iteration:
    1. Tree verification: backbone + LM head for tree_size tokens (no medusa)
    2. Medusa for next iter: ResBlocks + LM heads for 1 token only

    Total LM head calls per iter: tree_size + num_heads (e.g., 80 + 4 = 84)
    """
    print("\n" + "=" * 60)
    print(f"TEST: Component Breakdown (tree_size={tree_size})")
    print("=" * 60)

    num_heads = model.medusa_num_heads
    device = model.get_device()

    # Prefill to populate KV cache
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model.base_model.model(
            input_ids=input_tensor,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        prefill_hidden = outputs.last_hidden_state

    # Synchronize and warmup
    torch.cuda.synchronize()

    print(f"Prefill complete: {len(prompt_ids)} tokens cached")
    print(f"Model has {num_heads} Medusa heads")
    print(f"Running {n_iters} iterations...\n")

    # Create dummy tree input
    dummy_tokens = torch.zeros((1, tree_size), dtype=torch.long, device=device)
    single_token = torch.zeros((1, 1), dtype=torch.long, device=device)

    # Position IDs for tree tokens (starting after prefill)
    cache_len = len(prompt_ids)
    tree_position_ids = torch.arange(cache_len, cache_len + tree_size, device=device).unsqueeze(0)
    single_position_ids = torch.tensor([[cache_len]], device=device)

    # =========================================================================
    # 1. Profile backbone only (transformer, no LM head)
    # =========================================================================
    backbone_only_times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.base_model.model(
                input_ids=dummy_tokens,
                past_key_values=past_key_values,
                position_ids=tree_position_ids,
                use_cache=True,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        backbone_only_times.append(elapsed * 1000)

    backbone_only_avg = sum(backbone_only_times[3:]) / len(backbone_only_times[3:])

    # =========================================================================
    # 2. Profile tree verification: backbone + LM head for tree_size tokens
    # =========================================================================
    tree_verify_times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.base_model.model(
                input_ids=dummy_tokens,
                past_key_values=past_key_values,
                position_ids=tree_position_ids,
                use_cache=True,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            # LM head projection
            main_logits = model.base_model.lm_head(hidden)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tree_verify_times.append(elapsed * 1000)

    tree_verify_avg = sum(tree_verify_times[3:]) / len(tree_verify_times[3:])
    lm_head_tree_avg = tree_verify_avg - backbone_only_avg

    # =========================================================================
    # 3. Profile Medusa computation for 1 token only
    # =========================================================================
    # First get hidden states for 1 token
    with torch.no_grad():
        outputs = model.base_model.model(
            input_ids=single_token,
            past_key_values=past_key_values,
            position_ids=single_position_ids,
            use_cache=True,
            return_dict=True,
        )
        hidden_1tok = outputs.last_hidden_state
    torch.cuda.synchronize()

    medusa_1tok_times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            # Compute medusa logits (ResBlocks + LM heads)
            main_logits, medusa_logits = model._compute_logits(
                hidden_1tok, return_medusa=True
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        medusa_1tok_times.append(elapsed * 1000)

    medusa_1tok_avg = sum(medusa_1tok_times[3:]) / len(medusa_1tok_times[3:])

    # =========================================================================
    # 4. Simulated MTP iteration: tree verify + medusa for 1 token
    # =========================================================================
    mtp_iter_times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            # Step 1: Tree verification (backbone + LM head for tree_size tokens)
            outputs = model.base_model.model(
                input_ids=dummy_tokens,
                past_key_values=past_key_values,
                position_ids=tree_position_ids,
                use_cache=True,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            main_logits = model.base_model.lm_head(hidden)

            # Step 2: Medusa for last accepted position
            last_hidden = hidden[:, -1:, :]
            _, medusa_logits = model._compute_logits(last_hidden, return_medusa=True)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        mtp_iter_times.append(elapsed * 1000)

    mtp_iter_avg = sum(mtp_iter_times[3:]) / len(mtp_iter_times[3:])

    # =========================================================================
    # 5. Run actual generate_mtp to compare
    # =========================================================================
    n_tokens = 32

    # Build tree choices
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
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            out, stats = model.generate_mtp(
                input_ids=prompt_ids,
                max_new_tokens=n_tokens,
                tree_choices=tree_choices,
                eos_token_id=None,  # Don't stop early
            )

        torch.cuda.synchronize()
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
    if mtp_iter_avg > 0:
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
    parser = argparse.ArgumentParser(description="Profile CUDA speculation overhead")
    parser.add_argument("--checkpoint", type=str,
                        default="~/.cache/nanochat/gemma_medusa_270m_wildchat_100k_kl",
                        help="Medusa checkpoint path")
    parser.add_argument("--test", type=str, default="breakdown",
                        choices=["breakdown"],
                        help="Which test to run")
    parser.add_argument("--tree-size", type=int, default=80,
                        help="Tree size for breakdown test")
    parser.add_argument("--n-iters", type=int, default=25,
                        help="Number of iterations for profiling")
    args = parser.parse_args()

    # Expand checkpoint path
    checkpoint_path = os.path.expanduser(args.checkpoint)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = "cuda"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model, config = load_model_and_checkpoint(checkpoint_path, device)
    print("Model loaded!")

    # Create a test prompt using transformers tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.get('model_name', 'google/gemma-3-270m-it'))
    prompt = "<start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n"
    prompt_ids = tokenizer.encode(prompt)

    print(f"Prompt: {len(prompt_ids)} tokens")

    results = {}

    if args.test == "breakdown":
        results["component_breakdown"] = profile_component_breakdown(
            model, prompt_ids, tree_size=args.tree_size, n_iters=args.n_iters, device=device
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)


if __name__ == "__main__":
    main()
