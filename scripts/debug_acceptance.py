#!/usr/bin/env python
"""
Debug script to understand why acceptance rate is so low for attention mixer on GPU.
"""

import argparse
import json
import torch
from pathlib import Path


def load_model_and_checkpoint(checkpoint_dir: str, device: str):
    """Load model with checkpoint on specified device."""
    from nanochat.gemma_medusa.model import GemmaMedusaModel

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
        use_head_mixer=config.get('use_head_mixer', True),
        mixer_type=config.get('mixer_type', 'attention'),
        attn_num_layers=config.get('attn_num_layers', 2),
    )

    warnings = model.load_medusa_state_dict(checkpoint, strict=False)
    for w in warnings:
        print(f"  Warning: {w}")

    model.eval()
    return model, config


def debug_generation_step(model, input_ids, device_name):
    """Debug a single generation step."""
    from nanochat.gemma_medusa.model import (
        generate_tree_buffers,
        DEFAULT_TREES,
    )

    device = model.get_device()
    dtype = model._dtype
    num_heads = model.medusa_num_heads
    topk = 10
    temperature = 0.0

    # Get tree choices (use default tree for debugging)
    tree_choices = DEFAULT_TREES[num_heads]
    buffers = generate_tree_buffers(tree_choices, device, topk)
    max_speculation = max(len(c) for c in tree_choices) + 1

    print(f"\n=== DEBUG GENERATION STEP ({device_name}) ===")
    print(f"Input tokens: {input_ids}")
    print(f"Tree size: {len(tree_choices) + 1} nodes")

    # Get initial logits
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        main_logits, medusa_logits = model._compute_logits(
            model._get_hidden_states(input_tensor),
            return_medusa=True,
            last_only=True
        )

    # Take last position
    last_main = main_logits[:, 0, :]  # (B, vocab)
    last_medusa = medusa_logits[:, :, 0, :]  # (num_heads, B, vocab)

    # Show what tokens are predicted
    main_pred = last_main.argmax(dim=-1).item()
    print(f"\nBase model prediction: {main_pred}")

    medusa_preds = []
    for h in range(num_heads):
        topk_tokens = last_medusa[h, 0].topk(5).indices.tolist()
        medusa_preds.append(topk_tokens[0])
        print(f"Head {h} top-5: {topk_tokens}")

    # Generate candidates
    candidates, tree_candidates = model._generate_candidates(
        last_main, last_medusa, buffers, topk, temperature
    )

    print(f"\nGenerated {candidates.shape[0]} candidates, tree_candidates: {tree_candidates.shape}")
    print(f"First few candidates:")
    for i in range(min(5, candidates.shape[0])):
        print(f"  {i}: {candidates[i].tolist()}")

    # Now do a forward pass to verify
    with torch.inference_mode():
        tree_logits, ret_indices, valid_mask = model.forward_mtp(
            input_tensor, tree_candidates, buffers
        )

    print(f"\nTree logits shape: {tree_logits.shape}")
    print(f"Retrieve indices shape: {ret_indices.shape}")

    # Get predictions at each tree position
    tree_predictions = tree_logits.argmax(dim=-1)  # (tree_len,)
    print(f"\nTree predictions (first 20): {tree_predictions[:20].tolist()}")

    # Check first candidate acceptance
    for cand_idx in range(min(5, candidates.shape[0])):
        cand = candidates[cand_idx]
        ret_idx = ret_indices[cand_idx]
        mask = valid_mask[cand_idx]

        # Get predictions for this candidate
        safe_indices = ret_idx.clamp(min=0)
        cand_preds = tree_predictions[safe_indices]

        print(f"\nCandidate {cand_idx}:")
        print(f"  Tokens:      {cand.tolist()}")
        print(f"  Retrieve:    {ret_idx.tolist()}")
        print(f"  Valid mask:  {mask.tolist()}")
        print(f"  Tree preds:  {cand_preds.tolist()}")

        # Check matches
        matches = (cand[1:] == cand_preds[:-1])
        matches = matches & mask[1:]
        print(f"  Matches:     {matches.tolist()}")

        # Find accept length
        cumulative = torch.cumprod(matches.int(), dim=0)
        accept_len = cumulative.sum().item()
        print(f"  Accept length: {accept_len}")

    # Run the actual evaluation
    best_candidate, accept_length = model._evaluate_candidates_greedy_fast(
        tree_logits, candidates, ret_indices, valid_mask
    )

    print(f"\nFinal result: best_candidate={best_candidate}, accept_length={accept_length}")

    return accept_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str, required=True)
    parser.add_argument('--prompt', '-p', type=str, default="The quick brown fox")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-270m-it')
    input_ids = tokenizer.encode(args.prompt)

    print(f"Prompt: '{args.prompt}'")
    print(f"Input IDs: {input_ids}")

    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("TESTING GPU")
        print("="*60)
        model_gpu, _ = load_model_and_checkpoint(args.checkpoint, 'cuda')
        gpu_accept = debug_generation_step(model_gpu, input_ids, "GPU")
        del model_gpu
        torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("TESTING CPU")
    print("="*60)
    model_cpu, _ = load_model_and_checkpoint(args.checkpoint, 'cpu')
    cpu_accept = debug_generation_step(model_cpu, input_ids, "CPU")

    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"GPU accept_length: {gpu_accept}")
        print(f"CPU accept_length: {cpu_accept}")


if __name__ == "__main__":
    main()
