#!/usr/bin/env python3
"""Debug script to understand Medusa head prediction accuracy at inference time."""

import argparse
import torch
import torch.nn.functional as F
from nanochat.gemma_medusa.model import GemmaMedusaModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--medusa-num-heads', type=int, default=4)
    parser.add_argument('--medusa-num-layers', type=int, default=2)
    parser.add_argument('--lora-rank', type=int, default=64)
    parser.add_argument('--lora-alpha', type=int, default=128)
    parser.add_argument('--attn-num-layers', type=int, default=0)
    parser.add_argument('--num-tokens', type=int, default=100, help='Number of tokens to generate for testing')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model...")
    model = GemmaMedusaModel(
        model_name='google/gemma-3-270m-it',
        medusa_num_heads=args.medusa_num_heads,
        medusa_num_layers=args.medusa_num_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        attn_num_layers=args.attn_num_layers,
        device=device,
        dtype=torch.bfloat16,
    )
    # Load checkpoint
    import os
    checkpoint_path = os.path.join(args.checkpoint, "final", "medusa_heads.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(args.checkpoint, "medusa_heads.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.medusa_heads.load_state_dict(checkpoint['medusa_heads'])
    model.eval()

    # Get tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-270m-it')

    # Test prompt
    prompt = "The quick brown fox jumps over the lazy dog. This is a test of the Medusa speculative decoding system."
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)[0].tolist()

    print(f"Prompt: {prompt}")
    print(f"Prompt length: {len(input_ids)} tokens")
    print(f"\nGenerating {args.num_tokens} tokens and measuring head accuracy...\n")

    # Statistics
    # Note: The acceptance check compares head k's prediction to the base model's prediction
    # for position T+1+k (where T is current position). Head k is trained with shift=2+k,
    # meaning at position T-1 it predicts position T+1+k. So we need to compare head k
    # to the base model's prediction k+1 tokens ahead.
    head_correct_top1 = [0] * args.medusa_num_heads
    head_correct_top10 = [0] * args.medusa_num_heads
    total_predictions = [0] * args.medusa_num_heads

    current_tokens = list(input_ids)

    with torch.no_grad():
        for step in range(args.num_tokens):
            # Get logits from model at current position
            input_tensor = torch.tensor([current_tokens], device=device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                main_logits, medusa_logits = model.forward(input_tensor, return_medusa=True, last_only=True)

            # main_logits: (B, 1, vocab) - what the base model predicts for position T+1
            # medusa_logits: (num_heads, B, 1, vocab) - what each head predicts
            #
            # Critical insight: Head 0 is trained to predict position T+2 (shift=2+0=2 from position T).
            # But in inference with last_only=True, we're at position T-1 (last input position),
            # so head 0 predicts position T-1+2 = T+1.
            #
            # The acceptance logic compares:
            #   candidates[:, 1] (head 0's prediction) vs tree_logits[0].argmax()
            # where tree_logits[0] is the model's prediction AFTER seeing the speculated root token.
            #
            # So we need to:
            # 1. Get base_next_token = model prediction for position T
            # 2. Feed base_next_token and get prediction for position T+1
            # 3. Compare head 0 to that T+1 prediction

            # Get base model's next token (position T)
            base_next_token = main_logits[0, 0].argmax().item()

            # Feed the base token and get the model's prediction for position T+1
            extended_tokens = current_tokens + [base_next_token]
            extended_tensor = torch.tensor([extended_tokens], device=device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                extended_logits = model.forward(extended_tensor, return_medusa=False, last_only=True)

            # This is what the base model predicts for position T+1
            # This is the ground truth that head 0 should match for acceptance
            ground_truth_t1 = extended_logits[0, 0].argmax().item()

            # For head 0: compare to ground_truth_t1 (position T+1)
            # For head k: would need to extend further to get position T+1+k
            # For now, we focus on head 0 which is most important
            for h in range(args.medusa_num_heads):
                head_logits = medusa_logits[h, 0, 0]  # (vocab,)

                # For head 0, compare to ground_truth_t1
                # For head k>0, we'd need to generate k more tokens, but let's approximate
                # by noting that all heads compare against the verification at their respective positions
                if h == 0:
                    target_token = ground_truth_t1
                else:
                    # For simplicity, use base_next_token as an approximation for higher heads
                    # This won't be perfect but gives a sense of alignment
                    target_token = ground_truth_t1  # Use same for now (better than base_next_token)

                # Top-1 accuracy
                head_top1 = head_logits.argmax().item()
                if head_top1 == target_token:
                    head_correct_top1[h] += 1

                # Top-10 accuracy
                head_top10 = torch.topk(head_logits, 10).indices.tolist()
                if target_token in head_top10:
                    head_correct_top10[h] += 1

                total_predictions[h] += 1

            # Add base model's token to sequence (autoregressive)
            current_tokens.append(base_next_token)

            if step < 10 or step % 20 == 0:
                base_token_str = tokenizer.decode([base_next_token])
                gt_t1_str = tokenizer.decode([ground_truth_t1])
                head0_pred = medusa_logits[0, 0, 0].argmax().item()
                head0_pred_str = tokenizer.decode([head0_pred])
                match = "✓" if head0_pred == ground_truth_t1 else "✗"
                print(f"Step {step}: base='{base_token_str}' gt_t1='{gt_t1_str}' head0='{head0_pred_str}' {match}")

    print(f"\n" + "="*50)
    print("HEAD ACCURACY AT INFERENCE TIME")
    print("="*50)

    for h in range(args.medusa_num_heads):
        top1_acc = head_correct_top1[h] / total_predictions[h] if total_predictions[h] > 0 else 0
        top10_acc = head_correct_top10[h] / total_predictions[h] if total_predictions[h] > 0 else 0
        print(f"Head {h}: top1={top1_acc:.3f} ({head_correct_top1[h]}/{total_predictions[h]}), "
              f"top10={top10_acc:.3f} ({head_correct_top10[h]}/{total_predictions[h]})")

    print(f"\n" + "="*50)
    print("COMPARISON WITH TRAINING EVAL")
    print("="*50)
    print("Training eval measures: Given prefix at position T, does head k predict token at position T+2+k?")
    print("Inference acceptance: Given prefix + speculated tokens, does head k's prediction match verification?")
    print("\nKey insight: Head 0 predicts position T+1 (shift=2 from position T-1).")
    print("The acceptance check compares head 0's prediction to tree_logits[0].argmax(),")
    print("which is the base model's prediction AFTER seeing the speculated root token.")
    print("\nIf accuracy here differs from training eval:")
    print("  1. The verification context (with speculated tokens) differs from training")
    print("  2. Head predictions may not align with iterative verification")


def debug_single_speculation(model, tokenizer, prompt, device, num_heads):
    """Debug a single speculation step to understand acceptance."""
    from nanochat.gemma_medusa.model import get_default_tree_choices, generate_tree_buffers

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)[0].tolist()
    print(f"\n{'='*60}")
    print("DEBUGGING SINGLE SPECULATION STEP")
    print(f"{'='*60}")
    print(f"Prompt: {prompt[:50]}...")
    print(f"Using {num_heads} heads")

    # Get tree buffers for num_heads
    tree_choices = get_default_tree_choices(num_heads, topk=10)
    print(f"Tree choices ({len(tree_choices)} candidates): {tree_choices[:5]}...")
    buffers = generate_tree_buffers(tree_choices, device, topk=10)

    with torch.no_grad():
        input_tensor = torch.tensor([input_ids], device=device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            main_logits, medusa_logits = model.forward(input_tensor, return_medusa=True, last_only=True)

        # main_logits: (B, 1, vocab)
        # medusa_logits: (num_heads, B, 1, vocab)

        base_token = main_logits[0, 0].argmax().item()
        print(f"\nBase model next token: {base_token} = '{tokenizer.decode([base_token])}'")

        # Show what each Medusa head predicts
        print(f"\nMedusa head predictions (top-5):")
        for h in range(num_heads):
            head_logits = medusa_logits[h, 0, 0]
            top5 = torch.topk(head_logits, 5)
            top5_tokens = [tokenizer.decode([t.item()]) for t in top5.indices]
            top5_match = ["✓" if t.item() == base_token else "" for t in top5.indices]
            print(f"  Head {h}: {list(zip(top5_tokens, top5_match))}")

        # Now simulate what _generate_candidates does
        print(f"\nGenerating candidates...")
        candidates, tree_candidates = model._generate_candidates(
            main_logits[:, 0, :],  # (B, vocab)
            medusa_logits[:, :, 0, :],  # (num_heads, B, vocab)
            buffers,
            topk=10,
            temperature=0.0
        )
        print(f"Candidates shape: {candidates.shape}")  # (num_candidates, max_depth)
        print(f"Tree candidates shape: {tree_candidates.shape}")

        # Show first few candidates
        print(f"\nFirst 5 candidates (decoded):")
        for i in range(min(5, candidates.shape[0])):
            cand = candidates[i].tolist()
            cand_str = [tokenizer.decode([t]) if t > 0 else '<pad>' for t in cand]
            print(f"  {i}: {cand} = {cand_str}")

        # Now do verification forward pass
        print(f"\nRunning verification forward pass...")
        tree_logits, retrieve_indices, valid_mask = model.forward_mtp(
            input_tensor, tree_candidates, buffers
        )
        print(f"Tree logits shape: {tree_logits.shape}")  # (tree_len, vocab)

        # Get predictions at each tree position
        tree_predictions = tree_logits.argmax(dim=-1)  # (tree_len,)
        print(f"\nTree predictions (first 10):")
        for i in range(min(10, tree_predictions.shape[0])):
            pred = tree_predictions[i].item()
            print(f"  Position {i}: {pred} = '{tokenizer.decode([pred])}'")

        # Now trace through acceptance logic
        print(f"\n{'='*60}")
        print("ACCEPTANCE LOGIC TRACE")
        print(f"{'='*60}")

        # candidates[:, 1:] are the speculated tokens
        # tree_predictions indexed by retrieve_indices give predictions for those positions
        safe_indices = retrieve_indices.clamp(min=0)
        candidate_predictions = tree_predictions[safe_indices]  # (num_candidates, max_depth)

        print(f"For each candidate path:")
        for i in range(min(5, candidates.shape[0])):
            cand = candidates[i]
            preds = candidate_predictions[i]
            mask = valid_mask[i]

            print(f"\nCandidate {i}:")
            print(f"  Speculated tokens: {cand[1:].tolist()}")
            print(f"  Model predictions: {preds[:-1].tolist()}")
            print(f"  Valid mask: {mask[1:].tolist()}")

            # Check matches
            matches = (cand[1:] == preds[:-1]) & mask[1:]
            print(f"  Matches: {matches.tolist()}")

            # Cumulative accept
            cumulative = torch.cumprod(matches.int(), dim=0)
            accept_len = cumulative.sum().item()
            print(f"  Accept length: {accept_len}")


def debug_acceptance(model, tokenizer, device, num_heads):
    """Debug the acceptance logic step by step."""
    from nanochat.gemma_medusa.model import get_default_tree_choices, generate_tree_buffers

    prompt = "The quick brown fox"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)[0].tolist()

    print(f"\n{'='*60}")
    print(f"DEBUGGING ACCEPTANCE LOGIC (num_heads={num_heads})")
    print(f"{'='*60}")

    # Get tree buffers
    tree_choices = get_default_tree_choices(num_heads, topk=10)
    print(f"Tree choices: {tree_choices}")
    buffers = generate_tree_buffers(tree_choices, device, topk=10)

    print(f"\ntree_indices: {buffers['tree_indices'].tolist()}")
    print(f"retrieve_indices shape: {buffers['retrieve_indices'].shape}")
    print(f"retrieve_indices:\n{buffers['retrieve_indices']}")

    with torch.no_grad():
        input_tensor = torch.tensor([input_ids], device=device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            main_logits, medusa_logits = model.forward(input_tensor, return_medusa=True, last_only=True)

        print(f"\nmain_logits shape: {main_logits.shape}")
        print(f"medusa_logits shape: {medusa_logits.shape}")

        base_token = main_logits[0, 0].argmax().item()
        print(f"\nBase model prediction: {base_token} = '{tokenizer.decode([base_token])}'")

        # Head 0's top-10
        head0_top10 = torch.topk(medusa_logits[0, 0, 0], 10).indices
        print(f"Head 0 top-10: {head0_top10.tolist()}")
        print(f"  = {[tokenizer.decode([t.item()]) for t in head0_top10]}")

        # Check if base_token is in head0_top10
        is_in_top10 = base_token in head0_top10.tolist()
        print(f"\nIs base_token in head0_top10? {is_in_top10}")

        # Generate candidates
        candidates, tree_candidates = model._generate_candidates(
            main_logits[:, 0, :],
            medusa_logits[:num_heads, :, 0, :],  # Only use num_heads heads
            buffers,
            topk=10,
            temperature=0.0
        )

        print(f"\ntree_candidates: {tree_candidates.tolist()}")
        print(f"  = {[tokenizer.decode([t.item()]) for t in tree_candidates]}")

        print(f"\ncandidates shape: {candidates.shape}")
        print(f"candidates:\n{candidates}")

        # Do verification
        tree_logits, retrieve_indices, valid_mask = model.forward_mtp(
            input_tensor, tree_candidates, buffers
        )

        print(f"\ntree_logits shape: {tree_logits.shape}")

        # Get predictions
        tree_predictions = tree_logits.argmax(dim=-1)
        print(f"tree_predictions: {tree_predictions.tolist()}")
        print(f"  = {[tokenizer.decode([t.item()]) for t in tree_predictions]}")

        # Map to candidates
        safe_indices = retrieve_indices.clamp(min=0)
        candidate_predictions = tree_predictions[safe_indices]
        print(f"\ncandidate_predictions:\n{candidate_predictions}")

        # Check matches
        print(f"\nMatching candidates[:, 1:] against candidate_predictions[:, :-1]:")
        print(f"candidates[:, 1:] = {candidates[:, 1:].tolist()}")
        print(f"candidate_predictions[:, :-1] = {candidate_predictions[:, :-1].tolist()}")

        matches = (candidates[:, 1:] == candidate_predictions[:, :-1])
        matches = matches & valid_mask[:, 1:]
        print(f"matches:\n{matches}")

        cumulative_matches = torch.cumprod(matches.int(), dim=1)
        accept_lengths = cumulative_matches.sum(dim=1)
        print(f"accept_lengths: {accept_lengths.tolist()}")

        best_candidate = int(accept_lengths.argmax().item())
        accept_length = int(accept_lengths[best_candidate].item())
        print(f"\nBest candidate: {best_candidate}, accept_length: {accept_length}")


if __name__ == '__main__':
    # First run basic accuracy test
    main()

    # Then run detailed acceptance debug
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--medusa-num-heads', type=int, default=4)
    parser.add_argument('--medusa-num-layers', type=int, default=2)
    parser.add_argument('--lora-rank', type=int, default=64)
    parser.add_argument('--lora-alpha', type=int, default=128)
    parser.add_argument('--attn-num-layers', type=int, default=0)
    parser.add_argument('--num-tokens', type=int, default=100)
    args = parser.parse_args()

    import os as os_module
    device = torch.device('cuda')
    model = GemmaMedusaModel(
        model_name='google/gemma-3-270m-it',
        medusa_num_heads=args.medusa_num_heads,
        medusa_num_layers=args.medusa_num_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        attn_num_layers=args.attn_num_layers,
        device=device,
        dtype=torch.bfloat16,
    )
    checkpoint_path = os_module.path.join(args.checkpoint, "final", "medusa_heads.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.medusa_heads.load_state_dict(checkpoint['medusa_heads'])
    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-270m-it')

    debug_acceptance(model, tokenizer, device, num_heads=1)
