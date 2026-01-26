#!/usr/bin/env python3
"""
Debug script to diagnose EAGLE training vs inference alignment.

This script traces through:
1. Training forward pass alignment (position i sees what?)
2. Inference topk_generate alignment (position i sees what?)
3. Verifies they match
"""

import torch
import torch.nn.functional as F
from nanochat.gemma_eagle import GemmaEagleConfig, GemmaEagleModel


def debug_alignment():
    """Debug the alignment between training and inference."""
    print("=" * 60)
    print("EAGLE Alignment Debug")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    config = GemmaEagleConfig(
        base_model_name="google/gemma-3-270m-it",
        freeze_base=True,
        total_tokens=15,
        draft_depth=2,
    )

    print(f"\nLoading model on {device}...")
    model = GemmaEagleModel(config, device=device, dtype=dtype)
    model.eval()

    # Create a simple test input: [BOS, A, B, C, D]
    # Using token IDs that we know exist in the vocab
    test_input_ids = torch.tensor([[2, 100, 200, 300, 400]], device=device)
    print(f"\nTest input_ids: {test_input_ids}")
    print(f"  Shape: {test_input_ids.shape}")

    # CRITICAL: Verify training target alignment
    print("\n" + "=" * 60)
    print("TRAINING TARGET ALIGNMENT CHECK")
    print("=" * 60)
    print("""
    In EAGLE training:
    - fused_hidden[i] = base model hidden state after processing token i
    - shifted_ids[i] = token at position i+1 (shifted left)
    - target_logits[i] = base model prediction at position i+1 (shifted left)

    The draft model sees (fused_hidden[i], embed(shifted_ids[i])) and should
    predict what the base model predicts at position i+1, which is target_logits[i].

    At inference, the draft model's first prediction (depth 0) should match
    what the base model predicts AFTER the sample token.
    """)

    # =========================================================================
    # PART 1: Trace training forward
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 1: Training Forward Alignment")
    print("=" * 60)

    with torch.no_grad():
        # Step 1: Get base model hidden states
        fused_hidden, target_logits = model.get_base_hidden_states(test_input_ids)
        print(f"\nfused_hidden shape: {fused_hidden.shape}")  # (1, 5, H)
        print(f"target_logits shape: {target_logits.shape}")  # (1, 5, V)

        # What does base predict at each position?
        base_preds = target_logits.argmax(dim=-1)
        print(f"Base predictions at each position: {base_preds}")
        # base_preds[i] = token the base predicts AFTER position i

        # Step 2: Apply shift_left (as in training)
        def shift_left(tensor):
            zero_pad = torch.zeros_like(tensor[:, -1:])
            return torch.cat((tensor[:, 1:], zero_pad), dim=1)

        shifted_ids = shift_left(test_input_ids)
        shifted_target_logits = shift_left(target_logits)

        print(f"\nAfter shift_left:")
        print(f"  original input_ids:    {test_input_ids}")
        print(f"  shifted_ids:           {shifted_ids}")
        print(f"  shifted_target argmax: {shifted_target_logits.argmax(dim=-1)}")

        # Step 3: Run draft forward (step 0)
        hidden_out, _ = model.forward_draft(
            hidden_states=fused_hidden,
            input_ids=shifted_ids,
        )

        # Get draft predictions
        draft_logits = model.lm_head(model.norm(hidden_out))
        draft_preds = draft_logits.argmax(dim=-1)

        print(f"\nDraft predictions: {draft_preds}")
        print(f"Target (shifted):  {shifted_target_logits.argmax(dim=-1)}")

        # Check alignment at each position
        print("\nPosition-by-position alignment (training step 0):")
        for i in range(test_input_ids.shape[1]):
            h_i = fused_hidden[0, i]
            e_i = shifted_ids[0, i].item()
            target_i = shifted_target_logits[0, i].argmax().item()
            draft_pred_i = draft_preds[0, i].item()

            print(f"  Position {i}:")
            print(f"    hidden[{i}] = context after token {i}")
            print(f"    input embed = embed({e_i})")
            print(f"    target (base predicts at pos {i+1}) = {target_i}")
            print(f"    draft predicts = {draft_pred_i}")
            print(f"    match: {target_i == draft_pred_i}")

    # =========================================================================
    # PART 2: Trace inference topk_generate
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 2: Inference topk_generate Alignment")
    print("=" * 60)

    model.reset_kv()

    with torch.no_grad():
        # Step 1: Get base hidden states for original input
        fused_hidden, base_logits = model.get_base_hidden_states(test_input_ids)

        # Sample token from last position
        sample_logits = base_logits[:, -1:]
        sample_token = sample_logits.argmax(dim=-1)
        print(f"\nSampled token from base (after position {test_input_ids.shape[1]-1}): {sample_token}")

        # Append to input_ids
        input_ids_with_sample = torch.cat([test_input_ids, sample_token], dim=1)
        print(f"input_ids_with_sample: {input_ids_with_sample}")
        print(f"  Shape: {input_ids_with_sample.shape}")  # (1, 6)

        # Step 2: Simulate what topk_generate does
        # From topk_generate lines 828-836:
        draft_input_ids = input_ids_with_sample[:, 1:]  # Remove BOS
        min_len = min(draft_input_ids.shape[1], fused_hidden.shape[1])
        draft_input_ids = draft_input_ids[:, :min_len]
        hidden_states = fused_hidden[:, :min_len]

        print(f"\nIn topk_generate setup:")
        print(f"  draft_input_ids: {draft_input_ids}")
        print(f"  draft_input_ids shape: {draft_input_ids.shape}")
        print(f"  hidden_states shape: {hidden_states.shape}")

        # Step 3: Run draft forward
        out_hidden, _ = model.forward_draft(
            hidden_states, draft_input_ids, use_cache=True
        )

        # Get predictions from last position
        last_hidden = out_hidden[:, -1:]
        last_logits = model.lm_head(model.norm(last_hidden))

        top5 = torch.topk(last_logits[0, 0], 5)
        print(f"\nDraft top-5 predictions from last position:")
        print(f"  tokens: {top5.indices.tolist()}")
        print(f"  scores: {top5.values.tolist()}")

        # What does base predict after sample?
        # We need to run base model on extended sequence to see what it predicts
        _, base_extended_logits = model.get_base_hidden_states(input_ids_with_sample)
        base_next_pred = base_extended_logits[0, -1].argmax().item()
        print(f"\nBase model predicts after sample: {base_next_pred}")
        print(f"Draft top-1 prediction: {top5.indices[0].item()}")
        print(f"Match: {top5.indices[0].item() == base_next_pred}")

    # =========================================================================
    # PART 3: Compare training vs inference alignment at same position
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 3: Direct Comparison")
    print("=" * 60)

    model.reset_kv()

    with torch.no_grad():
        # In training, at position 3 (the last non-padding position):
        # hidden = fused_hidden[3] (after BOS, 100, 200, 300)
        # input_id = shifted_ids[3] = 400
        # target = shifted_target_logits[3] = what base predicts at position 4

        # In inference, at position 3 (last position in draft_input_ids):
        # hidden = fused_hidden[3] (same as training - after BOS, 100, 200, 300)
        # input_id = draft_input_ids[3] = ...

        # Let's verify the draft_input_ids
        print(f"\nVerifying position 3 alignment:")
        print(f"  Training input_id at pos 3: {shifted_ids[0, 3].item()}")
        print(f"  Inference input_id at pos 3: {draft_input_ids[0, 3].item()}")

        # These should be different! Training has token 400, inference has sample
        # This is expected because:
        # - Training: predicting future tokens given the actual sequence
        # - Inference: predicting future tokens given the sampled token

        # But wait - in training step 0, we're also using the actual tokens
        # Let me trace through more carefully...

        # Actually, the key insight is:
        # Training: at position i, we predict what base predicts at position i+1
        # Inference: at position i, we also predict what base would predict at position i+1
        #
        # The difference is that in inference, we're auto-regressively extending,
        # so position T-1 in inference corresponds to a NEW position (after the sample)
        # that wasn't in training

        print("\n" + "-" * 40)
        print("Key insight: Inference extends beyond training context")
        print("-" * 40)
        print(f"Training seq length: {test_input_ids.shape[1]}")
        print(f"Inference seq length (with sample): {input_ids_with_sample.shape[1]}")
        print("\nAt the last position in inference:")
        print(f"  hidden = context after {test_input_ids.shape[1]-1} original tokens")
        print(f"  input = embed(sample) = embed({sample_token[0,0].item()})")
        print("  This predicts what comes AFTER the sample")

        # The model is asked to predict the token after `sample`
        # In training, it learned to predict the next token given context + current token
        # So this should be correct!


def debug_with_checkpoint(checkpoint_path: str):
    """Debug with a trained checkpoint."""
    print("=" * 60)
    print(f"EAGLE Alignment Debug (with checkpoint)")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    config = GemmaEagleConfig(
        base_model_name="google/gemma-3-270m-it",
        freeze_base=True,
    )

    print(f"\nLoading model on {device}...")
    model = GemmaEagleModel(config, device=device, dtype=dtype)

    # Load checkpoint
    print(f"Loading checkpoint...")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_draft_state_dict(state)
    model.eval()

    # Run the same debug
    debug_single_sequence(model, device)


def debug_single_sequence(model, device):
    """Debug a single sequence through training and inference paths."""

    # Use a longer sequence
    test_input_ids = torch.tensor([[2, 100, 200, 300, 400, 500, 600]], device=device)
    print(f"\nTest input_ids: {test_input_ids}")

    with torch.no_grad():
        # Get base outputs
        fused_hidden, base_logits = model.get_base_hidden_states(test_input_ids)
        base_preds = base_logits.argmax(dim=-1)
        print(f"Base predictions: {base_preds}")

        # Training forward (step 0)
        def shift_left(tensor):
            zero_pad = torch.zeros_like(tensor[:, -1:])
            return torch.cat((tensor[:, 1:], zero_pad), dim=1)

        shifted_ids = shift_left(test_input_ids)
        shifted_base_logits = shift_left(base_logits)

        hidden_out, _ = model.forward_draft(fused_hidden, shifted_ids)
        draft_logits = model.lm_head(model.norm(hidden_out))
        draft_preds = draft_logits.argmax(dim=-1)

        targets = shifted_base_logits.argmax(dim=-1)

        print(f"\nTraining alignment (step 0):")
        print(f"  shifted_ids: {shifted_ids}")
        print(f"  targets:     {targets}")
        print(f"  draft_preds: {draft_preds}")

        matches = (draft_preds == targets).float()
        acc = matches.mean().item()
        print(f"  Accuracy: {acc:.2%}")

        # Check position by position
        print("\n  Per-position:")
        for i in range(test_input_ids.shape[1]):
            match = "✓" if draft_preds[0, i] == targets[0, i] else "✗"
            print(f"    pos {i}: input={shifted_ids[0,i].item():5d}, "
                  f"target={targets[0,i].item():5d}, "
                  f"pred={draft_preds[0,i].item():5d} {match}")


def debug_inference_verification(checkpoint_path: str = None):
    """
    Debug that inference verification matches training alignment.

    This test verifies that:
    1. The draft model's predictions are being correctly compared against
       what the base model would predict given the same context
    2. The verification step includes full context (not just tree candidates)
    """
    print("=" * 60)
    print("EAGLE Inference Verification Debug")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    config = GemmaEagleConfig(
        base_model_name="google/gemma-3-270m-it",
        freeze_base=True,
        total_tokens=15,
        draft_depth=2,
    )

    print(f"\nLoading model on {device}...")
    model = GemmaEagleModel(config, device=device, dtype=dtype)

    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_draft_state_dict(state)

    model.eval()

    # Test sequence
    test_input_ids = torch.tensor([[2, 100, 200, 300, 400]], device=device)
    print(f"\nTest input_ids: {test_input_ids}")

    with torch.no_grad():
        # Get base hidden states and sample a token
        fused_hidden, base_logits = model.get_base_hidden_states(test_input_ids)
        sample_token = base_logits[0, -1].argmax().item()
        print(f"Sample token (from base): {sample_token}")

        # Run topk_generate to get draft predictions
        model.reset_kv()
        input_ids_with_sample = torch.cat([
            test_input_ids,
            torch.tensor([[sample_token]], device=device)
        ], dim=1)

        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.topk_generate(
            fused_hidden, input_ids_with_sample
        )

        print(f"\nDraft tokens (tree structure): {draft_tokens}")
        print(f"Tree position IDs: {tree_position_ids}")

        # Build cart_candidates (candidate paths)
        tree_candidates_ext = torch.cat([
            draft_tokens.squeeze(0),
            torch.tensor([-1], device=device)
        ], dim=0)
        cart_candidates = tree_candidates_ext[retrieve_indices]

        print(f"\nCart candidates (paths): {cart_candidates}")

        # Verify each path by running base model on full context + path
        num_paths, max_depth = cart_candidates.shape
        print(f"\nVerifying {num_paths} paths (max_depth={max_depth}):")

        total_matches = 0
        total_positions = 0

        for path_idx in range(num_paths):
            path = cart_candidates[path_idx]
            path_suffix = path[1:].clone()  # Exclude root
            path_suffix[path_suffix < 0] = 0  # Replace -1 with pad

            # Build full sequence: prefix + suffix
            full_seq = torch.cat([
                test_input_ids[0],  # Original prefix (without sample yet)
                torch.tensor([sample_token], device=device),  # Sample
                path_suffix[path_suffix > 0]  # Non-padding suffix tokens
            ])

            print(f"\n  Path {path_idx}: {path.tolist()}")
            print(f"    Full sequence: {full_seq.tolist()}")

            # Run base model on this path
            _, path_base_logits = model.get_base_hidden_states(full_seq.unsqueeze(0))

            # Check what base predicts at each position of the suffix
            prefix_with_sample_len = test_input_ids.shape[1] + 1  # Original prefix + sample

            for depth in range(max_depth - 1):  # -1 because first position is root
                if path[depth + 1] < 0:  # Skip padding
                    continue

                # Position in full_seq where we predict path[depth+1]
                pos = prefix_with_sample_len + depth - 1

                if pos >= path_base_logits.shape[1]:
                    continue

                base_pred = path_base_logits[0, pos].argmax().item()
                draft_pred = path[depth + 1].item()

                match = "✓" if base_pred == draft_pred else "✗"
                total_positions += 1
                if base_pred == draft_pred:
                    total_matches += 1

                print(f"    Depth {depth}: draft={draft_pred:5d}, base={base_pred:5d} {match}")

        print(f"\n" + "=" * 60)
        print(f"Summary:")
        print(f"  Total positions verified: {total_positions}")
        print(f"  Matches: {total_matches}")
        print(f"  Accuracy: {total_matches / max(1, total_positions):.2%}")
        print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to EAGLE checkpoint")
    parser.add_argument("--verify", action="store_true",
                        help="Run inference verification debug")
    args = parser.parse_args()

    if args.verify:
        debug_inference_verification(args.checkpoint)
    elif args.checkpoint:
        debug_with_checkpoint(args.checkpoint)
    else:
        debug_alignment()
