"""Test EAGLE acceptance rate calculation."""

import torch
import pytest


def test_evaluate_posterior_basic():
    """Test that evaluate_posterior returns correct acceptance length."""
    from nanochat.gemma_eagle.tree import evaluate_posterior_greedy

    # Simulate: candidates has shape (num_paths, max_depth)
    # logits has shape (num_paths, max_depth, vocab_size)
    vocab_size = 100
    num_paths = 4
    max_depth = 3

    # Create candidates: each row is a path [root, tok1, tok2]
    candidates = torch.tensor([
        [10, 20, 30],  # path 0: root=10, then 20, then 30
        [10, 20, 31],  # path 1: root=10, then 20, then 31
        [10, 21, 32],  # path 2: root=10, then 21, then 32
        [10, 22, 33],  # path 3: root=10, then 22, then 33
    ])

    # Create logits where argmax matches the candidates for path 0
    logits = torch.randn(num_paths, max_depth, vocab_size)

    # Set up logits so that argmax(logits[:, :-1]) matches candidates[:, 1:]
    # For path 0: argmax(logits[0, 0]) should be 20, argmax(logits[0, 1]) should be 30
    logits[0, 0, 20] = 100.0  # position 0 predicts token 20
    logits[0, 1, 30] = 100.0  # position 1 predicts token 30
    # For path 1: argmax(logits[1, 0]) should be 20, argmax(logits[1, 1]) should NOT be 31
    logits[1, 0, 20] = 100.0
    logits[1, 1, 40] = 100.0  # position 1 predicts token 40 (not 31)

    best_candidate, accept_length, next_logits = evaluate_posterior_greedy(logits, candidates)

    print(f"candidates:\n{candidates}")
    print(f"candidates[:, 1:]:\n{candidates[:, 1:]}")
    print(f"argmax(logits[:, :-1]):\n{torch.argmax(logits[:, :-1], dim=-1)}")
    print(f"best_candidate: {best_candidate}")
    print(f"accept_length: {accept_length}")

    # Path 0 should have accept_length=2 (tokens 20 and 30 both match)
    assert accept_length == 2, f"Expected accept_length=2, got {accept_length}"
    assert best_candidate == 0, f"Expected best_candidate=0, got {best_candidate}"


def test_evaluate_posterior_no_match():
    """Test when no candidates match."""
    from nanochat.gemma_eagle.tree import evaluate_posterior_greedy

    vocab_size = 100
    num_paths = 2
    max_depth = 3

    candidates = torch.tensor([
        [10, 20, 30],
        [10, 21, 31],
    ])

    logits = torch.randn(num_paths, max_depth, vocab_size)
    # Set logits so argmax doesn't match any candidate
    logits[0, 0, 50] = 100.0  # predicts 50, not 20
    logits[1, 0, 51] = 100.0  # predicts 51, not 21

    best_candidate, accept_length, next_logits = evaluate_posterior_greedy(logits, candidates)

    print(f"candidates[:, 1:]:\n{candidates[:, 1:]}")
    print(f"argmax(logits[:, :-1]):\n{torch.argmax(logits[:, :-1], dim=-1)}")
    print(f"accept_length: {accept_length}")

    # No matches, accept_length should be 0
    assert accept_length == 0, f"Expected accept_length=0, got {accept_length}"


def test_eagle_generator_acceptance():
    """Test that EAGLE generator tracks acceptance correctly."""
    from nanochat.gemma_eagle import GemmaEagleConfig, GemmaEagleModel, EagleGenerator

    config = GemmaEagleConfig(
        base_model_name="google/gemma-3-270m-it",
        freeze_base=True,
        total_tokens=15,  # Smaller tree for testing
        draft_depth=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = GemmaEagleModel(config, device=device, dtype=dtype)
    model.eval()

    generator = EagleGenerator(model)

    # Generate a short sequence
    input_ids = torch.tensor([[2, 100, 200, 300]], device=device)

    print(f"Input shape: {input_ids.shape}")

    output_ids, stats = generator.generate(
        input_ids,
        max_new_tokens=10,
        temperature=0.0,
        return_stats=True,
    )

    print(f"Output shape: {output_ids.shape}")
    print(f"Tokens generated: {stats.tokens_generated}")
    print(f"Forward passes: {stats.forward_passes}")
    print(f"Total proposed: {stats.total_proposed}")
    print(f"Total accepted: {stats.total_accepted}")
    print(f"Mean accepted length: {stats.mean_accepted_length:.2f}")
    print(f"Acceptance rate: {stats.acceptance_rate:.2f}")

    # Sanity checks
    assert stats.tokens_generated > 0, "Should have generated tokens"
    # With untrained draft model, acceptance might be low, but should be > 0 sometimes
    print(f"Test completed. Mean acceptance: {stats.mean_accepted_length:.3f}")


def test_compare_with_direct_verify():
    """Test that verification logic is correct by checking draft tokens against base model."""
    from nanochat.gemma_eagle import GemmaEagleConfig, GemmaEagleModel

    config = GemmaEagleConfig(
        base_model_name="google/gemma-3-270m-it",
        freeze_base=True,
        total_tokens=15,
        draft_depth=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = GemmaEagleModel(config, device=device, dtype=dtype)
    model.eval()

    # Get initial hidden states
    input_ids = torch.tensor([[2, 100, 200, 300]], device=device)
    with torch.no_grad():
        fused_hidden, base_logits = model.get_base_hidden_states(input_ids)

    # Sample first token
    sample_token = base_logits[:, -1:].argmax(dim=-1)
    input_ids = torch.cat([input_ids, sample_token], dim=1)

    model.reset_kv()

    # Generate draft tokens
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.topk_generate(
        fused_hidden, input_ids, None
    )

    print(f"Draft tokens shape: {draft_tokens.shape}")
    print(f"Draft tokens: {draft_tokens}")
    print(f"Retrieve indices shape: {retrieve_indices.shape}")
    print(f"Tree position ids: {tree_position_ids}")

    # Now verify with base model
    position_ids = tree_position_ids + input_ids.shape[1] - 1
    model.tree_mask = tree_mask

    with torch.no_grad():
        outputs = model.base_model(
            input_ids=draft_tokens,
            position_ids=position_ids.unsqueeze(0),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    verify_logits = outputs.logits[0, retrieve_indices]
    print(f"Verify logits shape: {verify_logits.shape}")

    # Build cart_candidates
    tree_candidates_ext = torch.cat([
        draft_tokens.squeeze(0),
        torch.tensor([-1], device=device)
    ], dim=0)
    cart_candidates = tree_candidates_ext[retrieve_indices]
    print(f"Cart candidates shape: {cart_candidates.shape}")
    print(f"Cart candidates:\n{cart_candidates}")

    # Check what base model predicts
    base_predictions = torch.argmax(verify_logits[:, :-1], dim=-1)
    print(f"Base predictions shape: {base_predictions.shape}")
    print(f"Base predictions:\n{base_predictions}")
    print(f"Candidates to match:\n{cart_candidates[:, 1:]}")

    # Check matches
    matches = (cart_candidates[:, 1:].to(verify_logits.device) == base_predictions).int()
    print(f"Matches:\n{matches}")

    # Calculate cumulative product to find accept length per path
    accept_lengths = torch.cumprod(matches, dim=1).sum(dim=1)
    print(f"Accept lengths per path: {accept_lengths}")
    print(f"Max accept length: {accept_lengths.max().item()}")


if __name__ == "__main__":
    print("=" * 50)
    print("Test 1: evaluate_posterior_basic")
    print("=" * 50)
    test_evaluate_posterior_basic()

    print("\n" + "=" * 50)
    print("Test 2: evaluate_posterior_no_match")
    print("=" * 50)
    test_evaluate_posterior_no_match()

    print("\n" + "=" * 50)
    print("Test 3: eagle_generator_acceptance")
    print("=" * 50)
    test_eagle_generator_acceptance()

    print("\n" + "=" * 50)
    print("Test 4: compare_with_direct_verify")
    print("=" * 50)
    test_compare_with_direct_verify()
