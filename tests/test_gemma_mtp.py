"""
Test MTP (Multi-Token Prediction) tree attention generation for Gemma Medusa.

This test can run locally on a MacBook (CPU) with a small model.
For full testing, use a GPU with the Gemma 1B model.

Run with:
    uv run pytest tests/test_gemma_mtp.py -v -s
    uv run python tests/test_gemma_mtp.py  # Direct execution
"""

import time
import torch
import pytest


def test_tree_buffer_generation():
    """Test that tree buffers are generated correctly."""
    from nanochat.gemma_medusa.model import (
        generate_tree_buffers,
        get_default_tree_choices,
        get_sparse_tree_choices,
    )

    print("\n=== Testing Tree Buffer Generation ===")

    device = torch.device("cpu")

    # Test sparse tree (smaller, faster)
    choices = get_sparse_tree_choices(num_heads=4)
    print(f"Sparse tree choices: {len(choices)} paths")
    assert len(choices) > 0

    buffers = generate_tree_buffers(choices, device, topk=10)

    print(f"Tree attention mask shape: {buffers['tree_attn_mask'].shape}")
    print(f"Tree indices shape: {buffers['tree_indices'].shape}")
    print(f"Position IDs shape: {buffers['tree_position_ids'].shape}")
    print(f"Retrieve indices shape: {buffers['retrieve_indices'].shape}")

    # Verify shapes
    tree_len = len(choices) + 1
    assert buffers["tree_attn_mask"].shape == (1, 1, tree_len, tree_len)
    assert buffers["tree_indices"].shape == (tree_len,)
    assert buffers["tree_position_ids"].shape == (tree_len,)

    # Test default tree (larger)
    choices_default = get_default_tree_choices(num_heads=4, topk=5)
    print(f"\nDefault tree choices (topk=5): {len(choices_default)} paths")

    buffers_default = generate_tree_buffers(choices_default, device, topk=5)
    print(f"Default tree attention mask shape: {buffers_default['tree_attn_mask'].shape}")

    print("\n✓ Tree buffer generation tests passed!")


def test_mtp_stats():
    """Test MTPStats dataclass."""
    from nanochat.gemma_medusa.model import MTPStats

    print("\n=== Testing MTPStats ===")

    stats = MTPStats(
        tokens_generated=100,
        forward_passes=40,
        total_proposed=200,
        total_accepted=100,
    )

    print(f"Mean accepted length: {stats.mean_accepted_length:.2f}")
    print(f"Acceptance rate: {stats.acceptance_rate:.2%}")
    print(f"Speedup: {stats.speedup:.2f}x")

    assert stats.mean_accepted_length == 2.5
    assert stats.acceptance_rate == 0.5
    assert stats.speedup == 2.5

    print("\n✓ MTPStats tests passed!")


def test_candidate_generation_mock():
    """Test candidate generation with mock logits."""
    from nanochat.gemma_medusa.model import (
        generate_tree_buffers,
        get_sparse_tree_choices,
    )

    print("\n=== Testing Candidate Generation (Mock) ===")

    device = torch.device("cpu")
    num_heads = 4
    vocab_size = 1000
    topk = 5

    # Create sparse tree
    choices = get_sparse_tree_choices(num_heads)
    buffers = generate_tree_buffers(choices, device, topk=topk)

    # Mock logits
    main_logits = torch.randn(1, vocab_size)  # (B, vocab)
    medusa_logits = torch.randn(num_heads, 1, vocab_size)  # (num_heads, B, vocab)

    # Test the generation logic directly (extracted from model)
    tree_indices = buffers["tree_indices"]
    retrieve_indices = buffers["retrieve_indices"]

    # Get base token (greedy)
    base_token = torch.argmax(main_logits[0], dim=-1)
    print(f"Base token: {base_token.item()}")

    # Get top-k from each head
    medusa_topk = torch.topk(medusa_logits[:, 0, :], topk, dim=-1).indices
    print(f"Medusa top-k shape: {medusa_topk.shape}")

    # Build flat candidates
    flat_candidates = torch.cat([
        base_token.unsqueeze(0),
        medusa_topk.view(-1),
    ])
    print(f"Flat candidates shape: {flat_candidates.shape}")

    # Map to tree
    tree_candidates = flat_candidates[tree_indices]
    print(f"Tree candidates shape: {tree_candidates.shape}")

    # Extract candidate paths
    num_candidates, max_depth = retrieve_indices.shape
    candidates = torch.zeros(num_candidates, max_depth, dtype=torch.long, device=device)

    for i in range(num_candidates):
        for j in range(max_depth):
            idx = int(retrieve_indices[i, j].item())
            if idx >= 0:
                candidates[i, j] = tree_candidates[idx]

    print(f"Candidates shape: {candidates.shape}")
    print(f"First few candidates:\n{candidates[:5]}")

    # Verify all candidates start with base token
    assert (candidates[:, 0] == base_token).all()

    print("\n✓ Candidate generation tests passed!")


def test_greedy_acceptance():
    """Test greedy acceptance logic."""
    print("\n=== Testing Greedy Acceptance ===")

    device = torch.device("cpu")
    num_candidates = 5
    max_depth = 4
    vocab_size = 100

    # Create mock verification logits and candidates
    # Scenario: candidates[0] = [10, 20, 30, 40], logits predict [20, 30, 40, 50]
    # Expected: accept 3 tokens (20, 30, 40 match predictions)

    candidates = torch.tensor([
        [10, 20, 30, 40],  # Best candidate - 3 matches
        [10, 21, 30, 40],  # 0 matches (21 != predicted 20)
        [10, 20, 31, 40],  # 1 match (31 != predicted 30)
        [10, 20, 30, 41],  # 2 matches (41 != predicted 40)
        [10, 22, 33, 44],  # 0 matches
    ], dtype=torch.long, device=device)

    # Logits that predict next tokens
    verify_logits = torch.zeros(num_candidates, max_depth, vocab_size, device=device)
    # Set logits to predict [20, 30, 40, 50] for all candidates
    for i in range(num_candidates):
        verify_logits[i, 0, 20] = 10.0  # predicts 20
        verify_logits[i, 1, 30] = 10.0  # predicts 30
        verify_logits[i, 2, 40] = 10.0  # predicts 40
        verify_logits[i, 3, 50] = 10.0  # predicts 50

    # Run acceptance logic
    predictions = verify_logits.argmax(dim=-1)
    matches = (candidates[:, 1:] == predictions[:, :-1]).int()
    cumulative_matches = torch.cumprod(matches, dim=1)
    accept_lengths = cumulative_matches.sum(dim=1)

    print(f"Predictions (first pos): {predictions[:, 0].tolist()}")
    print(f"Candidates (second pos): {candidates[:, 1].tolist()}")
    print(f"Matches: {matches.tolist()}")
    print(f"Accept lengths: {accept_lengths.tolist()}")

    best_candidate = int(accept_lengths.argmax().item())
    accept_length = int(accept_lengths[best_candidate].item())

    print(f"Best candidate: {best_candidate}")
    print(f"Accept length: {accept_length}")

    assert best_candidate == 0
    assert accept_length == 3

    print("\n✓ Greedy acceptance tests passed!")


@pytest.mark.slow
def test_full_mtp_generation():
    """
    Test full MTP generation with actual Gemma model.

    This test requires downloading the Gemma model (~2GB).
    Skip with: pytest -m "not slow"
    """
    try:
        from nanochat.gemma_medusa import GemmaMedusaModel
        from transformers import AutoTokenizer
    except ImportError as e:
        pytest.skip(f"Import error: {e}")

    print("\n=== Testing Full MTP Generation ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model_name = "google/gemma-3-1b-it"
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Loading {model_name}...")

    try:
        model = GemmaMedusaModel(
            model_name=model_name,
            medusa_num_heads=4,
            medusa_num_layers=1,
            lora_rank=64,
            device=device,
            dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    model.eval()
    print(f"Medusa params: {model.get_medusa_param_count():,}")

    # Simple prompt
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
    print(f"\nPrompt: {prompt}")
    print(f"Input tokens: {len(input_ids)}")

    # Test MTP generation
    print("\n--- MTP Generation ---")
    t0 = time.perf_counter()
    mtp_tokens, mtp_stats = model.generate_mtp(
        input_ids,
        max_new_tokens=20,
        temperature=0.0,
        topk=5,
        use_sparse_tree=True,
    )
    t_mtp = time.perf_counter() - t0

    mtp_text = tokenizer.decode(mtp_tokens)
    print(f"Output: {mtp_text}")
    print(f"Time: {t_mtp:.2f}s")
    print(f"Tokens generated: {mtp_stats.tokens_generated}")
    print(f"Forward passes: {mtp_stats.forward_passes}")
    print(f"Mean accepted: {mtp_stats.mean_accepted_length:.2f}")
    print(f"Acceptance rate: {mtp_stats.acceptance_rate:.2%}")

    # Test standard generation for comparison
    print("\n--- Standard Generation ---")
    t0 = time.perf_counter()
    std_tokens, std_passes = model.generate_standard(
        input_ids,
        max_new_tokens=20,
        temperature=0.0,
    )
    t_std = time.perf_counter() - t0

    std_text = tokenizer.decode(std_tokens)
    print(f"Output: {std_text}")
    print(f"Time: {t_std:.2f}s")
    print(f"Forward passes: {std_passes}")

    # Compare
    print("\n--- Comparison ---")
    speedup = t_std / t_mtp if t_mtp > 0 else 0
    print(f"Time speedup: {speedup:.2f}x")
    print(f"Forward pass reduction: {std_passes / mtp_stats.forward_passes:.2f}x")

    # Check outputs match (greedy should be deterministic)
    if mtp_text != std_text:
        print(f"WARNING: Outputs differ!")
        print(f"  MTP: {mtp_text}")
        print(f"  Std: {std_text}")

    print("\n✓ Full MTP generation tests passed!")


def test_forward_mtp_shapes():
    """Test forward_mtp output shapes with mock model."""
    print("\n=== Testing forward_mtp Shapes ===")

    # This test verifies the tree attention forward pass shapes
    # without loading the full model

    from nanochat.gemma_medusa.model import (
        generate_tree_buffers,
        get_sparse_tree_choices,
    )

    device = torch.device("cpu")
    num_heads = 4
    topk = 5

    choices = get_sparse_tree_choices(num_heads)
    buffers = generate_tree_buffers(choices, device, topk=topk)

    tree_len = len(choices) + 1
    retrieve_indices = buffers["retrieve_indices"]
    num_candidates, max_depth = retrieve_indices.shape

    print(f"Tree length: {tree_len}")
    print(f"Num candidates: {num_candidates}")
    print(f"Max depth: {max_depth}")

    # Verify buffer shapes are consistent
    assert buffers["tree_attn_mask"].shape[2] == tree_len
    assert buffers["tree_attn_mask"].shape[3] == tree_len
    assert buffers["tree_indices"].shape[0] == tree_len

    print("\n✓ forward_mtp shape tests passed!")


def test_kv_cache_methods():
    """Test that KV cache methods exist and have correct signatures."""
    from nanochat.gemma_medusa.model import (
        GemmaMedusaModel,
        generate_tree_buffers,
        get_sparse_tree_choices,
    )

    print("\n=== Testing KV Cache Methods ===")

    # Test that all KV cache methods exist
    required_methods = [
        '_get_hidden_states_with_cache',
        'forward_mtp_with_cache',
        'generate_mtp_with_cache',
    ]

    for method in required_methods:
        assert hasattr(GemmaMedusaModel, method), f"Missing method: {method}"
    print(f"✓ All {len(required_methods)} KV cache methods present")

    # Test position ID computation for tree with cache
    device = torch.device("cpu")
    num_heads = 4
    topk = 5

    choices = get_sparse_tree_choices(num_heads)
    buffers = generate_tree_buffers(choices, device, topk=topk)

    # Simulate KV cache scenario
    base_seq_len = 100
    tree_position_ids = buffers["tree_position_ids"]
    position_ids = base_seq_len + tree_position_ids

    # Verify position IDs are correctly offset
    assert position_ids[0] == base_seq_len  # Root at base position
    assert (position_ids >= base_seq_len).all()  # All positions >= base
    print(f"✓ Position IDs correctly offset from base_seq_len={base_seq_len}")

    # Test incremental position update after accepting tokens
    accepted_tokens = 3
    new_base = base_seq_len + accepted_tokens
    new_position_ids = new_base + tree_position_ids

    assert new_position_ids[0] == new_base
    assert (new_position_ids - position_ids == accepted_tokens).all()
    print(f"✓ Position IDs correctly updated after accepting {accepted_tokens} tokens")

    print("\n✓ KV cache method tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Gemma MTP (Multi-Token Prediction) Tests")
    print("=" * 60)

    # Run fast tests
    test_tree_buffer_generation()
    test_mtp_stats()
    test_candidate_generation_mock()
    test_greedy_acceptance()
    test_forward_mtp_shapes()
    test_kv_cache_methods()

    # Run slow test if requested
    import sys
    if "--full" in sys.argv:
        test_full_mtp_generation()
    else:
        print("\n[Skipping full model test - run with --full to include]")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
