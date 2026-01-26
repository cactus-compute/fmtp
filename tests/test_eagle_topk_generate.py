"""Test EAGLE topk_generate tensor shape handling."""

import torch
import pytest


def test_topk_generate_shapes():
    """Test that topk_generate handles tensor shapes correctly."""
    from nanochat.gemma_eagle import GemmaEagleConfig, GemmaEagleModel

    # Use a small model for testing
    config = GemmaEagleConfig(
        base_model_name="google/gemma-3-270m-it",
        freeze_base=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = GemmaEagleModel(config, device=device, dtype=dtype)
    model.eval()

    # Simulate the generation flow
    # Original input: [BOS, tok1, tok2, tok3] = 4 tokens
    input_ids = torch.tensor([[2, 100, 200, 300]], device=device)  # BOS=2

    # Get fused hidden states (shape: B, T, H)
    with torch.no_grad():
        fused_hidden, base_logits = model.get_base_hidden_states(input_ids)

    print(f"Original input_ids shape: {input_ids.shape}")  # (1, 4)
    print(f"fused_hidden shape: {fused_hidden.shape}")  # (1, 4, H)

    # Sample first token (greedy)
    sample_token = base_logits[:, -1:, :].argmax(dim=-1)
    print(f"sample_token shape: {sample_token.shape}")  # (1, 1)

    # Append sampled token to input_ids
    input_ids_with_sample = torch.cat([input_ids, sample_token], dim=1)
    print(f"input_ids_with_sample shape: {input_ids_with_sample.shape}")  # (1, 5)

    # Reset KV cache
    model.reset_kv()

    # Now call topk_generate
    # It expects: hidden_states (B, T, H), input_ids (B, T+1)
    # Where T is original length (4), so input_ids should be (B, 5)
    print(f"\nCalling topk_generate with:")
    print(f"  hidden_states: {fused_hidden.shape}")
    print(f"  input_ids: {input_ids_with_sample.shape}")

    # This should work without error
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.topk_generate(
        fused_hidden, input_ids_with_sample, None
    )

    print(f"\nOutput shapes:")
    print(f"  draft_tokens: {draft_tokens.shape}")
    print(f"  retrieve_indices: {retrieve_indices.shape}")
    print(f"  tree_mask: {tree_mask.shape}")
    print(f"  tree_position_ids: {tree_position_ids.shape}")

    # Verify outputs are valid
    assert draft_tokens.dim() == 2
    assert draft_tokens.shape[0] == 1


def test_eagle_generator_multiple_iterations():
    """Test that EagleGenerator works for multiple generation iterations."""
    from nanochat.gemma_eagle import GemmaEagleConfig, GemmaEagleModel, EagleGenerator

    config = GemmaEagleConfig(
        base_model_name="google/gemma-3-270m-it",
        freeze_base=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = GemmaEagleModel(config, device=device, dtype=dtype)
    model.eval()

    generator = EagleGenerator(model)

    # Generate a short sequence
    input_ids = torch.tensor([[2, 100, 200, 300]], device=device)  # BOS=2

    print(f"Input shape: {input_ids.shape}")

    output_ids, stats = generator.generate(
        input_ids,
        max_new_tokens=20,
        temperature=0.0,
        return_stats=True,
    )

    print(f"Output shape: {output_ids.shape}")
    print(f"Tokens generated: {stats.tokens_generated}")
    print(f"Forward passes: {stats.forward_passes}")
    print(f"Mean accepted length: {stats.mean_accepted_length:.2f}")

    assert output_ids.shape[1] > input_ids.shape[1], "Should have generated some tokens"
    print("Test passed!")


if __name__ == "__main__":
    test_topk_generate_shapes()
    print("\n" + "="*50 + "\n")
    test_eagle_generator_multiple_iterations()
