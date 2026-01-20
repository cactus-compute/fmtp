"""
Speed comparison test for Gemma Medusa model.

Tests:
1. MedusaLoRAHead in isolation
2. GemmaMedusaModel (requires downloading Gemma model)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_medusa_head_isolated():
    """Test MedusaLoRAHead without loading full Gemma model."""
    from nanochat.gemma_medusa.heads import MedusaLoRAHead, MedusaResBlock

    print("\n=== Testing Medusa Heads (Isolated) ===")

    # Parameters matching Gemma 270M
    hidden_size = 1024
    vocab_size = 262144  # Gemma vocab
    num_heads = 4
    num_layers = 1
    lora_rank = 64
    B, T = 2, 128

    # Create mock hidden states
    hidden_states = torch.randn(B, T, hidden_size)

    # Create Medusa heads
    heads = nn.ModuleList([
        MedusaLoRAHead(hidden_size, vocab_size, num_layers, lora_rank)
        for _ in range(num_heads)
    ])

    total_params = sum(p.numel() for p in heads.parameters())
    print(f"Medusa params: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"  Per head: {total_params // num_heads:,}")

    # Simulate lm_head
    lm_weight = torch.randn(vocab_size, hidden_size)

    # Benchmark
    print(f"\nBenchmarking (B={B}, T={T})...")

    num_warmup = 3
    num_iters = 10

    # Warmup
    for _ in range(num_warmup):
        base_logits = F.linear(hidden_states, lm_weight)
        for head in heads:
            _ = head(hidden_states)

    # Time base lm_head only
    t0 = time.perf_counter()
    for _ in range(num_iters):
        base_logits = F.linear(hidden_states, lm_weight)
    t_base = (time.perf_counter() - t0) / num_iters * 1000

    # Time base + medusa heads
    t0 = time.perf_counter()
    for _ in range(num_iters):
        base_logits = F.linear(hidden_states, lm_weight)
        medusa_logits = torch.stack([
            base_logits + head(hidden_states)
            for head in heads
        ], dim=0)
    t_with_medusa = (time.perf_counter() - t0) / num_iters * 1000

    overhead = (t_with_medusa / t_base - 1) * 100

    print(f"\nResults:")
    print(f"  Base lm_head only:  {t_base:.2f} ms")
    print(f"  Base + {num_heads} Medusa:   {t_with_medusa:.2f} ms")
    print(f"  Overhead:           {overhead:.1f}%")

    # Verify shapes
    assert base_logits.shape == (B, T, vocab_size)
    assert medusa_logits.shape == (num_heads, B, T, vocab_size)

    print("\n✓ Shape tests passed!")


def test_gradient_flow():
    """Test that gradients flow through Medusa heads."""
    from nanochat.gemma_medusa.heads import MedusaLoRAHead

    print("\n=== Testing Gradient Flow ===")

    hidden_size = 256
    vocab_size = 1000
    B, T = 2, 16

    hidden_states = torch.randn(B, T, hidden_size)
    head = MedusaLoRAHead(hidden_size, vocab_size, num_layers=1, lora_rank=32)

    # Note: With zero-init of lora_B, gradients won't flow to lora_A on first pass.
    # This is expected - lora_B gets updated first, then gradients flow to lora_A.
    # Initialize lora_B with small non-zero values to verify gradient flow.
    nn.init.normal_(head.lora_B.weight, std=0.01)

    delta = head(hidden_states)
    targets = torch.randint(0, vocab_size, (B, T))
    loss = F.cross_entropy(delta.view(-1, vocab_size), targets.view(-1))
    loss.backward()

    print("Gradient check (with non-zero lora_B init):")
    for name, p in head.named_parameters():
        has_grad = p.grad is not None and p.grad.abs().sum() > 0
        print(f"  {name}: has_grad={has_grad}")
        assert has_grad, f"{name} should have gradients"

    print("\n✓ Gradient flow verified!")


def test_full_model():
    """Test full GemmaMedusaModel (requires downloading model)."""
    try:
        from nanochat.gemma_medusa import GemmaMedusaModel
    except ImportError as e:
        print(f"\n=== Skipping Full Model Test (import error: {e}) ===")
        return

    print("\n=== Testing Full GemmaMedusaModel ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model_name = "google/gemma-3-1b-it"

    print(f"Loading {model_name} (this may take a while)...")
    try:
        model = GemmaMedusaModel(
            model_name=model_name,
            medusa_num_heads=4,
            medusa_num_layers=1,
            lora_rank=64,
            device=device,
            dtype=dtype,
        )
    except Exception as e:
        print(f"  Skipping: {e}")
        return

    model.eval()
    print(f"Medusa params: {model.get_medusa_param_count():,}")

    B, T = 1, 64
    input_ids = torch.randint(0, model.config.vocab_size, (B, T), device=device)

    # Test forward
    with torch.no_grad():
        logits = model(input_ids, return_medusa=False)
        main_logits, medusa_logits = model(input_ids, return_medusa=True)

    print(f"Logits shape: {logits.shape}")
    print(f"Medusa logits shape: {medusa_logits.shape}")

    # Verify consistency
    diff = (logits - main_logits).abs().max().item()
    print(f"Max diff (return_medusa False vs True): {diff:.2e}")
    assert diff < 1e-4

    print("\n✓ Full model tests passed!")


if __name__ == "__main__":
    test_medusa_head_isolated()
    test_gradient_flow()
    test_full_model()
    print("\n✓ All tests passed!")
