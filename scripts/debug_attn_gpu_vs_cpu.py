#!/usr/bin/env python
"""
Debug script to compare attention mixer outputs on GPU vs CPU.

This helps diagnose why the 2-attention model has:
- GPU: mean_accepted=1.01 (broken)
- CPU: mean_accepted=1.72 (working)
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path


def load_model_and_checkpoint(checkpoint_dir: str, device: str):
    """Load model with checkpoint on specified device."""
    from nanochat.gemma_medusa.model import GemmaMedusaModel
    import json

    device = torch.device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Load checkpoint and config
    checkpoint_path = Path(checkpoint_dir) / "final" / "medusa_heads.pt"
    config_path = Path(checkpoint_dir) / "config.json"

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    # Try to load config from JSON file first, then from checkpoint
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = checkpoint.get('config', {})

    # Create model with same config
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

    # Load weights
    warnings = model.load_medusa_state_dict(checkpoint, strict=False)
    for w in warnings:
        print(f"  Warning: {w}")

    model.eval()
    return model, config


def get_attention_outputs(model, input_ids, device):
    """
    Get intermediate outputs from the attention mixer.

    Returns:
        - hidden_states: Raw transformer output
        - resblock_outputs: Outputs from each head's ResBlocks
        - post_attention: Outputs after the cross-head attention
        - medusa_logits: Final medusa logits
    """
    with torch.no_grad():
        # Get hidden states from base model
        hidden_states = model._get_hidden_states(input_ids)

        # Compute ResBlock outputs for each head
        resblock_outputs = []
        for head in model.medusa_heads:
            x = hidden_states
            for block in head.blocks:
                x = block(x)
            resblock_outputs.append(x)

        # Stack for attention
        stacked = torch.stack(resblock_outputs, dim=0)  # (num_heads, B, T, hidden)

        # Apply attention mixer
        if model.head_attention is not None:
            post_attention = model.head_attention(stacked)

            # Apply channel mixing
            if model.channel_mixer_fc is not None:
                post_channel = post_attention + F.silu(model.channel_mixer_fc(post_attention))
            else:
                post_channel = post_attention
        else:
            post_attention = stacked
            post_channel = stacked

        # Get final logits
        _, medusa_logits = model._compute_logits(hidden_states, return_medusa=True, last_only=True)

        return {
            'hidden_states': hidden_states,
            'stacked_resblock': stacked,
            'post_attention': post_attention,
            'post_channel': post_channel,
            'medusa_logits': medusa_logits,
        }


def compare_tensor_stats(name, t1, t2, device1='gpu', device2='cpu'):
    """Compare statistics of two tensors."""
    # Move to CPU for comparison
    t1_cpu = t1.float().cpu()
    t2_cpu = t2.float().cpu()

    # Basic stats
    diff = (t1_cpu - t2_cpu).abs()

    print(f"\n{name}:")
    print(f"  Shape: {tuple(t1.shape)}")
    print(f"  {device1} - mean: {t1_cpu.mean():.6f}, std: {t1_cpu.std():.6f}, min: {t1_cpu.min():.6f}, max: {t1_cpu.max():.6f}")
    print(f"  {device2} - mean: {t2_cpu.mean():.6f}, std: {t2_cpu.std():.6f}, min: {t2_cpu.min():.6f}, max: {t2_cpu.max():.6f}")
    print(f"  Diff - mean: {diff.mean():.6f}, max: {diff.max():.6f}")

    # Check for exact match
    if torch.allclose(t1_cpu, t2_cpu, atol=1e-4, rtol=1e-4):
        print(f"  ✓ Close match (atol=1e-4)")
    elif torch.allclose(t1_cpu, t2_cpu, atol=1e-2, rtol=1e-2):
        print(f"  ~ Approximate match (atol=1e-2)")
    else:
        print(f"  ✗ SIGNIFICANT DIFFERENCE")

        # Find where the differences are largest
        if len(t1.shape) >= 4:  # medusa logits: (num_heads, B, T, vocab)
            head_diffs = diff.mean(dim=(1, 2, 3))
            print(f"  Per-head mean diff: {head_diffs.tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--prompt', '-p', type=str, default="The quick brown fox",
                       help='Test prompt')
    args = parser.parse_args()

    # Check for CUDA
    if not torch.cuda.is_available():
        print("CUDA not available - this script requires GPU to compare GPU vs CPU")
        return

    print("Loading model on GPU...")
    model_gpu, config = load_model_and_checkpoint(args.checkpoint, 'cuda')

    print("\nLoading model on CPU...")
    model_cpu, _ = load_model_and_checkpoint(args.checkpoint, 'cpu')

    # Tokenize input
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.get('model_name', 'google/gemma-3-270m-it'))
    input_ids = tokenizer.encode(args.prompt, return_tensors='pt')

    print(f"\nPrompt: '{args.prompt}'")
    print(f"Input IDs shape: {input_ids.shape}")

    # Get outputs from both
    print("\n" + "="*60)
    print("Getting GPU outputs...")
    print("="*60)
    gpu_outputs = get_attention_outputs(model_gpu, input_ids.cuda(), 'cuda')

    print("\n" + "="*60)
    print("Getting CPU outputs...")
    print("="*60)
    cpu_outputs = get_attention_outputs(model_cpu, input_ids, 'cpu')

    # Compare
    print("\n" + "="*60)
    print("COMPARING GPU vs CPU OUTPUTS")
    print("="*60)

    compare_tensor_stats('Hidden States', gpu_outputs['hidden_states'], cpu_outputs['hidden_states'])
    compare_tensor_stats('Stacked ResBlock', gpu_outputs['stacked_resblock'], cpu_outputs['stacked_resblock'])
    compare_tensor_stats('Post Attention', gpu_outputs['post_attention'], cpu_outputs['post_attention'])
    compare_tensor_stats('Post Channel Mixing', gpu_outputs['post_channel'], cpu_outputs['post_channel'])
    compare_tensor_stats('Medusa Logits', gpu_outputs['medusa_logits'], cpu_outputs['medusa_logits'])

    # Check top-k predictions
    print("\n" + "="*60)
    print("TOP-K PREDICTION COMPARISON")
    print("="*60)

    gpu_logits = gpu_outputs['medusa_logits'][:, :, -1, :]  # (num_heads, B, vocab)
    cpu_logits = cpu_outputs['medusa_logits'][:, :, -1, :]

    for head_idx in range(gpu_logits.shape[0]):
        gpu_topk = gpu_logits[head_idx, 0].float().cpu().topk(5)
        cpu_topk = cpu_logits[head_idx, 0].float().cpu().topk(5)

        print(f"\nHead {head_idx}:")
        print(f"  GPU top-5 tokens: {gpu_topk.indices.tolist()}")
        print(f"  CPU top-5 tokens: {cpu_topk.indices.tolist()}")

        # Check if top-1 matches
        if gpu_topk.indices[0] == cpu_topk.indices[0]:
            print(f"  ✓ Top-1 match: {gpu_topk.indices[0].item()}")
        else:
            print(f"  ✗ Top-1 MISMATCH: GPU={gpu_topk.indices[0].item()}, CPU={cpu_topk.indices[0].item()}")

        # Check overlap in top-5
        gpu_set = set(gpu_topk.indices.tolist())
        cpu_set = set(cpu_topk.indices.tolist())
        overlap = len(gpu_set & cpu_set)
        print(f"  Top-5 overlap: {overlap}/5")


if __name__ == "__main__":
    main()
