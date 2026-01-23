"""
Measure Medusa head recall@k on WildChat validation split.

This script evaluates how well each Medusa head predicts future tokens
using the same validation split from training.

Head i predicts token at position t+i+1 given context up to position t.
Recall@k measures: does the correct token appear in the head's top-k predictions?

Usage (single GPU):
    python -m scripts.measure_head_accuracy_wildchat \
        --data-path data/wildchat_distill_v2_gemma3_270m.jsonl \
        --checkpoint ~/.cache/nanochat/gemma_medusa_270m_wildchat_100k \
        --max-samples 200 \
        --max-steps 50 \
        --topk 100 \
        -o head_acc_wildchat.json

Usage (multi-GPU):
    torchrun --standalone --nproc_per_node=8 -m scripts.measure_head_accuracy_wildchat \
        --data-path data/wildchat_distill_v2_gemma3_270m.jsonl \
        --checkpoint ~/.cache/nanochat/gemma_medusa_270m_wildchat_100k \
        --max-samples 1000 \
        --max-steps 50 \
        --topk 100 \
        -o head_acc_wildchat.json
"""

import argparse
import json
import os
import torch
import torch.distributed as dist
from tqdm import tqdm

from nanochat.gemma_medusa.model import load_gemma_medusa_model
from nanochat.gemma_medusa.tokenizer import GemmaTokenizerWrapper


def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        return rank, world_size, device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device


def print0(*args, **kwargs):
    """Print only from rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def load_wildchat_data(filepath: str):
    """
    Load WildChat-format JSONL data.

    Each line should be: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    conversations = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if 'messages' in item:
                conversations.append(item)
    return conversations


def compute_head_recall(
    model,
    tokenizer,
    conversations: list,
    sample_indices: list,
    max_steps: int,
    topk: int,
    device: torch.device,
    rank: int,
    world_size: int,
):
    """
    Compute recall@k for each Medusa head across the given sample indices.

    Returns:
        recall_counts: dict mapping head_idx -> k -> count of hits
        total_counts: dict mapping head_idx -> total predictions
    """
    num_heads = model.medusa_num_heads

    # recall_counts[head_idx][k] = number of times correct token was in top-k
    recall_counts = {h: {k: 0 for k in range(1, topk + 1)} for h in range(num_heads)}
    total_counts = {h: 0 for h in range(num_heads)}

    # Show progress bar only on rank 0
    iterator = tqdm(sample_indices, desc=f"Rank {rank}") if rank == 0 else sample_indices

    for sample_idx in iterator:
        conversation = conversations[sample_idx]

        # Render the prompt (without the assistant's response)
        prompt_ids = tokenizer.render_for_completion(conversation)

        # Get the ground truth completion tokens (full conversation)
        full_ids, _ = tokenizer.render_conversation(conversation)
        completion_ids = full_ids[len(prompt_ids):]

        if len(completion_ids) == 0:
            continue

        # Limit to max_steps
        num_steps = min(max_steps, len(completion_ids))

        # Process step by step
        input_ids = torch.tensor([prompt_ids], device=device)

        for step in range(num_steps):
            # Forward pass with medusa logits
            with torch.no_grad():
                main_logits, medusa_logits = model.forward(
                    input_ids, return_medusa=True, last_only=True
                )
                # main_logits: (B, 1, vocab_size)
                # medusa_logits: (num_heads, B, 1, vocab_size)

            # For each head, check if the correct future token is in top-k
            for head_idx in range(num_heads):
                # Head k predicts completion_ids[step + k + 1]
                future_pos = step + head_idx + 1

                if future_pos >= len(completion_ids):
                    # No more ground truth tokens for this head
                    continue

                target_token = completion_ids[future_pos]

                # Get head's top-k predictions
                head_logits = medusa_logits[head_idx, 0, 0, :]  # (vocab_size,)
                top_indices = head_logits.topk(topk, dim=-1).indices  # (topk,)

                # Check recall at each k
                for k in range(1, topk + 1):
                    if target_token in top_indices[:k]:
                        recall_counts[head_idx][k] += 1

                total_counts[head_idx] += 1

            # Autoregressive step: add the actual next token
            next_token = completion_ids[step]
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token]], device=device)
            ], dim=1)

    return recall_counts, total_counts


def reduce_counts(recall_counts, total_counts, num_heads, topk, device):
    """Reduce counts across all ranks."""
    if not dist.is_initialized():
        return recall_counts, total_counts

    # Flatten recall_counts to tensor for all_reduce
    recall_tensor = torch.zeros(num_heads, topk, dtype=torch.long, device=device)
    for h in range(num_heads):
        for k in range(1, topk + 1):
            recall_tensor[h, k - 1] = recall_counts[h][k]

    # Flatten total_counts
    total_tensor = torch.zeros(num_heads, dtype=torch.long, device=device)
    for h in range(num_heads):
        total_tensor[h] = total_counts[h]

    # All-reduce (sum across all ranks)
    dist.all_reduce(recall_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    # Convert back to dict
    reduced_recall = {h: {k: recall_tensor[h, k - 1].item() for k in range(1, topk + 1)} for h in range(num_heads)}
    reduced_total = {h: total_tensor[h].item() for h in range(num_heads)}

    return reduced_recall, reduced_total


def main():
    parser = argparse.ArgumentParser(description="Measure Medusa head recall@k on WildChat")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to WildChat JSONL file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to medusa checkpoint directory")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Maximum tokens per sample")
    parser.add_argument("--topk", type=int, default=100,
                        help="Maximum k to measure recall at")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output JSON path")
    parser.add_argument("--val-samples", type=int, default=960,
                        help="Number of validation samples (matches training default)")
    parser.add_argument("--medusa-num-heads", type=int, default=4,
                        help="Number of Medusa heads in the model")
    parser.add_argument("--medusa-num-layers", type=int, default=2,
                        help="Number of ResBlock layers per head")
    parser.add_argument("--lora-rank", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha (default: same as rank)")
    parser.add_argument("--model-name", type=str, default="google/gemma-3-270m-it",
                        help="Base model name")
    parser.add_argument("--use-head-mixer", action="store_true",
                        help="Use cross-head MLP mixer")
    parser.add_argument("--mixer-hidden", type=int, default=16,
                        help="Hidden dimension for the mixer")

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, device = setup_distributed()
    print0(f"Running with {world_size} GPU(s)")
    print0(f"Using device: {device}")

    # Load WildChat data
    print0(f"Loading WildChat data from: {args.data_path}")
    all_conversations = load_wildchat_data(args.data_path)
    print0(f"Loaded {len(all_conversations)} conversations")

    # Extract validation split (last val_samples, same as training)
    val_conversations = all_conversations[-args.val_samples:]
    print0(f"Using last {len(val_conversations)} conversations as validation set")

    # Load tokenizer
    print0("Loading tokenizer...")
    tokenizer = GemmaTokenizerWrapper(args.model_name)

    # Load model
    print0("Loading model...")
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank
    model = load_gemma_medusa_model(
        model_name=args.model_name,
        medusa_num_heads=args.medusa_num_heads,
        medusa_num_layers=args.medusa_num_layers,
        lora_rank=args.lora_rank,
        lora_alpha=lora_alpha,
        device=device,
        dtype=torch.bfloat16,
        freeze_base=True,
        use_head_mixer=args.use_head_mixer,
        mixer_hidden=args.mixer_hidden,
    )

    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint, "final", "medusa_heads.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(args.checkpoint, "medusa_heads.pt")
    print0(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.medusa_heads.load_state_dict(checkpoint['medusa_heads'])
    model.eval()
    print0(f"Medusa parameters: {model.get_medusa_param_count():,}")

    # Determine sample indices for this rank
    num_samples = min(args.max_samples, len(val_conversations))
    all_indices = list(range(num_samples))

    # Shard indices across ranks
    indices_per_rank = len(all_indices) // world_size
    start_idx = rank * indices_per_rank
    end_idx = start_idx + indices_per_rank if rank < world_size - 1 else len(all_indices)
    my_indices = all_indices[start_idx:end_idx]

    print0(f"Total samples: {num_samples}, samples per rank: ~{indices_per_rank}")
    if world_size > 1:
        print(f"Rank {rank}: processing indices {start_idx} to {end_idx} ({len(my_indices)} samples)")

    # Compute recall on this rank's shard
    print0(f"Computing recall@1-{args.topk} for {args.medusa_num_heads} heads...")
    recall_counts, total_counts = compute_head_recall(
        model=model,
        tokenizer=tokenizer,
        conversations=val_conversations,
        sample_indices=my_indices,
        max_steps=args.max_steps,
        topk=args.topk,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    # Reduce counts across all ranks
    recall_counts, total_counts = reduce_counts(
        recall_counts, total_counts,
        args.medusa_num_heads, args.topk, device
    )

    # Only rank 0 saves results
    if rank == 0:
        # Convert counts to recall rates
        recall_rates = {}
        for head_idx in range(args.medusa_num_heads):
            total = total_counts[head_idx]
            if total > 0:
                recall_rates[f"head_{head_idx}"] = {
                    str(k): recall_counts[head_idx][k] / total
                    for k in range(1, args.topk + 1)
                }
            else:
                recall_rates[f"head_{head_idx}"] = {
                    str(k): 0.0 for k in range(1, args.topk + 1)
                }

        # Build output
        output = {
            "data_path": args.data_path,
            "checkpoint": args.checkpoint,
            "model_name": args.model_name,
            "num_samples": num_samples,
            "val_samples": args.val_samples,
            "max_steps": args.max_steps,
            "num_gpus": world_size,
            "config": {
                "medusa_num_heads": args.medusa_num_heads,
                "medusa_num_layers": args.medusa_num_layers,
                "lora_rank": args.lora_rank,
                "lora_alpha": lora_alpha,
                "use_head_mixer": args.use_head_mixer,
                "mixer_hidden": args.mixer_hidden if args.use_head_mixer else None,
            },
            "total_predictions": total_counts,
            "recall": recall_rates,
        }

        # Save results
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")

        # Print summary
        print("\nRecall Summary (top-1 / top-10 / top-100):")
        for head_idx in range(args.medusa_num_heads):
            r1 = recall_rates[f"head_{head_idx}"]["1"]
            r10 = recall_rates[f"head_{head_idx}"]["10"]
            r100 = recall_rates[f"head_{head_idx}"][str(min(args.topk, 100))]
            print(f"  Head {head_idx}: {r1:.3f} / {r10:.3f} / {r100:.3f}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
