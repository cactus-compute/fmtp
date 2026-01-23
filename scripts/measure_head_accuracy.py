"""
Measure Medusa head recall@k for calibrating optimal tree structures.

This script evaluates how well each Medusa head predicts future tokens
by measuring recall@k (k=1 to topk) across training data.

Head i predicts token at position t+i+1 given context up to position t.
Recall@k measures: does the correct token appear in the head's top-k predictions?

Usage (single GPU):
    python -m scripts.measure_head_accuracy \
        --task gsm8k \
        --checkpoint /path/to/medusa_checkpoint \
        --max-samples 200 \
        --max-steps 50 \
        --topk 100 \
        -o head_acc_gsm8k.json

Usage (multi-GPU):
    torchrun --standalone --nproc_per_node=8 -m scripts.measure_head_accuracy \
        --task gsm8k \
        --checkpoint /path/to/medusa_checkpoint \
        --max-samples 1000 \
        --max-steps 50 \
        --topk 100 \
        -o head_acc_gsm8k.json
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


def load_task(task_name: str):
    """Load the specified task's training data."""
    if task_name == "gsm8k":
        from tasks.gsm8k import GSM8K
        return GSM8K(subset="main", split="train")
    elif task_name == "arc-challenge":
        from tasks.arc import ARC
        return ARC(subset="ARC-Challenge", split="train")
    elif task_name == "mmlu":
        from tasks.mmlu import MMLU
        return MMLU(subset="auxiliary_train", split="train")
    elif task_name == "mbpp":
        from tasks.mbpp import MBPP
        return MBPP(split="train")
    else:
        raise ValueError(f"Unknown task: {task_name}")


def compute_head_recall(
    model,
    tokenizer,
    task,
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
        conversation = task[sample_idx]

        # Render the prompt (without the assistant's response)
        prompt_ids = tokenizer.render_for_completion(conversation)

        # Get the ground truth completion tokens
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
                # Head k predicts token at position t + k + 2 (shift = 2 + k in training)
                # At step `step`, we've processed prompt + step tokens
                # The main model predicts completion_ids[step] (next token)
                # Head k predicts completion_ids[step + k + 1] (k+2 tokens ahead from last input)
                # But since we're at position step, the "next" token is step, so:
                # Head 0 predicts step+1, Head 1 predicts step+2, etc.
                # Wait - let's be precise:
                # - Main model at position t predicts token at t+1
                # - Head k at position t predicts token at t+k+2
                # So if we've fed prompt + step tokens (step tokens of completion),
                # the last position is prompt_len + step - 1
                # Main predicts completion_ids[step]
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
    # Shape: (num_heads, topk)
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
    parser = argparse.ArgumentParser(description="Measure Medusa head recall@k")
    parser.add_argument("--task", type=str, required=True,
                        choices=["gsm8k", "arc-challenge", "mmlu", "mbpp"],
                        help="Task to evaluate on")
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
    parser.add_argument("--medusa-num-heads", type=int, default=4,
                        help="Number of Medusa heads in the model")
    parser.add_argument("--medusa-num-layers", type=int, default=1,
                        help="Number of ResBlock layers per head")
    parser.add_argument("--lora-rank", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--model-name", type=str, default="google/gemma-3-1b-it",
                        help="Base model name")

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, device = setup_distributed()
    print0(f"Running with {world_size} GPU(s)")
    print0(f"Using device: {device}")

    # Load tokenizer
    print0("Loading tokenizer...")
    tokenizer = GemmaTokenizerWrapper(args.model_name)

    # Load model
    print0("Loading model...")
    model = load_gemma_medusa_model(
        model_name=args.model_name,
        medusa_num_heads=args.medusa_num_heads,
        medusa_num_layers=args.medusa_num_layers,
        lora_rank=args.lora_rank,
        device=device,
        dtype=torch.bfloat16,
        freeze_base=True,
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

    # Load task
    print0(f"Loading task: {args.task}")
    task = load_task(args.task)
    print0(f"Task size: {len(task)} examples")

    # Determine sample indices for this rank
    num_samples = min(args.max_samples, len(task))
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
        task=task,
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
            "task": args.task,
            "checkpoint": args.checkpoint,
            "num_samples": num_samples,
            "max_steps": args.max_steps,
            "num_gpus": world_size,
            "total_predictions": total_counts,
            "recall": recall_rates,
        }

        # Save results
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to: {args.output}")

        # Print summary
        print("\nRecall Summary (top-1 / top-10 / top-100):")
        for head_idx in range(args.medusa_num_heads):
            r1 = recall_rates[f"head_{head_idx}"]["1"]
            r10 = recall_rates[f"head_{head_idx}"]["10"]
            r100 = recall_rates[f"head_{head_idx}"][str(args.topk)]
            print(f"  Head {head_idx}: {r1:.3f} / {r10:.3f} / {r100:.3f}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
