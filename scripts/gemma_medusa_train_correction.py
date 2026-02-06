"""
Train a correction head for Medusa that specializes in predicting when head 0 is wrong.

Based on gemma_medusa_train.py but:
1. Loads existing checkpoint with head 0
2. Initializes head 1 from head 0's weights (not random)
3. Freezes head 0, trains only head 1
4. Masks gradient to zero when head 0 is correct

Example:
    # Single GPU test
    uv run python -m scripts.gemma_medusa_train_correction \
        --checkpoint ~/.cache/nanochat/gemma_medusa_270m_wildchat_100k_nolora \
        --data-path data/wildchat_100k.jsonl \
        --num-iterations 10

    # Multi-GPU training
    uv run torchrun --standalone --nproc_per_node=8 -m scripts.gemma_medusa_train_correction \
        --checkpoint ~/.cache/nanochat/gemma_medusa_270m_wildchat_100k_nolora \
        --data-path data/wildchat_100k.jsonl \
        --num-iterations 500
"""

import argparse
import json
import os
import time
from contextlib import nullcontext
from collections import defaultdict

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import wandb

from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    autodetect_device_type,
    DummyWandb,
)
from nanochat.gemma_medusa import (
    GemmaTokenizerWrapper,
    load_gemma_medusa_model,
)
from nanochat.gemma_common import (
    load_sharegpt_data,
    data_generator,
)


def compute_head_accuracy(
    model,
    tokenizer,
    conversations: list,
    sample_indices: list,
    max_steps: int,
    device: torch.device,
):
    """Compute accuracy for head 0 and head 1.

    Head k predicts token at position t+k+1 (matching original training).
    So head 0 predicts completion_ids[step + 1], head 1 predicts completion_ids[step + 2].
    """
    stats = defaultdict(int)

    for sample_idx in sample_indices:
        conversation = conversations[sample_idx]
        prompt_ids = tokenizer.render_for_completion(conversation)
        full_ids, _ = tokenizer.render_conversation(conversation)
        completion_ids = full_ids[len(prompt_ids):]

        if len(completion_ids) < 2:  # Need at least 2 tokens for head 0
            continue

        num_steps = min(max_steps, len(completion_ids) - 1)  # -1 because we check step+1
        input_ids = torch.tensor([prompt_ids], device=device)

        for step in range(num_steps):
            # Head 0 predicts token at step + 1
            h0_target_pos = step + 1
            if h0_target_pos >= len(completion_ids):
                break

            h0_target = completion_ids[h0_target_pos]

            with torch.no_grad():
                _, medusa_logits = model.forward(
                    input_ids, return_medusa=True, last_only=True
                )

            h0_top2 = medusa_logits[0, 0, 0, :].topk(2).indices.tolist()
            h0_top1_ok = h0_top2[0] == h0_target
            h0_top2_ok = h0_target in h0_top2

            # Head 1 predicts token at step + 2
            h1_target_pos = step + 2
            if model.medusa_num_heads > 1 and h1_target_pos < len(completion_ids):
                h1_target = completion_ids[h1_target_pos]
                h1_top2 = medusa_logits[1, 0, 0, :].topk(2).indices.tolist()
                h1_top1_ok = h1_top2[0] == h1_target
            else:
                h1_top1_ok = False

            stats['total'] += 1
            stats['h0_top1'] += int(h0_top1_ok)
            stats['h0_top2'] += int(h0_top2_ok)
            stats['h1_top1'] += int(h1_top1_ok)
            stats['combined'] += int(h0_top1_ok or h1_top1_ok)
            if not h0_top1_ok:
                stats['h0_wrong'] += 1
                stats['h1_when_h0_wrong'] += int(h1_top1_ok)

            # Add the actual next token (step) for autoregressive generation
            next_token = completion_ids[step]
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

    return stats


def reduce_stats(stats, device):
    if not dist.is_initialized():
        return stats
    keys = list(stats.keys())
    vals = torch.tensor([stats[k] for k in keys], dtype=torch.long, device=device)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    return {k: v.item() for k, v in zip(keys, vals)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--num-iterations", type=int, default=500)
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--total-batch-size", type=int, default=96)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    # Lower LR for fine-tuning aux head (not training from scratch)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--wandb-run", type=str, default="dummy",
                        help="wandb run name ('dummy' disables wandb logging)")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

    # wandb init
    use_dummy_wandb = args.wandb_run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
        project="gemma-medusa-correction",
        name=args.wandb_run,
        config=vars(args),
    )

    # Load checkpoint config
    checkpoint_path = os.path.expanduser(args.checkpoint)
    with open(os.path.join(checkpoint_path, "config.json")) as f:
        ckpt_config = json.load(f)

    base_model = ckpt_config["base_model"]
    medusa_num_layers = ckpt_config["medusa_num_layers"]
    lora_rank = ckpt_config.get("lora_rank", 0)
    lora_alpha = ckpt_config.get("lora_alpha", 512)

    print0(f"Base model: {base_model}")
    print0(f"Creating 2-head model from {ckpt_config['medusa_num_heads']}-head checkpoint")

    # Create model with 2 heads
    model = load_gemma_medusa_model(
        model_name=base_model,
        medusa_num_heads=2,
        medusa_num_layers=medusa_num_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=device,
        dtype=torch.bfloat16,
        freeze_base=True,
    )
    tokenizer = GemmaTokenizerWrapper(base_model)

    # Load checkpoint
    ckpt_file = os.path.join(checkpoint_path, "final", "medusa_heads.pt")
    if not os.path.exists(ckpt_file):
        ckpt_file = os.path.join(checkpoint_path, "step_1030", "medusa_heads.pt")
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    ckpt_state = checkpoint['medusa_heads']

    # Build state dict:
    # - Head 0: from checkpoint head 0 (frozen)
    # - Head 1: ALSO from checkpoint head 0 (will be fine-tuned)
    current_state = model.medusa_heads.state_dict()
    new_state = {}

    for key in current_state.keys():
        if key.startswith("0."):
            # Head 0 from checkpoint
            new_state[key] = ckpt_state[key]
        elif key.startswith("1."):
            # Head 1: copy from head 0 (same architecture)
            h0_key = "0." + key[2:]  # Replace "1." with "0."
            if h0_key in ckpt_state:
                new_state[key] = ckpt_state[h0_key].clone()
            else:
                print0(f"WARNING: {h0_key} not found, using random init")
                new_state[key] = current_state[key]
        else:
            new_state[key] = current_state[key]

    model.medusa_heads.load_state_dict(new_state)
    print0("Head 0: loaded from checkpoint (frozen)")
    print0("Head 1: initialized from head 0 weights (trainable)")

    # Freeze head 0
    for name, param in model.medusa_heads.named_parameters():
        if name.startswith("0."):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.medusa_heads.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.medusa_heads.parameters() if not p.requires_grad)
    print0(f"Trainable: {trainable:,}, Frozen: {frozen:,}")

    # Data
    print0(f"Loading data: {args.data_path}")
    data = load_sharegpt_data(args.data_path)

    import random
    random.seed(42)
    indices = list(range(len(data)))
    random.shuffle(indices)
    val_data = [data[i] for i in indices[:args.val_samples]]
    train_data = [data[i] for i in indices[args.val_samples:]]
    print0(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_loader = data_generator(
        train_data, tokenizer, args.device_batch_size, args.max_seq_len, device,
        ddp_rank=ddp_rank, ddp_world_size=ddp_world_size
    )

    # Optimizer - only head 1 params
    head1_params = [p for n, p in model.medusa_heads.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head1_params, lr=args.lr, weight_decay=args.weight_decay)

    examples_per_step = args.device_batch_size * ddp_world_size
    grad_accum_steps = max(1, args.total_batch_size // examples_per_step)
    print0(f"Grad accum: {grad_accum_steps}, effective batch: {examples_per_step * grad_accum_steps}")

    def get_lr_mult(step):
        warmup = int(args.warmup_ratio * args.num_iterations)
        if step < warmup:
            return (step + 1) / warmup
        # Linear decay from 1.0 to 0.0 over remaining steps
        decay_steps = args.num_iterations - warmup
        return 1.0 - (step - warmup) / decay_steps

    if args.output_dir is None:
        args.output_dir = checkpoint_path + "_correction"
    if master_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
            json.dump(vars(args), f, indent=2)

    print0(f"\n{'='*60}")
    print0("Training correction head (head 1)")
    print0("Gradient zeroed when head 0 is correct")
    print0(f"{'='*60}\n")

    model.train()
    smooth_loss = 0.0
    smooth_mask_ratio = 0.0
    total_time = 0.0

    for step in range(args.num_iterations):
        last_step = step == args.num_iterations - 1

        # Eval
        if step % args.eval_every == 0 or last_step:
            model.eval()
            n_eval = min(100, len(val_data))
            per_rank = max(1, n_eval // ddp_world_size)
            my_idx = list(range(ddp_rank * per_rank, min((ddp_rank + 1) * per_rank, n_eval)))

            with torch.no_grad(), autocast_ctx:
                stats = compute_head_accuracy(model, tokenizer, val_data, my_idx, 30, device)
            stats = reduce_stats(stats, device)

            if master_process and stats['total'] > 0:
                h0t1 = stats['h0_top1'] / stats['total']
                h0t2 = stats['h0_top2'] / stats['total']
                h1t1 = stats['h1_top1'] / stats['total']
                comb = stats['combined'] / stats['total']
                h1_h0w = stats['h1_when_h0_wrong'] / max(1, stats['h0_wrong'])
                print0(f"\n[Eval step {step}]")
                print0(f"  h0: top1={h0t1:.1%} top2={h0t2:.1%}")
                print0(f"  h1: top1={h1t1:.1%}")
                print0(f"  combined={comb:.1%} (+{comb-h0t2:.1%} vs h0-top2)")
                print0(f"  h1 when h0 wrong: {h1_h0w:.1%}\n")
                wandb_run.log({
                    'eval/h0_top1': h0t1, 'eval/h0_top2': h0t2,
                    'eval/h1_top1': h1t1, 'eval/combined': comb,
                    'eval/h1_when_h0_wrong': h1_h0w,
                    'eval/improvement': comb - h0t2,
                    'step': step,
                })
            model.train()

        if last_step:
            break

        synchronize()
        t0 = time.time()

        total_loss = 0.0
        total_masked = 0
        total_tokens = 0

        optimizer.zero_grad()

        for _ in range(grad_accum_steps):
            (inputs, targets), _ = next(train_loader)

            with autocast_ctx:
                # Forward through base model
                outputs = model.base_model(inputs, output_hidden_states=True, use_cache=False)
                hidden = outputs.hidden_states[-1]  # (B, T, D)

                # For lora_rank=0, medusa_heads return transformed hidden states, not logits
                # We need to apply lm_head to get actual logits
                lm_head = model.base_model.lm_head

                # Head 0 predictions (no grad)
                with torch.no_grad():
                    h0_transformed = model.medusa_heads[0](hidden)  # (B, T, hidden_size)
                    h0_logits = lm_head(h0_transformed)  # (B, T, V) - apply lm_head!
                    h0_preds = h0_logits.argmax(dim=-1)  # (B, T)

                # Head 1 predictions
                h1_transformed = model.medusa_heads[1](hidden)  # (B, T, hidden_size)
                h1_logits = lm_head(h1_transformed)  # (B, T, V) - apply lm_head!

                # Head 0 at position t predicts token at t+2 (one beyond base which predicts t+1)
                # Eval checks head 0 against targets[t+1] = inputs[t+2]
                # So we compare h0_preds[t] vs targets[t+1], meaning shift=1
                vocab_size = h1_logits.size(-1)
                B, T = targets.shape

                shift = 1  # Head 0 predicts t+2, so compare h0_preds[t] with targets[t+1]
                if T > shift:
                    h0_preds_shifted = h0_preds[:, :-shift]  # (B, T-1)
                    targets_shifted = targets[:, shift:]     # (B, T-1)
                    h1_logits_shifted = h1_logits[:, :-shift, :]  # (B, T-1, V)

                    valid = (targets_shifted >= 0) & (targets_shifted < vocab_size)
                    h0_wrong = (h0_preds_shifted != targets_shifted)
                    mask = (h0_wrong & valid)  # (B, T-2), boolean
                else:
                    # Sequence too short
                    valid = torch.zeros_like(targets, dtype=torch.bool)
                    mask = valid
                    targets_shifted = targets
                    h1_logits_shifted = h1_logits

                # Compute loss only on masked positions
                mask_flat = mask.view(-1)
                n_masked = mask_flat.sum().item()
                n_valid = valid.sum().item()

                total_tokens += n_valid

                if n_masked > 0:
                    # Select only positions where we want to train
                    # Use shifted tensors for proper alignment
                    h1_flat = h1_logits_shifted.reshape(-1, vocab_size)
                    targets_flat = targets_shifted.reshape(-1)

                    selected_logits = h1_flat[mask_flat]  # (N, V)
                    selected_targets = targets_flat[mask_flat]  # (N,)

                    # Defensive clamp in case of any edge cases
                    selected_targets = selected_targets.clamp(min=0, max=vocab_size - 1)

                    loss = F.cross_entropy(selected_logits, selected_targets)
                    total_loss += loss.item() * n_masked
                    total_masked += n_masked
                    (loss / grad_accum_steps).backward()

        # LR schedule
        lr_mult = get_lr_mult(step)
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr * lr_mult

        optimizer.step()

        synchronize()
        dt = time.time() - t0
        if step > 3:
            total_time += dt

        # Logging
        avg_loss = total_loss / max(1, total_masked)
        mask_ratio = total_masked / max(1, total_tokens)

        ema = 0.9
        smooth_loss = ema * smooth_loss + (1 - ema) * avg_loss
        smooth_mask_ratio = ema * smooth_mask_ratio + (1 - ema) * mask_ratio
        debias = 1 - ema ** (step + 1)

        if step % 5 == 0:
            print0(f"step {step:04d}/{args.num_iterations} | "
                   f"loss={smooth_loss/debias:.4f} | "
                   f"mask={smooth_mask_ratio/debias:.1%} | "
                   f"lr={lr_mult:.2f} | "
                   f"dt={dt*1000:.0f}ms")
            wandb_run.log({
                'train/loss': smooth_loss / debias,
                'train/mask_ratio': smooth_mask_ratio / debias,
                'train/lr': args.lr * lr_mult,
                'train/dt_ms': dt * 1000,
                'step': step,
            })

    # Final save
    if master_process:
        save_dir = os.path.join(args.output_dir, "final")
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'step': args.num_iterations,
            'medusa_heads': model.medusa_heads.state_dict(),
        }, os.path.join(save_dir, "medusa_heads.pt"))
        print0(f"Saved to {save_dir}")

    # Final eval
    print0(f"\n{'='*60}")
    print0("Final Evaluation")
    print0(f"{'='*60}")

    model.eval()
    n_eval = min(200, len(val_data))
    per_rank = max(1, n_eval // ddp_world_size)
    my_idx = list(range(ddp_rank * per_rank, min((ddp_rank + 1) * per_rank, n_eval)))

    with torch.no_grad(), autocast_ctx:
        stats = compute_head_accuracy(model, tokenizer, val_data, my_idx, 50, device)
    stats = reduce_stats(stats, device)

    if master_process and stats['total'] > 0:
        h0t1 = stats['h0_top1'] / stats['total']
        h0t2 = stats['h0_top2'] / stats['total']
        h1t1 = stats['h1_top1'] / stats['total']
        comb = stats['combined'] / stats['total']
        h1_h0w = stats['h1_when_h0_wrong'] / max(1, stats['h0_wrong'])

        print0(f"\nFinal ({stats['total']} tokens):")
        print0(f"  Head 0: top1={h0t1:.1%}, top2={h0t2:.1%}")
        print0(f"  Head 1: top1={h1t1:.1%}")
        print0(f"  Combined: {comb:.1%}")
        print0(f"  H1 when H0 wrong: {h1_h0w:.1%}")
        print0(f"  Improvement over h0-top2: {comb - h0t2:+.1%}")

        with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
            json.dump({
                'h0_top1': h0t1, 'h0_top2': h0t2, 'h1_top1': h1t1,
                'combined': comb, 'h1_when_h0_wrong': h1_h0w,
                'improvement': comb - h0t2,
            }, f, indent=2)

    print0(f"\nTotal time: {total_time/60:.1f}m")
    wandb_run.finish()
    compute_cleanup()
