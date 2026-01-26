"""
Training script for EAGLE-3 draft model on Gemma3.

Usage:
    # Single GPU
    uv run python -m scripts.eagle_train --base-model google/gemma-3-1b-it --data-path data/train.json

    # Multi-GPU with torchrun
    uv run torchrun --standalone --nproc_per_node=8 -m scripts.eagle_train \
        --base-model google/gemma-3-1b-it --data-path data/train.json

Based on EAGLE-3 training methodology:
- Multi-step prediction with weighted loss
- Multi-layer feature fusion
- Training-time testing simulation
"""

import argparse
import json
import os
import time
from contextlib import nullcontext

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
import wandb

from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    DummyWandb,
    autodetect_device_type,
)
from nanochat.gemma_eagle import GemmaEagleConfig, GemmaEagleModel
from nanochat.gemma_medusa import GemmaTokenizerWrapper
from nanochat.gemma_common import (
    load_sharegpt_data,
    filter_dataset,
    split_train_val,
    simple_data_generator,
    get_lr_scheduler,
    apply_lr_schedule,
    save_checkpoint,
    setup_output_dir,
    EMALoss,
    estimate_eta,
)


# -----------------------------------------------------------------------------
# Step accuracy evaluation (equivalent to Medusa head accuracy)

def compute_eagle_step_recall(
    model,
    tokenizer,
    conversations: list,
    sample_indices: list,
    max_steps: int,
    num_draft_steps: int,
    topk: int,
    device: torch.device,
    rank: int,
    world_size: int,
):
    """
    Compute recall@k for each EAGLE prediction step.

    Unlike Medusa where heads predict independently, EAGLE is autoregressive:
    - Step 0: Uses fused hidden from base model
    - Step k: Uses output hidden from step k-1

    Returns:
        recall_counts: dict mapping step_idx -> k -> count of hits
        total_counts: dict mapping step_idx -> total predictions
    """
    recall_counts = {s: {k: 0 for k in range(1, topk + 1)} for s in range(num_draft_steps)}
    total_counts = {s: 0 for s in range(num_draft_steps)}

    for sample_idx in sample_indices:
        conversation = conversations[sample_idx]

        # Render the prompt (without the assistant's response)
        prompt_ids = tokenizer.render_for_completion(conversation)

        # Get the ground truth completion tokens (full conversation)
        full_ids, _ = tokenizer.render_conversation(conversation)
        completion_ids = full_ids[len(prompt_ids):]

        if len(completion_ids) < num_draft_steps + 1:
            continue

        num_positions = min(max_steps, len(completion_ids) - num_draft_steps)
        input_ids = torch.tensor([prompt_ids], device=device)

        for pos in range(num_positions):
            with torch.no_grad():
                # Get base model hidden states (fused from multiple layers)
                fused_hidden, _ = model.get_base_hidden_states(input_ids)

                # Last position's fused hidden state
                hidden_states = fused_hidden[:, -1:, :]

                # Current token ID for embedding (last token in input)
                current_ids = input_ids[:, -1:]

                for step in range(num_draft_steps):
                    target_pos = pos + step
                    if target_pos >= len(completion_ids):
                        break

                    target_token = completion_ids[target_pos]

                    # Forward through draft model using forward_draft
                    hidden_out, _ = model.forward_draft(
                        hidden_states=hidden_states,
                        input_ids=current_ids,
                    )

                    # Get logits from last position
                    logits = model.lm_head(model.norm(hidden_out[:, -1, :]))
                    top_indices = logits.topk(topk, dim=-1).indices[0]

                    # Check recall at each k
                    for k in range(1, topk + 1):
                        if target_token in top_indices[:k]:
                            recall_counts[step][k] += 1
                    total_counts[step] += 1

                    # Prepare for next step (autoregressive with teacher forcing)
                    current_ids = torch.tensor([[target_token]], device=device)
                    hidden_states = hidden_out

            # Advance input for next position (autoregressive)
            next_token = completion_ids[pos]
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token]], device=device)
            ], dim=1)

    return recall_counts, total_counts


def reduce_eagle_step_recall_counts(recall_counts, total_counts, num_draft_steps, topk, device):
    """Reduce recall counts across all ranks."""
    if not dist.is_initialized():
        return recall_counts, total_counts

    # Flatten recall_counts to tensor for all_reduce
    recall_tensor = torch.zeros(num_draft_steps, topk, dtype=torch.long, device=device)
    for s in range(num_draft_steps):
        for k in range(1, topk + 1):
            recall_tensor[s, k - 1] = recall_counts[s][k]

    # Flatten total_counts
    total_tensor = torch.zeros(num_draft_steps, dtype=torch.long, device=device)
    for s in range(num_draft_steps):
        total_tensor[s] = total_counts[s]

    # All-reduce (sum across all ranks)
    dist.all_reduce(recall_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    # Convert back to dict
    reduced_recall = {s: {k: recall_tensor[s, k - 1].item() for k in range(1, topk + 1)} for s in range(num_draft_steps)}
    reduced_total = {s: total_tensor[s].item() for s in range(num_draft_steps)}

    return reduced_recall, reduced_total


def run_eagle_accuracy_eval(
    model,
    tokenizer,
    val_data,
    device,
    ddp_rank,
    ddp_world_size,
    num_draft_steps,
    max_samples=200,
    max_steps=50,
    topk=100,
):
    """
    Run step accuracy evaluation on validation data.

    Returns:
        recall_rates: dict mapping step_idx -> k -> recall rate
    """
    model.eval()

    # Determine sample indices for this rank
    num_samples = min(max_samples, len(val_data))
    all_indices = list(range(num_samples))

    # Shard indices across ranks
    indices_per_rank = len(all_indices) // ddp_world_size
    start_idx = ddp_rank * indices_per_rank
    end_idx = start_idx + indices_per_rank if ddp_rank < ddp_world_size - 1 else len(all_indices)
    my_indices = all_indices[start_idx:end_idx]

    # Compute recall on this rank's shard
    recall_counts, total_counts = compute_eagle_step_recall(
        model=model,
        tokenizer=tokenizer,
        conversations=val_data,
        sample_indices=my_indices,
        max_steps=max_steps,
        num_draft_steps=num_draft_steps,
        topk=topk,
        device=device,
        rank=ddp_rank,
        world_size=ddp_world_size,
    )

    # Reduce counts across all ranks
    recall_counts, total_counts = reduce_eagle_step_recall_counts(
        recall_counts, total_counts, num_draft_steps, topk, device
    )

    # Convert counts to recall rates
    recall_rates = {}
    for step_idx in range(num_draft_steps):
        total = total_counts[step_idx]
        if total > 0:
            recall_rates[f"step_{step_idx}"] = {
                str(k): recall_counts[step_idx][k] / total
                for k in range(1, topk + 1)
            }
        else:
            recall_rates[f"step_{step_idx}"] = {
                str(k): 0.0 for k in range(1, topk + 1)
            }

    return recall_rates, total_counts


# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train EAGLE-3 draft model for Gemma3")

    # Logging
    parser.add_argument("--wandb-run", type=str, default="dummy",
                        help="wandb run name ('dummy' disables wandb logging)")

    # Runtime
    parser.add_argument("--device-type", type=str, default="",
                        help="cuda|cpu|mps (empty = autodetect)")

    # Model
    parser.add_argument("--base-model", type=str, default="google/gemma-3-1b-it",
                        help="Base model name or path")
    parser.add_argument("--draft-checkpoint", type=str, default=None,
                        help="Resume from draft model checkpoint")

    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to training data (JSON or JSONL)")
    parser.add_argument("--val-data-path", type=str, default=None,
                        help="Path to validation data (optional)")
    parser.add_argument("--val-samples", type=int, default=960,
                        help="Number of samples to split off for validation")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--skip-filter", action="store_true",
                        help="Skip dataset filtering")

    # Training
    parser.add_argument("--device-batch-size", type=int, default=2,
                        help="Per-device batch size")
    parser.add_argument("--total-batch-size", type=int, default=64,
                        help="Total batch size for gradient accumulation")
    parser.add_argument("--matrix-lr", type=float, default=0.01,
                        help="Learning rate for matrix params (Muon)")
    parser.add_argument("--proj-lr", type=float, default=0.001,
                        help="Learning rate for projection params (AdamW)")
    parser.add_argument("--weight-decay", type=float, default=0.2,
                        help="Weight decay for Muon optimizer")
    parser.add_argument("--adam-weight-decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--adam-beta1", type=float, default=0.8,
                        help="Adam beta1")
    parser.add_argument("--adam-beta2", type=float, default=0.95,
                        help="Adam beta2")
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
                        help="Warmup ratio")
    parser.add_argument("--warmdown-ratio", type=float, default=0.95,
                        help="Warmdown start ratio")
    parser.add_argument("--num-iterations", type=int, default=-1,
                        help="Number of iterations (-1 = use num_epochs)")
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--num-steps", type=int, default=7,
                        help="Number of prediction steps per forward")
    parser.add_argument("--loss-decay", type=float, default=0.8,
                        help="Exponential decay for step losses")

    # Checkpointing
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--save-every", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--eval-steps", type=int, default=10,
                        help="Number of eval batches")
    parser.add_argument("--log-every", type=int, default=20,
                        help="Log detailed metrics every N steps")

    # Hardware
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 training")
    parser.add_argument("--no-chunked-loss", action="store_true",
                        help="Disable chunked loss computation")

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    user_config = vars(args).copy()

    # Compute initialization
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

    # wandb logging init
    use_dummy_wandb = args.wandb_run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
        project="gemma-eagle",
        name=args.wandb_run,
        config=user_config,
    )

    # Load tokenizer
    print0(f"Loading tokenizer from: {args.base_model}")
    tokenizer = GemmaTokenizerWrapper(args.base_model)

    # Create config and model
    print0(f"Loading base model: {args.base_model}")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    config = GemmaEagleConfig(
        base_model_name=args.base_model,
        freeze_base=True,
    )

    model = GemmaEagleModel(
        config,
        device=device,
        dtype=dtype,
    )

    # Load checkpoint if resuming
    if args.draft_checkpoint:
        print0(f"Loading draft checkpoint from {args.draft_checkpoint}")
        state = torch.load(args.draft_checkpoint, map_location=device)
        model.load_draft_state_dict(state)

    # Print model info
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    print0(f"Trainable parameters: {trainable_params:,}")

    # Optional compile
    if args.compile:
        print0("Compiling model with torch.compile...")
        model = torch.compile(model, dynamic=True)

    # Load data
    print0(f"Loading training data from: {args.data_path}")
    train_data = load_sharegpt_data(args.data_path)
    print0(f"Loaded {len(train_data)} training conversations")

    if not args.skip_filter:
        print0("Filtering conversations...")
        train_data, skipped = filter_dataset(train_data, tokenizer, args.max_seq_len, min_valid_tokens=1)
        print0(f"Filtered to {len(train_data)} conversations (skipped {skipped})")

    # Split validation data
    val_data = None
    if args.val_data_path:
        print0(f"Loading validation data from: {args.val_data_path}")
        val_data = load_sharegpt_data(args.val_data_path)
        if not args.skip_filter:
            val_data, _ = filter_dataset(val_data, tokenizer, args.max_seq_len, min_valid_tokens=1)
        print0(f"Loaded {len(val_data)} validation conversations")
    elif args.val_samples > 0:
        train_data, val_data = split_train_val(train_data, args.val_samples)
        print0(f"Split off {len(val_data)} validation samples")

    # Calculate training schedule
    examples_per_step = args.device_batch_size * ddp_world_size
    assert args.total_batch_size % examples_per_step == 0
    grad_accum_steps = args.total_batch_size // examples_per_step
    print0(f"Gradient accumulation steps: {grad_accum_steps}")

    if args.num_iterations > 0:
        num_iterations = args.num_iterations
    else:
        iterations_per_epoch = len(train_data) // args.total_batch_size
        num_iterations = iterations_per_epoch * args.num_epochs
    print0(f"Total iterations: {num_iterations:,}")

    # Setup optimizers (nanochat-style: Muon + AdamW)
    adam_betas = (args.adam_beta1, args.adam_beta2)
    optimizers = model.setup_optimizers(
        matrix_lr=args.matrix_lr,
        proj_lr=args.proj_lr,
        weight_decay=args.weight_decay,
        adam_weight_decay=args.adam_weight_decay,
        adam_betas=adam_betas,
    )

    # LR scheduler
    lr_fn = get_lr_scheduler(args.warmup_ratio, args.warmdown_ratio, 0.0)

    # Loss weights for multi-step prediction
    loss_weights = torch.tensor(
        [args.loss_decay ** i for i in range(args.num_steps)],
        device=device
    )
    loss_weights = loss_weights / loss_weights.sum()
    print0(f"Loss weights: {loss_weights.tolist()}")

    # Setup output directory
    model_short = args.base_model.split('/')[-1]
    output_dir = setup_output_dir(
        args.output_dir,
        f"gemma_eagle_{model_short}",
        user_config,
        master_process,
    )
    if master_process:
        print0(f"Output directory: {output_dir}")

    # Data loaders
    train_loader = simple_data_generator(
        train_data, tokenizer, args.device_batch_size, args.max_seq_len, device,
        ddp_rank=ddp_rank, ddp_world_size=ddp_world_size
    )

    def build_val_loader():
        if val_data is None:
            return None
        return simple_data_generator(
            val_data, tokenizer, args.device_batch_size, args.max_seq_len, device,
            ddp_rank=ddp_rank, ddp_world_size=ddp_world_size
        )

    # Training loop
    print0("\n" + "=" * 50)
    print0("Starting EAGLE training")
    print0("=" * 50 + "\n")

    model.train()
    ema_loss = EMALoss(beta=0.9)
    ema_accs = [EMALoss(beta=0.9) for _ in range(args.num_steps)]
    ema_step_losses = [EMALoss(beta=0.9) for _ in range(args.num_steps)]
    total_training_time = 0.0
    current_epoch = 0

    for step in range(num_iterations):
        last_step = step == num_iterations - 1

        # Evaluation
        if val_data is not None and args.eval_every > 0 and (last_step or step % args.eval_every == 0):
            model.eval()
            val_loader = build_val_loader()
            val_losses = []

            for eval_step in range(args.eval_steps):
                (val_input_ids, val_attention_mask, val_loss_mask), _ = next(val_loader)
                with torch.no_grad(), autocast_ctx:
                    losses, _ = model(
                        val_input_ids, val_attention_mask, val_loss_mask, args.num_steps
                    )
                    total_loss = sum(w * l for w, l in zip(loss_weights, losses))
                    val_losses.append(total_loss)

            val_loss = torch.stack(val_losses).mean()
            if ddp:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

            print0(f"Step {step:05d} | Val loss: {val_loss.item():.6f}")
            wandb_run.log({"step": step, "val/loss": val_loss.item()})
            model.train()

        # Save checkpoint
        if master_process and args.save_every > 0 and (last_step or (step > 0 and step % args.save_every == 0)):
            ckpt_path = save_checkpoint(
                output_dir, step, model.get_draft_state_dict(),
                optimizers=optimizers, config=user_config, filename="eagle_draft.pt"
            )
            print0(f"Saved checkpoint to {ckpt_path}")

        if last_step:
            break

        # Training step
        synchronize()
        t0 = time.time()

        total_loss = torch.tensor(0.0, device=device)
        total_accs = [0.0] * args.num_steps
        total_step_losses = [torch.tensor(0.0, device=device) for _ in range(args.num_steps)]

        for micro_step in range(grad_accum_steps):
            (input_ids, attention_mask, loss_mask), epoch = next(train_loader)
            current_epoch = epoch

            with autocast_ctx:
                losses, accuracies = model(
                    input_ids, attention_mask, loss_mask, args.num_steps,
                    use_chunked_loss=not args.no_chunked_loss,
                )
                loss = sum(w * l for w, l in zip(loss_weights, losses))
                total_loss += loss.detach()
                for i, acc in enumerate(accuracies):
                    total_accs[i] += acc
                for i, step_loss in enumerate(losses):
                    total_step_losses[i] += step_loss.detach()

            (loss / grad_accum_steps).backward()

        total_loss /= grad_accum_steps
        total_accs = [a / grad_accum_steps for a in total_accs]
        total_step_losses = [sl / grad_accum_steps for sl in total_step_losses]

        # LR schedule
        lrm = apply_lr_schedule(optimizers, step, num_iterations, lr_fn)

        # Compute gradient norms BEFORE optimizer step (for logging)
        if step % args.log_every == 0:
            grad_norms = {}
            total_grad_norm_sq = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm_sq += grad_norm ** 2
                    if 'fusion' in name:
                        grad_norms.setdefault('fusion', []).append(grad_norm)
                    elif 'self_attn' in name:
                        grad_norms.setdefault('attn', []).append(grad_norm)
                    elif 'mlp' in name:
                        grad_norms.setdefault('mlp', []).append(grad_norm)
                    elif 'norm' in name:
                        grad_norms.setdefault('norm', []).append(grad_norm)
            total_grad_norm = total_grad_norm_sq ** 0.5
        else:
            grad_norms = {}
            total_grad_norm = 0.0

        # Gradient step (Muon doesn't need clipping, AdamW benefits from it)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        synchronize()
        t1 = time.time()
        dt = t1 - t0
        if step > 10:
            total_training_time += dt

        # Logging
        loss_item = total_loss.item()
        debiased_loss = ema_loss.update(loss_item)
        debiased_accs = [ema.update(acc) for ema, acc in zip(ema_accs, total_accs)]
        debiased_step_losses = [ema.update(sl.item()) for ema, sl in zip(ema_step_losses, total_step_losses)]

        pct_done = 100 * step / num_iterations
        eta = estimate_eta(step, num_iterations, total_training_time, warmup_steps=10)
        eta_str = f" | eta: {eta}" if eta else ""

        acc_str = ", ".join([f"{a:.3f}" for a in debiased_accs[:3]])
        print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.1f}%) | loss: {debiased_loss:.4f} | acc: [{acc_str}] | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | total: {total_training_time/60:.1f}m{eta_str}")

        if step % args.log_every == 0:
            # Compute weight norms for logging (grad norms already computed above)
            weight_norms = {}
            weight_maxes = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    w_norm = param.norm().item()
                    w_max = param.abs().max().item()
                    if 'fusion' in name:
                        weight_norms.setdefault('fusion', []).append(w_norm)
                        weight_maxes.setdefault('fusion', []).append(w_max)
                    elif 'self_attn' in name:
                        weight_norms.setdefault('attn', []).append(w_norm)
                        weight_maxes.setdefault('attn', []).append(w_max)
                    elif 'mlp' in name:
                        weight_norms.setdefault('mlp', []).append(w_norm)
                        weight_maxes.setdefault('mlp', []).append(w_max)
                    elif 'norm' in name:
                        weight_norms.setdefault('norm', []).append(w_norm)
                        weight_maxes.setdefault('norm', []).append(w_max)

            log_data = {
                "step": step,
                "total_training_time": total_training_time,
                "train/loss": debiased_loss,
                "train/lrm": lrm,
                "train/dt": dt,
                "train/epoch": current_epoch,
                # Gradient norms
                "grad/total_norm": total_grad_norm,
                "grad/fusion_norm": sum(grad_norms.get('fusion', [0])) / max(len(grad_norms.get('fusion', [1])), 1),
                "grad/attn_norm": sum(grad_norms.get('attn', [0])) / max(len(grad_norms.get('attn', [1])), 1),
                "grad/mlp_norm": sum(grad_norms.get('mlp', [0])) / max(len(grad_norms.get('mlp', [1])), 1),
                "grad/norm_norm": sum(grad_norms.get('norm', [0])) / max(len(grad_norms.get('norm', [1])), 1),
                # Weight norms
                "weight/fusion_norm": sum(weight_norms.get('fusion', [0])) / max(len(weight_norms.get('fusion', [1])), 1),
                "weight/attn_norm": sum(weight_norms.get('attn', [0])) / max(len(weight_norms.get('attn', [1])), 1),
                "weight/mlp_norm": sum(weight_norms.get('mlp', [0])) / max(len(weight_norms.get('mlp', [1])), 1),
                "weight/norm_norm": sum(weight_norms.get('norm', [0])) / max(len(weight_norms.get('norm', [1])), 1),
                # Weight max values (can indicate overflow)
                "weight/fusion_max": max(weight_maxes.get('fusion', [0])),
                "weight/attn_max": max(weight_maxes.get('attn', [0])),
                "weight/mlp_max": max(weight_maxes.get('mlp', [0])),
                "weight/norm_max": max(weight_maxes.get('norm', [0])),
            }
            # Per-step accuracies
            for i, acc in enumerate(debiased_accs):
                log_data[f"train/acc_step{i}"] = acc
            # Per-step losses (equivalent to per-head losses in Medusa)
            for i, sl in enumerate(debiased_step_losses):
                log_data[f"train/step{i}_loss"] = sl
            wandb_run.log(log_data)

    # Final save
    if master_process:
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        checkpoint = {
            'step': num_iterations,
            **model.get_draft_state_dict(),
            'optimizers': [opt.state_dict() for opt in optimizers],
            'config': user_config,
        }
        torch.save(checkpoint, os.path.join(final_dir, "eagle_draft.pt"))
        print0(f"\nSaved final checkpoint to {final_dir}")

        # Save final metrics
        final_data = {
            'final_loss': debiased_loss,
            'final_accuracies': debiased_accs,
            'total_training_time': total_training_time,
            'num_iterations': num_iterations,
        }
        with open(os.path.join(output_dir, "final_loss.json"), 'w') as f:
            json.dump(final_data, f, indent=2)

    # -------------------------------------------------------------------------
    # Step accuracy evaluation on validation data

    if val_data is not None:
        print0("\n" + "=" * 50)
        print0("Running step accuracy evaluation on validation data...")
        print0("=" * 50)

        with torch.no_grad(), autocast_ctx:
            recall_rates, total_counts = run_eagle_accuracy_eval(
                model=model,
                tokenizer=tokenizer,
                val_data=val_data,
                device=device,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
                num_draft_steps=args.num_steps,
                max_samples=min(200, len(val_data)),
                max_steps=50,
                topk=100,
            )

        # Print and save results (only on master process)
        if master_process:
            print0("\nEAGLE Step Recall Summary (top-1 / top-10 / top-100):")
            for step_idx in range(args.num_steps):
                r1 = recall_rates[f"step_{step_idx}"]["1"]
                r10 = recall_rates[f"step_{step_idx}"]["10"]
                r100 = recall_rates[f"step_{step_idx}"]["100"]
                print0(f"  Step {step_idx}: {r1:.3f} / {r10:.3f} / {r100:.3f}")

            # Save step accuracy results
            step_acc_data = {
                "checkpoint": output_dir,
                "model_name": args.base_model,
                "num_samples": min(200, len(val_data)),
                "max_steps": 50,
                "num_draft_steps": args.num_steps,
                "config": {
                    "multi_layer_indices": list(model.multi_layer_indices),
                },
                "total_predictions": total_counts,
                "recall": recall_rates,
            }
            step_acc_path = os.path.join(output_dir, "eagle_step_acc.json")
            with open(step_acc_path, 'w') as f:
                json.dump(step_acc_data, f, indent=2)
            print0(f"\nSaved step accuracy to {step_acc_path}")

            # Log to wandb
            wandb_log_data = {}
            for step_idx in range(min(args.num_steps, 7)):  # Log first 7 steps
                wandb_log_data[f"step_acc/step{step_idx}_top1"] = recall_rates[f"step_{step_idx}"]["1"]
                wandb_log_data[f"step_acc/step{step_idx}_top10"] = recall_rates[f"step_{step_idx}"]["10"]
            wandb_run.log(wandb_log_data)

    # Summary
    print0("\n" + "=" * 50)
    print0("Training complete!")
    print0(f"Total training time: {total_training_time/60:.2f} minutes")
    print0(f"Final loss: {debiased_loss:.6f}")
    print0(f"Output directory: {output_dir}")
    print0("=" * 50)

    wandb_run.finish()
    compute_cleanup()
