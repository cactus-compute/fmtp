"""
Train Medusa heads for Gemma 3 models.

Train LoRA-based Medusa heads on a frozen Gemma base model for speculative decoding.

Example runs:
    # Single GPU training
    python -m scripts.gemma_medusa_train --data-path data/openhermes.json

    # Multi-GPU training (8x)
    torchrun --standalone --nproc_per_node=8 -m scripts.gemma_medusa_train \
        --data-path data/openhermes.json \
        --wandb-run my_run_name

    # Full configuration example
    torchrun --standalone --nproc_per_node=8 -m scripts.gemma_medusa_train \
        --base-model google/gemma-3-1b-it \
        --medusa-num-heads 4 \
        --medusa-num-layers 1 \
        --lora-rank 64 \
        --device-batch-size 4 \
        --total-batch-size 128 \
        --max-seq-len 2048 \
        --data-path data/openhermes.json \
        --wandb-run gemma_medusa_1b
"""

import argparse
import json
import os
import time
from contextlib import nullcontext
from datetime import datetime

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from nanochat.common import (
    compute_init,
    compute_cleanup,
    get_dist_info,
    print0,
    DummyWandb,
    autodetect_device_type,
    get_base_dir,
)
from nanochat.gemma_medusa import (
    GemmaTokenizerWrapper,
    load_gemma_medusa_model,
    GemmaMedusaModel,
)


# -----------------------------------------------------------------------------
# Data loading

def load_sharegpt_data(filepath):
    """Load ShareGPT-format JSON data.

    Supports both JSONL and JSON array formats.
    Expected format:
    {
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    or:
    {
        "messages": [...]
    }
    """
    conversations = []
    with open(filepath, 'r') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON array format
            data = json.loads(content)
            for item in data:
                if 'conversations' in item:
                    # ShareGPT format with "from"/"value" keys
                    messages = []
                    for turn in item['conversations']:
                        role = turn.get('from', turn.get('role'))
                        content = turn.get('value', turn.get('content'))
                        if role == 'human':
                            role = 'user'
                        elif role == 'gpt':
                            role = 'assistant'
                        messages.append({'role': role, 'content': content})
                    conversations.append({'messages': messages})
                elif 'messages' in item:
                    conversations.append(item)
        else:
            # JSONL format
            for line in content.split('\n'):
                if line.strip():
                    item = json.loads(line)
                    if 'conversations' in item:
                        messages = []
                        for turn in item['conversations']:
                            role = turn.get('from', turn.get('role'))
                            content = turn.get('value', turn.get('content'))
                            if role == 'human':
                                role = 'user'
                            elif role == 'gpt':
                                role = 'assistant'
                            messages.append({'role': role, 'content': content})
                        conversations.append({'messages': messages})
                    elif 'messages' in item:
                        conversations.append(item)
    return conversations


def filter_dataset(dataset, tokenizer, max_seq_len, min_valid_tokens=10):
    """
    Filter dataset to remove conversations without enough valid (assistant) tokens.

    Args:
        dataset: List of conversations
        tokenizer: Tokenizer wrapper
        max_seq_len: Maximum sequence length
        min_valid_tokens: Minimum number of valid (non-masked) tokens required

    Returns:
        Filtered dataset
    """
    filtered = []
    skipped = 0
    for doc in dataset:
        ids, mask = tokenizer.render_conversation(doc, max_tokens=max_seq_len)
        # Count valid tokens (mask=1 means assistant content we train on)
        # We check mask[1:] because targets are shifted by 1
        valid_count = sum(mask[1:])
        if valid_count >= min_valid_tokens:
            filtered.append(doc)
        else:
            skipped += 1
    return filtered, skipped


def medusa_data_generator(dataset, tokenizer, batch_size, max_seq_len, device, ddp_rank=0, ddp_world_size=1):
    """
    Generate batches of tokenized conversations for Medusa training.

    Yields (inputs, targets) where targets is shifted by 1 from inputs.
    The loss mask is applied via -1 in targets.
    """
    pad_token_id = tokenizer.hf_tokenizer.eos_token_id

    def collate_and_yield(batch):
        nrows = len(batch)
        # Truncate to max_seq_len and find max length
        batch = [(ids[:max_seq_len], mask[:max_seq_len]) for ids, mask in batch]
        ncols = max(len(ids) for ids, mask in batch) - 1  # seq of n creates inputs/targets of n-1

        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)  # -1 is ignore index

        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            # Apply mask: -1 where mask is 0
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets

        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets

    # Iterate over dataset in epochs
    epoch = 0
    while True:
        batch = []
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc, max_tokens=max_seq_len)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch), epoch
                batch = []
        epoch += 1


# -----------------------------------------------------------------------------
# Main training script

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medusa heads for Gemma 3")

    # Logging
    parser.add_argument("--wandb-run", type=str, default="dummy",
                        help="wandb run name ('dummy' disables wandb logging)")
    # Runtime
    parser.add_argument("--device-type", type=str, default="",
                        help="cuda|cpu|mps (empty = autodetect)")

    # Model configuration
    parser.add_argument("--base-model", type=str, default="google/gemma-3-270m-it",
                        help="HuggingFace base model name")
    parser.add_argument("--medusa-num-heads", type=int, default=4,
                        help="Number of Medusa prediction heads")
    parser.add_argument("--medusa-num-layers", type=int, default=2,
                        help="Number of ResBlock layers per head")
    parser.add_argument("--lora-rank", type=int, default=256,
                        help="LoRA rank for Medusa heads")
    parser.add_argument("--lora-alpha", type=int, default=512,
                        help="LoRA alpha scaling")
    parser.add_argument("--max-seq-len", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--zero-init-mtp-mlp", action="store_true", default=True,
                        help="Zero-initialize ResBlock MLP weights")
    parser.add_argument("--use-head-mixer", action="store_true",
                        help="Enable cross-head MLP mixer for improved multi-token prediction")
    parser.add_argument("--mixer-hidden", type=int, default=16,
                        help="Hidden dimension for cross-head mixer MLP (default: 16)")

    # Performance optimization
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for potential speedup (experimental)")

    # Medusa loss configuration
    parser.add_argument("--medusa-loss-weight", type=float, default=1.0,
                        help="Weight for Medusa head losses (constant or decay base)")
    parser.add_argument("--medusa-loss-scheme", type=str, default="constant",
                        choices=["constant", "decay"],
                        help="Weighting scheme: constant (all heads same) or decay (weight^k)")

    # Training horizon
    parser.add_argument("--num-iterations", type=int, default=1030,
                        help="Explicit number of optimization steps (-1 = use num_epochs)")
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Number of training epochs (used if num_iterations=-1)")

    # Optimization
    parser.add_argument("--device-batch-size", type=int, default=4,
                        help="Per-device batch size (4 works for A100-40GB with seq_len=1024)")
    parser.add_argument("--total-batch-size", type=int, default=96,
                        help="Total batch size (examples) for gradient accumulation")
    parser.add_argument("--matrix-lr", type=float, default=0.01,
                        help="Learning rate for matrix params (Muon)")
    parser.add_argument("--proj-lr", type=float, default=0.001,
                        help="Learning rate for projection params (Adam)")
    parser.add_argument("--weight-decay", type=float, default=0.2,
                        help="Weight decay for Muon optimizer (ResBlock params)")
    parser.add_argument("--adam-weight-decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer (LoRA params)")
    parser.add_argument("--adam-beta1", type=float, default=0.8,
                        help="Adam beta1 for projection params")
    parser.add_argument("--adam-beta2", type=float, default=0.95,
                        help="Adam beta2 for projection params")
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
                        help="Ratio of iterations for LR warmup")
    parser.add_argument("--warmdown-ratio", type=float, default=0.95,
                        help="Ratio of iterations for LR warmdown")
    parser.add_argument("--final-lr-frac", type=float, default=0.0,
                        help="Final LR as fraction of initial LR")

    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to training data (JSON/JSONL in ShareGPT format)")
    parser.add_argument("--val-data-path", type=str, default=None,
                        help="Path to validation data (optional)")
    parser.add_argument("--skip-filter", action="store_true",
                        help="Skip dataset filtering (use if data is pre-filtered)")
    parser.add_argument("--use-chunked-loss", action="store_true", default=True,
                        help="Compute loss in chunks to reduce memory (allows larger batch sizes)")
    parser.add_argument("--chunk-size", type=int, default=128,
                        help="Chunk size for chunked loss computation (default: 128)")

    # Evaluation
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Evaluate validation loss every N steps (-1 = disable)")
    parser.add_argument("--eval-steps", type=int, default=10,
                        help="Number of batches for validation evaluation")
    parser.add_argument("--val-samples", type=int, default=960,
                        help="Number of samples to split off for validation (default: 960 = 8 batches of 120)")
    parser.add_argument("--log-every", type=int, default=20,
                        help="Log detailed metrics (grad norms, weight norms) to wandb every N steps")
    parser.add_argument("--save-every", type=int, default=500,
                        help="Save checkpoint every N steps (-1 = only at end)")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from checkpoint directory")

    args = parser.parse_args()
    user_config = vars(args).copy()

    # -----------------------------------------------------------------------------
    # Compute initialization

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

    # wandb logging init
    use_dummy_wandb = args.wandb_run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
        project="gemma-medusa",
        name=args.wandb_run,
        config=user_config,
    )

    # -----------------------------------------------------------------------------
    # Load model and tokenizer

    print0(f"Loading base model: {args.base_model}")
    mixer_str = f", head_mixer={args.mixer_hidden}" if args.use_head_mixer else ""
    print0(f"Medusa config: {args.medusa_num_heads} heads, {args.medusa_num_layers} layers, lora_rank={args.lora_rank}{mixer_str}")

    model = load_gemma_medusa_model(
        model_name=args.base_model,
        medusa_num_heads=args.medusa_num_heads,
        medusa_num_layers=args.medusa_num_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=device,
        dtype=torch.bfloat16,
        freeze_base=True,
        zero_init_mlp=args.zero_init_mtp_mlp,
        use_head_mixer=args.use_head_mixer,
        mixer_hidden=args.mixer_hidden,
    )
    tokenizer = GemmaTokenizerWrapper(args.base_model)

    num_base_params = sum(p.numel() for p in model.base_model.parameters())
    num_medusa_params = model.get_medusa_param_count()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print0(f"Base model parameters: {num_base_params:,}")
    print0(f"Medusa parameters: {num_medusa_params:,}")
    print0(f"Trainable parameters: {trainable_params:,}")
    print0(f"Vocab size: {tokenizer.get_vocab_size()}")
    if args.use_chunked_loss:
        print0(f"Using chunked loss (chunk_size={args.chunk_size}) for memory efficiency")

    # Optional torch.compile
    if args.compile:
        print0("Compiling model with torch.compile...")
        model = torch.compile(model)

    # -----------------------------------------------------------------------------
    # Load data

    print0(f"Loading training data from: {args.data_path}")
    train_data = load_sharegpt_data(args.data_path)
    print0(f"Loaded {len(train_data)} training conversations")

    # Filter out conversations with 0 valid tokens (would cause NaN in cross-entropy)
    # This happens when user message is so long it gets truncated before assistant response
    # Skip if data is pre-filtered (e.g., from scripts/download_openhermes.py)
    if not args.skip_filter:
        print0("Filtering conversations with 0 valid assistant tokens...")
        train_data, skipped = filter_dataset(train_data, tokenizer, args.max_seq_len, min_valid_tokens=1)
        print0(f"Filtered to {len(train_data)} conversations (skipped {skipped})")
    else:
        print0("Skipping dataset filtering (--skip-filter)")

    # Split off validation data
    val_data = None
    if args.val_data_path:
        # Use explicit validation file if provided
        print0(f"Loading validation data from: {args.val_data_path}")
        val_data = load_sharegpt_data(args.val_data_path)
        print0(f"Loaded {len(val_data)} validation conversations")
        if not args.skip_filter:
            val_data, val_skipped = filter_dataset(val_data, tokenizer, args.max_seq_len, min_valid_tokens=1)
            print0(f"Filtered to {len(val_data)} validation conversations (skipped {val_skipped})")
    elif args.val_samples > 0:
        # Split off validation samples from training data
        import random
        random.seed(42)  # Deterministic split
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        val_indices = indices[:args.val_samples]
        train_indices = indices[args.val_samples:]
        val_data = [train_data[i] for i in val_indices]
        train_data = [train_data[i] for i in train_indices]
        print0(f"Split off {len(val_data)} validation samples from training data")
        print0(f"Remaining training samples: {len(train_data)}")

    # -----------------------------------------------------------------------------
    # Calculate training schedule

    examples_per_step = args.device_batch_size * ddp_world_size
    assert args.total_batch_size % examples_per_step == 0, \
        f"Total batch size {args.total_batch_size} must be divisible by examples per step {examples_per_step}"
    grad_accum_steps = args.total_batch_size // examples_per_step
    print0(f"Batch size: {args.device_batch_size} x {ddp_world_size} GPUs = {examples_per_step} examples/step")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")
    print0(f"Effective batch size: {args.total_batch_size}")

    if args.num_iterations > 0:
        num_iterations = args.num_iterations
        print0(f"Using user-provided number of iterations: {num_iterations:,}")
    else:
        # Calculate from epochs
        iterations_per_epoch = len(train_data) // args.total_batch_size
        num_iterations = iterations_per_epoch * args.num_epochs
        print0(f"Calculated iterations from {args.num_epochs} epochs: {num_iterations:,}")

    # -----------------------------------------------------------------------------
    # Setup optimizer (nanochat-style: Muon + AdamW)

    adam_betas = (args.adam_beta1, args.adam_beta2)
    optimizers = model.setup_optimizers(
        matrix_lr=args.matrix_lr,
        proj_lr=args.proj_lr,
        weight_decay=args.weight_decay,
        adam_weight_decay=args.adam_weight_decay,
        adam_betas=adam_betas,
    )

    # Learning rate scheduler (warmup -> constant -> warmdown)
    def get_lr_multiplier(it):
        warmup_iters = round(args.warmup_ratio * num_iterations)
        warmdown_iters = round(args.warmdown_ratio * num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        elif it <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * args.final_lr_frac

    # -----------------------------------------------------------------------------
    # Setup data loaders

    train_loader = medusa_data_generator(
        train_data, tokenizer, args.device_batch_size, args.max_seq_len, device,
        ddp_rank=ddp_rank, ddp_world_size=ddp_world_size
    )

    def build_val_loader():
        if val_data is None:
            return None
        return medusa_data_generator(
            val_data, tokenizer, args.device_batch_size, args.max_seq_len, device,
            ddp_rank=ddp_rank, ddp_world_size=ddp_world_size
        )

    # -----------------------------------------------------------------------------
    # Resume from checkpoint

    start_step = 0
    if args.resume_from:
        checkpoint_path = os.path.join(args.resume_from, "medusa_heads.pt")
        if os.path.exists(checkpoint_path):
            print0(f"Resuming from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.medusa_heads.load_state_dict(checkpoint['medusa_heads'])
            # Load optimizer states if available
            if 'optimizers' in checkpoint:
                for opt, state in zip(optimizers, checkpoint['optimizers']):
                    opt.load_state_dict(state)
            start_step = checkpoint['step']
            print0(f"Resumed from step {start_step}")
        else:
            print0(f"WARNING: Checkpoint not found at {checkpoint_path}, starting fresh")

    # -----------------------------------------------------------------------------
    # Setup output directory

    if args.output_dir is None:
        base_dir = get_base_dir()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_short = args.base_model.split('/')[-1]
        args.output_dir = os.path.join(base_dir, f"gemma_medusa_{model_short}_{timestamp}")

    if master_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # Save config
        with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
            json.dump(user_config, f, indent=2)
        print0(f"Output directory: {args.output_dir}")

    # -----------------------------------------------------------------------------
    # Training loop

    print0("\n" + "=" * 50)
    print0("Starting training")
    print0("=" * 50 + "\n")

    model.train()
    smooth_loss = 0.0
    smooth_main_loss = 0.0
    smooth_head_losses = [0.0] * args.medusa_num_heads
    total_training_time = 0.0
    current_epoch = 0

    for step in range(start_step, num_iterations):
        last_step = step == num_iterations - 1

        # Evaluate validation loss
        if val_data is not None and args.eval_every > 0 and (last_step or step % args.eval_every == 0):
            model.eval()
            val_loader = build_val_loader()
            val_losses = []
            val_main_losses = []
            val_head_losses_accum = [[] for _ in range(args.medusa_num_heads)]

            for eval_step in range(args.eval_steps):
                (val_inputs, val_targets), _ = next(val_loader)
                with torch.no_grad(), autocast_ctx:
                    main_loss, head_losses = model(
                        val_inputs, val_targets, return_medusa=True,
                        use_chunked_loss=args.use_chunked_loss, chunk_size=args.chunk_size
                    )
                    total_loss = main_loss.clone()
                    for k, head_loss in enumerate(head_losses):
                        if args.medusa_loss_scheme == "decay":
                            weight = args.medusa_loss_weight ** (k + 1)
                        else:
                            weight = args.medusa_loss_weight
                        total_loss = total_loss + weight * head_loss
                val_losses.append(total_loss)
                val_main_losses.append(main_loss)
                for k, hl in enumerate(head_losses):
                    val_head_losses_accum[k].append(hl)

            val_loss = torch.stack(val_losses).mean()
            val_main_loss = torch.stack(val_main_losses).mean()
            val_head_losses_mean = [torch.stack(hls).mean() for hls in val_head_losses_accum]

            if ddp:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(val_main_loss, op=dist.ReduceOp.AVG)
                for hl in val_head_losses_mean:
                    dist.all_reduce(hl, op=dist.ReduceOp.AVG)

            val_loss_item = val_loss.item()
            val_main_loss_item = val_main_loss.item()
            val_head_losses_items = [hl.item() for hl in val_head_losses_mean]

            print0(f"Step {step:05d} | Val loss: {val_loss_item:.6f} | Val main: {val_main_loss_item:.6f}")
            log_data = {
                "step": step,
                "val/loss": val_loss_item,
                "val/main_loss": val_main_loss_item,
            }
            for k, hl in enumerate(val_head_losses_items):
                log_data[f"val/head{k}_loss"] = hl
            wandb_run.log(log_data)
            model.train()

        # Save checkpoint
        if master_process and args.save_every > 0 and (last_step or (step > 0 and step % args.save_every == 0)):
            checkpoint_dir = os.path.join(args.output_dir, f"step_{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                'step': step,
                'medusa_heads': model.medusa_heads.state_dict(),
                'optimizers': [opt.state_dict() for opt in optimizers],
                'config': user_config,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, "medusa_heads.pt"))
            print0(f"Saved checkpoint to {checkpoint_dir}")

        if last_step:
            break

        # Training step
        synchronize()
        t0 = time.time()

        total_loss = torch.tensor(0.0, device=device)
        total_main_loss = torch.tensor(0.0, device=device)
        total_head_losses = [torch.tensor(0.0, device=device) for _ in range(args.medusa_num_heads)]

        for micro_step in range(grad_accum_steps):
            (train_inputs, train_targets), epoch = next(train_loader)
            current_epoch = epoch

            with autocast_ctx:
                main_loss, head_losses = model(
                    train_inputs, train_targets, return_medusa=True,
                    use_chunked_loss=args.use_chunked_loss, chunk_size=args.chunk_size
                )

                # Compute total loss with weighting
                loss = main_loss
                for k, head_loss in enumerate(head_losses):
                    if args.medusa_loss_scheme == "decay":
                        weight = args.medusa_loss_weight ** (k + 1)
                    else:
                        weight = args.medusa_loss_weight
                    loss = loss + weight * head_loss

                # Track for logging
                total_loss += loss.detach()
                total_main_loss += main_loss.detach()
                for k, hl in enumerate(head_losses):
                    total_head_losses[k] += hl.detach()

            # Normalize and backward
            (loss / grad_accum_steps).backward()

        # Average over accumulation steps
        total_loss /= grad_accum_steps
        total_main_loss /= grad_accum_steps
        total_head_losses = [hl / grad_accum_steps for hl in total_head_losses]

        # Learning rate schedule (nanochat-style linear warmdown)
        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group['lr'] = group['initial_lr'] * lrm

        # Compute gradient norms BEFORE optimizer step (for logging)
        if step % args.log_every == 0:
            grad_norms = {}
            total_grad_norm_sq = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm_sq += grad_norm ** 2
                    if 'lora_A' in name:
                        grad_norms.setdefault('lora_A', []).append(grad_norm)
                    elif 'lora_B' in name:
                        grad_norms.setdefault('lora_B', []).append(grad_norm)
                    elif 'blocks' in name:
                        grad_norms.setdefault('resblock', []).append(grad_norm)
            total_grad_norm = total_grad_norm_sq ** 0.5
        else:
            grad_norms = {}
            total_grad_norm = 0.0

        # Gradient step
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        synchronize()
        t1 = time.time()
        dt = t1 - t0
        if step > 10:
            total_training_time += dt

        # Logging with EMA
        ema_beta = 0.9
        loss_item = total_loss.item()
        main_loss_item = total_main_loss.item()
        head_losses_items = [hl.item() for hl in total_head_losses]

        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_item
        smooth_main_loss = ema_beta * smooth_main_loss + (1 - ema_beta) * main_loss_item
        for k, hl in enumerate(head_losses_items):
            smooth_head_losses[k] = ema_beta * smooth_head_losses[k] + (1 - ema_beta) * hl

        # Debias EMA
        debias = 1 - ema_beta ** (step - start_step + 1)
        debiased_loss = smooth_loss / debias
        debiased_main_loss = smooth_main_loss / debias
        debiased_head_losses = [shl / debias for shl in smooth_head_losses]

        pct_done = 100 * step / num_iterations

        # Calculate ETA
        steps_done = step - start_step
        if steps_done > 10:
            avg_time_per_step = total_training_time / (steps_done - 10)
            remaining_steps = num_iterations - step
            eta_seconds = remaining_steps * avg_time_per_step
            eta_str = f" | eta: {eta_seconds/60:.1f}m"
        else:
            eta_str = ""

        print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_loss:.6f} | main: {debiased_main_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | epoch: {current_epoch} | total time: {total_training_time/60:.2f}m{eta_str}")

        if step % args.log_every == 0:
            # Compute weight norms for logging (grad norms already computed above)
            weight_norms = {}
            weight_maxes = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    w_norm = param.norm().item()
                    w_max = param.abs().max().item()
                    if 'lora_A' in name:
                        weight_norms.setdefault('lora_A', []).append(w_norm)
                        weight_maxes.setdefault('lora_A', []).append(w_max)
                    elif 'lora_B' in name:
                        weight_norms.setdefault('lora_B', []).append(w_norm)
                        weight_maxes.setdefault('lora_B', []).append(w_max)
                    elif 'blocks' in name:
                        weight_norms.setdefault('resblock', []).append(w_norm)
                        weight_maxes.setdefault('resblock', []).append(w_max)

            log_data = {
                "step": step,
                "total_training_time": total_training_time,
                "train/loss": debiased_loss,
                "train/main_loss": debiased_main_loss,
                "train/lrm": lrm,
                "train/dt": dt,
                "train/epoch": current_epoch,
                # Gradient norms
                "grad/total_norm": total_grad_norm,
                "grad/lora_A_norm": sum(grad_norms.get('lora_A', [0])) / max(len(grad_norms.get('lora_A', [1])), 1),
                "grad/lora_B_norm": sum(grad_norms.get('lora_B', [0])) / max(len(grad_norms.get('lora_B', [1])), 1),
                "grad/resblock_norm": sum(grad_norms.get('resblock', [0])) / max(len(grad_norms.get('resblock', [1])), 1),
                # Weight norms
                "weight/lora_A_norm": sum(weight_norms.get('lora_A', [0])) / max(len(weight_norms.get('lora_A', [1])), 1),
                "weight/lora_B_norm": sum(weight_norms.get('lora_B', [0])) / max(len(weight_norms.get('lora_B', [1])), 1),
                "weight/resblock_norm": sum(weight_norms.get('resblock', [0])) / max(len(weight_norms.get('resblock', [1])), 1),
                # Weight max values (can indicate overflow)
                "weight/lora_A_max": max(weight_maxes.get('lora_A', [0])),
                "weight/lora_B_max": max(weight_maxes.get('lora_B', [0])),
                "weight/resblock_max": max(weight_maxes.get('resblock', [0])),
            }
            for k, hl in enumerate(debiased_head_losses):
                log_data[f"train/head{k}_loss"] = hl
            wandb_run.log(log_data)

    # -----------------------------------------------------------------------------
    # Final save

    if master_process:
        final_checkpoint_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_checkpoint_dir, exist_ok=True)
        checkpoint = {
            'step': num_iterations,
            'medusa_heads': model.medusa_heads.state_dict(),
            'optimizers': [opt.state_dict() for opt in optimizers],
            'config': user_config,
        }
        torch.save(checkpoint, os.path.join(final_checkpoint_dir, "medusa_heads.pt"))
        print0(f"\nSaved final checkpoint to {final_checkpoint_dir}")

        # Save final loss for ablation orchestration
        final_loss_data = {
            'final_loss': debiased_loss,
            'final_main_loss': debiased_main_loss,
            'final_head_losses': debiased_head_losses,
            'total_training_time': total_training_time,
            'num_iterations': num_iterations,
        }
        with open(os.path.join(args.output_dir, "final_loss.json"), 'w') as f:
            json.dump(final_loss_data, f, indent=2)
        print0(f"Saved final loss to {os.path.join(args.output_dir, 'final_loss.json')}")

    # Summary
    print0("\n" + "=" * 50)
    print0("Training complete!")
    print0(f"Total training time: {total_training_time/60:.2f} minutes")
    print0(f"Final loss: {debiased_loss:.6f}")
    print0(f"Output directory: {args.output_dir}")
    print0("=" * 50)

    # Cleanup
    wandb_run.finish()
    compute_cleanup()
