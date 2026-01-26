"""
Train the HST learned retrieval module.

The retrieval module predicts likely next tokens from TOKEN IDs (not hidden states).
It has its own learned embedding layer that captures token co-occurrence patterns.

Two-phase training:
1. Phase 1 - Pretraining (1B tokens from FineWeb-Edu):
   - Extract (context_token_ids, next_token_id) pairs by sliding window
   - Train the embedding layer + 3-layer MLP end-to-end
   - Loss: Cross-entropy with label smoothing (0.1)

2. Phase 2 - Fine-tuning (100k samples from WildChat distillations):
   - Adapts the module to conversational/instruction-following patterns
   - Lower learning rate (1e-5)

Usage:
    # Single GPU training
    uv run python -m scripts.hst_train_retrieval --phase 1 --tokens 100000000

    # Multi-GPU training with torchrun
    uv run torchrun --standalone --nproc_per_node=8 -m scripts.hst_train_retrieval --phase 1 --tokens 100000000

    # Use local parquet files from ~/.cache/nanochat/base_data/
    uv run torchrun --standalone --nproc_per_node=8 -m scripts.hst_train_retrieval --phase 1 --use-local-parquet --tokens 100000000
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from tqdm import tqdm
import time
import json

from transformers import AutoTokenizer


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Train HST retrieval module")

    # Training phase
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=1,
        help="Training phase: 1=pretrain, 2=finetune",
    )

    # Data
    parser.add_argument(
        "--tokens",
        type=int,
        default=1_000_000_000,
        help="Number of tokens to train on (phase 1)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100_000,
        help="Number of samples for fine-tuning (phase 2)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training data (jsonl for phase 2, or directory of parquet files)",
    )
    parser.add_argument(
        "--use-local-parquet",
        action="store_true",
        help="Use local parquet files from ~/.cache/nanochat/base_data/ instead of streaming",
    )

    # Model
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-3-1b-it",
        help="Base model (used for tokenizer only)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for 3-layer MLP",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=4,
        help="Number of context tokens (K)",
    )
    parser.add_argument(
        "--svd-rank",
        type=int,
        default=64,
        help="SVD rank for tied embeddings (input and output)",
    )

    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 1e-4 for phase 1, 1e-5 for phase 2)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Number of warmup steps",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1000,
        help="Evaluate every N steps",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


class FineWebEduDataset(IterableDataset):
    """
    Streaming dataset for FineWeb-Edu pretraining.

    Yields (context_tokens, next_token) pairs by sliding a window
    over tokenized documents.
    """

    def __init__(
        self,
        tokenizer,
        context_window: int = 4,
        max_tokens: int = 1_000_000_000,
        num_shards: int = 100,
    ):
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.num_shards = num_shards

    def __iter__(self):
        from datasets import load_dataset

        # Stream FineWeb-Edu
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        tokens_seen = 0
        for sample in dataset:
            text = sample["text"]

            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # Slide window to create training pairs
            for i in range(self.context_window, len(tokens)):
                if tokens_seen >= self.max_tokens:
                    return

                context = tokens[i - self.context_window : i]
                target = tokens[i]

                yield torch.tensor(context), torch.tensor(target)
                tokens_seen += 1


class LocalParquetDataset(IterableDataset):
    """
    Dataset that reads from local parquet files.

    Uses parquet files from ~/.cache/nanochat/base_data/ or a custom path.
    Supports distributed training by sharding files across ranks.
    """

    def __init__(
        self,
        tokenizer,
        context_window: int = 4,
        max_tokens: int = 1_000_000_000,
        data_path: str = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.rank = rank
        self.world_size = world_size

        if data_path is None:
            self.data_dir = Path.home() / ".cache" / "nanochat" / "base_data"
        else:
            self.data_dir = Path(data_path)

        all_parquet_files = sorted(self.data_dir.glob("shard_*.parquet"))
        if not all_parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")

        # Shard files across ranks
        self.parquet_files = [f for i, f in enumerate(all_parquet_files) if i % world_size == rank]
        if rank == 0:
            print(f"Found {len(all_parquet_files)} total parquet files, {len(self.parquet_files)} for this rank")

    def __iter__(self):
        import pandas as pd

        tokens_seen = 0
        for parquet_file in self.parquet_files:
            if tokens_seen >= self.max_tokens:
                return

            df = pd.read_parquet(parquet_file)

            for _, row in df.iterrows():
                if tokens_seen >= self.max_tokens:
                    return

                text = row["text"]

                # Tokenize
                tokens = self.tokenizer.encode(text, add_special_tokens=False)

                # Slide window to create training pairs
                for i in range(self.context_window, len(tokens)):
                    if tokens_seen >= self.max_tokens:
                        return

                    context = tokens[i - self.context_window : i]
                    target = tokens[i]

                    yield torch.tensor(context), torch.tensor(target)
                    tokens_seen += 1


class WildChatDataset(IterableDataset):
    """
    Dataset for WildChat fine-tuning.

    Loads from JSONL file with format: {"messages": [...]}

    For fine-tuning on model outputs (distillation data), we only train on
    assistant responses to learn the model's generation patterns.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        context_window: int = 4,
        max_samples: int = 100_000,
        assistant_only: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.max_samples = max_samples
        self.assistant_only = assistant_only
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        samples_seen = 0
        line_idx = 0

        with open(self.data_path) as f:
            for line in f:
                # Shard across workers
                if line_idx % self.world_size != self.rank:
                    line_idx += 1
                    continue
                line_idx += 1

                if samples_seen >= self.max_samples:
                    return

                data = json.loads(line)

                # Extract text based on format
                if "messages" in data:
                    if self.assistant_only:
                        # Only train on assistant responses (model outputs)
                        # This teaches retrieval to predict what the model generates
                        texts = [
                            m.get("content", "")
                            for m in data["messages"]
                            if m.get("role") == "assistant"
                        ]
                    else:
                        # Use all messages
                        texts = [m.get("content", "") for m in data["messages"]]
                elif "text" in data:
                    texts = [data["text"]]
                else:
                    continue

                # Process each text segment
                for text in texts:
                    if not text:
                        continue

                    # Tokenize
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)

                    # Create training pairs by sliding window
                    for i in range(self.context_window, len(tokens)):
                        if samples_seen >= self.max_samples:
                            return

                        context = tokens[i - self.context_window : i]
                        target = tokens[i]

                        yield torch.tensor(context), torch.tensor(target)
                        samples_seen += 1


def create_retrieval_module(args, vocab_size: int):
    """Create and initialize the retrieval module."""
    from nanochat.hst.retrieval import RetrievalMLP, load_svd_basis

    module = RetrievalMLP(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        context_window=args.context_window,
        svd_rank=args.svd_rank,
    )

    # Load SVD to initialize the tied embedding (will be fine-tuned during training)
    try:
        compressed_vocab = load_svd_basis(rank=args.svd_rank, model_name="gemma")
        module.load_svd(compressed_vocab)
        print(f"Loaded SVD basis (rank {args.svd_rank}) - tied embeddings will be fine-tuned")
    except FileNotFoundError:
        print(
            f"Warning: SVD basis not found. Run scripts/hst_compute_svd.py first."
        )
        print("Continuing with random initialization (for testing only).")
        # Initialize with random for testing
        with torch.no_grad():
            module.svd_embedding.normal_(0, 0.01)
        module._svd_initialized = True

    return module


def train_step(
    module: nn.Module,
    batch: tuple,
    optimizer: torch.optim.Optimizer,
    scheduler,
    label_smoothing: float,
    device: str,
    scaler=None,
):
    """Single training step."""
    context_ids, targets = batch
    context_ids = context_ids.to(device)
    targets = targets.to(device)

    # Forward through retrieval module (takes token IDs directly)
    if scaler is not None:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = module(context_ids)  # [B, vocab_size]
            loss = F.cross_entropy(
                logits, targets, label_smoothing=label_smoothing
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        logits = module(context_ids)
        loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
        loss.backward()
        optimizer.step()

    optimizer.zero_grad()
    if scheduler is not None:
        scheduler.step()

    return loss.item()


def evaluate(
    module: nn.Module,
    dataloader: DataLoader,
    device: str,
    max_batches: int = 100,
):
    """Evaluate Recall@K metrics."""
    module.eval()

    total = 0
    correct_at_1 = 0
    correct_at_5 = 0
    correct_at_10 = 0
    correct_at_50 = 0

    with torch.no_grad():
        for i, (context_ids, targets) in enumerate(dataloader):
            if i >= max_batches:
                break

            context_ids = context_ids.to(device)
            targets = targets.to(device)

            # Module takes token IDs directly
            logits = module(context_ids)

            # Compute Recall@K
            _, top_indices = torch.topk(logits, k=50, dim=-1)

            for j, target in enumerate(targets):
                total += 1
                if target in top_indices[j, :1]:
                    correct_at_1 += 1
                if target in top_indices[j, :5]:
                    correct_at_5 += 1
                if target in top_indices[j, :10]:
                    correct_at_10 += 1
                if target in top_indices[j, :50]:
                    correct_at_50 += 1

    module.train()

    return {
        "recall@1": correct_at_1 / max(total, 1),
        "recall@5": correct_at_5 / max(total, 1),
        "recall@10": correct_at_10 / max(total, 1),
        "recall@50": correct_at_50 / max(total, 1),
    }


def main():
    args = parse_args()

    # Set up distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    # Set default learning rate based on phase
    if args.lr is None:
        args.lr = 1e-4 if args.phase == 1 else 1e-5

    # Set device
    if world_size > 1:
        args.device = f"cuda:{local_rank}"
    elif args.device == "cuda" and torch.cuda.is_available():
        args.device = "cuda:0"

    # Set up output directory
    if args.output_dir is None:
        args.output_dir = Path.home() / ".cache" / "nanochat" / "hst_retrieval"
    else:
        args.output_dir = Path(args.output_dir)
    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"Phase {args.phase} training")
        print(f"Device: {args.device}")
        print(f"World size: {world_size}")
        print(f"Output directory: {args.output_dir}")

    # Load tokenizer only (no need for full base model!)
    if is_main:
        print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    vocab_size = tokenizer.vocab_size
    if is_main:
        print(f"Vocabulary size: {vocab_size}")

    # Create retrieval module (has its own learned embeddings)
    module = create_retrieval_module(args, vocab_size)
    module = module.to(args.device)

    # Wrap in DDP if distributed
    if world_size > 1:
        module = DDP(module, device_ids=[local_rank])

    # Log model info
    raw_module = module.module if hasattr(module, 'module') else module
    num_params = sum(p.numel() for p in raw_module.parameters())
    trainable_params = sum(p.numel() for p in raw_module.parameters() if p.requires_grad)
    if is_main:
        print(f"Model parameters: {num_params:,} total, {trainable_params:,} trainable")

    if args.resume:
        if is_main:
            print(f"Resuming from: {args.resume}")
        raw_module.load_state_dict(torch.load(args.resume, weights_only=True))

    if args.compile:
        module = torch.compile(module)

    # Adjust tokens per GPU for distributed training
    tokens_per_gpu = args.tokens // world_size

    # Create dataset
    if args.phase == 1:
        if args.use_local_parquet:
            dataset = LocalParquetDataset(
                tokenizer=tokenizer,
                context_window=args.context_window,
                max_tokens=tokens_per_gpu,
                data_path=args.data_path,
                rank=rank,
                world_size=world_size,
            )
        else:
            dataset = FineWebEduDataset(
                tokenizer=tokenizer,
                context_window=args.context_window,
                max_tokens=tokens_per_gpu,
            )
    else:
        if args.data_path is None:
            raise ValueError("--data-path required for phase 2 training")
        dataset = WildChatDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            context_window=args.context_window,
            max_samples=args.samples // world_size,
            assistant_only=True,  # Only train on model outputs for distillation
            rank=rank,
            world_size=world_size,
        )

    # Note: Using num_workers=0 for IterableDataset to avoid duplicate data issues
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    # Create a separate eval dataset (only on main process)
    eval_dataloader = None
    if is_main:
        if args.phase == 1:
            if args.use_local_parquet:
                eval_dataset = LocalParquetDataset(
                    tokenizer=tokenizer,
                    context_window=args.context_window,
                    max_tokens=min(args.tokens, 50000),
                    data_path=args.data_path,
                    rank=0,
                    world_size=1,  # eval only on single process
                )
            else:
                eval_dataset = FineWebEduDataset(
                    tokenizer=tokenizer,
                    context_window=args.context_window,
                    max_tokens=50000,
                )
        else:
            eval_dataset = WildChatDataset(
                data_path=args.data_path,
                tokenizer=tokenizer,
                context_window=args.context_window,
                max_samples=min(args.samples, 10000),
                assistant_only=True,
                rank=0,
                world_size=1,
            )

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler with warmup
    total_steps = tokens_per_gpu // args.batch_size

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if "cuda" in args.device else None

    # Training loop
    if is_main:
        print(f"Starting training...")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"Learning rate: {args.lr}")

    module.train()
    step = 0
    total_loss = 0.0
    start_time = time.time()

    pbar = tqdm(dataloader, desc="Training", disable=not is_main)
    for batch in pbar:
        loss = train_step(
            module=module,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            label_smoothing=args.label_smoothing,
            device=args.device,
            scaler=scaler,
        )

        total_loss += loss
        step += 1

        # Update progress bar
        if step % 10 == 0 and is_main:
            avg_loss = total_loss / step
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        # Evaluate (only on main)
        if step % args.eval_every == 0 and is_main:
            eval_module = raw_module  # Use unwrapped module for eval
            metrics = evaluate(eval_module, eval_dataloader, args.device)
            print(f"\nStep {step}: {metrics}")

            # Check target metric
            if metrics["recall@10"] > 0.5:
                print("Target Recall@10 > 50% achieved!")

        # Save checkpoint (only on main)
        if step % args.save_every == 0 and is_main:
            ckpt_path = args.output_dir / f"retrieval_phase{args.phase}_step{step}.pt"
            torch.save(raw_module.state_dict(), ckpt_path)
            print(f"\nSaved checkpoint: {ckpt_path}")

    # Final save (only on main)
    if is_main:
        final_path = args.output_dir / f"retrieval_phase{args.phase}_final.pt"
        torch.save(raw_module.state_dict(), final_path)
        print(f"Saved final model: {final_path}")

        # Final evaluation
        print("\nFinal evaluation:")
        metrics = evaluate(raw_module, eval_dataloader, args.device, max_batches=500)
        print(metrics)

        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed / 3600:.2f} hours")
        print(f"Total steps: {step}")

    # Clean up distributed
    cleanup_distributed()


if __name__ == "__main__":
    main()
