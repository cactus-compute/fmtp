#!/usr/bin/env python3
"""
Ablation study orchestrator for Gemma Medusa training.

Runs a sequential sweep of hyperparameters, using results from earlier
runs to inform later sweeps (greedy selection).

Usage:
    python -m scripts.run_ablations --data-path data/openhermes_filtered.json

    # Or with torchrun for multi-GPU
    python -m scripts.run_ablations --data-path data/openhermes_filtered.json --nproc 8
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class AblationConfig:
    """Single ablation run configuration."""
    name: str
    zero_init_mlp: bool = False
    proj_lr: float = 0.002  # baseline from user's command
    lora_rank: int = 32     # baseline from user's command
    lora_alpha: int | None = None  # None = same as rank
    medusa_num_layers: int = 1

    def to_args(self) -> list[str]:
        """Convert to CLI arguments."""
        args = [
            f"--proj-lr={self.proj_lr}",
            f"--lora-rank={self.lora_rank}",
            f"--medusa-num-layers={self.medusa_num_layers}",
        ]
        if self.zero_init_mlp:
            args.append("--zero-init-mtp-mlp")
        if self.lora_alpha is not None:
            args.append(f"--lora-alpha={self.lora_alpha}")
        return args


def generate_ablation_configs() -> list[AblationConfig]:
    """
    Generate ablation configurations in sweep order.

    Sweep order (greedy selection):
    1. Zero-init MLP: False vs True (pick best)
    2. LR sweep: 0.001, 0.002, 0.004 (pick best)
    3. Rank sweep: 32, 64, 128 (pick best)
    4. Alpha sweep: rank vs 2*rank (pick best)
    5. Layers sweep: 0, 1, 2 (pick best)
    """
    configs = []

    # Phase 1: Zero-init ablation
    configs.append(AblationConfig(name="baseline", zero_init_mlp=False))
    configs.append(AblationConfig(name="zero_init", zero_init_mlp=True))

    # Phase 2: LR sweep (will use best zero_init from phase 1)
    # baseline is 0.002, test 0.001 and 0.004
    for lr in [0.001, 0.004]:
        configs.append(AblationConfig(name=f"lr_{lr}", proj_lr=lr))

    # Phase 3: Rank sweep (will use best zero_init + lr)
    # baseline is 32, test 64 and 128
    for rank in [64, 128]:
        configs.append(AblationConfig(name=f"rank_{rank}", lora_rank=rank))

    # Phase 4: Alpha sweep (will use best config so far)
    # alpha = 2*rank (need to set dynamically based on best rank)
    configs.append(AblationConfig(name="alpha_2x", lora_alpha=-1))  # -1 = placeholder for 2*rank

    # Phase 5: Layers sweep (will use best config so far)
    for layers in [0, 2]:  # 1 is baseline
        configs.append(AblationConfig(name=f"layers_{layers}", medusa_num_layers=layers))

    return configs


def run_single_ablation(
    config: AblationConfig,
    base_args: list[str],
    output_dir: str,
    use_torchrun: bool = False,
    nproc: int = 1,
) -> dict:
    """
    Run a single ablation experiment.

    Returns dict with final loss and config.
    """
    run_output_dir = os.path.join(output_dir, config.name)

    cmd = []
    if use_torchrun:
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", "-m", "scripts.gemma_medusa_train"]
    else:
        cmd = [sys.executable, "-m", "scripts.gemma_medusa_train"]

    cmd.extend(base_args)
    cmd.extend(config.to_args())
    cmd.extend([f"--output-dir={run_output_dir}"])
    cmd.extend([f"--wandb-run=ablation_{config.name}"])

    print(f"\n{'='*60}")
    print(f"Running ablation: {config.name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Run the training
    result = subprocess.run(cmd, capture_output=False)

    # Parse results from output config
    results_file = os.path.join(run_output_dir, "final", "medusa_heads.pt")
    config_file = os.path.join(run_output_dir, "config.json")

    final_loss = float('inf')
    if os.path.exists(config_file):
        # Try to extract final loss from logs or checkpoint
        # For now, we'll rely on wandb or manual inspection
        pass

    return {
        "name": config.name,
        "config": vars(config),
        "output_dir": run_output_dir,
        "success": result.returncode == 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Gemma Medusa ablation study")

    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to training data (7.5%% of OpenHermes)")
    parser.add_argument("--val-data-path", type=str, default=None,
                        help="Path to validation data")

    # Training config (shared across all ablations)
    parser.add_argument("--num-iterations", type=int, default=-1,
                        help="Number of training iterations per ablation (-1 = use num-epochs)")
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Number of epochs per ablation (used if num-iterations=-1)")
    parser.add_argument("--device-batch-size", type=int, default=5,
                        help="Per-device batch size")
    parser.add_argument("--total-batch-size", type=int, default=120,
                        help="Total batch size")
    parser.add_argument("--max-seq-len", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--eval-every", type=int, default=1000,
                        help="Evaluate every N steps")
    parser.add_argument("--save-every", type=int, default=-1,
                        help="Save checkpoint every N steps (-1 = only at end)")

    # Model
    parser.add_argument("--base-model", type=str, default="google/gemma-3-270m-it",
                        help="Base model name")
    parser.add_argument("--medusa-num-heads", type=int, default=4,
                        help="Number of Medusa heads")

    # Optimizer defaults (can be overridden per-ablation)
    parser.add_argument("--matrix-lr", type=float, default=0.01,
                        help="Learning rate for matrix params (Muon)")
    parser.add_argument("--weight-decay", type=float, default=0.2,
                        help="Weight decay for Muon optimizer")
    parser.add_argument("--adam-weight-decay", type=float, default=0.0,
                        help="Weight decay for AdamW optimizer")

    # Schedule
    parser.add_argument("--warmup-ratio", type=float, default=0.0,
                        help="Warmup ratio")
    parser.add_argument("--warmdown-ratio", type=float, default=1.0,
                        help="Warmdown ratio")
    parser.add_argument("--final-lr-frac", type=float, default=0.0,
                        help="Final LR fraction")

    # Memory optimization
    parser.add_argument("--use-chunked-loss", action="store_true", default=True,
                        help="Use chunked loss computation")
    parser.add_argument("--chunk-size", type=int, default=128,
                        help="Chunk size for chunked loss")

    # Multi-GPU
    parser.add_argument("--nproc", type=int, default=8,
                        help="Number of GPUs (if > 1, uses torchrun)")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base output directory for all ablations")

    # Control
    parser.add_argument("--skip-until", type=str, default=None,
                        help="Skip ablations until this name (for resuming)")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="Only run these specific ablations")

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f"checkpoints/ablations_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Base arguments shared by all runs
    base_args = [
        f"--data-path={args.data_path}",
        f"--device-batch-size={args.device_batch_size}",
        f"--total-batch-size={args.total_batch_size}",
        f"--max-seq-len={args.max_seq_len}",
        f"--eval-every={args.eval_every}",
        f"--save-every={args.save_every}",
        f"--base-model={args.base_model}",
        f"--medusa-num-heads={args.medusa_num_heads}",
        f"--matrix-lr={args.matrix_lr}",
        f"--weight-decay={args.weight_decay}",
        f"--adam-weight-decay={args.adam_weight_decay}",
        f"--warmup-ratio={args.warmup_ratio}",
        f"--warmdown-ratio={args.warmdown_ratio}",
        f"--final-lr-frac={args.final_lr_frac}",
        "--skip-filter",  # Assume data is pre-filtered
    ]

    # Training horizon
    if args.num_iterations > 0:
        base_args.append(f"--num-iterations={args.num_iterations}")
    else:
        base_args.append(f"--num-epochs={args.num_epochs}")

    # Memory optimization
    if args.use_chunked_loss:
        base_args.append("--use-chunked-loss")
        base_args.append(f"--chunk-size={args.chunk_size}")

    if args.val_data_path:
        base_args.append(f"--val-data-path={args.val_data_path}")

    use_torchrun = args.nproc > 1

    # Generate and filter configs
    configs = generate_ablation_configs()

    if args.only:
        configs = [c for c in configs if c.name in args.only]

    skip_mode = args.skip_until is not None

    # Track results for greedy selection
    results = []
    best_config = AblationConfig(name="baseline")  # Start with defaults

    print(f"\n{'#'*60}")
    print(f"# Gemma Medusa Ablation Study")
    print(f"# Output: {args.output_dir}")
    print(f"# Base model: {args.base_model}")
    print(f"# Data: {args.data_path}")
    print(f"# GPUs: {args.nproc}")
    print(f"# Ablations to run: {len(configs)}")
    print(f"{'#'*60}\n")

    for i, config in enumerate(configs):
        if skip_mode:
            if config.name == args.skip_until:
                skip_mode = False
            else:
                print(f"Skipping {config.name}...")
                continue

        # Apply best settings from previous phases
        # Phase 2+: use best zero_init
        if i >= 2 and results:
            phase1_results = [r for r in results if r["name"] in ["baseline", "zero_init"]]
            if phase1_results:
                # For now, default to baseline unless zero_init clearly won
                pass

        # Phase 3+: use best lr
        if i >= 4 and config.proj_lr == 0.002:  # Only override if using default
            lr_results = [r for r in results if r["name"].startswith("lr_") or r["name"] == "baseline"]
            # Would need actual loss values to pick best

        # Handle alpha_2x special case
        if config.name == "alpha_2x":
            config.lora_alpha = best_config.lora_rank * 2

        # Merge best config with current ablation
        final_config = AblationConfig(
            name=config.name,
            zero_init_mlp=config.zero_init_mlp if config.name in ["baseline", "zero_init"] else best_config.zero_init_mlp,
            proj_lr=config.proj_lr if config.name.startswith("lr_") or config.name == "baseline" else best_config.proj_lr,
            lora_rank=config.lora_rank if config.name.startswith("rank_") or config.name == "baseline" else best_config.lora_rank,
            lora_alpha=config.lora_alpha,
            medusa_num_layers=config.medusa_num_layers if config.name.startswith("layers_") or config.name == "baseline" else best_config.medusa_num_layers,
        )

        result = run_single_ablation(
            final_config,
            base_args,
            args.output_dir,
            use_torchrun=use_torchrun,
            nproc=args.nproc,
        )
        results.append(result)

        # Save intermediate results
        results_file = os.path.join(args.output_dir, "ablation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n{'#'*60}")
    print(f"# Ablation study complete!")
    print(f"# Results saved to: {os.path.join(args.output_dir, 'ablation_results.json')}")
    print(f"{'#'*60}\n")

    # Print summary
    print("\nSummary:")
    print("-" * 40)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['name']}")


if __name__ == "__main__":
    main()
