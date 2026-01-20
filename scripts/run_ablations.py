#!/usr/bin/env python3
"""
Ablation study orchestrator for Gemma Medusa training.

Runs a sequential sweep of hyperparameters, using results from earlier
runs to inform later sweeps (greedy selection).

Usage:
    python -m scripts.run_ablations --data-path data/openhermes_75k.json

    # Or with torchrun for multi-GPU
    torchrun --standalone --nproc_per_node=8 -m scripts.run_ablations \
        --data-path data/openhermes_75k.json
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class AblationConfig:
    """Single ablation run configuration."""
    name: str
    zero_init_mlp: bool = False
    proj_lr: float = 0.004
    lora_rank: int = 64
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
    2. LR sweep: 0.001, 0.004, 0.01 (pick best)
    3. Rank sweep: 32, 64, 128 (pick best)
    4. Alpha sweep: rank vs 2*rank (pick best)
    5. Layers sweep: 0, 1, 2 (pick best)
    """
    configs = []

    # Phase 1: Zero-init ablation
    configs.append(AblationConfig(name="baseline", zero_init_mlp=False))
    configs.append(AblationConfig(name="zero_init", zero_init_mlp=True))

    # Phase 2: LR sweep (will use best zero_init from phase 1)
    for lr in [0.001, 0.01]:  # 0.004 is baseline
        configs.append(AblationConfig(name=f"lr_{lr}", proj_lr=lr))

    # Phase 3: Rank sweep (will use best zero_init + lr)
    for rank in [32, 128]:  # 64 is baseline
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
    parser.add_argument("--num-iterations", type=int, default=500,
                        help="Number of training iterations per ablation")
    parser.add_argument("--device-batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--total-batch-size", type=int, default=32,
                        help="Total batch size")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Evaluate every N steps")

    # Model
    parser.add_argument("--base-model", type=str, default="google/gemma-3-1b-it",
                        help="Base model name")

    # Multi-GPU
    parser.add_argument("--nproc", type=int, default=1,
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
        f"--num-iterations={args.num_iterations}",
        f"--device-batch-size={args.device_batch_size}",
        f"--total-batch-size={args.total_batch_size}",
        f"--max-seq-len={args.max_seq_len}",
        f"--eval-every={args.eval_every}",
        f"--base-model={args.base_model}",
        "--save-every=-1",  # Only save at end
    ]
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
        if i >= 4 and config.proj_lr == 0.004:  # Only override if using default
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
