#!/usr/bin/env python3
"""
Ablation study orchestrator for Gemma Medusa training.

Runs a sequential sweep of hyperparameters, using results from earlier
runs to inform later sweeps (greedy selection).

Usage:
    python -m scripts.run_ablations --data-path data/openhermes_filtered.json

    # Or with torchrun for multi-GPU
    python -m scripts.run_ablations --data-path data/openhermes_filtered.json --nproc 8

    # Use only 7.5% of the dataset
    python -m scripts.run_ablations --data-path data/openhermes.json --data-percent 7.5
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
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


def subset_data(input_path: str, percent: float, output_dir: str) -> str:
    """
    Subset a JSON/JSONL dataset to the first N percent of examples.

    Returns path to the subsetted data file.
    """
    print(f"Subsetting data to {percent}% of {input_path}...")

    # Load the data
    with open(input_path, 'r') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON array format
            data = json.loads(content)
        else:
            # JSONL format
            data = [json.loads(line) for line in content.split('\n') if line.strip()]

    total_samples = len(data)
    subset_size = int(total_samples * percent / 100)
    subset_data = data[:subset_size]

    print(f"  Total samples: {total_samples:,}")
    print(f"  Subset size: {subset_size:,} ({percent}%)")

    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(input_path)
    name, ext = os.path.splitext(basename)
    output_path = os.path.join(output_dir, f"{name}_{percent}pct{ext}")

    with open(output_path, 'w') as f:
        json.dump(subset_data, f)

    print(f"  Saved to: {output_path}")
    return output_path


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

    # Parse final loss from output
    final_loss_file = os.path.join(run_output_dir, "final_loss.json")
    final_loss = float('inf')
    final_main_loss = float('inf')
    final_head_losses = []
    parsed_results = False

    if os.path.exists(final_loss_file):
        with open(final_loss_file, 'r') as f:
            loss_data = json.load(f)
            final_loss = loss_data.get('final_loss', float('inf'))
            final_main_loss = loss_data.get('final_main_loss', float('inf'))
            final_head_losses = loss_data.get('final_head_losses', [])
            parsed_results = final_loss < float('inf')

    # Check that we were able to parse results for successful runs
    if result.returncode == 0 and not parsed_results:
        print(f"WARNING: Run '{config.name}' succeeded but could not parse final_loss.json!")
        print(f"  Expected file: {final_loss_file}")
        print(f"  File exists: {os.path.exists(final_loss_file)}")
    elif result.returncode != 0:
        print(f"WARNING: Run '{config.name}' failed with return code {result.returncode}")

    return {
        "name": config.name,
        "config": vars(config),
        "output_dir": run_output_dir,
        "success": result.returncode == 0 and parsed_results,
        "final_loss": final_loss,
        "final_main_loss": final_main_loss,
        "final_head_losses": final_head_losses,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Gemma Medusa ablation study")

    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to training data")
    parser.add_argument("--data-percent", type=float, default=None,
                        help="Subset data to first N%% (e.g., 7.5 for 7.5%%)")
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
    parser.add_argument("--warmdown-ratio", type=float, default=0.4,
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

    # Subset data if requested
    data_path = args.data_path
    if args.data_percent is not None:
        data_path = subset_data(args.data_path, args.data_percent, args.output_dir)

    # Base arguments shared by all runs
    base_args = [
        f"--data-path={data_path}",
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
    best_config = AblationConfig(name="best")  # Accumulates best settings

    print(f"\n{'#'*60}")
    print(f"# Gemma Medusa Ablation Study")
    print(f"# Output: {args.output_dir}")
    print(f"# Base model: {args.base_model}")
    print(f"# Data: {data_path}")
    print(f"# GPUs: {args.nproc}")
    print(f"# Ablations to run: {len(configs)}")
    print(f"{'#'*60}\n")

    def select_best(candidates: list[dict], metric: str = "final_loss") -> dict | None:
        """Select the result with lowest loss from candidates."""
        valid = [r for r in candidates if r["success"] and r.get(metric, float('inf')) < float('inf')]
        if not valid:
            return None
        return min(valid, key=lambda r: r[metric])

    def update_best_config_after_phase(phase_name: str, phase_results: list[dict]):
        """Update best_config based on phase results."""
        best_result = select_best(phase_results)
        if best_result is None:
            print(f"  Warning: No successful runs in {phase_name}, keeping previous best")
            return

        best_name = best_result["name"]
        best_loss = best_result["final_loss"]
        print(f"\n>>> Phase {phase_name} winner: {best_name} (loss={best_loss:.6f})")

        cfg = best_result["config"]
        if phase_name == "zero_init":
            best_config.zero_init_mlp = cfg["zero_init_mlp"]
        elif phase_name == "lr":
            best_config.proj_lr = cfg["proj_lr"]
        elif phase_name == "rank":
            best_config.lora_rank = cfg["lora_rank"]
        elif phase_name == "alpha":
            best_config.lora_alpha = cfg["lora_alpha"]
        elif phase_name == "layers":
            best_config.medusa_num_layers = cfg["medusa_num_layers"]

    # Phase tracking
    phase_results = []
    current_phase = None

    for i, config in enumerate(configs):
        if skip_mode:
            if config.name == args.skip_until:
                skip_mode = False
            else:
                print(f"Skipping {config.name}...")
                continue

        # Determine which phase this config belongs to
        if config.name in ["baseline", "zero_init"]:
            new_phase = "zero_init"
        elif config.name.startswith("lr_"):
            new_phase = "lr"
        elif config.name.startswith("rank_"):
            new_phase = "rank"
        elif config.name == "alpha_2x":
            new_phase = "alpha"
        elif config.name.startswith("layers_"):
            new_phase = "layers"
        else:
            new_phase = "unknown"

        # If phase changed, update best_config from previous phase
        if current_phase is not None and new_phase != current_phase and phase_results:
            update_best_config_after_phase(current_phase, phase_results)
            phase_results = []

        current_phase = new_phase

        # Build final config by merging current ablation with best_config
        # Each phase tests one variable while using best values for others
        final_config = AblationConfig(
            name=config.name,
            zero_init_mlp=config.zero_init_mlp if new_phase == "zero_init" else best_config.zero_init_mlp,
            proj_lr=config.proj_lr if new_phase == "lr" else best_config.proj_lr,
            lora_rank=config.lora_rank if new_phase == "rank" else best_config.lora_rank,
            lora_alpha=(best_config.lora_rank * 2) if config.name == "alpha_2x" else best_config.lora_alpha,
            medusa_num_layers=config.medusa_num_layers if new_phase == "layers" else best_config.medusa_num_layers,
        )

        # For LR phase, include baseline in comparison
        if new_phase == "lr" and config.name.startswith("lr_"):
            # baseline result should be used as the "lr=0.002" comparison point
            baseline_result = next((r for r in results if r["name"] == "baseline"), None)
            if baseline_result and baseline_result not in phase_results:
                phase_results.append(baseline_result)

        result = run_single_ablation(
            final_config,
            base_args,
            args.output_dir,
            use_torchrun=use_torchrun,
            nproc=args.nproc,
        )
        results.append(result)
        phase_results.append(result)

        # Print result
        if result["success"]:
            print(f"  -> {config.name}: loss={result['final_loss']:.6f}")
        else:
            print(f"  -> {config.name}: FAILED")

        # Save intermediate results
        results_file = os.path.join(args.output_dir, "ablation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Final phase update
    if phase_results:
        update_best_config_after_phase(current_phase, phase_results)

    # Save best config
    best_config_file = os.path.join(args.output_dir, "best_config.json")
    with open(best_config_file, 'w') as f:
        json.dump(vars(best_config), f, indent=2)

    print(f"\n{'#'*60}")
    print(f"# Ablation study complete!")
    print(f"# Results saved to: {os.path.join(args.output_dir, 'ablation_results.json')}")
    print(f"# Best config saved to: {best_config_file}")
    print(f"{'#'*60}\n")

    # Print summary
    print("\nSummary:")
    print("-" * 60)
    for r in results:
        status = "✓" if r["success"] else "✗"
        loss_str = f"loss={r['final_loss']:.6f}" if r["success"] and r.get("final_loss", float('inf')) < float('inf') else "N/A"
        print(f"  {status} {r['name']:20s} {loss_str}")

    print("\nBest configuration found:")
    print("-" * 60)
    print(f"  zero_init_mlp:     {best_config.zero_init_mlp}")
    print(f"  proj_lr:           {best_config.proj_lr}")
    print(f"  lora_rank:         {best_config.lora_rank}")
    print(f"  lora_alpha:        {best_config.lora_alpha}")
    print(f"  medusa_num_layers: {best_config.medusa_num_layers}")


if __name__ == "__main__":
    main()
