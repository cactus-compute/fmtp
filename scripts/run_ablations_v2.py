#!/usr/bin/env python3
"""
Ablation study orchestrator v2 for Gemma Medusa training.

Continues from best config found in v1:
  - zero_init_mlp: True
  - proj_lr: 0.001
  - lora_rank: 128
  - lora_alpha: 256 (2x rank)
  - medusa_num_layers: 2

Tests additional hyperparameters:
  1. Layers 3 - Does more depth continue to help?
  2. Rank 256 - More capacity
  3. Alpha 4x - Higher LoRA scaling
  4. LR 0.0005 - Lower LR for larger model

Usage:
    python -m scripts.run_ablations_v2 --data-path data/openhermes_filtered.json

    # Or with torchrun for multi-GPU
    python -m scripts.run_ablations_v2 --data-path data/openhermes_filtered.json --nproc 8

    # Use only 7.5% of the dataset
    python -m scripts.run_ablations_v2 --data-path data/openhermes.json --data-percent 7.5
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AblationConfig:
    """Single ablation run configuration."""
    name: str
    # Best config from v1 ablations as defaults
    zero_init_mlp: bool = True
    proj_lr: float = 0.001
    lora_rank: int = 128
    lora_alpha: int = 256  # 2x rank from v1
    medusa_num_layers: int = 2

    def to_args(self) -> list[str]:
        """Convert to CLI arguments."""
        args = [
            f"--proj-lr={self.proj_lr}",
            f"--lora-rank={self.lora_rank}",
            f"--medusa-num-layers={self.medusa_num_layers}",
            f"--lora-alpha={self.lora_alpha}",
        ]
        if self.zero_init_mlp:
            args.append("--zero-init-mtp-mlp")
        return args


def generate_ablation_configs() -> list[AblationConfig]:
    """
    Generate ablation configurations for v2 sweep.

    Starting from best v1 config, test in order of expected impact:
    1. Layers 3 - More depth (biggest win in v1 was layers=2)
    2. Rank 256 - More capacity (rank=128 beat 64, continue trend)
    3. Alpha 4x - Higher scaling (alpha=2x helped, try 4x)
    4. LR 0.0005 - Lower LR (larger model might need lower LR)
    """
    configs = []

    # Baseline: best config from v1
    configs.append(AblationConfig(name="v1_best"))

    # Phase 1: Test 3 layers (most impactful change in v1)
    configs.append(AblationConfig(name="layers_3", medusa_num_layers=3))

    # Phase 2: Test rank 256
    configs.append(AblationConfig(name="rank_256", lora_rank=256, lora_alpha=512))  # keep 2x ratio

    # Phase 3: Test alpha 4x (512 with rank=128)
    configs.append(AblationConfig(name="alpha_4x", lora_alpha=512))

    # Phase 4: Test lower LR
    configs.append(AblationConfig(name="lr_0.0005", proj_lr=0.0005))

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
    cmd.extend([f"--wandb-run=ablation_v2_{config.name}"])

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
    parser = argparse.ArgumentParser(description="Run Gemma Medusa ablation study v2")

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
        args.output_dir = f"checkpoints/ablations_v2_{timestamp}"

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

    # Track results
    results = []
    best_config = AblationConfig(name="best")  # Start with v1 best

    print(f"\n{'#'*60}")
    print(f"# Gemma Medusa Ablation Study v2")
    print(f"# Output: {args.output_dir}")
    print(f"# Base model: {args.base_model}")
    print(f"# Data: {data_path}")
    print(f"# GPUs: {args.nproc}")
    print(f"# Ablations to run: {len(configs)}")
    print(f"#")
    print(f"# Starting from v1 best config:")
    print(f"#   zero_init_mlp: True")
    print(f"#   proj_lr: 0.001")
    print(f"#   lora_rank: 128")
    print(f"#   lora_alpha: 256")
    print(f"#   medusa_num_layers: 2")
    print(f"{'#'*60}\n")

    def select_best(candidates: list[dict], metric: str = "final_loss") -> dict | None:
        """Select the result with lowest loss from candidates."""
        valid = [r for r in candidates if r["success"] and r.get(metric, float('inf')) < float('inf')]
        if not valid:
            return None
        return min(valid, key=lambda r: r[metric])

    # Run all ablations (simpler than v1 - no greedy phase updates, just compare all)
    for i, config in enumerate(configs):
        if skip_mode:
            if config.name == args.skip_until:
                skip_mode = False
            else:
                print(f"Skipping {config.name}...")
                continue

        result = run_single_ablation(
            config,
            base_args,
            args.output_dir,
            use_torchrun=use_torchrun,
            nproc=args.nproc,
        )
        results.append(result)

        # Print result
        if result["success"]:
            print(f"  -> {config.name}: loss={result['final_loss']:.6f}")
        else:
            print(f"  -> {config.name}: FAILED")

        # Save intermediate results
        results_file = os.path.join(args.output_dir, "ablation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Find overall best
    best_result = select_best(results)
    if best_result:
        best_config = AblationConfig(**{k: v for k, v in best_result["config"].items() if k != "name"}, name="best")

    # Save best config
    best_config_file = os.path.join(args.output_dir, "best_config.json")
    with open(best_config_file, 'w') as f:
        json.dump(vars(best_config), f, indent=2)

    print(f"\n{'#'*60}")
    print(f"# Ablation study v2 complete!")
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

    # Compare to v1 best
    v1_best_loss = None
    for r in results:
        if r["name"] == "v1_best" and r["success"]:
            v1_best_loss = r["final_loss"]
            break

    if v1_best_loss and best_result and best_result["name"] != "v1_best":
        improvement = v1_best_loss - best_result["final_loss"]
        pct_improvement = (improvement / v1_best_loss) * 100
        print(f"\n  Improvement over v1 best: {improvement:.6f} ({pct_improvement:.2f}%)")


if __name__ == "__main__":
    main()
