"""
Shared training utilities for Gemma Medusa and EAGLE training.

Includes:
- Learning rate scheduling
- Checkpoint saving/loading
- Logging utilities
- Common argument parsing
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import torch
import torch.distributed as dist


def get_lr_scheduler(
    warmup_ratio: float = 0.05,
    warmdown_ratio: float = 0.95,
    final_lr_frac: float = 0.0,
) -> Callable[[int, int], float]:
    """
    Create a learning rate scheduler function.

    Schedule: warmup -> constant -> warmdown (linear)

    Args:
        warmup_ratio: Ratio of total iterations for warmup
        warmdown_ratio: Ratio of total iterations before warmdown starts
        final_lr_frac: Final LR as fraction of initial LR

    Returns:
        Function that takes (current_step, total_steps) and returns LR multiplier
    """
    def get_lr_multiplier(step: int, total_steps: int) -> float:
        warmup_iters = round(warmup_ratio * total_steps)
        warmdown_iters = round((1.0 - warmdown_ratio) * total_steps)

        if step < warmup_iters:
            return (step + 1) / warmup_iters
        elif step < total_steps - warmdown_iters:
            return 1.0
        else:
            progress = (total_steps - step) / warmdown_iters
            return progress * 1.0 + (1 - progress) * final_lr_frac

    return get_lr_multiplier


def apply_lr_schedule(
    optimizers: List[torch.optim.Optimizer],
    step: int,
    total_steps: int,
    lr_fn: Callable[[int, int], float],
) -> float:
    """
    Apply learning rate schedule to optimizers.

    Args:
        optimizers: List of optimizers
        step: Current step
        total_steps: Total number of steps
        lr_fn: LR multiplier function

    Returns:
        Current LR multiplier
    """
    lrm = lr_fn(step, total_steps)
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * lrm
    return lrm


def save_checkpoint(
    output_dir: str,
    step: int,
    state_dict: Dict[str, Any],
    optimizers: Optional[List[torch.optim.Optimizer]] = None,
    config: Optional[Dict] = None,
    filename: str = "checkpoint.pt",
) -> str:
    """
    Save a training checkpoint.

    Args:
        output_dir: Output directory
        step: Current step
        state_dict: Model state dict (e.g., from get_medusa_state_dict or get_draft_state_dict)
        optimizers: List of optimizers (optional)
        config: Training config dict (optional)
        filename: Checkpoint filename

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = os.path.join(output_dir, f"step_{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'step': step,
        **state_dict,
    }
    if optimizers is not None:
        checkpoint['optimizers'] = [opt.state_dict() for opt in optimizers]
    if config is not None:
        checkpoint['config'] = config

    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load to

    Returns:
        Checkpoint dict
    """
    return torch.load(checkpoint_path, map_location=device)


def setup_output_dir(
    output_dir: Optional[str],
    base_name: str,
    config: Dict,
    master_process: bool = True,
) -> str:
    """
    Setup output directory for training.

    Args:
        output_dir: User-specified output dir (or None for auto)
        base_name: Base name for auto-generated dir (e.g., "gemma_medusa_1b")
        config: Config dict to save
        master_process: Whether this is the master process

    Returns:
        Output directory path
    """
    from nanochat.common import get_base_dir

    if output_dir is None:
        base_dir = get_base_dir()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(base_dir, f"{base_name}_{timestamp}")

    if master_process:
        os.makedirs(output_dir, exist_ok=True)
        # Save config
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

    return output_dir


def compute_grad_norms(
    model: torch.nn.Module,
    param_groups: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, float]:
    """
    Compute gradient norms for model parameters.

    Args:
        model: Model to compute norms for
        param_groups: Optional dict mapping group name to list of name patterns
                     Default groups: lora_A, lora_B, resblock, fusion, other

    Returns:
        Dict mapping group names to average grad norms
    """
    if param_groups is None:
        param_groups = {
            'lora_A': ['lora_A'],
            'lora_B': ['lora_B'],
            'resblock': ['blocks'],
            'fusion': ['fusion', 'fc'],
        }

    grad_norms = {name: [] for name in param_groups}
    grad_norms['other'] = []
    total_norm_sq = 0.0

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_norm_sq += grad_norm ** 2

            matched = False
            for group_name, patterns in param_groups.items():
                if any(p in name for p in patterns):
                    grad_norms[group_name].append(grad_norm)
                    matched = True
                    break
            if not matched:
                grad_norms['other'].append(grad_norm)

    # Compute averages
    result = {'total': total_norm_sq ** 0.5}
    for group_name, norms in grad_norms.items():
        if norms:
            result[group_name] = sum(norms) / len(norms)
        else:
            result[group_name] = 0.0

    return result


def compute_weight_norms(
    model: torch.nn.Module,
    param_groups: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute weight norms and max values for model parameters.

    Args:
        model: Model to compute norms for
        param_groups: Optional dict mapping group name to list of name patterns

    Returns:
        Dict with 'norm' and 'max' sub-dicts for each group
    """
    if param_groups is None:
        param_groups = {
            'lora_A': ['lora_A'],
            'lora_B': ['lora_B'],
            'resblock': ['blocks'],
            'fusion': ['fusion', 'fc'],
        }

    weight_norms = {name: [] for name in param_groups}
    weight_norms['other'] = []
    weight_maxes = {name: [] for name in param_groups}
    weight_maxes['other'] = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            w_norm = param.norm().item()
            w_max = param.abs().max().item()

            matched = False
            for group_name, patterns in param_groups.items():
                if any(p in name for p in patterns):
                    weight_norms[group_name].append(w_norm)
                    weight_maxes[group_name].append(w_max)
                    matched = True
                    break
            if not matched:
                weight_norms['other'].append(w_norm)
                weight_maxes['other'].append(w_max)

    # Compute averages
    result = {'norm': {}, 'max': {}}
    for group_name in list(param_groups.keys()) + ['other']:
        norms = weight_norms[group_name]
        maxes = weight_maxes[group_name]
        result['norm'][group_name] = sum(norms) / len(norms) if norms else 0.0
        result['max'][group_name] = max(maxes) if maxes else 0.0

    return result


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    All-reduce a tensor across distributed processes.

    Args:
        tensor: Tensor to reduce
        world_size: Number of processes

    Returns:
        Reduced tensor (averaged)
    """
    if not dist.is_initialized() or world_size == 1:
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


class EMALoss:
    """Exponential moving average for loss tracking."""

    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.value = 0.0
        self.steps = 0

    def update(self, loss: float) -> float:
        """Update EMA and return debiased value."""
        self.value = self.beta * self.value + (1 - self.beta) * loss
        self.steps += 1
        return self.get()

    def get(self) -> float:
        """Get debiased EMA value."""
        if self.steps == 0:
            return 0.0
        debias = 1 - self.beta ** self.steps
        return self.value / debias

    def reset(self):
        """Reset the EMA."""
        self.value = 0.0
        self.steps = 0


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def estimate_eta(
    current_step: int,
    total_steps: int,
    elapsed_time: float,
    warmup_steps: int = 10,
) -> Optional[str]:
    """
    Estimate time remaining.

    Args:
        current_step: Current training step
        total_steps: Total training steps
        elapsed_time: Time elapsed since warmup
        warmup_steps: Steps to skip for timing (warmup)

    Returns:
        Formatted ETA string or None if not enough data
    """
    if current_step <= warmup_steps:
        return None

    steps_done = current_step - warmup_steps
    if steps_done <= 0:
        return None

    avg_time_per_step = elapsed_time / steps_done
    remaining_steps = total_steps - current_step
    eta_seconds = remaining_steps * avg_time_per_step

    return format_time(eta_seconds)
