"""
Common utilities shared between Gemma Medusa and EAGLE training.

This module provides shared infrastructure for:
- Data loading (ShareGPT format)
- Training utilities (LR scheduling, checkpointing)
- Logging and metrics
"""

from .data import (
    load_sharegpt_data,
    filter_dataset,
    split_train_val,
    data_generator,
    simple_data_generator,
)

from .training import (
    get_lr_scheduler,
    apply_lr_schedule,
    save_checkpoint,
    load_checkpoint,
    setup_output_dir,
    compute_grad_norms,
    compute_weight_norms,
    reduce_tensor,
    EMALoss,
    format_time,
    estimate_eta,
)

from .speculative import (
    build_tree_attention_mask,
    update_kv_cache_from_tree,
)

__all__ = [
    # Data
    "load_sharegpt_data",
    "filter_dataset",
    "split_train_val",
    "data_generator",
    "simple_data_generator",
    # Training
    "get_lr_scheduler",
    "apply_lr_schedule",
    "save_checkpoint",
    "load_checkpoint",
    "setup_output_dir",
    "compute_grad_norms",
    "compute_weight_norms",
    "reduce_tensor",
    "EMALoss",
    "format_time",
    "estimate_eta",
    # Speculative decoding helpers
    "build_tree_attention_mask",
    "update_kv_cache_from_tree",
]
