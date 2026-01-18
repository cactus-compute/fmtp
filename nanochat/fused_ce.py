"""
Fused Linear Cross-Entropy using Liger Kernel.

This avoids materializing the full (B, T, vocab_size) logits tensor, which can be
16+ GiB for typical training configs. Instead, the matmul and cross-entropy are
fused into a single kernel that computes the loss without the intermediate.

Usage:
    from nanochat.fused_ce import fused_linear_cross_entropy

    # Instead of:
    #   logits = linear(hidden_states)
    #   loss = F.cross_entropy(logits, targets)
    # Use:
    #   loss = fused_linear_cross_entropy(hidden_states, linear.weight, targets)
"""
import torch
import torch.nn.functional as F

# Try to import Liger's fused kernel - use the functional API which is more stable
try:
    from liger_kernel.ops.fused_linear_cross_entropy import liger_fused_linear_cross_entropy
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False


def fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    softcap: float = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute cross-entropy loss with fused linear projection.

    Args:
        hidden_states: (B, T, hidden_dim) or (B*T, hidden_dim)
        weight: (vocab_size, hidden_dim) - the lm_head weight matrix
        targets: (B, T) or (B*T,) - target token ids
        bias: optional bias for the linear layer
        ignore_index: target value to ignore in loss computation
        softcap: if provided, apply logit soft-capping: softcap * tanh(logits / softcap)
        reduction: "mean", "sum", or "none"

    Returns:
        Loss tensor (scalar if reduction is "mean" or "sum")
    """
    # Flatten to 2D for the fused kernel
    original_shape = hidden_states.shape
    if hidden_states.ndim == 3:
        B, T, D = hidden_states.shape
        hidden_states = hidden_states.view(B * T, D)
        targets = targets.view(B * T)

    if HAS_LIGER:
        # Liger fused kernel - never materializes full logits
        loss = liger_fused_linear_cross_entropy(
            hidden_states,
            weight,
            targets,
            bias=bias,
            ignore_index=ignore_index,
            softcap=softcap,
            reduction=reduction,
        )
        return loss
    else:
        # Fallback: standard unfused computation
        logits = F.linear(hidden_states, weight, bias)
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction=reduction)
