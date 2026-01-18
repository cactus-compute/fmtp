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

# Try to import Liger's fused kernel - use the high-level transformers API
_liger_loss_fn = None
HAS_LIGER = False

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    _liger_loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-1, reduction="mean")
    HAS_LIGER = True
    print("[fused_ce] Liger kernel loaded successfully")
except ImportError as e:
    print(f"[fused_ce] Liger import failed (ImportError): {e}")
except Exception as e:
    print(f"[fused_ce] Liger import failed ({type(e).__name__}): {e}")


def fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -1,
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
    if hidden_states.ndim == 3:
        B, T, D = hidden_states.shape
        hidden_states = hidden_states.view(B * T, D)
        targets = targets.view(B * T)

    if HAS_LIGER:
        # LigerFusedLinearCrossEntropyLoss expects (weight, input, target)
        # Note: it creates loss_fn with fixed ignore_index/reduction at init time
        # For simplicity, we use the global one (ignore_index=-1, reduction="mean")
        # If you need different settings, create a new instance
        global _liger_loss_fn
        if ignore_index != -1 or reduction != "mean":
            # Create a custom loss fn for non-default settings
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
            return loss_fn(weight, hidden_states, targets)
        return _liger_loss_fn(weight, hidden_states, targets)
    else:
        # Fallback: standard unfused computation (WARNING: will OOM on large vocab!)
        import warnings
        warnings.warn(
            "Using unfused cross-entropy fallback - this will likely OOM! "
            "Install liger-kernel properly: pip install liger-kernel",
            RuntimeWarning
        )
        logits = F.linear(hidden_states, weight, bias)
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction=reduction)
