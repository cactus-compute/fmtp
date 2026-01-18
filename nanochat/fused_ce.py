"""
Fused Linear Cross-Entropy using Liger Kernel.

This avoids materializing the full (B, T, vocab_size) logits tensor, which can be
16+ GiB for typical training configs. Instead, the matmul and cross-entropy are
fused into a single kernel that computes the loss without the intermediate.

Architecture note: The fused CE is excluded from torch.compile via @torch.compiler.disable.
This is the CORRECT pattern - not a "battle" between Liger and torch.compile:
- The transformer backbone gets compiled (speedup from inductor)
- The fused CE runs as a Triton kernel outside compilation (memory efficiency from Liger)

This achieves ~31% MFU with 4 Medusa heads vs ~27% MFU with no compilation.

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
except ImportError as e:
    print(f"[fused_ce] Liger import failed (ImportError): {e}")
    print("[fused_ce] Falling back to unfused cross-entropy (will use much more memory with large vocabs)")
except Exception as e:
    print(f"[fused_ce] Liger import failed ({type(e).__name__}): {e}")
    print("[fused_ce] Falling back to unfused cross-entropy (will use much more memory with large vocabs)")


def _liger_ce(loss_fn, weight, hidden_states, targets):
    """Call Liger kernel."""
    return loss_fn(weight, hidden_states, targets)


# Disable torch.compile for this function to avoid recompilation on every
# different sequence length (dynamo treats each shape as a new graph).
# This is the CORRECT architecture: backbone compiles, fused CE doesn't.
@torch.compiler.disable
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
        global _liger_loss_fn
        if ignore_index != -1 or reduction != "mean":
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
            return _liger_ce(loss_fn, weight, hidden_states, targets)
        return _liger_ce(_liger_loss_fn, weight, hidden_states, targets)
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
