"""
Fused Linear Cross-Entropy implementation.

This avoids materializing the full (B, T, vocab_size) logits tensor, which can be
16+ GiB for typical training configs. Instead, the matmul and cross-entropy are
fused into a single kernel that computes the loss without the intermediate.

Supports multiple backends:
- "cce": Apple's Cut Cross-Entropy with torch.compile (recommended for torch.compile compatibility)
- "liger": Liger Kernel's fused CE (default, best raw performance)
- "unfused": Standard PyTorch fallback (will OOM on large vocabs)

Set via environment variable: FUSED_CE_BACKEND=cce|liger|unfused

Usage:
    from nanochat.fused_ce import fused_linear_cross_entropy

    # Instead of:
    #   logits = linear(hidden_states)
    #   loss = F.cross_entropy(logits, targets)
    # Use:
    #   loss = fused_linear_cross_entropy(hidden_states, linear.weight, targets)
"""
import os
import torch
import torch.nn.functional as F

# Backend selection via environment variable
FUSED_CE_BACKEND = os.environ.get("FUSED_CE_BACKEND", "liger").lower()

# Try to import Apple's Cut Cross-Entropy (torch.compile compatible)
HAS_CCE = False
cce_linear_cross_entropy = None
try:
    from cut_cross_entropy import linear_cross_entropy as cce_linear_cross_entropy
    HAS_CCE = True
except ImportError:
    pass

# Try to import Liger's fused kernel
_liger_loss_fn = None
HAS_LIGER = False
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    _liger_loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-1, reduction="mean")
    HAS_LIGER = True
except ImportError as e:
    if FUSED_CE_BACKEND == "liger":
        print(f"[fused_ce] Liger import failed (ImportError): {e}")
except Exception as e:
    if FUSED_CE_BACKEND == "liger":
        print(f"[fused_ce] Liger import failed ({type(e).__name__}): {e}")

# Report backend selection
if FUSED_CE_BACKEND == "cce":
    if HAS_CCE:
        print("[fused_ce] Using Apple Cut Cross-Entropy (torch.compile compatible)")
    else:
        print("[fused_ce] CCE requested but not available, falling back to Liger" if HAS_LIGER else "[fused_ce] CCE requested but not available, falling back to unfused")
elif FUSED_CE_BACKEND == "liger":
    if HAS_LIGER:
        print("[fused_ce] Using Liger Kernel fused CE")
    else:
        print("[fused_ce] Liger requested but not available, falling back to unfused")
elif FUSED_CE_BACKEND == "unfused":
    print("[fused_ce] Using unfused CE (WARNING: will use much more memory)")


def _liger_ce(loss_fn, weight, hidden_states, targets):
    """Call Liger kernel."""
    return loss_fn(weight, hidden_states, targets)


def _cce_impl(hidden_states, weight, targets, ignore_index, softcap, reduction):
    """
    Apple Cut Cross-Entropy implementation.
    Uses torch.compile backend for compatibility with compiled models.
    """
    # CCE expects: linear_cross_entropy(embeddings, classifier, labels, ...)
    # Note: CCE handles softcap differently - it uses "softcapping" parameter
    loss = cce_linear_cross_entropy(
        hidden_states,
        weight,
        targets,
        ignore_index=ignore_index,
        softcap=softcap,
        reduction=reduction,
        impl="torch_compile",  # Use torch.compile-compatible implementation
    )
    return loss


# Disable torch.compile for this function when using Liger to avoid recompilation
# on every different sequence length (dynamo treats each shape as a new graph).
# CCE with impl="torch_compile" doesn't need this decorator.
@torch.compiler.disable
def _fused_ce_liger(hidden_states, weight, targets, ignore_index, reduction):
    """Liger-based implementation with torch.compile disabled."""
    global _liger_loss_fn
    if ignore_index != -1 or reduction != "mean":
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        return _liger_ce(loss_fn, weight, hidden_states, targets)
    return _liger_ce(_liger_loss_fn, weight, hidden_states, targets)


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

    # Backend selection with fallbacks
    backend = FUSED_CE_BACKEND

    # Try CCE first if requested
    if backend == "cce" and HAS_CCE:
        return _cce_impl(hidden_states, weight, targets, ignore_index, softcap, reduction)

    # Try Liger if requested or as fallback from CCE
    if (backend == "liger" or (backend == "cce" and not HAS_CCE)) and HAS_LIGER:
        return _fused_ce_liger(hidden_states, weight, targets, ignore_index, reduction)

    # Unfused fallback (WARNING: will OOM on large vocab!)
    import warnings
    warnings.warn(
        "Using unfused cross-entropy fallback - this will likely OOM! "
        "Install cut-cross-entropy or liger-kernel for memory-efficient training.",
        RuntimeWarning
    )
    logits = F.linear(hidden_states, weight, bias)
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction=reduction)
