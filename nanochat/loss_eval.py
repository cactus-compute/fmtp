"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        total_tokens += loss2d.numel()
        loss2d = loss2d.view(-1) # flatten
        y = y.view(-1) # flatten
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # sum reduce across all ranks
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    # move to cpu, calculate bpb and ce, and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    total_tokens = total_tokens.item()
    if total_bytes == 0:
        return float('inf'), float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    ce = total_nats / total_tokens
    return bpb, ce


@torch.no_grad()
def evaluate_mtp_loss(model, batches, steps, medusa_loss_weight=1.0, medusa_loss_scheme="constant"):
    """
    Evaluate total MTP loss, first-token loss, and per-head losses for models with Medusa heads.

    Args:
        model: The model to evaluate
        batches: Iterator of (x, y) batches
        steps: Number of evaluation steps
        medusa_loss_weight: Weight for Medusa head losses
        medusa_loss_scheme: "constant" (all heads same weight) or "decay" (weight^k for head k)

    Returns:
        (total_loss, first_token_loss, head_losses): Average losses across all batches
        head_losses is a list of per-head losses for each Medusa head
    """
    device = model.get_device()
    total_loss_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
    first_token_loss_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
    head_loss_sums = None  # Will be initialized on first batch
    num_batches = torch.tensor(0, dtype=torch.int64, device=device)

    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        # Model returns (main_loss, medusa_losses) using fused CE
        first_token_loss, medusa_losses = model(x, y, return_medusa=True)

        # Initialize head_loss_sums on first batch
        if head_loss_sums is None:
            head_loss_sums = [torch.tensor(0.0, dtype=torch.float32, device=device) for _ in range(len(medusa_losses))]

        first_token_loss_sum += first_token_loss

        # Total loss (main head + Medusa heads)
        loss = first_token_loss
        for k, head_loss in enumerate(medusa_losses):
            head_loss_sums[k] += head_loss
            # Apply weighting scheme
            if medusa_loss_scheme == "decay":
                head_weight = medusa_loss_weight ** (k + 1)  # weight^1, weight^2, ...
            else:  # constant
                head_weight = medusa_loss_weight
            loss = loss + head_weight * head_loss
        total_loss_sum += loss
        num_batches += 1

    # sum reduce across all ranks
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(first_token_loss_sum, op=dist.ReduceOp.SUM)
        if head_loss_sums is not None:
            for head_loss_sum in head_loss_sums:
                dist.all_reduce(head_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)

    num_batches_val = num_batches.item()
    if num_batches_val == 0:
        return float('inf'), float('inf'), []
    total_loss = total_loss_sum.item() / num_batches_val
    first_token_loss = first_token_loss_sum.item() / num_batches_val
    head_losses = [h.item() / num_batches_val for h in head_loss_sums] if head_loss_sums is not None else []
    return total_loss, first_token_loss, head_losses
