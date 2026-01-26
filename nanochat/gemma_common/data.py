"""
Shared data loading utilities for Gemma Medusa and EAGLE training.

Supports ShareGPT and OpenAI message formats.
"""

import json
from typing import List, Dict, Tuple, Optional, Generator, Any
import torch


def load_sharegpt_data(filepath: str) -> List[Dict]:
    """
    Load ShareGPT-format JSON data.

    Supports both JSONL and JSON array formats.
    Expected format:
    {
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    or:
    {
        "messages": [...]
    }

    Args:
        filepath: Path to JSON or JSONL file

    Returns:
        List of conversation dicts with 'messages' key
    """
    conversations = []
    with open(filepath, 'r') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON array format
            data = json.loads(content)
            for item in data:
                if 'conversations' in item:
                    # ShareGPT format with "from"/"value" keys
                    messages = []
                    for turn in item['conversations']:
                        role = turn.get('from', turn.get('role'))
                        content = turn.get('value', turn.get('content'))
                        if role == 'human':
                            role = 'user'
                        elif role == 'gpt':
                            role = 'assistant'
                        messages.append({'role': role, 'content': content})
                    conversations.append({'messages': messages})
                elif 'messages' in item:
                    conversations.append(item)
        else:
            # JSONL format
            for line in content.split('\n'):
                if line.strip():
                    item = json.loads(line)
                    if 'conversations' in item:
                        messages = []
                        for turn in item['conversations']:
                            role = turn.get('from', turn.get('role'))
                            content = turn.get('value', turn.get('content'))
                            if role == 'human':
                                role = 'user'
                            elif role == 'gpt':
                                role = 'assistant'
                            messages.append({'role': role, 'content': content})
                        conversations.append({'messages': messages})
                    elif 'messages' in item:
                        conversations.append(item)
    return conversations


def filter_dataset(
    dataset: List[Dict],
    tokenizer,
    max_seq_len: int,
    min_valid_tokens: int = 10,
) -> Tuple[List[Dict], int]:
    """
    Filter dataset to remove conversations without enough valid (assistant) tokens.

    Args:
        dataset: List of conversations
        tokenizer: Tokenizer wrapper with render_conversation method
        max_seq_len: Maximum sequence length
        min_valid_tokens: Minimum number of valid (non-masked) tokens required

    Returns:
        Tuple of (filtered_dataset, num_skipped)
    """
    filtered = []
    skipped = 0
    for doc in dataset:
        ids, mask = tokenizer.render_conversation(doc, max_tokens=max_seq_len)
        # Count valid tokens (mask=1 means assistant content we train on)
        # We check mask[1:] because targets are shifted by 1
        valid_count = sum(mask[1:])
        if valid_count >= min_valid_tokens:
            filtered.append(doc)
        else:
            skipped += 1
    return filtered, skipped


def split_train_val(
    dataset: List[Dict],
    val_samples: int,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into train and validation sets.

    Args:
        dataset: Full dataset
        val_samples: Number of samples for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data)
    """
    import random
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    val_indices = indices[:val_samples]
    train_indices = indices[val_samples:]
    val_data = [dataset[i] for i in val_indices]
    train_data = [dataset[i] for i in train_indices]
    return train_data, val_data


def data_generator(
    dataset: List[Dict],
    tokenizer,
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    include_loss_mask: bool = False,
) -> Generator[Tuple[Tuple[torch.Tensor, ...], int], None, None]:
    """
    Generate batches of tokenized conversations.

    Yields (inputs, targets) or (inputs, targets, loss_mask) where targets is
    shifted by 1 from inputs. The loss mask indicates which positions to train on.

    Args:
        dataset: List of conversations
        tokenizer: Tokenizer wrapper with render_conversation method
        batch_size: Per-device batch size
        max_seq_len: Maximum sequence length
        device: Device to place tensors on
        ddp_rank: Rank in distributed training
        ddp_world_size: World size for distributed training
        include_loss_mask: Whether to include loss_mask in output

    Yields:
        Tuple of ((inputs, targets) or (inputs, targets, loss_mask), epoch)
    """
    pad_token_id = tokenizer.hf_tokenizer.eos_token_id

    def collate_and_yield(batch):
        nrows = len(batch)
        # Truncate to max_seq_len and find max length
        batch = [(ids[:max_seq_len], mask[:max_seq_len]) for ids, mask in batch]
        ncols = max(len(ids) for ids, mask in batch) - 1  # seq of n creates inputs/targets of n-1

        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)  # -1 is ignore index

        if include_loss_mask:
            loss_mask = torch.zeros((nrows, ncols), dtype=torch.float)

        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            # Apply mask: -1 where mask is 0
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets

            if include_loss_mask:
                loss_mask[i, :n-1] = mask_tensor.float()

        inputs = inputs.to(device)
        targets = targets.to(device)

        if include_loss_mask:
            loss_mask = loss_mask.to(device)
            return inputs, targets, loss_mask
        return inputs, targets

    # Iterate over dataset in epochs
    epoch = 0
    while True:
        batch = []
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc, max_tokens=max_seq_len)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch), epoch
                batch = []
        epoch += 1


def simple_data_generator(
    dataset: List[Dict],
    tokenizer,
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
) -> Generator[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], int], None, None]:
    """
    Simple data generator that always includes loss_mask.

    This is a convenience wrapper around data_generator for EAGLE training
    which always needs the loss mask.

    Yields:
        Tuple of ((input_ids, attention_mask, loss_mask), epoch)
    """
    pad_token_id = tokenizer.hf_tokenizer.eos_token_id

    def collate_batch(batch):
        nrows = len(batch)
        batch = [(ids[:max_seq_len], mask[:max_seq_len]) for ids, mask in batch]
        ncols = max(len(ids) for ids, mask in batch)

        input_ids = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((nrows, ncols), dtype=torch.long)
        loss_mask = torch.zeros((nrows, ncols), dtype=torch.float)

        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :n] = 1
            loss_mask[i, :n] = torch.tensor(mask, dtype=torch.float)

        return (
            input_ids.to(device),
            attention_mask.to(device),
            loss_mask.to(device),
        )

    epoch = 0
    while True:
        batch = []
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc, max_tokens=max_seq_len)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_batch(batch), epoch
                batch = []
        epoch += 1
