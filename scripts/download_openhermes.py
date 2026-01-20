#!/usr/bin/env python
"""
Download and pre-filter OpenHermes dataset for Medusa training.

Filters out conversations that would have 0 valid assistant tokens after truncation,
which would cause NaN in cross-entropy loss.

Usage:
    python -m scripts.download_openhermes --output data/openhermes_10k.json --samples 10000
    python -m scripts.download_openhermes --output data/openhermes.json  # full dataset
"""

import argparse
import json
from multiprocessing import Pool, cpu_count
from functools import partial
from datasets import load_dataset
from tqdm import tqdm

# Global tokenizer for worker processes
_tokenizer = None


def init_worker(model_name):
    """Initialize tokenizer in each worker process."""
    global _tokenizer
    from nanochat.gemma_medusa.tokenizer import GemmaTokenizerWrapper
    _tokenizer = GemmaTokenizerWrapper(model_name)


def check_conversation(doc, max_seq_len):
    """
    Check if a conversation has at least 1 valid assistant token after truncation.
    Returns (doc, is_valid) tuple.
    """
    global _tokenizer
    try:
        ids, mask = _tokenizer.render_conversation(doc, max_tokens=max_seq_len)
        valid_count = sum(mask[1:])  # shifted targets
        return (doc, valid_count >= 1)
    except Exception:
        return (doc, False)


def convert_to_messages_format(item):
    """Convert OpenHermes item to messages format."""
    if 'conversations' in item:
        messages = []
        for turn in item['conversations']:
            role = turn.get('from', turn.get('role', 'user'))
            content = turn.get('value', turn.get('content', ''))
            if role == 'human':
                role = 'user'
            elif role == 'gpt':
                role = 'assistant'
            messages.append({'role': role, 'content': content})
        return {'messages': messages}
    elif 'messages' in item:
        return item
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description='Download and filter OpenHermes dataset')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples (default: full dataset)')
    parser.add_argument('--max-seq-len', type=int, default=2048, help='Max sequence length for filtering')
    parser.add_argument('--model', type=str, default='google/gemma-3-1b-it', help='Model for tokenizer')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers (default: cpu_count)')
    args = parser.parse_args()

    num_workers = args.workers or cpu_count()
    print(f"Using {num_workers} workers")

    # Download dataset
    print("Downloading OpenHermes dataset...")
    if args.samples:
        ds = load_dataset('teknium/OpenHermes-2.5', split=f'train[:{args.samples}]')
    else:
        ds = load_dataset('teknium/OpenHermes-2.5', split='train')
    print(f"Downloaded {len(ds)} samples")

    # Convert to messages format
    print("Converting to messages format...")
    conversations = []
    for item in tqdm(ds, desc="Converting"):
        conv = convert_to_messages_format(item)
        if conv:
            conversations.append(conv)
    print(f"Converted {len(conversations)} conversations")

    # Filter using multiprocessing
    print(f"Filtering conversations with 0 valid tokens (max_seq_len={args.max_seq_len})...")
    check_fn = partial(check_conversation, max_seq_len=args.max_seq_len)

    filtered = []
    skipped = 0

    with Pool(num_workers, initializer=init_worker, initargs=(args.model,)) as pool:
        results = list(tqdm(
            pool.imap(check_fn, conversations, chunksize=100),
            total=len(conversations),
            desc="Filtering"
        ))

    for doc, is_valid in results:
        if is_valid:
            filtered.append(doc)
        else:
            skipped += 1

    print(f"Kept {len(filtered)} conversations, skipped {skipped} ({100*skipped/len(conversations):.2f}%)")

    # Save to JSON
    print(f"Saving to {args.output}...")
    with open(args.output, 'w') as f:
        for conv in filtered:
            f.write(json.dumps(conv) + '\n')

    print(f"Done! Saved {len(filtered)} conversations to {args.output}")


if __name__ == '__main__':
    main()
