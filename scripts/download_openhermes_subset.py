#!/usr/bin/env python3
"""
Download a subset of OpenHermes-2.5 for ablation studies.

Usage:
    python -m scripts.download_openhermes_subset --percent 7.5 --output data/openhermes_75k.json
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Download OpenHermes subset")
    parser.add_argument("--percent", type=float, default=7.5,
                        help="Percentage of full dataset to download (default: 7.5)")
    parser.add_argument("--output", type=str, default="data/openhermes_75k.json",
                        help="Output path for JSON file")
    parser.add_argument("--full", action="store_true",
                        help="Download full dataset instead of subset")
    args = parser.parse_args()

    from datasets import load_dataset

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.full:
        print("Downloading full OpenHermes-2.5 dataset (1M samples)...")
        ds = load_dataset('teknium/OpenHermes-2.5', split='train')
    else:
        # OpenHermes-2.5 has ~1M samples, so 7.5% = 75k
        num_samples = int(1_000_000 * args.percent / 100)
        print(f"Downloading {args.percent}% of OpenHermes-2.5 ({num_samples:,} samples)...")
        ds = load_dataset('teknium/OpenHermes-2.5', split=f'train[:{num_samples}]')

    print(f"Saving to {args.output}...")
    ds.to_json(args.output)
    print(f"Done! Saved {len(ds):,} samples.")


if __name__ == "__main__":
    main()
