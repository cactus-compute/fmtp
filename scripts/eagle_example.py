"""
Example script demonstrating EAGLE-3 speculative decoding for Gemma3.

This shows how to:
1. Load the base model with EAGLE draft model
2. Train the draft model (brief overview)
3. Generate with speculative decoding
4. Benchmark speedup

Usage:
    # Quick test (no trained draft - will show baseline generation)
    uv run python -m scripts.eagle_example --quick-test

    # With trained draft model
    uv run python -m scripts.eagle_example --draft-checkpoint checkpoints/eagle/best.pt
"""

import argparse
import torch

from nanochat.gemma_eagle import (
    GemmaEagleConfig,
    GemmaEagleModel,
    EagleGenerator,
    benchmark_eagle,
)


def parse_args():
    parser = argparse.ArgumentParser(description="EAGLE-3 example for Gemma3")
    parser.add_argument("--base-model", type=str, default="google/gemma-3-1b-it",
                        help="Base model name")
    parser.add_argument("--draft-checkpoint", type=str, default=None,
                        help="Path to trained draft model checkpoint")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run quick test without trained draft")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark on sample prompts")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?",
                        help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Device: {device}")
    print(f"Loading base model: {args.base_model}")

    # Create config
    config = GemmaEagleConfig(
        base_model_name=args.base_model,
        draft_depth=5,
        draft_top_k=8,
        total_tokens=63,
    )

    # Create model
    model = GemmaEagleModel(
        config,
        device=device,
        dtype=dtype,
    )

    # Load draft checkpoint if provided
    if args.draft_checkpoint:
        print(f"Loading draft checkpoint: {args.draft_checkpoint}")
        state = torch.load(args.draft_checkpoint, map_location=device)
        model.load_draft_state_dict(state)
        print("Draft model loaded successfully!")
    elif not args.quick_test:
        print("\nNote: No draft checkpoint provided.")
        print("The draft model is randomly initialized and will not provide speedup.")
        print("Train with: uv run python -m scripts.eagle_train --help")
        print()

    # Create generator
    generator = EagleGenerator(model)

    print(f"\nGenerating response for: {args.prompt}")
    print("-" * 50)

    # Generate
    output, stats = generator.generate_simple(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(f"Output: {output}")
    print("-" * 50)

    # Print stats
    print(f"\nGeneration Statistics:")
    print(f"  Tokens generated: {stats.tokens_generated}")
    print(f"  Base model forward passes: {stats.forward_passes}")
    print(f"  Draft model forward passes: {stats.draft_passes}")
    print(f"  Mean accepted length: {stats.mean_accepted_length:.2f}")
    print(f"  Acceptance rate: {stats.acceptance_rate:.2%}")
    print(f"  Time elapsed: {stats.time_elapsed:.2f}s")
    print(f"  Tokens/second: {stats.tokens_per_second:.1f}")

    # Run benchmark if requested
    if args.benchmark:
        print("\n" + "=" * 50)
        print("Running benchmark...")

        prompts = [
            "Explain the theory of relativity in simple terms.",
            "Write a short poem about the ocean.",
            "What are the main differences between Python and JavaScript?",
            "Describe the process of photosynthesis.",
            "How does a computer processor work?",
        ]

        results = benchmark_eagle(
            model,
            prompts,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            warmup_runs=1,
        )

        print(f"\nBenchmark Results ({len(prompts)} prompts):")
        print(f"  Total tokens: {results['total_tokens']}")
        print(f"  Total time: {results['total_time']:.2f}s")
        print(f"  Tokens/second: {results['tokens_per_second']:.1f}")
        print(f"  Mean accepted length: {results['mean_accepted_length']:.2f}")
        print(f"  Acceptance rate: {results['acceptance_rate']:.2%}")


if __name__ == "__main__":
    main()
