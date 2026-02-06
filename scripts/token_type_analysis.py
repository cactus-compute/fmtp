"""
Token Type Analysis for Medusa Head Accuracy.

Investigates whether there's a correlation between the type of token being predicted
and the accuracy of the Medusa head. This helps understand when speculation works
best and could inform selective speculation strategies.

Usage:
    python -m scripts.token_type_analysis --checkpoint ~/checkpoints/1h1t \
        --data-path data/wildchat_100k.jsonl --n 1000 --max-tokens 128

Token Types:
    - punctuation: ., , ; : ! ? { } ( ) [ ] etc.
    - whitespace: space, tab, newline
    - number: digits and numeric tokens
    - operator: + - * / = < > etc.
    - keyword: Python/code keywords (def, if, else, return, self, etc.)
    - bracket: { } ( ) [ ]
    - string_marker: quotes ' " ` '''
    - identifier: variable/function names
    - continuation: tokens that continue a word
    - other: anything else
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from nanochat.mlx.model import GemmaMedusaModel


# =============================================================================
# Token Type Classification
# =============================================================================

# Python/code keywords
KEYWORDS = {
    'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except',
    'finally', 'with', 'as', 'import', 'from', 'return', 'yield', 'raise',
    'pass', 'break', 'continue', 'and', 'or', 'not', 'in', 'is', 'None',
    'True', 'False', 'lambda', 'global', 'nonlocal', 'assert', 'del',
    'self', 'cls', 'async', 'await', 'print', 'len', 'range', 'str', 'int',
    'float', 'list', 'dict', 'set', 'tuple', 'type', 'isinstance', 'hasattr',
    # Common code patterns
    'function', 'const', 'let', 'var', 'public', 'private', 'static', 'void',
    # Common words in code comments/docstrings
    'the', 'a', 'an', 'is', 'are', 'to', 'of', 'and', 'or', 'this', 'that',
}

# Punctuation characters
PUNCTUATION = set('.,;:!?')

# Brackets
BRACKETS = set('{}()[]<>')

# Operators
OPERATORS = set('+-*/=<>%&|^~@#$')

# String markers
STRING_MARKERS = set('\'""`')


def classify_token(token_str: str) -> str:
    """
    Classify a token string into a type category.

    Args:
        token_str: The decoded token string

    Returns:
        Token type string
    """
    # Handle empty or whitespace-only tokens
    if not token_str:
        return "empty"

    stripped = token_str.strip()

    # Pure whitespace (including newlines)
    if not stripped:
        if '\n' in token_str:
            return "newline"
        elif '\t' in token_str:
            return "tab"
        else:
            return "space"

    # Check for continuation tokens (tokens that start without space and are lowercase)
    # These are typically word pieces like "ing", "tion", "ed"
    if not token_str[0].isspace() and token_str[0].islower() and len(stripped) <= 4:
        if stripped.isalpha():
            return "continuation"

    # Single character classifications
    if len(stripped) == 1:
        char = stripped
        if char in PUNCTUATION:
            return "punctuation"
        if char in BRACKETS:
            return "bracket"
        if char in OPERATORS:
            return "operator"
        if char in STRING_MARKERS:
            return "string_marker"
        if char.isdigit():
            return "number"
        if char.isalpha():
            # Single letter - could be variable or continuation
            return "identifier"
        return "other"

    # Multi-character tokens
    # Numbers (including floats and scientific notation)
    if re.match(r'^-?[\d,]+\.?\d*(?:e[+-]?\d+)?$', stripped, re.IGNORECASE):
        return "number"

    # Keywords (case-sensitive for Python, case-insensitive for others)
    if stripped.lower() in KEYWORDS or stripped in KEYWORDS:
        return "keyword"

    # Operators (multi-char like ==, !=, <=, etc.)
    if all(c in OPERATORS or c in '=' for c in stripped):
        return "operator"

    # String markers (multi-char like ''', """)
    if all(c in STRING_MARKERS for c in stripped):
        return "string_marker"

    # Brackets with possible content
    if stripped[0] in BRACKETS and stripped[-1] in BRACKETS:
        return "bracket"

    # Identifiers (variable/function names)
    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', stripped):
        return "identifier"

    # Mixed alphanumeric (could be identifiers with numbers)
    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', stripped.replace(' ', '')):
        return "identifier"

    # Indentation patterns
    if token_str.startswith('    ') or token_str.startswith('\t'):
        return "indentation"

    return "other"


@dataclass
class SpeculationRecord:
    """Record of a single speculation attempt."""
    position: int  # Position in generation
    prev_token_id: int
    prev_token_str: str
    prev_token_type: str
    head_prediction: int
    head_prediction_str: str
    head_prediction_type: str
    base_prediction: int  # What the base model actually wanted
    base_prediction_str: str
    base_prediction_type: str
    matched: bool


@dataclass
class TokenTypeStats:
    """Aggregated statistics for a token type."""
    total: int = 0
    matched: int = 0

    @property
    def accuracy(self) -> float:
        return self.matched / max(1, self.total)


@dataclass
class AnalysisResults:
    """Complete analysis results."""
    records: List[SpeculationRecord] = field(default_factory=list)
    # Stats by predicted token type (what the head predicted)
    by_predicted_type: Dict[str, TokenTypeStats] = field(default_factory=lambda: defaultdict(TokenTypeStats))
    # Stats by context token type (what came before)
    by_context_type: Dict[str, TokenTypeStats] = field(default_factory=lambda: defaultdict(TokenTypeStats))
    # Stats by position bucket
    by_position: Dict[str, TokenTypeStats] = field(default_factory=lambda: defaultdict(TokenTypeStats))
    # Joint stats: context_type -> predicted_type -> stats
    joint_stats: Dict[str, Dict[str, TokenTypeStats]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(TokenTypeStats))
    )

    # Overall stats
    total_speculations: int = 0
    total_matched: int = 0
    total_tokens_generated: int = 0
    total_time: float = 0.0


def generate_with_logging(
    model: GemmaMedusaModel,
    input_ids: List[int],
    max_new_tokens: int = 128,
) -> Tuple[List[int], List[SpeculationRecord]]:
    """
    Generate tokens using 1h1t speculation while logging each speculation attempt.

    Returns:
        Tuple of (output_tokens, speculation_records)
    """
    stop_tokens = model.tokenizer.eos_token_ids

    if model.medusa_num_heads == 0:
        raise ValueError("Model has no Medusa heads - cannot use speculation")

    cache = model.base_model.make_cache()
    records = []

    # Prefill
    input_array = mx.array([input_ids], dtype=mx.int32)
    h = model._get_hidden_states(input_array, cache=cache)
    main_logits, medusa_logits = model._compute_logits(
        h, return_medusa=True, last_only=True, num_active_heads=1
    )
    mx.eval(main_logits, medusa_logits)

    output_tokens = list(input_ids)
    num_generated = 0
    position = 0

    while num_generated < max_new_tokens:
        # Get base token prediction
        token1 = int(mx.argmax(main_logits[0, 0]).item())

        # Check stop tokens
        if stop_tokens and token1 in stop_tokens:
            break

        # Get medusa prediction for speculation
        token2 = int(mx.argmax(medusa_logits[0, 0, 0]).item())

        # Forward both tokens with standard causal attention
        spec_input = mx.array([[token1, token2]], dtype=mx.int32)
        h = model._get_hidden_states(spec_input, cache=cache)
        verify_logits, new_medusa = model._compute_logits(
            h, return_medusa=True, last_only=False, num_active_heads=1
        )
        mx.eval(verify_logits, new_medusa)

        # Check if token2 was correct
        verified_token2 = int(mx.argmax(verify_logits[0, 0]).item())

        # Decode tokens for classification
        token1_str = model.tokenizer.decode([token1])
        token2_str = model.tokenizer.decode([token2])
        verified_str = model.tokenizer.decode([verified_token2])

        # Classify tokens
        token1_type = classify_token(token1_str)
        token2_type = classify_token(token2_str)
        verified_type = classify_token(verified_str)

        matched = verified_token2 == token2

        # Record the speculation attempt
        record = SpeculationRecord(
            position=position,
            prev_token_id=token1,
            prev_token_str=token1_str,
            prev_token_type=token1_type,
            head_prediction=token2,
            head_prediction_str=token2_str,
            head_prediction_type=token2_type,
            base_prediction=verified_token2,
            base_prediction_str=verified_str,
            base_prediction_type=verified_type,
            matched=matched,
        )
        records.append(record)

        if matched:
            # Check stop tokens before adding
            if stop_tokens and token2 in stop_tokens:
                output_tokens.extend([token1, token2])
                num_generated += 2
                break

            # Both accepted
            output_tokens.extend([token1, token2])
            num_generated += 2
            position += 2

            # Use logits from position 1 for next iteration
            main_logits = verify_logits[:, 1:2, :]
            medusa_logits = new_medusa[:, :, 1:2, :]
        else:
            # Only token1 accepted - trim the second token from cache
            from nanochat.mlx.model import trim_cache
            trim_cache(cache, 1)

            output_tokens.append(token1)
            num_generated += 1
            position += 1

            # Check if the model wanted to generate a stop token after token1
            if stop_tokens and verified_token2 in stop_tokens:
                break

            # Use logits from position 0 for next iteration
            main_logits = verify_logits[:, 0:1, :]
            medusa_logits = new_medusa[:, :, 0:1, :]

    return output_tokens, records


def position_bucket(pos: int) -> str:
    """Bucket position into ranges."""
    if pos < 10:
        return "0-9"
    elif pos < 25:
        return "10-24"
    elif pos < 50:
        return "25-49"
    elif pos < 100:
        return "50-99"
    else:
        return "100+"


def analyze_records(records: List[SpeculationRecord], results: AnalysisResults):
    """Analyze speculation records and update results."""
    for record in records:
        results.records.append(record)
        results.total_speculations += 1
        if record.matched:
            results.total_matched += 1

        # By predicted token type
        stats = results.by_predicted_type[record.head_prediction_type]
        stats.total += 1
        if record.matched:
            stats.matched += 1

        # By context token type (the token that came before)
        stats = results.by_context_type[record.prev_token_type]
        stats.total += 1
        if record.matched:
            stats.matched += 1

        # By position bucket
        bucket = position_bucket(record.position)
        stats = results.by_position[bucket]
        stats.total += 1
        if record.matched:
            stats.matched += 1

        # Joint stats
        joint = results.joint_stats[record.prev_token_type][record.head_prediction_type]
        joint.total += 1
        if record.matched:
            joint.matched += 1


def load_wildchat_data(data_path: str, n_samples: int) -> List[Dict]:
    """Load n samples from wildchat dataset."""
    samples = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            try:
                data = json.loads(line.strip())
                samples.append(data)
            except json.JSONDecodeError:
                continue
    return samples


def format_prompt(conversation: Dict) -> str:
    """Format a wildchat conversation as a prompt."""
    messages = conversation.get("messages", [])
    if not messages:
        return ""

    # Take the first user message as the prompt
    prompt_parts = []
    for msg in messages:
        if msg["role"] == "user":
            prompt_parts.append(f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n")
        elif msg["role"] == "assistant":
            # Don't include assistant response - we want to generate
            break

    if prompt_parts:
        prompt_parts.append("<start_of_turn>model\n")
        return "".join(prompt_parts)
    return ""


def print_results(results: AnalysisResults):
    """Print analysis results in a formatted table."""
    print("\n" + "=" * 80)
    print("TOKEN TYPE ANALYSIS RESULTS")
    print("=" * 80)

    # Overall stats
    overall_acc = results.total_matched / max(1, results.total_speculations)
    print(f"\nOverall Statistics:")
    print(f"  Total speculations: {results.total_speculations:,}")
    print(f"  Total matched: {results.total_matched:,}")
    print(f"  Overall accuracy: {overall_acc:.1%}")
    print(f"  Tokens generated: {results.total_tokens_generated:,}")
    print(f"  Total time: {results.total_time:.1f}s")
    if results.total_time > 0:
        print(f"  Throughput: {results.total_tokens_generated / results.total_time:.1f} tok/s")

    # By predicted token type
    print(f"\n{'='*60}")
    print("ACCURACY BY PREDICTED TOKEN TYPE")
    print(f"{'='*60}")
    print(f"{'Type':<20} {'Total':>10} {'Matched':>10} {'Accuracy':>10}")
    print("-" * 50)

    sorted_types = sorted(
        results.by_predicted_type.items(),
        key=lambda x: x[1].total,
        reverse=True
    )
    for token_type, stats in sorted_types:
        print(f"{token_type:<20} {stats.total:>10} {stats.matched:>10} {stats.accuracy:>10.1%}")

    # By context token type
    print(f"\n{'='*60}")
    print("ACCURACY BY CONTEXT TOKEN TYPE (token before speculation)")
    print(f"{'='*60}")
    print(f"{'Type':<20} {'Total':>10} {'Matched':>10} {'Accuracy':>10}")
    print("-" * 50)

    sorted_context = sorted(
        results.by_context_type.items(),
        key=lambda x: x[1].total,
        reverse=True
    )
    for token_type, stats in sorted_context:
        print(f"{token_type:<20} {stats.total:>10} {stats.matched:>10} {stats.accuracy:>10.1%}")

    # By position
    print(f"\n{'='*60}")
    print("ACCURACY BY POSITION IN GENERATION")
    print(f"{'='*60}")
    print(f"{'Position':<20} {'Total':>10} {'Matched':>10} {'Accuracy':>10}")
    print("-" * 50)

    for bucket in ["0-9", "10-24", "25-49", "50-99", "100+"]:
        if bucket in results.by_position:
            stats = results.by_position[bucket]
            print(f"{bucket:<20} {stats.total:>10} {stats.matched:>10} {stats.accuracy:>10.1%}")

    # Top joint patterns (context -> predicted)
    print(f"\n{'='*60}")
    print("TOP JOINT PATTERNS (Context -> Predicted Type)")
    print(f"{'='*60}")
    print(f"{'Pattern':<40} {'Total':>10} {'Accuracy':>10}")
    print("-" * 60)

    # Flatten joint stats
    joint_flat = []
    for ctx_type, pred_dict in results.joint_stats.items():
        for pred_type, stats in pred_dict.items():
            if stats.total >= 10:  # Only show patterns with enough data
                joint_flat.append((f"{ctx_type} -> {pred_type}", stats))

    # Sort by total count
    joint_flat.sort(key=lambda x: x[1].total, reverse=True)
    for pattern, stats in joint_flat[:20]:
        print(f"{pattern:<40} {stats.total:>10} {stats.accuracy:>10.1%}")

    # High accuracy patterns (for selective speculation)
    print(f"\n{'='*60}")
    print("HIGH ACCURACY PATTERNS (>60% with n>=20)")
    print(f"{'='*60}")

    high_acc = [(p, s) for p, s in joint_flat if s.accuracy > 0.6 and s.total >= 20]
    high_acc.sort(key=lambda x: x[1].accuracy, reverse=True)
    for pattern, stats in high_acc[:15]:
        print(f"{pattern:<40} {stats.total:>10} {stats.accuracy:>10.1%}")

    # Low accuracy patterns (candidates for skipping)
    print(f"\n{'='*60}")
    print("LOW ACCURACY PATTERNS (<30% with n>=20)")
    print(f"{'='*60}")

    low_acc = [(p, s) for p, s in joint_flat if s.accuracy < 0.3 and s.total >= 20]
    low_acc.sort(key=lambda x: x[1].accuracy)
    for pattern, stats in low_acc[:15]:
        print(f"{pattern:<40} {stats.total:>10} {stats.accuracy:>10.1%}")


def save_results(results: AnalysisResults, output_path: str):
    """Save results to JSON for further analysis."""
    output = {
        "overall": {
            "total_speculations": results.total_speculations,
            "total_matched": results.total_matched,
            "accuracy": results.total_matched / max(1, results.total_speculations),
            "tokens_generated": results.total_tokens_generated,
            "time_seconds": results.total_time,
        },
        "by_predicted_type": {
            k: {"total": v.total, "matched": v.matched, "accuracy": v.accuracy}
            for k, v in results.by_predicted_type.items()
        },
        "by_context_type": {
            k: {"total": v.total, "matched": v.matched, "accuracy": v.accuracy}
            for k, v in results.by_context_type.items()
        },
        "by_position": {
            k: {"total": v.total, "matched": v.matched, "accuracy": v.accuracy}
            for k, v in results.by_position.items()
        },
        "joint_stats": {
            ctx: {
                pred: {"total": s.total, "matched": s.matched, "accuracy": s.accuracy}
                for pred, s in pred_dict.items()
            }
            for ctx, pred_dict in results.joint_stats.items()
        },
        # Sample of individual records for inspection
        "sample_records": [
            {
                "position": r.position,
                "prev_token": r.prev_token_str,
                "prev_type": r.prev_token_type,
                "head_pred": r.head_prediction_str,
                "head_pred_type": r.head_prediction_type,
                "base_pred": r.base_prediction_str,
                "matched": r.matched,
            }
            for r in results.records[:500]  # First 500 records
        ],
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Token type analysis for Medusa speculation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Medusa checkpoint path")
    parser.add_argument("--data-path", type=str, default="data/wildchat_100k.jsonl",
                        help="Path to wildchat data")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of samples to analyze")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum tokens to generate per sample")
    parser.add_argument("--output", type=str, default="token_type_analysis.json",
                        help="Output path for JSON results")
    args = parser.parse_args()

    # Load model
    checkpoint_path = os.path.expanduser(args.checkpoint)
    print(f"Loading model from {checkpoint_path}...")
    model = GemmaMedusaModel.from_checkpoint(
        checkpoint_path=checkpoint_path,
        lazy=True,
    )
    print(f"Model loaded: {model.medusa_num_heads} Medusa heads")

    # Load data
    print(f"\nLoading {args.n} samples from {args.data_path}...")
    samples = load_wildchat_data(args.data_path, args.n)
    print(f"Loaded {len(samples)} samples")

    # Run analysis
    results = AnalysisResults()

    print(f"\nRunning token type analysis on {len(samples)} samples...")
    print("-" * 60)

    start_time = time.perf_counter()

    for i, sample in enumerate(samples):
        prompt = format_prompt(sample)
        if not prompt:
            continue

        try:
            input_ids = model.tokenizer.encode(prompt)
            output_tokens, records = generate_with_logging(
                model, input_ids, max_new_tokens=args.max_tokens
            )

            # Analyze records
            analyze_records(records, results)
            results.total_tokens_generated += len(output_tokens) - len(input_ids)

            # Progress
            if (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - start_time
                current_acc = results.total_matched / max(1, results.total_speculations)
                print(f"  [{i+1}/{len(samples)}] "
                      f"specs={results.total_speculations:,} "
                      f"acc={current_acc:.1%} "
                      f"tokens={results.total_tokens_generated:,} "
                      f"time={elapsed:.1f}s")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    results.total_time = time.perf_counter() - start_time

    # Print and save results
    print_results(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
