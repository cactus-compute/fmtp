#!/usr/bin/env python
"""
Generate self-distillation dataset from WildChat prompts using a local model.

This script:
1. Downloads WildChat dataset (user prompts from real ChatGPT conversations)
2. Sends user prompts to a local vLLM server running your target model
3. Collects the model's own responses as training data for Medusa heads

Usage:
    # First, start a vLLM server with your model:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        uv run vllm serve google/gemma-3-270m-it \
        --host 0.0.0.0 --port 8000 \
        --dtype auto \
        --data-parallel-size 8 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --max-model-len 2048

    # Then run this script:
    python -m scripts.generate_wildchat_distill \
        --output data/wildchat_distill.jsonl \
        --num-samples 10000 \
        --num-threads 64

    # For multi-turn conversations (slower but more thorough):
    python -m scripts.generate_wildchat_distill \
        --output data/wildchat_distill.jsonl \
        --num-samples 10000 \
        --multi-turn
"""

import argparse
import json
import os
import concurrent.futures
from typing import Optional

import openai
from datasets import load_dataset
from tqdm import tqdm
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential


def discover_api_servers(base_port: int = 8000, max_servers: int = 10) -> list[str]:
    """Discover running vLLM API servers on consecutive ports."""
    api_bases = []
    for i in range(max_servers):
        api_base = f"http://localhost:{base_port + i}/v1"
        try:
            client = openai.OpenAI(api_key="EMPTY", base_url=api_base)
            models = client.models.list()
            if models.data:
                print(f"Found server at {api_base}: {models.data[0].id}")
                api_bases.append(api_base)
        except Exception:
            break
    return api_bases


def should_retry(exception):
    """Don't retry on BadRequestError - the input is invalid."""
    return not isinstance(exception, openai.BadRequestError)


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(should_retry),
)
def generate_single_turn(
    client: openai.OpenAI,
    model_name: str,
    user_content: str,
    system_content: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Optional[str]:
    """Generate a single assistant response for a user message."""
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if response.choices[0].finish_reason == "length":
        return None  # Skip truncated responses

    return response.choices[0].message.content.strip()


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(should_retry),
)
def generate_multi_turn(
    client: openai.OpenAI,
    model_name: str,
    user_messages: list[str],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Optional[list[dict]]:
    """Generate a full multi-turn conversation from user messages."""
    messages = []
    output_messages = []

    for user_content in user_messages:
        messages.append({"role": "user", "content": user_content})

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if response.choices[0].finish_reason == "length":
                break  # Stop if we hit length limit

            assistant_content = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": assistant_content})

            output_messages.append({"role": "user", "content": user_content})
            output_messages.append({"role": "assistant", "content": assistant_content})
        except Exception:
            break

    return output_messages if output_messages else None


def validate_message(content: str, max_chars: int = 8000) -> bool:
    """Check if a message is valid for the model."""
    if not content or not content.strip():
        return False
    # Skip very long messages (likely to exceed context)
    if len(content) > max_chars:
        return False
    return True


def estimate_tokens(text: str) -> int:
    """Conservative estimate of token count (~3 chars/token for mixed content)."""
    return len(text) // 3 + 1


def process_wildchat_sample(
    sample: dict,
    idx: int,
    api_bases: list[str],
    model_name: str,
    multi_turn: bool,
    temperature: float,
    max_seq_len: int,
    min_output_tokens: int = 64,
) -> Optional[dict]:
    """Process a single WildChat sample and generate model responses."""
    # Load balance across servers
    api_base = api_bases[idx % len(api_bases)]
    client = openai.OpenAI(api_key="EMPTY", base_url=api_base)

    # WildChat format: list of {"role": "user"/"assistant", "content": "..."}
    conversation = sample.get("conversation", [])
    if not conversation:
        return None

    # Extract and validate user messages
    user_messages = [
        turn["content"] for turn in conversation
        if turn.get("role") == "user" and validate_message(turn.get("content", ""))
    ]

    if not user_messages:
        return None

    try:
        if multi_turn:
            # For multi-turn, estimate total input and compute remaining budget
            total_input_tokens = sum(estimate_tokens(msg) for msg in user_messages)
            max_tokens = max_seq_len - total_input_tokens - 50  # 50 token buffer for formatting
            if max_tokens < min_output_tokens:
                return None  # Input too long
            output = generate_multi_turn(
                client, model_name, user_messages, temperature, max_tokens
            )
            if output:
                return {"messages": output}
        else:
            # Single turn: just use first user message
            input_tokens = estimate_tokens(user_messages[0])
            max_tokens = max_seq_len - input_tokens - 50  # 50 token buffer for formatting
            if max_tokens < min_output_tokens:
                return None  # Input too long
            response = generate_single_turn(
                client, model_name, user_messages[0], None, temperature, max_tokens
            )
            if response:
                return {
                    "messages": [
                        {"role": "user", "content": user_messages[0]},
                        {"role": "assistant", "content": response},
                    ]
                }
    except openai.BadRequestError as e:
        # Log the actual error message for debugging (but don't spam)
        if idx < 10:  # Only log first few to diagnose
            print(f"BadRequest sample {idx}: {e.message[:200] if hasattr(e, 'message') else str(e)[:200]}")
    except Exception as e:
        print(f"Error processing sample {idx}: {type(e).__name__}: {e}")

    return None


def main():
    parser = argparse.ArgumentParser(description="Generate self-distillation dataset from WildChat")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of WildChat samples to process")
    parser.add_argument("--num-threads", type=int, default=64, help="Number of concurrent threads")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max total sequence length (input + output)")
    parser.add_argument("--multi-turn", action="store_true", help="Generate full multi-turn conversations")
    parser.add_argument("--base-port", type=int, default=8000, help="Base port for vLLM servers")
    parser.add_argument("--dataset", type=str, default="allenai/WildChat-1M",
                        help="WildChat dataset variant (allenai/WildChat-1M, allenai/WildChat-nontoxic, etc.)")
    parser.add_argument("--language", type=str, default="English",
                        help="Filter by language (or 'all' for no filter)")
    args = parser.parse_args()

    # Discover API servers
    print("Discovering vLLM API servers...")
    api_bases = discover_api_servers(args.base_port)
    if not api_bases:
        print("ERROR: No vLLM servers found. Start one with:")
        print("  python -m vllm.entrypoints.openai.api_server --model google/gemma-3-1b-it --port 8000")
        return
    print(f"Found {len(api_bases)} server(s)")

    # Get model name from first server
    client = openai.OpenAI(api_key="EMPTY", base_url=api_bases[0])
    model_name = client.models.list().data[0].id
    print(f"Using model: {model_name}")

    # Download WildChat dataset
    print(f"Downloading {args.dataset}...")
    ds = load_dataset(args.dataset, split=f"train[:{args.num_samples * 2}]")  # Download extra for filtering
    print(f"Downloaded {len(ds)} samples")

    # Filter by language if specified
    if args.language != "all":
        print(f"Filtering for {args.language} conversations...")
        # WildChat has language detection per turn
        filtered = []
        for sample in ds:
            conv = sample.get("conversation", [])
            if conv and any(
                turn.get("language") == args.language
                for turn in conv
                if turn.get("role") == "user"
            ):
                filtered.append(sample)
                if len(filtered) >= args.num_samples:
                    break
        samples = filtered
        print(f"Filtered to {len(samples)} {args.language} samples")
    else:
        samples = list(ds)[:args.num_samples]

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Resume support: check existing output
    start_idx = 0
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            start_idx = sum(1 for _ in f)
        print(f"Resuming from sample {start_idx}")
        samples = samples[start_idx:]

    if not samples:
        print("All samples already processed!")
        return

    # Generate completions
    print(f"Generating completions for {len(samples)} samples...")

    completed = 0
    with open(args.output, "a") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = {
                executor.submit(
                    process_wildchat_sample,
                    sample,
                    idx + start_idx,
                    api_bases,
                    model_name,
                    args.multi_turn,
                    args.temperature,
                    args.max_seq_len,
                ): idx
                for idx, sample in enumerate(samples)
            }

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Generating"
            ):
                result = future.result()
                if result:
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    completed += 1

    print(f"Done! Generated {completed} conversations, saved to {args.output}")


if __name__ == "__main__":
    main()
