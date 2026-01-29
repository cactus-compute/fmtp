#!/usr/bin/env python
"""
Unified benchmarking/evaluation script for Medusa and Eagle Gemma models.

Evaluates both speculative decoding methods on 4 main benchmarks:
- MMLU (categorical): Multiple choice knowledge test
- ARC (categorical): Science reasoning multiple choice
- GSM8K (generative): Grade school math word problems
- HumanEval (generative): Python code completion

Designed to run locally on CPU for testing/development.

Usage:
    # Medusa model evaluation
    uv run python -m scripts.unified_eval --model-type medusa --checkpoint path/to/medusa_ckpt --task mmlu -n 100

    # Eagle model evaluation
    uv run python -m scripts.unified_eval --model-type eagle --checkpoint path/to/eagle_ckpt --task gsm8k -n 100

    # All benchmarks
    uv run python -m scripts.unified_eval --model-type medusa --checkpoint path/to/ckpt --task all -n 50

    # With custom base model
    uv run python -m scripts.unified_eval --model-type eagle --checkpoint path/to/ckpt --base-model google/gemma-3-1b-it --task arc

    # Baseline (no speculative decoding)
    uv run python -m scripts.unified_eval --model-type baseline --base-model google/gemma-3-270m-it --task mmlu -n 100
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from functools import partial
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F

from tasks.gsm8k import GSM8K
from tasks.arc import ARC
from tasks.mmlu import MMLU
from tasks.humaneval import HumanEval


@dataclass
class EvalResult:
    """Results from a single benchmark evaluation."""
    task: str
    num_samples: int
    num_correct: int
    accuracy: float
    # Speculative decoding metrics (optional)
    mean_acceptance: Optional[float] = None
    tokens_per_second: Optional[float] = None
    total_tokens: Optional[int] = None
    forward_passes: Optional[int] = None
    total_time_seconds: Optional[float] = None
    # Model info
    model_type: Optional[str] = None
    checkpoint: Optional[str] = None
    base_model: Optional[str] = None


def load_task(task_name: str):
    """Load a task by name."""
    task_name = task_name.lower()
    if task_name == 'gsm8k':
        return GSM8K(subset="main", split="test")
    elif task_name == 'arc-easy':
        return ARC(subset="ARC-Easy", split="test")
    elif task_name == 'arc-challenge' or task_name == 'arc':
        return ARC(subset="ARC-Challenge", split="test")
    elif task_name == 'mmlu':
        return MMLU(subset="all", split="test")
    elif task_name == 'humaneval':
        return HumanEval()
    else:
        raise ValueError(f"Unknown task: {task_name}. Supported: gsm8k, arc-easy, arc-challenge, arc, mmlu, humaneval")


def load_medusa_model(checkpoint_path: str, base_model: Optional[str], device: torch.device, dtype: torch.dtype, num_heads_override: Optional[int] = None):
    """Load a Medusa model from checkpoint."""
    from nanochat.gemma_medusa import GemmaMedusaModel, GemmaTokenizerWrapper

    # Load config from checkpoint if available
    # Handle both /path/to/ckpt and /path/to/ckpt/final
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path) and checkpoint_path.endswith("/final"):
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    if not os.path.exists(config_path):
        # Try parent directory
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")

    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    # Use provided base model or from config
    model_name = base_model or config.get('base_model', 'google/gemma-3-270m-it')

    # Get Medusa config from checkpoint
    medusa_num_heads = config.get('medusa_num_heads', 4)
    if num_heads_override is not None:
        medusa_num_heads = num_heads_override
    medusa_num_layers = config.get('medusa_num_layers', 1)
    lora_rank = config.get('lora_rank', 64)
    lora_alpha = config.get('lora_alpha', lora_rank)
    use_multi_layer = config.get('use_multi_layer', False)
    zero_init_mlp = config.get('zero_init_mtp_mlp', True)

    print(f"Loading Medusa model: {model_name}")
    print(f"  Heads: {medusa_num_heads}, Layers: {medusa_num_layers}, LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"  Multi-layer fusion: {use_multi_layer}")

    model = GemmaMedusaModel(
        model_name=model_name,
        medusa_num_heads=medusa_num_heads,
        medusa_num_layers=medusa_num_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        use_multi_layer=use_multi_layer,
        zero_init_mlp=zero_init_mlp,
        device=device,
        dtype=dtype,
        freeze_base=True,
    )

    # Load checkpoint weights
    checkpoint_file = os.path.join(checkpoint_path, "final", "medusa_heads.pt")
    if not os.path.exists(checkpoint_file):
        checkpoint_file = os.path.join(checkpoint_path, "medusa_heads.pt")
    if not os.path.exists(checkpoint_file):
        # Try step checkpoints
        step_dirs = sorted([d for d in os.listdir(checkpoint_path) if d.startswith("step_")])
        if step_dirs:
            checkpoint_file = os.path.join(checkpoint_path, step_dirs[-1], "medusa_heads.pt")

    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint: {checkpoint_file}")
        state = torch.load(checkpoint_file, map_location=device, weights_only=True)

        # Filter state dict to only include first N heads if num_heads_override is set
        if num_heads_override is not None and 'medusa_heads' in state:
            filtered_heads = {}
            for key, value in state['medusa_heads'].items():
                # Keys are like "0.blocks.0.linear.weight", "1.lora_A.weight", etc.
                head_idx = int(key.split('.')[0])
                if head_idx < num_heads_override:
                    filtered_heads[key] = value
            state['medusa_heads'] = filtered_heads

        model.load_medusa_state_dict(state)
        model._checkpoint_path = checkpoint_path
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}, using untrained Medusa heads")

    model.eval()
    tokenizer = GemmaTokenizerWrapper(model_name)

    return model, tokenizer, model_name


def load_eagle_model(checkpoint_path: str, base_model: Optional[str], device: torch.device, dtype: torch.dtype):
    """Load an Eagle model from checkpoint."""
    from nanochat.gemma_eagle import GemmaEagleConfig, GemmaEagleModel, EagleGenerator
    from nanochat.gemma_medusa import GemmaTokenizerWrapper

    # Load config from checkpoint if available
    config_path = os.path.join(checkpoint_path, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    model_name = base_model or config.get('base_model', 'google/gemma-3-270m-it')

    print(f"Loading Eagle model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")

    eagle_config = GemmaEagleConfig(
        base_model_name=model_name,
        freeze_base=True,
    )

    model = GemmaEagleModel(
        eagle_config,
        device=device,
        dtype=dtype,
    )

    # Load checkpoint weights
    checkpoint_file = os.path.join(checkpoint_path, "final", "eagle_draft.pt")
    if not os.path.exists(checkpoint_file):
        checkpoint_file = os.path.join(checkpoint_path, "eagle_draft.pt")
    if not os.path.exists(checkpoint_file):
        # Try step checkpoints
        step_dirs = sorted([d for d in os.listdir(checkpoint_path) if d.startswith("step_")])
        if step_dirs:
            checkpoint_file = os.path.join(checkpoint_path, step_dirs[-1], "eagle_draft.pt")

    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
        model.load_draft_state_dict(checkpoint)
    else:
        print(f"WARNING: No checkpoint found, using untrained draft model")

    model.eval()
    tokenizer = GemmaTokenizerWrapper(model_name)

    return model, tokenizer, model_name


def load_baseline_model(base_model: str, device: torch.device, dtype: torch.dtype):
    """Load a baseline Gemma model (no speculative decoding).

    Uses GemmaMedusaModel with 0 heads so we can use generate_standard_with_cache
    for fair comparison with Medusa's MTP generation.
    """
    from nanochat.gemma_medusa import GemmaMedusaModel, GemmaTokenizerWrapper

    print(f"Loading baseline model: {base_model}")

    # Use MedusaModel with 0 heads - this gives us access to generate_standard_with_cache
    model = GemmaMedusaModel(
        model_name=base_model,
        medusa_num_heads=0,  # No Medusa heads = baseline
        device=device,
        dtype=dtype,
        freeze_base=True,
    )
    model.eval()
    tokenizer = GemmaTokenizerWrapper(base_model)

    return model, tokenizer, base_model


@torch.inference_mode()
def generate_baseline(
    model,
    tokenizer,
    input_ids: List[int],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    eos_token_id: Optional[int] = None,
) -> Tuple[List[int], Dict]:
    """Generate tokens with manual KV caching (matches Medusa's generate_standard_with_cache)."""
    device = model.get_device()
    current_tokens = list(input_ids)

    # Initial prefill - process full prompt
    input_tensor = torch.tensor([current_tokens], dtype=torch.long, device=device)

    outputs = model.model.model(
        input_ids=input_tensor,
        use_cache=True,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state
    past_key_values = outputs.past_key_values

    # Get logits for last position
    logits = model.model.lm_head(hidden_states[:, -1:, :])
    last_logits = logits[0, 0, :]

    # Sample first token
    if temperature < 1e-5:
        next_token = int(last_logits.argmax().item())
    else:
        probs = F.softmax(last_logits / temperature, dim=-1)
        next_token = int(torch.multinomial(probs, num_samples=1).item())

    if eos_token_id is not None and next_token == eos_token_id:
        return current_tokens, {}

    current_tokens.append(next_token)
    current_seq_len = len(current_tokens)

    # Generate remaining tokens with KV cache
    for _ in range(max_new_tokens - 1):
        new_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        new_position_ids = torch.tensor([[current_seq_len - 1]], dtype=torch.long, device=device)

        outputs = model.model.model(
            input_ids=new_token_tensor,
            position_ids=new_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        past_key_values = outputs.past_key_values

        # Get logits
        logits = model.model.lm_head(hidden_states)
        last_logits = logits[0, 0, :]

        # Sample next token
        if temperature < 1e-5:
            next_token = int(last_logits.argmax().item())
        else:
            probs = F.softmax(last_logits / temperature, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())

        if eos_token_id is not None and next_token == eos_token_id:
            break

        current_tokens.append(next_token)
        current_seq_len += 1

    return current_tokens, {}


def run_categorical_eval(
    task_object,
    tokenizer,
    model,
    model_type: str,
    batch_size: int = 1,
    max_problems: Optional[int] = None,
) -> EvalResult:
    """
    Run categorical evaluation (MMLU, ARC).

    For categorical tasks, we don't need generation - just check logits for answer letters.
    Note: tokens/second is not meaningful for categorical tasks since no tokens are generated.
    """
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    letter_to_id_cache = {}
    num_passed, total = 0, 0

    start_time = time.time()

    for i in range(0, num_problems, batch_size):
        i0, i1 = i, min(i + batch_size, num_problems)

        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conv) for conv in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids]
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_tensor = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            if model_type == "medusa":
                logits, _ = model.forward(prompt_tensor, return_medusa=False)
            else:
                logits = model(prompt_tensor)

        for idx, conversation in enumerate(conversations):
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])

            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

        print(f"\r[{total}/{num_problems}] accuracy: {100*num_passed/total:.2f}%", end="", flush=True)

    print()
    elapsed = time.time() - start_time

    return EvalResult(
        task=task_object.__class__.__name__,
        num_samples=total,
        num_correct=num_passed,
        accuracy=num_passed / total,
        # No tokens_per_second for categorical - no generation happens
        total_time_seconds=elapsed,
        model_type=model_type,
    )


def run_generative_eval_medusa(
    task_object,
    tokenizer,
    model,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    max_problems: Optional[int] = None,
    use_cache: bool = True,
    topk: int = 10,
    tree_choices: Optional[List[Tuple[int, ...]]] = None,
) -> EvalResult:
    """Run generative evaluation using Medusa speculative decoding."""
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    eos_token_id = tokenizer.hf_tokenizer.eos_token_id
    total_accepted = 0
    total_forward_passes = 0
    total_tokens = 0
    total_gen_time = 0.0

    # Timing accumulators
    timing_totals = {
        "prefill_s": 0.0,
        "candidate_s": 0.0,
        "tree_verify_s": 0.0,
        "eval_s": 0.0,
        "kv_update_s": 0.0,
        "compute_logits_s": 0.0,
        "medusa_resblock_s": 0.0,
        "medusa_lmhead_s": 0.0,
    }

    num_passed, total = 0, 0

    for i in range(num_problems):
        conversation = task_object[i]
        input_ids = tokenizer.render_for_completion(conversation)

        # Time only generation
        t0 = time.perf_counter()
        if use_cache:
            output_ids, stats = model.generate_mtp_with_cache(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_token_id=eos_token_id,
                collect_timing=True,
                topk=topk,
                tree_choices=tree_choices,
            )
        else:
            output_ids, stats = model.generate_mtp(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_token_id=eos_token_id,
            )
        t1 = time.perf_counter()
        total_gen_time += (t1 - t0)

        # Accumulate timing stats
        if stats.timing:
            for key in timing_totals:
                timing_totals[key] += stats.timing.get(key, 0.0)

        # Decode completion
        completion = tokenizer.decode(output_ids[len(input_ids):])

        # Evaluate
        outcome = task_object.evaluate(conversation, completion)
        num_passed += int(outcome)
        total += 1

        # Accumulate stats - count tokens from output length directly
        gen_tokens = len(output_ids) - len(input_ids)
        total_accepted += stats.total_accepted
        total_forward_passes += stats.forward_passes
        total_tokens += gen_tokens

        mean_acc = total_accepted / total_forward_passes if total_forward_passes > 0 else 0
        tps = total_tokens / total_gen_time if total_gen_time > 0 else 0
        print(f"\r[{total}/{num_problems}] acc: {100*num_passed/total:.2f}%, mean_acc: {mean_acc:.2f}, tok/s: {tps:.1f}", end="", flush=True)

    print()

    # Print timing breakdown
    print("\nTiming breakdown:")
    for key, value in timing_totals.items():
        pct = 100 * value / total_gen_time if total_gen_time > 0 else 0
        print(f"  {key}: {value:.2f}s ({pct:.1f}%)")

    # Print tree info from last stats
    if stats.timing:
        print(f"\nTree info: tree_len={stats.timing.get('tree_len', 'N/A')}, max_speculation={stats.timing.get('max_speculation', 'N/A')}, topk={stats.timing.get('topk', 'N/A')}")

    return EvalResult(
        task=task_object.__class__.__name__,
        num_samples=total,
        num_correct=num_passed,
        accuracy=num_passed / total,
        mean_acceptance=total_accepted / total_forward_passes if total_forward_passes > 0 else None,
        tokens_per_second=total_tokens / total_gen_time if total_gen_time > 0 else None,
        total_tokens=total_tokens,
        forward_passes=total_forward_passes,
        total_time_seconds=total_gen_time,
        model_type="medusa",
    )


def run_generative_eval_eagle(
    task_object,
    tokenizer,
    model,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    max_problems: Optional[int] = None,
) -> EvalResult:
    """Run generative evaluation using Eagle speculative decoding."""
    from nanochat.gemma_eagle import EagleGenerator

    generator = EagleGenerator(model)

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    eos_token_id = tokenizer.hf_tokenizer.eos_token_id

    total_accepted = 0
    total_forward_passes = 0
    total_tokens = 0

    num_passed, total = 0, 0
    start_time = time.time()

    for i in range(num_problems):
        conversation = task_object[i]
        input_ids = tokenizer.render_for_completion(conversation)

        input_tensor = torch.tensor([input_ids], device=model._device)

        output_ids, stats = generator.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=eos_token_id,
            return_stats=True,
        )

        # Decode completion
        completion = tokenizer.decode(output_ids[0, len(input_ids):].tolist())

        # Evaluate
        outcome = task_object.evaluate(conversation, completion)
        num_passed += int(outcome)
        total += 1

        # Accumulate stats
        total_accepted += stats.total_accepted
        total_forward_passes += stats.forward_passes
        total_tokens += stats.tokens_generated

        mean_acc = total_accepted / total_forward_passes if total_forward_passes > 0 else 0
        print(f"\r[{total}/{num_problems}] acc: {100*num_passed/total:.2f}%, mean_acceptance: {mean_acc:.2f}", end="", flush=True)

    print()
    elapsed = time.time() - start_time

    return EvalResult(
        task=task_object.__class__.__name__,
        num_samples=total,
        num_correct=num_passed,
        accuracy=num_passed / total,
        mean_acceptance=total_accepted / total_forward_passes if total_forward_passes > 0 else None,
        tokens_per_second=total_tokens / elapsed if elapsed > 0 else None,
        total_tokens=total_tokens,
        forward_passes=total_forward_passes,
        total_time_seconds=elapsed,
        model_type="eagle",
    )


def run_generative_eval_baseline(
    task_object,
    tokenizer,
    model,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    max_problems: Optional[int] = None,
) -> EvalResult:
    """Run generative evaluation using standard autoregressive decoding with KV cache."""
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    eos_token_id = tokenizer.hf_tokenizer.eos_token_id

    total_tokens = 0
    total_gen_time = 0.0  # Only generation time, not tokenization/decoding
    num_passed, total = 0, 0

    for i in range(num_problems):
        conversation = task_object[i]
        input_ids = tokenizer.render_for_completion(conversation)

        # Time only the generation (use generate_standard_with_cache for fair comparison)
        t0 = time.perf_counter()
        output_ids, forward_passes = model.generate_standard_with_cache(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=eos_token_id,
        )
        t1 = time.perf_counter()

        # Count tokens generated (directly from output length)
        gen_tokens = len(output_ids) - len(input_ids)
        total_tokens += gen_tokens
        total_gen_time += (t1 - t0)

        # Decode completion
        completion = tokenizer.decode(output_ids[len(input_ids):])

        # Evaluate
        outcome = task_object.evaluate(conversation, completion)
        num_passed += int(outcome)
        total += 1

        tps = total_tokens / total_gen_time if total_gen_time > 0 else 0
        print(f"\r[{total}/{num_problems}] acc: {100*num_passed/total:.2f}%, tok/s: {tps:.1f}", end="", flush=True)

    print()

    return EvalResult(
        task=task_object.__class__.__name__,
        num_samples=total,
        num_correct=num_passed,
        accuracy=num_passed / total,
        tokens_per_second=total_tokens / total_gen_time if total_gen_time > 0 else None,
        total_tokens=total_tokens,
        total_time_seconds=total_gen_time,
        model_type="baseline",
    )


def run_evaluation(
    task_name: str,
    model,
    tokenizer,
    model_type: str,
    max_problems: Optional[int] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 1,
    topk: int = 10,
    tree_choices: Optional[List[Tuple[int, ...]]] = None,
) -> EvalResult:
    """Run evaluation on a single task."""
    task_object = load_task(task_name)

    if task_object.eval_type == 'categorical':
        return run_categorical_eval(
            task_object, tokenizer, model, model_type,
            batch_size=batch_size,
            max_problems=max_problems,
        )
    elif task_object.eval_type == 'generative':
        if model_type == 'medusa':
            return run_generative_eval_medusa(
                task_object, tokenizer, model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_problems=max_problems,
                topk=topk,
                tree_choices=tree_choices,
            )
        elif model_type == 'eagle':
            return run_generative_eval_eagle(
                task_object, tokenizer, model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_problems=max_problems,
            )
        else:  # baseline
            return run_generative_eval_baseline(
                task_object, tokenizer, model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_problems=max_problems,
            )
    else:
        raise ValueError(f"Unknown eval type: {task_object.eval_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmarking for Medusa and Eagle Gemma models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-type", "-m", type=str, required=True,
                       choices=["medusa", "eagle", "baseline"],
                       help="Model type: medusa, eagle, or baseline")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                       help="Path to model checkpoint (required for medusa/eagle)")
    parser.add_argument("--base-model", type=str, default=None,
                       help="Base model name (auto-detected from config if not provided)")
    parser.add_argument("--task", "-t", type=str, default="all",
                       help="Task to evaluate: mmlu, arc, arc-easy, arc-challenge, gsm8k, humaneval, or all")
    parser.add_argument("--num-samples", "-n", type=int, default=None,
                       help="Number of samples to evaluate (default: full dataset)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="Max new tokens to generate (for generative tasks)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (0=greedy)")
    parser.add_argument("--batch-size", "-b", type=int, default=4,
                       help="Batch size for categorical evaluation")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--device", type=str, default=None,
                       help="Device: cuda, cpu, or mps (auto-detected if not provided)")
    parser.add_argument("--dtype", type=str, default="auto",
                       choices=["auto", "float32", "bfloat16", "float16"],
                       help="Data type (auto = bfloat16 on CUDA, float32 on CPU)")
    parser.add_argument("--num-heads", type=int, default=None,
                       help="Override number of Medusa heads (for testing with fewer heads)")
    parser.add_argument("--topk", type=int, default=10,
                       help="Top-k predictions per Medusa head (smaller = faster, less accurate)")
    parser.add_argument("--sparse-tree", action="store_true",
                       help="Use sparse tree (12 nodes) instead of optimal tree (80 nodes)")
    args = parser.parse_args()

    # Validate arguments
    if args.model_type in ["medusa", "eagle"] and args.checkpoint is None:
        parser.error(f"--checkpoint is required for model type '{args.model_type}'")
    if args.model_type == "baseline" and args.base_model is None:
        parser.error("--base-model is required for model type 'baseline'")

    # Determine device (prefer CUDA > CPU > MPS for speed)
    # CPU is often faster than MPS for small models due to MPS overhead
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Determine dtype
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    elif args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Model type: {args.model_type}")

    # Load model
    if args.model_type == "medusa":
        model, tokenizer, base_model = load_medusa_model(args.checkpoint, args.base_model, device, dtype, args.num_heads)
    elif args.model_type == "eagle":
        model, tokenizer, base_model = load_eagle_model(args.checkpoint, args.base_model, device, dtype)
    else:  # baseline
        model, tokenizer, base_model = load_baseline_model(args.base_model, device, dtype)

    # Determine tasks to run
    all_tasks = ["mmlu", "arc", "gsm8k", "humaneval"]
    if args.task.lower() == "all":
        tasks = all_tasks
    else:
        tasks = [args.task.lower()]

    # Run evaluations
    results = []
    for task_name in tasks:
        print(f"\n{'='*60}")
        print(f"Evaluating {task_name.upper()}")
        print(f"{'='*60}")

        # Get tree choices if sparse tree is requested
        tree_choices = None
        if args.sparse_tree and args.model_type == 'medusa':
            from nanochat.gemma_medusa.model import SPARSE_TREES
            tree_choices = SPARSE_TREES.get(model.medusa_num_heads, SPARSE_TREES[4])

        result = run_evaluation(
            task_name,
            model, tokenizer,
            model_type=args.model_type,
            max_problems=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            topk=args.topk,
            tree_choices=tree_choices,
        )
        result.checkpoint = args.checkpoint
        result.base_model = base_model

        results.append(result)

        print(f"\nResults for {task_name.upper()}:")
        print(f"  Accuracy: {100*result.accuracy:.2f}% ({result.num_correct}/{result.num_samples})")
        if result.mean_acceptance is not None:
            print(f"  Mean acceptance: {result.mean_acceptance:.3f}")
        if result.tokens_per_second is not None:
            print(f"  Tokens/second: {result.tokens_per_second:.1f}")
        if result.total_time_seconds is not None:
            print(f"  Total time: {result.total_time_seconds:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for result in results:
        line = f"  {result.task}: {100*result.accuracy:.2f}%"
        if result.mean_acceptance is not None:
            line += f" (acceptance: {result.mean_acceptance:.2f})"
        print(line)

    # Calculate aggregate metrics
    if len(results) > 1:
        avg_accuracy = sum(r.accuracy for r in results) / len(results)
        print(f"\n  Average accuracy: {100*avg_accuracy:.2f}%")

        acceptance_results = [r for r in results if r.mean_acceptance is not None]
        if acceptance_results:
            avg_acceptance = sum(r.mean_acceptance for r in acceptance_results) / len(acceptance_results)
            print(f"  Average acceptance: {avg_acceptance:.3f}")

    # Save results
    if args.output:
        output_data = {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "base_model": base_model,
            "device": str(device),
            "dtype": str(dtype),
            "results": [asdict(r) for r in results],
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
