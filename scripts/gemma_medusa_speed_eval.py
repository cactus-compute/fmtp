"""
Evaluate Gemma Medusa model on benchmarks with MTP generation speed comparison.

Supported tasks:
- gsm8k: Math word problems (generative)
- arc-easy: ARC-Easy science questions (multiple choice)
- arc-challenge: ARC-Challenge science questions (multiple choice)
- mmlu: MMLU multi-task benchmark (multiple choice)
- humaneval: Code generation benchmark (code execution)

This script compares:
1. Standard autoregressive generation (HuggingFace generate)
2. MTP speculative decoding generation (Medusa)

Example:
    python -m scripts.gemma_medusa_speed_eval \
        --model-name google/gemma-3-270m-it \
        --checkpoint checkpoints/gemma_medusa_final \
        --medusa-num-heads 4 \
        --medusa-num-layers 2 \
        --lora-rank 256 \
        --lora-alpha 512 \
        --zero-init-mtp-mlp \
        --task gsm8k \
        -x 100
"""

import argparse
import json
import os
import re
import time
from contextlib import nullcontext
from datetime import datetime

import torch
import torch.nn.functional as F

from nanochat.common import print0, autodetect_device_type
from nanochat.gemma_medusa import GemmaTokenizerWrapper, load_gemma_medusa_model

from tasks.gsm8k import GSM8K, extract_answer as gsm_extract_answer
from tasks.arc import ARC
from tasks.mmlu import MMLU
from tasks.humaneval import HumanEval


# Extended answer extraction for Gemma (handles #### and \boxed{} formats)
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
FINAL_ANSWER_RE = re.compile(r"(?:final answer|answer)(?:\s+is)?[:\s]*\$?\\?boxed\{?([0-9,.\-]+)\}?\$?", re.IGNORECASE)
NUMBER_RE = re.compile(r"(\-?[0-9,]+(?:\.[0-9]+)?)")

def extract_answer_gemma(completion):
    """Extract numerical answer from Gemma's output."""
    # Try GSM8K format first (#### N)
    answer = gsm_extract_answer(completion)
    if answer:
        return answer

    # Try \boxed{N} format
    match = BOXED_RE.search(completion)
    if match:
        return match.group(1).replace(",", "").strip()

    # Try "final answer is N" pattern
    match = FINAL_ANSWER_RE.search(completion)
    if match:
        return match.group(1).replace(",", "").strip()

    # Last resort: find the last number in the text
    numbers = NUMBER_RE.findall(completion)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def extract_letter_answer(completion, valid_letters=('A', 'B', 'C', 'D')):
    """Extract letter answer from completion for multiple choice tasks."""
    completion = completion.strip()
    # Check if the completion starts with a valid letter
    if completion and completion[0].upper() in valid_letters:
        return completion[0].upper()
    # Look for a letter anywhere in the completion
    for char in completion:
        if char.upper() in valid_letters:
            return char.upper()
    return None


def load_task(task_name, max_problems=None):
    """Load a task by name."""
    task_name = task_name.lower()

    if task_name == 'gsm8k':
        task = GSM8K(subset="main", split="test")
        eval_type = 'generative'
    elif task_name == 'arc-easy':
        task = ARC(subset="ARC-Easy", split="test")
        eval_type = 'categorical'
    elif task_name == 'arc-challenge':
        task = ARC(subset="ARC-Challenge", split="test")
        eval_type = 'categorical'
    elif task_name == 'mmlu':
        task = MMLU(subset="all", split="test")
        eval_type = 'categorical'
    elif task_name == 'humaneval':
        task = HumanEval()
        eval_type = 'code'
    else:
        raise ValueError(f"Unknown task: {task_name}. Supported: gsm8k, arc-easy, arc-challenge, mmlu, humaneval")

    return task, eval_type


def evaluate_response(conversation, completion_text, eval_type, task=None):
    """Evaluate a model response based on task type."""
    if eval_type == 'generative':
        # GSM8K-style: extract numbers and compare
        assistant_message = conversation['messages'][-1]
        if isinstance(assistant_message['content'], list):
            last_text_part = assistant_message['content'][-1]['text']
        else:
            last_text_part = assistant_message['content']
        ref_answer = gsm_extract_answer(last_text_part)
        pred_answer = extract_answer_gemma(completion_text)
        return pred_answer == ref_answer, pred_answer, ref_answer
    elif eval_type == 'code':
        # HumanEval-style: execute code and check tests pass
        correct = task.evaluate(conversation, completion_text)
        return correct, None, None
    else:
        # Categorical: extract letter and compare
        valid_letters = conversation.get('letters', ('A', 'B', 'C', 'D'))
        ref_answer = conversation['messages'][-1]['content']
        pred_answer = extract_letter_answer(completion_text, valid_letters)
        return pred_answer == ref_answer, pred_answer, ref_answer


def run_mtp_eval(model, tokenizer, task, eval_type, max_problems, max_new_tokens, temperature,
                 use_fixed_size_tree=False, use_heuristic_tree=False, collect_timing=False):
    """Run evaluation using MTP speculative decoding."""
    num_problems = min(len(task), max_problems) if max_problems else len(task)

    total_tokens = 0
    total_time = 0.0
    total_forward_passes = 0
    num_passed = 0
    timing_totals = None
    timing_meta = None
    timing_iters = 0

    eos_token_id = tokenizer.hf_tokenizer.eos_token_id

    for i in range(num_problems):
        conversation = task[i]

        # Get prompt tokens
        prompt_ids = tokenizer.render_for_completion(conversation)

        # Generate with MTP
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        output_ids, stats = model.generate_mtp_with_cache(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=eos_token_id,
            use_fixed_size_tree=use_fixed_size_tree,
            use_heuristic_tree=use_heuristic_tree,
            collect_timing=collect_timing,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Decode completion
        completion_ids = output_ids[len(prompt_ids):]
        completion_text = tokenizer.decode(completion_ids)

        # Clean up end of turn marker
        if "<end_of_turn>" in completion_text:
            completion_text = completion_text.split("<end_of_turn>")[0]

        # Evaluate
        correct, _, _ = evaluate_response(conversation, completion_text, eval_type, task=task)
        num_passed += int(correct)

        # Stats
        gen_tokens = stats.tokens_generated
        total_tokens += gen_tokens
        total_time += (t1 - t0)
        total_forward_passes += stats.forward_passes
        if collect_timing and stats.timing is not None:
            if timing_totals is None:
                timing_totals = {k: 0.0 for k in stats.timing.keys() if k.endswith("_s")}
                timing_meta = {
                    k: stats.timing.get(k)
                    for k in ("tree_len", "max_speculation", "topk")
                    if k in stats.timing
                }
            for key, value in stats.timing.items():
                if key.endswith("_s"):
                    timing_totals[key] += float(value)
                elif key == "iterations":
                    timing_iters += int(value)

        print(f"\r[MTP] {i+1}/{num_problems} | {num_passed}/{i+1} ({100*num_passed/(i+1):.1f}%) | "
              f"mean_accepted={stats.mean_accepted_length:.2f}", end='', flush=True)

    print()

    accuracy = num_passed / num_problems
    tok_per_sec = total_tokens / total_time if total_time > 0 else 0
    mean_accepted = total_tokens / total_forward_passes if total_forward_passes > 0 else 0

    result = {
        'accuracy': accuracy,
        'tokens': total_tokens,
        'time_seconds': total_time,
        'tokens_per_second': tok_per_sec,
        'forward_passes': total_forward_passes,
        'mean_accepted_length': mean_accepted,
        'num_correct': num_passed,
        'num_total': num_problems,
    }
    if collect_timing and timing_totals is not None:
        decode_keys = ["candidate_s", "tree_verify_s", "eval_s", "kv_update_s", "compute_logits_s"]
        decode_s = sum(timing_totals.get(k, 0.0) for k in decode_keys)
        per_iter_ms = (decode_s / timing_iters * 1000.0) if timing_iters > 0 else 0.0
        per_token_ms = (decode_s / total_tokens * 1000.0) if total_tokens > 0 else 0.0
        result["timing"] = {
            "totals_s": timing_totals,
            "prefill_s": timing_totals.get("prefill_s", 0.0),
            "decode_s": decode_s,
            "iterations": timing_iters,
            "per_iter_ms": per_iter_ms,
            "per_token_ms": per_token_ms,
            "meta": timing_meta or {},
        }
        print0(f"Timing (MTP decode): {per_iter_ms:.3f} ms/iter, {per_token_ms:.3f} ms/token")
        if decode_s > 0:
            parts = []
            for key in decode_keys:
                if key in timing_totals:
                    parts.append(f"{key}={100*timing_totals[key]/decode_s:.1f}%")
            if parts:
                print0("Timing breakdown: " + ", ".join(parts))

    return result


def run_standard_eval(model, tokenizer, task, eval_type, max_problems, max_new_tokens, temperature):
    """Run evaluation using standard autoregressive generation."""
    num_problems = min(len(task), max_problems) if max_problems else len(task)

    total_tokens = 0
    total_time = 0.0
    total_forward_passes = 0
    num_passed = 0

    eos_token_id = tokenizer.hf_tokenizer.eos_token_id

    for i in range(num_problems):
        conversation = task[i]

        # Get prompt tokens
        prompt_ids = tokenizer.render_for_completion(conversation)

        # Generate with standard autoregressive
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        output_ids, forward_passes = model.generate_standard_with_cache(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=eos_token_id,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Decode completion
        completion_ids = output_ids[len(prompt_ids):]
        completion_text = tokenizer.decode(completion_ids)

        # Clean up end of turn marker
        if "<end_of_turn>" in completion_text:
            completion_text = completion_text.split("<end_of_turn>")[0]

        # Evaluate
        correct, _, _ = evaluate_response(conversation, completion_text, eval_type, task=task)
        num_passed += int(correct)

        # Stats
        gen_tokens = len(completion_ids)
        total_tokens += gen_tokens
        total_time += (t1 - t0)
        total_forward_passes += forward_passes

        print(f"\r[Standard] {i+1}/{num_problems} | {num_passed}/{i+1} ({100*num_passed/(i+1):.1f}%)",
              end='', flush=True)

    print()

    accuracy = num_passed / num_problems
    tok_per_sec = total_tokens / total_time if total_time > 0 else 0

    return {
        'accuracy': accuracy,
        'tokens': total_tokens,
        'time_seconds': total_time,
        'tokens_per_second': tok_per_sec,
        'forward_passes': total_forward_passes,
        'num_correct': num_passed,
        'num_total': num_problems,
    }


def load_checkpoint_config(checkpoint_path: str) -> dict:
    """Load config.json from checkpoint directory if it exists."""
    # Try to find config.json
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', type=str, default=None,
                        help='Model name (auto-detected from checkpoint config.json if not specified)')
    parser.add_argument('--medusa-num-heads', type=int, default=None,
                        help='Number of Medusa heads (auto-detected from checkpoint if not specified)')
    parser.add_argument('--medusa-num-layers', type=int, default=None,
                        help='ResBlock layers per head (auto-detected from checkpoint if not specified)')
    parser.add_argument('--lora-rank', type=int, default=None,
                        help='LoRA rank (auto-detected from checkpoint if not specified)')
    parser.add_argument('--lora-alpha', type=int, default=None,
                        help='LoRA alpha (auto-detected from checkpoint if not specified)')
    parser.add_argument('--checkpoint', type=str, default=None, required=True)
    parser.add_argument('--zero-init-mtp-mlp', action='store_true', default=None,
                        help='Use zero-init for MTP MLP (auto-detected from checkpoint if not specified)')
    parser.add_argument('-x', '--max-problems', type=int, default=100)
    parser.add_argument('--max-new-tokens', type=int, default=None,
                        help='Max tokens to generate (default: 512 for generative, 16 for categorical)')
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--task', type=str, default='gsm8k',
                        choices=['gsm8k', 'arc-easy', 'arc-challenge', 'mmlu', 'humaneval'],
                        help='Evaluation task (default: gsm8k)')
    parser.add_argument('--skip-standard', action='store_true', help='Skip standard generation baseline')
    parser.add_argument('--inference-num-heads', type=int, default=None,
                        help='Override number of heads to use during inference (for ablation testing)')
    parser.add_argument('--fixed-tree-size', action='store_true',
                        help='Use fixed 79-node tree for fair ablation comparison across head counts')
    parser.add_argument('--use-heuristic-tree', action='store_true',
                        help='Use heuristic tree instead of calibrated optimal tree (default: use optimal)')
    parser.add_argument('--use-head-mixer', action='store_true', default=None,
                        help='Use cross-head MLP mixer (auto-detected from checkpoint if not specified)')
    parser.add_argument('--mixer-hidden', type=int, default=None,
                        help='Hidden dimension for the cross-head mixer MLP (auto-detected from checkpoint)')
    parser.add_argument('--attn-num-layers', type=int, default=None,
                        help='Number of attention blocks for cross-head mixing (auto-detected from checkpoint)')
    parser.add_argument('--use-multi-layer', action='store_true', default=None,
                        help='Use multi-layer hidden state fusion (auto-detected from checkpoint)')
    parser.add_argument('--timing', action='store_true',
                        help='Collect detailed MTP timing (adds CUDA sync overhead)')
    args = parser.parse_args()

    # Load checkpoint config and merge with args (args take precedence if specified)
    ckpt_config = load_checkpoint_config(args.checkpoint)

    # Helper to get config value with fallback
    def get_cfg(arg_val, config_key, default):
        if arg_val is not None:
            return arg_val
        return ckpt_config.get(config_key, default)

    # Apply config with fallbacks (checkpoint config -> defaults)
    model_name = get_cfg(args.model_name, 'base_model', 'google/gemma-3-270m-it')
    medusa_num_heads = get_cfg(args.medusa_num_heads, 'medusa_num_heads', 4)
    medusa_num_layers = get_cfg(args.medusa_num_layers, 'medusa_num_layers', 2)
    lora_rank = get_cfg(args.lora_rank, 'lora_rank', 64)
    lora_alpha = get_cfg(args.lora_alpha, 'lora_alpha', lora_rank)
    zero_init_mlp = get_cfg(args.zero_init_mtp_mlp, 'zero_init_mtp_mlp', True)
    mixer_hidden = get_cfg(args.mixer_hidden, 'mlp_mixer_hidden', 16)
    attn_num_layers = get_cfg(args.attn_num_layers, 'attn_num_layers', 0)
    use_multi_layer = get_cfg(args.use_multi_layer, 'use_multi_layer', False)

    # Auto-detect head mixer type from config
    # Config has 'use_mlp_mixer' for MLP mixer and 'attn_num_layers' for attention mixer
    if args.use_head_mixer is not None:
        use_head_mixer = args.use_head_mixer
    else:
        use_head_mixer = ckpt_config.get('use_mlp_mixer', False) or attn_num_layers > 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print0(f"Loading model: {model_name}")
    if ckpt_config:
        print0(f"Auto-loaded config from: {args.checkpoint}/config.json")
    if args.timing:
        print0("Timing enabled: tok/s will include CUDA sync overhead")
    print0(f"Medusa config: {medusa_num_heads} heads, {medusa_num_layers} layers, rank={lora_rank}, alpha={lora_alpha}")

    # Determine mixer type
    mixer_type = "attention" if attn_num_layers > 0 else "mlp"
    if use_head_mixer:
        print0(f"Mixer: {mixer_type}, layers={attn_num_layers if mixer_type == 'attention' else 1}")

    model = load_gemma_medusa_model(
        model_name=model_name,
        medusa_num_heads=medusa_num_heads,
        medusa_num_layers=medusa_num_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=device,
        dtype=dtype,
        zero_init_mlp=zero_init_mlp,
        use_head_mixer=use_head_mixer,
        mixer_hidden=mixer_hidden,
        mixer_type=mixer_type,
        attn_num_layers=attn_num_layers,
        use_multi_layer=use_multi_layer,
    )

    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint, "final", "medusa_heads.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(args.checkpoint, "medusa_heads.pt")
    print0(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Load all Medusa weights using unified method
    warnings = model.load_medusa_state_dict(checkpoint, strict=True)
    for w in warnings:
        print0(f"WARNING: {w}")
    model._checkpoint_path = args.checkpoint  # Store for optimal tree generation from head_acc.json
    model.eval()
    print0(f"Medusa parameters: {model.get_medusa_param_count():,}")

    # Set inference num heads for ablation testing
    if args.inference_num_heads is not None:
        if args.inference_num_heads > medusa_num_heads:
            raise ValueError(f"inference_num_heads ({args.inference_num_heads}) cannot exceed medusa_num_heads ({medusa_num_heads})")
        # Simply override medusa_num_heads - tree buffers will be regenerated
        model.medusa_num_heads = args.inference_num_heads
        # Clear tree buffer cache to force regeneration
        model._tree_buffers_cache = None
        model._tree_buffers_config = None
        print0(f"Using {args.inference_num_heads} heads for inference (ablation mode)")

    tokenizer = GemmaTokenizerWrapper(model_name)

    # Load task
    task, eval_type = load_task(args.task)
    print0(f"{args.task.upper()} test set: {len(task)} problems")
    print0(f"Evaluating on {args.max_problems} problems")

    # Set default max_new_tokens based on task type
    if args.max_new_tokens is None:
        if eval_type == 'categorical':
            args.max_new_tokens = 16
        else:  # generative or code
            args.max_new_tokens = 512

    results = {}

    # Run standard generation baseline (optional)
    if not args.skip_standard:
        print0("\n" + "="*50)
        print0("Running Standard Autoregressive Generation")
        print0("="*50)
        with torch.amp.autocast('cuda', dtype=dtype):
            results['standard'] = run_standard_eval(
                model, tokenizer, task, eval_type, args.max_problems, args.max_new_tokens, args.temperature
            )
        print0(f"Standard: {100*results['standard']['accuracy']:.2f}% accuracy, "
               f"{results['standard']['tokens_per_second']:.1f} tok/s")

    # Run MTP generation
    effective_heads = args.inference_num_heads if args.inference_num_heads else medusa_num_heads
    print0("\n" + "="*50)
    tree_info = "fixed=79" if args.fixed_tree_size else "topk=10"
    tree_type = "heuristic" if args.use_heuristic_tree else "optimal"
    print0(f"Running MTP Speculative Decoding (heads={effective_heads}, {tree_info}, tree={tree_type})")
    print0("="*50)
    with torch.amp.autocast('cuda', dtype=dtype):
        results['mtp'] = run_mtp_eval(
            model, tokenizer, task, eval_type, args.max_problems, args.max_new_tokens, args.temperature,
            use_fixed_size_tree=args.fixed_tree_size,
            use_heuristic_tree=args.use_heuristic_tree,
            collect_timing=args.timing,
        )
    print0(f"MTP: {100*results['mtp']['accuracy']:.2f}% accuracy, "
           f"{results['mtp']['tokens_per_second']:.1f} tok/s, "
           f"mean_accepted={results['mtp']['mean_accepted_length']:.2f}")

    # Summary
    print0("\n" + "="*50)
    print0("SUMMARY")
    print0("="*50)

    if 'standard' in results:
        print0(f"Standard:  {100*results['standard']['accuracy']:.2f}% accuracy | "
               f"{results['standard']['tokens_per_second']:.1f} tok/s")
    print0(f"MTP:       {100*results['mtp']['accuracy']:.2f}% accuracy | "
           f"{results['mtp']['tokens_per_second']:.1f} tok/s | "
           f"mean_accepted={results['mtp']['mean_accepted_length']:.2f}")

    if 'standard' in results:
        speedup = results['mtp']['tokens_per_second'] / results['standard']['tokens_per_second']
        mean_accepted = results['mtp']['mean_accepted_length']
        # Tree overhead factor: how many single-token passes a tree verification costs
        # speedup = mean_accepted / tree_overhead, so tree_overhead = mean_accepted / speedup
        tree_overhead = mean_accepted / speedup if speedup > 0 else float('inf')
        print0(f"\nSpeedup: {speedup:.2f}x (wall-clock)")
        print0(f"Tree overhead factor: {tree_overhead:.2f}x (break-even at mean_accepted={tree_overhead:.2f})")

    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'task': args.task,
        'model_name': model_name,
        'checkpoint': args.checkpoint,
        'config': {
            'medusa_num_heads': medusa_num_heads,
            'inference_num_heads': effective_heads,
            'medusa_num_layers': medusa_num_layers,
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'max_problems': args.max_problems,
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'use_fixed_size_tree': args.fixed_tree_size,
            'use_heuristic_tree': args.use_heuristic_tree,
        },
        'results': results,
    }

    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"{args.task}_mtp_eval_{timestamp}.json"

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print0(f"\nResults saved to: {args.output}")
