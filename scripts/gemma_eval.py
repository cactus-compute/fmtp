"""
Evaluate Gemma 3 models using nanochat's evaluation infrastructure.

This is Phase 0 of the Gemma Medusa plan: verify Gemma 3 works with
the existing eval suite before adding MTP heads.

Example runs:
    # Categorical evaluation (no generation) - single GPU
    uv run python -m scripts.gemma_eval -a ARC-Easy -b 16

    # Generative evaluation with batching - single GPU
    uv run python -m scripts.gemma_eval -a GSM8K -n 1 --gen-batch-size 8

    # Multi-GPU with torchrun (8 GPUs)
    torchrun --standalone --nproc_per_node=8 -m scripts.gemma_eval -- -a ARC-Easy -b 16

    # With Medusa model (for speed comparison)
    uv run python -m scripts.gemma_eval --medusa -a GSM8K

    # All tasks
    uv run python -m scripts.gemma_eval
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from functools import partial
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanochat.common import compute_init, compute_cleanup, get_dist_info, print0, autodetect_device_type
from nanochat.gemma_medusa import GemmaTokenizerWrapper, load_gemma_model, load_gemma_medusa_model

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K, extract_answer as gsm_extract_answer
from tasks.spellingbee import SpellingBee


# Extended answer extraction for Gemma (handles #### and \boxed{} formats)
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
FINAL_ANSWER_RE = re.compile(r"(?:final answer|answer)(?:\s+is)?[:\s]*\$?\\?boxed\{?([0-9,.\-]+)\}?\$?", re.IGNORECASE)
NUMBER_RE = re.compile(r"(\-?[0-9,]+(?:\.[0-9]+)?)")

def extract_answer_gemma(completion):
    """
    Extract numerical answer from Gemma's output.
    Handles multiple formats: #### N, \\boxed{N}, "final answer is N", or last number.
    """
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


class GSM8KGemmaWrapper:
    """Wrapper around GSM8K that uses Gemma-compatible answer extraction."""

    def __init__(self, task):
        self.task = task

    def __len__(self):
        return len(self.task)

    def __getitem__(self, idx):
        return self.task[idx]

    @property
    def eval_type(self):
        return self.task.eval_type

    def evaluate(self, conversation, assistant_response):
        """Evaluate using Gemma-compatible extraction."""
        # Get ground truth
        assistant_message = conversation['messages'][-1]
        last_text_part = assistant_message['content'][-1]['text']
        ref_num = gsm_extract_answer(last_text_part)

        # Get predicted answer using extended extraction
        pred_num = extract_answer_gemma(assistant_response)

        return int(pred_num == ref_num)

# -----------------------------------------------------------------------------
# Generative evaluation loop

def get_hf_model(model):
    """Get the underlying HuggingFace model for generation.

    Works with both GemmaModelWrapper and GemmaMedusaModel.
    """
    if hasattr(model, 'base_model'):
        # GemmaMedusaModel - base_model is the HF model
        return model.base_model
    elif hasattr(model, 'model'):
        # GemmaModelWrapper - model is the HF model
        return model.model
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def run_generative_eval(task_object, tokenizer, model, num_samples, max_new_tokens, temperature, top_k, max_problems=None, batch_size=8, repetition_penalty=1.0):
    """Run batched generative evaluation using HuggingFace generate.

    Returns: (accuracy, total_tokens, total_time_seconds)
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # Get terminal tokens
    end_of_turn = tokenizer.hf_tokenizer.encode("<end_of_turn>", add_special_tokens=False)
    eos_token_id = tokenizer.hf_tokenizer.eos_token_id
    pad_token_id = tokenizer.hf_tokenizer.pad_token_id or eos_token_id

    # Set pad token for batched generation
    tokenizer.hf_tokenizer.pad_token_id = pad_token_id
    tokenizer.hf_tokenizer.padding_side = "left"  # Left-pad for generation

    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    num_passed, total = 0, 0
    total_generated_tokens = 0
    total_generation_time = 0.0

    # Get underlying HF model for generate()
    hf_model = get_hf_model(model)

    for batch_idx in range(ddp_rank, num_batches, ddp_world_size):
        i0 = batch_idx * batch_size
        i1 = min((batch_idx + 1) * batch_size, num_problems)
        batch_conversations = [task_object[i] for i in range(i0, i1)]
        actual_batch_size = len(batch_conversations)

        # Tokenize all prompts in the batch
        encoded_prompts = [tokenizer.render_for_completion(conv) for conv in batch_conversations]
        prompt_lengths = [len(p) for p in encoded_prompts]
        max_prompt_len = max(prompt_lengths)

        # Left-pad prompts to same length
        padded_prompts = []
        attention_masks = []
        for prompt in encoded_prompts:
            pad_len = max_prompt_len - len(prompt)
            padded_prompts.append([pad_token_id] * pad_len + prompt)
            attention_masks.append([0] * pad_len + [1] * len(prompt))

        prompt_ids = torch.tensor(padded_prompts, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

        # Generate completions for the entire batch with timing
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = hf_model.generate(
                prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_samples,
                temperature=temperature if temperature > 0 else 1.0,
                top_k=top_k if top_k else 50,
                do_sample=temperature > 0,
                pad_token_id=pad_token_id,
                eos_token_id=[eos_token_id] + end_of_turn if end_of_turn else [eos_token_id],
                repetition_penalty=repetition_penalty,
            )

        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.perf_counter()
        batch_time = t1 - t0
        total_generation_time += batch_time

        # Count generated tokens (exclude prompt)
        batch_generated_tokens = sum(len(out) - max_prompt_len for out in outputs)
        total_generated_tokens += batch_generated_tokens

        # Process outputs: each input generates num_samples outputs
        for idx, conversation in enumerate(batch_conversations):
            # Get outputs for this input (num_samples consecutive outputs)
            start_out_idx = idx * num_samples
            end_out_idx = start_out_idx + num_samples

            completions = []
            for out_idx in range(start_out_idx, end_out_idx):
                output = outputs[out_idx]
                # Find where actual content starts (skip left padding)
                completion_ids = output[max_prompt_len:].tolist()
                completion_text = tokenizer.decode(completion_ids)
                # Clean up end of turn marker
                if "<end_of_turn>" in completion_text:
                    completion_text = completion_text.split("<end_of_turn>")[0]
                completions.append(completion_text)

            # Evaluate
            outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
            passed = any(outcomes)

            total += 1
            num_passed += int(passed)

        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    print()

    # Aggregate across ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    # Aggregate timing stats across ranks
    if ddp:
        tokens_tensor = torch.tensor([total_generated_tokens], dtype=torch.long, device=device)
        time_tensor = torch.tensor([total_generation_time], dtype=torch.float64, device=device)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(time_tensor, op=dist.ReduceOp.SUM)
        total_generated_tokens = tokens_tensor.item()
        total_generation_time = time_tensor.item()

    tokens_per_second = total_generated_tokens / total_generation_time if total_generation_time > 0 else 0
    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100*num_passed/total:.2f}%)")
    print0(f"Generated {total_generated_tokens} tokens in {total_generation_time:.2f}s ({tokens_per_second:.1f} tok/s)")
    return num_passed / total, total_generated_tokens, total_generation_time

# -----------------------------------------------------------------------------
# Categorical evaluation loop

def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):
    """Run categorical evaluation (multiple choice)."""
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    letter_to_id_cache = {}
    num_passed, total = 0, 0

    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids]
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(prompt_ids)  # (B, T, V)

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

    # Aggregate across ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed / total
    print0(f"Final: {num_passed}/{total} ({100*average:.2f}%)")
    return average

# -----------------------------------------------------------------------------

def run_gemma_eval(task_name, model, tokenizer,
                   batch_size=1, num_samples=1, max_new_tokens=512, temperature=0.0, top_k=50,
                   max_problems=None, gen_batch_size=8, repetition_penalty=1.0):
    """Run evaluation on a specific task.

    Returns: dict with 'accuracy' and optionally 'tokens', 'time_seconds', 'tokens_per_second'
    """
    task_module = {
        'HumanEval': HumanEval,
        'MMLU': partial(MMLU, subset="all", split="test"),
        'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
        'GSM8K': partial(GSM8K, subset="main", split="test"),
        'SpellingBee': partial(SpellingBee, size=256, split="test"),
    }[task_name]
    task_object = task_module()

    # Wrap GSM8K with Gemma-compatible answer extraction
    if task_name == 'GSM8K':
        task_object = GSM8KGemmaWrapper(task_object)

    if task_object.eval_type == 'generative':
        acc, tokens, time_sec = run_generative_eval(
            task_object, tokenizer, model, num_samples, max_new_tokens, temperature, top_k,
            max_problems=max_problems, batch_size=gen_batch_size, repetition_penalty=repetition_penalty
        )
        return {
            'accuracy': acc,
            'tokens': tokens,
            'time_seconds': time_sec,
            'tokens_per_second': tokens / time_sec if time_sec > 0 else 0,
        }
    elif task_object.eval_type == 'categorical':
        acc = run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=max_problems)
        return {'accuracy': acc}
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', type=str, default='google/gemma-3-1b-it',
                        help="HuggingFace model name (default: google/gemma-3-1b-it)")
    parser.add_argument('-a', '--task-name', type=str, default=None,
                        help="Task name. Default = all tasks. Use | to split multiple tasks.")
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8,
                        help='Batch size for categorical evaluation')
    parser.add_argument('--gen-batch-size', type=int, default=8,
                        help='Batch size for generative evaluation')
    parser.add_argument('-x', '--max-problems', type=int, default=None,
                        help='Max problems to evaluate')
    parser.add_argument('--repetition-penalty', type=float, default=1.1,
                        help='Repetition penalty to discourage infinite loops (default: 1.1)')
    parser.add_argument('--device-type', type=str, default='',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device type: cuda|cpu|mps. Empty => autodetect')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output JSON file to save results (default: auto-generated)')
    parser.add_argument('--medusa', action='store_true',
                        help='Use GemmaMedusaModel instead of base GemmaModelWrapper')
    parser.add_argument('--medusa-num-heads', type=int, default=4,
                        help='Number of Medusa heads (default: 4)')
    parser.add_argument('--medusa-num-layers', type=int, default=1,
                        help='Number of ResBlock layers per Medusa head (default: 1)')
    parser.add_argument('--lora-rank', type=int, default=64,
                        help='LoRA rank for Medusa heads (default: 64)')
    parser.add_argument('--lora-alpha', type=int, default=None,
                        help='LoRA alpha (default: same as rank)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint directory to load trained Medusa heads')
    parser.add_argument('--zero-init-mtp-mlp', action='store_true',
                        help='Zero-initialize ResBlock MLP weights (must match training config)')
    args = parser.parse_args()

    # Initialize compute
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # Load Gemma model and tokenizer
    print0(f"Loading model: {args.model_name}")
    if args.medusa:
        lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank
        print0(f"Using GemmaMedusaModel with {args.medusa_num_heads} heads, {args.medusa_num_layers} layers, rank={args.lora_rank}, alpha={lora_alpha}")
        model = load_gemma_medusa_model(
            args.model_name,
            medusa_num_heads=args.medusa_num_heads,
            medusa_num_layers=args.medusa_num_layers,
            lora_rank=args.lora_rank,
            lora_alpha=lora_alpha,
            device=device,
            dtype=ptdtype,
            zero_init_mlp=args.zero_init_mtp_mlp,
        )
        print0(f"Medusa parameters: {model.get_medusa_param_count():,}")

        # Load checkpoint if provided
        if args.checkpoint:
            checkpoint_path = os.path.join(args.checkpoint, "final", "medusa_heads.pt")
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(args.checkpoint, "medusa_heads.pt")
            if os.path.exists(checkpoint_path):
                print0(f"Loading Medusa heads from: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.medusa_heads.load_state_dict(checkpoint['medusa_heads'])
                print0("Medusa heads loaded successfully")
            else:
                print0(f"WARNING: Checkpoint not found at {args.checkpoint}")
    else:
        model = load_gemma_model(args.model_name, device=device, dtype=ptdtype)
    tokenizer = GemmaTokenizerWrapper(args.model_name)
    print0(f"Model loaded. Config: {model.config}")
    print0(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Get tasks to evaluate
    all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
    baseline_accuracies = {
        'ARC-Easy': 0.25,
        'ARC-Challenge': 0.25,
        'MMLU': 0.25,
        'GSM8K': 0.0,
        'HumanEval': 0.0,
        'SpellingBee': 0.0,
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split('|')

    # Run evaluations
    results = {}
    for task_name in task_names:
        print0(f"\n{'='*50}")
        print0(f"Evaluating: {task_name}")
        print0(f"{'='*50}")
        with autocast_ctx:
            result = run_gemma_eval(
                task_name,
                model, tokenizer,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                max_problems=args.max_problems,
                gen_batch_size=args.gen_batch_size,
                repetition_penalty=args.repetition_penalty,
            )
            results[task_name] = result
            print0(f"{task_name} accuracy: {100 * result['accuracy']:.2f}%")

    # Summary
    print0(f"\n{'='*50}")
    print0("SUMMARY")
    print0(f"{'='*50}")
    for task_name, result in results.items():
        acc = result['accuracy']
        baseline = baseline_accuracies.get(task_name, 0.0)
        improvement = acc - baseline
        line = f"{task_name}: {100*acc:.2f}% (baseline: {100*baseline:.2f}%, +{100*improvement:.2f}%)"
        if 'tokens_per_second' in result:
            line += f" | {result['tokens_per_second']:.1f} tok/s"
        print0(line)

    # Calculate centered mean if all tasks evaluated
    chatcore_metric = None
    if all(task_name in results for task_name in all_tasks):
        centered_mean = 0
        for task_name, result in results.items():
            acc = result['accuracy']
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc) if baseline_acc < 1.0 else 0.0
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(results)
        print0(f"\nChatCORE metric: {chatcore_metric:.4f}")

    # Save results to JSON (only on rank 0)
    if ddp_rank == 0:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'model_name': args.model_name,
            'medusa': args.medusa,
            'config': {
                'temperature': args.temperature,
                'max_new_tokens': args.max_new_tokens,
                'num_samples': args.num_samples,
                'top_k': args.top_k,
                'batch_size': args.batch_size,
                'gen_batch_size': args.gen_batch_size,
                'repetition_penalty': args.repetition_penalty,
                'max_problems': args.max_problems,
                'dtype': args.dtype,
                'world_size': ddp_world_size,
            },
            'results': results,
            'chatcore_metric': chatcore_metric,
        }
        if args.medusa:
            output_data['medusa_config'] = {
                'num_heads': args.medusa_num_heads,
                'num_layers': args.medusa_num_layers,
                'lora_rank': args.lora_rank,
            }

        # Auto-generate output filename if not specified
        if args.output is None:
            model_short = args.model_name.split('/')[-1]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            medusa_suffix = "_medusa" if args.medusa else ""
            output_file = f"gemma_eval_{model_short}{medusa_suffix}_{timestamp}.json"
        else:
            output_file = args.output

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print0(f"\nResults saved to: {output_file}")

    compute_cleanup()
