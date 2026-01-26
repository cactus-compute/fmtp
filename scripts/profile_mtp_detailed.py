"""Detailed profiling of MTP generation to understand performance bottlenecks."""
import torch
import time
import argparse
from nanochat.gemma_medusa.model import GemmaMedusaModel, get_default_tree_choices, generate_tree_buffers

def profile_with_cuda_events(fn, warmup=5, iterations=50, name=""):
    """Profile a function using CUDA events for accurate GPU timing."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std, times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--medusa-num-heads', type=int, default=4)
    parser.add_argument('--medusa-num-layers', type=int, default=2)
    parser.add_argument('--lora-rank', type=int, default=64)
    parser.add_argument('--lora-alpha', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda')

    print("="*70)
    print("DETAILED MTP PERFORMANCE PROFILING")
    print("="*70)

    # Load model
    print(f"\nLoading model...")
    model = GemmaMedusaModel(
        model_name='google/gemma-3-270m-it',
        medusa_num_heads=args.medusa_num_heads,
        medusa_num_layers=args.medusa_num_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=device,
        dtype=torch.bfloat16,
    )

    import os
    checkpoint_path = os.path.join(args.checkpoint, "final", "medusa_heads.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(args.checkpoint, "medusa_heads.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Load all Medusa weights using unified method
    warnings = model.load_medusa_state_dict(checkpoint, strict=True)
    for w in warnings:
        print(f"WARNING: {w}")
    model.eval()

    print(f"Model loaded. Medusa config: {args.medusa_num_heads} heads, {args.medusa_num_layers} layers, rank={args.lora_rank}")

    # Setup
    prompt_len = 100
    prompt = torch.randint(0, 1000, (1, prompt_len), device=device)

    tree_choices = get_default_tree_choices(args.medusa_num_heads, topk=10)
    buffers = generate_tree_buffers(tree_choices, device, topk=10)
    tree_len = len(tree_choices) + 1

    print(f"\nTest setup:")
    print(f"  Prompt length: {prompt_len} tokens")
    print(f"  Tree size: {tree_len} nodes ({len(tree_choices)} candidates)")
    print(f"  Tree depth distribution: {dict((d, sum(1 for c in tree_choices if len(c)==d)) for d in range(1, args.medusa_num_heads+1))}")

    # =========================================================================
    # PART 1: Standard Generation Baseline
    # =========================================================================
    print("\n" + "="*70)
    print("PART 1: STANDARD GENERATION (single token)")
    print("="*70)

    # 1a. Full forward pass (no cache)
    def std_forward_no_cache():
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            return model.forward(prompt, return_medusa=False, last_only=True)

    avg, std, _ = profile_with_cuda_events(std_forward_no_cache, name="std_forward_no_cache")
    print(f"\n1a. Full forward (no cache, {prompt_len} tokens): {avg:.3f}ms ± {std:.3f}ms")

    # 1b. Initial cache fill
    def fill_cache():
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            return model._get_hidden_states_with_cache(prompt)

    avg_fill, std_fill, _ = profile_with_cuda_events(fill_cache, name="fill_cache")
    print(f"1b. Cache fill ({prompt_len} tokens): {avg_fill:.3f}ms ± {std_fill:.3f}ms")

    # Get a fresh cache for subsequent tests
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _, base_cache = model._get_hidden_states_with_cache(prompt)

    # 1c. Single token forward with cache
    single_token = torch.randint(0, 1000, (1, 1), device=device)
    pos_ids = torch.tensor([[prompt_len]], device=device)

    def std_single_token_cached():
        # Clone cache to avoid mutation issues
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            hs, _ = model._get_hidden_states_with_cache(single_token, past_key_values=base_cache, position_ids=pos_ids)
            return model._compute_logits(hs, return_medusa=False, last_only=True)

    avg_single, std_single, _ = profile_with_cuda_events(std_single_token_cached, name="std_single_token")
    print(f"1c. Single token forward (with cache): {avg_single:.3f}ms ± {std_single:.3f}ms")

    # Breakdown of single token
    def std_hidden_only():
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            return model._get_hidden_states_with_cache(single_token, past_key_values=base_cache, position_ids=pos_ids)

    def std_lm_head_only():
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            hs, _ = model._get_hidden_states_with_cache(single_token, past_key_values=base_cache, position_ids=pos_ids)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            return model._compute_logits(hs, return_medusa=False, last_only=True)

    avg_hs, _, _ = profile_with_cuda_events(std_hidden_only, name="hidden_only")
    print(f"    - Transformer forward: {avg_hs:.3f}ms")
    print(f"    - LM head: {avg_single - avg_hs:.3f}ms")

    # =========================================================================
    # PART 2: MTP Components Breakdown
    # =========================================================================
    print("\n" + "="*70)
    print("PART 2: MTP GENERATION COMPONENTS")
    print("="*70)

    # Get fresh cache and initial logits
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        hidden_states, fresh_cache = model._get_hidden_states_with_cache(prompt)
        main_logits, medusa_logits = model._compute_logits(hidden_states, return_medusa=True, last_only=True)

    # 2a. Generate candidates
    def gen_candidates():
        return model._generate_candidates(
            main_logits[:, 0, :], medusa_logits[:, :, 0, :], buffers, topk=10, temperature=0.0
        )

    avg_cand, std_cand, _ = profile_with_cuda_events(gen_candidates, name="gen_candidates")
    print(f"\n2a. Generate candidates: {avg_cand:.3f}ms ± {std_cand:.3f}ms")

    candidates, tree_candidates = gen_candidates()
    print(f"    - Candidates shape: {candidates.shape}")
    print(f"    - Tree candidates: {tree_candidates.shape[0]} tokens")

    # 2b. Tree verification forward pass (forward_mtp_with_cache)
    # Profile with fresh cache each iteration (to measure realistic tree verification cost)
    # Note: time.perf_counter based profiling like profile_mtp_old.py to match
    times = []
    for _ in range(50):
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, pv = model._get_hidden_states_with_cache(prompt)  # fresh cache
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            tree_logits_out, ret_indices_out, valid_mask_out, _, tree_hs_out = model.forward_mtp_with_cache(
                tree_candidates, buffers, pv, prompt_len
            )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    avg_tree = sum(times)/len(times)*1000
    std_tree = (sum((t*1000 - avg_tree) ** 2 for t in times) / len(times)) ** 0.5
    print(f"\n2b. Tree verification forward: {avg_tree:.3f}ms ± {std_tree:.3f}ms")
    print(f"    - Processing {tree_len} tree tokens with {prompt_len}-token cache")

    # Break down tree forward
    print(f"\n    Breakdown of tree forward pass:")

    # 2b-i. Attention mask construction
    tree_attn_mask = buffers["tree_attn_mask"]
    def build_attn_mask():
        cache_attn = torch.ones(1, 1, tree_len, prompt_len, device=device, dtype=tree_attn_mask.dtype)
        full_attn_mask = torch.cat([cache_attn, tree_attn_mask], dim=-1)
        hf_attn_mask = torch.where(
            full_attn_mask > 0.5,
            torch.zeros_like(full_attn_mask),
            torch.full_like(full_attn_mask, float('-inf'))
        )
        return hf_attn_mask

    avg_mask, _, _ = profile_with_cuda_events(build_attn_mask, name="attn_mask")
    print(f"    - Attention mask build: {avg_mask:.3f}ms")

    # 2b-ii. Tree transformer forward
    tree_input = tree_candidates.unsqueeze(0)
    tree_pos_ids = prompt_len + buffers["tree_position_ids"].unsqueeze(0)
    hf_attn_mask = build_attn_mask()

    def tree_transformer():
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, pv = model._get_hidden_states_with_cache(prompt)
            return model._get_hidden_states_with_cache(
                tree_input, past_key_values=pv, position_ids=tree_pos_ids, attention_mask=hf_attn_mask
            )

    avg_tree_tfm, _, _ = profile_with_cuda_events(tree_transformer, name="tree_transformer")
    print(f"    - Tree transformer ({tree_len} tokens): {avg_tree_tfm:.3f}ms")

    # 2b-iii. Tree LM head
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _, pv = model._get_hidden_states_with_cache(prompt)
        tree_hidden, _ = model._get_hidden_states_with_cache(
            tree_input, past_key_values=pv, position_ids=tree_pos_ids, attention_mask=hf_attn_mask
        )

    def tree_lm_head():
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            return model._compute_logits(tree_hidden, return_medusa=False)

    avg_tree_lm, _, _ = profile_with_cuda_events(tree_lm_head, name="tree_lm_head")
    print(f"    - Tree LM head ({tree_len} positions): {avg_tree_lm:.3f}ms")

    # 2c. Evaluate candidates
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _, pv = model._get_hidden_states_with_cache(prompt)
        tree_logits, ret_indices, valid_mask, _, tree_hs = model.forward_mtp_with_cache(
            tree_candidates, buffers, pv, prompt_len
        )

    def eval_candidates():
        return model._evaluate_candidates_greedy_fast(tree_logits, candidates, ret_indices, valid_mask)

    avg_eval, std_eval, _ = profile_with_cuda_events(eval_candidates, name="eval_candidates")
    print(f"\n2c. Evaluate candidates: {avg_eval:.3f}ms ± {std_eval:.3f}ms")

    best_cand, accept_len = eval_candidates()
    print(f"    - Best candidate: {best_cand}, accept_length: {accept_len}")

    # 2d. Compute Medusa heads for next iteration
    last_hidden = tree_hs[0].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden)

    def compute_medusa():
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            return model._compute_logits(last_hidden, return_medusa=True)

    avg_medusa, std_medusa, _ = profile_with_cuda_events(compute_medusa, name="compute_medusa")
    print(f"\n2d. Compute Medusa heads (next iter): {avg_medusa:.3f}ms ± {std_medusa:.3f}ms")

    # Break down Medusa head computation
    print(f"\n    Breakdown of Medusa head computation:")

    # 2d-i. LM head only
    def lm_head_only():
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            return model.base_model.lm_head(last_hidden.clone())

    avg_lm_only, _, _ = profile_with_cuda_events(lm_head_only, name="lm_head_only")
    print(f"    - LM head projection: {avg_lm_only:.3f}ms")

    # 2d-ii. ResBlocks for all heads
    head_input = last_hidden
    def resblocks():
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = []
            for head in model.medusa_heads:
                x = head_input
                for block in head.blocks:
                    x = block(x)
                outputs.append(x)
            return torch.stack(outputs, dim=0)

    avg_resblocks, _, _ = profile_with_cuda_events(resblocks, name="resblocks")
    print(f"    - ResBlocks ({args.medusa_num_heads} heads × {args.medusa_num_layers} layers): {avg_resblocks:.3f}ms")

    # 2d-iii. LoRA projections
    stacked_resblock = resblocks()

    def lora_projections():
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            lora_a_out = torch.einsum('hbti,hri->hbtr', stacked_resblock, model._stacked_lora_a)
            lora_deltas = torch.einsum('hbtr,hvr->hbtv', lora_a_out, model._stacked_lora_b)
            return lora_deltas

    avg_lora, _, _ = profile_with_cuda_events(lora_projections, name="lora_projections")
    print(f"    - LoRA projections (rank={args.lora_rank} -> vocab): {avg_lora:.3f}ms")

    # =========================================================================
    # PART 3: Summary and Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("PART 3: SUMMARY AND ANALYSIS")
    print("="*70)

    # Calculate totals
    mtp_total = avg_cand + avg_tree + avg_eval + avg_medusa
    std_total = avg_single

    print(f"\nPer-iteration costs:")
    print(f"  Standard generation (1 token): {std_total:.3f}ms")
    print(f"  MTP generation step: {mtp_total:.3f}ms")
    print(f"    - Generate candidates: {avg_cand:.3f}ms ({100*avg_cand/mtp_total:.1f}%)")
    print(f"    - Tree verification: {avg_tree:.3f}ms ({100*avg_tree/mtp_total:.1f}%)")
    print(f"    - Evaluate candidates: {avg_eval:.3f}ms ({100*avg_eval/mtp_total:.1f}%)")
    print(f"    - Compute Medusa heads: {avg_medusa:.3f}ms ({100*avg_medusa/mtp_total:.1f}%)")

    tree_overhead = mtp_total / std_total
    print(f"\n  Tree overhead factor: {tree_overhead:.2f}x")
    print(f"  Break-even mean_accepted: {tree_overhead:.2f}")

    print(f"\nSpeedup projections:")
    for ma in [1.5, 2.0, 2.5, 3.0, 4.0]:
        speedup = ma / tree_overhead
        tok_per_sec_std = 1000 / std_total
        tok_per_sec_mtp = tok_per_sec_std * speedup
        print(f"  mean_accepted={ma:.1f}: {speedup:.2f}x speedup ({tok_per_sec_mtp:.1f} vs {tok_per_sec_std:.1f} tok/s)")

    # Detailed breakdown
    print(f"\n" + "-"*70)
    print("DETAILED BREAKDOWN")
    print("-"*70)

    print(f"\nTree verification ({avg_tree:.3f}ms):")
    print(f"  - Attention mask: {avg_mask:.3f}ms ({100*avg_mask/avg_tree:.1f}%)")
    print(f"  - Transformer fwd: {avg_tree_tfm:.3f}ms ({100*avg_tree_tfm/avg_tree:.1f}%)")
    print(f"  - LM head: {avg_tree_lm:.3f}ms ({100*avg_tree_lm/avg_tree:.1f}%)")

    print(f"\nMedusa heads ({avg_medusa:.3f}ms):")
    print(f"  - LM head: {avg_lm_only:.3f}ms ({100*avg_lm_only/avg_medusa:.1f}%)")
    print(f"  - ResBlocks: {avg_resblocks:.3f}ms ({100*avg_resblocks/avg_medusa:.1f}%)")
    print(f"  - LoRA projections: {avg_lora:.3f}ms ({100*avg_lora/avg_medusa:.1f}%)")

    # Compare tree forward to N single-token forwards
    tokens_equivalent = avg_tree / avg_single
    print(f"\n" + "-"*70)
    print("EFFICIENCY ANALYSIS")
    print("-"*70)
    print(f"\nTree forward ({tree_len} tokens) = {tokens_equivalent:.2f} single-token forwards")
    print(f"Theoretical minimum (if parallelization were free): 1.0x")
    print(f"Actual: {tokens_equivalent:.2f}x")
    print(f"Parallelization efficiency: {100/tokens_equivalent:.1f}%")

    # What if we reduced tree size?
    print(f"\n" + "-"*70)
    print("WHAT-IF ANALYSIS")
    print("-"*70)

    # Estimate costs for different tree sizes (assume linear scaling for transformer, constant for other ops)
    transformer_per_token = avg_tree_tfm / tree_len
    lm_head_per_token = avg_tree_lm / tree_len

    print(f"\nEstimated costs per tree token:")
    print(f"  - Transformer: {transformer_per_token:.4f}ms/token")
    print(f"  - LM head: {lm_head_per_token:.4f}ms/token")

    for new_tree_len in [20, 40, 60, 79]:
        new_tree_tfm = transformer_per_token * new_tree_len + avg_mask  # mask is ~constant
        new_tree_lm = lm_head_per_token * new_tree_len
        new_tree_total = new_tree_tfm + new_tree_lm
        new_mtp_total = avg_cand + new_tree_total + avg_eval + avg_medusa
        new_overhead = new_mtp_total / std_total
        print(f"\n  Tree size {new_tree_len}:")
        print(f"    Tree forward: {new_tree_total:.3f}ms (vs {avg_tree:.3f}ms)")
        print(f"    MTP total: {new_mtp_total:.3f}ms")
        print(f"    Overhead factor: {new_overhead:.2f}x")
        print(f"    For 2.0 mean_accepted: {2.0/new_overhead:.2f}x speedup")

if __name__ == '__main__':
    main()
