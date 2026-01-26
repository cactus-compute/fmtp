"""Profile forward pass cost for different tree sizes at the +33% speedup commit."""
import torch
import time
from nanochat.gemma_medusa.model import GemmaMedusaModel, get_default_tree_choices, generate_tree_buffers

model = GemmaMedusaModel(
    model_name='google/gemma-3-270m-it',
    medusa_num_heads=4,
    medusa_num_layers=2,
    lora_rank=64,
    lora_alpha=64,
    device=torch.device('cuda'),
    dtype=torch.bfloat16,
)
model.eval()

# Warmup
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    x = torch.randint(0, 1000, (1, 100), device='cuda')
    for _ in range(10):
        _ = model.forward(x, return_medusa=False, last_only=True)
torch.cuda.synchronize()

# Profile the full MTP loop components
print("="*60)
print("Profiling MTP generation loop components (old commit):")
print("="*60)

# Get tree buffers
tree_choices = get_default_tree_choices(4, topk=10)
buffers = generate_tree_buffers(tree_choices, torch.device('cuda'), topk=10)
tree_len = len(tree_choices) + 1
print(f"Tree size: {tree_len} nodes")
print(f"Tree depth distribution: {dict((d, sum(1 for c in tree_choices if len(c)==d)) for d in range(1, 5))}")

# Profile individual components
prompt_len = 100
prompt = torch.randint(0, 1000, (1, prompt_len), device='cuda')

def fresh_cache():
    """Get fresh cache and logits."""
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        hidden_states, past_key_values = model._get_hidden_states_with_cache(prompt)
        main_logits, medusa_logits = model._compute_logits(hidden_states, return_medusa=True, last_only=True)
    return past_key_values, main_logits, medusa_logits

# Profile _generate_candidates
_, main_logits, medusa_logits = fresh_cache()
times = []
for _ in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    candidates, tree_candidates = model._generate_candidates(
        main_logits[:, 0, :], medusa_logits[:, :, 0, :], buffers, topk=10, temperature=0.0
    )
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
gen_cands_time = sum(times)/len(times)*1000
print(f"_generate_candidates: {gen_cands_time:.3f}ms")

# Profile forward_mtp_with_cache
times = []
for _ in range(50):
    past_key_values, _, _ = fresh_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        tree_logits, ret_indices, valid_mask, _, tree_hidden = model.forward_mtp_with_cache(
            tree_candidates, buffers, past_key_values, prompt_len
        )
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
fwd_mtp_time = sum(times)/len(times)*1000
print(f"forward_mtp_with_cache: {fwd_mtp_time:.3f}ms")

# Profile _evaluate_candidates_greedy_fast
past_key_values, main_logits, medusa_logits = fresh_cache()
candidates, tree_candidates = model._generate_candidates(
    main_logits[:, 0, :], medusa_logits[:, :, 0, :], buffers, topk=10, temperature=0.0
)
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    tree_logits, ret_indices, valid_mask, _, tree_hidden = model.forward_mtp_with_cache(
        tree_candidates, buffers, past_key_values, prompt_len
    )
times = []
for _ in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    best_candidate, accept_length = model._evaluate_candidates_greedy_fast(
        tree_logits, candidates, ret_indices, valid_mask
    )
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
eval_cands_time = sum(times)/len(times)*1000
print(f"_evaluate_candidates_greedy_fast: {eval_cands_time:.3f}ms")

# Profile _compute_logits for Medusa heads
last_hidden = tree_hidden[0].unsqueeze(0).unsqueeze(0)
times = []
for _ in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        main_logits_out, medusa_logits_out = model._compute_logits(last_hidden, return_medusa=True)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
compute_logits_time = sum(times)/len(times)*1000
print(f"_compute_logits (medusa heads): {compute_logits_time:.3f}ms")

# KV manip - estimate from earlier optimizations (~0.5ms)
kv_manip_time = 0.5  # Approximate from prior runs

# Standard generation single step for comparison
times = []
for _ in range(50):
    # Fresh cache
    _, pv = model._get_hidden_states_with_cache(prompt)

    # Single token forward
    token = torch.randint(0, 1000, (1, 1), device='cuda')
    position_ids = torch.tensor([[prompt_len]], device='cuda')

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        hs, _ = model._get_hidden_states_with_cache(token, past_key_values=pv, position_ids=position_ids)
        logits, _ = model._compute_logits(hs, return_medusa=False, last_only=True)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
std_step_time = sum(times)/len(times)*1000
print(f"\nStandard single-token generation step: {std_step_time:.3f}ms")

# Calculate expected speedup
print("\n" + "="*60)
print("Expected speedup calculation:")
print("="*60)
mtp_step_time = gen_cands_time + fwd_mtp_time + eval_cands_time + kv_manip_time + compute_logits_time
print(f"MTP step total: {mtp_step_time:.3f}ms")
print(f"  = {gen_cands_time:.3f} (gen_cands) + {fwd_mtp_time:.3f} (fwd_mtp) + {eval_cands_time:.3f} (eval) + {kv_manip_time:.3f} (kv) + {compute_logits_time:.3f} (logits)")
print(f"Standard step: {std_step_time:.3f}ms")
print()
tree_overhead = mtp_step_time / std_step_time
print(f"Tree overhead factor: {tree_overhead:.2f}x")
print(f"  (need mean_accepted > {tree_overhead:.2f} to see any speedup)")
print()
for mean_accepted in [1.5, 2.0, 2.5, 3.0]:
    expected_speedup = mean_accepted / tree_overhead
    print(f"  mean_accepted={mean_accepted:.1f} -> expected speedup={expected_speedup:.2f}x")
