"""Benchmark GemmaMedusaModel speed."""
import time
import torch
import sys
sys.path.insert(0, '.')

from nanochat.gemma_medusa import GemmaMedusaModel
from transformers import AutoTokenizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model_name = "google/gemma-3-1b-it"
    print(f"Loading {model_name} with Medusa heads...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GemmaMedusaModel(
        model_name=model_name,
        medusa_num_heads=4,
        medusa_num_layers=1,
        lora_rank=64,
        device=device,
        dtype=dtype,
        freeze_base=True,
    )
    model.eval()

    print(f"Medusa params: {model.get_medusa_param_count():,}")

    # Test prompt
    prompt = "What is the capital of France? Answer in one sentence."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print(f"Input length: {input_ids.shape[1]} tokens")
    print(f"Device: {device}, dtype: {dtype}")
    print()

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids, return_medusa=False)
            _ = model(input_ids, return_medusa=True)

    if device.type == "cuda":
        torch.cuda.synchronize()

    num_iters = 10

    # Benchmark forward pass WITHOUT Medusa
    print(f"Benchmarking forward pass WITHOUT Medusa ({num_iters} iterations)...")

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            outputs = model(input_ids, return_medusa=False)
            if device.type == "cuda":
                torch.cuda.synchronize()

    t_total = time.perf_counter() - t0
    t_no_medusa = t_total / num_iters * 1000

    # Benchmark forward pass WITH Medusa
    print(f"Benchmarking forward pass WITH Medusa ({num_iters} iterations)...")

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            main_logits, medusa_logits = model(input_ids, return_medusa=True)
            if device.type == "cuda":
                torch.cuda.synchronize()

    t_total = time.perf_counter() - t0
    t_with_medusa = t_total / num_iters * 1000

    print(f"\n=== GemmaMedusaModel Results ===")
    print(f"Forward (no Medusa):   {t_no_medusa:.2f} ms")
    print(f"Forward (with Medusa): {t_with_medusa:.2f} ms")
    print(f"Medusa overhead:       {t_with_medusa - t_no_medusa:.2f} ms ({(t_with_medusa/t_no_medusa - 1)*100:.1f}%)")
    print(f"Output logits shape: {outputs.shape}")
    print(f"Medusa logits shape: {medusa_logits.shape}")

    # Also benchmark with longer sequence
    long_input = torch.randint(0, 1000, (1, 128), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(long_input, return_medusa=False)
            _ = model(long_input, return_medusa=True)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            outputs = model(long_input, return_medusa=False)
            if device.type == "cuda":
                torch.cuda.synchronize()

    t_total = time.perf_counter() - t0
    t_no_medusa_long = t_total / num_iters * 1000

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            main_logits, medusa_logits = model(long_input, return_medusa=True)
            if device.type == "cuda":
                torch.cuda.synchronize()

    t_total = time.perf_counter() - t0
    t_with_medusa_long = t_total / num_iters * 1000

    print(f"\nWith 128 token input:")
    print(f"Forward (no Medusa):   {t_no_medusa_long:.2f} ms")
    print(f"Forward (with Medusa): {t_with_medusa_long:.2f} ms")
    print(f"Medusa overhead:       {t_with_medusa_long - t_no_medusa_long:.2f} ms ({(t_with_medusa_long/t_no_medusa_long - 1)*100:.1f}%)")

    # Benchmark with last_only=True (generation mode optimization)
    print(f"\n=== Generation Mode (last_only=True) ===")
    print("This simulates the decode step where we only need logits for the last token.")

    # Warmup with last_only
    with torch.no_grad():
        for _ in range(3):
            _ = model(long_input, return_medusa=False, last_only=True)
            _ = model(long_input, return_medusa=True, last_only=True)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark without Medusa, last_only=True
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            outputs = model(long_input, return_medusa=False, last_only=True)
            if device.type == "cuda":
                torch.cuda.synchronize()

    t_total = time.perf_counter() - t0
    t_no_medusa_last = t_total / num_iters * 1000

    # Benchmark with Medusa, last_only=True
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            main_logits, medusa_logits = model(long_input, return_medusa=True, last_only=True)
            if device.type == "cuda":
                torch.cuda.synchronize()

    t_total = time.perf_counter() - t0
    t_with_medusa_last = t_total / num_iters * 1000

    print(f"Forward (no Medusa, last_only):   {t_no_medusa_last:.2f} ms")
    print(f"Forward (with Medusa, last_only): {t_with_medusa_last:.2f} ms")
    print(f"Medusa overhead (last_only):      {t_with_medusa_last - t_no_medusa_last:.2f} ms ({(t_with_medusa_last/t_no_medusa_last - 1)*100:.1f}%)")
    print(f"\nSpeedup from last_only (with Medusa): {t_with_medusa_long / t_with_medusa_last:.1f}x")
    print(f"Output shape with last_only: {main_logits.shape}")
    print(f"Medusa shape with last_only: {medusa_logits.shape}")

if __name__ == "__main__":
    main()
