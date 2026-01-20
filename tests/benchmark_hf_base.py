"""Benchmark base HuggingFace Gemma model speed."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model_name = "google/gemma-3-1b-it"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

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
            _ = model(input_ids)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark forward pass (no generation, just single forward)
    num_iters = 10
    print(f"Benchmarking forward pass ({num_iters} iterations)...")

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            outputs = model(input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize()

    t_total = time.perf_counter() - t0
    t_per_iter = t_total / num_iters * 1000

    print(f"\n=== HF Base Model Results ===")
    print(f"Forward pass: {t_per_iter:.2f} ms")
    print(f"Output logits shape: {outputs.logits.shape}")

    # Also benchmark with longer sequence
    long_input = torch.randint(0, 1000, (1, 128), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(long_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            outputs = model(long_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    t_total = time.perf_counter() - t0
    t_per_iter_long = t_total / num_iters * 1000

    print(f"\nWith 128 token input:")
    print(f"Forward pass: {t_per_iter_long:.2f} ms")

if __name__ == "__main__":
    main()
