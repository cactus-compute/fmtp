"""
Compute SVD-compressed vocabulary matrices for HST retrieval module.

This script extracts the embedding matrix from a Gemma model and computes
truncated SVD for efficient vocabulary projection in the retrieval module.

Usage:
    # Compute rank-64 SVD (default)
    uv run python -m scripts.hst_compute_svd

    # Compute rank-128 SVD
    uv run python -m scripts.hst_compute_svd --rank 128

    # Use specific model
    uv run python -m scripts.hst_compute_svd --model google/gemma-3-1b-it

Output:
    Saves compressed vocabulary to ~/.cache/nanochat/svd/gemma_svd_{rank}.pt
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser(
        description="Compute SVD-compressed vocabulary for HST retrieval"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-1b-it",
        help="HuggingFace model ID to extract embeddings from",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="SVD truncation rank (64 or 128 recommended)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for SVD files (default: ~/.cache/nanochat/svd/)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemma",
        help="Short name for output file (gemma_svd_{rank}.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for SVD computation (cpu recommended for large matrices)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
        help="Data type for SVD computation",
    )
    args = parser.parse_args()

    # Set up output directory
    if args.output_dir is None:
        output_dir = Path.home() / ".cache" / "nanochat" / "svd"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    print(f"SVD rank: {args.rank}")
    print(f"Output directory: {output_dir}")

    # Load model to extract embedding weights
    model = AutoModel.from_pretrained(
        args.model,
        dtype=torch.float32,  # Always load as float32 for extraction
        device_map="cpu",  # Keep on CPU for embedding extraction
    )

    # Extract embedding matrix
    # Different models have different attribute names
    embed_weight = None

    # Try common attribute paths
    if hasattr(model, "embed_tokens"):
        embed_weight = model.embed_tokens.weight
    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        embed_weight = model.transformer.wte.weight
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_weight = model.model.embed_tokens.weight
    elif hasattr(model, "embeddings"):
        embed_weight = model.embeddings.word_embeddings.weight
    else:
        # Try to find it
        for name, param in model.named_parameters():
            if "embed" in name.lower() and "weight" in name.lower():
                if param.dim() == 2:
                    embed_weight = param
                    print(f"Found embedding at: {name}")
                    break

    if embed_weight is None:
        raise RuntimeError(
            f"Could not find embedding matrix in model. "
            f"Available parameters: {[n for n, _ in model.named_parameters()][:20]}..."
        )

    vocab_size, embed_dim = embed_weight.shape
    print(f"Embedding matrix shape: [{vocab_size}, {embed_dim}]")
    print(f"Embedding matrix size: {embed_weight.numel() * 4 / 1e6:.1f} MB (float32)")

    # Convert to computation dtype and device
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    embed_weight = embed_weight.to(device=args.device, dtype=dtype)

    print(f"Computing SVD (this may take a few minutes for large vocabularies)...")

    # Compute SVD
    # For very large matrices, consider using torch.svd_lowrank for approximation
    if vocab_size > 100000 and args.rank <= 256:
        print("Using randomized SVD for large vocabulary...")
        # Randomized SVD is much faster for low-rank approximations
        U, S, Vt = torch.svd_lowrank(embed_weight, q=args.rank + 10)
        U = U[:, :args.rank]
        S = S[:args.rank]
    else:
        # Full SVD
        U, S, Vt = torch.linalg.svd(embed_weight, full_matrices=False)
        U = U[:, :args.rank]
        S = S[:args.rank]

    # Compute compressed vocabulary: U[:, :k] @ diag(S[:k])
    compressed_vocab = U * S.unsqueeze(0)  # Broadcasting: [V, k] * [1, k]

    print(f"Compressed vocabulary shape: {compressed_vocab.shape}")
    print(f"Compressed size: {compressed_vocab.numel() * 4 / 1e6:.1f} MB (float32)")
    print(f"Compression ratio: {embed_weight.numel() / compressed_vocab.numel():.1f}x")

    # Compute reconstruction error
    reconstructed = compressed_vocab @ Vt[:args.rank, :]
    error = torch.norm(embed_weight - reconstructed) / torch.norm(embed_weight)
    print(f"Relative reconstruction error: {error.item():.4f}")

    # Variance explained
    total_variance = S.sum() ** 2  # Approximation using singular values
    captured_variance = S[:args.rank].sum() ** 2
    if total_variance > 0:
        variance_explained = captured_variance / total_variance
        print(f"Approximate variance explained: {variance_explained.item():.2%}")

    # Save compressed vocabulary
    output_path = output_dir / f"{args.model_name}_svd_{args.rank}.pt"
    compressed_vocab = compressed_vocab.to(torch.float32).cpu()  # Save as float32
    torch.save(compressed_vocab, output_path)
    print(f"Saved compressed vocabulary to: {output_path}")

    # Also save Vt for optional reconstruction/debugging
    vt_path = output_dir / f"{args.model_name}_vt_{args.rank}.pt"
    Vt_truncated = Vt[:args.rank, :].to(torch.float32).cpu()
    torch.save(Vt_truncated, vt_path)
    print(f"Saved Vt matrix to: {vt_path}")

    print("\nDone! You can now use the compressed vocabulary with RetrievalMLP:")
    print(f"  from nanochat.hst import load_svd_basis")
    print(f"  compressed_vocab = load_svd_basis(rank={args.rank})")
    print(f"  retrieval_module.load_svd(compressed_vocab)")


if __name__ == "__main__":
    main()
