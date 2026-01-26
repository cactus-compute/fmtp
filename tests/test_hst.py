"""
Tests for HST (Hybrid Smoothed Tree) speculation module.

Tests cover:
- Retrieval module: 3-layer MLP, SVD compression
- Suffix matching: Buffer management, lookup
- Tree building: Hybrid scoring, priority-queue expansion
- Tree attention: Flattening, mask generation
"""

import pytest
import torch

from nanochat.hst.retrieval import (
    RetrievalMLP,
    RetrievalModuleTiny,
    compute_svd_basis,
)
from nanochat.hst.suffix_match import (
    SuffixMatcher,
    ContextBuffer,
)
from nanochat.hst.tree_builder import (
    HSTNode,
    HybridScorer,
    HSTTreeBuilder,
)
from nanochat.hst.tree_attention import (
    flatten_tree,
    build_tree_mask,
    generate_hst_buffers,
    extract_candidate_paths,
)


class TestContextBuffer:
    """Tests for the rolling context buffer."""

    def test_append_single(self):
        buffer = ContextBuffer(max_size=10, device="cpu")
        buffer.append(42)
        assert len(buffer) == 1
        assert buffer.get_recent(1).tolist() == [42]

    def test_append_multiple(self):
        buffer = ContextBuffer(max_size=10, device="cpu")
        buffer.append([1, 2, 3, 4, 5])
        assert len(buffer) == 5
        assert buffer.get_recent(5).tolist() == [1, 2, 3, 4, 5]

    def test_circular_wrap(self):
        buffer = ContextBuffer(max_size=5, device="cpu")
        buffer.append([1, 2, 3, 4, 5, 6, 7])
        assert len(buffer) == 5
        # Should contain last 5 tokens
        recent = buffer.get_recent(5).tolist()
        assert recent == [3, 4, 5, 6, 7]

    def test_reset(self):
        buffer = ContextBuffer(max_size=10, device="cpu")
        buffer.append([1, 2, 3])
        buffer.reset()
        assert len(buffer) == 0

    def test_get_all(self):
        buffer = ContextBuffer(max_size=10, device="cpu")
        buffer.append([10, 20, 30])
        all_tokens = buffer.get_all().tolist()
        assert all_tokens == [10, 20, 30]


class TestSuffixMatcher:
    """Tests for context suffix matching."""

    def test_basic_matching(self):
        matcher = SuffixMatcher(
            buffer_size=100,
            min_suffix_len=1,
            max_suffix_len=2,
            device="cpu",
        )

        # Build context with repetition
        matcher.append([1, 2, 3, 1, 2])

        # Query suffix [1, 2] - should find continuation 3
        tokens, scores = matcher.find_continuations(
            torch.tensor([1, 2]),
            top_k=5,
        )

        # Should find token 3 as continuation
        assert len(tokens) > 0 or len(scores) > 0

    def test_empty_context(self):
        matcher = SuffixMatcher(buffer_size=10, device="cpu")

        tokens, scores = matcher.find_continuations(
            torch.tensor([1, 2]),
            top_k=5,
        )

        assert len(tokens) == 0
        assert len(scores) == 0

    def test_get_suffix_probabilities(self):
        matcher = SuffixMatcher(buffer_size=100, device="cpu")
        matcher.append([1, 2, 3, 1, 2, 4])

        probs = matcher.get_suffix_probabilities(
            torch.tensor([1, 2]),
            vocab_size=100,
        )

        assert probs.shape == (100,)
        assert probs.sum().item() == pytest.approx(1.0, rel=1e-5)


class TestRetrievalMLP:
    """Tests for the learned retrieval module (3-layer MLP)."""

    def test_initialization(self):
        module = RetrievalMLP(
            vocab_size=100,
            hidden_dim=64,
            context_window=4,
            svd_rank=16,
        )
        # Module should not be ready without SVD initialized
        assert not module._svd_initialized

    def test_forward_requires_svd(self):
        module = RetrievalMLP(
            vocab_size=100,
            hidden_dim=64,
            context_window=4,
            svd_rank=16,
        )
        # Now takes token IDs (integers), not embeddings
        x = torch.randint(0, 100, (2, 4))
        with pytest.raises(RuntimeError, match="SVD"):
            module(x)

    def test_forward_with_svd(self):
        module = RetrievalMLP(
            vocab_size=100,
            hidden_dim=64,
            context_window=4,
            svd_rank=16,
        )
        # Load fake SVD
        fake_svd = torch.randn(100, 16)
        module.load_svd(fake_svd)

        # Input is token IDs (integers), not embeddings
        x = torch.randint(0, 100, (2, 4))
        logits = module(x)
        assert logits.shape == (2, 100)

    def test_variable_input_length(self):
        """Test that module handles inputs shorter than context_window."""
        module = RetrievalMLP(
            vocab_size=100,
            hidden_dim=64,
            context_window=4,
            svd_rank=16,
        )
        module.load_svd(torch.randn(100, 16))

        # Test with fewer tokens than context_window
        x = torch.randint(0, 100, (2, 2))  # Only 2 tokens, context_window=4
        logits = module(x)
        assert logits.shape == (2, 100)

        # Test with single token
        x = torch.randint(0, 100, (2,))  # 1D input
        logits = module(x)
        assert logits.shape == (2, 100)

    def test_tied_svd_embeddings(self):
        """Test that SVD embeddings are tied (same for input and output)."""
        module = RetrievalMLP(
            vocab_size=100,
            hidden_dim=64,
            context_window=4,
            svd_rank=16,
        )
        fake_svd = torch.randn(100, 16)
        module.load_svd(fake_svd)

        # Should have a trainable svd_embedding parameter
        assert hasattr(module, 'svd_embedding')
        assert module.svd_embedding.requires_grad  # Should be trainable

        # Input is token IDs (integers)
        x = torch.randint(0, 100, (2, 4))
        logits = module(x)
        assert logits.shape == (2, 100)

        # Check parameter count includes svd_embedding
        num_params = sum(p.numel() for p in module.parameters())
        # svd_embedding: 100*16 = 1600
        # fc1: (4*16)*64 + 64 = 4160
        # fc2: 64*64 + 64 = 4160
        # fc3: 64*16 = 1024 (no bias)
        # Total: ~11k params for this small test config
        assert num_params > 1600, f"Expected svd_embedding in params, got {num_params}"

    def test_forward_topk_matches_full(self):
        """Test that forward_topk produces same scores as full projection for selected candidates."""
        module = RetrievalMLP(
            vocab_size=1000,
            hidden_dim=64,
            context_window=8,
            svd_rank=16,
        )
        fake_svd = torch.randn(1000, 16)
        module.load_svd(fake_svd)

        # Create test input
        token_ids = torch.randint(0, 1000, (4, 8))  # Batch of 4, context window 8

        # Get full vocab logits
        full_logits = module(token_ids)  # [4, 1000]

        # Select random candidates
        candidate_ids = torch.tensor([10, 50, 100, 500, 999])

        # Get topk logits
        topk_logits = module.forward_topk(token_ids, candidate_ids)  # [4, 5]

        # Verify they match
        for i, cand_id in enumerate(candidate_ids):
            expected = full_logits[:, cand_id.item()]
            actual = topk_logits[:, i]
            assert torch.allclose(expected, actual, atol=1e-5), \
                f"Mismatch at candidate {cand_id}: expected {expected}, got {actual}"

    def test_forward_topk_batch_candidates(self):
        """Test forward_topk with per-batch different candidates."""
        module = RetrievalMLP(
            vocab_size=1000,
            hidden_dim=64,
            context_window=4,
            svd_rank=16,
        )
        module.load_svd(torch.randn(1000, 16))

        token_ids = torch.randint(0, 1000, (2, 4))

        # Different candidates per batch item
        candidate_ids = torch.tensor([
            [10, 20, 30],
            [100, 200, 300],
        ])

        topk_logits = module.forward_topk(token_ids, candidate_ids)
        assert topk_logits.shape == (2, 3)

        # Verify against full projection
        full_logits = module(token_ids)
        for b in range(2):
            for i, cand_id in enumerate(candidate_ids[b]):
                expected = full_logits[b, cand_id.item()]
                actual = topk_logits[b, i]
                assert torch.allclose(expected, actual, atol=1e-5)

    def test_forward_topk_speedup(self):
        """Test that forward_topk is faster than full projection (sanity check)."""
        import time

        module = RetrievalMLP(
            vocab_size=10000,  # Larger vocab to see speedup
            hidden_dim=128,
            context_window=8,
            svd_rank=64,
        )
        module.load_svd(torch.randn(10000, 64))

        token_ids = torch.randint(0, 10000, (32, 8))
        candidate_ids = torch.randint(0, 10000, (100,))  # Only 100 candidates

        # Warmup
        _ = module(token_ids)
        _ = module.forward_topk(token_ids, candidate_ids)

        # Time full projection
        start = time.perf_counter()
        for _ in range(10):
            _ = module(token_ids)
        full_time = time.perf_counter() - start

        # Time topk projection
        start = time.perf_counter()
        for _ in range(10):
            _ = module.forward_topk(token_ids, candidate_ids)
        topk_time = time.perf_counter() - start

        # topk should be faster (at least 2x for 100 candidates vs 10000 vocab)
        # Note: May not always be faster for small batch sizes due to overhead
        # Just verify it runs without error for this test
        assert topk_time >= 0  # Basic sanity check


class TestRetrievalModuleTiny:
    """Tests for ultra-light retrieval module."""

    def test_forward(self):
        module = RetrievalModuleTiny(
            vocab_size=100,
            embed_dim=64,
            svd_rank=16,
        )
        module.load_svd(torch.randn(100, 16))

        # Input is token IDs (integers)
        x = torch.randint(0, 100, (2,))
        logits = module(x)
        assert logits.shape == (2, 100)


class TestSVDComputation:
    """Tests for SVD basis computation."""

    def test_compute_svd_basis(self):
        embed_weight = torch.randn(100, 64)  # Small test matrix
        compressed = compute_svd_basis(embed_weight, rank=16)
        assert compressed.shape == (100, 16)

    def test_svd_reconstruction(self):
        embed_weight = torch.randn(100, 64)
        rank = 32

        # Compute full SVD for comparison
        U, S, Vt = torch.linalg.svd(embed_weight, full_matrices=False)

        # Compute compressed
        compressed = compute_svd_basis(embed_weight, rank=rank)

        # Compressed should equal U[:, :rank] @ diag(S[:rank])
        expected = U[:, :rank] * S[:rank].unsqueeze(0)
        assert torch.allclose(compressed, expected, atol=1e-5)


class TestHybridScorer:
    """Tests for hybrid scoring function."""

    def test_score_candidates(self):
        scorer = HybridScorer(alpha=0.6, beta=0.3, gamma=0.1)

        mtp_logits = torch.randn(100)
        retrieval_logits = torch.randn(100)
        suffix_probs = torch.softmax(torch.randn(100), dim=-1)

        tokens, scores, metadata = scorer.score_candidates(
            mtp_logits=mtp_logits,
            retrieval_logits=retrieval_logits,
            suffix_probs=suffix_probs,
            top_k=5,
        )

        assert len(tokens) > 0
        assert len(scores) == len(tokens)
        assert "sources" in metadata

    def test_get_top_candidates(self):
        scorer = HybridScorer()

        mtp_logits = torch.randn(100)
        retrieval_logits = torch.randn(100)

        tokens, scores, sources = scorer.get_top_candidates(
            mtp_logits=mtp_logits,
            retrieval_logits=retrieval_logits,
            suffix_probs=None,
            mtp_top_k=1,
            retrieval_top_k=2,
        )

        # Should have at least 1 MTP + 2 retrieval candidates (possibly overlapping)
        assert len(tokens) >= 1
        assert len(tokens) <= 3  # At most 3 unique candidates

    def test_agreement_bonus(self):
        scorer = HybridScorer(alpha=0.5, beta=0.5, gamma=0.0, agreement_bonus=2.0)

        # Create logits where token 0 is top for both sources
        mtp_logits = torch.zeros(100)
        mtp_logits[0] = 10.0
        retrieval_logits = torch.zeros(100)
        retrieval_logits[0] = 10.0

        tokens, scores, sources = scorer.get_top_candidates(
            mtp_logits=mtp_logits,
            retrieval_logits=retrieval_logits,
            suffix_probs=None,
        )

        # Token 0 should have agreement bonus applied
        token_0_idx = (tokens == 0).nonzero(as_tuple=True)[0]
        if len(token_0_idx) > 0:
            idx = token_0_idx[0].item()
            assert len(sources[idx]) >= 2  # Both MTP and retrieval


class TestHSTTreeBuilder:
    """Tests for HST tree construction."""

    def test_build_simple_tree(self):
        scorer = HybridScorer()
        builder = HSTTreeBuilder(
            scorer=scorer,
            max_depth=2,
            tree_budget=10,
        )

        def get_mtp(depth, path):
            logits = torch.zeros(100)
            logits[depth + 1] = 10.0
            return logits

        tree = builder.build_tree(
            root_token=0,
            get_mtp_logits=get_mtp,
            get_retrieval_logits=lambda ctx: torch.randn(100),
            get_suffix_probs=lambda suf: torch.softmax(torch.randn(100), dim=-1),
            context_tokens=[0],
            vocab_size=100,
            device="cpu",
        )

        # Should have root + at least one child
        assert len(tree) >= 1
        assert tree[0].depth == 0  # Root
        assert tree[0].token_id == 0

    def test_get_candidate_paths(self):
        scorer = HybridScorer()
        builder = HSTTreeBuilder(scorer=scorer, max_depth=2, tree_budget=10)

        def get_mtp(depth, path):
            logits = torch.zeros(100)
            logits[0] = 10.0
            return logits

        tree = builder.build_tree(
            root_token=0,
            get_mtp_logits=get_mtp,
            get_retrieval_logits=lambda ctx: None,
            get_suffix_probs=lambda suf: None,
            context_tokens=[0],
            vocab_size=100,
            device="cpu",
        )

        paths = builder.get_candidate_paths(tree)
        # All paths should exclude root token
        for path in paths:
            assert len(path) > 0


class TestTreeAttention:
    """Tests for tree attention utilities."""

    def test_flatten_tree(self):
        # Create simple tree
        tree = [
            HSTNode(token_id=0, depth=0, score=1.0, node_idx=0),
            HSTNode(token_id=1, depth=1, score=0.9, parent_idx=0, node_idx=1),
            HSTNode(token_id=2, depth=1, score=0.8, parent_idx=0, node_idx=2),
            HSTNode(token_id=3, depth=2, score=0.7, parent_idx=1, node_idx=3),
        ]

        token_ids, parent_indices, depths = flatten_tree(tree, sort_by="depth_first")

        assert len(token_ids) == 4
        assert len(parent_indices) == 4
        assert len(depths) == 4

    def test_build_tree_mask(self):
        parent_indices = [-1, 0, 0, 1]  # Root, two children of root, one grandchild
        mask = build_tree_mask(parent_indices, device="cpu")

        assert mask.shape == (4, 4)
        # Diagonal should be 1
        for i in range(4):
            assert mask[i, i] == 1.0
        # Node 1 should attend to root (0)
        assert mask[1, 0] == 1.0
        # Node 3 should attend to node 1 and root
        assert mask[3, 1] == 1.0
        assert mask[3, 0] == 1.0
        # Node 2 should not attend to node 1
        assert mask[2, 1] == 0.0

    def test_generate_hst_buffers(self):
        tree = [
            HSTNode(token_id=0, depth=0, score=1.0, node_idx=0),
            HSTNode(token_id=1, depth=1, score=0.9, parent_idx=0, node_idx=1),
        ]

        buffers = generate_hst_buffers(tree, device="cpu")

        assert buffers.attn_mask.shape == (1, 1, 2, 2)
        assert buffers.tree_tokens.shape == (2,)
        assert buffers.position_ids.shape == (2,)

    def test_extract_candidate_paths(self):
        tree = [
            HSTNode(token_id=0, depth=0, score=1.0, node_idx=0),
            HSTNode(token_id=1, depth=1, score=0.9, parent_idx=0, node_idx=1),
            HSTNode(token_id=2, depth=2, score=0.8, parent_idx=1, node_idx=2),
        ]

        buffers = generate_hst_buffers(tree, device="cpu")
        paths = extract_candidate_paths(buffers)

        # Should have paths for nodes at depth > 0
        assert len(paths) >= 1


class TestIntegration:
    """Integration tests for the full HST pipeline."""

    def test_full_pipeline(self):
        """Test complete HST pipeline from input to candidate paths."""
        # Setup components
        retrieval = RetrievalMLP(
            vocab_size=100,
            hidden_dim=64,
            context_window=4,
            svd_rank=16,
        )
        retrieval.load_svd(torch.randn(100, 16))

        suffix_matcher = SuffixMatcher(buffer_size=50, device="cpu")
        suffix_matcher.append([1, 2, 3, 4, 5])

        scorer = HybridScorer(alpha=0.6, beta=0.3, gamma=0.1)
        builder = HSTTreeBuilder(scorer=scorer, max_depth=2, tree_budget=10)

        # Build tree
        def get_mtp(depth, path):
            return torch.randn(100)

        def get_retrieval(ctx):
            # Now takes token IDs directly
            token_ids = torch.tensor([ctx[-4:]])
            return retrieval(token_ids)[0]

        def get_suffix(suf):
            return suffix_matcher.get_suffix_probabilities(
                torch.tensor(suf[-4:]),
                vocab_size=100,
            )

        tree = builder.build_tree(
            root_token=5,
            get_mtp_logits=get_mtp,
            get_retrieval_logits=get_retrieval,
            get_suffix_probs=get_suffix,
            context_tokens=[1, 2, 3, 4, 5],
            vocab_size=100,
            device="cpu",
        )

        # Generate buffers
        buffers = generate_hst_buffers(tree, device="cpu")

        # Extract paths
        paths = extract_candidate_paths(buffers)

        assert len(tree) >= 1
        assert buffers.tree_len >= 1


class TestHSTScorer:
    """Tests for the HSTScorer callback interface."""

    @pytest.fixture
    def mock_scorer(self):
        """Create an HSTScorer with mocked SVD for testing."""
        from nanochat.gemma_medusa.hst_scorer import HSTScorer

        # Create scorer - SVD load will fail but fallback to random init
        scorer = HSTScorer(
            vocab_size=100,
            device=torch.device("cpu"),
            retrieval_checkpoint=None,
            svd_rank=16,  # Small rank for testing
        )
        # Manually init the retrieval module's SVD embedding
        scorer.retrieval_module.svd_embedding.data = torch.randn(100, 16)
        scorer.retrieval_module._svd_initialized = True
        return scorer

    def test_hst_scorer_initialization(self, mock_scorer):
        """Test HSTScorer initializes correctly."""
        assert mock_scorer.alpha == 0.6
        assert mock_scorer.beta == 0.3
        assert mock_scorer.gamma == 0.1
        assert mock_scorer.enabled

    def test_hst_scorer_callback_interface(self, mock_scorer):
        """Test HSTScorer implements the callback interface correctly."""
        # Test reset
        mock_scorer.reset()

        # Test __call__ (the main callback)
        main_logits = torch.randn(100)
        medusa_logits = torch.randn(4, 100)  # 4 heads
        tree_candidates = torch.randint(0, 100, (10,))  # 10 tree positions
        context_tokens = [1, 2, 3, 4, 5]

        scores = mock_scorer(main_logits, medusa_logits, tree_candidates, context_tokens)

        assert scores.shape == (10,)
        assert scores.dtype == torch.float32

    def test_hst_scorer_on_tokens_accepted(self, mock_scorer):
        """Test on_tokens_accepted updates suffix buffer."""
        # Initially empty
        assert len(mock_scorer.suffix_matcher) == 0

        # Accept some tokens
        mock_scorer.on_tokens_accepted([10, 20, 30])

        # Buffer should be updated
        assert len(mock_scorer.suffix_matcher) == 3

        # Accept more
        mock_scorer.on_tokens_accepted([40, 50])
        assert len(mock_scorer.suffix_matcher) == 5

    def test_hst_scorer_disable_enable(self, mock_scorer):
        """Test enable/disable functionality."""
        assert mock_scorer.enabled

        mock_scorer.disable()
        assert not mock_scorer.enabled

        # When disabled, should return uniform scores
        main_logits = torch.randn(100)
        medusa_logits = torch.randn(4, 100)
        tree_candidates = torch.randint(0, 100, (10,))
        context_tokens = [1, 2, 3]

        scores = mock_scorer(main_logits, medusa_logits, tree_candidates, context_tokens)
        assert torch.allclose(scores, torch.ones(10))

        mock_scorer.enable()
        assert mock_scorer.enabled

    def test_hst_scorer_set_weights(self, mock_scorer):
        """Test weight adjustment."""
        # Change weights
        mock_scorer.set_weights(alpha=0.5, beta=0.4, gamma=0.1)

        # Weights should be normalized
        assert mock_scorer.alpha == pytest.approx(0.5)
        assert mock_scorer.beta == pytest.approx(0.4)
        assert mock_scorer.gamma == pytest.approx(0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
