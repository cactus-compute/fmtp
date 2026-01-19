"""
Tests for Medusa Multi-Token Prediction (MTP) implementation.

Run: uv run pytest tests/test_medusa.py -v
"""

import pytest
import torch
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig, MedusaResBlock, MedusaHead, MedusaLoRAHead, IndependentMedusaHead


class TestMedusaResBlock:
    """Tests for MedusaResBlock class."""

    def test_resblock_output_shape(self):
        """ResBlock should preserve input shape."""
        hidden_size = 64
        block = MedusaResBlock(hidden_size)
        x = torch.randn(2, 10, hidden_size)
        out = block(x)
        assert out.shape == x.shape

    def test_resblock_identity_init(self):
        """ResBlock with zero-init should act approximately as identity."""
        hidden_size = 64
        block = MedusaResBlock(hidden_size)
        # Zero initialize the linear layer
        torch.nn.init.zeros_(block.linear.weight)

        x = torch.randn(2, 10, hidden_size)
        out = block(x)
        # With zero weights, silu(0) = 0, so output should equal input
        assert torch.allclose(out, x, atol=1e-6)

    def test_resblock_gradient_flow(self):
        """Gradients should flow through ResBlock."""
        hidden_size = 64
        block = MedusaResBlock(hidden_size)
        x = torch.randn(2, 10, hidden_size, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert block.linear.weight.grad is not None


class TestMedusaHead:
    """Tests for MedusaHead class."""

    def test_head_output_shape(self):
        """MedusaHead should output correct shape."""
        hidden_size = 64
        vocab_size = 1000
        num_layers = 2
        head = MedusaHead(hidden_size, vocab_size, num_layers)

        x = torch.randn(2, 10, hidden_size)
        out = head(x)
        assert out.shape == (2, 10, vocab_size)

    def test_head_single_layer(self):
        """MedusaHead with single layer should work."""
        hidden_size = 64
        vocab_size = 100
        head = MedusaHead(hidden_size, vocab_size, num_layers=1)

        x = torch.randn(2, 10, hidden_size)
        out = head(x)
        assert out.shape == (2, 10, vocab_size)

    def test_head_multiple_layers(self):
        """MedusaHead with multiple layers should work."""
        hidden_size = 64
        vocab_size = 100
        head = MedusaHead(hidden_size, vocab_size, num_layers=4)

        assert len(head.blocks) == 4
        x = torch.randn(2, 10, hidden_size)
        out = head(x)
        assert out.shape == (2, 10, vocab_size)


class TestGPTWithMedusa:
    """Tests for GPT model with Medusa heads."""

    @pytest.fixture
    def small_config_with_medusa(self):
        """Create a small GPT config with Medusa heads for testing."""
        return GPTConfig(
            sequence_len=64,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=64,
            medusa_num_heads=3,
            medusa_num_layers=1,
        )

    @pytest.fixture
    def small_config_no_medusa(self):
        """Create a small GPT config without Medusa heads."""
        return GPTConfig(
            sequence_len=64,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=64,
            medusa_num_heads=0,  # Disabled
        )

    def test_gpt_creates_medusa_heads(self, small_config_with_medusa):
        """GPT should create Medusa heads when configured."""
        with torch.device("cpu"):
            model = GPT(small_config_with_medusa)
        model.init_weights()

        assert model.medusa_heads is not None
        assert len(model.medusa_heads) == 3

    def test_gpt_no_medusa_heads_when_disabled(self, small_config_no_medusa):
        """GPT should not create Medusa heads when disabled."""
        with torch.device("cpu"):
            model = GPT(small_config_no_medusa)
        model.init_weights()

        assert model.medusa_heads is None

    def test_forward_without_medusa_flag(self, small_config_with_medusa):
        """Forward without return_medusa should return standard output."""
        with torch.device("cpu"):
            model = GPT(small_config_with_medusa)
        model.init_weights()
        model.eval()

        idx = torch.randint(0, 256, (2, 32))
        # Without return_medusa, should return only logits
        logits = model(idx)
        assert logits.shape == (2, 32, 256)

    def test_forward_with_medusa_flag_inference(self, small_config_with_medusa):
        """Forward with return_medusa should return logits and medusa_logits."""
        with torch.device("cpu"):
            model = GPT(small_config_with_medusa)
        model.init_weights()
        model.eval()

        idx = torch.randint(0, 256, (2, 32))
        logits, medusa_logits = model(idx, return_medusa=True)

        assert logits.shape == (2, 32, 256)
        assert medusa_logits.shape == (3, 2, 32, 256)  # (num_heads, B, T, vocab)

    def test_forward_with_targets(self, small_config_with_medusa):
        """Forward with targets should return loss."""
        with torch.device("cpu"):
            model = GPT(small_config_with_medusa)
        model.init_weights()

        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))

        # Without return_medusa
        loss = model(idx, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_forward_with_targets_and_medusa(self, small_config_with_medusa):
        """Forward with targets and return_medusa should return loss and medusa_losses."""
        with torch.device("cpu"):
            model = GPT(small_config_with_medusa)
        model.init_weights()

        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))

        # Training mode returns (loss, medusa_losses), not logits
        loss, medusa_losses = model(idx, targets, return_medusa=True)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert len(medusa_losses) == 3  # 3 Medusa heads
        for ml in medusa_losses:
            assert isinstance(ml, torch.Tensor)
            assert ml.shape == ()

    def test_backward_compatible_no_medusa(self, small_config_no_medusa):
        """GPT without Medusa should work exactly as before."""
        with torch.device("cpu"):
            model = GPT(small_config_no_medusa)
        model.init_weights()

        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))

        # Should return single loss value (not tuple)
        loss = model(idx, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()

        # Inference should return single tensor
        model.eval()
        logits = model(idx)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (2, 32, 256)


class TestTargetShifting:
    """Tests for Medusa target shifting logic."""

    def test_target_shifting_alignment(self):
        """Verify that target shifting aligns correctly for each head."""
        B, T = 2, 10
        vocab_size = 100
        num_heads = 3

        # Create sequential targets for easy verification
        targets = torch.arange(T).unsqueeze(0).expand(B, -1)  # [0,1,2,...,T-1]

        for k in range(num_heads):
            shift = 2 + k  # Head k predicts i+2+k

            # Truncate positions: predict from 0 to T-shift-1
            valid_positions = T - shift

            # Shifted targets should be [shift, shift+1, ..., T-1]
            shifted_targets = targets[:, shift:]

            assert shifted_targets.shape[1] == valid_positions
            assert shifted_targets[0, 0].item() == shift
            assert shifted_targets[0, -1].item() == T - 1

    def test_mtp_loss_computation(self):
        """Test the MTP loss computation logic."""
        B, T = 2, 20
        vocab_size = 100
        num_heads = 3

        # Create random logits and targets
        main_logits = torch.randn(B, T, vocab_size)
        medusa_logits = torch.randn(num_heads, B, T, vocab_size)
        targets = torch.randint(0, vocab_size, (B, T))

        # Compute main loss
        main_loss = F.cross_entropy(
            main_logits.view(-1, vocab_size),
            targets.view(-1)
        )

        # Compute Medusa losses with shifting
        total_loss = main_loss
        for k in range(num_heads):
            shift = 2 + k
            head_logits = medusa_logits[k, :, :-shift]
            shifted_targets = targets[:, shift:]

            # Shapes should match
            assert head_logits.shape[:2] == shifted_targets.shape

            head_loss = F.cross_entropy(
                head_logits.reshape(-1, vocab_size),
                shifted_targets.reshape(-1)
            )
            total_loss = total_loss + head_loss

        # Total loss should be larger than main loss alone
        assert total_loss.item() > main_loss.item()


class TestMedusaOptimizer:
    """Tests for optimizer setup with Medusa heads."""

    def test_setup_optimizers_includes_medusa(self):
        """setup_optimizers should split Medusa params: ResBlocks->Muon, proj->AdamW."""
        config = GPTConfig(
            sequence_len=64,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=64,
            medusa_num_heads=2,
            medusa_num_layers=2,  # 2 ResBlocks per head
        )
        with torch.device("cpu"):
            model = GPT(config)
        model.init_weights()

        # This should not raise an assertion error about parameter count
        optimizers = model.setup_optimizers()
        assert len(optimizers) == 2  # AdamW and Muon

        adamw, muon = optimizers

        # Collect all params from each optimizer
        all_adamw_params = []
        for group in adamw.param_groups:
            all_adamw_params.extend(group['params'])

        all_muon_params = []
        for group in muon.param_groups:
            all_muon_params.extend(group['params'])

        # Check that Medusa proj params are in AdamW (like lm_head)
        for head in model.medusa_heads:
            proj_weight = head.proj.weight
            assert any(p is proj_weight for p in all_adamw_params), "Medusa proj should be in AdamW"
            assert not any(p is proj_weight for p in all_muon_params), "Medusa proj should NOT be in Muon"

        # Check that Medusa ResBlock params are in Muon (like transformer linears)
        for head in model.medusa_heads:
            for block in head.blocks:
                linear_weight = block.linear.weight
                assert any(p is linear_weight for p in all_muon_params), "Medusa ResBlock should be in Muon"
                assert not any(p is linear_weight for p in all_adamw_params), "Medusa ResBlock should NOT be in AdamW"


class TestConfigDefaults:
    """Tests for GPTConfig Medusa defaults."""

    def test_default_medusa_disabled(self):
        """By default, Medusa should be disabled."""
        config = GPTConfig()
        assert config.medusa_num_heads == 0
        assert config.medusa_num_layers == 1

    def test_medusa_config_values(self):
        """Custom Medusa config values should be respected."""
        config = GPTConfig(medusa_num_heads=5, medusa_num_layers=3)
        assert config.medusa_num_heads == 5
        assert config.medusa_num_layers == 3

    def test_medusa_lora_rank_default(self):
        """medusa_lora_rank should default to 0 (full projection)."""
        config = GPTConfig()
        assert config.medusa_lora_rank == 0


class TestMedusaLoRAHead:
    """Tests for MedusaLoRAHead class (LoRA adapter over lm_head)."""

    def test_lora_head_output_shape(self):
        """MedusaLoRAHead should output correct shape."""
        hidden_size = 64
        vocab_size = 1000
        num_layers = 1
        lora_rank = 8
        head = MedusaLoRAHead(hidden_size, vocab_size, num_layers, lora_rank)

        x = torch.randn(2, 10, hidden_size)
        lm_head_weight = torch.randn(vocab_size, hidden_size)

        # Apply ResBlocks
        h = x
        for block in head.blocks:
            h = block(h)

        # Compute logits: base + lora_delta
        base_logits = F.linear(h, lm_head_weight)
        lora_delta = head.lora_B(head.lora_A(h))
        out = base_logits + lora_delta

        assert out.shape == (2, 10, vocab_size)

    def test_lora_head_has_correct_params(self):
        """MedusaLoRAHead should have lora_A and lora_B params."""
        hidden_size = 64
        vocab_size = 1000
        lora_rank = 16
        head = MedusaLoRAHead(hidden_size, vocab_size, num_layers=1, lora_rank=lora_rank)

        assert hasattr(head, 'lora_A')
        assert hasattr(head, 'lora_B')
        assert head.lora_A.weight.shape == (lora_rank, hidden_size)
        assert head.lora_B.weight.shape == (vocab_size, lora_rank)

    def test_lora_head_param_count_reduction(self):
        """LoRA head should have far fewer params than full projection."""
        hidden_size = 1280
        vocab_size = 65536
        lora_rank = 32

        # Full projection params
        full_head = MedusaHead(hidden_size, vocab_size, num_layers=1)
        full_proj_params = full_head.proj.weight.numel()  # vocab_size * hidden_size

        # LoRA params
        lora_head = MedusaLoRAHead(hidden_size, vocab_size, num_layers=1, lora_rank=lora_rank)
        lora_params = lora_head.lora_A.weight.numel() + lora_head.lora_B.weight.numel()

        # LoRA should be ~97% smaller
        assert lora_params < full_proj_params * 0.05  # Less than 5% of full

    def test_lora_merged_weight(self):
        """get_merged_weight should return W_base + B @ A."""
        hidden_size = 64
        vocab_size = 100
        lora_rank = 8
        head = MedusaLoRAHead(hidden_size, vocab_size, num_layers=1, lora_rank=lora_rank)

        lm_head_weight = torch.randn(vocab_size, hidden_size)
        merged = head.get_merged_weight(lm_head_weight)

        # Verify merged = lm_head_weight + lora_B.weight @ lora_A.weight
        expected = lm_head_weight + head.lora_B.weight @ head.lora_A.weight
        assert torch.allclose(merged, expected)

    def test_lora_identity_at_zero_init(self):
        """With B=zeros, LoRA should act as identity (output = lm_head(x))."""
        hidden_size = 64
        vocab_size = 100
        lora_rank = 8
        head = MedusaLoRAHead(hidden_size, vocab_size, num_layers=1, lora_rank=lora_rank)

        # Zero-init B (standard LoRA init)
        torch.nn.init.zeros_(head.lora_B.weight)
        # Also zero-init ResBlocks for identity
        for block in head.blocks:
            torch.nn.init.zeros_(block.linear.weight)

        x = torch.randn(2, 10, hidden_size)
        lm_head_weight = torch.randn(vocab_size, hidden_size)

        # With zero B, merged_weight should equal lm_head_weight
        merged = head.get_merged_weight(lm_head_weight)
        assert torch.allclose(merged, lm_head_weight)


class TestGPTWithLoRAMedusa:
    """Tests for GPT model with LoRA-based Medusa heads."""

    @pytest.fixture
    def small_config_lora_medusa(self):
        """Create a small GPT config with LoRA Medusa heads for testing."""
        return GPTConfig(
            sequence_len=64,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=64,
            medusa_num_heads=2,
            medusa_num_layers=1,
            medusa_lora_rank=8,  # LoRA enabled
        )

    def test_gpt_creates_lora_heads(self, small_config_lora_medusa):
        """GPT should create MedusaLoRAHead when lora_rank > 0."""
        with torch.device("cpu"):
            model = GPT(small_config_lora_medusa)
        model.init_weights()

        assert model.medusa_heads is not None
        for head in model.medusa_heads:
            assert isinstance(head, MedusaLoRAHead)
            assert hasattr(head, 'lora_A')
            assert hasattr(head, 'lora_B')

    def test_forward_inference_lora(self, small_config_lora_medusa):
        """Forward with LoRA heads should work for inference."""
        with torch.device("cpu"):
            model = GPT(small_config_lora_medusa)
        model.init_weights()
        model.eval()

        idx = torch.randint(0, 256, (2, 32))
        logits, medusa_logits = model(idx, return_medusa=True)

        assert logits.shape == (2, 32, 256)
        assert medusa_logits.shape == (2, 2, 32, 256)  # (num_heads, B, T, vocab)

    def test_forward_training_lora(self, small_config_lora_medusa):
        """Forward with LoRA heads should work for training."""
        with torch.device("cpu"):
            model = GPT(small_config_lora_medusa)
        model.init_weights()

        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))

        loss, medusa_losses = model(idx, targets, return_medusa=True)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert len(medusa_losses) == 2  # 2 Medusa heads

    def test_backward_lora(self, small_config_lora_medusa):
        """Gradients should flow through LoRA params."""
        with torch.device("cpu"):
            model = GPT(small_config_lora_medusa)
        model.init_weights()

        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))

        loss, medusa_losses = model(idx, targets, return_medusa=True)
        total_loss = loss + sum(medusa_losses)
        total_loss.backward()

        # Check gradients on LoRA params
        for head in model.medusa_heads:
            assert head.lora_A.weight.grad is not None
            assert head.lora_B.weight.grad is not None

    def test_optimizer_includes_lora_params(self, small_config_lora_medusa):
        """setup_optimizers should handle LoRA params: A->Muon, B->AdamW."""
        with torch.device("cpu"):
            model = GPT(small_config_lora_medusa)
        model.init_weights()

        optimizers = model.setup_optimizers()
        assert len(optimizers) == 2

        adamw, muon = optimizers

        # Collect all params from each optimizer
        all_adamw_params = []
        for group in adamw.param_groups:
            all_adamw_params.extend(group['params'])

        all_muon_params = []
        for group in muon.param_groups:
            all_muon_params.extend(group['params'])

        # Check that lora_A is in Muon, lora_B is in AdamW
        for head in model.medusa_heads:
            lora_a = head.lora_A.weight
            lora_b = head.lora_B.weight

            assert any(p is lora_a for p in all_muon_params), "lora_A should be in Muon"
            assert not any(p is lora_a for p in all_adamw_params), "lora_A should NOT be in AdamW"

            assert any(p is lora_b for p in all_adamw_params), "lora_B should be in AdamW"
            assert not any(p is lora_b for p in all_muon_params), "lora_B should NOT be in Muon"


class TestIndependentMedusaHead:
    """Tests for IndependentMedusaHead class (low-rank independent predictor).

    Architecture: ResBlocks -> W_a (hidden->rank) -> W_b (rank->vocab)
    The SiLU nonlinearity comes from ResBlocks, not a separate layer.
    """

    def test_independent_head_output_shape(self):
        """IndependentMedusaHead should output correct shape."""
        hidden_size = 64
        vocab_size = 1000
        num_layers = 1
        rank = 8
        head = IndependentMedusaHead(hidden_size, vocab_size, num_layers, rank)

        x = torch.randn(2, 10, hidden_size)
        out = head(x)
        assert out.shape == (2, 10, vocab_size)

    def test_independent_head_has_correct_params(self):
        """IndependentMedusaHead should have W_a, W_b params (not W_hidden)."""
        hidden_size = 64
        vocab_size = 1000
        rank = 8
        head = IndependentMedusaHead(hidden_size, vocab_size, num_layers=1,
                                      rank=rank)

        # Should NOT have W_hidden (SiLU comes from ResBlocks)
        assert not hasattr(head, 'W_hidden')
        assert hasattr(head, 'W_a')
        assert hasattr(head, 'W_b')
        assert head.W_a.weight.shape == (rank, hidden_size)
        assert head.W_b.weight.shape == (vocab_size, rank)

    def test_independent_head_param_count_reduction(self):
        """Independent head should have far fewer params than full projection."""
        hidden_size = 1280
        vocab_size = 65536
        rank = 32

        # Full projection params
        full_head = MedusaHead(hidden_size, vocab_size, num_layers=1)
        full_proj_params = full_head.proj.weight.numel()  # vocab_size * hidden_size

        # Independent head params (W_a + W_b only, no W_hidden)
        ind_head = IndependentMedusaHead(hidden_size, vocab_size, num_layers=1,
                                          rank=rank)
        ind_params = ind_head.W_a.weight.numel() + ind_head.W_b.weight.numel()

        # Independent head should be ~97% smaller
        assert ind_params < full_proj_params * 0.05  # Less than 5% of full

    def test_independent_head_gradient_flow(self):
        """Gradients should flow through both weight matrices."""
        hidden_size = 64
        vocab_size = 100
        head = IndependentMedusaHead(hidden_size, vocab_size, num_layers=1,
                                      rank=8)

        x = torch.randn(2, 10, hidden_size, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert head.W_a.weight.grad is not None
        assert head.W_b.weight.grad is not None

    def test_independent_head_resblocks(self):
        """Independent head should apply ResBlocks before low-rank projection."""
        hidden_size = 64
        vocab_size = 100
        num_layers = 2
        head = IndependentMedusaHead(hidden_size, vocab_size, num_layers,
                                      rank=8)

        assert len(head.blocks) == num_layers
        x = torch.randn(2, 10, hidden_size)
        out = head(x)
        assert out.shape == (2, 10, vocab_size)


class TestGPTWithIndependentMedusa:
    """Tests for GPT model with Independent low-rank Medusa heads."""

    @pytest.fixture
    def small_config_independent_medusa(self):
        """Create a small GPT config with Independent Medusa heads for testing."""
        return GPTConfig(
            sequence_len=64,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=64,
            medusa_num_heads=2,
            medusa_num_layers=1,
            medusa_independent=True,  # Independent heads enabled
            medusa_lora_rank=8,  # Used as bottleneck dimension for independent heads
        )

    def test_gpt_creates_independent_heads(self, small_config_independent_medusa):
        """GPT should create IndependentMedusaHead when medusa_independent=True."""
        with torch.device("cpu"):
            model = GPT(small_config_independent_medusa)
        model.init_weights()

        assert model.medusa_heads is not None
        for head in model.medusa_heads:
            assert isinstance(head, IndependentMedusaHead)
            # Should NOT have W_hidden (SiLU comes from ResBlocks)
            assert not hasattr(head, 'W_hidden')
            assert hasattr(head, 'W_a')
            assert hasattr(head, 'W_b')

    def test_forward_inference_independent(self, small_config_independent_medusa):
        """Forward with independent heads should work for inference."""
        with torch.device("cpu"):
            model = GPT(small_config_independent_medusa)
        model.init_weights()
        model.eval()

        idx = torch.randint(0, 256, (2, 32))
        logits, medusa_logits = model(idx, return_medusa=True)

        assert logits.shape == (2, 32, 256)
        assert medusa_logits.shape == (2, 2, 32, 256)  # (num_heads, B, T, vocab)

    def test_forward_training_independent(self, small_config_independent_medusa):
        """Forward with independent heads should work for training."""
        with torch.device("cpu"):
            model = GPT(small_config_independent_medusa)
        model.init_weights()

        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))

        loss, medusa_losses = model(idx, targets, return_medusa=True)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert len(medusa_losses) == 2  # 2 Medusa heads

    def test_backward_independent(self, small_config_independent_medusa):
        """Gradients should flow through independent head params."""
        with torch.device("cpu"):
            model = GPT(small_config_independent_medusa)
        model.init_weights()

        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))

        loss, medusa_losses = model(idx, targets, return_medusa=True)
        total_loss = loss + sum(medusa_losses)
        total_loss.backward()

        # Check gradients on independent head params (W_a and W_b only)
        for head in model.medusa_heads:
            assert head.W_a.weight.grad is not None
            assert head.W_b.weight.grad is not None

    def test_optimizer_includes_independent_params(self, small_config_independent_medusa):
        """setup_optimizers should handle independent head params: W_a->Muon, W_b->AdamW."""
        with torch.device("cpu"):
            model = GPT(small_config_independent_medusa)
        model.init_weights()

        optimizers = model.setup_optimizers()
        assert len(optimizers) == 2

        adamw, muon = optimizers

        # Collect all params from each optimizer
        all_adamw_params = []
        for group in adamw.param_groups:
            all_adamw_params.extend(group['params'])

        all_muon_params = []
        for group in muon.param_groups:
            all_muon_params.extend(group['params'])

        # Check that W_a is in Muon; W_b is in AdamW
        for head in model.medusa_heads:
            w_a = head.W_a.weight
            w_b = head.W_b.weight

            assert any(p is w_a for p in all_muon_params), "W_a should be in Muon"
            assert not any(p is w_a for p in all_adamw_params), "W_a should NOT be in AdamW"

            assert any(p is w_b for p in all_adamw_params), "W_b should be in AdamW"
            assert not any(p is w_b for p in all_muon_params), "W_b should NOT be in Muon"
