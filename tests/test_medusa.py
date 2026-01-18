"""
Tests for Medusa Multi-Token Prediction (MTP) implementation.

Run: uv run pytest tests/test_medusa.py -v
"""

import pytest
import torch
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig, MedusaResBlock, MedusaHead


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
        """Forward with targets and return_medusa should return loss, logits, medusa_logits."""
        with torch.device("cpu"):
            model = GPT(small_config_with_medusa)
        model.init_weights()

        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))

        loss, logits, medusa_logits = model(idx, targets, return_medusa=True)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert logits.shape == (2, 32, 256)
        assert medusa_logits.shape == (3, 2, 32, 256)

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
