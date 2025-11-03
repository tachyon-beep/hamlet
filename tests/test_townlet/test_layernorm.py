"""
Tests for LayerNorm additions to neural networks.

Following TDD: Write test first, watch it fail, then implement.
"""

import torch
import pytest

from townlet.agent.networks import SimpleQNetwork, RecurrentSpatialQNetwork


class TestSimpleQNetworkLayerNorm:
    """Test that SimpleQNetwork has LayerNorm layers."""

    def test_has_layernorm_after_first_layer(self):
        """SimpleQNetwork should have LayerNorm after first Linear layer."""
        obs_dim = 72
        action_dim = 5
        hidden_dim = 128

        network = SimpleQNetwork(obs_dim, action_dim, hidden_dim)

        # Check that network contains LayerNorm modules
        has_layernorm = any(isinstance(module, torch.nn.LayerNorm) for module in network.modules())
        assert has_layernorm, "SimpleQNetwork should contain LayerNorm layers"

    def test_has_two_layernorm_layers(self):
        """SimpleQNetwork should have exactly 2 LayerNorm layers (after each hidden layer)."""
        obs_dim = 72
        action_dim = 5
        hidden_dim = 128

        network = SimpleQNetwork(obs_dim, action_dim, hidden_dim)

        # Count LayerNorm modules
        layernorm_count = sum(1 for module in network.modules() if isinstance(module, torch.nn.LayerNorm))
        assert layernorm_count == 2, f"Expected 2 LayerNorm layers, found {layernorm_count}"

    def test_layernorm_has_correct_normalized_shape(self):
        """LayerNorm should normalize over hidden_dim."""
        obs_dim = 72
        action_dim = 5
        hidden_dim = 128

        network = SimpleQNetwork(obs_dim, action_dim, hidden_dim)

        # Find LayerNorm modules and check their normalized_shape
        layernorms = [module for module in network.modules() if isinstance(module, torch.nn.LayerNorm)]
        for ln in layernorms:
            assert ln.normalized_shape == (hidden_dim,), f"LayerNorm should normalize over ({hidden_dim},), got {ln.normalized_shape}"

    def test_forward_pass_works_with_layernorm(self):
        """Forward pass should work correctly with LayerNorm."""
        obs_dim = 72
        action_dim = 5
        hidden_dim = 128

        network = SimpleQNetwork(obs_dim, action_dim, hidden_dim)

        # Create dummy input
        batch_size = 4
        obs = torch.randn(batch_size, obs_dim)

        # Forward pass
        q_values = network(obs)

        # Check output shape
        assert q_values.shape == (batch_size, action_dim), f"Expected shape ({batch_size}, {action_dim}), got {q_values.shape}"

    def test_gradient_flow_with_layernorm(self):
        """Gradients should flow through LayerNorm layers."""
        obs_dim = 72
        action_dim = 5
        hidden_dim = 128

        network = SimpleQNetwork(obs_dim, action_dim, hidden_dim)

        # Forward + backward pass
        obs = torch.randn(1, obs_dim)
        q_values = network(obs)
        loss = q_values.sum()
        loss.backward()

        # Check that LayerNorm parameters have gradients
        layernorms = [module for module in network.modules() if isinstance(module, torch.nn.LayerNorm)]
        for ln in layernorms:
            assert ln.weight.grad is not None, "LayerNorm weight should have gradients"
            assert ln.bias.grad is not None, "LayerNorm bias should have gradients"


class TestRecurrentSpatialQNetworkLayerNorm:
    """Test that RecurrentSpatialQNetwork has LayerNorm layers."""

    def test_has_layernorm_modules(self):
        """RecurrentSpatialQNetwork should contain LayerNorm modules."""
        action_dim = 5
        window_size = 5

        network = RecurrentSpatialQNetwork(action_dim=action_dim, window_size=window_size)

        # Check that network contains LayerNorm modules
        has_layernorm = any(isinstance(module, torch.nn.LayerNorm) for module in network.modules())
        assert has_layernorm, "RecurrentSpatialQNetwork should contain LayerNorm layers"

    def test_has_three_layernorm_layers(self):
        """RecurrentSpatialQNetwork should have 3 LayerNorm layers (after vision encoder, after LSTM, in Q-head)."""
        action_dim = 5
        window_size = 5

        network = RecurrentSpatialQNetwork(action_dim=action_dim, window_size=window_size)

        # Count LayerNorm modules
        layernorm_count = sum(1 for module in network.modules() if isinstance(module, torch.nn.LayerNorm))
        assert layernorm_count == 3, f"Expected 3 LayerNorm layers, found {layernorm_count}"

    def test_vision_encoder_has_layernorm(self):
        """Vision encoder should have LayerNorm after final Linear layer."""
        action_dim = 5
        window_size = 5

        network = RecurrentSpatialQNetwork(action_dim=action_dim, window_size=window_size)

        # Check vision encoder contains LayerNorm
        vision_has_ln = any(isinstance(module, torch.nn.LayerNorm) for module in network.vision_encoder.modules())
        assert vision_has_ln, "Vision encoder should contain LayerNorm"

    def test_q_head_has_layernorm(self):
        """Q-head should have LayerNorm between layers."""
        action_dim = 5
        window_size = 5

        network = RecurrentSpatialQNetwork(action_dim=action_dim, window_size=window_size)

        # Check q_head contains LayerNorm
        qhead_has_ln = any(isinstance(module, torch.nn.LayerNorm) for module in network.q_head.modules())
        assert qhead_has_ln, "Q-head should contain LayerNorm"

    def test_has_lstm_norm_attribute(self):
        """Network should have lstm_norm attribute for normalizing LSTM output."""
        action_dim = 5
        window_size = 5

        network = RecurrentSpatialQNetwork(action_dim=action_dim, window_size=window_size)

        # Check lstm_norm exists
        assert hasattr(network, 'lstm_norm'), "Network should have lstm_norm attribute"
        assert isinstance(network.lstm_norm, torch.nn.LayerNorm), "lstm_norm should be a LayerNorm layer"

    def test_forward_pass_works_with_layernorm(self):
        """Forward pass should work correctly with LayerNorm."""
        action_dim = 5
        window_size = 5

        network = RecurrentSpatialQNetwork(action_dim=action_dim, window_size=window_size)

        # Create dummy input (obs_dim = 25 + 2 + 8 + 16 = 51 for default settings)
        batch_size = 4
        obs_dim = window_size * window_size + 2 + 8 + 16
        obs = torch.randn(batch_size, obs_dim)

        # Forward pass
        q_values, hidden = network(obs)

        # Check output shape
        assert q_values.shape == (batch_size, action_dim), f"Expected shape ({batch_size}, {action_dim}), got {q_values.shape}"
        assert hidden[0].shape == (1, batch_size, 256), f"Expected hidden shape (1, {batch_size}, 256), got {hidden[0].shape}"

    def test_gradient_flow_with_layernorm(self):
        """Gradients should flow through LayerNorm layers."""
        action_dim = 5
        window_size = 5

        network = RecurrentSpatialQNetwork(action_dim=action_dim, window_size=window_size)

        # Forward + backward pass
        obs_dim = window_size * window_size + 2 + 8 + 16
        obs = torch.randn(1, obs_dim)
        q_values, _ = network(obs)
        loss = q_values.sum()
        loss.backward()

        # Check that LayerNorm parameters have gradients
        layernorms = [module for module in network.modules() if isinstance(module, torch.nn.LayerNorm)]
        assert len(layernorms) == 3, f"Expected 3 LayerNorm modules, found {len(layernorms)}"

        for ln in layernorms:
            assert ln.weight.grad is not None, "LayerNorm weight should have gradients"
            assert ln.bias.grad is not None, "LayerNorm bias should have gradients"
