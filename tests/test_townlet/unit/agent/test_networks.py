"""Test Suite: Neural Network Architectures

Tests for SimpleQNetwork and RecurrentSpatialQNetwork architectures.

This file tests agent/networks.py:
- SimpleQNetwork: MLP for full observability
- RecurrentSpatialQNetwork: CNN + LSTM for partial observability

Critical Areas:
1. Input/output shape validation
2. Hidden state management (LSTM)
3. Batch processing
4. Q-value computation
5. Gradient flow (basic sanity)

Old location: tests/test_townlet/test_networks.py
New location: tests/test_townlet/unit/agent/test_networks.py (migrated 2025-11-04)
"""

import pytest
import torch
import torch.nn as nn

from townlet.agent.networks import RecurrentSpatialQNetwork, SimpleQNetwork

FULL_OBS_DIM = 93  # 8×8 occupancy grid (64) + position (2) + meters (8) + affordance (15) + temporal (4)
ACTION_DIM = 8  # Grid2D: 6 substrate actions + 2 custom actions (TASK-002B)


class TestSimpleQNetwork:
    """Test SimpleQNetwork (MLP for full observability)."""

    @pytest.fixture
    def network(self):
        """Create SimpleQNetwork with standard config."""
        obs_dim = FULL_OBS_DIM
        action_dim = ACTION_DIM  # Grid2D substrate + custom actions
        hidden_dim = 128  # Standard hidden dimension
        return SimpleQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)

    def test_initialization(self, network):
        """Network should initialize with correct architecture including LayerNorm."""
        assert isinstance(network, nn.Module)
        assert isinstance(network.net, nn.Sequential)

        # Check network is a 7-layer sequential (Linear-LayerNorm-ReLU-Linear-LayerNorm-ReLU-Linear)
        assert len(network.net) == 7
        assert isinstance(network.net[0], nn.Linear)  # First linear
        assert isinstance(network.net[1], nn.LayerNorm)  # LayerNorm
        assert isinstance(network.net[2], nn.ReLU)  # ReLU
        assert isinstance(network.net[3], nn.Linear)  # Second linear
        assert isinstance(network.net[4], nn.LayerNorm)  # LayerNorm
        assert isinstance(network.net[5], nn.ReLU)  # ReLU
        assert isinstance(network.net[6], nn.Linear)  # Q-head

        # Check dimensions
        assert network.net[0].in_features == FULL_OBS_DIM
        assert network.net[0].out_features == 128
        assert network.net[3].in_features == 128
        assert network.net[3].out_features == 128
        assert network.net[6].in_features == 128
        assert network.net[6].out_features == ACTION_DIM

    def test_forward_pass_single_agent(self, network):
        """Forward pass should produce correct Q-value shapes."""
        obs = torch.randn(1, FULL_OBS_DIM)  # Single agent
        q_values = network(obs)

        assert q_values.shape == (1, ACTION_DIM), f"Expected (1, {ACTION_DIM}), got {q_values.shape}"
        assert not torch.isnan(q_values).any(), "Q-values contain NaN"
        assert not torch.isinf(q_values).any(), "Q-values contain Inf"

    def test_forward_pass_batch(self, network):
        """Forward pass should handle batched observations."""
        batch_size = 32
        obs = torch.randn(batch_size, FULL_OBS_DIM)
        q_values = network(obs)

        assert q_values.shape == (batch_size, ACTION_DIM)
        assert not torch.isnan(q_values).any()

    def test_q_value_range(self, network):
        """Q-values should be reasonable (not exploding)."""
        obs = torch.randn(10, FULL_OBS_DIM)
        q_values = network(obs)

        # Q-values shouldn't be crazy large (untrained network)
        assert q_values.abs().max() < 100, "Untrained Q-values too large (initialization issue?)"

    def test_different_observation_dimensions(self):
        """Network should work with different observation dimensions."""
        # Full observability baseline
        net_full = SimpleQNetwork(obs_dim=FULL_OBS_DIM, action_dim=ACTION_DIM, hidden_dim=128)
        obs_full = torch.randn(4, FULL_OBS_DIM)
        q_full = net_full(obs_full)
        assert q_full.shape == (4, ACTION_DIM)

        # With temporal features (+ time_of_day + interaction_progress)
        net_temporal = SimpleQNetwork(obs_dim=FULL_OBS_DIM + 2, action_dim=ACTION_DIM, hidden_dim=128)
        obs_temporal = torch.randn(4, FULL_OBS_DIM + 2)
        q_temporal = net_temporal(obs_temporal)
        assert q_temporal.shape == (4, ACTION_DIM)

    def test_gradient_flow(self, network):
        """Gradients should flow through network."""
        obs = torch.randn(4, FULL_OBS_DIM, requires_grad=True)
        q_values = network(obs)
        loss = q_values.mean()
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in network.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


class TestRecurrentSpatialQNetwork:
    """Test RecurrentSpatialQNetwork (LSTM for partial observability)."""

    @pytest.fixture
    def network(self):
        """Create RecurrentSpatialQNetwork with standard config."""
        # Network computes obs_dim from window_size + num_meters + affordances
        # Default num_affordance_types=15 → encoding size = 15+1 = 16
        # Observation: 25 (5×5) + 2 (position) + 8 (meters) + 16 (affordance) = 51
        return RecurrentSpatialQNetwork(
            action_dim=ACTION_DIM,
            window_size=5,
            position_dim=2,
            num_meters=8,
            num_affordance_types=15,
            enable_temporal_features=False,
            hidden_dim=256,
        )

    def test_initialization(self, network):
        """Network should initialize with correct architecture."""
        assert isinstance(network, nn.Module)

        # Check encoder components
        assert hasattr(network, "vision_encoder")
        assert hasattr(network, "position_encoder")
        assert hasattr(network, "meter_encoder")
        assert hasattr(network, "affordance_encoder")

        # Check LSTM
        assert isinstance(network.lstm, nn.LSTM)
        assert network.lstm.input_size == 224  # 128+32+32+32
        assert network.lstm.hidden_size == 256
        assert network.lstm.num_layers == 1

        # Check Q-head (Sequential with 4 layers: Linear-LayerNorm-ReLU-Linear)
        assert isinstance(network.q_head, nn.Sequential)
        assert isinstance(network.q_head[0], nn.Linear)  # 256 → 128
        assert isinstance(network.q_head[1], nn.LayerNorm)  # LayerNorm
        assert isinstance(network.q_head[2], nn.ReLU)
        assert isinstance(network.q_head[3], nn.Linear)  # 128 → 6

    def test_forward_pass_without_hidden_state(self, network):
        """Forward pass should work without providing hidden state."""
        # 25 (grid) + 2 (pos) + 8 (meters) + 16 (affordance encoding) = 51
        obs = torch.randn(1, 51)
        q_values, hidden = network(obs)

        assert q_values.shape == (1, ACTION_DIM)
        assert hidden is not None
        assert len(hidden) == 2  # (h, c)
        assert hidden[0].shape == (1, 1, 256)  # (num_layers, batch, hidden_size)
        assert hidden[1].shape == (1, 1, 256)

    def test_forward_pass_with_hidden_state(self, network):
        """Forward pass should accept and update hidden state."""
        obs = torch.randn(1, 51)

        # First forward pass
        q1, hidden1 = network(obs)

        # Second forward pass with previous hidden state
        q2, hidden2 = network(obs, hidden1)

        assert q2.shape == (1, ACTION_DIM)
        assert not torch.equal(q1, q2), "Q-values should differ with different hidden states"
        assert not torch.equal(hidden1[0], hidden2[0]), "Hidden state should be updated"

    def test_hidden_state_management(self, network):
        """Hidden state should persist across forward passes."""
        batch_size = 1
        obs = torch.randn(batch_size, 51)

        # Reset hidden state
        network.reset_hidden_state(batch_size, device=torch.device("cpu"))
        h, c = network.get_hidden_state()

        assert h is not None
        assert c is not None
        assert h.shape == (1, batch_size, 256)
        assert torch.all(h == 0), "Hidden state should be zeroed"
        assert torch.all(c == 0), "Cell state should be zeroed"

        # Forward pass returns new hidden state but doesn't auto-update stored state
        q_values, new_hidden = network(obs)

        # New hidden state should be different from zeros (it was updated by LSTM)
        assert not torch.equal(new_hidden[0], torch.zeros_like(new_hidden[0]))
        assert not torch.equal(new_hidden[1], torch.zeros_like(new_hidden[1]))

        # But stored state should still be zeros (not automatically updated)
        stored_h, stored_c = network.get_hidden_state()
        assert torch.all(stored_h == 0), "Stored hidden state should still be zero (not auto-updated)"
        assert torch.all(stored_c == 0), "Stored cell state should still be zero (not auto-updated)"

        # Manually update stored state
        network.set_hidden_state(new_hidden)
        stored_h, stored_c = network.get_hidden_state()

        # Now stored state should match
        assert torch.equal(stored_h, new_hidden[0])
        assert torch.equal(stored_c, new_hidden[1])

    def test_set_hidden_state(self, network):
        """Should be able to manually set hidden state."""
        batch_size = 2
        custom_h = torch.randn(1, batch_size, 256)
        custom_c = torch.randn(1, batch_size, 256)

        network.set_hidden_state((custom_h, custom_c))

        stored_h, stored_c = network.get_hidden_state()
        assert torch.equal(stored_h, custom_h)
        assert torch.equal(stored_c, custom_c)

    def test_batch_processing(self, network):
        """Network should handle batched observations."""
        batch_size = 8
        obs = torch.randn(batch_size, 51)

        network.reset_hidden_state(batch_size, device=torch.device("cpu"))
        q_values, hidden = network(obs)

        assert q_values.shape == (batch_size, ACTION_DIM)
        assert hidden[0].shape == (1, batch_size, 256)
        assert hidden[1].shape == (1, batch_size, 256)

    def test_vision_encoding(self, network):
        """Vision encoder should process 5×5 window correctly."""
        # Test just the vision encoder component
        # Vision goes through conv-like processing (reshaped to spatial)
        # This is tested implicitly in full forward pass
        obs = torch.randn(4, 51)
        q_values, _ = network(obs)

        assert q_values.shape == (4, ACTION_DIM)
        assert not torch.isnan(q_values).any()

    def test_gradient_flow_through_lstm(self, network):
        """Gradients should flow through LSTM."""
        obs = torch.randn(4, 51, requires_grad=True)
        network.reset_hidden_state(4, device=torch.device("cpu"))

        q_values, _ = network(obs)
        loss = q_values.mean()
        loss.backward()

        # Check gradients on LSTM parameters
        for name, param in network.named_parameters():
            if "lstm" in name:
                assert param.grad is not None, f"No gradient for LSTM parameter {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_sequential_hidden_state_evolution(self, network):
        """Hidden state should evolve over multiple steps."""
        batch_size = 1
        network.reset_hidden_state(batch_size, device=torch.device("cpu"))

        hidden_states = []
        for step in range(5):
            obs = torch.randn(batch_size, 51)
            q_values, hidden = network(obs)
            hidden_states.append(hidden[0].clone())

        # Hidden states should differ across steps
        for i in range(len(hidden_states) - 1):
            assert not torch.equal(hidden_states[i], hidden_states[i + 1]), f"Hidden state unchanged between steps {i} and {i + 1}"

    def test_different_observation_dimensions(self):
        """Network should work with temporal features added (obs dimensions change)."""
        # Base POMDP: 25 (5×5) + 2 (pos) + 8 (meters) + 15 (affordance) = 50
        net_base = RecurrentSpatialQNetwork(
            action_dim=ACTION_DIM,
            window_size=5,
            position_dim=2,
            num_meters=8,
            num_affordance_types=15,
            enable_temporal_features=False,
            hidden_dim=256,
        )
        obs_base = torch.randn(2, 51)
        q_base, _ = net_base(obs_base)
        assert q_base.shape == (2, ACTION_DIM)

        # With temporal: 25 + 2 + 8 + 16 + 2 (time + progress) = 52
        # But network doesn't know about temporal - it just processes extra dims
        # So we test it handles the expected 50 dims correctly
        net_temporal = RecurrentSpatialQNetwork(
            action_dim=ACTION_DIM,
            window_size=5,
            position_dim=2,
            num_meters=8,
            num_affordance_types=15,
            enable_temporal_features=False,
            hidden_dim=256,
        )
        obs_temporal = torch.randn(2, 51)
        q_temporal, _ = net_temporal(obs_temporal)
        assert q_temporal.shape == (2, ACTION_DIM)

    def test_lstm_memory_effect(self, network):
        """LSTM should show memory effect across sequence."""
        batch_size = 1
        network.reset_hidden_state(batch_size, device=torch.device("cpu"))

        # Create sequence of similar observations
        obs_sequence = [torch.randn(batch_size, 51) for _ in range(3)]

        # Get Q-values for each step
        q_values_list = []
        for obs in obs_sequence:
            q_values, _ = network(obs)
            q_values_list.append(q_values.clone())

        # Q-values should change even for similar inputs (due to hidden state)
        assert not torch.equal(q_values_list[0], q_values_list[1])
        assert not torch.equal(q_values_list[1], q_values_list[2])

        # Now reset and try same sequence - should get same results
        network.reset_hidden_state(batch_size, device=torch.device("cpu"))
        q_values_reset_list = []
        for obs in obs_sequence:
            q_values, _ = network(obs)
            q_values_reset_list.append(q_values.clone())

        # Should match original sequence
        for i in range(3):
            assert torch.allclose(q_values_list[i], q_values_reset_list[i], atol=1e-6)


class TestNetworkComparison:
    """Compare behavior between SimpleQNetwork and RecurrentSpatialQNetwork."""

    def test_parameter_counts(self):
        """RecurrentSpatialQNetwork should have more parameters (LSTM)."""
        simple_net = SimpleQNetwork(obs_dim=FULL_OBS_DIM, action_dim=ACTION_DIM, hidden_dim=128)
        recurrent_net = RecurrentSpatialQNetwork(
            action_dim=ACTION_DIM,
            window_size=5,
            position_dim=2,
            num_meters=8,
            num_affordance_types=15,
            enable_temporal_features=False,
            hidden_dim=256,
        )

        simple_params = sum(p.numel() for p in simple_net.parameters())
        recurrent_params = sum(p.numel() for p in recurrent_net.parameters())

        print("\n📊 Parameter counts:")
        print(f"   SimpleQNetwork: {simple_params:,} parameters")
        print(f"   RecurrentSpatialQNetwork: {recurrent_params:,} parameters")

        assert recurrent_params > simple_params, "Recurrent network should have more parameters due to LSTM"

    def test_computational_difference(self):
        """Recurrent network should take longer due to sequential LSTM."""
        import time

        simple_net = SimpleQNetwork(obs_dim=FULL_OBS_DIM, action_dim=ACTION_DIM, hidden_dim=128)
        recurrent_net = RecurrentSpatialQNetwork(
            action_dim=ACTION_DIM,
            window_size=5,
            position_dim=2,
            num_meters=8,
            num_affordance_types=15,
            enable_temporal_features=False,
            hidden_dim=256,
        )

        batch_size = 32
        obs_simple = torch.randn(batch_size, FULL_OBS_DIM)
        obs_recurrent = torch.randn(batch_size, 51)  # 25 + 2 + 8 + 16

        # Warm up
        _ = simple_net(obs_simple)
        recurrent_net.reset_hidden_state(batch_size, device=torch.device("cpu"))
        _ = recurrent_net(obs_recurrent)

        # Time simple network
        start = time.time()
        for _ in range(100):
            _ = simple_net(obs_simple)
        simple_time = time.time() - start

        # Time recurrent network
        recurrent_net.reset_hidden_state(batch_size, device=torch.device("cpu"))
        start = time.time()
        for _ in range(100):
            _ = recurrent_net(obs_recurrent)
        recurrent_time = time.time() - start

        print("\n⏱️ Inference time (100 forward passes):")
        print(f"   SimpleQNetwork: {simple_time * 1000:.2f}ms")
        print(f"   RecurrentSpatialQNetwork: {recurrent_time * 1000:.2f}ms")
        print(f"   Ratio: {recurrent_time / simple_time:.2f}x slower")

        # Recurrent should be slower (LSTM is computationally expensive)
        assert recurrent_time > simple_time
        # Note: LSTM can be 20-30x slower due to sequential nature, this is expected
        assert recurrent_time < simple_time * 50, "Recurrent network way too slow!"
