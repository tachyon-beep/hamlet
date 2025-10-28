"""
Tests for neural network architectures.
"""

import pytest
import torch
import torch.nn as nn
from hamlet.agent.networks import QNetwork


def test_qnetwork_initialization():
    """Test that Q-Network initializes correctly."""
    network = QNetwork(state_dim=70, action_dim=5)

    assert network.state_dim == 70
    assert network.action_dim == 5
    assert network.layers is not None
    assert isinstance(network.layers, nn.Sequential)


def test_qnetwork_forward_single_state():
    """Test forward pass with single state."""
    network = QNetwork(state_dim=70, action_dim=5)

    # Single state
    state = torch.randn(70)
    q_values = network(state)

    assert q_values.shape == (5,)
    assert isinstance(q_values, torch.Tensor)


def test_qnetwork_forward_batch():
    """Test forward pass with batch of states."""
    network = QNetwork(state_dim=70, action_dim=5)

    # Batch of states
    batch_size = 32
    states = torch.randn(batch_size, 70)
    q_values = network(states)

    assert q_values.shape == (batch_size, 5)


def test_qnetwork_output_is_unbounded():
    """Test that Q-values are unbounded (no activation on output)."""
    network = QNetwork(state_dim=10, action_dim=3)

    # Create extreme input
    state = torch.ones(10) * 100.0
    q_values = network(state)

    # Q-values should be able to be large (no sigmoid/tanh)
    # Just check it doesn't crash and returns reasonable tensor
    assert q_values.shape == (3,)
    assert not torch.isnan(q_values).any()
    assert not torch.isinf(q_values).any()


def test_qnetwork_custom_hidden_dims():
    """Test network with custom hidden dimensions."""
    network = QNetwork(state_dim=20, action_dim=4, hidden_dims=[64, 64, 32])

    state = torch.randn(20)
    q_values = network(state)

    assert q_values.shape == (4,)


def test_qnetwork_default_hidden_dims():
    """Test that default hidden dims are [128, 128]."""
    network = QNetwork(state_dim=10, action_dim=3)

    # Count linear layers (should be 3: input->hidden1, hidden1->hidden2, hidden2->output)
    linear_layers = [m for m in network.layers if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 3

    # Check dimensions
    assert linear_layers[0].in_features == 10
    assert linear_layers[0].out_features == 128
    assert linear_layers[1].in_features == 128
    assert linear_layers[1].out_features == 128
    assert linear_layers[2].in_features == 128
    assert linear_layers[2].out_features == 3


def test_qnetwork_gradient_flow():
    """Test that gradients flow through network."""
    network = QNetwork(state_dim=10, action_dim=3)

    state = torch.randn(10, requires_grad=True)
    q_values = network(state)

    # Compute loss and backprop
    loss = q_values.sum()
    loss.backward()

    # Check that gradients exist
    assert state.grad is not None
    assert not torch.all(state.grad == 0)


def test_qnetwork_different_batch_sizes():
    """Test network handles different batch sizes."""
    network = QNetwork(state_dim=70, action_dim=5)

    for batch_size in [1, 8, 32, 64]:
        states = torch.randn(batch_size, 70)
        q_values = network(states)
        assert q_values.shape == (batch_size, 5)


def test_qnetwork_parameters_exist():
    """Test that network has trainable parameters."""
    network = QNetwork(state_dim=10, action_dim=3)

    params = list(network.parameters())
    assert len(params) > 0

    # Check all parameters require grad
    for param in params:
        assert param.requires_grad


def test_qnetwork_eval_mode():
    """Test network can switch to eval mode."""
    network = QNetwork(state_dim=10, action_dim=3)

    network.eval()

    state = torch.randn(10)
    with torch.no_grad():
        q_values = network(state)

    assert q_values.shape == (3,)


def test_qnetwork_deterministic_forward():
    """Test that forward pass is deterministic in eval mode."""
    network = QNetwork(state_dim=10, action_dim=3)
    network.eval()

    state = torch.randn(10)

    with torch.no_grad():
        q_values1 = network(state)
        q_values2 = network(state)

    assert torch.allclose(q_values1, q_values2)


def test_qnetwork_with_hamlet_dimensions():
    """Test network with actual Hamlet dimensions."""
    # Hamlet: 70-dim state (2 pos + 4 meters + 64 grid), 5 actions
    network = QNetwork(state_dim=70, action_dim=5)

    # Single state
    state = torch.randn(70)
    q_values = network(state)
    assert q_values.shape == (5,)

    # Batch
    batch_states = torch.randn(32, 70)
    batch_q_values = network(batch_states)
    assert batch_q_values.shape == (32, 5)


def test_qnetwork_can_select_action():
    """Test that network output can be used for action selection."""
    network = QNetwork(state_dim=70, action_dim=5)
    network.eval()

    state = torch.randn(70)

    with torch.no_grad():
        q_values = network(state)
        action = q_values.argmax().item()

    assert isinstance(action, int)
    assert 0 <= action < 5
