"""Tests for network factory."""

import torch

from townlet.agent.brain_config import (
    CNNEncoderConfig,
    FeedforwardConfig,
    LSTMConfig,
    MLPEncoderConfig,
    RecurrentConfig,
)
from townlet.agent.network_factory import NetworkFactory
from townlet.agent.networks import RecurrentSpatialQNetwork


def test_build_feedforward_basic():
    """NetworkFactory builds SimpleQNetwork from FeedforwardConfig."""
    config = FeedforwardConfig(
        hidden_layers=[128, 64],
        activation="relu",
        dropout=0.0,
        layer_norm=True,
    )

    network = NetworkFactory.build_feedforward(
        config=config,
        obs_dim=29,
        action_dim=8,
    )

    # Check output shape
    obs = torch.randn(4, 29)
    q_values = network(obs)
    assert q_values.shape == (4, 8)


def test_build_feedforward_multiple_layers():
    """NetworkFactory handles multiple hidden layers."""
    config = FeedforwardConfig(
        hidden_layers=[256, 128, 64],
        activation="gelu",
        dropout=0.1,
        layer_norm=False,
    )

    network = NetworkFactory.build_feedforward(
        config=config,
        obs_dim=54,
        action_dim=10,
    )

    obs = torch.randn(2, 54)
    q_values = network(obs)
    assert q_values.shape == (2, 10)


def test_build_feedforward_parameter_count():
    """NetworkFactory creates network with expected parameter count."""
    config = FeedforwardConfig(
        hidden_layers=[128],
        activation="relu",
        dropout=0.0,
        layer_norm=True,
    )

    network = NetworkFactory.build_feedforward(
        config=config,
        obs_dim=29,
        action_dim=8,
    )

    total_params = sum(p.numel() for p in network.parameters())
    # Rough sanity check (not exact)
    assert 1000 < total_params < 20000


def test_build_recurrent_basic():
    """NetworkFactory builds RecurrentSpatialQNetwork from RecurrentConfig."""
    config = RecurrentConfig(
        vision_encoder=CNNEncoderConfig(
            channels=[16, 32],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            padding=[1, 1],
            activation="relu",
        ),
        position_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        meter_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        affordance_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        lstm=LSTMConfig(
            hidden_size=256,
            num_layers=1,
            dropout=0.0,
        ),
        q_head=MLPEncoderConfig(
            hidden_sizes=[128],
            activation="relu",
        ),
    )

    network = NetworkFactory.build_recurrent(
        config=config,
        action_dim=8,
        window_size=5,
        position_dim=2,
        num_meters=8,
        num_affordance_types=14,
    )

    # Verify network type
    assert isinstance(network, RecurrentSpatialQNetwork)

    # Test forward pass with dummy observation
    batch_size = 4
    obs_dim = (5 * 5) + 2 + 8 + 15  # grid + position + meters + affordances
    obs = torch.randn(batch_size, obs_dim)

    q_values, hidden = network(obs)

    assert q_values.shape == (batch_size, 8)
    assert isinstance(hidden, tuple)
    assert len(hidden) == 2
    assert hidden[0].shape == (1, batch_size, 256)  # h
    assert hidden[1].shape == (1, batch_size, 256)  # c


def test_build_recurrent_parameter_count():
    """NetworkFactory creates recurrent network with expected parameter count."""
    config = RecurrentConfig(
        vision_encoder=CNNEncoderConfig(
            channels=[16, 32],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            padding=[1, 1],
            activation="relu",
        ),
        position_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        meter_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        affordance_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        lstm=LSTMConfig(
            hidden_size=256,
            num_layers=1,
            dropout=0.0,
        ),
        q_head=MLPEncoderConfig(
            hidden_sizes=[128],
            activation="relu",
        ),
    )

    network = NetworkFactory.build_recurrent(
        config=config,
        action_dim=8,
        window_size=5,
        position_dim=2,
        num_meters=8,
        num_affordance_types=14,
    )

    total_params = sum(p.numel() for p in network.parameters())
    # Recurrent networks are larger (~500K-700K params)
    assert 400_000 < total_params < 1_000_000


def test_build_recurrent_custom_lstm_size():
    """NetworkFactory respects custom LSTM hidden_size from config."""
    config = RecurrentConfig(
        vision_encoder=CNNEncoderConfig(
            channels=[16, 32],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            padding=[1, 1],
            activation="relu",
        ),
        position_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        meter_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        affordance_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        lstm=LSTMConfig(
            hidden_size=128,  # Smaller LSTM
            num_layers=1,
            dropout=0.0,
        ),
        q_head=MLPEncoderConfig(
            hidden_sizes=[64],
            activation="relu",
        ),
    )

    network = NetworkFactory.build_recurrent(
        config=config,
        action_dim=8,
        window_size=5,
        position_dim=2,
        num_meters=8,
        num_affordance_types=14,
    )

    # Verify LSTM hidden size is 128 (not default 256)
    assert network.hidden_dim == 128

    # Test forward pass
    batch_size = 2
    obs_dim = (5 * 5) + 2 + 8 + 15
    obs = torch.randn(batch_size, obs_dim)

    q_values, hidden = network(obs)

    assert q_values.shape == (batch_size, 8)
    assert hidden[0].shape == (1, batch_size, 128)  # h with custom size
    assert hidden[1].shape == (1, batch_size, 128)  # c with custom size


def test_build_recurrent_aspatial():
    """NetworkFactory builds recurrent network for aspatial substrate (position_dim=0)."""
    config = RecurrentConfig(
        vision_encoder=CNNEncoderConfig(
            channels=[16, 32],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            padding=[1, 1],
            activation="relu",
        ),
        position_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        meter_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        affordance_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        lstm=LSTMConfig(
            hidden_size=256,
            num_layers=1,
            dropout=0.0,
        ),
        q_head=MLPEncoderConfig(
            hidden_sizes=[128],
            activation="relu",
        ),
    )

    network = NetworkFactory.build_recurrent(
        config=config,
        action_dim=4,
        window_size=5,
        position_dim=0,  # Aspatial (no position)
        num_meters=8,
        num_affordance_types=14,
    )

    # Test forward pass without position component
    batch_size = 2
    obs_dim = (5 * 5) + 0 + 8 + 15  # grid + no position + meters + affordances
    obs = torch.randn(batch_size, obs_dim)

    q_values, hidden = network(obs)

    assert q_values.shape == (batch_size, 4)
    assert hidden[0].shape == (1, batch_size, 256)
    assert hidden[1].shape == (1, batch_size, 256)
