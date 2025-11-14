"""BrainConfig test fixtures for WP-C2 migration.

Provides standardized brain.yaml fixtures for all test scenarios:
- minimal_brain_config: SimpleQNetwork for unit tests
- recurrent_brain_config: RecurrentSpatialQNetwork for POMDP tests
- legacy_compatible_brain_config: Matches old hardcoded defaults

Usage:
    def test_something(minimal_brain_config):
        population = VectorizedPopulation(..., brain_config_path=minimal_brain_config)
"""

import pytest


@pytest.fixture
def minimal_brain_config(tmp_path):
    """Minimal brain.yaml for SimpleQNetwork testing.

    Use for: Unit tests requiring minimal network configuration.
    Architecture: SimpleQNetwork (MLP: obs_dim → 128 → 64 → action_dim)
    """
    brain_yaml = tmp_path / "brain.yaml"
    brain_yaml.write_text(
        """
architecture:
  type: simple_q
  hidden_dims: [128, 64]
  activation: relu

optimizer:
  type: adam
  learning_rate: 1e-3
  weight_decay: 0.0

q_learning:
  gamma: 0.99
  use_double_dqn: false
  target_update_frequency: 100

loss:
  type: smooth_l1
  beta: 1.0

replay:
  type: standard
  capacity: 10000
  batch_size: 32
"""
    )
    return brain_yaml


@pytest.fixture
def recurrent_brain_config(tmp_path):
    """Recurrent brain.yaml for LSTM testing.

    Use for: POMDP tests requiring RecurrentSpatialQNetwork.
    Architecture: RecurrentSpatialQNetwork (CNN+LSTM)
    """
    brain_yaml = tmp_path / "brain.yaml"
    brain_yaml.write_text(
        """
architecture:
  type: recurrent_spatial_q
  lstm_hidden_size: 256
  activation: relu

optimizer:
  type: adam
  learning_rate: 3e-4
  weight_decay: 1e-5

q_learning:
  gamma: 0.99
  use_double_dqn: true
  target_update_frequency: 200

loss:
  type: huber
  delta: 1.0

replay:
  type: sequential
  capacity: 5000
  batch_size: 16
  sequence_length: 8
"""
    )
    return brain_yaml


@pytest.fixture
def legacy_compatible_brain_config(tmp_path):
    """Brain.yaml matching old hardcoded defaults.

    Use for: Backward compatibility tests (legacy checkpoint loading).
    Matches old hardcoded values: hidden_dim=256, lr=3e-4, gamma=0.99
    """
    brain_yaml = tmp_path / "brain.yaml"
    brain_yaml.write_text(
        """
architecture:
  type: simple_q
  hidden_dims: [256, 128]
  activation: relu

optimizer:
  type: adam
  learning_rate: 3e-4
  weight_decay: 0.0

q_learning:
  gamma: 0.99
  use_double_dqn: false
  target_update_frequency: 100

loss:
  type: mse

replay:
  type: standard
  capacity: 50000
  batch_size: 64
"""
    )
    return brain_yaml
