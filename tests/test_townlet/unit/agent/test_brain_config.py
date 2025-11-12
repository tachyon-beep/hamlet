"""Tests for brain configuration DTOs."""

import pytest
from pydantic import ValidationError

from townlet.agent.brain_config import (
    ArchitectureConfig,
    BrainConfig,
    FeedforwardConfig,
    LossConfig,
    OptimizerConfig,
    QLearningConfig,
    compute_brain_hash,
    load_brain_config,
)


def test_feedforward_config_valid():
    """FeedforwardConfig accepts valid parameters."""
    config = FeedforwardConfig(
        hidden_layers=[256, 128],
        activation="relu",
        dropout=0.0,
        layer_norm=True,
    )
    assert config.hidden_layers == [256, 128]
    assert config.activation == "relu"
    assert config.dropout == 0.0
    assert config.layer_norm is True


def test_feedforward_config_rejects_empty_layers():
    """FeedforwardConfig requires at least one hidden layer."""
    with pytest.raises(ValidationError) as exc_info:
        FeedforwardConfig(
            hidden_layers=[],
            activation="relu",
            dropout=0.0,
            layer_norm=False,
        )
    assert "hidden_layers" in str(exc_info.value)


def test_feedforward_config_rejects_invalid_activation():
    """FeedforwardConfig rejects unsupported activation functions."""
    with pytest.raises(ValidationError) as exc_info:
        FeedforwardConfig(
            hidden_layers=[128],
            activation="invalid",
            dropout=0.0,
            layer_norm=False,
        )
    assert "activation" in str(exc_info.value)


def test_feedforward_config_rejects_negative_dropout():
    """FeedforwardConfig rejects dropout < 0."""
    with pytest.raises(ValidationError) as exc_info:
        FeedforwardConfig(
            hidden_layers=[128],
            activation="relu",
            dropout=-0.1,
            layer_norm=False,
        )
    assert "dropout" in str(exc_info.value)


def test_feedforward_config_rejects_dropout_gte_1():
    """FeedforwardConfig rejects dropout >= 1.0."""
    with pytest.raises(ValidationError) as exc_info:
        FeedforwardConfig(
            hidden_layers=[128],
            activation="relu",
            dropout=1.0,
            layer_norm=False,
        )
    assert "dropout" in str(exc_info.value)


def test_optimizer_config_adam():
    """OptimizerConfig accepts Adam configuration."""
    config = OptimizerConfig(
        type="adam",
        learning_rate=0.00025,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,
    )
    assert config.type == "adam"
    assert config.learning_rate == 0.00025


def test_optimizer_config_sgd():
    """OptimizerConfig accepts SGD configuration."""
    config = OptimizerConfig(
        type="sgd",
        learning_rate=0.01,
        sgd_momentum=0.9,
        sgd_nesterov=True,
        weight_decay=0.0,
    )
    assert config.type == "sgd"
    assert config.sgd_momentum == 0.9


def test_optimizer_config_rejects_negative_lr():
    """OptimizerConfig rejects negative learning rate."""
    with pytest.raises(ValidationError) as exc_info:
        OptimizerConfig(
            type="adam",
            learning_rate=-0.001,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_eps=1e-8,
            weight_decay=0.0,
        )
    assert "learning_rate" in str(exc_info.value)


def test_loss_config_mse():
    """LossConfig accepts MSE loss."""
    config = LossConfig(type="mse")
    assert config.type == "mse"


def test_loss_config_huber():
    """LossConfig accepts Huber loss with delta."""
    config = LossConfig(type="huber", huber_delta=1.0)
    assert config.type == "huber"
    assert config.huber_delta == 1.0


def test_loss_config_rejects_negative_huber_delta():
    """LossConfig rejects negative huber_delta."""
    with pytest.raises(ValidationError) as exc_info:
        LossConfig(type="huber", huber_delta=-1.0)
    assert "huber_delta" in str(exc_info.value)


def test_brain_config_feedforward():
    """BrainConfig accepts feedforward architecture."""
    config = BrainConfig(
        version="1.0",
        description="Simple feedforward Q-network",
        architecture=ArchitectureConfig(
            type="feedforward",
            feedforward=FeedforwardConfig(
                hidden_layers=[256, 128],
                activation="relu",
                dropout=0.0,
                layer_norm=True,
            ),
        ),
        optimizer=OptimizerConfig(
            type="adam",
            learning_rate=0.00025,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_eps=1e-8,
            weight_decay=0.0,
        ),
        loss=LossConfig(type="mse"),
        q_learning=QLearningConfig(
            gamma=0.99,
            target_update_frequency=100,
            use_double_dqn=False,
        ),
    )
    assert config.architecture.type == "feedforward"
    assert config.optimizer.type == "adam"


def test_brain_config_requires_feedforward_when_type_feedforward():
    """BrainConfig requires feedforward field when type=feedforward."""
    with pytest.raises(ValidationError) as exc_info:
        BrainConfig(
            version="1.0",
            description="Test",
            architecture=ArchitectureConfig(
                type="feedforward",
                # Missing feedforward field!
            ),
            optimizer=OptimizerConfig(
                type="adam",
                learning_rate=0.001,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                weight_decay=0.0,
            ),
            loss=LossConfig(type="mse"),
            q_learning=QLearningConfig(
                gamma=0.99,
                target_update_frequency=100,
                use_double_dqn=False,
            ),
        )
    assert "feedforward" in str(exc_info.value).lower()


def test_load_brain_config_valid(tmp_path):
    """load_brain_config loads valid brain.yaml."""
    brain_yaml = tmp_path / "brain.yaml"
    brain_yaml.write_text(
        """
version: "1.0"
description: "Test feedforward network"

architecture:
  type: feedforward
  feedforward:
    hidden_layers: [128, 64]
    activation: relu
    dropout: 0.0
    layer_norm: true

optimizer:
  type: adam
  learning_rate: 0.001
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_eps: 1.0e-8
  weight_decay: 0.0

loss:
  type: mse
  huber_delta: 1.0

q_learning:
  gamma: 0.99
  target_update_frequency: 100
  use_double_dqn: false
"""
    )

    config = load_brain_config(tmp_path)
    assert config.version == "1.0"
    assert config.architecture.feedforward.hidden_layers == [128, 64]
    assert config.optimizer.learning_rate == 0.001


def test_load_brain_config_missing_file(tmp_path):
    """load_brain_config raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_brain_config(tmp_path)
    assert "brain.yaml" in str(exc_info.value)


def test_load_brain_config_invalid_yaml(tmp_path):
    """load_brain_config raises ValueError for invalid YAML."""
    brain_yaml = tmp_path / "brain.yaml"
    brain_yaml.write_text(
        """
version: "1.0"
architecture:
  type: feedforward
  # Missing feedforward config!
optimizer:
  type: adam
  learning_rate: 0.001
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_eps: 1.0e-8
  weight_decay: 0.0
loss:
  type: mse
q_learning:
  gamma: 0.99
  target_update_frequency: 100
  use_double_dqn: false
"""
    )

    with pytest.raises(ValueError) as exc_info:
        load_brain_config(tmp_path)
    assert "invalid" in str(exc_info.value).lower()


def test_compute_brain_hash():
    """compute_brain_hash returns deterministic SHA256 hash."""
    config = BrainConfig(
        version="1.0",
        description="Test config",
        architecture=ArchitectureConfig(
            type="feedforward",
            feedforward=FeedforwardConfig(
                hidden_layers=[128, 64],
                activation="relu",
                dropout=0.0,
                layer_norm=True,
            ),
        ),
        optimizer=OptimizerConfig(
            type="adam",
            learning_rate=0.001,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_eps=1e-8,
            weight_decay=0.0,
        ),
        loss=LossConfig(type="mse"),
        q_learning=QLearningConfig(
            gamma=0.99,
            target_update_frequency=100,
            use_double_dqn=False,
        ),
    )

    hash1 = compute_brain_hash(config)
    hash2 = compute_brain_hash(config)

    # Hash should be deterministic
    assert hash1 == hash2
    # Hash should be 64-character hex string (SHA256)
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)


def test_compute_brain_hash_differs_for_different_configs():
    """compute_brain_hash produces different hashes for different configs."""
    config1 = BrainConfig(
        version="1.0",
        description="Config 1",
        architecture=ArchitectureConfig(
            type="feedforward",
            feedforward=FeedforwardConfig(
                hidden_layers=[128],
                activation="relu",
                dropout=0.0,
                layer_norm=True,
            ),
        ),
        optimizer=OptimizerConfig(
            type="adam",
            learning_rate=0.001,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_eps=1e-8,
            weight_decay=0.0,
        ),
        loss=LossConfig(type="mse"),
        q_learning=QLearningConfig(
            gamma=0.99,
            target_update_frequency=100,
            use_double_dqn=False,
        ),
    )

    config2 = BrainConfig(
        version="1.0",
        description="Config 2",
        architecture=ArchitectureConfig(
            type="feedforward",
            feedforward=FeedforwardConfig(
                hidden_layers=[256],  # Different!
                activation="relu",
                dropout=0.0,
                layer_norm=True,
            ),
        ),
        optimizer=OptimizerConfig(
            type="adam",
            learning_rate=0.001,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_eps=1e-8,
            weight_decay=0.0,
        ),
        loss=LossConfig(type="mse"),
        q_learning=QLearningConfig(
            gamma=0.99,
            target_update_frequency=100,
            use_double_dqn=False,
        ),
    )

    hash1 = compute_brain_hash(config1)
    hash2 = compute_brain_hash(config2)

    assert hash1 != hash2


def test_optimizer_config_adam_requires_adam_params():
    """OptimizerConfig type=adam requires adam_beta1, adam_beta2, adam_eps."""
    with pytest.raises(ValidationError) as exc_info:
        OptimizerConfig(
            type="adam",
            learning_rate=0.001,
            weight_decay=0.0,
            # Missing adam_beta1, adam_beta2, adam_eps!
        )
    error_str = str(exc_info.value)
    assert "adam_beta1" in error_str or "adam parameters required" in error_str.lower()


def test_optimizer_config_sgd_requires_sgd_params():
    """OptimizerConfig type=sgd requires sgd_momentum and sgd_nesterov."""
    with pytest.raises(ValidationError) as exc_info:
        OptimizerConfig(
            type="sgd",
            learning_rate=0.01,
            weight_decay=0.0,
            # Missing sgd_momentum and sgd_nesterov!
        )
    error_str = str(exc_info.value)
    assert "sgd_momentum" in error_str or "sgd parameters required" in error_str.lower()


def test_optimizer_config_rmsprop_requires_rmsprop_params():
    """OptimizerConfig type=rmsprop requires rmsprop_alpha and rmsprop_eps."""
    with pytest.raises(ValidationError) as exc_info:
        OptimizerConfig(
            type="rmsprop",
            learning_rate=0.001,
            weight_decay=0.0,
            # Missing rmsprop_alpha and rmsprop_eps!
        )
    error_str = str(exc_info.value)
    assert "rmsprop_alpha" in error_str or "rmsprop parameters required" in error_str.lower()


# TASK-005 Phase 2: Recurrent network configuration tests


def test_cnn_encoder_config_valid():
    """CNNEncoderConfig accepts valid CNN parameters."""
    from townlet.agent.brain_config import CNNEncoderConfig

    config = CNNEncoderConfig(
        channels=[16, 32],
        kernel_sizes=[3, 3],
        strides=[1, 1],
        padding=[1, 1],
        activation="relu",
    )
    assert config.channels == [16, 32]
    assert config.kernel_sizes == [3, 3]


def test_cnn_encoder_config_rejects_mismatched_lengths():
    """CNNEncoderConfig requires all lists to have same length."""
    from townlet.agent.brain_config import CNNEncoderConfig

    with pytest.raises(ValidationError) as exc_info:
        CNNEncoderConfig(
            channels=[16, 32],
            kernel_sizes=[3],  # Wrong length!
            strides=[1, 1],
            padding=[1, 1],
            activation="relu",
        )
    assert "same length" in str(exc_info.value).lower()


def test_mlp_encoder_config_valid():
    """MLPEncoderConfig accepts valid MLP parameters."""
    from townlet.agent.brain_config import MLPEncoderConfig

    config = MLPEncoderConfig(
        hidden_sizes=[32],
        activation="relu",
    )
    assert config.hidden_sizes == [32]


def test_lstm_config_valid():
    """LSTMConfig accepts valid LSTM parameters."""
    from townlet.agent.brain_config import LSTMConfig

    config = LSTMConfig(
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
    )
    assert config.hidden_size == 256
    assert config.num_layers == 1


def test_lstm_config_rejects_zero_hidden_size():
    """LSTMConfig rejects hidden_size=0."""
    from townlet.agent.brain_config import LSTMConfig

    with pytest.raises(ValidationError) as exc_info:
        LSTMConfig(
            hidden_size=0,
            num_layers=1,
            dropout=0.0,
        )
    assert "hidden_size" in str(exc_info.value)


def test_recurrent_config_valid():
    """RecurrentConfig accepts complete recurrent architecture."""
    from townlet.agent.brain_config import (
        CNNEncoderConfig,
        LSTMConfig,
        MLPEncoderConfig,
        RecurrentConfig,
    )

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
    assert config.lstm.hidden_size == 256
