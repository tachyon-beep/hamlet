"""Tests for brain configuration DTOs."""

import pytest
from pydantic import ValidationError

from townlet.agent.brain_config import FeedforwardConfig, LossConfig, OptimizerConfig


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
