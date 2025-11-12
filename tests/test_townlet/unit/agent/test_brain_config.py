"""Tests for brain configuration DTOs."""

import pytest
from pydantic import ValidationError

from townlet.agent.brain_config import FeedforwardConfig


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
