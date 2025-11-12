"""Tests for optimizer factory."""

import torch
import torch.nn as nn

from townlet.agent.brain_config import OptimizerConfig
from townlet.agent.optimizer_factory import OptimizerFactory


def test_build_adam():
    """OptimizerFactory builds Adam optimizer."""
    config = OptimizerConfig(
        type="adam",
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,
    )

    network = nn.Sequential(nn.Linear(10, 5))
    optimizer = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 0.001
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)


def test_build_sgd():
    """OptimizerFactory builds SGD optimizer."""
    config = OptimizerConfig(
        type="sgd",
        learning_rate=0.01,
        sgd_momentum=0.9,
        sgd_nesterov=True,
        weight_decay=1e-4,
    )

    network = nn.Sequential(nn.Linear(10, 5))
    optimizer = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["momentum"] == 0.9
    assert optimizer.param_groups[0]["nesterov"] is True


def test_build_adamw():
    """OptimizerFactory builds AdamW optimizer."""
    config = OptimizerConfig(
        type="adamw",
        learning_rate=0.0005,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.01,
    )

    network = nn.Sequential(nn.Linear(10, 5))
    optimizer = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["weight_decay"] == 0.01


def test_build_rmsprop():
    """OptimizerFactory builds RMSprop optimizer."""
    config = OptimizerConfig(
        type="rmsprop",
        learning_rate=0.001,
        rmsprop_alpha=0.99,
        rmsprop_eps=1e-8,
        weight_decay=0.0,
    )

    network = nn.Sequential(nn.Linear(10, 5))
    optimizer = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, torch.optim.RMSprop)
    assert optimizer.param_groups[0]["lr"] == 0.001
    assert optimizer.param_groups[0]["alpha"] == 0.99
