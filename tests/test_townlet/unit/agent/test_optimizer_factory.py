"""Tests for optimizer factory."""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR

from townlet.agent.brain_config import OptimizerConfig, ScheduleConfig
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
        schedule=ScheduleConfig(type="constant"),
    )

    network = nn.Sequential(nn.Linear(10, 5))
    optimizer, scheduler = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 0.001
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    assert scheduler is None  # constant schedule returns None


def test_build_sgd():
    """OptimizerFactory builds SGD optimizer."""
    config = OptimizerConfig(
        type="sgd",
        learning_rate=0.01,
        sgd_momentum=0.9,
        sgd_nesterov=True,
        weight_decay=1e-4,
        schedule=ScheduleConfig(type="constant"),
    )

    network = nn.Sequential(nn.Linear(10, 5))
    optimizer, scheduler = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["momentum"] == 0.9
    assert optimizer.param_groups[0]["nesterov"] is True
    assert scheduler is None


def test_build_adamw():
    """OptimizerFactory builds AdamW optimizer."""
    config = OptimizerConfig(
        type="adamw",
        learning_rate=0.0005,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.01,
        schedule=ScheduleConfig(type="constant"),
    )

    network = nn.Sequential(nn.Linear(10, 5))
    optimizer, scheduler = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert scheduler is None


def test_build_rmsprop():
    """OptimizerFactory builds RMSprop optimizer."""
    config = OptimizerConfig(
        type="rmsprop",
        learning_rate=0.001,
        rmsprop_alpha=0.99,
        rmsprop_eps=1e-8,
        weight_decay=0.0,
        schedule=ScheduleConfig(type="constant"),
    )

    network = nn.Sequential(nn.Linear(10, 5))
    optimizer, scheduler = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, torch.optim.RMSprop)
    assert optimizer.param_groups[0]["lr"] == 0.001
    assert optimizer.param_groups[0]["alpha"] == 0.99
    assert scheduler is None


def test_build_with_constant_schedule():
    """OptimizerFactory.build returns (optimizer, None) for constant schedule."""
    network = nn.Linear(10, 5)
    config = OptimizerConfig(
        type="adam",
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,
        schedule=ScheduleConfig(type="constant"),
    )

    optimizer, scheduler = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, torch.optim.Adam)
    assert scheduler is None


def test_build_with_step_decay_schedule():
    """OptimizerFactory.build returns (optimizer, StepLR) for step_decay."""
    network = nn.Linear(10, 5)
    config = OptimizerConfig(
        type="adam",
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,
        schedule=ScheduleConfig(type="step_decay", step_size=100, gamma=0.1),
    )

    optimizer, scheduler = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(scheduler, StepLR)
    assert scheduler.step_size == 100
    assert scheduler.gamma == 0.1


def test_build_with_cosine_schedule():
    """OptimizerFactory.build returns (optimizer, CosineAnnealingLR) for cosine."""
    network = nn.Linear(10, 5)
    config = OptimizerConfig(
        type="adam",
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,
        schedule=ScheduleConfig(type="cosine", t_max=1000, eta_min=0.00001),
    )

    optimizer, scheduler = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(scheduler, CosineAnnealingLR)
    assert scheduler.T_max == 1000
    assert scheduler.eta_min == 0.00001


def test_build_with_exponential_schedule():
    """OptimizerFactory.build returns (optimizer, ExponentialLR) for exponential."""
    network = nn.Linear(10, 5)
    config = OptimizerConfig(
        type="adam",
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,
        schedule=ScheduleConfig(type="exponential", gamma=0.9999),
    )

    optimizer, scheduler = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(scheduler, ExponentialLR)
    assert scheduler.gamma == 0.9999
