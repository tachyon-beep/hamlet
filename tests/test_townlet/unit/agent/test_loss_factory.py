"""Tests for loss factory."""

import torch
import torch.nn as nn

from townlet.agent.brain_config import LossConfig
from townlet.agent.loss_factory import LossFactory


def test_build_mse():
    """LossFactory builds MSELoss."""
    config = LossConfig(type="mse")
    loss_fn = LossFactory.build(config)

    assert isinstance(loss_fn, nn.MSELoss)

    # Test forward pass
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])
    loss = loss_fn(pred, target)
    assert loss.item() > 0.0


def test_build_huber():
    """LossFactory builds HuberLoss with delta."""
    config = LossConfig(type="huber", huber_delta=1.5)
    loss_fn = LossFactory.build(config)

    assert isinstance(loss_fn, nn.HuberLoss)

    # Test forward pass
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])
    loss = loss_fn(pred, target)
    assert loss.item() > 0.0


def test_build_smooth_l1():
    """LossFactory builds SmoothL1Loss."""
    config = LossConfig(type="smooth_l1")
    loss_fn = LossFactory.build(config)

    assert isinstance(loss_fn, nn.SmoothL1Loss)

    # Test forward pass
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])
    loss = loss_fn(pred, target)
    assert loss.item() > 0.0


def test_huber_delta_affects_loss():
    """LossFactory uses huber_delta parameter."""
    config_small = LossConfig(type="huber", huber_delta=0.1)
    config_large = LossConfig(type="huber", huber_delta=10.0)

    loss_small = LossFactory.build(config_small)
    loss_large = LossFactory.build(config_large)

    # Large errors should be affected by delta
    pred = torch.tensor([0.0])
    target = torch.tensor([5.0])

    loss_val_small = loss_small(pred, target).item()
    loss_val_large = loss_large(pred, target).item()

    # Smaller delta should produce smaller loss for large errors
    assert loss_val_small < loss_val_large
