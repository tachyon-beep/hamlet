"""Loss factory for building loss functions from configuration.

Builds PyTorch loss functions from BrainConfig specifications.
"""

import torch.nn as nn

from townlet.agent.brain_config import LossConfig


class LossFactory:
    """Factory for building loss functions from declarative configuration."""

    @staticmethod
    def build(config: LossConfig) -> nn.Module:
        """Build loss function from configuration.

        Args:
            config: Loss function configuration

        Returns:
            PyTorch loss module

        Example:
            >>> config = LossConfig(type="mse")
            >>> loss_fn = LossFactory.build(config)
            >>> isinstance(loss_fn, torch.nn.MSELoss)
            True
        """
        if config.type == "mse":
            return nn.MSELoss()

        elif config.type == "huber":
            return nn.HuberLoss(delta=config.huber_delta)

        elif config.type == "smooth_l1":
            return nn.SmoothL1Loss()

        else:
            raise ValueError(f"Unsupported loss type: {config.type}")
