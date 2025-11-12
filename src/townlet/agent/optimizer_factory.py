"""Optimizer factory for building optimizers from configuration.

Builds PyTorch optimizers from BrainConfig specifications.
Forward-compatible with future learning rate schedulers (Phase 2).
"""

from collections.abc import Iterator

import torch.nn as nn
import torch.optim as optim

from townlet.agent.brain_config import OptimizerConfig


class OptimizerFactory:
    """Factory for building optimizers from declarative configuration."""

    @staticmethod
    def build(
        config: OptimizerConfig,
        parameters: Iterator[nn.Parameter],
    ) -> optim.Optimizer:
        """Build optimizer from configuration.

        Args:
            config: Optimizer configuration
            parameters: Network parameters to optimize

        Returns:
            PyTorch optimizer

        Example:
            >>> config = OptimizerConfig(
            ...     type="adam",
            ...     learning_rate=0.001,
            ...     adam_beta1=0.9,
            ...     adam_beta2=0.999,
            ...     adam_eps=1e-8,
            ...     weight_decay=0.0,
            ... )
            >>> optimizer = OptimizerFactory.build(config, network.parameters())
            >>> isinstance(optimizer, torch.optim.Adam)
            True
        """
        if config.type == "adam":
            # Type assertions: validator ensures these are not None
            assert config.adam_beta1 is not None
            assert config.adam_beta2 is not None
            assert config.adam_eps is not None
            return optim.Adam(
                parameters,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
                weight_decay=config.weight_decay,
            )

        elif config.type == "adamw":
            # Type assertions: validator ensures these are not None
            assert config.adam_beta1 is not None
            assert config.adam_beta2 is not None
            assert config.adam_eps is not None
            return optim.AdamW(
                parameters,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
                weight_decay=config.weight_decay,
            )

        elif config.type == "sgd":
            # Type assertions: validator ensures these are not None
            assert config.sgd_momentum is not None
            assert config.sgd_nesterov is not None
            return optim.SGD(
                parameters,
                lr=config.learning_rate,
                momentum=config.sgd_momentum,
                nesterov=config.sgd_nesterov,
                weight_decay=config.weight_decay,
            )

        elif config.type == "rmsprop":
            # Type assertions: validator ensures these are not None
            assert config.rmsprop_alpha is not None
            assert config.rmsprop_eps is not None
            return optim.RMSprop(
                parameters,
                lr=config.learning_rate,
                alpha=config.rmsprop_alpha,
                eps=config.rmsprop_eps,
                weight_decay=config.weight_decay,
            )

        else:
            raise ValueError(f"Unsupported optimizer type: {config.type}")
