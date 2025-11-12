"""Optimizer factory for building optimizers from configuration.

Builds PyTorch optimizers and learning rate schedulers from BrainConfig specifications.
"""

from collections.abc import Iterator

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LRScheduler,
    StepLR,
)

from townlet.agent.brain_config import OptimizerConfig, ScheduleConfig


class OptimizerFactory:
    """Factory for building optimizers from declarative configuration."""

    @staticmethod
    def build(
        config: OptimizerConfig,
        parameters: Iterator[nn.Parameter],
    ) -> tuple[optim.Optimizer, LRScheduler | None]:
        """Build optimizer and optional scheduler from configuration.

        Args:
            config: Optimizer configuration (includes schedule)
            parameters: Network parameters to optimize

        Returns:
            Tuple of (optimizer, scheduler or None)

        Example:
            >>> config = OptimizerConfig(
            ...     type="adam",
            ...     learning_rate=0.001,
            ...     adam_beta1=0.9,
            ...     adam_beta2=0.999,
            ...     adam_eps=1e-8,
            ...     weight_decay=0.0,
            ...     schedule=ScheduleConfig(type="step_decay", step_size=100, gamma=0.1),
            ... )
            >>> optimizer, scheduler = OptimizerFactory.build(config, network.parameters())
            >>> isinstance(optimizer, torch.optim.Adam)
            True
            >>> isinstance(scheduler, StepLR)
            True
        """
        # Build optimizer (store in variable instead of returning immediately)
        if config.type == "adam":
            # Type assertions: validator ensures these are not None
            assert config.adam_beta1 is not None
            assert config.adam_beta2 is not None
            assert config.adam_eps is not None
            optimizer = optim.Adam(
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
            optimizer = optim.AdamW(
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
            optimizer = optim.SGD(
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
            optimizer = optim.RMSprop(
                parameters,
                lr=config.learning_rate,
                alpha=config.rmsprop_alpha,
                eps=config.rmsprop_eps,
                weight_decay=config.weight_decay,
            )

        else:
            raise ValueError(f"Unsupported optimizer type: {config.type}")

        # Build scheduler from schedule config
        scheduler = OptimizerFactory._build_scheduler(config.schedule, optimizer)

        return optimizer, scheduler

    @staticmethod
    def _build_scheduler(
        schedule_config: ScheduleConfig,
        optimizer: optim.Optimizer,
    ) -> LRScheduler | None:
        """Build learning rate scheduler from configuration.

        Args:
            schedule_config: Schedule configuration
            optimizer: PyTorch optimizer

        Returns:
            Scheduler or None (for constant)

        Example:
            >>> optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
            >>> schedule = ScheduleConfig(type="step_decay", step_size=100, gamma=0.1)
            >>> scheduler = OptimizerFactory._build_scheduler(schedule, optimizer)
            >>> isinstance(scheduler, StepLR)
            True
        """
        if schedule_config.type == "constant":
            return None
        elif schedule_config.type == "step_decay":
            # Type assertions: validator ensures these are not None
            assert schedule_config.step_size is not None
            assert schedule_config.gamma is not None
            return StepLR(
                optimizer,
                step_size=schedule_config.step_size,
                gamma=schedule_config.gamma,
            )
        elif schedule_config.type == "cosine":
            # Type assertions: validator ensures these are not None
            assert schedule_config.t_max is not None
            assert schedule_config.eta_min is not None
            return CosineAnnealingLR(
                optimizer,
                T_max=schedule_config.t_max,
                eta_min=schedule_config.eta_min,
            )
        elif schedule_config.type == "exponential":
            # Type assertions: validator ensures gamma is not None
            assert schedule_config.gamma is not None
            return ExponentialLR(
                optimizer,
                gamma=schedule_config.gamma,
            )
        else:
            raise ValueError(f"Unsupported schedule type: {schedule_config.type}")
