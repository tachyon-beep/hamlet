"""Network factory for building Q-networks from configuration.

Builds neural networks from BrainConfig specifications.
Forward-compatible with future SDA (Software Defined Agent) architecture.
"""

import torch.nn as nn

from townlet.agent.brain_config import FeedforwardConfig


class NetworkFactory:
    """Factory for building Q-networks from declarative configuration."""

    @staticmethod
    def build_feedforward(
        config: FeedforwardConfig,
        obs_dim: int,
        action_dim: int,
    ) -> nn.Module:
        """Build feedforward MLP Q-network from configuration.

        Args:
            config: Feedforward architecture configuration
            obs_dim: Observation dimension
            action_dim: Action dimension

        Returns:
            PyTorch module (feedforward Q-network)

        Example:
            >>> config = FeedforwardConfig(
            ...     hidden_layers=[256, 128],
            ...     activation="relu",
            ...     dropout=0.0,
            ...     layer_norm=True,
            ... )
            >>> network = NetworkFactory.build_feedforward(config, 29, 8)
            >>> network(torch.randn(4, 29)).shape
            torch.Size([4, 8])
        """
        layers: list[nn.Module] = []
        in_features = obs_dim

        # Build hidden layers
        for hidden_size in config.hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))

            if config.layer_norm:
                layers.append(nn.LayerNorm(hidden_size))

            layers.append(NetworkFactory._get_activation(config.activation))

            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))

            in_features = hidden_size

        # Output layer (Q-values)
        layers.append(nn.Linear(in_features, action_dim))

        return nn.Sequential(*layers)

    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        """Get activation function module from config string.

        Args:
            activation: Activation function name

        Returns:
            PyTorch activation module
        """
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),  # Swish = SiLU in PyTorch
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        return activations[activation]
