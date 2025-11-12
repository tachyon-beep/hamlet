"""Network factory for building Q-networks from configuration.

Builds neural networks from BrainConfig specifications.
Forward-compatible with future SDA (Software Defined Agent) architecture.
"""

import torch.nn as nn

from townlet.agent.brain_config import FeedforwardConfig, RecurrentConfig
from townlet.agent.networks import RecurrentSpatialQNetwork


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
    def build_recurrent(
        config: RecurrentConfig,
        action_dim: int,
        window_size: int,
        position_dim: int,
        num_meters: int,
        num_affordance_types: int,
    ) -> RecurrentSpatialQNetwork:
        """Build recurrent LSTM Q-network from configuration.

        Args:
            config: Recurrent architecture configuration
            action_dim: Number of actions
            window_size: Vision window size (5 for 5×5)
            position_dim: Position dimensionality (2 for Grid2D, 3 for Grid3D, 0 for Aspatial)
            num_meters: Number of meter values
            num_affordance_types: Number of affordance types

        Returns:
            RecurrentSpatialQNetwork

        Note:
            This builds a RecurrentSpatialQNetwork with configurable dimensions
            instead of hardcoded values. The network structure matches the original
            RecurrentSpatialQNetwork but dimensions come from config.

            Currently, the config's vision_encoder, position_encoder, meter_encoder,
            affordance_encoder, and q_head parameters are not used because
            RecurrentSpatialQNetwork has a fixed internal architecture. This is
            acceptable for Phase 2 (TASK-005). Future phases may make the network
            architecture fully configurable.

        Example:
            >>> config = RecurrentConfig(...)
            >>> network = NetworkFactory.build_recurrent(
            ...     config=config,
            ...     action_dim=8,
            ...     window_size=5,
            ...     position_dim=2,
            ...     num_meters=8,
            ...     num_affordance_types=14,
            ... )
        """
        # Extract LSTM hidden size from config
        lstm_hidden_size = config.lstm.hidden_size

        # Create RecurrentSpatialQNetwork with config-driven LSTM dimension
        # Note: The existing RecurrentSpatialQNetwork class has hardcoded encoder
        # architectures (CNN, MLPs). For Phase 2, we only make LSTM hidden_size
        # configurable via the hidden_dim parameter.
        # Future phases may make the entire architecture fully configurable.
        network = RecurrentSpatialQNetwork(
            action_dim=action_dim,
            window_size=window_size,
            position_dim=position_dim,
            num_meters=num_meters,
            num_affordance_types=num_affordance_types,
            enable_temporal_features=False,  # Will be determined by environment
            hidden_dim=lstm_hidden_size,  # From config instead of hardcoded!
        )

        return network

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
