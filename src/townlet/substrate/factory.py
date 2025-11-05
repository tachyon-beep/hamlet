"""Factory for building substrate instances from configuration."""

import torch

from townlet.substrate.aspatial import AspatialSubstrate
from townlet.substrate.base import SpatialSubstrate
from townlet.substrate.config import SubstrateConfig
from townlet.substrate.grid2d import Grid2DSubstrate


class SubstrateFactory:
    """Factory for building substrate instances from configuration.

    Converts SubstrateConfig (Pydantic DTO) into concrete SpatialSubstrate
    implementations (Grid2DSubstrate, AspatialSubstrate, etc.).
    """

    @staticmethod
    def build(config: SubstrateConfig, device: torch.device) -> SpatialSubstrate:
        """Build substrate instance from configuration.

        Args:
            config: Validated substrate configuration
            device: PyTorch device (cuda/cpu) for tensor operations

        Returns:
            Concrete SpatialSubstrate implementation

        Raises:
            ValueError: If substrate type is unknown

        Example:
            >>> config = load_substrate_config(Path("substrate.yaml"))
            >>> substrate = SubstrateFactory.build(config, torch.device("cuda"))
            >>> positions = substrate.initialize_positions(num_agents=100, device=device)
        """
        if config.type == "grid":
            assert config.grid is not None  # Validated by pydantic

            return Grid2DSubstrate(
                width=config.grid.width,
                height=config.grid.height,
                boundary=config.grid.boundary,
                distance_metric=config.grid.distance_metric,
            )

        elif config.type == "aspatial":
            assert config.aspatial is not None  # Validated by pydantic

            return AspatialSubstrate()

        else:
            raise ValueError(f"Unknown substrate type: {config.type}")
