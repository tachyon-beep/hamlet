"""Factory for building substrate instances from configuration."""

import torch

from townlet.substrate.aspatial import AspatialSubstrate
from townlet.substrate.base import SpatialSubstrate
from townlet.substrate.config import SubstrateConfig
from townlet.substrate.continuous import Continuous1DSubstrate, Continuous2DSubstrate, Continuous3DSubstrate
from townlet.substrate.continuousnd import ContinuousNDSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate
from townlet.substrate.gridnd import GridNDSubstrate


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
            device: PyTorch device (cuda/cpu) - NOTE: Device is specified per-operation
                   (e.g., initialize_positions), not at substrate construction time.
                   Substrates are device-agnostic; they create tensors on the device
                   specified when calling methods. This parameter is reserved for
                   future use.

        Returns:
            Concrete SpatialSubstrate implementation

        Raises:
            ValueError: If substrate type is unknown

        Example:
            >>> config = load_substrate_config(Path("substrate.yaml"))
            >>> substrate = SubstrateFactory.build(config, torch.device("cuda"))
            >>> # Device specified here, not at construction:
            >>> positions = substrate.initialize_positions(num_agents=100, device=device)
        """
        if config.type == "grid":
            assert config.grid is not None  # Validated by pydantic

            if config.grid.topology == "square":
                return Grid2DSubstrate(
                    width=config.grid.width,
                    height=config.grid.height,
                    boundary=config.grid.boundary,
                    distance_metric=config.grid.distance_metric,
                    observation_encoding=config.grid.observation_encoding,  # NEW: Phase 5C
                    topology=config.grid.topology,  # NEW: Pass topology from config
                )
            elif config.grid.topology == "cubic":
                if config.grid.depth is None:
                    raise ValueError("Cubic topology requires 'depth' parameter")
                return Grid3DSubstrate(
                    width=config.grid.width,
                    height=config.grid.height,
                    depth=config.grid.depth,
                    boundary=config.grid.boundary,
                    distance_metric=config.grid.distance_metric,
                    observation_encoding=config.grid.observation_encoding,  # NEW: Phase 5C
                    topology=config.grid.topology,  # NEW: Pass topology from config
                )
            else:
                raise ValueError(f"Unknown grid topology: {config.grid.topology}")

        elif config.type == "continuous":
            assert config.continuous is not None  # Validated by pydantic

            if config.continuous.dimensions == 1:
                (min_x, max_x) = config.continuous.bounds[0]
                return Continuous1DSubstrate(
                    min_x=min_x,
                    max_x=max_x,
                    boundary=config.continuous.boundary,
                    movement_delta=config.continuous.movement_delta,
                    interaction_radius=config.continuous.interaction_radius,
                    distance_metric=config.continuous.distance_metric,
                    observation_encoding=config.continuous.observation_encoding,  # NEW: Phase 5C
                )

            elif config.continuous.dimensions == 2:
                (min_x, max_x), (min_y, max_y) = config.continuous.bounds
                return Continuous2DSubstrate(
                    min_x=min_x,
                    max_x=max_x,
                    min_y=min_y,
                    max_y=max_y,
                    boundary=config.continuous.boundary,
                    movement_delta=config.continuous.movement_delta,
                    interaction_radius=config.continuous.interaction_radius,
                    distance_metric=config.continuous.distance_metric,
                    observation_encoding=config.continuous.observation_encoding,  # NEW: Phase 5C
                )

            elif config.continuous.dimensions == 3:
                (min_x, max_x), (min_y, max_y), (min_z, max_z) = config.continuous.bounds
                return Continuous3DSubstrate(
                    min_x=min_x,
                    max_x=max_x,
                    min_y=min_y,
                    max_y=max_y,
                    min_z=min_z,
                    max_z=max_z,
                    boundary=config.continuous.boundary,
                    movement_delta=config.continuous.movement_delta,
                    interaction_radius=config.continuous.interaction_radius,
                    distance_metric=config.continuous.distance_metric,
                    observation_encoding=config.continuous.observation_encoding,  # NEW: Phase 5C
                )
            else:
                raise ValueError(f"Unsupported continuous dimensions: {config.continuous.dimensions}")

        elif config.type == "gridnd":
            assert config.gridnd is not None  # Validated by pydantic

            return GridNDSubstrate(
                dimension_sizes=config.gridnd.dimension_sizes,
                boundary=config.gridnd.boundary,
                distance_metric=config.gridnd.distance_metric,
                observation_encoding=config.gridnd.observation_encoding,
                topology=config.gridnd.topology,  # NEW: Pass topology from config
            )

        elif config.type == "continuousnd":
            assert config.continuous is not None  # Validated by pydantic

            return ContinuousNDSubstrate(
                bounds=config.continuous.bounds,
                boundary=config.continuous.boundary,
                movement_delta=config.continuous.movement_delta,
                interaction_radius=config.continuous.interaction_radius,
                distance_metric=config.continuous.distance_metric,
                observation_encoding=config.continuous.observation_encoding,
            )

        elif config.type == "aspatial":
            assert config.aspatial is not None  # Validated by pydantic

            return AspatialSubstrate()

        else:
            raise ValueError(f"Unknown substrate type: {config.type}")
