"""Spatial substrate abstractions for UNIVERSE_AS_CODE.

The spatial substrate defines the coordinate system, topology, and distance metric
for agent positioning and navigation. This is an OPTIONAL component - aspatial
universes (pure resource management) are perfectly valid.
"""

from townlet.substrate.aspatial import AspatialSubstrate
from townlet.substrate.base import SpatialSubstrate
from townlet.substrate.config import SubstrateConfig, load_substrate_config
from townlet.substrate.factory import SubstrateFactory
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate

__all__ = [
    "SpatialSubstrate",
    "Grid2DSubstrate",
    "Grid3DSubstrate",
    "AspatialSubstrate",
    "SubstrateConfig",
    "load_substrate_config",
    "SubstrateFactory",
]
