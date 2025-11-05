"""Spatial substrate abstractions for UNIVERSE_AS_CODE.

The spatial substrate defines the coordinate system, topology, and distance metric
for agent positioning and navigation. This is an OPTIONAL component - aspatial
universes (pure resource management) are perfectly valid.
"""

from townlet.substrate.base import SpatialSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate

__all__ = ["SpatialSubstrate", "Grid2DSubstrate"]
