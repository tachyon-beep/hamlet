"""Substrate configuration schema (Pydantic DTOs).

Defines the YAML schema for substrate.yaml files, enforcing structure and
validating configuration at load time.

Design Principles (from TASK-001):
- No-Defaults Principle: All behavioral parameters must be explicit
- Conceptual Agnosticism: Don't assume 2D, grid-based, or Euclidean
- Structural Enforcement: Validate dimensions, boundary modes, metrics
- Permissive Semantics: Allow 3D, hex, continuous, graph, aspatial
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class GridConfig(BaseModel):
    """Configuration for grid-based substrates (2D square or 3D cubic).

    No-Defaults Principle: All fields required (no implicit defaults).
    """

    topology: Literal["square", "cubic"] = Field(description="Grid topology (square=2D, cubic=3D)")
    width: int = Field(gt=0, description="Grid width (X dimension)")
    height: int = Field(gt=0, description="Grid height (Y dimension)")
    depth: int | None = Field(None, gt=0, description="Grid depth (Z dimension) - required for cubic topology")
    boundary: Literal["clamp", "wrap", "bounce", "sticky"] = Field(description="Boundary handling mode")
    distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = Field(default="manhattan", description="Distance calculation method")

    @model_validator(mode="after")
    def validate_cubic_requires_depth(self) -> "GridConfig":
        """Cubic topology requires depth parameter."""
        if self.topology == "cubic" and self.depth is None:
            raise ValueError(
                "Cubic topology requires 'depth' parameter.\n"
                "Example:\n"
                "  topology: cubic\n"
                "  width: 8\n"
                "  height: 8\n"
                "  depth: 3  # Required for 3D\n"
            )
        if self.topology == "square" and self.depth is not None:
            raise ValueError("Square topology does not use 'depth' parameter. " "Remove 'depth' or use topology: cubic")
        return self


# Backward compatibility alias
Grid2DSubstrateConfig = GridConfig


class AspatialSubstrateConfig(BaseModel):
    """Configuration for aspatial substrate (no positioning).

    This is the simplest config - presence indicates aspatial mode is enabled.
    No additional configuration needed for aspatial substrates.
    """

    pass  # No configuration fields needed


class SubstrateConfig(BaseModel):
    """Complete substrate configuration.

    Exactly one substrate type must be specified via the 'type' field,
    and the corresponding config must be provided.

    No-Defaults Principle: version, description, type are all required.
    """

    version: str = Field(description="Config version (e.g., '1.0')")
    description: str = Field(description="Human-readable description")
    type: Literal["grid", "aspatial"] = Field(description="Substrate type selection")

    # Substrate-specific configs (only one should be populated)
    grid: GridConfig | None = Field(
        None,
        description="Grid substrate configuration (required if type='grid')",
    )
    aspatial: AspatialSubstrateConfig | None = Field(
        None,
        description="Aspatial substrate configuration (required if type='aspatial')",
    )

    @model_validator(mode="after")
    def validate_substrate_type_match(self) -> "SubstrateConfig":
        """Ensure substrate config matches declared type."""
        if self.type == "grid" and self.grid is None:
            raise ValueError("type='grid' requires grid configuration. " "Add grid: { topology: 'square', width: 8, height: 8, ... }")

        if self.type == "aspatial" and self.aspatial is None:
            raise ValueError("type='aspatial' requires aspatial configuration. " "Add aspatial: {}")

        # Ensure only one config is provided
        if self.type == "grid" and self.aspatial is not None:
            raise ValueError("type='grid' should not have aspatial configuration")

        if self.type == "aspatial" and self.grid is not None:
            raise ValueError("type='aspatial' should not have grid configuration")

        return self


def load_substrate_config(config_path: Path) -> SubstrateConfig:
    """Load and validate substrate configuration from YAML.

    Args:
        config_path: Path to substrate.yaml file

    Returns:
        SubstrateConfig: Validated substrate configuration

    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If config validation fails

    Example:
        >>> config = load_substrate_config(Path("configs/L1/substrate.yaml"))
        >>> print(f"Substrate type: {config.type}")
        >>> print(f"Grid size: {config.grid.width}×{config.grid.height}")
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Substrate config not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    try:
        return SubstrateConfig(**data)
    except Exception as e:
        raise ValueError(f"Invalid substrate config at {config_path}: {e}") from e
