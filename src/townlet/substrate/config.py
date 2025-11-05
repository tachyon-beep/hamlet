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


class Grid2DSubstrateConfig(BaseModel):
    """Configuration for 2D square grid substrate.

    No-Defaults Principle: All fields required (no implicit defaults).
    """

    topology: Literal["square"] = Field(description="Grid topology (must be 'square' for 2D)")
    width: int = Field(gt=0, description="Grid width (number of columns)")
    height: int = Field(gt=0, description="Grid height (number of rows)")
    boundary: Literal["clamp", "wrap", "bounce", "sticky"] = Field(description="Boundary handling mode")
    distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = Field(description="Distance metric for spatial queries")


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
    grid: Grid2DSubstrateConfig | None = Field(
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
