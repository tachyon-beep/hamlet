# TASK-000: Configurable Spatial Substrates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Abstract the hardcoded 2D square grid substrate into a configurable system supporting 2D/3D grids, hexagonal grids, graph topologies, and aspatial (pure state machine) universes.

**Architecture:** Create abstract `SpatialSubstrate` interface with concrete implementations (Grid2D, Grid3D, Aspatial), integrate with environment/observation/network layers, migrate all configs to `substrate.yaml` schema.

**Tech Stack:** Python 3.11+, Pydantic 2.x, PyTorch, YAML, Abstract Base Classes

**Research Findings Summary:**

- Spatial substrate hardcoded in ~15 core files
- Position tensors always `[num_agents, 2]` shape
- Manhattan distance used in 4 locations
- Observation dim depends on `grid_size²`
- Frontend assumes 2D SVG rendering
- Estimated effort: 15-22 hours

**Key Insight:** Meters (bars) are the true universe - spatial substrate is just an optional overlay for positioning and navigation.

---

## Phase 2: Substrate Configuration Schema

### Task 2.1: Create Substrate Config Pydantic Schema

**Files:**

- Create: `src/townlet/substrate/config.py`

**Step 1: Write test for substrate config schema**

Create: `tests/test_townlet/unit/test_substrate_config.py`

```python
"""Test substrate configuration schema."""
import pytest
from pathlib import Path
from townlet.substrate.config import (
    Grid2DSubstrateConfig,
    AspatialSubstrateConfig,
    SubstrateConfig,
    load_substrate_config,
)


def test_grid2d_config_valid():
    """Valid Grid2D config should parse successfully."""
    config_data = {
        "topology": "square",
        "width": 8,
        "height": 8,
        "boundary": "clamp",
        "distance_metric": "manhattan",
    }

    config = Grid2DSubstrateConfig(**config_data)

    assert config.width == 8
    assert config.height == 8
    assert config.boundary == "clamp"


def test_grid2d_config_invalid_dimensions():
    """Grid2D config with invalid dimensions should fail."""
    config_data = {
        "topology": "square",
        "width": 0,  # Invalid!
        "height": 8,
        "boundary": "clamp",
        "distance_metric": "manhattan",
    }

    with pytest.raises(ValueError, match="greater than 0"):
        Grid2DSubstrateConfig(**config_data)


def test_aspatial_config_valid():
    """Valid aspatial config should parse successfully."""
    config_data = {"enabled": True}

    config = AspatialSubstrateConfig(**config_data)

    assert config.enabled is True


def test_substrate_config_grid2d():
    """SubstrateConfig with type='grid' should require grid config."""
    config_data = {
        "version": "1.0",
        "description": "Test grid substrate",
        "type": "grid",
        "grid": {
            "topology": "square",
            "width": 8,
            "height": 8,
            "boundary": "clamp",
            "distance_metric": "manhattan",
        },
    }

    config = SubstrateConfig(**config_data)

    assert config.type == "grid"
    assert config.grid is not None
    assert config.grid.width == 8


def test_substrate_config_missing_grid():
    """SubstrateConfig with type='grid' but missing grid config should fail."""
    config_data = {
        "version": "1.0",
        "description": "Test",
        "type": "grid",
        # Missing grid config!
    }

    with pytest.raises(ValueError, match="grid config missing"):
        SubstrateConfig(**config_data)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_config.py::test_grid2d_config_valid -v
```

Expected: FAIL (module does not exist)

**Step 3: Implement substrate config schema**

Create: `src/townlet/substrate/config.py`

```python
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
from pydantic import BaseModel, Field, field_validator, model_validator


class Grid2DSubstrateConfig(BaseModel):
    """Configuration for 2D square grid substrate.

    No-Defaults Principle: All fields required (no implicit defaults).
    """

    topology: Literal["square"] = Field(description="Grid topology (must be 'square' for 2D)")
    width: int = Field(gt=0, description="Grid width (number of columns)")
    height: int = Field(gt=0, description="Grid height (number of rows)")
    boundary: Literal["clamp", "wrap", "bounce"] = Field(
        description="Boundary handling mode"
    )
    distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = Field(
        description="Distance metric for spatial queries"
    )


class AspatialSubstrateConfig(BaseModel):
    """Configuration for aspatial substrate (no positioning).

    This is the simplest config - just enable flag.
    """

    enabled: bool = Field(
        default=True,
        description="Enable aspatial mode (no spatial positioning)",
    )


class SubstrateConfig(BaseModel):
    """Complete substrate configuration.

    Exactly one substrate type must be specified via the 'type' field,
    and the corresponding config must be provided.

    No-Defaults Principle: version, description, type are all required.
    """

    version: str = Field(description="Config version (e.g., '1.0')")
    description: str = Field(description="Human-readable description")
    type: Literal["grid", "aspatial"] = Field(
        description="Substrate type selection"
    )

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
            raise ValueError(
                "type='grid' requires grid configuration. "
                "Add grid: { topology: 'square', width: 8, height: 8, ... }"
            )

        if self.type == "aspatial" and self.aspatial is None:
            raise ValueError(
                "type='aspatial' requires aspatial configuration. "
                "Add aspatial: { enabled: true }"
            )

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
```

**Step 4: Run tests to verify implementation**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_config.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/townlet/substrate/config.py tests/test_townlet/unit/test_substrate_config.py
git commit -m "feat: add Pydantic schema for substrate configuration

Defines SubstrateConfig DTOs for substrate.yaml validation:
- Grid2DSubstrateConfig: width, height, boundary, distance_metric
- AspatialSubstrateConfig: enabled flag
- SubstrateConfig: Top-level with type selection

Enforces:
- No-defaults principle (all fields required)
- Type validation (grid config for type='grid')
- Conceptual agnosticism (allows grid, aspatial, future 3D/hex)

Part of TASK-000 (Configurable Spatial Substrates)."
```

---

### Task 2.2: Create Substrate Factory

**Files:**

- Create: `src/townlet/substrate/factory.py`
- Modify: `src/townlet/substrate/__init__.py`

**Step 1: Write test for substrate factory**

Modify: `tests/test_townlet/unit/test_substrate_config.py`

Add to end of file:

```python
from townlet.substrate.factory import SubstrateFactory
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.aspatial import AspatialSubstrate


def test_factory_build_grid2d():
    """Factory should build Grid2DSubstrate from config."""
    config_data = {
        "version": "1.0",
        "description": "Test grid",
        "type": "grid",
        "grid": {
            "topology": "square",
            "width": 8,
            "height": 8,
            "boundary": "clamp",
            "distance_metric": "manhattan",
        },
    }

    config = SubstrateConfig(**config_data)
    substrate = SubstrateFactory.build(config, device=torch.device("cpu"))

    assert isinstance(substrate, Grid2DSubstrate)
    assert substrate.width == 8
    assert substrate.height == 8


def test_factory_build_aspatial():
    """Factory should build AspatialSubstrate from config."""
    config_data = {
        "version": "1.0",
        "description": "Test aspatial",
        "type": "aspatial",
        "aspatial": {"enabled": True},
    }

    config = SubstrateConfig(**config_data)
    substrate = SubstrateFactory.build(config, device=torch.device("cpu"))

    assert isinstance(substrate, AspatialSubstrate)
    assert substrate.position_dim == 0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_config.py::test_factory_build_grid2d -v
```

Expected: FAIL (SubstrateFactory not defined)

**Step 3: Implement substrate factory**

Create: `src/townlet/substrate/factory.py`

```python
"""Factory for building substrate instances from configuration."""

import torch
from townlet.substrate.base import SpatialSubstrate
from townlet.substrate.config import SubstrateConfig
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.aspatial import AspatialSubstrate


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
```

**Step 4: Update **init**.py**

Modify: `src/townlet/substrate/__init__.py`

```python
"""Spatial substrate abstractions for UNIVERSE_AS_CODE."""

from townlet.substrate.base import SpatialSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.aspatial import AspatialSubstrate
from townlet.substrate.config import SubstrateConfig, load_substrate_config
from townlet.substrate.factory import SubstrateFactory

__all__ = [
    "SpatialSubstrate",
    "Grid2DSubstrate",
    "AspatialSubstrate",
    "SubstrateConfig",
    "load_substrate_config",
    "SubstrateFactory",
]
```

**Step 5: Run tests to verify implementation**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_config.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/townlet/substrate/factory.py src/townlet/substrate/__init__.py tests/test_townlet/unit/test_substrate_config.py
git commit -m "feat: add SubstrateFactory for building substrates from config

Created factory that converts SubstrateConfig (Pydantic DTO) into concrete
SpatialSubstrate instances (Grid2DSubstrate, AspatialSubstrate).

Usage:
  config = load_substrate_config(path)
  substrate = SubstrateFactory.build(config, device)

Part of TASK-000 (Configurable Spatial Substrates)."
```

---
