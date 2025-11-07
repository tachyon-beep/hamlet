# Topology Metadata Consistency Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix topology metadata inconsistency by making topology a first-class substrate attribute, propagated from configuration through to WebSocket metadata.

**Architecture:** Add `topology` attribute to grid substrate classes (Grid2D/Grid3D/GridND), update factory to pass topology from config, modify WebSocket metadata builder to read topology from substrate instances, and update GridNDConfig schema to include topology field.

**Tech Stack:** Python 3.11, Pydantic (config validation), PyTorch (substrate system), WebSocket (live_inference), pytest (testing)

**Current State:** Topology is hardcoded in `_build_substrate_metadata()` for Grid2D ("square") and Grid3D ("cubic"), but missing for GridND/Continuous/Aspatial. The configured topology in substrate.yaml is read by factory but never stored in substrate instances.

**Target State:** All grid substrates have `.topology` attribute read from config. WebSocket metadata includes topology for grids, omits it for continuous/aspatial. Frontend receives consistent metadata structure.

---

## Task 1: Add Topology to Grid2DSubstrate (TDD)

**Files:**
- Test: `tests/test_townlet/test_substrate/test_grid2d.py`
- Modify: `src/townlet/substrate/grid2d.py:32-63`

**Step 1: Write failing tests for Grid2D topology storage**

Add to `tests/test_townlet/test_substrate/test_grid2d.py`:

```python
def test_grid2d_stores_topology_when_provided():
    """Grid2D should store topology attribute when explicitly provided."""
    substrate = Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="square",
    )
    assert substrate.topology == "square"


def test_grid2d_topology_defaults_to_square():
    """Grid2D topology should default to 'square' if not provided."""
    substrate = Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert substrate.topology == "square"


def test_grid2d_topology_attribute_exists():
    """Grid2D should have topology attribute (not inherited from base)."""
    substrate = Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert hasattr(substrate, "topology")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_townlet/test_substrate/test_grid2d.py::test_grid2d_stores_topology_when_provided -xvs
pytest tests/test_townlet/test_substrate/test_grid2d.py::test_grid2d_topology_defaults_to_square -xvs
pytest tests/test_townlet/test_substrate/test_grid2d.py::test_grid2d_topology_attribute_exists -xvs
```

Expected: `FAILED` - TypeError (unexpected keyword argument 'topology') or AttributeError ('Grid2DSubstrate' object has no attribute 'topology')

**Step 3: Implement topology parameter in Grid2DSubstrate**

Modify `src/townlet/substrate/grid2d.py:32-63`:

```python
def __init__(
    self,
    width: int,
    height: int,
    boundary: Literal["clamp", "wrap", "bounce", "sticky"],
    distance_metric: Literal["manhattan", "euclidean", "chebyshev"],
    observation_encoding: Literal["relative", "scaled", "absolute"] = "relative",
    topology: Literal["square"] = "square",  # NEW: Grid2D is always square topology
):
    """Initialize 2D grid substrate.

    Args:
        width: Grid width (number of columns)
        height: Grid height (number of rows)
        boundary: Boundary mode ("clamp", "wrap", "bounce", "sticky")
        distance_metric: Distance metric ("manhattan", "euclidean", "chebyshev")
        observation_encoding: Position encoding strategy ("relative", "scaled", "absolute")
        topology: Grid topology ("square" for 2D Cartesian grid)
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Grid dimensions must be positive: width={width}, height={height}")

    if boundary not in ("clamp", "wrap", "bounce", "sticky"):
        raise ValueError(f"Unknown boundary mode: {boundary}")

    if distance_metric not in ("manhattan", "euclidean", "chebyshev"):
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    self.width = width
    self.height = height
    self.boundary = boundary
    self.distance_metric = distance_metric
    self.observation_encoding = observation_encoding
    self.topology = topology  # NEW: Store topology
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/test_substrate/test_grid2d.py::test_grid2d_stores_topology_when_provided -xvs
pytest tests/test_townlet/test_substrate/test_grid2d.py::test_grid2d_topology_defaults_to_square -xvs
pytest tests/test_townlet/test_substrate/test_grid2d.py::test_grid2d_topology_attribute_exists -xvs
```

Expected: `PASSED` (3/3 tests)

**Step 5: Run full Grid2D test suite to ensure no regressions**

```bash
pytest tests/test_townlet/test_substrate/test_grid2d.py -v
```

Expected: All tests pass (existing tests should not break since topology has default)

**Step 6: Commit**

```bash
git add src/townlet/substrate/grid2d.py tests/test_townlet/test_substrate/test_grid2d.py
git commit -m "feat(substrate): add topology attribute to Grid2DSubstrate

- Add topology parameter with default 'square'
- Store topology as instance attribute
- Add tests for topology storage and default value
- Grid2D topology is always 'square' (4-connected 2D Cartesian grid)"
```

---

## Task 2: Add Topology to Grid3DSubstrate (TDD)

**Files:**
- Test: `tests/test_townlet/test_substrate/test_grid3d.py`
- Modify: `src/townlet/substrate/grid3d.py:40-72`

**Step 1: Write failing tests for Grid3D topology storage**

Add to `tests/test_townlet/test_substrate/test_grid3d.py`:

```python
def test_grid3d_stores_topology_when_provided():
    """Grid3D should store topology attribute when explicitly provided."""
    substrate = Grid3DSubstrate(
        width=8,
        height=8,
        depth=3,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="cubic",
    )
    assert substrate.topology == "cubic"


def test_grid3d_topology_defaults_to_cubic():
    """Grid3D topology should default to 'cubic' if not provided."""
    substrate = Grid3DSubstrate(
        width=8,
        height=8,
        depth=3,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert substrate.topology == "cubic"


def test_grid3d_topology_attribute_exists():
    """Grid3D should have topology attribute."""
    substrate = Grid3DSubstrate(
        width=8,
        height=8,
        depth=3,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert hasattr(substrate, "topology")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_townlet/test_substrate/test_grid3d.py::test_grid3d_stores_topology_when_provided -xvs
pytest tests/test_townlet/test_substrate/test_grid3d.py::test_grid3d_topology_defaults_to_cubic -xvs
pytest tests/test_townlet/test_substrate/test_grid3d.py::test_grid3d_topology_attribute_exists -xvs
```

Expected: `FAILED` - TypeError or AttributeError

**Step 3: Implement topology parameter in Grid3DSubstrate**

Modify `src/townlet/substrate/grid3d.py:40-72`:

```python
def __init__(
    self,
    width: int,
    height: int,
    depth: int,
    boundary: Literal["clamp", "wrap", "bounce", "sticky"],
    distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = "manhattan",
    observation_encoding: Literal["relative", "scaled", "absolute"] = "relative",
    topology: Literal["cubic"] = "cubic",  # NEW: Grid3D is always cubic topology
):
    """Initialize 3D cubic grid.

    Args:
        width: Number of cells in X dimension
        height: Number of cells in Y dimension
        depth: Number of cells in Z dimension (floors/layers)
        boundary: Boundary mode
        distance_metric: Distance calculation method
        observation_encoding: Position encoding strategy ("relative", "scaled", "absolute")
        topology: Grid topology ("cubic" for 3D Cartesian grid)
    """
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError(f"Grid dimensions must be positive: {width}×{height}×{depth}\nExample: width: 8, height: 8, depth: 3")
    if boundary not in ("clamp", "wrap", "bounce", "sticky"):
        raise ValueError(f"Unknown boundary mode: {boundary}")
    if distance_metric not in ("manhattan", "euclidean", "chebyshev"):
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    self.width = width
    self.height = height
    self.depth = depth
    self.boundary = boundary
    self.distance_metric = distance_metric
    self.observation_encoding = observation_encoding
    self.topology = topology  # NEW: Store topology
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/test_substrate/test_grid3d.py::test_grid3d_stores_topology_when_provided -xvs
pytest tests/test_townlet/test_substrate/test_grid3d.py::test_grid3d_topology_defaults_to_cubic -xvs
pytest tests/test_townlet/test_substrate/test_grid3d.py::test_grid3d_topology_attribute_exists -xvs
```

Expected: `PASSED` (3/3 tests)

**Step 5: Run full Grid3D test suite to ensure no regressions**

```bash
pytest tests/test_townlet/test_substrate/test_grid3d.py -v
```

Expected: All tests pass

**Step 6: Commit**

```bash
git add src/townlet/substrate/grid3d.py tests/test_townlet/test_substrate/test_grid3d.py
git commit -m "feat(substrate): add topology attribute to Grid3DSubstrate

- Add topology parameter with default 'cubic'
- Store topology as instance attribute
- Add tests for topology storage and default value
- Grid3D topology is always 'cubic' (6-connected 3D Cartesian grid)"
```

---

## Task 3: Add Topology to GridNDSubstrate (TDD)

**Files:**
- Test: `tests/test_townlet/test_substrate/test_gridnd.py`
- Modify: `src/townlet/substrate/gridnd.py:33-93`

**Step 1: Write failing tests for GridND topology storage**

Add to `tests/test_townlet/test_substrate/test_gridnd.py`:

```python
def test_gridnd_stores_topology_when_provided():
    """GridND should store topology attribute when explicitly provided."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="hypercube",
    )
    assert substrate.topology == "hypercube"


def test_gridnd_topology_defaults_to_hypercube():
    """GridND topology should default to 'hypercube' if not provided."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert substrate.topology == "hypercube"


def test_gridnd_topology_attribute_exists():
    """GridND should have topology attribute."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert hasattr(substrate, "topology")


def test_gridnd_topology_is_hypercube_for_all_dimensions():
    """GridND topology should be 'hypercube' regardless of dimensionality."""
    for num_dims in [4, 5, 7, 10]:
        substrate = GridNDSubstrate(
            dimension_sizes=[5] * num_dims,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )
        assert substrate.topology == "hypercube"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_townlet/test_substrate/test_gridnd.py::test_gridnd_stores_topology_when_provided -xvs
pytest tests/test_townlet/test_substrate/test_gridnd.py::test_gridnd_topology_defaults_to_hypercube -xvs
pytest tests/test_townlet/test_substrate/test_gridnd.py::test_gridnd_topology_attribute_exists -xvs
pytest tests/test_townlet/test_substrate/test_gridnd.py::test_gridnd_topology_is_hypercube_for_all_dimensions -xvs
```

Expected: `FAILED` - TypeError or AttributeError

**Step 3: Implement topology parameter in GridNDSubstrate**

Modify `src/townlet/substrate/gridnd.py:33-93`:

```python
def __init__(
    self,
    dimension_sizes: list[int],
    boundary: Literal["clamp", "wrap", "bounce", "sticky"],
    distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = "manhattan",
    observation_encoding: Literal["relative", "scaled", "absolute"] = "relative",
    topology: Literal["hypercube"] = "hypercube",  # NEW: GridND uses hypercube topology
):
    """Initialize N-dimensional grid substrate.

    Args:
        dimension_sizes: Size of each dimension [d0_size, d1_size, ..., dN_size]
        boundary: Boundary mode ("clamp", "wrap", "bounce", "sticky")
        distance_metric: Distance metric ("manhattan", "euclidean", "chebyshev")
        observation_encoding: Position encoding strategy ("relative", "scaled", "absolute")
        topology: Grid topology ("hypercube" for N-dimensional Cartesian grid)

    Raises:
        ValueError: If dimensions < 4 or any size <= 0

    Warnings:
        UserWarning: If dimensions >= 10 (action space size warning)
    """
    # Validate dimension count
    num_dims = len(dimension_sizes)
    if num_dims < 4:
        raise ValueError(
            f"GridND requires at least 4 dimensions, got {num_dims}. "
            f"Use Grid2DSubstrate (2D) or Grid3DSubstrate (3D) instead."
        )

    if num_dims > 100:
        raise ValueError(f"GridND dimension count ({num_dims}) exceeds limit (100)")

    # Warn at N≥10 (action space grows large)
    if num_dims >= 10:
        warnings.warn(
            f"GridND with {num_dims} dimensions has {2*num_dims+2} actions. "
            f"Large action spaces may be challenging to train. "
            f"Verify this is intentional for your research.",
            UserWarning,
        )

    # Validate dimension sizes
    for i, size in enumerate(dimension_sizes):
        if size <= 0:
            raise ValueError(f"Dimension sizes must be positive. Dimension {i} has size {size}.")

    # Validate parameters
    if boundary not in ("clamp", "wrap", "bounce", "sticky"):
        raise ValueError(f"Unknown boundary mode: {boundary}")

    if distance_metric not in ("manhattan", "euclidean", "chebyshev"):
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    if observation_encoding not in ("relative", "scaled", "absolute"):
        raise ValueError(f"Unknown observation encoding: {observation_encoding}")

    # Store configuration
    self.dimension_sizes = dimension_sizes
    self.boundary = boundary
    self.distance_metric = distance_metric
    self.observation_encoding = observation_encoding
    self.topology = topology  # NEW: Store topology
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/test_substrate/test_gridnd.py::test_gridnd_stores_topology_when_provided -xvs
pytest tests/test_townlet/test_substrate/test_gridnd.py::test_gridnd_topology_defaults_to_hypercube -xvs
pytest tests/test_townlet/test_substrate/test_gridnd.py::test_gridnd_topology_attribute_exists -xvs
pytest tests/test_townlet/test_substrate/test_gridnd.py::test_gridnd_topology_is_hypercube_for_all_dimensions -xvs
```

Expected: `PASSED` (4/4 tests)

**Step 5: Run full GridND test suite to ensure no regressions**

```bash
pytest tests/test_townlet/test_substrate/test_gridnd.py -v
```

Expected: All tests pass

**Step 6: Commit**

```bash
git add src/townlet/substrate/gridnd.py tests/test_townlet/test_substrate/test_gridnd.py
git commit -m "feat(substrate): add topology attribute to GridNDSubstrate

- Add topology parameter with default 'hypercube'
- Store topology as instance attribute
- Add tests for topology storage and default value
- GridND topology is always 'hypercube' (2N-connected N-dimensional grid)
- Topology is dimension-agnostic (same for 4D, 7D, 100D)"
```

---

## Task 4: Update GridNDConfig Schema (TDD)

**Files:**
- Test: `tests/test_townlet/test_substrate/test_config.py`
- Modify: `src/townlet/substrate/config.py:60-119`

**Step 1: Write failing test for GridNDConfig topology field**

Add to `tests/test_townlet/test_substrate/test_config.py`:

```python
def test_gridnd_config_includes_topology_field():
    """GridNDConfig should include topology field with default 'hypercube'."""
    config = GridNDConfig(
        dimension_sizes=[5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert hasattr(config, "topology")
    assert config.topology == "hypercube"


def test_gridnd_config_topology_can_be_overridden():
    """GridNDConfig should allow explicit topology specification."""
    config = GridNDConfig(
        dimension_sizes=[5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="hypercube",
    )
    assert config.topology == "hypercube"


def test_gridnd_config_validates_yaml_with_topology():
    """GridNDConfig should parse YAML with topology field."""
    yaml_data = {
        "dimension_sizes": [5, 5, 5, 5],
        "boundary": "clamp",
        "distance_metric": "manhattan",
        "observation_encoding": "relative",
        "topology": "hypercube",
    }
    config = GridNDConfig(**yaml_data)
    assert config.topology == "hypercube"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_townlet/test_substrate/test_config.py::test_gridnd_config_includes_topology_field -xvs
pytest tests/test_townlet/test_substrate/test_config.py::test_gridnd_config_topology_can_be_overridden -xvs
pytest tests/test_townlet/test_substrate/test_config.py::test_gridnd_config_validates_yaml_with_topology -xvs
```

Expected: `FAILED` - AttributeError ('GridNDConfig' object has no attribute 'topology')

**Step 3: Add topology field to GridNDConfig**

Modify `src/townlet/substrate/config.py:60-119`:

```python
class GridNDConfig(BaseModel):
    """Configuration for N-dimensional grid substrates (N≥4 dimensions).

    GridND supports 4D to 100D discrete grid substrates. For 2D/3D grids,
    use GridConfig (with topology="square" or "cubic") for better ergonomics.

    No-Defaults Principle: All fields required (no implicit defaults).
    """

    dimension_sizes: list[int] = Field(description="Size of each dimension [d0_size, d1_size, ..., dN_size]")

    boundary: Literal["clamp", "wrap", "bounce", "sticky"] = Field(description="Boundary handling mode")

    distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = Field(default="manhattan", description="Distance calculation method")

    # NEW: Phase 5C addition
    observation_encoding: Literal["relative", "scaled", "absolute"] = Field(
        default="relative",
        description="Position encoding strategy: relative (normalized [0,1]), scaled (normalized + sizes), absolute (raw coordinates)",
    )

    # NEW: Topology field (always 'hypercube' for now, explicit in config)
    topology: Literal["hypercube"] = Field(
        default="hypercube",
        description="Grid topology (hypercube for N-dimensional Cartesian grid)",
    )

    @model_validator(mode="after")
    def validate_dimension_sizes(self) -> "GridNDConfig":
        """Validate dimension count and sizes."""
        num_dims = len(self.dimension_sizes)

        if num_dims < 4:
            raise ValueError(
                f"GridND requires at least 4 dimensions, got {num_dims}.\n"
                f"For 2D grids, use:\n"
                f"  type: grid\n"
                f"  grid:\n"
                f"    topology: square\n"
                f"    width: 8\n"
                f"    height: 8\n"
                f"For 3D grids, use:\n"
                f"  type: grid\n"
                f"  grid:\n"
                f"    topology: cubic\n"
                f"    width: 8\n"
                f"    height: 8\n"
                f"    depth: 3\n"
            )

        if num_dims > 100:
            raise ValueError(
                f"GridND dimension count ({num_dims}) exceeds limit (100).\n"
                f"This is likely a configuration error. Verify dimension_sizes is correct."
            )

        # Validate all dimension sizes are positive
        for i, size in enumerate(self.dimension_sizes):
            if size <= 0:
                raise ValueError(
                    f"All dimension sizes must be positive. Dimension {i} has size {size}.\n"
                    f"Example: dimension_sizes: [8, 8, 8, 8]  # All positive"
                )

        return self
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/test_substrate/test_config.py::test_gridnd_config_includes_topology_field -xvs
pytest tests/test_townlet/test_substrate/test_config.py::test_gridnd_config_topology_can_be_overridden -xvs
pytest tests/test_townlet/test_substrate/test_config.py::test_gridnd_config_validates_yaml_with_topology -xvs
```

Expected: `PASSED` (3/3 tests)

**Step 5: Run full config test suite to ensure no regressions**

```bash
pytest tests/test_townlet/test_substrate/test_config.py -v
```

Expected: All tests pass

**Step 6: Commit**

```bash
git add src/townlet/substrate/config.py tests/test_townlet/test_substrate/test_config.py
git commit -m "feat(config): add topology field to GridNDConfig

- Add topology field with default 'hypercube'
- Explicit in config (no-defaults principle)
- Future-proof for other lattice types (simplex, BCC)
- Add tests for topology field validation"
```

---

## Task 5: Update SubstrateFactory to Propagate Topology (TDD)

**Files:**
- Test: `tests/test_townlet/test_substrate/test_factory.py`
- Modify: `src/townlet/substrate/factory.py:46-126`

**Step 1: Write failing tests for factory topology propagation**

Add to `tests/test_townlet/test_substrate/test_factory.py`:

```python
def test_factory_propagates_grid2d_topology():
    """Factory should pass topology from config to Grid2D substrate."""
    config = SubstrateConfig(
        version="1.0",
        description="Test Grid2D config",
        type="grid",
        grid=GridConfig(
            topology="square",
            width=8,
            height=8,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        ),
    )
    substrate = SubstrateFactory.build(config, torch.device("cpu"))
    assert isinstance(substrate, Grid2DSubstrate)
    assert substrate.topology == "square"


def test_factory_propagates_grid3d_topology():
    """Factory should pass topology from config to Grid3D substrate."""
    config = SubstrateConfig(
        version="1.0",
        description="Test Grid3D config",
        type="grid",
        grid=GridConfig(
            topology="cubic",
            width=8,
            height=8,
            depth=3,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        ),
    )
    substrate = SubstrateFactory.build(config, torch.device("cpu"))
    assert isinstance(substrate, Grid3DSubstrate)
    assert substrate.topology == "cubic"


def test_factory_propagates_gridnd_topology():
    """Factory should pass topology from config to GridND substrate."""
    config = SubstrateConfig(
        version="1.0",
        description="Test GridND config",
        type="gridnd",
        gridnd=GridNDConfig(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
            topology="hypercube",
        ),
    )
    substrate = SubstrateFactory.build(config, torch.device("cpu"))
    assert isinstance(substrate, GridNDSubstrate)
    assert substrate.topology == "hypercube"


def test_factory_sets_gridnd_topology_when_config_uses_default():
    """Factory should use GridND topology default when not specified in config."""
    config = SubstrateConfig(
        version="1.0",
        description="Test GridND config with default topology",
        type="gridnd",
        gridnd=GridNDConfig(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
            # topology not specified, uses default
        ),
    )
    substrate = SubstrateFactory.build(config, torch.device("cpu"))
    assert isinstance(substrate, GridNDSubstrate)
    assert substrate.topology == "hypercube"


def test_factory_continuous_substrates_have_no_topology():
    """Factory should create continuous substrates without topology attribute."""
    config = SubstrateConfig(
        version="1.0",
        description="Test Continuous2D config",
        type="continuous",
        continuous=ContinuousConfig(
            dimensions=2,
            bounds=[(0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
            distance_metric="euclidean",
            observation_encoding="relative",
        ),
    )
    substrate = SubstrateFactory.build(config, torch.device("cpu"))
    assert isinstance(substrate, Continuous2DSubstrate)
    assert not hasattr(substrate, "topology")


def test_factory_aspatial_substrate_has_no_topology():
    """Factory should create aspatial substrate without topology attribute."""
    config = SubstrateConfig(
        version="1.0",
        description="Test Aspatial config",
        type="aspatial",
        aspatial=AspatialSubstrateConfig(),
    )
    substrate = SubstrateFactory.build(config, torch.device("cpu"))
    assert isinstance(substrate, AspatialSubstrate)
    assert not hasattr(substrate, "topology")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_townlet/test_substrate/test_factory.py::test_factory_propagates_grid2d_topology -xvs
pytest tests/test_townlet/test_substrate/test_factory.py::test_factory_propagates_grid3d_topology -xvs
pytest tests/test_townlet/test_substrate/test_factory.py::test_factory_propagates_gridnd_topology -xvs
pytest tests/test_townlet/test_substrate/test_factory.py::test_factory_sets_gridnd_topology_when_config_uses_default -xvs
pytest tests/test_townlet/test_substrate/test_factory.py::test_factory_continuous_substrates_have_no_topology -xvs
pytest tests/test_townlet/test_substrate/test_factory.py::test_factory_aspatial_substrate_has_no_topology -xvs
```

Expected: `FAILED` - AssertionError (substrate.topology not matching expected value)

**Step 3: Update factory to pass topology to grid substrates**

Modify `src/townlet/substrate/factory.py:46-126`:

```python
if config.type == "grid":
    assert config.grid is not None  # Validated by pydantic

    if config.grid.topology == "square":
        return Grid2DSubstrate(
            width=config.grid.width,
            height=config.grid.height,
            boundary=config.grid.boundary,
            distance_metric=config.grid.distance_metric,
            observation_encoding=config.grid.observation_encoding,
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
            observation_encoding=config.grid.observation_encoding,
            topology=config.grid.topology,  # NEW: Pass topology from config
        )
    else:
        raise ValueError(f"Unknown grid topology: {config.grid.topology}")

# ... (continuous substrates unchanged) ...

elif config.type == "gridnd":
    assert config.gridnd is not None  # Validated by pydantic

    return GridNDSubstrate(
        dimension_sizes=config.gridnd.dimension_sizes,
        boundary=config.gridnd.boundary,
        distance_metric=config.gridnd.distance_metric,
        observation_encoding=config.gridnd.observation_encoding,
        topology=config.gridnd.topology,  # NEW: Pass topology from config
    )

# ... (continuousnd and aspatial unchanged) ...
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/test_substrate/test_factory.py::test_factory_propagates_grid2d_topology -xvs
pytest tests/test_townlet/test_substrate/test_factory.py::test_factory_propagates_grid3d_topology -xvs
pytest tests/test_townlet/test_substrate/test_factory.py::test_factory_propagates_gridnd_topology -xvs
pytest tests/test_townlet/test_substrate/test_factory.py::test_factory_sets_gridnd_topology_when_config_uses_default -xvs
pytest tests/test_townlet/test_substrate/test_factory.py::test_factory_continuous_substrates_have_no_topology -xvs
pytest tests/test_townlet/test_substrate/test_factory.py::test_factory_aspatial_substrate_has_no_topology -xvs
```

Expected: `PASSED` (6/6 tests)

**Step 5: Run full factory test suite to ensure no regressions**

```bash
pytest tests/test_townlet/test_substrate/test_factory.py -v
```

Expected: All tests pass

**Step 6: Commit**

```bash
git add src/townlet/substrate/factory.py tests/test_townlet/test_substrate/test_factory.py
git commit -m "feat(factory): propagate topology from config to substrates

- Pass topology parameter to Grid2D/Grid3D/GridND constructors
- Read topology from GridConfig and GridNDConfig
- Continuous/Aspatial substrates unchanged (no topology)
- Add tests for topology propagation and validation"
```

---

## Task 6: Update WebSocket Metadata Builder (TDD)

**Files:**
- Test: `tests/test_townlet/test_demo/test_live_inference.py`
- Modify: `src/townlet/demo/live_inference.py:155-192`

**Step 1: Write failing tests for WebSocket metadata topology handling**

Create or add to `tests/test_townlet/test_demo/test_live_inference.py`:

```python
import pytest
import torch
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate
from townlet.substrate.gridnd import GridNDSubstrate
from townlet.substrate.continuous import Continuous2DSubstrate
from townlet.substrate.aspatial import AspatialSubstrate
from townlet.demo.live_inference import LiveInferenceServer


class MockEnv:
    """Mock environment for testing metadata building."""
    def __init__(self, substrate):
        self.substrate = substrate


def test_metadata_includes_grid2d_topology():
    """WebSocket metadata should include topology for Grid2D."""
    substrate = Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="square",
    )

    # Mock the server just enough to test _build_substrate_metadata
    server = LiveInferenceServer(
        checkpoint_dir="",
        port=8766,
        speed_multiplier=1.0,
        total_episodes=1,
        training_config_path=None,
    )
    server.env = MockEnv(substrate)

    metadata = server._build_substrate_metadata()

    assert metadata["type"] == "grid2d"
    assert metadata["topology"] == "square"
    assert metadata["position_dim"] == 2
    assert metadata["width"] == 8
    assert metadata["height"] == 8


def test_metadata_includes_grid3d_topology():
    """WebSocket metadata should include topology for Grid3D."""
    substrate = Grid3DSubstrate(
        width=8,
        height=8,
        depth=3,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="cubic",
    )

    server = LiveInferenceServer(
        checkpoint_dir="",
        port=8766,
        speed_multiplier=1.0,
        total_episodes=1,
        training_config_path=None,
    )
    server.env = MockEnv(substrate)

    metadata = server._build_substrate_metadata()

    assert metadata["type"] == "grid3d"
    assert metadata["topology"] == "cubic"
    assert metadata["position_dim"] == 3
    assert metadata["depth"] == 3


def test_metadata_includes_gridnd_topology():
    """WebSocket metadata should include topology for GridND."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="hypercube",
    )

    server = LiveInferenceServer(
        checkpoint_dir="",
        port=8766,
        speed_multiplier=1.0,
        total_episodes=1,
        training_config_path=None,
    )
    server.env = MockEnv(substrate)

    metadata = server._build_substrate_metadata()

    assert metadata["type"] == "gridnd"
    assert metadata["topology"] == "hypercube"
    assert metadata["position_dim"] == 7
    assert metadata["dimension_sizes"] == [5, 5, 5, 5, 5, 5, 5]


def test_metadata_omits_topology_for_continuous():
    """WebSocket metadata should omit topology for Continuous substrates."""
    substrate = Continuous2DSubstrate(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    server = LiveInferenceServer(
        checkpoint_dir="",
        port=8766,
        speed_multiplier=1.0,
        total_episodes=1,
        training_config_path=None,
    )
    server.env = MockEnv(substrate)

    metadata = server._build_substrate_metadata()

    assert metadata["type"] == "continuous2d"
    assert "topology" not in metadata  # Should be omitted, not None
    assert metadata["position_dim"] == 2


def test_metadata_omits_topology_for_aspatial():
    """WebSocket metadata should omit topology for Aspatial substrate."""
    substrate = AspatialSubstrate()

    server = LiveInferenceServer(
        checkpoint_dir="",
        port=8766,
        speed_multiplier=1.0,
        total_episodes=1,
        training_config_path=None,
    )
    server.env = MockEnv(substrate)

    metadata = server._build_substrate_metadata()

    assert metadata["type"] == "aspatial"
    assert "topology" not in metadata  # Should be omitted, not None
    assert metadata["position_dim"] == 0


def test_metadata_topology_respects_substrate_attribute():
    """WebSocket metadata should read topology from substrate, not hardcode."""
    # This test verifies we're reading from substrate.topology, not hardcoding
    substrate = Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="square",  # Explicitly provided
    )

    server = LiveInferenceServer(
        checkpoint_dir="",
        port=8766,
        speed_multiplier=1.0,
        total_episodes=1,
        training_config_path=None,
    )
    server.env = MockEnv(substrate)

    metadata = server._build_substrate_metadata()

    # Should read from substrate.topology, not hardcode "square"
    assert metadata["topology"] == substrate.topology
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_townlet/test_demo/test_live_inference.py::test_metadata_includes_grid2d_topology -xvs
pytest tests/test_townlet/test_demo/test_live_inference.py::test_metadata_includes_grid3d_topology -xvs
pytest tests/test_townlet/test_demo/test_live_inference.py::test_metadata_includes_gridnd_topology -xvs
pytest tests/test_townlet/test_demo/test_live_inference.py::test_metadata_omits_topology_for_continuous -xvs
pytest tests/test_townlet/test_demo/test_live_inference.py::test_metadata_omits_topology_for_aspatial -xvs
pytest tests/test_townlet/test_demo/test_live_inference.py::test_metadata_topology_respects_substrate_attribute -xvs
```

Expected: `FAILED` - AssertionError (metadata structure doesn't match expected)

**Step 3: Update `_build_substrate_metadata()` to read topology from substrate**

Modify `src/townlet/demo/live_inference.py:155-192`:

```python
def _build_substrate_metadata(self) -> dict[str, Any]:
    """Build substrate metadata for WebSocket messages.

    Returns:
        Dict with substrate type, dimensions, and topology (if applicable).
        Used by frontend to dispatch correct renderer.

    Example:
        Grid2D: {"type": "grid2d", "position_dim": 2, "topology": "square", "width": 8, "height": 8, ...}
        GridND: {"type": "gridnd", "position_dim": 7, "topology": "hypercube", "dimension_sizes": [5,5,5,5,5,5,5], ...}
        Continuous2D: {"type": "continuous2d", "position_dim": 2, "bounds": [...], ...}
        Aspatial: {"type": "aspatial", "position_dim": 0}
    """
    if not self.env:
        return {"type": "unknown", "position_dim": 0}

    substrate = self.env.substrate

    # Derive substrate type from class name (Grid2DSubstrate -> "grid2d")
    substrate_type = type(substrate).__name__.lower().replace("substrate", "")

    metadata = {
        "type": substrate_type,
        "position_dim": substrate.position_dim,
    }

    # Add topology if substrate has it (grid substrates only)
    if hasattr(substrate, "topology"):
        metadata["topology"] = substrate.topology

    # Add type-specific metadata
    if substrate_type == "grid2d":
        metadata["width"] = substrate.width
        metadata["height"] = substrate.height
        metadata["boundary"] = substrate.boundary
        metadata["distance_metric"] = substrate.distance_metric

    elif substrate_type == "grid3d":
        metadata["width"] = substrate.width
        metadata["height"] = substrate.height
        metadata["depth"] = substrate.depth
        metadata["boundary"] = substrate.boundary
        metadata["distance_metric"] = substrate.distance_metric

    elif substrate_type == "gridnd":
        metadata["dimension_sizes"] = substrate.dimension_sizes
        metadata["boundary"] = substrate.boundary
        metadata["distance_metric"] = substrate.distance_metric

    elif substrate_type.startswith("continuous"):
        # Continuous substrates (1D/2D/3D/ND)
        metadata["bounds"] = substrate.bounds
        metadata["boundary"] = substrate.boundary
        metadata["movement_delta"] = substrate.movement_delta
        metadata["interaction_radius"] = substrate.interaction_radius
        metadata["distance_metric"] = substrate.distance_metric

    # Aspatial has no additional metadata

    return metadata
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/test_demo/test_live_inference.py::test_metadata_includes_grid2d_topology -xvs
pytest tests/test_townlet/test_demo/test_live_inference.py::test_metadata_includes_grid3d_topology -xvs
pytest tests/test_townlet/test_demo/test_live_inference.py::test_metadata_includes_gridnd_topology -xvs
pytest tests/test_townlet/test_demo/test_live_inference.py::test_metadata_omits_topology_for_continuous -xvs
pytest tests/test_townlet/test_demo/test_live_inference.py::test_metadata_omits_topology_for_aspatial -xvs
pytest tests/test_townlet/test_demo/test_live_inference.py::test_metadata_topology_respects_substrate_attribute -xvs
```

Expected: `PASSED` (6/6 tests)

**Step 5: Run full live_inference test suite to ensure no regressions**

```bash
pytest tests/test_townlet/test_demo/test_live_inference.py -v
```

Expected: All tests pass

**Step 6: Commit**

```bash
git add src/townlet/demo/live_inference.py tests/test_townlet/test_demo/test_live_inference.py
git commit -m "feat(websocket): read topology from substrate instead of hardcoding

- Remove hardcoded topology assignments ('square', 'cubic')
- Read topology from substrate.topology if attribute exists
- Omit topology field for continuous/aspatial substrates
- Add GridND and Continuous metadata building (preparatory)
- Add comprehensive tests for all substrate types"
```

---

## Task 7: Integration Testing (End-to-End)

**Files:**
- Test: `tests/test_townlet/test_integration.py`

**Step 1: Write integration test for config → factory → substrate → metadata flow**

Add to `tests/test_townlet/test_integration.py`:

```python
def test_topology_propagates_from_config_to_websocket_metadata_grid2d():
    """Integration test: topology flows from config → factory → substrate → metadata."""
    import tempfile
    from pathlib import Path
    import yaml
    from townlet.substrate.config import load_substrate_config
    from townlet.substrate.factory import SubstrateFactory
    from townlet.demo.live_inference import LiveInferenceServer

    # Step 1: Create substrate.yaml config
    config_data = {
        "version": "1.0",
        "description": "Integration test Grid2D config",
        "type": "grid",
        "grid": {
            "topology": "square",
            "width": 8,
            "height": 8,
            "boundary": "clamp",
            "distance_metric": "manhattan",
            "observation_encoding": "relative",
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "substrate.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Step 2: Load config
        config = load_substrate_config(config_path)
        assert config.grid.topology == "square"

        # Step 3: Build substrate via factory
        substrate = SubstrateFactory.build(config, torch.device("cpu"))
        assert hasattr(substrate, "topology")
        assert substrate.topology == "square"

        # Step 4: Build WebSocket metadata
        class MockEnv:
            def __init__(self, substrate):
                self.substrate = substrate

        server = LiveInferenceServer(
            checkpoint_dir="",
            port=8766,
            speed_multiplier=1.0,
            total_episodes=1,
            training_config_path=None,
        )
        server.env = MockEnv(substrate)

        metadata = server._build_substrate_metadata()

        # Step 5: Verify topology in metadata
        assert metadata["topology"] == "square"
        assert metadata["type"] == "grid2d"


def test_topology_propagates_from_config_to_websocket_metadata_gridnd():
    """Integration test: GridND topology flows through entire pipeline."""
    import tempfile
    from pathlib import Path
    import yaml
    from townlet.substrate.config import load_substrate_config
    from townlet.substrate.factory import SubstrateFactory
    from townlet.demo.live_inference import LiveInferenceServer

    # Step 1: Create substrate.yaml config for GridND
    config_data = {
        "version": "1.0",
        "description": "Integration test GridND config",
        "type": "gridnd",
        "gridnd": {
            "dimension_sizes": [5, 5, 5, 5, 5, 5, 5],
            "boundary": "clamp",
            "distance_metric": "manhattan",
            "observation_encoding": "relative",
            "topology": "hypercube",
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "substrate.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Step 2: Load config
        config = load_substrate_config(config_path)
        assert config.gridnd.topology == "hypercube"

        # Step 3: Build substrate via factory
        substrate = SubstrateFactory.build(config, torch.device("cpu"))
        assert hasattr(substrate, "topology")
        assert substrate.topology == "hypercube"

        # Step 4: Build WebSocket metadata
        class MockEnv:
            def __init__(self, substrate):
                self.substrate = substrate

        server = LiveInferenceServer(
            checkpoint_dir="",
            port=8766,
            speed_multiplier=1.0,
            total_episodes=1,
            training_config_path=None,
        )
        server.env = MockEnv(substrate)

        metadata = server._build_substrate_metadata()

        # Step 5: Verify topology in metadata
        assert metadata["topology"] == "hypercube"
        assert metadata["type"] == "gridnd"
        assert metadata["position_dim"] == 7


def test_continuous_substrate_has_no_topology_in_metadata():
    """Integration test: Continuous substrates omit topology throughout pipeline."""
    import tempfile
    from pathlib import Path
    import yaml
    from townlet.substrate.config import load_substrate_config
    from townlet.substrate.factory import SubstrateFactory
    from townlet.demo.live_inference import LiveInferenceServer

    # Step 1: Create substrate.yaml config for Continuous2D
    config_data = {
        "version": "1.0",
        "description": "Integration test Continuous2D config",
        "type": "continuous",
        "continuous": {
            "dimensions": 2,
            "bounds": [[0.0, 10.0], [0.0, 10.0]],
            "boundary": "clamp",
            "movement_delta": 0.5,
            "interaction_radius": 0.8,
            "distance_metric": "euclidean",
            "observation_encoding": "relative",
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "substrate.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Step 2: Load config
        config = load_substrate_config(config_path)
        assert not hasattr(config.continuous, "topology")

        # Step 3: Build substrate via factory
        substrate = SubstrateFactory.build(config, torch.device("cpu"))
        assert not hasattr(substrate, "topology")

        # Step 4: Build WebSocket metadata
        class MockEnv:
            def __init__(self, substrate):
                self.substrate = substrate

        server = LiveInferenceServer(
            checkpoint_dir="",
            port=8766,
            speed_multiplier=1.0,
            total_episodes=1,
            training_config_path=None,
        )
        server.env = MockEnv(substrate)

        metadata = server._build_substrate_metadata()

        # Step 5: Verify topology NOT in metadata
        assert "topology" not in metadata
        assert metadata["type"] == "continuous2d"
```

**Step 2: Run integration tests to verify they pass**

```bash
pytest tests/test_townlet/test_integration.py::test_topology_propagates_from_config_to_websocket_metadata_grid2d -xvs
pytest tests/test_townlet/test_integration.py::test_topology_propagates_from_config_to_websocket_metadata_gridnd -xvs
pytest tests/test_townlet/test_integration.py::test_continuous_substrate_has_no_topology_in_metadata -xvs
```

Expected: `PASSED` (3/3 tests)

**Step 3: Run full integration test suite**

```bash
pytest tests/test_townlet/test_integration.py -v
```

Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/test_townlet/test_integration.py
git commit -m "test(integration): add end-to-end topology propagation tests

- Test config → factory → substrate → metadata flow
- Verify Grid2D, GridND, and Continuous substrate handling
- Ensure topology present for grids, omitted for continuous
- Complete integration test coverage for topology feature"
```

---

## Task 8: Update Template Configs (Documentation)

**Files:**
- Modify: `configs/templates/substrate_gridnd.yaml`

**Step 1: Add topology field to GridND template config**

Modify `configs/templates/substrate_gridnd.yaml`:

```yaml
version: "1.0"
description: "N-dimensional hypercube grid (N≥4 dimensions)"
type: "gridnd"

gridnd:
  # Grid dimensions (one size per dimension)
  # Example: [5, 5, 5, 5] creates a 4D hypercube with 5 cells per dimension
  dimension_sizes: [5, 5, 5, 5]

  # Boundary handling
  # - clamp: Hard walls (agent clamped to grid edges)
  # - wrap: Toroidal wraparound (Pac-Man style)
  # - bounce: Elastic reflection (agent bounces off boundaries)
  # - sticky: Sticky walls (agent stays in place when hitting boundary)
  boundary: "clamp"

  # Distance metric
  # - manhattan: L1 norm, |x1-x2| + |y1-y2| + ... (matches movement actions)
  # - euclidean: L2 norm, sqrt((x1-x2)² + (y1-y2)² + ...) (straight-line distance)
  # - chebyshev: L∞ norm, max(|x1-x2|, |y1-y2|, ...) (diagonal movement)
  distance_metric: "manhattan"

  # Position encoding strategy
  # - relative: Normalized coordinates [0,1] per dimension (N dims, grid-size independent)
  # - scaled: Normalized + dimension sizes (2N dims, includes size metadata)
  # - absolute: Raw unnormalized coordinates (N dims, size-dependent)
  observation_encoding: "relative"

  # Grid topology (connectivity pattern)
  # - hypercube: 2N-connected N-dimensional Cartesian grid (default and only option)
  topology: "hypercube"

# Examples:
# 4D hypercube: dimension_sizes: [8, 8, 8, 8]
# 7D hypercube: dimension_sizes: [5, 5, 5, 5, 5, 5, 5]
# 10D hypercube: dimension_sizes: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

# Action space: 2N + 2 actions
# - 2N movement actions (±1 per dimension)
# - INTERACT (interact with affordance at position)
# - WAIT (no-op)

# Warning: N≥10 dimensions creates large action spaces (≥22 actions)
# Verify this is intentional for your research
```

**Step 2: Verify template config loads and validates correctly**

```bash
python -c "
from pathlib import Path
from townlet.substrate.config import load_substrate_config
config = load_substrate_config(Path('configs/templates/substrate_gridnd.yaml'))
print(f'Type: {config.type}')
print(f'Topology: {config.gridnd.topology}')
print(f'Dimensions: {len(config.gridnd.dimension_sizes)}')
"
```

Expected output:
```
Type: gridnd
Topology: hypercube
Dimensions: 4
```

**Step 3: Commit**

```bash
git add configs/templates/substrate_gridnd.yaml
git commit -m "docs(config): add topology field to GridND template config

- Add topology: 'hypercube' to template
- Document topology semantics (connectivity pattern)
- Add inline comments explaining hypercube topology
- Template now matches GridNDConfig schema"
```

---

## Task 9: Update Documentation

**Files:**
- Modify: `docs/architecture/substrate-system.md` (or create if doesn't exist)
- Modify: `/home/john/hamlet/CLAUDE.md:682-825` (Frontend Visualization section)

**Step 1: Document topology in substrate architecture docs**

Create or update `docs/architecture/substrate-system.md`:

```markdown
# Substrate System Architecture

## Topology Metadata

**Concept**: Topology describes the **connectivity pattern** of spatial substrates - how positions/cells connect to their neighbors.

### Grid Substrates

Grid substrates have discrete topology based on dimensionality:

- **Grid2D**: `topology = "square"` (4-connected 2D Cartesian grid)
  - Each cell connects to 4 neighbors: UP, DOWN, LEFT, RIGHT
  - Manhattan distance matches connectivity (L1 norm)

- **Grid3D**: `topology = "cubic"` (6-connected 3D Cartesian grid)
  - Each cell connects to 6 neighbors: ±X, ±Y, ±Z
  - Manhattan distance matches connectivity

- **GridND**: `topology = "hypercube"` (2N-connected N-dimensional grid)
  - Each cell connects to 2N neighbors (±1 per dimension)
  - Topology is dimension-agnostic (same for 4D, 7D, 100D)
  - Frontend checks `position_dim` for dimension-specific rendering

### Continuous Substrates

Continuous substrates (1D/2D/3D/ND) have **no discrete topology** - positions are continuous floats, not discrete cells. Topology field is **omitted** from metadata.

### Aspatial Substrates

Aspatial substrates have **no spatial structure** at all. Topology field is **omitted** from metadata.

### WebSocket Metadata Contract

**Grid substrates** (Grid2D/Grid3D/GridND):
```json
{
  "type": "grid2d",
  "position_dim": 2,
  "topology": "square",
  "width": 8,
  "height": 8,
  "boundary": "clamp",
  "distance_metric": "manhattan"
}
```

**Continuous substrates** (Continuous1D/2D/3D/ND):
```json
{
  "type": "continuous2d",
  "position_dim": 2,
  "bounds": [[0.0, 10.0], [0.0, 10.0]],
  "boundary": "clamp",
  "movement_delta": 0.5,
  "interaction_radius": 0.8,
  "distance_metric": "euclidean"
}
```
**Note**: No `topology` field - continuous spaces have no discrete connectivity.

**Aspatial substrate**:
```json
{
  "type": "aspatial",
  "position_dim": 0
}
```
**Note**: No `topology` field - no spatial structure.

### Configuration

Topology is configured in `substrate.yaml`:

**Grid2D/Grid3D** (`type: grid`):
```yaml
type: "grid"
grid:
  topology: "square"  # or "cubic" for 3D
  width: 8
  height: 8
  boundary: "clamp"
  distance_metric: "manhattan"
  observation_encoding: "relative"
```

**GridND** (`type: gridnd`):
```yaml
type: "gridnd"
gridnd:
  dimension_sizes: [5, 5, 5, 5, 5, 5, 5]
  topology: "hypercube"  # Always hypercube for N-dimensional grids
  boundary: "clamp"
  distance_metric: "manhattan"
  observation_encoding: "relative"
```

**Continuous substrates**: No topology field (not applicable).

### Implementation Flow

1. **Configuration** (`substrate.yaml`) → declares topology
2. **Config Schema** (`GridConfig`, `GridNDConfig`) → validates topology
3. **Factory** (`SubstrateFactory.build()`) → passes topology to substrate constructor
4. **Substrate** (Grid2D/Grid3D/GridND) → stores as `.topology` attribute
5. **WebSocket** (`live_inference._build_substrate_metadata()`) → reads `.topology` if present
6. **Frontend** → renders based on topology (square grid, cubic grid, etc.)

### Future Extensions

To add new topologies (e.g., simplex lattice, BCC lattice):

1. Add to `Literal["square", "cubic", "hypercube", "simplex"]` in config schema
2. Update factory to handle new topology value
3. Update substrate to accept new topology parameter
4. Update frontend to render new topology

Example:
```python
# Config
topology: Literal["square", "cubic", "hypercube", "simplex", "bcc"]

# Factory
if config.grid.topology == "simplex":
    return SimplexGridSubstrate(...)

# Substrate
class SimplexGridSubstrate(SpatialSubstrate):
    def __init__(self, ..., topology="simplex"):
        self.topology = topology
```
```

**Step 2: Update CLAUDE.md frontend visualization section**

Add to `/home/john/hamlet/CLAUDE.md` around line 707 (in WebSocket Contract section):

```markdown
**WebSocket Contract**:
```json
{
  "substrate": {
    "type": "grid2d",
    "position_dim": 2,
    "topology": "square",  // NEW: Grid connectivity pattern
    "width": 8,
    "height": 8,
    "boundary": "clamp",
    "distance_metric": "manhattan"
  },
  "grid": {
    "agents": [{"id": "agent_0", "x": 3, "y": 5}],
    "affordances": [{"type": "Bed", "x": 2, "y": 1}]
  }
}
```

**Topology Field** (Grid Substrates Only):
- **Grid2D**: `topology: "square"` (4-connected 2D grid)
- **Grid3D**: `topology: "cubic"` (6-connected 3D grid)
- **GridND**: `topology: "hypercube"` (2N-connected ND grid)
- **Continuous/Aspatial**: Topology field omitted (not applicable)

**Rationale**: Topology describes discrete connectivity pattern. Continuous spaces have no discrete cells, so topology is meaningless and omitted from metadata.
```

**Step 3: Verify documentation renders correctly**

```bash
# Check markdown syntax
cat docs/architecture/substrate-system.md
cat /home/john/hamlet/CLAUDE.md | grep -A 10 "topology"
```

**Step 4: Commit**

```bash
git add docs/architecture/substrate-system.md CLAUDE.md
git commit -m "docs: document topology metadata system

- Add substrate-system.md architecture doc
- Document topology semantics (connectivity pattern)
- Document config → factory → substrate → metadata flow
- Update CLAUDE.md WebSocket contract with topology field
- Document Grid vs Continuous vs Aspatial topology handling"
```

---

## Task 10: Run Full Test Suite and Verify

**Step 1: Run all substrate tests**

```bash
pytest tests/test_townlet/test_substrate/ -v
```

Expected: All tests pass

**Step 2: Run all demo tests**

```bash
pytest tests/test_townlet/test_demo/ -v
```

Expected: All tests pass

**Step 3: Run integration tests**

```bash
pytest tests/test_townlet/test_integration.py -v
```

Expected: All tests pass

**Step 4: Run full test suite**

```bash
pytest tests/test_townlet/ -v
```

Expected: All tests pass, no regressions

**Step 5: Verify type checking**

```bash
mypy src/townlet/substrate/ src/townlet/demo/live_inference.py
```

Expected: No type errors

**Step 6: Commit if any fixes were needed**

```bash
# Only if you had to fix anything
git add <fixed_files>
git commit -m "fix: address test failures and type errors"
```

---

## Task 11: Manual Testing (Optional but Recommended)

**Step 1: Test with real training config**

```bash
# Load an existing config and verify topology propagates
python -c "
from pathlib import Path
from townlet.substrate.config import load_substrate_config
from townlet.substrate.factory import SubstrateFactory
import torch

config = load_substrate_config(Path('configs/L1_full_observability/substrate.yaml'))
substrate = SubstrateFactory.build(config, torch.device('cpu'))
print(f'Substrate type: {type(substrate).__name__}')
print(f'Topology: {substrate.topology}')
print(f'Position dim: {substrate.position_dim}')
"
```

Expected output:
```
Substrate type: Grid2DSubstrate
Topology: square
Position dim: 2
```

**Step 2: Test WebSocket metadata building**

```bash
# Mock test of metadata builder
python -c "
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.demo.live_inference import LiveInferenceServer

substrate = Grid2DSubstrate(
    width=8, height=8,
    boundary='clamp',
    distance_metric='manhattan',
    observation_encoding='relative',
    topology='square'
)

class MockEnv:
    def __init__(self, substrate):
        self.substrate = substrate

server = LiveInferenceServer('', 8766, 1.0, 1, None)
server.env = MockEnv(substrate)
metadata = server._build_substrate_metadata()

import json
print(json.dumps(metadata, indent=2))
"
```

Expected output:
```json
{
  "type": "grid2d",
  "position_dim": 2,
  "topology": "square",
  "width": 8,
  "height": 8,
  "boundary": "clamp",
  "distance_metric": "manhattan"
}
```

**Step 3: Verify no topology for continuous**

```bash
python -c "
from townlet.substrate.continuous import Continuous2DSubstrate
from townlet.demo.live_inference import LiveInferenceServer

substrate = Continuous2DSubstrate(
    min_x=0.0, max_x=10.0,
    min_y=0.0, max_y=10.0,
    boundary='clamp',
    movement_delta=0.5,
    interaction_radius=0.8,
    distance_metric='euclidean',
    observation_encoding='relative'
)

class MockEnv:
    def __init__(self, substrate):
        self.substrate = substrate

server = LiveInferenceServer('', 8766, 1.0, 1, None)
server.env = MockEnv(substrate)
metadata = server._build_substrate_metadata()

import json
print(json.dumps(metadata, indent=2))
print(f\"Has topology: {'topology' in metadata}\")
"
```

Expected output:
```json
{
  "type": "continuous2d",
  "position_dim": 2,
  "bounds": [[0.0, 10.0], [0.0, 10.0]],
  "boundary": "clamp",
  "movement_delta": 0.5,
  "interaction_radius": 0.8,
  "distance_metric": "euclidean"
}
Has topology: False
```

---

## Task 12: Final Review and Cleanup

**Step 1: Review all changes**

```bash
git log --oneline --max-count=12
git diff origin/task-002a-configurable-spatial-substrates..HEAD --stat
```

**Step 2: Verify commit messages follow conventional commits**

All commits should follow pattern: `<type>(<scope>): <description>`

Example:
- `feat(substrate): add topology attribute to Grid2DSubstrate`
- `test(integration): add end-to-end topology propagation tests`
- `docs: document topology metadata system`

**Step 3: Squash commits if needed (optional)**

If you have many small "fix typo" commits, consider squashing:

```bash
# Interactive rebase to squash last N commits
git rebase -i HEAD~12

# In editor, mark commits to squash with 's'
# Save and exit
```

**Step 4: Final test run**

```bash
pytest tests/test_townlet/ -v --tb=short
```

Expected: All tests pass

**Step 5: Push to remote**

```bash
git push origin task-002a-configurable-spatial-substrates
```

---

## Summary

**What was implemented:**

1. **Substrate Changes**: Added `topology` attribute to Grid2D/Grid3D/GridND substrates
2. **Config Changes**: Added `topology` field to GridNDConfig schema
3. **Factory Changes**: Updated factory to propagate topology from config to substrates
4. **WebSocket Changes**: Updated metadata builder to read topology from substrate instances
5. **Testing**: Comprehensive TDD coverage (unit + integration tests)
6. **Documentation**: Architecture docs + template configs updated

**Key Principles Applied:**

- **TDD**: RED-GREEN-REFACTOR cycle for all changes
- **DRY**: Topology semantics defined once, propagated through system
- **YAGNI**: Only added topology where needed (grid substrates)
- **Separation of Concerns**: Topology = connectivity, position_dim = dimensionality
- **Explicit Configuration**: Topology in substrate.yaml (no hidden defaults)

**Verification:**

- All tests pass (unit + integration)
- Type checking passes
- Manual testing confirms metadata structure
- Documentation updated

**Next Steps:**

- Deploy to staging environment
- Test with live_inference server + frontend
- Verify frontend renders grids correctly with topology metadata
- Consider adding Grid3D/GridND frontend renderers (future work)
