# Phase 5C: Configurable Observation Encoding & N-Dimensional Substrates - Implementation Plan (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable observation encoding (relative/scaled/absolute) to all substrates and implement N-dimensional substrates (GridND, ContinuousND) for abstract state space research.

**Architecture:** Two-part implementation: (1) Retrofit observation_encoding parameter to existing Grid2D/3D and Continuous1D/2D/3D substrates, (2) Add GridND and ContinuousND classes for 4D+ dimensions with built-in configurable encoding.

**Tech Stack:** PyTorch, Pydantic v2, YAML configs, pytest

**Design Reference:** `/home/john/hamlet/docs/plans/task-002a-phase-5c-ndimensional-substrates.md` (v4)

**Prerequisites:**
- ✅ **Phase 5B MUST be complete** - Includes `action_space_size` property (2N+1 formula)
- ✅ Phase 5B implementation plan: `docs/plans/2025-11-06-action-space-size-property.md`

**BREAKING CHANGE:** Grid2D observation encoding changes from one-hot (64 dims for 8×8) to normalized coordinates (2 dims). Old checkpoints incompatible. See Migration section below.

**Estimated Total Time:** 17-21 hours (Part 1: 5-7h, Part 2: 12-14h)

**Note:** Tasks 1.1 and 1.7 completed in Phase 5B, reducing Part 1 time by ~1 hour.

---

## Migration Guide: Grid2D Encoding Change

**Phase 5A/5B (OLD):**
```python
# Grid2D used one-hot grid encoding
Grid2D(8×8).encode_observation() → [num_agents, 64]  # One cell per feature
Grid2D(10×10).encode_observation() → [num_agents, 100]
```

**Phase 5C (NEW - BREAKING):**
```python
# Grid2D uses normalized coordinate encoding (like Grid3D)
Grid2D(8×8, observation_encoding="relative").encode_observation() → [num_agents, 2]
Grid2D(8×8, observation_encoding="scaled").encode_observation() → [num_agents, 4]
Grid2D(8×8, observation_encoding="absolute").encode_observation() → [num_agents, 2]
```

**Impact:**
- **Checkpoints from Phase 5A/5B are INCOMPATIBLE** (observation_dim changed)
- **Networks need architecture update:** `obs_dim = 64 → 2` (or 4 for scaled)
- **Configs need observation_encoding field:** Add `observation_encoding: "relative"`

**Why This Change:**
- Consistency with Grid3D (already uses normalized in Phase 5B)
- Scalability: 2 dims regardless of grid size (vs 100 dims for 10×10 one-hot)
- Transfer learning: Grid2D ↔ Grid3D now compatible
- Prepares for N-dimensional substrates

**Migration Steps:**
1. Update network architecture: Change `obs_dim` to match new encoding
2. Add `observation_encoding: "relative"` to all Grid2D configs
3. Retrain from scratch (checkpoints incompatible)

---

## Part 1: Retrofit Existing Substrates (6-8 hours)

### ~~Task 1.1: action_space_size Property~~ ✅ DONE IN PHASE 5B

**Note:** The `action_space_size` property was implemented in Phase 5B as a prerequisite for Phase 5C.

**Implementation:** See `docs/plans/2025-11-06-action-space-size-property.md`

**Status:** ✅ Complete
- Property added to `SpatialSubstrate` base class
- Formula: `2*position_dim + 1` (aspatial = 1)
- Environment updated to use `substrate.action_space_size`
- All tests passing

**Verification:** Before starting Phase 5C, confirm Phase 5B is complete:
```bash
pytest tests/test_townlet/test_substrate/test_base.py::TestActionSpaceSizeProperty -v
```

---

### Task 1.2: Add observation_encoding Config Schema (45 min)

**Context:** Add observation_encoding parameter to GridConfig and ContinuousConfig with Pydantic validation.

**Files:**
- Modify: `src/townlet/substrate/config.py`
- Test: `tests/test_townlet/unit/test_substrate_config.py`

#### Step 1.2.1: Write failing test for observation_encoding config field

**Modify:** `tests/test_townlet/unit/test_substrate_config.py`

Add after existing Grid config tests (around line 150):

```python
def test_grid_config_observation_encoding_valid():
    """Test Grid config accepts valid observation_encoding values."""
    for encoding in ["relative", "scaled", "absolute"]:
        config = GridConfig(
            topology="square",
            width=8,
            height=8,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding=encoding,
        )
        assert config.observation_encoding == encoding


def test_grid_config_observation_encoding_default():
    """Test Grid config defaults to relative for backward compatibility."""
    config = GridConfig(
        topology="square",
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        # observation_encoding NOT provided
    )
    assert config.observation_encoding == "relative"


def test_grid_config_observation_encoding_invalid():
    """Test Grid config rejects invalid observation_encoding."""
    with pytest.raises(ValueError, match="observation_encoding"):
        GridConfig(
            topology="square",
            width=8,
            height=8,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="invalid",  # Not in Literal
        )


def test_continuous_config_observation_encoding_valid():
    """Test Continuous config accepts valid observation_encoding values."""
    for encoding in ["relative", "scaled", "absolute"]:
        config = ContinuousConfig(
            dimensions=2,
            bounds=[(0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding=encoding,
        )
        assert config.observation_encoding == encoding


def test_continuous_config_observation_encoding_default():
    """Test Continuous config defaults to relative for backward compatibility."""
    config = ContinuousConfig(
        dimensions=2,
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        distance_metric="euclidean",
        # observation_encoding NOT provided
    )
    assert config.observation_encoding == "relative"
```

**Run:** `uv run pytest tests/test_townlet/unit/test_substrate_config.py::test_grid_config_observation_encoding_valid -v`

**Expected:** FAIL - TypeError: __init__() got an unexpected keyword argument 'observation_encoding'

#### Step 1.2.2: Add observation_encoding field to GridConfig

**Modify:** `src/townlet/substrate/config.py`

In GridConfig class (around line 30):

```python
class GridConfig(BaseModel):
    """Configuration for grid-based substrates (2D square or 3D cubic)."""

    topology: Literal["square", "cubic"]
    width: int = Field(gt=0, description="Grid width (x-dimension)")
    height: int = Field(gt=0, description="Grid height (y-dimension)")
    depth: int | None = Field(None, gt=0, description="Grid depth (z-dimension, required for cubic)")
    boundary: Literal["clamp", "wrap", "bounce", "sticky"]
    distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = "manhattan"

    # NEW: Phase 5C addition
    observation_encoding: Literal["relative", "scaled", "absolute"] = Field(
        default="relative",
        description="Position encoding strategy: relative (normalized [0,1]), scaled (normalized + ranges), absolute (raw coordinates)",
    )

    # ... rest of validators ...
```

#### Step 1.2.3: Add observation_encoding field to ContinuousConfig

**Modify:** `src/townlet/substrate/config.py`

In ContinuousConfig class (around line 70):

```python
class ContinuousConfig(BaseModel):
    """Configuration for continuous-space substrates (1D/2D/3D)."""

    dimensions: int = Field(ge=1, le=3, description="Number of dimensions (1-3)")
    bounds: list[tuple[float, float]] = Field(description="[(min, max), ...] bounds per dimension")
    boundary: Literal["clamp", "wrap", "bounce", "sticky"]
    movement_delta: float = Field(gt=0, description="Distance moved per discrete action")
    interaction_radius: float = Field(gt=0, description="Proximity threshold for interactions")
    distance_metric: Literal["euclidean", "manhattan"] = "euclidean"

    # NEW: Phase 5C addition
    observation_encoding: Literal["relative", "scaled", "absolute"] = Field(
        default="relative",
        description="Position encoding strategy: relative (normalized [0,1]), scaled (normalized + ranges), absolute (raw coordinates)",
    )

    # ... rest of validators ...
```

**Run:** `uv run pytest tests/test_townlet/unit/test_substrate_config.py -k observation_encoding -v`

**Expected:** PASS (all 6 new tests)

#### Step 1.2.4: Commit

```bash
git add src/townlet/substrate/config.py tests/test_townlet/unit/test_substrate_config.py
git commit -m "feat(config): add observation_encoding parameter to substrate configs

- Add observation_encoding field to GridConfig and ContinuousConfig
- Three modes: relative (default), scaled, absolute
- Defaults to 'relative' for backward compatibility
- Pydantic validation with Literal type
- Part of Phase 5C Part 1

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.3: Update Factory to Pass observation_encoding (20 min)

**Context:** SubstrateFactory must pass observation_encoding to substrate constructors.

**Files:**
- Modify: `src/townlet/substrate/factory.py`
- Test: `tests/test_townlet/integration/test_substrate_factory.py`

#### Step 1.3.1: Write failing test for factory observation_encoding propagation

**Modify:** `tests/test_townlet/integration/test_substrate_factory.py`

Add after existing factory tests:

```python
def test_factory_passes_observation_encoding_to_grid2d():
    """Test factory passes observation_encoding to Grid2D substrate."""
    config = SubstrateConfig(
        version="1.0",
        description="Test",
        type="grid",
        grid=GridConfig(
            topology="square",
            width=8,
            height=8,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="scaled",  # Non-default
        ),
    )
    substrate = SubstrateFactory.build(config, device=torch.device("cpu"))
    assert substrate.observation_encoding == "scaled"


def test_factory_passes_observation_encoding_to_continuous():
    """Test factory passes observation_encoding to Continuous substrate."""
    config = SubstrateConfig(
        version="1.0",
        description="Test",
        type="continuous",
        continuous=ContinuousConfig(
            dimensions=2,
            bounds=[(0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="absolute",  # Non-default
        ),
    )
    substrate = SubstrateFactory.build(config, device=torch.device("cpu"))
    assert substrate.observation_encoding == "absolute"
```

**Run:** `uv run pytest tests/test_townlet/integration/test_substrate_factory.py::test_factory_passes_observation_encoding_to_grid2d -v`

**Expected:** FAIL - AttributeError: 'Grid2DSubstrate' object has no attribute 'observation_encoding'

#### Step 1.3.2: Update factory to pass observation_encoding parameter

**Modify:** `src/townlet/substrate/factory.py`

Update Grid2D instantiation (around line 45):

```python
if config.grid.topology == "square":
    return Grid2DSubstrate(
        width=config.grid.width,
        height=config.grid.height,
        boundary=config.grid.boundary,
        distance_metric=config.grid.distance_metric,
        observation_encoding=config.grid.observation_encoding,  # NEW
    )
```

Update Grid3D instantiation (around line 55):

```python
elif config.grid.topology == "cubic":
    return Grid3DSubstrate(
        width=config.grid.width,
        height=config.grid.height,
        depth=config.grid.depth,
        boundary=config.grid.boundary,
        distance_metric=config.grid.distance_metric,
        observation_encoding=config.grid.observation_encoding,  # NEW
    )
```

Update Continuous instantiations (around line 75):

```python
if config.continuous.dimensions == 1:
    return Continuous1DSubstrate(
        dimensions=1,
        bounds=config.continuous.bounds,
        boundary=config.continuous.boundary,
        movement_delta=config.continuous.movement_delta,
        interaction_radius=config.continuous.interaction_radius,
        distance_metric=config.continuous.distance_metric,
        observation_encoding=config.continuous.observation_encoding,  # NEW
    )
elif config.continuous.dimensions == 2:
    return Continuous2DSubstrate(
        dimensions=2,
        bounds=config.continuous.bounds,
        boundary=config.continuous.boundary,
        movement_delta=config.continuous.movement_delta,
        interaction_radius=config.continuous.interaction_radius,
        distance_metric=config.continuous.distance_metric,
        observation_encoding=config.continuous.observation_encoding,  # NEW
    )
elif config.continuous.dimensions == 3:
    return Continuous3DSubstrate(
        dimensions=3,
        bounds=config.continuous.bounds,
        boundary=config.continuous.boundary,
        movement_delta=config.continuous.movement_delta,
        interaction_radius=config.continuous.interaction_radius,
        distance_metric=config.continuous.distance_metric,
        observation_encoding=config.continuous.observation_encoding,  # NEW
    )
```

**Run:** `uv run pytest tests/test_townlet/integration/test_substrate_factory.py -k observation_encoding -v`

**Expected:** FAIL - TypeError: __init__() got an unexpected keyword argument 'observation_encoding'
(Substrates don't accept parameter yet)

#### Step 1.3.3: Commit factory changes (will fail until substrates updated)

```bash
git add src/townlet/substrate/factory.py tests/test_townlet/integration/test_substrate_factory.py
git commit -m "feat(factory): pass observation_encoding to substrate constructors

- Update SubstrateFactory.build() to pass observation_encoding
- Applies to Grid2D, Grid3D, Continuous1D/2D/3D
- Tests added (will fail until substrates accept parameter)
- Part of Phase 5C Part 1

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.4: Retrofit Grid2D with Configurable Encoding (90 min)

**Context:** Grid2D currently uses one-hot grid encoding (Phase 5A/5B). **BREAKING CHANGE:** Replace one-hot with normalized coordinate encoding (relative/scaled/absolute). Default to relative mode.

**Files:**
- Modify: `src/townlet/substrate/grid2d.py`
- Test: `tests/test_townlet/phase5/test_grid2d_observation_encoding.py` (NEW)

#### Step 1.4.1: Write failing tests for Grid2D observation encoding modes

**Create:** `tests/test_townlet/phase5/test_grid2d_observation_encoding.py`

```python
"""Test configurable observation encoding for Grid2D substrate."""
import pytest
import torch
from townlet.substrate.grid2d import Grid2DSubstrate


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def grid2d_relative(device):
    """Grid2D with relative encoding (normalized coordinates)."""
    return Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )


@pytest.fixture
def grid2d_scaled(device):
    """Grid2D with scaled encoding (normalized + ranges)."""
    return Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="scaled",
    )


@pytest.fixture
def grid2d_absolute(device):
    """Grid2D with absolute encoding (raw coordinates)."""
    return Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="absolute",
    )


def test_grid2d_relative_encoding_dimensions(grid2d_relative):
    """Relative encoding should return [num_agents, 2] normalized positions."""
    positions = torch.tensor([[0, 0], [7, 7], [3, 4]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_relative.encode_observation(positions, affordances)

    assert encoded.shape == (3, 2), "Should return [num_agents, 2]"
    assert encoded.dtype == torch.float32


def test_grid2d_relative_encoding_values(grid2d_relative):
    """Relative encoding should normalize to [0, 1] range."""
    positions = torch.tensor([[0, 0], [7, 7], [3, 4]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_relative.encode_observation(positions, affordances)

    # 0 / 7 = 0.0, 7 / 7 = 1.0, 3 / 7 = 0.428..., 4 / 7 = 0.571...
    assert torch.allclose(encoded[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(encoded[1], torch.tensor([1.0, 1.0]))
    assert torch.allclose(encoded[2], torch.tensor([3/7, 4/7]))


def test_grid2d_scaled_encoding_dimensions(grid2d_scaled):
    """Scaled encoding should return [num_agents, 4] (normalized + ranges)."""
    positions = torch.tensor([[0, 0], [7, 7]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_scaled.encode_observation(positions, affordances)

    assert encoded.shape == (2, 4), "Should return [num_agents, 4] (2 pos + 2 ranges)"
    assert encoded.dtype == torch.float32


def test_grid2d_scaled_encoding_values(grid2d_scaled):
    """Scaled encoding should have normalized positions + range metadata."""
    positions = torch.tensor([[3, 4]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_scaled.encode_observation(positions, affordances)

    # First 2 dims: normalized positions
    assert torch.allclose(encoded[0, :2], torch.tensor([3/7, 4/7]))
    # Last 2 dims: range sizes (width=8, height=8)
    assert torch.allclose(encoded[0, 2:], torch.tensor([8.0, 8.0]))


def test_grid2d_absolute_encoding_dimensions(grid2d_absolute):
    """Absolute encoding should return [num_agents, 2] raw coordinates."""
    positions = torch.tensor([[0, 0], [7, 7]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_absolute.encode_observation(positions, affordances)

    assert encoded.shape == (2, 2), "Should return [num_agents, 2]"
    assert encoded.dtype == torch.float32


def test_grid2d_absolute_encoding_values(grid2d_absolute):
    """Absolute encoding should return raw unnormalized coordinates."""
    positions = torch.tensor([[0, 0], [7, 7], [3, 4]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_absolute.encode_observation(positions, affordances)

    # Should be raw float coordinates
    assert torch.allclose(encoded[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(encoded[1], torch.tensor([7.0, 7.0]))
    assert torch.allclose(encoded[2], torch.tensor([3.0, 4.0]))


def test_grid2d_get_observation_dim_relative(grid2d_relative):
    """get_observation_dim() should return 2 for relative encoding."""
    assert grid2d_relative.get_observation_dim() == 2


def test_grid2d_get_observation_dim_scaled(grid2d_scaled):
    """get_observation_dim() should return 4 for scaled encoding."""
    assert grid2d_scaled.get_observation_dim() == 4


def test_grid2d_get_observation_dim_absolute(grid2d_absolute):
    """get_observation_dim() should return 2 for absolute encoding."""
    assert grid2d_absolute.get_observation_dim() == 2


def test_grid2d_default_encoding_is_relative():
    """Grid2D should default to relative encoding for backward compatibility."""
    substrate = Grid2DSubstrate(
        width=8, height=8, boundary="clamp", distance_metric="manhattan"
        # observation_encoding NOT provided
    )
    assert substrate.observation_encoding == "relative"
```

**Run:** `uv run pytest tests/test_townlet/phase5/test_grid2d_observation_encoding.py -v`

**Expected:** FAIL - TypeError: __init__() got an unexpected keyword argument 'observation_encoding'

#### Step 1.4.2: Add observation_encoding parameter to Grid2D constructor

**Modify:** `src/townlet/substrate/grid2d.py`

Update `__init__` method (around line 20):

```python
def __init__(
    self,
    width: int,
    height: int,
    boundary: str,
    distance_metric: str = "manhattan",
    observation_encoding: str = "relative",  # NEW: Phase 5C
):
    """Initialize Grid2D substrate.

    Args:
        width: Grid width (x-dimension)
        height: Grid height (y-dimension)
        boundary: Boundary mode ("clamp", "wrap", "bounce", "sticky")
        distance_metric: Distance metric ("manhattan", "euclidean", "chebyshev")
        observation_encoding: Position encoding strategy ("relative", "scaled", "absolute")
    """
    self.width = width
    self.height = height
    self.boundary = boundary
    self.distance_metric = distance_metric
    self.observation_encoding = observation_encoding  # NEW
```

**Run:** `uv run pytest tests/test_townlet/phase5/test_grid2d_observation_encoding.py::test_grid2d_default_encoding_is_relative -v`

**Expected:** PASS (parameter accepted, but encoding not implemented yet)

#### Step 1.4.3: Implement relative encoding helper method

**Modify:** `src/townlet/substrate/grid2d.py`

Add new method after `encode_observation()` (around line 180):

```python
def _encode_relative(
    self,
    positions: torch.Tensor,
    affordances: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Encode positions as normalized coordinates [0, 1].

    Args:
        positions: Agent positions [num_agents, 2]
        affordances: Affordance positions (currently unused)

    Returns:
        [num_agents, 2] normalized positions
    """
    num_agents = positions.shape[0]
    device = positions.device

    normalized = torch.zeros((num_agents, 2), dtype=torch.float32, device=device)
    normalized[:, 0] = positions[:, 0].float() / max(self.width - 1, 1)
    normalized[:, 1] = positions[:, 1].float() / max(self.height - 1, 1)

    return normalized
```

**Run:** `uv run pytest tests/test_townlet/phase5/test_grid2d_observation_encoding.py::test_grid2d_relative_encoding_values -v`

**Expected:** FAIL - Method exists but not called from encode_observation()

#### Step 1.4.4: Implement scaled encoding helper method

**Modify:** `src/townlet/substrate/grid2d.py`

Add after `_encode_relative()`:

```python
def _encode_scaled(
    self,
    positions: torch.Tensor,
    affordances: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Encode positions as normalized coordinates + range metadata.

    Args:
        positions: Agent positions [num_agents, 2]
        affordances: Affordance positions (currently unused)

    Returns:
        [num_agents, 4] normalized positions + range sizes
        First 2 dims: normalized [0, 1]
        Last 2 dims: (width, height)
    """
    num_agents = positions.shape[0]
    device = positions.device

    # Get normalized positions
    relative = self._encode_relative(positions, affordances)

    # Add range metadata
    ranges = torch.tensor(
        [float(self.width), float(self.height)],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).expand(num_agents, -1)

    return torch.cat([relative, ranges], dim=1)
```

#### Step 1.4.5: Implement absolute encoding helper method

**Modify:** `src/townlet/substrate/grid2d.py`

Add after `_encode_scaled()`:

```python
def _encode_absolute(
    self,
    positions: torch.Tensor,
    affordances: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Encode positions as raw unnormalized coordinates.

    Args:
        positions: Agent positions [num_agents, 2]
        affordances: Affordance positions (currently unused)

    Returns:
        [num_agents, 2] raw coordinates (as float)
    """
    return positions.float()
```

#### Step 1.4.6: Update encode_observation() to dispatch based on encoding mode

**Modify:** `src/townlet/substrate/grid2d.py`

Replace existing `encode_observation()` implementation (around line 150):

```python
def encode_observation(
    self,
    positions: torch.Tensor,
    affordances: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Encode agent positions and affordances into observation space.

    Args:
        positions: Agent positions [num_agents, 2]
        affordances: Dict mapping affordance names to positions [2]

    Returns:
        Encoded observations with dimensions based on encoding mode:
        - relative: [num_agents, 2]
        - scaled: [num_agents, 4]
        - absolute: [num_agents, 2]
    """
    if self.observation_encoding == "relative":
        return self._encode_relative(positions, affordances)
    elif self.observation_encoding == "scaled":
        return self._encode_scaled(positions, affordances)
    elif self.observation_encoding == "absolute":
        return self._encode_absolute(positions, affordances)
    else:
        raise ValueError(
            f"Invalid observation_encoding: {self.observation_encoding}. "
            f"Must be 'relative', 'scaled', or 'absolute'."
        )
```

**Run:** `uv run pytest tests/test_townlet/phase5/test_grid2d_observation_encoding.py -v`

**Expected:** PASS (most tests), but get_observation_dim() tests will fail

#### Step 1.4.7: Update get_observation_dim() to reflect encoding mode

**Modify:** `src/townlet/substrate/grid2d.py`

Replace existing `get_observation_dim()` (around line 220):

```python
def get_observation_dim(self) -> int:
    """Return dimensionality of position encoding.

    Returns:
        - relative: 2 (normalized x, y)
        - scaled: 4 (normalized x, y, width, height)
        - absolute: 2 (raw x, y)
    """
    if self.observation_encoding == "relative":
        return 2
    elif self.observation_encoding == "scaled":
        return 4
    elif self.observation_encoding == "absolute":
        return 2
    else:
        raise ValueError(f"Invalid observation_encoding: {self.observation_encoding}")
```

**Run:** `uv run pytest tests/test_townlet/phase5/test_grid2d_observation_encoding.py -v`

**Expected:** PASS (all tests)

#### Step 1.4.8: Run all Grid2D tests to check for regressions

**Run:** `uv run pytest tests/test_townlet/ -k grid2d -v`

**Expected:** PASS (no regressions from existing tests)

#### Step 1.4.9: Commit

```bash
git add src/townlet/substrate/grid2d.py tests/test_townlet/phase5/test_grid2d_observation_encoding.py
git commit -m "feat(grid2d): add configurable observation encoding

- Add observation_encoding parameter (relative/scaled/absolute)
- Default to 'relative' for backward compatibility
- relative: normalized [0,1] coordinates (2 dims)
- scaled: normalized + range metadata (4 dims)
- absolute: raw unnormalized coordinates (2 dims)
- Update get_observation_dim() to reflect encoding mode
- Part of Phase 5C Part 1

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.5: Retrofit Grid3D with Configurable Encoding (60 min)

**Context:** Grid3D already uses normalized encoding (relative). Add scaled and absolute modes.

**Files:**
- Modify: `src/townlet/substrate/grid3d.py`
- Test: `tests/test_townlet/phase5/test_grid3d_observation_encoding.py` (NEW)

**Implementation:** Follow same TDD pattern as Task 1.4:
1. Write failing tests for all three encoding modes
2. Add observation_encoding parameter to constructor
3. Implement _encode_relative(), _encode_scaled(), _encode_absolute()
4. Update encode_observation() to dispatch
5. Update get_observation_dim()
6. Verify no regressions
7. Commit

**Expected Results:**
- relative: [num_agents, 3] normalized
- scaled: [num_agents, 6] normalized + (width, height, depth)
- absolute: [num_agents, 3] raw coordinates

---

### Task 1.6: Retrofit Continuous Substrates with Configurable Encoding (90 min)

**Context:** Continuous1D/2D/3D already use normalized encoding. Add scaled and absolute modes.

**Files:**
- Modify: `src/townlet/substrate/continuous.py`
- Test: `tests/test_townlet/phase5/test_continuous_observation_encoding.py` (NEW)

**Implementation:** Follow same TDD pattern as Task 1.4, but for all three classes:
1. Write failing tests for Continuous1D, Continuous2D, Continuous3D
2. Add observation_encoding parameter to all three constructors
3. Implement _encode_relative(), _encode_scaled(), _encode_absolute() (shared logic)
4. Update encode_observation() to dispatch
5. Update get_observation_dim()
6. Verify no regressions
7. Commit

**Expected Results:**
- Continuous1D relative: [num_agents, 1]
- Continuous1D scaled: [num_agents, 2] (pos + 1 range)
- Continuous2D relative: [num_agents, 2]
- Continuous2D scaled: [num_agents, 4] (pos + 2 ranges)
- Continuous3D relative: [num_agents, 3]
- Continuous3D scaled: [num_agents, 6] (pos + 3 ranges)

---

### ~~Task 1.7: Environment action_space_size Integration~~ ✅ DONE IN PHASE 5B

**Note:** Environment integration with `substrate.action_space_size` property was implemented in Phase 5B as part of the action_space_size work.

**Implementation:** See `docs/plans/2025-11-06-action-space-size-property.md`

**Status:** ✅ Complete
- Environment updated to use `self.action_dim = self.substrate.action_space_size`
- Hardcoded if/elif chain removed from vectorized_env.py (lines 248-261)
- Single source of truth for action space sizing
- All integration tests passing

**Verification:** Confirm Phase 5B environment integration is complete:
```bash
pytest tests/test_townlet/integration/ -k "environment and action" -v
```

**Phase 5C Action:** No additional environment changes needed. Focus on substrate observation encoding only.

---

### Task 1.8: Update Config Examples and Documentation (30 min)

**Context:** Update all example configs and documentation to include observation_encoding parameter.

**Files:**
- Modify: `configs/templates/substrate.yaml`
- Modify: All configs in `configs/L*/substrate.yaml`
- Modify: `CLAUDE.md` (documentation)

#### Step 1.8.1: Update substrate config template

**Modify:** `configs/templates/substrate.yaml`

Add observation_encoding documentation:

```yaml
# Grid substrate example
type: "grid"
grid:
  topology: "square"  # or "cubic"
  width: 8
  height: 8
  depth: 3  # Required for cubic
  boundary: "clamp"  # clamp | wrap | bounce | sticky
  distance_metric: "manhattan"  # manhattan | euclidean | chebyshev

  # NEW: Phase 5C - Position encoding strategy
  observation_encoding: "relative"  # relative | scaled | absolute
  # - relative: Normalized [0,1] coordinates (default, backward compatible)
  # - scaled: Normalized [0,1] + range metadata (for heterogeneous dimensions)
  # - absolute: Raw unnormalized coordinates (for semantic meaning)

# Continuous substrate example
type: "continuous"
continuous:
  dimensions: 2  # 1-3
  bounds:
    - [0.0, 10.0]  # X dimension
    - [0.0, 10.0]  # Y dimension
  boundary: "clamp"
  movement_delta: 0.5
  interaction_radius: 1.0
  distance_metric: "euclidean"  # euclidean | manhattan

  # NEW: Phase 5C - Position encoding strategy
  observation_encoding: "relative"  # relative | scaled | absolute
```

#### Step 1.8.2: Update all curriculum level configs

**Modify:** All configs in `configs/L*/substrate.yaml`

Add `observation_encoding: "relative"` to preserve existing behavior:

```bash
for config in configs/L*/substrate.yaml; do
  # Add observation_encoding field to grid section
  # Use sed or manual editing
done
```

#### Step 1.8.3: Update CLAUDE.md documentation

**Modify:** `CLAUDE.md`

Add Phase 5C section in "Configuration System" (around line 150):

```markdown
### Observation Encoding (Phase 5C)

**Three encoding strategies** for position observations:

**relative** (default, backward compatible):
- Normalized coordinates [0, 1]
- Grid2D: 2 dims, Grid3D: 3 dims, Continuous: N dims
- Network learns: "I'm 50% across X axis, 30% across Y axis"

**scaled**:
- Normalized coordinates + range metadata
- Grid2D: 4 dims (x, y, width, height)
- Grid3D: 6 dims (x, y, z, width, height, depth)
- Network learns: "I'm 50% across a 100-unit range, 30% across a 50-unit range"

**absolute**:
- Raw unnormalized coordinates
- Grid2D: 2 dims, Grid3D: 3 dims
- Network learns: "I'm at absolute position [50, 15]"

**Configuration:**
```yaml
grid:
  # ...
  observation_encoding: "relative"  # or "scaled" or "absolute"
```
```

#### Step 1.8.4: Commit

```bash
git add configs/templates/substrate.yaml configs/L*/substrate.yaml CLAUDE.md
git commit -m "docs: update configs and docs for observation_encoding

- Add observation_encoding to substrate.yaml template
- Update all curriculum configs to use 'relative' (preserve behavior)
- Document three encoding modes in CLAUDE.md
- Part of Phase 5C Part 1

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.9: Integration Testing and Validation (60 min)

**Context:** Verify Part 1 works end-to-end with full training pipeline.

**Files:**
- Test: `tests/test_townlet/integration/test_phase5c_part1_integration.py` (NEW)

#### Step 1.9.1: Write integration test for full training with different encodings

**Create:** `tests/test_townlet/integration/test_phase5c_part1_integration.py`

```python
"""Integration tests for Phase 5C Part 1 - observation encoding retrofit."""
import pytest
import torch
from pathlib import Path
from townlet.demo.runner import DemoRunner


def test_training_with_relative_encoding(tmp_path):
    """Test full training pipeline with relative encoding."""
    # Create minimal config pack
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # substrate.yaml with relative encoding
    (config_dir / "substrate.yaml").write_text("""
version: "1.0"
description: "Test relative"
type: "grid"
grid:
  topology: "square"
  width: 3
  height: 3
  boundary: "clamp"
  distance_metric: "manhattan"
  observation_encoding: "relative"
""")

    # ... create other required configs (bars, affordances, training, etc.) ...

    # Run 10 episodes of training
    with DemoRunner(
        config_dir=config_dir,
        db_path=tmp_path / "test.db",
        checkpoint_dir=tmp_path / "checkpoints",
        max_episodes=10,
    ) as runner:
        runner.run()

    # Verify observations have correct dimensions
    assert runner.env.substrate.get_observation_dim() == 2  # Grid2D relative = 2


def test_training_with_scaled_encoding(tmp_path):
    """Test full training pipeline with scaled encoding."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # substrate.yaml with scaled encoding
    (config_dir / "substrate.yaml").write_text("""
version: "1.0"
description: "Test scaled"
type: "grid"
grid:
  topology: "square"
  width: 3
  height: 3
  boundary: "clamp"
  distance_metric: "manhattan"
  observation_encoding: "scaled"
""")

    # ... create other required configs ...

    with DemoRunner(
        config_dir=config_dir,
        db_path=tmp_path / "test.db",
        checkpoint_dir=tmp_path / "checkpoints",
        max_episodes=10,
    ) as runner:
        runner.run()

    # Verify observations have correct dimensions
    assert runner.env.substrate.get_observation_dim() == 4  # Grid2D scaled = 4


def test_backward_compatibility_missing_observation_encoding(tmp_path):
    """Test configs without observation_encoding still work (defaults to relative)."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # substrate.yaml WITHOUT observation_encoding field
    (config_dir / "substrate.yaml").write_text("""
version: "1.0"
description: "Legacy config"
type: "grid"
grid:
  topology: "square"
  width: 3
  height: 3
  boundary: "clamp"
  distance_metric: "manhattan"
""")

    # ... create other required configs ...

    with DemoRunner(
        config_dir=config_dir,
        db_path=tmp_path / "test.db",
        checkpoint_dir=tmp_path / "checkpoints",
        max_episodes=5,
    ) as runner:
        # Should default to relative
        assert runner.env.substrate.observation_encoding == "relative"
        assert runner.env.substrate.get_observation_dim() == 2
```

**Run:** `uv run pytest tests/test_townlet/integration/test_phase5c_part1_integration.py -v`

**Expected:** PASS

#### Step 1.9.2: Run full test suite

**Run:** `uv run pytest tests/test_townlet/ -v --cov=townlet.substrate --cov-report=term-missing`

**Expected:** PASS with good coverage

#### Step 1.9.3: Commit

```bash
git add tests/test_townlet/integration/test_phase5c_part1_integration.py
git commit -m "test: add Phase 5C Part 1 integration tests

- Test training with relative/scaled/absolute encodings
- Test backward compatibility (missing observation_encoding)
- Verify full pipeline works with new config parameter
- Part of Phase 5C Part 1

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Part 2: N-Dimensional Substrates (12-14 hours)

**Detailed Implementation Plan**: See `/home/john/hamlet/docs/plans/2025-11-06-phase5c-part2-detailed.md`

**Overview**: Part 2 adds GridND (4D-100D grids) and ContinuousND (4D+ continuous spaces) substrates for high-dimensional abstract state space research.

### Task 2.1: Implement GridND Base Class (120 min)

**What**: N-dimensional grid substrate (N≥4) with configurable encoding

**Key Implementation Points**:
- Dimension storage: `dimension_sizes: list[int]` instead of width/height/depth
- All boundary modes generalize via per-dimension loops (clamp/wrap/bounce/sticky)
- Distance metrics already dimension-agnostic (sum over dim=-1)
- Observation encoding: relative (N dims), scaled (2N dims), absolute (N dims)
- Neighbors: 2N cardinal directions (±1 per dimension)
- Warnings: N≥10 (action space size), get_all_positions() > 100K (memory)

**Test Dimensions**: 4D (core), 7D (mid-range), 10D (warning threshold), 20D (high-N stress)

**See detailed plan for 12-step TDD implementation** (lines 41-337 in part2-detailed.md)

---

### Task 2.2: Implement ContinuousND Class (90 min)

**What**: Remove `le=3` constraint from ContinuousSubstrate, support 4D-100D continuous spaces

**Key Implementation Points**:
- ContinuousSubstrate already uses `dimensions` parameter (just constrained to 1-3)
- All methods already generalized (loops over `range(self.dimensions)`)
- Main changes: Remove constraint, update coordinate semantics for N>3
- Warning: N≥10 (high-dimensional continuous spaces challenging to train)

**See detailed plan for TDD implementation** (part2-detailed.md)

---

### Task 2.3: Add GridND/ContinuousND to Config Schema (45 min)

**What**: Add GridNDConfig class, extend ContinuousConfig for N>3

**Strategy**:
- **GridND**: New config class `GridNDConfig` with `type: "gridnd"` (separate from GridConfig)
- **ContinuousND**: Extend existing ContinuousConfig by changing `dimensions: int = Field(ge=1, le=3)` → `le=100`

**Validation**: Pydantic validators for dimension ranges, bounds consistency, warning emissions

**See detailed plan for config schema design** (part2-detailed.md)

---

### Task 2.4: Update Factory for GridND/ContinuousND (30 min)

**What**: Add factory branches for gridnd and dimensions>3 continuous

**Implementation**:
```python
# In SubstrateFactory.build()
elif config.type == "gridnd":
    return GridNDSubstrate(
        dimension_sizes=config.gridnd.dimension_sizes,
        boundary=config.gridnd.boundary,
        distance_metric=config.gridnd.distance_metric,
        observation_encoding=config.gridnd.observation_encoding,
    )
```

**See detailed plan for factory integration** (part2-detailed.md)

---

### Task 2.5: Integration Testing for N-Dimensional Substrates (90 min)

**What**: Verify GridND/ContinuousND work end-to-end in training pipeline

**Tests**:
- Training with 4D GridND (full config pack)
- Training with 7D ContinuousND
- Action space sizing verification (9 actions for 4D, 15 for 7D)
- Observation dimensions correct for all encodings
- Boundary handling works in N dimensions

**See detailed plan for integration tests** (part2-detailed.md)

---

### Task 2.6: Documentation and Examples (60 min)

**What**: Document N-dimensional substrates with usage examples

**Content**:
- Add GridND/ContinuousND section to CLAUDE.md
- Create `configs/templates/substrate_nd.yaml` with examples
- Document pedagogical value (abstract state spaces, transfer learning)
- Update design doc status to "IMPLEMENTED"

**See detailed plan for documentation** (part2-detailed.md)

---

## Part 2 Summary

**Research Findings** (from Loop 3):
1. **GridND patterns**: All Grid2D/3D methods generalize via dimension loops
2. **ContinuousND patterns**: Already N-dimensional, just constrained to 1-3
3. **Config strategy**: New GridNDConfig class, extend ContinuousConfig
4. **Testing strategy**: 4D, 7D, 10D, 20D test dimensions with property-based tests

**Key Insight**: Most implementation is straightforward generalization from 2D/3D patterns. Main complexity is combinatorial explosion management (get_all_positions, observation windows).

---

## Final Validation

### Task 3.1: Full System Integration Test (60 min)

**Run complete test suite:**

```bash
# All unit tests
uv run pytest tests/test_townlet/unit/ -v

# All Phase 5 tests
uv run pytest tests/test_townlet/phase5/ -v

# All integration tests
uv run pytest tests/test_townlet/integration/ -v

# Coverage report
uv run pytest --cov=townlet.substrate --cov-report=html
```

**Expected:** All tests pass, coverage > 90%

### Task 3.2: Training Validation with All Encodings

**Run training with different encodings:**

```bash
# Test relative encoding (backward compatible)
uv run scripts/run_demo.py --config configs/L0_0_minimal

# Test scaled encoding (manually update substrate.yaml)
# ... validate training works ...

# Test GridND 7D (create test config)
# ... validate training works ...
```

### Task 3.3: Final Documentation Review

**Verify documentation is complete:**
- [ ] CLAUDE.md updated with Phase 5C details
- [ ] All config templates include observation_encoding
- [ ] Implementation plan marked complete
- [ ] Design document updated with "IMPLEMENTED" status

---

## Success Criteria

**Phase 5C Part 1 Complete:**
- [x] action_space_size property in base class
- [x] observation_encoding parameter in all configs
- [x] Grid2D/3D support relative/scaled/absolute encoding
- [x] Continuous1D/2D/3D support relative/scaled/absolute encoding
- [x] Environment uses substrate.action_space_size
- [x] All existing tests pass (backward compatible)
- [x] Documentation updated

**Phase 5C Part 2 Complete:**
- [x] GridND class (4D-arbitrary, warnings at N≥10)
- [x] ContinuousND class (4D-arbitrary, warning at N≥100)
- [x] Config schema supports N-dimensional substrates
- [x] Factory builds GridND/ContinuousND
- [x] Integration tests pass for 4D, 7D, 10D
- [x] Documentation with examples

---

## Commit Message Templates

**Feature commits:**
```
feat(substrate): add [feature]

- [What was added]
- [Why it matters]
- Part of Phase 5C Part [1|2]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Test commits:**
```
test: add [test coverage]

- [What tests cover]
- [Edge cases tested]
- Part of Phase 5C Part [1|2]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Documentation commits:**
```
docs: update [documentation]

- [What was documented]
- [Examples added]
- Part of Phase 5C Part [1|2]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Execution Options

**Plan saved to:** `docs/plans/2025-11-06-phase5c-implementation.md`

**Two execution options:**

**1. Subagent-Driven (this session)**
- Stay in this session
- Use superpowers:subagent-driven-development skill
- Dispatch fresh subagent per task
- Code review between tasks
- Fast iteration with quality gates

**2. Parallel Session (separate)**
- Open new session
- Use superpowers:executing-plans skill
- Batch execution with checkpoints
- Manual review between parts

Which approach would you like to use?
