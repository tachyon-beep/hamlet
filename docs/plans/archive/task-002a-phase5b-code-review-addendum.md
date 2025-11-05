# Phase 5B Code Review Addendum - Remaining Items

**Date**: 2025-11-05
**Status**: To be integrated into main Phase 5B plan
**Source**: Code review agent feedback

This document lists ALL remaining code review feedback items not yet incorporated into the Phase 5B plan.

---

## Summary of Remaining Items

### Already Integrated ✅
1. ✅ Phase 5 position hardcoding (Task 5B.0)
2. ✅ Grid3D observation encoding (normalized coordinates)
3. ✅ Missing position_dtype property (added to all substrate classes)
4. ✅ Continuous get_neighbors() error handling (raises NotImplementedError)

### To Be Added ⚠️

**Important Issues (5 items)**
- [ ] 5. Action delta mapping should return float
- [ ] 6. Config validation gaps (interaction_radius vs movement_delta)
- [ ] 7. Dtype isolation tests
- [ ] 8. Observation dim network compatibility test
- [ ] 9. Checkpoint version increment to 4

**Minor Issues (5 items)**
- [ ] 10. Config template files in deliverables
- [ ] 11. Bounce boundary specific tests
- [ ] 12. Frontend graceful degradation
- [ ] 13. Performance benchmarks
- [ ] 14. Audit error messages for examples

**Missing Items (2 items)**
- [ ] 15. Network observation dim validation
- [ ] 16. supports_get_all_positions() method

**Consistency Issues (2 items)**
- [ ] 17. Sticky boundary for continuous
- [ ] 18. Distance metric option for continuous

---

## IMPORTANT ISSUES (Must Add)

### Issue 5: Action Delta Mapping Returns Float

**Problem**: Current plan has `_action_to_deltas()` return `torch.long`, then continuous substrates cast to float. Inconsistent.

**Location**: Task 5B.2, Step 5 (Update action deltas)

**Better Approach**: Return float deltas, substrates cast to their dtype as needed.

**Add to Step 5**:

```python
# src/townlet/environment/vectorized_env.py

def _action_to_deltas(self, actions: torch.Tensor) -> torch.Tensor:
    """Map action indices to movement deltas.

    Returns float deltas. Substrates cast to their dtype as needed.

    Returns:
        [num_agents, 3] tensor of float deltas (padded to max dimension)
    """
    num_agents = actions.shape[0]
    # Return float32 (substrates will cast to long if needed)
    deltas = torch.zeros((num_agents, 3), dtype=torch.float32, device=self.device)

    # MOVE_X_NEGATIVE (0)
    deltas[actions == 0, 0] = -1.0
    # MOVE_X_POSITIVE (1)
    deltas[actions == 1, 0] = 1.0
    # MOVE_Y_NEGATIVE (2)
    deltas[actions == 2, 1] = -1.0
    # MOVE_Y_POSITIVE (3)
    deltas[actions == 3, 1] = 1.0
    # MOVE_Z_POSITIVE (4)
    deltas[actions == 4, 2] = 1.0
    # MOVE_Z_NEGATIVE (5)
    deltas[actions == 5, 2] = -1.0
    # INTERACT (6): no movement (already zeros)

    return deltas
```

**Update Grid substrates to cast to long**:

```python
# grid2d.py, grid3d.py
def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Apply movement deltas with boundary handling."""
    # Cast deltas to long for grid substrates
    new_positions = positions + deltas.long()
    # ... rest of logic
```

**Update Continuous substrates** (no cast needed):

```python
# continuous.py
def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Apply continuous movement with boundary handling."""
    # Deltas already float, no cast needed
    new_positions = positions + deltas.float()  # Ensure float (defensive)
    # ... rest of logic
```

**Effort**: Already in Step 5 (just update implementation details)

---

### Issue 6: Config Validation Gaps

**Problem**: ContinuousConfig doesn't validate interaction_radius vs movement_delta relationship.

**Location**: Task 5B.2, Step 2 (Update config schema)

**Add to ContinuousConfig validation**:

```python
# src/townlet/substrate/config.py

class ContinuousConfig(BaseModel):
    # ... existing fields ...

    @model_validator(mode="after")
    def validate_bounds_match_dimensions(self) -> "ContinuousConfig":
        """Validate bounds and interaction parameters."""
        # Existing bounds validation
        if len(self.bounds) != self.dimensions:
            raise ValueError(...)

        for i, (min_val, max_val) in enumerate(self.bounds):
            if min_val >= max_val:
                raise ValueError(
                    f"Bound {i} invalid: min ({min_val}) must be < max ({max_val})"
                )

            # NEW: Check space is large enough for interaction
            range_size = max_val - min_val
            if range_size < self.interaction_radius:
                raise ValueError(
                    f"Dimension {i} range ({range_size}) < interaction_radius ({self.interaction_radius}). "
                    f"Space too small for affordance interaction."
                )

        # NEW: Warn if interaction_radius < movement_delta
        if self.interaction_radius < self.movement_delta:
            import warnings
            warnings.warn(
                f"interaction_radius ({self.interaction_radius}) < movement_delta ({self.movement_delta}). "
                f"Agent may step over affordances without interaction. "
                f"This may be intentional for challenge, but verify configuration."
            )

        return self
```

**Effort**: Add to existing Step 2 validation section (+15 min)

---

### Issue 7: Dtype Isolation Tests

**Problem**: No test verifying Grid3D/Continuous don't contaminate each other's dtypes.

**Location**: Task 5B.1, Step 5 (Add unit tests)

**Add to test_substrate_grid3d.py**:

```python
def test_grid3d_dtype_isolation_from_continuous():
    """Grid3D should maintain long dtype independent of continuous substrates.

    This test ensures creating a continuous substrate (float32) doesn't
    affect grid substrates (long).
    """
    from townlet.substrate.continuous import Continuous2DSubstrate

    # Create continuous substrate (float positions)
    continuous = Continuous2DSubstrate(
        min_x=0.0, max_x=10.0,
        min_y=0.0, max_y=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )
    continuous_positions = continuous.initialize_positions(10, torch.device("cpu"))
    assert continuous_positions.dtype == torch.float32

    # Create grid3d substrate (long positions)
    grid3d = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
    grid3d_positions = grid3d.initialize_positions(10, torch.device("cpu"))

    # Should NOT be contaminated by continuous
    assert grid3d_positions.dtype == torch.long
    assert grid3d.position_dtype == torch.long
```

**Add to test_substrate_continuous.py**:

```python
def test_continuous_dtype_isolation_from_grid():
    """Continuous should maintain float32 dtype independent of grid substrates."""
    from townlet.substrate.grid3d import Grid3DSubstrate

    # Create grid substrate (long positions)
    grid = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
    grid_positions = grid.initialize_positions(10, torch.device("cpu"))
    assert grid_positions.dtype == torch.long

    # Create continuous substrate (float positions)
    continuous = Continuous2DSubstrate(
        min_x=0.0, max_x=10.0,
        min_y=0.0, max_y=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )
    continuous_positions = continuous.initialize_positions(10, torch.device("cpu"))

    # Should NOT be contaminated by grid
    assert continuous_positions.dtype == torch.float32
    assert continuous.position_dtype == torch.float32
```

**Effort**: Add to existing unit tests (+20 min)

---

### Issue 8: Observation Dim Network Compatibility Test

**Problem**: No test verifying network input dim matches environment observation output.

**Location**: Task 5B.1, Step 6 (Add integration tests)

**Add to test_substrate_migration.py**:

```python
@pytest.mark.parametrize("config_name,expected_obs_dim", [
    ("L1_full_observability", None),  # Computed from config
    ("L1_3D_house", None),  # Should be 3 (normalized) + 8 + 15 + 4 = 30
    ("L1_continuous_2D", None),  # Should be 2 (normalized) + 8 + 15 + 4 = 29
])
def test_observation_dim_matches_network_input(tmp_path, config_name, expected_obs_dim):
    """Network input dim should match actual observation size."""
    from pathlib import Path
    from townlet.demo.runner import DemoRunner

    config_dir = Path(f"configs/{config_name}")

    with DemoRunner(
        config_dir=config_dir,
        db_path=tmp_path / "test.db",
        checkpoint_dir=tmp_path / "checkpoints",
        max_episodes=1,
    ) as runner:
        # Get actual observation from environment
        runner.env.reset()
        obs = runner.env._get_observations()
        actual_obs_dim = obs.shape[1]

        # Get network expected input dim
        network = runner.population.q_network
        # Assume first layer is input (network.fc1.in_features or similar)
        if hasattr(network, 'fc1'):
            network_input_dim = network.fc1.in_features
        elif hasattr(network, 'layers') and len(network.layers) > 0:
            network_input_dim = network.layers[0].in_features
        else:
            pytest.skip("Cannot determine network input dim")

        assert actual_obs_dim == network_input_dim, \
            f"Observation dim mismatch for {config_name}: " \
            f"env produces {actual_obs_dim}, network expects {network_input_dim}"
```

**Effort**: Add to integration tests (+30 min)

---

### Issue 9: Checkpoint Version Increment

**Problem**: Phase 5B changes position dtypes and dimensions, should bump checkpoint version.

**Location**: Task 5B.1 or 5B.2 (add new step)

**Add Step 7 to Task 5B.2: Update checkpoint version**

**Action**: Increment checkpoint version to distinguish Phase 5B format

**Modify**: `src/townlet/training/state.py`

```python
# Update version constant
CHECKPOINT_VERSION = 4  # Phase 5B: Grid3D + Continuous support

# Previous versions:
# Version 1: Original format
# Version 2: Added metrics
# Version 3: Phase 5 - substrate support, position_dim
# Version 4: Phase 5B - Grid3D, Continuous, position_dtype
```

**Add validation in checkpoint loading**:

```python
def load_checkpoint(self, checkpoint_path: Path) -> PopulationCheckpoint:
    """Load checkpoint with version validation."""
    checkpoint = torch.load(checkpoint_path)

    if checkpoint["version"] < 4:
        # Check if substrate requires version 4
        if self.env.substrate.position_dim == 3:
            raise ValueError(
                f"Checkpoint version {checkpoint['version']} does not support "
                f"3D substrates (position_dim=3). "
                f"Phase 5B requires version 4 checkpoints. "
                f"Please retrain from scratch."
            )

        if hasattr(self.env.substrate, 'movement_delta'):
            raise ValueError(
                f"Checkpoint version {checkpoint['version']} does not support "
                f"continuous substrates. "
                f"Phase 5B requires version 4 checkpoints. "
                f"Please retrain from scratch."
            )

    # ... rest of loading logic
```

**Effort**: 30 minutes

---

## MINOR ISSUES (Should Add)

### Issue 10: Config Template Files

**Problem**: Plan mentions creating templates but doesn't specify as deliverable.

**Location**: Task 5B.1, Task 5B.2 - add explicit steps

**Add Step 7 to Task 5B.1: Create Grid3D config template**

**Create**: `configs/templates/substrate_grid3d.yaml`

```yaml
version: "1.0"
description: "3D cubic grid substrate template"

type: "grid"

grid:
  topology: "cubic"

  # Grid dimensions
  width: 8          # Number of cells in X dimension
  height: 8         # Number of cells in Y dimension
  depth: 3          # Number of cells in Z dimension (floors/layers)

  # Boundary handling
  # clamp: Hard walls (position clamped to edges)
  # wrap: Toroidal wraparound (Pac-Man in 3D)
  # bounce: Elastic reflection (agent bounces back)
  # sticky: Sticky walls (agent stays in place when hitting boundary)
  boundary: "clamp"

  # Distance metric
  # manhattan: L1 norm, |x1-x2| + |y1-y2| + |z1-z2| (matches 6-directional movement)
  # euclidean: L2 norm, sqrt((x1-x2)² + (y1-y2)² + (z1-z2)²)
  # chebyshev: L∞ norm, max(|x1-x2|, |y1-y2|, |z1-z2|)
  distance_metric: "manhattan"

# Use cases:
# - Multi-story buildings (house with 3 floors)
# - Apartment complexes (agents on different floors)
# - 3D navigation challenges
# - Vertical resource distribution

# Observation encoding:
# - Normalized coordinates [0, 1] for each dimension
# - Prevents dimension explosion (3 dims instead of width*height*depth)
# - Network learns spatial relationships

# Example 3-story house:
# width: 8, height: 8, depth: 3
# Ground floor (z=0): Kitchen, Living Room
# 2nd floor (z=1): Bedrooms
# Attic (z=2): Storage
```

**Add Step 9 to Task 5B.2: Create Continuous config templates**

**Create**: `configs/templates/substrate_continuous_2d.yaml`

```yaml
version: "1.0"
description: "Continuous 2D space template"

type: "continuous"

continuous:
  dimensions: 2

  # Bounds for each dimension [(min, max), ...]
  bounds:
    - [0.0, 10.0]  # X dimension: 0 to 10 units
    - [0.0, 10.0]  # Y dimension: 0 to 10 units

  # Boundary handling
  # clamp: Hard walls (position clamped to bounds)
  # wrap: Toroidal wraparound
  # bounce: Elastic reflection
  # sticky: Stay in place when hitting boundary
  boundary: "clamp"

  # Movement granularity (how far discrete actions move agent)
  # Smaller = finer control, Larger = faster movement
  # Example: 0.5 means MOVE_X_POSITIVE moves agent 0.5 units right
  movement_delta: 0.5

  # Interaction radius (distance threshold for affordance interaction)
  # Agent must be within this radius to interact
  # Recommendation: interaction_radius >= movement_delta
  # (prevents agent from "stepping over" affordances)
  interaction_radius: 0.8

  # Distance metric (optional, default: euclidean)
  # euclidean: sqrt((x1-x2)² + (y1-y2)²)
  # manhattan: |x1-x2| + |y1-y2|
  distance_metric: "euclidean"

# Use cases:
# - Robotics simulation (smooth movement)
# - Continuous control problems
# - Teaching discrete vs continuous control
# - Smooth navigation challenges

# Observation encoding:
# - Normalized positions [0, 1] for each dimension
# - Same encoding as Grid3D (consistent representation)

# Pedagogical notes:
# - Positions are float (torch.float32), not int
# - Proximity is radius-based, not exact position match
# - Discrete actions still used (pedagogical progression)
```

**Effort**: 30 minutes total

---

### Issue 11: Bounce Boundary Specific Tests

**Problem**: Bounce logic complex, tests only check "within bounds", not exact bounce positions.

**Location**: Task 5B.1, Step 5 (unit tests)

**Add specific bounce tests to test_substrate_grid3d.py**:

```python
def test_grid3d_bounce_exact_positions():
    """Test exact bounce positions for correctness."""
    substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="bounce")

    # Test case 1: Bounce from lower boundary
    # Position [0, 0, 0], delta [-2, 0, 0]
    # Expected: reflect from X=0 → new_pos = -(-2) = 2
    positions = torch.tensor([[0, 0, 0]], dtype=torch.long)
    deltas = torch.tensor([[-2, 0, 0]], dtype=torch.long)
    new_positions = substrate.apply_movement(positions, deltas)
    assert (new_positions == torch.tensor([[2, 0, 0]])).all(), \
        f"Expected [2, 0, 0], got {new_positions}"

    # Test case 2: Bounce from upper boundary
    # Position [7, 7, 2], delta [2, 0, 0]
    # Expected: reflect from X=8 → new_pos = 2*(8-1) - 9 = 5
    positions = torch.tensor([[7, 7, 2]], dtype=torch.long)
    deltas = torch.tensor([[2, 0, 0]], dtype=torch.long)
    new_positions = substrate.apply_movement(positions, deltas)
    assert (new_positions == torch.tensor([[5, 7, 2]])).all(), \
        f"Expected [5, 7, 2], got {new_positions}"

    # Test case 3: Bounce in Z dimension
    # Position [4, 4, 0], delta [0, 0, -3]
    # Expected: reflect from Z=0 → new_pos = -(-3) = 3, clamp to 2
    positions = torch.tensor([[4, 4, 0]], dtype=torch.long)
    deltas = torch.tensor([[0, 0, -3]], dtype=torch.long)
    new_positions = substrate.apply_movement(positions, deltas)
    # Reflection would give 3, but clamped to max depth (2)
    assert (new_positions == torch.tensor([[4, 4, 2]])).all(), \
        f"Expected [4, 4, 2], got {new_positions}"
```

**Add to test_substrate_continuous.py**:

```python
def test_continuous2d_bounce_exact_positions():
    """Test exact bounce positions for continuous space."""
    substrate = Continuous2DSubstrate(
        min_x=0.0, max_x=10.0,
        min_y=0.0, max_y=10.0,
        boundary="bounce",
        movement_delta=0.5,
        interaction_radius=0.8,
    )

    # Test: Bounce from lower bound
    # Position [0.2, 0.3], delta [-0.5, -0.5]
    # X: 0.2 - 0.5 = -0.3 → reflect: 2*0 - (-0.3) = 0.3
    # Y: 0.3 - 0.5 = -0.2 → reflect: 2*0 - (-0.2) = 0.2
    positions = torch.tensor([[0.2, 0.3]], dtype=torch.float32)
    deltas = torch.tensor([[-0.5, -0.5]], dtype=torch.float32)
    new_positions = substrate.apply_movement(positions, deltas)
    expected = torch.tensor([[0.3, 0.2]], dtype=torch.float32)
    assert torch.allclose(new_positions, expected, atol=1e-5), \
        f"Expected {expected}, got {new_positions}"
```

**Effort**: 30 minutes

---

### Issue 12: Frontend Graceful Degradation

**Problem**: Plan defers frontend to Phase 7, but should specify what happens NOW if user runs inference with 3D/continuous.

**Location**: Add new section to documentation

**Add to Phase 5B plan**:

#### Frontend Graceful Degradation (Phase 5B Interim Solution)

**Goal**: Prevent crashes when using live_inference.py with Grid3D or Continuous substrates.

**Interim Solution** (30 minutes):

**Modify**: `townlet/demo/live_inference.py`

```python
def format_positions_for_frontend(substrate, positions):
    """Format positions for frontend display with graceful degradation.

    Args:
        substrate: SpatialSubstrate instance
        positions: Agent positions tensor

    Returns:
        2D positions suitable for frontend rendering
    """
    if substrate.position_dim == 2:
        # 2D: return as-is
        return positions

    elif substrate.position_dim == 3:
        # 3D: Project to 2D (top-down view, ignore Z)
        print("[WARNING] 3D substrate detected. Showing top-down view (Z-axis ignored).")
        print(f"[INFO] Agents are on floors Z=0 to Z={substrate.depth-1}")
        return positions[:, :2]  # Just X, Y

    elif substrate.position_dim == 1:
        # 1D: Map to 2D line
        print("[WARNING] 1D substrate detected. Rendering as horizontal line.")
        # Create Y=0 for all positions
        y_coords = torch.zeros_like(positions)
        return torch.cat([positions, y_coords], dim=1)

    elif substrate.position_dim == 0:
        # Aspatial: No positions
        print("[WARNING] Aspatial substrate (no positioning). Using placeholder positions.")
        # Random positions for visualization
        return torch.rand((positions.shape[0], 2)) * 8

    # Continuous: Check if float positions
    if positions.dtype == torch.float32:
        print("[INFO] Continuous substrate detected. Rounding positions for display.")
        # Round to nearest grid cell for rendering
        return positions[:, :2].round()

    return positions
```

**Update live_inference main loop**:

```python
# In main loop, before sending to frontend
formatted_positions = format_positions_for_frontend(env.substrate, env.positions)

# Send formatted positions to frontend
websocket.send({
    "positions": formatted_positions.tolist(),
    "substrate_type": type(env.substrate).__name__,
    "position_dim": env.substrate.position_dim,
    # ... rest of data
})
```

**Frontend update** (if time permits, optional):

```javascript
// frontend/src/components/GridVisualization.vue
// Add warning banner for non-2D substrates
if (substrateType === 'Grid3DSubstrate') {
  showWarning('3D substrate: Showing top-down view (Z-axis hidden)');
} else if (substrateType.includes('Continuous')) {
  showWarning('Continuous substrate: Positions rounded for display');
}
```

**Effort**: 30 minutes

**Note**: Full 3D/continuous rendering deferred to Phase 7.

---

### Issue 13: Performance Benchmarks

**Problem**: Checklist says "no performance regression" but no test.

**Location**: Add new validation test

**Create**: `tests/test_townlet/performance/test_substrate_performance.py` (NEW)

```python
"""Performance benchmarks for substrate operations."""

import time
import torch
import pytest
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate
from townlet.substrate.continuous import Continuous2DSubstrate, Continuous3DSubstrate


def benchmark_substrate(substrate, num_agents=1000, num_iterations=1000, device="cuda"):
    """Benchmark substrate performance."""
    positions = substrate.initialize_positions(num_agents, torch.device(device))
    deltas = torch.randn((num_agents, substrate.position_dim), device=device)

    # Warmup
    for _ in range(10):
        substrate.apply_movement(positions, deltas)

    # Benchmark
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()

    for _ in range(num_iterations):
        new_positions = substrate.apply_movement(positions, deltas)

    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.time() - start

    ops_per_sec = (num_agents * num_iterations) / elapsed
    return elapsed, ops_per_sec


@pytest.mark.benchmark
def test_substrate_performance_comparison():
    """Compare Grid2D vs Grid3D vs Continuous performance."""

    substrates = {
        "Grid2D (8×8)": Grid2DSubstrate(8, 8, "clamp"),
        "Grid3D (8×8×3)": Grid3DSubstrate(8, 8, 3, "clamp"),
        "Continuous2D": Continuous2DSubstrate(
            0.0, 10.0, 0.0, 10.0, "clamp", 0.5, 0.8
        ),
        "Continuous3D": Continuous3DSubstrate(
            0.0, 10.0, 0.0, 10.0, 0.0, 10.0, "clamp", 0.5, 0.8
        ),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    for name, substrate in substrates.items():
        elapsed, ops_per_sec = benchmark_substrate(substrate, device=device)
        results[name] = {
            "elapsed": elapsed,
            "ops_per_sec": ops_per_sec,
        }
        print(f"{name}: {elapsed:.3f}s ({ops_per_sec/1e6:.2f}M ops/sec)")

    # Acceptance criteria: Grid3D should be ≤1.5x slower than Grid2D
    grid2d_ops = results["Grid2D (8×8)"]["ops_per_sec"]
    grid3d_ops = results["Grid3D (8×8×3)"]["ops_per_sec"]
    slowdown = grid2d_ops / grid3d_ops

    assert slowdown <= 1.5, \
        f"Grid3D too slow: {slowdown:.2f}x slower than Grid2D (max allowed: 1.5x)"

    print(f"\n✓ Grid3D slowdown: {slowdown:.2f}x (acceptable)")


@pytest.mark.benchmark
def test_observation_encoding_performance():
    """Benchmark observation encoding (normalized vs one-hot)."""

    # Grid3D with normalized encoding
    grid3d = Grid3DSubstrate(8, 8, 3, "clamp")
    positions_3d = grid3d.initialize_positions(1000, torch.device("cpu"))

    start = time.time()
    for _ in range(1000):
        obs = grid3d.encode_observation(positions_3d)
    elapsed_normalized = time.time() - start

    print(f"Normalized encoding (3 dims): {elapsed_normalized:.3f}s")
    print(f"Observation shape: {obs.shape}")

    # Hypothetical one-hot encoding (for comparison)
    # Would create 8*8*3=192 dimensional observations
    # Not implemented, but we can estimate overhead

    assert obs.shape[1] == 3, "Should produce 3-dim observations"
```

**Run benchmarks**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run pytest tests/test_townlet/performance/test_substrate_performance.py -v -m benchmark
```

**Effort**: 1 hour

---

## MISSING ITEMS

### Issue 15: Network Observation Dim Validation

**Already covered in Issue 8** ✅

---

### Issue 16: supports_get_all_positions() Method

**Problem**: Code uses `isinstance()` check for continuous substrates. Better to use abstract method.

**Location**: Task 5B.2, Step 4 (Handle affordance placement)

**Add to SpatialSubstrate interface**:

```python
# src/townlet/substrate/base.py

@abstractmethod
def supports_enumerable_positions(self) -> bool:
    """Whether this substrate has finite enumerable positions.

    Returns:
        True: Substrate has finite positions (use get_all_positions())
        False: Substrate has infinite positions (use random sampling)

    Examples:
        Grid substrates: True (finite grid cells)
        Continuous: False (infinite float positions)
        Aspatial: False (no positions at all)
    """
    pass
```

**Implement in each substrate**:

```python
# Grid2DSubstrate, Grid3DSubstrate
def supports_enumerable_positions(self) -> bool:
    return True

# ContinuousSubstrate
def supports_enumerable_positions(self) -> bool:
    return False

# AspatialSubstrate
def supports_enumerable_positions(self) -> bool:
    return False
```

**Update affordance randomization**:

```python
# src/townlet/environment/vectorized_env.py

def randomize_affordance_positions(self) -> None:
    """Randomize affordance positions using substrate."""

    # Aspatial substrates don't have positions
    if self.substrate.position_dim == 0:
        self.affordance_positions = torch.zeros(
            (len(self.affordances), 0),
            dtype=self.substrate.position_dtype,
            device=self.device
        )
        return

    # Check if substrate supports enumerable positions
    if self.substrate.supports_enumerable_positions():
        # Grid substrates: shuffle all positions
        all_positions = self.substrate.get_all_positions()

        if len(all_positions) < len(self.affordances):
            raise ValueError(
                f"Not enough positions for affordances. "
                f"Substrate has {len(all_positions)} positions, "
                f"but {len(self.affordances)} affordances enabled."
            )

        random.shuffle(all_positions)
        selected = all_positions[: len(self.affordances)]

        self.affordance_positions = torch.tensor(
            selected,
            dtype=self.substrate.position_dtype,
            device=self.device
        )
    else:
        # Continuous/Aspatial: random sampling
        self.affordance_positions = self.substrate.initialize_positions(
            num_agents=len(self.affordances),
            device=self.device
        )
```

**Benefits**:
- More explicit than `isinstance()` check
- Easier to extend to new substrate types
- Self-documenting intent

**Effort**: 30 minutes

---

## CONSISTENCY ISSUES

### Issue 17: Sticky Boundary for Continuous

**Problem**: Grid supports sticky, Continuous doesn't. Inconsistent.

**Location**: Task 5B.2, Step 1 (Create ContinuousSubstrate)

**Add sticky boundary to Continuous**:

```python
# src/townlet/substrate/continuous.py

class ContinuousSubstrate(SpatialSubstrate):
    def __init__(
        self,
        dimensions: int,
        bounds: list[tuple[float, float]],
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],  # Add sticky
        movement_delta: float,
        interaction_radius: float,
    ):
        # ... validation ...
        if boundary not in ("clamp", "wrap", "bounce", "sticky"):
            raise ValueError(f"Unknown boundary mode: {boundary}")
        # ...

    def apply_movement(
        self, positions: torch.Tensor, deltas: torch.Tensor
    ) -> torch.Tensor:
        """Apply continuous movement with boundary handling."""
        new_positions = positions + deltas.float()

        for dim in range(self.dimensions):
            min_val, max_val = self.bounds[dim]

            if self.boundary == "clamp":
                new_positions[:, dim] = torch.clamp(
                    new_positions[:, dim], min_val, max_val
                )

            elif self.boundary == "wrap":
                # ... existing wrap logic ...

            elif self.boundary == "bounce":
                # ... existing bounce logic ...

            elif self.boundary == "sticky":
                # NEW: Stay in place if out of bounds
                out_of_bounds = (new_positions[:, dim] < min_val) | (
                    new_positions[:, dim] > max_val
                )
                new_positions[out_of_bounds, dim] = positions[out_of_bounds, dim]

        return new_positions
```

**Update config schema**:

```python
# src/townlet/substrate/config.py

class ContinuousConfig(BaseModel):
    boundary: Literal["clamp", "wrap", "bounce", "sticky"] = Field(
        ...,
        description="Boundary handling mode"
    )
```

**Effort**: 15 minutes

---

### Issue 18: Distance Metric for Continuous

**Problem**: Grid has manhattan/euclidean/chebyshev options. Continuous is always Euclidean. Inconsistent.

**Location**: Task 5B.2, Step 1 (Create ContinuousSubstrate)

**Add distance_metric parameter**:

```python
# src/townlet/substrate/continuous.py

class ContinuousSubstrate(SpatialSubstrate):
    def __init__(
        self,
        dimensions: int,
        bounds: list[tuple[float, float]],
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        movement_delta: float,
        interaction_radius: float,
        distance_metric: Literal["euclidean", "manhattan"] = "euclidean",  # NEW
    ):
        # ... validation ...
        if distance_metric not in ("euclidean", "manhattan"):
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        self.distance_metric = distance_metric

    def compute_distance(
        self, pos1: torch.Tensor, pos2: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance in continuous space."""
        if self.distance_metric == "euclidean":
            return torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1))
        elif self.distance_metric == "manhattan":
            return torch.abs(pos1 - pos2).sum(dim=-1)
```

**Update config schema**:

```python
# src/townlet/substrate/config.py

class ContinuousConfig(BaseModel):
    # ... existing fields ...

    distance_metric: Literal["euclidean", "manhattan"] = Field(
        default="euclidean",
        description=(
            "Distance calculation method. "
            "euclidean: sqrt((x1-x2)² + ...), "
            "manhattan: |x1-x2| + |y1-y2| + ..."
        )
    )
```

**Update Continuous1D/2D/3D constructors**:

```python
class Continuous2DSubstrate(ContinuousSubstrate):
    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        movement_delta: float,
        interaction_radius: float,
        distance_metric: Literal["euclidean", "manhattan"] = "euclidean",  # NEW
    ):
        super().__init__(
            dimensions=2,
            bounds=[(min_x, max_x), (min_y, max_y)],
            boundary=boundary,
            movement_delta=movement_delta,
            interaction_radius=interaction_radius,
            distance_metric=distance_metric,  # Pass through
        )
        # ... store individual bounds for convenience ...
```

**Update factory**:

```python
# src/townlet/substrate/factory.py

elif config.type == "continuous":
    if config.continuous.dimensions == 2:
        (min_x, max_x), (min_y, max_y) = config.continuous.bounds
        return Continuous2DSubstrate(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            boundary=config.continuous.boundary,
            movement_delta=config.continuous.movement_delta,
            interaction_radius=config.continuous.interaction_radius,
            distance_metric=config.continuous.distance_metric,  # NEW
        )
```

**Effort**: 30 minutes

---

## Integration Instructions

1. **Merge into main Phase 5B plan**: Incorporate these items into appropriate tasks
2. **Update effort estimates**: Add ~3-4 hours for all new items
3. **Revised total**: 18-22h → **21-26 hours**
4. **Priority order**:
   - Important issues first (5-9)
   - Consistency issues (17-18) - low effort, high value
   - Minor issues (10-13) - as time permits
   - Missing items (15-16) - already covered or low priority

---

## Summary Checklist

### Important (Must Do)
- [ ] 5. Return float deltas from _action_to_deltas()
- [ ] 6. Add config validation (interaction_radius warnings)
- [ ] 7. Add dtype isolation tests
- [ ] 8. Add observation dim network test
- [ ] 9. Increment checkpoint version to 4

### Consistency (High Value/Low Effort)
- [ ] 17. Add sticky boundary to Continuous
- [ ] 18. Add distance_metric to Continuous

### Minor (Nice to Have)
- [ ] 10. Create config template files
- [ ] 11. Add bounce position tests
- [ ] 12. Add frontend graceful degradation
- [ ] 13. Add performance benchmarks

### Missing (Optional)
- [ ] 16. Add supports_enumerable_positions() method

---

**Total Additional Effort**: 3-4 hours
**Revised Phase 5B Total**: 21-26 hours (from 18-22h)
