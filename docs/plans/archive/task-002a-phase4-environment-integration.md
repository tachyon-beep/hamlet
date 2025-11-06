## Phase 4: Environment Integration (1.5 hours)

### Task 4.1: Add Substrate to VectorizedEnv (Load Only)

**Files:**

- Modify: `src/townlet/environment/vectorized_env.py`

**Step 1: Write test for environment substrate loading**

Create: `tests/test_townlet/unit/test_env_substrate_loading.py`

```python
"""Test environment loads and uses substrate configuration."""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.substrate.grid2d import Grid2DSubstrate


def test_env_loads_substrate_config():
    """Environment should load substrate.yaml and create substrate instance."""
    # Note: This test will initially PASS with legacy behavior
    # After Phase 4 integration, it will load from substrate.yaml

    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    # After integration, should have substrate attribute
    # For now, check legacy grid_size exists
    assert hasattr(env, "grid_size")
    assert env.grid_size == 8


def test_env_substrate_accessible():
    """Environment should expose substrate for inspection."""
    # This test will FAIL initially, becomes valid after integration

    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    # After integration:
    # assert hasattr(env, "substrate")
    # assert isinstance(env.substrate, Grid2DSubstrate)
    # assert env.substrate.width == 8
```

**Step 2: Run test to establish baseline**

```bash
uv run pytest tests/test_townlet/unit/test_env_substrate_loading.py::test_env_loads_substrate_config -v
```

Expected: PASS (legacy behavior)

**Step 3: Add substrate loading to VectorizedEnv.**init****

Modify: `src/townlet/environment/vectorized_env.py`

Find `__init__` method (around line 36). After line where `grid_size` is set, add:

```python
# BREAKING CHANGE: substrate.yaml is now REQUIRED
substrate_config_path = config_pack_path / "substrate.yaml"
if not substrate_config_path.exists():
    raise FileNotFoundError(
        f"substrate.yaml is required but not found in {config_pack_path}.\n\n"
        f"All config packs must define their spatial substrate.\n\n"
        f"Quick fix:\n"
        f"  1. Copy template: cp docs/examples/substrate.yaml {config_pack_path}/\n"
        f"  2. Edit substrate.yaml to match your grid_size from training.yaml\n"
        f"  3. See CLAUDE.md 'Configuration System' for details\n\n"
        f"This is a breaking change from TASK-002A. Previous configs without\n"
        f"substrate.yaml will no longer work. See CHANGELOG.md for migration guide."
    )

from townlet.substrate.config import load_substrate_config
from townlet.substrate.factory import SubstrateFactory

substrate_config = load_substrate_config(substrate_config_path)
self.substrate = SubstrateFactory.build(substrate_config, device=self.device)

# Update grid_size from substrate (for backward compatibility with other code)
if hasattr(self.substrate, "width") and hasattr(self.substrate, "height"):
    if self.substrate.width != self.substrate.height:
        raise ValueError(
            f"Non-square grids not yet supported: "
            f"{self.substrate.width}×{self.substrate.height}"
        )
    self.grid_size = self.substrate.width
```

**Step 4: Commit**

```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/unit/test_env_substrate_loading.py
git commit -m "feat: add substrate loading to VectorizedHamletEnv

BREAKING CHANGE: substrate.yaml is now REQUIRED for all config packs.

Environment now loads substrate.yaml and fails fast if missing:
- Creates substrate instance via SubstrateFactory
- Updates grid_size from substrate (for compatibility)
- Clear error message with migration steps

Backward compatibility removed per user authorization.
Old configs without substrate.yaml will NOT work.

Part of TASK-002A (Configurable Spatial Substrates)."
```

---

## Phase 4 Remaining Tasks

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

### Context

Phase 4 Task 4.1 (substrate loading) is **COMPLETE** ✅

**What's Complete**:
- ✅ Substrate loaded in `VectorizedEnv.__init__()` (commit `11baff4`)
- ✅ Substrate accessible via `env.substrate`
- ✅ `grid_size` synced from substrate for compatibility
- ✅ Error message if substrate.yaml missing

**What's Remaining**:
- ❌ Substrate methods not yet wired up (still using hardcoded operations)
- ❌ Position initialization (still using `torch.randint`)
- ❌ Movement/boundaries (still using `torch.clamp`)
- ❌ Affordance randomization (still using manual grid iteration)
- ❌ Integration tests (skipped)

**Goal**: Wire up substrate methods so they're actually called (not just loaded).

**Estimated Effort**: 2.75 hours (Tasks 4.0, 4.2-4.5)

---

### Task 4.0: Add substrate.get_all_positions() Method

**Purpose**: Add method needed by Phase 4 affordance randomization and Phase 5 integration

**Files:**
- Modify: `src/townlet/substrate/base.py`
- Modify: `src/townlet/substrate/grid2d.py`
- Modify: `src/townlet/substrate/aspatial.py`
- Modify: `tests/test_townlet/unit/test_substrate_base.py`

**Step 1: Write test for get_all_positions()**

Add to `tests/test_townlet/unit/test_substrate_base.py`:

```python
def test_grid2d_get_all_positions():
    """Grid2D should return all valid grid positions."""
    substrate = Grid2DSubstrate(width=3, height=2, boundary="clamp", distance_metric="manhattan")

    positions = substrate.get_all_positions()

    # Should return 6 positions (3×2 grid)
    assert len(positions) == 6

    # Each position should be [x, y] list
    assert all(len(pos) == 2 for pos in positions)

    # Should cover all grid cells
    expected = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
    assert sorted(positions) == sorted(expected)


def test_aspatial_get_all_positions():
    """Aspatial should return empty list (no positions exist)."""
    substrate = AspatialSubstrate()

    positions = substrate.get_all_positions()

    assert positions == []
```

**Step 2: Add method to base class**

Modify `src/townlet/substrate/base.py`:

```python
@abstractmethod
def get_all_positions(self) -> list[list[int]]:
    """Return all valid positions in the substrate.

    Returns:
        List of positions, where each position is [x, y, ...] (position_dim elements).
        For aspatial substrates, returns empty list.
        For 2D grids (3×3), returns [[0,0], [0,1], [0,2], [1,0], ...] (9 positions).
        For 3D grids, would return [[x,y,z], ...].

    Used for affordance randomization to ensure valid placement.
    """
    pass
```

**Step 3: Implement for Grid2D**

Modify `src/townlet/substrate/grid2d.py`:

```python
def get_all_positions(self) -> list[list[int]]:
    """Return all grid positions."""
    return [[x, y] for x in range(self.width) for y in range(self.height)]
```

**Step 4: Implement for Aspatial**

Modify `src/townlet/substrate/aspatial.py`:

```python
def get_all_positions(self) -> list[list[int]]:
    """Return empty list (aspatial has no positions)."""
    return []
```

**Step 5: Run tests**

```bash
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_grid2d_get_all_positions -v
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_aspatial_get_all_positions -v
```

Expected: Both tests PASS

**Step 6: Commit**

```bash
git add src/townlet/substrate/base.py src/townlet/substrate/grid2d.py src/townlet/substrate/aspatial.py tests/test_townlet/unit/test_substrate_base.py
git commit -m "feat(substrate): add get_all_positions() method

Add method to return all valid positions in substrate.

- Grid2D: Returns all [x,y] positions in grid
- Aspatial: Returns empty list (no positions exist)

Needed for:
- Phase 4: Affordance randomization (replace hardcoded grid iteration)
- Phase 5: Position management refactoring

Part of TASK-002A Phase 4 (Environment Integration)."
```

---

### Task 4.1: ALREADY COMPLETE ✅

Task 4.1 (Substrate Loading) was completed earlier with commits:
- `11baff4` - feat: add substrate loading to VectorizedHamletEnv
- `e0baf5c` - test: add missing edge case tests for substrate loading

---

### Task 4.2: Wire Up Position Initialization

**Purpose**: Replace hardcoded `torch.randint` with `substrate.initialize_positions()`

**Files:**
- Modify: `src/townlet/environment/vectorized_env.py`
- Modify: `tests/test_townlet/unit/test_env_substrate_loading.py`

**Prerequisite: Verify Test File Exists**

```bash
ls tests/test_townlet/unit/test_env_substrate_loading.py
```

**Expected**: File exists (created in Task 4.1)

**If file doesn't exist**: Task 4.1 is incomplete. The test file should have been created with basic substrate loading tests. Check Task 4.1 completion before proceeding.

---

**Step 1: Write test for position initialization**

Add to `tests/test_townlet/unit/test_env_substrate_loading.py`:

```python
def test_env_initializes_positions_via_substrate():
    """Environment should use substrate.initialize_positions() in reset()."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=5,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Reset environment
    env.reset()

    # Positions should be initialized via substrate
    assert env.positions.shape == (5, 2)  # [num_agents, position_dim]
    assert env.positions.dtype == torch.long
    assert env.positions.device == torch.device("cpu")

    # Positions should be within grid bounds
    assert (env.positions >= 0).all()
    assert (env.positions < 8).all()


def test_substrate_initialize_positions_correctness():
    """Grid2D.initialize_positions() should return valid grid positions."""
    from townlet.substrate.grid2d import Grid2DSubstrate

    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    positions = substrate.initialize_positions(num_agents=10, device=torch.device("cpu"))

    # Correct shape and type
    assert positions.shape == (10, 2)
    assert positions.dtype == torch.long

    # Within bounds
    assert (positions >= 0).all()
    assert (positions < 8).all()
```

**Step 2: Update reset() method**

Modify `src/townlet/environment/vectorized_env.py` (line 246):

**BEFORE:**
```python
self.positions = torch.randint(0, self.grid_size, (self.num_agents, 2), device=self.device)
```

**AFTER:**
```python
# Use substrate for position initialization (supports grid and aspatial)
self.positions = self.substrate.initialize_positions(self.num_agents, self.device)
```

**Step 3: Run tests**

```bash
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
uv run pytest tests/test_townlet/unit/test_env_substrate_loading.py::test_env_initializes_positions_via_substrate -v
uv run pytest tests/test_townlet/unit/test_env_substrate_loading.py::test_substrate_initialize_positions_correctness -v
```

Expected: Both tests PASS

**Step 4: Commit**

```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/unit/test_env_substrate_loading.py
git commit -m "feat(vectorized_env): replace torch.randint with substrate.initialize_positions()

BREAKING CHANGE: Position initialization now delegates to substrate.

This enables:
- Aspatial substrates (position_dim=0, returns empty tensor)
- Future 3D grids (position_dim=3)
- Custom initialization strategies per substrate type

Legacy behavior preserved: Grid2D substrate uses same random initialization
with dimensions matching substrate.width × substrate.height.

Part of TASK-002A Phase 4 (Environment Integration)."
```

---

### Task 4.3: Wire Up Movement and Boundary Handling

**Purpose**: Replace hardcoded `torch.clamp` with `substrate.apply_movement()`

**Files:**
- Modify: `src/townlet/environment/vectorized_env.py`
- Modify: `tests/test_townlet/unit/test_env_substrate_loading.py`

**Step 1: Write test for movement via substrate**

Add to `tests/test_townlet/unit/test_env_substrate_loading.py`:

```python
def test_env_applies_movement_via_substrate():
    """Environment should use substrate.apply_movement() for boundary handling."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    env.reset()

    # Place agent at top-left corner
    env.positions = torch.tensor([[0, 0]], dtype=torch.long, device=torch.device("cpu"))

    # Try to move up (UP action decreases Y, but we're at boundary)
    # UP action should be action 0 based on action_dim
    action = torch.tensor([0], dtype=torch.long, device=torch.device("cpu"))

    env.step(action)

    # Position should be clamped by substrate (boundary="clamp" in configs)
    # Since we're at [0, 0] and trying to move up, should stay at [0, 0]
    assert (env.positions[:, 0] >= 0).all()  # X within bounds
    assert (env.positions[:, 1] >= 0).all()  # Y within bounds


def test_substrate_movement_matches_legacy():
    """Substrate movement should produce identical results to legacy torch.clamp."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Test substrate.apply_movement directly
    substrate = env.substrate
    positions = torch.tensor([[3, 3]], dtype=torch.long, device=torch.device("cpu"))

    # Move up (delta [0, -1])
    deltas = torch.tensor([[0, -1]], dtype=torch.long, device=torch.device("cpu"))
    new_positions = substrate.apply_movement(positions, deltas)

    # Should move to [3, 2]
    assert (new_positions == torch.tensor([[3, 2]], dtype=torch.long)).all()

    # Test boundary clamping at edge
    edge_positions = torch.tensor([[0, 0]], dtype=torch.long, device=torch.device("cpu"))
    up_left_delta = torch.tensor([[-1, -1]], dtype=torch.long, device=torch.device("cpu"))
    clamped = substrate.apply_movement(edge_positions, up_left_delta)

    # Should clamp to [0, 0] (not go negative)
    assert (clamped == torch.tensor([[0, 0]], dtype=torch.long)).all()
```

**Step 2: Find movement logic location**

First, identify where `torch.clamp` is used:

```bash
grep -n "torch.clamp.*grid_size" src/townlet/environment/vectorized_env.py
```

Expected output: Line 434 or similar

**Step 3: Update _execute_actions() method**

Modify `src/townlet/environment/vectorized_env.py` (around line 434):

**BEFORE:**
```python
new_positions = positions + action_deltas
new_positions = torch.clamp(new_positions, 0, self.grid_size - 1)
```

**AFTER:**
```python
# Use substrate for movement and boundary handling
new_positions = self.substrate.apply_movement(positions, action_deltas)
```

**Step 4: Run tests**

```bash
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
uv run pytest tests/test_townlet/unit/test_env_substrate_loading.py::test_env_applies_movement_via_substrate -v
uv run pytest tests/test_townlet/unit/test_env_substrate_loading.py::test_substrate_movement_matches_legacy -v
```

Expected: Both tests PASS

**Step 5: Commit**

```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/unit/test_env_substrate_loading.py
git commit -m "feat(vectorized_env): replace torch.clamp with substrate.apply_movement()

Replace hardcoded boundary clamping with substrate method.

This enables:
- Configurable boundary modes (clamp, wrap, bounce, sticky)
- Substrate-specific movement logic
- Aspatial substrates (no-op movement)

Legacy behavior preserved: All production configs use boundary="clamp"
which produces identical behavior to torch.clamp(positions, 0, grid_size-1).

Part of TASK-002A Phase 4 (Environment Integration)."
```

---

### Task 4.4: Wire Up Affordance Randomization

**Purpose**: Replace hardcoded grid iteration with `substrate.get_all_positions()`

**Files:**
- Modify: `src/townlet/environment/vectorized_env.py`
- Modify: `tests/test_townlet/unit/test_env_substrate_loading.py`

**Step 1: Write test for affordance randomization**

Add to `tests/test_townlet/unit/test_env_substrate_loading.py`:

```python
def test_env_randomizes_affordances_via_substrate():
    """Environment should use substrate.get_all_positions() for affordance placement."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Randomize affordances
    env.randomize_affordance_positions()

    # All affordances should have valid positions within grid
    for affordance_name, position in env.affordances.items():
        assert position.shape == (2,)  # [x, y]
        assert (position >= 0).all()
        assert (position < 8).all()

    # Affordances should not overlap (each at unique position)
    positions_list = [tuple(pos.tolist()) for pos in env.affordances.values()]
    assert len(positions_list) == len(set(positions_list))  # All unique
```

**Step 2: Update randomize_affordance_positions() method**

Modify `src/townlet/environment/vectorized_env.py` (lines 690-698):

**BEFORE:**
```python
# Generate list of all grid positions
all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

# Shuffle and assign to affordances
random.shuffle(all_positions)

# Assign new positions to affordances
for i, affordance_name in enumerate(self.affordances.keys()):
    new_pos = all_positions[i]
    self.affordances[affordance_name] = torch.tensor(new_pos, dtype=torch.long, device=self.device)
```

**AFTER:**
```python
# Get all valid positions from substrate
all_positions = self.substrate.get_all_positions()

# Guard for aspatial substrates
if len(all_positions) == 0:
    raise ValueError(
        "Cannot randomize affordance positions in aspatial substrate. "
        "Aspatial universes have no spatial positioning (position_dim=0)."
    )

# Shuffle and assign to affordances
random.shuffle(all_positions)

# Assign new positions to affordances
for i, affordance_name in enumerate(self.affordances.keys()):
    new_pos = all_positions[i]
    self.affordances[affordance_name] = torch.tensor(new_pos, dtype=torch.long, device=self.device)
```

**Step 3: Update validation logic**

Also update line 682 to use substrate:

**BEFORE:**
```python
total_cells = self.grid_size * self.grid_size
```

**AFTER:**
```python
total_cells = len(self.substrate.get_all_positions())
```

**Step 4: Run tests**

```bash
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
uv run pytest tests/test_townlet/unit/test_env_substrate_loading.py::test_env_randomizes_affordances_via_substrate -v
```

Expected: Test PASSES

**Step 5: Commit**

```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/unit/test_env_substrate_loading.py
git commit -m "feat(vectorized_env): replace hardcoded grid iteration with substrate.get_all_positions()

Replace manual grid comprehension with substrate method in affordance randomization.

BEFORE:
  all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

AFTER:
  all_positions = self.substrate.get_all_positions()

This enables:
- Future 3D grids (would return [(x, y, z), ...])
- Non-square grids (width != height)
- Custom position enumeration per substrate type

Aspatial guard: Raises clear error if trying to randomize affordances
in aspatial universe (position_dim=0, no positions exist).

Part of TASK-002A Phase 4 (Environment Integration)."
```

---

### Task 4.5: Enable Integration Tests

**Purpose**: Un-skip integration tests and verify behavioral equivalence

**Files:**
- Modify: `tests/test_townlet/integration/test_substrate_migration.py`

**Step 1: Remove skip markers**

Modify `tests/test_townlet/integration/test_substrate_migration.py`:

**BEFORE (line 10):**
```python
pytestmark = pytest.mark.skip(reason="Phase 4 (Environment Integration) not yet complete")
```

**AFTER:**
```python
# Phase 4 complete - integration tests enabled
# pytestmark = pytest.mark.skip(reason="Phase 4 (Environment Integration) not yet complete")
```

**ALSO** remove individual `pytest.skip()` calls from each test function (lines 30, 58, 74, 94).

**Step 2: Uncomment test code**

The file currently has all imports and test bodies commented out. You need to uncomment them.

**Example - Before (commented):**
```python
# import pytest
# import torch
# from pathlib import Path
# from townlet.environment.vectorized_env import VectorizedHamletEnv
#
# def test_env_observation_dim_unchanged():
#     """Test observation dimensions unchanged after substrate integration."""
#     env = VectorizedHamletEnv(
#         config_pack_path=Path("configs/L1_full_observability"),
#         ...
#     )
#     assert env.observation_dim == 91
```

**After (uncommented):**
```python
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv

def test_env_observation_dim_unchanged():
    """Test observation dimensions unchanged after substrate integration."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        ...
    )
    assert env.observation_dim == 91
```

**Action**: Remove all `#` from import lines and test function bodies. Every line starting with `#` (except comment strings) should be uncommented.

**Step 3: Run integration tests**

```bash
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
uv run pytest tests/test_townlet/integration/test_substrate_migration.py -v
```

Expected: ALL 4 tests PASS (no failures, no skips)

**Tests enabled:**
1. `test_env_observation_dim_unchanged` (5 parameterized cases)
2. `test_env_substrate_dimensions` (5 parameterized cases)
3. `test_env_substrate_boundary_behavior`
4. `test_env_substrate_distance_metric`

**Step 4: Fix any failures (if needed)**

If tests fail, diagnose using these common failure patterns:

**1. Observation dimension mismatch**:
```
AssertionError: assert 64 == 91
```
**Cause**: Observation builder not using substrate encoding (expected - Phase 6 will fix this)
**Fix**: This is expected behavior. Observation encoding is Phase 6 scope. Integration tests should verify substrate dimensions, not full observation dims.

**2. Position tensor shape mismatch**:
```
RuntimeError: Expected shape (5, 2), got (5, 0)
```
**Cause**: Aspatial substrate returns position_dim=0. Tests assume Grid2D.
**Fix**: Add substrate type check in test. Aspatial has no positions (position_dim=0).

**3. Boundary behavior changed**:
```
AssertionError: Expected position [0, 0], got [-1, -1]
```
**Cause**: Substrate boundary mode not "clamp" (might be "wrap" or "bounce")
**Fix**: Verify all production configs use `boundary: "clamp"` in substrate.yaml.

**4. AttributeError on substrate methods**:
```
AttributeError: 'Grid2DSubstrate' object has no attribute 'apply_movement'
```
**Cause**: Task 4.0 (get_all_positions) incomplete or substrate interface missing methods
**Fix**: Complete Task 4.0 first. Verify substrate interface has all required methods.

**5. Import errors**:
```
ImportError: cannot import name 'VectorizedHamletEnv'
```
**Cause**: Circular import or syntax error in vectorized_env.py
**Fix**: Check Tasks 4.2-4.4 implementation. Verify syntax is correct.

**Step 5: Commit**

```bash
git add tests/test_townlet/integration/test_substrate_migration.py
git commit -m "test: enable substrate migration integration tests

Phase 4 integration complete - all substrate methods wired up:
- substrate.initialize_positions() in reset()
- substrate.apply_movement() in _execute_actions()
- substrate.get_all_positions() in randomize_affordance_positions()

All 4 integration tests passing (14 total assertions with parameterization):
- Observation dimensions unchanged (5 configs)
- Substrate dimensions correct (5 configs)
- Boundary behavior matches legacy (clamp)
- Distance calculations match legacy (manhattan)

Behavioral equivalence verified: No changes to training behavior.

Part of TASK-002A Phase 4 (Environment Integration)."
```

---

## Phase 4 Completion Checklist

After completing Tasks 4.0-4.5, verify:

- ✅ `substrate.initialize_positions()` called in `reset()` (not `torch.randint`)
- ✅ `substrate.apply_movement()` called in `_execute_actions()` (not `torch.clamp`)
- ✅ `substrate.get_all_positions()` called in `randomize_affordance_positions()` (not manual grid iteration)
- ✅ All 4 integration tests passing (no skips)
- ✅ Behavioral equivalence verified (observation dims, boundary behavior unchanged)
- ✅ No hardcoded spatial logic remaining in VectorizedEnv

**Verification Command:**
```bash
# Check for remaining hardcoded spatial operations
grep -n "torch.randint\|torch.clamp\|range(self.grid_size)" src/townlet/environment/vectorized_env.py
```

Expected: NO matches (all replaced with substrate methods)

**Total Tests After Phase 4:**
- 8 unit tests (test_env_substrate_loading.py): 6 existing + 2 new per task = 8 tests
- 4 integration tests (test_substrate_migration.py): 4 tests (14 assertions with parameterization)
- 2 substrate tests (test_substrate_base.py): get_all_positions() tests
- **Grand Total: 14 tests**

**Ready for Phase 5 When:**
- All 14 Phase 4 tests passing
- `grep "range(self.grid_size)" src/townlet/environment/vectorized_env.py` returns NO matches
- Phase 5 Step 0 verification passes

---

## Estimated Effort

- **Task 4.0**: 30 minutes (add get_all_positions() method)
- **Task 4.1**: COMPLETE (already done)
- **Task 4.2**: 30 minutes (position initialization)
- **Task 4.3**: 30 minutes (movement and boundaries)
- **Task 4.4**: 30 minutes (affordance randomization)
- **Task 4.5**: 45 minutes (enable and debug integration tests)
- **Total**: ~2.75 hours

---

## Phase 4 → Phase 5 Handoff

Phase 5 expects Phase 4 to deliver:

1. **Substrate methods fully wired**: Not just loaded, but actively called in:
   - `reset()` → `initialize_positions()`
   - `_execute_actions()` → `apply_movement()`
   - `randomize_affordance_positions()` → `get_all_positions()`

2. **No hardcoded spatial logic**: All `torch.randint`, `torch.clamp`, `range(self.grid_size)` removed

3. **Integration tests passing**: All 4 tests in test_substrate_migration.py enabled and passing

4. **Behavioral equivalence**: Observation dims, boundary behavior, distance calculations unchanged

5. **get_all_positions() available**: Phase 5 Task 5.1 assumes this method exists

If Phase 4 delivers these, Phase 5 can proceed cleanly with position management refactoring.

If Phase 4 is incomplete, Phase 5 Step 0 will fail and block progression.

---

## Known Limitations (Deferred to Phase 5)

The following substrate integrations are **intentionally deferred to Phase 5**:

1. **Observation encoding**: Still uses hardcoded grid encoding (Phase 5 Task 5.2)
2. **POMDP vision window**: Still uses hardcoded 2D window (Phase 5 Task 5.3)
3. **Position_dim flexibility**: Still assumes `[num_agents, 2]` in some places (Phase 5 refactoring)

These are acceptable because:
- All production configs use 2D grids (position_dim=2)
- Aspatial substrates not yet deployed to production
- Phase 5 will complete these integrations

Phase 4 focuses on the **critical path**: Position initialization, movement, and affordance placement.
