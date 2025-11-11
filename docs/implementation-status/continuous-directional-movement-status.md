# Continuous Directional Movement - Implementation Status

**Date**: 2025-11-11
**Status**: Phase 1 Complete, Phase 2 In Progress (60% done)

---

## Completed Work ✅

### 1. Fixed VFS Grid Encoding Bug (Grid3D + Continuous Substrates)

**Problem**:
- Grid3D: Compiler calculated `grid_cells = width * height` (ignored depth)
- Continuous: Action masking tried to use `grid_size` which was None

**Fixed Files**:
- `src/townlet/universe/compiler.py` (lines 318-398): Now handles Grid3D with `width * height * depth`
- `src/townlet/universe/compiler.py` (lines 1749-1778): `_derive_grid_dimensions()` returns correct values for Grid3D
- `src/townlet/environment/vectorized_env.py` (lines 799-826): Action masking checks `if self.grid_size is not None` before applying

**Tests**: All substrate migration tests now pass (Grid2D, Grid3D, Continuous1D/2D/3D)

---

### 2. Implemented Velocity Observation (VFS)

**What**: Agents can now observe their own velocity (direction + magnitude) from previous movement.

**Files Modified**:
- `src/townlet/universe/compiler.py` (lines 400-531):
  - Added velocity VFS variables for grid substrates: `velocity_x`, `velocity_y`, `velocity_z` (3D only), `velocity_magnitude`
  - Added velocity VFS variables for continuous substrates (lines 454-527)
  - Auto-generated, auto-exposed as observations via VFS

- `src/townlet/environment/vectorized_env.py` (lines 940-967):
  - Track `old_positions` before movement
  - Calculate `velocity = positions - old_positions`
  - Write velocity components to VFS registry
  - Calculate and write velocity magnitude

**Observation Impact**:
- Grid2D: 29 → **32 dims** (+3: velocity_x, velocity_y, velocity_magnitude)
- Grid3D: Similar (+4 with velocity_z)
- Continuous2D: Similar (+3)

**Test Status**: ✅ Verified working on Continuous2D substrate

---

### 3. Enabled Float Deltas in ActionConfig

**File**: `src/townlet/environment/action_config.py` (line 44)
- Changed: `delta: list[int] | None`
- To: `delta: list[int | float] | None`
- Allows continuous substrates to use float movement deltas

---

### 4. Added Action Discretization Config Schema

**File**: `src/townlet/substrate/config.py` (lines 156-161)
- Added to `ContinuousConfig`:
```python
action_discretization: dict[str, int] | None = Field(
    default=None,
    description="Discretize continuous action space: {'num_directions': 8-32, 'num_magnitudes': 3-7}. "
                "If None, uses legacy 8-way discrete actions.",
)
```

**Example Config** (for `substrate.yaml`):
```yaml
continuous:
  dimensions: 2
  bounds: [[0.0, 10.0], [0.0, 10.0]]
  boundary: "clamp"
  movement_delta: 0.5
  interaction_radius: 0.8
  distance_metric: "euclidean"
  observation_encoding: "relative"

  # NEW: Maximum freedom discretization
  action_discretization:
    num_directions: 32   # 11.25° resolution
    num_magnitudes: 7    # 0%, 16.7%, 33.3%, 50%, 66.7%, 83.3%, 100%
```

---

## Pending Work ⏳

### 5. Generate Discretized Continuous Actions (NOT YET IMPLEMENTED)

**What's Needed**: Modify action space builder to generate:
- **32 directions**: 0°, 11.25°, 22.5°, 33.75°, ..., 348.75° (360° / 32 = 11.25° resolution)
- **7 magnitudes**: 0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0
- **Total actions**: 1 STOP + 192 directional (32×6 magnitudes, exclude 0 magnitude for other dirs) + INTERACT + WAIT = **195 actions**

**Action Naming Scheme**:
- `STOP` (magnitude=0, any direction - only create once)
- `MOVE_0_1` through `MOVE_31_6` (direction index 0-31, magnitude index 1-6)
- `INTERACT`, `WAIT`

**Calculation Example**:
```python
import math

num_directions = 32
num_magnitudes = 7
magnitudes = [i / (num_magnitudes - 1) for i in range(num_magnitudes)]  # [0.0, 0.167, ..., 1.0]

actions = []
action_id = 0

# STOP action (magnitude=0)
actions.append(ActionConfig(
    id=action_id,
    name="STOP",
    type="passive",
    delta=None,
    costs={},
    effects={},
    enabled=True,
    source="substrate",
))
action_id += 1

# Directional movement actions
for dir_idx in range(num_directions):
    angle_rad = 2 * math.pi * dir_idx / num_directions
    dx_unit = math.cos(angle_rad)
    dy_unit = math.sin(angle_rad)

    for mag_idx in range(1, num_magnitudes):  # Skip magnitude=0 (already have STOP)
        magnitude = magnitudes[mag_idx]

        # Scaled delta = unit direction × magnitude × movement_delta
        delta_x = dx_unit * magnitude * movement_delta
        delta_y = dy_unit * magnitude * movement_delta

        actions.append(ActionConfig(
            id=action_id,
            name=f"MOVE_{dir_idx}_{mag_idx}",
            type="movement",
            delta=[delta_x, delta_y],  # Float deltas!
            costs={"energy": base_move_cost * magnitude},  # Scale cost by magnitude
            effects={},
            enabled=True,
            source="substrate",
        ))
        action_id += 1

# INTERACT and WAIT
# ... (add as usual)
```

**Where to Implement**:
Need to find action space builder - likely in one of these files:
- `src/townlet/environment/action_space.py` (if exists)
- `src/townlet/environment/vectorized_env.py` (look for `_build_action_space` or similar)
- `src/townlet/substrate/continuous.py` (substrate may provide actions)

**Search Command**:
```bash
grep -rn "def.*build.*action\|get_substrate_actions\|ActionConfig(" src/townlet/substrate/continuous.py src/townlet/environment/
```

---

### 6. Test Complete System (NOT YET DONE)

**Test Plan**:
1. Create test config: `configs/L1_continuous_2D_directional/`
2. Add `action_discretization: {num_directions: 32, num_magnitudes: 7}` to substrate.yaml
3. Clear cache: `rm -rf configs/L1_continuous_2D_directional/.compiled`
4. Run training: `uv run scripts/run_demo.py --config configs/L1_continuous_2D_directional`
5. Verify:
   - Action space has ~195 actions
   - Agents can navigate to arbitrary (x, y) positions
   - Velocity observations appear in state
   - Network adapts to new obs_dim automatically

---

## Architecture Notes

### VFS Auto-Exposure

Variables added to `_auto_generate_standard_variables()` are automatically:
1. Added to symbol table
2. Registered in VFS registry
3. Exposed as observations (via `_auto_generate_standard_exposures()`)
4. Passed to network via `obs_dim` parameter

**No manual wiring needed** - system is "self-healing"!

### Action Space Flexibility

The system can handle large action spaces:
- Current Grid2D: 8 actions (UP, DOWN, LEFT, RIGHT, INTERACT, WAIT, REST, MEDITATE)
- Target Continuous2D: **195 actions** (32×7 movement + meta-actions)
- Network will have output dim = 195 Q-values
- DQN training loop unchanged (just larger softmax)

### Backward Compatibility

**Grid2D/3D**: No changes, continue using discrete integer actions
**Existing Continuous configs**: Still work with legacy 8-way actions (action_discretization defaults to None)
**Breaking change**: If `action_discretization` is added, checkpoints are incompatible (action space changes)

---

## Design Document

Full design with 3 options (discretized, true continuous, hybrid):
- `/home/john/hamlet/docs/designs/continuous-directional-movement.md`

**Decision**: Implementing **Option A (Discretized)** for DQN compatibility

---

## Key Files Modified

```
src/townlet/universe/compiler.py
  - Lines 318-531: Auto-generate velocity VFS variables
  - Lines 1749-1778: Fix _derive_grid_dimensions for Grid3D

src/townlet/environment/vectorized_env.py
  - Lines 799-826: Fix action masking for continuous substrates
  - Lines 940-967: Track and write velocity to VFS

src/townlet/environment/action_config.py
  - Line 44: Allow float deltas

src/townlet/substrate/config.py
  - Lines 156-161: Add action_discretization schema
```

---

## Next Steps for Continuation

1. **Find action builder**: Search for where substrate actions are generated
2. **Implement discretization logic**: Generate 32×7 actions if `action_discretization` is set
3. **Handle 3D**: Extend to 3D with spherical coordinates (if needed later)
4. **Test**: Create test config and verify training
5. **Document**: Add example configs to `configs/templates/`

---

## Questions to Resolve

1. **Energy costs**: Should they scale linearly with magnitude? (Currently assumed linear)
2. **Action masking**: Do discretized continuous actions still need boundary masking? (Probably not - continuous substrates handle boundaries)
3. **Direction 0**: Should direction 0 be East (1, 0) or North (0, 1)? (Suggest East to match standard unit circle)
4. **STOP vs WAIT**: Should STOP (magnitude=0 movement) be separate from WAIT? (Suggest merge into WAIT)

---

## Testing Commands

```bash
# Clear caches
rm -rf configs/L1_continuous_*/.compiled

# Run substrate tests
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
export UV_CACHE_DIR=.uv-cache
uv run pytest tests/test_townlet/integration/test_substrate_migration.py -xvs

# Check observation dimensions
uv run python -m townlet.compiler inspect configs/L1_continuous_2D

# Run training
uv run scripts/run_demo.py --config configs/L1_continuous_2D
```

---

## Git Commit Message (When Complete)

```
feat(continuous): Add velocity observation and discretized directional actions

Implements Phase 1 and Phase 2A of continuous directional movement system:

**Velocity Observation (Complete)**:
- Add VFS variables: velocity_x, velocity_y, velocity_z, velocity_magnitude
- Track movement delta between steps in environment
- Auto-exposed as observations (+3-4 obs dims)
- Enables agents to remember movement direction and build navigation memory

**Bug Fixes**:
- Fix Grid3D grid_encoding dimension calculation (width×height×depth)
- Fix continuous substrate action masking (check grid_size is not None)
- Fix _derive_grid_dimensions to handle Grid3D correctly

**Action Space Foundation**:
- Change ActionConfig.delta from list[int] to list[int | float]
- Add action_discretization config schema to ContinuousConfig
- Enables 32 directions × 7 magnitudes = 195 fine-grained actions

**Breaking Changes**:
- Observation dimensions increased by +3-4 (velocity tracking)
- Old checkpoints incompatible with new obs space

**Next**: Implement discretized action generation in action space builder

See docs/designs/continuous-directional-movement.md for full design.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```
