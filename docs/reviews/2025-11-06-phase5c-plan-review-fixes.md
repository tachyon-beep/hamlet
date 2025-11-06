# Phase 5C Implementation Plan - Review Fixes

**Date**: 2025-11-06
**Review Loop**: 1
**Status**: CRITICAL ISSUES IDENTIFIED - NEEDS REVISION

---

## Executive Summary

Code review identified **5 critical issues** that must be fixed before implementation:

1. **Grid2D Encoding Mismatch**: Plan assumes normalized encoding, but Grid2D actually uses one-hot
2. **Action Space Formula Error**: Environment uses 2N+2 actions (includes WAIT), plan assumes 2N+1
3. **Test Infrastructure Missing**: phase5/ test directory doesn't exist
4. **Observation Dimensions Wrong**: All dimension calculations based on incorrect assumptions
5. **Grid3D Status Unclear**: Design doc says it already has normalized encoding, plan treats as new

**Recommended Action**: Fix critical issues, re-review, then proceed with implementation.

---

## Critical Issue 1: Grid2D Uses One-Hot Encoding (Not Normalized)

### Current Implementation

**File**: `src/townlet/substrate/grid2d.py:149-176`

```python
def encode_observation(self, positions, affordances):
    """Encode positions as one-hot grid cells."""
    grid_encoding = torch.zeros(num_agents, self.width * self.height, device=device)
    # Marks affordances and agents in flattened grid
    return grid_encoding  # Shape: [num_agents, width*height]
```

**Observation dimensions**:
- 3×3 grid: 9 dimensions
- 8×8 grid: 64 dimensions
- 10×10 grid: 100 dimensions

### Plan Assumption (INCORRECT)

Plan assumes Grid2D uses normalized encoding:
- Task 1.4.1: Tests expect `[num_agents, 2]` dimensions
- Plan: "relative encoding (default, backward compatible)"

### Root Cause

Confusion between:
- **Grid3D**: Uses normalized coordinates (3 dims) - confirmed in Phase 5B
- **Grid2D**: Still uses one-hot encoding (width×height dims) - not yet migrated

### Impact

- **All Grid2D tests in Task 1.4** will fail immediately
- **Observation dimension calculations** throughout plan are wrong
- **"Backward compatible default"** is misleading

### Proposed Fix

**Option A: Add one_hot as 4th Encoding (RECOMMENDED)**

```yaml
observation_encoding: "one_hot" | "relative" | "scaled" | "absolute"
# Default: "one_hot" (TRUE backward compatibility)
```

**Benefits**:
- Genuinely backward compatible
- Old configs work without modification
- Clear migration path (one_hot → relative)

**Option B: Break Backward Compatibility**

```yaml
observation_encoding: "relative"  # NEW DEFAULT (breaking change)
# Old behavior (one_hot) no longer available
```

**Benefits**:
- Cleaner API (3 modes instead of 4)
- Forces migration to better encoding

**Drawbacks**:
- All existing checkpoints incompatible
- All existing configs need update

### Recommendation

Choose **Option A** (add one_hot) for Phase 5C. Document one_hot as deprecated and plan removal in future phase.

---

## Critical Issue 2: Action Space Formula Mismatch

### Current Environment Implementation

**File**: `src/townlet/environment/vectorized_env.py:248-261`

```python
# Comments in code:
# 1D: 4 actions (MOVE_X_NEGATIVE, MOVE_X_POSITIVE, INTERACT, WAIT)
# 2D: 6 actions (UP, DOWN, LEFT, RIGHT, INTERACT, WAIT)
# 3D: 8 actions (±X, ±Y, ±Z, INTERACT, WAIT)

if self.substrate.position_dim == 1:
    self.action_dim = 4  # 2*1 + 2 (includes WAIT)
elif self.substrate.position_dim == 2:
    self.action_dim = 6  # 2*2 + 2 (includes WAIT)
elif self.substrate.position_dim == 3:
    self.action_dim = 8  # 2*3 + 2 (includes WAIT)
```

**Formula**: 2N + 2 (movement + INTERACT + WAIT)

### Plan Assumption (INCORRECT)

Plan uses formula: 2N + 1 (movement + INTERACT only)

```python
# Task 1.1.2:
return 2 * self.position_dim + 1  # Missing WAIT action
```

### Impact

- **action_space_size property** will return wrong values
- **Networks will be undersized** (missing WAIT action output)
- **Action masking** will break
- **Tests will fail** when comparing action counts

### Proposed Fix

**Update formula to 2N + 2**:

```python
@property
def action_space_size(self) -> int:
    """Return number of discrete actions.

    Returns:
        For spatial substrates: 2*position_dim + 2 (movement + INTERACT + WAIT)
        For aspatial: 2 (INTERACT + WAIT)

    Examples:
        Aspatial: 2 (INTERACT, WAIT)
        1D: 4 (±X, INTERACT, WAIT)
        2D: 6 (±X, ±Y, INTERACT, WAIT)
        3D: 8 (±X, ±Y, ±Z, INTERACT, WAIT)
    """
    if self.position_dim == 0:
        return 2  # Aspatial: INTERACT + WAIT
    return 2 * self.position_dim + 2  # Movement + INTERACT + WAIT
```

### Alternative Investigation

**Before changing formula**, verify WAIT action exists:
1. Check if WAIT is actually used in current codebase
2. Check if removing WAIT would break anything
3. Consider if WAIT should be removed (simplify to 2N+1)

If WAIT is vestigial, **Option B** might be better: Remove WAIT, use 2N+1 formula.

---

## Critical Issue 3: Test Directory Doesn't Exist

### Problem

Plan assumes `tests/test_townlet/phase5/` exists, but it needs to be created.

### Fix

Add as **Task 1.0** (before Task 1.1):

```markdown
### Task 1.0: Setup Phase 5C Test Infrastructure (5 min)

**Context:** Create test directory structure for Phase 5C tests.

**Files:**
- Create: `tests/test_townlet/phase5/__init__.py`

#### Step 1.0.1: Create test directory

```bash
mkdir -p tests/test_townlet/phase5
touch tests/test_townlet/phase5/__init__.py
```

#### Step 1.0.2: Verify directory exists

**Run:** `ls tests/test_townlet/phase5/__init__.py`

**Expected:** File exists

#### Step 1.0.3: Commit

```bash
git add tests/test_townlet/phase5/
git commit -m "chore: create Phase 5C test directory

- Add tests/test_townlet/phase5/ for Phase 5C tests
- Part of Phase 5C infrastructure setup

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```
```

---

## Critical Issue 4: Observation Dimension Calculations

### Problem

All observation dimension calculations assume:
- Grid2D relative: 2 dims (WRONG - currently width×height)
- Grid2D scaled: 4 dims (WRONG - currently width×height)

### Fix Table

| Substrate | Current Dims | Plan Relative | Plan Scaled | Plan Absolute |
|-----------|--------------|---------------|-------------|---------------|
| Grid2D (8×8) | 64 (one-hot) | 2 | 4 | 2 |
| Grid3D (8×8×3) | 3 (normalized) | 3 | 6 | 3 |
| Continuous1D | 1 (normalized) | 1 | 2 | 1 |
| Continuous2D | 2 (normalized) | 2 | 4 | 2 |
| Continuous3D | 3 (normalized) | 3 | 6 | 3 |

### Update Required

All tests in Task 1.4 need to be rewritten to account for:
1. Current one-hot encoding (64 dims for 8×8)
2. New relative encoding (2 dims)
3. Transition from one-hot to relative

---

## Critical Issue 5: Grid3D Status Unclear

### Design Doc Claims (v3, line 173)

> "CRITICAL: Corrected Grid3D documentation - acknowledged Phase 5B already uses normalized encoding"

### Verification Needed

Check `src/townlet/substrate/grid3d.py` to confirm:
1. Does Grid3D.encode_observation() already return normalized coordinates?
2. If yes, Task 1.5 only needs to add parameter + scaled/absolute modes
3. If no, Task 1.5 implements all three modes from scratch

### Proposed Fix

Add verification step to Task 1.5.1:

```python
# Step 1.5.0: Verify Grid3D current encoding
def test_grid3d_current_encoding_is_normalized():
    """Verify Grid3D already uses normalized encoding."""
    substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
    positions = torch.tensor([[0, 0, 0], [7, 7, 2]], dtype=torch.long)
    encoded = substrate.encode_observation(positions, {})

    # If normalized, should be [num_agents, 3] with values in [0, 1]
    assert encoded.shape == (2, 3), "Expected normalized coordinates"
    assert encoded.max() <= 1.0 and encoded.min() >= 0.0
```

---

## Minor Issue 6: Integration Test Config Incomplete

### Problem

Task 1.9.1 creates configs but doesn't specify minimum viable config pack:

```python
# ... create other required configs (bars, affordances, training, etc.) ...
```

### Fix

Provide complete minimal config pack template:

```python
def create_minimal_test_config(tmp_path: Path, observation_encoding: str = "relative"):
    """Create minimal valid config pack for testing."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # substrate.yaml
    (config_dir / "substrate.yaml").write_text(f"""
version: "1.0"
description: "Test config"
type: "grid"
grid:
  topology: "square"
  width: 3
  height: 3
  boundary: "clamp"
  distance_metric: "manhattan"
  observation_encoding: "{observation_encoding}"
""")

    # bars.yaml (minimal: energy only)
    (config_dir / "bars.yaml").write_text("""
version: "1.0"
bars:
  energy:
    initial_value: 100.0
    decay_rate: 0.1
    min_value: 0.0
    max_value: 100.0
""")

    # cascades.yaml (empty)
    (config_dir / "cascades.yaml").write_text("""
version: "1.0"
cascades: []
""")

    # affordances.yaml (one affordance: Bed)
    (config_dir / "affordances.yaml").write_text("""
version: "1.0"
affordances:
  Bed:
    effects:
      energy: 10.0
    cost: 0.0
    allowed_actions: ["INTERACT"]
""")

    # training.yaml (minimal hyperparams)
    (config_dir / "training.yaml").write_text("""
version: "1.0"
environment:
  partial_observability: false
  vision_range: 2
  enabled_affordances: ["Bed"]

population:
  num_agents: 1
  learning_rate: 0.001
  gamma: 0.99
  replay_buffer_capacity: 1000
  network_type: simple

curriculum:
  max_steps_per_episode: 100
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 100

exploration:
  embed_dim: 64
  initial_intrinsic_weight: 1.0
  variance_threshold: 100.0
  survival_window: 50

training:
  device: "cpu"
  max_episodes: 10
  train_frequency: 4
  target_update_frequency: 10
  batch_size: 16
  max_grad_norm: 10.0
  epsilon_start: 1.0
  epsilon_decay: 0.99
  epsilon_min: 0.01
""")

    return config_dir
```

---

## Recommended Action Plan

### Phase 1: Critical Fixes (2-3 hours)

1. **Verify Grid3D encoding** (30 min)
   - Read current implementation
   - Document status
   - Update Task 1.5 accordingly

2. **Fix action space formula** (30 min)
   - Investigate WAIT action usage
   - Choose 2N+1 vs 2N+2
   - Update Task 1.1.2 formula
   - Update all tests

3. **Fix Grid2D encoding strategy** (60 min)
   - Decision: Add "one_hot" as 4th option OR break compatibility
   - Update Task 1.2 config schema
   - Update Task 1.4 tests and implementation
   - Update all observation_dim calculations

4. **Add test infrastructure** (15 min)
   - Add Task 1.0 (create phase5/ directory)

5. **Complete integration test configs** (30 min)
   - Add create_minimal_test_config() helper
   - Update Task 1.9.1

### Phase 2: Re-Review (30 min)

- Have code-reviewer agent re-review updated plan
- Verify all critical issues resolved
- Check for new issues introduced by fixes

### Phase 3: Implementation (20-25 hours)

- Proceed with corrected plan
- Use TDD discipline throughout
- Commit frequently

---

## Approval Status

**Current Status**: ❌ NOT APPROVED

**Blocking Issues**:
1. Grid2D encoding mismatch
2. Action space formula error
3. Missing test infrastructure
4. Observation dimension errors
5. Grid3D status unclear

**Next Step**: Apply fixes from this document, then re-submit for review.

---

## Review Metadata

**Reviewer**: Code Review Agent (superpowers:code-reviewer)
**Review Date**: 2025-11-06
**Review Loop**: 1
**Verdict**: REQUEST REVISION
**Estimated Fix Time**: 2-3 hours
**Re-review Required**: Yes
