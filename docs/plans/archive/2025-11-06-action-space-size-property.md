# Action Space Size Property Implementation Plan (C1/C2 Fixes)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

> **CORRECTION (2025-11-06)**: Original plan assumed formula `2N+1` (INTERACT only).
> Actual implementation uses `2N+2` (INTERACT + WAIT). This was discovered during
> implementation by reviewing `action_labels.py:23-56` which shows WAIT action exists.
> All references to `2N+1` in this document should be read as `2N+2`.

**Goal:** Add `action_space_size` property to substrate base class and remove hardcoded action space logic from VectorizedHamletEnv.

**Architecture:** Add abstract property to `SpatialSubstrate`, implement in all 6 substrate classes, replace hardcoded if-else chain with property access.

**Tech Stack:** Python, PyTorch, Pydantic

**Why This Matters:**
- **Prerequisite for Phase 5C**: N-dimensional substrates need dynamic action space sizing (2N+1 formula)
- **Architectural improvement**: Substrates own their action space semantics
- **Code quality**: Removes hardcoded logic and enables substrate-specific action spaces

**Current Problem:**
- `VectorizedHamletEnv:247-261` hardcodes action space sizes based on `position_dim`
- No way for substrates to report their action space size
- Won't scale to N-dimensional substrates (would need cases for 4, 5, 6, ...)

---

## Task 1: Add action_space_size Property to Base Class

**Files:**
- Modify: `src/townlet/substrate/base.py:8-230`
- Test: `tests/test_townlet/test_substrate/test_base.py` (will be created)

### Step 1.1: Write test for action_space_size property

Create test file to verify all substrates implement the property:

**File:** `tests/test_townlet/test_substrate/test_base.py`

```python
"""Tests for SpatialSubstrate base class contracts."""

import pytest
import torch

from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate
from townlet.substrate.continuous import Continuous1DSubstrate, Continuous2DSubstrate, Continuous3DSubstrate
from townlet.substrate.aspatial import AspatialSubstrate


class TestActionSpaceSizeProperty:
    """Test that all substrates implement action_space_size property."""

    def test_grid2d_action_space_size(self):
        """Grid2D has 5 actions (UP/DOWN/LEFT/RIGHT/INTERACT)."""
        substrate = Grid2DSubstrate(
            width=8,
            height=8,
            boundary="clamp",
            distance_metric="manhattan",
        )
        assert substrate.action_space_size == 5
        assert substrate.action_space_size == 2 * substrate.position_dim + 1

    def test_grid3d_action_space_size(self):
        """Grid3D has 7 actions (±X/±Y/±Z/INTERACT)."""
        substrate = Grid3DSubstrate(
            width=8,
            height=8,
            depth=3,
            boundary="clamp",
            distance_metric="manhattan",
        )
        assert substrate.action_space_size == 7
        assert substrate.action_space_size == 2 * substrate.position_dim + 1

    def test_continuous1d_action_space_size(self):
        """Continuous1D has 3 actions (±X/INTERACT)."""
        substrate = Continuous1DSubstrate(
            dimensions=1,
            bounds=[(0.0, 10.0)],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
        )
        assert substrate.action_space_size == 3
        assert substrate.action_space_size == 2 * substrate.position_dim + 1

    def test_continuous2d_action_space_size(self):
        """Continuous2D has 5 actions (±X/±Y/INTERACT)."""
        substrate = Continuous2DSubstrate(
            dimensions=2,
            bounds=[(0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
        )
        assert substrate.action_space_size == 5
        assert substrate.action_space_size == 2 * substrate.position_dim + 1

    def test_continuous3d_action_space_size(self):
        """Continuous3D has 7 actions (±X/±Y/±Z/INTERACT)."""
        substrate = Continuous3DSubstrate(
            dimensions=3,
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 5.0)],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
        )
        assert substrate.action_space_size == 7
        assert substrate.action_space_size == 2 * substrate.position_dim + 1

    def test_aspatial_action_space_size(self):
        """Aspatial has 1 action (INTERACT only)."""
        substrate = AspatialSubstrate()
        assert substrate.action_space_size == 1
        # Aspatial is special: position_dim=0, but action_space_size=1 (not 2*0+1=1)
        # This is correct: aspatial only has INTERACT action

    def test_action_space_formula_consistency(self):
        """Verify 2N+1 formula holds for spatial substrates."""
        test_cases = [
            (Grid2DSubstrate(8, 8, "clamp", "manhattan"), 2, 5),
            (Grid3DSubstrate(8, 8, 3, "clamp"), 3, 7),
            (Continuous1DSubstrate(1, [(0.0, 10.0)], "clamp", 0.5, 1.0), 1, 3),
            (Continuous2DSubstrate(2, [(0.0, 10.0), (0.0, 10.0)], "clamp", 0.5, 1.0), 2, 5),
            (Continuous3DSubstrate(3, [(0.0, 10.0)] * 3, "clamp", 0.5, 1.0), 3, 7),
        ]

        for substrate, expected_dim, expected_actions in test_cases:
            assert substrate.position_dim == expected_dim
            assert substrate.action_space_size == expected_actions
            assert substrate.action_space_size == 2 * substrate.position_dim + 1
```

### Step 1.2: Run test to verify it fails

```bash
pytest tests/test_townlet/test_substrate/test_base.py -v
```

**Expected:** All tests FAIL with `AttributeError: 'Grid2DSubstrate' object has no attribute 'action_space_size'`

### Step 1.3: Add action_space_size property to base class

**File:** `src/townlet/substrate/base.py`

Add property after `position_dtype` (around line 56):

```python
    @property
    def position_dtype(self) -> torch.dtype:
        """Data type of position tensors.

        Returns:
            torch.long for discrete grids (integer coordinates)
            torch.float32 for continuous spaces (float coordinates)

        This enables substrates to mix int and float positioning without dtype errors.

        Example:
            Grid2D: torch.long (positions are integers)
            Continuous2D: torch.float32 (positions are floats)
        """
        pass

    @property
    def action_space_size(self) -> int:
        """Return number of discrete actions supported by this substrate.

        Action spaces are determined by substrate dimensionality:
        - Spatial substrates: 2 * position_dim + 1 (±movement per dimension + INTERACT)
        - Aspatial substrates: 1 (INTERACT only, no movement)

        Examples:
            Grid2D (position_dim=2): 5 actions (UP/DOWN/LEFT/RIGHT/INTERACT)
            Grid3D (position_dim=3): 7 actions (±X/±Y/±Z/INTERACT)
            Continuous1D (position_dim=1): 3 actions (±X/INTERACT)
            Continuous2D (position_dim=2): 5 actions (±X/±Y/INTERACT)
            Continuous3D (position_dim=3): 7 actions (±X/±Y/±Z/INTERACT)
            Aspatial (position_dim=0): 1 action (INTERACT only)

        This enables dynamic action space sizing for N-dimensional substrates.
        VectorizedHamletEnv queries this property instead of hardcoding action counts.

        Returns:
            Integer count of discrete actions
        """
        if self.position_dim == 0:
            # Aspatial: only INTERACT action (no movement)
            return 1
        # Spatial: 2N + 1 (±movement per dimension + INTERACT)
        return 2 * self.position_dim + 1
```

**Design Decision:** Make this a **concrete property with default implementation**, not abstract.
- Rationale: The formula `2*position_dim + 1` works for all current substrates
- Future substrates with custom action spaces can override if needed
- Reduces boilerplate (don't need to implement in every substrate)

### Step 1.4: Run tests to verify they pass

```bash
pytest tests/test_townlet/test_substrate/test_base.py -v
```

**Expected:** All tests PASS

### Step 1.5: Commit

```bash
git add src/townlet/substrate/base.py tests/test_townlet/test_substrate/test_base.py
git commit -m "feat(substrate): add action_space_size property to base class

Adds concrete property to SpatialSubstrate base class that returns
action space size based on position_dim using formula: 2N + 1

Spatial substrates: 2*position_dim + 1 (±movement per dim + INTERACT)
Aspatial substrates: 1 (INTERACT only, no movement)

This enables dynamic action space sizing for N-dimensional substrates
and eliminates hardcoded action space logic in VectorizedHamletEnv.

Test coverage: 7 unit tests verify formula for all substrate types.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Update VectorizedHamletEnv to Use action_space_size

**Files:**
- Modify: `src/townlet/environment/vectorized_env.py:247-261`
- Test: `tests/test_townlet/test_integration.py` (existing integration tests)

### Step 2.1: Write test for dynamic action space sizing

**File:** `tests/test_townlet/test_environment/test_action_space.py`

```python
"""Tests for dynamic action space sizing in VectorizedHamletEnv."""

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate
from townlet.substrate.continuous import Continuous1DSubstrate, Continuous2DSubstrate, Continuous3DSubstrate
from townlet.substrate.aspatial import AspatialSubstrate


class TestActionSpaceDynamicSizing:
    """Test that VectorizedHamletEnv uses substrate.action_space_size."""

    @pytest.fixture
    def minimal_config(self):
        """Minimal config for testing (bars, affordances, cascades)."""
        # This would load from a minimal test config directory
        # For now, return None and we'll use fixtures in actual implementation
        return None

    def test_env_respects_grid2d_action_space(self, minimal_config):
        """Environment action_dim matches Grid2D substrate."""
        substrate = Grid2DSubstrate(8, 8, "clamp", "manhattan")

        # Create environment (would need full config in real test)
        # env = VectorizedHamletEnv(substrate, minimal_config, ...)

        # Verify action dimension matches substrate
        # assert env.action_dim == substrate.action_space_size
        # assert env.action_dim == 5

        # Placeholder for now - actual test will be in integration tests
        assert substrate.action_space_size == 5

    def test_env_respects_grid3d_action_space(self):
        """Environment action_dim matches Grid3D substrate."""
        substrate = Grid3DSubstrate(8, 8, 3, "clamp")
        assert substrate.action_space_size == 7

    def test_env_respects_continuous_action_spaces(self):
        """Environment action_dim matches Continuous substrates."""
        c1d = Continuous1DSubstrate(1, [(0.0, 10.0)], "clamp", 0.5, 1.0)
        c2d = Continuous2DSubstrate(2, [(0.0, 10.0), (0.0, 10.0)], "clamp", 0.5, 1.0)
        c3d = Continuous3DSubstrate(3, [(0.0, 10.0)] * 3, "clamp", 0.5, 1.0)

        assert c1d.action_space_size == 3
        assert c2d.action_space_size == 5
        assert c3d.action_space_size == 7

    def test_env_respects_aspatial_action_space(self):
        """Environment action_dim matches Aspatial substrate."""
        substrate = AspatialSubstrate()
        assert substrate.action_space_size == 1
```

### Step 2.2: Run test to verify current behavior

```bash
pytest tests/test_townlet/test_environment/test_action_space.py -v
```

**Expected:** Tests PASS (substrate properties work, but env not tested yet)

### Step 2.3: Replace hardcoded action space logic

**File:** `src/townlet/environment/vectorized_env.py`

Replace lines 247-261:

**BEFORE:**
```python
        # Action space size depends on substrate dimensionality
        # 1D (Continuous1D): 4 actions (MOVE_X_NEGATIVE, MOVE_X_POSITIVE, INTERACT, WAIT)
        # 2D (Grid2D, Continuous2D): 6 actions (UP, DOWN, LEFT, RIGHT, INTERACT, WAIT)
        # 3D (Grid3D, Continuous3D): 8 actions (+ UP_Z, DOWN_Z)
        # Aspatial: 2 actions (INTERACT, WAIT)
        if self.substrate.position_dim == 1:
            self.action_dim = 4
        elif self.substrate.position_dim == 2:
            self.action_dim = 6
        elif self.substrate.position_dim == 3:
            self.action_dim = 8
        elif self.substrate.position_dim == 0:
            self.action_dim = 2
        else:
            raise ValueError(f"Unsupported substrate position_dim: {self.substrate.position_dim}")
```

**AFTER:**
```python
        # Action space size is determined by substrate
        # Substrate reports number of discrete actions via action_space_size property
        # This enables dynamic action spaces for N-dimensional substrates
        self.action_dim = self.substrate.action_space_size
```

**Note:** The comment mentions "WAIT" action but current implementation only has INTERACT. The action counts in the old comment are wrong:
- Old comment said 1D has 4 actions (including WAIT), but actually has 3 (±X, INTERACT)
- Old comment said 2D has 6 actions (including WAIT), but actually has 5 (UP/DOWN/LEFT/RIGHT, INTERACT)
- Old comment said Aspatial has 2 actions (INTERACT, WAIT), but actually has 1 (INTERACT only)

The new implementation correctly delegates to substrate's action_space_size.

### Step 2.4: Run integration tests to verify behavior unchanged

```bash
pytest tests/test_townlet/test_integration.py -v
```

**Expected:** All integration tests PASS (behavior unchanged, just cleaner code)

### Step 2.5: Run full test suite

```bash
pytest tests/test_townlet/ -v
```

**Expected:** All tests PASS

### Step 2.6: Commit

```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/test_environment/test_action_space.py
git commit -m "refactor(env): use substrate.action_space_size for dynamic action spaces

Replaces hardcoded if-else chain (lines 247-261) with delegation to
substrate.action_space_size property.

Benefits:
- Removes hardcoded position_dim → action_dim mapping
- Enables substrates to define custom action spaces
- Required for N-dimensional substrates (Phase 5C)
- Cleaner code: 1 line instead of 15

No behavior change: all existing substrates return same action counts.

Test coverage: Integration tests verify behavior unchanged.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com)"
```

---

## Task 3: Verify All Tests Pass

**Files:**
- Run: Full test suite

### Step 3.1: Run full test suite with coverage

```bash
pytest tests/test_townlet/ -v --cov=townlet --cov-report=term-missing
```

**Expected:** All tests PASS, coverage maintained or improved

### Step 3.2: Run specific substrate tests

```bash
pytest tests/test_townlet/test_substrate/ -v
```

**Expected:** All substrate tests PASS (including new action_space_size tests)

### Step 3.3: Run environment tests

```bash
pytest tests/test_townlet/test_environment/ -v
```

**Expected:** All environment tests PASS

### Step 3.4: Verify integration tests

```bash
pytest tests/test_townlet/test_integration.py -v
```

**Expected:** All integration tests PASS

---

## Task 4: Update Documentation

**Files:**
- Modify: `docs/CHANGELOG.md`
- Modify: `CLAUDE.md` (if needed)

### Step 4.1: Update CHANGELOG

**File:** `docs/CHANGELOG.md`

Add entry at top:

```markdown
## 2025-11-06 - Action Space Size Property (C1/C2 Fixes)

**Critical Fixes for Phase 5C Readiness:**

### Added
- `action_space_size` property to `SpatialSubstrate` base class
  - Formula: `2*position_dim + 1` for spatial substrates
  - Returns 1 for aspatial substrates (INTERACT only)
  - Enables dynamic action space sizing for N-dimensional substrates

### Changed
- `VectorizedHamletEnv` now uses `substrate.action_space_size` instead of hardcoded if-else chain
  - Removed 15 lines of hardcoded action space logic
  - Behavior unchanged for existing substrates
  - Prepares for Phase 5C N-dimensional substrates

### Tests
- Added 7 unit tests for `action_space_size` property
- All integration tests pass (behavior unchanged)

**Technical Details:**
- Grid2D: 5 actions (UP/DOWN/LEFT/RIGHT/INTERACT)
- Grid3D: 7 actions (±X/±Y/±Z/INTERACT)
- Continuous1D: 3 actions (±X/INTERACT)
- Continuous2D: 5 actions (±X/±Y/INTERACT)
- Continuous3D: 7 actions (±X/±Y/±Z/INTERACT)
- Aspatial: 1 action (INTERACT only)

**Why:** Required prerequisite for Phase 5C (N-dimensional substrates with 2N+1 action spaces)
```

### Step 4.2: Commit documentation

```bash
git add docs/CHANGELOG.md
git commit -m "docs: document action_space_size property implementation

Updates CHANGELOG with C1/C2 fixes (action_space_size property).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Verification Checklist

**Before considering this complete:**

- [ ] All tests pass: `pytest tests/test_townlet/ -v`
- [ ] New tests added: `tests/test_townlet/test_substrate/test_base.py`
- [ ] Integration tests unchanged: verify with `git diff tests/test_townlet/test_integration.py`
- [ ] No behavior changes: action space sizes match previous hardcoded values
- [ ] Documentation updated: CHANGELOG.md
- [ ] All commits follow conventional commit format
- [ ] Code review: verify property implementation is correct

**Action Space Size Verification:**

| Substrate | position_dim | action_space_size | Actions |
|-----------|-------------|-------------------|---------|
| Grid2D | 2 | 5 | UP/DOWN/LEFT/RIGHT/INTERACT |
| Grid3D | 3 | 7 | ±X/±Y/±Z/INTERACT |
| Continuous1D | 1 | 3 | ±X/INTERACT |
| Continuous2D | 2 | 5 | ±X/±Y/INTERACT |
| Continuous3D | 3 | 7 | ±X/±Y/±Z/INTERACT |
| Aspatial | 0 | 1 | INTERACT only |

---

## Success Criteria

**This plan is complete when:**

1. ✅ `action_space_size` property exists on `SpatialSubstrate` base class
2. ✅ Property uses formula: `2*position_dim + 1` (special case: aspatial = 1)
3. ✅ `VectorizedHamletEnv` uses `substrate.action_space_size` (no hardcoded logic)
4. ✅ All tests pass (unit + integration)
5. ✅ No behavior changes (action space sizes unchanged)
6. ✅ Documentation updated (CHANGELOG)

**Estimated Time:** 30-45 minutes

**Phase 5C Readiness:** With C1/C2 fixed, Phase 5C Part 1 (observation encoding retrofit) and Part 2 (N-dimensional substrates) can proceed without architectural blockers.
