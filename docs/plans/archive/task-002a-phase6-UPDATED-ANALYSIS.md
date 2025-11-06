# Phase 6 Updated Analysis - Alignment with Phase 5C Implementation

**Date**: 2025-11-06
**Status**: Under Review
**Author**: Claude (Plan Adaptation)

---

## Executive Summary

**Critical Finding**: Phase 5C already implemented 90% of Phase 6's goals!

The original Phase 6 plan expected to add substrate-based observation encoding, but Phase 5C already:
- ✅ Added `observation_encoding` parameter to all substrates
- ✅ Implemented `encode_observation()` and `get_observation_dim()` methods
- ✅ Integrated substrate methods into ObservationBuilder
- ✅ Updated VectorizedHamletEnv to use `substrate.get_observation_dim()`
- ✅ Created 229 passing tests for all substrate modes

**What's Left for Phase 6**:
1. Remove Grid2D-specific hardcoding in POMDP position normalization
2. Update outdated comments (still mention "one-hot")
3. Add observation_encoding mode tests for integration
4. Update documentation
5. Clean up unused parameters (grid_size)

This document proposes a simplified Phase 6 plan aligned with Phase 5C's implementation.

---

## Phase 5C vs Original Phase 6 Plan Comparison

### Design Differences

| Aspect | Original Plan | Phase 5C Implementation | Status |
|--------|--------------|------------------------|--------|
| **Parameter name** | `position_encoding` | `observation_encoding` | ✅ Better naming |
| **Encoding modes** | "onehot", "coords" | "relative", "scaled", "absolute" | ✅ More modes |
| **get_observation_dim()** | `(partial_observability, vision_range)` | No parameters | ✅ Simpler API |
| **POMDP encoding** | Combined in get_observation_dim() | Separate `encode_partial_observation()` | ✅ Better separation |
| **One-hot support** | Required for backward compat | Not implemented | ⚠️ Decision needed |

### Why Phase 5C's Design Is Better

1. **observation_encoding > position_encoding**:
   - More accurate: encodes positions AND metadata (scaled mode)
   - Consistent with substrate terminology elsewhere

2. **3 modes > 2 modes**:
   - "relative": Normalized [0,1] for transfer learning
   - "scaled": Normalized + metadata for size-aware strategies
   - "absolute": Raw coords for physical simulation
   - vs plan's "onehot" (infeasible for 3D) and "coords" (ambiguous)

3. **get_observation_dim() without parameters**:
   - Simpler: Encoding mode is substrate's responsibility
   - Cleaner: Partial obs handled by separate method
   - Decoupled: Substrate doesn't need to know about POMDP implementation details

4. **Separate encode_partial_observation()**:
   - Clear separation of concerns
   - POMDP-specific logic isolated
   - Already implemented and tested (229 tests)

### One-Hot Encoding Decision

**Original Plan Rationale**: Keep one-hot for backward compatibility with existing checkpoints

**Phase 5C Reality**: All configs already use "relative" encoding, so backward compatibility is already broken

**Options**:
1. **Add "onehot" as 4th mode** (moderate effort):
   - Pros: Supports original plan's backward compat goal
   - Cons: Only useful for 2D grids ≤8×8; infeasible for 3D/larger grids
   - Effort: ~4 hours (add mode, tests, docs)

2. **Keep current 3 modes** (recommended):
   - Pros: Simpler, already tested, supports all substrates
   - Cons: Can't load old checkpoints (but they're already incompatible)
   - Effort: 0 hours

**Recommendation**: Option 2. One-hot encoding is a dead end for TASK-002A's goal of supporting 3D, continuous, and N-dimensional substrates.

---

## What Phase 5C Already Achieved

### 1. Substrate Methods (Original Task 6.1) ✅ COMPLETE

All substrates implement:

```python
def encode_observation(positions, affordances) -> Tensor:
    """Encode positions using configurable observation_encoding mode."""
    if self.observation_encoding == "relative":
        return self._encode_relative(positions, affordances)
    elif self.observation_encoding == "scaled":
        return self._encode_scaled(positions, affordances)
    elif self.observation_encoding == "absolute":
        return self._encode_absolute(positions, affordances)

def get_observation_dim() -> int:
    """Return observation dimensionality based on encoding mode."""
    if self.observation_encoding == "relative":
        return self.position_dim  # 2 for Grid2D, 3 for Grid3D, N for GridND
    elif self.observation_encoding == "scaled":
        return 2 * self.position_dim  # Includes range metadata
    elif self.observation_encoding == "absolute":
        return self.position_dim

def encode_partial_observation(positions, affordances, vision_range) -> Tensor:
    """Encode local window for POMDP."""
    # Grid2D: Returns [num_agents, (2*vision_range+1)²] local grid
    # GridND: Raises ValueError (impractical for N≥4)
    # Aspatial: Returns empty tensor
```

**Status**: Implemented in Phase 5C for all 8 substrate types
- Grid2D, Grid3D ✅
- Continuous1D, Continuous2D, Continuous3D ✅
- GridND, ContinuousND ✅
- Aspatial ✅

**Tests**: 229 passing tests covering all modes and substrates

---

### 2. VectorizedHamletEnv Integration (Original Task 6.2) ✅ COMPLETE

File: `src/townlet/environment/vectorized_env.py`

**Observation dimension calculation** (lines 191-212):

```python
if partial_observability:
    window_size = 2 * vision_range + 1
    self.observation_dim = (
        window_size * window_size                      # Local grid
        + self.substrate.position_dim                  # Normalized position
        + meter_count                                  # 8 meters
        + (self.num_affordance_types + 1)             # Affordance one-hot
    )
else:
    self.observation_dim = (
        self.substrate.get_observation_dim()           # ✅ Uses substrate!
        + meter_count
        + (self.num_affordance_types + 1)
    )

self.observation_dim += 4  # Temporal features
```

**Status**: Already using `substrate.get_observation_dim()` for full observability

**Issue**: POMDP calculation uses `substrate.position_dim` instead of asking substrate for normalized position dims
- Currently assumes position_dim = normalized_position_dim
- Works for current substrates but not future-proof

---

### 3. ObservationBuilder Integration (Original Tasks 6.3-6.5) ✅ MOSTLY COMPLETE

File: `src/townlet/environment/observation_builder.py`

**Full observability** (line 129):
```python
grid_encoding = self.substrate.encode_observation(positions, affordances)
```
✅ Already delegating to substrate!

**Partial observability** (line 160):
```python
local_grids = self.substrate.encode_partial_observation(positions, affordances, vision_range)
```
✅ Already delegating to substrate!

**Issue**: Manual position normalization (lines 165-174):
```python
if hasattr(self.substrate, "width") and hasattr(self.substrate, "height"):
    # Grid2D-specific hardcoding! ⚠️
    normalized_positions = positions.float() / torch.tensor(
        [self.substrate.width - 1, self.substrate.height - 1],
        device=self.device,
    )
else:
    normalized_positions = positions.float()
```

**Problem**: Hardcoded Grid2D logic breaks for:
- Grid3D (needs width, height, depth)
- Continuous substrates (already normalized)
- GridND/ContinuousND (N-dimensional)

**Fix**: Use `substrate.encode_observation()` with "relative" mode instead

---

## What's Left for Phase 6 (Simplified)

### Task 6.1: Fix POMDP Position Normalization ⚠️ NEW

**Problem**: ObservationBuilder manually normalizes positions for POMDP (Grid2D-specific)

**Solution**: Use substrate's encode_observation() for normalized positions

**Files**:
- `src/townlet/environment/observation_builder.py`

**Changes**:
```python
# BEFORE (lines 165-174)
if hasattr(self.substrate, "width") and hasattr(self.substrate, "height"):
    normalized_positions = positions.float() / torch.tensor(...)
else:
    normalized_positions = positions.float()

# AFTER
# Use substrate's "relative" encoding for normalized positions
normalized_positions = self.substrate.encode_observation(positions, {})
# Returns [num_agents, position_dim] normalized to [0, 1]
```

**Edge case**: Aspatial substrate returns [num_agents, 0], which is correct

**Tests**: Update POMDP tests to verify position normalization for all substrates

---

### Task 6.2: Update Comments and Documentation

**Problem**: Comments still mention "one-hot" but code uses coordinate encoding

**Files**:
- `src/townlet/environment/observation_builder.py` (line 15, 122)
- `src/townlet/environment/vectorized_env.py` (line 204)
- `CLAUDE.md`

**Changes**:
- Update "One-hot grid position" → "Substrate-encoded position (relative/scaled/absolute)"
- Document observation_encoding modes
- Add examples for each mode

---

### Task 6.3: Add Integration Tests for observation_encoding Modes

**Problem**: Tests exist for individual substrates, but not end-to-end integration

**New test file**: `tests/test_townlet/integration/test_observation_encoding_modes.py`

**Test cases**:
1. Full observability with each mode (relative, scaled, absolute)
2. POMDP with relative mode (should work)
3. Observation dimensions match expected values
4. Network can train with different modes

**Estimated**: 15-20 tests

---

### Task 6.4: Clean Up Unused Parameters

**Problem**: ObservationBuilder stores `grid_size` but doesn't use it

**Changes**:
- Remove `grid_size` parameter from `__init__`
- Update VectorizedHamletEnv to not pass it
- Verify no other dependencies

**Risk**: Low (grid_size only used in old one-hot encoding logic)

---

### Task 6.5: Update Documentation (CLAUDE.md)

**Sections to add**:
1. Observation encoding modes (relative/scaled/absolute)
2. When to use each mode
3. Observation dimension calculations
4. Migration guide for old checkpoints

---

## Updated Task Breakdown

### Task 6.1: Fix POMDP Position Normalization (2 hours)

**Steps**:
1. Write test for POMDP position normalization using substrate
2. Update _build_partial_observations() to use substrate.encode_observation()
3. Verify test passes
4. Run full test suite

**Verification**:
```bash
uv run pytest tests/test_townlet/integration/test_observation_builder_pomdp.py -v
```

---

### Task 6.2: Update Comments (30 minutes)

**Steps**:
1. Find all "one-hot" references
2. Update to "substrate-encoded" with mode examples
3. Update docstrings

**Verification**: Manual review

---

### Task 6.3: Add Integration Tests (3 hours)

**Steps**:
1. Create test_observation_encoding_modes.py
2. Test each mode (relative/scaled/absolute) end-to-end
3. Test POMDP with different substrates
4. Verify observation dimensions

**Verification**:
```bash
uv run pytest tests/test_townlet/integration/test_observation_encoding_modes.py -v
```

---

### Task 6.4: Clean Up grid_size Parameter (1 hour)

**Steps**:
1. Remove grid_size from ObservationBuilder.__init__
2. Update VectorizedHamletEnv call
3. Search for any other uses
4. Run tests

**Verification**:
```bash
grep -r "grid_size" src/townlet/environment/
uv run pytest tests/test_townlet/ -v
```

---

### Task 6.5: Update Documentation (2 hours)

**Steps**:
1. Add observation_encoding section to CLAUDE.md
2. Document each mode with examples
3. Add observation dimension formulas
4. Update migration guide

**Verification**: Manual review

---

## Total Effort Estimate

| Task | Original Estimate | Updated Estimate | Savings |
|------|------------------|------------------|---------|
| 6.1 | 4 hours | 2 hours | 2 hours |
| 6.2 | 2 hours | 0.5 hours | 1.5 hours |
| 6.3 | 1 hour | 3 hours | -2 hours |
| 6.4 | 2 hours | 1 hour | 1 hour |
| 6.5 | 3 hours | SKIPPED (done in 5C) | 3 hours |
| 6.6 | 2 hours | SKIPPED (done in 5C) | 2 hours |
| 6.7 | 4 hours | 2 hours | 2 hours |
| 6.8 | 1 hour | 2 hours | -1 hour |
| **Total** | **19 hours** | **10.5 hours** | **8.5 hours saved** |

**Reason for savings**: Phase 5C already implemented substrate methods and integration

---

## Risks and Mitigations

### Risk 1: Breaking POMDP with position normalization change

**Mitigation**: Add comprehensive POMDP tests before making changes

### Risk 2: Observation dimensions mismatch after grid_size removal

**Mitigation**: Verify obs_dim calculation in all modes before removing grid_size

### Risk 3: Missing one-hot mode breaks existing use cases

**Mitigation**: Document that observation_encoding="relative" replaces all one-hot use cases

---

## Success Criteria

1. ✅ All 229 Phase 5C tests continue passing
2. ✅ POMDP position normalization uses substrate method (no hardcoding)
3. ✅ Integration tests verify all observation_encoding modes work end-to-end
4. ✅ Documentation updated with observation_encoding modes
5. ✅ No Grid2D-specific hardcoding remains in ObservationBuilder
6. ✅ Observation dimensions correct for all substrates and modes

---

## Recommendation

**Proceed with simplified Phase 6 plan**: 10.5 hours instead of 19 hours

**Key changes from original plan**:
1. Skip substrate method implementation (already done in 5C)
2. Skip VectorizedHamletEnv obs_dim update (already done in 5C)
3. Skip ObservationBuilder constructor refactor (already done in 5C)
4. Focus on cleanup: Remove hardcoding, update tests, document

**Next steps**:
1. Review agent validates this analysis
2. Execute simplified Phase 6 tasks using TDD + subagents
3. Verify all tests pass
4. Update documentation
