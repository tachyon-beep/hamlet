Title: Affordance randomization can collide with agent spawn positions

Severity: medium
Status: FIXED
Fixed Date: 2025-11-14
Fixed In: bug-fixing-001

Subsystem: environment/vectorized
Affected Files:
- `src/townlet/environment/vectorized_env.py:randomize_affordance_positions()` (modified)
- `src/townlet/environment/vectorized_env.py:reset()` (modified)

## Summary

Agents spawned on top of affordances with **95-97% collision probability** on small grids (3×3 with 6 affordances, 3 agents). This happened because affordances were placed first, then agents spawned randomly without knowledge of occupied positions.

## Root Cause

**Order of operations in `reset()`:**
1. Line 609: `randomize_affordance_positions()` - places affordances
2. Line 615: `initialize_positions(num_agents)` - spawns agents randomly

Agents spawn **after** affordances are placed, but `initialize_positions()` samples positions randomly without knowing which are already occupied by affordances.

## Investigation (Systematic Debugging)

### Phase 1: Root Cause Investigation

**Reproduction Test Results:**
```
3×3 grid (9 positions), 6 affordances, 3 agents
Collision rate: 95-97% over 100 runs
```

**Evidence:**
- Traced initialization flow in `reset()`
- Confirmed affordances placed first (line 609-610)
- Confirmed agents spawn afterward (line 615)
- `initialize_positions()` has no knowledge of occupied positions

### Phase 2: Pattern Analysis

Found that JANK-10 fix already implemented collision-free placement for affordances:
- Samples N positions with collision detection
- Retries up to 10 times
- Falls back to enumeration if needed

**Pattern:** Extend this to sample `(affordances + agents)` positions together, then split.

### Phase 3: Hypothesis

**Hypothesis:** If we sample `(num_affordances + num_agents)` positions together with collision detection, we guarantee zero collisions.

**Test:** Verified pattern works with 100% success rate on small, medium, and large grids.

### Phase 4: Implementation

**Changes:**

1. **Modified `randomize_affordance_positions()` signature:**
   - Old: `def randomize_affordance_positions(self) -> None`
   - New: `def randomize_affordance_positions(self) -> torch.Tensor | None`
   - Now returns agent spawn positions

2. **Extended sampling logic:**
   ```python
   # Sample (affordances + agents) positions together
   total_positions_needed = len(self.affordances) + self.num_agents
   sampled = substrate.initialize_positions(total_positions_needed, device)

   # Check for collisions among ALL positions
   # ... retry logic with fallback ...

   # Split: first N for affordances, remaining M for agents
   affordance_positions = sampled[:len(self.affordances)]
   agent_positions = sampled[len(self.affordances):]
   ```

3. **Updated `reset()` to use returned positions:**
   ```python
   if self.randomize_affordances:
       agent_positions = self.randomize_affordance_positions()
       self.positions = agent_positions  # Use collision-free positions
   else:
       self._apply_configured_affordance_positions()
       self.positions = self.substrate.initialize_positions(...)  # Still random
   ```

## Results

**Before Fix:**
- 3×3 grid, 6 affordances, 3 agents: **95% collision rate**

**After Fix:**
- Same configuration: **0% collision rate** (100 resets tested)
- L0_0_minimal (real environment): **0/100 collisions**

**Test Coverage:**
- ✅ All 59 environment unit tests passing
- ✅ All 5 affordance randomization tests passing
- ✅ Zero collisions verified on small, medium, and large grids

## Breaking Changes

**API Change:** `randomize_affordance_positions()` now returns agent positions instead of `None`.

**Migration Impact:**
- Internal method - no external callers
- `reset()` automatically uses returned positions
- Configured affordance positions still spawn agents randomly (TODO in code)

## Trade-offs

**Still Unresolved:**
- When using **configured** (non-randomized) affordance positions, agents still spawn randomly and CAN collide
- Added TODO comment for future enhancement
- Decision: Fix randomization case first (99% of use cases), defer configured case

**Why acceptable:**
- Randomization is the primary use case (all curriculum levels)
- Configured positions are rarely used
- Can be addressed in future PR if needed

## Future Improvements

1. **Handle configured affordance case:** Extend collision-free logic to configured positions
2. **Multi-agent property tests:** Add property-based tests for various grid sizes and agent counts
3. **Collision detection in step():** Consider runtime collision detection/resolution

## Performance Impact

**Minimal overhead:**
- Sampling `(N + M)` positions instead of `N` positions
- Collision detection overhead is same (already present from JANK-10)
- No measurable performance degradation

## Related Work

- **JANK-10**: Implemented collision-free affordance placement (foundation for this fix)
- **BUG-20**: Grid feasibility ignores num_agents (related capacity planning issue)

## Verification

**Test command:**
```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=/home/john/hamlet/src uv run pytest \
  tests/test_townlet/unit/environment/test_vectorized_env.py -v
```

**Result:** 59 passed ✅
