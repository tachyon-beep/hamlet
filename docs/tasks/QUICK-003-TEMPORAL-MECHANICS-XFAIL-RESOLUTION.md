# QUICK-003: Temporal Mechanics xfail Resolution

**Status**: In Progress
**Priority**: Medium
**Estimated Effort**: 2-4 hours
**Dependencies**: None
**Created**: 2025-11-04
**Started**: 2025-11-04
**Progress**: 1/14 tests fixed (observation dimensions)

**Keywords**: temporal-mechanics, testing, xfail, L3, operating-hours, multi-tick-interactions
**Subsystems**: environment, affordance_engine, observation_builder, recording
**Files**:
- `src/townlet/environment/affordance_engine.py`
- `src/townlet/environment/observation_builder.py`
- `tests/test_townlet/integration/test_temporal_mechanics.py`

---

## AI-Friendly Summary (Skim This First!)

**What**: Investigate and implement missing temporal mechanics functionality to resolve 14 xfail tests (1 fixed, 13 remaining)
**Why**: Temporal mechanics (L3) is partially implemented but 13 tests are marked as xfail, preventing validation
**Scope**: Complete implementation of time-of-day, operating hours, multi-tick interactions, and early exit mechanics

**UPDATE 2025-11-04**: Fixed `test_observation_dimensions_with_temporal` - was expecting 3 temporal features but implementation provides 4 (added lifetime_progress for forward compatibility). Test now passes.

**Quick Assessment**:

- Current State:
  - ✅ Temporal mechanics tests written (14 tests)
  - ✅ Basic time progression implemented
  - ❌ Operating hours masking not implemented
  - ❌ Multi-tick interaction system not implemented
  - ❌ Early exit mechanics not implemented
  - ❌ Temporal observation features incomplete

- Goal: All 14 temporal mechanics tests passing (remove xfail markers)

- Impact: Enables Level 3 training with time-based dynamics and multi-tick interactions

---

## Problem Statement

### Context

Level 3 (temporal mechanics) introduces time-based game dynamics:
- 24-tick day/night cycle with operating hours (Job 9am-5pm, Bar 6pm-2am)
- Multi-tick interactions with progressive benefits (75% linear + 25% completion bonus)
- Early exit mechanics (agents keep accumulated benefits)
- Temporal observation features (time_of_day, interaction_progress)

Currently, 14 tests are marked as xfail because these features are not fully implemented.

### Current Limitations

**What Doesn't Work**:

- Operating hours masking (affordances should be unavailable outside hours)
- Multi-tick interaction system (4-tick Job completion)
- Progressive benefit accumulation (75% linear rewards)
- Completion bonus (25% on final tick)
- Early exit from interactions (keep progress)
- Temporal observation features (sin/cos time, interaction_progress)

**What We're Missing**:

- Time-based action masking in environment
- Multi-tick interaction state tracking
- Progressive reward calculation
- Early exit handling with benefit preservation
- Temporal feature encoding in observations

### Use Cases

**Primary Use Case**:
Agent learns temporal planning - go to Job during daytime, go to Bar at night, optimize interaction timing

**Secondary Use Cases**:

- Opportunity cost learning (commit to 4-tick Job vs. move to Bed)
- Time-sensitive decision making (Bar closing at 4am)
- Benefit accumulation strategies (complete vs. early exit)

---

## Solution Design

### Overview

Investigate each failing test to identify missing implementation components, then implement them systematically using TDD (tests already written - just need to remove xfail and implement features).

### Technical Approach

**Investigation Phase** (30 minutes):

1. Run each xfail test individually to see exact failure mode
2. Identify missing features in affordance_engine.py, observation_builder.py
3. Document required changes per test category

**Implementation Steps**:

1. **Time Progression** (30 min)
   - Verify 24-tick cycle implementation
   - Add sin/cos time encoding to observations
   - Test: `test_observation_dimensions_with_temporal`

2. **Operating Hours Masking** (45 min)
   - Implement time-based action masking in affordance_engine
   - Add operating hours validation
   - Test: `test_operating_hours_mask_job`, `test_bar_wraparound_hours`, `test_24_hour_affordances`

3. **Multi-Tick Interactions** (60 min)
   - Add interaction state tracking (current_tick, total_ticks)
   - Implement progressive benefit calculation (75% linear)
   - Add completion bonus logic (25% on final tick)
   - Add per-tick cost charging
   - Add interaction_progress to observations
   - Test: `test_progressive_benefit_accumulation`, `test_completion_bonus`, `test_multi_tick_job_completion`, `test_money_charged_per_tick`, `test_interaction_progress_in_observations`, `test_completion_bonus_timing`

4. **Early Exit Mechanics** (30 min)
   - Implement early exit from interaction (non-INTERACT action)
   - Ensure accumulated benefits are preserved
   - Reset interaction state on exit
   - Test: `test_early_exit_from_interaction`, `test_early_exit_keeps_progress`

5. **Integration Validation** (15 min)
   - Test multi-agent temporal independence
   - Test curriculum integration
   - Test temporal disabled fallback
   - Test: `test_multi_agent_temporal_interactions`, `test_temporal_mechanics_with_curriculum`, `test_temporal_mechanics_disabled_fallback`

**Key Design Decisions**:

- **Decision 1**: Use existing test suite (TDD already done - just implement to spec)
- **Decision 2**: Implement in order of dependency (time → masking → interactions → exit)

### Edge Cases

**Must Handle**:

- Bar operating hours wrap midnight (6pm-4am): 18-23, 0-3
- Early exit on first tick (no benefits accumulated yet)
- Completion bonus only on final tick (not on early exit)
- Multi-agent independent interaction states
- Temporal mechanics disabled fallback (legacy configs)

---

## Implementation Plan

### Phase 1: Investigation (30 minutes)

**Approach**:

```bash
# Run each xfail test to see failure mode
uv run pytest tests/test_townlet/integration/test_temporal_mechanics.py::TestTimeProgression::test_observation_dimensions_with_temporal -xvs

# Document exact error for each test category
```

**Output**: Document exact missing features per test

### Phase 2: Time Encoding (30 minutes)

**File**: `src/townlet/environment/observation_builder.py`

**Changes**:
- Add sin/cos time encoding to temporal extras
- Ensure interaction_progress is included

**Testing**:

- [ ] Remove xfail: `test_observation_dimensions_with_temporal`
- [ ] Test passes

### Phase 3: Operating Hours (45 minutes)

**File**: `src/townlet/environment/affordance_engine.py`

**Changes**:
- Add `_is_affordance_available(affordance_id, current_time)` method
- Check operating hours from config
- Mask actions for unavailable affordances

**Testing**:

- [ ] Remove xfail: `test_operating_hours_mask_job`
- [ ] Remove xfail: `test_bar_wraparound_hours`
- [ ] Remove xfail: `test_24_hour_affordances`
- [ ] Tests pass

### Phase 4: Multi-Tick Interactions (60 minutes)

**File**: `src/townlet/environment/affordance_engine.py`

**Changes**:
- Track interaction state: `(affordance_id, current_tick, total_ticks)`
- Implement progressive benefit calculation
- Implement completion bonus
- Charge costs per tick (not on completion)
- Update interaction_progress each tick

**Testing**:

- [ ] Remove xfail: `test_progressive_benefit_accumulation`
- [ ] Remove xfail: `test_completion_bonus`
- [ ] Remove xfail: `test_multi_tick_job_completion`
- [ ] Remove xfail: `test_money_charged_per_tick`
- [ ] Remove xfail: `test_interaction_progress_in_observations`
- [ ] Remove xfail: `test_completion_bonus_timing`
- [ ] Tests pass

### Phase 5: Early Exit (30 minutes)

**File**: `src/townlet/environment/affordance_engine.py`

**Changes**:
- Detect non-INTERACT action during interaction
- Preserve accumulated benefits
- Reset interaction state

**Testing**:

- [ ] Remove xfail: `test_early_exit_from_interaction`
- [ ] Remove xfail: `test_early_exit_keeps_progress`
- [ ] Tests pass

### Phase 6: Integration Validation (15 minutes)

**File**: `tests/test_townlet/integration/test_temporal_mechanics.py`

**Verification Steps**:

1. [ ] Remove xfail: `test_multi_agent_temporal_interactions`
2. [ ] Remove xfail: `test_temporal_mechanics_with_curriculum`
3. [ ] Remove xfail: `test_temporal_mechanics_disabled_fallback`
4. [ ] All 14 tests pass
5. [ ] No regressions in other tests

---

## Testing Strategy

**Test Requirements**:

- **Unit Tests**: None needed (integration tests already written)
- **Integration Tests**: 14 tests in test_temporal_mechanics.py (already written, just xfailed)

**Coverage Target**: No new coverage requirement (existing code modified)

**Test-Driven Development**:

- ✅ Tests already written (RED phase done)
- [ ] Remove xfail markers one by one (GREEN phase)
- [ ] Implement minimal code to pass each test
- [ ] Refactor if needed

**Approach**: Un-xfail tests in dependency order (time → masking → interactions → exit)

---

## Acceptance Criteria

**Must Have**:

- [ ] All 14 xfail tests converted to passing tests
- [ ] No xfail markers remaining in test_temporal_mechanics.py
- [ ] All tests pass (no regressions)
- [ ] Temporal mechanics fully functional for L3 training

**Success Metrics**:

- Test suite output: "0 xfailed" (currently "14 xfailed")
- L3 config can train successfully with temporal mechanics

---

## Risk Assessment

**Technical Risks**:

- ⚠️ **MEDIUM**: Multi-tick interaction state may conflict with vectorized environment
  - Mitigation: Use per-agent state tracking (already vectorized)

- ✅ **LOW**: Tests are comprehensive and well-documented
  - Mitigation: Follow test specs exactly

**Migration**:

- No migration needed (new feature, doesn't break existing L0-L2)

---

## Future Enhancements (Out of Scope)

**Not Included**:

- Temporal mechanics visualization in frontend (separate task)
- Advanced temporal strategies (learning to wait for operating hours)
- Calendar/weekly cycles (beyond 24-tick day)

**Rationale**: This task focuses on core temporal mechanics implementation, not visualization or advanced features

---

## References

**Related Tasks**:

- TASK-001: Variable-size meter system (enabled temporal extras)
- Docs: `docs/architecture/TRAINING_LEVELS.md` (L3 specification)

**Code Files**:

- `src/townlet/environment/affordance_engine.py` - Interaction and masking logic
- `src/townlet/environment/observation_builder.py` - Temporal feature encoding
- `tests/test_townlet/integration/test_temporal_mechanics.py` - 14 integration tests
- `configs/L3_temporal_mechanics/` - L3 config pack

**Documentation**:

- `docs/architecture/TRAINING_LEVELS.md` - L3 temporal mechanics specification
- `tests/test_townlet/integration/test_temporal_mechanics.py:0-50` - Test organization and status

---

**END OF TASK SPECIFICATION**
