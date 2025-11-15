Title: Spatial feasibility check assumes only 1 agent; ignores `population.num_agents`

Severity: medium
Status: RESOLVED
Resolved Date: 2025-11-15

Subsystem: universe/compiler
Affected Version/Branch: main

Affected Files:
- `src/townlet/universe/compiler.py:1189` (required_cells = required + 1)

Description:
- The grid capacity check uses `required_cells = required + 1` (+1 agent), but does not account for multi-agent populations.
- For `num_agents > 1`, agents may exceed available cells even when affordances fit, if distinct spawn cells are required.

Reproduction:
1) Set `population.num_agents = 8` on a 3×3 grid (9 cells) with 2+ affordances.
2) Compiler passes feasibility check, but environment may collide at spawn or rely on overlapping positions.

Expected Behavior:
- When unique spawn positions are required, compiler should check `required_cells = required_affordances + num_agents`.

Actual Behavior:
- Always adds only 1 agent in feasibility calculation.

Root Cause:
- Hard-coded `+ 1` rather than reading `population.num_agents`.

Proposed Fix (Breaking OK):
- Use `num_agents = max(1, raw_configs.population.num_agents)` and compute `required_cells = required_affordances + num_agents`.
- If overlapping spawns are acceptable, make this a config switch and document behavior.

Migration Impact:
- Packs that relied on overlaps must update configs or set policy explicitly.

Alternatives Considered:
- Treat as warning; less safe and drifts from env behavior where overlaps are discouraged.

Tests:
- Add compiler test that fails when `required_cells` exceeds capacity for multi-agent cases.

Owner: compiler

---

## RESOLUTION

**Fixed By**: Systematic debugging process using superpowers:systematic-debugging skill
**Resolution Date**: 2025-11-15

### Implementation Summary

**Phase 1 (Root Cause Investigation)**: Identified that `required_cells = required + 1` on line 1188 hard-coded a single agent assumption, ignoring the actual `population.num_agents` from configuration. Available data confirmed `raw_configs.population.num_agents` is accessible and already used elsewhere in compiler (line 1898).

**Phase 2 (Pattern Analysis)**: Found that compiler correctly uses `num_agents` for VariableRegistry initialization at line 1898. Only spatial feasibility check was using hard-coded value. Existing test pattern uses config cloning + mutation + stage4 validation with UAC-VAL-001 error code.

**Phase 3 (Hypothesis)**: Confirmed that replacing `required + 1` with `required + raw_configs.population.num_agents` would correctly validate multi-agent spatial feasibility without breaking single-agent configs (backward compatible).

**Phase 4 (Implementation)**:

1. **Core Fix** (`src/townlet/universe/compiler.py:1188-1196`):
   ```python
   # BEFORE:
   required_cells = required + 1  # +1 for the agent

   # AFTER:
   num_agents = raw_configs.population.num_agents
   required_cells = required + num_agents
   ```

2. **Error Message Enhancement**:
   - Added proper pluralization: "agent" vs "agents"
   - Shows actual number of agents required in error message
   - Format: `"Grid has X cells but need Y (Z affordances + N agents)"`

3. **Docstring Update** (lines 1138-1142):
   - Updated from "affordances + agent" to "affordances + agents"
   - Added note: "Accounts for population.num_agents from training.yaml"

4. **Test Coverage** (`tests/test_townlet/unit/universe/test_grid_feasibility.py` - NEW):
   - `test_spatial_feasibility_single_agent_sufficient_space`: Backward compat validation
   - `test_spatial_feasibility_multi_agent_insufficient_space`: Core bug test (8 agents + 2 affordances on 3×3 grid)
   - `test_spatial_feasibility_multi_agent_exact_capacity`: Edge case (exact fit)
   - `test_spatial_feasibility_multi_agent_over_capacity_by_one`: Edge case (over by one)

### Test Results
- New tests: 4/4 PASSED ✓
- Regression test: 1/1 PASSED ✓
- Full universe test suite: 266/266 PASSED ✓

### Code Review
- Reviewer: feature-dev:code-reviewer subagent
- Status: ✅ APPROVED
- Score: 100/100 (Excellent)
- Findings:
  - Implementation correct, well-tested
  - Complete backward compatibility for all 19 existing single-agent configs
  - Comprehensive edge case coverage
  - No other code locations need similar fixes
  - No blocking issues found

### Files Modified
1. `src/townlet/universe/compiler.py` - Core fix + docstring (lines 1138-1142, 1188-1196)
2. `tests/test_townlet/unit/universe/test_grid_feasibility.py` - New comprehensive test suite (145 lines, 4 tests)

### Migration Notes
- All existing configs use `num_agents: 1` (verified 19 config packs)
- Zero breaking changes for single-agent configurations
- Multi-agent configs that previously passed incorrectly will now fail at compile-time (intended behavior)

### Impact
- ✅ Correctly accounts for all agents in spatial feasibility check
- ✅ Maintains backward compatibility for single-agent configs (num_agents=1)
- ✅ Clear error messages with proper pluralization
- ✅ Comprehensive test coverage for edge cases
- ✅ No other locations need similar fixes

### Example Error Messages

**Multi-agent insufficient space** (8 agents, 2 affordances, 3×3 grid):
```
Spatial impossibility: Grid has 9 cells (3×3) but need 10 (2 affordances + 8 agents).
```

**Single agent** (1 agent, 3 affordances, 1×1 grid):
```
Spatial impossibility: Grid has 1 cells (1×1) but need 4 (3 affordances + 1 agent).
```
