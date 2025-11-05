# QUICK-003: PDR-002 No-Defaults Whitelist Cleanup

**Status**: Planned
**Priority**: High
**Estimated Effort**: 12-17 days (1500+ LOC across 22 files)
**Dependencies**: PDR-002 (No-Defaults Principle adopted)
**Created**: 2025-11-05

**Keywords**: PDR-002, no-defaults, whitelist, config-driven, UAC, BAC, refactoring
**Subsystems**: All config-driven systems (environment, population, curriculum, exploration, training, demo)
**Files**: 22 files across src/townlet/ (see Phase breakdown)

---

## AI-Friendly Summary (Skim This First!)

**What**: Refactor codebase to remove UAC/BAC defaults from code, leaving only infrastructure defaults in whitelist
**Why**: Current whitelist exempts 85% UAC/BAC parameters (violates PDR-002), hides 226+ defaults across 22 files
**Scope**: Remove 17 non-compliant whitelist entries, refactor 22 files to fail-fast on missing UAC/BAC configs

**Quick Assessment**:

- Current State:
  - ❌ 19 whitelist entries total
  - ❌ 17 entries (85%) violate PDR-002 by exempting UAC/BAC parameters
  - ❌ Hides 226+ defaults (grid_size, epsilon, learning_rate, energy costs, etc.)
  - ✅ 2 entries compliant (recording, tensorboard - infrastructure only)

- Goal:
  - ✅ 17 compliant whitelist entries (0 UAC/BAC exemptions)
  - ✅ All UAC/BAC parameters fail-fast with clear error messages
  - ✅ Configs are complete and self-documenting
  - ✅ Infrastructure defaults remain (device, port, logging, metadata)

- Impact:
  - 100% PDR-002 compliance
  - Complete config reproducibility
  - Self-documenting configs
  - Fail-fast on missing UAC/BAC params

---

## Problem Statement

### Context

PDR-002 (No-Defaults Principle) establishes that all UAC/BAC parameters must be explicitly specified in config files with NO code defaults. The current `.defaults-whitelist.txt` was created before PDR-002 adoption and exempts many UAC/BAC parameters, violating the policy.

**Whitelist Review Findings** (see `docs/investigations/PDR-002-WHITELIST-REVIEW.md`):
- 85% of whitelist entries violate PDR-002
- 226+ hidden defaults across 22 files
- Most exemptions are UAC/BAC (universe/brain mechanics), not infrastructure

### Current Limitations

**What Doesn't Work**:

- Code can use defaults for UAC/BAC parameters (grid_size, epsilon, learning_rate, energy costs)
- Incomplete configs silently work (using hidden defaults)
- Old configs break when code defaults change
- Operators don't know actual parameter values being used

**What We're Missing**:

- Fail-fast validation for missing UAC/BAC parameters
- Clear error messages showing what's missing and how to fix
- Self-documenting configs (all params visible in YAML)
- Complete reproducibility (same config always produces same behavior)

### Use Cases

**Primary Use Case**:
Researcher shares config file → colleague loads it → gets identical behavior (no hidden defaults)

**Secondary Use Cases**:

- Student reads config to understand system behavior (no code diving needed)
- Operator changes parameter → knows exactly what they're changing
- Config validation catches missing params before training starts (fail-fast)

---

## Solution Design

### Overview

Systematically remove UAC/BAC defaults from 22 files over 4 phases, replacing `.get()` calls with fail-fast validation. Deploy compliant whitelist that only exempts infrastructure (device, port, logging, metadata). Update all example configs to be complete.

### Technical Approach

**Implementation Strategy**:

1. **Deploy Compliant Whitelist** - Replace `.defaults-whitelist.txt` with compliant version (0 UAC/BAC exemptions)
2. **Run Linter** - Expose all violations that were hidden by non-compliant whitelist
3. **Refactor by Phase** - Fix violations phase-by-phase (config loading → environment → networks → remaining)
4. **Verify Compliance** - Linter passes, all configs complete, tests pass

**Key Design Decisions**:

- **Decision 1: Fail-Fast Validation**: Replace `.get(key, default)` with explicit required checks + clear error messages
- **Decision 2: Phased Approach**: Fix in dependency order (config loading → environment → networks → remaining)
- **Decision 3: No Breaking Changes**: All example configs updated to remain valid
- **Decision 4: Infrastructure Exemptions**: Keep whitelist for device, port, logging, metadata (not UAC/BAC)

### Edge Cases

**Must Handle**:

- Computed/derived values (observation_dim calculated from grid_size) - NOT in config
- Infrastructure params (device="cpu", port=8080) - Can have defaults if whitelisted
- Optional visualization metadata (video export settings) - Can have defaults if whitelisted
- Legacy configs - Must update all example configs to be complete

---

## Implementation Plan

### Phase 1: Config Loading (TASK-003) (2-3 days)

**Files**:
- `src/townlet/demo/runner.py`
- `src/townlet/demo/unified_server.py`

**Changes**:
- Remove 50+ `.get()` calls with defaults
- Add Pydantic DTOs for config validation (TASK-003)
- Add fail-fast validation with clear error messages
- Update L0, L0.5, L1, L2, L3 configs to be complete

**Example Refactor**:

```python
# ❌ BEFORE (non-compliant)
def __init__(self, config: dict):
    self.grid_size = config.get("grid_size", 8)  # Hidden default!
    self.epsilon = config.get("epsilon", 0.1)  # Hidden default!

# ✅ AFTER (compliant)
def __init__(self, config: dict):
    required_params = ["grid_size", "epsilon"]
    for param in required_params:
        if param not in config:
            raise ValueError(
                f"Missing required parameter '{param}'. "
                f"Add to config: {param}: [value]"
            )
    self.grid_size = config["grid_size"]
    self.epsilon = config["epsilon"]
```

**Testing**:
- [ ] Linter passes for demo/ files
- [ ] All example configs load without errors
- [ ] Missing param triggers clear error message
- [ ] No regressions in training

**LOC**: 200+ lines changed

### Phase 2: Environment (TASK-004A) (3-4 days)

**Files** (8 files):
- `src/townlet/environment/affordance_config.py`
- `src/townlet/environment/affordance_engine.py`
- `src/townlet/environment/cascade_config.py`
- `src/townlet/environment/cascade_engine.py`
- `src/townlet/environment/meter_dynamics.py`
- `src/townlet/environment/observation_builder.py`
- `src/townlet/environment/reward_strategy.py`
- `src/townlet/environment/vectorized_env.py`

**Changes**:
- Remove UAC defaults (energy costs, grid properties, meter depletion rates)
- Add schema validation for bars.yaml, affordances.yaml, cascades.yaml
- Add fail-fast validation for environment config sections
- Update all config packs (L0, L0.5, L1, L2, L3) with complete environment params

**Testing**:
- [ ] Linter passes for environment/ files
- [ ] All config packs load successfully
- [ ] Missing environment param triggers clear error
- [ ] No regressions in environment tests

**LOC**: 400+ lines changed

### Phase 3: Networks (TASK-005) (2-3 days)

**Files**:
- `src/townlet/agent/networks.py`

**Changes**:
- Remove BAC defaults (hidden_dim, action_dim, activation functions)
- Add network config schema (BRAIN_AS_CODE)
- Add fail-fast validation for network architecture params
- Update configs with explicit network architecture

**Example Config Update**:

```yaml
# Add to training.yaml
network:
  type: simple  # or 'recurrent'
  hidden_dim: 256
  activation: relu
  # Recurrent-specific (if type=recurrent)
  lstm_hidden_dim: 256
  vision_channels: [32, 64]
```

**Testing**:
- [ ] Linter passes for agent/ files
- [ ] Network configs load successfully
- [ ] Missing network param triggers clear error
- [ ] No regressions in training

**LOC**: 100+ lines changed

### Phase 4: Remaining Systems (TASK-004A, TASK-005) (5-7 days)

**Files** (13 files):
- `src/townlet/curriculum/adversarial.py` (2 files)
- `src/townlet/curriculum/static.py`
- `src/townlet/exploration/adaptive_intrinsic.py` (4 files)
- `src/townlet/exploration/epsilon_greedy.py`
- `src/townlet/exploration/rnd.py`
- `src/townlet/exploration/base.py`
- `src/townlet/population/vectorized.py` (2 files)
- `src/townlet/training/replay_buffer.py`
- `src/townlet/training/sequential_replay_buffer.py`
- `src/townlet/training/state.py` (partial)

**Changes**:
- Remove BAC/UAC defaults (epsilon, RND params, curriculum thresholds, replay buffer size)
- Add config schemas for curriculum, exploration, population
- Add fail-fast validation for all subsystems
- Update configs with complete curriculum/exploration/population params

**Testing**:
- [ ] Linter passes for all src/townlet/ files
- [ ] All configs load successfully
- [ ] Missing param in any subsystem triggers clear error
- [ ] No regressions in full test suite

**LOC**: 800+ lines changed

### Phase 5: Validation (1 day)

**Verification Steps**:

1. [ ] Deploy compliant whitelist (`.defaults-whitelist-compliant.txt`)
2. [ ] Run linter: `python scripts/no_defaults_lint.py src/townlet/ --whitelist .defaults-whitelist.txt`
3. [ ] Expected: 0 violations (all UAC/BAC defaults removed)
4. [ ] Load all example configs (L0, L0.5, L1, L2, L3)
5. [ ] Expected: All load successfully without errors
6. [ ] Run full test suite: `uv run pytest`
7. [ ] Expected: All tests pass, no regressions
8. [ ] Verify fail-fast: Remove required param from config, expect clear error
9. [ ] Update PDR-002 status: "Phase 4 (Whitelist Cleanup) Completed"

---

## Testing Strategy

**Test Requirements**:

- **Linter Validation**: `python scripts/no_defaults_lint.py src/townlet/` must pass (0 violations)
- **Config Loading**: All example configs must load without errors
- **Fail-Fast Validation**: Missing UAC/BAC param must trigger clear error with example
- **Regression Prevention**: Full test suite must pass (no existing tests broken)

**Coverage Target**: No change (refactoring existing code, not adding new features)

**Test-Driven Development**:

This is refactoring work, not greenfield development:
- Tests already exist (full test suite)
- Goal: Keep all tests passing while removing defaults
- Add new tests for fail-fast error messages

---

## Acceptance Criteria

**Must Have**:

- [ ] Compliant whitelist deployed (`.defaults-whitelist-compliant.txt`)
- [ ] Linter passes with 0 violations
- [ ] All UAC/BAC defaults removed (17 non-compliant whitelist entries eliminated)
- [ ] All example configs complete (L0, L0.5, L1, L2, L3)
- [ ] Missing UAC/BAC param triggers clear fail-fast error
- [ ] All tests pass (no regressions)
- [ ] Infrastructure defaults remain whitelisted (device, port, logging, metadata)

**Success Metrics**:

- Whitelist compliance: 100% (0 UAC/BAC exemptions)
- Linter violations: 0 (down from 226+)
- Example configs: 100% complete (all required params explicit)
- Config reproducibility: 100% (same config always produces same behavior)

---

## Risk Assessment

**Technical Risks**:

- ⚠️ **MEDIUM**: Updating 22 files risks introducing regressions
  - Mitigation: Phase-by-phase approach, test after each phase

- ⚠️ **MEDIUM**: Old configs may break if not updated
  - Mitigation: Update all example configs during refactoring, provide migration guide

- ✅ **LOW**: Fail-fast validation is straightforward pattern
  - Mitigation: Use consistent pattern across all files (see Phase 1 example)

**Migration**:

All example configs will be updated to be complete, so no migration needed for users who copy from examples. Custom configs will need updating:

**Migration Guide for Custom Configs**:

```bash
# Test your custom config
uv run python -m townlet.demo.runner --config my_configs/custom/

# If it fails with "Missing required parameter 'X'":
# 1. Read error message (shows parameter name and example value)
# 2. Add parameter to your config YAML
# 3. Retest until config loads successfully
```

---

## Future Enhancements (Out of Scope)

**Not Included**:

- TASK-003: Pydantic DTO-based validation (complements this work, separate task)
- TASK-004A: Universe compilation pipeline (validates cross-file dependencies)
- TASK-005: Full BRAIN_AS_CODE implementation (network config driven)
- Config inheritance/templates (reduce repetition across config packs)

**Rationale**: This task focuses on removing defaults and achieving PDR-002 compliance. Schema validation and config inheritance are separate enhancements.

---

## References

**Related Tasks**:

- PDR-002: No-Defaults Principle (policy this task implements)
- TASK-003: UAC Core DTOs (Pydantic validation for configs)
- TASK-004A: Compiler Implementation (cross-file validation)
- TASK-005: BRAIN_AS_CODE (network config driven)

**Investigation Documents**:

- `docs/investigations/PDR-002-WHITELIST-REVIEW.md` - Comprehensive analysis (990 lines)
- `docs/investigations/PDR-002-WHITELIST-COMPARISON.md` - Before/after comparison (450 lines)
- `docs/investigations/PDR-002-WHITELIST-REVIEW-SUMMARY.md` - Executive brief (330 lines)
- `docs/investigations/PDR-002-WHITELIST-QUICK-REFERENCE.md` - Developer guide (450 lines)
- `.defaults-whitelist-compliant.txt` - Ready-to-deploy compliant whitelist (146 lines)

**Code Files**:

- `.defaults-whitelist.txt` - Current non-compliant whitelist (19 entries, 85% violate PDR-002)
- `.defaults-whitelist-compliant.txt` - Compliant whitelist (17 entries, 0 UAC/BAC exemptions)
- `scripts/no_defaults_lint.py` - Linter that enforces no-defaults principle
- 22 files across `src/townlet/` requiring refactoring (see Phase breakdown)

**Documentation**:

- `docs/decisions/PDR-002-NO-DEFAULTS-PRINCIPLE.md` - Policy specification
- `docs/development/LINT_ENFORCEMENT.md` - Linter usage guide
- `CLAUDE.md` - Project guidance (mentions UNIVERSE_AS_CODE / BRAIN_AS_CODE)

---

**Estimated Timeline**: 12-17 days full-time (1500+ LOC, 22 files, 4 phases + validation)

**END OF TASK SPECIFICATION**
