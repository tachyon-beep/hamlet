# Test Refactoring Methodology: Subagent-Driven Development with Review Checkpoints

**Version**: 1.0
**Date**: 2025-11-04
**Status**: PROVEN (validated on 5 tasks, 201 tests, 100% success rate)
**Context**: Developed during HAMLET Townlet test suite refactoring (Nov 2024)

---

## Executive Summary

This document describes a **high-precision, high-value methodology** for refactoring large test suites using AI-assisted subagent-driven development. The approach prioritizes quality over speed through mandatory research phases, code review checkpoints, and "measure twice, cut once" principles.

**Key Results**:

- ✅ 5 tasks completed, 201 tests created, 100% passing
- ✅ Zero critical issues across all tasks
- ✅ Average 2-4 subagents per task (~10min per task)
- ✅ Consistent quality grades (PASS or better on first review)

---

## Methodology Overview

### Core Principle: Parallel Build + Clean Cutover

**Strategy**: Build a complete new test suite in parallel, then perform a single clean cutover.

```
┌─────────────────────────────────────────────────────────────┐
│ OLD TEST SUITE (keep running)                               │
│ - 67+ unstructured test files                               │
│ - Still passing, no modifications during refactoring        │
│ - Frozen during development (no moving targets)             │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Build in parallel
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ NEW TEST SUITE (under construction)                         │
│ - Organized unit/integration/e2e structure                  │
│ - Built using subagent-driven development                   │
│ - Review checkpoints after each file                        │
│ - Sign off when complete                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Single cutover
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PRODUCTION TEST SUITE                                       │
│ - Delete old files                                          │
│ - Activate new structure                                    │
│ - Update CI/CD                                              │
└─────────────────────────────────────────────────────────────┘
```

### Why This Approach?

**Advantages**:

1. **Zero risk**: Old tests keep running, no disruption during refactoring
2. **Clean validation**: Can review new architecture before committing
3. **Clean git history**: One consolidated commit instead of incremental changes
4. **Rollback safety**: Can abandon and start over if needed
5. **Development freeze**: Old codebase frozen, no moving targets

**When NOT to use**:

- Small test suites (<10 files) - direct refactoring is faster
- Active development - parallel builds conflict with ongoing feature work
- Time pressure - this method prioritizes quality over speed

---

## The Four-Phase Process

### Phase 0: Research (MANDATORY)

**Purpose**: Understand what you're consolidating before writing any code.

**Steps**:

1. **Read ALL old test files** being consolidated
2. **Document findings**:
   - What does each file test?
   - Are all tests in scope (unit vs integration vs API)?
   - Any mixed concerns (e.g., environment tests mixed with population tests)?
   - What are the key test patterns?
   - What are the critical behaviors to preserve?
3. **Create coverage map**: List of test categories and expected coverage
4. **Identify scope issues**: Note any tests that belong elsewhere

**Time investment**: 5-10% of task time (e.g., 2 minutes for 20-minute task)

**Why mandatory**:

- **Prevents rework**: Catching scope issues early saves hours
- **Better organization**: Understanding patterns leads to cleaner structure
- **Complete coverage**: Coverage map ensures nothing is missed

**Example findings** (from Task 1):

```
test_action_selection.py (559 lines)
- Scope: ⚠️ MIXED
- Contains:
  - Environment masking tests (in scope) ✅
  - Population Q-value masking tests (OUT of scope) ❌
- Decision: Extract environment tests, note population tests for Task 10
```

**Example findings** (from Task 3):

```
test_affordance_integration.py (332 lines)
- Scope: ⚠️ Integration tests, not unit tests
- Decision: Exclude from unit/environment/, keep for integration/ consolidation
```

---

### Phase 1: Implementation

**Purpose**: Build the new consolidated test file based on research findings.

**Steps**:

1. **Plan structure** based on research:

   ```python
   """Consolidated tests for [component].

   This file consolidates:
   - [Concern 1]
   - [Concern 2]
   - [Concern 3]

   Old files consolidated: [file1.py], [file2.py], [file3.py]
   """

   class Test[Concern1]:
       """Test [concern 1] behaviors."""

   class Test[Concern2]:
       """Test [concern 2] behaviors."""
   ```

2. **Use fixtures from conftest.py** (never duplicate):

   ```python
   def test_something(basic_env, cpu_device):
       # Use existing fixtures, don't create new environments
       assert basic_env.grid_size == 8
   ```

3. **Apply lessons learned**:
   - Use `cpu_device` fixture for deterministic tests
   - Behavioral assertions over exact values
   - Clear test names that describe expected behavior
   - Comprehensive docstrings

4. **Run tests to verify**:

   ```bash
   pytest tests/path/to/new_file.py -v
   ```

5. **Report back**:
   - Research findings summary
   - Implementation structure (test classes, test count)
   - Test results (X/Y passing)
   - Coverage assessment
   - Any issues encountered

---

### Phase 2: Review

**Purpose**: Catch issues before proceeding to next task.

**Review Checklist**:

1. **Coverage verification**:
   - [ ] Are all critical tests from old files preserved?
   - [ ] What % of old functionality is covered?
   - [ ] Any significant gaps?

2. **Scope boundary verification**:
   - [ ] Are all tests at the correct abstraction level?
   - [ ] Any integration tests in unit tests?
   - [ ] Any API tests in environment tests?

3. **Quality verification**:
   - [ ] Is test organization logical and clear?
   - [ ] Are fixtures used correctly (no duplication)?
   - [ ] Are assertions behavioral (not brittle)?
   - [ ] Is documentation adequate?

4. **Technical verification**:
   - [ ] Do all tests pass?
   - [ ] Are deterministic tests using `cpu_device`?
   - [ ] Are there any flaky tests (run 5× to check)?

5. **Issue classification**:
   - **Critical**: Test failures, missing critical coverage, incorrect behavior
   - **Important**: Poor organization, missing edge cases, scope violations
   - **Minor**: Documentation improvements, naming suggestions

**Review outcome**:

- **PASS**: Ready to proceed to next task
- **NEEDS FIXES**: Fix issues before proceeding (go to Phase 3)
- **REJECT**: Start over with lessons learned (rare, only if fundamentally flawed)

---

### Phase 3: Fix (if needed)

**Purpose**: Address issues found during review.

**When to fix**:

- **Critical issues**: ALWAYS fix before proceeding
- **Important issues**: Fix if time permits, or document for Phase 2
- **Minor issues**: Document for Phase 2, not blockers

**Fix process**:

1. Dispatch fix subagent with specific instructions
2. Apply fixes
3. Re-run tests
4. Dispatch final reviewer to validate fixes

**Example** (from Task 2):

```
Issue: Flaky tests due to random agent spawning + CUDA non-determinism
Fix: Use cpu_device + force agent position to center of grid
Validation: Run tests 5× to confirm no more flakes
Result: 100% pass rate across 5 runs
```

---

### Phase 4: Mark Complete

**Purpose**: Document results and proceed.

**Steps**:

1. **Update task tracking** (todos, status board)
2. **Document results**:
   - Test count (new vs old)
   - Coverage % (what was preserved)
   - Quality grade (PASS, A+, etc.)
   - Issues found and resolved
3. **Proceed to next task**

---

## Key Principles

### 1. Research First, Always

**Rule**: Never write code before reading all old test files.

**Why**:

- Prevents scope violations
- Identifies mixed concerns early
- Leads to better organization
- Ensures complete coverage

**Evidence**:

- Task 1 (no research): Scope violations found during review
- Tasks 2-5 (with research): Clean scope boundaries, zero violations

---

### 2. CPU Fixtures for Determinism

**Rule**: Unit tests must use `cpu_device` fixture, not default device.

**Problem**: CUDA introduces non-determinism through:

- Random agent spawn positions
- Floating-point operation ordering
- Kernel execution timing

**Solution**:

```python
def test_movement_updates_position(cpu_device, test_config_pack_path):
    """Test that movement changes agent position."""
    env = VectorizedHamletEnv(
        num_agents=1,
        device=cpu_device,  # ← Force CPU for determinism
        ...
    )
    env.reset()
    env.positions[0] = torch.tensor([4, 4], device=cpu_device)  # ← Control position

    actions = torch.tensor([0], device=cpu_device)  # ← UP action
    env.step(actions)

    assert env.positions[0, 0] == 3  # ← Deterministic result
```

**When to use GPU**:

- Integration tests (testing GPU performance)
- Performance benchmarks
- E2E tests (production environment)

---

### 3. Behavioral Assertions > Exact Values

**Rule**: Test behavior, not implementation details.

**Good patterns**:

```python
# Good: Behavioral assertion
assert meters[0] > initial_value, "Energy should increase after rest"

# Good: Tolerance for known values
assert abs(result - expected) < 0.01, "Within acceptable tolerance"

# Good: Using mock config for exact values
from fixtures.mock_config import EXPECTED_ENERGY_RESTORATION
assert result == pytest.approx(EXPECTED_ENERGY_RESTORATION, rel=1e-6)
```

**Anti-patterns**:

```python
# Bad: Brittle exact value
assert meters[0] == 0.1875  # Breaks when config changes

# Bad: Testing implementation
assert len(env._grid_cache) == 64  # Internal implementation detail

# Bad: Magic numbers
assert obs.shape[1] == 89  # What is 89? Where did it come from?
```

**Exception**: Mock config values can be exact when documented:

```python
# From mock_config.yaml (frozen 2025-11-04)
EXPECTED_BED_RESTORATION = 0.50  # +50% energy
assert result == pytest.approx(EXPECTED_BED_RESTORATION, rel=1e-6)
```

---

### 4. Proper Scope Boundaries

**Rule**: Tests must be at the correct abstraction level.

**Scope hierarchy**:

```
Unit Tests (unit/)
├─ Environment: Test env mechanics via env.step()
├─ Agent: Test network forward pass, Q-value computation
├─ Population: Test training logic, action selection
├─ Curriculum: Test stage progression logic
├─ Exploration: Test exploration strategies
└─ Training: Test replay buffer, checkpointing

Integration Tests (integration/)
├─ Component interactions: Test A → B communication
├─ Multi-component flows: Test A → B → C sequences
└─ Subsystem integration: Test training loop without full system

E2E Tests (e2e/)
├─ Complete training runs: Full system from start to finish
├─ Inference server: Live visualization with checkpoints
└─ Production scenarios: Realistic end-to-end workflows
```

**Scope violations** (examples from our refactoring):

| Test | Old Location | Issue | Correct Location |
|------|-------------|-------|------------------|
| Q-value masking | `test_action_selection.py` | Population logic in env tests | `unit/population/test_action_selection.py` |
| Engine initialization | `test_affordance_engine.py` | API tests in env tests | `unit/environment/test_affordance_engine_api.py` |
| End-to-end training | `test_affordance_integration.py` | Integration tests in unit tests | `integration/test_affordance_integration.py` |

**How to identify scope violations**:

1. **Read the test**: What is it actually testing?
2. **Check the abstraction**: Is it testing component A or component A→B interaction?
3. **Ask**: "Could this test pass if component B is broken?"
   - If YES → unit test
   - If NO → integration test

---

### 5. Fixture Usage from conftest.py

**Rule**: Always check conftest.py before creating fixtures.

**Available fixtures** (example from HAMLET):

```python
# Configuration
mock_config_path, test_config_pack_path, temp_config_pack

# Devices
device (CUDA if available), cpu_device (force CPU)

# Environments
basic_env (1 agent, full obs), pomdp_env (POMDP), temporal_env (temporal mechanics)

# Networks
simple_qnetwork, recurrent_qnetwork

# Training components
replay_buffer, adversarial_curriculum, epsilon_greedy_exploration

# Utilities
sample_observations, sample_actions
```

**When to add a fixture**:

- Used by 2+ test files
- Complex setup (>5 lines)
- Shared configuration

**Never**:

- Duplicate fixtures across files
- Create one-off fixtures in test files
- Hardcode paths or values that could be fixtures

---

## Common Pitfalls & Solutions

### Pitfall 1: CUDA Non-Determinism

**Symptom**: Tests pass ~80% of the time, fail intermittently

**Cause**: GPU introduces randomness in:

- Agent spawn positions
- Floating-point operation ordering
- Kernel execution timing

**Solution**:

```python
def test_something(cpu_device):  # ← Force CPU
    env = VectorizedHamletEnv(..., device=cpu_device)
    env.reset()
    env.positions[0] = torch.tensor([4, 4], device=cpu_device)  # ← Control position
```

**Example**: Task 2 had flaky tests that failed when agents spawned at grid edges where movement was blocked. Fixed by forcing CPU device and centering agents.

---

### Pitfall 2: Random Agent Spawning

**Symptom**: Movement tests fail when agent spawns at grid edge

**Cause**: `env.reset()` places agents randomly. If agent spawns at top edge, UP action is masked.

**Solution**:

```python
env.reset()
# Force agent to center where all movements are valid
env.positions[0] = torch.tensor([4, 4], device=cpu_device)

actions = torch.tensor([0], device=cpu_device)  # UP - guaranteed valid from center
obs, _, _, _ = env.step(actions)

assert env.positions[0, 0] == 3, "Agent should have moved UP"
```

---

### Pitfall 3: Mixed Scope Concerns

**Symptom**: Reviewer finds tests that belong in different directories

**Cause**: Legacy tests had environment + population + integration mixed together

**Solution**: Research phase identifies these early

```
Research findings:
- test_action_selection.py (559 lines)
  - Lines 1-200: Environment masking (✅ in scope)
  - Lines 201-559: Population Q-value masking (❌ out of scope)

Decision: Extract environment tests, note population tests for Task 10
```

---

### Pitfall 4: Exact Value Assertions

**Symptom**: Tests break when tuning hyperparameters or config values

**Cause**: Hardcoded exact values like `assert result == 0.1875`

**Solution**: Use behavioral assertions or mock config

```python
# Good: Behavioral
assert result > initial_value, "Should increase"

# Good: Tolerance
assert abs(result - expected) < 0.01

# Good: Mock config (documented exact value)
from fixtures.mock_config import EXPECTED_VALUE
assert result == pytest.approx(EXPECTED_VALUE, rel=1e-6)
```

---

### Pitfall 5: Fixture Duplication

**Symptom**: Same environment setup repeated across multiple tests

**Cause**: Not checking conftest.py before creating test-specific setup

**Solution**: Extract to conftest.py

```python
# Before (duplicated in 5 test files)
def test_something():
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=torch.device("cpu"),
        ...
    )
    # ... test logic

# After (conftest.py)
@pytest.fixture
def basic_env(cpu_device, test_config_pack_path):
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=cpu_device,
        config_pack_path=test_config_pack_path,
    )

# In test file
def test_something(basic_env):
    # ... test logic
```

---

## Subagent Workflow

### Roles and Responsibilities

**Dev Subagent**:

- Conducts research phase (reads old files)
- Implements new consolidated file
- Runs tests to verify
- Reports findings and results

**Reviewer Subagent**:

- Audits implementation against requirements
- Verifies coverage and quality
- Classifies issues (Critical/Important/Minor)
- Recommends PASS / NEEDS FIXES / REJECT

**Fix Subagent** (if needed):

- Addresses specific issues from review
- Re-runs tests to validate fixes
- Reports what was changed

**Final Reviewer Subagent** (if fixes applied):

- Validates that fixes resolved issues
- Confirms no new issues introduced
- Gives final PASS verdict

---

### Typical Task Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 0: RESEARCH                                                │
├──────────────────────────────────────────────────────────────────┤
│ 1. Dispatch dev subagent                                         │
│ 2. Subagent reads ALL old test files                            │
│ 3. Subagent documents:                                           │
│    - What each file tests                                        │
│    - Scope boundaries (unit/integration/API)                     │
│    - Mixed concerns                                              │
│    - Coverage map                                                │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 1: IMPLEMENTATION                                          │
├──────────────────────────────────────────────────────────────────┤
│ 1. Subagent plans structure based on research                   │
│ 2. Subagent builds consolidated file                            │
│ 3. Subagent runs tests (pytest -v)                              │
│ 4. Subagent reports back:                                        │
│    - Research findings                                           │
│    - Test structure (classes, count)                             │
│    - Test results (X/Y passing)                                  │
│    - Coverage assessment                                         │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 2: REVIEW                                                  │
├──────────────────────────────────────────────────────────────────┤
│ 1. Dispatch reviewer subagent                                    │
│ 2. Reviewer reads new file                                       │
│ 3. Reviewer compares against old files                           │
│ 4. Reviewer verifies:                                            │
│    - Coverage preserved                                          │
│    - Scope boundaries correct                                    │
│    - Quality standards met                                       │
│ 5. Reviewer reports: PASS / NEEDS FIXES / REJECT                │
└──────────────────────────────────────────────────────────────────┘
                            ↓
              ┌─────────────────────────┐
              │  Issues found?          │
              └─────────────────────────┘
                     ↙        ↘
              YES (NEEDS FIXES)    NO (PASS)
                ↓                      ↓
┌────────────────────────────┐  ┌──────────────────────────────┐
│ PHASE 3: FIX               │  │ PHASE 4: MARK COMPLETE       │
├────────────────────────────┤  ├──────────────────────────────┤
│ 1. Dispatch fix subagent   │  │ 1. Update task tracking      │
│ 2. Apply fixes             │  │ 2. Document results          │
│ 3. Re-run tests            │  │ 3. Proceed to next task      │
│ 4. Dispatch final reviewer │  └──────────────────────────────┘
│ 5. Validate fixes          │
│ 6. Give final PASS         │
└────────────────────────────┘
                ↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 4: MARK COMPLETE                                           │
├──────────────────────────────────────────────────────────────────┤
│ 1. Update task tracking                                          │
│ 2. Document results                                              │
│ 3. Proceed to next task                                          │
└──────────────────────────────────────────────────────────────────┘
```

**Typical subagent count**:

- No issues: 2 subagents (dev + reviewer)
- With fixes: 4 subagents (dev + reviewer + fix + final reviewer)

**Average time**: ~10 minutes per task (5 tasks in 50 minutes)

---

## Quality Metrics

### Success Indicators

**Per-Task Metrics**:

- ✅ All tests pass (100% pass rate)
- ✅ Coverage preserved (≥85% of old functionality)
- ✅ Zero critical issues
- ✅ ≤2 important issues (non-blocking)
- ✅ Clear documentation (traceability to old tests)

**Overall Metrics** (from HAMLET refactoring):

- ✅ 5/5 tasks completed successfully
- ✅ 201 tests created (all passing)
- ✅ 0 critical issues across all tasks
- ✅ Average 2-4 subagents per task
- ✅ Consistent quality (PASS or better on first review)

### Failure Modes

**Critical failure** (restart task):

- Tests don't pass
- Major coverage gaps (>20% missing)
- Scope violations not caught in research

**Important failure** (fix before proceeding):

- Missing edge cases
- Poor organization
- Fixture duplication

**Minor failure** (document for later):

- Documentation improvements
- Naming suggestions
- Non-blocking enhancements

---

## Templates

### Dev Subagent Prompt Template

```markdown
You are implementing **Task N: Build [test_file.py]**.

# Your Mission

Build a **NEW** comprehensive test file that consolidates [X] from multiple old files.
This is a CLEAN BUILD - do NOT modify old files, build the new one from scratch.

# MANDATORY: Research Phase First!

**STEP 1**: Before writing ANY code, read all [N] old files and document:
- What does each file test?
- Are all tests [scope]-level (e.g., environment-level)?
- Any mixed concerns or integration tests?
- What are the key test patterns?
- What are the critical behaviors to preserve?

Document your findings before proceeding.

# Context

Location: `/path/to/tests/`

Old files being consolidated (READ FIRST):
- `old_file_1.py` ([lines] lines)
- `old_file_2.py` ([lines] lines)
- ...

New file location: `/path/to/new/test_file.py`

# Available Fixtures

From `conftest.py`:
- `fixture_1`: [description]
- `fixture_2`: [description]
- `cpu_device`: Force CPU for deterministic tests

# Expected Structure (adjust after research)

```python
"""Consolidated tests for [component].

This file consolidates:
- [Concern 1]
- [Concern 2]
- [Concern 3]

Old files consolidated: [file1.py], [file2.py], [file3.py]
"""

class Test[Concern1]:
    """Test [concern 1] behaviors."""

class Test[Concern2]:
    """Test [concern 2] behaviors."""
```

# Key Lessons from Previous Tasks

✅ **Research first**: Read all old files, understand scope, plan structure
✅ **Use CPU for deterministic tests**: Avoid CUDA randomness
✅ **Behavioral assertions**: Focus on "what should happen" not exact values
✅ **Proper fixture usage**: Leverage conftest.py, no duplication
✅ **Clear documentation**: Explain what old tests are consolidated
✅ **Check for scope violations**: Don't include integration/API tests

# Your Tasks

1. **RESEARCH PHASE**:
   - Read all [N] old files thoroughly
   - Document key behaviors tested
   - Identify any scope issues
   - Note critical test patterns

2. **IMPLEMENTATION PHASE**:
   - Build new consolidated file based on research
   - Use fixtures from conftest.py
   - Apply lessons learned

3. **VERIFICATION PHASE**:
   - Run pytest:

     ```bash
     pytest path/to/new_file.py -v
     ```

4. **REPORT BACK**:
   - **Research findings**: What you learned
   - **Implementation summary**: Number of tests, structure
   - **Test results**: X/Y passing
   - **Coverage**: What old tests consolidated
   - **Any issues**: Scope problems? Missing tests?

# DO NOT

- Modify old test files
- Create new fixtures (use conftest.py)
- Add TODOs or incomplete tests
- Skip the research phase!

Work from: `/path/to/repo/`

**BEGIN WITH RESEARCH PHASE**

```

---

### Reviewer Subagent Prompt Template

```markdown
You are reviewing Task N implementation: **[test_file.py]**

# Context

A subagent just built a new consolidated test file to replace [N] legacy test files.

## What Was Implemented

Location: `/path/to/new/test_file.py`

The subagent created a consolidated test file with:
- **X tests across Y test classes**
- TestClass1 (N1 tests)
- TestClass2 (N2 tests)
- ...

**All X tests pass.**

## Research Phase Findings (from subagent)

[Include subagent's research findings summary]

## Requirements (from Plan)

**Goal**: [Goal statement]

**Must have**:
1. [Requirement 1]
2. [Requirement 2]
3. [Requirement 3]

## Lessons from Previous Tasks

- ✅ Research phase before implementation
- ✅ CPU fixtures for deterministic tests
- ✅ Behavioral assertions
- ✅ Proper fixture usage
- ✅ Clear documentation

## Your Review Mission

Working directory: `/path/to/repo/`

**Audit the implementation:**

1. **Read the new file**: `/path/to/new/test_file.py`

2. **Compare against old files**:
   - `/path/to/old_file_1.py`
   - `/path/to/old_file_2.py`

3. **Verify**:
   - ✅ Are critical test cases from old files preserved?
   - ✅ Is test organization logical and clear?
   - ✅ Are fixtures used correctly?
   - ✅ Are assertions behavioral (not brittle)?
   - ✅ Is documentation adequate?
   - ✅ Are there any gaps in coverage?

4. **Run tests**:
   ```bash
   pytest path/to/new_file.py -v
   ```

5. **Check for issues**:
   - **Critical**: Test failures, missing critical coverage, incorrect behavior
   - **Important**: Poor organization, unclear tests, missing edge cases
   - **Minor**: Documentation improvements, naming suggestions

## Report Format

### Strengths

- What the implementation does well

### Issues Found

**Critical** (must fix):

- [List critical issues]

**Important** (should fix):

- [List important issues]

**Minor** (nice to have):

- [List minor improvements]

### Coverage Assessment

- What % of old test functionality is preserved?
- Any significant gaps?

### Overall Assessment

- **PASS**: Ready to proceed to next task
- **NEEDS FIXES**: Fix issues before proceeding
- **REJECT**: Start over with lessons learned

### Recommendations

- Specific changes needed (if any)

**BEGIN REVIEW**

```

---

## Appendix: Case Studies

### Case Study 1: Task 2 - Flaky Tests

**Problem**: Tests failed intermittently (~20% failure rate)

**Root cause analysis**:
1. Random agent spawn positions (agents could spawn at grid edges)
2. CUDA non-determinism in floating-point operations
3. Tests assumed agent moved, but movement was blocked at boundaries

**Solution applied**:
```python
# Before (flaky)
def test_movement_updates_grid(basic_env):
    obs1 = basic_env.reset()  # Random spawn
    obs2, _, _, _ = basic_env.step(torch.tensor([0]))  # UP
    assert not torch.equal(obs1, obs2)  # FAILS if agent at top edge

# After (deterministic)
def test_movement_updates_grid(cpu_device, test_config_pack_path):
    env = VectorizedHamletEnv(..., device=cpu_device)
    env.reset()
    env.positions[0] = torch.tensor([4, 4], device=cpu_device)  # Center
    obs1 = env._get_observations()
    env.step(torch.tensor([0], device=cpu_device))  # UP from center
    obs2 = env._get_observations()
    assert not torch.equal(obs1, obs2)  # Always passes
```

**Validation**: Ran tests 5× consecutively, 100% pass rate

**Lesson**: Always use `cpu_device` + controlled positions for movement tests

---

### Case Study 2: Task 1 - Scope Violations

**Problem**: Reviewer found population-level tests in environment test file

**Root cause**: No research phase, blindly consolidated all tests from `test_action_selection.py`

**Discovery**:

```
test_action_selection.py (559 lines):
- Lines 1-200: Environment action masking (✅ in scope)
- Lines 201-559: Population Q-value masking (❌ out of scope)
```

**Resolution**:

- Kept environment tests in Task 1
- Noted population tests for Task 10 (unit/population/test_action_selection.py)

**Lesson**: Research phase is mandatory to catch mixed concerns

---

### Case Study 3: Task 4 - Enhanced Coverage

**Problem**: How to improve coverage beyond just consolidating old tests?

**Approach**: Identify gaps during research, add tests to fill them

**Added tests** (6 new tests not in old files):

1. `test_no_negative_values_after_depletion` - Multi-cycle clamping
2. `test_no_overflow_above_one` - Upper bound clamping
3. `test_cascade_engine_respects_bounds` - Engine-level clamping
4. `test_agents_have_independent_meters` - Multi-agent isolation
5. `test_full_cascade_via_engine` - Engine integration
6. `test_equivalence_multi_agent_batch` - Batch equivalence

**Result**: 114% coverage (41 tests vs 36 in old files)

**Lesson**: Research phase reveals gaps that can be filled proactively

---

## Glossary

**Subagent**: An AI assistant dispatched to complete a specific task (dev, reviewer, fix)

**Research phase**: Mandatory step where dev subagent reads all old files before implementation

**Scope boundary**: The abstraction level a test operates at (unit/integration/e2e)

**Behavioral assertion**: Test that verifies observable behavior, not implementation

**CPU fixture**: pytest fixture that forces CPU device for deterministic tests

**Mock config**: Frozen configuration file for tests requiring exact values

**Parallel build**: Building new test suite without modifying old tests

**Clean cutover**: Single transition from old test suite to new test suite

**Coverage preservation**: % of old test functionality retained in new tests

---

## References

- HAMLET test refactoring (Nov 2024): 5 tasks, 201 tests, 100% success
- Subagent-driven development: See `/home/john/.claude/plugins/cache/superpowers/skills/subagent-driven-development/`
- Test-driven development: See `/home/john/.claude/plugins/cache/superpowers/skills/test-driven-development/`
- Code review process: See `/home/john/.claude/plugins/cache/superpowers/skills/requesting-code-review/`

---

**END OF METHODOLOGY**
