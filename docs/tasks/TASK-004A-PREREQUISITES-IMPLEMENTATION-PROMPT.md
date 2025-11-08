# IMPLEMENTATION PROMPT: TASK-004A-PREREQUISITES

**Copy this entire prompt to Claude tomorrow morning to begin implementation.**

---

## Role & Methodology

You are an expert Python programmer, debugger, and test-driven development (TDD) practitioner with deep expertise in:

- **Clean Architecture**: Separation of concerns, dependency inversion, single responsibility
- **Test-Driven Development**: RED → GREEN → REFACTOR cycle, comprehensive test coverage
- **Python Best Practices**: Type hints, dataclasses, Pydantic models, pathlib, proper error handling
- **Debugging**: Systematic root cause analysis, hypothesis-driven investigation, minimal reproduction
- **Git Workflow**: Atomic commits, clear messages, incremental progress tracking

## Your Mission

Implement **TASK-004A-PREREQUISITES** using strict TDD methodology to unblock TASK-004A Universe Compiler implementation.

**Critical Context**: The TASK-004A implementation plan contains 5 critical blockers where assumptions don't match repository reality. You must resolve all 5 blockers before TASK-004A can proceed.

## Required Reading (Read These First)

**Primary Reference** (your implementation spec):
- `/home/user/hamlet/docs/tasks/TASK-004A-PREREQUISITES-TASKING.md` (1,030 lines - READ IN FULL)

**Supporting Documentation** (context and validation):
- `/home/user/hamlet/docs/tasks/TASK-004A-PREREQUISITES.md` (blockers explanation)
- `/home/user/hamlet/docs/tasks/TASK-004A-COMPILER-IMPLEMENTATION.md` (what this unblocks)
- `/home/user/hamlet/docs/plans/2025-11-08-task-004a-tdd-implementation-plan.md` (TDD approach)
- `/home/user/hamlet/docs/architecture/COMPILER_ARCHITECTURE.md` (architectural context)

**Action**: Read the tasking statement fully before writing any code. Understand all 5 parts and how they interrelate.

## TDD Workflow (Mandatory)

For each function/class, follow this exact cycle:

### Phase 1: RED (Write Failing Test)

1. **Create test file** (if not exists)
2. **Write ONE specific test case** for ONE behavior
3. **Run test** - verify it FAILS for the right reason
4. **Commit** (optional): `test: Add failing test for <behavior>`

Example:
```python
def test_load_variables_reference_config_returns_list():
    """Should return list of VariableDef instances."""
    result = load_variables_reference_config(Path("configs/L0_0_minimal"))
    assert isinstance(result, list)
    assert all(isinstance(v, VariableDef) for v in result)
    # This should FAIL because function doesn't exist yet
```

### Phase 2: GREEN (Minimal Implementation)

1. **Write MINIMAL code** to make test pass (not perfect, just passing)
2. **Run test** - verify it PASSES
3. **Run all related tests** - verify no regressions
4. **Commit**: `feat: Implement <function/class> (passing <test_name>)`

Example:
```python
def load_variables_reference_config(config_dir: Path) -> list[VariableDef]:
    # Minimal implementation - just make test pass
    yaml_path = config_dir / "variables_reference.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return [VariableDef(**var) for var in data["variables"]]
```

### Phase 3: REFACTOR (Improve Without Breaking)

1. **Improve code quality** (error handling, type hints, documentation)
2. **Run all tests** - verify still passing
3. **Commit**: `refactor: Improve <function/class> with <improvement>`

Example:
```python
def load_variables_reference_config(config_dir: Path) -> list[VariableDef]:
    """Load VFS variables from variables_reference.yaml.

    Args:
        config_dir: Config pack directory (e.g., configs/L1_full_observability)

    Returns:
        List of VariableDef instances

    Raises:
        FileNotFoundError: If variables_reference.yaml not found
        ValidationError: If YAML schema invalid
    """
    yaml_path = config_dir / "variables_reference.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"variables_reference.yaml not found in {config_dir}"
        )

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {yaml_path}: {e}")

    if "variables" not in data:
        raise ValueError(f"Missing 'variables' section in {yaml_path}")

    try:
        return [VariableDef(**var) for var in data["variables"]]
    except ValidationError as e:
        raise ValueError(f"Invalid variable schema in {yaml_path}: {e}")
```

### Phase 4: VERIFY (Run Full Test Suite)

```bash
# Run all related tests
uv run pytest tests/test_townlet/unit/vfs/test_load_variables_reference.py -v

# Run broader test suite to check for regressions
uv run pytest tests/test_townlet/unit/vfs/ -v

# If all pass, continue to next test case
```

**Repeat for each test case until feature complete.**

## Implementation Plan (5 Parts Sequential)

### Part 1: Config Schema Alignment (3-4 hours)

**Goal**: Create `load_variables_reference_config()` and `load_global_actions_config()`.

**TDD Sequence**:

1. **Test 1**: `test_load_variables_reference_config_returns_list`
   - RED: Write test
   - GREEN: Minimal implementation
   - REFACTOR: Add error handling
   - COMMIT: `feat(vfs): Add load_variables_reference_config()`

2. **Test 2**: `test_load_variables_reference_missing_file_raises`
   - RED: Test FileNotFoundError
   - GREEN: Add file existence check
   - COMMIT: `test(vfs): Add missing file error test`

3. **Test 3**: `test_load_variables_reference_invalid_schema_raises`
   - RED: Test ValidationError
   - GREEN: Add schema validation
   - COMMIT: `test(vfs): Add schema validation test`

4. **Test 4**: `test_load_variables_reference_preserves_order`
   - RED: Test order preservation
   - GREEN: Ensure list order matches YAML
   - COMMIT: `test(vfs): Add order preservation test`

5. **Tests 5-8**: Repeat TDD cycle for `load_global_actions_config()`
   - Same pattern: returns list, missing file, invalid schema, composition

6. **Tests 9-10**: `ActionSpaceConfig` class tests
   - `test_action_space_config_get_by_id`
   - `test_action_space_config_get_by_name`

**Commit After Part 1**:
```bash
git add src/townlet/vfs/schema.py \
        src/townlet/environment/action_space_builder.py \
        tests/test_townlet/unit/vfs/test_load_variables_reference.py \
        tests/test_townlet/unit/environment/test_load_global_actions.py

git commit -m "feat(task-004a-prereq): Part 1 - Config schema alignment

- Add load_variables_reference_config() to vfs/schema.py
- Add load_global_actions_config() to action_space_builder.py
- Create ActionSpaceConfig wrapper class
- Add 10 unit tests (all passing)
- Resolves Blocker #1 (config filename mismatch)

TDD: All tests written first, all passing"
```

### Part 2: DTO Consolidation (2-3 hours)

**Goal**: Add type aliases for `BarsConfig` and export `ActionSpaceConfig`.

**TDD Sequence**:

1. **Test 1**: `test_bars_config_importable_from_config_bar`
   - RED: Test import from `townlet.config.bar`
   - GREEN: Add type alias
   - COMMIT: `feat(config): Add BarsConfig type alias`

2. **Test 2**: `test_bars_config_is_same_class`
   - RED: Test alias points to original
   - GREEN: Verify alias correctness
   - COMMIT: `test(config): Verify BarsConfig alias correctness`

**Commit After Part 2**:
```bash
git add src/townlet/config/bar.py \
        tests/test_townlet/unit/config/test_bars_config_alias.py

git commit -m "feat(task-004a-prereq): Part 2 - DTO consolidation

- Add BarsConfig type alias to townlet.config.bar
- Export ActionSpaceConfig from action_space_builder
- Add 2 unit tests (all passing)
- Resolves Blocker #2 (DTO package mismatch)

TDD: All tests written first, all passing"
```

### Part 3: ObservationSpec Adapter (2-3 hours)

**Goal**: Create adapter to convert VFS ObservationField → Universe ObservationSpec.

**Critical**: This is the most complex part. Use TDD rigorously.

**TDD Sequence**:

1. **Test 1**: `test_flatten_1d_shape`
   - RED: Test `shape=(8,) → dims=8`
   - GREEN: Implement shape flattening helper
   - COMMIT: `feat(universe): Add shape flattening`

2. **Test 2**: `test_flatten_2d_shape`
   - RED: Test `shape=(5,5) → dims=25`
   - GREEN: Extend flattening for 2D
   - COMMIT: `test(universe): Add 2D shape flattening`

3. **Test 3**: `test_compute_contiguous_indices`
   - RED: Test start_index/end_index calculation
   - GREEN: Implement index accumulation
   - COMMIT: `feat(universe): Add index computation`

4. **Test 4-6**: Semantic type inference
   - RED: Test position inference
   - RED: Test meter inference
   - RED: Test affordance inference
   - GREEN: Implement `_infer_semantic_type()`
   - COMMIT: `feat(universe): Add semantic type inference`

5. **Test 7-8**: Field type inference
   - RED: Test scalar type
   - RED: Test spatial_grid type
   - GREEN: Implement `_infer_field_type()`
   - COMMIT: `feat(universe): Add field type inference`

6. **Test 9**: `test_adapter_preserves_field_order`
   - RED: Test field order preservation
   - GREEN: Ensure adapter maintains order
   - COMMIT: `test(universe): Verify field order preservation`

7. **Test 10**: `test_adapter_total_dims_correct`
   - RED: Test total_dims calculation
   - GREEN: Verify sum matches
   - COMMIT: `test(universe): Verify total dims calculation`

**Commit After Part 3**:
```bash
git add src/townlet/universe/adapters/vfs_adapter.py \
        src/townlet/universe/dto/observation_spec.py \
        tests/test_townlet/unit/universe/adapters/test_vfs_adapter.py

git commit -m "feat(task-004a-prereq): Part 3 - ObservationSpec adapter

- Create vfs_to_universe_observation_spec() adapter
- Create Universe ObservationSpec DTO
- Implement shape flattening and index computation
- Add semantic type inference
- Add field type inference
- Add 10 unit tests (all passing)
- Resolves Blocker #3 (ObservationSpec incompatibility)

TDD: All tests written first, all passing"
```

### Part 4: HamletConfig Integration (1-2 hours)

**Goal**: Document compiler builds on HamletConfig strategy.

**No TDD** (documentation only), but still verify:

1. **Create documentation**: `docs/architecture/COMPILER-HAMLET-INTEGRATION.md`
2. **Include**:
   - Architecture decision
   - Rationale
   - Data flow diagram (ASCII art or Mermaid)
   - Integration points
   - RawConfigs structure
   - What compiler adds beyond HamletConfig
3. **Update**: `docs/tasks/TASK-004A-PREREQUISITES.md` Part 4 status

**Commit After Part 4**:
```bash
git add docs/architecture/COMPILER-HAMLET-INTEGRATION.md \
        docs/tasks/TASK-004A-PREREQUISITES.md

git commit -m "docs(task-004a-prereq): Part 4 - HamletConfig integration

- Document compiler builds on HamletConfig strategy
- Add data flow diagram
- Specify RawConfigs structure
- Clarify separation of concerns
- Resolves Blocker #5 (HamletConfig duplication)"
```

### Part 5: Update TASK-004A Spec (1-2 hours)

**Goal**: Verify TASK-004A plan has correct imports and assumptions.

**Verification Checklist**:

1. **Scan TASK-004A plan**: Look for old function names
   - Search for: `load_variables_config` (should be `load_variables_reference_config`)
   - Search for: `variables.yaml` (should be `variables_reference.yaml`)
   - Search for: `raw_configs.variables` (should be `raw_configs.variables_reference`)

2. **Scan TDD plan**: Same search patterns

3. **Update if found**: Use Edit tool to correct

4. **Verify imports**: Check all import statements match Part 1-3 implementations

**Commit After Part 5**:
```bash
git add docs/tasks/TASK-004A-COMPILER-IMPLEMENTATION.md \
        docs/plans/2025-11-08-task-004a-tdd-implementation-plan.md

git commit -m "docs(task-004a-prereq): Part 5 - Update TASK-004A spec

- Verify all imports corrected in TASK-004A plan
- Update TDD plan with correct function names
- Ensure no references to non-existent functions
- Resolves Blocker #4 (missing load functions)"
```

## Integration Testing (After All 5 Parts)

**Create integration test**: `tests/test_townlet/integration/test_prerequisites_integration.py`

**Test specification** (from tasking statement):

```python
def test_prerequisites_end_to_end():
    """Verify complete data flow: HamletConfig → VFS → Adapter → Universe DTO."""
    from pathlib import Path
    from townlet.config.hamlet import HamletConfig
    from townlet.vfs.schema import load_variables_reference_config
    from townlet.environment.action_space_builder import load_global_actions_config
    from townlet.vfs.observation_builder import VFSObservationSpecBuilder
    from townlet.vfs.registry import VariableRegistry
    from townlet.universe.adapters.vfs_adapter import vfs_to_universe_observation_spec

    config_dir = Path("configs/L0_0_minimal")

    # Part 1: Load functions work
    hamlet_config = HamletConfig.load(config_dir)
    variables_reference = load_variables_reference_config(config_dir)
    global_actions = load_global_actions_config()

    assert hamlet_config is not None
    assert len(variables_reference) > 0
    assert len(global_actions.actions) > 0

    # Part 2: Type alias works
    from townlet.config.bar import BarsConfig
    assert isinstance(hamlet_config.bars, BarsConfig)

    # Part 3: VFS → Universe adapter works
    temp_registry = VariableRegistry()
    for var_def in variables_reference:
        temp_registry.register(var_def)

    builder = VFSObservationSpecBuilder(
        variable_registry=temp_registry,
        substrate=hamlet_config.substrate
    )
    vfs_fields = builder.build_observation_spec()  # Adjust method name if needed

    universe_obs_spec = vfs_to_universe_observation_spec(vfs_fields)

    assert universe_obs_spec.total_dims > 0
    assert len(universe_obs_spec.fields) > 0

    print(f"✅ Prerequisites integration test passed!")
    print(f"   Observation dims: {universe_obs_spec.total_dims}")
    print(f"   Actions: {len(global_actions.actions)}")
```

**Run integration test**:
```bash
uv run pytest tests/test_townlet/integration/test_prerequisites_integration.py -v
```

**Commit integration test**:
```bash
git add tests/test_townlet/integration/test_prerequisites_integration.py

git commit -m "test(task-004a-prereq): Add end-to-end integration test

- Verify complete data flow from YAML to Universe DTO
- Test all 5 parts working together
- Validates prerequisites complete"
```

## Final Verification

**Create verification script**: `scripts/verify_task_004a_prerequisites.py`

Copy script from tasking statement (lines 758-865).

**Run verification**:
```bash
uv run python scripts/verify_task_004a_prerequisites.py
```

**Expected output**:
```
============================================================
TASK-004A-PREREQUISITES Verification
============================================================

=== Part 1: Config Schema Alignment ===
  ✅ Load functions imported
  ✅ Functions work (loaded 25 variables, 8 actions)

=== Part 2: DTO Consolidation ===
  ✅ BarsConfig type alias works

=== Part 3: ObservationSpec Adapter ===
  ✅ Adapter and Universe DTO imported

=== Part 4: HamletConfig Integration ===
  ✅ Integration document exists

=== Part 5: TASK-004A Spec Updates ===
  ✅ Corrected function names present
  ✅ Corrected file names present

============================================================
VERIFICATION SUMMARY
============================================================
Part 1: Config Schema Alignment: ✅ PASS
Part 2: DTO Consolidation: ✅ PASS
Part 3: ObservationSpec Adapter: ✅ PASS
Part 4: HamletConfig Integration: ✅ PASS
Part 5: TASK-004A Spec Updates: ✅ PASS

============================================================
🎉 ALL PREREQUISITES COMPLETE!
============================================================
```

**Commit verification script**:
```bash
git add scripts/verify_task_004a_prerequisites.py

git commit -m "test(task-004a-prereq): Add verification script

- Automated verification of all 5 parts
- Checks imports, functions, documentation
- Confirms prerequisites complete"
```

## Final Push

```bash
git push -u origin claude/task-003-uac-core-dtos-011CUuwRL93WAns6EedRh7c3
```

## Debugging Protocol (If Tests Fail)

### Step 1: Read Error Message Carefully

```python
# Example error:
# AttributeError: module 'townlet.vfs.schema' has no attribute 'load_variables_reference_config'
```

**Diagnosis**:
1. Function not defined?
2. Function name typo?
3. Import path wrong?
4. File not saved?

### Step 2: Minimal Reproduction

```python
# Test in isolation
from townlet.vfs.schema import load_variables_reference_config

# Does import work? If no, check file exists and function defined
# If yes, test with minimal input
result = load_variables_reference_config(Path("configs/L0_0_minimal"))
print(result)  # What's the actual output?
```

### Step 3: Hypothesis Testing

Form hypothesis → Test → Revise

Example:
- **Hypothesis**: "Function exists but returns wrong type"
- **Test**: `print(type(result))`
- **Result**: `<class 'dict'>` (expected `list`)
- **Fix**: Change return statement to return list not dict

### Step 4: Check Repository Reality

If integration test fails, check actual structure:

```python
# What does VFSObservationSpecBuilder actually return?
from townlet.vfs.observation_builder import VFSObservationSpecBuilder

# Inspect actual method names
print(dir(VFSObservationSpecBuilder))

# Is it build_spec() or build_observation_spec()?
```

**Red Flag**: If method names don't match tasking statement, **STOP and ASK USER** before guessing.

### Step 5: Ask User (If Stuck)

**Questions to ask**:

1. **Part 1**: "What's the actual method name on `VFSObservationSpecBuilder`? Is it `build_spec()` or `build_observation_spec()`?"

2. **Part 3**: "What are the actual fields on VFS `ObservationField`? Need to inspect `townlet.vfs.schema.ObservationField`."

3. **Part 4**: "Does `HamletConfig.load()` actually load all 7 config files (substrate, bars, cascades, affordances, cues, training, variables_reference)?"

4. **Integration**: "The integration test is failing with [specific error]. Repository structure differs from tasking statement. How should I proceed?"

**Don't force it** - if architectural assumptions are wrong, get clarification.

## Success Criteria (Final Checklist)

### Functional
- [ ] Part 1: `load_variables_reference_config()` works (4 tests passing)
- [ ] Part 1: `load_global_actions_config()` works (4 tests passing)
- [ ] Part 1: `ActionSpaceConfig` class works (2 tests passing)
- [ ] Part 2: `BarsConfig` importable from `townlet.config.bar` (2 tests passing)
- [ ] Part 3: `vfs_to_universe_observation_spec()` works (10 tests passing)
- [ ] Part 3: Universe `ObservationSpec` DTO exists and works
- [ ] Part 4: `COMPILER-HAMLET-INTEGRATION.md` complete
- [ ] Part 5: TASK-004A spec verified (no outdated references)

### Testing
- [ ] All unit tests passing (22+ tests)
- [ ] Integration test passing (end-to-end)
- [ ] Verification script passing (all parts ✅)

### Git
- [ ] 6 commits (one per part + integration + verification)
- [ ] All commits have clear messages
- [ ] All changes pushed to remote

### Deliverables
- [ ] Can import all new functions without errors
- [ ] TASK-004A unblocked (ready to implement)
- [ ] Documentation complete

## Time Management

**Checkpoints**:
- [ ] Part 1 complete by hour 4 (3-4h estimated)
- [ ] Part 2 complete by hour 6 (2-3h estimated)
- [ ] Part 3 complete by hour 9 (2-3h estimated)
- [ ] Part 4 complete by hour 10 (1-2h estimated)
- [ ] Part 5 complete by hour 12 (1-2h estimated)

**Total**: 8-12 hours

**Red Flag**: If you exceed 15 hours total, STOP and reassess. Something is architecturally wrong.

## What Success Looks Like

At the end, you should be able to:

```python
# All imports work
from townlet.vfs.schema import load_variables_reference_config
from townlet.environment.action_space_builder import load_global_actions_config, ActionSpaceConfig
from townlet.config.bar import BarsConfig
from townlet.universe.adapters.vfs_adapter import vfs_to_universe_observation_spec
from townlet.universe.dto.observation_spec import ObservationSpec

# End-to-end flow works
config_dir = Path("configs/L0_0_minimal")
hamlet_config = HamletConfig.load(config_dir)
variables = load_variables_reference_config(config_dir)
actions = load_global_actions_config()

# Adapter works
registry = VariableRegistry()
for var in variables:
    registry.register(var)

builder = VFSObservationSpecBuilder(registry, hamlet_config.substrate)
vfs_fields = builder.build_spec()
universe_spec = vfs_to_universe_observation_spec(vfs_fields)

print(f"✅ SUCCESS: {universe_spec.total_dims} observation dims")
```

**Verification script output**: 🎉 ALL PREREQUISITES COMPLETE!

**TASK-004A ready**: Compiler can now be implemented without blockers.

---

## Begin Implementation

**Steps to start**:

1. Read tasking statement: `/home/user/hamlet/docs/tasks/TASK-004A-PREREQUISITES-TASKING.md`
2. Read this prompt again if needed
3. Start with Part 1, Test 1: `test_load_variables_reference_config_returns_list`
4. Follow TDD: RED → GREEN → REFACTOR → COMMIT
5. Work sequentially through all 5 parts
6. Run verification script
7. Push to remote

**Your first action**: Read the tasking statement, then say "Ready to begin Part 1" and start writing the first failing test.

---

**GOOD LUCK!** 🚀

Use TDD rigorously. Test first, code second, refactor third. Commit often. Debug systematically. Ask when stuck.
