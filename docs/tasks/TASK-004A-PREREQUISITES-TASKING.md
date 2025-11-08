# TASKING STATEMENT: TASK-004A-PREREQUISITES Implementation

**Date**: 2025-11-08
**Branch**: `claude/task-003-uac-core-dtos-011CUuwRL93WAns6EedRh7c3`
**Status**: Ready for Implementation
**Estimated Effort**: 8-12 hours
**Priority**: CRITICAL (Blocks TASK-004A Universe Compiler)

---

## Executive Summary

**DO THIS FIRST**: Before implementing TASK-004A Universe Compiler, you MUST resolve 5 critical blockers where the implementation plan assumptions don't match repository reality.

**Your Task**: Implement TASK-004A-PREREQUISITES in 5 parts to unblock TASK-004A.

**Success Criteria**: All 5 parts complete, all tests passing, verification script confirms readiness for TASK-004A.

---

## Context: Why These Prerequisites Are Critical

An external peer reviewer identified 5 blockers in the TASK-004A plan:

1. **Config Filename Mismatch**: Plan expects `variables.yaml`, repo has `variables_reference.yaml`
2. **DTO Package Mismatch**: Plan expects `BarsConfig` in `townlet.config.bar`, actually in `townlet.environment.cascade_config`
3. **Missing Load Functions**: Plan expects `load_variables_config()` that doesn't exist
4. **ObservationSpec Incompatibility**: VFS `ObservationField` ≠ Universe `ObservationSpec` (incompatible DTOs)
5. **HamletConfig Duplication**: Unclear how compiler relates to existing `HamletConfig.load()`

**Without these fixes**, TASK-004A implementation will fail with import errors, missing functions, and incompatible data contracts.

---

## Prerequisites Document Location

**Primary Reference**: `/home/user/hamlet/docs/tasks/TASK-004A-PREREQUISITES.md`

Read this document fully before starting. It contains complete specifications for all 5 parts.

---

## Implementation Order (Sequential)

Execute in this exact order:

```
Part 1: Config Schema Alignment (3-4h)
  └─> Part 2: DTO Consolidation (2-3h)
      └─> Part 3: ObservationSpec Adapter (2-3h)
          └─> Part 4: HamletConfig Integration (1-2h)
              └─> Part 5: Update TASK-004A Spec (1-2h)
```

---

## Part 1: Config Schema Alignment (3-4 hours)

### Goal
Create missing load functions for `variables_reference.yaml` and `configs/global_actions.yaml`.

### Tasks

#### Task 1.1: Create `load_variables_reference_config()`

**File**: `src/townlet/vfs/schema.py` (add to existing file)

**Function Signature**:
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
```

**Implementation Requirements**:
1. Look for `config_dir / "variables_reference.yaml"`
2. Load YAML, extract `variables` section
3. Validate each variable dict against `VariableDef` schema
4. Return `list[VariableDef]`
5. Clear error messages for missing file or invalid schema

**Test File**: `tests/test_townlet/unit/vfs/test_load_variables_reference.py`

**Test Cases**:
```python
def test_load_variables_reference_valid()
def test_load_variables_reference_missing_file_raises()
def test_load_variables_reference_invalid_schema_raises()
def test_load_variables_reference_preserves_order()
```

#### Task 1.2: Create `load_global_actions_config()`

**File**: `src/townlet/environment/action_space_builder.py` (NEW file)

**Function Signature**:
```python
def load_global_actions_config() -> ActionSpaceConfig:
    """Load global action vocabulary from configs/global_actions.yaml.

    Loads from repository-wide shared config (NOT per-config-pack).

    Returns:
        ActionSpaceConfig with substrate + custom actions

    Raises:
        FileNotFoundError: If configs/global_actions.yaml not found
        ValidationError: If YAML schema invalid
    """
```

**Implementation Requirements**:
1. Load from `configs/global_actions.yaml` (repository root, NOT config pack)
2. Extract `substrate_actions` and `custom_actions` sections
3. Compose into single `ActionSpaceConfig` wrapper
4. Return unified action space

**New Class**: `ActionSpaceConfig` wrapper

```python
from dataclasses import dataclass
from townlet.environment.action_config import ActionConfig

@dataclass(frozen=True)
class ActionSpaceConfig:
    """Wrapper for global action vocabulary (substrate + custom)."""

    actions: list[ActionConfig]

    def get_action_by_id(self, action_id: int) -> ActionConfig:
        """Lookup action by ID."""
        for action in self.actions:
            if action.id == action_id:
                return action
        raise KeyError(f"Action ID {action_id} not found")

    def get_action_by_name(self, name: str) -> ActionConfig:
        """Lookup action by name."""
        for action in self.actions:
            if action.name == name:
                return action
        raise KeyError(f"Action '{name}' not found")
```

**Test File**: `tests/test_townlet/unit/environment/test_load_global_actions.py`

**Test Cases**:
```python
def test_load_global_actions_valid()
def test_load_global_actions_missing_file_raises()
def test_action_space_config_get_by_id()
def test_action_space_config_get_by_name()
def test_action_space_config_composed_order()  # substrate actions before custom
```

### Success Criteria Part 1

- [ ] `load_variables_reference_config()` exists and works
- [ ] `load_global_actions_config()` exists and works
- [ ] `ActionSpaceConfig` class exists with lookup methods
- [ ] All tests passing (10 tests minimum)
- [ ] Functions callable from TASK-004A compiler

---

## Part 2: DTO Consolidation (2-3 hours)

### Goal
Add type aliases and compatibility layers for DTO package mismatches.

### Tasks

#### Task 2.1: Create `BarsConfig` Type Alias

**File**: `src/townlet/config/bar.py` (modify existing)

**Add to Exports**:
```python
from townlet.environment.cascade_config import BarsConfig

__all__ = ["BarConfig", "load_bars_config", "BarsConfig"]
```

**Rationale**: TASK-004A plan expects `from townlet.config.bar import BarsConfig`, but `BarsConfig` actually lives in `townlet.environment.cascade_config`. Type alias maintains compatibility.

**Test File**: `tests/test_townlet/unit/config/test_bars_config_alias.py`

**Test Cases**:
```python
def test_bars_config_importable_from_config_bar():
    """Verify BarsConfig can be imported from townlet.config.bar."""
    from townlet.config.bar import BarsConfig
    assert BarsConfig is not None

def test_bars_config_is_same_class():
    """Verify alias points to correct class."""
    from townlet.config.bar import BarsConfig as Alias
    from townlet.environment.cascade_config import BarsConfig as Original
    assert Alias is Original
```

#### Task 2.2: Export `ActionSpaceConfig` from `action_space_builder.py`

**File**: `src/townlet/environment/action_space_builder.py` (from Part 1)

**Ensure Exports**:
```python
__all__ = ["load_global_actions_config", "ActionSpaceConfig"]
```

### Success Criteria Part 2

- [ ] `BarsConfig` importable from `townlet.config.bar`
- [ ] Type alias points to correct class
- [ ] `ActionSpaceConfig` importable from `action_space_builder`
- [ ] All tests passing (2 tests minimum)

---

## Part 3: ObservationSpec Adapter (2-3 hours)

### Goal
Bridge incompatible VFS and Universe ObservationSpec DTOs.

### Problem

**VFS ObservationField** (from `VFSObservationSpecBuilder.build_spec()`):
```python
@dataclass
class ObservationField:
    id: str
    source_variable: str
    shape: tuple[int, ...]  # Multi-dimensional
    normalization: NormalizationSpec | None
    exposed_to: list[str]
```

**Universe ObservationSpec** (expected by TASK-004A):
```python
@dataclass
class ObservationField:
    name: str
    type: Literal["scalar", "vector", "categorical", "spatial_grid"]
    dims: int  # Flattened
    start_index: int
    end_index: int
    scope: Literal["global", "agent", "agent_private"]
    semantic_type: str | None
```

**They're incompatible!** Need adapter.

### Tasks

#### Task 3.1: Create VFS Adapter Module

**File**: `src/townlet/universe/adapters/vfs_adapter.py` (NEW)

**Function Signature**:
```python
def vfs_to_universe_observation_spec(
    vfs_fields: list[VFSObservationField]
) -> UniverseObservationSpec:
    """Convert VFS ObservationField list to Universe ObservationSpec.

    Performs:
    1. Flatten shapes → dims (product of shape dimensions)
    2. Compute contiguous start_index, end_index
    3. Infer semantic_type from field names and source_variable
    4. Map VFS scope to Universe scope

    Args:
        vfs_fields: List of VFS ObservationField from VFSObservationSpecBuilder

    Returns:
        Universe ObservationSpec with flattened fields
    """
```

**Implementation Requirements**:

1. **Flatten Shapes**: `shape=(8,) → dims=8`, `shape=(5,5) → dims=25`
2. **Compute Indices**: Accumulate running index for start/end
3. **Infer Semantic Types**:
   ```python
   def _infer_semantic_type(field_id: str, source_variable: str) -> str:
       """Infer semantic type from field metadata."""
       if "position" in field_id.lower():
           return "position"
       elif "meter" in source_variable or field_id in METER_NAMES:
           return "meter"
       elif "affordance" in field_id.lower():
           return "affordance"
       elif "time" in field_id.lower() or "temporal" in field_id.lower():
           return "temporal"
       else:
           return "unknown"
   ```
4. **Infer Field Types**:
   ```python
   def _infer_field_type(shape: tuple, semantic_type: str) -> str:
       """Infer field type from shape and semantics."""
       if len(shape) == 0 or (len(shape) == 1 and shape[0] == 1):
           return "scalar"
       elif len(shape) == 2 and shape[0] == shape[1]:
           return "spatial_grid"  # Square grid (e.g., 5×5 local vision)
       else:
           return "vector"
   ```

**New DTO**: `UniverseObservationSpec`

**File**: `src/townlet/universe/dto/observation_spec.py` (create directory if needed)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ObservationField:
    """Single field in flattened observation vector."""
    name: str
    type: str  # "scalar", "vector", "categorical", "spatial_grid"
    dims: int
    start_index: int
    end_index: int
    scope: str  # "global", "agent", "agent_private"
    description: str
    semantic_type: str | None = None
    categorical_labels: tuple[str, ...] | None = None

@dataclass(frozen=True)
class ObservationSpec:
    """Complete observation specification."""
    total_dims: int
    fields: tuple[ObservationField, ...]
    encoding_version: str = "1.0"

    @classmethod
    def from_fields(cls, fields: list[ObservationField], encoding_version: str = "1.0"):
        """Build from field list, computing total_dims."""
        total_dims = sum(f.dims for f in fields)
        return cls(total_dims=total_dims, fields=tuple(fields), encoding_version=encoding_version)

    def get_field_by_name(self, name: str) -> ObservationField:
        """Lookup field by name."""
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(f"Field '{name}' not found")
```

**Test File**: `tests/test_townlet/unit/universe/adapters/test_vfs_adapter.py`

**Test Cases**:
```python
def test_flatten_1d_shape()
def test_flatten_2d_shape()  # (5,5) → 25
def test_compute_contiguous_indices()
def test_infer_semantic_type_position()
def test_infer_semantic_type_meter()
def test_infer_semantic_type_affordance()
def test_infer_field_type_scalar()
def test_infer_field_type_spatial_grid()
def test_adapter_preserves_field_order()
def test_adapter_total_dims_correct()
```

### Success Criteria Part 3

- [ ] `vfs_to_universe_observation_spec()` exists and works
- [ ] Universe `ObservationSpec` DTO exists
- [ ] Adapter correctly flattens shapes → dims
- [ ] Adapter computes contiguous indices
- [ ] Adapter infers semantic types correctly
- [ ] All tests passing (10 tests minimum)
- [ ] VFS output → adapter → Universe DTO (end-to-end works)

---

## Part 4: HamletConfig Integration Strategy (1-2 hours)

### Goal
Document how compiler builds on existing `HamletConfig.load()`.

### Current State

**File**: `src/townlet/config/hamlet.py`

`HamletConfig` already exists and loads:
- `substrate.yaml`
- `bars.yaml`
- `cascades.yaml`
- `affordances.yaml`
- `cues.yaml`
- `training.yaml`

`HamletConfig.load(config_dir)` performs validation and returns master config.

### Strategy

**Compiler BUILDS ON `HamletConfig`, doesn't replace it.**

```
Data Flow:
YAML Files → HamletConfig.load() → Validation → HamletConfig Instance
                                                      ↓
                                 Additional Loaders (VFS, actions)
                                                      ↓
                                        UniverseCompiler.compile()
                                                      ↓
                                   Additional Validation + Metadata
                                                      ↓
                                          CompiledUniverse
```

### Tasks

#### Task 4.1: Document Integration Pattern

**File**: `docs/architecture/COMPILER-HAMLET-INTEGRATION.md` (NEW)

**Content Template**:
```markdown
# Universe Compiler & HamletConfig Integration

## Architecture Decision

**Decision**: Universe Compiler builds ON HamletConfig, not parallel to it.

## Rationale

1. **Avoid Duplication**: HamletConfig already loads 7/9 config files
2. **Preserve Validation**: HamletConfig performs field-level validation
3. **Backward Compatibility**: DemoRunner uses HamletConfig today
4. **Clear Separation**: HamletConfig = file loading, Compiler = cross-validation

## Data Flow

[Include diagram]

## Integration Points

### Stage 1: Parse Individual Files

Compiler uses `HamletConfig.load()` as PRIMARY loader:

```python
def _stage_1_parse_individual_files(config_dir: Path) -> RawConfigs:
    # PRIMARY: Load 7 core config files via HamletConfig
    hamlet_config = HamletConfig.load(config_dir)

    # ADDITIONAL: Load VFS and action configs
    variables_reference = load_variables_reference_config(config_dir)
    global_actions = load_global_actions_config()

    return RawConfigs(
        hamlet_config=hamlet_config,
        variables_reference=variables_reference,
        global_actions=global_actions
    )
```

### RawConfigs Structure

```python
@dataclass
class RawConfigs:
    hamlet_config: HamletConfig  # Master config
    variables_reference: list[VariableDef]
    global_actions: ActionSpaceConfig

    # Convenience properties
    @property
    def substrate(self) -> SubstrateConfig:
        return self.hamlet_config.substrate

    # ... (bars, cascades, affordances, cues, training)
```

### What Compiler Adds

Compiler adds BEYOND HamletConfig:
1. Cross-file reference resolution (Stage 3)
2. Semantic validation (Stage 4)
3. Metadata computation (Stage 5)
4. Optimization data (Stage 6)
5. Caching (Stage 9)
```

#### Task 4.2: Update TASK-004A-PREREQUISITES.md

**File**: `docs/tasks/TASK-004A-PREREQUISITES.md`

In Part 4, add:
```markdown
**Implementation Status**: ✅ DOCUMENTED

See: `docs/architecture/COMPILER-HAMLET-INTEGRATION.md`
```

### Success Criteria Part 4

- [ ] `COMPILER-HAMLET-INTEGRATION.md` created
- [ ] Integration pattern clearly documented
- [ ] Data flow diagram included
- [ ] RawConfigs structure specified
- [ ] Clear separation of concerns documented

---

## Part 5: Update TASK-004A Spec (1-2 hours)

### Goal
Correct all imports and assumptions in TASK-004A plan.

### Tasks

#### Task 5.1: Verify TASK-004A Updates

**File**: `docs/tasks/TASK-004A-COMPILER-IMPLEMENTATION.md`

**Already Updated** (completed in previous session):
- ✅ Prerequisites warning added at top
- ✅ `RawConfigs` structure updated to use `hamlet_config`
- ✅ Stage 1 updated to use `HamletConfig.load()`
- ✅ All imports corrected (`BarsConfig`, `load_variables_reference_config`, etc.)
- ✅ VFS adapter usage added to Stage 5
- ✅ Effort estimate updated (60-84h total)

**Verify These Sections** (read through, confirm correctness):

1. **Lines 177-180**: Import statements
   ```python
   from townlet.config.bar import load_bars_config, BarsConfig  # Type alias
   from townlet.vfs.schema import load_variables_reference_config
   from townlet.environment.action_space_builder import load_global_actions_config
   ```

2. **Lines 244-280**: `RawConfigs` structure with `hamlet_config`

3. **Lines 283-343**: Stage 1 using `HamletConfig.load()`

4. **Lines 1150-1176**: VFS adapter usage in Stage 5

5. **Lines 1229-1253**: Stage 5 metadata computation with adapter

#### Task 5.2: Update TDD Plan If Needed

**File**: `docs/plans/2025-11-08-task-004a-tdd-implementation-plan.md`

**Scan for References** to old assumptions:
- Search for `load_variables_config` (should be `load_variables_reference_config`)
- Search for `variables.yaml` (should be `variables_reference.yaml`)
- Search for `raw_configs.variables` (should be `raw_configs.variables_reference`)

**Update if Found** (use Edit tool to correct)

### Success Criteria Part 5

- [ ] TASK-004A plan verified (all imports correct)
- [ ] TDD plan scanned and updated if needed
- [ ] No references to non-existent functions
- [ ] No references to incorrect file names
- [ ] Integration approach matches Part 4 documentation

---

## Testing Strategy

### Unit Tests (Each Part)

Each part has specific unit tests (listed above). Run after completing each part:

```bash
# Part 1
uv run pytest tests/test_townlet/unit/vfs/test_load_variables_reference.py -v
uv run pytest tests/test_townlet/unit/environment/test_load_global_actions.py -v

# Part 2
uv run pytest tests/test_townlet/unit/config/test_bars_config_alias.py -v

# Part 3
uv run pytest tests/test_townlet/unit/universe/adapters/test_vfs_adapter.py -v

# Part 5 (if changes made)
uv run pytest tests/test_townlet/unit/config/ -v
```

### Integration Test (Final Verification)

**File**: `tests/test_townlet/integration/test_prerequisites_integration.py` (NEW)

```python
"""Integration test: Verify all prerequisites work together."""

def test_prerequisites_end_to_end():
    """Verify complete data flow: HamletConfig → VFS → Adapter → Universe DTO."""
    from pathlib import Path
    from townlet.config.hamlet import HamletConfig
    from townlet.vfs.schema import load_variables_reference_config
    from townlet.environment.action_space_builder import load_global_actions_config
    from townlet.vfs.observation_builder import VFSObservationSpecBuilder
    from townlet.vfs.registry import VariableRegistry
    from townlet.universe.adapters.vfs_adapter import vfs_to_universe_observation_spec

    # Load test config
    config_dir = Path("configs/L0_0_minimal")

    # Part 1: Load functions
    hamlet_config = HamletConfig.load(config_dir)
    variables_reference = load_variables_reference_config(config_dir)
    global_actions = load_global_actions_config()

    assert hamlet_config is not None
    assert len(variables_reference) > 0
    assert len(global_actions.actions) > 0

    # Part 2: Type alias
    from townlet.config.bar import BarsConfig
    assert isinstance(hamlet_config.bars, BarsConfig)

    # Part 3: VFS → Universe adapter
    temp_registry = VariableRegistry()
    for var_def in variables_reference:
        temp_registry.register(var_def)

    builder = VFSObservationSpecBuilder(
        variable_registry=temp_registry,
        substrate=hamlet_config.substrate
    )
    vfs_fields = builder.build_observation_spec()  # or .build_spec()

    universe_obs_spec = vfs_to_universe_observation_spec(vfs_fields)

    assert universe_obs_spec.total_dims > 0
    assert len(universe_obs_spec.fields) > 0

    print(f"✅ Prerequisites integration test passed!")
    print(f"   Observation dims: {universe_obs_spec.total_dims}")
    print(f"   Actions: {len(global_actions.actions)}")
```

Run after completing all 5 parts:

```bash
uv run pytest tests/test_townlet/integration/test_prerequisites_integration.py -v
```

---

## Final Verification Script

**After completing all 5 parts**, run comprehensive verification:

**File**: `scripts/verify_task_004a_prerequisites.py` (NEW)

```python
"""Verify TASK-004A-PREREQUISITES completion."""
import sys
from pathlib import Path


def verify_part_1():
    """Verify Part 1: Config schema alignment."""
    print("\n=== Part 1: Config Schema Alignment ===")

    try:
        from townlet.vfs.schema import load_variables_reference_config
        from townlet.environment.action_space_builder import load_global_actions_config, ActionSpaceConfig
        print("  ✅ Load functions imported")

        # Test with L0_0_minimal
        config_dir = Path("configs/L0_0_minimal")
        variables = load_variables_reference_config(config_dir)
        actions = load_global_actions_config()

        assert len(variables) > 0
        assert len(actions.actions) > 0
        print(f"  ✅ Functions work (loaded {len(variables)} variables, {len(actions.actions)} actions)")

        return True
    except Exception as e:
        print(f"  ❌ Part 1 failed: {e}")
        return False


def verify_part_2():
    """Verify Part 2: DTO consolidation."""
    print("\n=== Part 2: DTO Consolidation ===")

    try:
        from townlet.config.bar import BarsConfig
        from townlet.environment.cascade_config import BarsConfig as Original

        assert BarsConfig is Original
        print("  ✅ BarsConfig type alias works")

        return True
    except Exception as e:
        print(f"  ❌ Part 2 failed: {e}")
        return False


def verify_part_3():
    """Verify Part 3: ObservationSpec adapter."""
    print("\n=== Part 3: ObservationSpec Adapter ===")

    try:
        from townlet.universe.adapters.vfs_adapter import vfs_to_universe_observation_spec
        from townlet.universe.dto.observation_spec import ObservationSpec
        print("  ✅ Adapter and Universe DTO imported")

        # TODO: Test with real VFS fields
        print("  ⚠️  Integration test required (run test_prerequisites_integration.py)")

        return True
    except Exception as e:
        print(f"  ❌ Part 3 failed: {e}")
        return False


def verify_part_4():
    """Verify Part 4: HamletConfig integration."""
    print("\n=== Part 4: HamletConfig Integration ===")

    doc_path = Path("docs/architecture/COMPILER-HAMLET-INTEGRATION.md")
    if doc_path.exists():
        print(f"  ✅ Integration document exists")
        return True
    else:
        print(f"  ❌ Missing: {doc_path}")
        return False


def verify_part_5():
    """Verify Part 5: TASK-004A spec updates."""
    print("\n=== Part 5: TASK-004A Spec Updates ===")

    task_path = Path("docs/tasks/TASK-004A-COMPILER-IMPLEMENTATION.md")
    if not task_path.exists():
        print("  ❌ TASK-004A spec not found")
        return False

    content = task_path.read_text()

    # Check for corrected imports
    if "load_variables_reference_config" in content:
        print("  ✅ Corrected function names present")
    else:
        print("  ❌ TASK-004A still has old function names")
        return False

    if "variables_reference.yaml" in content:
        print("  ✅ Corrected file names present")
    else:
        print("  ❌ TASK-004A still has old file names")
        return False

    return True


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("TASK-004A-PREREQUISITES Verification")
    print("=" * 60)

    results = {
        "Part 1: Config Schema Alignment": verify_part_1(),
        "Part 2: DTO Consolidation": verify_part_2(),
        "Part 3: ObservationSpec Adapter": verify_part_3(),
        "Part 4: HamletConfig Integration": verify_part_4(),
        "Part 5: TASK-004A Spec Updates": verify_part_5(),
    }

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    for part, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{part}: {status}")

    if all(results.values()):
        print("\n" + "=" * 60)
        print("🎉 ALL PREREQUISITES COMPLETE!")
        print("=" * 60)
        print("\nNext Steps:")
        print("1. Run integration test:")
        print("   uv run pytest tests/test_townlet/integration/test_prerequisites_integration.py -v")
        print("\n2. Run full test suite:")
        print("   uv run pytest tests/test_townlet/ -v")
        print("\n3. Commit changes:")
        print("   git add .")
        print("   git commit -m 'feat(task-004a): Complete prerequisites for Universe Compiler'")
        print("\n4. Push to remote:")
        print("   git push -u origin claude/task-003-uac-core-dtos-011CUuwRL93WAns6EedRh7c3")
        print("\n5. Begin TASK-004A implementation!")
        sys.exit(0)
    else:
        print("\n⚠️  Some parts incomplete - fix before proceeding")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## Commit Strategy

### After Each Part

```bash
# Part 1
git add src/townlet/vfs/schema.py \
        src/townlet/environment/action_space_builder.py \
        tests/test_townlet/unit/vfs/test_load_variables_reference.py \
        tests/test_townlet/unit/environment/test_load_global_actions.py

git commit -m "feat(task-004a-prereq): Part 1 - Config schema alignment

- Add load_variables_reference_config() to vfs/schema.py
- Add load_global_actions_config() to action_space_builder.py
- Create ActionSpaceConfig wrapper class
- Add 10 unit tests for load functions
- Resolves Blocker #1 (config filename mismatch)"

# Part 2
git add src/townlet/config/bar.py \
        tests/test_townlet/unit/config/test_bars_config_alias.py

git commit -m "feat(task-004a-prereq): Part 2 - DTO consolidation

- Add BarsConfig type alias to townlet.config.bar
- Export ActionSpaceConfig from action_space_builder
- Add 2 unit tests for type alias
- Resolves Blocker #2 (DTO package mismatch)"

# Part 3
git add src/townlet/universe/adapters/vfs_adapter.py \
        src/townlet/universe/dto/observation_spec.py \
        tests/test_townlet/unit/universe/adapters/test_vfs_adapter.py

git commit -m "feat(task-004a-prereq): Part 3 - ObservationSpec adapter

- Create vfs_to_universe_observation_spec() adapter
- Create Universe ObservationSpec DTO
- Implement shape flattening and index computation
- Add semantic type inference
- Add 10 unit tests for adapter
- Resolves Blocker #3 (ObservationSpec incompatibility)"

# Part 4
git add docs/architecture/COMPILER-HAMLET-INTEGRATION.md \
        docs/tasks/TASK-004A-PREREQUISITES.md

git commit -m "docs(task-004a-prereq): Part 4 - HamletConfig integration

- Document compiler builds on HamletConfig strategy
- Add data flow diagram
- Specify RawConfigs structure
- Clarify separation of concerns
- Resolves Blocker #5 (HamletConfig duplication)"

# Part 5
git add docs/tasks/TASK-004A-COMPILER-IMPLEMENTATION.md \
        docs/plans/2025-11-08-task-004a-tdd-implementation-plan.md

git commit -m "docs(task-004a-prereq): Part 5 - Update TASK-004A spec

- Verify all imports corrected in TASK-004A plan
- Update TDD plan with correct function names
- Ensure no references to non-existent functions
- Resolves Blocker #4 (missing load functions)"

# Final Commit
git add tests/test_townlet/integration/test_prerequisites_integration.py \
        scripts/verify_task_004a_prerequisites.py

git commit -m "test(task-004a-prereq): Add integration test and verification

- Add end-to-end integration test
- Add comprehensive verification script
- All 5 prerequisites complete and tested"
```

### Final Push

```bash
git push -u origin claude/task-003-uac-core-dtos-011CUuwRL93WAns6EedRh7c3
```

---

## Success Criteria (Overall)

### Functional Requirements

- [ ] Part 1: `load_variables_reference_config()` works
- [ ] Part 1: `load_global_actions_config()` works
- [ ] Part 1: `ActionSpaceConfig` class works
- [ ] Part 2: `BarsConfig` importable from `townlet.config.bar`
- [ ] Part 3: `vfs_to_universe_observation_spec()` works
- [ ] Part 3: Universe `ObservationSpec` DTO exists
- [ ] Part 4: Integration documentation complete
- [ ] Part 5: TASK-004A spec verified/updated

### Testing Requirements

- [ ] 10+ unit tests for Part 1 (all passing)
- [ ] 2+ unit tests for Part 2 (all passing)
- [ ] 10+ unit tests for Part 3 (all passing)
- [ ] Integration test passing
- [ ] Verification script passes

### Documentation Requirements

- [ ] COMPILER-HAMLET-INTEGRATION.md complete
- [ ] TASK-004A plan verified (correct imports)
- [ ] TDD plan updated (if needed)
- [ ] All commit messages clear and descriptive

---

## What Success Looks Like

When you're done:

1. Run verification script:
   ```bash
   uv run python scripts/verify_task_004a_prerequisites.py
   ```
   Output: `🎉 ALL PREREQUISITES COMPLETE!`

2. Run integration test:
   ```bash
   uv run pytest tests/test_townlet/integration/test_prerequisites_integration.py -v
   ```
   Output: All tests passing ✅

3. Can import all new functions without errors:
   ```python
   from townlet.vfs.schema import load_variables_reference_config
   from townlet.environment.action_space_builder import load_global_actions_config
   from townlet.config.bar import BarsConfig
   from townlet.universe.adapters.vfs_adapter import vfs_to_universe_observation_spec
   ```

4. TASK-004A implementation can proceed without blockers

---

## Red Flags (Stop and Ask)

If you encounter:

1. **VFSObservationSpecBuilder API different than expected**
   - Check actual method name: `build_spec()` vs `build_observation_spec()`
   - Adapter signature in Part 3 and read actual VFS ObservationField structure

2. **HamletConfig doesn't load all expected files**
   - Verify which files HamletConfig actually loads
   - May need to add more to `load_variables_reference_config()`

3. **Tests fail consistently**
   - Don't force it - understand root cause
   - Ask for clarification if architectural assumptions wrong

4. **Circular imports**
   - Check import order
   - May need to restructure module boundaries

---

## Time Checkpoint

After each part, note actual time spent:

- [ ] Part 1: ____ hours (estimated 3-4h)
- [ ] Part 2: ____ hours (estimated 2-3h)
- [ ] Part 3: ____ hours (estimated 2-3h)
- [ ] Part 4: ____ hours (estimated 1-2h)
- [ ] Part 5: ____ hours (estimated 1-2h)

**Total: ____ hours (estimated 8-12h)**

If running significantly over (>15h), stop and reassess.

---

## Final Deliverables Checklist

- [ ] All 5 parts implemented
- [ ] All unit tests passing (22+ tests)
- [ ] Integration test passing
- [ ] Verification script passing
- [ ] Documentation complete
- [ ] All changes committed
- [ ] Changes pushed to remote
- [ ] TASK-004A unblocked and ready to implement

---

## Questions to Ask User If Blocked

1. **Part 1**: "VFSObservationSpecBuilder actual method name? Is it `build_spec()` or `build_observation_spec()`?"

2. **Part 3**: "What's the actual structure of VFS ObservationField? Need to inspect `townlet.vfs.schema.ObservationField` fields."

3. **Part 4**: "Does HamletConfig.load() handle all 7 config files? Which ones does it actually load?"

4. **Part 5**: "Should I update TDD plan even if TASK-004A plan already correct?"

---

**READY TO START**: You have all information needed. Begin with Part 1 tomorrow morning.

**GOOD LUCK!** 🚀
