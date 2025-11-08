# TASK-004A-PREREQUISITES: Universe Compiler Prerequisites

**Task**: Resolve blockers before TASK-004A Universe Compiler implementation
**Priority**: CRITICAL (Blocks TASK-004A)
**Effort**: 8-12 hours
**Status**: Not Started
**Created**: 2025-11-08
**Depends On**: TASK-003 (Complete ✅)
**Blocks**: TASK-004A Universe Compiler Implementation

---

## Executive Summary

Independent peer review of TASK-004A identified **5 critical blockers** where the implementation plan assumptions don't match the actual repository state. These must be resolved before TASK-004A implementation can proceed.

**Key Issues**:
1. Config filenames mismatch (plan expects `variables.yaml`, repo has `variables_reference.yaml`)
2. DTO package location mismatch (collection DTOs in wrong modules)
3. ObservationSpec contract mismatch (VFS vs Universe DTO versions)
4. HamletConfig relationship undefined (duplication risk)
5. Missing adapters between VFS and Universe DTOs

**This task resolves all blockers and updates TASK-004A spec to match reality.**

---

## Problem Statement

### Blocker #1: Config Filenames Don't Match Repository

**Plan Expects**:
```python
# TASK-004A Stage 1 assumes:
variables = load_variables_config(config_dir / "variables.yaml")  # ❌ FILE DOESN'T EXIST
actions = load_action_space_config(config_dir / "actions.yaml")   # ❌ FILE DOESN'T EXIST
```

**Repository Reality**:
```bash
# Actual files:
configs/L0_0_minimal/variables_reference.yaml  # ✅ EXISTS
configs/global_actions.yaml                     # ✅ EXISTS (shared, not per-pack)
```

**Impact**: Stage 1 parse will fail immediately - can't find config files.

### Blocker #2: DTO Package Locations Inconsistent

**Plan Expects**:
```python
from townlet.config.bar import BarsConfig  # ❌ DOESN'T EXIST
from townlet.config.cascade import CascadesConfig  # ❌ DOESN'T EXIST
from townlet.config.affordance import AffordanceConfigCollection  # ❌ DOESN'T EXIST
```

**Repository Reality**:
```python
# Collection DTOs are in environment package:
from townlet.environment.cascade_config import BarsConfig  # ✅ EXISTS
from townlet.environment.affordance_config import AffordanceConfigCollection  # ✅ EXISTS

# Individual DTOs + loaders are in config package:
from townlet.config.bar import BarConfig, load_bars_config  # ✅ EXISTS (returns list[BarConfig])
```

**Impact**: RawConfigs dataclass can't be constructed - wrong imports.

### Blocker #3: ObservationSpec Has Two Incompatible Versions

**VFS ObservationField** (`src/townlet/vfs/schema.py`):
```python
@dataclass
class ObservationField:
    id: str
    source_variable: str
    shape: list[int]  # ❌ NOT flattened
    normalization: NormalizationSpec | None
    # ❌ Missing: start_index, end_index, semantic_type
```

**Universe DTO ObservationField** (`src/townlet/universe/dto/observation_spec.py`):
```python
@dataclass(frozen=True)
class ObservationField:
    name: str
    dims: int  # ✅ Flattened
    start_index: int  # ✅ Has indices
    end_index: int
    semantic_type: str | None  # ✅ Has semantic tags
```

**Impact**: Stage 5 can't convert VFS builder output to Universe DTO - no adapter exists.

### Blocker #4: HamletConfig Already Does Validation

`HamletConfig.load()` already:
- Loads all 9 config files
- Performs cross-validation (batch_size vs buffer, network type vs observability, grid capacity)
- Composes master config

**DemoRunner uses it today**. Plan doesn't specify:
- Will compiler replace HamletConfig?
- Build on top of it?
- Run in parallel (duplication/divergence risk)?

**Impact**: Risk of duplicate validation logic, inconsistent error messages, maintenance burden.

### Blocker #5: Missing load_variables_config Function

**Plan Expects**:
```python
from townlet.vfs.schema import load_variables_config  # ❌ DOESN'T EXIST
```

**Repository Reality**:
```python
# src/townlet/vfs/schema.py has VariableDef but NO load function
# Variables loaded manually in VectorizedHamletEnv.__init__
```

**Impact**: Can't load variables in Stage 1.

---

## Solution: 5-Part Prerequisites Task

### Part 1: Config Schema Alignment (3-4 hours)

**Decision**: Keep current structure (`variables_reference.yaml` + `configs/global_actions.yaml`)

**Why**: Changing would require:
- Renaming files in all 13 config packs
- Updating VectorizedHamletEnv to look for new names
- Migrating all existing configs
- Higher risk, no benefit

**Actions**:

1. **Create `load_variables_reference_config()` function** (30min)

**File**: `src/townlet/vfs/schema.py`

```python
def load_variables_reference_config(config_dir: Path) -> list[VariableDef]:
    """Load variables from variables_reference.yaml.

    Args:
        config_dir: Directory containing variables_reference.yaml

    Returns:
        List of VariableDef instances

    Raises:
        FileNotFoundError: If variables_reference.yaml not found
        ValidationError: If YAML invalid

    Example:
        >>> variables = load_variables_reference_config(Path("configs/L0_0_minimal"))
        >>> len(variables)
        12
    """
    from townlet.config.base import load_yaml_section

    yaml_path = config_dir / "variables_reference.yaml"
    data = load_yaml_section(yaml_path, "variables")

    variables = []
    for var_data in data:
        var = VariableDef(**var_data)
        variables.append(var)

    return variables
```

2. **Create `load_global_actions_config()` function** (30min)

**File**: `src/townlet/environment/action_config.py`

```python
def load_global_actions_config() -> ActionSpaceConfig:
    """Load global action vocabulary from configs/global_actions.yaml.

    Returns:
        ActionSpaceConfig with substrate + custom actions

    Raises:
        FileNotFoundError: If global_actions.yaml not found

    Example:
        >>> actions = load_global_actions_config()
        >>> len(actions.actions)
        8  # Grid2D: 6 substrate + 2 custom
    """
    from pathlib import Path
    from townlet.config.base import load_yaml_section

    yaml_path = Path("configs/global_actions.yaml")

    # Load substrate actions
    substrate_actions_data = load_yaml_section(yaml_path, "substrate_actions")
    substrate_actions = [ActionConfig(**a) for a in substrate_actions_data]

    # Load custom actions
    custom_actions_data = load_yaml_section(yaml_path, "custom_actions")
    custom_actions = [ActionConfig(**a) for a in custom_actions_data]

    return ActionSpaceConfig(
        actions=substrate_actions + custom_actions
    )
```

3. **Add tests** (1 hour)

**File**: `tests/test_townlet/unit/vfs/test_load_variables_reference.py`

```python
"""Test load_variables_reference_config function."""
import pytest
from pathlib import Path
from townlet.vfs.schema import load_variables_reference_config


def test_load_variables_reference_l0():
    """Verify loading variables_reference.yaml from L0_0_minimal."""
    config_dir = Path("configs/L0_0_minimal")

    variables = load_variables_reference_config(config_dir)

    assert len(variables) > 0
    assert all(hasattr(v, "id") for v in variables)
    assert all(hasattr(v, "scope") for v in variables)


def test_load_variables_reference_missing_file():
    """Verify error when variables_reference.yaml missing."""
    config_dir = Path("configs/nonexistent")

    with pytest.raises(FileNotFoundError):
        load_variables_reference_config(config_dir)
```

**File**: `tests/test_townlet/unit/environment/test_load_global_actions.py`

```python
"""Test load_global_actions_config function."""
import pytest
from townlet.environment.action_config import load_global_actions_config


def test_load_global_actions():
    """Verify loading global_actions.yaml."""
    actions_config = load_global_actions_config()

    assert len(actions_config.actions) > 0
    assert all(hasattr(a, "id") for a in actions_config.actions)
    assert all(hasattr(a, "name") for a in actions_config.actions)


def test_global_actions_has_substrate_and_custom():
    """Verify both substrate and custom actions loaded."""
    actions_config = load_global_actions_config()

    # Check for substrate actions (e.g., UP, DOWN, LEFT, RIGHT, INTERACT, WAIT)
    substrate_names = {a.name for a in actions_config.actions if a.type == "movement"}
    assert len(substrate_names) > 0

    # Check for custom actions (e.g., REST, MEDITATE)
    custom_actions = [a for a in actions_config.actions if a.source == "custom"]
    assert len(custom_actions) >= 2  # REST, MEDITATE
```

4. **Update verification script** (30min)

Update `scripts/verify_compiler_dependencies.py` to use new function names:

```python
def verify_load_functions():
    """Verify all load functions exist and have correct signatures."""
    print("\nVerifying load functions...")

    try:
        from townlet.config.training import load_training_config
        from townlet.config.bar import load_bars_config
        from townlet.config.cascade import load_cascades_config
        from townlet.config.affordance import load_affordances_config
        from townlet.config.cue import load_cues_config
        from townlet.substrate.config import load_substrate_config
        from townlet.environment.action_config import load_global_actions_config  # ✅ NEW
        from townlet.vfs.schema import load_variables_reference_config  # ✅ NEW

        print("  ✅ All load functions imported successfully")

        # Test with actual config
        test_config = Path("configs/L0_0_minimal")
        if test_config.exists():
            print(f"\n  Testing load functions with {test_config}...")

            variables = load_variables_reference_config(test_config)
            print(f"    ✅ load_variables_reference_config returned {len(variables)} variables")

            actions = load_global_actions_config()
            print(f"    ✅ load_global_actions_config returned {len(actions.actions)} actions")

        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
```

**Success Criteria**:
- [ ] `load_variables_reference_config()` exists and loads variables
- [ ] `load_global_actions_config()` exists and loads actions
- [ ] Tests pass for both functions
- [ ] Verification script updated and passes

---

### Part 2: DTO Consolidation (2-3 hours)

**Decision**: Keep DTOs in current locations, create type aliases in `townlet.config.*` for compatibility

**Why**: Moving would require:
- Refactoring all imports across codebase
- Risk of breaking existing code
- Type aliases provide compatibility layer with minimal changes

**Actions**:

1. **Create collection type aliases** (1 hour)

**File**: `src/townlet/config/bar.py`

```python
# Add at end of file after load_bars_config():

# Type alias for collection (compatibility with legacy code)
from townlet.environment.cascade_config import BarsConfig

__all__ = ["BarConfig", "load_bars_config", "BarsConfig"]
```

**File**: `src/townlet/config/cascade.py`

```python
# Add at end of file:

from townlet.environment.cascade_config import CascadesConfig

__all__ = ["CascadeConfig", "load_cascades_config", "CascadesConfig"]
```

**File**: `src/townlet/config/affordance.py`

```python
# Add at end of file:

from townlet.environment.affordance_config import AffordanceConfigCollection

__all__ = ["AffordanceConfig", "load_affordances_config", "AffordanceConfigCollection"]
```

2. **Create ActionSpaceConfig wrapper** (1 hour)

**File**: `src/townlet/environment/action_config.py`

```python
# Add after existing ActionConfig:

from pydantic import BaseModel
from typing import List

class ActionSpaceConfig(BaseModel):
    """Collection of all actions (substrate + custom).

    Composed by ActionSpaceBuilder from substrate actions + custom actions.

    Attributes:
        actions: All actions in canonical order
    """
    actions: List[ActionConfig]

    def get_action_by_id(self, action_id: int) -> ActionConfig:
        """Lookup action by ID."""
        for action in self.actions:
            if action.id == action_id:
                return action
        raise KeyError(f"Action ID {action_id} not found")

    def get_action_count(self) -> int:
        """Total action count."""
        return len(self.actions)
```

3. **Add tests** (1 hour)

**File**: `tests/test_townlet/unit/config/test_dto_aliases.py`

```python
"""Test DTO collection type aliases."""
import pytest


def test_bars_config_alias():
    """Verify BarsConfig importable from townlet.config.bar."""
    from townlet.config.bar import BarsConfig
    from townlet.environment.cascade_config import BarsConfig as OriginalBarsConfig

    assert BarsConfig is OriginalBarsConfig


def test_cascades_config_alias():
    """Verify CascadesConfig importable from townlet.config.cascade."""
    from townlet.config.cascade import CascadesConfig
    from townlet.environment.cascade_config import CascadesConfig as OriginalCascadesConfig

    assert CascadesConfig is OriginalCascadesConfig


def test_affordance_collection_alias():
    """Verify AffordanceConfigCollection importable from townlet.config.affordance."""
    from townlet.config.affordance import AffordanceConfigCollection
    from townlet.environment.affordance_config import AffordanceConfigCollection as Original

    assert AffordanceConfigCollection is Original


def test_action_space_config_creation():
    """Verify ActionSpaceConfig can be created."""
    from townlet.environment.action_config import ActionConfig, ActionSpaceConfig

    actions = [
        ActionConfig(id=0, name="UP", type="movement", delta=[0, -1], costs=[]),
        ActionConfig(id=1, name="DOWN", type="movement", delta=[0, 1], costs=[]),
    ]

    config = ActionSpaceConfig(actions=actions)

    assert len(config.actions) == 2
    assert config.get_action_count() == 2
    assert config.get_action_by_id(0).name == "UP"
```

**Success Criteria**:
- [ ] Type aliases created in `townlet.config.*` modules
- [ ] `ActionSpaceConfig` wrapper created
- [ ] Tests pass proving aliases work
- [ ] Imports work: `from townlet.config.bar import BarsConfig`

---

### Part 3: ObservationSpec Adapter (2-3 hours)

**Decision**: Create adapter to convert VFS ObservationField → Universe DTO ObservationField

**Actions**:

1. **Create VFS adapter module** (1.5 hours)

**File**: `src/townlet/universe/adapters/__init__.py`

```python
"""Adapters for converting between different DTO representations."""

from townlet.universe.adapters.vfs_adapter import vfs_to_universe_observation_spec

__all__ = ["vfs_to_universe_observation_spec"]
```

**File**: `src/townlet/universe/adapters/vfs_adapter.py`

```python
"""Adapter to convert VFS ObservationField to Universe DTO ObservationSpec.

Bridges the gap between:
- VFS builder output (list of vfs.ObservationField with shapes)
- Universe DTO (universe.dto.ObservationField with flattened dims + indices)
"""

from typing import Literal
from townlet.vfs.schema import ObservationField as VFSObservationField
from townlet.universe.dto.observation_spec import ObservationField as UniverseObservationField
from townlet.universe.dto.observation_spec import ObservationSpec as UniverseObservationSpec


def _infer_semantic_type(field_id: str, source_variable: str) -> str:
    """Infer semantic type from field ID and source variable.

    Args:
        field_id: Field identifier (e.g., "obs_position", "obs_energy")
        source_variable: Source variable name

    Returns:
        Semantic type string (e.g., "position", "meter", "affordance", "temporal")
    """
    # Heuristic mapping based on naming conventions
    if "position" in field_id.lower():
        return "position"
    elif "energy" in source_variable or "health" in source_variable or \
         "satiation" in source_variable or "money" in source_variable or \
         "mood" in source_variable or "social" in source_variable or \
         "fitness" in source_variable or "hygiene" in source_variable:
        return "meter"
    elif "affordance" in field_id.lower():
        return "affordance"
    elif "time" in field_id.lower() or "temporal" in field_id.lower():
        return "temporal"
    else:
        return "unknown"


def _flatten_shape(shape: list[int]) -> int:
    """Flatten VFS shape to single dimension count.

    Args:
        shape: VFS shape (e.g., [2], [8], [5, 5])

    Returns:
        Total dimensions (product of shape)

    Examples:
        >>> _flatten_shape([2])
        2
        >>> _flatten_shape([8])
        8
        >>> _flatten_shape([5, 5])
        25
    """
    dims = 1
    for dim in shape:
        dims *= dim
    return dims


def _infer_field_type(dims: int, categorical_labels: tuple[str, ...] | None) -> Literal["scalar", "vector", "categorical", "spatial_grid"]:
    """Infer field type from dimensions and labels.

    Args:
        dims: Number of dimensions
        categorical_labels: Optional categorical labels

    Returns:
        Field type
    """
    if categorical_labels:
        return "categorical"
    elif dims == 1:
        return "scalar"
    elif dims > 9:  # Likely spatial grid (e.g., 5x5 = 25)
        return "spatial_grid"
    else:
        return "vector"


def vfs_to_universe_observation_spec(
    vfs_fields: list[VFSObservationField]
) -> UniverseObservationSpec:
    """Convert VFS ObservationField list to Universe ObservationSpec.

    Performs:
    1. Flattens shapes → dims
    2. Computes contiguous start_index, end_index
    3. Infers semantic_type from field names
    4. Wraps in UniverseObservationSpec

    Args:
        vfs_fields: List of VFS ObservationField instances

    Returns:
        UniverseObservationSpec with flattened fields and indices

    Example:
        >>> vfs_fields = [
        ...     VFSObservationField(id="obs_position", source_variable="position", shape=[2], ...),
        ...     VFSObservationField(id="obs_energy", source_variable="energy", shape=[1], ...),
        ... ]
        >>> spec = vfs_to_universe_observation_spec(vfs_fields)
        >>> spec.total_dims
        3
        >>> spec.fields[0].start_index
        0
        >>> spec.fields[0].end_index
        2
    """
    universe_fields = []
    current_index = 0

    for vfs_field in vfs_fields:
        # Flatten shape to dims
        dims = _flatten_shape(vfs_field.shape)

        # Compute indices
        start_index = current_index
        end_index = current_index + dims

        # Infer semantic type
        semantic_type = _infer_semantic_type(vfs_field.id, vfs_field.source_variable)

        # Infer field type
        field_type = _infer_field_type(dims, None)  # VFS doesn't have categorical labels

        # Create Universe DTO field
        universe_field = UniverseObservationField(
            name=vfs_field.id,
            type=field_type,
            dims=dims,
            start_index=start_index,
            end_index=end_index,
            scope=vfs_field.exposed_to[0] if vfs_field.exposed_to else "agent",  # Use first exposed_to
            description=f"Observation field from variable {vfs_field.source_variable}",
            semantic_type=semantic_type,
        )

        universe_fields.append(universe_field)
        current_index = end_index

    # Build ObservationSpec
    return UniverseObservationSpec.from_fields(universe_fields)
```

2. **Add tests** (1 hour)

**File**: `tests/test_townlet/unit/universe/adapters/test_vfs_adapter.py`

```python
"""Test VFS to Universe DTO adapter."""
import pytest
from townlet.vfs.schema import ObservationField as VFSObservationField
from townlet.universe.adapters.vfs_adapter import (
    vfs_to_universe_observation_spec,
    _flatten_shape,
    _infer_semantic_type,
)


def test_flatten_shape_scalar():
    """Verify flattening scalar shape."""
    assert _flatten_shape([1]) == 1


def test_flatten_shape_vector():
    """Verify flattening vector shape."""
    assert _flatten_shape([8]) == 8


def test_flatten_shape_grid():
    """Verify flattening 2D grid shape."""
    assert _flatten_shape([5, 5]) == 25


def test_infer_semantic_type_position():
    """Verify position semantic type inference."""
    assert _infer_semantic_type("obs_position", "position") == "position"


def test_infer_semantic_type_meter():
    """Verify meter semantic type inference."""
    assert _infer_semantic_type("obs_energy", "energy") == "meter"
    assert _infer_semantic_type("obs_health", "health") == "meter"


def test_vfs_to_universe_conversion():
    """Verify full VFS → Universe conversion."""
    # Create VFS fields
    vfs_fields = [
        VFSObservationField(
            id="obs_position",
            source_variable="position",
            exposed_to=["agent"],
            shape=[2],
            normalization=None,
        ),
        VFSObservationField(
            id="obs_energy",
            source_variable="energy",
            exposed_to=["agent"],
            shape=[1],
            normalization=None,
        ),
    ]

    # Convert
    spec = vfs_to_universe_observation_spec(vfs_fields)

    # Verify spec
    assert spec.total_dims == 3  # 2 + 1
    assert len(spec.fields) == 2

    # Verify first field
    pos_field = spec.fields[0]
    assert pos_field.name == "obs_position"
    assert pos_field.dims == 2
    assert pos_field.start_index == 0
    assert pos_field.end_index == 2
    assert pos_field.semantic_type == "position"

    # Verify second field
    energy_field = spec.fields[1]
    assert energy_field.name == "obs_energy"
    assert energy_field.dims == 1
    assert energy_field.start_index == 2
    assert energy_field.end_index == 3
    assert energy_field.semantic_type == "meter"


def test_vfs_to_universe_with_grid():
    """Verify conversion with spatial grid field."""
    vfs_fields = [
        VFSObservationField(
            id="local_grid",
            source_variable="local_affordances",
            exposed_to=["agent"],
            shape=[5, 5],  # 2D grid
            normalization=None,
        ),
    ]

    spec = vfs_to_universe_observation_spec(vfs_fields)

    assert spec.total_dims == 25
    assert spec.fields[0].dims == 25
    assert spec.fields[0].type == "spatial_grid"
```

**Success Criteria**:
- [ ] Adapter module created
- [ ] Conversion function handles scalar, vector, grid shapes
- [ ] Semantic type inference works for position, meter, affordance, temporal
- [ ] Tests pass with 100% coverage

---

### Part 4: HamletConfig Integration Strategy (1-2 hours)

**Decision**: Compiler builds on top of HamletConfig (reuses existing loader + validation)

**Why**:
- Avoids duplicate validation logic
- Leverages existing cross-config checks
- Maintains backward compatibility with DemoRunner
- Compiler adds additional compile-time checks on top

**Actions**:

1. **Document integration strategy** (1 hour)

**File**: `docs/architecture/COMPILER_HAMLETCONFIG_INTEGRATION.md`

```markdown
# Universe Compiler + HamletConfig Integration

## Strategy: Compiler Builds on HamletConfig

The Universe Compiler **uses HamletConfig as input**, not a replacement.

### Data Flow

```
YAML Files → HamletConfig.load() → Validation → HamletConfig Instance
                                                        ↓
                                              UniverseCompiler.compile()
                                                        ↓
                                         Additional Validation + Metadata
                                                        ↓
                                                CompiledUniverse
```

### Responsibilities

**HamletConfig.load()** (existing):
- Load all 9 YAML files
- Basic structural validation (Pydantic schemas)
- Cross-config validation:
  - batch_size ≤ replay_buffer_capacity
  - network_type matches observability mode (warning)
  - grid capacity check (warning)

**UniverseCompiler.compile()** (new):
- Accept HamletConfig as input
- Additional compile-time validation:
  - Circular cascade detection (DFS)
  - Cue range coverage validation
  - Cross-file reference resolution (affordances → meters)
  - VFS observation spec generation
- Generate rich metadata:
  - ObservationSpec for BAC
  - ActionSpaceMetadata, MeterMetadata, AffordanceMetadata
- Compute config hash for checkpoint compatibility
- Cache compiled universe (MessagePack)

### API

```python
# Load config using existing HamletConfig
from townlet.config.hamlet import HamletConfig

hamlet_config = HamletConfig.load(Path("configs/L0_0_minimal"))

# Compile into immutable universe
from townlet.universe.compiler import UniverseCompiler

compiler = UniverseCompiler()
universe = compiler.compile(hamlet_config)  # ✅ Takes HamletConfig, not Path

# Use compiled universe
env = universe.create_environment(num_agents=100, device="cuda")
```

### Migration Path

**Phase 1** (current): DemoRunner uses HamletConfig directly
```python
class DemoRunner:
    def __init__(self, config_dir: Path):
        self.hamlet_config = HamletConfig.load(config_dir)
        # Use hamlet_config directly
```

**Phase 2** (with compiler): DemoRunner compiles before use
```python
class DemoRunner:
    def __init__(self, config_dir: Path, use_compiler: bool = False):
        self.hamlet_config = HamletConfig.load(config_dir)

        if use_compiler:
            compiler = UniverseCompiler()
            self.universe = compiler.compile(self.hamlet_config)
            # Use compiled universe
        else:
            # Legacy path (still works)
            self.universe = None
```

**Phase 3** (full adoption): Compiler always used
```python
class DemoRunner:
    def __init__(self, config_dir: Path):
        hamlet_config = HamletConfig.load(config_dir)
        compiler = UniverseCompiler()
        self.universe = compiler.compile(hamlet_config)
```

### Benefits

✅ **No duplication**: Compiler builds on HamletConfig validation, not parallel to it
✅ **Backward compatible**: Existing code continues to work
✅ **Incremental adoption**: Can opt-in to compiler gradually
✅ **Clear responsibilities**: HamletConfig = load + basic validation, Compiler = advanced validation + metadata
✅ **Single source of truth**: HamletConfig remains authoritative for config loading

### Validation Layering

```
Layer 1 (Pydantic): Structural validation
  ↓
Layer 2 (HamletConfig): Cross-config validation
  ↓
Layer 3 (Compiler): Compile-time validation + metadata generation
```

Each layer builds on the previous, no replacement.
```

2. **Update RawConfigs to use HamletConfig** (30min)

Update plan to use:

```python
@dataclass
class RawConfigs:
    """Container for raw config objects loaded from YAML.

    Built from HamletConfig - reuses existing loader.
    """
    hamlet_config: HamletConfig  # Source of all configs

    @property
    def substrate(self) -> SubstrateConfig:
        return self.hamlet_config.substrate

    @property
    def bars(self) -> tuple[BarConfig, ...]:
        return self.hamlet_config.bars

    @property
    def cascades(self) -> tuple[CascadeConfig, ...]:
        return self.hamlet_config.cascades

    # ... etc for all configs
```

**Success Criteria**:
- [ ] Integration strategy documented
- [ ] RawConfigs updated to use HamletConfig
- [ ] Migration path clear (3 phases)

---

### Part 5: Update TASK-004A Specification (1-2 hours)

**Actions**:

1. **Update Stage 1 to use actual filenames** (30min)

```python
def _stage_1_parse_individual_files(self, config_dir: Path) -> RawConfigs:
    """Stage 1: Load configs using HamletConfig.

    Leverages existing HamletConfig.load() instead of reimplementing.
    """
    hamlet_config = HamletConfig.load(config_dir)

    # Load variables (not in HamletConfig yet)
    variables = load_variables_reference_config(config_dir)

    # Load global actions
    actions = load_global_actions_config()

    return RawConfigs(
        hamlet_config=hamlet_config,
        variables=variables,
        actions=actions,
    )
```

2. **Update imports in plan** (30min)

Change all:
```python
from townlet.config.bar import BarsConfig  # ✅ NOW WORKS (type alias)
from townlet.config.cascade import CascadesConfig  # ✅ NOW WORKS
from townlet.config.affordance import AffordanceConfigCollection  # ✅ NOW WORKS
from townlet.environment.action_config import ActionSpaceConfig  # ✅ NOW WORKS
from townlet.vfs.schema import load_variables_reference_config  # ✅ NOW WORKS
```

3. **Add VFS adapter to Phase 7** (30min)

Update Phase 7 (Compute Metadata) to include:

```python
# Stage 5: Compute metadata
def _stage_5_compute_metadata(...):
    # ... existing code ...

    # Build observation spec from VFS
    from townlet.vfs.observation_builder import VFSObservationSpecBuilder
    from townlet.universe.adapters.vfs_adapter import vfs_to_universe_observation_spec

    vfs_builder = VFSObservationSpecBuilder()
    vfs_fields = vfs_builder.build_observation_spec(
        variables=raw_configs.variables,
        exposures=...,  # From variables_reference.yaml
    )

    # Convert VFS → Universe DTO
    observation_spec = vfs_to_universe_observation_spec(vfs_fields)

    obs_dim = observation_spec.total_dims

    # ... rest of metadata computation ...
```

4. **Update effort estimate for VectorizedHamletEnv refactor** (15min)

Change in plan:
```markdown
### Phase 8: Stage 6/7 Optimize & Emit (6-10h)

**Note**: This phase does NOT include VectorizedHamletEnv refactor.
Environment integration deferred to separate follow-up task (est. 20-30h).

Compiler delivers `universe.create_environment()` helper stub that
forwards to existing VectorizedHamletEnv constructor for now.
```

**Success Criteria**:
- [ ] TASK-004A plan updated with correct imports
- [ ] Stage 1 uses HamletConfig.load()
- [ ] Phase 7 includes VFS adapter
- [ ] VectorizedHamletEnv refactor effort honest (20-30h, separate task)

---

## Testing Strategy

### Unit Tests

- `test_load_variables_reference.py` - Test variables loader
- `test_load_global_actions.py` - Test actions loader
- `test_dto_aliases.py` - Test type aliases work
- `test_vfs_adapter.py` - Test VFS → Universe DTO conversion

### Integration Tests

- `test_compiler_with_hamlet_config.py` - Test compiler accepts HamletConfig
- `test_verification_script.py` - Test verification script passes

### Validation

Run updated verification script:

```bash
uv run python scripts/verify_compiler_dependencies.py

# Expected output:
# ============================================================
# TASK-004A Pre-Implementation Dependency Verification
# ============================================================
#
# Verifying DTO files...
#   ✅ All DTO files exist
#
# Verifying load functions...
#   ✅ All load functions imported successfully
#
#   Testing load functions with configs/L0_0_minimal...
#     ✅ load_variables_reference_config returned 12 variables
#     ✅ load_global_actions_config returned 8 actions
#
# Verifying VFS builder...
#   ✅ VFSObservationSpecBuilder imported successfully
#   ✅ VFSObservationSpecBuilder.build_observation_spec exists
#   ✅ ObservationField class imported successfully
#
# ============================================================
# VERIFICATION SUMMARY
# ============================================================
# DTO Files: ✅ PASS
# Load Functions: ✅ PASS
# VFS Builder: ✅ PASS
#
# 🎉 All dependencies verified - ready for implementation!
```

---

## Acceptance Criteria

**This task is complete when:**

✅ **Part 1**: Config schema alignment
- [ ] `load_variables_reference_config()` implemented and tested
- [ ] `load_global_actions_config()` implemented and tested
- [ ] Verification script updated and passes

✅ **Part 2**: DTO consolidation
- [ ] Type aliases created in `townlet.config.*`
- [ ] `ActionSpaceConfig` wrapper created
- [ ] All imports work correctly

✅ **Part 3**: ObservationSpec adapter
- [ ] VFS adapter module created
- [ ] Conversion function tested with scalar/vector/grid
- [ ] Semantic type inference works

✅ **Part 4**: HamletConfig integration
- [ ] Integration strategy documented
- [ ] Migration path defined (3 phases)
- [ ] RawConfigs updated to use HamletConfig

✅ **Part 5**: TASK-004A updates
- [ ] Plan updated with correct imports
- [ ] Stage 1 uses HamletConfig
- [ ] VFS adapter integrated into Phase 7
- [ ] Effort estimate honest about VectorizedHamletEnv

**Gate**: TASK-004A implementation can ONLY start after ALL 5 parts complete ✅

---

## Implementation Order

**Day 1** (4-5 hours):
1. Part 1: Config schema alignment (3-4h)
2. Part 2: DTO consolidation (start, 1h)

**Day 2** (4-5 hours):
1. Part 2: DTO consolidation (finish, 1-2h)
2. Part 3: ObservationSpec adapter (2-3h)

**Day 3** (2-3 hours):
1. Part 4: HamletConfig integration strategy (1-2h)
2. Part 5: Update TASK-004A spec (1-2h)
3. Run full verification suite

**Total**: 8-12 hours over 3 days

---

## Related Documents

- **[TASK-004A-COMPILER-IMPLEMENTATION.md](TASK-004A-COMPILER-IMPLEMENTATION.md)** - Main compiler implementation (blocked until this complete)
- **[COMPILER_ARCHITECTURE.md](../architecture/COMPILER_ARCHITECTURE.md)** - Authoritative architecture spec
- **[vfs-integration-guide.md](../vfs-integration-guide.md)** - VFS integration patterns

---

## Commit Strategy

**Incremental commits per part**:

1. `feat(vfs): Add load_variables_reference_config and load_global_actions_config`
2. `feat(config): Add DTO collection type aliases for compiler compatibility`
3. `feat(universe): Add VFS to Universe DTO adapter`
4. `docs(compiler): Document HamletConfig integration strategy`
5. `docs(task-004a): Update plan to match repository reality`

Each part is independently useful and testable.
