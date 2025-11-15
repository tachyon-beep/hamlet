# Config v2.1 Implementation Plan
## Master Implementation Plan for ENH-28

**Status**: Ready to Execute
**Branch**: feature/config-v2.1
**Estimated Duration**: 1-2 days focused work
**Owner**: compiler + config subsystems

---

## Overview

**Goal**: Migrate from flat config structure to v2.1 hierarchical structure without breaking the project

**Strategy**: Clean-break migration on feature branch with test-driven validation

**Success Criteria**:
- All 5 curriculum levels load with v2.1 structure
- All 144 tests passing
- obs_dim unchanged (L1=L2=121 for transfer learning)
- No old config code remains
- Reference config matches actual configs

**Development Approach**:
- Break it deliberately, fix it systematically
- Test-driven: let failures guide us
- One reference level (L1) to validate pattern
- Mechanical replication for remaining levels
- No backwards compatibility needed (zero users)

---

## Phase 1: Setup & Safety Net (30 minutes)

### Objectives
- Create feature branch
- Archive old configs
- Establish baseline
- Create implementation checklist

### Activities

**1.1 Create feature branch**
```bash
git checkout -b feature/config-v2.1
```

**1.2 Archive old configs**
```bash
mkdir configs_archive
git mv configs/* configs_archive/
git commit -m "archive: move old flat configs to archive"
```

**1.3 Capture baseline test results**
```bash
pytest tests/test_townlet/ -q --tb=no | tee docs/bugs/BUNDLE-01-curriculum-observation-architecture/baseline-test-results.txt
```

**1.4 Create implementation checklist**

See `implementation-checklist.md` in this bundle.

### Deliverables
- [x] Branch: `feature/config-v2.1`
- [x] `configs_archive/` with old flat configs
- [x] `BUNDLE-01/baseline-test-results.txt` (144 passing)
- [x] `BUNDLE-01/implementation-checklist.md`

### Success Criteria
- Baseline captured: 144 tests passing
- Old configs safely archived
- Clean slate in `configs/`

---

## Phase 2: Create Model Config from Template (1 hour)

### Objectives
- Extract clean v2.1 configs from reference template
- Create one complete working example (L1)

### Activities

**2.1 Create directory structure**
```bash
mkdir -p configs/default_curriculum/levels/L1_full_observability
```

**2.2 Extract experiment-level configs**

From `reference-config-v2.1-complete.yaml`, extract sections:

1. **experiment.yaml** (lines 36-49, Section 1)
   - Version, metadata, curriculum_levels list

2. **stratum.yaml** (lines 56-133, Section 2)
   - Substrate definition (Grid2D 8×8)
   - Behavioral params (boundary, distance_metric, observation_encoding)
   - vision_support: both
   - temporal_support: enabled

3. **environment.yaml** (lines 140-333, Section 3)
   - 8 meters with range_type metadata
   - cascade_graph structure
   - modulation_graph structure
   - 14 affordances
   - 4 VFS variables

4. **actions.yaml** (lines 340-382, Section 4)
   - substrate_actions: inherit true
   - custom_actions: INTERACT, WAIT, REST, MEDITATE
   - labels: gaming

5. **agent.yaml** (lines 389-531, Section 5)
   - perception settings
   - drive (DAC configuration)
   - brain (feedforward, optimizer, q_learning defaults)

**2.3 Extract curriculum-level configs**

From `reference-config-v2.1-complete.yaml`, extract sections:

6. **curriculum.yaml** (lines 538-601, use L1 example)
   - active_vision: global
   - active_temporal: false
   - vision_range: 0.5 (not used but required)
   - day_length: null

7. **bars.yaml** (lines 608-697, Section 7)
   - 8 meter parameters
   - cascade parameters

8. **affordances.yaml** (lines 704-806, Section 8)
   - 14 affordance parameters
   - modulation parameters

9. **training.yaml** (lines 813-898, Section 9)
   - population, q_learning, replay_buffer, exploration settings

**2.4 Clean up extracted files**
- Remove all comments (keep only actual config)
- Verify all required fields present
- Ensure proper YAML formatting

### Deliverables
- [x] `configs/default_curriculum/experiment.yaml`
- [x] `configs/default_curriculum/stratum.yaml`
- [x] `configs/default_curriculum/environment.yaml`
- [x] `configs/default_curriculum/actions.yaml`
- [x] `configs/default_curriculum/agent.yaml`
- [x] `configs/default_curriculum/levels/L1_full_observability/curriculum.yaml`
- [x] `configs/default_curriculum/levels/L1_full_observability/bars.yaml`
- [x] `configs/default_curriculum/levels/L1_full_observability/affordances.yaml`
- [x] `configs/default_curriculum/levels/L1_full_observability/training.yaml`

### Success Criteria
- 9 clean YAML files created
- All match reference config structure
- Valid YAML syntax (verify with yamllint if available)

---

## Phase 3: DTO Creation (2-3 hours)

### Objectives
- Create Pydantic DTOs matching v2.1 structure
- Enforce no-defaults principle
- Add cross-field validation

### Activities

**3.1 Create experiment_config.py**

```python
from pydantic import BaseModel, Field
from typing import List

class ExperimentMetadata(BaseModel):
    name: str
    description: str
    author: str
    created: str
    tags: List[str] = Field(default_factory=list)

class ExperimentConfig(BaseModel):
    version: str
    metadata: ExperimentMetadata
    curriculum_levels: List[str]

    class Config:
        extra = "forbid"  # Strict: fail on unknown fields
```

**3.2 Create stratum_config.py**

```python
from pydantic import BaseModel
from typing import Literal

class GridConfig(BaseModel):
    topology: Literal["square", "hexagonal"]
    width: int
    height: int
    boundary: Literal["clamp", "wrap", "bounce", "sticky"]
    distance_metric: Literal["manhattan", "euclidean", "chebyshev"]
    observation_encoding: Literal["relative", "scaled", "absolute"]

    class Config:
        extra = "forbid"

class SubstrateConfig(BaseModel):
    type: Literal["grid", "grid3d", "gridnd", "continuous", "continuousnd", "aspatial"]
    grid: GridConfig  # Only for grid types

    class Config:
        extra = "forbid"

class StratumConfig(BaseModel):
    version: str
    substrate: SubstrateConfig
    vision_support: Literal["global", "partial", "both", "none"]
    temporal_support: Literal["enabled", "disabled"]

    class Config:
        extra = "forbid"
```

**3.3 Create environment_config.py**

```python
from pydantic import BaseModel
from typing import List, Literal, Optional

class MeterDefinition(BaseModel):
    name: str
    description: str
    range_type: Literal["normalized", "unbounded", "integer"]

    class Config:
        extra = "forbid"

class CascadeGraphEdge(BaseModel):
    source: str
    target: str
    description: str

    class Config:
        extra = "forbid"

class ModulationGraphEdge(BaseModel):
    bar: str
    affordances: List[str]
    description: str

    class Config:
        extra = "forbid"

class AffordanceDefinition(BaseModel):
    name: str
    description: str
    category: str

    class Config:
        extra = "forbid"

class VFSNormalizationSpec(BaseModel):
    method: Literal["clip", "normalize", "standardize", "none"]
    range: List[float]  # [min, max]

    class Config:
        extra = "forbid"

class VFSVariableDefinition(BaseModel):
    name: str
    type: Literal["scalar", "vector"]
    dims: int
    scope: Literal["global", "agent", "agent_private"]
    description: str
    normalization: VFSNormalizationSpec

    class Config:
        extra = "forbid"

class CueDefinition(BaseModel):
    # TODO: Define cue structure
    pass

class EnvironmentConfig(BaseModel):
    version: str
    meters: List[MeterDefinition]
    cascade_graph: List[CascadeGraphEdge]
    modulation_graph: List[ModulationGraphEdge]
    affordances: List[AffordanceDefinition]
    variables: List[VFSVariableDefinition]
    cues: Optional[List[CueDefinition]] = None

    class Config:
        extra = "forbid"
```

**3.4 Create actions_config.py**

```python
from pydantic import BaseModel
from typing import List, Literal

class CustomActionDefinition(BaseModel):
    name: str
    description: str
    enabled_by_default: bool

    class Config:
        extra = "forbid"

class SubstrateActionsConfig(BaseModel):
    inherit: bool

    class Config:
        extra = "forbid"

class ActionsConfig(BaseModel):
    version: str
    substrate_actions: SubstrateActionsConfig
    custom_actions: List[CustomActionDefinition]
    labels: Literal["gaming", "6dof", "cardinal", "math"]

    class Config:
        extra = "forbid"
```

**3.5 Create agent_config.py**

```python
from pydantic import BaseModel
from typing import List, Literal, Optional, Dict, Any

class DriveConfig(BaseModel):
    # TODO: Define DAC structure matching drive_as_code.yaml
    pass

class FeedforwardConfig(BaseModel):
    hidden_sizes: List[int]
    activation: Literal["relu", "tanh", "elu", "gelu"]

    class Config:
        extra = "forbid"

class RecurrentConfig(BaseModel):
    # TODO: Define recurrent network structure
    pass

class OptimizerConfig(BaseModel):
    type: Literal["adam", "sgd", "rmsprop"]
    learning_rate: float

    class Config:
        extra = "forbid"

class QLearningConfig(BaseModel):
    algorithm: Literal["dqn", "double_dqn"]
    gamma: float
    target_update_frequency: int

    class Config:
        extra = "forbid"

class BrainConfig(BaseModel):
    architecture: Literal["feedforward", "recurrent"]
    feedforward: Optional[FeedforwardConfig] = None
    recurrent: Optional[RecurrentConfig] = None
    optimizer: OptimizerConfig
    q_learning: QLearningConfig

    class Config:
        extra = "forbid"

class AgentConfig(BaseModel):
    version: str
    drive: DriveConfig
    brain: BrainConfig

    class Config:
        extra = "forbid"
```

**3.6 Create curriculum_config.py**

```python
from pydantic import BaseModel, validator
from typing import Literal, Optional

class CurriculumConfig(BaseModel):
    version: str
    active_vision: Literal["global", "partial"]
    active_temporal: bool
    vision_range: float  # Normalized [0.0, 1.0]
    day_length: Optional[int]  # Required when active_temporal=true, null otherwise

    @validator('vision_range')
    def validate_vision_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("vision_range must be in [0.0, 1.0]")
        return v

    @validator('day_length')
    def validate_day_length(cls, v, values):
        active_temporal = values.get('active_temporal')
        if active_temporal and v is None:
            raise ValueError("day_length required when active_temporal=true")
        if not active_temporal and v is not None:
            raise ValueError("day_length must be null when active_temporal=false")
        return v

    class Config:
        extra = "forbid"
```

### Deliverables
- [x] `src/townlet/config/experiment_config.py`
- [x] `src/townlet/config/stratum_config.py`
- [x] `src/townlet/config/environment_config.py`
- [x] `src/townlet/config/actions_config.py`
- [x] `src/townlet/config/agent_config.py`
- [x] `src/townlet/config/curriculum_config.py`

### Success Criteria
- All DTOs compile without errors
- Pydantic validation works (test with sample data)
- No-defaults principle enforced
- Cross-field validators working

---

## Phase 4: Compiler Updates (2-3 hours)

### Objectives
- Replace flat-config loading with hierarchical
- Implement cross-curriculum vocabulary validation
- Update observation spec for support/active pattern
- Delete all old config code

### Activities

**4.1 Update compiler.py main entry point**

Replace existing `compile()` function:

```python
def compile(experiment_dir: Path) -> CompiledUniverse:
    """
    Compile v2.1 hierarchical config structure.

    Args:
        experiment_dir: Path to experiment directory (e.g., configs/default_curriculum/)

    Returns:
        CompiledUniverse with all levels compiled
    """
    # Stage 1: Load hierarchical structure
    experiment, stratum, environment, actions, agent, levels = _load_hierarchical_configs(experiment_dir)

    # Stage 2: Cross-curriculum vocabulary validation
    _validate_vocabulary_consistency(environment, levels)

    # Stage 3-4: Existing stages (symbol table, resolution)
    # ...

    # Stage 5: Build observation spec with support/active pattern
    for level_name, level_config in levels.items():
        obs_spec = _build_observation_spec(stratum, environment, level_config["curriculum"])
        # ...
```

**4.2 Implement Stage 1: Load hierarchical structure**

```python
def _load_hierarchical_configs(experiment_dir: Path):
    """Load all v2.1 config files."""

    # Load experiment-level configs (shared across curriculum)
    experiment = ExperimentConfig.from_yaml(experiment_dir / "experiment.yaml")
    stratum = StratumConfig.from_yaml(experiment_dir / "stratum.yaml")
    environment = EnvironmentConfig.from_yaml(experiment_dir / "environment.yaml")
    actions = ActionsConfig.from_yaml(experiment_dir / "actions.yaml")
    agent = AgentConfig.from_yaml(experiment_dir / "agent.yaml")

    # Load curriculum-level configs for each level
    levels = {}
    for level_name in experiment.curriculum_levels:
        level_dir = experiment_dir / "levels" / level_name
        levels[level_name] = {
            "curriculum": CurriculumConfig.from_yaml(level_dir / "curriculum.yaml"),
            "bars": BarsConfig.from_yaml(level_dir / "bars.yaml"),
            "affordances": AffordancesConfig.from_yaml(level_dir / "affordances.yaml"),
            "training": TrainingConfig.from_yaml(level_dir / "training.yaml"),
        }

    return experiment, stratum, environment, actions, agent, levels
```

**4.3 Implement Stage 2: Cross-curriculum vocabulary validation**

```python
def _validate_vocabulary_consistency(environment: EnvironmentConfig, levels: dict):
    """
    Validate all curriculum levels have same vocabulary as environment.yaml.

    STRICT validation:
    - All levels must have exact same meters as environment.meters
    - All levels must have exact same affordances as environment.affordances
    - All levels must implement all cascades from environment.cascade_graph
    """

    env_meter_names = set(m.name for m in environment.meters)
    env_affordance_names = set(a.name for a in environment.affordances)
    env_cascade_edges = set((c.source, c.target) for c in environment.cascade_graph)

    for level_name, level_config in levels.items():
        # Validate meters
        level_meter_names = set(level_config["bars"].meters.keys())
        if level_meter_names != env_meter_names:
            raise ValueError(
                f"Level {level_name} meter vocabulary mismatch.\n"
                f"Environment: {env_meter_names}\n"
                f"Level: {level_meter_names}"
            )

        # Validate affordances
        level_affordance_names = set(level_config["affordances"].affordances.keys())
        if level_affordance_names != env_affordance_names:
            raise ValueError(
                f"Level {level_name} affordance vocabulary mismatch.\n"
                f"Environment: {env_affordance_names}\n"
                f"Level: {level_affordance_names}"
            )

        # Validate cascades (must implement all, can disable with strength: 0.0)
        level_cascade_edges = set(
            (c.source, c.target) for c in level_config["bars"].cascades
        )
        if level_cascade_edges != env_cascade_edges:
            raise ValueError(
                f"Level {level_name} cascade graph mismatch.\n"
                f"Environment: {env_cascade_edges}\n"
                f"Level: {level_cascade_edges}"
            )
```

**4.4 Update Stage 5: Observation spec with support/active pattern**

```python
def _build_observation_spec(
    stratum: StratumConfig,
    environment: EnvironmentConfig,
    curriculum: CurriculumConfig
) -> ObservationSpec:
    """
    Build observation spec using support/active pattern.

    Support (stratum): Which fields CAN exist
    Active (curriculum): Which fields ARE active vs masked
    """

    fields = []

    # Vision fields (support/active pattern)
    if stratum.vision_support in ["both", "global"]:
        grid_dims = stratum.substrate.grid.width * stratum.substrate.grid.height
        fields.append(ObservationField(
            name="obs_grid_encoding",
            dims=grid_dims,
            curriculum_active=(curriculum.active_vision == "global")  # Masking control
        ))

    if stratum.vision_support in ["both", "partial"]:
        # Compute window dimensions from normalized vision_range
        grid_width = stratum.substrate.grid.width
        grid_height = stratum.substrate.grid.height

        radius_x = math.ceil(curriculum.vision_range * (grid_width / 2))
        radius_y = math.ceil(curriculum.vision_range * (grid_height / 2))

        window_width = radius_x * 2 + 1
        window_height = radius_y * 2 + 1
        window_dims = window_width * window_height

        fields.append(ObservationField(
            name="obs_local_window",
            dims=window_dims,
            curriculum_active=(curriculum.active_vision == "partial")
        ))

    # Implicit substrate fields (always present)
    # Grid2D: position (2 dims) + velocity (2 dims)
    fields.append(ObservationField(
        name="obs_agent_position",
        dims=2,  # x, y for Grid2D
        curriculum_active=True  # Always active
    ))

    fields.append(ObservationField(
        name="obs_agent_velocity",
        dims=2,  # dx, dy for Grid2D
        curriculum_active=True  # Always active
    ))

    # Meter fields (always active)
    for meter in environment.meters:
        fields.append(ObservationField(
            name=f"obs_meter_{meter.name}",
            dims=1,
            curriculum_active=True
        ))

    # Affordance fields (binary visibility, always active)
    for affordance in environment.affordances:
        fields.append(ObservationField(
            name=f"obs_affordance_{affordance.name}",
            dims=1,
            curriculum_active=True
        ))

    # VFS variable fields (always active)
    for variable in environment.variables:
        fields.append(ObservationField(
            name=f"obs_vfs_{variable.name}",
            dims=variable.dims,
            curriculum_active=True
        ))

    # Temporal fields (support/active pattern)
    if stratum.temporal_support == "enabled":
        fields.append(ObservationField(
            name="obs_time_sin",
            dims=1,
            curriculum_active=curriculum.active_temporal
        ))
        fields.append(ObservationField(
            name="obs_time_cos",
            dims=1,
            curriculum_active=curriculum.active_temporal
        ))

    return ObservationSpec(fields=fields)
```

**4.5 Delete old config loading code**

Search for and remove:
- Old flat-config DTO imports
- Old Stage 1 loading logic
- Any SubstrateConfig references (replaced by StratumConfig)
- Any backwards compatibility checks

### Deliverables
- [x] Updated `src/townlet/universe/compiler.py`
- [x] New Stage 1: Hierarchical config loading
- [x] New Stage 2: Cross-curriculum vocabulary validation
- [x] Updated Stage 5: Observation spec with support/active pattern
- [x] Deleted: All old flat-config loading code

### Success Criteria
- Compiler compiles without syntax errors
- L1 config loads successfully
- Observation spec includes support/active masking
- Cross-curriculum validation enforces vocabulary consistency

---

## Phase 5: Test Updates (2-3 hours)

### Objectives
- Fix all broken tests
- Update test fixtures to v2.1 structure
- Validate obs_dim unchanged

### Activities

**5.1 Run tests to see what breaks**
```bash
pytest tests/test_townlet/ -x --tb=line
```

**5.2 Update compiler tests**

Update tests in `tests/test_townlet/unit/universe/`:
- Change config paths to v2.1 structure
- Update expectations for new DTOs
- Update observation spec assertions

**5.3 Update config DTO tests**

Update tests in `tests/test_townlet/unit/config/`:
- Test new DTOs (ExperimentConfig, StratumConfig, etc.)
- Test Pydantic validators (vision_range, day_length)
- Test strict validation (extra fields forbidden)

**5.4 Update integration tests**

Update tests in `tests/test_townlet/integration/`:
- Update config fixtures to v2.1 structure
- Validate L1 loads successfully
- Validate obs_dim = 121 for L1

**5.5 Fix tests incrementally**

As tests fail:
1. Read the error message
2. Identify what needs updating (config path, DTO field, etc.)
3. Fix the test
4. Re-run to validate fix
5. Repeat until all tests pass

### Deliverables
- [x] All compiler tests passing
- [x] All config DTO tests passing
- [x] All integration tests passing
- [x] L1 obs_dim validated = 121

### Success Criteria
- All tests passing for L1
- No test skips or xfails
- obs_dim unchanged from baseline

---

## Phase 6: Remaining Levels (1-2 hours)

### Objectives
- Convert L0_0, L0_5, L2, L3 to v2.1
- Mechanical copy of L1 pattern

### Activities

**6.1 Create L0_0_minimal**

Copy L1 pattern, adjust:
- Grid size: 3×3
- Fewer affordances
- Simpler meter dynamics
- curriculum.yaml: active_vision: global, active_temporal: false

**6.2 Create L0_5_dual_resource**

Copy L1 pattern, adjust:
- Grid size: 7×7
- Dual resource mechanics
- curriculum.yaml: active_vision: global, active_temporal: false

**6.3 Create L2_partial_observability**

Copy L1 pattern, adjust:
- curriculum.yaml: active_vision: partial, vision_range: 0.5
- curriculum.yaml: active_temporal: false
- Validate: obs_dim = 121 (same as L1!)

**6.4 Create L3_temporal_mechanics**

Copy L1 pattern, adjust:
- curriculum.yaml: active_vision: global
- curriculum.yaml: active_temporal: true, day_length: 24
- Validate: obs_dim = 121 (same as L1 and L2!)

**6.5 Run tests for each level**

After each level conversion:
```bash
pytest tests/test_townlet/ -q --tb=no
```

### Deliverables
- [x] `configs/default_curriculum/levels/L0_0_minimal/` (4 files)
- [x] `configs/default_curriculum/levels/L0_5_dual_resource/` (4 files)
- [x] `configs/default_curriculum/levels/L2_partial_observability/` (4 files)
- [x] `configs/default_curriculum/levels/L3_temporal_mechanics/` (4 files)

### Success Criteria
- All 5 levels compile successfully
- All 144 tests passing
- L1 = L2 = L3 = 121 obs_dim (transfer learning validated)

---

## Phase 7: Cleanup & Validation (30 minutes)

### Objectives
- Remove archived configs
- Update documentation
- Final test run
- Merge to main

### Activities

**7.1 Remove archived configs**
```bash
rm -rf configs_archive/
git add -A
git commit -m "cleanup: remove archived flat configs"
```

**7.2 Update documentation**

Update `BUNDLE-01/README.md`:
- Change status to: "Complete (v2.1 implemented)"
- Add implementation completion date

Update `ENH-28-experiment-level-configuration-hierarchy.md`:
- Change status to: "implemented"
- Add implementation notes

**7.3 Final validation**

```bash
# All tests passing
pytest tests/test_townlet/ -q --tb=no

# No old config references
grep -r "SubstrateConfig" src/townlet/universe/  # Should find nothing

# Reference config matches
# Manual review: compare reference-config-v2.1-complete.yaml with actual configs
```

**7.4 Update checklist**

Mark all items complete in `implementation-checklist.md`

**7.5 Commit and merge**

```bash
git add -A
git commit -m "feat(config): implement v2.1 hierarchical config system

BREAKING CHANGE: Config structure migrated from flat to hierarchical

- Implements experiment-level hierarchy (experiment, stratum, environment, actions, agent)
- Implements curriculum-level configs (curriculum, bars, affordances, training)
- Implements support/active pattern for observation field control
- All 5 curriculum levels migrated (L0_0, L0_5, L1, L2, L3)
- All 144 tests passing
- obs_dim unchanged (L1=L2=L3=121 for transfer learning)

Implements: ENH-28
Design: BUNDLE-01/target-config-design-v2.md
Reference: BUNDLE-01/reference-config-v2.1-complete.yaml"

git checkout main
git merge feature/config-v2.1
git push
```

### Deliverables
- [x] Deleted: `configs_archive/`
- [x] Updated: BUNDLE-01/README.md
- [x] Updated: ENH-28-experiment-level-configuration-hierarchy.md
- [x] Merged to main

### Success Criteria
- All 144 tests passing on main
- No old config code remains
- Reference config accurate
- Checklist 100% complete
- Clean commit history

---

## Risk Mitigation

### Identified Risks

1. **Breaking unexpected code paths**
   - Mitigation: Let tests scream, fix incrementally
   - Rollback: Feature branch = safe sandbox

2. **Missing config fields**
   - Mitigation: Strict DTO validation catches immediately
   - Rollback: Reference config is ground truth

3. **obs_dim mismatch**
   - Mitigation: Validate after each level conversion
   - Rollback: Compare with baseline obs_dim

4. **Forgotten old code**
   - Mitigation: grep for old DTO names before merge
   - Rollback: Code review catches stray references

### Rollback Strategy

- Feature branch = safe sandbox (can revert entire branch)
- Git revert if catastrophic failure
- Can cherry-pick working phases if partial success
- Baseline test results captured for comparison

---

## Verification Checklist

Before merge to main:

- [ ] All 144 tests passing
- [ ] All 5 levels load successfully
- [ ] L1 = L2 = L3 = 121 obs_dim
- [ ] No old config code references (grep verification)
- [ ] Reference config matches actual configs
- [ ] Implementation checklist 100% complete
- [ ] Documentation updated
- [ ] Clean commit message

---

## Next Steps After Implementation

1. **Create actual experiment configs from archived configs**
   - Extract real values from `configs_archive/`
   - Replace template placeholders in model config
   - Validate against baseline behavior

2. **Write migration script for future config updates**
   - Script to convert flat → hierarchical
   - Useful for external users (when they exist)

3. **Update Universe Compiler documentation**
   - Document new loading process
   - Document support/active pattern
   - Document validation rules

4. **Create config templates**
   - `configs/templates/` with minimal examples
   - Template for each substrate type
   - Template for each observation mode

---

## References

- **Design Spec**: `target-config-design-v2.md`
- **Reference Config**: `reference-config-v2.1-complete.yaml`
- **Design Review**: `archive/design-v2-changes-summary.md`
- **v2.1 Patch**: `archive/target-config-design-v2.1-patch.md`
- **CLAUDE.md**: Pre-release status, zero backwards compatibility required
