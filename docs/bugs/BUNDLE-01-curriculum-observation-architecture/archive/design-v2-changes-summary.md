# Configuration Design v2: Changes Summary
## Response to Code Review Feedback

**Reviewer**: feature-dev:code-reviewer (cold read)
**Design v1**: `target-config-design.md`
**Design v2**: `target-config-design-v2.md`

---

## Critical Issues Fixed

### Issue #1: Observation Structure Contradiction ✅ FIXED

**Reviewer's Finding**:
> The design makes grid_encoding and local_window mutually exclusive, but BUG-43's solution includes BOTH fields (one active, one masked). This would produce L1=96 dims, L2=57 dims, breaking checkpoint transfer.

**v1 Code (BROKEN)**:
```python
if config.agent.perception.partial_observability:
    # Include local_window ONLY
else:
    # Include grid_encoding ONLY
```

**v2 Fix: Support/Active Pattern**:

**Experiment-level** (`stratum.yaml`):
```yaml
vision_support: both  # Creates BOTH fields in obs_dim
```

**Curriculum-level** (`curriculum.yaml`):
```yaml
# L1
active_vision: global  # grid_encoding active, local_window masked

# L2
active_vision: partial  # local_window active, grid_encoding masked
```

**v2 Code (FIXED)**:
```python
if config.stratum.vision_support in ["both", "global"]:
    observation_fields.append(ObservationField(
        name="obs_grid_encoding",
        dims=grid_dims,
        curriculum_active=(curriculum.active_vision == "global")  # Masking control
    ))

if config.stratum.vision_support in ["both", "partial"]:
    observation_fields.append(ObservationField(
        name="obs_local_window",
        dims=window_size ** 2,
        curriculum_active=(curriculum.active_vision == "partial")  # Masking control
    ))
```

**Result**: L1 and L2 both have obs_dim=121 (both fields present), enabling checkpoint transfer.

**Power User Mode**: `vision_support: global` creates ONLY grid_encoding (no local_window overhead).

---

### Issue #2: Perception Can't Vary Across Curriculum ✅ FIXED

**Reviewer's Finding**:
> `vision_range` in `agent.yaml` (experiment-level, shared) creates impossibility: L1 needs full observability, L2 needs partial observability with vision_range=2. Cannot have single agent.yaml for both.

**v1 Structure (BROKEN)**:
```
agent.yaml (experiment-level, shared by all curriculum levels)
  perception:
    partial_observability: ???  # Can't be both true and false
    vision_range: ???  # Can't be both 8 and 2
```

**v2 Fix**: Moved to curriculum-level

```
levels/L1/curriculum.yaml:
  active_vision: global
  vision_range: 8  # Not used (global mode)

levels/L2/curriculum.yaml:
  active_vision: partial
  vision_range: 2  # 5×5 local window
```

**Result**: Each curriculum level controls its own perception mode independently.

---

### Issue #3: Temporal Mechanics All-or-Nothing ✅ FIXED

**Reviewer's Finding**:
> Temporal mechanics in stratum.yaml (experiment-level) forces ALL levels to have temporal observations, but only L3 uses temporal mechanics. L0/L1/L2 don't have temporal fields.

**v1 Structure (BROKEN)**:
```yaml
# stratum.yaml (experiment-level)
temporal_mechanics:
  enabled: true  # Forces temporal fields in ALL levels

# But L0/L1/L2 don't want temporal!
```

**v2 Fix: Support/Active Pattern**:

**Experiment-level**:
```yaml
# stratum.yaml
temporal_support: enabled  # Creates time_sin, time_cos fields in obs_dim
```

**Curriculum-level**:
```yaml
# L0/L1/L2 curriculum.yaml
active_temporal: false  # Temporal fields MASKED

# L3 curriculum.yaml
active_temporal: true  # Temporal fields ACTIVE
day_length: 24
```

**Result**: All levels have same obs_dim (temporal fields present), but L0/L1/L2 have them masked, L3 has them active.

**Power User Mode**: `temporal_support: disabled` removes temporal fields entirely.

---

### Issue #4: Global Actions Missing ✅ FIXED

**Reviewer's Finding**:
> `global_actions.yaml` exists but has no home in the architecture. Actions are vocabulary (breaking) but not allocated.

**v1 Structure (BROKEN)**:
- No `actions.yaml` in directory structure
- No mention of action vocabulary

**v2 Fix**: Added `actions.yaml` at experiment level

```
configs/default_curriculum/
├── actions.yaml  # NEW: Action vocabulary
```

```yaml
# actions.yaml
substrate_actions:
  inherit: true  # UP, DOWN, LEFT, RIGHT from Grid2D

custom_actions:
  - name: INTERACT
    enabled_by_default: true
  - name: WAIT
    enabled_by_default: true
  - name: REST
    enabled_by_default: false
  - name: MEDITATE
    enabled_by_default: false

labels:
  preset: gaming
```

**Result**: Action vocabulary defined at experiment level, curriculum can enable/disable subset.

---

### Issue #5: Cascade Semantics ✅ ADDRESSED

**Reviewer's Finding**:
> Cascades describe relationships between bars (structure) and parameters (threshold, strength). Design only puts cascades at curriculum level - structure vs parameters not separated.

**v1 Structure**:
```yaml
# bars.yaml (curriculum-level only)
cascades:
  - source: satiation
    target: health
    threshold: 0.3  # Parameters
    strength: 0.004
```

**v2 Fix**: Split structure and parameters

```yaml
# environment.yaml (experiment-level)
cascade_graph:
  - source: satiation
    target: health
    description: "Low hunger damages health"
  # Defines WHICH cascades exist (structure)

# bars.yaml (curriculum-level)
cascades:
  - source: satiation  # Must match cascade_graph
    target: health
    threshold: 0.3  # Curriculum parameter
    strength: 0.004  # Curriculum parameter
```

**Compiler Validation**: All curriculum levels must implement all cascades from environment.yaml cascade_graph.

---

### Issue #6: VFS Implementation ✅ CLARIFIED

**Reviewer's Finding**:
> VFS variables defined in environment.yaml but computation logic unclear. Where does "deficit_energy" get computed?

**v2 Clarification**:

```yaml
# environment.yaml (vocabulary)
variables:
  - name: deficit_energy
    type: scalar
    dims: 1
    description: "How far below target energy"
  # Just declares "this field exists in observations"

# Computation logic lives in VFS compiler, not in config
# VFS system handles runtime computation based on variable definitions
```

**Result**: Clean separation - configs declare vocabulary, VFS subsystem handles computation.

---

### Issue #7: Action Labels and Custom Actions ✅ FIXED

**Reviewer's Finding**:
> No clear home for action labels (gaming, 6dof), custom actions (REST, MEDITATE).

**v2 Fix**: All in `actions.yaml` (see Issue #4 fix above)

---

## Major Concerns Addressed

### Concern #8: Migration Complexity ✅ ACKNOWLEDGED

**Reviewer's Finding**:
> substrate.yaml has settings spanning both levels: type/width (breaking) vs boundary/distance_metric (non-breaking). Migration must split these.

**v2 Acknowledgement**:
- Added to Open Questions #1: "Where do substrate behavioral parameters live?"
- Options: curriculum.yaml, stratum.yaml with behavioral section, new file
- Requires decision before implementation

---

## New Patterns Introduced

### Support vs Active Pattern

**Core Idea**: Experiment declares which fields CAN exist (support), curriculum declares which ARE active vs masked (active).

**Benefits**:
1. **Preserves BUG-43 masking**: `vision_support: both` enables transfer learning
2. **Enables power user optimization**: `vision_support: global` minimizes obs_dim
3. **Per-level perception control**: Each curriculum level sets own active_vision
4. **Clear boundaries**: Support = vocabulary (experiment), Active = behavior (curriculum)

**Applied to**:
- Vision: `vision_support` + `active_vision`
- Temporal: `temporal_support` + `active_temporal`
- Future: Meters, VFS variables?

---

## File Structure Changes

### v1 → v2 Additions

- ✅ `actions.yaml` (experiment-level action vocabulary)
- ✅ `curriculum.yaml` (per-level active field control)

### v1 → v2 Modifications

**stratum.yaml**:
```diff
+ vision_support: both  # NEW: Support/Active pattern
+ temporal_support: enabled
```

**environment.yaml**:
```diff
+ cascade_graph:  # NEW: Cascade structure (was in bars.yaml)
+   - source: satiation
+     target: health
+ modulation_graph:  # NEW: Modulation structure
```

**agent.yaml**:
```diff
  perception:
    observation_encoding: relative
-   partial_observability: false  # REMOVED: Now in curriculum.yaml
-   vision_range: 2                # REMOVED: Now in curriculum.yaml
```

---

## Example: Default Curriculum

### Experiment-Level Files

**stratum.yaml**:
```yaml
substrate:
  type: grid
  grid: {width: 8, height: 8}
vision_support: both  # Enable transfer learning
temporal_support: enabled
```

**environment.yaml**: 8 meters, 14 affordances, cascade graph, VFS variables

**actions.yaml**: Substrate actions + INTERACT + WAIT + REST + MEDITATE

**agent.yaml**: Perception encoding + Drive (multiplicative) + Brain (feedforward)

### Curriculum-Level Files

**L1/curriculum.yaml**:
```yaml
active_vision: global  # Full grid, mask local window
active_temporal: false  # Mask temporal
vision_range: 8
```

**L2/curriculum.yaml**:
```yaml
active_vision: partial  # Local window, mask grid
active_temporal: false
vision_range: 2  # 5×5 window
```

**L3/curriculum.yaml**:
```yaml
active_vision: global
active_temporal: true  # Activate temporal
day_length: 24
```

**Result**: L1, L2, L3 all have **identical obs_dim=121** → checkpoint transfer works!

---

## Open Questions Carried Forward

1. **Substrate behavioral parameters**: boundary, distance_metric - where do they live?
2. **Active meter control**: Can curriculum levels mask certain meters?
3. **Cascade enforcement**: STRICT (all must be present) or LENIENT (can disable with strength=0)?
4. **Expert mode**: Allow manual observation spec definition with no guardrails?

---

## Future: Expert Mode

**Note from user**: The Support/Active pattern is "middle ground". Future "expert mode" allows manual observation spec definition:

```yaml
# stratum.yaml (expert mode)
observation_mode: manual

manual_observation_spec:
  fields:
    - name: custom_field_1
      dims: 42
    - name: custom_field_2
      dims: 7
  # If this doesn't match network architecture, everything explodes
  # "You keep both pieces"
```

**Design decision**: Deferred to future enhancement.

---

## Recommendation

**Design v2 is ready for implementation** pending resolution of Open Question #1 (substrate behavioral parameters).

All critical issues from code review are resolved. The Support/Active pattern provides clean separation while preserving BUG-43's transfer learning capability.
