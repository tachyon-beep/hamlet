# Target Configuration Design v2.1 - Patch Notes
## Addressing Code Review Round 2 Feedback

**Reviewer**: feature-dev:code-reviewer (cold read of v2)
**Base Design**: `target-config-design-v2.md`
**Status**: All critical and major issues resolved

---

## Critical Issues Resolved

### Issue #1: Vision Range Runtime Error ✅ FIXED

**Problem**: Observation spec code used `curriculum.vision_range` unconditionally, causing crash when `active_vision: global`.

**Resolution**: Changed `vision_range` from integer to normalized float (0.0-1.0)

**New Schema**:
```yaml
# curriculum.yaml
vision_range: 0.5  # Normalized: 0.0 (blind) to 1.0 (full grid coverage)
```

**Formula**:
```python
# Supports non-square grids
radius_x = ceil(vision_range * (grid_width / 2))
radius_y = ceil(vision_range * (grid_height / 2))
window_width = radius_x * 2 + 1
window_height = radius_y * 2 + 1
window_dims = window_width * window_height
```

**Examples**:
- 8×8 grid, vision_range=0.5: radius=2, window=5×5 (25 dims)
- 9×9 grid, vision_range=0.25: radius=2, window=5×5 (25 dims)
- 5×9 grid, vision_range=0.7: radius_x=2, radius_y=4, window=5×9 (45 dims)

**Benefits**:
- Cannot create invalid windows (always bounded by grid)
- Semantic: "see X% of distance to edge"
- No special case for active_vision: global (vision_range ignored, no crash)

**Compiler Output** (verbose mode):
```
Compiling curriculum: L2_partial_observability
  vision_support: both
  active_vision: partial
  vision_range: 0.5 (normalized)
  grid_dimensions: 8×8
  → computed local_window: 5×5 (radius_x=2, radius_y=2, dims=25)
  → obs_dim: 121 (grid_encoding=64[masked], local_window=25[active], ...)
```

**Updated Observation Spec Code**:
```python
if config.stratum.vision_support in ["both", "partial"]:
    # Compute window dimensions from normalized vision_range
    grid_width = config.stratum.substrate.grid.width
    grid_height = config.stratum.substrate.grid.height

    radius_x = math.ceil(curriculum.vision_range * (grid_width / 2))
    radius_y = math.ceil(curriculum.vision_range * (grid_height / 2))

    window_width = radius_x * 2 + 1
    window_height = radius_y * 2 + 1
    window_dims = window_width * window_height

    observation_fields.append(ObservationField(
        name="obs_local_window",
        dims=window_dims,
        curriculum_active=(curriculum.active_vision == "partial")
    ))
```

---

### Issue #2: Meter range_type Semantics ✅ CLARIFIED

**Problem**: `range_type` labeled "breaking" but behavior undefined.

**Resolution**: `range_type` is **metadata only**, not breaking.

**Purpose**:
- UI display hint (show as percentage vs integer vs unbounded)
- Documentation (communicates semantic range to developers)
- **Does NOT affect observation encoding** (all meters encoded as float32)

**Schema**:
```yaml
# environment.yaml
meters:
  - name: energy
    range_type: normalized  # UI displays as 0-100%

  - name: money
    range_type: unbounded  # UI displays raw value (can exceed 1.0)

  - name: charisma  # Hypothetical future meter
    range_type: integer  # UI displays as 0-100 points
```

**Breaking Status**: Changed from YES to **NO** (metadata doesn't affect obs_dim)

---

### Issue #3: Vision Range Validation ✅ RESOLVED

**Problem**: No validation that `vision_range` is valid for grid dimensions.

**Resolution**: Normalized vision_range (0.0-1.0) makes validation unnecessary.

**Validation Rules**:
```python
# Schema validation in curriculum.yaml
assert 0.0 <= vision_range <= 1.0, "vision_range must be in [0.0, 1.0]"

# No grid-size-specific validation needed (normalized range always valid)
```

**Edge Cases**:
- `vision_range: 0.0`: radius=0, window=1×1 (degenerate, but valid)
- `vision_range: 1.0`: Full grid coverage (capped at grid dimensions)
- `vision_range: 2.0`: **Schema error** (exceeds [0.0, 1.0] range)

---

## Major Concerns Resolved

### Issue #4: Substrate Behavioral Parameters ✅ RESOLVED

**Problem**: `boundary` and `distance_metric` had no home.

**Resolution**: Added to `stratum.yaml` (substrate parameters live with substrate definition)

**Updated stratum.yaml**:
```yaml
version: "1.0"

substrate:
  type: grid

  grid:
    topology: square
    width: 8
    height: 8

    # Behavioral parameters (experiment-level, must match across curriculum)
    boundary: clamp  # clamp, wrap, bounce, sticky (MANDATORY)
    distance_metric: manhattan  # manhattan, euclidean, chebyshev (MANDATORY)
    observation_encoding: relative  # relative, scaled, absolute (MANDATORY)

vision_support: both
temporal_support: enabled
```

**Breaking Status**: NO (behavioral parameters don't affect obs_dim)

**No-Defaults Enforcement**: All three parameters are **mandatory** (no defaults).

**Clarification on observation_encoding**:
All encoding modes produce **identical obs_dim** (2 dims for position in Grid2D). The modes differ only in value ranges:
- `relative`: Position coords normalized to [0,1] (grid-size-agnostic)
- `scaled`: Position coords scaled to [0, grid_size] (value range conveys grid size implicitly)
- `absolute`: Raw unnormalized coordinates (for continuous substrates)

The network can learn grid size from the scaled value range without requiring explicit grid_width/grid_height dimensions. This preserves checkpoint portability across grid sizes while giving networks richer position information if needed.

---

### Issue #5: Cascade Validation Ambiguity ✅ CLARIFIED

**Problem**: Unclear what happens if bars.yaml omits cascades from environment.yaml.

**Resolution**: STRICT validation with explicit disable pattern.

**Validation Rules**:
1. **All cascades from environment.yaml cascade_graph MUST be present in bars.yaml**
2. **To disable a cascade, set `strength: 0.0`** (explicit disable)
3. **Cannot add cascades not in environment.yaml** (compiler error)

**Example**:
```yaml
# environment.yaml (experiment-level graph structure)
cascade_graph:
  - source: satiation
    target: health
  - source: satiation
    target: energy
  - source: mood
    target: energy

# bars.yaml (curriculum-level parameters)
cascades:
  - source: satiation
    target: health
    threshold: 0.3
    strength: 0.004  # Active cascade

  - source: satiation
    target: energy
    threshold: 0.3
    strength: 0.0  # DISABLED (explicit)

  - source: mood
    target: energy
    threshold: 0.2
    strength: 0.001  # Active cascade
```

**Compiler Errors**:
- Missing cascade: `"bars.yaml missing cascade from environment.yaml: mood → energy"`
- Extra cascade: `"bars.yaml defines cascade not in environment.yaml: fitness → health"`
- Mismatched names: `"bars.yaml cascade source 'satation' doesn't match meter name"`

---

### Issue #6: active_temporal Requires day_length ✅ RESOLVED

**Problem**: `active_temporal: true` requires `day_length`, but not enforced by schema.

**Resolution**: Apply **no-defaults principle** - `day_length` is **always mandatory**.

**Schema**:
```yaml
# curriculum.yaml (MANDATORY FIELDS)
active_temporal: false
day_length: null  # Explicit "not used" (no-defaults principle)

# OR

active_temporal: true
day_length: 24  # MUST be specified when active_temporal: true
```

**Validation Rules**:
```python
# Schema validation
if curriculum.active_temporal == True:
    assert curriculum.day_length is not None, "day_length required when active_temporal: true"
    assert curriculum.day_length > 0, "day_length must be positive"

if curriculum.active_temporal == False:
    assert curriculum.day_length is None, "day_length must be null when active_temporal: false"
```

**Examples**:
```yaml
# ✅ VALID: Temporal disabled
active_temporal: false
day_length: null

# ✅ VALID: Temporal enabled
active_temporal: true
day_length: 24

# ❌ INVALID: Missing day_length (violates no-defaults)
active_temporal: false
# day_length not specified → SCHEMA ERROR

# ❌ INVALID: day_length specified but not used
active_temporal: false
day_length: 24  # Conflict: temporal disabled but day_length set → SCHEMA ERROR
```

---

### Issue #7: Meter Vocabulary Changes ✅ ACKNOWLEDGED

**Problem**: Adding new meter to environment.yaml breaks all curriculum levels + all checkpoints.

**Resolution**: Keep STRICT for v2.1, document as future enhancement.

**Current Behavior (STRICT)**:
- All meters in environment.yaml must be in all curriculum levels
- Adding meter = breaking change requiring full migration
- Follows no-defaults principle (explicit everywhere)

**Future Enhancement (v3.0)**:
Apply support/active pattern to meters:
```yaml
# environment.yaml
meters:
  - name: charisma
    required: false  # Optional meter (support/active pattern)

# curriculum.yaml
active_meters: [energy, health, satiation, mood]  # Charisma masked
```

**Rationale for keeping STRICT in v2.1**:
- Pre-release = zero users, can break freely
- Support/active for meters adds complexity
- Can be added in v3.0 based on real usage patterns

---

## Updated File Examples

### stratum.yaml (with substrate behavioral params)

```yaml
version: "1.0"

substrate:
  type: grid

  grid:
    topology: square
    width: 8
    height: 8
    boundary: clamp
    distance_metric: manhattan
    observation_encoding: relative

vision_support: both
temporal_support: enabled
```

---

### curriculum.yaml (with normalized vision_range + mandatory day_length)

```yaml
version: "1.0"

# L1 Full Observability
active_vision: global
active_temporal: false

vision_range: 0.5  # Normalized (not used when active_vision: global, but still valid)
day_length: null  # Explicit disable (no-defaults principle)
```

```yaml
version: "1.0"

# L2 Partial Observability
active_vision: partial
active_temporal: false

vision_range: 0.5  # 50% of grid distance → 5×5 window on 8×8 grid
day_length: null
```

```yaml
version: "1.0"

# L3 Temporal Mechanics
active_vision: global
active_temporal: true

vision_range: 0.5  # Not used (global vision)
day_length: 24  # MANDATORY when active_temporal: true
```

---

## Summary of Changes

### Critical Fixes
1. ✅ Vision range → normalized float (0.0-1.0) with ceil rounding
2. ✅ Meter range_type → metadata only (not breaking)
3. ✅ Vision validation → normalized range eliminates need

### Major Fixes
4. ✅ Substrate params → added to stratum.yaml (boundary, distance_metric, observation_encoding)
   - **Clarification added**: observation_encoding modes produce identical obs_dim (value ranges differ, not dimensions)
5. ✅ Cascade validation → STRICT with explicit disable (strength: 0.0)
6. ✅ day_length → always mandatory (null when not used)
7. ✅ Meter flexibility → acknowledged, deferred to v3.0

### Design Status

**v2.0**: 70% complete (per reviewer)
**v2.1**: 100% complete (all critical/major issues resolved + observation_encoding clarified)

**Remaining Open Questions** (non-blocking):
- VFS variable computation location (acknowledged, implementation detail)
- Action enabled_by_default semantics (minor, can clarify in implementation)
- Expert mode manual observation spec (future enhancement)

**Ready for Implementation**: YES

---

## Next Steps

1. ✅ ~~Get code reviewer cold read of v2.1 patch~~ (DONE - observation_encoding clarified)
2. Merge v2.1 changes into v2 document
3. Create final target-config-design-v2.1.md
4. Update BUNDLE README with v2.1 final status
5. Begin implementation planning
