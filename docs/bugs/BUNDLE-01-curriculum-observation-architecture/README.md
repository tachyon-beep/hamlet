# BUNDLE-01: Curriculum Observation Architecture

**Common Concern**: Observation structure, curriculum masking, and transfer learning across curriculum levels

**Status**: Design Complete (BUG-43 resolved, ENH-28 design v2.1 approved for implementation)

## Overview

This bundle addresses the architectural tension between:
1. **Transfer learning**: Checkpoints trained on one curriculum level (e.g., L1 full obs) should work on another (e.g., L2 POMDP)
2. **Optimization**: Power users want minimal obs_dim for single-level experiments without curriculum overhead
3. **Configuration hierarchy**: Experiment-level concerns (observation policy, stratum) should be separate from curriculum-level concerns (substrate, bars, affordances)

## Components

### BUG-43: Partial observability global view masking ✅ CLOSED
- **Problem**: Full obs and partial obs had different observation dimensions (L1=96, L2=57), preventing checkpoint transfer
- **Solution**: Always include both grid_encoding and local_window in observation vector (one active, one masked)
- **Result**: L1 and L2 now have identical obs_dim=121, enabling transfer learning
- **Trade-off**: Wasted dimensions for single-level experiments (L1 wastes 25 dims on masked local_window)
- **Status**: Resolved in commit fe7f29d
- **Tests**: `tests/test_townlet/unit/universe/test_partial_obs_curriculum_masking.py`

### ENH-28: Experiment-level configuration hierarchy ✅ DESIGN v2.1 COMPLETE
- **Problem**: BUG-43 forces all configs to pay obs_dim cost of curriculum superset, even single-level experiments
- **Solution**: Four-layer architecture + Support/Active pattern for observation field control
- **Architecture**: experiment.yaml + stratum.yaml + environment.yaml + actions.yaml + agent.yaml + curriculum.yaml per level
- **Key Pattern**: Support (experiment: which fields exist) vs Active (curriculum: which fields active vs masked)
- **Migration**: Backwards compatible with existing flat `configs/` structure (legacy mode)
- **Status**: Design v2.1 complete (code review approved 100/100 confidence), ready for implementation
- **Implementation Reference**: `reference-config-v2.1-complete.yaml` (600+ line complete example)
- **Owner**: compiler + config
- **Key Insights**:
  - Support/Active pattern preserves BUG-43 masking while enabling power user optimization
  - WHAT vs HOW split: vocabulary (breaking) vs parameters (non-breaking)
  - Normalized vision_range (0.0-1.0) eliminates validation complexity
  - All observation_encoding modes produce identical obs_dim (value ranges differ)

### Test Fixes: Observation structure adaptation ✅ COMPLETE
- **Problem**: 10 tests had hardcoded observation dimension calculations that broke after BUG-43
- **Solution**: Replaced hardcoded indices with `env.universe.observation_spec.fields` lookups
- **Pattern**: Use compiled observation spec as source of truth instead of manual calculations
- **Status**: All 144 observation tests passing
- **Files Modified**:
  - `tests/test_townlet/unit/environment/test_observations.py` (4 tests)
  - `tests/test_townlet/integration/test_env_substrate.py` (5 tests)
  - `tests/test_townlet/integration/test_data_flows.py` (1 test)

## Timeline

1. **2025-11-15**: BUG-43 implemented (curriculum masking)
2. **2025-11-15**: Test suite fixed (10 observation tests adapted)
3. **2025-11-15**: ENH-28 documented (experiment-level hierarchy)
4. **Future**: ENH-28 implementation

## Key Insights

### Observation Structure Evolution

**Pre-BUG-43 (broken transfer learning)**:
```
L1 (full obs):    obs_dim=96  (grid_encoding + position + velocity + meters + affordances + temporal)
L2 (partial obs): obs_dim=57  (local_window + position + velocity + meters + affordances + temporal)
Problem: Cannot transfer L1 checkpoint to L2 (different input shapes)
```

**Post-BUG-43 (curriculum superset)**:
```
L1 (full obs):    obs_dim=121 (grid_encoding [active] + local_window [masked] + position + velocity + meters + affordances + temporal)
L2 (partial obs): obs_dim=121 (grid_encoding [masked] + local_window [active] + position + velocity + meters + affordances + temporal)
Benefit: Checkpoint transfer works (same input shape)
Cost: Wasted dimensions (L1 wastes 25, L2 wastes 64)
```

**Future-ENH-28 (configurable policy)**:
```
Mode: curriculum_superset (default, current behavior)
  L1: obs_dim=121 (transfer learning enabled, wasted dims)
  L2: obs_dim=121

Mode: minimal (power user optimization)
  L1: obs_dim=96  (no wasted dims, no transfer)
  L2: obs_dim=57

Mode: explicit (manual specification)
  User specifies exact fields, full control
```

### Test Fragility Lessons

**Antipattern** (hardcoded calculations):
```python
# Breaks when observation structure changes
substrate_dim = env.substrate.get_observation_dim()
affordance_start = substrate_dim + 3 + env.meter_count
affordance = obs[0, affordance_start:affordance_start+15]
```

**Correct pattern** (use observation spec):
```python
# Robust to observation structure changes
field = next(f for f in env.universe.observation_spec.fields if f.name == "obs_affordance_at_position")
affordance = obs[0, field.start_index:field.end_index]
```

## Related Work

- **VFS (Variable & Feature System)**: Provides the infrastructure for observation field metadata
- **Curriculum masking**: `curriculum_active` flag on observation fields enables masked dimensions
- **Compiler Stage 5**: Builds `ObservationSpec` with field boundaries and metadata
- **Transfer learning research**: Common observation space across curriculum levels

## Future Extensions

1. **Multi-experiment workspaces**: `experiments/` directory with multiple `experiment.yaml` files
2. **Observation policy validation**: Compiler errors if policy conflicts with curriculum requirements
3. **Dynamic observation masking**: Runtime control over which fields are active (beyond curriculum)
4. **Observation compression**: Automatic field removal for deployment (strip masked dims)

## Documentation

### Active Documents

- **BUG-43**: `BUG-43-partial-observability-global-view-masking-and-obs-dim.md` - Closed issue that enabled this work
- **ENH-28**: `ENH-28-experiment-level-configuration-hierarchy.md` - Enhancement tracker
- **Target Design v2**: `target-config-design-v2.md` - Complete design specification (v2 + v2.1 integrated)
- **Reference Config**: `reference-config-v2.1-complete.yaml` - 600+ line implementation reference with all options

### Historical Documents

See `archive/README.md` for design iteration history:
- Design v1 (superseded by v2)
- v2.1 patch notes (merged into v2)
- v1 → v2 changes summary
- Brainstorming artifacts (semantic categories, settings audit)

## Contact

**Owner**: compiler + config + environment subsystems
**Questions**: See individual component files for detailed documentation
