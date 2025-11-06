# Retired: TASK-002B UAC Action Space (Original)

**Retirement Date**: 2025-11-06

**Reason**: Architecture conflict with TASK-002A substrate abstraction

---

## Why This Was Retired

The original TASK-002B plans were written **before** TASK-002A (Configurable Spatial Substrates) was completed. TASK-002A introduced fundamental changes that made the original 002B approach incompatible:

### What Changed in TASK-002A

1. **Substrate-defined action spaces**: Grid2D→6 actions, Grid3D→8, GridND(7D)→16, Aspatial→2
2. **Dynamic action dimensionality**: No longer hardcoded to 6 actions
3. **Action space determined by substrate geometry**: Substrates own their movement actions

### Why Original 002B Conflicted

The original plan assumed:
- ❌ Fixed 6-action space (UP/DOWN/LEFT/RIGHT/INTERACT/WAIT)
- ❌ Actions defined entirely in config (conflicted with substrate authority)
- ❌ Manual action configs for all substrate types (impractical for GridND)

### The Architectural Conflict

**Original 002B approach** (config owns action space):
```yaml
# actions.yaml defines ALL actions
actions:
  - {id: 0, name: "UP", delta: [0, -1]}
  - {id: 1, name: "DOWN", delta: [0, 1]}
  # ... total: 6 actions
```

**Post-002A reality** (substrate owns action space):
```python
# Substrate auto-generates actions based on dimensionality
self.action_dim = self.substrate.action_space_size
# Grid2D: 6, Grid3D: 8, GridND(7D): 16
```

**Result**: Cannot have two sources of truth for action space size.

---

## What Replaced It

**New Task**: [TASK-002B-COMPOSABLE-ACTION-SPACE](../../TASK-002B-COMPOSABLE-ACTION-SPACE.md)

**Key Insight**: Actions are **composed** from multiple sources, not defined in a single place.

```
Action Space = Substrate Actions + Custom Actions + (Future) Affordance Actions
              (REQUIRED)          (OPTIONAL)         (DEFERRED)
```

**Benefits**:
- ✅ Substrate owns its movement actions (no conflict)
- ✅ Operators can add custom actions (REST, MEDITATE, TELEPORT)
- ✅ Scales to N-dimensional substrates (no manual config for 100D grids)
- ✅ Backward compatible (custom_actions.yaml is optional)

---

## Files Archived

- `TASK-002B-UAC-ACTION-SPACE.md` - Original task definition
- `plan-task-002b-uac-action-space.md` - Original implementation plan (28K tokens)
- `plan-task-002b-uac-action-space-addendum.md` - Performance optimization addendum

---

## Historical Value

These documents remain useful for:
- Understanding the evolution of the action space design
- Seeing what approaches were considered and rejected
- Performance optimization insights (addendum's vectorization patterns still apply)
- Lessons about planning tasks before dependencies are complete

---

**Status**: Archived for historical reference only. Do not implement.

**See instead**: [TASK-002B-COMPOSABLE-ACTION-SPACE](../../TASK-002B-COMPOSABLE-ACTION-SPACE.md)
