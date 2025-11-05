# Substrate Migration Guide

**Date**: 2025-11-05
**Status**: Active
**Version**: 1.0
**Applies to**: TASK-002A (Configurable Spatial Substrates)

---

## Overview

This guide explains how to migrate between different spatial substrates in HAMLET/Townlet. Substrate changes affect observation dimensions, network architecture, and checkpoint compatibility.

**Key Insight**: Changing substrate type or encoding strategy **requires retraining from scratch**. Checkpoints are substrate-specific and cannot be transferred.

---

## Substrate Types

HAMLET supports three substrate types:

| Substrate | Position Dim | Use Case | Observation Encoding |
|-----------|--------------|----------|---------------------|
| **Grid2D** | 2 | Standard 2D grids (8×8, 16×16, etc.) | One-hot or Coordinate |
| **Grid3D** | 3 | 3D voxel worlds (future) | Coordinate only |
| **Aspatial** | 0 | Pure resource management (no positioning) | None |

---

## Migration Scenarios

### Scenario 1: Grid2D → Grid2D (Different Size)

**Example**: 8×8 grid → 16×16 grid

**Impact**:
- ⚠️ **Observation dimension changes** (one-hot encoding)
- ❌ **Checkpoints incompatible** (network input size changes)
- ✅ **No code changes needed** (substrate handles size)

**One-Hot Encoding**:
- 8×8: obs_dim = 64 (grid) + 27 (meters/affordances/temporal) = 91
- 16×16: obs_dim = 256 (grid) + 27 = 283

**Coordinate Encoding** (recommended for large grids):
- 8×8: obs_dim = 2 (normalized x,y) + 27 = 29
- 16×16: obs_dim = 2 (normalized x,y) + 27 = 29 ✅ **Same dimension!**

**Migration Steps**:

1. **Choose encoding strategy**:
   ```yaml
   # configs/my_large_grid/substrate.yaml
   type: "grid2d"
   grid:
     width: 16
     height: 16
     position_encoding: "coords"  # ← Use coords for grids >8×8
     topology: "square"
     boundary: "clamp"
     distance_metric: "manhattan"
   ```

2. **Delete old checkpoints**:
   ```bash
   rm -rf checkpoints_old/
   ```

3. **Retrain from scratch**:
   ```bash
   python -m townlet.demo.runner --config configs/my_large_grid
   ```

**Performance Comparison**:
- **One-hot**: More sample-efficient (stronger inductive bias), but infeasible for large grids
- **Coordinate**: Works for any grid size, enables transfer learning, but may need more samples

---

### Scenario 2: Grid2D (One-Hot) → Grid2D (Coordinate)

**Example**: Switch from one-hot to coordinate encoding on same grid

**Impact**:
- ⚠️ **Observation dimension changes drastically**
- ❌ **Checkpoints incompatible**
- ✅ **Enables transfer learning across grid sizes**

**Before (One-Hot)**:
```yaml
# substrate.yaml
grid:
  width: 8
  height: 8
  position_encoding: "onehot"  # Default for backward compat
```
- Obs dim: 64 + 27 = 91

**After (Coordinate)**:
```yaml
# substrate.yaml
grid:
  width: 8
  height: 8
  position_encoding: "coords"  # New encoding
```
- Obs dim: 2 + 27 = 29

**Migration Steps**:

1. **Update substrate.yaml** (add `position_encoding: "coords"`)
2. **Delete checkpoints** (`rm -rf checkpoints/`)
3. **Retrain from scratch**

**When to Use Coordinate Encoding**:
- ✅ Grid size >8×8 (one-hot becomes infeasible)
- ✅ Want transfer learning across grid sizes
- ✅ Training on large grids, deploying on small grids
- ❌ Small grids where one-hot works well (may reduce sample efficiency)

---

### Scenario 3: Grid2D → Aspatial

**Example**: Remove positioning from pure resource management game

**Impact**:
- ⚠️ **Observation dimension changes** (no grid encoding)
- ⚠️ **Affordances have no positions** (interaction mechanics change)
- ❌ **Checkpoints incompatible**
- ✅ **Simpler state space** (faster training)

**Before (Grid2D)**:
```yaml
# substrate.yaml
type: "grid2d"
grid:
  width: 8
  height: 8
```
- Obs dim: 64 + 27 = 91
- Agent has position (x, y)
- Affordances have positions
- Movement actions matter

**After (Aspatial)**:
```yaml
# substrate.yaml
type: "aspatial"
aspatial: {}
```
- Obs dim: 0 + 27 = 27
- Agent has no position
- Affordances have no positions (always available)
- Movement actions do nothing (or disabled)

**Migration Steps**:

1. **Update substrate.yaml**:
   ```yaml
   type: "aspatial"
   aspatial: {}
   ```

2. **Update training.yaml** (adjust action space if needed):
   ```yaml
   environment:
     move_energy_cost: 0.0  # No movement in aspatial
   ```

3. **Delete checkpoints** (`rm -rf checkpoints/`)

4. **Retrain from scratch**

**Use Cases for Aspatial**:
- ✅ Pure resource management (no spatial strategy)
- ✅ Testing meter dynamics without positioning
- ✅ Baseline for comparing spatial vs aspatial learning

---

### Scenario 4: Grid2D → Grid3D (Future)

**Example**: 8×8×1 (2D) → 8×8×3 (3D voxel world)

**Impact**:
- ⚠️ **Observation dimension changes**
- ⚠️ **Network architecture may need updates** (3D convolutions)
- ❌ **Checkpoints incompatible**
- ⚠️ **Movement actions expand** (UP/DOWN/LEFT/RIGHT → 6 directions)

**Migration Steps** (when Grid3D is implemented):

1. **Update substrate.yaml**:
   ```yaml
   type: "grid3d"
   grid3d:
     width: 8
     height: 8
     depth: 3
     position_encoding: "coords"  # Required for 3D
     boundary: "clamp"
     distance_metric: "euclidean"
   ```

2. **Update network type** (may need 3D convolutions):
   ```yaml
   # training.yaml
   population:
     network_type: "spatial3d"  # Future network type
   ```

3. **Delete checkpoints**, retrain

---

## Checkpoint Compatibility Matrix

| From/To | Grid2D (One-Hot) | Grid2D (Coords) | Grid3D | Aspatial |
|---------|------------------|-----------------|--------|----------|
| **Grid2D (One-Hot)** | ✅ (same size only) | ❌ | ❌ | ❌ |
| **Grid2D (Coords)** | ❌ | ✅ (any size!) | ❌ | ❌ |
| **Grid3D** | ❌ | ❌ | ✅ (same dims) | ❌ |
| **Aspatial** | ❌ | ❌ | ❌ | ✅ |

**Key Takeaway**: Only coordinate-encoded Grid2D checkpoints are size-agnostic. All other combinations require retraining.

---

## Observation Dimension Calculator

Use this formula to predict observation dimension after substrate change:

```
obs_dim = position_encoding_dim + meter_dim + affordance_dim + temporal_dim
```

Where:
- **position_encoding_dim**: Substrate-specific
  - Grid2D (one-hot): `width × height`
  - Grid2D (coords): `2`
  - Grid3D (coords): `3`
  - Aspatial: `0`
- **meter_dim**: Number of meters (8 for standard Townlet)
- **affordance_dim**: 15 (14 affordances + "none")
- **temporal_dim**: 4 (time_of_day, retirement_age, interaction_progress, interaction_ticks)

**Examples**:
- Grid2D 8×8 (one-hot): `64 + 8 + 15 + 4 = 91`
- Grid2D 8×8 (coords): `2 + 8 + 15 + 4 = 29`
- Grid3D 8×8×3 (coords): `3 + 8 + 15 + 4 = 30`
- Aspatial: `0 + 8 + 15 + 4 = 27`

---

## Performance Considerations

### Grid Size Impact on Training

| Grid Size | One-Hot Dim | Coord Dim | Training Speed | Sample Efficiency |
|-----------|-------------|-----------|----------------|-------------------|
| 3×3 | 9 | 2 | ⚡⚡⚡ Fast | ✅✅✅ High |
| 8×8 | 64 | 2 | ⚡⚡ Medium | ✅✅ Medium |
| 16×16 | 256 | 2 | ⚡ Slow | ✅ Low |
| 32×32 | 1024 | 2 | 🐢 Very slow | ⚠️ Very low |

**Recommendation**: Use **coordinate encoding** for grids >8×8. One-hot becomes infeasible.

### Position Encoding Trade-offs

**One-Hot Encoding**:
- ✅ Stronger inductive bias (network sees full grid state)
- ✅ Faster learning on small grids (3×3, 8×8)
- ❌ Observation dimension explodes with grid size
- ❌ Cannot transfer between grid sizes

**Coordinate Encoding**:
- ✅ Constant observation dimension (2 dims for any grid)
- ✅ Transfer learning across grid sizes
- ✅ Scales to large grids (16×16, 32×32)
- ❌ Weaker inductive bias (network must learn spatial relationships)
- ❌ May require more samples on small grids

**Rule of Thumb**:
- Grids ≤8×8: One-hot (unless you need transfer learning)
- Grids >8×8: Coordinate (one-hot infeasible)
- 3D grids: Coordinate only
- Aspatial: No encoding

---

## Common Migration Errors

### Error 1: "Checkpoint observation dimension mismatch"

**Symptom**:
```
RuntimeError: Checkpoint has obs_dim=91 but environment has obs_dim=29
```

**Cause**: Loaded checkpoint from different substrate configuration

**Fix**: Delete checkpoint and retrain from scratch

---

### Error 2: "Grid has 64 cells but 70 affordances need space"

**Symptom**:
```
ValueError: Grid has 64 cells but 70 affordances + 1 agent need space.
```

**Cause**: Grid too small for enabled affordances

**Fix**: Either:
1. Increase grid size in `substrate.yaml`
2. Reduce `enabled_affordances` in `training.yaml`
3. Use aspatial substrate (no positioning constraint)

---

### Error 3: "Cannot randomize affordance positions in aspatial substrate"

**Symptom**:
```
ValueError: Cannot randomize affordance positions in aspatial substrate.
```

**Cause**: Trying to use aspatial substrate with position-based affordance logic

**Fix**: This is expected behavior - aspatial substrates have no positions. Affordances are always "available" without positioning.

---

## Migration Checklist

Before migrating to a new substrate:

- [ ] **Understand impact**: Will observation dimension change?
- [ ] **Back up checkpoints**: `cp -r checkpoints/ checkpoints_backup/`
- [ ] **Update substrate.yaml**: Change substrate type/size/encoding
- [ ] **Validate config**: `python scripts/validate_substrate_configs.py`
- [ ] **Run runtime validation**: `python scripts/validate_substrate_runtime.py --config configs/my_config`
- [ ] **Delete old checkpoints**: `rm -rf checkpoints/`
- [ ] **Retrain from scratch**: `python -m townlet.demo.runner --config configs/my_config`
- [ ] **Monitor training**: Watch for unexpected obs_dim, network errors
- [ ] **Document changes**: Note substrate change in experiment log

---

## Best Practices

### When to Migrate Substrates

✅ **Do migrate**:
- Experimenting with different grid sizes
- Testing spatial vs aspatial learning
- Moving to 3D when available
- Optimizing for large grids (switch to coords)

❌ **Don't migrate mid-training**:
- Never change substrate while training in progress
- Always complete current training run first
- Checkpoints are substrate-specific

### Position Encoding Strategy

**Default**: Use one-hot for backward compatibility (existing configs use this)

**Recommended**:
```yaml
# For small grids (≤8×8) where sample efficiency matters
position_encoding: "onehot"

# For large grids (>8×8) where one-hot is infeasible
position_encoding: "coords"

# For transfer learning experiments
position_encoding: "coords"  # Train on 16×16, deploy on 8×8
```

---

## Future Substrate Types

### Hexagonal Grids (Planned)

```yaml
type: "hexagonal"
hexagonal:
  radius: 5
  orientation: "pointy"  # or "flat"
  position_encoding: "coords"
```

### Graph Substrates (Planned)

```yaml
type: "graph"
graph:
  num_nodes: 20
  connectivity: "erdos_renyi"
  edge_probability: 0.3
  position_encoding: "node_id"
```

---

## Summary

**Key Takeaways**:

1. **Substrate changes break checkpoints** - always retrain from scratch
2. **One-hot encoding** best for small grids (≤8×8)
3. **Coordinate encoding** required for large grids (>8×8)
4. **Aspatial substrates** simplify learning but remove spatial strategy
5. **Use validation scripts** before training to catch config errors

**Quick Reference**:
```bash
# Validate config
python scripts/validate_substrate_configs.py

# Validate runtime
python scripts/validate_substrate_runtime.py --config configs/my_config

# Delete old checkpoints
rm -rf checkpoints/

# Retrain
python -m townlet.demo.runner --config configs/my_config
```

---

**For more information**:
- TASK-002A Implementation Plan: `docs/tasks/TASK-002A-CONFIGURABLE-SPATIAL-SUBSTRATES.md`
- Substrate Template: `configs/templates/substrate.yaml`
- Runtime Validation: `scripts/validate_substrate_runtime.py`
