# Manual Observation Dimension Validation

**Purpose**: Manually calculate expected observation dimensions for each config to validate against VFS implementation (Cycle 5).

**Date**: 2025-11-07
**Cycle**: 0.3 (Observation Structure Reverse Engineering)

---

## Calculation Method

For each config, calculate:
```
Total = substrate_dim + meter_count + affordance_dims + temporal_dims
```

Where:
- **substrate_dim**: From substrate.get_observation_dim()
  - Full observability: grid_cells + position_dims
  - POMDP: window_size^position_dim (no grid encoding in POMDP)
- **meter_count**: Always 8 (energy, health, satiation, money, mood, social, fitness, hygiene)
- **affordance_dims**: Always 15 (14 affordances + "none" one-hot)
- **temporal_dims**: Always 4 (time_sin, time_cos, interaction_progress, lifetime_progress)

---

## L0_0_minimal

**Config**: 3×3 Grid2D, full observability

### Substrate Encoding
- Grid size: 3×3 = 9 cells
- Position: 2 dims (normalized x, y)
- **substrate.get_observation_dim()**: 9 + 2 = **11 dims**

### Meters
- 8 scalar variables: energy, health, satiation, money, mood, social, fitness, hygiene
- **Total**: **8 dims**

### Affordance Encoding
- 14 affordance types + 1 "none" = 15-way one-hot
- **Total**: **15 dims**

### Temporal Features
- time_sin: 1 dim
- time_cos: 1 dim
- interaction_progress: 1 dim
- lifetime_progress: 1 dim
- **Total**: **4 dims**

### Total Calculation
```
11 (substrate) + 8 (meters) + 15 (affordance) + 4 (temporal) = 38 dims ✓
```

### Variable Count Validation
From `variables_reference.yaml`:
- grid_encoding: 9 dims (vecNf, dims=9)
- position: 2 dims (vecNf, dims=2)
- 8 meters: 8 × 1 dim = 8 dims
- affordance_at_position: 15 dims (vecNf, dims=15)
- 4 temporal: 4 × 1 dim = 4 dims
**TOTAL**: 9 + 2 + 8 + 15 + 4 = **38 dims** ✓

---

## L0_5_dual_resource

**Config**: 7×7 Grid2D, full observability

### Substrate Encoding
- Grid size: 7×7 = 49 cells
- Position: 2 dims
- **substrate.get_observation_dim()**: 49 + 2 = **51 dims**

### Meters: **8 dims**
### Affordance: **15 dims**
### Temporal: **4 dims**

### Total Calculation
```
51 (substrate) + 8 (meters) + 15 (affordance) + 4 (temporal) = 78 dims ✓
```

### Variable Count Validation
- grid_encoding: 49 dims
- position: 2 dims
- 8 meters: 8 dims
- affordance_at_position: 15 dims
- 4 temporal: 4 dims
**TOTAL**: 49 + 2 + 8 + 15 + 4 = **78 dims** ✓

---

## L1_full_observability

**Config**: 8×8 Grid2D, full observability

### Substrate Encoding
- Grid size: 8×8 = 64 cells
- Position: 2 dims
- **substrate.get_observation_dim()**: 64 + 2 = **66 dims**

### Meters: **8 dims**
### Affordance: **15 dims**
### Temporal: **4 dims**

### Total Calculation
```
66 (substrate) + 8 (meters) + 15 (affordance) + 4 (temporal) = 93 dims ✓
```

### Variable Count Validation
- grid_encoding: 64 dims
- position: 2 dims
- 8 meters: 8 dims
- affordance_at_position: 15 dims
- 4 temporal: 4 dims
**TOTAL**: 64 + 2 + 8 + 15 + 4 = **93 dims** ✓

---

## L2_partial_observability

**Config**: 8×8 Grid2D, POMDP with 5×5 vision window

### Local Window Encoding (POMDP)
- Vision range: 2 → window_size = 2×2 + 1 = 5
- Position dimensions: 2 (Grid2D)
- Window: 5^2 = **25 cells**
- **NOTE**: POMDP uses local window, NOT full grid encoding

### Position Encoding
- Normalized position: 2 dims (x, y in [0, 1])
- **Total**: **2 dims**

### Meters: **8 dims**
### Affordance: **15 dims**
### Temporal: **4 dims**

### Total Calculation
```
25 (window) + 2 (position) + 8 (meters) + 15 (affordance) + 4 (temporal) = 54 dims ✓
```

### Variable Count Validation
- local_window: 25 dims (vecNf, dims=25)
- position: 2 dims (vecNf, dims=2)
- 8 meters: 8 dims
- affordance_at_position: 15 dims
- 4 temporal: 4 dims
**TOTAL**: 25 + 2 + 8 + 15 + 4 = **54 dims** ✓

---

## L3_temporal_mechanics

**Config**: 8×8 Grid2D, full observability, temporal mechanics enabled

### Substrate Encoding
- Grid size: 8×8 = 64 cells
- Position: 2 dims
- **substrate.get_observation_dim()**: 64 + 2 = **66 dims**

### Meters: **8 dims**
### Affordance: **15 dims**
### Temporal: **4 dims**

**NOTE**: Temporal mechanics adds operating hours and multi-tick interactions, but does NOT change observation dimensions. Time features are always present.

### Total Calculation
```
66 (substrate) + 8 (meters) + 15 (affordance) + 4 (temporal) = 93 dims ✓
```

### Variable Count Validation
- grid_encoding: 64 dims
- position: 2 dims
- 8 meters: 8 dims
- affordance_at_position: 15 dims
- 4 temporal: 4 dims
**TOTAL**: 64 + 2 + 8 + 15 + 4 = **93 dims** ✓

---

## Summary Table

| Config | Grid | Observability | Substrate | Meters | Affordance | Temporal | **Total** | **Expected** | **Match** |
|--------|------|---------------|-----------|--------|------------|----------|-----------|--------------|-----------|
| L0_0_minimal | 3×3 | Full | 11 | 8 | 15 | 4 | **38** | 38 | ✓ |
| L0_5_dual_resource | 7×7 | Full | 51 | 8 | 15 | 4 | **78** | 78 | ✓ |
| L1_full_observability | 8×8 | Full | 66 | 8 | 15 | 4 | **93** | 93 | ✓ |
| L2_partial_observability | 8×8 | POMDP | 25+2 | 8 | 15 | 4 | **54** | 54 | ✓ |
| L3_temporal_mechanics | 8×8 | Full | 66 | 8 | 15 | 4 | **93** | 93 | ✓ |

**All calculations match expected dimensions!** ✓

---

## Key Insights

### Insight 1: Grid Size Affects Dimensions (Full Observability)
Full observability configs have different dimensions based on grid size:
- 3×3 grid → 38 dims
- 7×7 grid → 78 dims
- 8×8 grid → 93 dims

**Implication**: Checkpoint transfer only possible between configs with same grid size (e.g., L1 ↔ L3).

### Insight 2: POMDP Uses Local Window, Not Full Grid
POMDP observation includes:
- **Local window**: 25 cells (5×5 window around agent)
- **Position**: 2 dims (normalized, for spatial context)
- NOT the full 64-cell grid

**Implication**: POMDP dimensions (54) much smaller than full observability (93) despite same grid size.

### Insight 3: Temporal Mechanics Don't Change Dimensions
L3 has same dimensions as L1 (both 93) because:
- Temporal features (time_sin, time_cos) are ALWAYS present in all configs
- Operating hours and multi-tick interactions are runtime behavior, not observation changes

---

## Critical Validation Points for Cycle 5

When implementing VFS (Cycles 1-3) and testing regression (Cycle 5), ensure:

1. **VFS dimension calculation matches these manual calculations**
   ```python
   vfs_dim = sum(field.shape_size for field in obs_spec.fields)
   assert vfs_dim == expected_dim  # Must match!
   ```

2. **Variable shape inference correct**
   - vecNf with dims=N → contributes N to observation_dim
   - scalar → contributes 1 to observation_dim

3. **Substrate encoding handled correctly**
   - Full observability: grid_cells + position_dims
   - POMDP: window_cells (no full grid) + position_dims

4. **All configs tested**
   - Regression tests MUST cover all 5 configs
   - Any mismatch breaks checkpoint compatibility

---

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Status**: Manual calculations complete, ready for Cycle 0.4 verification
