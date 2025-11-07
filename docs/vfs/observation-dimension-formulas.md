# Current Observation Dimension Formulas

**Purpose**: Document existing observation dimension calculations to ensure VFS implementation maintains checkpoint compatibility.

**Source**: `src/townlet/environment/vectorized_env.py` lines 240-264

---

## Full Observability Formula

```python
obs_dim = substrate.get_observation_dim() + meter_count + (num_affordances + 1) + 4
```

### Components

1. **`substrate.get_observation_dim()`**: Substrate-specific grid + position encoding
   - **Grid2D (relative)**: grid_cells + 2 (e.g., 8×8 = 64 + 2 = 66)
   - **Grid2D (scaled)**: grid_cells + 4 (grid + normalized x, y, width, height)
   - **Grid2D (absolute)**: grid_cells + 2 (grid + raw x, y)
   - **Grid3D (relative)**: grid_cells + 3 (e.g., 8×8×8 = 512 + 3 = 515)
   - **Grid3D (scaled)**: grid_cells + 6 (grid + normalized x, y, z, width, height, depth)
   - **GridND (relative, N dims)**: product(dim_sizes) + N (grid + normalized coordinates)
   - **GridND (scaled, N dims)**: product(dim_sizes) + 2N (grid + normalized + sizes)
   - **Continuous (relative)**: dimensions (normalized positions, no grid)
   - **Continuous (scaled)**: dimensions * 2 (normalized + range sizes, no grid)
   - **Aspatial**: 0 (no position encoding)

2. **`meter_count`**: Always **8** meters
   - energy, health, satiation, money, mood, social, fitness, hygiene

3. **`num_affordances + 1`**: One-hot affordance encoding
   - **14 affordances** (vocabulary size) + 1 ("none") = **15 dimensions**
   - Vocabulary: Bed, Hospital, HomeMeal, Job, Gym, Shower, Bar, Restaurant, Theatre, Park, SuperMarket, Bank, FastFood, CoffeShop

4. **Temporal features**: Always **4** dimensions
   - `time_sin`: sin(2π * time_of_day / 24) - cyclical time encoding
   - `time_cos`: cos(2π * time_of_day / 24) - cyclical time encoding
   - `interaction_progress`: normalized ticks completed [0-1]
   - `lifetime_progress`: normalized lifetime [0-1] (0 = birth, 1 = retirement)

### Example: Grid2D (8×8, relative encoding)

```
obs_dim = 66 + 8 + 15 + 4 = 93 dimensions
```

Breakdown:
- Substrate: 66 dims (64 grid cells + 2 normalized position)
- Meters: 8 dims
- Affordance: 15 dims (one-hot)
- Temporal: 4 dims
- **Total**: 93 dims

**Note**: substrate.get_observation_dim() for Grid2D returns `grid_cells + position_dims`

---

## Partial Observability (POMDP) Formula

```python
obs_dim = window_size^position_dim + position_dim + meter_count + (num_affordances + 1) + 4
```

### Components

1. **`window_size^position_dim`**: Local vision window
   - `window_size = 2 * vision_range + 1`
   - For `vision_range=2`: window_size = 5 (5×5 window)
   - **Grid2D**: 5^2 = **25 dimensions** (5×5 local grid)
   - **Grid3D**: 5^3 = **125 dimensions** (5×5×5 local cube)

2. **`position_dim`**: Normalized position coordinates
   - **Grid2D**: 2 dims (normalized x, y)
   - **Grid3D**: 3 dims (normalized x, y, z)
   - **Note**: Always uses normalized positions (substrate.normalize_positions())

3. **`meter_count`**: Same as full observability - **8 dimensions**

4. **`num_affordances + 1`**: Same as full observability - **15 dimensions**

5. **Temporal features**: Same as full observability - **4 dimensions**

### Example: Grid2D POMDP (8×8 grid, vision_range=2)

```
obs_dim = 25 + 2 + 8 + 15 + 4 = 54 dimensions
```

Breakdown:
- Local window: 25 dims (5×5 grid)
- Position: 2 dims (normalized x, y)
- Meters: 8 dims
- Affordance: 15 dims (one-hot)
- Temporal: 4 dims
- **Total**: 54 dims

---

## Current Dimensions by Config

| Config | Observability | Grid | Vision Range | Substrate Encoding | Calculation | **Total Dims** |
|--------|---------------|------|--------------|-------------------|-------------|----------------|
| **L0_0_minimal** | Full | 3×3 Grid2D | N/A | relative | 11 + 8 + 15 + 4 | **38** |
| **L0_5_dual_resource** | Full | 7×7 Grid2D | N/A | relative | 51 + 8 + 15 + 4 | **78** |
| **L1_full_observability** | Full | 8×8 Grid2D | N/A | relative | 66 + 8 + 15 + 4 | **93** |
| **L2_partial_observability** | POMDP | 8×8 Grid2D | 2 (5×5) | relative | 25 + 2 + 8 + 15 + 4 | **54** |
| **L3_temporal_mechanics** | Full | 8×8 Grid2D | N/A | relative | 66 + 8 + 15 + 4 | **93** |

**Key Insight**: Observation dimensions **vary by grid size** for full observability! Only configs with the same grid size (L1 and L3, both 8×8) have identical dimensions (93). This means **checkpoint transfer is only possible between configs with the same grid size**.

---

## Implementation Details

### Where Dimensions Are Calculated

**File**: `src/townlet/environment/vectorized_env.py`

**Full Observability** (lines 256-260):
```python
self.observation_dim = (
    self.substrate.get_observation_dim() +
    meter_count +
    (self.num_affordance_types + 1)
)
self.observation_dim += 4  # Temporal features
```

**Partial Observability** (lines 241-254):
```python
window_size = 2 * vision_range + 1
local_window_dim = window_size**self.substrate.position_dim
self.observation_dim = (
    local_window_dim +
    self.substrate.position_dim +  # Normalized position
    meter_count +
    (self.num_affordance_types + 1)
)
self.observation_dim += 4  # Temporal features
```

### Substrate Position Encoding

**File**: `src/townlet/substrate/base.py` (abstract method)

Each substrate implements `get_observation_dim()`:
- **Grid2D**: Returns grid_cells + position_dims (e.g., 64 + 2 = 66 for 8×8 "relative")
  - Includes the full grid encoding (one cell per grid position)
  - Plus position coordinates (2 dims for relative, 4 for scaled)
- **Grid3D**: Returns grid_cells + position_dims (e.g., 512 + 3 = 515 for 8×8×8 "relative")
- **GridND**: Returns product(dim_sizes) + N (grid hypercube + normalized coordinates)
- **Continuous**: Returns position_dims only (e.g., 2 for 2D space, no grid)
- **Aspatial**: Returns 0 (no position encoding)

---

## Critical Constraints for VFS

### ⚠️ **CHECKPOINT COMPATIBILITY REQUIREMENT**

**VFS-generated observation dimensions MUST exactly match these calculations.**

Any deviation breaks checkpoint compatibility:
- Different `observation_dim` → Q-network input shape mismatch
- Q-network shape mismatch → cannot load saved checkpoints
- Cannot load checkpoints → **months of training lost**

### Validation Strategy

1. **Cycle 0** (current): Document formulas and expected dimensions
2. **Cycle 5**: Automated regression tests validate VFS matches environment
3. **Validation script**: `scripts/validate_vfs_dimensions.py` (created in Cycle 5)

---

## Notes for VFS Implementation

### Position Encoding (Cycles 1-3)

VFS variables for position:
```yaml
# Grid2D with relative encoding
- id: "position"
  scope: "agent"
  type: "vec2i"
  # ... exposes as 2 dims (normalized x, y)
```

### Meter Encoding (Cycles 1-3)

VFS variables for meters (8 scalar variables):
```yaml
- id: "energy"
  scope: "agent"
  type: "scalar"  # 1 dim
# ... repeat for health, satiation, money, mood, social, fitness, hygiene
```

### Affordance Encoding (Cycles 1-3)

VFS variable for affordance:
```yaml
- id: "affordance_at_position"
  scope: "agent"
  type: "categorical"  # NOT IMPLEMENTED IN PHASE 1
  # Phase 1 workaround: Use 15 separate boolean variables or single vec15i
  # ... exposes as 15 dims (one-hot encoding)
```

**Phase 1 Limitation**: Categorical type not implemented. Use:
- Option A: 15 boolean variables (energy_at_position, health_at_position, ...)
- Option B: Single fixed-size vector variable (type: vec15i)

### Temporal Encoding (Cycles 1-3)

VFS variables for temporal features (4 scalar variables):
```yaml
- id: "time_sin"
  scope: "global"
  type: "scalar"  # 1 dim

- id: "time_cos"
  scope: "global"
  type: "scalar"  # 1 dim

- id: "interaction_progress"
  scope: "agent"
  type: "scalar"  # 1 dim

- id: "lifetime_progress"
  scope: "agent"
  type: "scalar"  # 1 dim
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Cycle**: 0.1 (Observation Structure Reverse Engineering)
