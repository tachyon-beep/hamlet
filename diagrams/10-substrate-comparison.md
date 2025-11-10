# Substrate Feature Comparison Matrix

## Substrate Type Overview

```mermaid
graph TB
    substrate_base["Substrate<br/>Abstract Base Class"]

    discrete["Discrete Substrates<br/>Integer positions"]
    continuous_sub["Continuous Substrates<br/>Float positions"]

    grid2d["Grid2D<br/>2D square/hex grid"]
    grid3d["Grid3D<br/>3D cubic grid"]
    gridnd["GridND<br/>4D-100D grid"]

    continuous1d["Continuous<br/>1D/2D/3D space"]
    continuousnd["ContinuousND<br/>4D-100D space"]

    aspatial["Aspatial<br/>No position concept"]

    substrate_base --> discrete
    substrate_base --> continuous_sub
    substrate_base --> aspatial

    discrete --> grid2d
    discrete --> grid3d
    discrete --> gridnd

    continuous_sub --> continuous1d
    continuous_sub --> continuousnd

    style substrate_base fill:#fff9c4
    style discrete fill:#e1f5fe
    style continuous_sub fill:#c8e6c9
    style aspatial fill:#f3e5f5
```

## Feature Comparison Table

| Feature | Grid2D | Grid3D | GridND | Continuous | ContinuousND | Aspatial |
|---------|--------|--------|--------|------------|--------------|----------|
| **Position Dim** | 2 | 3 | 4-100 | 1-3 | 4-100 | 0 |
| **Position Type** | Integer | Integer | Integer | Float | Float | N/A |
| **Topologies** | square, hex | cubic | hypercube | euclidean | euclidean | N/A |
| **Boundaries** | clamp, wrap, bounce, sticky | clamp, wrap | clamp, wrap | clamp, wrap, bounce | clamp, wrap | N/A |
| **Distance Metrics** | manhattan, euclidean, chebyshev | manhattan, euclidean | manhattan, euclidean | euclidean | euclidean | 0.0 |
| **POMDP Support** | ✅ Yes | ✅ Yes (limited) | ❌ No (window too large) | ✅ Yes | ❌ No | ✅ Special case |
| **Action Space** | 6-8 actions | 8-10 actions | 2N+2 actions | 2-6 actions | 2N+2 actions | 2 actions |
| **Observation Encoding** | relative, scaled, absolute | relative, scaled, absolute | relative | coordinate | coordinate | N/A |
| **Transfer Learning** | ✅ Full support | ✅ Full support | ⚠️ Position dim varies | ⚠️ Position dim varies | ❌ Dimension-specific | ✅ No position |
| **Typical Use Case** | Standard RL | 3D navigation | High-dim optimization | Physics simulation | ML optimization | Resource management |
| **Example Config** | L1_full_observability | None (future) | None (future) | None (future) | None (future) | None (future) |

## Grid2D Detailed Features

```mermaid
graph TB
    grid2d_root["Grid2DSubstrate"]

    subgraph "Topologies"
        square["Square Grid<br/>4-connected or 8-connected"]
        hex["Hex Grid<br/>Axial coordinates (q, r)"]
    end

    subgraph "Boundary Modes"
        clamp["clamp: Hard walls<br/>position clamped to valid range"]
        wrap["wrap: Toroidal<br/>edges connect (Pac-Man)"]
        bounce["bounce: Elastic<br/>reflect off boundaries"]
        sticky["sticky: One-way<br/>can't leave once at edge"]
    end

    subgraph "Distance Metrics"
        manhattan["manhattan (L1)<br/>|x1-x2| + |y1-y2|"]
        euclidean["euclidean (L2)<br/>sqrt((x1-x2)² + (y1-y2)²)"]
        chebyshev["chebyshev (L∞)<br/>max(|x1-x2|, |y1-y2|)"]
    end

    subgraph "Observation Encodings"
        relative["relative: [0,1] normalized<br/>Best for transfer learning"]
        scaled["scaled: normalized + metadata<br/>Network learns grid size"]
        absolute["absolute: raw coords<br/>For physical simulation"]
    end

    subgraph "Action Space (Square)"
        up["UP: [0, -1]"]
        down["DOWN: [0, 1]"]
        left["LEFT: [-1, 0]"]
        right["RIGHT: [1, 0]"]
        interact["INTERACT: special"]
        wait["WAIT: [0, 0]"]
        custom["Custom actions (e.g., REST)"]
    end

    grid2d_root --> square
    grid2d_root --> hex

    grid2d_root --> clamp
    grid2d_root --> wrap
    grid2d_root --> bounce
    grid2d_root --> sticky

    grid2d_root --> manhattan
    grid2d_root --> euclidean
    grid2d_root --> chebyshev

    grid2d_root --> relative
    grid2d_root --> scaled
    grid2d_root --> absolute

    grid2d_root --> up
    grid2d_root --> down
    grid2d_root --> left
    grid2d_root --> right
    grid2d_root --> interact
    grid2d_root --> wait
    grid2d_root --> custom

    style grid2d_root fill:#e1f5fe
    style relative fill:#c8e6c9
    style up fill:#fff9c4
```

## POMDP Compatibility

```mermaid
graph TB
    pomdp_root["POMDP Support Matrix"]

    subgraph "Supported Substrates"
        grid2d_pomdp["Grid2D<br/>✅ Full support<br/>5×5 window (vision_range=2)"]
        grid3d_pomdp["Grid3D<br/>✅ Limited support<br/>3×3×3 window max"]
        aspatial_pomdp["Aspatial<br/>✅ Special case<br/>No spatial window needed"]
    end

    subgraph "Not Supported"
        gridnd_no["GridND (N≥4)<br/>❌ Window too large<br/>2N+1 per dim = exponential"]
        continuous_no["Continuous<br/>❌ Window concept unclear<br/>Radius-based instead?"]
        continuousnd_no["ContinuousND<br/>❌ Same as GridND"]
    end

    subgraph "Window Size Calculations"
        grid2d_calc["Grid2D: (2r+1)²<br/>r=2 → 5×5 = 25 cells"]
        grid3d_calc["Grid3D: (2r+1)³<br/>r=1 → 3×3×3 = 27 cells"]
        gridnd_calc["GridND: (2r+1)^N<br/>N=4, r=1 → 3⁴ = 81 cells<br/>N=7, r=1 → 3⁷ = 2187 cells!"]
    end

    pomdp_root --> grid2d_pomdp
    pomdp_root --> grid3d_pomdp
    pomdp_root --> aspatial_pomdp

    pomdp_root --> gridnd_no
    pomdp_root --> continuous_no
    pomdp_root --> continuousnd_no

    grid2d_pomdp --> grid2d_calc
    grid3d_pomdp --> grid3d_calc
    gridnd_no --> gridnd_calc

    style grid2d_pomdp fill:#c8e6c9
    style gridnd_no fill:#ffccbc
```

## Position Representation

```mermaid
graph TB
    subgraph "Grid2D Positions"
        grid2d_pos["Stored: Tensor[num_agents, 2]<br/>dtype: int64<br/>Example: [[2, 3], [4, 5]]"]
        grid2d_norm["Normalized: divide by grid_size<br/>[[2/8, 3/8], [4/8, 5/8]]<br/>= [[0.25, 0.375], [0.5, 0.625]]"]
    end

    subgraph "Grid3D Positions"
        grid3d_pos["Stored: Tensor[num_agents, 3]<br/>dtype: int64<br/>Example: [[2, 3, 1], [4, 5, 2]]"]
        grid3d_norm["Normalized: divide by dims<br/>[[2/8, 3/8, 1/4], ...]"]
    end

    subgraph "GridND Positions"
        gridnd_pos["Stored: Tensor[num_agents, N]<br/>dtype: int64<br/>N can be 4-100"]
        gridnd_norm["Normalized: per-dimension<br/>pos[i] / dimension_sizes[i]"]
    end

    subgraph "Continuous Positions"
        continuous_pos["Stored: Tensor[num_agents, dims]<br/>dtype: float32<br/>Example: [[1.5, 2.3], [4.1, 5.9]]"]
        continuous_norm["Normalized: (pos - min) / (max - min)<br/>Based on bounds"]
    end

    subgraph "Aspatial Positions"
        aspatial_pos["No positions!<br/>Tensor[num_agents, 0]<br/>Empty tensor"]
        aspatial_note["All agents at 'same' location<br/>is_adjacent() always returns True"]
    end

    grid2d_pos --> grid2d_norm
    grid3d_pos --> grid3d_norm
    gridnd_pos --> gridnd_norm
    continuous_pos --> continuous_norm

    style grid2d_pos fill:#e1f5fe
    style continuous_pos fill:#c8e6c9
    style aspatial_pos fill:#f3e5f5
```

## Action Space Comparison

```mermaid
graph TB
    subgraph "Grid2D Action Space"
        grid2d_actions["Total: 8 actions<br/>(6 substrate + 2 custom)"]
        grid2d_movement["Movement: 4 cardinal<br/>UP, DOWN, LEFT, RIGHT"]
        grid2d_special["Special: INTERACT, WAIT"]
        grid2d_custom["Custom: REST, MEDITATE"]
    end

    subgraph "Grid3D Action Space"
        grid3d_actions["Total: 10 actions<br/>(8 substrate + 2 custom)"]
        grid3d_movement["Movement: 6 directions<br/>±X, ±Y, ±Z"]
        grid3d_special["Special: INTERACT, WAIT"]
        grid3d_custom["Custom: REST, MEDITATE"]
    end

    subgraph "GridND Action Space"
        gridnd_actions["Total: 2N+2 actions<br/>N dimensions"]
        gridnd_movement["Movement: 2N directions<br/>±dim[i] for each dim"]
        gridnd_special["Special: INTERACT, WAIT"]
    end

    subgraph "Continuous Action Space"
        continuous_actions["Varies by dimensions<br/>1D: 4 actions<br/>2D: 6 actions<br/>3D: 8 actions"]
        continuous_movement["Movement: ±delta per axis<br/>delta from config"]
        continuous_special["Special: INTERACT, WAIT"]
    end

    subgraph "Aspatial Action Space"
        aspatial_actions["Total: 2 actions<br/>(0 movement + 2 custom)"]
        aspatial_no_move["No movement actions<br/>(no position concept)"]
        aspatial_custom["Custom: REST, MEDITATE<br/>OR INTERACT (affordances)"]
    end

    grid2d_actions --> grid2d_movement
    grid2d_actions --> grid2d_special
    grid2d_actions --> grid2d_custom

    grid3d_actions --> grid3d_movement
    grid3d_actions --> grid3d_special
    grid3d_actions --> grid3d_custom

    gridnd_actions --> gridnd_movement
    gridnd_actions --> gridnd_special

    continuous_actions --> continuous_movement
    continuous_actions --> continuous_special

    aspatial_actions --> aspatial_no_move
    aspatial_actions --> aspatial_custom

    style grid2d_actions fill:#e1f5fe
    style aspatial_actions fill:#f3e5f5
```

## Transfer Learning Compatibility

```mermaid
flowchart TD
    checkpoint["Checkpoint from Source Config"]

    check_substrate{"Same<br/>substrate type?"}
    check_position_dim{"Same<br/>position_dim?"}
    check_obs_dim{"Same<br/>observation_dim?"}

    compatible["✅ Compatible<br/>Can load checkpoint"]
    incompatible_substrate["❌ Incompatible<br/>Different substrate types"]
    incompatible_pos["❌ Incompatible<br/>Different position dimensions"]
    incompatible_obs["❌ Incompatible<br/>Different observation dimensions"]

    subgraph "Compatible Scenarios"
        grid_same["Grid2D → Grid2D<br/>✅ Even different grid sizes<br/>(using 'relative' encoding)"]
        aspatial_same["Aspatial → Aspatial<br/>✅ No position to worry about"]
    end

    subgraph "Incompatible Scenarios"
        grid_to_continuous["Grid2D → Continuous<br/>❌ Different substrate types"]
        grid2d_to_grid3d["Grid2D → Grid3D<br/>❌ position_dim: 2 → 3"]
        different_affordances["Different enabled_affordances<br/>❌ Different obs_dim"]
    end

    checkpoint --> check_substrate
    check_substrate -->|Yes| check_position_dim
    check_substrate -->|No| incompatible_substrate

    check_position_dim -->|Yes| check_obs_dim
    check_position_dim -->|No| incompatible_pos

    check_obs_dim -->|Yes| compatible
    check_obs_dim -->|No| incompatible_obs

    compatible --> grid_same
    compatible --> aspatial_same

    incompatible_substrate --> grid_to_continuous
    incompatible_pos --> grid2d_to_grid3d
    incompatible_obs --> different_affordances

    style compatible fill:#c8e6c9
    style incompatible_substrate fill:#ffccbc
```

## Performance Characteristics

### Memory Footprint (4 agents)

| Substrate | Position Storage | Typical Size |
|-----------|------------------|--------------|
| Grid2D | `[4, 2]` int64 | 64 bytes |
| Grid3D | `[4, 3]` int64 | 96 bytes |
| GridND (N=7) | `[4, 7]` int64 | 224 bytes |
| Continuous (3D) | `[4, 3]` float32 | 48 bytes |
| Aspatial | `[4, 0]` - | 0 bytes |

### Movement Computation Cost

| Substrate | Operation | Cost |
|-----------|-----------|------|
| Grid2D | Add delta, clamp bounds | O(1) per agent |
| Grid3D | Add delta, clamp bounds | O(1) per agent |
| GridND | Add delta, clamp N dims | O(N) per agent |
| Continuous | Add delta, check bounds | O(dims) per agent |
| Aspatial | No-op (always succeeds) | O(1) |

### Distance Computation

| Substrate | Metric | Formula |
|-----------|--------|---------|
| Grid2D | Manhattan | `|x1-x2| + |y1-y2|` |
| Grid2D | Euclidean | `sqrt((x1-x2)² + (y1-y2)²)` |
| Grid2D | Chebyshev | `max(|x1-x2|, |y1-y2|)` |
| Grid3D | Euclidean | `sqrt((x1-x2)² + (y1-y2)² + (z1-z2)²)` |
| Continuous | Euclidean | `sqrt(sum((p1[i]-p2[i])²))` |
| Aspatial | Always 0 | `0.0` |

## Use Case Recommendations

```mermaid
graph TB
    use_case["Use Case"]

    standard_rl["Standard RL Training<br/>Discrete navigation"]
    pomdp_training["POMDP / Memory Tasks<br/>Partial observability"]
    high_dim["High-Dimensional<br/>Optimization"]
    physics["Physics Simulation<br/>Continuous dynamics"]
    resource["Pure Resource<br/>Management"]

    recommend_grid2d["✅ Grid2D<br/>Most common choice<br/>Full tooling support"]
    recommend_grid2d_pomdp["✅ Grid2D + POMDP<br/>5×5 vision window<br/>LSTM network"]
    recommend_gridnd["✅ GridND<br/>Future: ML benchmarks"]
    recommend_continuous["✅ Continuous<br/>Future: robotics"]
    recommend_aspatial["✅ Aspatial<br/>No spatial component"]

    use_case --> standard_rl
    use_case --> pomdp_training
    use_case --> high_dim
    use_case --> physics
    use_case --> resource

    standard_rl --> recommend_grid2d
    pomdp_training --> recommend_grid2d_pomdp
    high_dim --> recommend_gridnd
    physics --> recommend_continuous
    resource --> recommend_aspatial

    style recommend_grid2d fill:#c8e6c9
    style recommend_grid2d_pomdp fill:#e1f5fe
    style recommend_aspatial fill:#fff9c4
```

## Configuration Examples

### Grid2D (Square, 8×8)
```yaml
type: "grid"
grid:
  topology: "square"
  width: 8
  height: 8
  boundary: "clamp"
  distance_metric: "manhattan"
  observation_encoding: "relative"
```

### Grid2D (Hex)
```yaml
type: "grid"
grid:
  topology: "hex"
  width: 10
  height: 10
  boundary: "wrap"
  distance_metric: "manhattan"
  observation_encoding: "relative"
```

### Grid3D (Cubic)
```yaml
type: "grid3d"
grid3d:
  width: 5
  height: 5
  depth: 5
  boundary: "clamp"
```

### GridND (7D)
```yaml
type: "gridnd"
gridnd:
  dimension_sizes: [3, 3, 3, 3, 3, 3, 3]
  boundary: "wrap"
```

### Continuous (2D)
```yaml
type: "continuous"
continuous:
  dimensions: 2
  bounds: [[0.0, 10.0], [0.0, 10.0]]
  movement_delta: 0.1
  interaction_radius: 0.5
```

### Aspatial
```yaml
type: "aspatial"
# No additional configuration needed
```

## Future Substrate Extensions

1. **Graph Substrate**: Arbitrary graph topology
2. **Hybrid Substrate**: Mixed discrete/continuous
3. **Dynamic Substrate**: Changing topology over time
4. **Multi-Scale Substrate**: Hierarchical grids
5. **Wrapped Continuous**: Toroidal continuous space
