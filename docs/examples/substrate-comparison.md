# Substrate Configuration Comparison

Quick reference for comparing different substrate configurations.

## Boundary Modes

| Boundary | Behavior | Use Cases |
|----------|----------|-----------|
| `clamp` | Agent position clamped to grid edges (hard walls) | Standard spatial planning, clear boundaries |
| `wrap` | Toroidal wraparound (Pac-Man style) | Infinite grid feel, no corners |
| `bounce` | Elastic reflection (agent bounces back from boundary) | Realistic physics, collision response |
| `sticky` | Agent stays in place when hitting boundary | Similar to clamp but different semantics |

## Distance Metrics

| Metric | Formula | Characteristics |
|--------|---------|-----------------|
| `manhattan` | \|x1-x2\| + \|y1-y2\| | L1 norm, matches 4-directional movement |
| `euclidean` | sqrt((x1-x2)² + (y1-y2)²) | L2 norm, straight-line distance |
| `chebyshev` | max(\|x1-x2\|, \|y1-y2\|) | L∞ norm, 8-directional movement |

## Observation Dimensions

| Grid Size | Grid Encoding | Total Obs Dim (with meters) |
|-----------|---------------|------------------------------|
| 3×3 | 9 | 36 (9 + 8 + 15 + 4) |
| 7×7 | 49 | 76 (49 + 8 + 15 + 4) |
| 8×8 | 64 | 91 (64 + 8 + 15 + 4) |
| Aspatial | 0 | 27 (0 + 8 + 15 + 4) |

**Components**:
- Grid encoding: width × height
- Meters: 8 (energy, health, satiation, money, mood, social, fitness, hygiene)
- Affordances: 15 (14 affordances + "none")
- Temporal: 4 (time_of_day, retirement_age, interaction_progress, interaction_ticks)

## Examples

See example files:
- `substrate-toroidal-grid.yaml` - Wrap boundary (Pac-Man style)
- `substrate-aspatial.yaml` - No positioning (pure resource management)
- `substrate-euclidean-distance.yaml` - Diagonal-aware proximity
