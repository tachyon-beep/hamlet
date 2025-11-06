# Substrate System Architecture

## Topology Metadata

**Concept**: Topology describes the **connectivity pattern** of spatial substrates - how positions/cells connect to their neighbors.

### Grid Substrates

Grid substrates have discrete topology based on dimensionality:

- **Grid2D**: `topology = "square"` (4-connected 2D Cartesian grid)
  - Each cell connects to 4 neighbors: UP, DOWN, LEFT, RIGHT
  - Manhattan distance matches connectivity (L1 norm)

- **Grid3D**: `topology = "cubic"` (6-connected 3D Cartesian grid)
  - Each cell connects to 6 neighbors: ±X, ±Y, ±Z
  - Manhattan distance matches connectivity

- **GridND**: `topology = "hypercube"` (2N-connected N-dimensional grid)
  - Each cell connects to 2N neighbors (±1 per dimension)
  - Topology is dimension-agnostic (same for 4D, 7D, 100D)
  - Frontend checks `position_dim` for dimension-specific rendering

### Continuous Substrates

Continuous substrates (1D/2D/3D/ND) have **no discrete topology** - positions are continuous floats, not discrete cells. Topology field is **omitted** from metadata.

### Aspatial Substrates

Aspatial substrates have **no spatial structure** at all. Topology field is **omitted** from metadata.

### WebSocket Metadata Contract

**Grid substrates** (Grid2D/Grid3D/GridND):
```json
{
  "type": "grid2d",
  "position_dim": 2,
  "topology": "square",
  "width": 8,
  "height": 8,
  "boundary": "clamp",
  "distance_metric": "manhattan"
}
```

**Continuous substrates** (Continuous1D/2D/3D/ND):
```json
{
  "type": "continuous2d",
  "position_dim": 2,
  "bounds": [[0.0, 10.0], [0.0, 10.0]],
  "boundary": "clamp",
  "movement_delta": 0.5,
  "interaction_radius": 0.8,
  "distance_metric": "euclidean"
}
```
**Note**: No `topology` field - continuous spaces have no discrete connectivity.

**Aspatial substrate**:
```json
{
  "type": "aspatial",
  "position_dim": 0
}
```
**Note**: No `topology` field - no spatial structure.

### Configuration

Topology is configured in `substrate.yaml`:

**Grid2D/Grid3D** (`type: grid`):
```yaml
type: "grid"
grid:
  topology: "square"  # or "cubic" for 3D
  width: 8
  height: 8
  boundary: "clamp"
  distance_metric: "manhattan"
  observation_encoding: "relative"
```

**GridND** (`type: gridnd`):
```yaml
type: "gridnd"
gridnd:
  dimension_sizes: [5, 5, 5, 5, 5, 5, 5]
  topology: "hypercube"  # Always hypercube for N-dimensional grids
  boundary: "clamp"
  distance_metric: "manhattan"
  observation_encoding: "relative"
```

**Continuous substrates**: No topology field (not applicable).

### Implementation Flow

1. **Configuration** (`substrate.yaml`) → declares topology
2. **Config Schema** (`GridConfig`, `GridNDConfig`) → validates topology
3. **Factory** (`SubstrateFactory.build()`) → passes topology to substrate constructor
4. **Substrate** (Grid2D/Grid3D/GridND) → stores as `.topology` attribute
5. **WebSocket** (`live_inference._build_substrate_metadata()`) → reads `.topology` if present
6. **Frontend** → renders based on topology (square grid, cubic grid, etc.)

### Future Extensions

To add new topologies (e.g., simplex lattice, BCC lattice):

1. Add to `Literal["square", "cubic", "hypercube", "simplex"]` in config schema
2. Update factory to handle new topology value
3. Update substrate to accept new topology parameter
4. Update frontend to render new topology

Example:
```python
# Config
topology: Literal["square", "cubic", "hypercube", "simplex", "bcc"]

# Factory
if config.grid.topology == "simplex":
    return SimplexGridSubstrate(...)

# Substrate
class SimplexGridSubstrate(SpatialSubstrate):
    def __init__(self, ..., topology="simplex"):
        self.topology = topology
```
