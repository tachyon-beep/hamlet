# Research: Observation Encoding Strategy (One-Hot vs Coordinates)

## Executive Summary

**Question**: As HAMLET moves toward configurable spatial substrates (TASK-000), which position encoding strategy should be used for observations: one-hot grid encoding vs coordinate encoding?

**Answer**: **Hybrid approach with auto-selection** - Use coordinate encoding for large grids (≥16×16) and 3D substrates, one-hot for small grids (≤8×8), and no encoding for aspatial substrates. Allow substrate config to override.

**Key Insight**: Coordinate encoding is **critical for 3D substrates** (one-hot would require 512+ dims for 8×8×3) and enables **transfer learning** across grid sizes (same obs_dim). One-hot is acceptable for small 2D grids where network can learn per-cell patterns.

**Impact**: **CRITICAL** - Without coordinate encoding, 3D substrates are infeasible. This research unblocks TASK-000 Phase 2 (3D cubic grids).

---

## Problem Statement

### Current Implementation: One-Hot Encoding

The current observation encoding uses **one-hot encoding** for grid position:

```python
# src/townlet/environment/observation_builder.py
# Full observability: Grid position as one-hot
flat_indices = positions[:, 1] * self.grid_size + positions[:, 0]
grid_encoding = torch.zeros(self.num_agents, self.grid_size * self.grid_size, device=self.device)
grid_encoding.scatter_(1, flat_indices.unsqueeze(1), 1.0)

# Partial observability: Uses normalized coordinates!
normalized_positions = positions.float() / (self.grid_size - 1)
```

**Current observation dimensions:**

- **L0_minimal (3×3)**: 36 dims (9 grid + 8 meters + 15 affordances + 4 temporal)
- **L0_5_dual_resource (7×7)**: 76 dims (49 grid + 8 meters + 15 affordances + 4 temporal)
- **L1_full_observability (8×8)**: 91 dims (64 grid + 8 meters + 15 affordances + 4 temporal)
- **L2_partial_observability (5×5 window)**: 54 dims (25 local grid + 2 position + 8 meters + 15 affordances + 4 temporal)

**Key observation**: L2 POMDP **already uses coordinate encoding** (normalized_positions) instead of one-hot!

### The Explosion Problem

One-hot encoding **explodes** for larger substrates:

| Substrate Type | Grid Size | One-Hot Dims | Coordinate Dims | Ratio |
|----------------|-----------|--------------|-----------------|-------|
| 2D small | 8×8 | 64 | 2 | 32× |
| 2D medium | 16×16 | 256 | 2 | 128× |
| 2D large | 32×32 | 1024 | 2 | 512× |
| 3D small | 8×8×3 | **512** | 3 | 170× |
| 3D medium | 8×8×8 | **512** | 3 | 170× |
| 3D large | 16×16×4 | **1024** | 3 | 341× |
| Hexagonal | 8×8 (axial) | 64 | 2 | 32× |
| Graph | 16 nodes | 16 | 1 | 16× |
| Aspatial | N/A | 0 | 0 | N/A |

**Critical insight**: 3D substrates are **infeasible** with one-hot encoding. An 8×8×3 grid requires **512 dimensions** just for position!

---

## Design Space: Encoding Strategies

### Option A: One-Hot Encoding (Current for Full Obs)

**Implementation**:

```python
def encode_position_onehot(positions: torch.Tensor, grid_size: int) -> torch.Tensor:
    """One-hot encoding of grid position.
    
    Args:
        positions: [batch, 2] (x, y) coordinates
        grid_size: Size of square grid
    
    Returns:
        [batch, grid_size²] one-hot vectors
    """
    flat_indices = positions[:, 1] * grid_size + positions[:, 0]
    one_hot = torch.zeros((positions.shape[0], grid_size * grid_size))
    one_hot.scatter_(1, flat_indices.unsqueeze(1), 1)
    return one_hot
```

**Pros**:

- Discrete representation - each cell is independent feature
- Network can learn per-cell patterns (e.g., "bottom-right corner is dangerous")
- No continuous values - network doesn't need to learn spatial relationships
- Easy to interpret - activation directly corresponds to grid cell

**Cons**:

- **Explodes for large grids**: 32×32 = 1024 dims
- **Infeasible for 3D**: 8×8×3 = 512 dims, 16×16×4 = 1024 dims
- **No transfer learning**: Network trained on 8×8 can't work on 16×16 (different obs_dim)
- **Doesn't generalize**: Network must learn spatial structure from scratch
- **Sparse activation**: Only 1 dimension is active at a time (inefficient)

**When to use**:

- Small 2D grids (≤8×8 = 64 dims)
- Tasks where per-cell learning is valuable (e.g., each cell has unique properties)
- Baseline for comparison

### Option B: Coordinate Encoding (Normalized Floats)

**Implementation**:

```python
def encode_position_coords(positions: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Coordinate encoding with normalized floats.
    
    Args:
        positions: [batch, 2] (x, y) coordinates
        grid_size: Size of square grid
    
    Returns:
        [batch, 2] normalized coordinates in [0, 1]
    """
    return positions.float() / (grid_size - 1)
```

**Pros**:

- **Compact**: Always 2 dims for 2D, 3 dims for 3D (regardless of grid size!)
- **Scales to any size**: 8×8 and 32×32 both use 2 dims
- **Enables transfer learning**: Network trained on 8×8 works on 16×16 (same obs_dim)
- **3D-ready**: 8×8×3 uses only 3 dims (vs 512 for one-hot)
- **Dense activation**: All dimensions convey information

**Cons**:

- **Network must learn spatial reasoning**: Must understand (0.5, 0.5) is center
- **Continuous values**: Network needs capacity to process floats
- **No per-cell patterns**: Can't learn "cell (3, 5) has special property"
- **May need deeper networks**: Spatial reasoning requires more layers

**When to use**:

- Large 2D grids (≥16×16)
- All 3D grids (cubic, etc.)
- When transfer learning across grid sizes is desired
- Hexagonal grids (axial coordinates q, r)

**Empirical evidence**: L2 POMDP **already uses this** successfully! RecurrentSpatialQNetwork processes normalized coordinates.

### Option C: Learned Position Embeddings

**Implementation**:

```python
class LearnedPositionEmbedding(nn.Module):
    def __init__(self, grid_size: int, embed_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(grid_size * grid_size, embed_dim)
    
    def forward(self, positions: torch.Tensor, grid_size: int) -> torch.Tensor:
        flat_indices = positions[:, 1] * grid_size + positions[:, 0]
        return self.embedding(flat_indices)
```

**Pros**:

- **Flexible dimensionality**: Can choose embed_dim (e.g., 32) independent of grid size
- **Learned spatial structure**: Network learns similarity between nearby cells
- **Better than one-hot for large grids**: 32 dims vs 1024 dims for 32×32
- **Can capture patterns**: Embeddings can encode "corner", "edge", "center" concepts

**Cons**:

- **Requires training**: Must learn embeddings before Q-learning (pre-training overhead)
- **No transfer across grid sizes**: 8×8 embedding table doesn't work for 16×16
- **More parameters**: Adds parameters to network (grid_size² × embed_dim)
- **Still tied to specific grid**: Can't use same network for different substrates
- **Opaque**: Hard to interpret what embeddings have learned

**When to use**:

- Large 2D grids where coordinate encoding isn't working
- When you have time/data for pre-training embeddings
- Research into spatial representation learning

**Research evidence**: Some RL papers (ICLR 2023 SNAC) use pre-trained position estimation modules with DRQNs. Shows promise but adds complexity.

### Option D: Fourier/Sinusoidal Position Encoding

**Implementation**:

```python
def encode_position_fourier(positions: torch.Tensor, grid_size: int, num_freqs: int = 10) -> torch.Tensor:
    """Fourier position encoding (inspired by Transformers, NeRF).
    
    Args:
        positions: [batch, 2] (x, y) coordinates
        grid_size: Size of square grid
        num_freqs: Number of frequency bands
    
    Returns:
        [batch, 4 * num_freqs] encoded positions
            For each dim (x, y): [sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^(num_freqs-1) * pi * x), cos(2^(num_freqs-1) * pi * x)]
    """
    normalized = positions.float() / (grid_size - 1)  # [0, 1]
    freqs = 2.0 ** torch.arange(num_freqs, device=positions.device) * torch.pi
    
    encoded = []
    for i in range(normalized.shape[1]):  # x, y
        coord = normalized[:, i:i+1]  # [batch, 1]
        angles = coord * freqs  # [batch, num_freqs]
        encoded.append(torch.sin(angles))
        encoded.append(torch.cos(angles))
    
    return torch.cat(encoded, dim=1)  # [batch, 4 * num_freqs]
```

**Pros**:

- **Fixed dimensionality**: Always 4 × num_freqs dims (e.g., 40 for num_freqs=10)
- **Captures spatial frequencies**: sin/cos encode position at multiple scales
- **No learnable parameters**: Deterministic encoding
- **Proven in other domains**: NeRF, Transformers use this successfully
- **Better than raw coordinates**: More informative for networks

**Cons**:

- **Higher dimensional than coordinates**: 40 dims vs 2 dims
- **Hyperparameter tuning**: Need to choose num_freqs
- **Overkill for grids?**: Spatial frequencies matter more for continuous spaces
- **Complexity**: Harder to implement and debug

**When to use**:

- Continuous substrates (smooth movement)
- Very large grids where network struggles with raw coordinates
- Research into representation learning

**Research evidence**: NeurIPS 2024, NeRF papers show sinusoidal encoding improves spatial reasoning. Worth exploring for continuous substrates.

### Option E: No Position Encoding (Aspatial)

**Implementation**:

```python
def encode_position_aspatial() -> torch.Tensor:
    """No position encoding for aspatial substrates.
    
    Returns:
        [batch, 0] empty tensor
    """
    return torch.zeros((batch_size, 0))
```

**Pros**:

- **Simplest**: No position = no encoding
- **Reveals deep insight**: Position is optional, meters are fundamental
- **Forces focus on resource management**: No spatial navigation crutch

**Cons**:

- **Only for aspatial substrates**: Doesn't apply to grids

**When to use**:

- Aspatial substrates (pure state machines)
- Economic simulations without spatial component

---

## Tradeoffs Analysis

### Dimension Scaling

| Encoding | 2D (8×8) | 2D (16×16) | 2D (32×32) | 3D (8×8×3) | 3D (16×16×4) |
|----------|----------|------------|------------|------------|--------------|
| One-hot | 64 | 256 | 1024 | **512** | **1024** |
| Coordinates | 2 | 2 | 2 | 3 | 3 |
| Learned (32d) | 32 | 32 | 32 | 32 | 32 |
| Fourier (10 freqs) | 40 | 40 | 40 | 60 | 60 |

**Verdict**: Coordinate encoding **wins on scalability**. 3D substrates are infeasible with one-hot.

### Transfer Learning

Can a network trained on 8×8 grid work on 16×16 grid?

| Encoding | Transfer Possible? | Notes |
|----------|-------------------|-------|
| One-hot | ❌ No | obs_dim changes (64 → 256), incompatible |
| Coordinates | ✅ Yes | obs_dim stays 2, network generalizes |
| Learned | ❌ No | Embedding table size changes |
| Fourier | ✅ Yes | Fixed num_freqs, works across sizes |
| Aspatial | ✅ N/A | No position, always transfers |

**Verdict**: Coordinate/Fourier encoding **enable curriculum progression** where agent trains on small grid then transfers to larger grid.

### Network Capacity Required

How much network capacity is needed to learn spatial reasoning?

| Encoding | Capacity | Rationale |
|----------|----------|-----------|
| One-hot | Low | Direct lookup, no reasoning needed |
| Coordinates | **Medium** | Must learn (0.5, 0.5) = center, edges at 0/1 |
| Learned | Medium | Embeddings capture structure, Q-head still learns |
| Fourier | Medium | Frequency decomposition helps network |

**Empirical evidence**: RecurrentSpatialQNetwork (L2 POMDP) uses **coordinates** with 256-dim LSTM. Works successfully! Suggests medium capacity is sufficient.

### Training Difficulty

| Encoding | Exploration Burden | Learning Speed |
|----------|-------------------|----------------|
| One-hot | High | Slower (sparse) |
| Coordinates | Medium | Faster (dense) |
| Learned | Medium | Moderate (pre-train then learn) |
| Fourier | Medium | Faster (informative) |

**Hypothesis**: Coordinate encoding may actually **speed up learning** because network gets dense signals about position rather than sparse one-hot.

---

## Recommendation: Hybrid Approach with Auto-Selection

### Strategy

**Substrate automatically selects encoding based on type and size:**

```yaml
# substrate.yaml
substrate:
  type: "grid"
  grid:
    topology: "square"
    dimensions: [8, 8]
    boundary: "clamp"
    distance_metric: "manhattan"
    position_encoding: "auto"  # auto, onehot, coords, fourier, learned, none
```

**Auto-selection logic:**

1. **Aspatial substrate** → `encoding = "none"` (0 dims)
2. **2D grid ≤ 8×8** → `encoding = "onehot"` (64 dims, keeps current behavior)
3. **2D grid > 8×8** → `encoding = "coords"` (2 dims, prevents explosion)
4. **3D grid (any size)** → `encoding = "coords"` (3 dims, only feasible option)
5. **Hexagonal grid** → `encoding = "coords"` (2 dims, axial coords q,r)
6. **Graph substrate** → `encoding = "coords"` (1 dim, node ID normalized)
7. **Continuous substrate** → `encoding = "fourier"` (40 dims, better for continuous)

**Override**: Operators can override auto-selection (e.g., `position_encoding: "fourier"`) for experimentation.

### Implementation Sketch

```python
# src/townlet/environment/substrate.py

class SpatialSubstrate(ABC):
    @abstractmethod
    def get_position_encoding_strategy(self) -> str:
        """Return encoding strategy: 'onehot', 'coords', 'fourier', 'learned', 'none'."""
        pass
    
    @abstractmethod
    def get_position_dim(self) -> int:
        """Return dimensionality of position vectors (2 for 2D, 3 for 3D, 0 for aspatial)."""
        pass
    
    @abstractmethod
    def get_observation_dim(self) -> int:
        """Return dimensionality of position encoding in observations."""
        pass
    
    @abstractmethod
    def encode_position(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode positions into observation space.
        
        Args:
            positions: [batch, position_dim] raw positions
        
        Returns:
            [batch, observation_dim] encoded positions
        """
        pass


class SquareGridSubstrate(SpatialSubstrate):
    def __init__(self, width: int, height: int, boundary: str = "clamp", 
                 position_encoding: str = "auto"):
        self.width = width
        self.height = height
        self.boundary = boundary
        
        # Auto-select encoding
        if position_encoding == "auto":
            if width * height <= 64:  # 8×8 or smaller
                self.position_encoding = "onehot"
            else:
                self.position_encoding = "coords"
        else:
            self.position_encoding = position_encoding
    
    def get_position_encoding_strategy(self) -> str:
        return self.position_encoding
    
    def get_position_dim(self) -> int:
        return 2  # (x, y)
    
    def get_observation_dim(self) -> int:
        if self.position_encoding == "onehot":
            return self.width * self.height
        elif self.position_encoding == "coords":
            return 2
        elif self.position_encoding == "fourier":
            return 40  # 2 dims × 10 freqs × 2 (sin/cos)
        else:
            raise ValueError(f"Unknown encoding: {self.position_encoding}")
    
    def encode_position(self, positions: torch.Tensor) -> torch.Tensor:
        if self.position_encoding == "onehot":
            return self._encode_onehot(positions)
        elif self.position_encoding == "coords":
            return self._encode_coords(positions)
        elif self.position_encoding == "fourier":
            return self._encode_fourier(positions)
        else:
            raise ValueError(f"Unknown encoding: {self.position_encoding}")
    
    def _encode_onehot(self, positions: torch.Tensor) -> torch.Tensor:
        flat_indices = positions[:, 1] * self.width + positions[:, 0]
        one_hot = torch.zeros((positions.shape[0], self.width * self.height), 
                             device=positions.device)
        one_hot.scatter_(1, flat_indices.unsqueeze(1), 1)
        return one_hot
    
    def _encode_coords(self, positions: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        normalized_x = positions[:, 0].float() / (self.width - 1)
        normalized_y = positions[:, 1].float() / (self.height - 1)
        return torch.stack([normalized_x, normalized_y], dim=1)
    
    def _encode_fourier(self, positions: torch.Tensor, num_freqs: int = 10) -> torch.Tensor:
        normalized = positions.float() / torch.tensor([self.width - 1, self.height - 1], 
                                                       device=positions.device)
        freqs = 2.0 ** torch.arange(num_freqs, device=positions.device) * torch.pi
        
        encoded = []
        for i in range(2):  # x, y
            coord = normalized[:, i:i+1]
            angles = coord * freqs
            encoded.append(torch.sin(angles))
            encoded.append(torch.cos(angles))
        
        return torch.cat(encoded, dim=1)


class CubicGridSubstrate(SpatialSubstrate):
    def __init__(self, width: int, height: int, depth: int, boundary: str = "clamp",
                 position_encoding: str = "auto"):
        self.width = width
        self.height = height
        self.depth = depth
        self.boundary = boundary
        
        # Force coordinate encoding for 3D (one-hot would be 512+ dims!)
        if position_encoding == "auto":
            self.position_encoding = "coords"
        elif position_encoding == "onehot":
            raise ValueError("One-hot encoding infeasible for 3D grids (512+ dims). Use 'coords' or 'fourier'.")
        else:
            self.position_encoding = position_encoding
    
    def get_position_dim(self) -> int:
        return 3  # (x, y, z)
    
    def get_observation_dim(self) -> int:
        if self.position_encoding == "coords":
            return 3
        elif self.position_encoding == "fourier":
            return 60  # 3 dims × 10 freqs × 2 (sin/cos)
        else:
            raise ValueError(f"Unknown encoding: {self.position_encoding}")
    
    def encode_position(self, positions: torch.Tensor) -> torch.Tensor:
        if self.position_encoding == "coords":
            normalized_x = positions[:, 0].float() / (self.width - 1)
            normalized_y = positions[:, 1].float() / (self.height - 1)
            normalized_z = positions[:, 2].float() / (self.depth - 1)
            return torch.stack([normalized_x, normalized_y, normalized_z], dim=1)
        elif self.position_encoding == "fourier":
            return self._encode_fourier(positions)
        else:
            raise ValueError(f"Unknown encoding: {self.position_encoding}")


class AspatialSubstrate(SpatialSubstrate):
    def get_position_dim(self) -> int:
        return 0  # No position!
    
    def get_observation_dim(self) -> int:
        return 0  # No encoding
    
    def encode_position(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.zeros((positions.shape[0], 0), device=positions.device)
```

### Migration Path

**Phase 1**: Keep current one-hot for existing configs (backward compatible)

- L0, L0.5, L1 continue using one-hot (≤8×8 grids)
- L2 POMDP already uses coordinates, no change needed

**Phase 2**: Add coordinate encoding for 3D substrates

- New config packs with 3D cubic grids auto-select coordinates
- Test transfer learning: Train on 8×8, evaluate on 16×16

**Phase 3**: Experiment with Fourier encoding for continuous substrates

- Create config pack with continuous substrate
- Compare Fourier vs coordinates for continuous control

---

## Transfer Learning Impact

### Scenario: Curriculum Progression Across Grid Sizes

**Current limitation**: Can't train on small grid then transfer to large grid (obs_dim mismatch).

**With coordinate encoding**:

1. Train agent on L0 (3×3 grid, 2-dim coords)
2. Transfer to L0.5 (7×7 grid, still 2-dim coords)
3. Transfer to L1 (8×8 grid, still 2-dim coords)
4. Transfer to L1_large (16×16 grid, still 2-dim coords)

Network architecture **never changes**! obs_dim stays constant.

**Pedagogical value**: Students see **true transfer learning** - agent learns spatial reasoning on small grid, applies to larger grid without retraining.

### Scenario: 2D → 3D Transfer

**Question**: Can agent trained on 2D grid transfer to 3D grid?

**With coordinate encoding**:

- 2D network: obs_dim = 2 (x, y) + 8 (meters) + 15 (affordances) + 4 (temporal) = 29
- 3D network: obs_dim = 3 (x, y, z) + 8 (meters) + 15 (affordances) + 4 (temporal) = 30

Almost compatible! Could:

1. Train on 2D with frozen z=0
2. Fine-tune on 3D by adding z dimension
3. See if spatial reasoning transfers

**With one-hot encoding**:

- 2D (8×8): obs_dim = 64 + 8 + 15 + 4 = 91
- 3D (8×8×3): obs_dim = 512 + 8 + 15 + 4 = 539

Completely incompatible. No transfer possible.

---

## Performance Considerations

### Network Forward Pass

**One-hot (8×8 = 64 dims)**:

- Input layer: 64 → 256 (16,384 weights)
- Sparse activation: Only 1 of 64 dims is 1.0
- Fast lookup, but sparse gradients

**Coordinate (2 dims)**:

- Input layer: 2 → 256 (512 weights)
- Dense activation: Both dims convey information
- 32× fewer input weights!

**Memory savings**: Coordinate encoding uses **32× less memory** for input layer.

### Training Time

**Hypothesis**: Coordinate encoding may train **faster** despite requiring spatial reasoning, because:

1. Fewer parameters (smaller network)
2. Dense gradients (all dims contribute)
3. Generalization across positions (learns "near edge" not "cell 17")

**Empirical test needed**: Compare L1 (one-hot) vs L1_coords (coordinates) on same task.

### Inference Speed

**One-hot**: Fast encoding (integer index → scatter)
**Coordinates**: Faster encoding (division only)

Both are negligible compared to network forward pass. No meaningful difference.

---

## Critical Questions Answered

### Q1: Is one-hot encoding a hard requirement?

**A**: No. L2 POMDP **already uses coordinate encoding** successfully (see `observation_builder.py` line 201). RecurrentSpatialQNetwork processes normalized positions without issues.

**Implication**: Current architecture does NOT require one-hot. Can switch to coordinates.

### Q2: Can networks learn spatial reasoning from coordinates?

**A**: Yes, with sufficient capacity. L2 POMDP uses 256-dim LSTM to process coordinates. Research (ICLR 2023 SNAC) shows pre-trained position estimation helps, but not required.

**Capacity recommendation**:

- Simple grids: 128-dim hidden layer sufficient
- Complex 3D: 256-dim LSTM recommended
- Continuous substrates: Consider Fourier encoding to help network

### Q3: Should encoding strategy be configurable or auto-selected?

**A**: **Both**. Auto-select for ease of use, but allow override for experimentation.

Default logic:

- Small 2D (≤8×8): one-hot (current behavior)
- Large 2D (>8×8): coords (prevents explosion)
- 3D (any): coords (only option)
- Continuous: fourier (better for continuous)
- Aspatial: none (no position)

Override: `position_encoding: "coords"` in substrate.yaml

---

## Estimated Effort

### Phase 1: Coordinate Encoding for Grid Substrates

- Add `encode_position()` to `SpatialSubstrate` interface: **1 hour**
- Implement coordinate encoding in `SquareGridSubstrate`: **1 hour**
- Implement coordinate encoding in `CubicGridSubstrate`: **1 hour**
- Update `ObservationBuilder` to use substrate encoding: **2 hours**
- **Total Phase 1: 5 hours**

### Phase 2: Auto-Selection Logic

- Add `position_encoding` field to substrate config schema: **1 hour**
- Implement auto-selection in substrate factory: **1 hour**
- Add config validation (reject one-hot for 3D): **1 hour**
- **Total Phase 2: 3 hours**

### Phase 3: Fourier Encoding (Optional)

- Implement Fourier encoding function: **2 hours**
- Test on continuous substrate: **2 hours**
- **Total Phase 3: 4 hours**

### Phase 4: Testing & Validation

- Test coordinate encoding on L1 (compare to one-hot): **3 hours**
- Test 3D substrate with coordinate encoding: **3 hours**
- Test transfer learning (8×8 → 16×16): **2 hours**
- **Total Phase 4: 8 hours**

### Total Effort: 20 hours (16 hours required, 4 hours optional)

---

## Priority & Value

### Priority: **CRITICAL**

**Blocks**: TASK-000 Phase 2 (3D cubic grids)

Without coordinate encoding, 3D substrates require 512+ dims (infeasible). This research **unblocks** the highest-value substrate type.

### Value: **VERY HIGH**

**Enables**:

1. **3D cubic grids** (8×8×3, 16×16×4) - multi-story buildings, Minecraft-like
2. **Large 2D grids** (16×16, 32×32) - bigger worlds
3. **Transfer learning** - train on small grid, deploy on large grid
4. **Memory efficiency** - 32× fewer input parameters
5. **Curriculum progression** - L0 (3×3) → L1 (8×8) → L1_large (16×16) with same network

**Pedagogical value**:

- Students see transfer learning in action
- Teaches spatial reasoning vs memorization
- Reveals position encoding choices (not invisible)

---

## Recommendations

### Immediate Actions (Before TASK-000 Phase 2)

1. **Add `encode_position()` to substrate interface** - Required for 3D
2. **Implement coordinate encoding in SquareGrid** - Test backward compatibility
3. **Validate L2 POMDP still works** - Already uses coordinates
4. **Test 3D substrate with coordinates** - Prove feasibility

### Short-Term (TASK-000 Phase 2)

5. **Add auto-selection logic** - Easy of use for operators
6. **Create L1_coords config** - Compare one-hot vs coords on same task
7. **Test transfer learning** - L0 (3×3) → L1 (8×8)

### Long-Term (Post TASK-000)

8. **Experiment with Fourier encoding** - Continuous substrates
9. **Research learned embeddings** - When coordinates aren't sufficient
10. **Document encoding strategies** - Add to HAMLET teaching materials

---

## Conclusion

**One-hot encoding is a legacy artifact from small 2D grids.** It prevents 3D substrates, wastes memory, and blocks transfer learning.

**Coordinate encoding is the path forward.** It's compact (2-3 dims), scalable (any grid size), and enables transfer learning. The L2 POMDP already proves it works.

**Hybrid approach balances pragmatism and flexibility.** Keep one-hot for tiny grids (≤8×8) to preserve current behavior, use coordinates for everything else, allow override for experimentation.

**Impact**: This research **unblocks TASK-000 Phase 2** (3D cubic grids) and enables future curriculum levels (L4 multi-zone, L5 large worlds).

**Recommendation**: Implement coordinate encoding **before** 3D substrates. Test transfer learning as proof of concept.
