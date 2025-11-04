# Research: Substrate-Agnostic Visualization Architecture

## Executive Summary

**Question**: Does TASK-000 (Configurable Spatial Substrates) require frontend visualization changes, or can it be deferred?

**Answer**: **DEFERRABLE** - Text-based visualization is sufficient for Phase 1 implementation and validation. Advanced substrate-specific rendering (3D, hex, graph) should be **separate follow-up projects** with significantly lower priority than core UAC infrastructure.

**Key Insight**: The primary value of TASK-000 is **backend flexibility** (training agents on alternative substrates), not real-time visualization. Students can experiment with 3D/hex/graph substrates using terminal output and post-training analysis before investing in complex rendering systems.

---

## Problem Statement

TASK-000 introduces multiple spatial substrate types beyond the current 2D square grid:

- **3D cubic grids**: Multi-story buildings (8×8×3)
- **Hexagonal grids**: 6-way movement with axial coordinates
- **Graph substrates**: Node-edge topologies (subway systems)
- **Aspatial substrates**: No spatial positioning (pure resource management)
- **Continuous spaces**: Smooth movement with float coordinates

**Current frontend assumes hardcoded 2D square grid**:

- `Grid.vue`: Renders 8×8 SVG grid with square cells
- Agent positions: `(x, y)` integer coordinates
- Affordances: Placed at `(x, y)` grid cells
- Cell rendering: Fixed 75×75px squares

**Challenge**: How to visualize alternative substrates without blocking TASK-000 implementation?

---

## Design Space: Visualization Strategies

### Option A: Substrate-Specific Vue Components

**Architecture**:

```
frontend/src/components/
  ├── Grid2D.vue          # Current square grid renderer
  ├── Grid3D.vue          # 3D cubic grid with floor selection
  ├── GridHex.vue         # Hexagonal tile renderer
  ├── GraphViz.vue        # Node-edge graph visualization
  └── AspatialDash.vue    # Meters-only dashboard (no spatial rendering)
```

**Substrate detection**:

```javascript
// App.vue
const substrateType = computed(() => store.substrateConfig?.type || 'grid')
const gridTopology = computed(() => store.substrateConfig?.grid?.topology || 'square')

// Render appropriate component
<Grid2D v-if="substrateType === 'grid' && gridTopology === 'square'" />
<Grid3D v-if="substrateType === 'grid' && gridTopology === 'cubic'" />
<GridHex v-if="substrateType === 'grid' && gridTopology === 'hexagonal'" />
<GraphViz v-if="substrateType === 'graph'" />
<AspatialDash v-if="substrateType === 'aspatial'" />
```

**Pros**:

- Clean separation of concerns (each renderer is self-contained)
- Easy to implement incrementally (add components as needed)
- No shared rendering complexity

**Cons**:

- Code duplication (agent markers, affordance icons, meters)
- 5+ separate components to maintain
- Higher total implementation effort

**Effort**:

- Grid3D: 12-16 hours (floor selector, camera controls, occlusion)
- GridHex: 8-12 hours (axial coordinate mapping, hex tile SVG)
- GraphViz: 10-14 hours (force-directed layout, edge rendering)
- AspatialDash: 2-4 hours (meters-only UI, no spatial rendering)
- **Total**: 32-46 hours

---

### Option B: Unified Component with Substrate Renderers

**Architecture**:

```javascript
// SubstrateVisualization.vue
<template>
  <component :is="rendererComponent" v-bind="rendererProps" />
</template>

<script setup>
const rendererComponent = computed(() => {
  switch (props.substrateType) {
    case 'grid':
      if (props.topology === 'cubic') return Grid3DRenderer
      if (props.topology === 'hexagonal') return HexRenderer
      return SquareGridRenderer
    case 'graph': return GraphRenderer
    case 'aspatial': return AspatialRenderer
    default: return SquareGridRenderer
  }
})
</script>
```

**Pros**:

- Single entry point (easier to reason about)
- Shared props/state management
- Easier to add new substrates (just add renderer)

**Cons**:

- More complex abstraction layer
- Still requires separate renderer implementations
- Similar total effort to Option A

**Effort**: 30-40 hours (similar to Option A, slightly less duplication)

---

### Option C: Text-Based Visualization (ASCII/Terminal)

**Architecture**:

```python
# src/townlet/demo/text_viz.py

def render_substrate_text(env):
    """Render current environment state as ASCII art."""
    if env.substrate.type == 'grid':
        if env.substrate.topology == 'square':
            return render_2d_grid(env)
        elif env.substrate.topology == 'cubic':
            return render_3d_grid_floors(env)
        elif env.substrate.topology == 'hexagonal':
            return render_hex_grid(env)
    elif env.substrate.type == 'graph':
        return render_graph_nodes(env)
    elif env.substrate.type == 'aspatial':
        return render_meters_only(env)

# Example 2D square grid:
# ┌─┬─┬─┬─┬─┬─┬─┬─┐
# │ │ │B│ │ │ │ │ │  B = Bed
# ├─┼─┼─┼─┼─┼─┼─┼─┤  H = Hospital
# │ │A│ │ │ │ │ │ │  A = Agent
# ├─┼─┼─┼─┼─┼─┼─┼─┤
# │ │ │H│ │ │ │ │ │
# └─┴─┴─┴─┴─┴─┴─┴─┘

# Example 3D cubic grid (3 floors):
# Floor 0: ┌─┬─┬─┐  Floor 1: ┌─┬─┬─┐  Floor 2: ┌─┬─┬─┐
#          │B│ │ │           │ │ │ │           │ │ │ │
#          ├─┼─┼─┤           ├─┼─┼─┤           ├─┼─┼─┤
#          │A│H│ │           │ │ │ │           │ │ │ │
#          └─┴─┴─┘           └─┴─┴─┘           └─┴─┴─┘

# Example graph substrate:
# Nodes: [0: Home (A), 1: Work, 2: Gym, 3: Shop, 4: Hospital]
# Edges: 0→1, 0→3, 1→2, 3→4
# Agent at node 0 (Home), meters: E=0.8, H=0.9
```

**Pros**:

- **ZERO frontend work required**
- Works immediately with all substrates
- Easy to implement (Python string formatting)
- Sufficient for validating substrate mechanics
- Students can experiment with 3D/hex/graph substrates NOW

**Cons**:

- Not as visually appealing as GUI
- Limited interactivity
- No real-time visualization (print on step or episode end)

**Effort**: 4-6 hours (text rendering for all substrate types)

**Critical Advantage**: **Unblocks TASK-000 immediately** - can implement and test all substrates without waiting for frontend work.

---

### Option D: WebGL/Three.js for 3D, SVG for 2D/Hex

**Architecture**:

- **3D cubic grid**: Three.js with camera controls, floor transparency, agent meshes
- **2D square grid**: Current SVG implementation (no changes)
- **Hexagonal grid**: SVG with hex tile paths
- **Graph**: D3.js force-directed layout
- **Aspatial**: Meters dashboard (no spatial rendering)

**Example 3D rendering**:

```javascript
// Grid3D.vue (Three.js)
import * as THREE from 'three'

const scene = new THREE.Scene()
const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000)
const renderer = new THREE.WebGLRenderer()

// Render 3D grid cells as boxes
for (let z = 0; z < depth; z++) {
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const geometry = new THREE.BoxGeometry(1, 1, 1)
      const material = new THREE.MeshBasicMaterial({ color: 0x2a2a2a, wireframe: true })
      const cube = new THREE.Mesh(geometry, material)
      cube.position.set(x, y, z)
      scene.add(cube)
    }
  }
}

// Render agent as sphere
const agentGeometry = new THREE.SphereGeometry(0.4, 32, 32)
const agentMaterial = new THREE.MeshBasicMaterial({ color: 0xff5555 })
const agentMesh = new THREE.Mesh(agentGeometry, agentMaterial)
```

**Pros**:

- High-quality 3D visualization
- Professional appearance
- Interactive camera controls

**Cons**:

- **Very high implementation effort** (20+ hours for 3D alone)
- Complex state management (Three.js scene updates)
- Performance concerns (many objects in scene)
- Requires learning Three.js/WebGL

**Effort**:

- 3D cubic (Three.js): 20-30 hours
- Hexagonal (SVG): 8-12 hours
- Graph (D3.js): 10-14 hours
- **Total**: 38-56 hours

---

## Tradeoffs Analysis

| Option | Frontend Effort | Blocks TASK-000? | Quality | Flexibility | Maintenance |
|--------|----------------|------------------|---------|-------------|-------------|
| **A: Substrate-Specific Components** | 32-46h | Yes | High | Medium | Medium (5+ components) |
| **B: Unified Component** | 30-40h | Yes | High | High | Medium (complex abstraction) |
| **C: Text-Based Viz** | 4-6h | **NO** | Low | High | Low (Python only) |
| **D: WebGL/Three.js** | 38-56h | Yes | **Very High** | Medium | High (complex rendering) |

**Key Insight**: Options A, B, D all **block TASK-000 implementation** by requiring 30-56 hours of frontend work before substrates can be tested. Option C **unblocks immediately** with minimal effort.

---

## Recommendation: Phased Approach

### Phase 1: Text-Based Visualization (TASK-000)

**Implement**: Terminal/ASCII rendering for all substrates
**Effort**: 4-6 hours
**Deliverable**: `src/townlet/demo/text_viz.py`

**Example usage**:

```bash
# Train on 3D cubic grid with text visualization
python -m townlet.demo.runner --config configs/L1_3D_house --viz text

# Output (printed every 10 steps):
# Episode 42, Step 127/500
# Floor 0: ┌─┬─┬─┐  Floor 1: ┌─┬─┬─┐  Floor 2: ┌─┬─┬─┐
#          │B│ │ │           │ │A│ │           │ │ │ │
#          ├─┼─┼─┤           ├─┼─┼─┤           ├─┼─┼─┤
#          │ │H│ │           │J│ │ │           │ │ │ │
#          └─┴─┴─┘           └─┴─┴─┘           └─┴─┴─┘
# Meters: E=0.72, H=0.85, S=0.64, $=15.20
```

**Benefits**:

- ✅ Unblocks TASK-000 implementation (no frontend dependency)
- ✅ Works with ALL substrates (2D, 3D, hex, graph, aspatial)
- ✅ Sufficient for validating substrate mechanics
- ✅ Students can experiment NOW (don't need GUI)
- ✅ Minimal effort (4-6 hours)

**Limitations**:

- ❌ No real-time GUI visualization
- ❌ No interactivity (can't click/zoom)
- ❌ Less pedagogically engaging than GUI

**Acceptance Criteria**:

- Student can train agents on 3D cubic grid and see floor-by-floor ASCII output
- Student can train on hexagonal grid and see axial coordinate ASCII output
- Student can train on graph substrate and see node-based ASCII output
- Student can train on aspatial substrate and see meters-only output

---

### Phase 2: 2D Substrate Extensions (Separate Project)

**Implement**: Hexagonal SVG rendering
**Effort**: 8-12 hours
**Priority**: **MEDIUM** (nice-to-have, not critical)

**Why defer?**:

- Hexagonal grids are **less common** in pedagogical scenarios than 3D
- SVG hex tile rendering is **non-trivial** (axial coordinate mapping)
- Students can use text viz to experiment with hex substrates in the meantime

**Deliverable**: `frontend/src/components/GridHex.vue`

---

### Phase 3: 3D Visualization (Separate Project)

**Implement**: Three.js 3D cubic grid renderer with floor selection
**Effort**: 20-30 hours
**Priority**: **LOW** (significant effort, marginal pedagogical value over text viz)

**Why defer?**:

- **Very high implementation cost** (20-30 hours)
- **Maintenance burden** (Three.js scene management, camera controls, occlusion)
- Students can use text viz or 2D floor projections in the meantime
- Real pedagogical value is **training on 3D substrates**, not visualizing them

**Deliverable**: `frontend/src/components/Grid3D.vue`

**Example floor selector**:

```
┌─────────────────────────┐
│ Floor: [0] [1] [2]      │ ← Radio buttons to select floor
│                         │
│   ┌─────────────┐       │
│   │ Floor 1     │       │ ← Show selected floor in 2D
│   │   A   B     │       │
│   │             │       │
│   │   H   J     │       │
│   └─────────────┘       │
└─────────────────────────┘
```

**Simpler alternative**: 2D floor projection (no Three.js needed!)

- Render each floor as separate 2D grid
- Stack floors vertically in UI
- Agent appears on correct floor
- **Effort**: 6-8 hours (much simpler than full 3D)

---

### Phase 4: Graph/Aspatial Visualization (On Demand)

**Implement**:

- D3.js force-directed graph layout
- Meters-only dashboard (aspatial)

**Effort**: 12-16 hours
**Priority**: **VERY LOW** (implement only if student specifically requests)

**Why defer?**:

- Graph substrates are **rare in pedagogical scenarios** (advanced topic)
- Force-directed layout is **complex** and **slow** for large graphs
- Text viz shows node IDs and edges clearly enough for validation
- Aspatial substrates need **no spatial rendering** (just meters dashboard, trivial)

---

## Critical Questions & Answers

### Q1: Is text-based visualization acceptable for Phase 1?

**Answer**: **YES** - Text visualization is sufficient for:

- ✅ Validating substrate mechanics (does 3D movement work?)
- ✅ Training agents (does the agent learn?)
- ✅ Debugging config errors (are affordances placed correctly?)
- ✅ Student experimentation ("What if my world was a cube?")

Text viz is **NOT** sufficient for:

- ❌ Real-time interactive demos (need GUI)
- ❌ Recording videos for presentations (need GUI)
- ❌ Public showcases (text looks "unfinished")

**Mitigation**: Use text viz for development/validation, implement GUI rendering as separate follow-up project when needed for demos.

---

### Q2: Should 3D/hex/graph visualization be separate projects?

**Answer**: **YES** - Here's why:

**Evidence from TASK-000**:
> "Risks & Mitigations: Visualization: Frontend assumes 2D square grid. Mitigation: **Phase 1 uses text-based viz, 3D viz comes later**"

**TASK-000 already acknowledges this tradeoff**. The primary value is **backend flexibility**, not frontend rendering.

**Effort comparison**:

- TASK-000 core implementation: 15-22 hours
- Substrate-agnostic GUI rendering: 32-56 hours

**GUI rendering would cost MORE than the entire substrate system!**

**Priority ordering**:

1. **TASK-000**: Implement substrate abstraction (15-22h)
2. **TASK-001**: Schema validation with DTOs (8-12h)
3. **TASK-002**: Action space configuration (10-15h)
4. **TASK-003**: Universe compilation pipeline (12-18h)
5. **TASK-004**: BRAIN_AS_CODE configuration (15-20h)
6. **GUI rendering**: 3D/hex/graph visualization (32-56h) ← LOWEST PRIORITY

**Rationale**: Core UAC infrastructure enables **all future work**. GUI rendering is **nice-to-have**, not critical path.

---

### Q3: Can we defer visualization entirely?

**Answer**: **YES, initially** - Here's the workflow:

**Development Phase** (TASK-000 implementation):

- Use text viz for validation ("does 3D movement work?")
- Use pytest for automated testing
- Use post-training analysis (episode logs, checkpoints)

**Student Experimentation Phase**:

- Students edit `substrate.yaml` to try 3D/hex/graph
- Students see text output during training
- Students analyze training metrics (survival steps, reward)
- Students learn substrate concepts WITHOUT needing GUI

**Demo/Presentation Phase** (later):

- Implement GUI rendering as separate project
- Record videos with GUI for showcases
- Use GUI for interactive demonstrations

**Key Insight**: The pedagogical value of TASK-000 is **backend flexibility** (training on alternative substrates), not real-time visualization. Students learn by **experimenting with configs and analyzing results**, not by watching pretty animations.

---

## Implementation Sketch: Text-Based Visualization

### Substrate Detection

```python
# src/townlet/demo/text_viz.py

def render_state_text(env):
    """Render environment state as ASCII text."""
    substrate_type = env.substrate_config.type

    if substrate_type == "grid":
        topology = env.substrate_config.grid.topology
        if topology == "square":
            return render_2d_square_grid(env)
        elif topology == "cubic":
            return render_3d_cubic_grid(env)
        elif topology == "hexagonal":
            return render_hex_grid(env)
    elif substrate_type == "graph":
        return render_graph_substrate(env)
    elif substrate_type == "aspatial":
        return render_aspatial_state(env)
```

---

### Example: 2D Square Grid (Current)

```python
def render_2d_square_grid(env):
    """Render 2D square grid as ASCII."""
    width, height = env.substrate.dimensions
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Place affordances
    for aff in env.affordances:
        x, y = aff.position
        grid[y][x] = aff.icon[0].upper()  # "B" for Bed, "H" for Hospital

    # Place agents
    for agent in env.agents:
        if agent.alive:
            x, y = agent.position
            grid[y][x] = "A"

    # Render with box drawing characters
    lines = ["┌" + "─┬" * (width - 1) + "─┐"]
    for row in grid:
        lines.append("│" + "│".join(row) + "│")
        lines.append("├" + "─┼" * (width - 1) + "─┤")
    lines[-1] = "└" + "─┴" * (width - 1) + "─┘"

    return "\n".join(lines)
```

**Output**:

```
┌─┬─┬─┬─┬─┬─┬─┬─┐
│ │ │B│ │ │ │ │ │
├─┼─┼─┼─┼─┼─┼─┼─┤
│ │A│ │ │ │ │ │ │
├─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │H│ │ │ │J│ │
└─┴─┴─┴─┴─┴─┴─┴─┘
```

---

### Example: 3D Cubic Grid

```python
def render_3d_cubic_grid(env):
    """Render 3D cubic grid as multiple 2D floor projections."""
    width, height, depth = env.substrate.dimensions
    floors = []

    for z in range(depth):
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Place affordances on this floor
        for aff in env.affordances:
            x, y, z_aff = aff.position
            if z_aff == z:
                grid[y][x] = aff.icon[0].upper()

        # Place agents on this floor
        for agent in env.agents:
            if agent.alive:
                x, y, z_agent = agent.position
                if z_agent == z:
                    grid[y][x] = "A"

        # Render floor
        floor_str = f"Floor {z}:\n"
        floor_str += "┌" + "─┬" * (width - 1) + "─┐\n"
        for row in grid:
            floor_str += "│" + "│".join(row) + "│\n"
        floor_str += "└" + "─┴" * (width - 1) + "─┘"
        floors.append(floor_str)

    return "\n\n".join(floors)
```

**Output**:

```
Floor 0:
┌─┬─┬─┐
│B│ │ │
├─┼─┼─┤
│ │H│ │
└─┴─┴─┘

Floor 1:
┌─┬─┬─┐
│ │A│ │
├─┼─┼─┤
│J│ │ │
└─┴─┴─┘

Floor 2:
┌─┬─┬─┐
│ │ │ │
├─┼─┼─┤
│ │ │ │
└─┴─┴─┘
```

---

### Example: Hexagonal Grid

```python
def render_hex_grid(env):
    """Render hexagonal grid using axial coordinates."""
    # Simplified ASCII representation (offset rows)
    # Full implementation would use proper hex ASCII art
    q_max, r_max = env.substrate.dimensions

    lines = []
    for r in range(r_max):
        # Offset odd rows for hex layout
        offset = "  " if r % 2 == 1 else ""
        row = offset

        for q in range(q_max):
            # Find what's at (q, r)
            cell = " "
            for aff in env.affordances:
                if aff.position == (q, r):
                    cell = aff.icon[0].upper()
            for agent in env.agents:
                if agent.alive and agent.position == (q, r):
                    cell = "A"

            row += f"[{cell}] "
        lines.append(row)

    return "\n".join(lines)
```

**Output**:

```
[B] [ ] [ ] [ ]
  [ ] [A] [ ] [ ]
[H] [ ] [J] [ ]
  [ ] [ ] [ ] [ ]
```

---

### Example: Graph Substrate

```python
def render_graph_substrate(env):
    """Render graph substrate as node list with edges."""
    lines = []

    # List nodes
    lines.append("Nodes:")
    for node_id in range(env.substrate.num_nodes):
        # Find what's at this node
        affordances = [aff.type for aff in env.affordances if aff.position == node_id]
        agents = [agent.id for agent in env.agents if agent.alive and agent.position == node_id]

        node_desc = f"  {node_id}: "
        if affordances:
            node_desc += f"[{', '.join(affordances)}]"
        if agents:
            node_desc += f" (Agent: {', '.join(agents)})"
        lines.append(node_desc)

    # Show edges
    lines.append("\nEdges:")
    for i, j in env.substrate.edges:
        lines.append(f"  {i} ←→ {j}")

    return "\n".join(lines)
```

**Output**:

```
Nodes:
  0: [Bed] (Agent: agent_0)
  1: [Hospital]
  2: [Job]
  3: [Gym]
  4:

Edges:
  0 ←→ 1
  0 ←→ 2
  1 ←→ 3
  2 ←→ 4
```

---

### Example: Aspatial Substrate

```python
def render_aspatial_state(env):
    """Render aspatial substrate (meters only, no spatial info)."""
    lines = ["=== ASPATIAL UNIVERSE ==="]
    lines.append("(No spatial positioning)")
    lines.append("")

    # Show agent meters
    for agent in env.agents:
        if agent.alive:
            lines.append(f"Agent {agent.id}:")
            for meter_name, value in agent.meters.items():
                bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
                lines.append(f"  {meter_name:12s}: [{bar}] {value:.2f}")

    # Show available affordances (no positions)
    lines.append("")
    lines.append("Available affordances:")
    for aff in env.affordances:
        lines.append(f"  - {aff.type}")

    return "\n".join(lines)
```

**Output**:

```
=== ASPATIAL UNIVERSE ===
(No spatial positioning)

Agent agent_0:
  energy      : [████████████░░░░░░░░] 0.72
  health      : [█████████████████░░░] 0.85
  satiation   : [████████████░░░░░░░░] 0.64
  money       : [██████░░░░░░░░░░░░░░] 0.30

Available affordances:
  - Bed
  - Hospital
  - HomeMeal
  - Job
```

---

## Priority & Value Assessment

### Is substrate-agnostic visualization critical for TASK-000?

**Answer**: **NO** - Text visualization is sufficient.

**Evidence**:

1. **TASK-000 goals**: "Enable config-driven spatial substrates" ← Backend flexibility
2. **Not a goal**: "Real-time GUI visualization of all substrates" ← Frontend polish
3. **Success criteria**: "Can switch between 2D/3D by editing substrate.yaml" ← Config-driven, no GUI mention
4. **Risk mitigation**: "Phase 1 uses text-based viz, 3D viz comes later" ← Already acknowledged

**Pedagogical Value Comparison**:

- ✅ **High value**: Training agents on 3D/hex/graph substrates (teaches spatial reasoning, exploration, action space design)
- ✅ **Medium value**: Analyzing training results (survival steps, reward, learned policy)
- ✅ **Low value**: Real-time GUI visualization (nice-to-have, not critical for learning)

**Key Insight**: Students learn by **experimenting with substrate configs** and **analyzing training results**, not by watching real-time 3D animations. Text viz enables experimentation NOW without blocking TASK-000 on 30+ hours of frontend work.

---

### Should this be a separate TASK after TASK-004?

**Answer**: **YES** - Create `TASK-005: Substrate Visualization Suite` as separate project.

**Rationale**:

1. **Dependency**: Visualization depends on TASK-000 being complete (substrate system exists)
2. **Effort**: 32-56 hours (larger than most individual TASKs)
3. **Priority**: Lower than core UAC infrastructure (TASK-001 through TASK-004)
4. **Scope**: Self-contained (doesn't block other work)

**Proposed TASK-005**:

```markdown
# TASK-005: Substrate Visualization Suite

## Problem
Frontend assumes 2D square grid. Cannot visualize 3D cubic, hexagonal, graph, or aspatial substrates.

## Solution
Implement substrate-specific Vue components for each substrate type.

## Dependencies
- TASK-000 (spatial substrates must exist)
- Text visualization (validates substrate mechanics first)

## Implementation Plan
### Phase 1: Hexagonal Grid (8-12h)
- GridHex.vue with axial coordinate mapping
- SVG hex tile rendering

### Phase 2: 3D Floor Projection (6-8h)
- Grid3DFloors.vue with floor selector
- Stack multiple 2D grids vertically (simpler than full 3D)

### Phase 3: Graph Visualization (10-14h)
- GraphViz.vue with D3.js force-directed layout
- Node-edge rendering

### Phase 4: Full 3D Rendering (20-30h, optional)
- Grid3D.vue with Three.js
- Camera controls, floor transparency, occlusion

## Estimated Effort
- Phases 1-3: 24-34 hours
- Phase 4 (optional): +20-30 hours
- Total: 24-64 hours

## Priority
**LOW** - Implement only when needed for demos or student requests.
```

---

## Estimated Effort Summary

| Phase | Component | Effort | Priority | Blocks TASK-000? |
|-------|-----------|--------|----------|------------------|
| **Phase 1: Text Viz** | `text_viz.py` | 4-6h | **HIGH** | **NO** |
| Phase 2: Hex SVG | `GridHex.vue` | 8-12h | MEDIUM | Yes |
| Phase 3a: 3D Floors | `Grid3DFloors.vue` | 6-8h | MEDIUM | Yes |
| Phase 3b: 3D WebGL | `Grid3D.vue` | 20-30h | LOW | Yes |
| Phase 4: Graph | `GraphViz.vue` | 10-14h | LOW | Yes |
| Phase 4: Aspatial | `AspatialDash.vue` | 2-4h | LOW | Yes |

**Critical Path**: Phase 1 only (4-6 hours, does NOT block TASK-000)

**Follow-up work** (Phases 2-4): 46-68 hours, implement as separate TASK-005 when needed for demos.

---

## Conclusion

### Summary

**Substrate-agnostic visualization is NOT critical for TASK-000 implementation.** Text-based visualization is sufficient for:

- Validating substrate mechanics
- Training agents on alternative substrates
- Student experimentation with 3D/hex/graph topologies
- Debugging config errors

**Recommendation**:

1. ✅ **Implement text visualization in TASK-000** (4-6 hours, unblocks immediately)
2. ✅ **Defer GUI rendering to TASK-005** (24-64 hours, implement when needed for demos)
3. ✅ **Use 2D floor projections instead of full 3D** (6-8h vs 20-30h, simpler and sufficient)
4. ✅ **Implement hex/graph visualization on demand** (only if student specifically requests)

**Key Insight**: The value of TASK-000 is **backend flexibility** (config-driven substrates), not **frontend polish** (real-time GUI rendering). Text viz enables the former without blocking on the latter.

**Pedagogical Philosophy**: "Fuck around and find out" applies to **substrate mechanics**, not **visualization aesthetics**. Students learn by experimenting with substrate configs and analyzing results, not by watching fancy 3D animations.

**Final Answer**: Substrate-agnostic visualization is **DEFERRABLE** - implement text viz in Phase 1, defer GUI rendering to separate follow-up project with significantly lower priority than core UAC infrastructure (TASK-001 through TASK-004).
