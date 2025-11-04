# TASK-006: Substrate-Agnostic Visualization

**Status**: Low Priority (Deferred)
**Created**: 2025-11-04 (from research findings)
**Dependencies**: TASK-002A (Spatial Substrates), TASK-002B (Action Space)
**Related Research**: `docs/research/RESEARCH-SUBSTRATE-AGNOSTIC-VISUALIZATION.md`

---

## Problem Statement

Current HAMLET frontend assumes 2D square grid with hardcoded Vue components:

- `GridVisualization.vue` renders 8×8 square grid only
- Agent positions use `(x, y)` coordinates
- Affordance icons placed at grid coordinates
- No support for 3D, hexagonal, graph, or aspatial substrates

**Challenge**: How to render alternative substrates introduced in TASK-002A?

### Substrate Types Requiring Visualization

| Substrate Type | Current Support | Needed |
|----------------|----------------|---------|
| 2D square grid | ✅ Yes (hardcoded) | Generalize |
| 3D cubic grid | ❌ No | Floor selection + 3D rendering |
| Hexagonal grid | ❌ No | Hex tile SVG rendering |
| Graph substrate | ❌ No | Node-edge D3.js visualization |
| Aspatial | ❌ No | Meters-only dashboard (no spatial rendering) |

---

## Solution Overview

Implement substrate-agnostic GUI rendering for all substrate types.

**Key Insight** (from research): Text visualization (implemented in TASK-002A Phase 4) is sufficient for validation, training, and debugging. GUI rendering is "nice-to-have" for demos, not critical for learning.

**Priority**: **Low** - Defer after TASK-001 through TASK-005 complete.

---

## Implementation Plan

### Phase 1: Hexagonal Grid Rendering (8-12 hours)

**Goal**: Render hexagonal grids with axial coordinate system.

#### 1.1: Hex Tile SVG Component

```vue
<!-- HexTile.vue -->
<template>
  <polygon
    :points="hexPoints"
    :class="tileClass"
    @click="handleClick"
  />
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  q: Number,  // Axial coordinate q
  r: Number,  // Axial coordinate r
  size: Number,
  type: String,  // 'empty', 'affordance', 'agent'
})

const hexPoints = computed(() => {
  // Flat-top hexagon vertices
  const size = props.size
  const angles = [0, 60, 120, 180, 240, 300]
  return angles
    .map(angle => {
      const rad = (Math.PI / 180) * angle
      const x = size * Math.cos(rad)
      const y = size * Math.sin(rad)
      return `${x},${y}`
    })
    .join(' ')
})
</script>
```

#### 1.2: Hex Grid Layout

- Convert axial coordinates (q, r) to screen coordinates (x, y)
- Offset rows for proper hex spacing
- Render affordances at hex centers
- Agent movement along hex edges

**Files to create**:

- `frontend/src/components/visualizations/HexTile.vue`
- `frontend/src/components/visualizations/HexGrid.vue`
- `frontend/src/utils/hexCoordinates.ts`

**Effort**: 8-12 hours

---

### Phase 2: 3D Floor Projection (6-8 hours)

**Goal**: Render 3D cubic grids as stacked 2D floor views.

#### 2.1: Floor Selection UI

```vue
<!-- FloorSelector.vue -->
<template>
  <div class="floor-selector">
    <button
      v-for="floor in numFloors"
      :key="floor"
      :class="{ active: currentFloor === floor }"
      @click="selectFloor(floor)"
    >
      Floor {{ floor }}
    </button>
  </div>
</template>
```

#### 2.2: Multi-Floor Grid View

- Render current floor as 2D square grid (reuse existing `GridVisualization.vue`)
- Show vertical movement indicators (stairs, elevators)
- Highlight affordances on other floors (translucent)
- Agent appears on current floor only

**Approach**: Simpler than full 3D rendering

- No camera controls needed
- No occlusion handling needed
- Easy to implement (flat stacking)

**Alternative** (optional, Phase 4): Full 3D WebGL rendering with Three.js

**Files to modify**:

- `frontend/src/components/visualizations/GridVisualization.vue` (add floor prop)
- `frontend/src/components/FloorSelector.vue` (new)

**Effort**: 6-8 hours

---

### Phase 3: Graph Visualization (10-14 hours)

**Goal**: Render graph substrates (subway systems, social networks) as node-edge diagrams.

#### 3.1: D3.js Force-Directed Graph

```typescript
// graphRenderer.ts
import * as d3 from 'd3'

interface GraphNode {
  id: number
  label: string
  affordances: string[]  // Affordances at this node
}

interface GraphEdge {
  source: number
  target: number
}

export function renderGraph(
  container: HTMLElement,
  nodes: GraphNode[],
  edges: GraphEdge[],
  agentPosition: number
) {
  // D3 force simulation
  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(edges).id((d: any) => d.id))
    .force('charge', d3.forceManyBody().strength(-300))
    .force('center', d3.forceCenter(width / 2, height / 2))

  // Render nodes
  const nodeElements = svg.selectAll('.node')
    .data(nodes)
    .enter().append('circle')
    .attr('class', 'node')
    .attr('r', 20)
    .style('fill', d => d.id === agentPosition ? 'red' : 'lightblue')

  // Render edges
  const linkElements = svg.selectAll('.link')
    .data(edges)
    .enter().append('line')
    .attr('class', 'link')

  // Update positions on tick
  simulation.on('tick', () => {
    linkElements
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y)

    nodeElements
      .attr('cx', d => d.x)
      .attr('cy', d => d.y)
  })
}
```

#### 3.2: Node Labels and Affordances

- Display node IDs/names
- Show affordances at each node (icons)
- Highlight agent current node
- Show available movement edges

**Files to create**:

- `frontend/src/components/visualizations/GraphVisualization.vue`
- `frontend/src/utils/graphRenderer.ts`

**Effort**: 10-14 hours

---

### Phase 4: Full 3D WebGL Rendering (20-30 hours, OPTIONAL)

**Goal**: True 3D rendering with camera controls, occlusion, lighting.

#### 4.1: Three.js Scene Setup

```typescript
// scene3d.ts
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

export class Scene3D {
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  renderer: THREE.WebGLRenderer
  controls: OrbitControls

  constructor(container: HTMLElement) {
    this.scene = new THREE.Scene()

    this.camera = new THREE.PerspectiveCamera(
      75,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    )
    this.camera.position.set(8, 8, 8)

    this.renderer = new THREE.WebGLRenderer({ antialias: true })
    this.renderer.setSize(container.clientWidth, container.clientHeight)
    container.appendChild(this.renderer.domElement)

    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
  }

  renderGrid(width: number, height: number, depth: number) {
    // Render floor planes
    for (let z = 0; z < depth; z++) {
      const geometry = new THREE.PlaneGeometry(width, height)
      const material = new THREE.MeshBasicMaterial({
        color: 0x808080,
        transparent: true,
        opacity: 0.3,
        side: THREE.DoubleSide
      })
      const plane = new THREE.Mesh(geometry, material)
      plane.rotation.x = -Math.PI / 2
      plane.position.y = z
      this.scene.add(plane)
    }

    // Render grid lines
    const gridHelper = new THREE.GridHelper(width, width)
    for (let z = 0; z < depth; z++) {
      const grid = gridHelper.clone()
      grid.position.y = z
      this.scene.add(grid)
    }
  }

  renderAffordance(x: number, y: number, z: number, color: number) {
    const geometry = new THREE.BoxGeometry(0.8, 0.8, 0.8)
    const material = new THREE.MeshStandardMaterial({ color })
    const mesh = new THREE.Mesh(geometry, material)
    mesh.position.set(x, z, y)
    this.scene.add(mesh)
  }

  animate() {
    requestAnimationFrame(() => this.animate())
    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }
}
```

#### 4.2: Camera Controls

- Orbit controls (rotate, zoom, pan)
- Snap to floor view (top-down)
- Follow agent camera mode

#### 4.3: Occlusion and Transparency

- Show/hide floors
- Transparent upper floors
- X-ray mode (see through walls)

**Effort**: 20-30 hours

**Priority**: **LOWEST** - Only if needed for impressive demos/videos.

---

## Aspatial Substrate Rendering (Included in TASK-002A)

**No spatial visualization needed** - just show meters dashboard.

**Already handled by**: Existing meters display (progress bars, numerical values).

**Effort**: 0 hours (already works!)

---

## Benefits

1. **Substrate variety support**: Students can experiment with 3D, hex, graph topologies
2. **Pedagogical visualization**: See agents navigate different spatial structures
3. **Debugging aid**: Visual feedback for substrate behavior
4. **Demo quality**: Impressive 3D renderings for presentations

---

## Dependencies

### TASK-002A (Spatial Substrates) - REQUIRED

Must be complete before starting this task:

- Substrate types defined (2D, 3D, hex, graph, aspatial)
- Substrate config loading (`substrate.yaml`)
- Backend substrate implementation

### TASK-002B (Action Space) - REQUIRED

Needed for rendering agent movement:

- Action deltas (how agent moves)
- Action availability (which actions valid where)

### TASK-003 (Compiler) - OPTIONAL

Helpful but not blocking:

- Compiled universe provides metadata for rendering

---

## Success Criteria

### Phase 1: Hexagonal Grid

- [ ] Hex tiles render correctly with axial coordinates
- [ ] Affordances placed at hex centers
- [ ] Agent moves along hex edges
- [ ] Hover/click interactions work

### Phase 2: 3D Floor Projection

- [ ] Floor selector UI works
- [ ] Current floor renders as 2D grid
- [ ] Agent appears on correct floor
- [ ] Vertical movement indicators visible
- [ ] Can switch floors and see agent on different levels

### Phase 3: Graph Visualization

- [ ] D3 force-directed graph renders
- [ ] Nodes labeled with IDs
- [ ] Edges show connections
- [ ] Agent node highlighted
- [ ] Affordances shown at nodes

### Phase 4: Full 3D WebGL (OPTIONAL)

- [ ] Three.js scene renders 3D grid
- [ ] Camera controls work (orbit, zoom, pan)
- [ ] Agent renders as 3D object
- [ ] Affordances render as 3D objects
- [ ] Floor transparency works
- [ ] Performance >30 FPS on target hardware

---

## Estimated Effort

### Minimum Viable (Hex + 3D Projection)

- **Phase 1** (Hex): 8-12 hours
- **Phase 2** (3D projection): 6-8 hours
- **Total**: **14-20 hours**

### Complete (+ Graph)

- **Phase 1** (Hex): 8-12 hours
- **Phase 2** (3D projection): 6-8 hours
- **Phase 3** (Graph): 10-14 hours
- **Total**: **24-34 hours**

### Full (+ WebGL 3D)

- **Phase 1** (Hex): 8-12 hours
- **Phase 2** (3D projection): 6-8 hours
- **Phase 3** (Graph): 10-14 hours
- **Phase 4** (Full 3D): 20-30 hours
- **Total**: **44-64 hours**

**Recommended starting point**: Phase 1 + Phase 2 (14-20h) - covers most common substrates.

---

## Risks & Mitigations

### Risk: High Implementation Complexity

**Phase 4 (Full 3D) is expensive** (20-30h) and has limited pedagogical value.

**Mitigation**: Start with Phase 2 (floor projection) which is simpler (6-8h) and provides 80% of the value.

### Risk: Performance on Low-End Hardware

3D rendering can be slow on Chromebooks, budget laptops.

**Mitigation**:

- Keep Phase 2 as fallback (2D floor views)
- Optimize Phase 4 (LOD, culling, instancing)
- Provide "low performance mode" toggle

### Risk: Browser Compatibility

WebGL support varies across browsers, especially older ones.

**Mitigation**:

- Detect WebGL support, fallback to Phase 2 if unavailable
- Test on Firefox, Chrome, Safari, Edge

### Risk: Maintenance Burden

Three.js/D3.js have breaking changes between major versions.

**Mitigation**:

- Pin dependency versions
- Document which versions tested
- Keep visualization separate from core logic (loose coupling)

---

## Priority Justification: Why LOW Priority?

### Text Visualization is Sufficient (from research)

**TASK-002A Phase 4 implements text-based visualization** (4-6 hours) that provides:

- ✅ Debugging: See agent position, affordances, meters
- ✅ Validation: Confirm substrate behavior correct
- ✅ Experimentation: Test 3D, hex, graph substrates without GUI
- ✅ Fast: No WebGL setup, works in any terminal

**Example text output** (3D cubic, floor 1):

```
Floor 1:
. . . B . . . .
. . . . . . . .
. A . . . . H .
. . . . J . . .

Floor 2:
. . . . . . . .
. . . . . . . .
. . . . . . . .
```

### GUI is "Nice-to-Have"

**GUI rendering is valuable for**:

- Impressive demos/videos for stakeholders
- Making substrate differences visually obvious
- Student engagement (visual learners)

**BUT NOT CRITICAL FOR**:

- Learning RL concepts (students analyze metrics, not graphics)
- Training agents (backend only)
- Debugging substrate logic (text sufficient)
- Core UAC infrastructure (TASK-001 through TASK-005)

### Effort vs Value Tradeoff

| Feature | Effort | Pedagogical Value | Priority |
|---------|--------|-------------------|----------|
| Text viz | 4-6h | High (debugging) | ✅ High (TASK-002A) |
| Hex SVG | 8-12h | Medium (topology learning) | ⚠️ Medium |
| 3D projection | 6-8h | Medium (spatial reasoning) | ⚠️ Medium |
| Graph D3 | 10-14h | Medium (abstract spaces) | ⚠️ Medium |
| Full 3D WebGL | 20-30h | **Low (eye candy)** | ❌ Low |

**Conclusion**: Defer GUI to TASK-006 as separate low-priority project. Focus on core UAC infrastructure first.

---

## Relationship to Other Tasks

**After TASK-002A** (Spatial Substrates):

- Substrate types defined
- Backend substrate implementation complete
- Text visualization available

**After TASK-002B** (Action Space):

- Action definitions available
- Agent movement logic defined

**After TASK-003** (Compiler):

- Compiled universe metadata available
- Config loading stable

**After TASK-005** (BRAIN_AS_CODE):

- Network architecture configurable
- Full UAC system complete

**THEN TASK-006** (This task):

- Add GUI rendering for visual appeal
- Enhance student engagement
- Enable impressive demos

**Recommended sequence**:

```
TASK-001 → TASK-002A → TASK-002B → TASK-003 → TASK-004A → TASK-004B → TASK-005 → TASK-006
                                                                                      ↑
                                                                               (lowest priority)
```

---

## Design Principles

### Substrate Detection

Frontend must detect substrate type from config/compiled universe:

```typescript
// substrateRenderer.ts
export function renderSubstrate(substrate: SubstrateConfig) {
  switch (substrate.type) {
    case 'grid':
      if (substrate.grid.topology === 'square' && substrate.grid.dimensions.length === 2) {
        return <GridVisualization2D />
      } else if (substrate.grid.topology === 'cubic' || substrate.grid.dimensions.length === 3) {
        return <GridVisualization3D />
      } else if (substrate.grid.topology === 'hexagonal') {
        return <HexGridVisualization />
      }
    case 'graph':
      return <GraphVisualization />
    case 'aspatial':
      return <MetersOnlyDashboard />
    default:
      return <TextVisualization />  // Fallback
  }
}
```

### Fallback Strategy

Always provide text visualization as fallback:

- If WebGL unsupported
- If substrate type unknown
- If rendering errors occur
- If performance too low

---

## Future Extensions

### Multi-Agent Visualization

Show multiple agents on same grid (color-coded).

### Replay Visualization

Render recorded episodes for analysis.

### Comparative Visualization

Show agent behavior on different substrates side-by-side.

### VR/AR Rendering

3D substrates in VR headset (distant future, research project).

---

## Conclusion

TASK-006 implements GUI rendering for alternative spatial substrates (hex, 3D, graph). While valuable for demos and visual learners, it is **not critical** for core HAMLET functionality and is **deferred to low priority** after UAC infrastructure (TASK-001 through TASK-005) is complete.

**Minimum viable**: Phases 1-2 (14-20h)
**Complete**: Phases 1-3 (24-34h)
**Full**: Phases 1-4 (44-64h)

**Status**: **DEFER** - Text visualization from TASK-002A is sufficient for now.
