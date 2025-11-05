# Frontend Visualization Architecture (HLD)

**Date**: 2025-11-06
**Status**: Implemented (TASK-002A Phase 7)
**Version**: 2.0 (Multi-Substrate Support)

---

## Overview

The HAMLET frontend provides real-time visualization of agent behavior via WebSocket. It supports **two rendering modes** based on substrate type: **Spatial** (Grid2D) and **Aspatial**.

**Technology Stack**:
- Vue 3 (Composition API)
- Pinia (state management)
- SVG (spatial rendering)
- WebSocket (real-time communication)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Backend (Python)                        │
│                                                             │
│  VectorizedHamletEnv ──→ LiveInferenceServer ──→ WebSocket │
│          │                      │                           │
│          │                      │                           │
│       Substrate              Substrate                      │
│       Metadata               Serialization                  │
└─────────────────────────────────────────────────────────────┘
                               │
                               │ WebSocket
                               │ (JSON messages)
                               ↓
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Vue 3)                         │
│                                                             │
│  WebSocket ──→ Simulation Store ──→ App.vue                │
│                  (Pinia)                 │                  │
│                     │                    │                  │
│                     │          ┌─────────┴─────────┐        │
│                     │          │                   │        │
│                     ↓          ↓                   ↓        │
│                Grid.vue    AspatialView.vue   Other         │
│                  (SVG)       (Dashboard)      Components    │
└─────────────────────────────────────────────────────────────┘
```

---

## Rendering Pipeline

### 1. WebSocket Message Receipt

**Message Types**:
- `connected`: Initial handshake, substrate metadata
- `episode_start`: New episode, reset state
- `state_update`: Step update, agent/affordance positions, meters

**Substrate Metadata** (in all messages):
```json
{
  "substrate": {
    "type": "grid2d",
    "position_dim": 2,
    "width": 8,
    "height": 8,
    "topology": "square"
  }
}
```

---

### 2. State Storage (Pinia)

**Store** (`simulation.js`):
```javascript
const substrateType = ref('grid2d')
const substrateMetadata = ref({...})
const gridWidth = ref(8)
const gridHeight = ref(8)
const agents = ref([...])
const affordances = ref([...])
const agentMeters = ref({...})
```

**Update Flow**:
```
WebSocket message → Parse JSON → Update refs → Trigger Vue reactivity
```

---

### 3. Component Rendering

**Dispatcher** (`App.vue`):
```vue
<Grid v-if="substrateType === 'grid2d'" ... />
<AspatialView v-else-if="substrateType === 'aspatial'" ... />
```

**Props Flow**:
```
Store (reactive refs) → App.vue (pass as props) → Child component (render)
```

---

## Spatial Mode (Grid2D)

**Component**: `Grid.vue`

**Rendering Strategy**: SVG-based 2D grid

**Data Requirements**:
- Agent positions: `[{id, x, y, color}]`
- Affordance positions: `[{type, x, y}]`
- Grid dimensions: `width, height`

**Visual Elements**:
1. **Grid cells** (background): `<rect>` for each (x, y)
2. **Heat map** (optional): Colored overlay for position visit frequency
3. **Affordances**: Icons at affordance positions
4. **Agent trails**: Last 3 positions with fading opacity
5. **Agents**: Circles at current position with pulse animation

**Performance**:
- ~100 SVG elements (8×8 grid + agents + affordances)
- Hardware-accelerated rendering
- 60 FPS at 50 steps/sec

---

## Aspatial Mode

**Component**: `AspatialView.vue`

**Rendering Strategy**: HTML dashboard (no SVG)

**Data Requirements**:
- Agent meters: `{energy: 0.8, health: 0.5, ...}`
- Affordances: `[{type: "Bed"}, {type: "Job"}]` (no positions)
- Last action: `4` (INTERACT)

**Visual Elements**:
1. **Large meter bars**: Color-coded by value (critical/warning/healthy)
2. **Affordance list**: Cards showing available interactions
3. **Action history**: Log of recent actions (last 10)

**Layout** (responsive):
- **Mobile**: Single column (meters → affordances → history)
- **Tablet+**: Two columns (meters left, affordances/history right)

**Performance**:
- ~20 HTML elements (8 meters + affordance cards)
- Simpler than spatial mode (no SVG complexity)
- Better performance on low-end devices

---

## Backward Compatibility

**Legacy Checkpoints** (without substrate metadata):

**Detection**:
```javascript
if (!message.substrate) {
  // Assume legacy spatial behavior
  substrateType.value = 'grid2d'
  substrateMetadata.value = {
    type: 'grid2d',
    position_dim: 2,
    width: message.grid.width,
    height: message.grid.height
  }
}
```

**Fallback Behavior**:
- Render spatial view (Grid.vue)
- Use grid dimensions from `grid.width/height`
- No crashes or errors

---

## Feature Matrix

| Feature | Spatial (Grid2D) | Aspatial |
|---------|------------------|----------|
| Grid cells | ✅ Yes | ❌ No |
| Agent positions | ✅ (x, y) | ❌ No concept |
| Affordance positions | ✅ (x, y) | ❌ No concept |
| Meter bars | ✅ Small panel | ✅ Large display |
| Heat map | ✅ Position visits | ❌ Spatial feature |
| Agent trails | ✅ Last 3 positions | ❌ Spatial feature |
| Affordance list | ❌ Not needed | ✅ Card layout |
| Action history | ❌ Not shown | ✅ Text log |
| Novelty heatmap | ✅ RND exploration | ❌ Spatial feature |

---

## Testing Strategy

### Unit Tests

**Grid.vue**:
- Renders correct number of grid cells
- Positions agents correctly at (x, y)
- Handles missing heat map gracefully

**AspatialView.vue**:
- Renders all meters with correct values
- Color-codes meters by threshold (critical/warning/healthy)
- Updates action history on new actions

**Simulation Store**:
- Stores substrate metadata from WebSocket
- Falls back to spatial mode if substrate missing
- Passes substrate to components via props

---

### Integration Tests

**End-to-End Spatial**:
1. Start Grid2D inference server
2. Connect frontend
3. Verify SVG grid renders
4. Verify agents move on grid

**End-to-End Aspatial**:
1. Start Aspatial inference server
2. Connect frontend
3. Verify dashboard renders (no grid)
4. Verify meters update in real-time

**Backward Compatibility**:
1. Connect to legacy checkpoint (no substrate)
2. Verify spatial view renders
3. Verify no errors in console

---

## Future Enhancements

**Possible Extensions** (out of scope for Phase 7):

1. **3D Grid Substrates**: WebGL/Three.js renderer for 3×3×3 grids
2. **Graph Substrates**: D3.js force-directed graph for node-based universes
3. **Multi-Agent Visualization**: Color-coded agents with ID labels
4. **Affordance Operating Hours**: Show open/closed status in UI
5. **Interaction Progress Ring**: Animated ring for multi-tick interactions (already implemented for spatial)

---

## Maintenance Notes

**Adding New Substrate Types**:

1. Add substrate type to backend (`SpatialSubstrate` subclass)
2. Update WebSocket protocol to include substrate metadata
3. Create new Vue component for rendering (e.g., `Graph3DView.vue`)
4. Add dispatcher case in `App.vue`:
   ```vue
   <NewView v-else-if="substrateType === 'newtype'" ... />
   ```

**Modifying Affordance Icons**:

Edit `frontend/src/utils/constants.js`:
```javascript
export const AFFORDANCE_ICONS = {
  NewAffordance: '🆕',  // Add new icon here
}
```

**Changing Meter Colors**:

Edit `frontend/src/components/AspatialView.vue`:
```javascript
function getMeterClass(name, value) {
  if (value < 0.1) return 'meter-critical'  // Adjust threshold
  if (value < 0.4) return 'meter-warning'
  return 'meter-healthy'
}
```

---

## References

- **TASK-002A**: Substrate abstraction implementation
- **PDR-002**: No-defaults principle (explicit substrate config)
- **Vue 3 Docs**: https://vuejs.org/guide/introduction.html
- **Pinia Docs**: https://pinia.vuejs.org/
