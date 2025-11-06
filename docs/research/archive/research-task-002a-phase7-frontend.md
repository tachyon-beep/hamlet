# TASK-002A Phase 7: Frontend Visualization - Research Document

**Date**: 2025-11-05
**Status**: Research Complete
**Phase**: 7 of 8 (Frontend Visualization)
**Estimated Effort**: 4-6 hours

---

## Executive Summary

Phase 7 adapts the frontend visualization to handle multiple substrate types (Grid2D, Aspatial). Currently the frontend **assumes** 2D spatial grid - all rendering logic is hardcoded for `x,y` coordinates.

**Key Finding**: The frontend must support **two rendering modes**:
1. **Spatial Mode** (Grid2D): Current SVG-based grid visualization
2. **Aspatial Mode** (Aspatial): Meters-only dashboard (no grid)

**Impact**: Moderate complexity. Grid.vue is cleanly isolated and already uses props-based architecture. Changes are additive (no removal of existing spatial code).

**Critical Discovery**: WebSocket protocol already sends grid dimensions (`grid.width`, `grid.height`) but does NOT send substrate metadata. We must add substrate type to WebSocket messages.

---

## Research Questions

### Q1: Where is the frontend code located?

**Answer**: `frontend/src/` directory with Vue 3 + Pinia architecture

**Key Files**:
- `frontend/src/components/Grid.vue` - SVG grid renderer (MODIFY)
- `frontend/src/components/MeterPanel.vue` - Meter dashboard (ENHANCE)
- `frontend/src/stores/simulation.js` - Pinia store for WebSocket state (MODIFY)
- `frontend/src/App.vue` - Main layout (MINOR MODIFY)

**Architecture**:
- **Grid.vue** (269 lines): Receives props from parent, renders SVG grid
- **Props used**: `gridWidth`, `gridHeight`, `agents`, `affordances`, `heatMap`
- **SVG structure**: Grid cells, affordances (icons), agents (circles), trails

**Finding**: Grid.vue is **perfectly isolated** - all data comes via props. We can swap rendering mode based on substrate metadata without touching other components.

---

### Q2: How does Grid.vue currently render?

**Answer**: SVG-based 2D grid with hardcoded spatial assumptions

**Current Rendering Pipeline**:

```vue
<!-- Grid.vue structure -->
<svg viewBox="...">
  <!-- 1. Grid cells (background) -->
  <rect v-for="(x,y) in grid" :x="x * cellSize" :y="y * cellSize" />

  <!-- 2. Heat map overlay (optional) -->
  <rect v-for="pos in heatMap" :opacity="intensity" />

  <!-- 3. Affordances (icons) -->
  <rect v-for="aff in affordances" :x="aff.x * cellSize" :y="aff.y * cellSize" />

  <!-- 4. Agent trails (last 3 positions) -->
  <circle v-for="pos in trail" :cx="pos.x" :cy="pos.y" />

  <!-- 5. Agents (current position) -->
  <circle v-for="agent in agents" :cx="agent.x" :cy="agent.y" />
</svg>
```

**Spatial Assumptions** (MUST FIX for aspatial):
1. Line 18: `viewBox` computed from `gridWidth × cellSize`
2. Lines 28-34: Grid cells iterate `x in gridWidth, y in gridHeight`
3. Lines 54-60: Affordances positioned at `affordance.x * cellSize`
4. Lines 99-104: Agents positioned at `agent.x * cellSize`
5. Lines 74-83: Agent trails use `pos.x, pos.y`

**Heat Map Logic** (Lines 37-49):
- Parses `heatMap` object: `{"0,1": 0.8, "2,3": 0.5}`
- Extracts x,y from key: `key.split(',').map(Number)`
- Renders colored overlay at grid position

**Finding**: All rendering assumes `(x, y)` coordinates exist. Aspatial mode needs completely different renderer.

---

### Q3: What substrate metadata needs to be sent via WebSocket?

**Answer**: Backend must send substrate type and dimensions in state updates

**Current WebSocket Protocol** (live_inference.py, line 731):

```python
update = {
    "type": "state_update",
    "step": self.current_step,
    "grid": {
        "width": self.env.grid_size,  # ← Already sent
        "height": self.env.grid_size,  # ← Already sent
        "agents": [...],
        "affordances": [...]
    },
    # Missing: substrate metadata!
}
```

**Required Additions**:

```python
update = {
    "type": "state_update",
    "substrate": {  # ← NEW
        "type": "grid2d",  # or "aspatial"
        "position_dim": 2,  # 2 for grid, 0 for aspatial
        "topology": "square",  # grid2d only
        "width": 8,  # grid2d only
        "height": 8  # grid2d only
    },
    "grid": {
        # Keep existing for backward compat during migration
        "width": self.env.grid_size,
        "height": self.env.grid_size,
        "agents": [...],
        "affordances": [...]
    },
}
```

**Rationale**:
- `substrate.type`: Frontend dispatches to correct renderer
- `substrate.position_dim`: Validates agent/affordance data shape
- `substrate.topology/width/height`: Grid2D rendering parameters

**Backward Compatibility**:
Keep existing `grid.width/height` during migration. Frontend can detect substrate metadata presence and fall back to legacy behavior if missing.

---

### Q4: How to handle aspatial substrates (no grid to render)?

**Answer**: Create new `<AspatialView>` component, swap in App.vue based on substrate type

**Aspatial Design**:

```vue
<!-- AspatialView.vue (NEW COMPONENT) -->
<template>
  <div class="aspatial-container">
    <!-- Large meter panel (no grid) -->
    <MeterPanel :agent-meters="agentMeters" :large-mode="true" />

    <!-- Affordance availability list -->
    <div class="affordance-list">
      <div v-for="aff in availableAffordances" class="affordance-card">
        <span class="icon">{{ getAffordanceIcon(aff.type) }}</span>
        <span class="name">{{ aff.type }}</span>
        <span class="status">Available</span>
      </div>
    </div>

    <!-- Action history (recent interactions) -->
    <div class="action-history">
      <div v-for="action in recentActions" class="action-item">
        {{ action.affordance }} (Step {{ action.step }})
      </div>
    </div>
  </div>
</template>
```

**Rendering Strategy**:
1. **No grid**: Remove spatial visualization entirely
2. **Focus on meters**: Large meter bars with labels
3. **Affordance list**: Show all affordances as cards (no positions)
4. **Action history**: Text log of recent interactions

**Layout**:
- Full-width meter panel at top
- Affordance cards in grid layout (CSS grid, not spatial grid)
- Action history scrollable panel at bottom

**Why This Design**:
- Aspatial universes have **no concept of position** - showing grid would be misleading
- Meters are **primary signal** for aspatial learning (no spatial exploration)
- Affordance availability is **binary** (available vs unavailable) - no distance
- Action history provides **temporal context** without spatial context

**Implementation Notes**:
- Reuse `<MeterPanel>` component with `large-mode` prop
- Reuse `AFFORDANCE_ICONS` from `utils/constants.js`
- Store action history in simulation.js store (last 10 actions)

---

## Critical Findings

### Finding 1: WebSocket Protocol Gap

**Issue**: Backend does NOT send substrate metadata in state updates

**Evidence**: `live_inference.py` line 731 sends:
```python
"grid": {"width": 8, "height": 8, ...}
```

But does not send:
```python
"substrate": {"type": "grid2d", "position_dim": 2}
```

**Impact**: Frontend cannot detect substrate type → cannot dispatch to correct renderer

**Fix**: Add `substrate` field to ALL WebSocket messages:
- `state_update` (every step)
- `episode_start` (episode begin)
- `connected` (initial handshake)

---

### Finding 2: Props-Based Architecture is Perfect

**Advantage**: Grid.vue already receives all data via props → easy to swap renderers

**Current Props**:
```vue
defineProps({
  gridWidth: Number,
  gridHeight: Number,
  agents: Array,
  affordances: Array,
  heatMap: Object
})
```

**Proposed Enhancement**:
```vue
defineProps({
  substrate: Object,  // ← NEW (includes type, dims)
  gridWidth: Number,   // ← Keep for backward compat
  gridHeight: Number,
  agents: Array,
  affordances: Array,
  heatMap: Object
})
```

**Implementation**:
```vue
<template>
  <GridView v-if="substrate.type === 'grid2d'" :substrate="substrate" ... />
  <AspatialView v-else-if="substrate.type === 'aspatial'" ... />
</template>
```

---

### Finding 3: Minimal Changes Required

**Scope**: Only 4 files need modification:

1. **Backend** (`live_inference.py`): Add substrate metadata to WebSocket messages (50 lines)
2. **Store** (`simulation.js`): Store substrate state, pass to components (20 lines)
3. **Grid.vue**: Detect substrate type, dispatch renderer (30 lines refactor)
4. **AspatialView.vue**: NEW component (150 lines)

**Total Estimated Changes**: ~250 lines (very small for a rendering mode change)

**Why So Small**:
- Existing components are well-isolated (props-based)
- No changes to other components (MeterPanel, StatsPanel, etc.)
- Aspatial mode is additive (Grid.vue keeps all spatial logic)

---

## Substrate Type Detection Strategy

**Option 1: Explicit substrate field** (RECOMMENDED)
```json
{
  "substrate": {
    "type": "grid2d",
    "position_dim": 2
  }
}
```

**Pros**: Clear, explicit, extensible
**Cons**: Requires backend changes

**Option 2: Infer from position_dim** (FALLBACK)
```javascript
const isAspatial = agents.length === 0 || agents[0].x === undefined
```

**Pros**: Works without backend changes
**Cons**: Fragile (breaks if agent data missing for other reasons)

**Recommendation**: Use Option 1 (explicit substrate field). Cleaner contract, easier to debug.

---

## WebSocket Message Contract (Proposed)

### Message: `connected`

**Current**:
```json
{
  "type": "connected",
  "mode": "inference",
  "checkpoint": "checkpoint_ep00100"
}
```

**Proposed**:
```json
{
  "type": "connected",
  "mode": "inference",
  "checkpoint": "checkpoint_ep00100",
  "substrate": {
    "type": "grid2d",
    "position_dim": 2,
    "topology": "square",
    "width": 8,
    "height": 8
  }
}
```

---

### Message: `episode_start`

**Current**:
```json
{
  "type": "episode_start",
  "episode": 42,
  "checkpoint_episode": 100
}
```

**Proposed**:
```json
{
  "type": "episode_start",
  "episode": 42,
  "checkpoint_episode": 100,
  "substrate": {
    "type": "grid2d",
    "position_dim": 2,
    "width": 8,
    "height": 8
  }
}
```

---

### Message: `state_update`

**Current**:
```json
{
  "type": "state_update",
  "step": 15,
  "grid": {
    "width": 8,
    "height": 8,
    "agents": [{"id": "agent_0", "x": 3, "y": 5}],
    "affordances": [{"type": "Bed", "x": 2, "y": 1}]
  }
}
```

**Proposed**:
```json
{
  "type": "state_update",
  "step": 15,
  "substrate": {
    "type": "grid2d",
    "position_dim": 2,
    "width": 8,
    "height": 8
  },
  "grid": {
    "width": 8,
    "height": 8,
    "agents": [{"id": "agent_0", "x": 3, "y": 5}],
    "affordances": [{"type": "Bed", "x": 2, "y": 1}]
  }
}
```

**For Aspatial**:
```json
{
  "type": "state_update",
  "step": 15,
  "substrate": {
    "type": "aspatial",
    "position_dim": 0
  },
  "grid": {
    "agents": [{"id": "agent_0"}],  // No x, y
    "affordances": [{"type": "Bed"}]  // No x, y
  }
}
```

---

## Component Interaction Flow

**Spatial (Grid2D)**:
```
WebSocket → Store → App.vue → Grid.vue → SVG Renderer
                                       ↓
                                  (x,y) positions
```

**Aspatial**:
```
WebSocket → Store → App.vue → AspatialView.vue → Dashboard Renderer
                                               ↓
                                          No positions
```

**Detection Point**: `App.vue` checks `substrate.type` and renders correct component

---

## Risk Assessment

### Risk 1: Backend Substrate Metadata Not Available

**Scenario**: `VectorizedHamletEnv` doesn't expose substrate type in Phase 7

**Mitigation**: Phase 4 already added `env.substrate` attribute. We can access it in live_inference.py:
```python
substrate_type = self.env.substrate.__class__.__name__  # "Grid2DSubstrate"
substrate_metadata = self.env.substrate.to_dict()  # Phase 1 method
```

**Likelihood**: LOW (Phase 4 already provides this)

---

### Risk 2: Frontend Breaks for Existing Checkpoints

**Scenario**: Old checkpoints without substrate metadata crash frontend

**Mitigation**: Backward compatibility fallback:
```javascript
const substrate = message.substrate || {
  type: "grid2d",  // Assume legacy behavior
  position_dim: 2,
  width: message.grid.width,
  height: message.grid.height
}
```

**Likelihood**: MEDIUM (will happen during rollout)

---

### Risk 3: Heat Map Breaks in Aspatial Mode

**Scenario**: Heat map assumes grid positions, crashes with aspatial

**Mitigation**: Disable heat map for aspatial:
```vue
<button v-if="substrate.type === 'grid2d' && heatMap" @click="toggleHeatMap">
  Show Heat Map
</button>
```

**Likelihood**: HIGH (heat map is fundamentally spatial)

---

## Documentation Requirements

### CLAUDE.md Updates

**Add Section**: "Frontend Visualization Modes"

```markdown
### Frontend Visualization (src/frontend)

**Rendering Modes**:

1. **Spatial Mode** (Grid2D substrates)
   - SVG-based 2D grid visualization
   - Agent positions rendered as circles
   - Affordances rendered as icons at (x, y)
   - Heat map overlay for position visit frequency
   - Agent trails (last 3 positions)

2. **Aspatial Mode** (Aspatial substrates)
   - Meters-only dashboard (no grid)
   - Affordance availability list (no positions)
   - Action history log (temporal context)
   - No heat map (position-based feature)

**Substrate Detection**:
Frontend detects substrate type from WebSocket messages:
- `message.substrate.type === "grid2d"` → Spatial mode
- `message.substrate.type === "aspatial"` → Aspatial mode

**Backward Compatibility**:
If `substrate` field missing (legacy checkpoints), frontend assumes Grid2D.
```

---

## Testing Strategy

### Unit Tests

1. **Grid.vue substrate detection**:
   - Renders spatial view when `substrate.type === "grid2d"`
   - Renders aspatial view when `substrate.type === "aspatial"`
   - Falls back to spatial when substrate missing (legacy)

2. **AspatialView.vue rendering**:
   - Renders meter panel in large mode
   - Renders affordance list without positions
   - Renders action history with timestamps

3. **Simulation store substrate handling**:
   - Stores substrate metadata from WebSocket
   - Passes substrate to components via props
   - Handles missing substrate gracefully

---

### Integration Tests

1. **End-to-end spatial rendering**:
   - Connect to Grid2D environment
   - Verify SVG grid renders
   - Verify agents positioned correctly

2. **End-to-end aspatial rendering**:
   - Connect to Aspatial environment
   - Verify dashboard renders (no grid)
   - Verify affordance list displays

3. **Backward compatibility**:
   - Connect to legacy checkpoint (no substrate metadata)
   - Verify frontend falls back to spatial mode
   - Verify no crashes or errors

---

### Manual Testing Checklist

- [ ] Grid2D environment renders spatial view
- [ ] Aspatial environment renders dashboard view
- [ ] Switching between environments updates UI correctly
- [ ] Heat map disabled in aspatial mode
- [ ] Agent trails disabled in aspatial mode
- [ ] Affordance list shows all affordances in aspatial
- [ ] Action history updates in real-time
- [ ] Legacy checkpoints render without crashes

---

## Performance Considerations

**Spatial Rendering** (Current):
- SVG elements: ~100 (grid cells + affordances + agents)
- Re-renders: Every WebSocket update (~5-50 Hz)
- Performance: Excellent (SVG is hardware-accelerated)

**Aspatial Rendering** (New):
- HTML elements: ~20 (meters + affordance cards)
- Re-renders: Every WebSocket update (~5-50 Hz)
- Performance: Better than spatial (fewer DOM elements)

**Finding**: Aspatial mode will be **faster** than spatial mode (simpler rendering).

---

## Alternative Designs Considered

### Alternative 1: Show Grid for Aspatial (Rejected)

**Idea**: Render agents/affordances in fake grid for visual consistency

**Rejected Because**:
- **Misleading**: Implies spatial relationships that don't exist
- **Pedagogically harmful**: Students learn incorrect intuitions
- **Unnecessary**: Meters alone provide sufficient information

---

### Alternative 2: Text-Only Rendering (Rejected)

**Idea**: Pure text log of state (no graphics)

**Rejected Because**:
- **Poor UX**: Hard to scan/parse quickly
- **Less engaging**: Students lose interest
- **Underutilizes meters**: Meter bars convey info faster than numbers

---

### Alternative 3: 3D Visualization for Aspatial (Rejected)

**Idea**: Use abstract 3D space to show "conceptual" relationships

**Rejected Because**:
- **Overengineered**: Adds complexity without value
- **Slow**: 3D rendering overhead
- **No semantic meaning**: 3D positions arbitrary in aspatial

---

## Open Questions

### Q1: Should aspatial mode show Q-values?

**Context**: Grid2D shows Q-values for all 6 actions (UP, DOWN, LEFT, RIGHT, INTERACT, WAIT). In aspatial, only INTERACT and WAIT make sense.

**Options**:
1. Show all 6 Q-values (with MOVEMENT actions greyed out)
2. Show only INTERACT and WAIT Q-values
3. Hide Q-values entirely

**Recommendation**: Option 2 (show only valid actions). Cleaner, less confusing.

---

### Q2: Should heat map work for aspatial?

**Context**: Heat map tracks position visit frequency. Aspatial has no positions.

**Options**:
1. Disable heat map for aspatial (current plan)
2. Reinterpret heat map as "affordance interaction frequency"

**Recommendation**: Option 1 (disable). Heat map is fundamentally spatial - reinterpreting it is confusing.

---

### Q3: Should agent trails work for aspatial?

**Context**: Agent trails show last 3 positions. Aspatial has no positions.

**Options**:
1. Disable trails for aspatial (current plan)
2. Show "action sequence" instead (last 3 interactions)

**Recommendation**: Option 2 is interesting but out of scope. Stick with Option 1 for Phase 7.

---

## Conclusion

Phase 7 is **well-scoped** and **low-risk**:

1. **Minimal Backend Changes**: Add substrate metadata to 3 WebSocket messages (~50 lines)
2. **Isolated Frontend Changes**: Create AspatialView component, add dispatcher to App.vue (~200 lines)
3. **Strong Backward Compatibility**: Legacy checkpoints work without crashes
4. **Performance Improvement**: Aspatial mode is faster than spatial

**Critical Success Factors**:
- Backend sends substrate metadata in all relevant messages
- Frontend gracefully handles missing substrate (legacy fallback)
- Aspatial mode provides useful information without misleading spatial metaphors

**Estimated Effort**: 4-6 hours (smaller than initial estimate due to clean architecture)

**Readiness**: Ready for implementation. All questions answered, design validated.
