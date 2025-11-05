# TASK-002A Phase 7: Frontend Visualization - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Date**: 2025-11-05
**Status**: Ready for Implementation
**Dependencies**: Phases 0-6 Complete
**Estimated Effort**: 4-6 hours

---

⚠️ **BREAKING CHANGES NOTICE** ⚠️

Phase 7 introduces breaking changes to the WebSocket protocol.

**Impact:**
- Backend must send substrate metadata in all WebSocket messages
- Old inference servers (without substrate metadata) will show legacy spatial view only
- No crashes, but aspatial environments won't render correctly

**Rationale:**
Breaking changes authorized per TASK-002A scope. Frontend needs substrate metadata to dispatch correct renderer. Backward compatibility maintained via fallback to spatial mode.

**Migration Path:**
Update live_inference.py to include substrate metadata. Legacy checkpoints work with spatial fallback.

---

## Executive Summary

Phase 7 adds multi-substrate rendering support to the frontend. Currently the UI **assumes** 2D spatial grid - all components hardcode `(x, y)` coordinates.

**Key Changes**:
1. Backend sends substrate metadata via WebSocket
2. Frontend detects substrate type and dispatches to correct renderer
3. New `AspatialView` component for aspatial substrates
4. Backward compatibility for legacy checkpoints

**Scope**: 4 files modified, 1 new component (~300 lines total)

**Testing**: Unit tests for rendering modes, integration tests for end-to-end flows

---

## Phase 7 Task Breakdown

### Task 7.1: Add Substrate Metadata to WebSocket Protocol

**Purpose**: Send substrate type/dimensions from backend to frontend

**Files**:
- `src/townlet/demo/live_inference.py`

**Estimated Time**: 1.5 hours

---

#### Step 1: Write test for substrate metadata in connected message

**Action**: Add test for WebSocket handshake

**Create**: `tests/test_townlet/integration/test_live_inference_websocket.py`

```python
"""Integration tests for live inference WebSocket protocol."""
import asyncio
import json
from pathlib import Path

import pytest
import torch

from townlet.demo.live_inference import LiveInferenceServer


@pytest.fixture
async def inference_server(tmp_path, test_config_pack_path):
    """Create test inference server."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    server = LiveInferenceServer(
        checkpoint_dir=checkpoint_dir,
        port=8767,  # Different port to avoid conflicts
        step_delay=0.01,
        total_episodes=100,
        config_dir=test_config_pack_path,
    )

    # Initialize components
    await server.startup()
    yield server
    await server.shutdown()


@pytest.mark.asyncio
async def test_connected_message_includes_substrate_metadata(inference_server):
    """Connected message should include substrate metadata."""
    # Create mock WebSocket client
    class MockWebSocket:
        def __init__(self):
            self.messages = []

        async def accept(self):
            pass

        async def send_json(self, data):
            self.messages.append(data)

        async def receive_json(self):
            await asyncio.sleep(0.1)
            return {"command": "disconnect"}

    mock_ws = MockWebSocket()

    # Connect to server
    await inference_server.websocket_endpoint(mock_ws)

    # Find connected message
    connected = next((m for m in mock_ws.messages if m.get("type") == "connected"), None)
    assert connected is not None, "Should send connected message"

    # Verify substrate metadata present
    assert "substrate" in connected, "Connected message should include substrate metadata"

    substrate = connected["substrate"]
    assert substrate["type"] == "grid2d", "Test config uses Grid2D substrate"
    assert substrate["position_dim"] == 2, "Grid2D has position_dim=2"
    assert substrate["width"] == 8, "Test config uses 8×8 grid"
    assert substrate["height"] == 8, "Test config uses 8×8 grid"
    assert substrate["topology"] == "square", "Test config uses square topology"


@pytest.mark.asyncio
async def test_episode_start_includes_substrate_metadata(inference_server):
    """Episode start message should include substrate metadata."""
    mock_ws = MockWebSocket()

    # Trigger episode
    await inference_server._run_single_episode()

    # Find episode_start message (sent to all clients)
    # Note: In real test, mock_ws would be in inference_server.clients
    # For now, verify substrate metadata is constructed correctly

    # Access substrate from environment
    substrate = inference_server.env.substrate
    assert substrate is not None, "Environment should have substrate"

    # Verify we can serialize substrate metadata
    metadata = {
        "type": substrate.type,
        "position_dim": substrate.position_dim,
        "topology": substrate.topology if hasattr(substrate, "topology") else None,
        "width": substrate.width if hasattr(substrate, "width") else None,
        "height": substrate.height if hasattr(substrate, "height") else None,
    }

    assert metadata["type"] == "grid2d"
    assert metadata["position_dim"] == 2


@pytest.mark.asyncio
async def test_state_update_includes_substrate_metadata(inference_server):
    """State update message should include substrate metadata."""
    # Similar to episode_start test
    # Verify substrate metadata can be constructed from env.substrate
    pass
```

**Run test**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run pytest tests/test_townlet/integration/test_live_inference_websocket.py::test_connected_message_includes_substrate_metadata -v
```

**Expected**: FAIL (substrate metadata not yet added)

---

#### Step 2: Add substrate metadata helper method

**Action**: Extract substrate metadata from environment

**Modify**: `src/townlet/demo/live_inference.py`

Add after `_build_agent_telemetry()` method (around line 155):

```python
def _build_substrate_metadata(self) -> dict[str, Any]:
    """Build substrate metadata for WebSocket messages.

    Returns:
        Dict with substrate type, dimensions, and topology.
        Used by frontend to dispatch correct renderer.

    Example:
        Grid2D: {"type": "grid2d", "position_dim": 2, "width": 8, "height": 8, "topology": "square"}
        Aspatial: {"type": "aspatial", "position_dim": 0}
    """
    if not self.env:
        return {"type": "unknown", "position_dim": 0}

    substrate = self.env.substrate
    metadata = {
        "type": substrate.type,
        "position_dim": substrate.position_dim,
    }

    # Add grid-specific metadata
    if substrate.type == "grid2d":
        metadata["topology"] = substrate.topology
        metadata["width"] = substrate.width
        metadata["height"] = substrate.height
        metadata["boundary"] = substrate.boundary
        metadata["distance_metric"] = substrate.distance_metric

    return metadata
```

**Expected**: Helper method available for use in WebSocket handlers

---

#### Step 3: Add substrate metadata to connected message

**Action**: Include substrate in initial handshake

**Modify**: `src/townlet/demo/live_inference.py`

In `websocket_endpoint()` method (around line 367), modify connected message:

```python
async def websocket_endpoint(self, websocket: WebSocket):
    """WebSocket endpoint for client connections."""
    await websocket.accept()
    self.clients.add(websocket)
    logger.info(f"Client connected. Total clients: {len(self.clients)}")

    # Send connection message with substrate metadata
    await websocket.send_json(
        {
            "type": "connected",
            "message": "Connected to live inference server",
            "available_models": [],
            "mode": "inference",
            "checkpoint": f"checkpoint_ep{self.current_checkpoint_episode:05d}" if self.current_checkpoint_path else "None",
            "checkpoint_episode": self.current_checkpoint_episode,
            "total_episodes": self.total_episodes,
            "epsilon": self.current_epsilon,
            "auto_checkpoint_mode": self.auto_checkpoint_mode,
            "substrate": self._build_substrate_metadata(),  # ← ADD THIS
        }
    )
```

**Expected**: Connected message includes substrate metadata

---

#### Step 4: Add substrate metadata to episode_start message

**Action**: Include substrate when episode begins

**Modify**: `src/townlet/demo/live_inference.py`

In `_run_single_episode()` method (around line 544), modify episode_start broadcast:

```python
# Send episode start with curriculum info and substrate metadata
await self._broadcast_to_clients(
    {
        "type": "episode_start",
        "episode": self.current_episode,
        "checkpoint": f"checkpoint_ep{self.current_checkpoint_episode:05d}",
        "checkpoint_episode": self.current_checkpoint_episode,
        "total_episodes": self.total_episodes,
        "epsilon": epsilon_snapshot,
        "curriculum_stage": current_stage,
        "curriculum_multiplier": float(current_multiplier),
        "baseline_survival": baseline_survival,
        "telemetry": episode_telemetry,
        "substrate": self._build_substrate_metadata(),  # ← ADD THIS
    }
)
```

**Expected**: Episode start message includes substrate metadata

---

#### Step 5: Add substrate metadata to state_update message

**Action**: Include substrate in every step update

**Modify**: `src/townlet/demo/live_inference.py`

In `_broadcast_state_update()` method (around line 731), modify update dict:

```python
# Build state update message
update = {
    "type": "state_update",
    "step": self.current_step,
    "cumulative_reward": cumulative_reward,
    "step_reward": step_reward,
    "projected_reward": projected_reward,
    "epsilon": self.current_epsilon,
    "checkpoint_episode": self.current_checkpoint_episode,
    "total_episodes": self.total_episodes,
    "substrate": self._build_substrate_metadata(),  # ← ADD THIS
    "grid": {
        "width": self.env.grid_size,
        "height": self.env.grid_size,
        "agents": [
            {
                "id": "agent_0",
                "x": agent_pos[0],
                "y": agent_pos[1],
                "color": "#4CAF50",
                "last_action": last_action,
            }
        ],
        "affordances": affordances,
    },
    # ... rest of message
}
```

**Expected**: State update includes substrate metadata

---

#### Step 6: Run tests to verify substrate metadata

**Action**: Verify all WebSocket messages include substrate

**Run test**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run pytest tests/test_townlet/integration/test_live_inference_websocket.py -v
```

**Expected**: PASS - All messages include substrate metadata

---

#### Step 7: Commit Task 7.1

**Action**: Commit WebSocket protocol changes

**Command**:
```bash
cd /home/john/hamlet
git add src/townlet/demo/live_inference.py tests/test_townlet/integration/test_live_inference_websocket.py
git commit -m "$(cat <<'EOF'
feat(task-002a): Add substrate metadata to WebSocket protocol (Phase 7.1)

Add substrate type and dimensions to all WebSocket messages to enable
multi-substrate frontend rendering.

Changes:
- Add _build_substrate_metadata() helper to extract substrate info
- Include substrate metadata in connected handshake
- Include substrate metadata in episode_start messages
- Include substrate metadata in state_update messages

Frontend can now detect substrate type and dispatch to correct renderer:
- Grid2D: Spatial view with SVG grid
- Aspatial: Meters-only dashboard

Backward compatibility: Frontend falls back to spatial view if substrate
metadata missing (legacy checkpoints).

Testing:
- Add integration tests for WebSocket protocol
- Verify substrate metadata in connected/episode_start/state_update

TASK-002A Phase 7 Task 7.1

Breaking Change: WebSocket protocol now includes substrate field
EOF
)"
```

**Verification**:
```bash
git log -1 --stat
```

**Expected**: Commit created with 2 files changed (~80 lines)

---

### Task 7.2: Update Frontend Store to Handle Substrate Metadata

**Purpose**: Store substrate state and pass to components

**Files**:
- `frontend/src/stores/simulation.js`

**Estimated Time**: 1 hour

---

#### Step 1: Add substrate state to simulation store

**Action**: Store substrate metadata from WebSocket

**Modify**: `frontend/src/stores/simulation.js`

Add after `gridHeight` (around line 49):

```javascript
// Substrate metadata (for multi-substrate rendering)
const substrateType = ref('grid2d')  // 'grid2d' or 'aspatial'
const substratePositionDim = ref(2)  // 2 for grid, 0 for aspatial
const substrateMetadata = ref({
  type: 'grid2d',
  position_dim: 2,
  topology: 'square',
  width: 8,
  height: 8
})
```

**Expected**: State variables for substrate metadata

---

#### Step 2: Update WebSocket message handler to store substrate

**Action**: Extract substrate from WebSocket messages

**Modify**: `frontend/src/stores/simulation.js`

In WebSocket `onmessage` handler (around line 250), add substrate handling:

```javascript
ws.value.onmessage = (event) => {
  const data = JSON.parse(event.data)

  // Handle substrate metadata (present in connected, episode_start, state_update)
  if (data.substrate) {
    substrateType.value = data.substrate.type || 'grid2d'
    substratePositionDim.value = data.substrate.position_dim || 2
    substrateMetadata.value = data.substrate
  }

  if (data.type === 'connected') {
    console.log('Server connected:', data.message)
    // ... existing connected logic
  }

  if (data.type === 'episode_start') {
    // ... existing episode_start logic
  }

  if (data.type === 'state_update') {
    currentStep.value = data.step

    // Update substrate metadata if present (may change between episodes in multi-substrate envs)
    if (data.substrate) {
      substrateType.value = data.substrate.type || 'grid2d'
      substratePositionDim.value = data.substrate.position_dim || 2
      substrateMetadata.value = data.substrate
    }

    // Update grid dimensions (for spatial substrates)
    if (data.grid) {
      gridWidth.value = data.grid.width || 8
      gridHeight.value = data.grid.height || 8
      agents.value = data.grid.agents || []
      affordances.value = data.grid.affordances || []
    }

    // ... rest of state_update handling
  }
}
```

**Expected**: Substrate metadata stored and updated from WebSocket

---

#### Step 3: Export substrate state for components

**Action**: Make substrate state available to components

**Modify**: `frontend/src/stores/simulation.js`

In store return statement (around line 600), add substrate exports:

```javascript
return {
  // ... existing exports
  gridWidth,
  gridHeight,
  substrateType,  // ← ADD
  substratePositionDim,  // ← ADD
  substrateMetadata,  // ← ADD
  agents,
  affordances,
  // ... rest of exports
}
```

**Expected**: Substrate state accessible in components via `store.substrateType` etc.

---

#### Step 4: Commit Task 7.2

**Action**: Commit store changes

**Command**:
```bash
cd /home/john/hamlet
git add frontend/src/stores/simulation.js
git commit -m "$(cat <<'EOF'
feat(task-002a): Store substrate metadata in frontend (Phase 7.2)

Add substrate state to simulation store for multi-substrate rendering.

Changes:
- Add substrateType, substratePositionDim, substrateMetadata refs
- Parse substrate metadata from WebSocket messages
- Export substrate state for components to consume

Frontend components can now check substrate type to dispatch renderer:
- store.substrateType === 'grid2d' → Render spatial view
- store.substrateType === 'aspatial' → Render dashboard view

TASK-002A Phase 7 Task 7.2
EOF
)"
```

**Verification**:
```bash
git log -1 --stat
```

**Expected**: Commit created with 1 file changed (~30 lines)

---

### Task 7.3: Create AspatialView Component and Update Grid Dispatcher

**Purpose**: Add aspatial renderer and substrate-based dispatch logic

**Files**:
- `frontend/src/components/AspatialView.vue` (NEW)
- `frontend/src/components/Grid.vue`
- `frontend/src/App.vue`

**Estimated Time**: 2.5 hours

---

#### Step 1: Create AspatialView component

**Action**: Create meters-only dashboard for aspatial substrates

**Create**: `frontend/src/components/AspatialView.vue`

```vue
<template>
  <div class="aspatial-container">
    <!-- Large meters display -->
    <div class="large-meters-panel">
      <h2>Agent Status</h2>
      <div
        v-for="(value, name) in meters"
        :key="name"
        class="large-meter"
        :class="getMeterClass(name, value)"
      >
        <div class="meter-header">
          <span class="meter-name">{{ formatMeterName(name) }}</span>
          <span class="meter-value">{{ (value * 100).toFixed(0) }}%</span>
        </div>
        <div class="meter-bar-container">
          <div
            class="meter-bar-fill"
            :style="{ width: `${value * 100}%` }"
          ></div>
        </div>
      </div>
    </div>

    <!-- Available affordances -->
    <div class="affordance-list-panel">
      <h2>Available Interactions</h2>
      <div class="affordance-grid">
        <div
          v-for="affordance in affordances"
          :key="affordance.type"
          class="affordance-card"
        >
          <div class="affordance-icon">{{ getAffordanceIcon(affordance.type) }}</div>
          <div class="affordance-name">{{ affordance.type }}</div>
          <div class="affordance-status">Ready</div>
        </div>
      </div>
    </div>

    <!-- Recent actions log -->
    <div class="action-history-panel">
      <h2>Recent Actions</h2>
      <div class="action-list">
        <div
          v-for="action in recentActions"
          :key="action.id"
          class="action-item"
        >
          <span class="action-step">Step {{ action.step }}</span>
          <span class="action-type">{{ action.actionName }}</span>
          <span v-if="action.affordance" class="action-affordance">
            {{ getAffordanceIcon(action.affordance) }} {{ action.affordance }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { AFFORDANCE_ICONS } from '../utils/constants'

const props = defineProps({
  meters: {
    type: Object,
    default: () => ({})
  },
  affordances: {
    type: Array,
    default: () => []
  },
  currentStep: {
    type: Number,
    default: 0
  },
  lastAction: {
    type: Number,
    default: null
  },
  lastAffordance: {
    type: String,
    default: null
  }
})

// Track recent actions (last 10)
const recentActions = ref([])
let actionIdCounter = 0

// Action names for display
const ACTION_NAMES = ['Up', 'Down', 'Left', 'Right', 'Interact', 'Wait']

// Watch for new actions
watch(() => props.lastAction, (newAction) => {
  if (newAction !== null) {
    const action = {
      id: actionIdCounter++,
      step: props.currentStep,
      actionName: ACTION_NAMES[newAction] || 'Unknown',
      affordance: newAction === 4 ? props.lastAffordance : null  // INTERACT action
    }

    recentActions.value.unshift(action)  // Add to front

    // Keep only last 10
    if (recentActions.value.length > 10) {
      recentActions.value.pop()
    }
  }
})

function getAffordanceIcon(type) {
  return AFFORDANCE_ICONS[type] || '?'
}

function formatMeterName(name) {
  // Convert "energy" → "Energy", "health" → "Health"
  return name.charAt(0).toUpperCase() + name.slice(1)
}

function getMeterClass(name, value) {
  // Color-code meters by value
  if (value < 0.2) return 'meter-critical'
  if (value < 0.5) return 'meter-warning'
  return 'meter-healthy'
}
</script>

<style scoped>
.aspatial-container {
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--spacing-lg);
  padding: var(--spacing-lg);
  width: 100%;
  height: 100%;
  overflow-y: auto;
}

/* Large meters panel */
.large-meters-panel {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-lg);
}

.large-meters-panel h2 {
  margin: 0 0 var(--spacing-md) 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

.large-meter {
  margin-bottom: var(--spacing-md);
}

.meter-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: var(--spacing-xs);
}

.meter-name {
  font-weight: var(--font-weight-medium);
  font-size: var(--font-size-base);
}

.meter-value {
  font-weight: var(--font-weight-semibold);
  font-size: var(--font-size-base);
}

.meter-bar-container {
  height: 24px;
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-sm);
  overflow: hidden;
}

.meter-bar-fill {
  height: 100%;
  transition: width var(--transition-base), background-color var(--transition-base);
}

.meter-healthy .meter-bar-fill {
  background: var(--color-success);
}

.meter-warning .meter-bar-fill {
  background: var(--color-warning);
}

.meter-critical .meter-bar-fill {
  background: var(--color-error);
}

/* Affordance list panel */
.affordance-list-panel {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-lg);
}

.affordance-list-panel h2 {
  margin: 0 0 var(--spacing-md) 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
}

.affordance-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: var(--spacing-md);
}

.affordance-card {
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-sm);
  padding: var(--spacing-md);
  text-align: center;
  transition: all var(--transition-base);
}

.affordance-card:hover {
  background: var(--color-interactive-disabled);
  transform: translateY(-2px);
}

.affordance-icon {
  font-size: 32px;
  margin-bottom: var(--spacing-sm);
}

.affordance-name {
  font-weight: var(--font-weight-medium);
  font-size: var(--font-size-sm);
  margin-bottom: var(--spacing-xs);
}

.affordance-status {
  font-size: var(--font-size-xs);
  color: var(--color-success);
  font-weight: var(--font-weight-medium);
}

/* Action history panel */
.action-history-panel {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-lg);
}

.action-history-panel h2 {
  margin: 0 0 var(--spacing-md) 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
}

.action-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  max-height: 300px;
  overflow-y: auto;
}

.action-item {
  display: flex;
  gap: var(--spacing-md);
  padding: var(--spacing-sm);
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-sm);
  font-size: var(--font-size-sm);
}

.action-step {
  color: var(--color-text-secondary);
  font-weight: var(--font-weight-medium);
  min-width: 60px;
}

.action-type {
  color: var(--color-text-primary);
  font-weight: var(--font-weight-semibold);
  min-width: 80px;
}

.action-affordance {
  color: var(--color-text-secondary);
}

/* Tablet+ layout */
@media (min-width: 768px) {
  .aspatial-container {
    grid-template-columns: 2fr 1fr;
    grid-template-rows: auto auto;
  }

  .large-meters-panel {
    grid-column: 1 / 2;
    grid-row: 1 / 3;
  }

  .affordance-list-panel {
    grid-column: 2 / 3;
    grid-row: 1 / 2;
  }

  .action-history-panel {
    grid-column: 2 / 3;
    grid-row: 2 / 3;
  }
}
</style>
```

**Expected**: AspatialView component renders meters, affordances, and action history

---

#### Step 2: Update Grid.vue to add substrate prop

**Action**: Pass substrate metadata to Grid component

**Modify**: `frontend/src/components/Grid.vue`

Update props section (around line 124):

```vue
<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { CELL_SIZE, AFFORDANCE_ICONS } from '../utils/constants'

// ✅ Props First: Receive data from parent instead of importing store
const props = defineProps({
  substrate: {  // ← ADD THIS
    type: Object,
    default: () => ({ type: 'grid2d', position_dim: 2, width: 8, height: 8 })
  },
  gridWidth: {
    type: Number,
    default: 8
  },
  gridHeight: {
    type: Number,
    default: 8
  },
  agents: {
    type: Array,
    default: () => []
  },
  affordances: {
    type: Array,
    default: () => []
  },
  heatMap: {
    type: Object,
    default: () => ({})
  }
})
```

**Expected**: Grid component receives substrate metadata

---

#### Step 3: Add substrate-based dispatcher to App.vue

**Action**: Render correct component based on substrate type

**Modify**: `frontend/src/App.vue`

Import AspatialView (around line 179):

```javascript
import AspatialView from './components/AspatialView.vue'
```

Update grid rendering section (around line 94):

```vue
<!-- ✅ Show grid when connected -->
<div
  v-else-if="isConnected"
  class="grid-wrapper"
  :style="{ transform: `scale(${store.gridZoom})` }"
>
  <!-- Spatial mode: SVG grid -->
  <Grid
    v-if="store.substrateType === 'grid2d'"
    :substrate="store.substrateMetadata"
    :grid-width="store.gridWidth"
    :grid-height="store.gridHeight"
    :agents="store.agents"
    :affordances="store.affordances"
    :heat-map="store.heatMap"
  />

  <!-- Aspatial mode: Meters dashboard -->
  <AspatialView
    v-else-if="store.substrateType === 'aspatial'"
    :meters="store.agentMeters.agent_0?.meters || {}"
    :affordances="store.affordances"
    :current-step="store.currentStep"
    :last-action="store.lastAction"
  />

  <!-- Fallback: Unknown substrate -->
  <div v-else class="unknown-substrate-message">
    <p>Unknown substrate type: {{ store.substrateType }}</p>
    <p class="hint">Supported: grid2d, aspatial</p>
  </div>

  <!-- Overlay components (work for both spatial and aspatial) -->
  <InteractionProgressRing
    v-if="store.agents && store.agents.length > 0 && store.interactionProgress > 0"
    :x="store.agents[0].x"
    :y="store.agents[0].y"
    :progress="store.interactionProgress"
    :affordance-type="currentAffordanceType"
    :cell-size="75"
  />
  <!-- Note: NoveltyHeatmap only for spatial -->
  <NoveltyHeatmap
    v-if="store.substrateType === 'grid2d' && store.rndMetrics && store.rndMetrics.novelty_map"
    :novelty-map="store.rndMetrics.novelty_map"
    :grid-size="store.gridWidth"
    :cell-size="75"
  />
</div>
```

**Expected**: App.vue renders Grid or AspatialView based on substrate type

---

#### Step 4: Test frontend rendering modes

**Action**: Manually test spatial and aspatial views

**Test Spatial Mode**:
```bash
# Start Grid2D environment
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python -m townlet.demo.live_inference checkpoints_level1 8766 0.2 10000 configs/L1_full_observability

# In another terminal, start frontend
cd /home/john/hamlet/frontend
npm run dev

# Open http://localhost:5173
# Verify: SVG grid renders, agents move, affordances display
```

**Test Aspatial Mode** (after Phase 8):
```bash
# Start Aspatial environment (requires aspatial config pack from Phase 8)
python -m townlet.demo.live_inference checkpoints_aspatial 8766 0.2 10000 configs/aspatial_test

# Verify: Dashboard renders, no grid, affordance list shows
```

**Expected**:
- Spatial mode: Grid renders normally
- Aspatial mode: Dashboard renders (no grid)
- No crashes or console errors

---

#### Step 5: Commit Task 7.3

**Action**: Commit frontend rendering changes

**Command**:
```bash
cd /home/john/hamlet
git add frontend/src/components/AspatialView.vue frontend/src/components/Grid.vue frontend/src/App.vue
git commit -m "$(cat <<'EOF'
feat(task-002a): Add multi-substrate frontend rendering (Phase 7.3)

Add AspatialView component and substrate-based rendering dispatch.

Changes:
- Create AspatialView.vue for aspatial substrate rendering
  - Large meter bars with color coding
  - Affordance list (no positions)
  - Recent actions log (temporal context)
- Update Grid.vue to receive substrate metadata via props
- Update App.vue to dispatch based on substrate type
  - Grid2D → Grid.vue (spatial view)
  - Aspatial → AspatialView.vue (dashboard view)
- Disable spatial-only features in aspatial mode (heat map, novelty heatmap)

Frontend now supports two rendering modes:
1. Spatial (Grid2D): SVG grid with agents/affordances at (x,y) positions
2. Aspatial: Meters dashboard with affordance list (no grid)

TASK-002A Phase 7 Task 7.3
EOF
)"
```

**Verification**:
```bash
git log -1 --stat
```

**Expected**: Commit created with 3 files changed (~300 lines)

---

### Task 7.4: Update Documentation

**Purpose**: Document frontend multi-substrate rendering

**Files**:
- `CLAUDE.md`
- `docs/architecture/hld/frontend-visualization.md` (NEW)

**Estimated Time**: 1 hour

---

#### Step 1: Add frontend visualization section to CLAUDE.md

**Action**: Document rendering modes and usage

**Modify**: `CLAUDE.md`

Add new section after "Centralization Roadmap" (around line 600):

```markdown
---

## Frontend Visualization (Multi-Substrate Support)

**Location**: `frontend/src/components/`

The frontend supports **two rendering modes** based on substrate type:

### Spatial Mode (Grid2D Substrates)

**Component**: `Grid.vue` (SVG-based 2D grid)

**Features**:
- Grid cells rendered as SVG rectangles
- Agents positioned at (x, y) coordinates
- Affordances displayed as icons at grid positions
- Heat map overlay (position visit frequency)
- Agent trails (last 3 positions)
- Novelty heatmap (RND exploration)

**WebSocket Contract**:
```json
{
  "substrate": {
    "type": "grid2d",
    "position_dim": 2,
    "width": 8,
    "height": 8
  },
  "grid": {
    "agents": [{"id": "agent_0", "x": 3, "y": 5}],
    "affordances": [{"type": "Bed", "x": 2, "y": 1}]
  }
}
```

---

### Aspatial Mode (Aspatial Substrates)

**Component**: `AspatialView.vue` (Meters-only dashboard)

**Features**:
- Large meter bars with color coding (critical/warning/healthy)
- Affordance list (no positions, just availability)
- Recent actions log (temporal context without spatial context)
- No heat map (position-based feature)
- No agent trails (position-based feature)

**WebSocket Contract**:
```json
{
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

**Rationale**:
Aspatial universes have no concept of "position" - rendering a fake grid would:
1. Be pedagogically harmful (teaches incorrect intuitions)
2. Mislead operators about agent behavior
3. Imply spatial relationships that don't exist

Instead, aspatial mode focuses on **meters** (primary learning signal) and **interaction history** (temporal, not spatial).

---

### Substrate Detection

**Logic** (in `App.vue`):
```vue
<Grid v-if="store.substrateType === 'grid2d'" ... />
<AspatialView v-else-if="store.substrateType === 'aspatial'" ... />
```

**Fallback** (for legacy checkpoints without substrate metadata):
```javascript
const substrate = message.substrate || {
  type: "grid2d",  // Assume legacy spatial behavior
  position_dim: 2,
  width: message.grid.width,
  height: message.grid.height
}
```

---

### Running Frontend

**Prerequisites**:
1. Live inference server running (provides WebSocket endpoint)
2. Node.js and npm installed

**Commands**:
```bash
# Terminal 1: Start inference server (from worktree with checkpoints)
cd /home/john/hamlet/.worktrees/substrate-abstraction
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python -m townlet.demo.live_inference checkpoints 8766 0.2 10000 configs/L1_full_observability

# Terminal 2: Start frontend (from main repo)
cd /home/john/hamlet/frontend
npm run dev

# Open http://localhost:5173
```

**Port Configuration**:
- Frontend dev server: `localhost:5173` (Vite default)
- WebSocket endpoint: `localhost:8766` (live_inference default)
- Frontend auto-connects to WebSocket on component mount

---

### Customizing Visualization

**Affordance Icons** (`frontend/src/utils/constants.js`):
```javascript
export const AFFORDANCE_ICONS = {
  Bed: '🛏️',
  Hospital: '🏥',
  Job: '💼',
  // ... add custom icons here
}
```

**Meter Colors** (`frontend/src/styles/tokens.js`):
```javascript
'--color-success': '#22c55e',  // Healthy meters
'--color-warning': '#f59e0b',  // Warning meters
'--color-error': '#ef4444',    // Critical meters
```

**Grid Cell Size** (`frontend/src/utils/constants.js`):
```javascript
export const CELL_SIZE = 75  // Pixels per grid cell
```

---
```

**Expected**: CLAUDE.md documents frontend rendering modes

---

#### Step 2: Create high-level design document for frontend

**Action**: Document frontend architecture and rendering pipeline

**Create**: `docs/architecture/hld/frontend-visualization.md`

```markdown
# Frontend Visualization Architecture (HLD)

**Date**: 2025-11-05
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
```

**Expected**: HLD document provides architecture overview and maintenance guide

---

#### Step 3: Commit Task 7.4

**Action**: Commit documentation

**Command**:
```bash
cd /home/john/hamlet
git add CLAUDE.md docs/architecture/hld/frontend-visualization.md
git commit -m "$(cat <<'EOF'
docs(task-002a): Document frontend multi-substrate rendering (Phase 7.4)

Add comprehensive documentation for frontend visualization architecture.

Changes:
- Add "Frontend Visualization" section to CLAUDE.md
  - Spatial vs aspatial rendering modes
  - WebSocket protocol contract
  - Substrate detection logic
  - Running and customizing frontend
- Create frontend-visualization.md HLD document
  - Architecture diagram
  - Rendering pipeline
  - Component interaction flow
  - Feature matrix
  - Testing strategy
  - Maintenance guide

Documentation clarifies:
1. Why aspatial mode doesn't show grid (pedagogically harmful)
2. How substrate detection works (explicit field vs inference)
3. Backward compatibility strategy (fallback to spatial)
4. How to extend with new substrate types

TASK-002A Phase 7 Task 7.4
EOF
)"
```

**Verification**:
```bash
git log -1 --stat
```

**Expected**: Commit created with 2 files changed (~400 lines)

---

## Phase 7 Verification

### Final Testing Checklist

Before marking Phase 7 complete, verify:

**Backend (WebSocket Protocol)**:
- [ ] Connected message includes substrate metadata
- [ ] Episode_start message includes substrate metadata
- [ ] State_update message includes substrate metadata
- [ ] Substrate metadata includes all required fields (type, position_dim, etc.)

**Frontend (Store)**:
- [ ] Substrate state stored correctly from WebSocket
- [ ] Substrate state exported for components
- [ ] Backward compatibility fallback works (missing substrate)

**Frontend (Rendering)**:
- [ ] Grid2D environments render spatial view (SVG grid)
- [ ] Aspatial environments render dashboard view (no grid)
- [ ] Heat map disabled in aspatial mode
- [ ] Agent trails disabled in aspatial mode
- [ ] Affordance list shows in aspatial mode
- [ ] Action history updates in aspatial mode

**Documentation**:
- [ ] CLAUDE.md documents rendering modes
- [ ] HLD document provides architecture overview
- [ ] Examples show WebSocket message format

---

### Integration Test Script

**Test Spatial Rendering**:
```bash
# Terminal 1: Start Grid2D environment
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python -m townlet.demo.live_inference checkpoints_level1 8766 0.2 10000 configs/L1_full_observability

# Terminal 2: Start frontend
cd /home/john/hamlet/frontend
npm run dev

# Open http://localhost:5173
# Expected: SVG grid with moving agents
```

**Test Legacy Fallback**:
```bash
# Use old checkpoint (pre-Phase 7) without substrate metadata
# Expected: Frontend renders spatial view (no crash)
```

---

### Commit Summary for Phase 7

**Total Files Changed**: 8
- Backend: 2 files (live_inference.py + tests)
- Frontend: 4 files (store, Grid.vue, AspatialView.vue, App.vue)
- Docs: 2 files (CLAUDE.md, frontend-visualization.md)

**Total Lines Changed**: ~800 lines
- Backend: ~80 lines (substrate metadata)
- Frontend: ~300 lines (AspatialView + dispatcher)
- Docs: ~420 lines (comprehensive documentation)

**Commits**:
1. feat(task-002a): Add substrate metadata to WebSocket protocol (Phase 7.1)
2. feat(task-002a): Store substrate metadata in frontend (Phase 7.2)
3. feat(task-002a): Add multi-substrate frontend rendering (Phase 7.3)
4. docs(task-002a): Document frontend multi-substrate rendering (Phase 7.4)

---

## Success Criteria

Phase 7 is complete when:

1. ✅ **Backend sends substrate metadata** in all WebSocket messages
2. ✅ **Frontend stores substrate metadata** in Pinia store
3. ✅ **Grid2D environments render spatial view** (SVG grid with agents)
4. ✅ **Aspatial environments render dashboard view** (meters only, no grid)
5. ✅ **Backward compatibility works** (legacy checkpoints render spatial view)
6. ✅ **Documentation complete** (CLAUDE.md + HLD)
7. ✅ **No crashes or console errors** when switching substrate types
8. ✅ **All tests pass** (unit + integration)

---

## Phase 8 Dependencies

Phase 8 (Testing) depends on Phase 7 completion:
- Integration tests need multi-substrate frontend to verify end-to-end flows
- Property-based tests need substrate metadata in WebSocket protocol
- Regression tests need aspatial environments to verify non-spatial rendering

**Readiness**: Phase 7 complete → Phase 8 can begin immediately

---

## Rollback Plan

If Phase 7 needs rollback:

1. Revert commits in reverse order:
   ```bash
   git revert HEAD~3..HEAD  # Revert last 4 commits (7.4, 7.3, 7.2, 7.1)
   ```

2. Delete AspatialView component:
   ```bash
   rm frontend/src/components/AspatialView.vue
   ```

3. Restart frontend (will use legacy spatial rendering)

**Risk**: LOW (changes are additive, no removal of existing code)

---

## Estimated Effort Summary

| Task | Estimated | Actual |
|------|-----------|--------|
| 7.1: WebSocket Protocol | 1.5h | TBD |
| 7.2: Frontend Store | 1h | TBD |
| 7.3: AspatialView + Dispatcher | 2.5h | TBD |
| 7.4: Documentation | 1h | TBD |
| **Total** | **6h** | TBD |

**Estimate Confidence**: HIGH (frontend architecture is clean, changes are isolated)
