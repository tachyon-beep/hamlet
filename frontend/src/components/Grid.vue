<template>
  <div class="grid-container">
    <!-- ✅ Heat map toggle with keyboard accessibility -->
    <button
      v-if="Object.keys(props.heatMap).length > 0"
      @click="toggleHeatMap"
      class="heat-map-toggle"
      :class="{ active: showHeatMap }"
      :aria-label="`${showHeatMap ? 'Hide' : 'Show'} heat map showing position visit frequency. Keyboard shortcut: H`"
      :aria-pressed="showHeatMap"
      title="Toggle heat map (Keyboard: H)"
    >
      {{ showHeatMap ? 'Hide' : 'Show' }} Heat Map
    </button>

    <!-- ✅ SVG with role and comprehensive aria-label -->
    <svg
      :viewBox="`0 0 ${props.gridWidth * cellSize} ${props.gridHeight * cellSize}`"
      class="grid-svg"
      role="img"
      :aria-label="gridAriaLabel"
    >
      <!-- Grid cells -->
      <g v-for="y in props.gridHeight" :key="`row-${y}`">
        <rect
          v-for="x in props.gridWidth"
          :key="`cell-${x}-${y}`"
          :x="(x - 1) * cellSize"
          :y="(y - 1) * cellSize"
          :width="cellSize"
          :height="cellSize"
          class="grid-cell"
        />
      </g>

      <!-- Heat map overlay (position visit frequency) -->
      <g v-if="showHeatMap && Object.keys(props.heatMap).length > 0" class="heat-map-layer">
        <rect
          v-for="(intensity, key) in props.heatMap"
          :key="`heat-${key}`"
          :x="getHeatX(key) * cellSize"
          :y="getHeatY(key) * cellSize"
          :width="cellSize"
          :height="cellSize"
          :fill="getHeatColor(intensity)"
          :opacity="0.6"
          class="heat-map-cell"
        />
      </g>

      <!-- Affordances -->
    <g v-for="affordance in props.affordances" :key="`affordance-${affordance.x}-${affordance.y}`">
      <rect
        :x="affordance.x * cellSize + cellSize * 0.1"
        :y="affordance.y * cellSize + cellSize * 0.1"
        :width="cellSize * 0.8"
        :height="cellSize * 0.8"
        :class="['affordance', `affordance-${affordance.type.toLowerCase()}`]"
        rx="8"
      />
      <text
        :x="affordance.x * cellSize + cellSize / 2"
        :y="affordance.y * cellSize + cellSize / 2"
        class="affordance-label"
        text-anchor="middle"
        dominant-baseline="middle"
      >
        {{ getAffordanceIcon(affordance.type) }}
      </text>
    </g>

    <!-- Agent trails (last 3 positions with fading opacity) -->
    <g v-for="agent in validAgents" :key="`trail-${agent.id}`" class="agent-trail-group">
      <circle
        v-for="(pos, index) in getAgentTrail(agent.id)"
        :key="`trail-${agent.id}-${index}`"
        :cx="pos.x * cellSize + cellSize / 2"
        :cy="pos.y * cellSize + cellSize / 2"
        :r="cellSize * 0.2"
        :fill="agent.color"
        :opacity="(index + 1) * 0.2"
        class="trail-dot"
      />
    </g>

    <!-- Agents -->
    <g v-for="agent in validAgents" :key="agent.id" class="agent-group">
      <!-- Outer glow with pulse animation -->
      <circle
        :cx="agent.x * cellSize + cellSize / 2"
        :cy="agent.y * cellSize + cellSize / 2"
        :r="cellSize * 0.45"
        :fill="agent.color"
        class="agent-glow"
        opacity="0.3"
      />
      <!-- Main agent circle (increased from 0.3 to 0.35) -->
      <circle
        :cx="agent.x * cellSize + cellSize / 2"
        :cy="agent.y * cellSize + cellSize / 2"
        :r="cellSize * 0.35"
        :fill="agent.color"
        class="agent-circle"
      />
      <text
        :x="agent.x * cellSize + cellSize / 2"
        :y="agent.y * cellSize + cellSize / 2"
        class="agent-label"
        text-anchor="middle"
        dominant-baseline="middle"
      >
        {{ agent.id ? agent.id.replace('agent_', '') : '?' }}
      </text>
    </g>
  </svg>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { CELL_SIZE, AFFORDANCE_ICONS } from '../utils/constants'

// ✅ Props First: Receive data from parent instead of importing store
const props = defineProps({
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
  },
  substrate: {
    type: Object,
    default: () => ({ type: 'grid2d', position_dim: 2 })
  }
})

// Filter valid agents
const validAgents = computed(() => (props.agents || []).filter(a => a && a.id))

// Agent trail tracking (last 3 positions for each agent)
const agentTrails = ref({})

// Watch for agent position changes and update trail
watch(() => props.agents, (newAgents, oldAgents) => {
  if (!newAgents || newAgents.length === 0) return

  newAgents.forEach(agent => {
    if (!agent || !agent.id) return

    const currentPos = { x: agent.x, y: agent.y }

    // Initialize trail for new agents
    if (!agentTrails.value[agent.id]) {
      agentTrails.value[agent.id] = []
    }

    const trail = agentTrails.value[agent.id]

    // Check if position changed
    const lastPos = trail.length > 0 ? trail[trail.length - 1] : null
    const positionChanged = !lastPos || lastPos.x !== currentPos.x || lastPos.y !== currentPos.y

    if (positionChanged) {
      // Add current position to trail
      trail.push(currentPos)

      // Keep only last 3 positions
      if (trail.length > 3) {
        trail.shift()
      }
    }
  })
}, { deep: true })

// ✅ Use imported constant
const cellSize = CELL_SIZE

const showHeatMap = ref(true) // Show heat map by default

// Get trail positions for an agent (excluding current position)
function getAgentTrail(agentId) {
  const trail = agentTrails.value[agentId]
  if (!trail || trail.length <= 1) return []
  // Return all but the last position (last position is current agent position)
  return trail.slice(0, -1)
}

// ✅ Computed aria-label for screen readers
const gridAriaLabel = computed(() => {
  const agentCount = validAgents.value.length
  const affordanceCount = props.affordances.length
  const heatMapStatus = showHeatMap.value ? 'Heat map is visible showing agent movement patterns' : 'Heat map is hidden'

  return `${props.gridWidth} by ${props.gridHeight} simulation grid. ${agentCount} agent${agentCount !== 1 ? 's' : ''} visible. ${affordanceCount} affordance${affordanceCount !== 1 ? 's' : ''} available: ${props.affordances.map(a => a.type).join(', ')}. ${heatMapStatus}.`
})

// ✅ Toggle heat map function
function toggleHeatMap() {
  showHeatMap.value = !showHeatMap.value
}

// ✅ Keyboard shortcut handler
function handleKeyPress(event) {
  // H key toggles heat map
  if (event.key === 'h' || event.key === 'H') {
    // Only trigger if not typing in an input
    if (event.target.tagName !== 'INPUT' && event.target.tagName !== 'TEXTAREA') {
      toggleHeatMap()
      event.preventDefault()
    }
  }
}

// ✅ Add/remove keyboard listener
onMounted(() => {
  window.addEventListener('keydown', handleKeyPress)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyPress)
})

// ✅ Use imported constant from utils
function getAffordanceIcon(type) {
  return AFFORDANCE_ICONS[type] || '?'
}

// Heat map helpers
function getHeatX(key) {
  const [x] = key.split(',').map(Number)
  return x
}

function getHeatY(key) {
  const [, y] = key.split(',').map(Number)
  return y
}

function getHeatColor(intensity) {
  // Intensity is 0-1, map to color gradient: blue → cyan → yellow → red
  if (intensity < 0.25) {
    // Blue to cyan
    const t = intensity / 0.25
    return `rgb(${Math.round(0 + t * 0)}, ${Math.round(100 + t * 150)}, ${Math.round(255 - t * 55)})`
  } else if (intensity < 0.5) {
    // Cyan to green
    const t = (intensity - 0.25) / 0.25
    return `rgb(${Math.round(0 + t * 0)}, ${Math.round(250 - t * 50)}, ${Math.round(200 - t * 200)})`
  } else if (intensity < 0.75) {
    // Green to yellow
    const t = (intensity - 0.5) / 0.25
    return `rgb(${Math.round(0 + t * 255)}, ${Math.round(200 + t * 55)}, ${Math.round(0)})`
  } else {
    // Yellow to red
    const t = (intensity - 0.75) / 0.25
    return `rgb(${Math.round(255)}, ${Math.round(255 - t * 255)}, ${Math.round(0)})`
  }
}
</script>

<style scoped>
/* ✅ Refactored to use design tokens */
.grid-container {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.heat-map-toggle {
  position: absolute;
  top: var(--spacing-sm);
  right: var(--spacing-sm);
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-interactive-disabled);
  border-radius: var(--border-radius-sm);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
  cursor: pointer;
  transition: all var(--transition-base);
  z-index: var(--z-index-dropdown);
}

.heat-map-toggle:hover {
  background: var(--color-interactive-disabled);
  color: var(--color-text-primary);
  border-color: var(--color-text-tertiary);
}

.heat-map-toggle.active {
  background: var(--color-success);
  color: white;
  border-color: var(--color-interactive-hover);
}

/* ✅ Mobile-first: responsive grid sizing */
.grid-svg {
  width: 100%;
  height: auto;
  max-width: 100%;
  max-height: 400px;
}

@media (min-width: 768px) {
  .grid-svg {
    max-height: 500px;
  }
}

@media (min-width: 1024px) {
  .grid-svg {
    max-width: var(--layout-max-grid-size);
    max-height: var(--layout-max-grid-size);
  }
}

.grid-cell {
  fill: var(--color-bg-secondary);
  stroke: var(--color-bg-tertiary);
  stroke-width: 1;
}

.heat-map-cell {
  pointer-events: none;
  transition: opacity var(--transition-base);
}

.affordance {
  stroke-width: 2;
  transition: all var(--transition-base);
}

/*
 * Affordance colors - SYSTEMATIC by category
 * Each category uses shades of a common color family
 */

/* === REST/SLEEP (Purple family) === */
.affordance-bed {
  fill: #6366f1;  /* Indigo - standard rest */
  stroke: #818cf8;
}

.affordance-luxurybed {
  fill: #6b21a8;  /* Deep purple - premium rest */
  stroke: #7e22ce;
}

/* === HYGIENE (Cyan/Teal family) === */
.affordance-shower {
  fill: #0891b2;  /* Dark cyan */
  stroke: #06b6d4;
}

/* === FOOD (Orange/Amber family) === */
.affordance-homemeal {
  fill: #f59e0b;  /* Amber - home cooking */
  stroke: #fbbf24;
}

.affordance-fastfood {
  fill: #ef4444;  /* Red - unhealthy fast food */
  stroke: #f87171;
}

/* === WORK/INCOME (Professional colors) === */
.affordance-job {
  fill: #8b5cf6;  /* Purple - office/professional */
  stroke: #a78bfa;
}

.affordance-labor {
  fill: #f97316;  /* Orange - physical labor */
  stroke: #fb923c;
}

/* === SOCIAL (Pink/Magenta family) === */
.affordance-bar {
  fill: #ec4899;  /* Pink - social venue */
  stroke: #f472b6;
}

/* === FITNESS (Green family) === */
.affordance-gym {
  fill: #10b981;  /* Emerald - active fitness */
  stroke: #34d399;
}

.affordance-park {
  fill: #059669;  /* Darker emerald - outdoor fitness */
  stroke: #10b981;
}

/* === MOOD (Blue/Fuchsia family) === */
.affordance-recreation {
  fill: #3b82f6;  /* Blue - casual entertainment */
  stroke: #60a5fa;
}

.affordance-therapist {
  fill: #c026d3;  /* Fuchsia - professional mental health */
  stroke: #d946ef;
}

/* === HEALTH/MEDICAL (Dark red/maroon family - dark for light icons) === */
.affordance-doctor {
  fill: #991b1b;  /* Dark red - medical tier 1 */
  stroke: #b91c1c;
}

.affordance-hospital {
  fill: #7f1d1d;  /* Darker maroon - medical tier 2 (emergency) */
  stroke: #991b1b;
}

.affordance-label {
  font-size: 32px;
  pointer-events: none;
  user-select: none;
}

.agent-trail-group {
  pointer-events: none;
}

.trail-dot {
  transition: opacity var(--transition-base);
}

.agent-group {
  transition: all var(--transition-base);
}

.agent-glow {
  animation: agent-pulse 2s ease-in-out infinite;
  transform-box: fill-box;
  transform-origin: center;
}

@keyframes agent-pulse {
  0%, 100% {
    opacity: 0.3;
    transform: scale(1);
  }
  50% {
    opacity: 0.5;
    transform: scale(1.1);
  }
}

.agent-circle {
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
}

.agent-label {
  fill: white;
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  pointer-events: none;
  user-select: none;
}
</style>
