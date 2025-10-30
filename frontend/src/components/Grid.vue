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

    <!-- Agents -->
    <g v-for="agent in validAgents" :key="agent.id" class="agent-group">
      <circle
        :cx="agent.x * cellSize + cellSize / 2"
        :cy="agent.y * cellSize + cellSize / 2"
        :r="cellSize * 0.3"
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
import { ref, computed, onMounted, onUnmounted } from 'vue'
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
  }
})

// Filter valid agents
const validAgents = computed(() => (props.agents || []).filter(a => a && a.id))

// ✅ Use imported constant
const cellSize = CELL_SIZE

const showHeatMap = ref(true) // Show heat map by default

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

/* Affordance colors - using semantic colors */
.affordance-bed {
  fill: #6366f1;
  stroke: var(--color-affordance-bed-stroke);
}

.affordance-shower {
  fill: var(--color-meter-hygiene);
  stroke: var(--color-affordance-shower-stroke);
}

.affordance-homemeal {
  fill: var(--color-meter-satiation);
  stroke: var(--color-affordance-homemeal-stroke);
}

.affordance-fastfood {
  fill: var(--color-error);
  stroke: var(--color-affordance-fastfood-stroke);
}

.affordance-job {
  fill: var(--color-meter-money);
  stroke: var(--color-affordance-job-stroke);
}

.affordance-recreation {
  fill: var(--color-success);
  stroke: var(--color-interactive-hover);
}

.affordance-bar {
  fill: var(--color-meter-social);
  stroke: var(--color-affordance-bar-stroke);
}

.affordance-gym {
  fill: var(--color-meter-mood-high);
  stroke: var(--color-affordance-gym-stroke);
}

/* New affordances from coupled cascade architecture */
.affordance-luxurybed {
  fill: #8b5cf6;  /* Brighter purple than Bed */
  stroke: var(--color-affordance-bed-stroke);
}

.affordance-labor {
  fill: #f59e0b;  /* Orange-gold for physical work */
  stroke: var(--color-warning);
}

.affordance-park {
  fill: #10b981;  /* Green for nature/outdoor */
  stroke: var(--color-success);
}

.affordance-therapist {
  fill: #ec4899;  /* Pink for mental health professional */
  stroke: var(--color-error);
}

.affordance-doctor {
  fill: #3b82f6;  /* Blue for medical (tier 1) */
  stroke: var(--color-chart-primary);
}

.affordance-hospital {
  fill: #2563eb;  /* Darker blue for medical (tier 2) */
  stroke: var(--color-chart-primary);
}

.affordance-label {
  font-size: 32px;
  pointer-events: none;
  user-select: none;
}

.agent-group {
  transition: all var(--transition-base);
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
