<template>
  <div class="grid-container">
    <!-- Heat map toggle button -->
    <button
      v-if="Object.keys(heatMap).length > 0"
      @click="showHeatMap = !showHeatMap"
      class="heat-map-toggle"
      :class="{ active: showHeatMap }"
    >
      {{ showHeatMap ? 'Hide' : 'Show' }} Heat Map
    </button>

    <svg
      :viewBox="`0 0 ${gridWidth * cellSize} ${gridHeight * cellSize}`"
      class="grid-svg"
    >
      <!-- Grid cells -->
      <g v-for="y in gridHeight" :key="`row-${y}`">
        <rect
          v-for="x in gridWidth"
          :key="`cell-${x}-${y}`"
          :x="(x - 1) * cellSize"
          :y="(y - 1) * cellSize"
          :width="cellSize"
          :height="cellSize"
          class="grid-cell"
        />
      </g>

      <!-- Heat map overlay (position visit frequency) -->
      <g v-if="showHeatMap && Object.keys(heatMap).length > 0" class="heat-map-layer">
        <rect
          v-for="(intensity, key) in heatMap"
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
    <g v-for="affordance in affordances" :key="`affordance-${affordance.x}-${affordance.y}`">
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
    <g v-for="agent in agents" :key="agent.id" class="agent-group">
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
import { ref, computed } from 'vue'
import { useSimulationStore } from '../stores/simulation'

const store = useSimulationStore()

const gridWidth = computed(() => store.gridWidth)
const gridHeight = computed(() => store.gridHeight)
const agents = computed(() => (store.agents || []).filter(a => a && a.id))
const affordances = computed(() => store.affordances)
const heatMap = computed(() => store.heatMap || {})

const cellSize = 75 // pixels per cell
const showHeatMap = ref(true) // Show heat map by default

function getAffordanceIcon(type) {
  const icons = {
    'Bed': '🛏️',
    'Shower': '🚿',
    'HomeMeal': '🥘',    // Home cooking - cheap, healthy
    'FastFood': '🍔',    // Fast food - expensive, convenient
    'Job': '💼',
    'Recreation': '🎮',
    'Bar': '🍺'  // Beer mug - social gathering
  }
  return icons[type] || '?'
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
  top: 10px;
  right: 10px;
  padding: 0.5rem 1rem;
  background: #3a3a4e;
  color: #a0a0b0;
  border: 1px solid #4a4a5e;
  border-radius: 4px;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  z-index: 10;
}

.heat-map-toggle:hover {
  background: #4a4a5e;
  color: #e0e0e0;
  border-color: #5a5a6e;
}

.heat-map-toggle.active {
  background: #10b981;
  color: white;
  border-color: #34d399;
}

.grid-svg {
  width: 100%;
  height: 100%;
  max-width: 600px;
  max-height: 600px;
}

.grid-cell {
  fill: #2a2a3e;
  stroke: #3a3a4e;
  stroke-width: 1;
}

.heat-map-cell {
  pointer-events: none;
  transition: opacity 0.3s ease;
}

.affordance {
  stroke-width: 2;
  transition: all 0.3s ease;
}

.affordance-bed {
  fill: #6366f1;
  stroke: #818cf8;
}

.affordance-shower {
  fill: #06b6d4;
  stroke: #22d3ee;
}

.affordance-homemeal {
  fill: #f59e0b;
  stroke: #fbbf24;
}

.affordance-fastfood {
  fill: #ef4444;
  stroke: #f87171;
}

.affordance-job {
  fill: #8b5cf6;
  stroke: #a78bfa;
}

.affordance-recreation {
  fill: #10b981;
  stroke: #34d399;
}

.affordance-bar {
  fill: #ec4899;
  stroke: #f472b6;
}

.affordance-label {
  font-size: 32px;
  pointer-events: none;
  user-select: none;
}

.agent-group {
  transition: all 0.3s ease-out;
}

.agent-circle {
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
}

.agent-label {
  fill: white;
  font-size: 14px;
  font-weight: 600;
  pointer-events: none;
  user-select: none;
}
</style>
