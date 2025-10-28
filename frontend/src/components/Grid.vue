<template>
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
</template>

<script setup>
import { computed } from 'vue'
import { useSimulationStore } from '../stores/simulation'

const store = useSimulationStore()

const gridWidth = computed(() => store.gridWidth)
const gridHeight = computed(() => store.gridHeight)
const agents = computed(() => (store.agents || []).filter(a => a && a.id))
const affordances = computed(() => store.affordances)

const cellSize = 75 // pixels per cell

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
</script>

<style scoped>
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
