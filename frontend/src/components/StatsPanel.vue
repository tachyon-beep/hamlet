<template>
  <div class="stats-panel">
    <h3>Episode Info</h3>

    <div class="stats-grid">
      <div class="stat-item">
        <span class="stat-label">Episode</span>
        <span class="stat-value">#{{ currentEpisode }}</span>
      </div>

      <div class="stat-item">
        <span class="stat-label">Steps</span>
        <span class="stat-value">{{ currentStep }}</span>
      </div>

      <div class="stat-item">
        <span class="stat-label">Reward</span>
        <span class="stat-value" :class="{ positive: cumulativeReward > 0, negative: cumulativeReward < 0 }">
          {{ cumulativeReward.toFixed(1) }}
        </span>
      </div>

      <div class="stat-item">
        <span class="stat-label">Action</span>
        <span class="stat-value action">{{ formatAction(lastAction) }}</span>
      </div>
    </div>

    <div v-if="episodeHistory.length > 0" class="history">
      <h4>Performance</h4>

      <div class="history-chart">
        <svg viewBox="0 0 300 100" class="chart-svg">
          <!-- Grid lines -->
          <line
            v-for="i in 5"
            :key="`grid-${i}`"
            :x1="0"
            :y1="i * 20"
            :x2="300"
            :y2="i * 20"
            stroke="#3a3a4e"
            stroke-width="1"
          />

          <!-- Line chart -->
          <polyline
            :points="chartPoints"
            fill="none"
            stroke="#3b82f6"
            stroke-width="2"
          />

          <!-- Data points -->
          <circle
            v-for="(point, i) in chartData"
            :key="`point-${i}`"
            :cx="point.x"
            :cy="point.y"
            r="3"
            fill="#3b82f6"
          />
        </svg>
      </div>

      <div class="history-stats">
        <div class="history-stat">
          <span class="label">Avg Survival:</span>
          <span class="value">{{ averageSurvivalTime }} steps</span>
        </div>
        <div class="history-stat">
          <span class="label">Last Episode:</span>
          <span class="value">{{ lastEpisodeSteps }} steps</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useSimulationStore } from '../stores/simulation'

const store = useSimulationStore()

const currentEpisode = computed(() => store.currentEpisode)
const currentStep = computed(() => store.currentStep)
const cumulativeReward = computed(() => store.cumulativeReward)
const lastAction = computed(() => store.lastAction)
const episodeHistory = computed(() => store.episodeHistory)
const averageSurvivalTime = computed(() => store.averageSurvivalTime)

const lastEpisodeSteps = computed(() => {
  if (episodeHistory.value.length === 0) return 0
  return episodeHistory.value[episodeHistory.value.length - 1].steps
})

// Chart data processing
const chartData = computed(() => {
  if (episodeHistory.value.length === 0) return []

  const history = episodeHistory.value.slice(-10) // Last 10 episodes
  const maxSteps = Math.max(...history.map(ep => ep.steps), 1)

  return history.map((ep, i) => ({
    x: (i / Math.max(history.length - 1, 1)) * 300,
    y: 100 - (ep.steps / maxSteps) * 100
  }))
})

const chartPoints = computed(() => {
  return chartData.value.map(p => `${p.x},${p.y}`).join(' ')
})

function formatAction(action) {
  if (!action) return '—'

  const actionIcons = {
    'up': '↑ Up',
    'down': '↓ Down',
    'left': '← Left',
    'right': '→ Right',
    'interact': '⚡ Interact'
  }

  return actionIcons[action] || action
}
</script>

<style scoped>
.stats-panel {
  background: #2a2a3e;
  border-radius: 8px;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.stats-panel h3 {
  margin: 0;
  font-size: 1.125rem;
  font-weight: 600;
  color: #e0e0e0;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.stat-label {
  font-size: 0.75rem;
  color: #a0a0b0;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.stat-value {
  font-size: 1.25rem;
  font-weight: 600;
  color: #e0e0e0;
}

.stat-value.positive {
  color: #10b981;
}

.stat-value.negative {
  color: #ef4444;
}

.stat-value.action {
  font-size: 1rem;
}

.history h4 {
  margin: 0 0 0.75rem 0;
  font-size: 0.875rem;
  font-weight: 600;
  color: #e0e0e0;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.history-chart {
  background: #1e1e2e;
  border-radius: 6px;
  padding: 1rem;
  margin-bottom: 1rem;
}

.chart-svg {
  width: 100%;
  height: auto;
}

.history-stats {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.history-stat {
  display: flex;
  justify-content: space-between;
  font-size: 0.875rem;
}

.history-stat .label {
  color: #a0a0b0;
}

.history-stat .value {
  color: #e0e0e0;
  font-weight: 600;
}
</style>
