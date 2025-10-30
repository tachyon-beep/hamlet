<template>
  <!-- ✅ Semantic HTML: section instead of div -->
  <section class="stats-panel" aria-labelledby="stats-heading">
    <h3 id="stats-heading">Episode Info</h3>

    <div class="stats-grid">
      <div class="stat-item">
        <span class="stat-label">Steps</span>
        <!-- ✅ ARIA live region for real-time updates -->
        <span
          class="stat-value compact-value"
          aria-live="polite"
          aria-atomic="true"
          role="status"
        >
          {{ props.currentStep }}
        </span>
      </div>

      <div class="stat-item">
        <span class="stat-label">Episode</span>
        <!-- ✅ ARIA live region for real-time updates -->
        <span
          class="stat-value compact-value"
          aria-live="polite"
          aria-atomic="true"
          role="status"
        >
          {{ displayEpisode }}
        </span>
      </div>

      <div v-if="props.checkpointEpisode > 0" class="stat-item checkpoint-ref">
        <span class="stat-label">Checkpoint</span>
        <span class="stat-value compact-value">
          {{ props.checkpointEpisode }}
        </span>
      </div>
    </div>

    <div v-if="props.episodeHistory.length > 0" class="history">
      <h4>Performance</h4>

      <!-- ✅ Chart with accessibility: role="img" and aria-label -->
      <div
        class="history-chart"
        role="img"
        :aria-label="`Episode performance chart showing survival time over last ${chartData.length} episodes, ranging from ${minSurvival} to ${maxSurvival} steps`"
      >
        <svg viewBox="0 0 300 100" class="chart-svg" aria-hidden="true">
          <!-- Grid lines -->
          <line
            v-for="i in 5"
            :key="`grid-${i}`"
            :x1="0"
            :y1="i * 20"
            :x2="300"
            :y2="i * 20"
            stroke="var(--color-chart-grid)"
            stroke-width="1"
          />

          <!-- Line chart -->
          <polyline
            :points="chartPoints"
            fill="none"
            stroke="var(--color-chart-primary)"
            stroke-width="2"
          />

          <!-- Data points -->
          <circle
            v-for="(point, i) in chartData"
            :key="`point-${i}`"
            :cx="point.x"
            :cy="point.y"
            r="3"
            fill="var(--color-chart-primary)"
          />
        </svg>

        <!-- ✅ Screen reader alternative text -->
        <div class="sr-only">
          Chart data: Last {{ chartData.length }} episodes with survival times ranging from
          {{ minSurvival }} to {{ maxSurvival }} steps.
          Average survival: {{ averageSurvivalTime }} steps.
        </div>
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
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { ACTION_ICONS } from '../utils/constants'

// ✅ Props First: Receive data from parent instead of importing store
const props = defineProps({
  currentEpisode: {
    type: Number,
    default: 0
  },
  currentStep: {
    type: Number,
    default: 0
  },
  cumulativeReward: {
    type: Number,
    default: 0
  },
  lastAction: {
    type: [Number, String],  // Accept both Number (from server) and String (legacy)
    default: null
  },
  episodeHistory: {
    type: Array,
    default: () => []
  },
  checkpointEpisode: {
    type: Number,
    default: 0
  }
})

// Computed for derived values

// ✅ Display episode starting at 1 (backend sends 0-indexed)
const displayEpisode = computed(() => {
  return props.currentEpisode + 1
})

const averageSurvivalTime = computed(() => {
  if (props.episodeHistory.length === 0) return 0
  const sum = props.episodeHistory.reduce((acc, ep) => acc + ep.steps, 0)
  return Math.round(sum / props.episodeHistory.length)
})

// ✅ Extract toFixed() from template to computed property
const formattedReward = computed(() => {
  return props.cumulativeReward.toFixed(1)
})

const lastEpisodeSteps = computed(() => {
  if (props.episodeHistory.length === 0) return 0
  return props.episodeHistory[props.episodeHistory.length - 1].steps
})

// ✅ Min/max survival for accessibility aria-label
const minSurvival = computed(() => {
  if (props.episodeHistory.length === 0) return 0
  const history = props.episodeHistory.slice(-10)
  return Math.min(...history.map(ep => ep.steps))
})

const maxSurvival = computed(() => {
  if (props.episodeHistory.length === 0) return 0
  const history = props.episodeHistory.slice(-10)
  return Math.max(...history.map(ep => ep.steps))
})

// Chart data processing
const chartData = computed(() => {
  if (props.episodeHistory.length === 0) return []

  const history = props.episodeHistory.slice(-10) // Last 10 episodes
  const maxSteps = Math.max(...history.map(ep => ep.steps), 1)

  return history.map((ep, i) => ({
    x: (i / Math.max(history.length - 1, 1)) * 300,
    y: 100 - (ep.steps / maxSteps) * 100
  }))
})

const chartPoints = computed(() => {
  return chartData.value.map(p => `${p.x},${p.y}`).join(' ')
})

// ✅ Use imported constant from utils
function formatAction(action) {
  if (!action) return '—'
  return ACTION_ICONS[action] || action
}
</script>

<style scoped>
/* ✅ Refactored to use design tokens */
.stats-panel {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-lg);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
}

.stats-panel h3 {
  margin: 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--spacing-md);
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.stat-label {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.stat-value {
  font-size: var(--font-size-xl);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

.stat-value.positive {
  color: var(--color-success);
}

.checkpoint-ref {
  opacity: 0.8;
}

.stat-value.compact-value {
  font-size: var(--font-size-lg);
}

.stat-value.negative {
  color: var(--color-error);
}

.stat-value.action {
  font-size: var(--font-size-base);
}

.history h4 {
  margin: 0 0 var(--spacing-sm) 0;
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.history-chart {
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
  padding: var(--spacing-md);
  margin-bottom: var(--spacing-md);
}

.chart-svg {
  width: 100%;
  height: auto;
}

.history-stats {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.history-stat {
  display: flex;
  justify-content: space-between;
  font-size: var(--font-size-sm);
}

.history-stat .label {
  color: var(--color-text-secondary);
}

.history-stat .value {
  color: var(--color-text-primary);
  font-weight: var(--font-weight-semibold);
}

/* ✅ Screen reader only utility class */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}
</style>
