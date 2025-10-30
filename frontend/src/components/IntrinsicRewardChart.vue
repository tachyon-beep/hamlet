<template>
  <div class="intrinsic-reward-chart">
    <h3>Reward Streams (Last 100 Steps)</h3>

    <!-- Empty state when no data -->
    <EmptyState
      v-if="extrinsicHistory.length === 0"
      icon="📈"
      title="No Reward Data"
      message="Start the simulation to see reward streams over time."
    />

    <!-- Chart when data is available -->
    <svg v-else :width="width" :height="height" class="chart-svg" @mouseleave="hoveredIndex = null">
      <!-- Axes -->
      <line :x1="margin" :y1="height - margin" :x2="width - margin" :y2="height - margin" class="axis-line" />
      <line :x1="margin" :y1="margin" :x2="margin" :y2="height - margin" class="axis-line" />

      <!-- Extrinsic line (blue) -->
      <polyline
        :points="extrinsicPoints"
        fill="none"
        class="line-extrinsic"
        stroke-width="2"
      />

      <!-- Intrinsic line (orange) -->
      <polyline
        :points="intrinsicPoints"
        fill="none"
        class="line-intrinsic"
        stroke-width="2"
      />

      <!-- Hover areas for data points -->
      <g v-for="(point, idx) in dataPoints" :key="`hover-${idx}`">
        <circle
          :cx="point.x"
          :cy="point.extrinsicY"
          r="6"
          class="hover-circle"
          @mouseenter="hoveredIndex = idx"
        />
        <circle
          :cx="point.x"
          :cy="point.intrinsicY"
          r="6"
          class="hover-circle"
          @mouseenter="hoveredIndex = idx"
        />
      </g>

      <!-- Highlight circles for hovered point -->
      <g v-if="hoveredIndex !== null && dataPoints[hoveredIndex]">
        <circle
          :cx="dataPoints[hoveredIndex].x"
          :cy="dataPoints[hoveredIndex].extrinsicY"
          r="5"
          class="highlight-circle extrinsic"
        />
        <circle
          :cx="dataPoints[hoveredIndex].x"
          :cy="dataPoints[hoveredIndex].intrinsicY"
          r="5"
          class="highlight-circle intrinsic"
        />
      </g>

      <!-- Legend -->
      <g transform="translate(50, 20)">
        <line x1="0" y1="0" x2="30" y2="0" class="line-extrinsic" stroke-width="2" />
        <text x="35" y="5" class="legend-text">Extrinsic</text>

        <line x1="0" y1="20" x2="30" y2="20" class="line-intrinsic" stroke-width="2" />
        <text x="35" y="25" class="legend-text">Intrinsic</text>
      </g>
    </svg>

    <!-- Tooltip -->
    <div
      v-if="hoveredIndex !== null && dataPoints[hoveredIndex]"
      class="chart-tooltip"
      :style="{
        left: dataPoints[hoveredIndex].x + 'px',
        top: '0px'
      }"
    >
      <div class="tooltip-content">
        <div class="tooltip-header">Step {{ hoveredIndex }}</div>
        <div class="tooltip-row extrinsic">
          <span class="tooltip-label">Extrinsic:</span>
          <span class="tooltip-value">{{ extrinsicHistory[hoveredIndex].toFixed(2) }}</span>
        </div>
        <div class="tooltip-row intrinsic">
          <span class="tooltip-label">Intrinsic:</span>
          <span class="tooltip-value">{{ intrinsicHistory[hoveredIndex].toFixed(2) }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import EmptyState from './EmptyState.vue'

export default {
  name: 'IntrinsicRewardChart',
  components: {
    EmptyState
  },
  props: {
    extrinsicHistory: {
      type: Array,
      default: () => [],
    },
    intrinsicHistory: {
      type: Array,
      default: () => [],
    },
    width: {
      type: Number,
      default: 600,
    },
    height: {
      type: Number,
      default: 200,
    },
  },
  data() {
    return {
      margin: 40,
      hoveredIndex: null,
    }
  },
  computed: {
    plotWidth() {
      return this.width - 2 * this.margin
    },
    plotHeight() {
      return this.height - 2 * this.margin
    },
    extrinsicPoints() {
      return this.getPoints(this.extrinsicHistory)
    },
    intrinsicPoints() {
      return this.getPoints(this.intrinsicHistory)
    },
    dataPoints() {
      // Compute x,y coordinates for each data point (for hover areas)
      if (this.extrinsicHistory.length === 0) return []

      const allData = [...this.extrinsicHistory, ...this.intrinsicHistory]
      const maxValue = Math.max(...allData.map(Math.abs), 1)
      const minValue = -maxValue

      return this.extrinsicHistory.map((extrinsicValue, idx) => {
        const intrinsicValue = this.intrinsicHistory[idx] || 0
        const x = this.margin + (idx / (this.extrinsicHistory.length - 1 || 1)) * this.plotWidth

        const extrinsicNorm = (extrinsicValue - minValue) / (maxValue - minValue || 1)
        const extrinsicY = this.height - this.margin - extrinsicNorm * this.plotHeight

        const intrinsicNorm = (intrinsicValue - minValue) / (maxValue - minValue || 1)
        const intrinsicY = this.height - this.margin - intrinsicNorm * this.plotHeight

        return { x, extrinsicY, intrinsicY }
      })
    },
  },
  methods: {
    getPoints(data) {
      if (data.length === 0) return ''

      const maxValue = Math.max(...data.map(Math.abs), 1)
      const minValue = -maxValue

      return data.map((value, idx) => {
        const x = this.margin + (idx / (data.length - 1 || 1)) * this.plotWidth
        const normalized = (value - minValue) / (maxValue - minValue || 1)
        const y = this.height - this.margin - normalized * this.plotHeight
        return `${x},${y}`
      }).join(' ')
    },
  },
}
</script>

<style scoped>
.intrinsic-reward-chart {
  position: relative;
  margin: var(--spacing-sm) 0;
  padding: var(--spacing-md);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius-md);
  background: var(--color-bg-secondary);
}

.intrinsic-reward-chart h3 {
  margin: 0 0 var(--spacing-sm) 0;
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-secondary);
}

.chart-svg {
  display: block;
  width: 100%;
}

.axis-line {
  stroke: var(--color-chart-grid);
  stroke-width: 1;
}

.line-extrinsic {
  stroke: var(--color-chart-primary);
}

.line-intrinsic {
  stroke: var(--color-warning);
}

.legend-text {
  fill: var(--color-text-secondary);
  font-size: var(--font-size-xs);
}

/* Hover interaction elements */
.hover-circle {
  fill: transparent;
  cursor: pointer;
  transition: fill var(--transition-base);
}

.hover-circle:hover {
  fill: rgba(255, 255, 255, 0.1);
}

.highlight-circle {
  stroke-width: 2;
  transition: all var(--transition-base);
  animation: highlight-pulse 1s ease-in-out infinite;
}

.highlight-circle.extrinsic {
  fill: var(--color-chart-primary);
  stroke: white;
}

.highlight-circle.intrinsic {
  fill: var(--color-warning);
  stroke: white;
}

@keyframes highlight-pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
}

/* Tooltip styling */
.chart-tooltip {
  position: absolute;
  pointer-events: none;
  z-index: var(--z-index-tooltip);
  transform: translateX(-50%);
}

.tooltip-content {
  background: var(--color-bg-primary);
  border: 2px solid var(--color-border);
  border-radius: var(--border-radius-sm);
  padding: var(--spacing-sm);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  min-width: 150px;
}

.tooltip-header {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
  margin-bottom: var(--spacing-xs);
  text-align: center;
  border-bottom: 1px solid var(--color-border);
  padding-bottom: var(--spacing-xs);
}

.tooltip-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: var(--font-size-xs);
  margin: var(--spacing-xs) 0;
  padding: 2px var(--spacing-xs);
  border-radius: var(--border-radius-sm);
}

.tooltip-row.extrinsic {
  background: rgba(59, 130, 246, 0.1);
}

.tooltip-row.intrinsic {
  background: rgba(234, 179, 8, 0.1);
}

.tooltip-label {
  color: var(--color-text-secondary);
  font-weight: var(--font-weight-medium);
}

.tooltip-value {
  color: var(--color-text-primary);
  font-weight: var(--font-weight-semibold);
  font-family: 'Monaco', 'Courier New', monospace;
}
</style>
