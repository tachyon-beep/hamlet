<template>
  <div class="survival-trend-chart">
    <h3>Survival Time Trend (Avg per 100 Episodes)</h3>

    <!-- Empty state when no data -->
    <EmptyState
      v-if="trendData.length === 0"
      icon="📊"
      title="No Survival Data"
      message="Connect to the simulation to see survival trends."
    />

    <!-- Chart when data is available -->
    <svg v-else :width="width" :height="height" class="chart-svg" @mouseleave="hoveredIndex = null">
      <!-- Axes -->
      <line :x1="margin" :y1="height - margin" :x2="width - margin" :y2="height - margin" class="axis-line" />
      <line :x1="margin" :y1="margin" :x2="margin" :y2="height - margin" class="axis-line" />

      <!-- Trend line -->
      <polyline
        :points="trendPoints"
        fill="none"
        class="trend-line"
        stroke-width="3"
      />

      <!-- Hover areas for data points -->
      <g v-for="(point, idx) in dataPoints" :key="`hover-${idx}`">
        <circle
          :cx="point.x"
          :cy="point.y"
          r="8"
          class="hover-circle"
          @mouseenter="hoveredIndex = idx"
        />
      </g>

      <!-- Highlight circle for hovered point -->
      <g v-if="hoveredIndex !== null && dataPoints[hoveredIndex]">
        <circle
          :cx="dataPoints[hoveredIndex].x"
          :cy="dataPoints[hoveredIndex].y"
          r="6"
          class="highlight-circle"
        />
      </g>

      <!-- Axis labels -->
      <text :x="width / 2" :y="height - 5" text-anchor="middle" class="axis-label">Episodes</text>
      <text :x="10" :y="height / 2" text-anchor="middle" class="axis-label" transform="rotate(-90, 10, 100)">
        Avg Survival (steps)
      </text>
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
        <div class="tooltip-header">Episode {{ hoveredIndex * 100 }}</div>
        <div class="tooltip-row">
          <span class="tooltip-label">Avg Survival:</span>
          <span class="tooltip-value">{{ trendData[hoveredIndex].toFixed(1) }} steps</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import EmptyState from './EmptyState.vue'

export default {
  name: 'SurvivalTrendChart',
  components: {
    EmptyState
  },
  props: {
    trendData: {
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
    trendPoints() {
      if (this.trendData.length === 0) return ''

      const maxValue = Math.max(...this.trendData, 1)

      return this.trendData.map((value, idx) => {
        const x = this.margin + (idx / (this.trendData.length - 1 || 1)) * this.plotWidth
        const normalized = value / maxValue
        const y = this.height - this.margin - normalized * this.plotHeight
        return `${x},${y}`
      }).join(' ')
    },
    dataPoints() {
      // Compute x,y coordinates for each data point (for hover areas)
      if (this.trendData.length === 0) return []

      const maxValue = Math.max(...this.trendData, 1)

      return this.trendData.map((value, idx) => {
        const x = this.margin + (idx / (this.trendData.length - 1 || 1)) * this.plotWidth
        const normalized = value / maxValue
        const y = this.height - this.margin - normalized * this.plotHeight
        return { x, y }
      })
    },
  },
}
</script>

<style scoped>
.survival-trend-chart {
  position: relative;
  margin: var(--spacing-sm) 0;
  padding: var(--spacing-md);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius-md);
  background: var(--color-bg-secondary);
}

.survival-trend-chart h3 {
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

.trend-line {
  stroke: var(--color-success);
}

.axis-label {
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
  fill: var(--color-success);
  stroke: white;
  stroke-width: 2;
  transition: all var(--transition-base);
  animation: highlight-pulse 1s ease-in-out infinite;
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
  background: rgba(34, 197, 94, 0.1);
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
