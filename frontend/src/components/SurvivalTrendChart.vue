<template>
  <div class="survival-trend-chart">
    <h3>Survival Time Trend (Avg per 100 Episodes)</h3>
    <svg :width="width" :height="height" class="chart-svg">
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

      <!-- Axis labels -->
      <text :x="width / 2" :y="height - 5" text-anchor="middle" class="axis-label">Episodes</text>
      <text :x="10" :y="height / 2" text-anchor="middle" class="axis-label" transform="rotate(-90, 10, 100)">
        Avg Survival (steps)
      </text>
    </svg>
  </div>
</template>

<script>
export default {
  name: 'SurvivalTrendChart',
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
  },
}
</script>

<style scoped>
.survival-trend-chart {
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
</style>
