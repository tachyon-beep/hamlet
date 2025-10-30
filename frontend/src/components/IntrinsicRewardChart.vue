<template>
  <div class="intrinsic-reward-chart">
    <h3>Reward Streams (Last 100 Steps)</h3>
    <svg :width="width" :height="height" class="chart-svg">
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

      <!-- Legend -->
      <g transform="translate(50, 20)">
        <line x1="0" y1="0" x2="30" y2="0" class="line-extrinsic" stroke-width="2" />
        <text x="35" y="5" class="legend-text">Extrinsic</text>

        <line x1="0" y1="20" x2="30" y2="20" class="line-intrinsic" stroke-width="2" />
        <text x="35" y="25" class="legend-text">Intrinsic</text>
      </g>
    </svg>
  </div>
</template>

<script>
export default {
  name: 'IntrinsicRewardChart',
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
</style>
