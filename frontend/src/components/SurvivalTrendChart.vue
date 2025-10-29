<template>
  <div class="survival-trend-chart">
    <h3>Survival Time Trend (Avg per 100 Episodes)</h3>
    <svg :width="width" :height="height">
      <!-- Axes -->
      <line :x1="margin" :y1="height - margin" :x2="width - margin" :y2="height - margin" stroke="#ccc" />
      <line :x1="margin" :y1="margin" :x2="margin" :y2="height - margin" stroke="#ccc" />

      <!-- Trend line -->
      <polyline
        :points="trendPoints"
        fill="none"
        stroke="#10b981"
        stroke-width="3"
      />

      <!-- Axis labels -->
      <text :x="width / 2" :y="height - 5" text-anchor="middle" font-size="12">Episodes</text>
      <text :x="10" :y="height / 2" text-anchor="middle" font-size="12" transform="rotate(-90, 10, 100)">
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
  margin: 10px 0;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
}
</style>
