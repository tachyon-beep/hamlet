<template>
  <div class="intrinsic-reward-chart">
    <h3>Reward Streams (Last 100 Steps)</h3>
    <svg :width="width" :height="height">
      <!-- Axes -->
      <line :x1="margin" :y1="height - margin" :x2="width - margin" :y2="height - margin" stroke="#ccc" />
      <line :x1="margin" :y1="margin" :x2="margin" :y2="height - margin" stroke="#ccc" />

      <!-- Extrinsic line (blue) -->
      <polyline
        :points="extrinsicPoints"
        fill="none"
        stroke="#3b82f6"
        stroke-width="2"
      />

      <!-- Intrinsic line (orange) -->
      <polyline
        :points="intrinsicPoints"
        fill="none"
        stroke="#f59e0b"
        stroke-width="2"
      />

      <!-- Legend -->
      <g transform="translate(50, 20)">
        <line x1="0" y1="0" x2="30" y2="0" stroke="#3b82f6" stroke-width="2" />
        <text x="35" y="5" font-size="12">Extrinsic</text>

        <line x1="0" y1="20" x2="30" y2="20" stroke="#f59e0b" stroke-width="2" />
        <text x="35" y="25" font-size="12">Intrinsic</text>
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
  margin: 10px 0;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
}
</style>
