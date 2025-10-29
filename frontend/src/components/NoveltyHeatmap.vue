<template>
  <div class="novelty-heatmap">
    <svg :width="width" :height="height">
      <g v-for="(row, rowIdx) in noveltyGrid" :key="rowIdx">
        <rect
          v-for="(novelty, colIdx) in row"
          :key="colIdx"
          :x="colIdx * cellSize"
          :y="rowIdx * cellSize"
          :width="cellSize"
          :height="cellSize"
          :fill="getNoveltyColor(novelty)"
          :opacity="0.6"
        />
      </g>
    </svg>
  </div>
</template>

<script>
export default {
  name: 'NoveltyHeatmap',
  props: {
    noveltyMap: {
      type: Array,
      required: true,
      default: () => Array(8).fill(null).map(() => Array(8).fill(0)),
    },
    gridSize: {
      type: Number,
      default: 8,
    },
    cellSize: {
      type: Number,
      default: 75,
    },
  },
  computed: {
    width() {
      return this.gridSize * this.cellSize
    },
    height() {
      return this.gridSize * this.cellSize
    },
    noveltyGrid() {
      return this.noveltyMap
    },
  },
  methods: {
    getNoveltyColor(novelty) {
      // Map novelty (0-high) to color gradient: blue (familiar) -> yellow -> red (novel)
      // Normalize novelty to [0, 1]
      const maxNovelty = Math.max(...this.noveltyMap.flat())
      const normalized = maxNovelty > 0 ? novelty / maxNovelty : 0

      if (normalized < 0.5) {
        // Blue to yellow
        const t = normalized * 2
        const r = Math.round(t * 255)
        const g = Math.round(t * 255)
        const b = Math.round((1 - t) * 255)
        return `rgb(${r}, ${g}, ${b})`
      } else {
        // Yellow to red
        const t = (normalized - 0.5) * 2
        const r = 255
        const g = Math.round((1 - t) * 255)
        const b = 0
        return `rgb(${r}, ${g}, ${b})`
      }
    },
  },
}
</script>

<style scoped>
.novelty-heatmap {
  position: absolute;
  top: 0;
  left: 0;
  pointer-events: none;
}
</style>
