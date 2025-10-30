<template>
  <div class="affordance-graph">
    <h4>Learned Routines (Affordance Transitions)</h4>
    <svg :width="width" :height="height" v-if="hasData" class="graph-svg">
      <!-- Nodes (affordances) -->
      <g v-for="(node, name) in nodes" :key="name">
        <circle
          :cx="node.x"
          :cy="node.y"
          :r="30"
          :fill="getNodeColor(name)"
          class="node-circle"
          stroke-width="2"
        />
        <text
          :x="node.x"
          :y="node.y + 5"
          text-anchor="middle"
          class="node-label"
        >
          {{ name }}
        </text>
      </g>

      <!-- Edges (transitions) -->
      <g v-for="(edge, idx) in edges" :key="idx">
        <line
          :x1="edge.x1"
          :y1="edge.y1"
          :x2="edge.x2"
          :y2="edge.y2"
          :stroke="edge.color"
          :stroke-width="edge.width"
          stroke-opacity="0.6"
          marker-end="url(#arrowhead)"
        />
      </g>

      <!-- Arrow marker definition -->
      <defs>
        <marker
          id="arrowhead"
          markerWidth="10"
          markerHeight="10"
          refX="8"
          refY="3"
          orient="auto"
        >
          <polygon points="0 0, 10 3, 0 6" class="arrow-head" />
        </marker>
      </defs>
    </svg>
    <div v-else class="no-data">
      No transition data yet...
    </div>
  </div>
</template>

<script>
export default {
  name: 'AffordanceGraph',
  props: {
    transitionData: {
      type: Object,
      default: () => ({}),
    },
    width: {
      type: Number,
      default: 400,
    },
    height: {
      type: Number,
      default: 300,
    },
  },
  computed: {
    hasData() {
      return Object.keys(this.transitionData).length > 0
    },
    nodes() {
      // Position nodes in a circle
      const names = ['Bed', 'Job', 'Shower', 'Fridge']
      const centerX = this.width / 2
      const centerY = this.height / 2
      const radius = Math.min(this.width, this.height) / 3

      const nodes = {}
      names.forEach((name, idx) => {
        const angle = (idx / names.length) * 2 * Math.PI - Math.PI / 2
        nodes[name] = {
          x: centerX + radius * Math.cos(angle),
          y: centerY + radius * Math.sin(angle),
        }
      })

      return nodes
    },
    edges() {
      const edges = []

      // Build edges from transition data
      for (const [from, toDict] of Object.entries(this.transitionData)) {
        for (const [to, count] of Object.entries(toDict)) {
          if (from === to) continue // Skip self-loops

          const fromNode = this.nodes[from]
          const toNode = this.nodes[to]

          if (!fromNode || !toNode) continue

          // Edge thickness based on count (log scale)
          const width = Math.log(count + 1) * 2 + 1

          // Edge color based on recency (for now, just use count)
          const color = count > 10 ? '#10b981' : '#3b82f6'

          edges.push({
            x1: fromNode.x,
            y1: fromNode.y,
            x2: toNode.x,
            y2: toNode.y,
            width,
            color,
          })
        }
      }

      return edges
    },
  },
  methods: {
    getNodeColor(name) {
      // Use CSS custom properties for colors (matches MeterPanel colors)
      const colors = {
        'Bed': 'var(--color-meter-energy)',        // Purple for energy
        'Job': 'var(--color-meter-money)',          // Gold for money
        'Shower': 'var(--color-meter-hygiene)',     // Blue for hygiene
        'Fridge': 'var(--color-meter-satiation)',   // Green for satiation
      }
      // Get computed style to resolve CSS variables
      const style = getComputedStyle(document.documentElement)
      const colorVar = colors[name] || 'var(--color-text-muted)'
      // Extract variable name from var() syntax
      const match = colorVar.match(/var\((--[\w-]+)\)/)
      if (match) {
        return style.getPropertyValue(match[1]).trim()
      }
      return colorVar
    },
  },
}
</script>

<style scoped>
.affordance-graph {
  margin: var(--spacing-sm) 0;
  padding: var(--spacing-md);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius-md);
  background: var(--color-bg-secondary);
}

.affordance-graph h4 {
  margin: 0 0 var(--spacing-sm) 0;
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-secondary);
}

.graph-svg {
  display: block;
  width: 100%;
}

.node-circle {
  stroke: var(--color-text-primary);
}

.node-label {
  fill: white;
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-semibold);
}

.arrow-head {
  fill: var(--color-text-muted);
}

.no-data {
  text-align: center;
  padding: var(--spacing-2xl);
  color: var(--color-text-tertiary);
  font-style: italic;
  font-size: var(--font-size-sm);
}
</style>
