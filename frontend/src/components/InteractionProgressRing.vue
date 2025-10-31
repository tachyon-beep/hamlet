<template>
  <!-- Progress ring around agent during multi-tick interactions -->
  <svg
    v-if="isActive"
    :width="ringSize"
    :height="ringSize"
    :style="{
      position: 'absolute',
      left: `${x * cellSize - ringSize / 2 + cellSize / 2}px`,
      top: `${y * cellSize - ringSize / 2 + cellSize / 2}px`,
      pointerEvents: 'none',
      zIndex: 50,
    }"
    class="progress-ring"
  >
    <!-- Background circle (subtle) -->
    <circle
      :cx="ringSize / 2"
      :cy="ringSize / 2"
      :r="radius"
      fill="none"
      stroke="rgba(255, 255, 255, 0.2)"
      :stroke-width="strokeWidth"
    />

    <!-- Progress arc (colored and glowing) -->
    <circle
      :cx="ringSize / 2"
      :cy="ringSize / 2"
      :r="radius"
      fill="none"
      :stroke="progressColor"
      :stroke-width="strokeWidth"
      :stroke-dasharray="circumference"
      :stroke-dashoffset="dashOffset"
      stroke-linecap="round"
      class="progress-arc"
      :style="{
        filter: `drop-shadow(0 0 ${glowIntensity}px ${progressColor})`,
      }"
    />

    <!-- Center pulse effect -->
    <circle
      :cx="ringSize / 2"
      :cy="ringSize / 2"
      :r="radius * 0.4"
      :fill="progressColor"
      opacity="0.2"
      class="center-pulse"
    />

    <!-- Progress percentage text (optional, for large rings) -->
    <text
      v-if="ringSize >= 60"
      :x="ringSize / 2"
      :y="ringSize / 2"
      text-anchor="middle"
      dominant-baseline="middle"
      :font-size="ringSize * 0.2"
      :fill="progressColor"
      font-weight="bold"
      font-family="Monaco, monospace"
      class="progress-text"
    >
      {{ Math.round(progressPercent) }}%
    </text>
  </svg>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  // Agent position
  x: {
    type: Number,
    required: true
  },
  y: {
    type: Number,
    required: true
  },
  // Progress (0-1 normalized)
  progress: {
    type: Number,
    required: true,
    validator: (value) => value >= 0 && value <= 1
  },
  // Current affordance type (for color coding)
  affordanceType: {
    type: String,
    default: null
  },
  // Grid cell size
  cellSize: {
    type: Number,
    default: 75
  }
})

// Only show when there's active progress
const isActive = computed(() => props.progress > 0)

// Ring dimensions
const ringSize = computed(() => props.cellSize * 1.4)
const strokeWidth = computed(() => props.cellSize * 0.08)
const radius = computed(() => (ringSize.value - strokeWidth.value) / 2)
const circumference = computed(() => 2 * Math.PI * radius.value)

// Progress as percentage
const progressPercent = computed(() => props.progress * 100)

// Dash offset for arc (starts at top, goes clockwise)
const dashOffset = computed(() => {
  return circumference.value * (1 - props.progress)
})

// Glow intensity increases with progress
const glowIntensity = computed(() => {
  return 4 + props.progress * 8
})

// Color based on affordance type
const progressColor = computed(() => {
  const type = props.affordanceType?.toLowerCase()

  // Affordance color mapping
  const colorMap = {
    bed: '#9c27b0',           // Purple (rest/sleep)
    shower: '#2196f3',        // Blue (water)
    fridge: '#4caf50',        // Green (food)
    job: '#ff9800',           // Orange (work)
    gym: '#f44336',           // Red (exercise)
    bar: '#e91e63',           // Pink (social/fun)
    coffeeshop: '#795548',    // Brown (coffee)
    clinic: '#00bcd4',        // Cyan (health)
    park: '#8bc34a',          // Light green (nature)
    library: '#673ab7',       // Deep purple (learning)
    restaurant: '#ff5722',    // Deep orange (dining)
    mall: '#9e9e9e',          // Grey (shopping)
    therapist: '#3f51b5',     // Indigo (mental health)
    home: '#607d8b',          // Blue grey (comfort)
  }

  return colorMap[type] || '#ffeb3b' // Default to yellow
})
</script>

<style scoped>
.progress-ring {
  animation: fade-in 0.3s ease-out;
}

@keyframes fade-in {
  from {
    opacity: 0;
    transform: scale(0.8);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

/* Progress arc animation */
.progress-arc {
  transform-origin: center;
  transform: rotate(-90deg); /* Start from top */
  transition: stroke-dashoffset 0.3s ease-out;
  animation: pulse-ring 2s ease-in-out infinite;
}

@keyframes pulse-ring {
  0%, 100% {
    opacity: 0.9;
  }
  50% {
    opacity: 1;
  }
}

/* Center pulse effect */
.center-pulse {
  animation: pulse-center 2s ease-in-out infinite;
}

@keyframes pulse-center {
  0%, 100% {
    opacity: 0.1;
    transform: scale(1);
  }
  50% {
    opacity: 0.3;
    transform: scale(1.2);
  }
}

/* Progress text */
.progress-text {
  animation: fade-pulse 2s ease-in-out infinite;
  text-shadow: 0 0 4px rgba(0, 0, 0, 0.5);
}

@keyframes fade-pulse {
  0%, 100% {
    opacity: 0.8;
  }
  50% {
    opacity: 1;
  }
}
</style>
