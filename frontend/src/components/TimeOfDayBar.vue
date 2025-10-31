<template>
  <!-- Time of day indicator bar (top-left) -->
  <div class="time-of-day-bar" role="region" aria-label="Time of day indicator">
    <!-- Gradient bar showing day/night cycle -->
    <div class="time-bar-container">
      <div class="time-bar-background">
        <!-- Progress indicator showing current time -->
        <div
          class="time-progress"
          :style="{ width: `${progressPercent}%` }"
        ></div>
      </div>

      <!-- Time label -->
      <div class="time-label">
        <span class="time-icon" aria-hidden="true">{{ timeIcon }}</span>
        <span class="time-text">{{ formattedTime }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  timeOfDay: {
    type: Number,
    default: 0,
    validator: (value) => value >= 0 && value < 24
  }
})

// Progress as percentage (0-100%)
const progressPercent = computed(() => {
  return (props.timeOfDay / 24) * 100
})

// Format time as 12-hour with AM/PM
const formattedTime = computed(() => {
  const hour = props.timeOfDay
  if (hour === 0) return '12 AM'
  if (hour === 12) return '12 PM'
  if (hour < 12) return `${hour} AM`
  return `${hour - 12} PM`
})

// Time icon based on period
const timeIcon = computed(() => {
  const hour = props.timeOfDay
  if (hour >= 6 && hour < 18) {
    return '☀️'  // Daytime (6am-6pm)
  } else {
    return '🌙'  // Nighttime (6pm-6am)
  }
})
</script>

<style scoped>
/* Time of day indicator (top-left corner) */
.time-of-day-bar {
  position: absolute;
  top: var(--spacing-md);
  left: var(--spacing-md);
  z-index: 100;
}

.time-bar-container {
  background: var(--color-bg-secondary);
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--border-radius-md);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  backdrop-filter: blur(8px);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  min-width: 180px;
}

/* Gradient bar showing day/night cycle */
.time-bar-background {
  width: 100%;
  height: 12px;
  border-radius: var(--border-radius-sm);
  background: linear-gradient(
    to right,
    /* Midnight to 6am: black to yellow (night ending) */
    #000000 0%,
    #1a1a00 12.5%,
    #ffeb3b 25%,
    /* 6am to 6pm: yellow to black (day to dusk) */
    #ffeb3b 25%,
    #ffd700 37.5%,
    #ff6b35 50%,
    #8b4513 62.5%,
    #000000 75%,
    /* 6pm to midnight: black to yellow start (night) */
    #000000 75%,
    #0a0a00 87.5%,
    #000000 100%
  );
  position: relative;
  overflow: hidden;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
}

/* Progress indicator (fills from left) */
.time-progress {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  background: rgba(255, 255, 255, 0.3);
  backdrop-filter: brightness(1.2);
  border-right: 2px solid rgba(255, 255, 255, 0.8);
  transition: width 0.3s ease-out;
  box-shadow: 2px 0 6px rgba(255, 255, 255, 0.4);
}

/* Time label with icon */
.time-label {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
  font-family: 'Monaco', 'Courier New', monospace;
}

.time-icon {
  font-size: var(--font-size-lg);
  line-height: 1;
}

.time-text {
  flex: 1;
  text-align: left;
}

/* Responsive adjustments */
@media (max-width: 767px) {
  .time-of-day-bar {
    top: var(--spacing-xs);
    left: var(--spacing-xs);
  }

  .time-bar-container {
    padding: var(--spacing-xs) var(--spacing-sm);
    min-width: 140px;
  }

  .time-bar-background {
    height: 10px;
  }

  .time-label {
    font-size: var(--font-size-xs);
  }

  .time-icon {
    font-size: var(--font-size-base);
  }
}
</style>
