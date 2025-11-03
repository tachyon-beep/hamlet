<template>
  <!-- Time of day indicator bar (top-left) -->
  <div class="time-of-day-bar" role="region" aria-label="Time of day indicator">
    <!-- Gradient bar showing day/night cycle -->
    <div class="time-bar-container">
      <!-- Small label above bar -->
      <div class="time-label-header">TIME OF DAY</div>

      <div class="time-bar-background">
        <!-- Tick marks for major hours -->
        <div class="tick-mark" style="left: 25%"></div>  <!-- 6am -->
        <div class="tick-mark" style="left: 50%"></div>  <!-- 12pm -->
        <div class="tick-mark" style="left: 75%"></div>  <!-- 6pm -->

        <!-- Shimmer effect -->
        <div class="shimmer"></div>

        <!-- Progress indicator (no glow, clean appearance) -->
        <div
          class="time-progress"
          :style="{ width: `${progressPercent}%` }"
        >
          <!-- Marker dot at current time -->
          <div class="time-marker" :style="{ background: markerColor }"></div>
        </div>

        <!-- Bright green clock bar sliding across -->
        <div
          class="clock-bar"
          :style="{ left: `${progressPercent}%` }"
        ></div>
      </div>

      <!-- Time label with animated icon and day counter -->
      <div class="time-label">
        <span class="time-icon" aria-hidden="true">
          <transition name="icon-fade" mode="out-in">
            <span :key="timeIcon">{{ timeIcon }}</span>
          </transition>
        </span>
        <span class="time-text-main">{{ formattedTime }}</span>
        <span class="day-counter">Day {{ currentDay }}</span>
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
  },
  currentStep: {
    type: Number,
    default: 0
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

// Calculate current day (step / 24, since one step = one hour)
const currentDay = computed(() => {
  return Math.floor(props.currentStep / 24) + 1  // +1 so days start at 1, not 0
})

// No longer using dynamic glow color - removed for cleaner appearance

// Marker color matches time of day
const markerColor = computed(() => {
  const hour = props.timeOfDay
  if (hour >= 4 && hour < 7) return '#ff8c3c'  // Dawn orange
  if (hour >= 7 && hour < 17) return '#ffeb3b' // Day yellow
  if (hour >= 17 && hour < 19) return '#ff6eb4' // Dusk pink
  return '#6495ed' // Night blue
})

// Time icon based on period with sunrise/sunset
const timeIcon = computed(() => {
  const hour = props.timeOfDay

  if (hour >= 5 && hour < 7) {
    return '🌅'  // Sunrise (5-7am)
  } else if (hour >= 7 && hour < 17) {
    return '☀️'  // Daytime (7am-5pm)
  } else if (hour >= 17 && hour < 19) {
    return '🌆'  // Sunset (5-7pm)
  } else {
    return '🌙'  // Nighttime (7pm-5am)
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
  padding: var(--spacing-md) var(--spacing-lg);
  border-radius: var(--border-radius-md);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  backdrop-filter: blur(8px);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  min-width: 270px;
}

/* Small header label */
.time-label-header {
  font-size: 12px;  /* Increased from 10px - minimum readable size for streaming */
  font-weight: var(--font-weight-bold);
  color: rgba(255, 255, 255, 0.85);  /* Increased from secondary + 0.7 opacity */
  letter-spacing: 1.5px;
  text-align: center;
  font-family: 'Monaco', 'Courier New', monospace;
  margin-bottom: 4px;
}

/* Gradient bar showing day/night cycle */
.time-bar-background {
  width: 100%;
  height: 24px;  /* Increased from 18px - 33% larger for streaming visibility */
  border-radius: var(--border-radius-sm);
  background: linear-gradient(
    to right,
    /* Night (12am-4am): Deep midnight */
    #0a0a1a 0%,
    #1a1a3a 16.67%,
    /* Dawn (4am-7am): Blue to golden */
    #2d3561 20%,
    #ff6b4a 25%,
    #ffb347 29.17%,
    /* Day (7am-5pm): Bright golden to warm */
    #ffeb3b 29.17%,
    #ffd700 41.67%,
    #ffb84d 54.17%,
    #ff8c42 66.67%,
    /* Dusk (5pm-7pm): Warm to purple */
    #ff6b9d 70.83%,
    #9b59b6 75%,
    #5a3d7a 79.17%,
    /* Night (7pm-12am): Purple to deep midnight */
    #2d2d4a 83.33%,
    #1a1a3a 91.67%,
    #0a0a1a 100%
  );
  position: relative;
  overflow: hidden;
  /* Inner glow effect */
  box-shadow:
    inset 0 2px 4px rgba(0, 0, 0, 0.3),
    inset 0 -1px 2px rgba(255, 255, 255, 0.1),
    0 2px 8px rgba(0, 0, 0, 0.3);  /* Added outer shadow for depth */
}

/* Top shine effect for glassy look */
.time-bar-background::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 50%;
  background: linear-gradient(
    to bottom,
    rgba(255, 255, 255, 0.15) 0%,
    rgba(255, 255, 255, 0) 100%
  );
  border-radius: var(--border-radius-sm) var(--border-radius-sm) 0 0;
  pointer-events: none;
}

/* Shimmer animation */
.shimmer {
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(255, 255, 255, 0.2) 50%,
    transparent 100%
  );
  animation: shimmer 8s infinite;
  pointer-events: none;
}

@keyframes shimmer {
  0% {
    left: -100%;
  }
  50%, 100% {
    left: 200%;
  }
}

/* Tick marks for major hours */
.tick-mark {
  position: absolute;
  top: 0;
  width: 2px;
  height: 100%;
  background: rgba(255, 255, 255, 0.3);
  box-shadow: 0 0 4px rgba(255, 255, 255, 0.5);
  pointer-events: none;
  z-index: 1;
}

/* Bright green clock bar sliding across */
.clock-bar {
  position: absolute;
  top: 0;
  width: 4px;  /* Increased from 3px - more prominent for streaming */
  height: 100%;
  background: #00ff00;
  box-shadow:
    0 0 12px #00ff00,
    0 0 24px #00ff00,
    0 0 36px rgba(0, 255, 0, 0.8);  /* Stronger glow */
  pointer-events: none;
  z-index: 3;
  transform: translateX(-50%);
  transition: left 0.3s ease-out;
  animation: pulse-clock 2s ease-in-out infinite;
}

@keyframes pulse-clock {
  0%, 100% {
    opacity: 1;
    box-shadow:
      0 0 12px #00ff00,
      0 0 24px #00ff00,
      0 0 36px rgba(0, 255, 0, 0.8);
  }
  50% {
    opacity: 0.85;
    box-shadow:
      0 0 16px #00ff00,
      0 0 32px #00ff00,
      0 0 48px rgba(0, 255, 0, 1.0);
  }
}

/* Progress indicator (clean, no glow) */
.time-progress {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  background: rgba(255, 255, 255, 0.15);
  transition: width 0.3s ease-out;
}

/* Time marker dot */
.time-marker {
  position: absolute;
  right: -6px;
  top: 50%;
  transform: translateY(-50%);
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #fff;
  box-shadow:
    0 0 8px currentColor,
    0 0 16px rgba(255, 255, 255, 0.8),
    0 2px 4px rgba(0, 0, 0, 0.3);
  animation: pulse-marker 2s ease-in-out infinite;
}

@keyframes pulse-marker {
  0%, 100% {
    transform: translateY(-50%) scale(1);
  }
  50% {
    transform: translateY(-50%) scale(1.15);
  }
}

/* Time label with icon and day counter */
.time-label {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--spacing-sm);
}

.time-icon {
  font-size: 28px;  /* Increased from 20px - prominent visual anchor for streaming */
  line-height: 1;
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.5));
  flex-shrink: 0;
}

/* Icon fade transition */
.icon-fade-enter-active,
.icon-fade-leave-active {
  transition: all 0.5s ease;
}

.icon-fade-enter-from {
  opacity: 0;
  transform: scale(0.8) rotate(-10deg);
}

.icon-fade-leave-to {
  opacity: 0;
  transform: scale(0.8) rotate(10deg);
}

/* Time text - same size as emoji */
.time-text-main {
  font-size: 24px;  /* Increased from 20px - primary information for streaming */
  font-weight: var(--font-weight-bold);
  color: var(--color-text-primary);
  font-family: 'Monaco', 'Courier New', monospace;
  line-height: 1;
  text-shadow:
    0 2px 4px rgba(0, 0, 0, 0.5),
    0 0 8px rgba(255, 255, 255, 0.3);  /* Dual shadow for pop against gradient */
  flex-grow: 1;
  text-align: center;
}

/* Day counter */
.day-counter {
  font-size: var(--font-size-lg);  /* 18px - increased from 16px */
  font-weight: var(--font-weight-bold);
  color: var(--color-text-primary);  /* Changed from secondary - more prominent */
  font-family: 'Monaco', 'Courier New', monospace;
  line-height: 1;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
  opacity: 1;  /* Full opacity */
  flex-shrink: 0;
  background: rgba(0, 0, 0, 0.3);  /* Background pill for contrast */
  padding: 4px 8px;
  border-radius: 6px;
}

/* Responsive adjustments */
@media (max-width: 767px) {
  .time-of-day-bar {
    top: var(--spacing-xs);
    left: var(--spacing-xs);
  }

  .time-bar-container {
    padding: var(--spacing-sm) var(--spacing-md);
    min-width: 210px;
  }

  .time-bar-background {
    height: 15px;
  }

  .time-icon {
    font-size: var(--font-size-lg);
  }

  .time-text-main {
    font-size: var(--font-size-lg);
  }

  .day-counter {
    font-size: var(--font-size-sm);
  }

  .time-marker {
    width: 10px;
    height: 10px;
    right: -5px;
  }
}
</style>
