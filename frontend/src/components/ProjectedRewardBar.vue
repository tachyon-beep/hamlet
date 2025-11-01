<template>
  <!-- Projected reward indicator bar (below time of day) -->
  <div class="projected-reward-bar" role="region" aria-label="Projected reward indicator">
    <!-- Gradient bar showing learning progress -->
    <div class="reward-bar-container">
      <!-- Small label above bar -->
      <div class="reward-label-header">PROJECTED REWARD</div>

      <div class="reward-bar-background">
        <!-- Zero line marker (baseline) -->
        <div class="baseline-marker" :style="{ left: `${baselinePercent}%` }">
          <div class="baseline-tick"></div>
        </div>

        <!-- Progress indicator with glow -->
        <div
          class="reward-progress"
          :style="{
            width: `${progressPercent}%`,
            background: progressGlowColor
          }"
        >
          <!-- Marker dot at current progress -->
          <div class="reward-marker" :style="{ background: markerColor }"></div>
        </div>
      </div>

      <!-- Status label with icon -->
      <div class="reward-label">
        <span class="reward-icon" aria-hidden="true">
          <transition name="icon-fade" mode="out-in">
            <span :key="statusIcon">{{ statusIcon }}</span>
          </transition>
        </span>
        <div class="reward-text-container">
          <span class="reward-text-main" :class="rewardStatusClass">{{ formattedReward }}</span>
          <span class="reward-text-sub">{{ statusText }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  projectedReward: {
    type: Number,
    default: 0
  },
  currentStep: {
    type: Number,
    default: 0
  },
  baselineSurvival: {
    type: Number,
    default: 100
  }
})

// Calculate tier (0-4) based on how many times we've exceeded baseline
const currentTier = computed(() => {
  if (props.baselineSurvival === 0) return 0
  const tier = Math.floor(props.currentStep / props.baselineSurvival)
  return Math.min(tier, 3)  // Cap at tier 3 (4 total tiers: 0, 1, 2, 3)
})

// Calculate progress percentage within current tier (wraps at 100%)
const progressPercent = computed(() => {
  if (props.baselineSurvival === 0) return 0
  const progressInTier = props.currentStep % props.baselineSurvival
  return (progressInTier / props.baselineSurvival) * 100
})

// Baseline position (always at 100%)
const baselinePercent = computed(() => {
  return 100  // Baseline is always at 100% mark (where it wraps)
})

// Format reward with sign
const formattedReward = computed(() => {
  const reward = props.projectedReward
  if (reward > 0) return `+${reward.toFixed(1)}`
  return reward.toFixed(1)
})

// Status text based on tier
const statusText = computed(() => {
  const tier = currentTier.value
  if (tier === 0) return 'Struggling'
  if (tier === 1) return 'Learning'
  if (tier === 2) return 'Thriving'
  return 'Mastered'  // tier 3
})

// Status icon based on tier
const statusIcon = computed(() => {
  const tier = currentTier.value
  if (tier === 0) return '🔴'
  if (tier === 1) return '🟡'
  if (tier === 2) return '�'
  return '�'  // tier 3
})

// CSS class for reward text color based on tier
const rewardStatusClass = computed(() => {
  const tier = currentTier.value
  if (tier === 0) return 'reward-tier-0'
  if (tier === 1) return 'reward-tier-1'
  if (tier === 2) return 'reward-tier-2'
  return 'reward-tier-3'
})

// Dynamic progress glow color based on tier
const progressGlowColor = computed(() => {
  const tier = currentTier.value

  // Tier 0: Red (struggling)
  if (tier === 0) {
    return 'radial-gradient(ellipse at right, rgba(244, 67, 54, 0.7), rgba(200, 50, 50, 0.4))'
  }
  // Tier 1: Yellow (learning)
  if (tier === 1) {
    return 'radial-gradient(ellipse at right, rgba(255, 235, 59, 0.7), rgba(255, 200, 50, 0.4))'
  }
  // Tier 2: Green (thriving)
  if (tier === 2) {
    return 'radial-gradient(ellipse at right, rgba(76, 175, 80, 0.7), rgba(50, 150, 70, 0.4))'
  }
  // Tier 3: Blue (mastered)
  return 'radial-gradient(ellipse at right, rgba(33, 150, 243, 0.7), rgba(20, 100, 200, 0.4))'
})

// Marker color matches tier
const markerColor = computed(() => {
  const tier = currentTier.value
  if (tier === 0) return '#f44336'  // Red
  if (tier === 1) return '#ffeb3b'  // Yellow
  if (tier === 2) return '#4caf50'  // Green
  return '#2196f3'  // Blue
})
</script>

<style scoped>
.projected-reward-bar {
  position: absolute;
  top: 130px;  /* Position below TimeOfDayBar with spacing */
  left: var(--spacing-md);  /* Match TimeOfDayBar left alignment */
  min-width: 270px;  /* Match TimeOfDayBar width */
  z-index: 100;
  pointer-events: none;
}

.reward-bar-container {
  background: var(--color-bg-secondary);
  padding: var(--spacing-md) var(--spacing-lg);
  border-radius: var(--border-radius-md);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  backdrop-filter: blur(8px);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.reward-label-header {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1.5px;
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 8px;
  text-align: left;
}

.reward-bar-background {
  position: relative;
  height: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  overflow: visible;
  margin-bottom: 10px;
}

.baseline-marker {
  position: absolute;
  top: -4px;
  transform: translateX(-50%);
  z-index: 2;
}

.baseline-tick {
  width: 2px;
  height: 16px;
  background: rgba(255, 255, 255, 0.6);
  border-radius: 1px;
  box-shadow: 0 0 4px rgba(255, 255, 255, 0.3);
}

.reward-progress {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s ease, background 0.5s ease;
}

.reward-marker {
  position: absolute;
  right: -6px;
  top: 50%;
  transform: translateY(-50%);
  width: 12px;
  height: 12px;
  border-radius: 50%;
  border: 2px solid rgba(0, 0, 0, 0.3);
  box-shadow: 0 0 8px currentColor, 0 2px 4px rgba(0, 0, 0, 0.3);
  transition: background 0.5s ease;
}

.reward-label {
  display: flex;
  align-items: center;
  gap: 8px;
}

.reward-icon {
  font-size: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 24px;
}

.icon-fade-enter-active,
.icon-fade-leave-active {
  transition: opacity 0.3s ease;
}

.icon-fade-enter-from,
.icon-fade-leave-to {
  opacity: 0;
}

.reward-text-container {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex: 1;
}

.reward-text-main {
  font-size: 20px;
  font-weight: 700;
  letter-spacing: 0.5px;
  transition: color 0.3s ease;
}

.reward-tier-0 {
  color: #f44336;  /* Red - Struggling */
  text-shadow: 0 0 8px rgba(244, 67, 54, 0.5);
}

.reward-tier-1 {
  color: #ffeb3b;  /* Yellow - Learning */
  text-shadow: 0 0 8px rgba(255, 235, 59, 0.5);
}

.reward-tier-2 {
  color: #4caf50;  /* Green - Thriving */
  text-shadow: 0 0 8px rgba(76, 175, 80, 0.5);
}

.reward-tier-3 {
  color: #2196f3;  /* Blue - Mastered */
  text-shadow: 0 0 8px rgba(33, 150, 243, 0.5);
}

.reward-text-sub {
  font-size: 11px;
  font-weight: 500;
  color: rgba(255, 255, 255, 0.6);
  letter-spacing: 0.5px;
}
</style>
