<template>
  <!-- Projected reward indicator bar (below time of day) -->
  <div class="projected-reward-bar" role="region" aria-label="Projected reward indicator">
    <!-- Gradient bar showing learning progress -->
    <div class="reward-bar-container">
      <!-- Small label above bar -->
      <div class="reward-label-header">
        <span>🎁</span>
        <span>PROJECTED REWARD</span>
      </div>

      <div class="reward-bar-background">
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

// Baseline position - shows where the bar will wrap (always at 100% of current tier)
const baselinePercent = computed(() => {
  // The baseline marker stays at 100% to show where the wrap happens
  // This is the target the agent is trying to reach in the current tier
  return 100
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

// Dynamic progress glow color based on tier - gradient to next tier color
const progressGlowColor = computed(() => {
  const tier = currentTier.value

  // Tier 0: Red → Yellow (struggling → learning)
  if (tier === 0) {
    return 'linear-gradient(to right, rgba(244, 67, 54, 0.9), rgba(255, 235, 59, 0.7))'
  }
  // Tier 1: Yellow → Green (learning → thriving)
  if (tier === 1) {
    return 'linear-gradient(to right, rgba(255, 235, 59, 0.9), rgba(76, 175, 80, 0.7))'
  }
  // Tier 2: Green → Blue (thriving → mastered)
  if (tier === 2) {
    return 'linear-gradient(to right, rgba(76, 175, 80, 0.9), rgba(33, 150, 243, 0.7))'
  }
  // Tier 3: Blue (mastered) - stays blue
  return 'linear-gradient(to right, rgba(33, 150, 243, 0.9), rgba(33, 150, 243, 0.7))'
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
  width: 100%;
  margin-top: 13px;
}

.reward-bar-container {
  background: var(--color-bg-secondary);
  padding: var(--spacing-md) var(--spacing-lg);
  border-radius: var(--border-radius-md);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  backdrop-filter: blur(8px);
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.reward-label-header {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-bold);
  letter-spacing: 0.05em;
  color: rgba(255, 255, 255, 0.7);
  margin-bottom: 3px;
  text-align: left;
  display: flex;
  align-items: center;
  gap: 6px;
}

.reward-bar-background {
  position: relative;
  height: 24px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  overflow: visible;
  margin-bottom: 3px;
}

.baseline-marker {
  position: absolute;
  top: -4px;
  transform: translateX(-50%);
  z-index: 2;
}

.baseline-tick {
  width: 2px;
  height: 32px;
  background: rgba(255, 255, 255, 0.6);
  border-radius: 1px;
  box-shadow: 0 0 4px rgba(255, 255, 255, 0.3);
}

.reward-progress {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  border-radius: 10px;
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
  font-size: 24px;
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
  font-size: 12px;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.7);
  letter-spacing: 0.5px;
  text-transform: uppercase;
}
</style>
