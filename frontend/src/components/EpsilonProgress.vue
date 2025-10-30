<template>
  <div class="epsilon-progress">
    <div class="epsilon-header">
      <div class="header-left">
        <h4>Exploration → Exploitation</h4>
        <InfoTooltip
          title="What is Epsilon (ε)?"
          text="Epsilon controls explore vs exploit tradeoff. High ε = random actions (exploration). Low ε = learned policy (exploitation). Decays over time as agent learns optimal strategies."
          position="right"
        />
      </div>
      <span class="epsilon-value">ε = {{ epsilonFormatted }}</span>
    </div>

    <div class="progress-container">
      <div class="progress-bar-wrapper">
        <div
          class="progress-bar"
          :style="{
            width: epsilonPercent + '%',
            background: epsilonColor
          }"
        >
          <div class="progress-glow"></div>
        </div>
      </div>
    </div>

    <div class="epsilon-footer">
      <span class="label-explore">Explore (Random)</span>
      <span class="label-exploit">Exploit (Learned)</span>
    </div>

    <div class="epsilon-description">
      <p v-if="epsilonValue > 0.7" class="phase exploring">
        <span class="phase-icon">🎲</span>
        <span>High exploration - agent trying random actions to discover optimal strategies</span>
      </p>
      <p v-else-if="epsilonValue > 0.3" class="phase learning">
        <span class="phase-icon">🧠</span>
        <span>Balanced phase - mixing exploration with learned behaviours</span>
      </p>
      <p v-else class="phase exploiting">
        <span class="phase-icon">🎯</span>
        <span>Low exploration - agent following learned policy, converging to optimal behaviour</span>
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import InfoTooltip from './InfoTooltip.vue'

const props = defineProps({
  epsilon: {
    type: Number,
    default: 1.0
  }
})

const epsilonValue = computed(() => props.epsilon)
const epsilonPercent = computed(() => props.epsilon * 100)
const epsilonFormatted = computed(() => props.epsilon.toFixed(3))

const epsilonColor = computed(() => {
  const eps = props.epsilon
  // High epsilon (>0.7): Blue (exploration)
  // Mid epsilon (0.3-0.7): Cyan/Teal (transition)
  // Low epsilon (<0.3): Green (exploitation)

  if (eps > 0.7) {
    return '#3b82f6' // Blue - exploration
  } else if (eps > 0.3) {
    // Gradient from blue to teal
    const t = (eps - 0.3) / 0.4
    return `rgb(${Math.round(59 + (16 - 59) * (1 - t))}, ${Math.round(130 + (185 - 130) * (1 - t))}, ${Math.round(246 + (129 - 246) * (1 - t))})`
  } else {
    // Gradient from teal to green
    const t = eps / 0.3
    return `rgb(${Math.round(16 + (34 - 16) * (1 - t))}, ${Math.round(185 + (197 - 185) * (1 - t))}, ${Math.round(129 + (94 - 129) * (1 - t))})`
  }
})
</script>

<style scoped>
.epsilon-progress {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  margin: var(--spacing-sm) 0;
}

.epsilon-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-sm);
}

.header-left {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.epsilon-header h4 {
  margin: 0;
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

.epsilon-value {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  font-family: 'Monaco', 'Courier New', monospace;
  color: var(--color-text-secondary);
  background: var(--color-bg-tertiary);
  padding: 2px 8px;
  border-radius: var(--border-radius-sm);
}

.progress-container {
  margin-bottom: var(--spacing-sm);
}

.progress-bar-wrapper {
  height: 24px;
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-full);
  overflow: hidden;
  position: relative;
}

.progress-bar {
  height: 100%;
  border-radius: var(--border-radius-full);
  transition: width var(--transition-base), background 0.5s ease;
  position: relative;
  overflow: hidden;
}

.progress-glow {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
  animation: shimmer 2s infinite;
}

@keyframes shimmer {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(100%);
  }
}

.epsilon-footer {
  display: flex;
  justify-content: space-between;
  font-size: var(--font-size-xs);
  margin-bottom: var(--spacing-md);
}

.label-explore {
  color: #3b82f6;
  font-weight: var(--font-weight-medium);
}

.label-exploit {
  color: var(--color-success);
  font-weight: var(--font-weight-medium);
}

.epsilon-description {
  margin-top: var(--spacing-md);
  padding: var(--spacing-sm);
  border-radius: var(--border-radius-sm);
  background: var(--color-bg-tertiary);
}

.epsilon-description p {
  margin: 0;
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  display: flex;
  align-items: flex-start;
  gap: var(--spacing-sm);
}

.phase-icon {
  font-size: var(--font-size-base);
  flex-shrink: 0;
}

.phase.exploring {
  border-left: 3px solid #3b82f6;
  padding-left: var(--spacing-sm);
}

.phase.learning {
  border-left: 3px solid #10b981;
  padding-left: var(--spacing-sm);
}

.phase.exploiting {
  border-left: 3px solid var(--color-success);
  padding-left: var(--spacing-sm);
}
</style>
