<template>
  <div class="agent-behavior-panel">
    <div class="panel-header">
      <h3>Agent Behaviour</h3>
    </div>

    <!-- SECTION 1: NOW (Episode Performance) -->
    <section class="section performance-section">
      <h4 class="section-header">
        <span class="section-icon">⚡</span>
        NOW
      </h4>
      <div class="immediate-info">
        <div class="reward-display">
          <span class="label">Reward</span>
          <span class="value" :class="rewardClass">{{ formattedReward }}</span>
        </div>
        <div class="action-display">
          <span class="action-icon">{{ actionIcon }}</span>
          <span class="action-name">{{ actionName }}</span>
        </div>
      </div>
    </section>

    <!-- SECTION 2: MODE (Exploration → Exploitation) -->
    <section class="section exploration-section">
      <h4 class="section-header">
        <span class="section-icon">🎲</span>
        MODE
      </h4>
      <div class="context-info">
        <div class="epsilon-display">
          <span class="epsilon-value">ε = {{ formattedEpsilon }}</span>
          <span class="epsilon-label">{{ epsilonLabel }}</span>
        </div>
        <div class="epsilon-bar">
          <div class="bar-track">
            <div class="bar-fill" :style="{ width: `${epsilonPercent}%` }"></div>
          </div>
          <div class="bar-labels">
            <span>🎯 Exploit</span>
            <span>🎲 Explore</span>
          </div>
        </div>
      </div>
      <p class="explanation">{{ epsilonExplanation }}</p>
    </section>

    <!-- SECTION 3: TRAINING (Learning Progress) -->
    <section class="section learning-section">
      <h4 class="section-header">
        <span class="section-icon">📈</span>
        TRAINING
      </h4>
      <div class="context-info">
        <div class="progress-display">
          <div class="progress-bar">
            <div class="bar-track">
              <div class="bar-fill" :style="{ width: `${progressPercent}%`, background: progressGradient }">
                <div class="bar-shimmer"></div>
              </div>
            </div>
          </div>
          <div class="progress-labels">
            <span class="label-start">Early Training</span>
            <span class="label-mid">Learning</span>
            <span class="label-end">Converged</span>
          </div>
          <div class="progress-status">
            <span class="status-icon">{{ progressIcon }}</span>
            <span class="status-text">{{ progressStatus }}</span>
          </div>
        </div>
      </div>
      <p class="explanation">{{ progressExplanation }}</p>
    </section>

    <!-- Permanent explanation text -->
    <div class="panel-explanation">
      <p>Shows what the agent is doing right now, its learning mode, and training progress. Watch how behaviour evolves as the agent learns!</p>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  // Episode performance
  cumulativeReward: {
    type: Number,
    default: 0
  },
  lastAction: {
    type: Number,
    default: null
  },

  // Exploration mode
  epsilon: {
    type: Number,
    default: 1.0
  },

  // Training progress
  checkpointEpisode: {
    type: Number,
    default: 0
  },
  totalEpisodes: {
    type: Number,
    default: 0
  }
})

// Action mapping
const actionMap = {
  0: { icon: '⬆️', name: 'Move Up' },
  1: { icon: '⬇️', name: 'Move Down' },
  2: { icon: '⬅️', name: 'Move Left' },
  3: { icon: '➡️', name: 'Move Right' },
  4: { icon: '⚡', name: 'Interact' }
}

const actionIcon = computed(() => {
  if (props.lastAction === null) return '⏸️'
  return actionMap[props.lastAction]?.icon || '❓'
})

const actionName = computed(() => {
  if (props.lastAction === null) return 'Waiting'
  return actionMap[props.lastAction]?.name || 'Unknown'
})

// Reward formatting
const formattedReward = computed(() => {
  const val = props.cumulativeReward
  if (val >= 0) return `+${val.toFixed(1)}`
  return val.toFixed(1)
})

const rewardClass = computed(() => {
  if (props.cumulativeReward > 0) return 'positive'
  if (props.cumulativeReward < 0) return 'negative'
  return 'neutral'
})

// Epsilon formatting
const formattedEpsilon = computed(() => props.epsilon.toFixed(3))

const epsilonPercent = computed(() => props.epsilon * 100)

const epsilonLabel = computed(() => {
  const e = props.epsilon
  if (e > 0.7) return 'High Exploration'
  if (e > 0.3) return 'Balanced Mix'
  return 'Low Exploration'
})

const epsilonExplanation = computed(() => {
  const e = props.epsilon
  if (e > 0.7) return '🎲 High exploration - agent discovering new strategies'
  if (e > 0.3) return '🎲 Moderate exploration - agent balancing discovery with learned policy'
  return '🎯 Low exploration - agent following learned policy, converging to optimal behaviour'
})

// Training progress
const progressPercent = computed(() => {
  if (props.totalEpisodes === 0) return 0
  return Math.min((props.checkpointEpisode / props.totalEpisodes) * 100, 100)
})

const progressGradient = computed(() => {
  const percent = progressPercent.value
  if (percent < 33) {
    return 'linear-gradient(90deg, #3b82f6, #06b6d4)'
  } else if (percent < 66) {
    return 'linear-gradient(90deg, #06b6d4, #10b981)'
  } else {
    return 'linear-gradient(90deg, #10b981, #22c55e)'
  }
})

const progressIcon = computed(() => {
  const percent = progressPercent.value
  if (percent < 20) return '🌱'
  if (percent < 50) return '📈'
  if (percent < 80) return '🎯'
  return '✨'
})

const progressStatus = computed(() => {
  const percent = progressPercent.value
  if (percent < 20) return 'Early exploration phase'
  if (percent < 50) return 'Discovering strategies'
  if (percent < 80) return 'Refining policy'
  return 'Near convergence'
})

const progressExplanation = computed(() => {
  const percent = progressPercent.value
  if (percent < 20) return '🌱 Early exploration - random actions, building experience'
  if (percent < 50) return '📈 Discovering strategies - learning which affordances help survival'
  if (percent < 80) return '🎯 Refining policy - optimising action sequences'
  return '✨ Near convergence - consistent survival strategy established'
})
</script>

<style scoped>
.agent-behavior-panel {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
  height: 100%;
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-sm);
}

.panel-header h3 {
  margin: 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

/* Sections */
.section {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  padding-bottom: var(--spacing-md);
}

.section:not(:last-child) {
  border-bottom: 1px solid var(--color-bg-tertiary);
}

.section-header {
  margin: 0 0 var(--spacing-sm) 0;
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-bold);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-text-secondary);
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
}

.section-icon {
  font-size: var(--font-size-base);
}

/* Performance section accent */
.performance-section .section-header {
  color: var(--color-success);
}

.exploration-section .section-header {
  color: #f59e0b; /* Amber */
}

.learning-section .section-header {
  color: #3b82f6; /* Blue */
}

/* TIER 1: Immediate Info (Performance) */
.immediate-info {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
  border-left: 4px solid var(--color-success);
}

.reward-display {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}

.reward-display .label {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
  font-weight: var(--font-weight-semibold);
}

.reward-display .value {
  font-size: 2rem;
  font-weight: var(--font-weight-bold);
  font-family: 'Monaco', 'Courier New', monospace;
  font-variant-numeric: tabular-nums;
}

.reward-display .value.positive { color: var(--color-success); }
.reward-display .value.negative { color: var(--color-error); }
.reward-display .value.neutral { color: var(--color-text-secondary); }

.action-display {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding-top: var(--spacing-sm);
  border-top: 1px solid var(--color-bg-tertiary);
}

.action-icon {
  font-size: 1.5rem;
}

.action-name {
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

/* TIER 2: Context Info (Exploration, Training) */
.context-info {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.epsilon-display {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}

.epsilon-value {
  font-size: var(--font-size-xl);
  font-weight: var(--font-weight-bold);
  color: var(--color-text-primary);
  font-family: 'Monaco', 'Courier New', monospace;
}

.epsilon-label {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
  font-weight: var(--font-weight-semibold);
}

/* Epsilon bar */
.epsilon-bar {
  margin-top: var(--spacing-xs);
}

.epsilon-bar .bar-track {
  height: 24px;
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-full);
  overflow: hidden;
  position: relative;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
}

.epsilon-bar .bar-fill {
  height: 100%;
  background: linear-gradient(to right, #3b82f6, #f59e0b);
  transition: width 0.3s ease;
  position: relative;
  overflow: hidden;
}

.bar-labels {
  display: flex;
  justify-content: space-between;
  margin-top: var(--spacing-xs);
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
}

/* Progress bar */
.progress-display {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.progress-bar {
  margin-bottom: var(--spacing-xs);
}

.progress-bar .bar-track {
  height: 24px;
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-full);
  overflow: hidden;
  position: relative;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
}

.progress-bar .bar-fill {
  height: 100%;
  border-radius: var(--border-radius-full);
  transition: width 0.5s ease, background 0.5s ease;
  position: relative;
  overflow: hidden;
}

.bar-shimmer {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
  animation: shimmer 2s infinite;
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

.progress-labels {
  display: flex;
  justify-content: space-between;
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
}

.label-start { color: #3b82f6; }
.label-mid { color: #06b6d4; }
.label-end { color: var(--color-success); }

.progress-status {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-xs) var(--spacing-sm);
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
}

.status-icon {
  font-size: 1rem;
}

.status-text {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  font-style: italic;
}

/* TIER 3: Explanations */
.explanation {
  margin: var(--spacing-xs) 0 0 0;
  padding: var(--spacing-sm);
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
  font-size: var(--font-size-sm);
  color: var(--color-text-tertiary);
  font-style: italic;
  line-height: 1.4;
  opacity: 0.9;
}

/* Panel explanation */
.panel-explanation {
  margin-top: auto;
  padding: var(--spacing-sm);
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
  border-top: 1px solid var(--color-border);
}

.panel-explanation p {
  margin: 0;
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
  font-style: italic;
  line-height: 1.4;
  opacity: 0.9;
}
</style>
