<template>
  <div class="agent-behavior-panel">
    <div class="panel-header">
      <h3>Agent Behaviour</h3>
    </div>

    <!-- SECTION 1: ACTION (Current Action) -->
    <section class="section performance-section">
      <h4 class="section-header">
        <span class="section-icon">⚡</span>
        ACTION
      </h4>
      <div class="immediate-info">
        <div class="action-display">
          <div class="action-current">
            <span class="action-icon">{{ actionIcon }}</span>
            <span class="action-name">{{ actionName }}</span>
          </div>
          <div v-if="recentActions.length > 0" class="action-trail">
            <span
              v-for="(action, idx) in recentActions"
              :key="idx"
              class="trail-icon"
              :style="{ opacity: 1 - (idx * 0.2) }"
            >
              {{ action.icon }}
            </span>
          </div>
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
        </div>
        <div class="epsilon-bar">
          <div class="bar-track">
            <div class="bar-fill" :style="{ width: `${epsilonPercent}%` }"></div>
          </div>
          <div class="bar-labels">
            <span>🎲 Explore</span>
            <span>🎯 Exploit</span>
          </div>
        </div>
      </div>
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
        </div>
      </div>
    </section>

    <!-- SECTION 4: Q-VALUES (What agent thinks) -->
    <section class="section q-values-section">
      <h4 class="section-header">
        <span class="section-icon">🧠</span>
        Q-VALUES
      </h4>
      <div class="q-values-list">
        <div
          v-for="(prob, index) in actionProbabilities"
          :key="index"
          class="q-value-item"
          :class="{ 
            'best-action': index === bestActionIndex && actionMasks?.[index],
            'action-masked': !actionMasks?.[index]
          }"
        >
          <div class="q-value-label">
            <span class="action-icon-small">{{ actionMap[index]?.icon }}</span>
            <span class="action-name-small">{{ actionMap[index]?.name }}</span>
          </div>
          <div class="q-value-bar-container">
            <div class="q-value-bar" :style="{ width: `${prob}%` }"></div>
          </div>
          <span class="q-value-percent">{{ prob.toFixed(1) }}%</span>
        </div>
      </div>
    </section>

    <!-- SECTION 5: ACTION CONFIDENCE -->
    <section class="section confidence-section">
      <h4 class="section-header">
        <span class="section-icon">🎯</span>
        CONFIDENCE
      </h4>
      <div class="confidence-display">
        <div class="confidence-bar-container">
          <div class="confidence-bar" :style="{ width: `${mockConfidence}%` }"></div>
        </div>
        <div class="confidence-label">
          <span class="confidence-value">{{ mockConfidence }}%</span>
          <span class="confidence-text">{{ confidenceText }}</span>
        </div>
      </div>
    </section>

    <!-- SECTION 6: TOP 3 FAVOURITES -->
    <section v-if="topFavorites.length > 0" class="section affordances-section">
      <h4 class="section-header">
        <span class="section-icon">⭐</span>
        FAVOURITES
      </h4>
      <div class="favorites-list">
        <div v-for="(favorite, index) in topFavorites" :key="favorite.name" class="favorite-item">
          <span class="favorite-rank">{{ index + 1 }}</span>
          <span class="affordance-icon">{{ favorite.icon }}</span>
          <div class="favorite-info">
            <span class="affordance-name">{{ favorite.name }}</span>
            <span class="affordance-count">{{ favorite.count }} uses</span>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'

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
  currentStep: {
    type: Number,
    default: 0
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
  },

  // Q-values and affordance stats (from backend)
  qValues: {
    type: Array,
    default: () => []
  },
  actionMasks: {
    type: Array,
    default: () => [true, true, true, true, true, true]  // All 6 actions valid by default
  },
  affordanceStats: {
    type: Array,
    default: () => []
  }
})

// Action mapping
const actionMap = {
  0: { icon: '⬆️', name: 'Move Up' },
  1: { icon: '⬇️', name: 'Move Down' },
  2: { icon: '⬅️', name: 'Move Left' },
  3: { icon: '➡️', name: 'Move Right' },
  4: { icon: '⚡', name: 'Interact' },
  5: { icon: '⏸️', name: 'Wait' }
}

// Affordance icon mapping
const affordanceIconMap = {
  'Bed': '🛏️',
  'Shower': '🚿',
  'Fridge': '🍔',
  'Job': '💼'
}

// Action history trail (last 5 actions)
const recentActions = ref([])
const lastProcessedStep = ref(0)

// Watch for step changes to capture each action (including repeated ones)
watch(() => props.currentStep, (newStep, oldStep) => {
  // Clear trail when episode restarts
  if (newStep === 0 || (newStep < oldStep)) {
    recentActions.value = []
    lastProcessedStep.value = 0
    return
  }

  // Add action only if this is a new step we haven't processed yet
  if (newStep > lastProcessedStep.value && newStep > 0 && props.lastAction !== null) {
    recentActions.value.unshift({
      icon: actionMap[props.lastAction]?.icon || '❓',
      timestamp: Date.now()
    })

    // Keep last 5 actions
    if (recentActions.value.length > 5) {
      recentActions.value.pop()
    }

    lastProcessedStep.value = newStep
  }
})

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

const rewardIcon = computed(() => {
  if (props.cumulativeReward > 0) return '↗'
  if (props.cumulativeReward < 0) return '↘'
  return '→'
})

// Epsilon formatting
const formattedEpsilon = computed(() => {
  console.log('Epsilon value:', props.epsilon)
  return props.epsilon.toFixed(3)
})

// Bar shows exploitation progress (1 - ε): starts at 0 when ε=1, trends to 100 when ε→0
const epsilonPercent = computed(() => (1 - props.epsilon) * 100)

// Training progress
const progressPercent = computed(() => {
  console.log('Training progress:', {
    checkpointEpisode: props.checkpointEpisode,
    totalEpisodes: props.totalEpisodes,
    percent: (props.checkpointEpisode / props.totalEpisodes) * 100
  })

  if (props.totalEpisodes === 0) return 0
  // Prevent showing progress > 100% if checkpoint > total (data error)
  if (props.checkpointEpisode > props.totalEpisodes) return 100
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

// Q-Values (from backend via props)
const displayQValues = computed(() => {
  // Use real Q-values if available, otherwise return zeros
  if (props.qValues && props.qValues.length >= 5) {
    // If backend sends 5 values (old), pad with 0 for Wait action
    if (props.qValues.length === 5) {
      return [...props.qValues, 0]
    }
    return props.qValues
  }
  return [0, 0, 0, 0, 0, 0]  // 6 actions: UP, DOWN, LEFT, RIGHT, INTERACT, WAIT
})

// Action probabilities - show relative strength of each Q-value
const actionProbabilities = computed(() => {
  const qvals = displayQValues.value
  const masks = props.actionMasks
  
  // Safety check for empty or invalid data
  if (!qvals || qvals.length === 0 || !masks || masks.length === 0) {
    return [0, 0, 0, 0, 0, 0]
  }
  
  // Filter to only valid (unmasked) actions for normalization
  const validQValues = qvals.filter((q, i) => masks[i])
  
  if (validQValues.length === 0) {
    return qvals.map(() => 0)
  }
  
  // Find max and min Q-values among valid actions only
  const maxQ = Math.max(...validQValues)
  const minQ = Math.min(...validQValues)
  
  // If all Q-values are the same, return uniform distribution for valid actions
  if (maxQ === minQ) {
    return qvals.map((q, i) => masks[i] ? 100 / validQValues.length : 0)
  }
  
  // Normalize to 0-100 scale based on min-max range
  // Masked actions always get 0%
  const range = maxQ - minQ
  return qvals.map((q, i) => masks[i] ? ((q - minQ) / range) * 100 : 0)
})

const bestActionIndex = computed(() => {
  const probs = actionProbabilities.value
  if (probs.length === 0) return -1
  return probs.indexOf(Math.max(...probs))
})

// Confidence (combines Q-value spread and epsilon)
// High Q-value difference + low epsilon = confident
// Similar Q-values or high epsilon = uncertain
const mockConfidence = computed(() => {
  const probs = actionProbabilities.value
  if (probs.length === 0) return 0
  
  // Filter out masked actions (they have 0% probability)
  const validProbs = probs.filter(p => p > 0)
  
  if (validProbs.length === 0) return 0
  
  // Sort to get top 2 probabilities among valid actions
  const sortedProbs = [...validProbs].sort((a, b) => b - a)
  const bestProb = sortedProbs[0]
  const secondBestProb = sortedProbs[1] || 0
  
  // Calculate separation between best and second-best (0-100)
  const separation = bestProb - secondBestProb
  
  // Factor in epsilon (high epsilon = low confidence due to exploration)
  // Confidence = separation * (1 - epsilon)
  // At epsilon=1.0 (full exploration), confidence = 0%
  // At epsilon=0.0 (full exploitation), confidence = separation
  const exploitationFactor = 1 - props.epsilon
  const confidence = separation * exploitationFactor
  
  console.log('Confidence calc:', { 
    validCount: validProbs.length,
    bestProb, 
    secondBestProb, 
    separation, 
    epsilon: props.epsilon, 
    exploitationFactor, 
    confidence 
  })
  
  return Math.round(confidence)
})

const confidenceText = computed(() => {
  const conf = mockConfidence.value
  if (conf > 80) return 'Very confident'
  if (conf > 60) return 'Confident'
  if (conf > 40) return 'Moderate'
  if (conf > 20) return 'Uncertain'
  return 'Very uncertain'
})

// Favorites (from backend via props)
const mockFavorites = computed(() => {
  if (!props.affordanceStats || props.affordanceStats.length === 0) {
    // Return empty array if no data yet
    return []
  }

  // Map backend data to display format with icons
  return props.affordanceStats.map(stat => ({
    icon: affordanceIconMap[stat.name] || '❓',
    name: stat.name,
    count: stat.count
  }))
})

// Top 3 favorites - the most-used affordances
const topFavorites = computed(() => {
  const favorites = mockFavorites.value
  return favorites.slice(0, 3)  // Backend already sorts by count descending
})

// Keep topFavorite for backward compatibility (shows section if any favorites exist)
const topFavorite = computed(() => {
  return topFavorites.value.length > 0 ? topFavorites.value[0] : null
})
</script>

<style scoped>
.agent-behavior-panel {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-lg);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
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
  gap: var(--spacing-md);
  padding: var(--spacing-lg);
  background: linear-gradient(135deg,
    rgba(16, 185, 129, 0.05),
    rgba(16, 185, 129, 0.02)
  );
  border-radius: var(--border-radius-md);
  border-left: 4px solid var(--color-success);
  position: relative;
  overflow: hidden;
}

/* Animated background pulse for NOW section */
.immediate-info::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: radial-gradient(circle at top left,
    rgba(16, 185, 129, 0.1),
    transparent 70%
  );
  animation: pulse-bg 3s ease-in-out infinite;
  pointer-events: none;
}

@keyframes pulse-bg {
  0%, 100% { opacity: 0.5; }
  50% { opacity: 0.8; }
}

/* Ensure content stays above background */
.immediate-info > * {
  position: relative;
  z-index: 1;
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

.value-container {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
}

.reward-icon {
  font-size: 1.5rem;
  font-weight: var(--font-weight-bold);
  transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.reward-icon.positive {
  color: var(--color-success);
  animation: bounce-up 0.6s ease;
}

.reward-icon.negative {
  color: var(--color-error);
  animation: bounce-down 0.6s ease;
}

@keyframes bounce-up {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-4px); }
}

@keyframes bounce-down {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(4px); }
}

.reward-display .value {
  font-size: 2rem;
  font-weight: var(--font-weight-bold);
  font-family: 'Monaco', 'Courier New', monospace;
  font-variant-numeric: tabular-nums;
  min-width: 5ch;
  text-align: right;
}

.reward-display .value.positive { color: var(--color-success); }
.reward-display .value.negative { color: var(--color-error); }
.reward-display .value.neutral { color: var(--color-text-secondary); }

.action-display {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.action-current {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.action-icon {
  font-size: 1.75rem;
  animation: icon-bounce 0.3s ease;
}

@keyframes icon-bounce {
  0% { transform: scale(0.8) rotate(-10deg); }
  50% { transform: scale(1.1) rotate(5deg); }
  100% { transform: scale(1) rotate(0deg); }
}

.action-name {
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

.action-trail {
  display: flex;
  gap: var(--spacing-xs);
  padding-left: var(--spacing-sm);
  height: 20px;
  overflow: hidden;
}

.trail-icon {
  font-size: 0.875rem;
  transition: all 0.3s ease;
  filter: grayscale(0.5);
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
  font-variant-numeric: tabular-nums;
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
  will-change: transform;
  transform: translateZ(0);
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

/* TIER 3: Status Box (Combined label + description) */
.status-box {
  margin: var(--spacing-sm) 0 0 0;
  padding: var(--spacing-sm);
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.status-label {
  font-size: var(--font-size-sm);
  color: var(--color-text-primary);
  font-weight: var(--font-weight-medium);
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
}

.status-description {
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

/* Q-VALUES SECTION */
.q-values-section .section-header {
  color: #a78bfa; /* Purple */
}

.q-values-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.q-value-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-xs) var(--spacing-sm);
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
  border-left: 2px solid var(--color-bg-tertiary);
  transition: all var(--transition-base);
}

.q-value-item.best-action {
  border-left-color: var(--color-success);
  background: rgba(16, 185, 129, 0.05);
}

.q-value-label {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  min-width: 90px;
}

.action-icon-small {
  font-size: 1rem;
  flex-shrink: 0;
}

.action-name-small {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
}

.q-value-bar-container {
  flex: 1;
  height: 14px;
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-full);
  overflow: hidden;
  box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2);
}

.q-value-bar {
  height: 100%;
  background: linear-gradient(90deg, #a78bfa, #c4b5fd);
  transition: width 0.3s ease;
  border-radius: var(--border-radius-full);
}

.q-value-item.best-action .q-value-bar {
  background: linear-gradient(90deg, #10b981, #34d399);
}

.q-value-percent {
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-bold);
  font-family: 'Monaco', 'Courier New', monospace;
  font-variant-numeric: tabular-nums;
  color: var(--color-text-primary);
  min-width: 40px;
  text-align: right;
}

.q-value-item.best-action .q-value-percent {
  color: var(--color-success);
}

.q-value-item.action-masked {
  opacity: 0.3;
  border-left-color: var(--color-bg-tertiary);
  background: var(--color-bg-primary);
}

.q-value-item.action-masked .action-name-small {
  text-decoration: line-through;
  color: var(--color-text-tertiary);
}

.q-value-item.action-masked .q-value-bar {
  background: rgba(128, 128, 128, 0.3);
}

.q-value-item.action-masked .q-value-percent {
  color: var(--color-text-tertiary);
}

/* CONFIDENCE SECTION */
.confidence-section .section-header {
  color: #f59e0b; /* Amber */
}

.confidence-display {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.confidence-bar-container {
  height: 20px;
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-full);
  overflow: hidden;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
}

.confidence-bar {
  height: 100%;
  background: linear-gradient(90deg, #f59e0b, #22c55e);
  transition: width 0.3s ease;
  border-radius: var(--border-radius-full);
}

.confidence-label {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}

.confidence-value {
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-bold);
  color: var(--color-text-primary);
  font-family: 'Monaco', 'Courier New', monospace;
  font-variant-numeric: tabular-nums;
}

.confidence-text {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
  font-style: italic;
}

/* AFFORDANCES SECTION */
.affordances-section .section-header {
  color: #ec4899; /* Pink */
}

.favorites-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.favorite-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: linear-gradient(135deg,
    rgba(236, 72, 153, 0.1),
    rgba(236, 72, 153, 0.05)
  );
  border-radius: var(--border-radius-md);
  border-left: 3px solid #ec4899;
}

.favorite-rank {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-bold);
  color: #ec4899;
  min-width: 1.5rem;
  text-align: center;
}

.affordance-icon {
  font-size: 1.5rem;
  flex-shrink: 0;
}

.favorite-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex: 1;
}

.affordance-name {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

.affordance-count {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  font-family: 'Monaco', 'Courier New', monospace;
  font-variant-numeric: tabular-nums;
}

/* Respect user's reduced motion preference */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
</style>
