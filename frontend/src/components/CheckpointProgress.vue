<template>
  <div class="checkpoint-progress">
    <div class="checkpoint-header">
      <h4>Training Checkpoint</h4>
      <InfoTooltip
        title="What is this?"
        text="Shows which training checkpoint the inference server is using. The agent's behaviour reflects learning up to this episode. Newer checkpoints = more training = better performance."
        position="left"
      />
    </div>

    <!-- Empty state when no checkpoint info -->
    <EmptyState
      v-if="!hasCheckpointInfo"
      icon="🔄"
      title="No Checkpoint Loaded"
      message="Waiting for inference server to load a trained agent..."
    />

    <!-- Checkpoint gauge -->
    <div v-else class="checkpoint-content">
      <div class="checkpoint-info">
        <span class="episode-current">Episode {{ currentEpisode }}</span>
        <span class="episode-total">/ {{ totalEpisodes }}</span>
      </div>

      <div class="gauge-container">
        <div class="gauge-track">
          <div
            class="gauge-fill"
            :style="{
              width: progressPercent + '%',
              background: gaugeGradient
            }"
          >
            <div class="gauge-shimmer"></div>
          </div>
        </div>
      </div>

      <div class="progress-labels">
        <span class="label-start">Early Training</span>
        <span class="label-mid">Learning</span>
        <span class="label-end">Converged</span>
      </div>

      <div class="checkpoint-status">
        <span class="status-icon">{{ statusIcon }}</span>
        <span class="status-text">{{ statusText }}</span>
      </div>

      <!-- Checkpoint controls -->
      <div class="checkpoint-controls">
        <button
          @click="$emit('refresh')"
          class="control-button refresh-button"
          title="Manually check for and load the latest checkpoint"
        >
          🔄 Refresh
        </button>
        <button
          @click="$emit('toggle-auto')"
          :class="['control-button', 'auto-button', autoMode ? 'active' : 'inactive']"
          :title="autoMode ? 'Auto mode ON - will load new checkpoints automatically' : 'Auto mode OFF - click to enable automatic checkpoint loading'"
        >
          {{ autoMode ? '⚡ Auto ON' : '⏸ Auto OFF' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import EmptyState from './EmptyState.vue'
import InfoTooltip from './InfoTooltip.vue'

const props = defineProps({
  currentEpisode: {
    type: Number,
    default: 0
  },
  totalEpisodes: {
    type: Number,
    default: 0
  },
  autoMode: {
    type: Boolean,
    default: false
  }
})

defineEmits(['refresh', 'toggle-auto'])

const hasCheckpointInfo = computed(() => {
  return props.currentEpisode > 0 && props.totalEpisodes > 0
})

const progressPercent = computed(() => {
  if (props.totalEpisodes === 0) return 0
  return Math.min((props.currentEpisode / props.totalEpisodes) * 100, 100)
})

const gaugeGradient = computed(() => {
  const percent = progressPercent.value
  // Early: Blue → Mid: Cyan → Late: Green
  if (percent < 33) {
    return 'linear-gradient(90deg, #3b82f6, #06b6d4)'
  } else if (percent < 66) {
    return 'linear-gradient(90deg, #06b6d4, #10b981)'
  } else {
    return 'linear-gradient(90deg, #10b981, #22c55e)'
  }
})

const statusIcon = computed(() => {
  const percent = progressPercent.value
  if (percent < 20) return '🌱'
  if (percent < 50) return '📈'
  if (percent < 80) return '🎯'
  return '✨'
})

const statusText = computed(() => {
  const percent = progressPercent.value
  if (percent < 20) return 'Early exploration phase'
  if (percent < 50) return 'Discovering strategies'
  if (percent < 80) return 'Refining policy'
  return 'Near convergence'
})
</script>

<style scoped>
.checkpoint-progress {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  margin: var(--spacing-sm) 0;
}

.checkpoint-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-md);
}

.checkpoint-header h4 {
  margin: 0;
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

.checkpoint-content {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.checkpoint-info {
  display: flex;
  justify-content: center;
  align-items: baseline;
  gap: var(--spacing-xs);
}

.episode-current {
  font-size: var(--font-size-xl);
  font-weight: var(--font-weight-bold);
  color: var(--color-text-primary);
  font-family: 'Monaco', 'Courier New', monospace;
}

.episode-total {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
  font-family: 'Monaco', 'Courier New', monospace;
}

.gauge-container {
  margin: var(--spacing-sm) 0;
}

.gauge-track {
  height: 24px;
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-full);
  overflow: hidden;
  position: relative;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
}

.gauge-fill {
  height: 100%;
  border-radius: var(--border-radius-full);
  transition: width 0.5s ease, background 0.5s ease;
  position: relative;
  overflow: hidden;
}

.gauge-shimmer {
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

.progress-labels {
  display: flex;
  justify-content: space-between;
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
}

.label-start {
  color: #3b82f6;
}

.label-mid {
  color: #06b6d4;
}

.label-end {
  color: var(--color-success);
}

.checkpoint-status {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-xs) var(--spacing-sm);
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
  margin-top: var(--spacing-xs);
}

.status-icon {
  font-size: 1rem;
}

.status-text {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  font-style: italic;
}

.checkpoint-controls {
  display: flex;
  gap: var(--spacing-xs);
  margin-top: var(--spacing-sm);
}

.control-button {
  flex: 1;
  padding: var(--spacing-xs) var(--spacing-sm);
  border: none;
  border-radius: var(--border-radius-sm);
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-semibold);
  cursor: pointer;
  transition: all var(--transition-base);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-xs);
}

.refresh-button {
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
}

.refresh-button:hover {
  background: var(--color-interactive-hover);
  transform: translateY(-1px);
}

.auto-button.inactive {
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
}

.auto-button.inactive:hover {
  background: var(--color-warning);
  color: var(--color-text-on-dark);
}

.auto-button.active {
  background: var(--color-success);
  color: var(--color-text-on-dark);
  box-shadow: 0 2px 4px rgba(34, 197, 94, 0.3);
}

.auto-button.active:hover {
  background: var(--color-warning);
  color: var(--color-text-on-dark);
}

.control-button:active {
  transform: translateY(0);
}
</style>
