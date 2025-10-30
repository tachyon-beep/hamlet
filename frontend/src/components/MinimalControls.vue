<template>
  <!-- Ultra-minimal top-right controls -->
  <div class="minimal-controls" role="region" aria-label="Simulation controls">
    <!-- Refresh checkpoint button -->
    <button
      @click="refreshCheckpoint"
      class="control-button refresh-button"
      aria-label="Manually refresh to latest checkpoint"
      title="Refresh checkpoint"
    >
      <span aria-hidden="true">🔄</span>
    </button>

    <!-- Auto checkpoint toggle -->
    <button
      @click="toggleAutoCheckpoint"
      :class="['control-button', 'auto-button', autoMode ? 'active' : 'inactive']"
      :aria-label="autoMode ? 'Auto checkpoint mode ON' : 'Auto checkpoint mode OFF'"
      :title="autoMode ? 'Auto ON - loads new checkpoints automatically' : 'Auto OFF - click to enable'"
    >
      <span aria-hidden="true">{{ autoMode ? '⚡' : '⏸' }}</span>
    </button>

    <!-- Speed control -->
    <div class="speed-control">
      <label for="speed" class="sr-only">Playback speed</label>
      <span class="speed-label">{{ speedValue }}x</span>
      <input
        id="speed"
        type="range"
        min="0.5"
        max="10"
        step="0.5"
        v-model.number="speedValue"
        @input="onSpeedChange"
        class="speed-slider"
        aria-label="Adjust playback speed"
      />
    </div>

    <!-- Disconnect button -->
    <button
      @click="disconnect"
      class="disconnect-button"
      aria-label="Disconnect from server"
      title="Disconnect"
    >
      <span aria-hidden="true">✕</span>
    </button>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  isConnected: {
    type: Boolean,
    default: false
  },
  autoMode: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['disconnect', 'set-speed', 'refresh-checkpoint', 'toggle-auto-checkpoint'])

const speedValue = ref(1.0)

function onSpeedChange() {
  emit('set-speed', speedValue.value)
}

function disconnect() {
  emit('disconnect')
}

function refreshCheckpoint() {
  emit('refresh-checkpoint')
}

function toggleAutoCheckpoint() {
  emit('toggle-auto-checkpoint')
}
</script>

<style scoped>
/* Ultra-minimal design for top-right corner */
.minimal-controls {
  position: absolute;
  top: var(--spacing-md);
  right: var(--spacing-md);
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  background: var(--color-bg-secondary);
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--border-radius-md);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  z-index: 100;
  backdrop-filter: blur(8px);
}

@media (min-width: 768px) {
  .minimal-controls {
    padding: var(--spacing-sm) var(--spacing-lg);
  }
}

/* Speed control */
.speed-control {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.speed-label {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
  font-family: 'Monaco', 'Courier New', monospace;
  min-width: 40px;
  text-align: right;
}

.speed-slider {
  width: 120px;
  height: 6px;
  border-radius: var(--border-radius-sm);
  background: var(--color-bg-tertiary);
  outline: none;
  -webkit-appearance: none;
  cursor: pointer;
}

/* Slider thumb - Webkit */
.speed-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: var(--border-radius-full);
  background: var(--color-success);
  cursor: pointer;
  transition: all var(--transition-base);
  box-shadow: 0 2px 4px rgba(34, 197, 94, 0.4);
}

.speed-slider::-webkit-slider-thumb:hover {
  transform: scale(1.15);
  box-shadow: 0 2px 6px rgba(34, 197, 94, 0.6);
}

/* Slider thumb - Firefox */
.speed-slider::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border-radius: var(--border-radius-full);
  background: var(--color-success);
  cursor: pointer;
  border: none;
  transition: all var(--transition-base);
  box-shadow: 0 2px 4px rgba(34, 197, 94, 0.4);
}

.speed-slider::-moz-range-thumb:hover {
  transform: scale(1.15);
  box-shadow: 0 2px 6px rgba(34, 197, 94, 0.6);
}

/* Control buttons (refresh, auto) */
.control-button {
  padding: var(--spacing-xs) var(--spacing-sm);
  border: none;
  border-radius: var(--border-radius-sm);
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-bold);
  cursor: pointer;
  transition: all var(--transition-base);
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 36px;
  min-height: 36px;
}

.refresh-button {
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
}

.refresh-button:hover {
  background: var(--color-interactive-hover);
  transform: scale(1.05);
}

.auto-button.inactive {
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
}

.auto-button.inactive:hover {
  background: var(--color-warning);
  color: var(--color-text-on-dark);
  transform: scale(1.05);
}

.auto-button.active {
  background: var(--color-success);
  color: var(--color-text-on-dark);
  box-shadow: 0 2px 4px rgba(34, 197, 94, 0.3);
}

.auto-button.active:hover {
  background: var(--color-warning);
  color: var(--color-text-on-dark);
  transform: scale(1.05);
}

/* Disconnect button */
.disconnect-button {
  padding: var(--spacing-xs) var(--spacing-sm);
  background: var(--color-error);
  border: none;
  border-radius: var(--border-radius-sm);
  color: var(--color-text-on-dark);
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-bold);
  cursor: pointer;
  transition: all var(--transition-base);
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 36px;
  min-height: 36px;
}

.disconnect-button:hover {
  background: var(--color-error-hover);
  transform: scale(1.05);
}

.control-button:active,
.disconnect-button:active {
  transform: scale(0.98);
}

/* Screen reader only utility */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

/* Responsive adjustments */
@media (max-width: 767px) {
  .minimal-controls {
    top: var(--spacing-xs);
    right: var(--spacing-xs);
    padding: var(--spacing-xs) var(--spacing-sm);
    gap: var(--spacing-sm);
  }

  .speed-slider {
    width: 80px;
  }

  .speed-label {
    font-size: var(--font-size-xs);
    min-width: 32px;
  }

  .disconnect-button {
    min-width: 32px;
    min-height: 32px;
    font-size: var(--font-size-base);
  }
}
</style>
