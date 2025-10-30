<template>
  <!-- ✅ Semantic HTML: section instead of div -->
  <section class="controls-panel" aria-labelledby="controls-heading">
    <h3 id="controls-heading">Controls</h3>

    <!-- Connection status indicator -->
    <div class="connection-status" v-if="props.isConnected">
      <span class="status-indicator connected"></span>
      <span>Live Inference (Latest Checkpoint)</span>
    </div>

    <!-- ✅ Connection Control with better semantics -->
    <div class="connection-control">
      <button
        v-if="!props.isConnected"
        @click="connect"
        class="primary-button"
        aria-label="Connect to live inference server"
      >
        Connect
      </button>
      <button
        v-else
        @click="disconnect"
        class="secondary-button"
        aria-label="Disconnect from live inference server"
      >
        Disconnect
      </button>
    </div>

    <!-- ✅ Playback Controls (shown when connected) -->
    <div v-if="props.isConnected" class="inference-controls">
      <div class="button-group" role="group" aria-label="Playback controls">
        <button
          @click="emit('play')"
          class="control-button play"
          aria-label="Play simulation"
        >
          <span aria-hidden="true">▶</span>
        </button>

        <button
          @click="emit('pause')"
          class="control-button pause"
          aria-label="Pause simulation"
        >
          <span aria-hidden="true">⏸</span>
        </button>

        <button
          @click="emit('step')"
          class="control-button step"
          aria-label="Step forward one action"
        >
          <span aria-hidden="true">⏭</span>
        </button>

        <button
          @click="emit('reset')"
          class="control-button reset"
          aria-label="Reset episode"
        >
          <span aria-hidden="true">↻</span>
        </button>
      </div>

      <div class="speed-control">
        <label for="speed">Speed: {{ speedValue }}x</label>
        <input
          id="speed"
          type="range"
          min="0.5"
          max="10"
          step="0.5"
          v-model.number="speedValue"
          @input="onSpeedChange"
          class="speed-slider"
        />
        <div class="speed-labels">
          <span>0.5x</span>
          <span>1x</span>
          <span>5x</span>
          <span>10x</span>
        </div>
      </div>

      <div class="model-controls">
        <div v-if="props.availableModels.length > 0" class="model-selector">
          <label for="model">Model:</label>
          <select
            id="model"
            v-model="selectedModel"
            @change="onModelChange"
            class="model-select"
          >
            <option v-for="model in props.availableModels" :key="model" :value="model">
              {{ model }}
            </option>
          </select>
        </div>
      </div>
    </div>

    <!-- ✅ Training Mode Controls with semantic form -->
    <!-- Training controls removed - live inference only -->
    <div v-if="false" class="training-controls">
      <form v-if="!props.isTraining" class="training-config" @submit.prevent="startTraining">
        <fieldset>
          <legend class="sr-only">Training Configuration</legend>

          <div class="form-group">
            <label for="num-episodes">Episodes:</label>
            <input
              id="num-episodes"
              v-model.number="trainingConfig.numEpisodes"
              type="number"
              min="1"
              max="10000"
              class="number-input"
              aria-required="true"
            />
          </div>

          <div class="form-group">
            <label for="batch-size">Batch Size:</label>
            <input
              id="batch-size"
              v-model.number="trainingConfig.batchSize"
              type="number"
              min="1"
              max="256"
              class="number-input"
              aria-required="true"
            />
          </div>

          <div class="form-group">
            <label for="buffer-capacity">Buffer Capacity:</label>
            <input
              id="buffer-capacity"
              v-model.number="trainingConfig.bufferCapacity"
              type="number"
              min="1000"
              max="100000"
              step="1000"
              class="number-input"
              aria-required="true"
            />
          </div>

          <div class="form-group">
            <label for="show-every">Show Every N Episodes:</label>
            <input
              id="show-every"
              v-model.number="trainingConfig.showEvery"
              type="number"
              min="1"
              max="100"
              class="number-input"
              aria-required="true"
              aria-describedby="show-every-hint"
            />
            <span id="show-every-hint" class="hint">1 = show all episodes, 5 = show every 5th episode (faster)</span>
          </div>

          <div class="form-group">
            <label for="step-delay">Step Delay (seconds):</label>
            <input
              id="step-delay"
              v-model.number="trainingConfig.stepDelay"
              type="number"
              min="0.05"
              max="2"
              step="0.05"
              class="number-input"
              aria-required="true"
              aria-describedby="step-delay-hint"
            />
            <span id="step-delay-hint" class="hint">Time between steps when visualizing (0.2s = smooth)</span>
          </div>

          <button type="submit" class="primary-button">
            Start Training
          </button>
        </fieldset>
      </form>

      <!-- ✅ Training status with accessibility -->
      <div v-else class="training-status">
        <div class="training-progress">
          <!-- ✅ Progress text with live region -->
          <div
            class="progress-text"
            aria-live="polite"
            aria-atomic="true"
            role="status"
          >
            Episode {{ props.currentEpisode }} / {{ props.totalEpisodes }}
          </div>
          <!-- ✅ Progress bar with proper role and ARIA -->
          <div class="progress-bar-container">
            <div
              class="progress-bar"
              role="progressbar"
              :aria-valuenow="props.currentEpisode"
              aria-valuemin="0"
              :aria-valuemax="props.totalEpisodes"
              :aria-label="`Training progress: ${props.currentEpisode} of ${props.totalEpisodes} episodes completed`"
              :style="{ width: `${trainingProgressPercentage}%` }"
            ></div>
          </div>
        </div>

        <!-- ✅ Training metrics with ARIA live regions -->
        <div class="training-metrics">
          <div class="metric">
            <span class="metric-label">Avg Reward (5):</span>
            <span
              class="metric-value"
              aria-live="polite"
              role="status"
            >
              {{ formattedAvgReward }}
            </span>
          </div>
          <div class="metric">
            <span class="metric-label">Avg Length (5):</span>
            <span
              class="metric-value"
              aria-live="polite"
              role="status"
            >
              {{ formattedAvgLength }}
            </span>
          </div>
          <div class="metric">
            <span class="metric-label">Avg Loss (5):</span>
            <span
              class="metric-value"
              aria-live="polite"
              role="status"
            >
              {{ formattedAvgLoss }}
            </span>
          </div>
          <div class="metric">
            <span class="metric-label">Epsilon:</span>
            <span
              class="metric-value"
              aria-live="polite"
              role="status"
            >
              {{ formattedEpsilon }}
            </span>
          </div>
          <div class="metric">
            <span class="metric-label">Buffer Size:</span>
            <span
              class="metric-value"
              aria-live="polite"
              role="status"
            >
              {{ formatNumber(trainingMetrics.bufferSize) }}
            </span>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref, computed } from 'vue'
import { formatTrainingMetric, formatNumber } from '../utils/formatting'

// ✅ Props First: Receive data from parent instead of importing store
const props = defineProps({
  isConnected: {
    type: Boolean,
    default: false
  },
  isTraining: {
    type: Boolean,
    default: false
  },
  availableModels: {
    type: Array,
    default: () => []
  },
  currentEpisode: {
    type: Number,
    default: 0
  },
  totalEpisodes: {
    type: Number,
    default: 0
  },
  trainingMetrics: {
    type: Object,
    default: () => ({
      avgReward5: 0,
      avgLength5: 0,
      avgLoss5: 0,
      epsilon: 1.0,
      bufferSize: 0
    })
  },
  serverAvailability: {
    type: Object,
    default: () => ({
      inference: false,
      training: false,
      checked: false
    })
  }
})

// ✅ Emit events instead of calling store methods directly
const emit = defineEmits([
  'connect',
  'disconnect',
  'play',
  'pause',
  'step',
  'reset',
  'set-speed',
  'load-model',
  'start-training'
])

// Mode removed - always run live inference on latest checkpoint
const speedValue = ref(1.0)
const selectedModel = ref(null)
const defaultModelName = 'trained_agent.pt'

const trainingConfig = ref({
  numEpisodes: 100,
  batchSize: 32,
  bufferCapacity: 10000,
  showEvery: 1,  // Show every episode by default
  stepDelay: 0.2,  // 200ms delay between steps
})

// ✅ Extract toFixed() from template to computed properties
const trainingProgressPercentage = computed(() => {
  if (props.totalEpisodes === 0) return 0
  return (props.currentEpisode / props.totalEpisodes) * 100
})

// ✅ Use imported formatting utility
const formattedAvgReward = computed(() => {
  return formatTrainingMetric(props.trainingMetrics.avgReward5, 'reward')
})

const formattedAvgLength = computed(() => {
  return formatTrainingMetric(props.trainingMetrics.avgLength5, 'length')
})

const formattedAvgLoss = computed(() => {
  return formatTrainingMetric(props.trainingMetrics.avgLoss5, 'loss')
})

const formattedEpsilon = computed(() => {
  return formatTrainingMetric(props.trainingMetrics.epsilon, 'epsilon')
})

// Set initial model when available
if (props.availableModels.length > 0) {
  selectedModel.value = props.availableModels[0]
}

// ✅ Emit events instead of calling store methods
function connect() {
  emit('connect', 'inference')  // Always connect in inference mode
}

function disconnect() {
  emit('disconnect')
}

function startTraining() {
  emit('start-training', {
    numEpisodes: trainingConfig.value.numEpisodes,
    batchSize: trainingConfig.value.batchSize,
    bufferCapacity: trainingConfig.value.bufferCapacity,
    showEvery: trainingConfig.value.showEvery,
    stepDelay: trainingConfig.value.stepDelay
  })
}

function onSpeedChange() {
  emit('set-speed', speedValue.value)
}

function onModelChange() {
  if (selectedModel.value) {
    emit('load-model', selectedModel.value)
  }
}
</script>

<style scoped>
/* ✅ Mobile-first: Refactored to use design tokens */
.controls-panel {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

@media (min-width: 768px) {
  .controls-panel {
    padding: var(--spacing-lg);
    gap: var(--spacing-lg);
  }
}

.controls-panel h3 {
  margin: 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

/* Connection Status Indicator */
.connection-status {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-sm);
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
}

.status-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
}

.status-indicator.connected {
  background: var(--color-success);
  box-shadow: 0 0 8px var(--color-success);
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.6;
  }
}

/* Mode Selector (deprecated - kept for compatibility) */
.mode-selector {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  border: none; /* Remove default fieldset border */
  padding: 0;
  margin: 0;
}

.mode-selector label {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
  color: var(--color-text-secondary);
}

.mode-buttons {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--spacing-sm);
}

.mode-button {
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-bg-tertiary);
  border: 2px solid transparent;
  border-radius: var(--border-radius-sm);
  color: var(--color-text-secondary);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
  cursor: pointer;
  transition: all var(--transition-base);
}

.mode-button:hover:not(:disabled) {
  background: var(--color-interactive-disabled);
}

.mode-button.active {
  background: var(--color-mode-inference);
  color: var(--color-text-on-dark);
  border-color: var(--color-interactive-focus);
}

.mode-button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* Connection Control */
.connection-control {
  display: flex;
}

.primary-button {
  width: 100%;
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-success);
  border: none;
  border-radius: var(--border-radius-sm);
  color: var(--color-text-on-dark);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  cursor: pointer;
  transition: all var(--transition-base);
}

.primary-button:hover {
  background: var(--color-interactive-hover);
  transform: translateY(-1px);
}

.primary-button:active {
  transform: translateY(0);
}

.secondary-button {
  width: 100%;
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-error);
  border: none;
  border-radius: var(--border-radius-sm);
  color: var(--color-text-on-dark);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  cursor: pointer;
  transition: all var(--transition-base);
}

.secondary-button:hover {
  background: var(--color-error-hover);
  transform: translateY(-1px);
}

.secondary-button:active {
  transform: translateY(0);
}

/* Inference Controls */
.inference-controls {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

@media (min-width: 768px) {
  .inference-controls {
    gap: var(--spacing-lg);
  }
}

/* ✅ Mobile-first: 2x2 grid on mobile, 4 columns on tablet+ */
.button-group {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--spacing-sm);
}

@media (min-width: 768px) {
  .button-group {
    grid-template-columns: repeat(4, 1fr);
  }
}

.control-button {
  padding: var(--spacing-sm) var(--spacing-md);
  font-size: var(--font-size-xl);
  background: var(--color-bg-tertiary);
  border: none;
  border-radius: var(--border-radius-sm);
  color: var(--color-text-primary);
  cursor: pointer;
  transition: all var(--transition-base);
}

.control-button:hover:not(:disabled) {
  background: var(--color-interactive-disabled);
  transform: translateY(-1px);
}

.control-button:active:not(:disabled) {
  transform: translateY(0);
}

.control-button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.control-button.play:hover:not(:disabled) {
  background: var(--color-success);
}

.control-button.pause:hover:not(:disabled) {
  background: var(--color-warning);
}

.control-button.step:hover:not(:disabled) {
  background: var(--color-info);
}

.control-button.reset:hover:not(:disabled) {
  background: var(--color-error);
}

.speed-control {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.speed-control label {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
  color: var(--color-text-secondary);
}

.speed-slider {
  width: 100%;
  height: 6px;
  border-radius: var(--border-radius-sm);
  background: var(--color-bg-tertiary);
  outline: none;
  -webkit-appearance: none;
}

.speed-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: var(--border-radius-full);
  background: var(--color-info);
  cursor: pointer;
  transition: background var(--transition-base);
}

.speed-slider::-webkit-slider-thumb:hover {
  background: var(--color-interactive-focus);
}

.speed-slider::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: var(--border-radius-full);
  background: var(--color-info);
  cursor: pointer;
  border: none;
  transition: background var(--transition-base);
}

.speed-slider::-moz-range-thumb:hover {
  background: var(--color-interactive-focus);
}

.speed-slider:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.speed-labels {
  display: flex;
  justify-content: space-between;
  font-size: var(--font-size-xs);
  color: var(--color-text-muted);
}

.model-selector {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.model-selector label {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
  color: var(--color-text-secondary);
}

/* ✅ Mobile-friendly: proper touch target sizing */
.model-select {
  padding: var(--spacing-sm);
  min-height: var(--a11y-min-touch-target);
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-interactive-disabled);
  border-radius: var(--border-radius-sm);
  color: var(--color-text-primary);
  font-size: var(--font-size-base);
  cursor: pointer;
  outline: none;
}

@media (min-width: 768px) {
  .model-select {
    font-size: var(--font-size-sm);
  }
}

.model-select:hover:not(:disabled) {
  border-color: var(--color-text-tertiary);
}

.model-select:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* Training Controls */
.training-controls {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.training-config {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.training-config fieldset {
  border: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.form-group label {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
  color: var(--color-text-secondary);
}

/* ✅ Mobile-friendly: proper touch target sizing */
.number-input {
  padding: var(--spacing-sm);
  min-height: var(--a11y-min-touch-target);
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-interactive-disabled);
  border-radius: var(--border-radius-sm);
  color: var(--color-text-primary);
  font-size: var(--font-size-base);
  outline: none;
}

@media (min-width: 768px) {
  .number-input {
    font-size: var(--font-size-sm);
  }
}

.number-input:focus {
  border-color: var(--color-info);
}

.number-input:hover:not(:disabled) {
  border-color: var(--color-text-tertiary);
}

.hint {
  font-size: var(--font-size-xs);
  color: var(--color-text-muted);
  font-style: italic;
  margin-top: var(--spacing-xs);
}

/* Training Status */
.training-status {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.training-progress {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.progress-text {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
  color: var(--color-text-primary);
  text-align: center;
}

/* ✅ Updated to match template changes */
.progress-bar-container {
  height: 8px;
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-sm);
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, var(--color-info), var(--color-success));
  border-radius: var(--border-radius-sm);
  transition: width var(--transition-base);
}

.training-metrics {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
}

.metric {
  display: flex;
  justify-content: space-between;
  font-size: var(--font-size-sm);
}

.metric-label {
  color: var(--color-text-secondary);
}

.metric-value {
  color: var(--color-text-primary);
  font-weight: var(--font-weight-semibold);
  font-family: 'Monaco', 'Courier New', monospace;
}

/* ✅ Screen reader only utility class */
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
</style>
