<template>
  <div class="controls-panel">
    <h3>Controls</h3>

    <!-- Mode Selector -->
    <div class="mode-selector">
      <label>Mode:</label>
      <div class="mode-buttons">
        <button
          @click="selectedMode = 'inference'"
          :class="['mode-button', { active: selectedMode === 'inference' }]"
          :disabled="isConnected"
        >
          Inference
        </button>
        <button
          @click="selectedMode = 'training'"
          :class="['mode-button', { active: selectedMode === 'training' }]"
          :disabled="isConnected"
        >
          Training
        </button>
      </div>
    </div>

    <!-- Connection Control -->
    <div class="connection-control">
      <button
        v-if="!isConnected"
        @click="connect"
        class="primary-button"
      >
        Connect
      </button>
      <button
        v-else
        @click="disconnect"
        class="secondary-button"
      >
        Disconnect
      </button>
    </div>

    <!-- Inference Mode Controls -->
    <div v-if="selectedMode === 'inference' && isConnected" class="inference-controls">
      <div class="button-group">
        <button
          @click="store.play()"
          class="control-button play"
          title="Play"
        >
          ▶
        </button>

        <button
          @click="store.pause()"
          class="control-button pause"
          title="Pause"
        >
          ⏸
        </button>

        <button
          @click="store.step()"
          class="control-button step"
          title="Step Forward"
        >
          ⏭
        </button>

        <button
          @click="store.reset()"
          class="control-button reset"
          title="Reset Episode"
        >
          ↻
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

      <div v-if="availableModels.length > 0" class="model-selector">
        <label for="model">Model:</label>
        <select
          id="model"
          v-model="selectedModel"
          @change="onModelChange"
          class="model-select"
        >
          <option v-for="model in availableModels" :key="model" :value="model">
            {{ model }}
          </option>
        </select>
      </div>
    </div>

    <!-- Training Mode Controls -->
    <div v-if="selectedMode === 'training' && isConnected" class="training-controls">
      <div v-if="!isTraining" class="training-config">
        <div class="form-group">
          <label for="num-episodes">Episodes:</label>
          <input
            id="num-episodes"
            v-model.number="trainingConfig.numEpisodes"
            type="number"
            min="1"
            max="10000"
            class="number-input"
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
          />
          <span class="hint">1 = show all episodes, 5 = show every 5th episode (faster)</span>
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
          />
          <span class="hint">Time between steps when visualizing (0.2s = smooth)</span>
        </div>

        <button @click="startTraining" class="primary-button">
          Start Training
        </button>
      </div>

      <div v-else class="training-status">
        <div class="training-progress">
          <div class="progress-text">
            Episode {{ currentEpisode }} / {{ totalEpisodes }}
          </div>
          <div class="progress-bar">
            <div
              class="progress-fill"
              :style="{ width: `${(currentEpisode / totalEpisodes) * 100}%` }"
            ></div>
          </div>
        </div>

        <div class="training-metrics">
          <div class="metric">
            <span class="metric-label">Avg Reward (5):</span>
            <span class="metric-value">{{ trainingMetrics.avgReward5.toFixed(2) }}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Avg Length (5):</span>
            <span class="metric-value">{{ trainingMetrics.avgLength5.toFixed(1) }}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Avg Loss (5):</span>
            <span class="metric-value">{{ trainingMetrics.avgLoss5.toFixed(4) }}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Epsilon:</span>
            <span class="metric-value">{{ trainingMetrics.epsilon.toFixed(3) }}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Buffer Size:</span>
            <span class="metric-value">{{ trainingMetrics.bufferSize }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useSimulationStore } from '../stores/simulation'

const store = useSimulationStore()

const isConnected = computed(() => store.isConnected)
const availableModels = computed(() => store.availableModels)
const isTraining = computed(() => store.isTraining)
const currentEpisode = computed(() => store.currentEpisode)
const totalEpisodes = computed(() => store.totalEpisodes)
const trainingMetrics = computed(() => store.trainingMetrics)

const selectedMode = ref('inference')
const speedValue = ref(1.0)
const selectedModel = ref(null)

const trainingConfig = ref({
  numEpisodes: 100,
  batchSize: 32,
  bufferCapacity: 10000,
  showEvery: 1,  // Show every episode by default
  stepDelay: 0.2,  // 200ms delay between steps
})

// Set initial model when available
if (availableModels.value.length > 0) {
  selectedModel.value = availableModels.value[0]
}

function connect() {
  store.connect(selectedMode.value)
}

function disconnect() {
  store.disconnect()
}

function startTraining() {
  store.startTraining(
    trainingConfig.value.numEpisodes,
    trainingConfig.value.batchSize,
    trainingConfig.value.bufferCapacity,
    trainingConfig.value.showEvery,
    trainingConfig.value.stepDelay
  )
}

function onSpeedChange() {
  store.setSpeed(speedValue.value)
}

function onModelChange() {
  if (selectedModel.value) {
    store.loadModel(selectedModel.value)
  }
}
</script>

<style scoped>
.controls-panel {
  background: #2a2a3e;
  border-radius: 8px;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.controls-panel h3 {
  margin: 0;
  font-size: 1.125rem;
  font-weight: 600;
  color: #e0e0e0;
}

/* Mode Selector */
.mode-selector {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.mode-selector label {
  font-size: 0.875rem;
  font-weight: 500;
  color: #a0a0b0;
}

.mode-buttons {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
}

.mode-button {
  padding: 0.75rem;
  background: #3a3a4e;
  border: 2px solid transparent;
  border-radius: 6px;
  color: #a0a0b0;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.mode-button:hover:not(:disabled) {
  background: #4a4a5e;
}

.mode-button.active {
  background: #3b82f6;
  color: #ffffff;
  border-color: #60a5fa;
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
  padding: 0.75rem;
  background: #10b981;
  border: none;
  border-radius: 6px;
  color: #ffffff;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.primary-button:hover {
  background: #059669;
  transform: translateY(-1px);
}

.primary-button:active {
  transform: translateY(0);
}

.secondary-button {
  width: 100%;
  padding: 0.75rem;
  background: #ef4444;
  border: none;
  border-radius: 6px;
  color: #ffffff;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.secondary-button:hover {
  background: #dc2626;
  transform: translateY(-1px);
}

.secondary-button:active {
  transform: translateY(0);
}

/* Inference Controls */
.inference-controls {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.button-group {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.5rem;
}

.control-button {
  padding: 0.75rem;
  font-size: 1.25rem;
  background: #3a3a4e;
  border: none;
  border-radius: 6px;
  color: #e0e0e0;
  cursor: pointer;
  transition: all 0.2s ease;
}

.control-button:hover:not(:disabled) {
  background: #4a4a5e;
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
  background: #10b981;
}

.control-button.pause:hover:not(:disabled) {
  background: #f59e0b;
}

.control-button.step:hover:not(:disabled) {
  background: #3b82f6;
}

.control-button.reset:hover:not(:disabled) {
  background: #ef4444;
}

.speed-control {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.speed-control label {
  font-size: 0.875rem;
  font-weight: 500;
  color: #a0a0b0;
}

.speed-slider {
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: #3a3a4e;
  outline: none;
  -webkit-appearance: none;
}

.speed-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #3b82f6;
  cursor: pointer;
  transition: background 0.2s ease;
}

.speed-slider::-webkit-slider-thumb:hover {
  background: #60a5fa;
}

.speed-slider::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #3b82f6;
  cursor: pointer;
  border: none;
  transition: background 0.2s ease;
}

.speed-slider::-moz-range-thumb:hover {
  background: #60a5fa;
}

.speed-slider:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.speed-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: #6a6a7a;
}

.model-selector {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.model-selector label {
  font-size: 0.875rem;
  font-weight: 500;
  color: #a0a0b0;
}

.model-select {
  padding: 0.5rem;
  background: #3a3a4e;
  border: 1px solid #4a4a5e;
  border-radius: 6px;
  color: #e0e0e0;
  font-size: 0.875rem;
  cursor: pointer;
  outline: none;
}

.model-select:hover:not(:disabled) {
  border-color: #5a5a6e;
}

.model-select:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* Training Controls */
.training-controls {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.training-config {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-group label {
  font-size: 0.875rem;
  font-weight: 500;
  color: #a0a0b0;
}

.number-input {
  padding: 0.5rem;
  background: #3a3a4e;
  border: 1px solid #4a4a5e;
  border-radius: 6px;
  color: #e0e0e0;
  font-size: 0.875rem;
  outline: none;
}

.number-input:focus {
  border-color: #3b82f6;
}

.number-input:hover:not(:disabled) {
  border-color: #5a5a6e;
}

.hint {
  font-size: 0.75rem;
  color: #6a6a7a;
  font-style: italic;
  margin-top: 0.25rem;
}

/* Training Status */
.training-status {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.training-progress {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.progress-text {
  font-size: 0.875rem;
  font-weight: 500;
  color: #e0e0e0;
  text-align: center;
}

.progress-bar {
  height: 8px;
  background: #3a3a4e;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #10b981);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.training-metrics {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 1rem;
  background: #1e1e2e;
  border-radius: 6px;
}

.metric {
  display: flex;
  justify-content: space-between;
  font-size: 0.875rem;
}

.metric-label {
  color: #a0a0b0;
}

.metric-value {
  color: #e0e0e0;
  font-weight: 600;
  font-family: 'Monaco', 'Courier New', monospace;
}
</style>
