<template>
  <div class="app">
    <!-- ✅ Semantic HTML: header with role="banner" -->
    <header class="header" role="banner">
      <h1>HAMLET - DRL Visualization</h1>
      <div class="connection-status" :class="{ connected: isConnected }" role="status">
        {{ isConnected ? 'Connected' : 'Disconnected' }}
      </div>
    </header>

    <!-- ✅ Semantic HTML: main with role="main" -->
    <main class="main-container" role="main">
      <!-- ✅ Semantic HTML: aside for left panel -->
      <aside class="left-panel" aria-label="Agent status panels">
        <MeterPanel :agent-meters="store.agentMeters" />
        <StatsPanel
          :current-episode="store.currentEpisode"
          :current-step="store.currentStep"
          :cumulative-reward="store.cumulativeReward"
          :last-action="store.lastAction"
          :episode-history="store.episodeHistory"
        />
      </aside>

      <!-- ✅ Semantic HTML: region for grid visualization -->
      <div class="grid-container" role="region" aria-label="Simulation grid">
        <!-- ✅ Show loading state while connecting -->
        <LoadingState
          v-if="isConnecting"
          message="Connecting to simulation server..."
        />

        <!-- ✅ Show error state if connection failed -->
        <ErrorState
          v-else-if="connectionError"
          :title="connectionError.title"
          :message="connectionError.message"
          @retry="retryConnection"
        />

        <!-- ✅ Show grid when connected -->
        <Grid
          v-else-if="isConnected"
          :grid-width="store.gridWidth"
          :grid-height="store.gridHeight"
          :agents="store.agents"
          :affordances="store.affordances"
          :heat-map="store.heatMap"
        />

        <!-- ✅ Show empty state when not connected -->
        <div v-else class="not-connected-message">
          <p>Not connected to simulation server.</p>
          <p class="hint">Use the "Connect (Inference)" or "Connect (Training)" buttons in the controls panel to start.</p>
        </div>
      </div>

      <!-- ✅ Semantic HTML: aside for right panel -->
      <aside class="right-panel" aria-label="Simulation controls">
        <Controls
          :is-connected="store.isConnected"
          :is-training="store.isTraining"
          :available-models="store.availableModels"
          :current-episode="store.currentEpisode"
          :total-episodes="store.totalEpisodes"
          :training-metrics="store.trainingMetrics"
          :server-availability="store.serverAvailability"
          @connect="handleConnect"
          @disconnect="store.disconnect"
          @play="store.play"
          @pause="store.pause"
          @step="store.step"
          @reset="store.reset"
          @set-speed="store.setSpeed"
          @load-model="store.loadModel"
          @start-training="handleStartTraining"
        />
      </aside>
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useSimulationStore } from './stores/simulation'
import Grid from './components/Grid.vue'
import MeterPanel from './components/MeterPanel.vue'
import Controls from './components/Controls.vue'
import StatsPanel from './components/StatsPanel.vue'
import LoadingState from './components/LoadingState.vue'
import ErrorState from './components/ErrorState.vue'

const store = useSimulationStore()
const isConnected = computed(() => store.isConnected)
const isConnecting = computed(() => store.isConnecting)
const connectionError = computed(() => store.connectionError)

// Check which servers are available on mount
onMounted(() => {
  store.checkServerAvailability()
})

// ✅ Retry connection on error
function retryConnection() {
  store.connect(store.mode)
}

// ✅ Handle connect event from Controls (with mode parameter)
function handleConnect(mode) {
  store.connect(mode)
}

// ✅ Handle start training event from Controls
function handleStartTraining(config) {
  store.startTraining(
    config.numEpisodes,
    config.batchSize,
    config.bufferCapacity,
    config.showEvery,
    config.stepDelay
  )
}
</script>

<style scoped>
/* ✅ Refactored to use design tokens from variables.css */
.app {
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  overflow: hidden;
}

/* ✅ Mobile-first header */
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-sm) var(--spacing-md);
  border-bottom: 1px solid var(--color-bg-tertiary);
  flex-wrap: wrap;
  gap: var(--spacing-sm);
}

.header h1 {
  margin: 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
}

/* ✅ Larger header on tablet+ */
@media (min-width: 768px) {
  .header {
    padding: var(--spacing-md) var(--spacing-lg);
    flex-wrap: nowrap;
  }

  .header h1 {
    font-size: var(--font-size-xl);
  }
}

/* ✅ Full size header on desktop */
@media (min-width: 1024px) {
  .header {
    padding: var(--spacing-md) var(--spacing-xl);
  }

  .header h1 {
    font-size: var(--font-size-2xl);
  }
}

.connection-status {
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--border-radius-sm);
  background: var(--color-error);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
  transition: var(--transition-base);
}

.connection-status.connected {
  background: var(--color-success);
}

/* ✅ Mobile-first responsive layout */
.main-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  overflow: hidden;
}

.left-panel {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  overflow-y: auto;
}

.grid-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  min-width: 0;
  min-height: 300px;
}

.right-panel {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  overflow-y: auto;
}

/* ✅ Tablet breakpoint: side-by-side panels */
@media (min-width: 768px) {
  .main-container {
    flex-direction: row;
    gap: var(--spacing-lg);
    padding: var(--spacing-lg);
  }

  .left-panel {
    width: 280px;
    gap: var(--spacing-lg);
  }

  .grid-container {
    padding: var(--spacing-lg);
  }

  .right-panel {
    width: 320px;
    gap: var(--spacing-lg);
  }
}

/* ✅ Desktop breakpoint: full layout */
@media (min-width: 1024px) {
  .main-container {
    padding: var(--spacing-xl);
  }

  .left-panel {
    width: var(--layout-left-panel-width);
  }

  .grid-container {
    padding: var(--spacing-xl);
  }

  .right-panel {
    width: var(--layout-right-panel-width);
  }
}

/* ✅ Not connected message styling */
.not-connected-message {
  text-align: center;
  color: var(--color-text-secondary);
  padding: var(--spacing-2xl);
}

.not-connected-message p {
  margin: 0 0 var(--spacing-md) 0;
  font-size: var(--font-size-base);
}

.not-connected-message .hint {
  font-size: var(--font-size-sm);
  color: var(--color-text-tertiary);
  max-width: 400px;
  margin: 0 auto;
}
</style>
