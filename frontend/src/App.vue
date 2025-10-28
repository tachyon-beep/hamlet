<template>
  <div class="app">
    <header class="header">
      <h1>HAMLET - DRL Visualization</h1>
      <div class="connection-status" :class="{ connected: isConnected }">
        {{ isConnected ? 'Connected' : 'Disconnected' }}
      </div>
    </header>

    <div class="main-container">
      <!-- Left Panel: Meters + Episode Info -->
      <div class="left-panel">
        <MeterPanel />
        <StatsPanel />
      </div>

      <!-- Center: Grid -->
      <div class="grid-container">
        <Grid />
      </div>

      <!-- Right Panel: Controls -->
      <div class="right-panel">
        <Controls />
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useSimulationStore } from './stores/simulation'
import Grid from './components/Grid.vue'
import MeterPanel from './components/MeterPanel.vue'
import Controls from './components/Controls.vue'
import StatsPanel from './components/StatsPanel.vue'

const store = useSimulationStore()
const isConnected = computed(() => store.isConnected)

// User will connect manually via Controls panel
</script>

<style scoped>
.app {
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: #1e1e2e;
  color: #e0e0e0;
  overflow: hidden;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  border-bottom: 1px solid #3a3a4e;
}

.header h1 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
}

.connection-status {
  padding: 0.5rem 1rem;
  border-radius: 4px;
  background: #ef4444;
  font-size: 0.875rem;
  font-weight: 500;
}

.connection-status.connected {
  background: #10b981;
}

.main-container {
  flex: 1;
  display: flex;
  gap: 1.5rem;
  padding: 2rem;
  overflow: hidden;
}

.left-panel {
  width: 320px;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  overflow-y: auto;
}

.grid-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #2a2a3e;
  border-radius: 8px;
  padding: 2rem;
  min-width: 0; /* Allow flex shrinking */
}

.right-panel {
  width: 380px;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  overflow-y: auto;
}
</style>
