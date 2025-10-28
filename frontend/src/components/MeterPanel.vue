<template>
  <div class="meter-panel">
    <h3>Agent Meters</h3>

    <div v-if="meters" class="meters">
      <div
        v-for="(value, name) in meters"
        :key="name"
        class="meter"
        :class="{
          critical: isCritical(name, value),
          strobe: name === 'stress' && isLonely()
        }"
      >
        <div class="meter-header">
          <span class="meter-name">{{ capitalize(name) }}</span>
          <span class="meter-value">{{ formatValue(name, value) }}</span>
        </div>
        <div class="meter-bar-container">
          <div
            class="meter-bar"
            :style="{
              width: getPercentage(name, value) + '%',
              background: getMeterColor(name, value)
            }"
          ></div>
        </div>
      </div>
    </div>

    <div v-else class="no-data">
      Waiting for simulation data...
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useSimulationStore } from '../stores/simulation'

const store = useSimulationStore()

const meters = computed(() => {
  const agent = store.agentMeters['agent_0']
  return agent ? agent.meters : null
})

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1)
}

function formatValue(name, value) {
  if (name === 'money') {
    return `$${Math.round(value)}`
  }
  if (name === 'stress') {
    // Stress is 0-100 (higher = worse)
    return `${Math.round(value)}`
  }
  // Other meters are normalized 0-1, convert to percentage
  return `${Math.round(value * 100)}%`
}

function getPercentage(name, value) {
  if (name === 'money' || name === 'stress') {
    // Money and stress are already 0-100
    return Math.max(0, Math.min(100, value))
  }
  // Other meters are normalized 0-1, convert to percentage
  return Math.max(0, Math.min(100, value * 100))
}

function isCritical(name, value) {
  const percentage = name === 'money' ? value : (name === 'stress' ? value : value * 100)
  // Stress counts UP - high stress (>80) is critical
  if (name === 'stress') {
    return percentage > 80
  }
  // Other meters - low values (<20) are critical
  return percentage < 20
}

function isLonely() {
  // Check if social is at 0 (causes stress to increase) AND stress is high
  if (!meters.value) return false
  const social = meters.value.social
  const stress = meters.value.stress
  // Only strobe when BOTH lonely (social ≈ 0) AND stressed (stress > 80)
  return social <= 0.01 && stress > 80
}

function getMeterColor(name, value) {
  const percentage = name === 'money' ? value : (name === 'stress' ? value : value * 100)

  // Color mapping
  const colors = {
    energy: { high: '#10b981', mid: '#f59e0b', low: '#ef4444' },
    hygiene: { high: '#06b6d4', mid: '#f59e0b', low: '#ef4444' },
    satiation: { high: '#f59e0b', mid: '#f59e0b', low: '#ef4444' },
    money: { high: '#8b5cf6', mid: '#a78bfa', low: '#ef4444' },
    stress: { low: '#10b981', mid: '#f59e0b', high: '#ef4444' },  // Stress: low=green, high=red
    social: { high: '#ec4899', mid: '#f472b6', low: '#ef4444' }   // Pink - social connections
  }

  const colorSet = colors[name] || colors.energy

  // Stress counts UP (0 = good, 100 = bad)
  if (name === 'stress') {
    if (percentage < 30) return colorSet.low   // Low stress (0-30) = green
    if (percentage < 70) return colorSet.mid   // Medium stress (30-70) = yellow
    return colorSet.high                       // High stress (70-100) = red
  }

  // Normal meters - HIGH is good
  if (percentage > 60) return colorSet.high
  if (percentage > 30) return colorSet.mid
  return colorSet.low
}
</script>

<style scoped>
.meter-panel {
  background: #2a2a3e;
  border-radius: 8px;
  padding: 1.5rem;
}

.meter-panel h3 {
  margin: 0 0 1rem 0;
  font-size: 1.125rem;
  font-weight: 600;
  color: #e0e0e0;
}

.meters {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.meter {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.meter.critical {
  animation: pulse 1s ease-in-out infinite;
}

.meter.strobe {
  animation: strobe 0.5s ease-in-out infinite !important;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.6;
  }
}

@keyframes strobe {
  0%, 100% {
    opacity: 1;
  }
  25% {
    opacity: 0.2;
  }
  50% {
    opacity: 1;
  }
  75% {
    opacity: 0.2;
  }
}

.meter-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.meter-name {
  font-size: 0.875rem;
  font-weight: 500;
  color: #a0a0b0;
}

.meter-value {
  font-size: 0.875rem;
  font-weight: 600;
  color: #e0e0e0;
}

.meter-bar-container {
  width: 100%;
  height: 20px;
  background: #1e1e2e;
  border-radius: 10px;
  overflow: hidden;
}

.meter-bar {
  height: 100%;
  border-radius: 10px;
  transition: width 0.3s ease, background 0.3s ease;
}

.no-data {
  color: #a0a0b0;
  font-size: 0.875rem;
  font-style: italic;
}
</style>
