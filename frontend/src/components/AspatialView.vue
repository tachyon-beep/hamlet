<template>
  <div class="aspatial-container">
    <!-- Large meters display -->
    <div class="large-meters-panel">
      <h2>Agent Status</h2>
      <div
        v-for="(value, name) in meters"
        :key="name"
        class="large-meter"
        :class="getMeterClass(name, value)"
      >
        <div class="meter-header">
          <span class="meter-name">{{ formatMeterName(name) }}</span>
          <span class="meter-value">{{ (value * 100).toFixed(0) }}%</span>
        </div>
        <div class="meter-bar-container">
          <div
            class="meter-bar-fill"
            :style="{ width: `${value * 100}%` }"
          ></div>
        </div>
      </div>
    </div>

    <!-- Available affordances -->
    <div class="affordance-list-panel">
      <h2>Available Interactions</h2>
      <div class="affordance-grid">
        <div
          v-for="affordance in affordances"
          :key="affordance.type"
          class="affordance-card"
        >
          <div class="affordance-icon">{{ getAffordanceIcon(affordance.type) }}</div>
          <div class="affordance-name">{{ affordance.type }}</div>
          <div class="affordance-status">Ready</div>
        </div>
      </div>
    </div>

    <!-- Recent actions log -->
    <div class="action-history-panel">
      <h2>Recent Actions</h2>
      <div class="action-list">
        <div
          v-for="action in recentActions"
          :key="action.id"
          class="action-item"
        >
          <span class="action-step">Step {{ action.step }}</span>
          <span class="action-type">{{ action.actionName }}</span>
          <span v-if="action.affordance" class="action-affordance">
            {{ getAffordanceIcon(action.affordance) }} {{ action.affordance }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { AFFORDANCE_ICONS } from '../utils/constants'

const props = defineProps({
  meters: {
    type: Object,
    default: () => ({})
  },
  affordances: {
    type: Array,
    default: () => []
  },
  currentStep: {
    type: Number,
    default: 0
  },
  lastAction: {
    type: Number,
    default: null
  },
  lastAffordance: {
    type: String,
    default: null
  }
})

// Track recent actions (last 10)
const recentActions = ref([])
let actionIdCounter = 0

// Action names for display
const ACTION_NAMES = ['Up', 'Down', 'Left', 'Right', 'Interact', 'Wait']

// Watch for new actions
watch(() => props.lastAction, (newAction) => {
  if (newAction !== null) {
    const action = {
      id: actionIdCounter++,
      step: props.currentStep,
      actionName: ACTION_NAMES[newAction] || 'Unknown',
      affordance: newAction === 4 ? props.lastAffordance : null  // INTERACT action
    }

    recentActions.value.unshift(action)  // Add to front

    // Keep only last 10
    if (recentActions.value.length > 10) {
      recentActions.value.pop()
    }
  }
})

function getAffordanceIcon(type) {
  return AFFORDANCE_ICONS[type] || '?'
}

function formatMeterName(name) {
  // Convert "energy" → "Energy", "health" → "Health"
  return name.charAt(0).toUpperCase() + name.slice(1)
}

function getMeterClass(name, value) {
  // Color-code meters by value
  if (value < 0.2) return 'meter-critical'
  if (value < 0.5) return 'meter-warning'
  return 'meter-healthy'
}
</script>

<style scoped>
.aspatial-container {
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--spacing-lg);
  padding: var(--spacing-lg);
  width: 100%;
  height: 100%;
  overflow-y: auto;
}

/* Large meters panel */
.large-meters-panel {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-lg);
}

.large-meters-panel h2 {
  margin: 0 0 var(--spacing-md) 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

.large-meter {
  margin-bottom: var(--spacing-md);
}

.meter-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: var(--spacing-xs);
}

.meter-name {
  font-weight: var(--font-weight-medium);
  font-size: var(--font-size-base);
}

.meter-value {
  font-weight: var(--font-weight-semibold);
  font-size: var(--font-size-base);
}

.meter-bar-container {
  height: 24px;
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-sm);
  overflow: hidden;
}

.meter-bar-fill {
  height: 100%;
  transition: width var(--transition-base), background-color var(--transition-base);
}

.meter-healthy .meter-bar-fill {
  background: var(--color-success);
}

.meter-warning .meter-bar-fill {
  background: var(--color-warning);
}

.meter-critical .meter-bar-fill {
  background: var(--color-error);
}

/* Affordance list panel */
.affordance-list-panel {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-lg);
}

.affordance-list-panel h2 {
  margin: 0 0 var(--spacing-md) 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
}

.affordance-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: var(--spacing-md);
}

.affordance-card {
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-sm);
  padding: var(--spacing-md);
  text-align: center;
  transition: all var(--transition-base);
}

.affordance-card:hover {
  background: var(--color-interactive-disabled);
  transform: translateY(-2px);
}

.affordance-icon {
  font-size: 32px;
  margin-bottom: var(--spacing-sm);
}

.affordance-name {
  font-weight: var(--font-weight-medium);
  font-size: var(--font-size-sm);
  margin-bottom: var(--spacing-xs);
}

.affordance-status {
  font-size: var(--font-size-xs);
  color: var(--color-success);
  font-weight: var(--font-weight-medium);
}

/* Action history panel */
.action-history-panel {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-lg);
}

.action-history-panel h2 {
  margin: 0 0 var(--spacing-md) 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
}

.action-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  max-height: 300px;
  overflow-y: auto;
}

.action-item {
  display: flex;
  gap: var(--spacing-md);
  padding: var(--spacing-sm);
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-sm);
  font-size: var(--font-size-sm);
}

.action-step {
  color: var(--color-text-secondary);
  font-weight: var(--font-weight-medium);
  min-width: 60px;
}

.action-type {
  color: var(--color-text-primary);
  font-weight: var(--font-weight-semibold);
  min-width: 80px;
}

.action-affordance {
  color: var(--color-text-secondary);
}

/* Tablet+ layout */
@media (min-width: 768px) {
  .aspatial-container {
    grid-template-columns: 2fr 1fr;
    grid-template-rows: auto auto;
  }

  .large-meters-panel {
    grid-column: 1 / 2;
    grid-row: 1 / 3;
  }

  .affordance-list-panel {
    grid-column: 2 / 3;
    grid-row: 1 / 2;
  }

  .action-history-panel {
    grid-column: 2 / 3;
    grid-row: 2 / 3;
  }
}
</style>
