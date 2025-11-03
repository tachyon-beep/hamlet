<template>
  <div class="critical-event-log">
    <div class="log-header">
      <h4>Critical Events</h4>
    </div>

    <!-- Health status indicator (always visible, aligned with MeterPanel) -->
    <div
      class="health-status"
      :data-urgency="healthUrgency"
      :role="events.length > 0 ? 'alert' : 'status'"
    >
      <span class="status-icon">{{ healthIcon }}</span>
      <span class="status-text">
        <div class="status-title">{{ healthTitle }}</div>
        <div class="status-message">{{ healthMessage }}</div>
      </span>
    </div>

    <!-- Event log (below status indicator) -->
    <div v-if="sortedEvents.length > 0" class="event-list">
      <TransitionGroup name="event-slide">
        <div
          v-for="event in sortedEvents"
          :key="event.id"
          class="event-item"
          :class="`tier-${event.tier}`"
        >
          <div class="event-icon">
            <span v-if="event.tier === 'primary'">🔴</span>
            <span v-else-if="event.tier === 'secondary'">🟡</span>
            <span v-else>🟠</span>
          </div>
          <div class="event-content">
            <div class="event-header-row">
              <span class="event-meter">{{ event.meterName }}</span>
              <span class="event-value">{{ event.value }}%</span>
            </div>
            <div class="event-timestamp">{{ formatTimestamp(event.timestamp) }}</div>
            <div v-if="event.cascade" class="event-cascade">
              <span class="cascade-label">Cascade:</span>
              <span class="cascade-text">{{ event.cascade }}</span>
            </div>
          </div>
        </div>
      </TransitionGroup>
    </div>

    <!-- Clear button when there are events -->
    <button
      v-if="sortedEvents.length > 0"
      @click="$emit('clear')"
      class="clear-button"
      aria-label="Clear event log"
    >
      Clear Log
    </button>
  </div>
</template>

<script setup>
import { computed, shallowRef, watch } from 'vue'

const props = defineProps({
  events: {
    type: Array,
    default: () => []
  }
})

defineEmits(['clear'])

// Use shallowRef to avoid deep reactivity for sorted events
const sortedEvents = shallowRef([])

// Watch events and update sortedEvents only when needed
watch(
  () => props.events,
  (newEvents) => {
    // Sort and slice in one operation, store result
    const sorted = [...newEvents].sort((a, b) => b.timestamp - a.timestamp)
    sortedEvents.value = sorted.slice(0, 10) // Keep only last 10
  },
  { immediate: true, deep: true }
)

// Calculate health urgency level (matching MeterPanel logic)
const healthUrgency = computed(() => {
  const count = props.events.length
  if (count === 0) return 'none'
  if (count <= 2) return 'low'
  if (count <= 4) return 'medium'
  return 'high'
})

// Get icon based on urgency - memoized with v-memo in template would be better,
// but keep computed for simplicity
const healthIcon = computed(() => {
  return healthUrgency.value === 'none' ? '✅' : '⚠️'
})

// Get title based on urgency
const healthTitle = computed(() => {
  if (healthUrgency.value === 'none') return 'All meters healthy'
  const count = props.events.length
  return `${count} critical meter${count > 1 ? 's' : ''}`
})

// Get detailed message based on urgency
const healthMessage = computed(() => {
  const urgency = healthUrgency.value

  if (urgency === 'none') {
    return 'Agent maintaining homeostasis. No cascade risks detected.'
  } else if (urgency === 'low') {
    return 'Minor cascade risk. Monitor affected meters to prevent escalation.'
  } else if (urgency === 'medium') {
    return 'Moderate cascade detected. Multiple systems affected. Intervention advised.'
  } else { // high
    return 'Severe cascade in progress. Death risk imminent without immediate action.'
  }
})

function formatTimestamp(timestamp) {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  })
}
</script>

<style scoped>
.critical-event-log {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  flex: 1;
  min-height: 200px;
  overflow: hidden;
}

.log-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--spacing-sm);
}

.log-header h4 {
  margin: 0;
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

/* Health status indicator (aligned with MeterPanel critical-alert style) */
.health-status {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  background: var(--color-success);
  color: white;
  border-radius: var(--border-radius-md);
  margin-bottom: var(--spacing-md);
  font-weight: var(--font-weight-semibold);
  transition: background var(--transition-base), color var(--transition-base);
  min-height: 90px;

  /* GPU acceleration - use transform for animations */
  will-change: transform;
  transform: translateZ(0);
  backface-visibility: hidden;
}

/* Escalating urgency levels (matching MeterPanel) */
.health-status[data-urgency="low"] {
  animation: alert-pulse-gentle 2s ease-in-out infinite;
  background: var(--color-warning);
}

.health-status[data-urgency="medium"] {
  animation: alert-pulse-moderate 1.5s ease-in-out infinite;
  background: linear-gradient(135deg, var(--color-warning), var(--color-error));
}

.health-status[data-urgency="high"] {
  animation: alert-pulse-urgent 1s ease-in-out infinite;
  background: var(--color-error);
  box-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
}

@keyframes alert-pulse-gentle {
  0%, 100% { transform: scale3d(1, 1, 1); }
  50% { transform: scale3d(1.01, 1.01, 1); }
}

@keyframes alert-pulse-moderate {
  0%, 100% { transform: scale3d(1, 1, 1); }
  50% { transform: scale3d(1.015, 1.015, 1); }
}

@keyframes alert-pulse-urgent {
  0%, 100% { transform: scale3d(1, 1, 1) rotate(0deg); }
  25% { transform: scale3d(1.02, 1.02, 1) rotate(-0.5deg); }
  75% { transform: scale3d(1.02, 1.02, 1) rotate(0.5deg); }
}

.status-icon {
  font-size: var(--font-size-xl);
  animation: shake 0.5s ease-in-out infinite;

  /* GPU acceleration for icon animation */
  will-change: transform;
  transform: translateZ(0);
}

/* Don't shake when healthy */
.health-status[data-urgency="none"] .status-icon {
  animation: none;
}

@keyframes shake {
  0%, 100% { transform: translate3d(0, 0, 0); }
  25% { transform: translate3d(-4px, 0, 0); }
  75% { transform: translate3d(4px, 0, 0); }
}

.status-text {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.status-title {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-bold);
  color: white;
}

.status-message {
  font-size: var(--font-size-xs);
  color: rgba(255, 255, 255, 0.9);
  line-height: 1.4;
}

.event-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  padding-right: var(--spacing-xs);
}

/* Custom scrollbar */
.event-list::-webkit-scrollbar {
  width: 6px;
}

.event-list::-webkit-scrollbar-track {
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-sm);
}

.event-list::-webkit-scrollbar-thumb {
  background: var(--color-interactive-disabled);
  border-radius: var(--border-radius-sm);
}

.event-list::-webkit-scrollbar-thumb:hover {
  background: var(--color-text-tertiary);
}

.event-item {
  display: flex;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
  border-left: 3px solid transparent;
  transition: all var(--transition-base);

  /* GPU acceleration for smooth list updates */
  will-change: transform, opacity;
  transform: translateZ(0);
  backface-visibility: hidden;
}

.event-item.tier-primary {
  border-left-color: var(--color-error);
  background: rgba(239, 68, 68, 0.05);
}

.event-item.tier-secondary {
  border-left-color: var(--color-warning);
  background: rgba(245, 158, 11, 0.05);
}

.event-item.tier-tertiary {
  border-left-color: #f97316; /* Orange for tertiary */
  background: rgba(249, 115, 22, 0.05);
}

.event-icon {
  font-size: 1.25rem;
  flex-shrink: 0;
  line-height: 1;
}

.event-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.event-header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.event-meter {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
  text-transform: capitalize;
}

.event-value {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  font-family: 'Monaco', 'Courier New', monospace;
  color: var(--color-error);
}

.event-timestamp {
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
  font-family: 'Monaco', 'Courier New', monospace;
}

.event-cascade {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  margin-top: 2px;
  padding-top: 4px;
  border-top: 1px solid var(--color-border);
}

.cascade-label {
  font-weight: var(--font-weight-semibold);
  color: var(--color-warning);
  margin-right: var(--spacing-xs);
}

.cascade-text {
  font-style: italic;
}

.clear-button {
  width: 100%;
  padding: var(--spacing-xs) var(--spacing-sm);
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius-sm);
  color: var(--color-text-secondary);
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-medium);
  cursor: pointer;
  transition: all var(--transition-base);
}

.clear-button:hover {
  background: var(--color-bg-hover);
  color: var(--color-text-primary);
  border-color: var(--color-text-tertiary);
}

/* Transition animations - GPU-accelerated with transform only */
.event-slide-enter-active {
  transition: transform 0.3s ease, opacity 0.3s ease;
}

.event-slide-leave-active {
  transition: transform 0.2s ease, opacity 0.2s ease;
}

.event-slide-enter-from {
  opacity: 0;
  transform: translate3d(-20px, 0, 0);
}

.event-slide-leave-to {
  opacity: 0;
  transform: translate3d(20px, 0, 0);
}

.event-slide-move {
  transition: transform 0.3s ease;
}
</style>
