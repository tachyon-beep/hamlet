<template>
  <div class="critical-event-log">
    <div class="log-header">
      <h4>Critical Events</h4>
      <InfoTooltip
        title="What are critical events?"
        text="Logs when meters drop below 20% (critical threshold). Shows cascading failures: tertiary meters (hygiene, social) → secondary (satiation, money) → primary (energy, health) → death."
        position="left"
      />
    </div>

    <!-- All good state (green) -->
    <div v-if="events.length === 0" class="all-good-state">
      <div class="all-good-icon">✨</div>
      <div class="all-good-content">
        <div class="all-good-title">All Meters Healthy</div>
        <div class="all-good-message">
          Agent maintaining homeostasis. No cascade risks detected.
        </div>
      </div>
    </div>

    <!-- Event log -->
    <div v-else class="event-list">
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
      v-if="events.length > 0"
      @click="$emit('clear')"
      class="clear-button"
      aria-label="Clear event log"
    >
      Clear Log
    </button>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import InfoTooltip from './InfoTooltip.vue'

const props = defineProps({
  events: {
    type: Array,
    default: () => []
  }
})

defineEmits(['clear'])

// Sort events by timestamp (most recent first)
const sortedEvents = computed(() => {
  return [...props.events].sort((a, b) => b.timestamp - a.timestamp).slice(0, 10) // Keep only last 10
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

/* All good state (green, always visible) */
.all-good-state {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-lg);
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(34, 197, 94, 0.1));
  border: 2px solid rgba(16, 185, 129, 0.3);
  border-radius: var(--border-radius-md);
  min-height: 80px;
}

.all-good-icon {
  font-size: 2rem;
  flex-shrink: 0;
  animation: gentle-pulse 3s ease-in-out infinite;
}

@keyframes gentle-pulse {
  0%, 100% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.1);
    opacity: 0.8;
  }
}

.all-good-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.all-good-title {
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-semibold);
  color: var(--color-success);
}

.all-good-message {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
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

/* Transition animations */
.event-slide-enter-active {
  transition: all 0.3s ease;
}

.event-slide-leave-active {
  transition: all 0.2s ease;
}

.event-slide-enter-from {
  opacity: 0;
  transform: translateX(-20px);
}

.event-slide-leave-to {
  opacity: 0;
  transform: translateX(20px);
}

.event-slide-move {
  transition: transform 0.3s ease;
}
</style>
