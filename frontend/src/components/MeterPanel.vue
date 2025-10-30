<template>
  <!-- ✅ Semantic HTML: section instead of div -->
  <section class="meter-panel" aria-labelledby="meter-heading">
    <div class="panel-header">
      <h3 id="meter-heading">Agent Meters</h3>
    </div>

    <!-- ✅ Critical state alert banner (always visible) -->
    <div
      v-if="meters"
      class="critical-alert"
      :class="{ 'all-healthy': criticalMetersCount === 0 }"
      :role="criticalMetersCount > 0 ? 'alert' : 'status'"
    >
      <span class="alert-icon">{{ criticalMetersCount > 0 ? '⚠️' : '✅' }}</span>
      <span class="alert-text">
        <template v-if="criticalMetersCount > 0">
          {{ criticalMetersCount }} critical meter{{ criticalMetersCount > 1 ? 's' : '' }}
        </template>
        <template v-else>
          All meters healthy
        </template>
      </span>
    </div>

    <!-- ✅ Meters as semantic list -->
    <div v-if="meters" class="meters" role="list">
      <div
        v-for="(value, name) in meters"
        :key="name"
        class="meter"
        role="listitem"
        :class="{
          critical: isCritical(name, value),
          'strobe-slow': name === 'mood' && isLonely() && !isMoodCritical(),
          'strobe-fast': name === 'mood' && isLonely() && isMoodCritical()
        }"
        :aria-label="`${capitalize(name)}: ${formatMeterValue(name, value)}`"
      >
        <div class="meter-header">
          <span class="meter-name">{{ capitalize(name) }}</span>
          <!-- ✅ ARIA live region for real-time meter updates -->
          <span
            class="meter-value"
            aria-live="polite"
            aria-atomic="true"
            role="status"
          >
            {{ formatMeterValue(name, value) }}
          </span>
        </div>
        <!-- ✅ Meter bar as progressbar with ARIA attributes -->
        <div class="meter-bar-container">
          <div
            class="meter-bar"
            role="progressbar"
            :aria-valuenow="getMeterPercentage(name, value)"
            aria-valuemin="0"
            aria-valuemax="100"
            :aria-label="`${capitalize(name)} level: ${getMeterPercentage(name, value).toFixed(0)}%`"
            :style="{
              width: getMeterPercentage(name, value) + '%',
              background: getMeterColor(name, value)
            }"
          ></div>
        </div>
      </div>
    </div>

    <!-- ✅ Empty state when no meter data available -->
    <EmptyState
      v-else
      icon="📊"
      title="No Agent Data"
      message="Connect to the simulation to see agent meters."
    />

    <!-- Permanent explanation text -->
    <div v-if="meters" class="panel-explanation">
      <p>Meters represent the agent's needs (0-100%). Critical (<20%) triggers death risk. Primary meters (energy, health) are survival-critical. Secondary meters (satiation, money) feed into primary. Tertiary meters (hygiene, social, fitness, mood) accelerate all pathways.</p>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import EmptyState from './EmptyState.vue'
import { capitalize, formatMeterValue, getMeterPercentage } from '../utils/formatting'

// ✅ Props First: Receive data from parent instead of importing store
const props = defineProps({
  agentMeters: {
    type: Object,
    default: () => ({})
  }
})

const meters = computed(() => {
  const agent = props.agentMeters['agent_0']
  return agent ? agent.meters : null
})

// Count critical meters for alert banner
const criticalMetersCount = computed(() => {
  if (!meters.value) return 0
  return Object.entries(meters.value).filter(([name, value]) =>
    isCritical(name, value)
  ).length
})

// ✅ Use imported formatting utilities
// (capitalize, formatMeterValue, getMeterPercentage are imported above)

function isCritical(name, value) {
  // All meters are 0-1 normalized, convert to percentage
  const percentage = value * 100
  // Low mood or other meters trigger critical state when percentage < 20
  return percentage < 20
}

function isLonely() {
  // Check if social is at 0 (causes mood to drop rapidly)
  if (!meters.value) return false
  const social = meters.value.social
  return social <= 0.01
}

function isMoodCritical() {
  // Check if mood is dangerously low
  if (!meters.value) return false
  const mood = meters.value.mood
  return mood < 20
}

// ✅ Extract meter color logic using CSS variables
function getMeterColor(name, value) {
  // All meters are 0-1 normalized, convert to percentage
  const percentage = value * 100

  // Color mapping using CSS custom properties
  const colors = {
    energy: {
      high: 'var(--color-meter-energy)',
      mid: 'var(--color-warning)',
      low: 'var(--color-error)'
    },
    hygiene: {
      high: 'var(--color-meter-hygiene)',
      mid: 'var(--color-warning)',
      low: 'var(--color-error)'
    },
    satiation: {
      high: 'var(--color-meter-satiation)',
      mid: 'var(--color-warning)',
      low: 'var(--color-error)'
    },
    money: {
      high: 'var(--color-meter-money)',
      mid: 'var(--color-meter-money)',
      low: 'var(--color-error)'
    },
    health: {
      high: '#10b981',  // Green - healthy
      mid: 'var(--color-warning)',
      low: 'var(--color-error)'
    },
    mood: {
      high: 'var(--color-meter-mood-high)',
      mid: 'var(--color-meter-mood-mid)',
      low: 'var(--color-meter-mood-low)'
    },
    social: {
      high: 'var(--color-meter-social)',
      mid: 'var(--color-meter-social)',
      low: 'var(--color-error)'
    },
    fitness: {
      high: '#8b5cf6',  // Purple - athletic
      mid: 'var(--color-warning)',
      low: 'var(--color-error)'
    }
  }

  const colorSet = colors[name] || colors.energy

  // Normal meters - HIGH is good
  if (percentage > 60) return colorSet.high
  if (percentage > 30) return colorSet.mid
  return colorSet.low
}
</script>

<style scoped>
/* ✅ Refactored to use design tokens */
.meter-panel {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-lg);
  display: flex;
  flex-direction: column;
  height: 100%;
}

.panel-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-md);
}

.meter-panel h3 {
  margin: 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

.meters {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  flex: 1;
  overflow-y: auto;
}

.meter {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.meter.critical {
  animation: pulse 1s ease-in-out infinite;
}

.meter.strobe-slow {
  animation: strobe-slow 2s ease-in-out infinite !important;
}

.meter.strobe-fast {
  animation: strobe-fast 0.6s ease-in-out infinite !important;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.6;
  }
}

@keyframes strobe-slow {
  0%, 100% {
    opacity: 1;
  }
  25% {
    opacity: 0.3;
  }
  50% {
    opacity: 1;
  }
  75% {
    opacity: 0.3;
  }
}

@keyframes strobe-fast {
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
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
  color: var(--color-text-secondary);
}

.meter-value {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

.meter-bar-container {
  width: 100%;
  height: 20px;
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-full);
  overflow: hidden;
}

.meter-bar {
  height: 100%;
  border-radius: var(--border-radius-full);
  transition: width var(--transition-base), background var(--transition-base);
}

/* ✅ Critical alert banner (always visible) */
.critical-alert {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  background: var(--color-error);
  color: white;
  border-radius: var(--border-radius-md);
  margin-bottom: var(--spacing-md);
  font-weight: var(--font-weight-semibold);
  animation: alert-pulse 1s ease-in-out infinite;
  transition: background var(--transition-base), color var(--transition-base);
}

/* Green state when all meters are healthy */
.critical-alert.all-healthy {
  background: var(--color-success);
  animation: none;
}

@keyframes alert-pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.02); }
}

.alert-icon {
  font-size: var(--font-size-xl);
  animation: shake 0.5s ease-in-out infinite;
}

/* Don't shake when healthy */
.critical-alert.all-healthy .alert-icon {
  animation: none;
}

@keyframes shake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-4px); }
  75% { transform: translateX(4px); }
}

.alert-text {
  flex: 1;
  font-size: var(--font-size-sm);
}

/* Panel explanation */
.panel-explanation {
  margin-top: auto;
  padding: var(--spacing-sm);
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
  border-top: 1px solid var(--color-border);
}

.panel-explanation p {
  margin: 0;
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
  font-style: italic;
  line-height: 1.4;
  opacity: 0.9;
}
</style>
