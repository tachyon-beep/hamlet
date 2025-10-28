<template>
  <!-- ✅ Semantic HTML: section instead of div -->
  <section class="meter-panel" aria-labelledby="meter-heading">
    <h3 id="meter-heading">Agent Meters</h3>

    <!-- ✅ Meters as semantic list -->
    <div v-if="meters" class="meters" role="list">
      <div
        v-for="(value, name) in meters"
        :key="name"
        class="meter"
        role="listitem"
        :class="{
          critical: isCritical(name, value),
          'strobe-slow': name === 'stress' && isLonely() && !isHighStress(),
          'strobe-fast': name === 'stress' && isLonely() && isHighStress()
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

// ✅ Use imported formatting utilities
// (capitalize, formatMeterValue, getMeterPercentage are imported above)

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
  // Check if social is at 0 (causes stress to increase)
  if (!meters.value) return false
  const social = meters.value.social
  return social <= 0.01
}

function isHighStress() {
  // Check if stress is dangerously high
  if (!meters.value) return false
  const stress = meters.value.stress
  return stress > 80
}

// ✅ Extract meter color logic using CSS variables
function getMeterColor(name, value) {
  const percentage = name === 'money' ? value : (name === 'stress' ? value : value * 100)

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
    stress: {
      low: 'var(--color-meter-stress-low)',
      mid: 'var(--color-meter-stress-mid)',
      high: 'var(--color-meter-stress-high)'
    },
    social: {
      high: 'var(--color-meter-social)',
      mid: 'var(--color-meter-social)',
      low: 'var(--color-error)'
    }
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
/* ✅ Refactored to use design tokens */
.meter-panel {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-lg);
}

.meter-panel h3 {
  margin: 0 0 var(--spacing-md) 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

.meters {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
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
</style>
