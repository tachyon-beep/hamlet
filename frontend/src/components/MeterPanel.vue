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
      :data-urgency="alertUrgency"
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

    <!-- ✅ Grouped meter tiers showing cascading relationships -->
    <div v-if="meters" class="meters" role="list">
      <!-- Tier 1: Primary (Survival-Critical) -->
      <div class="meter-tier" data-tier-level="primary">
        <div class="tier-header">
          <span class="tier-icon">🎯</span>
          <h4 class="tier-name">SURVIVAL-CRITICAL</h4>
        </div>
        <div class="tier-meters">
          <template v-for="name in primaryMeters" :key="name">
          <div
            v-if="meters && meters[name] !== undefined"
            class="meter"
            role="listitem"
            :data-meter="name"
            :class="{
              critical: isCritical(name, meters[name])
            }"
            :aria-label="`${capitalize(name)}: ${formatMeterValue(name, meters[name])}`"
          >
            <div class="meter-header">
              <span class="meter-name">{{ capitalize(name) }}</span>
              <span class="meter-value" aria-live="polite" aria-atomic="true" role="status">
                {{ formatMeterValue(name, meters[name]) }}
              </span>
            </div>
            <div class="meter-bar-container">
              <div
                class="meter-bar"
                role="progressbar"
                :aria-valuenow="getMeterPercentage(name, meters[name])"
                aria-valuemin="0"
                aria-valuemax="100"
                :style="{
                  width: getMeterPercentage(name, meters[name]) + '%',
                  background: getMeterColor(name, meters[name])
                }"
              ></div>
            </div>
          </div>
          </template>
        </div>
      </div>

      <!-- Tier 2: Secondary (Modifiers) -->
      <div class="meter-tier" data-tier-level="secondary">
        <div class="tier-header">
          <span class="tier-icon">🔄</span>
          <h4 class="tier-name">MODIFIERS</h4>
          <span class="tier-description">affect primary</span>
        </div>
        <div class="tier-meters">
          <template v-for="name in secondaryMeters" :key="name">
          <div
            v-if="meters && meters[name] !== undefined"
            class="meter"
            role="listitem"
            :data-meter="name"
            :data-affects="getMeterAffects(name)"
            :class="{
              critical: isCritical(name, meters[name]),
              'strobe-slow': name === 'mood' && isLonely() && !isMoodCritical(),
              'strobe-fast': name === 'mood' && isLonely() && isMoodCritical()
            }"
            :aria-label="`${capitalize(name)}: ${formatMeterValue(name, meters[name])}`"
          >
            <div class="meter-header">
              <span class="meter-name">
                {{ capitalize(name) }}
              </span>
              <span class="meter-value" aria-live="polite" aria-atomic="true" role="status">
                {{ formatMeterValue(name, meters[name]) }}
              </span>
            </div>
            <div class="meter-relationship">
              → {{ getRelationshipText(name) }}
            </div>
            <div class="meter-bar-container">
              <div
                class="meter-bar"
                role="progressbar"
                :aria-valuenow="getMeterPercentage(name, meters[name])"
                aria-valuemin="0"
                aria-valuemax="100"
                :style="{
                  width: getMeterPercentage(name, meters[name]) + '%',
                  background: getMeterColor(name, meters[name])
                }"
              ></div>
            </div>
          </div>
          </template>
        </div>
      </div>

      <!-- Tier 3: Tertiary (Accelerators) -->
      <div class="meter-tier" data-tier-level="tertiary">
        <div class="tier-header">
          <span class="tier-icon">⚡</span>
          <h4 class="tier-name">ACCELERATORS</h4>
          <span class="tier-description">speed up effects</span>
        </div>
        <div class="tier-meters">
          <template v-for="name in tertiaryMeters" :key="name">
          <div
            v-if="meters && meters[name] !== undefined"
            class="meter"
            role="listitem"
            :data-meter="name"
            :data-affects="getMeterAffects(name)"
            :class="{
              critical: isCritical(name, meters[name])
            }"
            :aria-label="`${capitalize(name)}: ${formatMeterValue(name, meters[name])}`"
          >
            <div class="meter-header">
              <span class="meter-name">
                {{ capitalize(name) }}
              </span>
              <span class="meter-value" aria-live="polite" aria-atomic="true" role="status">
                {{ formatMeterValue(name, meters[name]) }}
              </span>
            </div>
            <div class="meter-relationship accelerator">
              ⚡ {{ getRelationshipText(name) }}
            </div>
            <div class="meter-bar-container">
              <div
                class="meter-bar"
                role="progressbar"
                :aria-valuenow="getMeterPercentage(name, meters[name])"
                aria-valuemin="0"
                aria-valuemax="100"
                :style="{
                  width: getMeterPercentage(name, meters[name]) + '%',
                  background: getMeterColor(name, meters[name])
                }"
              ></div>
            </div>
          </div>
          </template>
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
import { computed, ref } from 'vue'
import EmptyState from './EmptyState.vue'
import { capitalize, formatMeterValue, getMeterPercentage } from '../utils/formatting'

// ✅ Props First: Receive data from parent instead of importing store
const props = defineProps({
  agentMeters: {
    type: Object,
    default: () => ({})
  }
})

// Meter tier arrays (matching actual game code, secondary in alphabetical order)
const primaryMeters = ['energy', 'health', 'money']
const secondaryMeters = ['fitness', 'mood', 'satiation']
const tertiaryMeters = ['hygiene', 'social']

// Relationship map: meter → what it affects (matching actual game code)
const meterRelationships = {
  // Tertiary → Secondary
  hygiene: ['satiation', 'fitness', 'mood'],
  social: ['mood'],
  // Secondary → Primary
  mood: ['energy'],
  satiation: ['energy', 'health'],
  fitness: ['health']
}

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

// Calculate alert urgency level
const alertUrgency = computed(() => {
  const count = criticalMetersCount.value
  if (count === 0) return 'none'
  if (count <= 2) return 'low'
  if (count <= 4) return 'medium'
  return 'high'
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

function getMeterTier(name) {
  // Updated classification based on actual game mechanics:
  // Primary: Direct survival-critical (death risk if <20%)
  // Secondary: Direct modifiers of primary meters
  // Tertiary: Accelerators/multipliers
  const primary = ['energy', 'health', 'money']
  const secondary = ['mood', 'satiation', 'hygiene']
  // tertiary: fitness, social

  if (primary.includes(name)) return 'primary'
  if (secondary.includes(name)) return 'secondary'
  return 'tertiary'
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

// Get what meters this meter affects (for data-affects attribute)
function getMeterAffects(name) {
  return (meterRelationships[name] || []).join(',')
}

// Get compact relationship text (always visible)
function getRelationshipText(name) {
  const affects = meterRelationships[name] || []
  if (affects.length === 0) return ''

  // Capitalize and join with "+"
  return affects.map(m => capitalize(m)).join(' + ')
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
  gap: 0;
  flex: 1;
  overflow-y: auto;
}

.meter {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  position: relative;
  transition: all var(--transition-base);
}

/* ===== Tier Container Styling ===== */
.meter-tier {
  margin-bottom: var(--spacing-sm);
  border-left: 4px solid var(--tier-color);
  padding-left: var(--spacing-md);
  transition: all var(--transition-base);
}

.meter-tier[data-tier-level="primary"] {
  --tier-color: #fbbf24; /* Gold */
}

.meter-tier[data-tier-level="secondary"] {
  --tier-color: #06b6d4; /* Cyan */
  margin-top: var(--spacing-md);
}

.meter-tier[data-tier-level="tertiary"] {
  --tier-color: #a78bfa; /* Purple */
  margin-top: var(--spacing-md);
}

/* Tier Header Styling */
.tier-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  margin-bottom: var(--spacing-sm);
}

.tier-icon {
  font-size: var(--font-size-base);
}

.tier-name {
  margin: 0;
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-bold);
  color: var(--tier-color);
  letter-spacing: 0.05em;
}

.tier-description {
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
  font-style: italic;
  margin-left: auto;
}

/* Primary tier gets more visual weight */
.meter-tier[data-tier-level="primary"] .tier-name {
  font-size: var(--font-size-base);
}

/* Tier Meters Container */
.tier-meters {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

/* ===== Meter Sizing by Tier ===== */
.meter-tier[data-tier-level="primary"] .meter {
  padding: var(--spacing-md);
  background: rgba(255, 255, 255, 0.02);
  border-radius: var(--border-radius-sm);
}

.meter-tier[data-tier-level="primary"] .meter-bar-container {
  height: 22px;
}

.meter-tier[data-tier-level="secondary"] .meter-bar-container {
  height: 22px;
}

.meter-tier[data-tier-level="tertiary"] .meter-bar-container {
  height: 22px;
}

/* ===== Relationship Text (Always Visible) ===== */
.meter-relationship {
  font-size: var(--font-size-xs);
  color: var(--tier-color);
  margin-top: var(--spacing-xs);
  padding: calc(var(--spacing-xs) / 2) var(--spacing-xs);
  background: rgba(255, 255, 255, 0.03);
  border-radius: var(--border-radius-sm);
  border-left: 2px solid var(--tier-color);
  font-weight: var(--font-weight-medium);
  opacity: 0.85;
  line-height: 1.2;
}

.meter-relationship.accelerator {
  color: #a78bfa;
  border-left-color: #a78bfa;
}

.meter.critical {
  animation: pulse 1s ease-in-out infinite;
  will-change: transform, opacity;
  transform: translateZ(0);
}

.meter.strobe-slow {
  animation: strobe-slow 3s ease-in-out infinite !important;
  will-change: transform, opacity;
  transform: translateZ(0);
}

.meter.strobe-fast {
  animation: strobe-fast 1s ease-in-out infinite !important;
  will-change: transform, opacity;
  transform: translateZ(0);
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
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-bold);
  color: var(--color-text-primary);
  font-family: 'Monaco', 'Courier New', monospace;
  font-variant-numeric: tabular-nums;
  min-width: 3ch;
  text-align: right;
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
  will-change: transform, opacity;
  transform: translateZ(0);
}

/* Green state when all meters are healthy */
.critical-alert.all-healthy {
  background: var(--color-success);
  animation: none;
}

/* Escalating urgency levels */
.critical-alert[data-urgency="low"] {
  animation: alert-pulse-gentle 2s ease-in-out infinite;
  background: var(--color-warning);
}

.critical-alert[data-urgency="medium"] {
  animation: alert-pulse-moderate 1.5s ease-in-out infinite;
  background: linear-gradient(135deg, var(--color-warning), var(--color-error));
}

.critical-alert[data-urgency="high"] {
  animation: alert-pulse-urgent 1s ease-in-out infinite;
  background: var(--color-error);
  box-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
}

@keyframes alert-pulse-gentle {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.01); opacity: 0.95; }
}

@keyframes alert-pulse-moderate {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.015); }
}

@keyframes alert-pulse-urgent {
  0%, 100% { transform: scale(1); }
  25% { transform: scale(1.02) rotate(-0.5deg); }
  75% { transform: scale(1.02) rotate(0.5deg); }
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

/* Respect user's reduced motion preference */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
</style>
