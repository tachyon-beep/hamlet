<template>
  <div class="reference-panel" :class="{ 'collapsed': isCollapsed }">
    <!-- Toggle Button -->
    <button
      class="toggle-btn"
      @click="toggleCollapse"
      :aria-expanded="!isCollapsed"
      aria-label="Toggle affordance reference"
    >
      <span class="toggle-icon">{{ isCollapsed ? '▲' : '▼' }}</span>
      <span class="toggle-label">{{ isCollapsed ? 'Show' : 'Hide' }} Affordance Reference</span>
    </button>

    <!-- Collapsible Content -->
    <transition name="slide-fade">
      <div v-show="!isCollapsed" class="reference-content">
        <div class="reference-grid">
          <!-- Compact Legend (Horizontally Laid Out) -->
          <div class="legend-section">
            <h3>Affordances Guide</h3>

            <!-- ✅ PEDAGOGY-FIRST: Lead with Strategic Insights -->
            <div class="strategic-insights-header">
              <h4>🎯 Key Strategic Insights</h4>
              <div class="insights-grid">
                <div class="insight-card">
                  <span class="insight-icon">⛓️</span>
                  <div class="insight-content">
                    <strong>Meter Hierarchy</strong>
                    <p>PRIMARY (death): Energy, Health. SECONDARY: Satiation → Energy + Health (critical!), Fitness → Health, Mood → Energy. TERTIARY: Hygiene, Social.</p>
                  </div>
                </div>
                <div class="insight-card">
                  <span class="insight-icon">💀</span>
                  <div class="insight-content">
                    <strong>Death Conditions</strong>
                    <p>Die if Energy ≤ 0 OR Health ≤ 0. Satiation is foundational—hunger makes you sick AND exhausted!</p>
                  </div>
                </div>
                <div class="insight-card">
                  <span class="insight-icon">⚠️</span>
                  <div class="insight-content">
                    <strong>Economic Traps</strong>
                    <p>FastFood damages fitness & health. Labor ($30) damages health. Bar costs $15 but drains energy & hygiene.</p>
                  </div>
                </div>
                <div class="insight-card">
                  <span class="insight-icon">🛡️</span>
                  <div class="insight-content">
                    <strong>Prevention Strategy</strong>
                    <p>Gym ($8) builds fitness → prevents health decline → saves Hospital ($15). Park is FREE multi-benefit!</p>
                  </div>
                </div>
              </div>
            </div>

            <!-- Divider -->
            <div class="section-divider">
              <span>Individual Affordances</span>
            </div>

            <div class="legend-categories-horizontal">
              <!-- Basic Survival -->
              <div class="category-column">
                <h4>Basic Survival</h4>
                <div class="affordances-compact">
                  <div class="affordance-compact">
                    <span class="icon">🛏️</span>
                    <span class="name">Bed</span>
                    <span class="effects">+50% energy, +2% health, -$5</span>
                  </div>
                  <div class="affordance-compact">
                    <span class="icon">🛌</span>
                    <span class="name">Luxury Bed</span>
                    <span class="effects">+75% energy, +5% health, -$11</span>
                  </div>
                  <div class="affordance-compact">
                    <span class="icon">🚿</span>
                    <span class="name">Shower</span>
                    <span class="effects">+40% hygiene, -$3</span>
                  </div>
                  <div class="affordance-compact">
                    <span class="icon">🥘</span>
                    <span class="name">HomeMeal</span>
                    <span class="effects">+45% satiation, +3% health, -$3</span>
                  </div>
                  <div class="affordance-compact">
                    <span class="icon">🍔</span>
                    <span class="name">FastFood</span>
                    <span class="effects">+45% satiation, +15% energy, -3% fitness, -2% health, -$10</span>
                  </div>
                </div>
              </div>

              <!-- Income -->
              <div class="category-column">
                <h4>Income Sources</h4>
                <div class="affordances-compact">
                  <div class="affordance-compact">
                    <span class="icon">💼</span>
                    <span class="name">Job</span>
                    <span class="effects">+$22.5, -15% energy, +2% social, -3% health</span>
                  </div>
                  <div class="affordance-compact">
                    <span class="icon">🔨</span>
                    <span class="name">Labor</span>
                    <span class="effects">+$30, -20% energy, -5% fitness, -5% health</span>
                  </div>
                </div>

                <h4 style="margin-top: 16px;">Fitness & Social</h4>
                <div class="affordances-compact">
                  <div class="affordance-compact">
                    <span class="icon">💪</span>
                    <span class="name">Gym</span>
                    <span class="effects">+30% fitness, -8% energy, -$8</span>
                  </div>
                  <div class="affordance-compact">
                    <span class="icon">🍺</span>
                    <span class="name">Bar</span>
                    <span class="effects">+50% social, +25% mood, +30% satiation, -20% energy, -15% hygiene, -5% health, -$15</span>
                  </div>
                  <div class="affordance-compact">
                    <span class="icon">🌳</span>
                    <span class="name">Park</span>
                    <span class="effects">+20% fitness, +15% social, +15% mood, -15% energy (FREE)</span>
                  </div>
                </div>
              </div>

              <!-- Mental Health & Physical Health -->
              <div class="category-column">
                <h4>Mental Health</h4>
                <div class="affordances-compact">
                  <div class="affordance-compact">
                    <span class="icon">🎮</span>
                    <span class="name">Recreation</span>
                    <span class="effects">+25% mood, +12% energy, -$6</span>
                  </div>
                  <div class="affordance-compact">
                    <span class="icon">🧠</span>
                    <span class="name">Therapist</span>
                    <span class="effects">+40% mood, -$15</span>
                  </div>
                </div>

                <h4 style="margin-top: 16px;">Physical Health</h4>
                <div class="affordances-compact">
                  <div class="affordance-compact">
                    <span class="icon">⚕️</span>
                    <span class="name">Doctor</span>
                    <span class="effects">+25% health, -$8</span>
                  </div>
                  <div class="affordance-compact">
                    <span class="icon">🏥</span>
                    <span class="name">Hospital</span>
                    <span class="effects">+40% health, -$15</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref } from 'vue'

// Start collapsed to maximize initial view of real-time data
const isCollapsed = ref(true)

function toggleCollapse() {
  isCollapsed.value = !isCollapsed.value
}
</script>

<style scoped>
/* Panel container - bottom of screen */
.reference-panel {
  position: relative;
  width: 100%;
  background: var(--color-bg-secondary);
  border-top: 1px solid var(--color-border);
  transition: all var(--transition-base);
}

.reference-panel.collapsed {
  /* Minimal height when collapsed */
  height: auto;
}

/* Toggle button - always visible */
.toggle-btn {
  width: 100%;
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-bg-tertiary);
  border: none;
  color: var(--color-text-secondary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-sm);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
  transition: all var(--transition-base);
}

.toggle-btn:hover {
  background: var(--color-bg-hover);
  color: var(--color-text-primary);
}

.toggle-btn:focus {
  outline: 2px solid var(--color-primary);
  outline-offset: -2px;
}

.toggle-icon {
  font-size: 0.875rem;
  transition: transform var(--transition-base);
}

/* Content area - slides in/out */
.reference-content {
  padding: var(--spacing-lg) var(--spacing-md);
  max-height: 50vh;
  overflow-y: auto;
  overflow-x: hidden;
}

/* Horizontal grid layout */
.reference-grid {
  width: 100%;
}

.legend-section h3 {
  margin: 0 0 var(--spacing-md) 0;
  font-size: var(--font-size-md);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-secondary); /* Subdued - this is reference data */
  text-align: center;
}

/* ✅ Strategic Insights Header - Pedagogy First */
.strategic-insights-header {
  margin-bottom: var(--spacing-xl);
}

.strategic-insights-header h4 {
  margin: 0 0 var(--spacing-md) 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-bold);
  color: var(--color-text-primary);
  text-align: center;
}

.insights-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: var(--spacing-md);
}

.insight-card {
  display: flex;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-md);
  border-left: 3px solid var(--color-primary);
  transition: all var(--transition-base);
}

.insight-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  border-left-color: var(--color-interactive-hover);
}

.insight-icon {
  font-size: 1.5rem;
  flex-shrink: 0;
  line-height: 1;
}

.insight-content {
  flex: 1;
}

.insight-content strong {
  display: block;
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
  margin-bottom: var(--spacing-xs);
}

.insight-content p {
  margin: 0;
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  line-height: 1.5;
}

/* Section Divider */
.section-divider {
  display: flex;
  align-items: center;
  text-align: center;
  margin: var(--spacing-xl) 0;
  color: var(--color-text-tertiary);
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-medium);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.section-divider::before,
.section-divider::after {
  content: '';
  flex: 1;
  border-bottom: 1px solid var(--color-border);
}

.section-divider span {
  padding: 0 var(--spacing-md);
}

/* Four-column layout */
.legend-categories-horizontal {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: var(--spacing-lg);
  margin-top: var(--spacing-md);
}

.category-column h4 {
  margin: 0 0 var(--spacing-xs) 0;
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-tertiary); /* Even more subdued */
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border-bottom: 1px solid var(--color-border);
  padding-bottom: 4px;
}

/* Compact affordance list */
.affordances-compact {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.affordance-compact {
  display: grid;
  grid-template-columns: 24px auto;
  grid-template-rows: auto auto;
  gap: 4px var(--spacing-xs);
  padding: var(--spacing-xs);
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
  font-size: var(--font-size-xs);
  line-height: 1.4;
}

.affordance-compact .icon {
  grid-row: 1 / 3;
  font-size: 1.25rem;
}

.affordance-compact .name {
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-secondary);
}

.affordance-compact .effects {
  grid-column: 2;
  color: var(--color-text-tertiary);
  font-size: 0.6875rem; /* 11px */
}

/* Slide-fade transition */
.slide-fade-enter-active,
.slide-fade-leave-active {
  transition: all 0.3s ease;
}

.slide-fade-enter-from,
.slide-fade-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

/* Responsive: Stack on narrow screens */
@media (max-width: 1024px) {
  .legend-categories-horizontal {
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  }
}

@media (max-width: 768px) {
  .legend-categories-horizontal {
    grid-template-columns: 1fr;
  }
}
</style>
