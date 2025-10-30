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
                    <span class="effects">+50% hygiene, -$2</span>
                  </div>
                  <div class="affordance-compact">
                    <span class="icon">🥘</span>
                    <span class="name">HomeMeal</span>
                    <span class="effects">+40% satiation, +3% health, -$3</span>
                  </div>
                  <div class="affordance-compact">
                    <span class="icon">🍔</span>
                    <span class="name">FastFood</span>
                    <span class="effects">+45% satiation, -3% fitness, -$10</span>
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
                    <span class="effects">+$22.5, -15% energy, +2% social</span>
                  </div>
                  <div class="affordance-compact">
                    <span class="icon">🔨</span>
                    <span class="name">Labor</span>
                    <span class="effects">+$30, -20% energy, -5% fitness</span>
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
                    <span class="effects">+50% social, +25% mood, -$15</span>
                  </div>
                  <div class="affordance-compact">
                    <span class="icon">🌳</span>
                    <span class="name">Park</span>
                    <span class="effects">+20% fitness, +15% social (FREE)</span>
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

              <!-- Strategy Notes (Right Column) -->
              <div class="category-column strategy-column">
                <h4>Strategic Insights</h4>
                <div class="strategy-compact">
                  <p><strong>Meter Coupling:</strong> Satiation affects Health & Energy. Tertiary meters (hygiene/social) accelerate all pathways.</p>
                  <p><strong>Cascading Death:</strong> Neglect tertiary → secondary tanks → primary fails → death.</p>
                  <p><strong>Traps:</strong> FastFood damages fitness pathway. Labor high pay but damages health long-term.</p>
                  <p><strong>Prevention:</strong> Gym ($8) prevents Hospital ($15) by maintaining fitness buffer.</p>
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

/* Strategy notes column */
.strategy-column {
  background: var(--color-bg-primary);
  padding: var(--spacing-sm);
  border-radius: var(--border-radius-sm);
  border-left: 2px solid var(--color-primary);
}

.strategy-compact {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.strategy-compact p {
  margin: 0;
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
  line-height: 1.5;
}

.strategy-compact strong {
  color: var(--color-text-secondary);
  font-weight: var(--font-weight-semibold);
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
