<template>
  <section class="legend-panel" aria-labelledby="legend-heading">
    <h3 id="legend-heading">Affordances Guide</h3>

    <div class="legend-categories">
      <!-- Basic Survival -->
      <div class="category">
        <h4>Basic Survival</h4>
        <div class="affordances">
          <div class="affordance-item">
            <span class="icon">🛏️</span>
            <div class="details">
              <strong>Bed</strong>
              <span class="effects">+60% energy, -$5</span>
            </div>
          </div>
          <div class="affordance-item">
            <span class="icon">🚿</span>
            <div class="details">
              <strong>Shower</strong>
              <span class="effects">+50% hygiene, -$2</span>
            </div>
          </div>
          <div class="affordance-item">
            <span class="icon">🥘</span>
            <div class="details">
              <strong>HomeMeal</strong>
              <span class="effects">+40% satiation, +3% health, -$3</span>
            </div>
          </div>
          <div class="affordance-item">
            <span class="icon">🍔</span>
            <div class="details">
              <strong>FastFood</strong>
              <span class="effects">+45% satiation, -3% fitness, -2% health, -$10</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Income -->
      <div class="category">
        <h4>Income Sources</h4>
        <div class="affordances">
          <div class="affordance-item">
            <span class="icon">💼</span>
            <div class="details">
              <strong>Job</strong>
              <span class="effects">+$22.5, -15% energy, +2% social, -3% health</span>
            </div>
          </div>
          <div class="affordance-item">
            <span class="icon">🔨</span>
            <div class="details">
              <strong>Labor</strong>
              <span class="effects">+$30, -20% energy, -5% fitness, -5% health</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Secondary Meters -->
      <div class="category">
        <h4>Fitness & Social</h4>
        <div class="affordances">
          <div class="affordance-item">
            <span class="icon">💪</span>
            <div class="details">
              <strong>Gym</strong>
              <span class="effects">+30% fitness, -8% energy, -$8</span>
            </div>
          </div>
          <div class="affordance-item">
            <span class="icon">🍺</span>
            <div class="details">
              <strong>Bar</strong>
              <span class="effects">+50% social, +25% mood, -5% health, -$15</span>
            </div>
          </div>
          <div class="affordance-item">
            <span class="icon">🌳</span>
            <div class="details">
              <strong>Park</strong>
              <span class="effects">+20% fitness, +15% social, +15% mood, -15% energy (FREE)</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Mood Care -->
      <div class="category">
        <h4>Mental Health</h4>
        <div class="affordances">
          <div class="affordance-item">
            <span class="icon">🎮</span>
            <div class="details">
              <strong>Recreation</strong>
              <span class="effects">+25% mood, +12% energy, -$6</span>
            </div>
          </div>
          <div class="affordance-item">
            <span class="icon">🧠</span>
            <div class="details">
              <strong>Therapist</strong>
              <span class="effects">+40% mood, -$15</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Physical Health -->
      <div class="category">
        <h4>Physical Health</h4>
        <div class="affordances">
          <div class="affordance-item">
            <span class="icon">⚕️</span>
            <div class="details">
              <strong>Doctor</strong>
              <span class="effects">+25% health, -$8</span>
            </div>
          </div>
          <div class="affordance-item">
            <span class="icon">🏥</span>
            <div class="details">
              <strong>Hospital</strong>
              <span class="effects">+40% health, -$15</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Strategy Notes -->
    <div class="strategy-notes">
      <h4>Strategy Tips</h4>
      <ul>
        <li><strong>Primary Meters:</strong> Mood & Health need direct restoration</li>
        <li><strong>Secondary Meters:</strong> Low Social → Mood penalty, Low Fitness → Health decline 3x faster</li>
        <li><strong>Prevention vs Cure:</strong> Gym ($8) + Park (free) prevent expensive Hospital visits ($15)</li>
        <li><strong>Vice Tradeoffs:</strong> Bar & FastFood solve immediate needs but damage long-term health</li>
      </ul>
    </div>
  </section>
</template>

<script setup>
// No props needed - static legend
</script>

<style scoped>
.legend-panel {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-lg);
  max-height: 80vh;
  overflow-y: auto;
}

.legend-panel h3 {
  margin: 0 0 var(--spacing-md) 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

.legend-categories {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
}

.category {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.category h4 {
  margin: 0;
  font-size: var(--font-size-md);
  font-weight: var(--font-weight-medium);
  color: var(--color-text-secondary);
  border-bottom: 1px solid var(--color-border);
  padding-bottom: var(--spacing-xs);
}

.affordances {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.affordance-item {
  display: flex;
  align-items: flex-start;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
  transition: background var(--transition-base);
}

.affordance-item:hover {
  background: var(--color-bg-tertiary);
}

.icon {
  font-size: 1.5rem;
  flex-shrink: 0;
}

.details {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex: 1;
}

.details strong {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

.effects {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  line-height: 1.4;
}

.strategy-notes {
  margin-top: var(--spacing-lg);
  padding: var(--spacing-md);
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
  border-left: 3px solid var(--color-primary);
}

.strategy-notes h4 {
  margin: 0 0 var(--spacing-sm) 0;
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  color: var(--color-primary);
}

.strategy-notes ul {
  margin: 0;
  padding-left: var(--spacing-lg);
  list-style-type: disc;
}

.strategy-notes li {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  line-height: 1.6;
  margin-bottom: var(--spacing-xs);
}

.strategy-notes strong {
  color: var(--color-text-primary);
}
</style>
