<template>
  <div class="death-certificates">
    <div class="log-header">
      <h4>💀 Death Certificates</h4>
      <InfoTooltip
        title="What are death certificates?"
        text="Records of agent deaths showing final meter states and interaction patterns. Top 3 longest survival times plus most recent death."
        position="left"
      />
    </div>

    <!-- No deaths yet state -->
    <div v-if="certificates.length === 0" class="no-deaths-state">
      <div class="no-deaths-icon">🌱</div>
      <div class="no-deaths-content">
        <div class="no-deaths-title">No Deaths Yet</div>
        <div class="no-deaths-message">
          Agent is alive. Death certificates will appear here.
        </div>
      </div>
    </div>

    <!-- Death certificates list -->
    <div v-else class="certificate-list">
      <TransitionGroup name="cert-slide">
        <div
          v-for="cert in displayedCertificates"
          :key="cert.id"
          class="certificate-item"
          :class="{ 'is-recent': cert.isRecent, 'is-best': cert.isBest }"
        >
          <!-- Header: Episode and survival time -->
          <div class="cert-header">
            <div class="cert-badge">
              <span v-if="cert.isBest" class="badge-icon">🏆</span>
              <span v-if="cert.isRecent" class="badge-icon">⚡</span>
              <span class="cert-episode">Ep {{ cert.episode }}</span>
            </div>
            <div class="cert-survival">
              <span class="survival-steps">{{ cert.survivalSteps }}</span>
              <span class="survival-label">steps</span>
            </div>
          </div>

          <!-- Critical meters at death -->
          <div class="cert-meters">
            <div class="meters-label">Critical Meters:</div>
            <div class="meter-chips">
              <div
                v-for="meter in cert.criticalMeters"
                :key="meter.name"
                class="meter-chip"
                :class="`meter-${meter.severity}`"
              >
                <span class="meter-name">{{ meter.name }}</span>
                <span class="meter-value">{{ meter.value }}%</span>
              </div>
            </div>
          </div>

          <!-- Affordance usage -->
          <div class="cert-affordances">
            <div class="affordances-label">Affordance Usage:</div>
            <div class="affordance-list">
              <div
                v-for="aff in cert.topAffordances"
                :key="aff.name"
                class="affordance-row"
              >
                <span class="aff-name">{{ aff.name }}</span>
                <span class="aff-count">×{{ aff.count }}</span>
              </div>
            </div>
          </div>
        </div>
      </TransitionGroup>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import InfoTooltip from './InfoTooltip.vue'

const props = defineProps({
  certificates: {
    type: Array,
    default: () => []
  }
})

// Display most recent at top, then best 3 underneath
const displayedCertificates = computed(() => {
  if (props.certificates.length === 0) return []

  // Get most recent death
  const mostRecent = { ...props.certificates[props.certificates.length - 1], isRecent: true, isBest: false }

  // Sort by survival time (descending) for best 3
  const sorted = [...props.certificates].sort((a, b) => b.survivalSteps - a.survivalSteps)
  const top3 = sorted.slice(0, 3).map(cert => ({ ...cert, isBest: true, isRecent: false }))

  // Check if most recent is in top 3
  const recentInTop3 = top3.some(cert => cert.id === mostRecent.id)

  if (recentInTop3) {
    // Most recent is in top 3 - mark it as both
    return [
      { ...mostRecent, isBest: true, isRecent: true },
      ...top3.filter(cert => cert.id !== mostRecent.id)
    ]
  } else {
    // Most recent is not in top 3 - show it first, then top 3
    return [mostRecent, ...top3]
  }
})
</script>

<style scoped>
.death-certificates {
  position: absolute;
  top: 80px; /* Below MinimalControls */
  right: var(--spacing-md);
  width: 320px;
  height: 620px; /* Fixed height - fits header + 4 certificates nicely */
  background: var(--color-bg-secondary);
  border: 2px solid var(--color-border);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  backdrop-filter: blur(8px);
  z-index: 90;
  opacity: 1; /* Fully opaque */
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

/* No deaths state */
.no-deaths-state {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-lg);
  background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(74, 222, 128, 0.1));
  border: 2px solid rgba(34, 197, 94, 0.3);
  border-radius: var(--border-radius-md);
  min-height: 80px;
}

.no-deaths-icon {
  font-size: 2rem;
  flex-shrink: 0;
}

.no-deaths-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.no-deaths-title {
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-semibold);
  color: var(--color-success);
}

.no-deaths-message {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
  line-height: 1.4;
}

/* Certificate list */
.certificate-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  padding-right: var(--spacing-xs);
}

/* Custom scrollbar */
.certificate-list::-webkit-scrollbar {
  width: 6px;
}

.certificate-list::-webkit-scrollbar-track {
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-sm);
}

.certificate-list::-webkit-scrollbar-thumb {
  background: var(--color-interactive-disabled);
  border-radius: var(--border-radius-sm);
}

.certificate-list::-webkit-scrollbar-thumb:hover {
  background: var(--color-text-tertiary);
}

/* Certificate item */
.certificate-item {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  padding: var(--spacing-sm);
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
  border-left: 3px solid var(--color-text-tertiary);
  transition: all var(--transition-base);
}

.certificate-item.is-best {
  border-left-color: #fbbf24; /* Gold */
  background: rgba(251, 191, 36, 0.05);
}

.certificate-item.is-recent {
  border-left-color: #c0c0c0; /* Silver */
  background: rgba(192, 192, 192, 0.05);
}

.certificate-item.is-best.is-recent {
  border-left-color: #fbbf24; /* Gold (best trumps recent) */
  background: rgba(251, 191, 36, 0.05);
}

/* Certificate header */
.cert-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.cert-badge {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
}

.badge-icon {
  font-size: 1rem;
  line-height: 1;
}

.cert-episode {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
  font-family: 'Monaco', 'Courier New', monospace;
}

.cert-survival {
  display: flex;
  align-items: baseline;
  gap: 4px;
}

.survival-steps {
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-bold);
  color: var(--color-text-primary);
  font-family: 'Monaco', 'Courier New', monospace;
}

.survival-label {
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
}

/* Critical meters */
.cert-meters {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.meters-label {
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.meter-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.meter-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-medium);
  font-family: 'Monaco', 'Courier New', monospace;
}

.meter-chip.meter-critical {
  background: rgba(239, 68, 68, 0.2);
  color: var(--color-error);
}

.meter-chip.meter-low {
  background: rgba(245, 158, 11, 0.2);
  color: var(--color-warning);
}

.meter-name {
  text-transform: capitalize;
}

.meter-value {
  font-weight: var(--font-weight-bold);
}

/* Affordance usage */
.cert-affordances {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding-top: 4px;
  border-top: 1px solid var(--color-border);
}

.affordances-label {
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.affordance-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.affordance-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: var(--font-size-xs);
}

.aff-name {
  color: var(--color-text-secondary);
  text-transform: capitalize;
}

.aff-count {
  font-family: 'Monaco', 'Courier New', monospace;
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

/* Transition animations */
.cert-slide-enter-active {
  transition: all 0.3s ease;
}

.cert-slide-leave-active {
  transition: all 0.2s ease;
}

.cert-slide-enter-from {
  opacity: 0;
  transform: translateY(-10px);
}

.cert-slide-leave-to {
  opacity: 0;
  transform: translateY(10px);
}

.cert-slide-move {
  transition: transform 0.3s ease;
}
</style>
