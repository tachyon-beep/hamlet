<template>
  <div class="death-certificates">
    <div class="log-header">
      <h4>💀 Death Certificates</h4>
    </div>

    <!-- No deaths yet state -->
    <div v-if="displayedCertificates.length === 0" class="no-deaths-state">
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
          :key="cert.displayKey"
          class="certificate-item"
          :class="{ 'is-recent': cert.isRecent && !cert.isBest, 'is-best': cert.isBest }"
        >
          <!-- Header: Episode and survival time -->
          <div class="cert-header">
            <div class="cert-badge">
              <span v-if="cert.isBest" class="badge-icon">🏆</span>
              <span v-if="cert.isRecent" class="badge-icon">⚡</span>
              <span class="cert-episode">Ep {{ cert.episode }}</span>
            </div>
            <div class="cert-stats">
              <div class="cert-survival">
                <span class="survival-steps">{{ cert.survivalSteps }}</span>
                <span class="survival-label">steps</span>
              </div>
              <div class="cert-reward">
                <span class="reward-value">{{ cert.totalReward?.toFixed(1) || '0.0' }}</span>
                <span class="reward-label">reward</span>
              </div>
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
import { computed, shallowRef, watch } from 'vue'

const props = defineProps({
  certificates: {
    type: Array,
    default: () => []
  }
})

// Use shallowRef to store computed certificates to avoid unnecessary deep reactivity
const displayedCertificates = shallowRef([])

// Watch certificates and update only when actually needed
watch(
  () => props.certificates,
  (newCerts) => {
    if (newCerts.length === 0) {
      displayedCertificates.value = []
      return
    }

    // Find the best (longest survival) by iterating once
    let best = newCerts[0]
    let bestIndex = 0
    for (let i = 1; i < newCerts.length; i++) {
      if (newCerts[i].survivalSteps > best.survivalSteps) {
        best = newCerts[i]
        bestIndex = i
      }
    }

    // For "recent", show the second-most-recent death (if it exists)
    const recentIndex = newCerts.length >= 2 ? newCerts.length - 2 : newCerts.length - 1
    const recent = newCerts[recentIndex]

    // Check if we only have one certificate, or if recent is also the best
    if (newCerts.length === 1 || recent.id === best.id) {
      // Show only the best certificate
      displayedCertificates.value = [
        { ...best, isRecent: false, isBest: true, displayKey: 'best' }
      ]
    } else {
      // Show best first (stable at top), then recent second-to-last
      displayedCertificates.value = [
        { ...best, isBest: true, isRecent: false, displayKey: 'best' },
        { ...recent, isRecent: true, isBest: false, displayKey: 'recent' }
      ]
    }
  },
  { immediate: true, deep: true }
)
</script>

<style scoped>
.death-certificates {
  position: absolute;
  top: 80px; /* Below MinimalControls */
  right: var(--spacing-md);
  bottom: 70px; /* Stop above zoom slider (16px bottom + ~50px slider height + gap) */
  width: 320px;
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
  font-size: var(--font-size-xl);  /* 20px - increased from 16px for streaming */
  font-weight: var(--font-weight-bold);  /* Bolder */
  color: var(--color-text-primary);
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
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
  gap: var(--spacing-md);  /* 16px - increased from 8px for better scannability */
  padding-right: var(--spacing-xs);
  position: relative;  /* For absolute positioning during transitions */
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
  gap: var(--spacing-sm);  /* 8px - increased from 4px */
  padding: var(--spacing-md);  /* 16px - increased from 8px */
  background: var(--color-bg-primary);
  border-radius: var(--border-radius-sm);
  border-left: 5px solid var(--color-text-tertiary);  /* Thicker border for streaming */
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);  /* Added shadow for depth */

  /* GPU acceleration for smooth rendering */
  will-change: transform, opacity;
  transform: translateZ(0);
  backface-visibility: hidden;

  /* Smooth color transitions when promoted from recent to best */
  transition:
    border-left-color 1.5s ease-in-out,
    border-left-width 1.5s ease-in-out,
    background 1.5s ease-in-out,
    transform 0.3s ease,
    opacity 0.3s ease;
}

.certificate-item.is-best {
  border-left-color: #fbbf24; /* Gold */
  border-left-width: 6px;  /* Even thicker for best */
  background: rgba(251, 191, 36, 0.08);  /* Slightly stronger tint */
}

.certificate-item.is-recent {
  border-left-color: #00d9ff; /* Neon blue (electricity) */
  border-left-width: 6px;  /* Match best thickness */
  background: rgba(0, 217, 255, 0.08);
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
  font-size: 1.25rem;  /* 20px - increased from 16px for streaming */
  line-height: 1;
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.4));  /* Added depth */
}

.cert-episode {
  font-size: var(--font-size-base);  /* 16px - increased from 14px */
  font-weight: var(--font-weight-bold);  /* Bolder */
  color: var(--color-text-primary);
  font-family: 'Monaco', 'Courier New', monospace;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
}

.cert-stats {
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: flex-end;
}

.cert-survival {
  display: flex;
  align-items: baseline;
  gap: 4px;
}

.survival-steps {
  font-size: 22px;  /* Increased from 18px - most important metric for streaming */
  font-weight: var(--font-weight-bold);
  color: var(--color-text-primary);
  font-family: 'Monaco', 'Courier New', monospace;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.4);
}

.survival-label {
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
}

.cert-reward {
  display: flex;
  align-items: baseline;
  gap: 4px;
}

.reward-value {
  font-size: 18px;  /* Slightly smaller than steps */
  font-weight: var(--font-weight-semibold);
  color: #4caf50;  /* Green to indicate efficiency/reward */
  font-family: 'Monaco', 'Courier New', monospace;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.4);
}

.reward-label {
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
  font-size: 13px;  /* Increased from 12px for streaming legibility */
  font-weight: var(--font-weight-bold);  /* Bolder */
  color: var(--color-text-primary);  /* Changed from secondary - more contrast */
  text-transform: uppercase;
  letter-spacing: 0.8px;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
  margin-bottom: 2px;  /* Add spacing below label */
}

.meter-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.meter-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;  /* Larger gap */
  padding: 4px 8px;  /* More padding for visibility */
  border-radius: 6px;
  font-size: 13px;  /* Increased from 12px for streaming */
  font-weight: var(--font-weight-semibold);  /* Bolder */
  font-family: 'Monaco', 'Courier New', monospace;
  border: 1px solid currentColor;  /* Added border for definition */
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
  font-size: 13px;  /* Increased from 12px for streaming legibility */
  font-weight: var(--font-weight-bold);  /* Bolder */
  color: var(--color-text-primary);  /* Changed from secondary - more contrast */
  text-transform: uppercase;
  letter-spacing: 0.8px;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
  margin-bottom: 2px;  /* Add spacing below label */
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
  font-size: 13px;  /* Increased from 12px for streaming */
  padding: 2px 0;  /* Added vertical padding for spacing */
}

.aff-name {
  color: var(--color-text-secondary);
  text-transform: capitalize;
  font-weight: var(--font-weight-medium);  /* Added weight */
}

.aff-count {
  font-family: 'Monaco', 'Courier New', monospace;
  font-weight: var(--font-weight-bold);  /* Changed from semibold */
  color: var(--color-text-primary);
  font-size: 14px;  /* Slightly larger than label for hierarchy */
}

/* Transition animations - GPU-accelerated with transform only */
.cert-slide-enter-active,
.cert-slide-leave-active {
  position: absolute;
  width: 100%;
}

.cert-slide-enter-active {
  transition: all 1.5s ease-out;
}

.cert-slide-leave-active {
  transition: all 1s ease-in;
}

.cert-slide-enter-from {
  opacity: 0;
  transform: translate3d(0, 30px, 0);
}

.cert-slide-leave-to {
  opacity: 0;
  transform: translate3d(0, -30px, 0);
}

.cert-slide-move {
  transition: transform 1.5s ease-in-out;
}
</style>
