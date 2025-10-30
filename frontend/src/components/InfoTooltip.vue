<template>
  <div class="info-tooltip-wrapper">
    <button
      class="info-icon"
      @mouseenter="showTooltip = true"
      @mouseleave="showTooltip = false"
      @focus="showTooltip = true"
      @blur="showTooltip = false"
      :aria-label="`Information: ${title}`"
      type="button"
    >
      <span aria-hidden="true">ℹ️</span>
    </button>
    <Transition name="tooltip-fade">
      <div v-if="showTooltip" class="tooltip-popup" :class="position" role="tooltip">
        <div class="tooltip-title" v-if="title">{{ title }}</div>
        <div class="tooltip-text">{{ text }}</div>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref } from 'vue'

defineProps({
  title: {
    type: String,
    default: ''
  },
  text: {
    type: String,
    required: true
  },
  position: {
    type: String,
    default: 'top',
    validator: (value) => ['top', 'bottom', 'left', 'right'].includes(value)
  }
})

const showTooltip = ref(false)
</script>

<style scoped>
.info-tooltip-wrapper {
  position: relative;
  display: inline-block;
}

.info-icon {
  background: none;
  border: none;
  padding: 0;
  margin: 0;
  cursor: help;
  font-size: 0.875rem;
  line-height: 1;
  opacity: 0.6;
  transition: opacity var(--transition-base);
}

.info-icon:hover,
.info-icon:focus {
  opacity: 1;
}

.info-icon:focus {
  outline: 2px solid var(--color-interactive-focus);
  outline-offset: 2px;
  border-radius: 2px;
}

.tooltip-popup {
  position: absolute;
  z-index: var(--z-index-tooltip);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius-sm);
  padding: var(--spacing-sm);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  width: max-content;
  max-width: 250px;
  pointer-events: none;
}

/* Position variants */
.tooltip-popup.top {
  bottom: calc(100% + 8px);
  left: 50%;
  transform: translateX(-50%);
}

.tooltip-popup.bottom {
  top: calc(100% + 8px);
  left: 50%;
  transform: translateX(-50%);
}

.tooltip-popup.left {
  right: calc(100% + 8px);
  top: 50%;
  transform: translateY(-50%);
}

.tooltip-popup.right {
  left: calc(100% + 8px);
  top: 50%;
  transform: translateY(-50%);
}

.tooltip-title {
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
  margin-bottom: 4px;
  border-bottom: 1px solid var(--color-border);
  padding-bottom: 4px;
}

.tooltip-text {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  line-height: 1.5;
}

/* Fade transition */
.tooltip-fade-enter-active,
.tooltip-fade-leave-active {
  transition: opacity 0.2s ease;
}

.tooltip-fade-enter-from,
.tooltip-fade-leave-to {
  opacity: 0;
}
</style>
