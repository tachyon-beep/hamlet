<template>
  <div class="zoom-control">
    <span class="zoom-label">{{ Math.round(zoomValue * 100) }}%</span>
    <input
      type="range"
      v-model.number="zoomValue"
      @input="onZoomChange"
      min="0.5"
      max="2.0"
      step="0.1"
      class="zoom-slider"
      aria-label="Grid zoom level"
    />
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  zoom: {
    type: Number,
    default: 1.0
  }
})

const emit = defineEmits(['update:zoom'])

const zoomValue = ref(props.zoom)

// Watch for external zoom changes
watch(() => props.zoom, (newZoom) => {
  zoomValue.value = newZoom
})

function onZoomChange() {
  emit('update:zoom', zoomValue.value)
}
</script>

<style scoped>
.zoom-control {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(8px);
  border: 1px solid var(--color-border);
}

.zoom-label {
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-secondary);
  font-family: 'Monaco', 'Courier New', monospace;
  font-variant-numeric: tabular-nums;
  min-width: 3ch;
}

.zoom-slider {
  width: 100px;
  height: 4px;
  border-radius: var(--border-radius-full);
  background: var(--color-bg-tertiary);
  outline: none;
  -webkit-appearance: none;
  appearance: none;
  cursor: pointer;
}

.zoom-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--color-interactive);
  cursor: pointer;
  transition: all var(--transition-base);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.zoom-slider::-webkit-slider-thumb:hover {
  background: var(--color-interactive-hover);
  transform: scale(1.1);
}

.zoom-slider::-webkit-slider-thumb:active {
  transform: scale(0.95);
}

.zoom-slider::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--color-interactive);
  cursor: pointer;
  border: none;
  transition: all var(--transition-base);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.zoom-slider::-moz-range-thumb:hover {
  background: var(--color-interactive-hover);
  transform: scale(1.1);
}

.zoom-slider::-moz-range-thumb:active {
  transform: scale(0.95);
}

.zoom-slider::-moz-range-track {
  width: 100%;
  height: 4px;
  background: var(--color-bg-tertiary);
  border-radius: var(--border-radius-full);
}
</style>
