<template>
  <!-- ✅ Error state with ARIA alert role -->
  <div class="error-state" role="alert">
    <div class="error-icon" aria-hidden="true">⚠️</div>
    <h3 class="error-title">{{ title }}</h3>
    <p class="error-message">{{ message }}</p>
    <button
      v-if="showRetry"
      @click="$emit('retry')"
      class="retry-button"
      aria-label="Retry the failed operation"
    >
      {{ retryText }}
    </button>
  </div>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    default: 'Something went wrong'
  },
  message: {
    type: String,
    default: 'An error occurred. Please try again.'
  },
  showRetry: {
    type: Boolean,
    default: true
  },
  retryText: {
    type: String,
    default: 'Retry'
  }
})

defineEmits(['retry'])
</script>

<style scoped>
.error-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-md);
  padding: var(--spacing-2xl);
  text-align: center;
  color: var(--color-text-primary);
}

.error-icon {
  font-size: 48px;
  line-height: 1;
}

.error-title {
  margin: 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  color: var(--color-error);
}

.error-message {
  margin: 0;
  font-size: var(--font-size-base);
  color: var(--color-text-secondary);
  max-width: 400px;
}

.retry-button {
  margin-top: var(--spacing-sm);
  padding: var(--spacing-sm) var(--spacing-lg);
  background: var(--color-error);
  color: white;
  border: none;
  border-radius: var(--border-radius-sm);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  cursor: pointer;
  transition: all var(--transition-base);
}

.retry-button:hover {
  background: var(--color-error-hover);
  transform: translateY(-1px);
}

.retry-button:active {
  transform: translateY(0);
}
</style>
