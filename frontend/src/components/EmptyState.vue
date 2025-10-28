<template>
  <!-- ✅ Empty state with proper semantics -->
  <div class="empty-state">
    <div class="empty-icon" aria-hidden="true">{{ icon }}</div>
    <h3 class="empty-title">{{ title }}</h3>
    <p class="empty-message">{{ message }}</p>
    <button
      v-if="showAction"
      @click="$emit('action')"
      class="action-button"
      :aria-label="actionLabel"
    >
      {{ actionText }}
    </button>
  </div>
</template>

<script setup>
defineProps({
  icon: {
    type: String,
    default: '📭'
  },
  title: {
    type: String,
    default: 'No data available'
  },
  message: {
    type: String,
    default: 'There is nothing to display yet.'
  },
  showAction: {
    type: Boolean,
    default: false
  },
  actionText: {
    type: String,
    default: 'Get started'
  },
  actionLabel: {
    type: String,
    default: ''
  }
})

defineEmits(['action'])
</script>

<style scoped>
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-md);
  padding: var(--spacing-2xl);
  text-align: center;
  color: var(--color-text-secondary);
}

.empty-icon {
  font-size: 64px;
  line-height: 1;
  opacity: 0.7;
}

.empty-title {
  margin: 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

.empty-message {
  margin: 0;
  font-size: var(--font-size-base);
  color: var(--color-text-secondary);
  max-width: 400px;
}

.action-button {
  margin-top: var(--spacing-sm);
  padding: var(--spacing-sm) var(--spacing-lg);
  background: var(--color-interactive);
  color: white;
  border: none;
  border-radius: var(--border-radius-sm);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  cursor: pointer;
  transition: all var(--transition-base);
}

.action-button:hover {
  background: var(--color-interactive-hover);
  transform: translateY(-1px);
}

.action-button:active {
  transform: translateY(0);
}
</style>
