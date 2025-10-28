# Hamlet Frontend Implementation Plan
## From Audit to Production Quality

**Based on**: Frontend Audit Report (vue3-ux-design skill v1.0.0)
**Current Score**: 55% (C-)
**Target Score**: 85%+ (B+)
**Total Estimated Time**: 35 hours
**Created**: 2025-10-28

---

## Table of Contents

1. [Week 1: Foundations (18 hours)](#week-1-foundations)
   - Task 1.1: Create Design Tokens System
   - Task 1.2: Implement Accessibility Standards
   - Task 1.3: Add Error/Loading/Empty States
2. [Week 2: Architecture (14 hours)](#week-2-architecture)
   - Task 2.1: Decouple Components from Store
   - Task 2.2: Implement Mobile Responsiveness
3. [Week 3: Polish (3 hours)](#week-3-polish)
   - Task 3.1: Clean Up Template Logic
4. [Testing Checklist](#testing-checklist)
5. [Acceptance Criteria](#acceptance-criteria)

---

## Week 1: Foundations (18 hours)

### Task 1.1: Create Design Tokens System 🔴 BLOCKER
**Priority**: Critical
**Time**: 4 hours
**Blocks**: All other tasks

#### Subtask 1.1.1: Create tokens.js file (1 hour)

**File**: `frontend/src/styles/tokens.js`

```javascript
/**
 * Design Tokens - Single source of truth for all design decisions
 *
 * Usage:
 *   import { tokens } from '@/styles/tokens'
 *   const color = tokens.colors.interactive
 */

export const tokens = {
  // ============================================================================
  // Colors - Semantic names describing PURPOSE, not appearance
  // ============================================================================
  colors: {
    // Background surfaces
    backgroundPrimary: '#1e1e2e',     // Main app background
    backgroundSecondary: '#2a2a3e',   // Panel backgrounds
    backgroundTertiary: '#3a3a4e',    // Nested elements, borders

    // Text colors
    textPrimary: '#e0e0e0',           // Main text
    textSecondary: '#a0a0b0',         // Labels, secondary info
    textTertiary: '#808090',          // Disabled, hints
    textMuted: '#6a6a7a',             // Very subtle text

    // Interactive elements
    interactive: '#10b981',            // Primary interactive (green)
    interactiveHover: '#34d399',       // Hover state
    interactiveFocus: '#6ee7b7',       // Focus ring
    interactiveDisabled: '#4a4a5e',    // Disabled state

    // Status colors
    success: '#10b981',                // Success state
    warning: '#f59e0b',                // Warning state
    error: '#ef4444',                  // Error state
    info: '#3b82f6',                   // Info state (blue)

    // Chart colors
    chartPrimary: '#3b82f6',           // Main chart color (blue)
    chartSecondary: '#8b5cf6',         // Secondary chart color (purple)
    chartGrid: '#3a3a4e',              // Chart grid lines

    // Meter-specific colors (from MeterPanel)
    meterEnergy: '#10b981',            // Green
    meterHygiene: '#06b6d4',           // Cyan
    meterSatiation: '#f59e0b',         // Orange
    meterMoney: '#8b5cf6',             // Purple
    meterStress: {
      low: '#10b981',                  // Low stress = green
      mid: '#f59e0b',                  // Medium stress = yellow
      high: '#ef4444',                 // High stress = red
    },
    meterSocial: '#ec4899',            // Pink

    // Mode-specific colors
    modeInference: '#3b82f6',          // Blue
    modeTraining: '#3b82f6',           // Same blue
  },

  // ============================================================================
  // Spacing - Consistent rhythm throughout the app
  // ============================================================================
  spacing: {
    xs: '0.25rem',    // 4px  - Tiny gaps
    sm: '0.5rem',     // 8px  - Small gaps
    md: '1rem',       // 16px - Standard gap
    lg: '1.5rem',     // 24px - Large gap
    xl: '2rem',       // 32px - Extra large gap
    '2xl': '3rem',    // 48px - Section spacing
  },

  // ============================================================================
  // Typography - Font sizes and weights
  // ============================================================================
  fontSize: {
    xs: '0.75rem',    // 12px - Small labels, hints
    sm: '0.875rem',   // 14px - Secondary text, buttons
    base: '1rem',     // 16px - Body text
    lg: '1.125rem',   // 18px - Large text, headings
    xl: '1.25rem',    // 20px - Large values, stats
    '2xl': '1.5rem',  // 24px - Page titles
  },

  fontWeight: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },

  // ============================================================================
  // Layout - Border radius, transitions, shadows
  // ============================================================================
  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '12px',
    full: '9999px',   // Circular
  },

  transition: {
    fast: '0.15s ease',
    base: '0.2s ease',   // Changed from 0.3s to match current code
    slow: '0.5s ease',
  },

  // ============================================================================
  // Responsive breakpoints - Mobile-first approach
  // ============================================================================
  breakpoints: {
    sm: '640px',      // Small tablets
    md: '768px',      // Tablets
    lg: '1024px',     // Small desktops
    xl: '1280px',     // Large desktops
  },

  // ============================================================================
  // Layout dimensions - Component widths, panel sizes
  // ============================================================================
  layout: {
    leftPanelWidth: '320px',
    rightPanelWidth: '380px',
    maxGridSize: '600px',
    headerHeight: '70px',
  },

  // ============================================================================
  // Accessibility - WCAG compliant sizes
  // ============================================================================
  accessibility: {
    minTouchTarget: '44px',  // WCAG minimum
    focusOutlineWidth: '2px',
    focusOutlineOffset: '2px',
  },

  // ============================================================================
  // Z-index layers - Stacking order
  // ============================================================================
  zIndex: {
    base: 1,
    dropdown: 10,
    modal: 100,
    toast: 1000,
  },
}

/**
 * Helper function to convert tokens to CSS custom properties
 * Usage in your CSS:
 *   import { tokensToCSS } from '@/styles/tokens'
 *   :root { tokensToCSS() }
 */
export function tokensToCSS() {
  // This will be implemented in subtask 1.1.2
  return ''
}
```

**Acceptance Criteria**:
- [ ] File created at `frontend/src/styles/tokens.js`
- [ ] All 20+ colors extracted from audit report
- [ ] All spacing values extracted
- [ ] Can import in any component

#### Subtask 1.1.2: Create CSS custom properties (1 hour)

**File**: `frontend/src/styles/variables.css`

```css
/**
 * CSS Custom Properties - Auto-generated from tokens.js
 *
 * Usage in Vue components:
 *   color: var(--color-text-primary);
 *   padding: var(--spacing-md);
 */

:root {
  /* Colors */
  --color-bg-primary: #1e1e2e;
  --color-bg-secondary: #2a2a3e;
  --color-bg-tertiary: #3a3a4e;

  --color-text-primary: #e0e0e0;
  --color-text-secondary: #a0a0b0;
  --color-text-tertiary: #808090;
  --color-text-muted: #6a6a7a;

  --color-interactive: #10b981;
  --color-interactive-hover: #34d399;
  --color-interactive-focus: #6ee7b7;
  --color-interactive-disabled: #4a4a5e;

  --color-success: #10b981;
  --color-warning: #f59e0b;
  --color-error: #ef4444;
  --color-info: #3b82f6;

  --color-chart-primary: #3b82f6;
  --color-chart-secondary: #8b5cf6;
  --color-chart-grid: #3a3a4e;

  --color-meter-energy: #10b981;
  --color-meter-hygiene: #06b6d4;
  --color-meter-satiation: #f59e0b;
  --color-meter-money: #8b5cf6;
  --color-meter-stress-low: #10b981;
  --color-meter-stress-mid: #f59e0b;
  --color-meter-stress-high: #ef4444;
  --color-meter-social: #ec4899;

  /* Spacing */
  --spacing-xs: 0.25rem;
  --spacing-sm: 0.5rem;
  --spacing-md: 1rem;
  --spacing-lg: 1.5rem;
  --spacing-xl: 2rem;
  --spacing-2xl: 3rem;

  /* Typography */
  --font-size-xs: 0.75rem;
  --font-size-sm: 0.875rem;
  --font-size-base: 1rem;
  --font-size-lg: 1.125rem;
  --font-size-xl: 1.25rem;
  --font-size-2xl: 1.5rem;

  --font-weight-normal: 400;
  --font-weight-medium: 500;
  --font-weight-semibold: 600;
  --font-weight-bold: 700;

  /* Layout */
  --border-radius-sm: 4px;
  --border-radius-md: 8px;
  --border-radius-lg: 12px;
  --border-radius-full: 9999px;

  --transition-fast: 0.15s ease;
  --transition-base: 0.2s ease;
  --transition-slow: 0.5s ease;

  /* Dimensions */
  --layout-left-panel-width: 320px;
  --layout-right-panel-width: 380px;
  --layout-max-grid-size: 600px;
  --layout-header-height: 70px;

  /* Accessibility */
  --a11y-min-touch-target: 44px;
  --a11y-focus-outline-width: 2px;
  --a11y-focus-outline-offset: 2px;
}
```

**Acceptance Criteria**:
- [ ] File created at `frontend/src/styles/variables.css`
- [ ] Imported in `main.js`
- [ ] Can use `var(--color-bg-primary)` in components

#### Subtask 1.1.3: Import in main.js (15 min)

**File**: `frontend/src/main.js`

```javascript
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'

// Import global styles and design tokens
import './assets/styles.css'
import './styles/variables.css'  // ← ADD THIS

const app = createApp(App)
app.use(createPinia())
app.mount('#app')
```

**Acceptance Criteria**:
- [ ] variables.css imported globally
- [ ] All CSS custom properties available in components

#### Subtask 1.1.4: Refactor one component as example (1.75 hours)

**File**: `frontend/src/components/App.vue` (refactored)

**BEFORE**:
```vue
<style scoped>
.header {
  padding: 1rem 2rem;
  border-bottom: 1px solid #3a3a4e;
}
.left-panel {
  width: 320px;
}
</style>
```

**AFTER**:
```vue
<style scoped>
.header {
  padding: var(--spacing-md) var(--spacing-xl);
  border-bottom: 1px solid var(--color-bg-tertiary);
}
.left-panel {
  width: var(--layout-left-panel-width);
}
</style>
```

**Complete refactoring checklist for App.vue**:
- [ ] Replace all hardcoded colors with CSS variables
- [ ] Replace all spacing values with spacing tokens
- [ ] Replace font sizes with typography tokens
- [ ] Replace border-radius with layout tokens
- [ ] Test that UI looks identical

**Acceptance Criteria**:
- [ ] App.vue uses zero hardcoded values
- [ ] Visual appearance unchanged
- [ ] Can change theme by modifying variables.css

---

### Task 1.2: Implement Accessibility Standards 🔴 CRITICAL
**Priority**: Critical (Legal/ethical requirement)
**Time**: 8 hours
**Dependencies**: None (can start immediately)

#### Subtask 1.2.1: Add ARIA live regions to real-time components (2 hours)

**Components to update**: StatsPanel.vue, MeterPanel.vue, Controls.vue (training metrics)

**File**: `frontend/src/components/StatsPanel.vue` (partial example)

**ADD** aria-live regions:

```vue
<template>
  <div class="stats-panel">
    <h3>Episode Info</h3>

    <div class="stats-grid">
      <div class="stat-item">
        <span class="stat-label">Episode</span>
        <!-- ADD aria-live for screen readers -->
        <span
          class="stat-value"
          aria-live="polite"
          aria-atomic="true"
          role="status"
        >
          #{{ currentEpisode }}
        </span>
      </div>

      <div class="stat-item">
        <span class="stat-label">Steps</span>
        <span
          class="stat-value"
          aria-live="polite"
          aria-atomic="true"
          role="status"
        >
          {{ currentStep }}
        </span>
      </div>

      <div class="stat-item">
        <span class="stat-label">Reward</span>
        <span
          class="stat-value"
          :class="{ positive: cumulativeReward > 0, negative: cumulativeReward < 0 }"
          aria-live="polite"
          aria-atomic="true"
          role="status"
        >
          {{ formattedReward }}  <!-- Use computed property -->
        </span>
      </div>

      <div class="stat-item">
        <span class="stat-label">Action</span>
        <span
          class="stat-value action"
          aria-live="polite"
          aria-atomic="true"
          role="status"
        >
          {{ formatAction(lastAction) }}
        </span>
      </div>
    </div>

    <!-- Chart with proper ARIA -->
    <div v-if="episodeHistory.length > 0" class="history">
      <h4>Performance</h4>

      <div
        class="history-chart"
        role="img"
        aria-label="Episode performance chart showing survival time over last 10 episodes"
      >
        <svg viewBox="0 0 300 100" class="chart-svg" aria-hidden="true">
          <!-- SVG content -->
        </svg>

        <!-- Screen reader alternative -->
        <div class="sr-only">
          Chart data: Last 10 episodes with survival times ranging from
          {{ minSurvival }} to {{ maxSurvival }} steps.
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
// ADD screen reader only class utility
const formattedReward = computed(() => {
  return cumulativeReward.value.toFixed(1)
})

const minSurvival = computed(() => {
  if (episodeHistory.value.length === 0) return 0
  return Math.min(...episodeHistory.value.map(ep => ep.steps))
})

const maxSurvival = computed(() => {
  if (episodeHistory.value.length === 0) return 0
  return Math.max(...episodeHistory.value.map(ep => ep.steps))
})
</script>

<style scoped>
/* Screen reader only content */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}
</style>
```

**Checklist for StatsPanel.vue**:
- [ ] Add `aria-live="polite"` to all stat values
- [ ] Add `role="status"` to real-time values
- [ ] Add `role="img"` to chart with descriptive `aria-label`
- [ ] Add screen-reader text describing chart data
- [ ] Add `.sr-only` utility class

**Repeat for**:
- [ ] MeterPanel.vue (all meter values)
- [ ] Controls.vue (training progress metrics)

**Acceptance Criteria**:
- [ ] Screen reader announces stat changes
- [ ] Chart has descriptive alternative text
- [ ] Test with VoiceOver (Mac) or NVDA (Windows)

#### Subtask 1.2.2: Add semantic HTML to all components (3 hours)

**Target components**: All 5 components

**Example transformations**:

**MeterPanel.vue** - Convert to semantic structure:

**BEFORE**:
```vue
<div class="meter-panel">
  <h3>Agent Meters</h3>
  <div class="meters">
    <!-- meters -->
  </div>
</div>
```

**AFTER**:
```vue
<section class="meter-panel" aria-labelledby="meter-heading">
  <h3 id="meter-heading">Agent Meters</h3>
  <div class="meters" role="list">
    <div
      v-for="(value, name) in meters"
      :key="name"
      class="meter"
      role="listitem"
      :aria-label="`${capitalize(name)}: ${formatValue(name, value)}`"
    >
      <!-- meter content -->
    </div>
  </div>
</section>
```

**App.vue** - Add semantic landmarks:

**BEFORE**:
```vue
<div class="app">
  <div class="header">
    <!-- header content -->
  </div>
  <div class="main-container">
    <!-- panels -->
  </div>
</div>
```

**AFTER**:
```vue
<div class="app">
  <header class="header" role="banner">
    <h1>HAMLET - DRL Visualization</h1>
    <div class="connection-status" :class="{ connected: isConnected }" role="status">
      {{ isConnected ? 'Connected' : 'Disconnected' }}
    </div>
  </header>

  <main class="main-container" role="main">
    <aside class="left-panel" aria-label="Agent status panels">
      <MeterPanel />
      <StatsPanel />
    </aside>

    <div class="grid-container" role="region" aria-label="Simulation grid">
      <Grid />
    </div>

    <aside class="right-panel" aria-label="Simulation controls">
      <Controls />
    </aside>
  </main>
</div>
```

**Controls.vue** - Wrap training form semantically:

**BEFORE**:
```vue
<div class="training-config">
  <div class="form-group">
    <label for="num-episodes">Episodes:</label>
    <input id="num-episodes" ... />
  </div>
</div>
```

**AFTER**:
```vue
<form class="training-config" @submit.prevent="startTraining">
  <fieldset>
    <legend class="sr-only">Training Configuration</legend>

    <div class="form-group">
      <label for="num-episodes">Episodes:</label>
      <input
        id="num-episodes"
        type="number"
        aria-required="true"
        aria-describedby="episodes-hint"
        ...
      />
      <span id="episodes-hint" class="hint">Number of training episodes</span>
    </div>

    <!-- More inputs -->
  </fieldset>

  <button type="submit" class="primary-button">
    Start Training
  </button>
</form>
```

**Checklist**:
- [ ] App.vue: Add `<header>`, `<main>`, `<aside>` tags
- [ ] MeterPanel.vue: Use `<section>` with role="list"
- [ ] StatsPanel.vue: Use `<section>` and proper headings
- [ ] Controls.vue: Wrap training config in `<form>`
- [ ] Grid.vue: Already good (SVG semantic)

**Acceptance Criteria**:
- [ ] All interactive regions have semantic tags
- [ ] Heading hierarchy is correct (h1 → h2 → h3)
- [ ] Form uses `<form>`, `<fieldset>`, `<legend>`
- [ ] Test with browser reader mode

#### Subtask 1.2.3: Add keyboard navigation support (2 hours)

**Components to update**: Controls.vue, Grid.vue

**Controls.vue** - Ensure all buttons are keyboard accessible:

```vue
<!-- Play/Pause/Step/Reset buttons - already good, using <button> -->

<!-- But ADD keyboard shortcuts: -->
<template>
  <div class="controls-panel" @keydown="handleKeyboardShortcuts">
    <!-- ... buttons ... -->
  </div>
</template>

<script setup>
// ADD keyboard shortcuts
function handleKeyboardShortcuts(event) {
  // Only when connected and in inference mode
  if (!isConnected.value || selectedMode.value !== 'inference') return

  // Don't capture if typing in input
  if (event.target.tagName === 'INPUT') return

  switch (event.key) {
    case ' ':  // Spacebar = play/pause
      event.preventDefault()
      store.pause()  // Toggle based on current state
      break
    case 'ArrowRight':  // → = step forward
      event.preventDefault()
      store.step()
      break
    case 'r':  // R = reset
      if (event.ctrlKey || event.metaKey) return  // Don't override browser
      event.preventDefault()
      store.reset()
      break
  }
}
</script>
```

**Grid.vue** - ADD keyboard controls for exploration:

```vue
<template>
  <div
    class="grid-container"
    tabindex="0"
    role="application"
    aria-label="Simulation grid. Use arrow keys to explore."
    @keydown="handleGridKeyboard"
  >
    <!-- heat map toggle - ensure focusable -->
    <button
      v-if="Object.keys(heatMap).length > 0"
      @click="showHeatMap = !showHeatMap"
      class="heat-map-toggle"
      :class="{ active: showHeatMap }"
      :aria-pressed="showHeatMap"
    >
      {{ showHeatMap ? 'Hide' : 'Show' }} Heat Map
    </button>

    <svg ... >
      <!-- Grid content -->
    </svg>
  </div>
</template>

<script setup>
// ADD keyboard navigation for grid
function handleGridKeyboard(event) {
  // Could add keyboard controls for camera pan, zoom, etc.
  // For now, just ensure heat map toggle is keyboard accessible
  if (event.key === 'h' && !event.ctrlKey && !event.metaKey) {
    event.preventDefault()
    showHeatMap.value = !showHeatMap.value
  }
}
</script>
```

**Checklist**:
- [ ] Controls.vue: Add keyboard shortcuts (spacebar, arrows, R)
- [ ] Grid.vue: Make heat map toggle keyboard accessible
- [ ] All interactive elements reachable with Tab
- [ ] Focus indicators visible on all buttons

**Acceptance Criteria**:
- [ ] Can navigate entire app with keyboard only
- [ ] Spacebar plays/pauses simulation
- [ ] Arrow key steps forward
- [ ] Tab order makes sense
- [ ] Focus indicators are visible (not `outline: none`)

#### Subtask 1.2.4: Add touch target sizing (1 hour)

**All components with buttons** - Ensure 44px minimum:

**Global utility** in `frontend/src/assets/styles.css`:

```css
/* Accessibility - Touch targets MUST be 44x44px minimum */
button,
input[type="button"],
input[type="submit"],
input[type="reset"],
a.button,
.button,
.control-button,
.mode-button {
  min-height: 44px;
  min-width: 44px;
}

/* Ensure slider thumbs are touch-friendly */
input[type="range"]::-webkit-slider-thumb {
  width: 44px;
  height: 44px;
}

input[type="range"]::-moz-range-thumb {
  width: 44px;
  height: 44px;
}

/* Focus indicators must be visible */
*:focus-visible {
  outline: var(--a11y-focus-outline-width) solid var(--color-interactive-focus);
  outline-offset: var(--a11y-focus-outline-offset);
}

/* Never remove outlines */
*:focus {
  outline-color: var(--color-interactive-focus);
}
```

**Update Controls.vue sliders**:

**BEFORE**:
```css
.speed-slider::-webkit-slider-thumb {
  width: 18px;  /* ❌ Too small */
  height: 18px;
}
```

**AFTER**:
```css
.speed-slider::-webkit-slider-thumb {
  width: 44px;  /* ✅ Touch-friendly */
  height: 44px;
  /* Make it look good at larger size */
  border-radius: 50%;
  background: var(--color-info);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  cursor: pointer;
  transition: var(--transition-fast);
}

.speed-slider::-webkit-slider-thumb:hover {
  transform: scale(1.1);
  background: var(--color-interactive-hover);
}
```

**Checklist**:
- [ ] Add touch target rules to global styles
- [ ] Verify all buttons are 44x44px or larger
- [ ] Update slider thumbs to 44x44px
- [ ] Add visible focus indicators
- [ ] Test on mobile device or browser dev tools

**Acceptance Criteria**:
- [ ] All interactive elements meet 44x44px minimum
- [ ] Sliders are easy to grab on mobile
- [ ] Focus indicators visible on all elements
- [ ] Test with Chrome DevTools mobile emulation

---

### Task 1.3: Add Error/Loading/Empty States 🟡 HIGH PRIORITY
**Priority**: High (Major UX improvement)
**Time**: 6 hours
**Dependencies**: Task 1.1 (design tokens)

#### Subtask 1.3.1: Create reusable state components (2 hours)

**File**: `frontend/src/components/base/LoadingState.vue` (NEW)

```vue
<template>
  <div class="loading-state" role="status" aria-live="polite">
    <div class="spinner" aria-hidden="true"></div>
    <p class="loading-message">{{ message }}</p>
  </div>
</template>

<script setup>
defineProps({
  message: {
    type: String,
    default: 'Loading...'
  }
})
</script>

<style scoped>
.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-md);
  padding: var(--spacing-2xl);
  min-height: 200px;
}

.spinner {
  width: 48px;
  height: 48px;
  border: 4px solid var(--color-bg-tertiary);
  border-top-color: var(--color-interactive);
  border-radius: var(--border-radius-full);
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.loading-message {
  color: var(--color-text-secondary);
  font-size: var(--font-size-sm);
  margin: 0;
}
</style>
```

**File**: `frontend/src/components/base/ErrorState.vue` (NEW)

```vue
<template>
  <div class="error-state" role="alert" aria-live="assertive">
    <svg class="error-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path
        fill="currentColor"
        d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"
      />
    </svg>
    <p class="error-message">{{ message }}</p>
    <button
      v-if="onRetry"
      @click="onRetry"
      class="retry-button"
      type="button"
    >
      {{ retryText }}
    </button>
  </div>
</template>

<script setup>
defineProps({
  message: {
    type: String,
    required: true
  },
  onRetry: {
    type: Function,
    default: null
  },
  retryText: {
    type: String,
    default: 'Retry'
  }
})
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
  min-height: 200px;
}

.error-icon {
  width: 64px;
  height: 64px;
  color: var(--color-error);
}

.error-message {
  color: var(--color-text-primary);
  font-size: var(--font-size-base);
  max-width: 400px;
  margin: 0;
}

.retry-button {
  min-height: var(--a11y-min-touch-target);
  padding: var(--spacing-sm) var(--spacing-lg);
  background: var(--color-interactive);
  color: var(--color-text-primary);
  border: none;
  border-radius: var(--border-radius-md);
  font-weight: var(--font-weight-medium);
  cursor: pointer;
  transition: var(--transition-base);
}

.retry-button:hover {
  background: var(--color-interactive-hover);
  transform: translateY(-1px);
}

.retry-button:focus-visible {
  outline: var(--a11y-focus-outline-width) solid var(--color-interactive-focus);
  outline-offset: var(--a11y-focus-outline-offset);
}
</style>
```

**File**: `frontend/src/components/base/EmptyState.vue` (NEW)

```vue
<template>
  <div class="empty-state">
    <svg class="empty-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path
        fill="currentColor"
        d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14zm-7-2h2v-4h4v-2h-4V7h-2v4H7v2h4z"
      />
    </svg>
    <p class="empty-message">{{ message }}</p>
    <button
      v-if="actionText && onAction"
      @click="onAction"
      class="action-button"
      type="button"
    >
      {{ actionText }}
    </button>
  </div>
</template>

<script setup>
defineProps({
  message: {
    type: String,
    required: true
  },
  actionText: {
    type: String,
    default: null
  },
  onAction: {
    type: Function,
    default: null
  }
})
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
  min-height: 200px;
}

.empty-icon {
  width: 64px;
  height: 64px;
  color: var(--color-text-tertiary);
}

.empty-message {
  color: var(--color-text-secondary);
  font-size: var(--font-size-base);
  font-style: italic;
  max-width: 400px;
  margin: 0;
}

.action-button {
  min-height: var(--a11y-min-touch-target);
  padding: var(--spacing-sm) var(--spacing-lg);
  background: var(--color-interactive);
  color: var(--color-text-primary);
  border: none;
  border-radius: var(--border-radius-md);
  font-weight: var(--font-weight-medium);
  cursor: pointer;
  transition: var(--transition-base);
}

.action-button:hover {
  background: var(--color-interactive-hover);
  transform: translateY(-1px);
}

.action-button:focus-visible {
  outline: var(--a11y-focus-outline-width) solid var(--color-interactive-focus);
  outline-offset: var(--a11y-focus-outline-offset);
}
</style>
```

**Checklist**:
- [ ] Create `frontend/src/components/base/` directory
- [ ] Create LoadingState.vue
- [ ] Create ErrorState.vue
- [ ] Create EmptyState.vue
- [ ] Test each component in isolation

**Acceptance Criteria**:
- [ ] Three reusable state components created
- [ ] All use design tokens
- [ ] All are accessible (ARIA, semantic HTML)
- [ ] Can be imported and used in any component

#### Subtask 1.3.2: Add states to App.vue (1 hour)

**File**: `frontend/src/components/App.vue` (updated)

```vue
<template>
  <div class="app">
    <header class="header" role="banner">
      <h1>HAMLET - DRL Visualization</h1>
      <div class="connection-status" :class="{ connected: isConnected }" role="status">
        {{ isConnected ? 'Connected' : 'Disconnected' }}
      </div>
    </header>

    <main class="main-container" role="main">
      <!-- ADD: Show states based on connection status -->

      <!-- Loading State: Connecting to server -->
      <LoadingState
        v-if="isConnecting"
        message="Connecting to simulation server..."
      />

      <!-- Error State: Connection failed -->
      <ErrorState
        v-else-if="connectionError"
        :message="connectionError"
        :on-retry="retryConnection"
        retry-text="Reconnect"
      />

      <!-- Empty State: Not connected -->
      <EmptyState
        v-else-if="!isConnected"
        message="Not connected to simulation. Use the controls panel to connect."
      />

      <!-- Data State: Connected and ready -->
      <template v-else>
        <aside class="left-panel" aria-label="Agent status panels">
          <MeterPanel />
          <StatsPanel />
        </aside>

        <div class="grid-container" role="region" aria-label="Simulation grid">
          <Grid />
        </div>

        <aside class="right-panel" aria-label="Simulation controls">
          <Controls />
        </aside>
      </template>
    </main>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useSimulationStore } from './stores/simulation'
import Grid from './components/Grid.vue'
import MeterPanel from './components/MeterPanel.vue'
import Controls from './components/Controls.vue'
import StatsPanel from './components/StatsPanel.vue'
import LoadingState from './components/base/LoadingState.vue'
import ErrorState from './components/base/ErrorState.vue'
import EmptyState from './components/base/EmptyState.vue'

const store = useSimulationStore()
const isConnected = computed(() => store.isConnected)

// ADD: Connection state tracking
const isConnecting = computed(() => store.isConnecting)
const connectionError = computed(() => store.connectionError)

function retryConnection() {
  store.clearConnectionError()
  store.connect()
}
</script>
```

**UPDATE**: `frontend/src/stores/simulation.js` (add connection states)

```javascript
// ADD to store state
const isConnecting = ref(false)
const connectionError = ref(null)

// UPDATE connect function
function connect(connectionMode = 'inference') {
  mode.value = connectionMode
  manualDisconnect.value = false
  isConnecting.value = true  // ← ADD
  connectionError.value = null  // ← ADD

  // ... existing WebSocket setup ...

  ws.value.onopen = () => {
    console.log('WebSocket connected')
    isConnected.value = true
    isConnecting.value = false  // ← ADD
    reconnectAttempts.value = 0
  }

  ws.value.onerror = (error) => {
    console.error('WebSocket error:', error)
    isConnecting.value = false  // ← ADD
    connectionError.value = 'Failed to connect to simulation server. Is it running?'  // ← ADD
  }

  // ... rest of code ...
}

// ADD helper function
function clearConnectionError() {
  connectionError.value = null
}

// ADD to return statement
return {
  // ... existing exports ...
  isConnecting,
  connectionError,
  clearConnectionError,
}
```

**Checklist**:
- [ ] Import state components in App.vue
- [ ] Add isConnecting, connectionError to store
- [ ] Update connect() to set loading/error states
- [ ] Test connection flow: loading → connected
- [ ] Test connection error: loading → error → retry

**Acceptance Criteria**:
- [ ] Shows loading spinner during connection
- [ ] Shows error message if connection fails
- [ ] Retry button works
- [ ] Empty state shown when not connected

#### Subtask 1.3.3: Add states to MeterPanel.vue (1 hour)

**File**: `frontend/src/components/MeterPanel.vue` (updated)

```vue
<template>
  <section class="meter-panel" aria-labelledby="meter-heading">
    <h3 id="meter-heading">Agent Meters</h3>

    <!-- Loading State -->
    <LoadingState
      v-if="isLoading"
      message="Waiting for agent data..."
    />

    <!-- Error State -->
    <ErrorState
      v-else-if="error"
      :message="error"
      :on-retry="retryData"
    />

    <!-- Empty State -->
    <EmptyState
      v-else-if="!meters || Object.keys(meters).length === 0"
      message="No agent data available. Start the simulation to see meters."
    />

    <!-- Data State -->
    <div v-else class="meters" role="list">
      <div
        v-for="(value, name) in meters"
        :key="name"
        class="meter"
        role="listitem"
        :class="{
          critical: isCritical(name, value),
          'strobe-slow': name === 'stress' && isLonely() && !isHighStress(),
          'strobe-fast': name === 'stress' && isLonely() && isHighStress()
        }"
        :aria-label="`${capitalize(name)}: ${formatValue(name, value)}`"
      >
        <div class="meter-header">
          <span class="meter-name">{{ capitalize(name) }}</span>
          <span
            class="meter-value"
            aria-live="polite"
            aria-atomic="true"
          >
            {{ formatValue(name, value) }}
          </span>
        </div>
        <div class="meter-bar-container">
          <div
            class="meter-bar"
            :style="{
              width: getPercentage(name, value) + '%',
              background: getMeterColor(name, value)
            }"
            role="progressbar"
            :aria-valuenow="getPercentage(name, value)"
            aria-valuemin="0"
            aria-valuemax="100"
          ></div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useSimulationStore } from '../stores/simulation'
import LoadingState from './base/LoadingState.vue'
import ErrorState from './base/ErrorState.vue'
import EmptyState from './base/EmptyState.vue'

const store = useSimulationStore()

// ADD state tracking
const isLoading = ref(false)
const error = ref(null)

const meters = computed(() => {
  const agent = store.agentMeters['agent_0']
  return agent ? agent.meters : null
})

function retryData() {
  error.value = null
  // Could trigger data refresh here
}

// ... rest of existing code ...
</script>
```

**Checklist**:
- [ ] Import state components
- [ ] Add loading/error/empty state conditions
- [ ] Add aria-live to meter values
- [ ] Add role="progressbar" to meter bars
- [ ] Test all states

**Acceptance Criteria**:
- [ ] Shows loading while waiting for data
- [ ] Shows empty state if no meters
- [ ] Meter updates announced to screen readers

#### Subtask 1.3.4: Add states to StatsPanel.vue (1 hour)

Similar to MeterPanel - add LoadingState, ErrorState, EmptyState.

**Focus on**:
- Empty state when `episodeHistory.length === 0`
- Loading state during initial data fetch
- Chart shows meaningful message when no data

**Checklist**:
- [ ] Import state components
- [ ] Handle empty episode history gracefully
- [ ] Show helpful message about starting simulation
- [ ] Test chart behavior with 0-10 episodes

**Acceptance Criteria**:
- [ ] Empty state shows before any episodes
- [ ] Chart appears smoothly as episodes arrive
- [ ] No console errors with empty data

#### Subtask 1.3.5: Add error handling to Controls.vue (1 hour)

**File**: `frontend/src/components/Controls.vue` (updated)

```vue
<script setup>
// ... existing imports ...

// ADD error handling
const commandError = ref(null)

// UPDATE connection functions with try/catch
async function connect() {
  try {
    commandError.value = null
    await store.connect(selectedMode.value)
  } catch (err) {
    commandError.value = 'Failed to connect: ' + err.message
  }
}

async function startTraining() {
  try {
    commandError.value = null
    await store.startTraining(
      trainingConfig.value.numEpisodes,
      trainingConfig.value.batchSize,
      trainingConfig.value.bufferCapacity,
      trainingConfig.value.showEvery,
      trainingConfig.value.stepDelay
    )
  } catch (err) {
    commandError.value = 'Failed to start training: ' + err.message
  }
}

// ... rest of code ...
</script>

<template>
  <div class="controls-panel">
    <h3>Controls</h3>

    <!-- ADD error display -->
    <div v-if="commandError" class="command-error" role="alert">
      <span>{{ commandError }}</span>
      <button @click="commandError = null" class="dismiss-button">✕</button>
    </div>

    <!-- ... rest of template ... -->
  </div>
</template>

<style scoped>
.command-error {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm) var(--spacing-md);
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid var(--color-error);
  border-radius: var(--border-radius-md);
  color: var(--color-error);
  font-size: var(--font-size-sm);
}

.dismiss-button {
  background: none;
  border: none;
  color: inherit;
  font-size: var(--font-size-lg);
  cursor: pointer;
  padding: 0;
  min-width: var(--a11y-min-touch-target);
  min-height: var(--a11y-min-touch-target);
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
```

**Checklist**:
- [ ] Add error state variable
- [ ] Wrap commands in try/catch
- [ ] Display errors with role="alert"
- [ ] Add dismiss button
- [ ] Test error scenarios

**Acceptance Criteria**:
- [ ] Connection errors shown to user
- [ ] Training errors shown to user
- [ ] Errors are dismissible
- [ ] Errors announced to screen readers

---

## Week 2: Architecture (14 hours)

### Task 2.1: Decouple Components from Store 🔴 HIGH PRIORITY
**Priority**: High (Architecture improvement)
**Time**: 8 hours
**Dependencies**: None (independent)

#### Subtask 2.1.1: Refactor MeterPanel to use props (1.5 hours)

**GOAL**: Remove direct store import, accept data via props

**File**: `frontend/src/components/MeterPanel.vue` (refactored)

**BEFORE**:
```vue
<script setup>
import { useSimulationStore } from '../stores/simulation'
const store = useSimulationStore()
const meters = computed(() => {
  const agent = store.agentMeters['agent_0']
  return agent ? agent.meters : null
})
</script>
```

**AFTER**:
```vue
<script setup>
import { computed } from 'vue'

// ✅ Accept data via props
const props = defineProps({
  meters: {
    type: Object,
    default: null,
    validator: (value) => {
      if (!value) return true
      // Validate meter structure
      return typeof value === 'object'
    }
  },
  agentId: {
    type: String,
    default: 'agent_0'
  }
})

// Component logic works with props, not store
const hasMeterData = computed(() => {
  return props.meters && Object.keys(props.meters).length > 0
})

// ... rest of component uses props.meters instead of store ...
</script>
```

**UPDATE**: `App.vue` to pass data to MeterPanel:

```vue
<template>
  <!-- ... -->
  <aside class="left-panel" aria-label="Agent status panels">
    <!-- Pass store data via props -->
    <MeterPanel
      :meters="agentMeters"
      agent-id="agent_0"
    />
    <StatsPanel
      :currentEpisode="currentEpisode"
      :currentStep="currentStep"
      :cumulativeReward="cumulativeReward"
      :lastAction="lastAction"
      :episodeHistory="episodeHistory"
    />
  </aside>
  <!-- ... -->
</template>

<script setup>
import { computed } from 'vue'
import { useSimulationStore } from './stores/simulation'

const store = useSimulationStore()

// Extract data from store for props
const agentMeters = computed(() => {
  const agent = store.agentMeters['agent_0']
  return agent ? agent.meters : null
})

const currentEpisode = computed(() => store.currentEpisode)
const currentStep = computed(() => store.currentStep)
const cumulativeReward = computed(() => store.cumulativeReward)
const lastAction = computed(() => store.lastAction)
const episodeHistory = computed(() => store.episodeHistory)
</script>
```

**Checklist**:
- [ ] Remove `useSimulationStore` import from MeterPanel
- [ ] Add props definition to MeterPanel
- [ ] Update App.vue to pass data via props
- [ ] Test that meters still display correctly
- [ ] Verify MeterPanel can now be tested with mock data

**Acceptance Criteria**:
- [ ] MeterPanel has no store import
- [ ] MeterPanel defines props interface
- [ ] App.vue passes all needed data via props
- [ ] Visual appearance unchanged
- [ ] Can test MeterPanel in isolation with mock data

#### Subtask 2.1.2: Refactor StatsPanel to use props (1.5 hours)

Similar to MeterPanel:

**Props to define**:
```javascript
defineProps({
  currentEpisode: { type: Number, default: 0 },
  currentStep: { type: Number, default: 0 },
  cumulativeReward: { type: Number, default: 0 },
  lastAction: { type: String, default: null },
  episodeHistory: { type: Array, default: () => [] }
})
```

**Checklist**:
- [ ] Remove store import
- [ ] Define props
- [ ] Update App.vue to pass data
- [ ] Test chart rendering
- [ ] Verify no regressions

#### Subtask 2.1.3: Refactor Grid to use props (1.5 hours)

**Props to define**:
```javascript
defineProps({
  gridWidth: { type: Number, required: true },
  gridHeight: { type: Number, required: true },
  agents: { type: Array, default: () => [] },
  affordances: { type: Array, default: () => [] },
  heatMap: { type: Object, default: () => ({}) }
})
```

**Checklist**:
- [ ] Remove store import
- [ ] Define props
- [ ] Update App.vue to pass grid data
- [ ] Test grid rendering
- [ ] Verify heat map still works

#### Subtask 2.1.4: Refactor Controls to use props + events (2 hours)

**This is more complex** - Controls needs both data IN (props) and actions OUT (events).

**Props to define**:
```javascript
defineProps({
  isConnected: { type: Boolean, default: false },
  availableModels: { type: Array, default: () => [] },
  isTraining: { type: Boolean, default: false },
  currentEpisode: { type: Number, default: 0 },
  totalEpisodes: { type: Number, default: 0 },
  trainingMetrics: {
    type: Object,
    default: () => ({
      avgReward5: 0,
      avgLength5: 0,
      avgLoss5: 0,
      epsilon: 0,
      bufferSize: 0
    })
  }
})
```

**Events to emit**:
```javascript
const emit = defineEmits([
  'connect',
  'disconnect',
  'play',
  'pause',
  'step',
  'reset',
  'setSpeed',
  'loadModel',
  'startTraining'
])
```

**Update methods to emit instead of calling store**:

**BEFORE**:
```javascript
function connect() {
  store.connect(selectedMode.value)
}
```

**AFTER**:
```javascript
function connect() {
  emit('connect', selectedMode.value)
}
```

**Update App.vue to handle events**:

```vue
<template>
  <Controls
    :is-connected="isConnected"
    :available-models="availableModels"
    :is-training="isTraining"
    :current-episode="currentEpisode"
    :total-episodes="totalEpisodes"
    :training-metrics="trainingMetrics"
    @connect="handleConnect"
    @disconnect="handleDisconnect"
    @play="handlePlay"
    @pause="handlePause"
    @step="handleStep"
    @reset="handleReset"
    @set-speed="handleSetSpeed"
    @load-model="handleLoadModel"
    @start-training="handleStartTraining"
  />
</template>

<script setup>
// Handle all control events
function handleConnect(mode) {
  store.connect(mode)
}

function handleDisconnect() {
  store.disconnect()
}

function handlePlay() {
  store.play()
}

// ... etc for all events ...
</script>
```

**Checklist**:
- [ ] Remove store import from Controls
- [ ] Define props
- [ ] Define emits
- [ ] Update all methods to emit events
- [ ] Update App.vue to handle all events
- [ ] Test all control actions work
- [ ] Verify training flow works

#### Subtask 2.1.5: Final integration testing (1.5 hours)

**Test scenarios**:
1. Connect to simulation server (both modes)
2. Play/pause/step/reset controls
3. Change speed slider
4. Load different models
5. Start training
6. Monitor training metrics
7. Disconnect

**Create test utilities** for mock data:

**File**: `frontend/src/utils/mockData.js` (NEW)

```javascript
/**
 * Mock data utilities for testing components in isolation
 */

export const mockMeters = {
  energy: 0.8,
  hygiene: 0.6,
  satiation: 0.7,
  money: 50,
  stress: 30,
  social: 0.9
}

export const mockEpisodeHistory = [
  { episode: 1, steps: 45, reward: 12.3 },
  { episode: 2, steps: 52, reward: 15.1 },
  { episode: 3, steps: 38, reward: 9.8 },
  { episode: 4, steps: 61, reward: 18.4 },
  { episode: 5, steps: 55, reward: 16.2 }
]

export const mockAgents = [
  { id: 'agent_0', x: 3, y: 4, color: '#10b981' }
]

export const mockAffordances = [
  { x: 1, y: 1, type: 'Bed' },
  { x: 5, y: 2, type: 'Shower' },
  { x: 3, y: 6, type: 'HomeMeal' }
]

export const mockTrainingMetrics = {
  avgReward5: 14.2,
  avgLength5: 48.5,
  avgLoss5: 0.0234,
  epsilon: 0.456,
  bufferSize: 2341
}
```

**Checklist**:
- [ ] Create mockData.js
- [ ] Test MeterPanel with mock meters
- [ ] Test StatsPanel with mock history
- [ ] Test Grid with mock agents
- [ ] Test Controls with mock state
- [ ] Verify all props interfaces documented

**Acceptance Criteria**:
- [ ] All 5 components decoupled from store
- [ ] Only App.vue imports useSimulationStore
- [ ] Components can be tested with mock data
- [ ] All functionality still works
- [ ] No visual regressions

---

### Task 2.2: Implement Mobile Responsiveness 🟡 HIGH PRIORITY
**Priority**: High (Usability on mobile)
**Time**: 6 hours
**Dependencies**: Task 1.1 (design tokens)

#### Subtask 2.2.1: Add responsive breakpoints to App.vue (2 hours)

**Goal**: Make layout responsive - panels stack on mobile

**File**: `frontend/src/components/App.vue` (add responsive CSS)

**Current layout**: 3-column (fixed widths)
```
┌────────┬──────────┬────────┐
│ Left   │  Grid    │ Right  │
│ 320px  │  flex    │ 380px  │
└────────┴──────────┴────────┘
```

**Mobile layout**: Stacked vertically
```
┌──────────────────┐
│     Header       │
├──────────────────┤
│    Controls      │  (most important on mobile)
├──────────────────┤
│      Grid        │
├──────────────────┤
│     Meters       │
├──────────────────┤
│     Stats        │
└──────────────────┘
```

**AFTER** (add media queries):

```vue
<template>
  <div class="app">
    <header class="header" role="banner">
      <!-- Add hamburger menu for mobile -->
      <button
        class="mobile-menu-button"
        @click="mobileMenuOpen = !mobileMenuOpen"
        :aria-expanded="mobileMenuOpen"
        aria-label="Toggle menu"
      >
        ☰
      </button>

      <h1>HAMLET</h1>  <!-- Shorter title on mobile -->

      <div class="connection-status" :class="{ connected: isConnected }" role="status">
        {{ isConnected ? '●' : '○' }}  <!-- Icon only on mobile -->
        <span class="connection-text">{{ isConnected ? 'Connected' : 'Disconnected' }}</span>
      </div>
    </header>

    <main class="main-container" role="main">
      <!-- On mobile, controls come first -->
      <aside
        class="right-panel"
        aria-label="Simulation controls"
        :class="{ 'mobile-hidden': !mobileMenuOpen }"
      >
        <Controls
          :is-connected="isConnected"
          @connect="handleConnect"
          @disconnect="handleDisconnect"
          <!-- ... events ... -->
        />
      </aside>

      <div class="grid-container" role="region" aria-label="Simulation grid">
        <Grid
          :grid-width="gridWidth"
          :grid-height="gridHeight"
          :agents="agents"
          :affordances="affordances"
          :heat-map="heatMap"
        />
      </div>

      <aside
        class="left-panel"
        aria-label="Agent status panels"
        :class="{ 'mobile-hidden': !mobileMenuOpen }"
      >
        <MeterPanel :meters="agentMeters" agent-id="agent_0" />
        <StatsPanel
          :current-episode="currentEpisode"
          :current-step="currentStep"
          :cumulative-reward="cumulativeReward"
          :last-action="lastAction"
          :episode-history="episodeHistory"
        />
      </aside>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue'
// ... existing imports ...

const mobileMenuOpen = ref(false)
</script>

<style scoped>
/* Mobile-first base styles */
.app {
  width: 100%;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  overflow: hidden;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-md);  /* Reduced on mobile */
  border-bottom: 1px solid var(--color-bg-tertiary);
  position: sticky;
  top: 0;
  z-index: 10;
  background: var(--color-bg-primary);
}

.mobile-menu-button {
  display: flex;  /* Visible on mobile */
  align-items: center;
  justify-content: center;
  min-width: var(--a11y-min-touch-target);
  min-height: var(--a11y-min-touch-target);
  background: none;
  border: none;
  color: var(--color-text-primary);
  font-size: var(--font-size-2xl);
  cursor: pointer;
}

.header h1 {
  margin: 0;
  font-size: var(--font-size-lg);  /* Smaller on mobile */
  font-weight: var(--font-weight-semibold);
}

.connection-status {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--border-radius-md);
  background: var(--color-error);
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-medium);
}

.connection-status.connected {
  background: var(--color-success);
}

.connection-text {
  display: none;  /* Hidden on mobile */
}

/* Mobile layout: Stack vertically */
.main-container {
  flex: 1;
  display: flex;
  flex-direction: column;  /* Stack on mobile */
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  overflow-y: auto;
}

.left-panel,
.right-panel {
  width: 100%;  /* Full width on mobile */
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  transition: var(--transition-base);
}

.mobile-hidden {
  display: none;  /* Can be toggled via hamburger menu */
}

.grid-container {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  min-height: 300px;  /* Smaller on mobile */
}

/* ========================================
   TABLET: 768px and up
   ======================================== */
@media (min-width: 768px) {
  .header {
    padding: var(--spacing-md) var(--spacing-xl);
  }

  .mobile-menu-button {
    display: none;  /* Hide hamburger on tablet+ */
  }

  .header h1 {
    font-size: var(--font-size-2xl);  /* Full title */
  }

  .connection-text {
    display: inline;  /* Show full text */
  }

  .main-container {
    padding: var(--spacing-lg);
  }

  .left-panel,
  .right-panel {
    display: flex !important;  /* Always visible */
  }

  .grid-container {
    padding: var(--spacing-xl);
    min-height: 400px;
  }
}

/* ========================================
   DESKTOP: 1024px and up
   ======================================== */
@media (min-width: 1024px) {
  .main-container {
    flex-direction: row;  /* Side-by-side layout */
    padding: var(--spacing-xl);
    overflow: hidden;
  }

  .left-panel {
    width: var(--layout-left-panel-width);
    overflow-y: auto;
  }

  .grid-container {
    flex: 1;
    min-width: 0;  /* Allow flex shrinking */
  }

  .right-panel {
    width: var(--layout-right-panel-width);
    overflow-y: auto;
  }
}

/* ========================================
   LARGE DESKTOP: 1280px and up
   ======================================== */
@media (min-width: 1280px) {
  .main-container {
    gap: var(--spacing-2xl);
  }
}
</style>
```

**Checklist**:
- [ ] Add mobile-first base styles
- [ ] Add hamburger menu button
- [ ] Stack panels vertically on mobile
- [ ] Add @media queries for tablet (768px)
- [ ] Add @media queries for desktop (1024px)
- [ ] Test on mobile emulator (iPhone SE, Pixel 5)
- [ ] Test on tablet emulator (iPad)
- [ ] Test on desktop

**Acceptance Criteria**:
- [ ] Layout works on 320px width
- [ ] Panels stack on mobile, side-by-side on desktop
- [ ] Hamburger menu toggles panels on mobile
- [ ] All touch targets 44px+
- [ ] No horizontal scrolling at any size

#### Subtask 2.2.2: Make Grid responsive (1.5 hours)

**File**: `frontend/src/components/Grid.vue` (add responsive sizing)

**Goal**: Grid scales down on mobile, readable at all sizes

```vue
<script setup>
import { computed } from 'vue'
import { useWindowSize } from '@vueuse/core'

const props = defineProps({
  gridWidth: { type: Number, required: true },
  gridHeight: { type: Number, required: true },
  // ... other props ...
})

// ADD responsive cell sizing
const { width: windowWidth } = useWindowSize()

const cellSize = computed(() => {
  // Mobile: smaller cells
  if (windowWidth.value < 640) {
    return 40  // 40px cells on mobile
  }
  // Tablet: medium cells
  if (windowWidth.value < 1024) {
    return 60  // 60px cells on tablet
  }
  // Desktop: full size cells
  return 75  // 75px cells on desktop
})

const svgWidth = computed(() => props.gridWidth * cellSize.value)
const svgHeight = computed(() => props.gridHeight * cellSize.value)
</script>

<style scoped>
.grid-container {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.heat-map-toggle {
  position: absolute;
  top: var(--spacing-sm);
  right: var(--spacing-sm);
  min-height: var(--a11y-min-touch-target);  /* Touch-friendly */
  min-width: var(--a11y-min-touch-target);
  padding: var(--spacing-xs) var(--spacing-md);
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-bg-tertiary);
  border-radius: var(--border-radius-sm);
  font-size: var(--font-size-xs);  /* Smaller on mobile */
  font-weight: var(--font-weight-medium);
  cursor: pointer;
  transition: var(--transition-base);
  z-index: 10;
}

.heat-map-toggle:hover {
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
}

.heat-map-toggle.active {
  background: var(--color-interactive);
  color: var(--color-text-primary);
  border-color: var(--color-interactive-hover);
}

.grid-svg {
  width: 100%;
  height: 100%;
  max-width: 90vw;  /* Responsive max width */
  max-height: 50vh;  /* Don't take full mobile screen */
}

/* ========================================
   TABLET: 768px and up
   ======================================== */
@media (min-width: 768px) {
  .heat-map-toggle {
    top: var(--spacing-md);
    right: var(--spacing-md);
    font-size: var(--font-size-sm);
  }

  .grid-svg {
    max-width: 600px;  /* Standard desktop size */
    max-height: 600px;
  }
}
</style>
```

**Checklist**:
- [ ] Install @vueuse/core: `npm install @vueuse/core`
- [ ] Add responsive cell sizing
- [ ] Make SVG responsive
- [ ] Ensure heat map button is touch-friendly
- [ ] Test grid at different sizes

**Acceptance Criteria**:
- [ ] Grid readable on mobile (320px)
- [ ] Cell size scales appropriately
- [ ] Heat map toggle is 44px+
- [ ] Grid centered and doesn't overflow

#### Subtask 2.2.3: Make components touch-friendly (1 hour)

**Goal**: All interactive elements are 44px+ on all screen sizes

**Files to update**: Controls.vue, MeterPanel.vue, StatsPanel.vue

**Key changes**:
```css
/* Ensure buttons are always 44px+ */
button,
.button,
.control-button,
.mode-button {
  min-height: var(--a11y-min-touch-target);
  min-width: var(--a11y-min-touch-target);
  /* Add comfortable tap spacing */
  margin: var(--spacing-xs);
}

/* Larger tap areas on mobile */
@media (max-width: 767px) {
  button,
  .control-button {
    padding: var(--spacing-md);  /* More padding on mobile */
  }
}

/* Range sliders touch-friendly */
input[type="range"] {
  min-height: var(--a11y-min-touch-target);
  cursor: pointer;
  -webkit-tap-highlight-color: rgba(16, 185, 129, 0.3);
}
```

**Checklist**:
- [ ] Verify all buttons are 44px+
- [ ] Add comfortable spacing around interactive elements
- [ ] Test tap highlight color on mobile
- [ ] Ensure controls don't overlap

**Acceptance Criteria**:
- [ ] All interactive elements 44px+
- [ ] Easy to tap on mobile
- [ ] No accidental taps on adjacent controls

#### Subtask 2.2.4: Test on real devices (1.5 hours)

**Testing matrix**:
| Device | Screen Size | Browser | Result |
|--------|-------------|---------|--------|
| iPhone SE | 375 x 667 | Safari | |
| iPhone 12 | 390 x 844 | Safari | |
| Pixel 5 | 393 x 851 | Chrome | |
| iPad | 768 x 1024 | Safari | |
| Desktop | 1920 x 1080 | Chrome | |

**Test scenarios**:
1. **Mobile (320-767px)**
   - [ ] Header fits, hamburger menu works
   - [ ] Panels stack vertically
   - [ ] Grid is readable
   - [ ] All controls accessible via menu
   - [ ] Touch targets easy to hit

2. **Tablet (768-1023px)**
   - [ ] Panels visible (no hamburger menu)
   - [ ] Grid larger and clearer
   - [ ] Two-column or stacked layout works

3. **Desktop (1024px+)**
   - [ ] Three-column layout
   - [ ] Grid full size
   - [ ] All panels visible simultaneously

**Checklist**:
- [ ] Test on Chrome DevTools mobile emulator
- [ ] Test on real iPhone (if available)
- [ ] Test on real Android (if available)
- [ ] Test on iPad (if available)
- [ ] Take screenshots at each breakpoint
- [ ] Document any issues found

**Acceptance Criteria**:
- [ ] All layouts work at all breakpoints
- [ ] No horizontal scrolling anywhere
- [ ] Touch controls work on real devices
- [ ] Performance acceptable on mobile

---

## Week 3: Polish (3 hours)

### Task 3.1: Clean Up Template Logic 🟢 MEDIUM PRIORITY
**Priority**: Medium (Code quality)
**Time**: 3 hours
**Dependencies**: None

#### Subtask 3.1.1: Extract toFixed() calls (1 hour)

**Goal**: Move all formatting logic from templates to computed properties

**Example**: `StatsPanel.vue`

**BEFORE**:
```vue
<template>
  <span class="stat-value">
    {{ cumulativeReward.toFixed(1) }}
  </span>
</template>
```

**AFTER**:
```vue
<template>
  <span class="stat-value">
    {{ formattedReward }}
  </span>
</template>

<script setup>
// Extract constant
const REWARD_DECIMAL_PLACES = 1

const formattedReward = computed(() => {
  return props.cumulativeReward.toFixed(REWARD_DECIMAL_PLACES)
})
</script>
```

**Checklist for StatsPanel**:
- [ ] Extract `cumulativeReward.toFixed(1)` → `formattedReward`
- [ ] Extract reward formatting constant
- [ ] Ensure all numeric displays use computed properties

**Repeat for**:
- [ ] Controls.vue (training metrics formatting)
- [ ] MeterPanel.vue (meter value formatting)

**Acceptance Criteria**:
- [ ] Zero `.toFixed()` calls in templates
- [ ] All formatting in computed properties
- [ ] Constants defined for decimal places

#### Subtask 3.1.2: Extract magic numbers (1 hour)

**Goal**: Create constants file for all magic numbers

**File**: `frontend/src/utils/constants.js` (NEW)

```javascript
/**
 * Application constants
 */

// Chart configuration
export const CHART_HISTORY_LENGTH = 10
export const CHART_WIDTH = 300
export const CHART_HEIGHT = 100

// Formatting
export const REWARD_DECIMAL_PLACES = 1
export const METRICS_DECIMAL_PLACES = 2
export const LOSS_DECIMAL_PLACES = 4
export const EPSILON_DECIMAL_PLACES = 3

// Training defaults
export const DEFAULT_EPISODES = 100
export const DEFAULT_BATCH_SIZE = 32
export const DEFAULT_BUFFER_CAPACITY = 10000
export const DEFAULT_SHOW_EVERY = 1
export const DEFAULT_STEP_DELAY = 0.2

// Meter thresholds
export const METER_THRESHOLD_CRITICAL = 20
export const METER_THRESHOLD_WARNING = 60
export const STRESS_THRESHOLD_HIGH = 80
export const PERCENTAGE_MULTIPLIER = 100

// Grid
export const DEFAULT_GRID_WIDTH = 8
export const DEFAULT_GRID_HEIGHT = 8
export const CELL_SIZE_MOBILE = 40
export const CELL_SIZE_TABLET = 60
export const CELL_SIZE_DESKTOP = 75
```

**Update components to use constants**:

```vue
<script setup>
import { CHART_HISTORY_LENGTH, REWARD_DECIMAL_PLACES } from '@/utils/constants'

const chartData = computed(() => {
  const history = props.episodeHistory.slice(-CHART_HISTORY_LENGTH)
  // ... use constant instead of hardcoded 10 ...
})

const formattedReward = computed(() => {
  return props.cumulativeReward.toFixed(REWARD_DECIMAL_PLACES)
})
</script>
```

**Checklist**:
- [ ] Create constants.js
- [ ] Extract all magic numbers from components
- [ ] Update components to import and use constants
- [ ] Update Controls.vue training defaults
- [ ] Update StatsPanel.vue chart constants
- [ ] Update MeterPanel.vue threshold constants

**Acceptance Criteria**:
- [ ] All magic numbers extracted to constants
- [ ] Constants file well-documented
- [ ] All components import and use constants
- [ ] Behavior unchanged

#### Subtask 3.1.3: Extract complex computed logic (1 hour)

**Goal**: Move complex logic to utility functions

**File**: `frontend/src/utils/formatting.js` (NEW)

```javascript
/**
 * Formatting utilities
 */

import {
  REWARD_DECIMAL_PLACES,
  METRICS_DECIMAL_PLACES,
  LOSS_DECIMAL_PLACES,
  EPSILON_DECIMAL_PLACES,
  PERCENTAGE_MULTIPLIER
} from './constants'

export function formatReward(value) {
  return value.toFixed(REWARD_DECIMAL_PLACES)
}

export function formatMetric(value) {
  return value.toFixed(METRICS_DECIMAL_PLACES)
}

export function formatLoss(value) {
  return value.toFixed(LOSS_DECIMAL_PLACES)
}

export function formatEpsilon(value) {
  return value.toFixed(EPSILON_DECIMAL_PLACES)
}

export function formatPercentage(value) {
  return Math.round(value * PERCENTAGE_MULTIPLIER)
}

export function getMeterStatusClass(name, value, thresholds) {
  const percentage = name === 'money' || name === 'stress'
    ? value
    : value * PERCENTAGE_MULTIPLIER

  if (name === 'stress') {
    return percentage > thresholds.high ? 'critical'
         : percentage > thresholds.mid ? 'warning'
         : 'good'
  }

  return percentage < thresholds.critical ? 'critical'
       : percentage < thresholds.warning ? 'warning'
       : 'good'
}
```

**File**: `frontend/src/utils/actions.js` (NEW)

```javascript
/**
 * Action formatting utilities
 */

const ACTION_ICONS = {
  'up': '↑ Up',
  'down': '↓ Down',
  'left': '← Left',
  'right': '→ Right',
  'interact': '⚡ Interact'
}

export function formatAction(action) {
  if (!action) return '—'
  return ACTION_ICONS[action] || action
}

export function getAffordanceIcon(type) {
  const icons = {
    'Bed': '🛏️',
    'Shower': '🚿',
    'HomeMeal': '🥘',
    'FastFood': '🍔',
    'Job': '💼',
    'Recreation': '🎮',
    'Bar': '🍺'
  }
  return icons[type] || '?'
}
```

**Update components to use utilities**:

```vue
<script setup>
import { formatReward, formatMetric } from '@/utils/formatting'
import { formatAction } from '@/utils/actions'

// Simple computed properties using utilities
const formattedReward = computed(() => formatReward(props.cumulativeReward))
const formattedMetric = computed(() => formatMetric(props.avgReward5))
const actionText = computed(() => formatAction(props.lastAction))
</script>
```

**Checklist**:
- [ ] Create formatting.js utilities
- [ ] Create actions.js utilities
- [ ] Update StatsPanel to use utilities
- [ ] Update Controls to use utilities
- [ ] Update MeterPanel to use utilities
- [ ] Update Grid to use utilities

**Acceptance Criteria**:
- [ ] All formatting logic extracted
- [ ] Components use simple, readable computed properties
- [ ] Utilities are well-tested
- [ ] Code is DRY (Don't Repeat Yourself)

---

## Testing Checklist

After completing all tasks, run through this comprehensive checklist:

### Functional Testing
- [ ] WebSocket connection works (inference mode)
- [ ] WebSocket connection works (training mode)
- [ ] Play/pause/step/reset controls work
- [ ] Speed slider adjusts playback speed
- [ ] Model switching works
- [ ] Training starts and shows progress
- [ ] Training metrics update in real-time
- [ ] Heat map toggle works
- [ ] All meters update in real-time
- [ ] Episode stats chart updates
- [ ] Disconnect works properly

### Accessibility Testing
- [ ] All buttons keyboard accessible (Tab navigation)
- [ ] Spacebar plays/pauses simulation
- [ ] Arrow keys work for controls
- [ ] Screen reader announces real-time updates (VoiceOver/NVDA)
- [ ] All images have alt text or aria-label
- [ ] Color contrast passes WCAG AA
- [ ] Focus indicators visible on all elements
- [ ] Form labels properly associated with inputs

### Responsive Testing
- [ ] Works at 320px width (iPhone SE)
- [ ] Works at 375px width (iPhone 12)
- [ ] Works at 768px width (iPad portrait)
- [ ] Works at 1024px width (iPad landscape)
- [ ] Works at 1920px width (desktop)
- [ ] No horizontal scrolling at any size
- [ ] Touch targets 44px+ on mobile
- [ ] Grid readable at all sizes
- [ ] Hamburger menu works on mobile

### Visual Regression Testing
- [ ] All colors use design tokens
- [ ] All spacing uses design tokens
- [ ] Typography consistent throughout
- [ ] Meters display correctly
- [ ] Chart renders properly
- [ ] Grid visualization accurate
- [ ] No layout shifts during loading

### Error Handling Testing
- [ ] Connection failure shows error + retry
- [ ] Training failure shows error message
- [ ] Invalid input shows validation errors
- [ ] WebSocket disconnect handled gracefully
- [ ] Loading states show during operations

### Performance Testing
- [ ] Initial load time < 2 seconds
- [ ] Smooth animations (60fps)
- [ ] No lag during real-time updates
- [ ] Grid renders smoothly
- [ ] Chart updates don't cause jank

---

## Acceptance Criteria

### Overall Success Metrics

**Before** (Current State):
- Compliance Score: 55% (C-)
- Design Tokens: 0%
- Accessibility: 5%
- State Management: 20%
- Error States: 10%
- Template Cleanliness: 55%
- Responsive Design: 5%

**After** (Target State):
- Compliance Score: ≥85% (B+)
- Design Tokens: 100%
- Accessibility: ≥90%
- State Management: ≥85%
- Error States: ≥90%
- Template Cleanliness: ≥85%
- Responsive Design: ≥90%

### Definition of Done

A task is considered complete when:
1. ✅ All subtasks checked off
2. ✅ All acceptance criteria met
3. ✅ Manual testing passed
4. ✅ No regressions introduced
5. ✅ Code reviewed (self-review against skill)
6. ✅ Documentation updated (if needed)

### Final Deliverables

Upon completion of all tasks:
- [ ] All components use design tokens
- [ ] All components are accessible (WCAG AA)
- [ ] All components decoupled from store (Props First)
- [ ] All components handle loading/error/empty states
- [ ] All templates are clean (no business logic)
- [ ] All layouts responsive (mobile-first)
- [ ] All touch targets ≥44px
- [ ] All magic numbers extracted
- [ ] Complete test coverage (manual)
- [ ] Updated README with testing instructions
- [ ] Git commit with comprehensive changes
- [ ] Optional: Screenshot comparison (before/after)

---

## Progress Tracking

Use this table to track progress:

| Task | Priority | Time | Status | Completed |
|------|----------|------|--------|-----------|
| 1.1 Design Tokens | 🔴 | 4h | ☐ | YYYY-MM-DD |
| 1.2 Accessibility | 🔴 | 8h | ☐ | YYYY-MM-DD |
| 1.3 Error States | 🟡 | 6h | ☐ | YYYY-MM-DD |
| 2.1 Decouple Components | 🔴 | 8h | ☐ | YYYY-MM-DD |
| 2.2 Mobile Responsive | 🟡 | 6h | ☐ | YYYY-MM-DD |
| 3.1 Template Cleanup | 🟢 | 3h | ☐ | YYYY-MM-DD |

**Total**: 35 hours estimated

---

## Notes

- This plan is comprehensive but flexible - adjust as needed
- Prioritize blockers first (design tokens, accessibility)
- Test incrementally - don't wait until the end
- Commit after each major task
- Ask for help if blocked for >30 minutes
- Celebrate small wins! 🎉

---

## References

- [Frontend Audit Report](frontend-audit-report.md) - Original audit findings
- [vue3-ux-design Skill](/.claude/skills/vue3-ux-design.md) - Skill definition
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/) - Accessibility standards
- [Vue 3 Docs](https://vuejs.org/) - Vue documentation
- [Pinia Docs](https://pinia.vuejs.org/) - State management
