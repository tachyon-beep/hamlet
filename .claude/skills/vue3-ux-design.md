---
name: vue3-ux-design
version: 1.0.0
stack: Vue 3.x + Vite + Pinia
description: Use when building Vue 3 components, creating dashboards, implementing forms, optimizing data-heavy interfaces, or making designs responsive. Mandates design tokens, accessibility, proper state management, error states, and template cleanliness for maintainable UX.
---

<EXTREMELY_IMPORTANT>
You MUST read the entire skill. Every section exists because agents failed without it.
Skipping sections = guaranteed failure. This skill is 900+ lines because each line prevents a real mistake.
</EXTREMELY_IMPORTANT>

# Vue 3 UX Design - Production Quality Standards

## When This Skill Applies

**MANDATORY for**:
- Building any Vue 3 component
- Creating real-time dashboards or data visualizations
- Implementing forms with validation
- Rendering large datasets (tables, lists)
- Making interfaces mobile-responsive
- Any UI work in Vue 3 + Vite + Pinia stack

**Not optional. If you're writing Vue 3 code, you use this skill.**

---

## Critical Rule: Time Pressure is NOT Permission

<EXTREMELY_IMPORTANT>
If you catch yourself thinking ANY of these thoughts, STOP. You are rationalizing away quality.

❌ "Time pressure means skip error handling"
❌ "This is simple enough for inline styles"
❌ "I'm tired, here's the quick fix"
❌ "Users expect simple interface" (as excuse to skip semantic HTML)
❌ "Performance is critical" (as excuse for premature optimization)
❌ "This is just a prototype"

**Users don't care about your timeline or fatigue. They need working, accessible software.**
</EXTREMELY_IMPORTANT>

---

## Rationalization Counter Table

**Before skipping ANY step, check this table:**

| If You're Thinking... | Why You're Wrong | What You Must Do |
|----------------------|------------------|------------------|
| "Time pressure means skip error handling" | Users hit errors immediately. No error handling = broken software | Add try/catch and loading states FIRST, before any features |
| "This is simple enough for inline styles" | Every "simple" component becomes complex. Refactoring later is 10x harder | Extract to computed properties and design tokens from the start |
| "I'm tired, here's the quick fix" | Technical debt is permanent. Your "quick fix" ships to production | Take a break OR follow the process. No shortcuts. |
| "Users expect simple interface" | Simple UX requires complex implementation. "Simple" ≠ skip accessibility | Semantic HTML + ARIA are non-negotiable regardless of "simplicity" |
| "Performance is critical so I'll optimize now" | Premature optimization creates unmaintainable code | Profile first with Vue DevTools, optimize second with data |
| "Grid CSS is straightforward" | CSS complexity is invisible until debugging at 2am | Use design tokens and extract magic numbers before writing CSS |
| "This is just a prototype" | Prototypes become production. Always. | Build it right the first time or don't build it |
| "Store makes this simpler than props" | Tight coupling prevents reuse and testing | Props + callbacks first. Store only for truly global state. |

**If a rationalization isn't in this table, it's still wrong. Check yourself.**

---

## Phase 1: Design Tokens BEFORE Any Code

**DO THIS FIRST. Not optional. Not later. NOW.**

### 1.1 Create Design Token File

Before writing ANY component code, create or update your design tokens file:

**File**: `src/styles/tokens.js`

```javascript
// Design Tokens - Single source of truth for design decisions
export const tokens = {
  // Colors - Semantic names, not visual descriptions
  colors: {
    // Surfaces
    backgroundPrimary: '#1e1e2e',
    backgroundSecondary: '#2a2a3e',
    backgroundTertiary: '#3a3a4e',

    // Text
    textPrimary: '#e0e0e0',
    textSecondary: '#a0a0b0',
    textTertiary: '#808090',

    // Interactive
    interactive: '#10b981',
    interactiveHover: '#34d399',
    interactiveFocus: '#6ee7b7',

    // Status
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#06b6d4',
  },

  // Spacing - Consistent rhythm
  spacing: {
    xs: '0.25rem',   // 4px
    sm: '0.5rem',    // 8px
    md: '1rem',      // 16px
    lg: '1.5rem',    // 24px
    xl: '2rem',      // 32px
    '2xl': '3rem',   // 48px
  },

  // Typography
  fontSize: {
    xs: '0.75rem',   // 12px
    sm: '0.875rem',  // 14px
    base: '1rem',    // 16px
    lg: '1.125rem',  // 18px
    xl: '1.25rem',   // 20px
    '2xl': '1.5rem', // 24px
  },

  fontWeight: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },

  // Layout
  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '12px',
    full: '9999px',
  },

  // Transitions
  transition: {
    fast: '0.15s ease',
    base: '0.3s ease',
    slow: '0.5s ease',
  },

  // Responsive breakpoints
  breakpoints: {
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
  },

  // Accessibility
  touchTarget: {
    minSize: '44px',  // WCAG minimum
  },
}
```

### 1.2 Import Tokens in Components

```vue
<script setup>
import { tokens } from '@/styles/tokens'

// Use tokens for any dynamic styling
const cardStyle = computed(() => ({
  backgroundColor: tokens.colors.backgroundSecondary,
  padding: tokens.spacing.lg,
  borderRadius: tokens.borderRadius.md,
}))
</script>
```

### 1.3 Use CSS Custom Properties

**File**: `src/styles/variables.css`

```css
:root {
  /* Import from tokens.js or define here */
  --color-bg-primary: #1e1e2e;
  --color-bg-secondary: #2a2a3e;
  --color-text-primary: #e0e0e0;

  --spacing-sm: 0.5rem;
  --spacing-md: 1rem;
  --spacing-lg: 1.5rem;

  --font-size-base: 1rem;
  --font-size-sm: 0.875rem;

  --border-radius-md: 8px;
  --transition-base: 0.3s ease;
}
```

**Why**: Magic numbers scattered in code are unmaintainable. One source of truth.

**Failure Mode**: Skipping this because "it's extra work upfront" → leads to inconsistent spacing, colors everywhere, impossible to theme.

---

## Phase 2: Accessibility Checklist (NON-NEGOTIABLE)

**Every component MUST pass this checklist. No exceptions.**

### 2.1 Semantic HTML First

❌ **NEVER**:
```vue
<div class="button" @click="submit">Submit</div>
<div class="input-wrapper">
  <div class="label">Name</div>
  <div class="input" @input="..."></div>
</div>
```

✅ **ALWAYS**:
```vue
<button type="submit" @click="submit">Submit</button>
<form @submit.prevent="handleSubmit">
  <label for="name">Name</label>
  <input id="name" type="text" v-model="name" />
</form>
```

**Semantic elements**: `<button>`, `<form>`, `<label>`, `<input>`, `<select>`, `<nav>`, `<main>`, `<article>`, `<section>`, `<header>`, `<footer>`, `<aside>`

**Why**: Screen readers rely on semantic HTML. Div soup is inaccessible.

### 2.2 ARIA Attributes

**Interactive elements**:
```vue
<!-- Buttons with icons only -->
<button aria-label="Close dialog" @click="close">
  <XIcon />
</button>

<!-- Toggle buttons -->
<button
  :aria-pressed="isActive"
  @click="toggle"
>
  Toggle Feature
</button>

<!-- Form validation -->
<input
  id="email"
  type="email"
  v-model="email"
  :aria-invalid="hasError"
  :aria-describedby="hasError ? 'email-error' : undefined"
/>
<span v-if="hasError" id="email-error" role="alert">
  {{ errorMessage }}
</span>
```

**Data tables**:
```vue
<table>
  <thead>
    <tr>
      <th scope="col" role="columnheader" :aria-sort="sortState">
        Episode
      </th>
    </tr>
  </thead>
</table>
```

**Live regions** (for real-time updates):
```vue
<div aria-live="polite" aria-atomic="true">
  Episode {{ currentEpisode }} completed: {{ reward }} reward
</div>
```

### 2.3 Keyboard Navigation

**All interactive elements MUST be keyboard accessible**:

```vue
<div
  role="button"
  tabindex="0"
  @click="handleClick"
  @keydown.enter="handleClick"
  @keydown.space.prevent="handleClick"
>
  Custom Button
</div>
```

**Focus management**:
```javascript
const firstInputRef = ref(null)

onMounted(() => {
  // Focus first input when dialog opens
  firstInputRef.value?.focus()
})
```

### 2.4 Touch Targets

**Minimum 44x44px for all interactive elements (WCAG guideline)**:

```css
.button {
  min-height: 44px;
  min-width: 44px;
  padding: var(--spacing-sm) var(--spacing-md);
}
```

### 2.5 Color Contrast

**WCAG AA minimum**: 4.5:1 for normal text, 3:1 for large text

```javascript
// Bad - low contrast
color: { bg: '#2a2a3e', text: '#4a4a5e' }  // Fails WCAG

// Good - sufficient contrast
color: { bg: '#2a2a3e', text: '#e0e0e0' }  // Passes WCAG AA
```

**Check contrast**: Use browser DevTools or https://webaim.org/resources/contrastchecker/

### 2.6 Test With Screen Readers

**Before marking component complete**:
- [ ] Test with VoiceOver (macOS) or NVDA (Windows)
- [ ] Navigate with keyboard only (no mouse)
- [ ] Verify all interactive elements are reachable
- [ ] Verify form errors are announced
- [ ] Verify dynamic content changes are announced

**If you skip this, your component is incomplete.**

---

## Phase 3: State Management Patterns

### 3.1 Decision Tree: Pinia vs Props vs Composables

```
Is the state needed by 3+ unrelated components?
├─ Yes → Consider Pinia store
└─ No ↓

Is the state specific to this component tree?
├─ Yes → Use props + events
└─ No ↓

Is the state reusable logic with reactivity?
├─ Yes → Create composable
└─ No → Local ref/reactive
```

### 3.2 Props First (Decouple from Store)

❌ **NEVER** (tight coupling):
```vue
<script setup>
import { useSimulationStore } from '@/stores/simulation'

const store = useSimulationStore()
const data = computed(() => store.episodeHistory)
</script>
```

✅ **ALWAYS** (loose coupling):
```vue
<script setup>
// Component accepts data via props
defineProps({
  episodes: {
    type: Array,
    required: true
  }
})
</script>

<!-- Parent passes data from store -->
<EpisodeTable :episodes="store.episodeHistory" />
```

**Why**: Component can be reused with different data sources, tested with mock data, moved to different projects.

### 3.3 When to Use Pinia

**✅ Good use cases**:
- Authentication state (current user, token)
- WebSocket connection status
- Application-wide settings (theme, locale)
- Large shared datasets (cached API responses)

**❌ Bad use cases**:
- Component-local state (use `ref`)
- Parent-child communication (use props/events)
- Form state (use local state + validation library)
- Derived data (use computed props)

### 3.4 Composables for Reusable Logic

```javascript
// composables/useWebSocket.js
export function useWebSocket(url) {
  const isConnected = ref(false)
  const error = ref(null)
  const data = ref(null)

  const connect = () => {
    // WebSocket logic
  }

  const disconnect = () => {
    // Cleanup
  }

  onUnmounted(() => {
    disconnect()
  })

  return {
    isConnected,
    error,
    data,
    connect,
    disconnect,
  }
}
```

**Use in component**:
```vue
<script setup>
import { useWebSocket } from '@/composables/useWebSocket'

const { isConnected, data, connect } = useWebSocket('ws://localhost:8765')
</script>
```

---

## Phase 4: Error/Loading/Empty States (The 3-State Pattern)

**EVERY data operation needs 3 states. No exceptions.**

### 4.1 The Template Pattern

```vue
<template>
  <div class="data-container">
    <!-- Loading State -->
    <div v-if="isLoading" class="loading-state">
      <SpinnerIcon />
      <p>Loading episodes...</p>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="error-state" role="alert">
      <AlertIcon />
      <p>{{ error.message }}</p>
      <button @click="retry">Retry</button>
    </div>

    <!-- Empty State -->
    <div v-else-if="!data || data.length === 0" class="empty-state">
      <InboxIcon />
      <p>No episodes yet. Start training to see results.</p>
    </div>

    <!-- Data State -->
    <div v-else class="data-state">
      <!-- Render actual data -->
    </div>
  </div>
</template>
```

### 4.2 The Script Pattern

```javascript
const isLoading = ref(false)
const error = ref(null)
const data = ref(null)

async function fetchData() {
  isLoading.value = true
  error.value = null

  try {
    const response = await fetch('/api/episodes')
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    data.value = await response.json()
  } catch (err) {
    error.value = err
    console.error('Failed to fetch episodes:', err)
  } finally {
    isLoading.value = false
  }
}

function retry() {
  fetchData()
}
```

### 4.3 Store Data Pattern

**If using Pinia store**, expose loading/error states:

```javascript
// stores/simulation.js
export const useSimulationStore = defineStore('simulation', () => {
  const episodes = ref([])
  const isLoading = ref(false)
  const error = ref(null)

  async function fetchEpisodes() {
    isLoading.value = true
    error.value = null
    try {
      // fetch logic
    } catch (err) {
      error.value = err
    } finally {
      isLoading.value = false
    }
  }

  return {
    episodes,
    isLoading,
    error,
    fetchEpisodes,
  }
})
```

**Why**: Users need feedback. Silent failures are broken experiences.

**Failure Mode**: "I'll add error handling later" → it never happens → production breaks silently.

---

## Phase 5: Template Cleanliness

### 5.1 No Business Logic in Templates

❌ **NEVER**:
```vue
<template>
  <div :style="{ width: (epsilon * 100) + '%' }">
  <div :class="percentage < 20 ? 'critical' : percentage < 60 ? 'warning' : 'good'">
  <div v-for="item in data.filter(x => x.reward > 10).sort((a,b) => b.reward - a.reward)">
</template>
```

✅ **ALWAYS**:
```vue
<template>
  <div :style="{ width: epsilonPercentage }">
  <div :class="statusClass">
  <div v-for="item in filteredSortedData" :key="item.id">
</template>

<script setup>
// Move calculations to computed
const epsilonPercentage = computed(() => `${epsilon.value * 100}%`)

const statusClass = computed(() => {
  if (percentage.value < 20) return 'critical'
  if (percentage.value < 60) return 'warning'
  return 'good'
})

const filteredSortedData = computed(() => {
  return props.data
    .filter(item => item.reward > REWARD_THRESHOLD)
    .sort((a, b) => b.reward - a.reward)
})
</script>
```

### 5.2 Extract Constants

❌ **NEVER**:
```javascript
if (percentage < 20) return 'critical'
if (stress > 80) showAlert()
fetchData({ limit: 100 })
```

✅ **ALWAYS**:
```javascript
const THRESHOLD_CRITICAL = 20
const THRESHOLD_HIGH_STRESS = 80
const DEFAULT_PAGE_SIZE = 100

if (percentage < THRESHOLD_CRITICAL) return 'critical'
if (stress > THRESHOLD_HIGH_STRESS) showAlert()
fetchData({ limit: DEFAULT_PAGE_SIZE })
```

### 5.3 Extract Complex Computed

If computed has >5 lines of logic, extract to utility function:

```javascript
// utils/dataTransforms.js
export function calculateMetricStatus(value, thresholds) {
  if (value < thresholds.critical) return 'critical'
  if (value < thresholds.warning) return 'warning'
  return 'good'
}

// Component
import { calculateMetricStatus } from '@/utils/dataTransforms'

const statusClass = computed(() =>
  calculateMetricStatus(percentage.value, METRIC_THRESHOLDS)
)
```

---

## Phase 6: Responsive Design Standards

### 6.1 Mobile-First CSS

❌ **NEVER** (desktop-first):
```css
.container {
  width: 1200px;
  padding: 2rem;
}

@media (max-width: 768px) {
  .container {
    width: 100%;
    padding: 1rem;
  }
}
```

✅ **ALWAYS** (mobile-first):
```css
.container {
  width: 100%;
  padding: 1rem;
}

@media (min-width: 768px) {
  .container {
    width: 1200px;
    padding: 2rem;
  }
}
```

### 6.2 Standard Breakpoints

Use consistent breakpoints (from tokens.js):

```css
/* Mobile: 0-639px (default styles) */

@media (min-width: 640px) {
  /* Small tablets and up */
}

@media (min-width: 768px) {
  /* Tablets and up */
}

@media (min-width: 1024px) {
  /* Desktops and up */
}

@media (min-width: 1280px) {
  /* Large desktops */
}
```

### 6.3 Touch-Friendly Controls

**All interactive elements 44x44px minimum**:

```css
.button, .input, .select {
  min-height: 44px;
  min-width: 44px;
  /* Padding can make it larger */
}
```

### 6.4 Responsive Typography

```css
.heading {
  font-size: 1.25rem; /* 20px mobile */
}

@media (min-width: 768px) {
  .heading {
    font-size: 1.5rem; /* 24px tablet+ */
  }
}
```

### 6.5 Container Queries (Modern Approach)

For component-level responsiveness:

```css
.card-container {
  container-type: inline-size;
}

.card {
  padding: 1rem;
}

@container (min-width: 400px) {
  .card {
    padding: 2rem;
  }
}
```

### 6.6 Test Responsive Behavior

**Before marking complete**:
- [ ] Test at 320px (small phone)
- [ ] Test at 375px (iPhone SE)
- [ ] Test at 768px (tablet portrait)
- [ ] Test at 1024px (tablet landscape / small laptop)
- [ ] Verify touch targets are 44px+
- [ ] Verify no horizontal scrolling at any breakpoint

---

## Phase 7: Performance Patterns

### 7.1 When to Optimize

**❌ DO NOT optimize until you have**:
1. Measured performance with Vue DevTools
2. Identified actual bottleneck
3. Confirmed it's a user-facing issue

**✅ DO optimize when**:
- Rendering >1000 items in list
- Running heavy computation on every reactive change
- Loading large datasets without pagination

### 7.2 Virtual Scrolling (Large Lists Only)

**Only use if list has 1000+ items**:

```vue
<!-- Install: npm install vue-virtual-scroller -->
<template>
  <RecycleScroller
    :items="episodes"
    :item-size="36"
    key-field="id"
    v-slot="{ item }"
  >
    <div class="episode-row">
      {{ item.episode }}: {{ item.reward }}
    </div>
  </RecycleScroller>
</template>

<script setup>
import { RecycleScroller } from 'vue-virtual-scroller'
import 'vue-virtual-scroller/dist/vue-virtual-scroller.css'
</script>
```

### 7.3 Debounce User Input

For search/filter inputs:

```javascript
import { useDebounceFn } from '@vueuse/core'

const searchQuery = ref('')
const debouncedSearch = useDebounceFn(() => {
  performSearch(searchQuery.value)
}, 300) // 300ms delay

watch(searchQuery, debouncedSearch)
```

### 7.4 Lazy Load Heavy Components

```vue
<script setup>
import { defineAsyncComponent } from 'vue'

const HeavyChart = defineAsyncComponent(() =>
  import('./components/HeavyChart.vue')
)
</script>

<template>
  <Suspense>
    <template #default>
      <HeavyChart :data="chartData" />
    </template>
    <template #fallback>
      <LoadingSpinner />
    </template>
  </Suspense>
</template>
```

### 7.5 Optimize Computed Dependencies

```javascript
// ❌ Bad - recomputes on ANY store change
const filteredData = computed(() =>
  store.allData.filter(x => x.type === props.filter)
)

// ✅ Good - only recomputes when specific dependencies change
const filteredData = computed(() => {
  const data = store.allData  // Track only this
  const filter = props.filter  // And this
  return data.filter(x => x.type === filter)
})
```

---

## Phase 8: Component Structure Standards

### 8.1 File Organization

```
src/
├── components/
│   ├── base/           # Reusable primitives (Button, Input, Card)
│   ├── features/       # Feature-specific (Dashboard, EpisodeTable)
│   └── layouts/        # Layout components (AppLayout, GridLayout)
├── composables/        # Reusable reactive logic
├── stores/             # Pinia stores
├── utils/              # Pure functions (no reactivity)
├── styles/
│   ├── tokens.js       # Design tokens
│   ├── variables.css   # CSS custom properties
│   └── global.css      # Global styles
└── views/              # Route pages
```

### 8.2 Component Template

```vue
<template>
  <!-- Template here -->
</template>

<script setup>
// 1. Imports
import { ref, computed, watch, onMounted } from 'vue'
import { useStore } from '@/stores/myStore'
import { tokens } from '@/styles/tokens'

// 2. Props & Emits
const props = defineProps({
  data: {
    type: Array,
    required: true
  }
})

const emit = defineEmits(['update', 'delete'])

// 3. Stores
const store = useStore()

// 4. Local state
const isLoading = ref(false)
const error = ref(null)

// 5. Computed
const processedData = computed(() => {
  return props.data.map(transform)
})

// 6. Methods
function handleClick() {
  emit('update', processedData.value)
}

// 7. Lifecycle
onMounted(() => {
  // Initialization
})

// 8. Watchers
watch(() => props.data, (newData) => {
  // React to prop changes
})
</script>

<style scoped>
/* Component styles */
/* Use tokens, not magic numbers */
</style>
```

---

## Phase 9: Testing & Verification

### 9.1 Manual Testing Checklist

Before marking component complete:

**Functionality**:
- [ ] All features work as expected
- [ ] Error states display correctly
- [ ] Loading states show and hide properly
- [ ] Empty states render when no data

**Accessibility**:
- [ ] Keyboard navigation works (Tab, Enter, Space, Arrows)
- [ ] Screen reader announces changes (test with VoiceOver/NVDA)
- [ ] Focus indicators visible
- [ ] Color contrast passes WCAG AA
- [ ] Touch targets 44px+

**Responsive**:
- [ ] Works at 320px width (small phone)
- [ ] Works at 768px width (tablet)
- [ ] Works at 1280px width (desktop)
- [ ] No horizontal scrolling
- [ ] Touch controls work on mobile

**Performance**:
- [ ] No lag when interacting
- [ ] Large lists render smoothly
- [ ] No unnecessary re-renders (check Vue DevTools)

### 9.2 Code Review Checklist

**Before submitting**:
- [ ] No magic numbers (all extracted to constants/tokens)
- [ ] No inline styles (moved to computed or CSS)
- [ ] No business logic in template
- [ ] Error/loading/empty states present
- [ ] Accessibility attributes added (ARIA, semantic HTML)
- [ ] Props used instead of direct store access (where appropriate)
- [ ] Component can be reused in different contexts
- [ ] Comments explain "why", not "what"

---

## Summary: The Golden Path

**For every Vue 3 component**:

1. **Design Tokens** → Create/update tokens.js with colors, spacing, typography
2. **Accessibility** → Semantic HTML, ARIA attributes, keyboard navigation
3. **Props First** → Accept data via props, emit events, minimize store coupling
4. **3-State Pattern** → Loading, error, empty states for all data operations
5. **Clean Templates** → Business logic in computed/methods, not templates
6. **Mobile-First** → Responsive CSS with standard breakpoints, 44px touch targets
7. **Measure First** → Profile before optimizing, use virtual scroll only when needed
8. **Test Everything** → Manual testing checklist before marking complete

**If you skip any of these steps, your component is incomplete.**

**Time pressure, exhaustion, or "this is simple" are NOT valid reasons to skip steps.**

---

## Enforcement

**How to use this skill**:

1. Announce you're using the vue3-ux-design skill
2. Reference design tokens FIRST (before any component code)
3. As you write each component, explicitly check against:
   - Accessibility checklist
   - 3-state pattern
   - Template cleanliness rules
   - Responsive standards
4. Before marking component complete, run through testing checklist

**If you catch yourself rationalizing away any step, STOP and re-read the Rationalization Counter Table.**

**This skill is mandatory for all Vue 3 work. Not optional.**

---

## Long Conversation Reminder

If this is a long conversation and you're approaching context limits:

1. **STOP** and re-read this skill before continuing
2. Verify you're still following ALL phases (not just some)
3. Check the Rationalization Counter Table again
4. Review the testing checklist

**Context loss = quality loss.** If you can't remember the full skill, re-read it. Don't continue from memory.
