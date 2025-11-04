# P0-1: Reduce Cognitive Load with Progressive Disclosure

## Problem

Information overload: 8 meters + 15 affordances + 5 charts simultaneously visible.

## Solution: Tabbed Panels with Contextual Focus

### Left Panel - Refactor to Tabs

**File**: `/home/john/hamlet/frontend/src/App.vue:11-21`

```vue
<aside class="left-panel" aria-label="Agent status panels">
  <!-- Tab Navigation -->
  <nav class="panel-tabs" role="tablist">
    <button
      role="tab"
      :aria-selected="activeTab === 'meters'"
      @click="activeTab = 'meters'"
      class="tab-button"
    >
      Meters
      <span v-if="criticalMetersCount > 0" class="badge critical">
        {{ criticalMetersCount }}
      </span>
    </button>
    <button
      role="tab"
      :aria-selected="activeTab === 'affordances'"
      @click="activeTab = 'affordances'"
      class="tab-button"
    >
      System
    </button>
  </nav>

  <!-- Tab Content -->
  <div class="tab-content">
    <MeterPanel
      v-show="activeTab === 'meters'"
      :agent-meters="store.agentMeters"
      @critical-count="criticalMetersCount = $event"
    />
    <AffordanceLegend v-show="activeTab === 'affordances'" />
  </div>
</aside>
```

### CSS (add to App.vue scoped styles)

```css
.panel-tabs {
  display: flex;
  gap: var(--spacing-xs);
  border-bottom: 2px solid var(--color-bg-tertiary);
  margin-bottom: var(--spacing-md);
}

.tab-button {
  flex: 1;
  padding: var(--spacing-sm) var(--spacing-md);
  background: transparent;
  border: none;
  border-bottom: 3px solid transparent;
  color: var(--color-text-secondary);
  font-weight: var(--font-weight-medium);
  cursor: pointer;
  transition: all var(--transition-base);
  position: relative;
}

.tab-button[aria-selected="true"] {
  color: var(--color-text-primary);
  border-bottom-color: var(--color-primary);
}

.badge {
  position: absolute;
  top: var(--spacing-xs);
  right: var(--spacing-xs);
  padding: 2px 6px;
  border-radius: var(--border-radius-full);
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-bold);
  background: var(--color-error);
  color: white;
}
```

### Right Panel - Collapsible Charts

**File**: `/home/john/hamlet/frontend/src/App.vue:91-110`

```vue
<!-- Collapsible Charts Section -->
<details class="collapsible-section" open>
  <summary class="section-header">
    <span>Learning Progress</span>
    <span class="chevron" aria-hidden="true">▼</span>
  </summary>
  <div class="section-content">
    <IntrinsicRewardChart ... />
    <SurvivalTrendChart ... />
    <AffordanceGraph ... />
  </div>
</details>
```

### CSS

```css
.collapsible-section {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
  list-style: none; /* Remove default marker */
}

.section-header .chevron {
  transition: transform var(--transition-base);
}

details[open] .chevron {
  transform: rotate(180deg);
}

.section-content {
  margin-top: var(--spacing-md);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}
```

## Pedagogical Benefit

- **Focused attention**: Students see meters first (survival priority)
- **Critical alerts**: Badge shows dangerous states without switching tabs
- **Progressive learning**: System complexity hidden until needed
- **Reduced scrolling**: Charts collapsed by default for cleaner interface
