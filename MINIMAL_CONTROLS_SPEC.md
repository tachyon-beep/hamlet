# MinimalControls Component Technical Specification

## Component Overview

**File:** `/home/john/hamlet/frontend/src/components/MinimalControls.vue`
**Purpose:** Ultra-minimal playback controls for HAMLET visualizer
**Dependencies:** Vue 3 Composition API
**Parent:** `App.vue` (grid-container)

## API

### Props

```typescript
interface Props {
  isConnected: boolean  // Whether WebSocket is connected (default: false)
}
```

**Usage:**
```vue
<MinimalControls
  :is-connected="store.isConnected"
  @disconnect="store.disconnect"
  @set-speed="store.setSpeed"
/>
```

### Events

```typescript
interface Events {
  disconnect: () => void      // Emitted when disconnect button clicked
  'set-speed': (speed: number) => void  // Emitted when speed slider changes
}
```

**Event flow:**
1. User adjusts slider → `@input` → `onSpeedChange()` → `emit('set-speed', speedValue)`
2. User clicks X → `@click` → `disconnect()` → `emit('disconnect')`

## Component Structure

### Template Hierarchy
```
div.minimal-controls (root)
├── div.speed-control
│   ├── label#speed.sr-only (screen reader only)
│   ├── span.speed-label (displays "1.0x")
│   └── input#speed.speed-slider (range input)
└── button.disconnect-button (✕ button)
```

### Reactive State

```javascript
const speedValue = ref(1.0)  // Current speed multiplier (0.5-10.0)
```

**No computed properties** - Simple component with minimal state.

### Methods

```javascript
function onSpeedChange() {
  emit('set-speed', speedValue.value)
}

function disconnect() {
  emit('disconnect')
}
```

## Styling

### Positioning Strategy

**Position:** `absolute`
**Location:** Top-right corner of `.grid-container`
**Z-index:** 100 (above grid content)

```css
.minimal-controls {
  position: absolute;
  top: var(--spacing-md);      /* 16px */
  right: var(--spacing-md);    /* 16px */
  z-index: 100;
}
```

**Why absolute?**
- Doesn't affect grid layout calculations
- Can be shown/hidden without reflow
- Floats above content
- Minimal visual footprint

### Visual Design

**Background:**
```css
background: var(--color-bg-secondary);
backdrop-filter: blur(8px);
box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
border-radius: var(--border-radius-md);
```

**Spacing:**
- Padding: `var(--spacing-sm) var(--spacing-md)` (8px 16px)
- Gap: `var(--spacing-md)` (16px between controls)

**Layout:**
```css
display: flex;
align-items: center;
gap: var(--spacing-md);
```

### Speed Control

**Structure:**
```
┌────────────────────────┐
│ 1.0x  [━━━━━●━━━━━━━] │
└────────────────────────┘
```

**Components:**
1. **Speed label** (`span.speed-label`)
   - Font: Monospace (Monaco, Courier New)
   - Size: `var(--font-size-sm)` (14px)
   - Weight: `var(--font-weight-semibold)` (600)
   - Min-width: 40px (right-aligned)

2. **Speed slider** (`input[type="range"]`)
   - Width: 120px (desktop), 80px (mobile)
   - Height: 6px
   - Min: 0.5, Max: 10.0, Step: 0.5
   - Color: Green (`--color-success`)

**Slider thumb styling:**
```css
/* Webkit (Chrome, Safari) */
.speed-slider::-webkit-slider-thumb {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--color-success);
  box-shadow: 0 2px 4px rgba(34, 197, 94, 0.4);
  cursor: pointer;
}

.speed-slider::-webkit-slider-thumb:hover {
  transform: scale(1.15);
  box-shadow: 0 2px 6px rgba(34, 197, 94, 0.6);
}

/* Firefox */
.speed-slider::-moz-range-thumb {
  /* Same styling as webkit */
}
```

### Disconnect Button

**Visual:**
```
┌────┐
│ ✕  │  ← Red button with X
└────�┘
```

**Styling:**
```css
.disconnect-button {
  padding: var(--spacing-xs) var(--spacing-sm);
  background: var(--color-error);              /* Red */
  border: none;
  border-radius: var(--border-radius-sm);
  color: var(--color-text-on-dark);           /* White */
  font-size: var(--font-size-lg);             /* 18px */
  min-width: 36px;
  min-height: 36px;
  cursor: pointer;
}

.disconnect-button:hover {
  background: var(--color-error-hover);
  transform: scale(1.05);
}

.disconnect-button:active {
  transform: scale(0.98);
}
```

**Icon:** `✕` (Unicode character, not emoji - better cross-platform)

## Responsive Behavior

### Breakpoints

**Mobile (< 768px):**
```css
.minimal-controls {
  top: var(--spacing-xs);     /* 8px - closer to edge */
  right: var(--spacing-xs);
  padding: var(--spacing-xs) var(--spacing-sm);
  gap: var(--spacing-sm);
}

.speed-slider {
  width: 80px;                /* Narrower slider */
}

.speed-label {
  font-size: var(--font-size-xs);
  min-width: 32px;
}

.disconnect-button {
  min-width: 32px;
  min-height: 32px;
  font-size: var(--font-size-base);
}
```

**Tablet/Desktop (≥ 768px):**
```css
.minimal-controls {
  padding: var(--spacing-sm) var(--spacing-lg);
}

.speed-slider {
  width: 120px;               /* Full-width slider */
}
```

### Layout Behavior

**When not connected:**
- Component is not rendered (`v-if="isConnected"`)

**When connected:**
- Component fades in (via CSS transition)
- Positioned absolutely in top-right
- Does not affect grid layout

**On mobile:**
- Scales down proportionally
- Maintains usability (touch targets ≥ 32px)
- Slightly closer to edge (8px vs 16px)

## Accessibility

### ARIA Attributes

```html
<!-- Root region -->
<div class="minimal-controls"
     role="region"
     aria-label="Simulation controls">

  <!-- Speed slider -->
  <label for="speed" class="sr-only">Playback speed</label>
  <input id="speed"
         type="range"
         aria-label="Adjust playback speed" />

  <!-- Disconnect button -->
  <button aria-label="Disconnect from server"
          title="Disconnect">
    <span aria-hidden="true">✕</span>
  </button>
</div>
```

### Screen Reader Support

**Speed control:**
1. Label read: "Playback speed"
2. Value announced: "1.0" (current value)
3. Range announced: "0.5 to 10, step 0.5"

**Disconnect button:**
1. Label read: "Disconnect from server"
2. Role: "button"
3. State: (none - simple button)

### Keyboard Navigation

**Tab order:**
1. Speed slider (first focus)
2. Disconnect button (second focus)

**Keyboard controls:**
- **Tab/Shift+Tab:** Navigate between controls
- **Arrow keys (on slider):** Adjust speed by 0.5
- **Home/End (on slider):** Jump to min/max
- **Enter/Space (on button):** Trigger disconnect

### Focus Indicators

```css
.speed-slider:focus-visible {
  outline: 2px solid var(--color-interactive-focus);
  outline-offset: 2px;
}

.disconnect-button:focus-visible {
  outline: 2px solid var(--color-interactive-focus);
  outline-offset: 2px;
}
```

## Integration with App.vue

### Mounting Location

**Parent container:**
```vue
<div class="grid-container" role="region">
  <MinimalControls
    v-if="isConnected"
    :is-connected="store.isConnected"
    @disconnect="store.disconnect"
    @set-speed="store.setSpeed"
  />
  <!-- Grid content below -->
</div>
```

**Why inside grid-container?**
- Absolute positioning relative to grid area
- Semantic grouping (controls for grid)
- Easy to show/hide with connection state

### CSS Requirements

**Parent must have:**
```css
.grid-container {
  position: relative;  /* For absolute positioning of MinimalControls */
}
```

**Without this:** MinimalControls would position relative to viewport, not grid.

### Event Handling

**App.vue script:**
```javascript
const store = useSimulationStore()

// MinimalControls emits → Store actions
// @disconnect → store.disconnect()
// @set-speed → store.setSpeed(speed)
```

**No intermediate handlers needed** - Events map directly to store methods.

## Store Integration

### Required Store Methods

```javascript
// In simulation.js store
export const useSimulationStore = defineStore('simulation', () => {

  function disconnect() {
    manualDisconnect.value = true
    if (ws.value) {
      ws.value.close()
    }
    isConnected.value = false
  }

  function setSpeed(speed) {
    sendCommand('set_speed', { speed })
  }

  return {
    isConnected,
    disconnect,
    setSpeed,
    // ...
  }
})
```

### WebSocket Protocol

**Speed change command:**
```json
{
  "command": "set_speed",
  "speed": 1.5
}
```

**Disconnect:**
- No command sent
- WebSocket closed directly
- Backend detects disconnect via `onclose` handler

## Performance Considerations

### Rendering Frequency

**Speed slider:**
- `@input` event fires on every slider movement
- `emit('set-speed')` called for each update
- Backend may throttle/debounce actual speed changes

**Recommendation:** Consider debouncing if network traffic is concern.

```javascript
import { ref } from 'vue'
import { useDebounceFn } from '@vueuse/core'

const debouncedSpeedChange = useDebounceFn(() => {
  emit('set-speed', speedValue.value)
}, 200)  // 200ms debounce
```

**Current implementation:** No debounce (immediate feedback preferred for learning tool)

### Re-render Triggers

Component re-renders when:
1. `isConnected` prop changes (show/hide)
2. `speedValue` changes (user interaction)

**No re-renders from:**
- Grid updates (independent component)
- Meter changes (no shared state)
- Parent state changes (unless affecting props)

### DOM Operations

**Mount/unmount:**
- `v-if="isConnected"` completely removes from DOM when disconnected
- Cleaner than `v-show` (no hidden elements)
- Slight cost on mount/unmount (acceptable for infrequent operation)

## Testing Strategy

### Unit Tests (Vitest)

```javascript
import { mount } from '@vue/test-utils'
import MinimalControls from './MinimalControls.vue'

describe('MinimalControls', () => {
  it('renders speed slider with default value', () => {
    const wrapper = mount(MinimalControls, {
      props: { isConnected: true }
    })
    const slider = wrapper.find('.speed-slider')
    expect(slider.element.value).toBe('1')
  })

  it('emits set-speed when slider changes', async () => {
    const wrapper = mount(MinimalControls, {
      props: { isConnected: true }
    })
    const slider = wrapper.find('.speed-slider')
    await slider.setValue('2.5')
    expect(wrapper.emitted('set-speed')).toBeTruthy()
    expect(wrapper.emitted('set-speed')[0]).toEqual([2.5])
  })

  it('emits disconnect when button clicked', async () => {
    const wrapper = mount(MinimalControls, {
      props: { isConnected: true }
    })
    await wrapper.find('.disconnect-button').trigger('click')
    expect(wrapper.emitted('disconnect')).toBeTruthy()
  })
})
```

### Integration Tests

```javascript
describe('MinimalControls in App', () => {
  it('appears when connected', async () => {
    const wrapper = mount(App)
    expect(wrapper.find('.minimal-controls').exists()).toBe(false)

    await wrapper.vm.handleConnect('inference')
    await wrapper.vm.$nextTick()

    expect(wrapper.find('.minimal-controls').exists()).toBe(true)
  })

  it('disconnects when X clicked', async () => {
    const wrapper = mount(App)
    await wrapper.vm.handleConnect('inference')

    const disconnectBtn = wrapper.find('.disconnect-button')
    await disconnectBtn.trigger('click')

    expect(wrapper.vm.store.isConnected).toBe(false)
  })
})
```

### Manual Testing Checklist

**Visual:**
- [ ] Controls appear in top-right when connected
- [ ] Speed label updates when slider moves
- [ ] Disconnect button turns red on hover
- [ ] Controls have subtle shadow/backdrop blur
- [ ] Responsive sizing on mobile

**Functional:**
- [ ] Speed slider adjusts from 0.5x to 10x
- [ ] Speed changes affect simulation speed
- [ ] Disconnect button stops simulation
- [ ] Controls disappear when disconnected
- [ ] Keyboard navigation works

**Accessibility:**
- [ ] Screen reader announces "Playback speed"
- [ ] Tab key navigates between controls
- [ ] Arrow keys adjust slider
- [ ] Focus indicators visible
- [ ] Button has accessible label

## Design Tokens Used

### Colors
```css
--color-bg-secondary       /* Background */
--color-text-primary       /* Speed label text */
--color-text-on-dark       /* Button text (white) */
--color-success            /* Slider thumb (green) */
--color-error              /* Disconnect button (red) */
--color-error-hover        /* Disconnect button hover */
--color-interactive-focus  /* Focus outline */
```

### Spacing
```css
--spacing-xs      /* 8px  - mobile padding */
--spacing-sm      /* 12px - button padding */
--spacing-md      /* 16px - default padding/gap */
--spacing-lg      /* 24px - desktop padding */
```

### Typography
```css
--font-size-xs        /* 12px - mobile speed label */
--font-size-sm        /* 14px - desktop speed label */
--font-size-base      /* 16px - mobile button */
--font-size-lg        /* 18px - desktop button */
--font-weight-semibold /* 600 - speed label */
--font-weight-bold     /* 700 - button */
```

### Border Radius
```css
--border-radius-sm    /* 4px  - button */
--border-radius-md    /* 8px  - controls container */
--border-radius-full  /* 50%  - slider thumb */
```

### Transitions
```css
--transition-base     /* 150ms ease - all interactions */
```

## Browser Compatibility

### Supported Browsers
- Chrome 90+ ✓
- Firefox 88+ ✓
- Safari 14+ ✓
- Edge 90+ ✓

### Known Issues
**None** - Uses only standard web APIs:
- `<input type="range">` (universal support)
- Flexbox (universal support)
- CSS custom properties (universal support)
- Backdrop filter (graceful degradation)

### Fallbacks

**Backdrop filter not supported:**
```css
@supports not (backdrop-filter: blur(8px)) {
  .minimal-controls {
    background: rgba(17, 24, 39, 0.95);  /* Solid background */
  }
}
```

## Future Enhancements

### Potential Additions
1. **Model selector dropdown** (if multiple models common)
2. **Speed presets** (buttons for 0.5x, 1x, 2x, 5x)
3. **Pause button** (if user testing shows need)
4. **Tooltips** (on hover, show keyboard shortcuts)

### NOT Recommended
- ❌ Step button (conflicts with auto-play philosophy)
- ❌ Reset button (disconnect/reconnect achieves same)
- ❌ Play/pause (redundant with connect/disconnect)

## Summary

**MinimalControls** is a purpose-built component that:
- Provides essential playback controls (speed + disconnect)
- Floats unobtrusively in top-right corner
- Auto-shows when connected, hides when not
- Maintains full accessibility support
- Uses modern CSS for polished feel
- Requires minimal state/logic (simple is better)

**Philosophy:** The best control panel is the one you barely notice.

---

**Key Files:**
- Component: `/home/john/hamlet/frontend/src/components/MinimalControls.vue`
- Parent: `/home/john/hamlet/frontend/src/App.vue`
- Store: `/home/john/hamlet/frontend/src/stores/simulation.js`
- Summary: `/home/john/hamlet/SIMPLIFIED_CONTROLS_SUMMARY.md`
- Comparison: `/home/john/hamlet/CONTROLS_BEFORE_AFTER.md`

**Build status:** ✓ Successful (570ms, 123.65 kB bundle)
