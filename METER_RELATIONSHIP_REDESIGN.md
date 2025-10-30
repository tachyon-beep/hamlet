# Meter Relationship Visualization Redesign

## Problem Statement

The original implementation used **hover-based cascade highlighting** which was inappropriate for a **real-time streaming visualization** where users need to continuously monitor all meter values simultaneously.

### Issues with Hover-Based Design

1. **Information hiding**: Dimming unrelated meters to 30% opacity obscured critical real-time data
2. **Attention splitting**: Users couldn't hover while monitoring live state changes
3. **Single-target limitation**: Can't show multiple relationships simultaneously
4. **Anti-pattern for streaming**: Requires active exploration instead of passive monitoring

## Solution: Static Visual Language

Redesigned relationship indicators to be **always visible** and work for peripheral vision monitoring during real-time streams.

### Changes Made

#### 1. Removed Interactive Elements

**Before:**
```vue
@mouseenter="highlightCascade(name)"
@mouseleave="clearHighlight"
:data-highlighted="highlightedMeters.has(name)"
:data-dimmed="highlightedMeters.size > 0 && !highlightedMeters.has(name)"
```

**After:**
```vue
<!-- No event handlers, no dimming attributes -->
```

#### 2. Added Always-Visible Relationship Text

**Before:** (hover-only arrow indicator)
```vue
<span class="relationship-indicator" :title="getRelationshipTooltip(name)">→</span>
```

**After:** (permanent inline text)
```vue
<div class="meter-relationship">
  → {{ getRelationshipText(name) }}
</div>
```

#### 3. Visual Hierarchy for Passive Scanning

**Secondary meters (modifiers):**
```
Mood                 87%
→ Energy                    [cyan colored line]
████████████░░░░ (progress bar)
```

**Tertiary meters (accelerators):**
```
Social              45%
⚡ Mood                      [purple colored line]
██████░░░░░░ (progress bar)
```

### Design Principles Applied

1. **Pre-attentive attributes**: Color and position encode tier relationships
2. **Peripheral vision first**: Can detect cascades without focusing attention
3. **No occlusion**: All meter values remain fully visible at all times
4. **Graceful degradation**: Quick glance shows structure, closer look shows details

### CSS Implementation

```css
.meter-relationship {
  font-size: var(--font-size-xs);
  color: var(--tier-color);           /* Inherits cyan/purple from tier */
  margin-top: var(--spacing-xs);
  padding: calc(var(--spacing-xs) / 2) var(--spacing-xs);
  background: rgba(255, 255, 255, 0.03);  /* Subtle container */
  border-radius: var(--border-radius-sm);
  border-left: 2px solid var(--tier-color);  /* Color-coded accent */
  font-weight: var(--font-weight-medium);
  opacity: 0.85;                      /* Visible but not competing with values */
  line-height: 1.2;
}

.meter-relationship.accelerator {
  color: #a78bfa;                     /* Purple for tertiary tier */
  border-left-color: #a78bfa;
}
```

### Typography Choices

- **Arrow (→)** for secondary modifiers: Direct, immediate effects
- **Lightning (⚡)** for tertiary accelerators: Speed, multiplicative effects
- **Compact text**: "Energy + Health" not "Affects energy and health"
- **Color consistency**: Matches tier color (cyan for secondary, purple for tertiary)

### Removed Code

- `highlightedMeters` reactive state (no longer needed)
- `highlightCascade()` function (removed hover logic)
- `clearHighlight()` function (removed hover logic)
- `getRelationshipTooltip()` function (replaced with always-visible text)
- `.meter[data-highlighted]` CSS (removed highlight box styles)
- `.meter[data-dimmed]` CSS (removed opacity dimming)
- `.relationship-indicator` hover animations (removed pulse-arrow)

### Accessibility Improvements

- **No hover dependency**: Works for keyboard, touch, and mouse users equally
- **Screen reader friendly**: Relationship text is normal DOM text (not tooltip)
- **High contrast**: 0.85 opacity on dark background meets WCAG AA
- **Consistent ARIA**: Existing aria-label still describes full state

### Performance Benefits

- **No event handlers**: Removed mouseenter/mouseleave on 5 meters × 2 events = 10 listeners
- **No reactive state updates**: Removed `highlightedMeters` Set mutations
- **Simpler rendering**: No conditional dimming/highlighting calculations
- **Smaller bundle**: Removed ~40 lines of JS + ~40 lines of CSS

## Visual Example

### Old Design (Hover Required)
```
Mood: 87%  →              ← Arrow only visible on hover
████████████░░░░           ← Dimmed to 30% when other meter hovered
```

### New Design (Always Visible)
```
Mood                 87%
→ Energy                    ← Always visible, color-coded cyan
████████████░░░░           ← Never dimmed, always readable
```

## Testing Verification

Run the visualization to verify:
```bash
# Terminal 1: Backend
uv run python demo_visualization.py

# Terminal 2: Frontend
cd frontend && npm run dev
```

Open browser to http://localhost:5174 and verify:
- [ ] All meter relationships visible without hovering
- [ ] No meters dim when hovering over others
- [ ] Secondary meters show "→ Energy" or similar text
- [ ] Tertiary meters show "⚡ Mood" or similar text
- [ ] Color coding: cyan for secondary, purple for tertiary
- [ ] Text remains readable during rapid value updates

## Design Rationale

### Why Static Over Interactive?

**Use case:** Monitoring real-time simulation where agent state updates 10+ times per second

**User goal:** Track all meter values simultaneously to understand agent survival strategy

**Interaction model:**
- **Wrong**: Hover to learn → Monitor to use
- **Right**: Learn structure immediately → Monitor continuously

### Why Inline Text Over Visual Lines?

Considered alternatives:
1. **SVG lines connecting meters**: Complex layout, hard to read at 300px width
2. **Color-coded badges**: Takes up too much space, unclear direction
3. **Mini dependency diagram**: Separate from live values, attention split
4. **Icon-only indicators**: Too cryptic, requires memorization

**Winner: Inline text** because:
- Compact (fits in 300px sidebar)
- Self-documenting (no need to memorize icon meanings)
- Scannable (arrow/lightning prefix makes structure obvious)
- Flexible (works with variable number of targets: "Energy" vs "Energy + Health")

### Color Coding Strategy

| Tier | Color | Rationale |
|------|-------|-----------|
| Primary | Gold (#fbbf24) | High value, survival-critical |
| Secondary | Cyan (#06b6d4) | Cool, modifier effects |
| Tertiary | Purple (#a78bfa) | Accent, accelerator effects |

Colors chosen for:
- **Distinctiveness**: Easy to tell apart in peripheral vision
- **Hierarchy**: Gold > Cyan > Purple matches importance
- **Accessibility**: All meet WCAG AA contrast on dark backgrounds

## Future Considerations

### If More Relationships Are Added

Current system scales to:
- 2 targets per meter: "Energy + Health" (readable)
- 3+ targets: May need abbreviations "Eng + Hth + Mny"

### If Multi-Level Cascades Need Emphasis

Could add subtle visual indicator for transitive relationships:
```
Social              45%
⚡ Mood → Energy           ← Shows two-level cascade
██████░░░░░░
```

But test with users first - may be too complex.

### If Space Becomes Critical

Could use icons only with hover tooltips as fallback:
```
Social              45%
⚡ M E                      ← Icons with aria-label
██████░░░░░░
```

But this returns to hover dependency - avoid unless absolutely necessary.

## Conclusion

This redesign transforms meter relationships from **interactive exploration** (wrong for streaming) to **ambient awareness** (right for monitoring). Users can now track all meter values and their dependencies simultaneously without any interaction required.

**Key insight:** In real-time visualization, every piece of information should be designed for peripheral vision first, focused attention second.
