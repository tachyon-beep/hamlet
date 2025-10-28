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
    textOnDark: '#ffffff',            // White text on dark backgrounds

    // Interactive elements
    interactive: '#10b981',            // Primary interactive (green)
    interactiveHover: '#34d399',       // Hover state
    interactiveFocus: '#6ee7b7',       // Focus ring
    interactiveDisabled: '#4a4a5e',    // Disabled state

    // Status colors
    success: '#10b981',                // Success state
    warning: '#f59e0b',                // Warning state
    error: '#ef4444',                  // Error state
    errorHover: '#dc2626',             // Error hover state (darker red)
    errorDark: '#b91c1c',              // Error dark state (darkest red)
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

    // Affordance stroke colors (for Grid.vue affordances)
    affordanceBedStroke: '#818cf8',    // Light indigo
    affordanceShowerStroke: '#22d3ee', // Cyan
    affordanceHomeMealStroke: '#fbbf24', // Amber
    affordanceFastFoodStroke: '#f87171', // Light red
    affordanceJobStroke: '#a78bfa',    // Light purple
    affordanceBarStroke: '#f472b6',    // Pink

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
 *
 * Note: variables.css is manually maintained for better control and readability.
 * This function is provided as a convenience for programmatic generation if needed.
 */
export function tokensToCSS() {
  const cssVars = []

  // Helper to convert camelCase to kebab-case
  const toKebab = (str) => str.replace(/([a-z0-9])([A-Z])/g, '$1-$2').toLowerCase()

  // Process colors
  Object.entries(tokens.colors).forEach(([key, value]) => {
    if (typeof value === 'object' && !Array.isArray(value)) {
      // Handle nested objects like meterStress
      Object.entries(value).forEach(([subKey, subValue]) => {
        cssVars.push(`  --color-${toKebab(key)}-${toKebab(subKey)}: ${subValue};`)
      })
    } else {
      cssVars.push(`  --color-${toKebab(key)}: ${value};`)
    }
  })

  // Process spacing
  Object.entries(tokens.spacing).forEach(([key, value]) => {
    cssVars.push(`  --spacing-${key}: ${value};`)
  })

  // Process font sizes
  Object.entries(tokens.fontSize).forEach(([key, value]) => {
    cssVars.push(`  --font-size-${key}: ${value};`)
  })

  // Process font weights
  Object.entries(tokens.fontWeight).forEach(([key, value]) => {
    cssVars.push(`  --font-weight-${key}: ${value};`)
  })

  // Process border radius
  Object.entries(tokens.borderRadius).forEach(([key, value]) => {
    cssVars.push(`  --border-radius-${key}: ${value};`)
  })

  // Process transitions
  Object.entries(tokens.transition).forEach(([key, value]) => {
    cssVars.push(`  --transition-${key}: ${value};`)
  })

  // Process layout dimensions
  Object.entries(tokens.layout).forEach(([key, value]) => {
    cssVars.push(`  --layout-${toKebab(key)}: ${value};`)
  })

  // Process accessibility
  Object.entries(tokens.accessibility).forEach(([key, value]) => {
    cssVars.push(`  --a11y-${toKebab(key)}: ${value};`)
  })

  // Process z-index
  Object.entries(tokens.zIndex).forEach(([key, value]) => {
    cssVars.push(`  --z-index-${key}: ${value};`)
  })

  return cssVars.join('\n')
}
