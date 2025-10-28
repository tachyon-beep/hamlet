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
