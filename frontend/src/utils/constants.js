/**
 * Application Constants
 *
 * Centralized constants used across components
 */

// Grid rendering
export const CELL_SIZE = 75 // pixels per cell

// Action display names (support both numeric and string keys)
export const ACTION_ICONS = {
  // Numeric mappings (from action space)
  0: '↑ Up',
  1: '↓ Down',
  2: '← Left',
  3: '→ Right',
  4: '⚡ Interact',
  // String mappings (legacy support)
  up: '↑ Up',
  down: '↓ Down',
  left: '← Left',
  right: '→ Right',
  interact: '⚡ Interact'
}

// Affordance display
export const AFFORDANCE_ICONS = {
  Bed: '🛏️',
  Shower: '🚿',
  Fridge: '🥘',     // Main food source in current env
  HomeMeal: '🥘',  // Legacy support
  FastFood: '🍔',
  Job: '💼',
  Recreation: '🎮',
  Bar: '🍺',
  Gym: '💪'
}

// Meter thresholds
export const METER_THRESHOLDS = {
  CRITICAL: 20,    // Below this = critical (red)
  LOW: 30,         // Below this = low (warning)
  MODERATE: 60,    // Below this = moderate
  HEALTHY: 80      // Above this = healthy (green)
}

// Mood meter thresholds (higher is better)
export const MOOD_THRESHOLDS = {
  LOW: 30,         // Below this = low mood (red)
  MODERATE: 60,    // Between LOW and MODERATE = warning
  HIGH: 80         // Above this = healthy mood
}
