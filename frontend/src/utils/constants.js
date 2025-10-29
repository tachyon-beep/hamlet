/**
 * Application Constants
 *
 * Centralized constants used across components
 */

// Grid rendering
export const CELL_SIZE = 75 // pixels per cell

// Action display names
export const ACTION_ICONS = {
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
  HomeMeal: '🥘',
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
