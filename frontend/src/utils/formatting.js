/**
 * Formatting Utilities
 *
 * Reusable formatting functions for consistent display
 */

/**
 * Capitalize the first letter of a string
 * @param {string} str - String to capitalize
 * @returns {string} Capitalized string
 */
export function capitalize(str) {
  if (!str) return ''
  return str.charAt(0).toUpperCase() + str.slice(1)
}

/**
 * Format meter value for display
 * @param {string} name - Meter name (money, stress, or normalized meter)
 * @param {number} value - Raw meter value
 * @returns {string} Formatted display value
 */
export function formatMeterValue(name, value) {
  if (name === 'money') {
    return `$${Math.round(value)}`
  }
  if (name === 'stress') {
    // Stress is 0-100 (higher = worse)
    return `${Math.round(value)}`
  }
  // Other meters are normalized 0-1, convert to percentage
  return `${Math.round(value * 100)}%`
}

/**
 * Get meter percentage for progress bars
 * @param {string} name - Meter name
 * @param {number} value - Raw meter value
 * @returns {number} Percentage value (0-100)
 */
export function getMeterPercentage(name, value) {
  if (name === 'money' || name === 'stress') {
    // Money and stress are already 0-100
    return Math.max(0, Math.min(100, value))
  }
  // Other meters are normalized 0-1, convert to percentage
  return Math.max(0, Math.min(100, value * 100))
}

/**
 * Format reward value with sign
 * @param {number} reward - Reward value
 * @param {number} decimals - Number of decimal places (default: 1)
 * @returns {string} Formatted reward with sign
 */
export function formatReward(reward, decimals = 1) {
  const formatted = reward.toFixed(decimals)
  return reward > 0 ? `+${formatted}` : formatted
}

/**
 * Format number with commas for readability
 * @param {number} num - Number to format
 * @returns {string} Formatted number with commas
 */
export function formatNumber(num) {
  return num.toLocaleString('en-US')
}

/**
 * Format training metric with appropriate precision
 * @param {number} value - Metric value
 * @param {string} type - Metric type ('reward', 'length', 'loss', 'epsilon')
 * @returns {string} Formatted metric
 */
export function formatTrainingMetric(value, type) {
  switch (type) {
    case 'reward':
      return value.toFixed(2)
    case 'length':
      return value.toFixed(1)
    case 'loss':
      return value.toFixed(4)
    case 'epsilon':
      return value.toFixed(3)
    default:
      return value.toString()
  }
}
