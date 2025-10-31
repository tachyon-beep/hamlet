import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useSimulationStore = defineStore('simulation', () => {
  // WebSocket connection
  const ws = ref(null)
  const isConnected = ref(false)
  const reconnectAttempts = ref(0)
  const maxReconnectAttempts = 10
  const reconnectDelay = 3000 // ms
  const mode = ref('inference') // 'inference' or 'training'
  const manualDisconnect = ref(false) // Track if user manually disconnected

  // ✅ Connection states for UI feedback
  const isConnecting = ref(false)
  const connectionError = ref(null)

  // Server availability detection
  const serverAvailability = ref({
    inference: false,
    training: false,
    checked: false
  })

  // Simulation state
  const currentEpisode = ref(0)
  const currentStep = ref(0)
  const cumulativeReward = ref(0)
  const lastAction = ref(null)

  // Training-specific state
  const isTraining = ref(false)
  const totalEpisodes = ref(0)
  const trainingMetrics = ref({
    avgReward5: 0,
    avgLength5: 0,
    avgLoss5: 0,
    epsilon: 1.0,
    bufferSize: 0,
  })

  // Checkpoint progress (for inference mode)
  const checkpointEpisode = ref(0)
  const checkpointTotalEpisodes = ref(0)
  const autoCheckpointMode = ref(false)  // Auto-update to latest checkpoint after each episode

  // Grid state
  const gridWidth = ref(8)
  const gridHeight = ref(8)
  const agents = ref([])
  const affordances = ref([])
  const gridZoom = ref(1.0) // Zoom level for grid scaling (0.5 - 2.0)

  // Agent meters
  const agentMeters = ref({})

  // Heat map (position visit frequencies)
  const heatMap = ref({})

  // History
  const episodeHistory = ref([])
  const maxHistoryLength = 10

  // Available models
  const availableModels = ref([])

  // RND metrics (for Phase 3 visualization)
  const rndMetrics = ref(null)

  // Affordance transition data (for garden path visualization)
  const transitionData = ref(null)

  // Q-values and affordance stats (for agent behavior panel)
  const qValues = ref([])  // Q-values for all 5 actions
  const affordanceStats = ref([])  // Affordance interaction counts

  // Temporal mechanics state
  const timeOfDay = ref(0)  // Current time of day (0-23)
  const interactionProgress = ref(0)  // Current interaction progress (0-1)

  // Computed
  const averageSurvivalTime = computed(() => {
    if (episodeHistory.value.length === 0) return 0
    const sum = episodeHistory.value.reduce((acc, ep) => acc + ep.steps, 0)
    return Math.round(sum / episodeHistory.value.length)
  })

  // Check which servers are available
  async function checkServerAvailability() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.hostname
    const port = 8766  // Live inference server port

    const checkEndpoint = (endpoint) => {
      return new Promise((resolve) => {
        const wsUrl = `${protocol}//${host}:${port}${endpoint}`
        const testWs = new WebSocket(wsUrl)

        const timeout = setTimeout(() => {
          testWs.close()
          resolve(false)
        }, 2000) // 2 second timeout

        testWs.onopen = () => {
          clearTimeout(timeout)
          testWs.close()
          resolve(true)
        }

        testWs.onerror = () => {
          clearTimeout(timeout)
          resolve(false)
        }
      })
    }

    // Check both endpoints (they're the same in live_inference, but kept for compatibility)
    serverAvailability.value.inference = await checkEndpoint('/ws')
    serverAvailability.value.training = await checkEndpoint('/ws/training')
    serverAvailability.value.checked = true

    console.log('Server availability:', serverAvailability.value)
  }

  // WebSocket connection
  function connect(connectionMode = 'inference') {
    mode.value = connectionMode
    manualDisconnect.value = false // User is connecting, not disconnecting

    // ✅ Set connecting state and clear previous errors
    isConnecting.value = true
    connectionError.value = null

    // Use same host as frontend, allowing remote access
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.hostname

    // Connect to live inference server on port 8766
    const port = 8766
    const endpoint = '/ws'  // Both modes use same endpoint now
    const wsUrl = `${protocol}//${host}:${port}${endpoint}`

    console.log(`Connecting to live inference server:`, wsUrl)

    ws.value = new WebSocket(wsUrl)

    ws.value.onopen = () => {
      console.log('WebSocket connected')
      isConnected.value = true
      isConnecting.value = false  // ✅ Stop loading state
      reconnectAttempts.value = 0

      // Auto-start simulation (no manual play button needed)
      // Wait for next tick to ensure WebSocket is fully ready
      setTimeout(() => {
        console.log('Auto-starting simulation...')
        sendCommand('play')
      }, 100)
    }

    ws.value.onclose = () => {
      console.log('WebSocket disconnected')
      isConnected.value = false
      isConnecting.value = false  // ✅ Stop loading state

      // Only attempt reconnection if it wasn't a manual disconnect
      if (!manualDisconnect.value && reconnectAttempts.value < maxReconnectAttempts) {
        reconnectAttempts.value++
        console.log(`Reconnecting... attempt ${reconnectAttempts.value}`)
        setTimeout(() => {
          connect(mode.value)
        }, reconnectDelay)
      } else if (manualDisconnect.value) {
        console.log('Manual disconnect - not reconnecting')
        reconnectAttempts.value = 0
        connectionError.value = null  // ✅ Clear error on manual disconnect
      } else {
        console.error('Max reconnect attempts reached')
        // ✅ Set error state when max reconnects reached
        connectionError.value = {
          title: 'Connection Lost',
          message: `Failed to reconnect after ${maxReconnectAttempts} attempts. Please check if the backend server is running.`
        }
      }
    }

    ws.value.onerror = (error) => {
      console.error('WebSocket error:', error)
      // ✅ Set error state for UI feedback
      connectionError.value = {
        title: 'Connection Failed',
        message: `Could not connect to ${connectionMode} server. Please check if the backend server is running on port 8765.`
      }
      isConnecting.value = false  // ✅ Stop loading state
    }

    ws.value.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        handleMessage(message)
      } catch (error) {
        console.error('Error handling message:', error, event.data)
      }
    }
  }

  function disconnect() {
    if (ws.value) {
      manualDisconnect.value = true // Mark as manual disconnect
      connectionError.value = null  // ✅ Clear error state
      ws.value.close()
      ws.value = null
    }
  }

  function handleMessage(message) {
    console.log('Received message:', message.type)

    switch (message.type) {
      case 'connected':
        availableModels.value = message.available_models || []
        // Handle checkpoint progress (inference mode)
        if (message.checkpoint_episode !== undefined) {
          checkpointEpisode.value = message.checkpoint_episode
        }
        if (message.total_episodes !== undefined) {
          checkpointTotalEpisodes.value = message.total_episodes
        }
        if (message.epsilon !== undefined) {
          trainingMetrics.value.epsilon = message.epsilon
        }
        if (message.auto_checkpoint_mode !== undefined) {
          autoCheckpointMode.value = message.auto_checkpoint_mode
        }
        break

      case 'training_status':
        isTraining.value = message.is_training
        currentEpisode.value = message.current_episode || 0
        totalEpisodes.value = message.total_episodes || 0
        break

      case 'training_started':
        isTraining.value = true
        totalEpisodes.value = message.num_episodes
        console.log(`Training started: ${message.num_episodes} episodes`)
        break

      case 'episode_start':
        currentEpisode.value = message.episode
        currentStep.value = 0
        cumulativeReward.value = 0
        lastAction.value = null
        if (message.epsilon !== undefined) {
          trainingMetrics.value.epsilon = message.epsilon
        }
        // Handle checkpoint progress (inference mode)
        if (message.checkpoint_episode !== undefined) {
          checkpointEpisode.value = message.checkpoint_episode
        }
        if (message.total_episodes !== undefined) {
          checkpointTotalEpisodes.value = message.total_episodes
        }
        console.log(`Episode ${message.episode} started`)
        break

      case 'state_update':
        updateState(message)
        break

      case 'episode_complete':
        handleEpisodeComplete(message)
        break

      case 'episode_end':
        handleEpisodeEnd(message)
        break

      case 'training_complete':
        isTraining.value = false
        console.log('Training complete!', message)
        break

      case 'model_loaded':
        console.log(`Model loaded: ${message.model}`)
        // Handle checkpoint progress update
        if (message.episode !== undefined) {
          checkpointEpisode.value = message.episode
        }
        if (message.total_episodes !== undefined) {
          checkpointTotalEpisodes.value = message.total_episodes
        }
        if (message.epsilon !== undefined) {
          trainingMetrics.value.epsilon = message.epsilon
        }
        break

      case 'auto_checkpoint_mode':
        autoCheckpointMode.value = message.enabled
        console.log(`Auto checkpoint mode ${message.enabled ? 'enabled' : 'disabled'}`)
        break

      case 'paused':
        console.log('Training paused')
        break

      case 'resumed':
        console.log('Training resumed')
        break

      case 'error':
        console.error('Server error:', message.message)
        break

      default:
        console.warn('Unhandled message type:', message.type, message)
    }
  }

  function updateState(message) {
    currentStep.value = message.step
    cumulativeReward.value = message.cumulative_reward || message.reward || 0

    // Update grid
    if (message.grid) {
      gridWidth.value = message.grid.width
      gridHeight.value = message.grid.height
      affordances.value = message.affordances || message.grid.affordances || []

      const gridAgents = message.grid.agents || []
      if (Array.isArray(gridAgents) && gridAgents.length > 0) {
        agents.value = gridAgents
        lastAction.value = gridAgents[0].last_action || lastAction.value
      } else if (Array.isArray(message.agents)) {
        agents.value = message.agents
        lastAction.value = message.agents[0]?.last_action || lastAction.value
      } else if (message.agents && typeof message.agents === 'object') {
        const values = Object.values(message.agents)
        agents.value = values
        lastAction.value = values[0]?.last_action || lastAction.value
      } else {
        agents.value = []
      }

      if (message.agent_meters) {
        agentMeters.value = message.agent_meters
      } else if (message.agents && typeof message.agents === 'object' && !Array.isArray(message.agents)) {
        agentMeters.value = message.agents
      } else {
        agentMeters.value = {}
      }
    }

    // Handle heat map data
    if (message.heat_map) {
      heatMap.value = message.heat_map
    }

    // Handle RND metrics (Phase 3)
    if (message.rnd_metrics) {
      rndMetrics.value = message.rnd_metrics
    }

    // Handle affordance transition data (garden path)
    if (message.affordance_graph) {
      transitionData.value = message.affordance_graph
    }

    // Handle Q-values and affordance stats (agent behavior panel)
    if (message.q_values) {
      qValues.value = message.q_values
    }
    if (message.affordance_stats) {
      affordanceStats.value = message.affordance_stats
    }

    // Handle temporal mechanics data
    if (message.temporal) {
      timeOfDay.value = message.temporal.time_of_day
      interactionProgress.value = message.temporal.interaction_progress
    }
  }

  function handleEpisodeComplete(message) {
    console.log(`Episode ${message.episode} complete: ${message.length} steps, reward: ${message.reward}`)

    // Update training metrics
    trainingMetrics.value = {
      avgReward5: message.avg_reward_5 || 0,
      avgLength5: message.avg_length_5 || 0,
      avgLoss5: message.avg_loss_5 || 0,
      epsilon: message.epsilon || 0,
      bufferSize: message.buffer_size || 0,
    }

    // Add to history
    episodeHistory.value.push({
      episode: message.episode,
      steps: message.length,
      reward: message.reward,
      loss: message.loss,
    })

    // Keep only last N episodes
    if (episodeHistory.value.length > maxHistoryLength) {
      episodeHistory.value = episodeHistory.value.slice(-maxHistoryLength)
    }
  }

  function handleEpisodeEnd(message) {
    console.log(`Episode ${message.episode} ended: ${message.steps} steps, reward: ${message.total_reward}`)

    // Add to history
    episodeHistory.value.push({
      episode: message.episode,
      steps: message.steps,
      reward: message.total_reward,
      reason: message.reason,
    })

    // Keep only last N episodes
    if (episodeHistory.value.length > maxHistoryLength) {
      episodeHistory.value = episodeHistory.value.slice(-maxHistoryLength)
    }
  }

  // Control commands
  function sendCommand(command, params = {}) {
    if (!ws.value || !isConnected.value) {
      console.error('WebSocket not connected')
      return
    }

    ws.value.send(JSON.stringify({
      type: 'control',
      command,
      ...params
    }))
  }

  // Auto-plays on connect, no need for play/pause/step/reset controls

  function setSpeed(speed) {
    sendCommand('set_speed', { speed })
  }

  function loadModel(modelName) {
    sendCommand('load_model', { model: modelName })
  }

  function refreshCheckpoint() {
    sendCommand('refresh_checkpoint')
  }

  function toggleAutoCheckpoint() {
    sendCommand('toggle_auto_checkpoint')
  }

  // Training commands
  function startTraining(numEpisodes = 100, batchSize = 32, bufferCapacity = 10000, showEvery = 5, stepDelay = 0.2) {
    if (!ws.value || !isConnected.value) {
      console.error('WebSocket not connected')
      return
    }

    ws.value.send(JSON.stringify({
      command: 'start_training',
      num_episodes: numEpisodes,
      batch_size: batchSize,
      buffer_capacity: bufferCapacity,
      show_every: showEvery,
      step_delay: stepDelay,
    }))
  }

  function setZoom(zoom) {
    gridZoom.value = Math.max(0.5, Math.min(2.0, zoom))
  }

  return {
    // State
    isConnected,
    isConnecting,
    connectionError,
    serverAvailability,
    mode,
    currentEpisode,
    currentStep,
    cumulativeReward,
    lastAction,
    gridWidth,
    gridHeight,
    agents,
    affordances,
    gridZoom,
    agentMeters,
    heatMap,
    episodeHistory,
    availableModels,
    averageSurvivalTime,
    rndMetrics,
    transitionData,
    qValues,
    affordanceStats,
    timeOfDay,
    interactionProgress,

    // Training state
    isTraining,
    totalEpisodes,
    trainingMetrics,

    // Checkpoint progress
    checkpointEpisode,
    checkpointTotalEpisodes,
    autoCheckpointMode,

    // Actions
    checkServerAvailability,
    connect,
    disconnect,
    setSpeed,
    setZoom,
    loadModel,
    refreshCheckpoint,
    toggleAutoCheckpoint,
    startTraining,
  }
})
