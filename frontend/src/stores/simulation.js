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

  // Grid state
  const gridWidth = ref(8)
  const gridHeight = ref(8)
  const agents = ref([])
  const affordances = ref([])

  // Agent meters
  const agentMeters = ref({})

  // History
  const episodeHistory = ref([])
  const maxHistoryLength = 10

  // Available models
  const availableModels = ref([])

  // Computed
  const averageSurvivalTime = computed(() => {
    if (episodeHistory.value.length === 0) return 0
    const sum = episodeHistory.value.reduce((acc, ep) => acc + ep.steps, 0)
    return Math.round(sum / episodeHistory.value.length)
  })

  // WebSocket connection
  function connect(connectionMode = 'inference') {
    mode.value = connectionMode
    manualDisconnect.value = false // User is connecting, not disconnecting

    // Use same host as frontend, allowing remote access
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.hostname

    // Both modes use port 8765, but different endpoints
    const port = 8765
    const endpoint = connectionMode === 'training' ? '/ws/training' : '/ws'
    const wsUrl = `${protocol}//${host}:${port}${endpoint}`

    console.log(`Connecting to ${connectionMode} mode:`, wsUrl)

    ws.value = new WebSocket(wsUrl)

    ws.value.onopen = () => {
      console.log('WebSocket connected')
      isConnected.value = true
      reconnectAttempts.value = 0
    }

    ws.value.onclose = () => {
      console.log('WebSocket disconnected')
      isConnected.value = false

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
      } else {
        console.error('Max reconnect attempts reached')
      }
    }

    ws.value.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.value.onmessage = (event) => {
      const message = JSON.parse(event.data)
      handleMessage(message)
    }
  }

  function disconnect() {
    if (ws.value) {
      manualDisconnect.value = true // Mark as manual disconnect
      ws.value.close()
      ws.value = null
    }
  }

  function handleMessage(message) {
    console.log('Received message:', message.type)

    switch (message.type) {
      case 'connected':
        availableModels.value = message.available_models || []
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
    }
  }

  function updateState(message) {
    currentStep.value = message.step
    cumulativeReward.value = message.cumulative_reward || message.reward || 0

    // Update grid
    if (message.grid) {
      gridWidth.value = message.grid.width
      gridHeight.value = message.grid.height
      affordances.value = message.affordances || []

      // Handle agents - should be an array from training server
      if (message.agents) {
        if (Array.isArray(message.agents)) {
          agents.value = message.agents
        } else {
          // Fallback: convert dict to array
          agents.value = Object.values(message.agents)
        }

        // Extract last action from first agent
        const firstAgent = agents.value[0]
        if (firstAgent) {
          lastAction.value = firstAgent.last_action
        }
      } else {
        agents.value = []
      }

      // Handle agent meters (separate from agents array)
      if (message.agent_meters) {
        agentMeters.value = message.agent_meters
      } else if (message.agents && !Array.isArray(message.agents)) {
        // Fallback for old format
        agentMeters.value = message.agents
      }
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

  function play() {
    sendCommand('play')
  }

  function pause() {
    sendCommand('pause')
  }

  function step() {
    sendCommand('step')
  }

  function reset() {
    sendCommand('reset')
  }

  function setSpeed(speed) {
    sendCommand('set_speed', { speed })
  }

  function loadModel(modelName) {
    sendCommand('load_model', { model: modelName })
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

  return {
    // State
    isConnected,
    mode,
    currentEpisode,
    currentStep,
    cumulativeReward,
    lastAction,
    gridWidth,
    gridHeight,
    agents,
    affordances,
    agentMeters,
    episodeHistory,
    availableModels,
    averageSurvivalTime,

    // Training state
    isTraining,
    totalEpisodes,
    trainingMetrics,

    // Actions
    connect,
    disconnect,
    play,
    pause,
    step,
    reset,
    setSpeed,
    loadModel,
    startTraining,
  }
})
