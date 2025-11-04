<template>
  <div class="app">
    <!-- ✅ Semantic HTML: header with role="banner" -->
    <header class="header" role="banner">
      <h1>HAMLET - DRL Visualization</h1>
    </header>

    <!-- ✅ Semantic HTML: main with role="main" -->
    <main class="main-container" role="main">
      <!-- ✅ Top row: three-column layout (meters, grid, stats) -->
      <div class="top-row">
        <!-- ✅ Semantic HTML: aside for meters panel (left) -->
        <aside class="meters-panel" aria-label="Agent status panels">
          <MeterPanel
            :agent-meters="store.agentMeters"
            :lifetime-progress="store.lifetimeProgress"
            :agent-age="store.agentAge"
          />

          <!-- ✅ Projected reward bar (below meters) - always shown when connected -->
          <ProjectedRewardBar
            v-if="isConnected"
            :current-step="store.currentStep"
            :baseline-survival="store.baselineSurvival"
            :step-reward="store.stepReward"
          />
        </aside>

        <!-- Agent Behaviour (consolidated panel - centre left) -->
        <aside class="behavior-panel" aria-label="Agent behaviour">
          <AgentBehaviorPanel
            v-if="store.isConnected"
            :cumulative-reward="store.cumulativeReward"
            :last-action="store.lastAction"
            :current-step="store.currentStep"
            :epsilon="store.trainingMetrics.epsilon"
            :checkpoint-episode="store.checkpointEpisode"
            :total-episodes="store.checkpointTotalEpisodes"
            :q-values="store.qValues"
            :action-masks="store.actionMasks"
            :affordance-stats="store.affordanceStats"
          />
        </aside>

      <!-- ✅ Semantic HTML: region for grid visualization -->
      <div class="grid-container" role="region" aria-label="Simulation grid">
        <!-- ✅ Time of day bar (top-left) - always shown when connected (cycles naturally) -->
        <TimeOfDayBar
          v-if="isConnected"
          :time-of-day="store.timeOfDay"
          :current-step="store.currentStep"
        />

        <!-- ✅ Minimal top-right controls (only when connected) -->
        <MinimalControls
          v-if="isConnected"
          :is-connected="store.isConnected"
          :auto-mode="store.autoCheckpointMode"
          @disconnect="store.disconnect"
          @set-speed="store.setSpeed"
          @refresh-checkpoint="store.refreshCheckpoint"
          @toggle-auto-checkpoint="store.toggleAutoCheckpoint"
        />

        <!-- Death Certificates (below speed controls) -->
        <DeathCertificates
          v-if="isConnected"
          :certificates="store.deathCertificates"
        />

        <!-- ✅ Zoom control in bottom right (only when connected) -->
        <ZoomControl
          v-if="isConnected"
          :zoom="store.gridZoom"
          @update:zoom="store.setZoom"
          class="zoom-control-position"
        />

        <!-- ✅ Show loading state while connecting -->
        <LoadingState
          v-if="isConnecting"
          message="Connecting to simulation server..."
        />

        <!-- ✅ Show error state if connection failed -->
        <ErrorState
          v-else-if="connectionError"
          :title="connectionError.title"
          :message="connectionError.message"
          @retry="retryConnection"
        />

        <!-- ✅ Show grid when connected -->
        <div
          v-else-if="isConnected"
          class="grid-wrapper"
          :style="{ transform: `scale(${store.gridZoom})` }"
        >
          <Grid
            :grid-width="store.gridWidth"
            :grid-height="store.gridHeight"
            :agents="store.agents"
            :affordances="store.affordances"
            :heat-map="store.heatMap"
          />
          <!-- Interaction progress ring overlay (temporal mechanics) -->
          <InteractionProgressRing
            v-if="store.agents && store.agents.length > 0 && store.interactionProgress > 0"
            :x="store.agents[0].x"
            :y="store.agents[0].y"
            :progress="store.interactionProgress"
            :affordance-type="currentAffordanceType"
            :cell-size="75"
          />
          <!-- Phase 3: Novelty heatmap overlay -->
          <NoveltyHeatmap
            v-if="store.rndMetrics && store.rndMetrics.novelty_map"
            :novelty-map="store.rndMetrics.novelty_map"
            :grid-size="store.gridWidth"
            :cell-size="75"
          />
        </div>

        <!-- ✅ Show empty state when not connected -->
        <div v-else class="not-connected-message">
          <p>Not connected to simulation server.</p>
          <p class="hint">Click here to connect and start the simulation.</p>
          <button @click="handleConnect('inference')" class="connect-button">
            Connect
          </button>
        </div>
      </div>

        <!-- ✅ Semantic HTML: aside for right panel (stats and visualizations) -->
        <aside class="right-panel" aria-label="Episode statistics and visualizations">
          <StatsPanel
            :current-episode="store.currentEpisode"
            :current-step="store.currentStep"
            :cumulative-reward="store.cumulativeReward"
            :last-action="store.lastAction"
            :episode-history="store.episodeHistory"
            :checkpoint-episode="store.checkpointEpisode"
          />
          <!-- Critical Event Log -->
          <CriticalEventLog
            v-if="store.isConnected"
            :events="criticalEvents"
            @clear="clearCriticalEvents"
          />
          <!-- Phase 3: Reward charts -->
          <IntrinsicRewardChart
            v-if="store.rndMetrics"
            :extrinsic-history="extrinsicHistory"
            :intrinsic-history="intrinsicHistory"
            :width="300"
            :height="150"
          />
          <SurvivalTrendChart
            v-if="survivalTrend.length > 0"
            :trend-data="survivalTrend"
            :width="300"
            :height="150"
          />
          <AffordanceGraph
            v-if="store.transitionData"
            :transition-data="store.transitionData"
            :width="300"
            :height="250"
          />
        </aside>
      </div>
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useSimulationStore } from './stores/simulation'
import Grid from './components/Grid.vue'
import MeterPanel from './components/MeterPanel.vue'
import MinimalControls from './components/MinimalControls.vue'
import DeathCertificates from './components/DeathCertificates.vue'
import TimeOfDayBar from './components/TimeOfDayBar.vue'
import ProjectedRewardBar from './components/ProjectedRewardBar.vue'
import InteractionProgressRing from './components/InteractionProgressRing.vue'
import StatsPanel from './components/StatsPanel.vue'
import LoadingState from './components/LoadingState.vue'
import ErrorState from './components/ErrorState.vue'
import NoveltyHeatmap from './components/NoveltyHeatmap.vue'
import IntrinsicRewardChart from './components/IntrinsicRewardChart.vue'
import CurriculumTracker from './components/CurriculumTracker.vue'
import AgentBehaviorPanel from './components/AgentBehaviorPanel.vue'
import CriticalEventLog from './components/CriticalEventLog.vue'
import SurvivalTrendChart from './components/SurvivalTrendChart.vue'
import AffordanceGraph from './components/AffordanceGraph.vue'
import ZoomControl from './components/ZoomControl.vue'

const store = useSimulationStore()
const isConnected = computed(() => store.isConnected)
const isConnecting = computed(() => store.isConnecting)
const connectionError = computed(() => store.connectionError)

// Compute current affordance agent is on (for interaction progress ring)
const currentAffordanceType = computed(() => {
  if (!store.agents || store.agents.length === 0) return null
  if (!store.affordances || store.affordances.length === 0) return null

  const agent = store.agents[0]
  const affordance = store.affordances.find(
    a => a.x === agent.x && a.y === agent.y
  )
  return affordance ? affordance.type : null
})

// Phase 3: Track reward histories and survival trend
const extrinsicHistory = ref([])
const intrinsicHistory = ref([])
const survivalTrend = ref([])

// Critical event log
const criticalEvents = ref([])
let eventIdCounter = 0

// Meter tier classification for cascade detection
// Updated to match actual game code mechanics (secondary in alphabetical order)
const meterTiers = {
  primary: ['energy', 'health', 'money'],
  secondary: ['fitness', 'mood', 'satiation'],
  tertiary: ['hygiene', 'social']
}

// Cascade explanations (matching actual game code)
const cascadeMessages = {
  // Tertiary → Secondary (accelerators)
  hygiene: 'Poor hygiene → depletes satiation, fitness & mood',
  social: 'Loneliness → accelerates mood decline',

  // Secondary → Primary (direct modifiers)
  mood: 'Low mood → depletes energy',
  satiation: 'Hunger → depletes energy & health',
  fitness: 'Poor fitness → accelerates health decline',

  // Primary (survival-critical)
  energy: 'CRITICAL: Energy depletion → imminent death risk',
  health: 'CRITICAL: Health failure → death imminent',
  money: 'CRITICAL: No money → cannot afford survival needs'
}

// Watch for RND metrics updates
watch(() => store.rndMetrics, (newMetrics) => {
  if (newMetrics) {
    // Update reward histories (keep last 100)
    if (newMetrics.extrinsic_reward !== undefined) {
      extrinsicHistory.value.push(newMetrics.extrinsic_reward)
      if (extrinsicHistory.value.length > 100) {
        extrinsicHistory.value.shift()
      }
    }

    if (newMetrics.intrinsic_reward !== undefined) {
      intrinsicHistory.value.push(newMetrics.intrinsic_reward)
      if (intrinsicHistory.value.length > 100) {
        intrinsicHistory.value.shift()
      }
    }

    // Update survival trend (avg per 100 episodes)
    if (newMetrics.avg_survival_last_100 !== undefined) {
      survivalTrend.value.push(newMetrics.avg_survival_last_100)
    }
  }
})

// Watch for episode end to clear critical events
watch(() => store.currentEpisode, (newEpisode, oldEpisode) => {
  // Clear events when episode changes (new episode started)
  if (newEpisode !== oldEpisode && oldEpisode > 0) {
    criticalEvents.value = []
  }
})

// Watch for critical meter events
watch(() => store.agentMeters, (newMeters) => {
  if (!newMeters || !newMeters.agent_0) return

  const meters = newMeters.agent_0.meters
  if (!meters) return

  // Check each meter for critical threshold (<20%)
  Object.entries(meters).forEach(([meterName, value]) => {
    const percentage = value * 100

    if (percentage < 20 && percentage > 0) {
      // Find meter tier
      let tier = 'tertiary'
      if (meterTiers.primary.includes(meterName)) tier = 'primary'
      else if (meterTiers.secondary.includes(meterName)) tier = 'secondary'

      // Create event
      const event = {
        id: eventIdCounter++,
        meterName,
        value: Math.round(percentage),
        tier,
        cascade: cascadeMessages[meterName],
        timestamp: Date.now()
      }

      // Avoid duplicate events for same meter in quick succession
      const recentEvent = criticalEvents.value.find(
        e => e.meterName === meterName && (Date.now() - e.timestamp) < 5000
      )

      if (!recentEvent) {
        criticalEvents.value.push(event)
      }
    }
  })
}, { deep: true })

// Check which servers are available on mount
onMounted(() => {
  store.checkServerAvailability()
})

// ✅ Retry connection on error
function retryConnection() {
  store.connect(store.mode)
}

// Clear critical event log
function clearCriticalEvents() {
  criticalEvents.value = []
}

// ✅ Handle connect event (always inference mode - auto-plays on connect)
function handleConnect(mode = 'inference') {
  store.connect(mode)
}
</script>

<style scoped>
/* ✅ Refactored to use design tokens from variables.css */
.app {
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  overflow: hidden;
}

/* ✅ Mobile-first header */
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-sm) var(--spacing-md);
  border-bottom: 1px solid var(--color-bg-tertiary);
  flex-wrap: wrap;
  gap: var(--spacing-sm);
}

.header h1 {
  margin: 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
}

/* ✅ Larger header on tablet+ */
@media (min-width: 768px) {
  .header {
    padding: var(--spacing-md) var(--spacing-lg);
    flex-wrap: nowrap;
  }

  .header h1 {
    font-size: var(--font-size-xl);
  }
}

/* ✅ Full size header on desktop */
@media (min-width: 1024px) {
  .header {
    padding: var(--spacing-md) var(--spacing-xl);
  }

  .header h1 {
    font-size: var(--font-size-2xl);
  }
}

/* ✅ Mobile-first responsive layout - now with top row + bottom panel */
.main-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0; /* No gap - bottom panel handles its own spacing */
  padding: 0;
  overflow: hidden;
}

/* Top row contains the three-column layout (meters, grid, stats) */
.top-row {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  overflow-y: auto;
  overflow-x: hidden;
}

/* Meters panel (left) */
.meters-panel {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 2px;
  height: 100%;
  min-height: 0; /* Allow flex children to shrink */
}

/* Behaviour panel (centre-left) */
.behavior-panel {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  height: 100%;
  min-height: 0; /* Allow flex children to shrink */
}

.grid-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  min-width: 0;
  height: 100%;
  min-height: 0; /* Allow flex children to shrink */
  position: relative; /* For absolute positioning of MinimalControls */
}

.grid-wrapper {
  position: relative;
  transition: transform var(--transition-base);
  transform-origin: center;
}

/* ✅ Position zoom control in bottom right corner of grid container */
.zoom-control-position {
  position: absolute;
  bottom: var(--spacing-md);
  right: var(--spacing-md);
  z-index: var(--z-index-dropdown);
}

.right-panel {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  height: 100%;
  min-height: 0; /* Allow flex children to shrink */
}

/* ✅ Tablet breakpoint: side-by-side panels */
@media (min-width: 768px) {
  .top-row {
    flex-direction: row;
    gap: var(--spacing-lg);
    padding: var(--spacing-lg);
  }

  .meters-panel {
    width: 300px;
    flex-shrink: 0;
    gap: 2px;
  }

  .behavior-panel {
    width: 300px;
    flex-shrink: 0;
    gap: var(--spacing-lg);
  }

  .grid-container {
    padding: var(--spacing-lg);
  }

  .right-panel {
    width: 300px;
    flex-shrink: 0;
    gap: var(--spacing-lg);
  }
}

/* ✅ Desktop breakpoint: full layout */
@media (min-width: 1024px) {
  .top-row {
    padding: var(--spacing-xl);
  }

  .meters-panel {
    width: 300px;
  }

  .behavior-panel {
    width: 300px;
  }

  .grid-container {
    padding: var(--spacing-xl);
  }

  .right-panel {
    width: 300px;
  }
}

/* ✅ Not connected message styling */
.not-connected-message {
  text-align: center;
  color: var(--color-text-secondary);
  padding: var(--spacing-2xl);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-md);
}

.not-connected-message p {
  margin: 0;
  font-size: var(--font-size-base);
}

.not-connected-message .hint {
  font-size: var(--font-size-sm);
  color: var(--color-text-tertiary);
  max-width: 400px;
}

.connect-button {
  padding: var(--spacing-md) var(--spacing-xl);
  background: var(--color-success);
  border: none;
  border-radius: var(--border-radius-md);
  color: var(--color-text-on-dark);
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-semibold);
  cursor: pointer;
  transition: all var(--transition-base);
  margin-top: var(--spacing-md);
}

.connect-button:hover {
  background: var(--color-interactive-hover);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
}

.connect-button:active {
  transform: translateY(0);
}
</style>
