<template>
  <section class="failure-panel" aria-labelledby="failure-heading">
    <div class="panel-header">
      <h3 id="failure-heading">Failure Log</h3>
      <button class="refresh-button" type="button" @click="refresh" :disabled="isLoading">
        {{ isLoading ? 'Loading…' : 'Refresh' }}
      </button>
    </div>

    <form class="filters" @submit.prevent>
      <label>
        <span>Agent</span>
        <input
          v-model.trim="filters.agent"
          type="text"
          placeholder="All"
          aria-label="Filter by agent id"
        />
      </label>
      <label>
        <span>Reason</span>
        <input
          v-model.trim="filters.reason"
          type="text"
          placeholder="Any"
          aria-label="Filter by failure reason"
        />
      </label>
      <label>
        <span>Limit</span>
        <select v-model.number="filters.limit" aria-label="Limit failures">
          <option v-for="option in limitOptions" :key="option" :value="option">
            {{ option }}
          </option>
        </select>
      </label>
    </form>

    <p v-if="error" class="error" role="alert">{{ error }}</p>

    <div v-if="summary.length" class="summary" role="region" aria-label="Failure summary">
      <h4>Top Reasons</h4>
      <ul>
        <li v-for="row in summary" :key="`${row.agent_id}-${row.reason}`">
          <span class="summary-reason">{{ row.reason }}</span>
          <span class="summary-count">{{ row.count }}</span>
          <span class="summary-agent">{{ row.agent_id }}</span>
          <span class="summary-episode">Last ep {{ row.last_episode }}</span>
        </li>
      </ul>
    </div>

    <div class="table-wrapper" role="region" aria-label="Failure events table">
      <table v-if="failures.length">
        <thead>
          <tr>
            <th scope="col">Episode</th>
            <th scope="col">Agent</th>
            <th scope="col">Reason</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="failure in failures" :key="failure.id ?? failure.timestamp + failure.episode">
            <td>#{{ failure.episode }}</td>
            <td>{{ failure.agent_id }}</td>
            <td><code>{{ failure.reason }}</code></td>
          </tr>
        </tbody>
      </table>
      <p v-else-if="isLoading" class="placeholder">Loading failure events…</p>
      <p v-else class="placeholder">No failure events recorded yet.</p>
    </div>
  </section>
</template>

<script setup>
import { onMounted, onBeforeUnmount, reactive, ref, watch } from 'vue'

const filters = reactive({
  agent: '',
  reason: '',
  limit: 10,
})

const limitOptions = [5, 10, 20, 50]

const failures = ref([])
const summary = ref([])
const isLoading = ref(false)
const error = ref(null)

let debounceTimer = null

async function fetchFailures() {
  const params = new URLSearchParams()
  if (filters.agent) params.append('agent', filters.agent)
  if (filters.reason) params.append('reason', filters.reason)
  if (filters.limit) params.append('limit', String(filters.limit))

  const summaryParams = new URLSearchParams()
  if (filters.agent) summaryParams.append('agent', filters.agent)
  if (filters.reason) summaryParams.append('reason', filters.reason)
  summaryParams.append('top', '5')

  const [failuresResp, summaryResp] = await Promise.all([
    fetch(`/api/failures?${params.toString()}`),
    fetch(`/api/failure_summary?${summaryParams.toString()}`)
  ])

  if (failuresResp.status === 404 || summaryResp.status === 404) {
    failures.value = []
    summary.value = []
    error.value = 'Failure tracking is not enabled on this server.'
    return
  }

  if (!failuresResp.ok) {
    throw new Error(`Failed to load failures (${failuresResp.status})`)
  }
  if (!summaryResp.ok) {
    throw new Error(`Failed to load failure summary (${summaryResp.status})`)
  }

  const failuresData = await failuresResp.json()
  const summaryData = await summaryResp.json()

  failures.value = failuresData.failures ?? []
  summary.value = summaryData.summary ?? []
}

async function refresh() {
  isLoading.value = true
  error.value = null
  try {
    await fetchFailures()
  } catch (err) {
    console.error(err)
    error.value = err instanceof Error ? err.message : 'Failed to load failure events'
  } finally {
    isLoading.value = false
  }
}

function scheduleRefresh(delay = 0) {
  clearTimeout(debounceTimer)
  debounceTimer = setTimeout(() => {
    refresh()
  }, delay)
}

watch(() => [filters.agent, filters.reason], () => scheduleRefresh(400))
watch(() => filters.limit, () => scheduleRefresh(0))

onMounted(() => {
  refresh()
})

onBeforeUnmount(() => {
  clearTimeout(debounceTimer)
})

function formatTimestamp() {
  return ''
}
</script>

<style scoped>
.failure-panel {
  margin-top: var(--spacing-lg);
  padding: var(--spacing-md);
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius-md);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  box-shadow: var(--shadow-sm);
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--spacing-sm);
}

.panel-header h3 {
  margin: 0;
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
}

.refresh-button {
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--border-radius-sm);
  background: var(--color-meter-mood-high);
  color: var(--color-text-inverse, #fff);
  border: none;
  font-size: var(--font-size-sm);
  cursor: pointer;
  transition: var(--transition-base);
}

.refresh-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.filters {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: var(--spacing-sm);
}

.filters label {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
}

.filters input,
.filters select {
  padding: var(--spacing-xs);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--color-bg-tertiary);
  background: var(--color-bg-primary);
  color: inherit;
}

.error {
  color: var(--color-error);
  font-size: var(--font-size-sm);
}

.summary ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.summary li {
  display: flex;
  gap: var(--spacing-sm);
  align-items: baseline;
  font-size: var(--font-size-sm);
}

.summary-reason {
  font-weight: var(--font-weight-semibold);
}

.summary-count {
  color: var(--color-text-secondary);
}

.summary-agent {
  margin-left: auto;
  color: var(--color-text-tertiary);
}

.summary-episode {
  color: var(--color-text-tertiary);
}

.table-wrapper {
  overflow-x: auto;
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--color-bg-tertiary);
}

.table-wrapper table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--font-size-sm);
}

.table-wrapper th,
.table-wrapper td {
  padding: var(--spacing-xs) var(--spacing-sm);
  border-bottom: 1px solid var(--color-bg-tertiary);
  text-align: left;
}

.table-wrapper th {
  background: var(--color-bg-tertiary);
  font-weight: var(--font-weight-semibold);
}

.table-wrapper code {
  font-family: var(--font-family-mono, 'Fira Code', monospace);
  background: var(--color-bg-tertiary);
  padding: 0 var(--spacing-xxs, 2px);
  border-radius: var(--border-radius-xs, 3px);
}

.placeholder {
  margin: var(--spacing-md) 0;
  text-align: center;
  color: var(--color-text-secondary);
}

@media (max-width: 768px) {
  .failure-panel {
    margin-top: var(--spacing-md);
  }
}
</style>
