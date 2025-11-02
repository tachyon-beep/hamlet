# Live Inference Telemetry Schema

**Schema Version:** 1.0.0  
**Last Updated:** 2025-11-02  
**Source:** `src/townlet/demo/live_inference.py`

## Envelope

All telemetry payloads exposed in live inference (`episode_start`, `state_update`, `episode_end`) now include a `telemetry` field:

```json
{
  "schema_version": "1.0.0",
  "episode_index": 512,
  "agents": [
    {
      "agent_id": "agent-0",
      "baseline_survival_steps": 100.0,
      "survival_time": 32,
      "curriculum_stage": 2,
      "epsilon": 0.05,
      "intrinsic_weight": 0.2,
      "timestamp_unix": 1762065600.123
    }
  ]
}
```

## Field Definitions

- `schema_version` (str)  
  Semantic version string (e.g., `"1.0.0"`). Increment the major component for breaking changes. Clients must branch on this value for backwards compatibility.

- `episode_index` (int | null)  
  Episode number when the snapshot was generated. `null` indicates the emitter could not determine the current episode (e.g., inference idle state).

- `agents` (list)  
  One entry per agent managed by the runtime registry.

### Agent Entry

- `agent_id` (str)  
  Stable identifier matching the agent’s index in the population (`agent-<idx>` by default).

- `baseline_survival_steps` (float)  
  Baseline survival expectation (R) used for reward shaping at the time of snapshot. Units: environment steps.

- `survival_time` (int)  
  Steps survived in the current episode at the moment of snapshot.

- `curriculum_stage` (int)  
  Current curriculum difficulty stage (1..5).

- `epsilon` (float)  
  Exploration rate applied to the agent when the snapshot was generated.

- `intrinsic_weight` (float)  
  Current scaling factor applied to intrinsic reward (e.g., RND weight).

- `timestamp_unix` (float)  
  Wall-clock timestamp (seconds since Unix epoch) when the snapshot was materialised.

## Generation

Snapshots are produced by `VectorizedPopulation.build_telemetry_snapshot()`, which calls `AgentRuntimeRegistry.get_snapshot_for_agent()` for each agent and normalises the payload to JSON-safe primitives.  
`build_agent_telemetry_payload()` (live inference) forwards this snapshot and attaches the schema version.

## Client Guidance

1. **Version branch:** Treat `schema_version` as authoritative. Unknown versions should degrade gracefully (log + safely ignore).
2. **Optional fields:** Future versions may add optional keys. Do not assume additional keys are absent.
3. **Timestamps:** Use `timestamp_unix` for ordering snapshots if message latency varies.
4. **Audit logging:** Store the entire telemetry envelope per message to enable retrospective incident analysis.

## Change Log

- **v1.0.0 (2025-11-02):** Initial versioned schema introducing structured agent payloads.
