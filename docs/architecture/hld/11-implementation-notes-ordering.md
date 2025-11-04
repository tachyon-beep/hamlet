## 11. Implementation notes (ordering)

This section is about "what order do we do this in so we don't set ourselves on fire". It's the recommended build sequence for Townlet v2.5.

You do these in order. If you jump around, the audit story collapses and you'll end up duct-taping provenance on later, which never works.

### 11.1 Snapshot discipline first

Goal: lock down provenance from day one.

- Create `configs/<run_name>/` with all 5 YAMLs:

  - `config.yaml`
  - `universe_as_code.yaml`
  - `cognitive_topology.yaml` (Layer 1)
  - `agent_architecture.yaml` (Layer 2)
  - `execution_graph.yaml` (Layer 3)

- Write the launcher so that when you "start run", it immediately:

  - creates `runs/<run_name>__<timestamp>/`
  - copies those 5 YAMLs byte-for-byte into `runs/<run_name>__<timestamp>/config_snapshot/`
  - creates empty subdirs: `checkpoints/`, `telemetry/`, `logs/`

Rules:

- Snapshot is a physical copy, not a symlink.
- After launch, the live process never silently re-reads from `configs/<run_name>/`. The snapshot is now truth.
- All provenance, audit, and replay logic assume the snapshot is the canonical contract for that run.

Why this is first:

- If you don't freeze the world and the mind at launch, you can't prove anything later. Governance dies right here.
- Also: the rest of the system (factory, hashing, checkpoints) all builds on the assumption that the snapshot is the single source of truth.

---

### 11.2 Build the minimal GraphAgent pipeline

Goal: replace the old monolithic RL agent class with a graph-driven brain that can think() once.

Deliverables:

- `agent/factory.py`

  - Reads the run's `config_snapshot/`
  - Builds each module declared in `agent_architecture.yaml` (perception_encoder, world_model, social_model, hierarchical_policy, panic_controller, EthicsFilter, etc)
  - Wires in behavioural knobs from Layer 1 (panic_thresholds, forbid_actions, rollout_depth, social_model.enabled)
  - Verifies interface dims declared in `interfaces` (belief_distribution_dim, action_space_dim, etc)
  - Assembles a registry of modules (e.g. an `nn.ModuleDict`)

- `agent/graph_executor.py`

  - Reads `execution_graph.yaml`
  - Compiles it into a deterministic ordered step list with explicit dataflow
  - Resolves each `"@modules.*"` and `"@config.L1.*"` reference into actual callables/values
  - Knows how to run one tick: perception → policy → panic_controller → EthicsFilter → final_action
  - Produces named outputs (`final_action`, `new_recurrent_state`) and intermediate signals for telemetry

- `agent/graph_agent.py`

  - Owns the module registry and the executor
  - Stores persistent recurrent state
  - Exposes `think(raw_observation, prev_recurrent_state) -> { final_action, new_recurrent_state }`

For the first cut:

- world_model_service can just be a stub
- social_model_service can return "disabled"
- panic_controller can just pass through
- EthicsFilter can just pass through

Why this is second:

- Until you have a callable brain built from YAML + snapshot, you can't hash cognition, you can't checkpoint provenance, you can't expose the think loop, you can't do glass box UI. Everything else depends on this.

---

### 11.3 Cognitive hash

Goal: give the instantiated mind a provable identity.

Implement `cognitive_hash.txt` generator. This hash (for example SHA-256) must deterministically cover:

1. The exact bytes of all 5 YAMLs in the run's `config_snapshot/`, concatenated in a defined order:

   - `config.yaml`
   - `universe_as_code.yaml`
   - `cognitive_topology.yaml` (Layer 1)
   - `agent_architecture.yaml` (Layer 2)
   - `execution_graph.yaml` (Layer 3)

2. The compiled execution graph:

   - After `graph_executor` resolves bindings like `@modules.world_model` and `@config.L1.panic_thresholds`
   - After it expands the step order and knows exactly which module is called, in what sequence, with what inputs, and which veto gates get applied

3. The instantiated architectures:

   - For each module (perception_encoder, world_model, etc):

     - type (MLP, CNN, GRU, etc)
     - layer sizes / hidden dims
     - optimiser type and learning rate
     - interface dimensions (e.g. `belief_distribution_dim: 128`)

If any of those change, the hash changes. That's the whole point. You cannot secretly "just tweak panic thresholds" and pretend it's the same mind.

Why we do it here:

- Hashing has to exist before checkpoints so you can stamp checkpoints with identity.
- Hashing also feeds telemetry: telemetry every tick logs `full_cognitive_hash` so you can prove "this exact mind did this".

---

### 11.4 Checkpoint writer and resume

Goal: pause/replay/fork without lying to audit.

The checkpoint writer must emit, under `runs/<run_id>/checkpoints/step_<N>/`:

- `weights.pt`
  - all module weights from the GraphAgent (including EthicsFilter, panic_controller, etc)
- `optimizers.pt`
  - optimiser states for each trainable module
- `rng_state.json`
  - RNG state for both sim and agent
- `config_snapshot/`
  - deep copy of the snapshot as of this checkpoint (not a pointer to `configs/`)
- `cognitive_hash.txt`
  - the full hash at this checkpoint

Resume rules:

- Resume never consults `configs/<run_name>/`.
- Resume loads only from the checkpoint directory.
- Resume starts a new run folder named `..._resume_<timestamp>/` with the restored snapshot.
- If you haven't touched the snapshot, the resumed brain produces the same cognitive hash.

Branching:

- If you edit the snapshot before resuming (e.g. change `panic_thresholds`, disable `social_model.enabled`, lower `greed`, change rollout_depth), that is a fork. New hash, new run_id. We do not lie about continuity.

This gives you:

- Long training jobs across interruptions
- Honest ablations ("same weights, same world, except panic disabled")
- True line of custody for behaviour

---

### 11.5 Telemetry and UI

Goal: make cognition observable in real time and scrubbable after the fact.

Two deliverables here:

1. Telemetry writer

   - For every tick, write a structured record to `runs/.../telemetry/` with:

     - `run_id`
     - `tick_index`
     - `full_cognitive_hash`
     - `current_goal` (engine truth)
     - `agent_claimed_reason` (if enabled)
     - `panic_state`
     - `candidate_action`
     - `panic_adjusted_action` (+ `panic_reason`)
     - `final_action`
     - `ethics_veto_applied` (+ `veto_reason`)
     - short summaries of belief uncertainty, world model expectation, social inference
     - planning_depth
     - social_model.enabled

2. Live Run Context Panel

   - Show at runtime:

     - `run_id`
     - short_cognitive_hash (shortened hash)
     - tick / planned_run_length
     - current_goal
     - panic_state
     - planning_depth
     - social_model.enabled
     - panic_override_last_tick (+ panic_reason)
     - ethics_veto_last_tick (+ veto_reason)
     - agent_claimed_reason (if introspection.publish_goal_reason is true)

At this stage the panel provides an auditable narrative: the agent is in SURVIVAL, panic overruled the planner, EthicsFilter blocked `steal`, the planning depth is six ticks, and the agent claims "I'm going to work for money."

---

### 11.6 Panic and ethics for real

Goal: safety and survival must be enforced in-graph rather than remaining comments in YAML.

At this stage you replace the stub panic_controller and EthicsFilter in the execution graph with the real ones.

- `panic_controller`:

  - Reads `panic_thresholds` from Layer 1 (e.g. energy < 0.15)
  - Can override `candidate_action` with an emergency survival action (`call_ambulance`, `go_to_bed_now`, etc)
  - Emits `panic_override_applied` and `panic_reason`
  - Logged to telemetry and surfaced in the UI

- `EthicsFilter`:

  - Reads `forbid_actions` and `penalize_actions` from Layer 1 compliance
  - Blocks forbidden actions outright, substitutes something allowed, and emits `ethics_veto_applied` + `veto_reason`
  - Logged to telemetry and surfaced in UI

Important: EthicsFilter is final. Panic can escalate urgency, but panic cannot legalise a forbidden act. If panic tries "steal" as an emergency move, EthicsFilter still vetoes it. Ethics wins.

By the end of this step:

- panic is an explicit, logged controller in the loop
- ethics is an explicit, logged controller in the loop
- and we have a clean override chain:
  hierarchical_policy → panic_controller → EthicsFilter → final_action

At this point we can brief governance stakeholders using the recorded override trace rather than informal assurances.

---
