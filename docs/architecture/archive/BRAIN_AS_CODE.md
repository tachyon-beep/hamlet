# Townlet v2.5: Software Defined Agent (SDA) and Run Provenance Specification

Document Version: 1.1
Date: 3 November 2025
Status: APPROVED FOR IMPLEMENTATION
Owner: Principal Technical Advisor (AI)
---------------------------------------

1. Executive Summary

Hamlet v2.5 replaces the old hardcoded `RecurrentSpatialQNetwork` with a Software Defined Agent (SDA). The SDA is no longer "an RL model" in code, it's a graph of cognitive modules that is assembled at runtime from declarative configs.

In parallel, every training / sim "run" is now a first-class object with clear provenance, so that every behaviour we record can be tied back to:

- which world it lived in
- which brain it was running
- which safety constraints were in force
- which hyperparameters and curriculum settings applied

The goals:

- Interpretability (for teaching and debugging)
- Reproducibility (for research)
- Accountability / governance (for policy)

We achieve this through:

1. A three-layer cognitive config stack (L1, L2, L3) that defines the agent's mind.

2. A per-run directory structure that captures world + brain + runtime config.

3. A checkpoint format that embeds the full config snapshot + cognitive hash.

4. Telemetry and UI surfacing of that identity in real time.

5. Philosophy: "The Glass Box"

v1.0: The brain is a black box Python class. You ask "why did it do that?" and the answer is ¯\*(ツ)*/¯.

v2.5: The brain is a graph defined in YAML. You can literally open the run folder and read:

- which faculties are turned on (perception, world model, social model, hierarchical policy)
- how those faculties are internally built (GRU vs LSTM etc)
- how they talk to each other (execution graph steps)
- what panic rules and ethics vetoes exist

Students, auditors, reviewers, everyone can see the mind and the rules that govern it.

We call this "Software Defined Agent" (SDA).

3. The Three Cognitive Layers

The SDA brain is defined by three YAML files, each aimed at a different audience.

3.1 Layer 1: cognitive_topology.yaml
Audience: student / policy / non-expert
"Character sheet"

Controls:

- Which faculties are on or off (perception, world model, social model, hierarchical policy)
- Planning depth, social reasoning on/off, panic thresholds
- Personality knobs (greed, curiosity, etc)
- Compliance / ethics (forbid_actions etc)
- Introspection + UI hints

Example:

```yaml
# === Layer 1: The "Character Sheet" ===
# High-level behaviour and personality. Safe to edit.

perception:
  enabled: true
  uncertainty_awareness: true  # Can it report "I'm not sure"?

world_model:
  enabled: true
  rollout_depth: 6        # How many steps ahead it can "imagine"
  num_candidates: 4       # How many futures it evaluates per tick

social_model:
  enabled: true           # false = sociopath mode
  use_family_channel: true

hierarchical_policy:
  enabled: true
  meta_controller_period: 50  # Re-evaluate strategic goal every 50 ticks
  world_model_proposals:
    strategy: "shortest_path_to_goal"
    num_candidates: 3

personality:
  greed: 0.7              # money priority
  agreeableness: 0.3      # social harmony priority
  curiosity: 0.8          # intrinsic exploration drive
  neuroticism: 0.6        # risk aversion

panic_thresholds:
  energy: 15
  health: 25
  satiation: 10

compliance:
  forbid_actions:
    - "attack"
    - "steal"

  penalize_actions:
    - { action: "shove", penalty: -5.0 }

introspection:
  visible_in_ui: "research"  # beginner | intermediate | research
  publish_goal_reason: true  # Show "why I chose this goal" in UI
```

Notes:

- This file is what you show in class.
- This is also what an ethics reviewer will look at to confirm the "rules of engagement".

3.2 Layer 2: agent_architecture.yaml
Audience: grad student / engineer
"The Blueprint"

Controls:

- What each module actually is architecturally (CNN/MLP/GRU, hidden size, optimiser)
- What each module predicts / outputs (the heads)
- The shared interfaces, so they wire together cleanly

It also encodes pretraining tasks (curriculum) for each module.

Example:

```yaml
# === Layer 2: The "Engineering Blueprint" ===
# Internal structure of each module.

interfaces:
  belief_distribution_dim: 128      # Output of perception
  imagined_future_dim: 256          # Summary vector from world model
  social_prediction_dim: 128        # Summary vector from social model
  goal_vector_dim: 16               # Encoded goal representation
  action_space_dim: 6               # up,down,left,right,interact,wait

modules:

  perception_encoder:  # Module A
    spatial_frontend:
      type: "CNN"
      channels: [16, 32, 32]
      kernel_sizes: [3, 3, 3]

    vector_frontend:
      type: "MLP"
      layers: [64]
      input_features: "auto"  # infer from obs spec

    core:
      type: "GRU"
      hidden_dim: 512
      num_layers: 2

    heads:
      belief_dim: 128  # must match interfaces.belief_distribution_dim

    optimizer: { type: "Adam", lr: 0.0001 }

    pretraining:
      objective: "reconstruction+next_step"
      dataset: "observation_rollout_buffer"

  world_model:  # Module B
    core_network:
      type: "MLP"
      layers: [256, 256]
      activation: "ReLU"

    heads:
      next_state_belief: { dim: 128 }   # == belief_distribution_dim
      next_reward:       { dim: 1 }
      next_done:         { dim: 1 }
      next_value:        { dim: 1 }

    optimizer: { type: "Adam", lr: 0.00005 }

    pretraining:
      objective: "dynamics+value"
      dataset: "sdw_ground_truth_logs"

  social_model:  # Module C
    core_network:
      type: "GRU"
      hidden_dim: 128

    inputs:
      use_public_cues: true
      use_family_channel: true
      history_window: 12  # ticks of observation history

    heads:
      goal_distribution:     { dim: 16 }  # == goal_vector_dim
      next_action_dist:      { dim: 6 }   # == action_space_dim

    optimizer: { type: "Adam", lr: 0.0001 }

    pretraining:
      objective: "ctde_intent_prediction"
      dataset: "sdw_ground_truth_logs"

  hierarchical_policy:  # Module D
    meta_controller:
      network: { type: "MLP", layers: [256, 128], activation: "ReLU" }
      heads:
        goal_output: { dim: 16 }  # == goal_vector_dim

    controller:
      network: { type: "MLP", layers: [256, 128], activation: "ReLU" }
      heads:
        action_output: { dim: 6 }  # == action_space_dim

    optimizer: { type: "Adam", lr: 0.0003 }

    pretraining:
      objective: "behavioral_cloning"
      dataset: "v1_agent_trajectories"
```

Notes:

- This is where we decide GRU vs LSTM vs Transformer.
- This is also where we declare optimiser LR, so training is traceable.

3.3 Layer 3: execution_graph.yaml
Audience: researcher / you
"The Glue"

Controls:

- The actual cognitive dataflow per tick
- The order modules run
- What inputs are passed to them
- Where compliance/ethics gates happen
- What gets returned to the environment

Example:

```yaml
# === Layer 3: Execution Graph ===
# The ordered "think loop" DAG.

inputs:
  - "raw_observation"
  - "prev_recurrent_state"

services:
  - "world_model_service": "@modules.world_model"
  - "social_model_service": "@modules.social_model"

steps:

  - name: "perception_packet"
    node: "@modules.perception_encoder"
    inputs:
      - "@graph.raw_observation"
      - "@graph.prev_recurrent_state"

  - name: "belief_distribution"
    node: "@utils.unpack"
    input: "@steps.perception_packet"
    key: "belief"

  - name: "new_recurrent_state"
    node: "@utils.unpack"
    input: "@steps.perception_packet"
    key: "state"

  - name: "policy_packet"
    node: "@modules.hierarchical_policy"
    inputs:
      - "@steps.belief_distribution"
      - "@services.world_model_service"
      - "@services.social_model_service"

  - name: "candidate_action"
    node: "@utils.unpack"
    input: "@steps.policy_packet"
    key: "action"

  - name: "final_action"
    node: "@modules.ethics_filter"
    inputs:
      - "@steps.candidate_action"
      - "@config.L1.compliance.forbid_actions"
    outputs:
      - "action"
      - "veto_reason"

outputs:
  - "final_action": "@steps.final_action.action"
  - "new_recurrent_state": "@steps.new_recurrent_state"
```

Notes:

- `ethics_filter` is where L1's `forbid_actions` is actually enforced.
- You can insert new nodes here without touching model code: e.g. add `Module E: Ethics`, or bypass the world model entirely for ablation.
- This file is the "wiring harness" of cognition.

4. Run Bundles

A run is now defined by a folder in `configs/`.

For example:

```
configs/
  L99_Inventing_ASI/
    config.yaml
    software_defined_world.yaml
    cognitive_topology.yaml        # SDA L1
    agent_architecture.yaml        # SDA L2
    execution_graph.yaml           # SDA L3
```

4.1 Meaning of each file in a run bundle

- config.yaml
  Simulation/runtime knobs. Examples:

  - run_length_ticks
  - tick_rate_hz
  - max_population
  - random_seed
  - logging_frequency
  - curriculum flags (e.g. "start with 1 agent, add 2nd rival at tick 10k")
  - training mode vs eval mode

- software_defined_world.yaml
  The Software Defined World (SDW): bars/meters, affordances, map layout, cues, time-of-day availability, etc.
  All affordances are declarative:

  - capacity
  - exclusive / interruptible
  - per-tick bar deltas
  - cost (money, hygiene, etc)
  - optional whitelisted effect_type like teleport
  - panic semantics not here, that's cognitive

- cognitive_topology.yaml
  The L1 character sheet for this run's agent.

- agent_architecture.yaml
  The L2 module internals.

- execution_graph.yaml
  The L3 dataflow.

4.2 Launching a run

When you do `run L99_Inventing_ASI`, the system:

1. Creates a new output directory under `runs/` with timestamp.
2. Copies the entire contents of `configs/L99_Inventing_ASI/` into a subfolder `config_snapshot/` in that run directory.

Result:

```
runs/
  L99_Inventing_ASI__2025-11-03-12-14-22/
    config_snapshot/
      config.yaml
      software_defined_world.yaml
      cognitive_topology.yaml
      agent_architecture.yaml
      execution_graph.yaml
    checkpoints/
    telemetry/
    logs/
```

This snapshot is now the source of truth for this run.
After launch, changing anything in configs/L99_Inventing_ASI/ does not mutate this run.
This is critical for reproducible science and audit.

5. Checkpoints

Every checkpoint in a run MUST include:

- module weights (all nn.Modules in SDA, plus optimiser/GRU states etc)
- optimiser states
- RNG state(s)
- a copy of the config snapshot at that moment
- a cognitive hash

Example layout:

```
runs/
  L99_Inventing_ASI__2025-11-03-12-14-22/
    checkpoints/
      step_000500/
        weights.pt
        optimizers.pt
        rng_state.json
        config_snapshot/
          config.yaml
          software_defined_world.yaml
          cognitive_topology.yaml
          agent_architecture.yaml
          execution_graph.yaml
        cognitive_hash.txt
```

5.1 Cognitive Hash

`cognitive_hash.txt` is a deterministic hash (e.g. SHA256) computed over:

1. The five config_snapshot YAML files concatenated.
2. The resolved execution graph after compilation (the actual ordered steps and the resolved node→input mapping).
3. The instantiated architectures (layer types and dims for each module, as built by the factory).

That hash is the ID of "this exact mind".

Why we care:

- Telemetry can say "this behaviour happened under hash X".
- You can resume training later and prove you resumed the same mind, not a slightly different brain.
- You can evaluate two different policies in the exact same world and have evidence they really were different minds, not just unlucky RNG.

6. Resume Semantics

Resuming from a checkpoint MUST use the checkpoint snapshot, not the live configs.

When you resume from `step_000500`:

1. You restore weights.pt, optimizers.pt, rng_state.json.
2. You restore the config_snapshot/ from inside that checkpoint.
3. You DO NOT reread anything from `configs/L99_Inventing_ASI/`.

You may create a new output directory for the resumed run, e.g.

`runs/L99_Inventing_ASI__2025-11-03-12-14-22_resume_2025-11-03-13-40-09/`

but that resumed run is still considered to be running the same "mind identity" (same cognitive hash) until/unless you deliberately edit the snapshot and thereby change the hash.

This gives you:

- Continuity for long trainings
- Branching for ablations ("same brain, tweak greed only, now it's a fork")
- Defensible lineage for reporting / teaching

7. Runtime Engine Components

We replace the old monolithic agent code with general factories and an execution engine.

7.1 agent/factory.py

- Input: the run's config_snapshot (cognitive_topology.yaml, agent_architecture.yaml, execution_graph.yaml, plus software_defined_world.yaml and config.yaml)
- Responsibilities:

  - Build each nn.Module described in agent_architecture.yaml.
  - Validate that interface dims line up.
  - Return a `GraphAgent`.

7.2 agent/graph_agent.py

- A generic nn.Module replacing `RecurrentSpatialQNetwork`.
- Holds:

  - Module registry (perception_encoder, world_model, social_model, hierarchical_policy, ethics_filter, etc) in an nn.ModuleDict.
  - A `GraphExecutor` which knows how to run the execution graph.
  - The current recurrent state(s).
- Exposes:

  - `think(raw_observation, prev_recurrent_state) -> { final_action, new_recurrent_state }`

Critically:
`GraphAgent` is instantiated against the frozen snapshot, not against mutable on-disk configs. This enforces provenance.

7.3 agent/graph_executor.py

- Input: execution_graph.yaml and the module registry
- On init:

  - Parses and "compiles" the execution_graph into an ordered step list with resolved inputs/outputs and service bindings.
  - This compiled form is part of the cognitive hash.
- On run():

  - Maintains a data_cache for intermediate tensors.
  - Executes each step in order (perception, unpack, policy, ethics filter, etc).
  - Returns the declared outputs.

7.4 ethics_filter module

- A small module that enforces the `forbid_actions` and `penalize_actions` rules declared in L1 cognitive_topology.yaml.
- Outputs:

  - possibly vetoed action
  - veto_reason (string or code)
- This makes "the agent wanted to do X, but we blocked it" first-class telemetry.

8. Telemetry and UI Surfacing

We want two layers of visibility.

8.1 Run Context Panel (live UI)

At runtime the UI must surface:

- run_id (e.g. `L99_Inventing_ASI__2025-11-03-12-14-22`)
- cognitive_hash (short form, like first 8 chars)
- tick: current_tick / planned_run_length_ticks
- current_goal (from hierarchical_policy.meta_controller)
- panic_state (true/false derived from panic_thresholds in L1)
- ethics_veto_last_tick (bool + veto_reason if any)
- planning_depth (world_model.rollout_depth)
- social_model.enabled (true/false)

This is the "at a glance" view for instructors and demos:
"See here? The agent is currently in SURVIVAL mode because energy < 15, it's allowed to plan 6 steps ahead, and last tick its chosen action got vetoed because it tried to 'steal'."

8.2 Per-tick Trace Logging (telemetry/)

We also persist a structured event log (one row per tick or per n ticks) into `runs/.../telemetry/`.

Each record contains at minimum:

- run_id
- tick_index
- cognitive_hash
- chosen_goal (Goal struct ID, e.g. THRIVING, SURVIVAL, SOCIAL)
- candidate_action (pre-ethics)
- final_action (post-ethics)
- veto_reason (if any)
- summary of belief_uncertainty (e.g. "energy bar confidence 0.42")
- summary of world_model expectation (e.g. predicted_reward_next_step)
- summary of social_model inference (e.g. "Agent_2 intent: use_fridge")

This lets us:

- Reconstruct interesting decisions later
- Build curriculum labs: "Explain why the agent starved while standing at the fridge"
- Publish introspection traces without needing to rerun live sim

9. Declarative Goals / Termination Conditions

Module D1 (meta_controller) outputs a Goal like SURVIVAL or THRIVING. That goal needs a termination condition, but YAML can't store Python lambdas.

We solve it via a small DSL in cognitive_topology.yaml:

```yaml
goal_definitions:
  - id: "survive_energy"
    termination:
      all:
        - { bar: "energy", op: ">=", val: 80 }
        - { bar: "health", op: ">=", val: 70 }

  - id: "get_money"
    termination:
      any:
        - { bar: "money", op: ">=", val: 100 }
        - { time_elapsed_ticks: ">=", val: 500 }
```

The engine will ship simple evaluators for `all` / `any` sets of {metric, op, value} rules.

This gives us:

- Interpretable stopping logic
- No inline Python
- Stable replay semantics

10. Affordance Semantics in software_defined_world.yaml

All affordances and world interactions live in software_defined_world.yaml. Rules:

- All effects are expressed as bar deltas per tick, or via whitelisted special effect types (`teleport`, etc).
- Affordance fields include:

  - `capacity`
  - `exclusive`
  - `interruptible`
  - `distance_limit`
  - `effects_per_tick` (list of bar deltas)
  - `costs` (bars to deduct up front)
  - optional `effect_type` blocks for engine-handled specials like teleport
- Reservation / contention / interrupt semantics are handled deterministically by the engine (reservation token, tie-break on stable rule, atomic application of bar deltas per tick).

We do NOT persist ownership state in the object itself. Occupancy is ephemeral per tick. This keeps the world simple and learnable for the world_model.

11. Success Criteria

Technical success:
[ ] We can launch a run from configs/ and get a timestamped runs/ folder with a frozen config_snapshot.
[ ] agent/graph_agent.py can build itself entirely from that snapshot.
[ ] checkpoints embed weights + optimiser + rng + config_snapshot + cognitive_hash.
[ ] resume uses only checkpoint snapshot, not mutable configs/.
[ ] telemetry logs include run_id and cognitive_hash.
[ ] UI shows run context, panic state, veto reasons, planning depth, etc.

Pedagogical success:
[ ] A student (L1 level) can answer "Why did it refuse to steal?" just by reading the UI and cognitive_topology.yaml.
[ ] An advanced student (L2 level) can modify `agent_architecture.yaml` to swap GRU for LSTM and see if memory changes.
[ ] A researcher (L3 level) can rewire the execution_graph to remove the world_model and watch the agent become more impulsive.
[ ] We can take any cool emergent behaviour clip and prove what brain and rules made it happen, using cognitive_hash and the embedded snapshot.

Governance success:
[ ] We can prove, to an auditor, that a given agent in a demo had `forbid_actions: ["attack","steal"]` at the time of behaviour, and that the ethics filter was actually in the loop.
[ ] We can replay that exact cognitive state later from the checkpoint alone, without hand waving.

12. Implementation Notes (ordering)

Recommended build order once you start L1 → L4 runs and get a dev breather:

1. Folder discipline:

   - configs/<run_name>/ with all 5 YAMLs
   - runs/<run_name>__<timestamp>/config_snapshot/ created on launch

2. Snapshot copy on launch (no symlink tricks at runtime; symlinks okay for developer comfort pre-launch)

3. agent/factory.py + agent/graph_agent.py + agent/graph_executor.py minimal viable path

   - Can be stubby first, doesn't need full social model yet
   - Just prove we can load from snapshot and tick

4. Checkpoint writer that saves:

   - model state dicts
   - optimiser
   - rng
   - config_snapshot/
   - cognitive_hash.txt

5. Resume path that trusts checkpoint first, configs never

6. Telemetry log format with run_id + cognitive_hash

7. UI Run Context panel surfacing:

   - run_id
   - cognitive_hash (short)
   - tick / run_length
   - current_goal
   - panic state
   - veto info
