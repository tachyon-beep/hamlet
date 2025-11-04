# Townlet v2.5: Brain as Code

Document version: 1.1
Date: 3 November 2025
Status: Approved for Implementation
Owner: Principal Technical Advisor (AI)

---

## 1. Executive Summary

Brain as Code (BAC) replaces Townlet’s monolithic recurrent policy with a Software Defined Agent built from declarative configuration. The agent’s cognition is assembled at runtime from modular components whose wiring is described in YAML, allowing us to inspect, diff, govern, and reproduce minds with the same rigour we apply to world configuration.

### 1.1 Modular brains

Earlier releases fused perception, memory, planning, and action into a single network. BAC decomposes the mind into explicit modules—perception, world modelling, social modelling, panic control, hierarchical planning, policy heads—connected through a configured execution graph. Each module can be parameterised (e.g., GRU hidden size, planning horizon) and re-used across variants. This enables targeted changes such as inserting a panic gate or swapping a planner without rebuilding the entire policy.

### 1.2 Provenance-aware runs

Every run records which Universe-as-Code pack, brain configuration, safety rules, and curriculum step were in effect. Snapshots and cognitive hashes serve as the audit boundary: if behaviour occurs, we can prove which mind under which conditions produced it. Runs are therefore concrete artefacts (“BrainConfig X in WorldConfig Y at time T”), not vaguely described policies.

### 1.3 Identity-preserving checkpoints

Checkpoints now bundle live weights, cognitive configuration, and a hash. Reloading a checkpoint restores the full mind design, not just performance. This supports forensic analysis—tracking behaviours back to module combinations or safety settings—and eliminates unexplained regressions caused by opaque weight files.

### 1.4 Glass-box telemetry

Universe as Code exposes body metrics; Brain as Code surfaces cognition. Telemetry indicates which module drove action selection, whether panic overrides engaged, and whether EthicsFilter vetoed proposed actions. Auditors can interrogate decisions (“was it hunger, depression, fear, or economics?”) with evidence, supporting safety, governance, and pedagogy. We refer to this transparency as the glass-box model.

### 1.5 Why it matters

BAC aligns technical architecture with governance requirements:

- **Interpretability** — Behaviour is explainable at the module level.
- **Reproducibility** — Behaviour is a snapshot-plus-checkpoint pair that can be rerun.
- **Accountability** — Misbehaviour maps to misconfigured subgraphs rather than “the AI decided”.
- **Pedagogy** — Students can compare minds experimentally in identical worlds.

In short, the mind is a first-class content asset rather than a by-product of training.

---

## 2. The three cognitive layers

Brain as Code says: an agent’s mind is not mystical. It's three YAML files.

Those three files line up with three audiences:

- People who care what the agent is allowed to do.
- People who care how the agent is built.
- People who care how the agent actually thinks, step by step.

Together, these three layers are the SDA brain. Change the files, you change the mind.

### 2.1 Layer 1: cognitive_topology.yaml

Audience: policy, instructors, simulation designers
Nickname: the character sheet

Layer 1 is the public-facing definition of the agent as a character. This is the part you would show in a classroom, or put in front of an ethics reviewer, or attach to a safety case.

It answers questions like:

- Does this agent have social reasoning enabled?
- How paranoid is it?
- How greedy is it about money?
- When does it panic and override normal behaviour?
- What actions is it forbidden to take, full stop?

In other words: this layer defines how the agent is meant to behave, not how it's implemented under the hood.

Example (trimmed for clarity):

```yaml
# === Layer 1: The "Character Sheet" ===
# High-level behaviour and personality. Safe to edit.

perception:
  enabled: true
  uncertainty_awareness: true  # Can it admit "I'm not sure"?

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
  curiosity: 0.8          # exploration drive
  neuroticism: 0.6        # risk aversion / anxiety

panic_thresholds:
  energy: 0.15        # panic if energy < 15%
  health: 0.25        # panic if health < 25%
  satiation: 0.10     # panic if satiation < 10%

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

How to read this:

- `perception.enabled: true` tells us the agent is allowed to build an internal model of the world. If this were false, you'd get something more blind and reflexive, which is valid for ablations or "zombie baseline" training.

- `world_model.rollout_depth: 6` says the agent is literally allowed to imagine up to 6 ticks into the future before acting. This is now a dial. You can ship a short-horizon impulsive teenager or a long-horizon planner without touching code.

- `social_model.enabled: true` means it reasons about other entities as having minds/goals. Turn it off and you're simulating someone who does not model other people, which is sometimes exactly what you want.

- `personality` is where we admit we're doing psychology sliders. We are not pretending it's "just reward shaping". We say outright: this agent is greedy (0.7), curious (0.8), not very agreeable (0.3), and somewhat anxious (0.6). That will affect how it trades off money vs comfort vs safety.

- `panic_thresholds` is survival mode. Below these bars, the normal planner can be overridden by emergency behaviour. Panic is not a magic mystery anymore, it's parameterised in a file you can read.

- `compliance.forbid_actions` is the hard veto list. These are actions the agent is simply not allowed to take, ever, regardless of how desperate it is. This is where safety policy literally binds behaviour.

- `introspection.publish_goal_reason: true` means the agent will tell you (the human observer) why it thinks it's doing what it's doing. This is how we get "Glass Box" style auditability in UI. It's not just acting, it's narrating intent.

This is Layer 1’s job: describe what the mind is supposed to be like, in human terms.

This is the layer a policy person signs off on. This is the layer you show to governance. This is where you answer "what kind of entity did you just put in my sim?" in plain language.

### 2.2 Layer 2: agent_architecture.yaml

Audience: engineers, grad students
Nickname: the blueprint

Layer 2 defines the implementation behind Layer 1 faculties. For example, if Layer 1 enables a world model, Layer 2 specifies its network architecture, optimiser, and training setup.

Layer 2 covers:

- Network types (GRU, MLP, CNN, etc).
- Hidden sizes and layer widths.
- What each module outputs, and the dimensionality of those outputs.
- Which optimiser and what learning rate we are allowed to use.
- What pretraining task we ran to get the module started.

This separation supports reproducibility and pedagogy. Two brains can share Layer 1 behaviour while diverging in Layer 2 architecture, enabling controlled experiments.

Example (trimmed):

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

Key points:

- `interfaces` provide a contract for tensor dimensions, ensuring modules remain swappable. If the world model exports a 256-dimensional belief, the policy expects 256 dimensions by specification, not convention.
- `optimizer` and `pretraining` live in configuration. Any claim that behaviour stemmed from hidden hyperparameter tweaks can be validated directly. This supports auditability.
- Layer 2 modules correspond to Layer 1 enablement flags. Disabling a module in Layer 1 leaves its blueprint available in Layer 2 for future configurations.

Layer 2 converts the mind from concept to engineering specification.

### 2.3 Layer 3: execution_graph.yaml

Audience: researchers, auditors, you at 2 am
Nickname: the glue / the think loop

Layer 3 records the cognition loop—execution order, data flow, veto points, and final outputs.

It defines:

- The order modules are called every tick.
- Which outputs get passed forward as inputs to others.
- Where emergency logic (panic) and ethics/compliance veto kicks in.
- What final signal goes back into the environment.

In effect, Layer 3 is the wiring diagram of thought.

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

  - name: "panic_adjustment"
    node: "@modules.panic_controller"
    inputs:
      - "@steps.candidate_action"
      - "@config.L1.panic_thresholds"
    outputs:
      - "panic_action"
      - "panic_reason"

  - name: "final_action"
    node: "@modules.EthicsFilter"
    inputs:
      - "@steps.panic_adjustment.panic_action"
      - "@config.L1.compliance.forbid_actions"
    outputs:
      - "action"
      - "veto_reason"

outputs:
  - "final_action": "@steps.final_action.action"
  - "new_recurrent_state": "@steps.new_recurrent_state"
```

Execution flow:

1. The agent takes in the current world state (`raw_observation`) and its own memory of what just happened (`prev_recurrent_state`).

2. The perception module digests that into two things:

   - a cleaned-up internal belief about "what's going on right now",
   - an updated recurrent state to carry forward.

3. The hierarchical policy then plans an action. But it doesn't just look at perception. It also consults:

   - the world model service (what do I think will happen next if I do X),
   - the social model service (what do I think other agents are about to do).

   The agent therefore simulates futures within the rollout horizon specified in Layer 1.

4. That gives us a candidate_action. This is "what I want to do".

5. `panic_controller` evaluates crisis thresholds from Layer 1 and may override the candidate action with an emergency response, logging the reason.

6. `EthicsFilter` enforces compliance rules from Layer 1. It is the final authority; panic cannot legitimise forbidden actions. Vetoes (and substitutions) are logged for telemetry and UI.

7. The execution graph outputs:

- the final chosen action,
- the updated recurrent state for next tick.

Layer 3 matters because:

- Panic overrides live here, with explicit guardrails enforcing survival priorities without bypassing ethics.
- Auditing hooks can log candidate actions prior to EthicsFilter review.
- Cause and effect become demonstrable; telemetry can trace an ambulance call to panic firing, survival overrides, and ethics approval.

This is the glass-box philosophy in practice—thought is a directed acyclic graph, not an anecdote.

## 3. Run bundles

So far we've talked about how we describe a brain.

This section is about how we package and track an actual run of that brain in an actual world, so that we can:

- reproduce it,
- audit it,
- and teach from it later.

In Townlet, a "run" is now a bundle with identity. It's not just "I pressed go on some script and a tensor popped out". It’s a traceable experiment.

A run bundle is defined by a directory under `configs/`, and it looks like this:

```text
configs/
  L99_Inventing_ASI/
    config.yaml
    universe_as_code.yaml
    cognitive_topology.yaml        # SDA Layer 1
    agent_architecture.yaml        # SDA Layer 2
    execution_graph.yaml           # SDA Layer 3
```

Each file in the bundle serves a specific role:

### 3.1 config.yaml

Defines the simulation runtime envelope:

- How long does this run last? (`run_length_ticks`)
- How fast do we tick? (`tick_rate_hz`)
- How many agents are active? (`max_population`)
- What's the random seed?
- Are we training or just evaluating?
- What's the curriculum schedule (for example: "start solo, introduce rival agent at tick 10k")?

### 3.2 universe_as_code.yaml

Defines the world configuration for the run, including:

- the meters/bars (energy, health, money, mood, etc),
- what kills you (terminal conditions),
- the affordances (Bed, Job, Hospital, Bar, etc),
- opening hours,
- costs,
- per-tick effects,
- map layout,
- time-of-day mechanics,
- social cues if applicable.

Panic logic resides in Brain as Code; the world defines the problem, not the coping strategy.

#### 3.3 cognitive_topology.yaml

Layer 1 (behavioural intent): specifies active faculties, planning depth, priorities, panic thresholds, and prohibitions. Suitable for reviewers and safety sign-off.

#### 3.4 agent_architecture.yaml

Layer 2 (architectural blueprint): details network types, hidden dimensions, optimisers, and pretraining datasets, ensuring the brain can be rebuilt and audited.

#### 3.5 execution_graph.yaml

Layer 3 (execution flow): specifies the tick-by-tick wiring of cognition, including veto points. It answers governance questions such as “under what conditions may the agent abandon long-term planning for survival?”

### 3.6 Launching a run

Launching a bundle such as `L99_Inventing_ASI` snapshots configuration before simulation begins. The launcher creates a run folder:

```text
runs/
  L99_Inventing_ASI__2025-11-03-12-14-22/
    config_snapshot/
      config.yaml
      universe_as_code.yaml
      cognitive_topology.yaml
      agent_architecture.yaml
      execution_graph.yaml
    checkpoints/
    telemetry/
    logs/
```

Key properties:

1. `config_snapshot/` stores a copy of the bundle, sealing historical provenance even if `configs/...` is edited later.
2. `checkpoints/` accumulates weights and metadata (including cognitive hashes) over time.
3. `telemetry/` records module activity, vetoes, and panic events for audit and teaching.
4. `logs/` captures console/state output for debugging and replay.

Run bundles deliver reproducibility, accountability, and curriculum science: each folder can be rehydrated elsewhere, inspected for permitted behaviour, and diffed against sibling runs. Every agent becomes an artefact with chain of custody.

---

## 4. Checkpoints

A checkpoint is not just "model weights at step 500". A checkpoint is a frozen moment of mind and context.

Every checkpoint in a run must include all the ingredients required to:

- resume training honestly,
- audit the agent honestly,
- and prove what cognitive configuration produced a given behaviour.

That means each checkpoint directory contains:

- The live neural state

  - module weights (all nn.Modules in the SDA brain: perception, world model, social model, policy, EthicsFilter, etc)
  - optimiser states for each trainable module
  - any recurrent / GRU / memory state you need to carry forward

- The sim random number generators at that instant

  - RNG state(s)

- The rules the brain believed governed the world and itself at that instant

  - a full config snapshot (the same 5 YAMLs you launched with)

- A cryptographic identity for that specific mind

  - a cognitive hash

Example layout:

```text
runs/
  L99_Inventing_ASI__2025-11-03-12-14-22/
    checkpoints/
      step_000500/
        weights.pt
        optimizers.pt
        rng_state.json
        config_snapshot/
          config.yaml
          universe_as_code.yaml
          cognitive_topology.yaml
          agent_architecture.yaml
          execution_graph.yaml
        cognitive_hash.txt
```

The intent is governance-grade provenance. At `step_000500` we can answer, with evidence:

- what world the agent thought it lived in,
- which modules existed in its head,
- which rules constrained its behaviour,
- and which exact tensors represented its "brain state".

This separates anecdotal observations from provable mind identities.

### 4.1 Cognitive hash

`cognitive_hash.txt` stores the agent’s identity as a collision-resistant digest (for example, SHA-256) computed over:

1. The five config_snapshot YAML files concatenated in a stable order:

   - config.yaml
   - universe_as_code.yaml
   - cognitive_topology.yaml (Layer 1)
   - agent_architecture.yaml (Layer 2)
   - execution_graph.yaml (Layer 3)

2. The resolved execution graph after compilation, capturing the ordered steps and wiring once all `@modules.*` and `@graph.*` references are bound.

3. The instantiated architectures.
   Specifically: layer types, hidden dims, interface dimensions, etc, exactly as built by the factory for this run. That means if someone silently swaps GRU → LSTM in `agent_architecture.yaml`, the hash changes. Good.

Together, these inputs define “this exact mind in this exact configuration executing this loop.” Any mutation changes the cognitive hash; hotpatching without identity change is unsupported.

Implications:

- Telemetry labels behaviour with `full_cognitive_hash` for attribution.
- Resuming a run preserves identity if the hash remains unchanged.
- A/B experiments generate distinct hashes when cognition differs.

The hash functions as a brain fingerprint.

---

## 5. Resume semantics

Resuming is part of the audit trail. When resuming from `step_000500`, treat the checkpoint as authoritative; do not consult mutable configs in `configs/.../`.

Resume rules:

1. Restore `weights.pt`, `optimizers.pt`, and `rng_state.json` to recover the same brain, optimiser momentum, and randomness.
2. Restore the checkpoint’s `config_snapshot/`; it defines the world, topology, and execution graph at that moment.
3. Ignore live configuration directories. They may have drifted since the original launch.

Resumes write into a new run directory, for example:

```text
runs/
  L99_Inventing_ASI__2025-11-03-12-14-22_resume_2025-11-03-13-40-09/
    config_snapshot/        # copied from checkpoint, not from configs/
    checkpoints/
    telemetry/
    logs/
```

Consequences:

- **Lineage** — If the cognitive hash remains unchanged, the resumed run shares identity with the original run.
- **Branching** — Editing the snapshot before resume (e.g., lowering greed, adjusting panic thresholds) produces a new hash and run_id; this is an intentional fork.

Resumes therefore support long trainings, controlled experiments, and verifiable lineage. Any cognitive change creates a new mind by definition.

---

## 6. Runtime engine components

Under the hood, the classic "big RL agent class" is gone. We replace it with: factories, a graph agent, and an execution engine.

This gives us two wins:

1. We can build any brain that matches the YAML spec without changing code.
2. We can prove that the thing we built is the thing we ran.

### 6.1 agent/factory.py

This is the constructor of the mind.

Inputs:

- the run's frozen `config_snapshot/`

  - cognitive_topology.yaml (Layer 1: behaviour contract)
  - agent_architecture.yaml (Layer 2: module blueprints)
  - execution_graph.yaml (Layer 3: wiring order)
  - universe_as_code.yaml (for observation/action space alignment)
  - config.yaml (runtime envelope)

Responsibilities:

- Build each nn.Module exactly as described in `agent_architecture.yaml`
  (perception_encoder, world_model, social_model, hierarchical_policy, EthicsFilter, etc).
- Verify that interface dimensions match what `interfaces` promised.
  No silent broadcasting, no mystery reshapes.
- Package all of that into a `GraphAgent` instance.

So `factory.py` is basically: "From these YAMLs, instantiate the brain the YAMLs describe."

This is where the cognitive hash gets finalised, because this is where abstract config turns into actual module graphs.

### 6.2 agent/graph_agent.py

`GraphAgent` is the living brain object that replaces the old `RecurrentSpatialQNetwork`.

It's a generic nn.Module that:

- Owns all submodules in a registry, e.g. an `nn.ModuleDict` keyed by name (`perception_encoder`, `world_model`, etc).
- Holds whatever recurrent / memory state is persistent across ticks.
- Holds a `GraphExecutor` that knows how to actually run cognition for one tick.

It exposes a clean interface, roughly:

```python
think(raw_observation, prev_recurrent_state)
  -> { final_action, new_recurrent_state }
```

In other words: given what I see and what I remember, what do I do next, and what will I remember after I do it?

Key rule:
`GraphAgent` is always instantiated against the frozen `config_snapshot/` (from the run folder or checkpoint), never the mutable live configs. That enforces provenance and prevents "oops I hotpatched the policy on a live demo brain without updating the record".

### 6.3 agent/graph_executor.py

This is the cognition runner.

Inputs:

- `execution_graph.yaml` from Layer 3
- the module registry built by the factory

On init:

- It parses the execution graph YAML and "compiles" it into an ordered list of steps with explicitly resolved inputs and outputs.
- It resolves service bindings like `"@modules.world_model"` into actual module handles.
- It validates that all required data dependencies exist.

This compiled form (the ordered think loop with concrete wiring) is part of the cognitive hash. That means two graphs that look similar but schedule steps differently will hash differently, which is exactly what we want.

On run():

- It creates a scratchpad / data_cache.
- It executes each step in order:

- run perception
- unpack belief and new recurrent state
- run hierarchical policy with world/social context
- get candidate action
- run panic_controller
- run EthicsFilter
- emit final_action and new_recurrent_state

- It emits whatever the graph declared as outputs (final action and updated recurrent state, at minimum).

`graph_executor.py` therefore acts as the cognition microkernel, making the reasoning loop explicit, deterministic, and inspectable.

### 6.4 EthicsFilter module

This is the "seatbelt".

EthicsFilter is a module in the registry, just like perception or policy, except its job is to enforce the behavioural contract in Layer 1.

Inputs:

- candidate action from policy
- panic-adjusted action from panic_controller

Outputs:

- possibly substituted `final_action`
- `veto_reason` (why we overruled policy, if we did)

Because this happens inside the execution graph, we can log it every tick. That gives you a literal audit trail:

"Tick 842: agent tried to STEAL. EthicsFilter vetoed. Reason: compliance.forbid_actions."

This is how we move from "trust us, it's safe" to "here is a line in telemetry where it tried to steal and got blocked".

---

## 7. Telemetry and UI surfacing

Transparency is a runtime requirement. We expose two levels of visibility: a live panel for observers and per-tick trace logs for replay, teaching, and forensics.

### 7.1 Run Context Panel (live UI)

At runtime, the UI shows a compact panel for the observed agent containing:

- `run_id`
  For example: `L99_Inventing_ASI__2025-11-03-12-14-22`

- `short_cognitive_hash` (prettified form of the full_cognitive_hash suitable for UX, e.g. first 8 chars)
- `run_id` — e.g., `L99_Inventing_ASI__2025-11-03-12-14-22`
- `short_cognitive_hash` — abbreviated identity for UI reference
- `tick` — current tick versus planned length
- `current_goal` — meta-controller selection (SURVIVAL, THRIVING, SOCIAL, etc.)
- `panic_state` — emergency-mode indicator derived from Layer 1 thresholds
- `ethics_veto_last_tick` — prior tick veto status and reason
- `planning_depth` — allowable rollout horizon
- `social_model.enabled` — whether social reasoning is active
- `panic_override_last_tick` — prior tick panic override and reason
- `agent_claimed_reason` — self-reported rationale when introspection is enabled

Together these fields provide immediate glass-box context.

### 7.2 Per-tick trace logging (telemetry/)

Telemetry records (`runs/.../telemetry/`) capture each tick (or batched intervals) with at least:

- `run_id`
- `tick_index`
- `full_cognitive_hash`
- `current_goal`
- `agent_claimed_reason` (if enabled)
- `panic_state`
- `planning_depth`
- `social_model.enabled`
- `candidate_action`
- `panic_adjusted_action` (+ `panic_override_applied`, `panic_reason`)
- `final_action`
- `ethics_veto_applied` (+ `veto_reason`)
- `belief_uncertainty_summary`
- `world_model_expectation_summary`
- `social_model_inference_summary`

Distinctions:

- `current_goal` reflects engine truth from the meta-controller; `agent_claimed_reason` is a self-report.
- `panic_adjusted_action` is pre-ethics; `final_action` is post-ethics.

These logs answer retrospective questions (e.g., starvation events, panic responses, veto activity, social inference errors) and supply authentic teaching material. Words of estimative probability that the telemetry contract matches the current implementation: high (~90 percent).

---

## 8. Declarative goals and termination conditions

The hierarchical policy selects discrete goals (e.g., SURVIVAL, THRIVING). Each goal requires an inspectable termination condition to signal completion. To maintain reproducibility, we express these conditions in YAML via a constrained DSL.

Example:

```yaml
goal_definitions:
  - id: "survive_energy"
    termination:
      all:
        - { bar: "energy", op: ">=", val: 0.8 }
        - { bar: "health", op: ">=", val: 0.7 }

  - id: "get_money"
    termination:
      any:
        - { bar: "money", op: ">=", val: 1.0 }    # money 1.0 = $100
        - { time_elapsed_ticks: ">=", val: 500 }
```

All comparisons use normalised [0.0, 1.0] values from `universe_as_code.yaml` (money defaults to $100 at 1.0).

Mechanics:

- Each goal provides an `id` and a `termination` tree composed of `all`/`any` nodes.
- Leaves compare either bar values (energy, money, health) or runtime metrics (e.g., `time_elapsed_ticks`).

At runtime a lightweight interpreter evaluates the tree; satisfying the `all` or `any` clause completes the goal.

Benefits:

- Goals are readable without touching code.
- Historical runs can be replayed with identical logic because termination is data, not Python.
- Behavioural incentives are diffable (“SURVIVAL now requires energy ≥ 0.8 instead of ≥ 0.6”).

This also supports curriculum: early stages can require energy ≥ 0.5 for SURVIVAL, later stages ≥ 0.8. The change becomes an auditable YAML diff.

---

## 9. Affordance semantics in universe_as_code.yaml

Agent actions are defined as affordances in `universe_as_code.yaml`. Entries specify costs, per-tick effects, exclusivity, interruptibility, distance limits, and any permitted special effects (e.g., teleport).

Key fields:

- `capacity` — concurrent users allowed.
- `exclusive` — whether the affordance is single-occupancy.
- `interruptible` — whether progress can be abandoned mid-way.
- `distance_limit` — interaction range (e.g., call ambulance from adjacent tiles).
- `effects_per_tick` — bar deltas applied each tick.
- `costs` / `costs_per_tick` — upfront or ongoing expenditures.
- `effect_type` — whitelisted special handlers such as `teleport`.

Engine guarantees:

- Contention is resolved deterministically.
- Effects apply atomically each tick.
- Occupancy is ephemeral, keeping world-model learning tractable.

Division of responsibility:

- Universe as Code specifies “Hospital heals health at cost X”.
- Brain as Code decides whether panic or strategy warrants visiting the hospital.

This separation keeps world design and brain design individually reviewable.

---

## 10. Success criteria

Success is evaluated across technical, pedagogical, and governance axes.

### Technical success

[ ] We can launch a run from `configs/<run_name>/` and automatically create `runs/<run_name>__<timestamp>/` with a frozen `config_snapshot/`.

[ ] `agent/graph_agent.py` can be fully built from that snapshot alone (through `agent/factory.py`) and can tick.

[ ] Each checkpoint includes:

- weights
- optimiser state
- RNG state
- a nested `config_snapshot/`
- `full_cognitive_hash.txt`

[ ] Resume uses only the checkpoint snapshot and reproduces the same `full_cognitive_hash` (unless deliberately changed).

[ ] Telemetry logs include `run_id` and `cognitive_hash` so you can tie behaviour to mind identity.

[ ] UI surfaces run context (panic, veto, etc) live.

### Pedagogical success

[ ] A beginner can answer "Why didn't it steal the food?" just by:

- looking at the UI panel (veto_reason),
- looking at `cognitive_topology.yaml` (forbid_actions).

[ ] An intermediate student can change `agent_architecture.yaml` (for example GRU → LSTM) and rerun, and then actually observe the memory/behaviour consequences without touching core code.

[ ] A researcher can edit `execution_graph.yaml` to bypass `world_model_service` and watch the agent become more impulsive. That experiment is now "change this line in YAML", not "pull apart 4k lines of torch spaghetti".

[ ] We can take any interesting emergent behaviour clip, grab its run folder, and show precisely:

- which mind (hash),
- which world rules,
- which ethics rules,
- which panic thresholds,
were in play at that moment.

### Governance success

[ ] We can prove to an auditor that, at tick T in run R, the agent had `forbid_actions: ["attack","steal"]` and that `EthicsFilter` did in fact run in the execution graph.

[ ] We can replay that agent exactly as it existed at tick T using only the checkpoint directory, with no "trust me" developer hand waving.

Meeting these criteria demonstrates that Brain as Code operates as an accountable system rather than a slogan.

---

## 11. Implementation notes (ordering)

Recommended staged approach:

1. Establish folder discipline

   - Create `configs/<run_name>/` with all 5 YAMLs:

     - config.yaml
     - universe_as_code.yaml
     - cognitive_topology.yaml
     - agent_architecture.yaml
     - execution_graph.yaml

   - On launch, auto-generate:

     - `runs/<run_name>__<timestamp>/config_snapshot/`
       which is a deep copy, not a symlink.

   This locks down provenance from day one.

2. Snapshot copy on launch

   - Symlinks are permissible during development but configurations are copied on launch.

3. Build the core runtime stack:

   - `agent/factory.py`
   - `agent/graph_agent.py`
   - `agent/graph_executor.py`

   At this stage stubs are acceptable; the goal is to validate the pipeline:

   - load snapshot,
   - assemble modules,
   - run one think() tick.

4. Checkpoint writer

   - Able to dump:

     - model state dicts,
     - optimiser states,
     - RNG state,
     - config_snapshot/,
     - cognitive_hash.txt
   - This is where you define what "the mind" means in practice.

5. Resume path

   - Load from checkpoint only.
   - Spin up a new `runs/..._resume_<timestamp>/`.
   - Preserve or recompute the same cognitive hash unless you intentionally mutate the snapshot.

6. Telemetry format

   - Start logging per-tick (or per-n-ticks) JSON or CSV under `runs/.../telemetry/`.
   - Each record must include run_id and cognitive_hash.

7. UI Run Context panel

   - Show:

     - run_id
     - short cognitive_hash
     - tick / planned_run_length
     - current_goal
     - panic_state
     - ethics veto info (bool + reason)
     - planning_depth
     - social_model.enabled

Presenting this information in UI enables observers to state with evidence: the action reflected intent, ethics filtering, and traceable identity.

---

## 12. Implementation order

Audience: engineering leads, curriculum/tooling, anyone who has to actually ship this without catching fire

This section is the boot sequence. If you follow this order, you get a working glass-box agent with provenance, panic, ethics, telemetry, and replay. If you jump around and half-implement bits, you’ll hate your life and the audit story won’t hold.

### 12.1 Snapshots and run folders first

Goal: lock down provenance on day one.

- Create the bundle layout under `configs/<run_name>/` with the 5 YAMLs:

  - `config.yaml`
  - `universe_as_code.yaml`
  - `cognitive_topology.yaml`
  - `agent_architecture.yaml`
  - `execution_graph.yaml`

- Write the launcher that, when you “start run”, immediately:

  - creates `runs/<run_name>__<timestamp>/`
  - copies all 5 YAMLs byte-for-byte into `runs/<run_name>__<timestamp>/config_snapshot/`
  - makes `checkpoints/`, `telemetry/`, `logs/` subdirs

Rules:

- This is a physical copy, not a symlink.
- After snapshot, the live run must never silently re-read from `configs/<run_name>/` at runtime. The snapshot is now truth.

Why this is first:

- Everything else (the agent, the EthicsFilter, telemetry, resume) leans on snapshot provenance.
- If you don’t freeze config at launch, you can’t make any governance claims later.

### 12.2 Minimal GraphAgent pipeline

Goal: replace the old monolithic RL class with GraphAgent that can tick once.

Deliverables:

- `agent/factory.py`

  - Reads the run’s `config_snapshot/`
  - Instantiates modules declared in Layer 2 (`agent_architecture.yaml`)
  - Verifies interface dimensions declared in `interfaces`
  - Injects Layer 1 knobs (panic_thresholds, forbid_actions, etc) into the appropriate runtime modules
  - Assembles an `nn.ModuleDict` containing the submodules

- `agent/graph_executor.py`

  - Reads Layer 3 (`execution_graph.yaml`)
  - Compiles it into an ordered list of steps with explicit wiring
  - Validates that every step’s inputs resolve to something that actually exists
  - Produces a callable `run_step(data_cache)` that executes the loop in order

- `agent/graph_agent.py`

  - Owns module registry + executor
  - Maintains recurrent state
  - Exposes `think(raw_observation, prev_recurrent_state) -> { final_action, new_recurrent_state }`
  - Calls executor to actually do perception → policy → panic_controller → EthicsFilter

For v0 of this pipeline you can stub:

- world_model service to just echo nonsense
- social_model service to return “disabled”
- panic_controller to just pass through
- EthicsFilter to just pass through

Why this is second:

- You need a callable brain before you can talk about panic escalation, veto, telemetry, etc.
- You also need this working before you can hash cognition.

### 12.3 Cognitive hash

Goal: give the brain a verifiable identity.

Implement `cognitive_hash.txt` generator. The hash (for example SHA-256 digest hex string) must deterministically encode:

1. The text of all 5 YAMLs in `config_snapshot/` concatenated in a stable order
   (config.yaml, universe_as_code.yaml, cognitive_topology.yaml, agent_architecture.yaml, execution_graph.yaml)

2. The compiled execution graph
   After `graph_executor` resolves all `@modules.*`, `@graph.*`, and service bindings and produces an ordered step list with concrete inputs and outputs.

3. The instantiated architectures
   For each module in `agent_architecture.yaml`, capture:

   - class/type (GRU, MLP, CNN)
   - layer sizes / hidden dims
   - optimiser type and LR
   - interface dimensions it exposes/consumes

That bundle becomes the canonical “mind fingerprint”.

Why we do it here:

- You want hashing before checkpoints, so you can stamp every checkpoint with the hash.
- Also, telemetry needs to be logging this hash from the very first tick of any run.

Note:

- If anything in (1), (2), or (3) changes, the hash changes.
  That is what lets you prove “same brain” or “forked brain” to an auditor.

### 12.4 Checkpoint writer and resume

Goal: honest pause/resume and experimental forking.

Checkpoint writer must output, at minimum, into `runs/<run_id>/checkpoints/step_<N>/`:

- `weights.pt`

  - All module weights from GraphAgent (including EthicsFilter etc)

- `optimizers.pt`

  - All optimiser states

- `rng_state.json`

  - Environment RNG state and agent RNG / torch RNG etc

- `config_snapshot/`

  - Deep copy of the snapshot as of this checkpoint (it may match launch snapshot exactly, but store it again so we can resume from here even if the world evolved under curriculum)

- `cognitive_hash.txt`

  - The full hash for this checkpoint

Resume rules:

- Resume never reaches back into `configs/<run_name>/`.
- Resume reloads only from the checkpoint subfolder.
- Resume writes into a new run folder, e.g.
  `runs/<run_name>__<launch_ts>_resume_<resume_ts>/`
  with a cloned snapshot from the checkpoint including cognitive_hash.txt.

Branching:

- If you mutate that cloned snapshot before resuming (for example you edit `cognitive_topology.yaml` to drop `greed` from 0.7 to 0.4), you are creating a sibling mind with a new cognitive_hash.
- That is allowed, but it’s recorded as a fork, not a continuation.

This gets us:

- long trainings that survive GPU pre-emption,
- forensics (“replay mind at tick 842, unchanged”),
- honest ablations (“same weights, same world, except panic disabled”).

### 12.5 Telemetry and UI

Goal: observable cognition.

Implement:

- the per-tick `telemetry/` writer described in section 7.2,
- the live Run Context Panel described in section 7.1.

Refer to the Minimum fields to surface/log each tick are described in Section 7.2.

By this step, you’ve got a glass-box demo.

### 12.6 Panic and ethics for real

Goal: safety and survival actually do something.

Replace the stub panic_controller and EthicsFilter in `graph_executor` with working modules that:

- panic_controller:

  - Reads Layer 1 `panic_thresholds`
  - Checks bars from observation / belief (“energy below 0.15? health below 0.25?” etc)
  - If panic is active, is allowed to override the policy’s `candidate_action` with an emergency action (e.g. “call_ambulance”, “go_to_bed_now”)
  - Logs that override and reason so the UI + telemetry can say “panic override applied: energy_critical”

  Important: panic is allowed to reprioritise survival, but it cannot authorise illegal actions. Panic fires before ethics. Ethics is final.

- EthicsFilter:

  - Reads Layer 1 `forbid_actions` and `penalize_actions`
  - Blocks forbidden actions, substitutes a safe fallback, and emits `veto_reason`
  - Optionally soft-penalises “allowed but discouraged” actions like `shove` by biasing toward an alternative or tagging the action for a negative shaping reward
  - Emits `ethics_veto_applied` and `veto_reason` into telemetry

By the end of this step:

- panic is visible and auditable,
- ethics is visible and auditable,
- and we have an accountable override chain:
  policy → panic_controller → EthicsFilter → final_action

And that’s basically the system.

---

## 13. Open questions and forward extensions

Audience: systems, safety, curriculum, future-you

These items are deferred by design—either high-risk or high-effort. They should be tracked explicitly rather than implemented ad hoc.

### 13.1 Panic, ethics, and conflict

Right now the chain is:

1. hierarchical_policy proposes `candidate_action`
2. panic_controller may escalate it for survival
3. EthicsFilter can still veto the result

The current rule is “ethics beats panic.” Allowing panic to bypass ethics (e.g., stealing as a last resort) would be a policy decision requiring governance approval and a cognitive hash change. Leaving the rule ambiguous creates governance risk.

### 13.2 Curriculum pressure vs identity drift

Curriculum may:

- spawn new instances of existing affordances (a second Bed, a rival agent),
- change wages, prices, or availability schedules over time,
- introduce social pressure (e.g. add an NPC that hoards food).

Curriculum must not silently rewrite physics. Mutating affordance definitions or base depletion/terminal rules constitutes a new world and must trigger a snapshot, run_id, and hash change. Spawning additional instances of existing affordances is acceptable under the same definitions. The launcher should enforce this distinction.

### 13.3 Social model and privacy

Layer 1 can turn `social_model.enabled` off completely (“sociopath mode”), which is useful for baselines or certain curriculum phases.

But we haven’t addressed:

- Should an agent be allowed to model other agents’ *goals* if policy says it’s not allowed to model other agents’ *intentions*?
- Should some roles in the sim (say, “civilian NPCs”) have reduced social observability for privacy/safety reasons?

Potential extension: per-entity visibility controls, for example:

- social_model can see family,
- social_model can’t reason about strangers beyond “occupied/unoccupied affordance”,
- social_model can’t infer exact plan of special protected agents.

That would likely live in Layer 1 under social_model, e.g.

```yaml
social_model:
  enabled: true
  privilege:
    infer_goals_for:
      - "family"
      - "co_workers"
    forbid_intent_prediction_for:
      - "civilians"
      - "children"
```

and be enforced in the execution graph by passing censored summaries from social_model_service.

Such controls matter for privacy and ethics in multi-agent environments.

### 13.4 Introspection and self-reporting honesty

Layer 1 has:

```yaml
introspection:
  visible_in_ui: "research"
  publish_goal_reason: true
```

Right now, “publish_goal_reason: true” means “tell the observer why you think you’re doing what you’re doing”.

Agents may misreport motives. UI and telemetry should distinguish between engine truth (`current_goal`) and self-report (`agent_claimed_reason`). Divergence is informative but must be labelled.

- `current_goal: SURVIVAL (engine truth)`
- `agent_claimed_reason: "I'm trying to get a better job"`

If those don’t match, that’s actually gold for teaching: we’ve caught cognitive dissonance. But we need both channels logged.

Action:

- Add `agent_claimed_reason` to telemetry.
- Add `current_goal` (engine truth) to telemetry.
- Add both to the live UI panel if `introspection.publish_goal_reason: true`.

### 13.5 Compliance richness

Layer 1 compliance currently offers two tools:

- hard veto (`forbid_actions`)
- soft discourage (`penalize_actions` with a penalty)

We may also want:

- situational bans, like “steal is forbidden unless the target is flagged as `abandoned_property`”
- contextual norms, like “calling ambulance while mood is high and health is high is considered abuse of system, apply social penalty”

This drifts toward rule-based reasoning. Options include:

Option A: expand `penalize_actions` to allow structured conditions:

```yaml
penalize_actions:
  - action: "call_ambulance"
    if:
      all:
        - { bar: "health", op: ">=", val: 0.7 }
        - { bar: "mood", op: ">=", val: 0.8 }
    penalty: -5.0
    note: "stop faking emergencies"
```

Option B: introduce a tiny Compliance DSL parallel to goal termination DSL. That DSL would be evaluated inside EthicsFilter before veto/penalty, similar to how panic_controller checks thresholds.

Whatever approach we choose must remain declarative and hashable; introducing arbitrary code undermines governance.

---

At this point we’ve got:

- A world defined in YAML (Universe as Code).
- A brain defined in YAML (Brain as Code, Layers 1–3).
- A run that snapshots both.
- A mind identity (`full_cognitive_hash`) that’s cryptographically bound to that snapshot + compiled wiring.
- A panic controller and an EthicsFilter that are explicit modules in the think loop.
- Telemetry and UI that expose what the mind tried to do and why it was (or wasn’t) allowed.

That means: the sim stops being vibes and starts being evidence.

And that’s the whole game.
