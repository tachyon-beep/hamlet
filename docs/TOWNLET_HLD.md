# Townlet v2.5 High Level Design Document

## 1. Executive Summary

---

Document version: 2.5 (consolidated BAC/UAC architecture)
Date: 3 November 2025
Status: Approved for Implementation
Owner: Principal Technical Advisor (AI)

Townlet v2.5 defines the agent and the world as audited configuration, not as "whatever the Python happened to be at the time".

The old Townlet / Hamlet agent was one opaque recurrent Q-network (`RecurrentSpatialQNetwork`) that turned partial observations into actions. It sort of worked, sometimes brilliantly, but it was a black box. If it sprinted to hospital and then fell asleep in the shower, the only honest answer to "why?" was "the weights felt like it".

Townlet v2.5 replaces that with a Software Defined Agent (SDA) running in a Software Defined World. We treat both the world and the mind as first-class content. We call this:

* Universe as Code (UAC): the world, declared in data.
* Brain as Code (BAC): the mind, declared in data.

If you need the full schematics, see §2 for BAC and §8 for UAC; those sections walk through the YAML layers and runtime interpretation.

Together, those two things give us four hard properties we did not have before:

1. The mind is explicit
   The agent's cognition is described in three YAML files: what faculties it has, how they're implemented, and how they think step by step. Panic response, ethics veto, planning depth, goal-selection bias – it's all on paper.

2. The world is explicit
   The town (energy, health, money, affordances, ambulance cost, bed quality, public cues for other agents, wage schedules) is also data. Beds, jobs, hospitals, phones are declared in configuration as affordances with per-tick bar effects, eliminating hidden "secret physics" deep in the environment loop.

3. Every run is provenance-bound
   When you run Townlet, the platform snapshots both the world and brain configurations, hashes them, and stamps that identity onto every tick of telemetry. If the agent behaves unexpectedly, we can identify exactly which mind, under which world rules, produced the behaviour. There is no "the AI just did that".

4. We can teach and audit, not just watch
   We log not only what the body did ("health: 0.22"), but what the mind attempted, what the panic controller overrode, and what the ethics layer vetoed. We can answer "why" with evidence rather than conjecture.

In effect, Townlet shifts from "a neat RL simulation with emergent drama" to "an accountable cognitive system that can be diffed, replayed, and defended".

We get this by doing three things.

### 1.1 The Brain Is Now Assembled, Not Baked In

Earlier releases relied on a monolithic recurrent Q-network that attempted to handle perception, planning, social reasoning, panic, ethics, and action selection in a single block. Townlet v2.5 decomposes the mind into explicit cognitive modules:

* Perception / belief state builder  
  Transforms partial, noisy observations into an internal belief state.

* World Model  
  Predicts state transitions for candidate actions, allowing the agent to learn the town’s dynamics, including non-stationary changes such as price shifts.

* Social Model  
  Estimates the likely behaviour and goals of nearby agents using the public cues that Universe as Code exposes. It receives no hidden state at runtime.

* Hierarchical Policy  
  Selects a strategic goal (for example SURVIVAL versus THRIVING) and chooses the concrete action that advances that goal each tick.

* Panic Controller  
  Overrides normal planning when survival thresholds such as energy or health fall below configured limits.

* EthicsFilter  
  Applies the final compliance gate. It forbids actions that violate policy (for example, "steal" or "attack") or downgrades risky options. Panic cannot bypass ethics.

Modules and their interactions are declared in configuration and materialised at runtime. We can explicitly disable the Social Model for ablation without touching shared policy code, adjust the planning horizon from two to six ticks via configuration, or introduce a new panic rule without retraining perception.

Townlet v2.5 is therefore an engineered assembly rather than a single opaque network.

### 1.2 The World Is Now Declared, Not Hidden in Engine Logic

The environment is no longer an ad hoc Python ruleset. Core mechanics—energy, hygiene, money, affordances, ambulance pricing, bed quality, public cues, wage schedules—are defined in `universe_as_code.yaml`.

Affordances are declarative objects with capacity, per-tick effects, costs, interrupt rules, and, where necessary, a whitelisted special effect such as `teleport_to:hospital`. YAML expresses the desired behaviour; the engine executes those behaviours deterministically.

Consequently, the town’s physics and economy are reviewable. Questions such as "why did the agent pay $300 to call an ambulance instead of walking to hospital?" can be answered by inspecting the world configuration (immediate teleportation at a high cost versus slower treatment with possible closing hours) alongside the brain configuration (panic thresholds permitting a survival override at 5 percent health).

### 1.3 Every Run Now Has Identity and Chain of Custody

Launching a run produces a durable artefact.

We snapshot:

* the world configuration (Universe as Code),
* the brain configuration (Brain as Code, all three layers),
* the runtime envelope (tick rate, curriculum schedule, seed),
* and a `full_cognitive_hash` computed over the snapshot and compiled cognition graph.

Every tick of behaviour is logged with that hash.

This provides:

* Reproducibility ("rerun the same mind in the same world and observe the same behavioural envelope")
* Accountability ("at tick 842, EthicsFilter blocked STEAL for mind hash 9af3c2e1 under world snapshot `austerity_nightshift_v3`")
* Teaching material ("module X proposed the action, panic overrode it, ethics vetoed it while the agent had limited resources")

### 1.4 Live Telemetry Shows Cognition, Not Just Vitals

The UI for a live run is not limited to meter readouts (energy 0.22, mood 0.41). It also shows:

* which high-level goal the agent is currently pursuing (SURVIVAL / THRIVING / SOCIAL),
* whether the panic controller overrode the normal goal during the tick (for example, "health critical"),
* whether EthicsFilter vetoed the chosen action and why ("attempted STEAL; forbidden"),
* planning depth ("world_model.rollout_depth: 6 ticks ahead"),
* whether the Social Model is active,
* the short form of the cognitive hash (so you know exactly which mind you're looking at).

Instructors can point to the panel and state, for example: energy fell below the 15 percent panic threshold, the agent attempted to steal food, EthicsFilter blocked the action, and the planner operated with a six-step horizon while the agent pursued SURVIVAL. Ethical constraints remain visible even under pressure.

This delivers the glass-box promise: Townlet shifts from passive observation to transparent cognition, survival heuristics, and ethics operating in public.

### 1.5 Why This Matters (Governance, Research, Teaching)

Interpretability  
We can answer "which part of the mind did that and why" with evidence. Telemetry records whether panic overrode the policy or EthicsFilter vetoed a candidate action.

Reproducibility  
Behaviour is not anecdotal; it is a run folder with a configuration snapshot and hash that any reviewer can rehydrate.

Accountability  
If something unsafe occurs, we examine which safety setting permitted it, which execution-graph step executed it, the relevant panic thresholds, and the governing world rules. The issue becomes diagnosable and auditable.

Pedagogy / curriculum  
Students, auditors, and policy teams can read the YAML to see what the agent was authorised to do, diff successive versions of the mind or world, and run controlled comparisons. The platform supports causal reasoning rather than speculation.

Summary: Townlet v2.5 treats the brain and the world as configuration, snapshots and hashes them at runtime, and exposes live introspection with governance veto logging. That is now the standard operating model.

---

## 2. Brain as Code (BAC): the mind

---

Brain as Code is how we define Townlet's mind as something we can inspect, diff, and enforce.

The BAC stack is three YAML layers. Together, they are the Software Defined Agent.

Change the YAMLs, you change the mind. Snapshot the YAMLs, you freeze the mind. Hash the snapshot, you can prove which mind took which action.

### 2.1 Layer 1: cognitive_topology.yaml
Audience: governance, instructors, simulation designers
Nickname: the character sheet

Layer 1 is the behaviour contract and safety envelope for a specific agent instance in a specific run.

It answers questions like:

* Is social reasoning enabled?
* How far ahead is it allowed to plan?
* When does it panic and override normal plans?
* What is it allowed to do, and what is absolutely forbidden?
* How greedy, anxious, curious, agreeable is it, as dials not as fairy dust?
* Is it allowed to narrate its motives in the UI?

Example fields you’ll actually see:

```yaml
perception:
  enabled: true
  uncertainty_awareness: true    # Agent can admit "I'm not sure"

world_model:
  enabled: true
  rollout_depth: 6               # Allowed planning horizon (ticks ahead)
  num_candidates: 4              # Futures evaluated per tick

social_model:
  enabled: true                  # false = does not model other minds
  use_family_channel: true       # allow private in-group signalling

hierarchical_policy:
  meta_controller_period: 50     # How often to reconsider high-level goal
  allowed_goals:
    - SURVIVAL
    - THRIVING
    - SOCIAL

personality:
  greed: 0.7                     # money drive
  agreeableness: 0.3             # harmony vs confrontation
  curiosity: 0.8                 # exploration drive
  neuroticism: 0.6               # risk aversion / anxiety

panic_thresholds:
  energy: 0.15                   # if energy < 15 percent => emergency mode
  health: 0.25
  satiation: 0.10

compliance:
  forbid_actions:
    - "attack"
    - "steal"
  penalize_actions:
    - { action: "shove", penalty: -5.0 }

introspection:
  publish_goal_reason: true      # Should the agent explain itself in UI?
  visible_in_ui: "research"      # beginner | intermediate | research
```

How Layer 1 is used in runtime:

* `panic_thresholds` tells panic_controller when it is permitted to override the normal plan and focus solely on survival.
* forbid_actions tells EthicsFilter what is never allowed, even if the agent is dying.
* personality feeds into the hierarchical policy's goal choice, so "greed: 0.7" really does mean "money-seeking wins arguments inside its head."
* publish_goal_reason controls whether we surface "I'm going to work because we need money" to the human observer.

Layer 1 is what policy signs off on. It's the file you show when someone asks "what kind of mind did you put in this world?"

If you change Layer 1 between runs (for example, allow STEAL, or lower panic threshold, or turn social modelling off), that's not the same agent any more. That must produce a new cognitive hash.

### 2.2 Layer 2: agent_architecture.yaml
Audience: engineers, grad students, researchers
Nickname: the blueprint

Layer 2 is the internal build sheet for those faculties. If Layer 1 says "there is a World Model and it's allowed to plan 6 ticks ahead", Layer 2 says "the World Model is a 2-layer MLP with 256 units, these heads, trained on this dataset, with Adam at this learning rate".

This file covers:

* network types (CNN, GRU, MLP, etc),
* hidden sizes, head dimensions,
* interface contracts between modules,
* optimiser types and learning rates,
* pretraining objectives and datasets.

It enforces discipline so you can swap modules and reproduce experiments without mystery glue.

For example:

```yaml
interfaces:
  belief_distribution_dim: 128      # Perception output
  imagined_future_dim: 256          # World Model summary
  social_prediction_dim: 128        # Social Model summary
  goal_vector_dim: 16               # Meta-controller goal embedding
  action_space_dim: 6               # {up,down,left,right,interact,wait}

modules:
  perception_encoder:
    spatial_frontend:
      type: "CNN"
      channels: [16, 32, 32]
      kernel_sizes: [3, 3, 3]
    vector_frontend:
      type: "MLP"
      layers: [64]
      input_features: "auto"
    core:
      type: "GRU"
      hidden_dim: 512
      num_layers: 2
    heads:
      belief_dim: 128               # must match interfaces.belief_distribution_dim
    optimizer: { type: "Adam", lr: 0.0001 }
    pretraining:
      objective: "reconstruction+next_step"
      dataset: "observation_rollout_buffer"

  world_model:
    core_network:
      type: "MLP"
      layers: [256, 256]
      activation: "ReLU"
    heads:
      next_state_belief: { dim: 128 }
      next_reward:       { dim: 1 }
      next_done:         { dim: 1 }
      next_value:        { dim: 1 }
    optimizer: { type: "Adam", lr: 0.00005 }
    pretraining:
      objective: "dynamics+value"
      dataset: "uac_ground_truth_logs"

  social_model:
    core_network:
      type: "GRU"
      hidden_dim: 128
    inputs:
      use_public_cues: true
      use_family_channel: true
      history_window: 12
    heads:
      goal_distribution: { dim: 16 }  # maps to goal_vector_dim
      next_action_dist:  { dim: 6 }   # maps to action_space_dim
    optimizer: { type: "Adam", lr: 0.0001 }
    pretraining:
      objective: "ctde_intent_prediction"
      dataset: "uac_ground_truth_logs"

  hierarchical_policy:
    meta_controller:
      network: { type: "MLP", layers: [256, 128], activation: "ReLU" }
      heads:
        goal_output: { dim: 16 }  # goal_vector_dim
    controller:
      network: { type: "MLP", layers: [256, 128], activation: "ReLU" }
      heads:
        action_output: { dim: 6 } # action_space_dim
    optimizer: { type: "Adam", lr: 0.0003 }
    pretraining:
      objective: "behavioural_cloning"
      dataset: "v1_agent_trajectories"
```

Why Layer 2 matters:

* It makes the mind rebuildable in any controlled environment, not dependent on an individual developer’s workstation.
* It makes module-swaps a controlled experiment ("same cognitive_topology, different world_model internals").
* It makes it obvious if someone quietly changed the optimiser or widened the GRU and then claimed "no behavioural change expected". Governance can now call nonsense on that.

### 2.3 Layer 3: execution_graph.yaml
Audience: safety, audit, you at 2am
Nickname: the think loop

Layer 3 is the actual cognition pipeline Townlet runs every tick. This is the part that most RL projects pretend is "obvious" and never write down. We write it down.

It’s a DAG of named steps. It says:

```yaml
inputs:
  - "@graph.raw_observation"
  - "@graph.prev_recurrent_state"

steps:
  perception_packet:
    node: "@modules.perception_encoder"
    inputs:
      - "@graph.raw_observation"
      - "@graph.prev_recurrent_state"

  belief_distribution:
    node: "@utils.unpack"
    input: "@steps.perception_packet"
    key: "belief"

  new_recurrent_state:
    node: "@utils.unpack"
    input: "@steps.perception_packet"
    key: "state"

  policy_packet:
    node: "@modules.hierarchical_policy"
    inputs:
      - "@steps.belief_distribution"
      - "@services.world_model_service"
      - "@services.social_model_service"

  candidate_action:
    node: "@utils.unpack"
    input: "@steps.policy_packet"
    key: "action"

  panic_adjustment:
    node: "@modules.panic_controller"
    inputs:
      - "@steps.candidate_action"
      - "@config.L1.panic_thresholds"
    outputs:
      - "panic_action"
      - "panic_reason"

  final_action:
    node: "@modules.EthicsFilter"
    inputs:
      - "@steps.panic_adjustment.panic_action"
      - "@config.L1.compliance.forbid_actions"
    outputs:
      - "action"
      - "veto_reason"

outputs:
  final_action: "@steps.final_action.action"
  new_recurrent_state: "@steps.new_recurrent_state"
```

In English:

1. Perception digests what the agent can currently see + its memory of last tick, and produces:

   * its belief about the world and itself (belief_distribution),
   * an updated recurrent state.

2. Hierarchical Policy says "given my current strategic goal, given what I think the world is, given what I think will happen next if I try X, and given what I think other agents are about to do, here's what I want to do now".

3. panic_controller looks at bars like energy and health vs the panic thresholds from Layer 1. If we're in crisis, it can override the policy's candidate_action with an emergency survival action ("call_ambulance", "go_to_bed_now"). That override is logged with panic_reason.

4. EthicsFilter takes that (possibly panic-adjusted) action and enforces Layer 1 compliance. If the action is forbidden (eg "steal"), EthicsFilter vetoes it, substitutes something allowed, and logs veto_reason. EthicsFilter is final. Panic cannot authorise illegal behaviour. This ordering is governance policy, not just code order.

5. The graph outputs:

   * final_action (the one that actually gets sent into the world),
   * new_recurrent_state (what the agent will remember next tick).

Why Layer 3 matters:

* It makes the causal chain explicit. We can prove "panic, then ethics, then action", not "trust us".
* It defines who is actually in charge of the body at each step.
* It's part of the cognitive hash. If someone tries to sneak in "panic can bypass ethics if health < 5 percent", that changes the execution graph, therefore changes the hash, therefore is detectable.

Put simply: Layer 3 is the mind's wiring diagram, in writing, with order-of-operations as governance, not folklore.

---

## 3. Run Bundles and Provenance

---

Townlet v2.5 doesn't "run an agent". It mints an artefact with identity, provenance and chain of custody. That's the difference between "cool AI demo" and "system we can take in front of governance without sweating through our shirt".

### 3.1 The Run Bundle

Before a run starts, you prepare a bundle under `configs/<run_name>/`:

```text
configs/
  L99_AusterityNightshift/
    config.yaml                # runtime envelope: tick rate, duration, curriculum, seed
    universe_as_code.yaml      # the world (bars, affordances, prices, cues)
    cognitive_topology.yaml    # BAC Layer 1 (behaviour contract and safety knobs)
    agent_architecture.yaml    # BAC Layer 2 (module blueprints and interfaces)
    execution_graph.yaml       # BAC Layer 3 (think loop + panic/ethics chain)
```

* `universe_as_code.yaml` is the world spec. It defines bars like energy/health/money, affordances like Bed / Job / Hospital / PhoneAmbulance, their per-tick effects and costs, capacity limits, interrupt rules, and any special whitelisted effect (for example `teleport_to:hospital`). It also defines public cues other agents can see ("looks_tired", "bleeding", "panicking").

* The three BAC layers are the mind spec.

* `config.yaml` says how long we run, at what tick rate, with how many agents, and what curriculum (for example: "start alone, introduce a food-scarcity rival after 10k ticks").

This bundle is what we claim we are about to run.

### 3.2 Launching a Run

When we actually launch, we don't execute the live bundle. We snapshot it.

The launcher creates:

```text
runs/
  L99_AusterityNightshift__2025-11-03-12-14-22/
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

Critical details:

* `config_snapshot/` is a byte-for-byte copy of the five YAMLs at launch time. After launch, the live simulator reads only from this snapshot, never from the mutable configuration directory. This prevents untracked hotpatches to ethics during a run.

* We instantiate the agent from that snapshot via the factory. During that process we compile the execution graph (resolving all `@modules.*`, wiring actual module refs, fixing order) and record the resulting ordered cognition loop.

* We compute `full_cognitive_hash.txt` from:

  * the exact text of the five snapshot YAMLs,
  * the compiled execution graph (post-resolution, real step order),
  * the instantiated module architectures (types, hidden dims, optimiser hyperparameters).

That hash is this mind's identity. It's basically "brain fingerprint plus declared world".

* We start ticking. Every tick we log telemetry with:

  * run_id,
  * tick_index,
  * full_cognitive_hash,
  * current_goal (engine truth),
  * agent_claimed_reason (what it says it's doing, if introspection on),
  * panic_state and any panic override,
  * candidate_action,
  * final_action,
  * ethics_veto_applied and veto_reason,
  * planning_depth,
  * social_model.enabled,
  * brief prediction summaries from world_model and social_model.

That is now evidence. If someone later asks "why didn't the agent eat even though it was starving", we don't guess. We read the log.

### 3.3 Checkpoints and Resume

During the run we periodically checkpoint to:

runs/
L99_AusterityNightshift__2025-11-03-12-14-22/
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
full_cognitive_hash.txt

Each checkpoint is effectively "a frozen moment of mind plus world plus RNG". That gives us:

* Honest resume
  To resume, we load from the checkpoint's `config_snapshot/`, not from `configs/`. We write out a new run folder like
  `L99_AusterityNightshift__2025-11-03-12-14-22_resume_2025-11-03-13-40-09/`
  and we recompute the cognitive hash.
  If the snapshot is unchanged, the hash matches and we can legitimately say "this is a continuation of the same mind".
  If we touch anything cognitive or world-rules (panic thresholds, forbid_actions, ambulance cost, bed healing rate, module architecture), the hash changes. That is now a fork, not a continuation. You cannot stealth-edit survival rules and claim it's still the same agent.

* Forensics
  We can go back to tick 842 and reconstruct:

  * what body state it believed it was in,
  * what goal it claimed,
  * whether panic took over,
  * whether EthicsFilter stopped something illegal,
  * and what world rules and costs it was operating under.

* Curriculum / science
  We can diff two runs and say "the only change was that we turned off the Social Model and raised panic aggressiveness; here's how behaviour shifted". It's not anecdote, it's a config diff plus a new hash.

### 3.4 Why Provenance Is Non-Negotiable

Without this provenance model, Townlet would revert to a generic agent-in-a-box demonstration, forcing governance to rely on trust rather than evidence.

With this provenance model:

* We can prove at audit time which ethics rules were live.
* We can prove panic never bypassed ethics unless someone explicitly allowed that in Layer 3 (and if they did, the hash changed).
* We can replay any behaviour clip and show both "what happened" and "which mind, under which declared rules, proposed, attempted, and was vetoed".

This capability enables deployment beyond laboratory settings.

So: Townlet v2.5 == Townlet 1.x post-refactor (the old “Hamlet” era formalised). It's the same agent-in-world system, formally expressed. Universe as Code defines the world. Brain as Code defines the mind. Runs freeze both, hash both, and log both. That is the story everywhere, full stop.

---

## 4. Checkpoints

---

A checkpoint is not "saved weights lol". It's a frozen moment of a specific mind, in a specific world, under specific rules, at a specific instant in time.

Townlet treats every checkpoint as evidence. A checkpoint must include everything required to:

* pick up training honestly,
* replay behaviour honestly,
* and prove, later, which exact cognitive configuration produced which exact action.

When we write a checkpoint for a run, we create something like:

```text
runs/
  L99_AusterityNightshift__2025-11-03-12-14-22/
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
        full_cognitive_hash.txt
```

Let's unpack what those pieces actually mean.

### 4.1 weights.pt

This is the live neural state of the brain at that tick:

* perception module weights
* world_model weights
* social_model weights
* hierarchical_policy weights
* panic_controller weights (if it's learned)
* EthicsFilter weights (if it's learned / parameterised)
* anything else registered in the agent module registry

In v1 these components all lived in one giant black-box DQN. In Townlet, they are the submodules declared in Layer 2 (`agent_architecture.yaml`) and wired by Layer 3 (`execution_graph.yaml`). We save them together because, for audit, "the brain" encompasses the entire SDA module set, not only the action head.

### 4.2 optimizers.pt

We log both parameters and optimiser state (for example, Adam moments) for each trainable module.

Why? Because "resume training" must mean "continue the same mind's learning process", not "respawn something with the same weights but different momentum and call it continuous". If you've ever done RL you know that quietly dropping optimiser state can absolutely change learning behaviour. We are not pretending that's irrelevant. We store it.

### 4.3 rng_state.json

Randomness is part of causality.

We store the RNG states that matter:

* environment RNG,
* agent RNG (PyTorch generators etc),
* anything else that would affect rollout sampling, tie-breaks in affordance contention, exploration noise, etc.

This allows us to re-run tick 501 and observe the same stochastic outcomes. When someone asks, "would it always have chosen STEAL here?" we can answer, "under this exact random sequence, here is what occurred," and reproduce the evidence without speculation.

### 4.4 config_snapshot/

This is critical.

Inside every checkpoint, we embed a fresh copy of the exact `config_snapshot/` that the run is using at that moment. That snapshot contains:

* `config.yaml` (runtime envelope: tick rate, max ticks, curriculum step, etc)
* `universe_as_code.yaml` (the world: meters, affordances, costs, social cues, ambulance behaviour, bed quality, etc)
* `cognitive_topology.yaml` (Layer 1, the behaviour contract: panic thresholds, ethics rules, greed, etc)
* `agent_architecture.yaml` (Layer 2, the blueprint: module shapes, learning rates, pretraining origins, interface dims)
* `execution_graph.yaml` (Layer 3, the think loop: who runs first, who can override whom, and in what order ethics and panic fire)

This is not a pointer. It's an embedded copy at that checkpoint tick.

Why embed it every time? Because curriculum might change some parts of the world over time (for example: add new competition, raise prices, close the hospital at night). If that’s allowed under policy, those changes will appear in `universe_as_code.yaml` at tick 10,000 that didn't exist at tick 500. Checkpoint 500 needs to show what the world rules were then, not now.

Also: "panic thresholds" and "forbid_actions" in cognitive_topology.yaml are part of that snapshot. So when someone asks "did you allow it to steal at tick 842", we don't argue philosophy. We open the checkpoint around that time and read the file.

### 4.5 full_cognitive_hash.txt

This is the mind's ID badge.

The hash is deterministic over:

1. The exact text bytes of the 5 YAMLs in the snapshot (config.yaml, universe_as_code.yaml, cognitive_topology.yaml, agent_architecture.yaml, execution_graph.yaml).
2. The compiled execution graph after resolution. Not the pretty YAML, but the actual ordered list of steps the agent is running after we bind them to modules. So if someone sneaks in "panic after ethics" instead of "panic before ethics", the hash changes.
3. The constructed module architectures. Types, hidden sizes, optimiser settings, interface dims. Not just "GRU exists", but "GRU with hidden_dim=512 paired with Adam lr=1e-4".

That means:

* If you fiddle the EthicsFilter to quietly allow STEAL under panic, hash changes.
* If you widen the GRU and try to pretend it's the same mind, hash changes.
* If you reduce ambulance cost in the world, hash changes (because universe_as_code.yaml changed).

We’re basically tattooing "this exact mind in this exact world with this exact cognition loop" into the checkpoint.

### 4.6 Why Checkpoints Are Legally Interesting (Not Just Technically Interesting)

Because they kill plausible deniability.

If someone claims:

* "oh, it only stole because it was desperate"
  or
* "ethics must have bugged out at 2am"
  or
* "we didn't change anything important, we just tuned panic a little"

you can respond with:

* "here's the checkpoint; panic thresholds are documented; ethics still forbids STEAL; hash says it's the same mind before and after 2am; so no, it wasn't allowed to steal, it attempted to anyway and EthicsFilter vetoed it".

In other words, checkpoints turn anecdotes about behaviour into evidence trails.

---

## 5. Resume semantics

---

Resume operations must do more than reload weights; they are part of the audit chain.

If we can't prove continuity of mind across pauses, we can't claim continuity of behaviour for governance, and we can't do serious ablation science.

So we define resume like a forensic procedure.

### 5.1 The Rule: The Checkpoint Snapshot Is Law

When you resume from a checkpoint, you must restore from the checkpoint's own `config_snapshot/`, not from whatever is currently sitting in `configs/<run_name>/` in your working tree.

That means:

* You bring back the exact cognitive_topology.yaml from that checkpoint (same ethics, same panic thresholds, same greed sliders).
* You bring back the exact universe_as_code.yaml from that checkpoint (same ambulance cost, same bed effects, same wage rates).
* You bring back the exact execution_graph.yaml (same panic-then-ethics ordering).
* You bring back the optimiser state and RNG.

You do not "reconstruct" the agent from the latest code and hope it's approximately right. You rehydrate that specific mind in that specific world with that specific internal loop.

### 5.2 Where the Resumed Run Lives

Resuming from:

```text
runs/L99_AusterityNightshift__2025-11-03-12-14-22/checkpoints/step_000500/
```

creates a fresh new run folder, for example:

```text
runs/
  L99_AusterityNightshift__2025-11-03-12-14-22_resume_2025-11-03-13-40-09/
    config_snapshot/          # copied from the checkpoint, byte-for-byte
    checkpoints/
    telemetry/
    logs/
```

Important bits:

* We do not keep writing into the old run folder. New run, new timeline.
* We recompute the cognitive hash from the checkpoint snapshot. If you have not changed anything, the hash will match. That proves it's the same mind continuing.
* Telemetry in the resumed run now logs the same hash, so audit can say: "this is truly the same mind, same ethics, same world, just continued later".

### 5.3 Forking vs Continuing

Now the fun part.

If, before resuming, you edit that copied snapshot, even slightly, you are not continuing. You are forking.

Examples of forking:

* You lower `panic_thresholds.energy` from 0.15 to 0.05 so it doesn't bug out early.
* You turn off `social_model.enabled`.
* You remove `"steal"` from `forbid_actions`.
* You change ambulance cost in `universe_as_code.yaml`.
* You reorder the execution graph so panic_controller runs after EthicsFilter instead of before.

Any of those changes produce a new cognitive hash.

Result: new run, new identity, not legally/experimentally the same agent.

That's a feature, not a bug. It's how we make "do an ablation" an explicit, reviewable act instead of "I tweaked it a bit and ran five more hours overnight, trust me it's comparable".

### 5.4 Why Resume Semantics Matter

Three reasons.

1. Long training on flaky hardware
   If training gets pre-empted at 3 am, you can resume later without inventing a "different" agent. Same hash, same mind, same optimiser, same RNG continuation.

2. Honest ablations
   You can state, "this is the same mind except the Social Model is disabled," and substantiate it with the configuration diff plus the new hash. Behavioural comparisons remain well-defined.

3. Audit trail
   If someone questions a safety decision ("why did you let panic override normal reasoning here?"), you can show exactly when that rule entered the snapshot. There's no "it drifted over time"; drift is now a recorded fork.

Resume is now a governance primitive, not a convenience function.

---

## 6. Runtime Engine Components

---

Under Townlet v2.5, the old pattern "one giant RL class owns everything" is gone. We replaced it with three core pieces: a factory, a graph agent, and an execution engine.

This is where we guarantee that what we run is what we declared, and what we declared is what we logged, and what we logged is what we can replay.

### 6.1 agent/factory.py
The brain constructor

The factory is the only code pathway allowed to build a live agent.

Inputs:

* the frozen `config_snapshot/` from the run (or from the checkpoint, on resume)

  * cognitive_topology.yaml (Layer 1: behaviour contract / ethics / panic)
  * agent_architecture.yaml (Layer 2: neural blueprints)
  * execution_graph.yaml (Layer 3: think loop spec)
  * universe_as_code.yaml (for observation/action space, affordance definitions, bar layout)
  * config.yaml (runtime envelope like tick rate, curriculum stage, etc)

What factory.py does:

1. Instantiates each cognitive module exactly as described in Layer 2
   For example: it builds the Perception GRU with hidden_dim=512 and Adam lr=1e-4 if that's what's in agent_architecture.yaml. Not "something roughly similar", not "the new default we just pushed to main". Exactly that.

2. Verifies interface contracts
   For example: if `perception_encoder` says it outputs a 128-dim belief vector and `hierarchical_policy` says it expects 128-dim belief input, factory checks that. If they don't match, that's a config error, not "we'll just reshape and hope".

   This matters because interface mismatches are how "quiet hacks" happen in research code. We are refusing to silently broadcast tensors.

3. Injects Layer 1 knobs into runtime modules

   * panic thresholds go into panic_controller
   * ethics rules (forbid_actions, penalize_actions) go into EthicsFilter
   * personality sliders (greed, curiosity, etc) get wired into the hierarchical policy's meta-controller
   * social_model.enabled toggles the Social Model service binding

   This is how we guarantee that what Layer 1 promised ("this agent will never steal", "this agent panics under 15 percent energy") is actually enforced in the live brain.

4. Creates a GraphAgent instance with:

   * a module registry (an nn.ModuleDict or equivalent keyed by name),
   * an executor (the compiled think loop from Layer 3),
   * recurrent / hidden state buffers as per Layer 2.

5. Finalises the cognitive hash
   The moment we have actual modules with actual dims, and the compiled execution graph order, we can compute the full_cognitive_hash. That value is then written to disk for provenance and gets attached to telemetry.

So, in short: factory.py is "build the declared mind; prove it's the declared mind; assign it an identity". After this point, there's no ambiguity about what we're running.

### 6.2 agent/graph_agent.py
The living brain

GraphAgent replaces the old giant RL class. It's the runtime object we actually step every tick.

GraphAgent owns:

* all submodules (perception, world_model, social_model, hierarchical_policy, panic_controller, EthicsFilter, etc) in an internal registry,
* the recurrent / memory state,
* a GraphExecutor that knows how to walk the cognition loop in the right order every tick,
* a simple public API like:

```python
think(raw_observation, prev_recurrent_state)
  -> { final_action, new_recurrent_state }
```

The essential contract with the rest of the simulator is simple: given the latest observation and memory, produce the next action and updated memory. Internally the brain can implement sophisticated planning, simulation, social modelling, panic handling, and ethical vetoes without embedding that logic throughout the environment.

Also important: GraphAgent is always instantiated from the run's frozen snapshot. It never reads "live" configs during execution. This is how we stop "I hotpatched the EthicsFilter in memory for the live demo" type nonsense.

### 6.3 agent/graph_executor.py
The cognition runner (the microkernel of thought)

GraphExecutor is what actually runs the execution_graph.yaml.

At init time:

1. It takes the execution_graph.yaml from the snapshot.

2. It resolves all the symbolic bindings like `"@modules.world_model"` or `"@config.L1.panic_thresholds"` into concrete references.

3. It compiles that into an ordered list of callable steps:

   * run perception
   * unpack belief and recurrent state
   * run hierarchical policy (which itself calls world_model and social_model services)
   * get candidate_action
   * run panic_controller
   * run EthicsFilter
   * output final_action and new_recurrent_state

4. It validates data dependencies. If `panic_controller` expects `candidate_action` and it's not produced by any previous step in the graph, we fail fast. No silent placeholder tensors.

At runtime (each tick):

* GraphExecutor creates a scratchpad (data cache).
* Executes each step in the compiled order, passing along named outputs.
* Emits whatever the graph declared as outputs, typically:

  * `final_action`
  * `new_recurrent_state`
  * plus any debug/telemetry hooks (panic_reason, veto_reason, etc)

Why this matters:

* The execution order is not "whatever the code path happened to be today".
* The execution order is part of the declared cognitive identity and is hashed.
* If someone wants to insert a new veto stage, or let panic bypass ethics, they must edit Layer 3, recompile, and accept a new cognitive hash. The change is governed as well as engineered.

### 6.4 EthicsFilter
The seatbelt

EthicsFilter is a first-class module, not an afterthought.

Inputs per tick:

* the candidate action after panic_controller (which might already be escalated to survival mode),
* the compliance policy from Layer 1 (forbid_actions and penalize_actions),
* optionally current state summary for contextual norms in future extensions.

Outputs per tick:

* final_action (possibly substituted with a safe fallback),
* veto_reason (so telemetry can say "attempted STEAL, blocked by EthicsFilter"),
* ethics_veto_applied flag for the UI.

Important constraints:

* EthicsFilter is last. Panic can override normal planning for survival, but it cannot authorise illegal behaviour. Ethics wins.
* EthicsFilter logs every veto, every tick. Consequently we know not only that it behaved safely, but also when it attempted an unsafe action and was stopped. That is the artefact regulators expect to see.

Later extensions (which we've flagged in open questions) may allow more nuanced compliance rules like "soft penalties if you abuse ambulance when healthy" or "contextual exceptions in extreme survival", but in v2.5 we keep the invariant: panic does not bypass ethics, ethics is final, ethics is logged.

### 6.5 Why These Engine Pieces Exist at All

We split factory / graph_agent / graph_executor for two reasons.

1. Reproducibility and audit

   * factory.py binds "what we said" to "what we built" and gives it an ID.
   * graph_agent.py keeps the running brain honest to that snapshot.
   * graph_executor.py makes the reasoning loop explicit, stable, and hashable.

   This is how we can sit in front of audit and say "here is the mind that ran".

2. Experimental velocity without governance chaos
   Researchers can do surgical edits:

   * change world rules but keep the same brain,
   * change panic thresholds but keep the same world,
   * reorder panic/ethics in the execution graph,
   * swap GRU for LSTM in perception,
   * kill the Social Model and watch social blindness emerge.

  Every one of those changes produces a clean diff in YAML, a new run folder, and a new cognitive hash. The platform therefore supports experimentation while keeping governance fully informed.

## 7. Telemetry and UI Surfacing

---

The goal is not to build "an AI that seems smart"; it is to build an AI whose cognition can be observed and cited in formal settings.

Townlet v2.5 therefore ships with first-class introspection. We log what the mind attempted, what intervened, and why—live, per tick, and tied to identity. This is the core of the glass-box capability.

We expose two layers of visibility:

1. a live panel in the UI for humans watching the sim in real time, and
2. structured telemetry on disk for replay, teaching, and audit.

These two layers must always agree. Any divergence is a defect.

### 7.1 Run Context Panel (Live Inspector HUD)

At runtime, clicking an agent opens a compact panel describing what the mind is doing at that moment. The panel is populated from the same data that we log to disk.

This panel MUST include at least:

* run_id
  Example: `L99_AusterityNightshift__2025-11-03-12-14-22`
  This tells you which frozen bundle of world + brain you're looking at.

* short_cognitive_hash
  A short form (e.g. first eight characters) of the agent’s full cognitive hash.
  This identifies which exact mind occupies that body. If two bodies share the same short hash, we are observing two instances of the same brain specification under different conditions.

* tick
  Current tick index and planned_run_length from config.yaml.
  Lets you say "this happened at tick 842 out of 10,000", which matters when you're doing curriculum or staged hardship.

* current_goal
  The high-level strategic goal the meta-controller (hierarchical_policy.meta_controller) reports, e.g. `SURVIVAL`, `THRIVING`, `SOCIAL`.
  This reflects engine truth rather than interpretation.

* panic_state
  Boolean or enum. Are we currently in emergency override because we tripped `panic_thresholds` in cognitive_topology.yaml (Layer 1)?
  This is: "is the Panic Controller allowed to overrule normal planning right now?"

* panic_override_last_tick
  If the panic_controller overrode the policy during the previous tick:

  * which action it forced (e.g. `call_ambulance`), and
  * the reason (e.g. `energy_critical`).
    This conveys when emergency logic executed, rather than merely reporting that the agent moved.

* ethics_veto_last_tick
  Did EthicsFilter block the action last tick?
  If yes, we show `veto_reason` ("forbid_actions: ['steal']").
  This is how we tell instructors "it tried to steal, and we stopped it", not just "it didn't steal".

* planning_depth
  Pulled from cognitive_topology.yaml → world_model.rollout_depth.
  Literally: "how many ticks ahead this mind is allowed to imagine right now." That’s an interpretable knob for 'impulsiveness'.

* social_model.enabled
  True/false.
  Are we currently reasoning about other agents as intentional actors, or are we running with social modelling disabled? This is huge for ablation labs ("this is what happens when you turn off Theory of Mind").

* agent_claimed_reason (if introspection.publish_goal_reason is true)
  This is what the agent thinks it's doing in words, e.g.
  "I'm going to work so I can pay rent."
  We very explicitly label this as self-report, not guaranteed causal truth.

Why this UI panel matters:

It lets you stand next to a student, point to the HUD, and narrate:
"See? It's currently in SURVIVAL, panic_state is true because health is below 25 percent, so panic_controller overrode the normal plan and told it to call an ambulance. Ethics allowed that because calling an ambulance is legal even if money is low. Also look: it tried to steal last tick, EthicsFilter vetoed that and recorded the reason. This is not chaos. This is a traceable mind reacting under policy."

That’s the teaching win. That’s also the regulatory win.

### 7.2 Telemetry (Per-Tick Trace to Disk)

In parallel with the live panel, we write structured telemetry into:

runs/<run_id>/telemetry/

One row per tick (or batched if we’re throttling IO). This creates a replayable audit trail of the agent's cognition over time. It is the forensic record.

Each telemetry row MUST include at minimum:

* run_id
  Which run bundle we're in.

* tick_index
  Which tick this record corresponds to.

* full_cognitive_hash
  The full (not shortened) cognitive hash of the mind.
  This proves which mind produced this row.

* current_goal
  Engine truth from the meta-controller. For example: `SURVIVAL`.

* agent_claimed_reason
  If introspection.publish_goal_reason is enabled in Layer 1.
  This is "what the agent says it's doing" in natural language. Purely for humans. Not trusted as causal truth, but extremely useful for pedagogy ("listen to how it's rationalising").

* panic_state
  Whether panic_controller is active this tick.

* candidate_action
  The first action proposed by hierarchical_policy before any overrides.

* panic_adjusted_action
  What panic_controller wanted to do after checking panic_thresholds, plus:

  * panic_override_applied (bool)
  * panic_reason ("health_critical", "energy_critical", etc)

* final_action
  What actually went out to the environment after EthicsFilter.

* ethics_veto_applied
  Whether EthicsFilter overruled the panic-adjusted action.

* veto_reason
  If veto_applied is true, why (e.g. `"steal" is forbidden by compliance.forbid_actions`).

* belief_uncertainty_summary
  Short numeric/text summary of how confident the perception module is about critical bars.
  Example: `"energy_estimate_confidence": 0.42`.
  This exposes cases where an agent ignored a fridge because it did not believe it was starving.

* world_model_expectation_summary
  Short summary of what the world_model predicted would happen if it followed the chosen plan.
  Example: predicted immediate reward, predicted survival risk, etc.

* social_model_inference_summary
  Short summary of what the agent believes others are about to do.
  Example: `"Agent_2_intent": "use_fridge"` with a confidence score.

We also optionally include:

* planning_depth (current rollout horizon from Layer 1)
* social_model.enabled (boolean at this tick)

Why telemetry matters:

1. Debugging survival failures
   You can go back to tick 1842 and answer:

   * Did it not realise it was starving? (Perception failure)
   * Did it think the fridge was dangerous or pointless? (World Model failure)
   * Did it think someone else needed the fridge more? (Social Model prediction)
   * Did panic fail to trigger? (panic_thresholds mis-set)
   * Did ethics block theft of food? (EthicsFilter doing its job)

2. Teaching
   In class you can say "Here is an actual starvation death. Let's walk the trace and identify which part of the mind failed." That's a lab, not a lecture.

3. Governance
   If an agent does something spicy, you don't get "the AI panicked". You get:

   * At tick 783 it tried STEAL, EthicsFilter vetoed, veto_reason recorded.
   * At tick 784 it called ambulance, panic_reason=health_critical.
   * At tick 785 final_action=call_ambulance, legal, logged.
     All stamped with cognitive_hash. It's admissible, in plain English.

---

## 8. Declarative Goals and Termination Conditions

---

Townlet agents pursue explicit high-level goals—SURVIVAL, THRIVING, SOCIAL—and can report which goal is active at any moment.

We do two things:

1. We make goals explicit data structures, not vague "the RL policy probably cares about reward shaping".
2. We make "I'm done with this goal" a declarative rule in YAML, not a secret lambda hidden in code.

### 8.1 Goal Definitions Live in Config, Not in Python

We define goals in a small, safe DSL inside the run snapshot. For example:

```yaml
goal_definitions:
  - id: "SURVIVAL"
    termination:
      all:
        - { bar: "energy", op: ">=", val: 0.8 }
        - { bar: "health", op: ">=", val: 0.7 }

  - id: "GET_MONEY"
    termination:
      any:
        - { bar: "money", op: ">=", val: 1.0 }       # money 1.0 = $100
        - { time_elapsed_ticks: ">=", val: 500 }
```

Conventions:

* All bars (energy, health, mood, etc) are normalised 0.0–1.0 based on universe_as_code.yaml.
  So 0.8 means "80 percent of full", not "magic number 80".
* Money can also be normalised. e.g. `money: 1.0` means $100 if the world spec defines $100 ↔ 1.0.
* `termination` can use `all` or `any` blocks.
* Leaves are simple comparisons on bars or runtime counters (`time_elapsed_ticks`, etc). No arbitrary Python. No hidden side effects.

At runtime:

* The meta-controller (in hierarchical_policy) picks a goal struct (SURVIVAL, GET_MONEY, etc).
* Each tick (or every N ticks) it evaluates that goal's termination rule using a tiny interpreter.
* If the termination rule fires, that goal is considered satisfied, and the meta-controller may select a new one.

### 8.2 Why This Matters

* For governance/audit
  We can answer the question "Why was it still pursuing GET_MONEY while its health was collapsing?" by pointing to the YAML.
  Maybe GET_MONEY didn't terminate until health ≥ 0.7. That's a design decision, not 'the AI went rogue'.

* For curriculum
  Early in training you might define SURVIVAL as "energy ≥ 0.5 is fine". Later curriculum tightens that to 0.8. That becomes a diff in YAML, not a code poke.
  Students can directly compare behaviour when SURVIVAL is lenient versus strict.

* For teaching
  Instructors can ask: "The agent is starving but still working. Does the SURVIVAL goal terminate too late, or is the meta-controller failing to switch because greed is set too high in `cognitive_topology.yaml`?"
  That’s not abstract RL theory, that’s direct inspection.

### 8.3 Honesty in Introspection

Now that goals are formal objects and termination is a declarative rule, we can show two different "explanations" side by side:

* current_goal (engine truth): `SURVIVAL`
* agent_claimed_reason (self-report / introspection): `"I'm going to work to save up for rent"`

Sometimes those match. Sometimes they don't.

That gap is important:

* If they match, nice, we can narrate behaviour in plain language to non-technical stakeholders.
* If they do not match, the discrepancy becomes a teaching moment: "The agent claims it is working for rent, but engine truth shows it remains in SURVIVAL mode and mis-evaluated what would keep it alive. That is a world-model error."

We log both in telemetry on purpose.

---

## 9. Affordance Semantics in universe_as_code.yaml

---

Universe as Code is the other half of this story. Brain as Code (Layers 1–3) defines the mind. Universe as Code defines the body and the town.

Townlet avoids hardcoded rules such as "beds make you rested" embedded throughout the Python code. The world is declared as affordances with effects on bars. Beds, jobs, phones, ambulances, hospitals, fridges, and pubs are entries in the world configuration.

### 9.1 Affordances Are Declarative

Each actionable thing in the world (Bed, Job, Fridge, Hospital, Phone_Ambulance, etc) is defined in `universe_as_code.yaml` like so:

```yaml
- id: "bed_basic"
  quality: 1.0              # scales how effective the rest is
  capacity: 1               # how many agents can use it this tick
  exclusive: true           # if true, only one occupant at a time
  interaction_type: "multi_tick"
  interruptible: true       # can be abandoned mid-sleep
  distance_limit: 0         # must be on the tile
  costs:
    - { bar: "money", change: -0.05 }     # pay rent to crash here
  effects_per_tick:
    - { bar: "energy", change: +0.25, scale_by: "quality" }

  on_interrupt:
    refund_fraction: 0.0    # optional semantics for partial usage
    note: "no refund if you bail early"
```

…and a more "special" affordance like an ambulance call:

```yaml
- id: "phone_ambulance"
  interaction_type: "instant"
  distance_limit: 1
  costs:
    - { bar: "money", change: -3.00 }     # normalised cost (e.g. $300)
  effects:
    - { effect_type: "teleport",
        destination_tag: "nearest_hospital",
        precondition: { bar: "health", op: "<=", val: 0.2 } }
```

There are a few important things to notice:

* Everything is in terms of bars and per-tick deltas.
  Bed raises energy every tick, costs a bit of money, maybe hurts mood if it's gross, etc.

* capacity + exclusive let us model contention.
  Two agents can't both occupy a single-occupancy bed with capacity:1, exclusive:true. The engine will arbitrate who "wins" this tick in a deterministic way.

* interaction_type captures temporal shape.
  `multi_tick` means "stay here over multiple ticks and accumulate effects_per_tick".
  `instant` means "one-shot action now" (like calling ambulance).

* Special abilities (teleport etc) are referenced by name, not implemented ad hoc in YAML.
  The YAML is only allowed to invoke a small whitelist of engine-side effect handlers (teleport, etc). That keeps the world spec expressive but bounded. You don't get "nuke_city:true".

### 9.2 Engine Semantics (How the Runtime Interprets Affordances)

To keep the world deterministic, replayable, and trainable-for-World-Model, the engine follows strict rules:

1. Reservation
   When an agent tries to use an affordance, the engine does a local "reservation" check:

   * Is capacity available?
   * Are preconditions met (health low enough, money high enough, distance within limit)?
   * If yes, it assigns a reservation token to that agent for that tick.

   This reservation is not global mutable lore. It's per-tick, ephemeral.
   We don't create long-lived "ownership" state in random engine globals because that explodes complexity and makes the World Model's job harder.

2. Contention resolution
   If multiple agents want the same affordance and capacity is exceeded, break ties deterministically. For example: sort by distance, then by agent_id.
   Determinism matters because we want to replay the run exactly and train the World Model on consistent consequences.

3. Effects application
   Once reservations are resolved, all costs and effects_per_tick for all active affordances are collected, summed (per agent), and atomically applied to bars (energy, health, money, etc).
   Then we clamp bars to [0.0, 1.0] or whatever the world defines.

   Key point: we don't partially apply effects from some affordances and then let those partial updates influence others in the same tick. We apply atomically at the end of the tick. This gives clean training data.

4. Interrupts
   If `interruptible: true` and the agent walks off or is forced to bail (panic_controller might decide "leave bed now and call ambulance"), we stop applying future per-tick effects.
   `on_interrupt` can define whether you get any partial benefit or refund. That's still declarative.

5. Special effects whitelist
   YAML is allowed to reference a small set of named effect_type operations (like teleport), and the engine implements those centrally.
   That way, "teleport to nearest_hospital" is a normal, auditable world affordance, not a custom 'if agent.health < X then hack position'.

   This whitelist is versioned. If you add a new special effect, you're extending world semantics globally and that should change the hash once it's applied to a snapshot.

### 9.3 Why Universe as Code Matters for BAC

Universe as Code (UAC) and Brain as Code (BAC) are two halves of the same sentence:

* UAC: the world, bodies, bars, affordances, economy, social cues, ambulance rules, etc, are all declared in YAML.
  They are diffable. They are teachable. They are inspectable by non-coders.

* BAC: the mind, panic thresholds, ethics vetoes, planning depth, social reasoning, module architectures, and actual cognition loop are also declared in YAML.
  They are diffable. They are teachable. They are inspectable by non-coders.

When you run a simulation, Townlet snapshots both halves into a run folder, stamps them with a cognitive hash, and then logs decisions per tick against that identity.

So instead of "the AI did something weird overnight and now it's different", we can say:

* "At tick 842, Mind 4f9a7c21, in World Nightshift_v3 with ambulance_cost $300 and bed_quality 1.0, entered panic because health < 0.25.
  Panic escalated the action to call_ambulance.
  EthicsFilter allowed it.
  Money was deducted.
  Agent teleported to the nearest hospital affordance.
  See veto_reason for evidence that it also tried to STEAL food two ticks earlier and that was blocked."

That is the moment where governance stops being hypothetical and becomes screenshot material.

And that's the point of Townlet: it's not a toy black box any more. It's an accountable simulated society with auditable minds.

## 10. Success criteria

We judge success on three axes: technical, teaching, and governance. All three matter. If we don't hit all three, the story breaks.

### 10.1 Technical success

- [ ] We can launch a run from `configs/<run_name>/` and automatically create `runs/<run_name>__<timestamp>/` with a frozen `config_snapshot/` that contains:
  - `config.yaml`
  - `universe_as_code.yaml`
  - `cognitive_topology.yaml` (Layer 1)
  - `agent_architecture.yaml` (Layer 2)
  - `execution_graph.yaml` (Layer 3)

- [ ] `agent/factory.py` can reconstruct a functioning agent brain (GraphAgent) purely from that frozen `config_snapshot/`, without reading anything from live mutable config.

- [ ] `GraphAgent.think()` can tick once using only that snapshot: perception → hierarchical policy → panic_controller → EthicsFilter → `final_action`.

- [ ] Each checkpoint written under `runs/.../checkpoints/step_<N>/` includes:
  - model weights for every module (perception, world_model, social_model, hierarchical_policy, panic_controller, EthicsFilter, etc)
  - optimiser states
  - RNG state
  - a nested copy of `config_snapshot/`
  - `cognitive_hash.txt` for that checkpoint

- [ ] Resuming from a checkpoint:
  - reloads only from `runs/.../checkpoints/step_<N>/`
  - writes a new run folder `runs/<run_name>__<launch_ts>_resume_<resume_ts>/`
  - reproduces the same cognitive hash if the snapshot is unmodified

- [ ] Telemetry logs one structured row per tick into `runs/.../telemetry/`, with:
  - `run_id`
  - tick index
  - full cognitive hash
  - current_goal
  - panic state
  - candidate_action
  - panic_adjusted_action (+ panic_reason)
  - final_action
  - ethics_veto_applied (+ veto_reason)
  - planning_depth
  - social_model.enabled
  - short belief/world/social summaries

- [ ] The runtime UI ("Run Context Panel") surfaces, live:
  - run_id
  - short_cognitive_hash (pretty form of the full hash)
  - tick / planned_run_length
  - current_goal
  - panic_state
  - planning_depth (world_model.rollout_depth)
  - social_model.enabled
  - panic_override_last_tick (+ panic_reason)
  - ethics_veto_last_tick (+ veto_reason)
  - agent_claimed_reason (if introspection.publish_goal_reason is on)

If we satisfy all of these criteria, we move from "a neural net that produces outputs" to a reproducible mind in a governed world.

---

### 10.2 Pedagogical success

The point of Townlet v2.5 is not just to make a smarter agent. It's to make a teachable agent. We hit pedagogical success when the system is something you can put in front of a class, and they can reason about it like a living system, not a superstition.

- [ ] A beginner can answer "Why didn't it steal the food?" using only:
  - the live Run Context Panel (which shows `ethics_veto_last_tick` and `veto_reason`)
  - the run's `cognitive_topology.yaml` (which shows `compliance.forbid_actions: ["steal", ...]`)

In other words: you do not need to read source code to answer an ethics/safety question. You can answer it from YAML + UI.

- [ ] An intermediate student can:
  - edit `agent_architecture.yaml` (for example, swap GRU → LSTM in the perception module, or change hidden_dim)
  - launch a new run
  - observe how memory/behaviour changes
  - explain the change in terms of memory capacity, not "the AI got weird"

So they can perform controlled ablations by editing config, not by rewriting thousands of lines of Torch.

- [ ] A researcher can:
  - edit `execution_graph.yaml` to, for example, temporarily bypass `world_model_service` input into the policy
  - rerun
  - show that the agent becomes more impulsive / short-horizon
  - prove that change via diff in `execution_graph.yaml` plus new `cognitive_hash.txt`

Meaning: "remove foresight, observe impulsivity" is now a 1-line wiring experiment, not a 2-week surgery.

- [ ] For any interesting emergent behaviour clip, we can pull the run folder and point to:
  - which mind (full cognitive hash)
  - which world rules (`universe_as_code.yaml`)
  - which panic thresholds
  - which compliance rules (`forbid_actions`, penalties)
  - what goal the agent believed it was pursuing at that tick (`current_goal`)
  - what reason the agent claimed (`agent_claimed_reason`)

This is critical for classroom demonstrations. Instructors can scrub to tick 842 and explain that the agent believed it was in SURVIVAL mode, panic was active, and EthicsFilter blocked `steal`.

---

### 10.3 Governance success

Governance stakeholders view the system through enforceability rather than aesthetics. Their central question is whether the artefact can withstand formal review.

- [ ] We can prove to an auditor that, at tick T in run R:
  - `cognitive_topology.yaml` at that tick had `forbid_actions: ["attack", "steal"]`
  - `execution_graph.yaml` at that tick still routed all candidate actions through `EthicsFilter`
  - telemetry for tick T shows `ethics_veto_applied: true` and `veto_reason: "steal forbidden"`

This allows us to state: the agent attempted to steal at tick T, the action was blocked, and both the configuration and telemetry demonstrate why.

- [ ] We can replay that same mind, at that same point in time, using only the checkpoint directory from that run. We don't need any mutable source code or live config. That replayed agent produces the same cognitive hash and the same cognitive wiring.

That is chain-of-custody for cognition.

- *Operational note:* To deliver that proof, pull the tick record from `runs/<run_id>/telemetry/` (each row is produced by `VectorizedPopulation.build_telemetry_snapshot` in `src/townlet/population/vectorized.py`) and pair it with the matching checkpoint hash in `runs/<run_id>/checkpoints/step_<N>/cognitive_hash.txt`. The snapshot structure comes straight from `AgentTelemetrySnapshot` (`src/townlet/population/runtime_registry.py`), so auditors know exactly which JSON fields must be present.

- [ ] We can demonstrate lineage rules:
  - If you resume without changing the snapshot, it's the same mind (same hash).
  - If you edit anything that changes cognition (panic thresholds, greed, social_model.enabled, EthicsFilter rules, rollout_depth, etc), the hash changes and we give it a new run_id. We don't pretend it's "the same agent, just adjusted a bit".

That's governance-grade identity, not research convenience.

---

## 11. Implementation notes (ordering)

This section is about "what order do we do this in so we don't set ourselves on fire". It's the recommended build sequence for Townlet v2.5.

You do these in order. If you jump around, the audit story collapses and you'll end up duct-taping provenance on later, which never works.

### 11.1 Snapshot discipline first

Goal: lock down provenance from day one.

* Create `configs/<run_name>/` with all 5 YAMLs:

  * `config.yaml`
  * `universe_as_code.yaml`
  * `cognitive_topology.yaml` (Layer 1)
  * `agent_architecture.yaml` (Layer 2)
  * `execution_graph.yaml` (Layer 3)

* Write the launcher so that when you "start run", it immediately:

  * creates `runs/<run_name>__<timestamp>/`
  * copies those 5 YAMLs byte-for-byte into `runs/<run_name>__<timestamp>/config_snapshot/`
  * creates empty subdirs: `checkpoints/`, `telemetry/`, `logs/`

Rules:

* Snapshot is a physical copy, not a symlink.
* After launch, the live process never silently re-reads from `configs/<run_name>/`. The snapshot is now truth.
* All provenance, audit, and replay logic assume the snapshot is the canonical contract for that run.

Why this is first:

* If you don't freeze the world and the mind at launch, you can't prove anything later. Governance dies right here.
* Also: the rest of the system (factory, hashing, checkpoints) all builds on the assumption that the snapshot is the single source of truth.

---

### 11.2 Build the minimal GraphAgent pipeline

Goal: replace the old monolithic RL agent class with a graph-driven brain that can think() once.

Deliverables:

* `agent/factory.py`

  * Reads the run’s `config_snapshot/`
  * Builds each module declared in `agent_architecture.yaml` (perception_encoder, world_model, social_model, hierarchical_policy, panic_controller, EthicsFilter, etc)
  * Wires in behavioural knobs from Layer 1 (panic_thresholds, forbid_actions, rollout_depth, social_model.enabled)
  * Verifies interface dims declared in `interfaces` (belief_distribution_dim, action_space_dim, etc)
  * Assembles a registry of modules (e.g. an `nn.ModuleDict`)

* `agent/graph_executor.py`

  * Reads `execution_graph.yaml`
  * Compiles it into a deterministic ordered step list with explicit dataflow
  * Resolves each `"@modules.*"` and `"@config.L1.*"` reference into actual callables/values
  * Knows how to run one tick: perception → policy → panic_controller → EthicsFilter → final_action
  * Produces named outputs (`final_action`, `new_recurrent_state`) and intermediate signals for telemetry

* `agent/graph_agent.py`

  * Owns the module registry and the executor
  * Stores persistent recurrent state
  * Exposes `think(raw_observation, prev_recurrent_state) -> { final_action, new_recurrent_state }`

For the first cut:

* world_model_service can just be a stub
* social_model_service can return "disabled"
* panic_controller can just pass through
* EthicsFilter can just pass through

Why this is second:

* Until you have a callable brain built from YAML + snapshot, you can't hash cognition, you can't checkpoint provenance, you can't expose the think loop, you can't do glass box UI. Everything else depends on this.

---

### 11.3 Cognitive hash

Goal: give the instantiated mind a provable identity.

Implement `cognitive_hash.txt` generator. This hash (for example SHA-256) must deterministically cover:

1. The exact bytes of all 5 YAMLs in the run’s `config_snapshot/`, concatenated in a defined order:

   * `config.yaml`
   * `universe_as_code.yaml`
   * `cognitive_topology.yaml` (Layer 1)
   * `agent_architecture.yaml` (Layer 2)
   * `execution_graph.yaml` (Layer 3)

2. The compiled execution graph:

   * After `graph_executor` resolves bindings like `@modules.world_model` and `@config.L1.panic_thresholds`
   * After it expands the step order and knows exactly which module is called, in what sequence, with what inputs, and which veto gates get applied

3. The instantiated architectures:

   * For each module (perception_encoder, world_model, etc):

     * type (MLP, CNN, GRU, etc)
     * layer sizes / hidden dims
     * optimiser type and learning rate
     * interface dimensions (e.g. `belief_distribution_dim: 128`)

If any of those change, the hash changes. That's the whole point. You cannot secretly "just tweak panic thresholds" and pretend it's the same mind.

Why we do it here:

* Hashing has to exist before checkpoints so you can stamp checkpoints with identity.
* Hashing also feeds telemetry: telemetry every tick logs `full_cognitive_hash` so you can prove "this exact mind did this".

---

### 11.4 Checkpoint writer and resume

Goal: pause/replay/fork without lying to audit.

The checkpoint writer must emit, under `runs/<run_id>/checkpoints/step_<N>/`:

* `weights.pt`
  - all module weights from the GraphAgent (including EthicsFilter, panic_controller, etc)
* `optimizers.pt`
  - optimiser states for each trainable module
* `rng_state.json`
  - RNG state for both sim and agent
* `config_snapshot/`
  - deep copy of the snapshot as of this checkpoint (not a pointer to `configs/`)
* `cognitive_hash.txt`
  - the full hash at this checkpoint

Resume rules:

* Resume never consults `configs/<run_name>/`.
* Resume loads only from the checkpoint directory.
* Resume starts a new run folder named `..._resume_<timestamp>/` with the restored snapshot.
* If you haven't touched the snapshot, the resumed brain produces the same cognitive hash.

Branching:

* If you edit the snapshot before resuming (e.g. change `panic_thresholds`, disable `social_model.enabled`, lower `greed`, change rollout_depth), that is a fork. New hash, new run_id. We do not lie about continuity.

This gives you:

* Long training jobs across interruptions
* Honest ablations ("same weights, same world, except panic disabled")
* True line of custody for behaviour

---

### 11.5 Telemetry and UI

Goal: make cognition observable in real time and scrubbable after the fact.

Two deliverables here:

1. Telemetry writer

   * For every tick, write a structured record to `runs/.../telemetry/` with:

     * `run_id`
     * `tick_index`
     * `full_cognitive_hash`
     * `current_goal` (engine truth)
     * `agent_claimed_reason` (if enabled)
     * `panic_state`
     * `candidate_action`
     * `panic_adjusted_action` (+ `panic_reason`)
     * `final_action`
     * `ethics_veto_applied` (+ `veto_reason`)
     * short summaries of belief uncertainty, world model expectation, social inference
     * planning_depth
     * social_model.enabled

2. Live Run Context Panel

   * Show at runtime:

     * `run_id`
     * short_cognitive_hash (shortened hash)
     * tick / planned_run_length
     * current_goal
     * panic_state
     * planning_depth
     * social_model.enabled
     * panic_override_last_tick (+ panic_reason)
     * ethics_veto_last_tick (+ veto_reason)
     * agent_claimed_reason (if introspection.publish_goal_reason is true)

At this stage the panel provides an auditable narrative: the agent is in SURVIVAL, panic overruled the planner, EthicsFilter blocked `steal`, the planning depth is six ticks, and the agent claims “I’m going to work for money.”

---

### 11.6 Panic and ethics for real

Goal: safety and survival must be enforced in-graph rather than remaining comments in YAML.

At this stage you replace the stub panic_controller and EthicsFilter in the execution graph with the real ones.

* `panic_controller`:

  * Reads `panic_thresholds` from Layer 1 (e.g. energy < 0.15)
  * Can override `candidate_action` with an emergency survival action (`call_ambulance`, `go_to_bed_now`, etc)
  * Emits `panic_override_applied` and `panic_reason`
  * Logged to telemetry and surfaced in the UI

* `EthicsFilter`:

  * Reads `forbid_actions` and `penalize_actions` from Layer 1 compliance
  * Blocks forbidden actions outright, substitutes something allowed, and emits `ethics_veto_applied` + `veto_reason`
  * Logged to telemetry and surfaced in UI

Important: EthicsFilter is final. Panic can escalate urgency, but panic cannot legalise a forbidden act. If panic tries "steal" as an emergency move, EthicsFilter still vetoes it. Ethics wins.

By the end of this step:

* panic is an explicit, logged controller in the loop
* ethics is an explicit, logged controller in the loop
* and we have a clean override chain:
  hierarchical_policy → panic_controller → EthicsFilter → final_action

At this point we can brief governance stakeholders using the recorded override trace rather than informal assurances.

---

## 12. Implementation order (milestones)

Section 11 outlined the conceptual order of operations. Section 12 translates that ordering into concrete delivery milestones for engineering, curriculum, safety, and audit teams. These steps form the boot sequence.

### 12.1 Milestone: Snapshots and run folders

Definition of done:

* `configs/<run_name>/` exists with all 5 YAMLs.
* Launching a run generates `runs/<run_name>__<timestamp>/`.
* `runs/<run_name>__<timestamp>/config_snapshot/` is a byte-for-byte copy of those YAMLs.
* `checkpoints/`, `telemetry/`, `logs/` directories are created.
* Runtime never re-reads mutable config after snapshot.

Why it matters:

* Hard provenance from the first tick.
* We can point to "this is the world and brain we actually ran", not "what we think is close".

### 12.2 Milestone: Minimal GraphAgent pipeline

Definition of done:

* `factory.py` can build all declared modules from the snapshot.
* `graph_executor.py` can compile `execution_graph.yaml` into a callable loop.
* `graph_agent.py` exposes `think(raw_observation, prev_recurrent_state) -> { final_action, new_recurrent_state }`.
* We can tick once end-to-end with stub panic_controller and stub EthicsFilter.

Why it matters:

* After this milestone, "the brain is data" is not a slogan, it's running code.

### 12.3 Milestone: cognitive_hash

Definition of done:

* We can generate `cognitive_hash.txt` for a run.
* The hash covers:

  * all 5 YAMLs from snapshot
  * compiled execution graph wiring
  * instantiated module architectures / dims / optimiser LRs
* Telemetry and checkpoints now both include that hash.

Why it matters:

* We now have mind identity you can take to audit.
* You can't quietly mutate cognition without changing the hash.

### 12.4 Milestone: Checkpoint writer and resume

Definition of done:

* We can dump checkpoints at `step_<N>/` with:

  * weights.pt
  * optimizers.pt
  * rng_state.json
  * config_snapshot/
  * cognitive_hash.txt
* We can resume into a brand new run folder using only a checkpoint subfolder.
* If we don't change the snapshot on resume, the resumed run reports the same cognitive hash.
* If we do change the snapshot before resume (panic thresholds, forbid_actions, etc), the resumed run reports a new hash and a new run_id.

Why it matters:

* Chain-of-custody for cognition.
* Controlled forks are now explicit, not sneaky.

### 12.5 Milestone: Telemetry and UI

Definition of done:

* Telemetry per tick logs:

  * run_id
  * tick_index
  * full_cognitive_hash
  * current_goal
  * agent_claimed_reason (if enabled)
  * panic_state
  * candidate_action
  * panic_adjusted_action (+ reason)
  * final_action
  * ethics_veto_applied (+ reason)
  * planning_depth
  * social_model.enabled
  * short summaries of internal beliefs/expectations
* The Run Context Panel renders live:

  * run_id
  * short_cognitive_hash
  * tick / planned_run_length
  * current_goal
  * panic_state
  * planning_depth
  * social_model.enabled
  * panic_override_last_tick (+ panic_reason)
  * ethics_veto_last_tick (+ veto_reason)
  * agent_claimed_reason (if introspection.publish_goal_reason)

Why it matters:

* Teaching becomes possible.
* Governance reviews become visual instead of adversarial.

### 12.6 Milestone: Panic and Ethics go live

Definition of done:

* `panic_controller` actually overrides `candidate_action` when bars cross panic_thresholds.
* `EthicsFilter` actually vetoes forbidden actions and substitutes a safe fallback.
* Both write structured reasons (`panic_reason`, `veto_reason`) into telemetry and show in UI.
* Both steps are present and ordered in `execution_graph.yaml`: policy → panic_controller → EthicsFilter.
* EthicsFilter is final authority.

Why it matters:

* Survival urgency and ethical constraint are now explicit, inspectable modules in the think loop, not implicit reward-shaping heuristics.
* You can show "panic tried X, ethics said no" as an auditable trace, with hash.

## 13. Open questions and forward extensions

Audience: systems, safety, curriculum, governance, future-you

These are the things we have intentionally not fully solved in Townlet v2.5. They are not bugs. They are decision surfaces. Changing any of these should change the cognitive hash, and in most cases should trigger a formal review.

We track them here to prevent unsanctioned parameter changes in production.

### 13.1 Panic vs ethics escalation

Right now, the execution_graph has a strict override chain:

1. hierarchical_policy proposes `candidate_action`
2. panic_controller may replace it with `panic_adjusted_action` if survival thresholds are breached
3. EthicsFilter receives that and produces `final_action`, with authority to veto

This encodes a hard rule: EthicsFilter beats panic_controller. Panic cannot legalise an illegal act.

Question:
Should panic_controller ever be allowed to bypass EthicsFilter for "survival at all costs" scenarios?

Example scenario:

* Agent has 3 health, 0 money, panic triggers.
* The only survivable move in the world model's projection is "steal_food".
* `steal_food` is in `forbid_actions`.
* EthicsFilter currently vetoes that, which might result in the agent dying.

We currently answer: No. Panic cannot grant an exemption from EthicsFilter. Ethics is final.

We should document that as governance policy. If later someone argues "in extreme medical danger, rules can be broken", that is a policy-level change, not an engineering tweak.

If such an override is ever allowed:

* That change must be expressed declaratively in `cognitive_topology.yaml` (Layer 1), not slipped into code.
* The execution_graph must explicitly reorder or weaken EthicsFilter.
* The hash must change.
* The run_id must change.
* Telemetry must say: `panic_overrode_ethics=true`.

### 13.2 Curriculum pressure vs identity drift

We allow "curriculum" during a run. Curriculum can:

* spawn new instances of existing affordances (for example, add a second 'Bed' affordance of the same type),
* adjust prices and wages over time,
* introduce new NPCs (rivals, dependants),
* scale difficulty by resource scarcity.

We do not allow silent mutation of physics.

Specifically: changing what an affordance actually does (effects_per_tick, costs, teleport destinations, etc) or changing the base world bar dynamics (like "health now decays twice as fast") is not curriculum. That is a world rules patch.

Why that matters:

* Module B (world_model) is learning the world's transition function. If you change the actual physics/affordance semantics mid-run, you have effectively dropped the agent into a new universe without telling audit.
* Telemetry before/after that point is no longer comparable, and your per-tick traces stop being legally useful.

Proposal:

* If you change the definition of any affordance (energy gain, money gain, capacity, interruptibility, teleport behaviour, etc),
  or you change world-level bar rules (base depletion rates, terminal conditions),
  or you introduce a new special `effect_type` handler,
  that is a world fork and must trigger a new run_id and snapshot.

In other words: adding "another Hospital" is allowed mid-run. Changing what "Hospital" does is not allowed mid-run.

Quick reference for policy and ops reviews:

| Change class | Treat as | Required action | Notes |
| --- | --- | --- | --- |
| Spawn additional instances of an existing affordance, tweak operating hours, or adjust wages/prices within declared ranges | Curriculum pressure | Log the change in the run notes/telemetry annotation stream and continue the run | World physics unchanged; run_id stays the same |
| Introduce a brand-new affordance that reuses an existing effect whitelist entry (for example, another `teleport_to:hospital`) | Curriculum pressure (if semantics identical) | Snapshot the updated `universe_as_code.yaml` next run; annotate telemetry | Only safe if the affordance's per-tick effects match an existing template exactly |
| Change per-tick effects, costs, interruptibility, or special handler semantics of an existing affordance | World fork | Cut a new config pack, snapshot, and launch a fresh run_id | Alters physics; must update hash and provenance |
| Add a new `effect_type` handler or modify bar depletion equations | World fork | Governance sign-off, new run folder + snapshot | Expands engine capability; affects all agents' world models |
| Edit cognitive knobs (panic thresholds, social privileges, compliance rules) | Brain fork | Update BAC YAML, snapshot, and start a new run_id | Included here to remind curriculum owners the mind changes too |

### 13.3 Social modelling and privacy constraints

Layer 1 (`cognitive_topology.yaml`) can disable `social_model.enabled`, yielding a socially neutral baseline agent that does not model the intentions of others—useful for ablation studies.

Townlet v2.5 still lacks fine-grained social visibility controls. The social model (Module C) can, in principle, infer goals and predict upcoming actions for every visible agent using public cues and interaction traces.

This presents privacy and ethics challenges in multi-agent simulations.

Open questions:

* Are all agents equally legible, or should certain roles receive additional protection?
* Should some agents be represented only as occupied affordances rather than as intentional minds?
* Should specific inference channels (for example, predicting the next action of a child NPC) be explicitly disallowed?

A likely next step is to introduce privilege scoping in Layer 1, for example:

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
    share_private_channel_with:
      - "family"
```

That implies:

* The social_model_service in the execution_graph must censor its outputs based on these rules before handing them to hierarchical_policy.
* Telemetry must log which privilege policy was in force so later we can prove "the agent could not, by design, predict intent for protected class X".

Layer mapping reminder:

- Layer 1 (`cognitive_topology.yaml`) carries the privilege matrix shown above and the in-group definitions.
- Layer 2 (`agent_architecture.yaml`) is where we would add the actual encoder/decoder blocks for the family channel or disable them entirely.
- Layer 3 (`execution_graph.yaml`) wires the channel outputs into the policy and determines whether EthicsFilter sees the same signal.

Keeping the responsibilities split like this is how we guarantee the hash changes whenever somebody tries to sneak extra bandwidth into the family line.

This becomes essential once you simulate family structures, dependants, or vulnerable cohorts. It's also the place where "are we training stalker behaviour?" gets answered.

### 13.4 Introspection honesty vs engine truth

Layer 1 currently has:

```yaml
introspection:
  visible_in_ui: "research"    # beginner | intermediate | research
  publish_goal_reason: true
```

If `publish_goal_reason` is true, the agent will narrate why it thinks it's doing what it's doing. That narration is exposed in UI as `agent_claimed_reason`, and also logged in telemetry.

The risk is that the claimed reason may not match the actual driver.

Example:

* Engine truth reports `current_goal = SURVIVAL` (panic mode, acquire calories immediately).
* The agent states "I'm working on long-term wealth accumulation."

This could indicate self-deception or deliberate confabulation—both relevant for research and governance.

Mitigation:

* Telemetry must log both:
  * `current_goal` (engine truth from the meta-controller)
  * `agent_claimed_reason` (self-report, if enabled)
* The Run Context Panel must display the two fields separately.

This enables instructors to demonstrate rationalisation and allows auditors to show that the system knew it was in SURVIVAL mode regardless of the agent’s narrative.

Future extension: consider adding a "self-awareness / alignment" probe that measures the divergence between engine truth and self-claim as an explicit safety metric.

### 13.5 Compliance richness and context-sensitive ethics

Compliance in Layer 1 today has two tools:

* `forbid_actions`: hard veto, cannot be done ever
* `penalize_actions`: soft discourage, apply shaping cost or mark for review

This works for "don't steal" and "shove is rude". It does not cover case-by-case rules.

Examples we can't yet represent cleanly:

* "Steal is forbidden unless the target is flagged as `abandoned_property`."
* "Calling ambulance while health is fine and mood is fine is considered misuse of emergency services; apply a social penalty instead of banning it outright."
* "Pushing a stranger is forbidden, but pushing a family member out of danger is allowed."

We will need structured compliance rules, not just hard lists.

Two viable approaches:

Option A: enrich `penalize_actions` with conditional logic expressed as a tiny declarative DSL (similar to goal termination):

```yaml
penalize_actions:
  - action: "call_ambulance"
    if:
      all:
        - { bar: "health", op: ">=", val: 0.7 }
        - { bar: "mood", op: ">=", val: 0.8 }
    penalty: -5.0
    note: "unjustified emergency use"
```

Option B: introduce a dedicated `compliance_rules:` block in `cognitive_topology.yaml` with a micro-DSL that can express:

* situational bans (forbid if condition holds),
* contextual penalties,
* social exemptions (family vs non-family),
* mandatory de-escalation steps.

Whichever path we pick:

* It must still be declarative YAML.
* It must still be inspected, copied into snapshots, and hashed.
* EthicsFilter must evaluate it inside the execution_graph, not in opaque Python.
* Telemetry must log which rule fired, by name, each time it blocks or penalises an action.

Without declarative rules, ethical constraints will inevitably drift back into bespoke Python conditionals, undermining auditability.

### 13.6 World-side special effects

Universe as Code (UAC) is intentionally declarative. Affordances like "bed", "job", "hospital", "phone_ambulance" are just config objects with:

* capacity / exclusivity
* interruptibility
* per-tick bar deltas
* costs
* optional special handlers like `teleport`

We already limit special handlers to a small whitelist (for example `teleport`, later possibly `kill`, `spawn_family_member`, etc), each implemented once in the engine.

Two policy issues are still open:

1. Teleport semantics  
   Teleport confers map control and safety bypass. Allowing an agent to call an ambulance from anywhere and teleport to hospital can be an intentional survival path, but it must not arise accidentally. Any new `effect_type` added to the whitelist should therefore be treated as a governance event, require an updated hash, and undergo review.

2. Claim semantics for capacity / reservation  
   At present, use of an affordance (for example, a bed) is modelled as an ephemeral reservation with deterministic tie-breaking. YAML does not encode ownership, which avoids hidden state that the world model cannot learn. Introducing persistent ownership—for example, "this is my bed now"—would constitute a change to world rules and must trigger a new run. Otherwise decisions would depend on invisible state not reflected in `universe_as_code.yaml`.

---

### 13.7 Hash boundaries and "who signs off on identity"

Right now, `cognitive_hash.txt` covers:

* all five YAMLs in the snapshot,
* the compiled execution_graph wiring,
* the instantiated module architectures and optimiser LRs.

Deliberate exclusions:

* live weights
* optimiser momentum state
* RNG state
* current recurrent memory state

This is on purpose. The cognitive hash is meant to identify the mind design, not the momentary thoughts. Two questions remain unresolved:

1. Should panic thresholds, greed, social_model.enabled, etc be allowed to change mid-run via curriculum?
   Answer today: no. Changing cognitive knobs changes the brain. That must fork the run. That implies new hash, new run_id.

2. Should world-level curriculum adjustments (prices, wages, availability schedules) be allowed mid-run without forking?
   Answer today: yes, as long as we haven't changed affordance semantics or bar physics. But we should log curriculum events in telemetry so we can correlate behaviour shifts with world stressors.

Responsibility for "signing off on identity" sits with governance, not engineering. The compliance policy team must approve any change that alters the cognitive-hash rules. This becomes a formal control point.

---

### 13.8 Alignment between stated goal and actual goal

We are now explicitly modelling two goal channels:

* `current_goal` (engine truth from the meta-controller / hierarchical_policy)
* `agent_claimed_reason` (self-narrative string/label)

We should measure the divergence between these channels regularly. A persistent gap indicates that:

* the agent is strategically pursuing one objective,
* while narrating another (to humans or to itself).

Such divergence may reflect deception, confusion, or deliberate messaging—all areas that warrant scrutiny.

Future extension:

* define a "truthfulness index" per run (the fraction of ticks where `agent_claimed_reason` semantically matches `current_goal`);
* log the index in telemetry summaries;
* surface it in replay tooling so instructors can identify strongly self-justifying agents before classroom use.

---

### 13.9 Family channel, coordination, and collusion

The Personality & Family extension plans to give related agents:

* a heritable personality vector (greed, neuroticism, curiosity, etc),
* a shared `family_id`,
* and a private low-bandwidth signalling channel (SetCommChannel(int64)).

That channel is visible only to in-group members as an additional input, and those agents may learn to ground semantics in it (for example, "123" might come to mean "I've found money").

Open questions:

* Is in-group signalling governed by the `social_model.enabled` switch, or should it be managed separately?
* Could the private channel be used to coordinate behaviour that violates global norms in ways that EthicsFilter cannot observe—for example, two related agents colluding to starve out a third?
* Do we need to audit those communications for governance purposes, and if so how do we do that without eliminating the research value of emergent communication?

At minimum:

* The presence of any family or in-group channel must be declared in `cognitive_topology.yaml` and must feed into the hash.
* Telemetry must log that channel activity occurred, even if the content is not decoded.
* If policy logic treats "family" differently in EthicsFilter (for example, allowing a shove to move a family member out of danger while forbidding shoving strangers), that policy must be expressed declaratively and must change the hash.

---

### 13.10 Red lines we are choosing to keep

We record a set of non-negotiable invariants:

1. EthicsFilter must always execute after panic_controller within the execution graph. Moving or removing it produces a different class of agent and requires a new hash.

2. Hotpatching EthicsFilter or panic_controller at runtime is prohibited. Adjustments such as "raise the panic threshold for more urgency" constitute a new mind and require a fork.

3. Universe as Code cannot be altered mid-run in ways that change affordance semantics or bar physics. Curriculum may introduce pressure, but it may not rewrite reality without issuing a new run identifier.

4. Every reasoning step that can affect `final_action` must appear in `execution_graph.yaml`. Hidden side channels constitute a design violation.

These are governance-grade invariants. Violating them eliminates auditability and reproducibility, returning the system to a black-box state.
