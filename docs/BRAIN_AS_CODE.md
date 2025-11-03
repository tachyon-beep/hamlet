# Townlet v2.5: Brain as Code

Document version: 1.1
Date: 3 November 2025
Status: Approved for Implementation
Owner: Principal Technical Advisor (AI)

---

## 1. Executive Summary

We are moving from "the agent is a neural net we hope behaves" to "the agent is an explicitly described mind we can inspect, diff, and govern".

Townlet v2.5 replaces the old hardwired `RecurrentSpatialQNetwork` with a Software Defined Agent (SDA). Instead of one opaque RL policy baked into Python, we now build the agent's mind at runtime from declarative config. We call this Brain as Code (BAC).

Brain as Code does three big things.

### 1.1 Brains are now assembled, not baked in

In the old model, you got one big lump of policy: perception, memory, decision-making, all fused into a single recurrent Q-network. If you wanted to know why the agent sprinted to the hospital and then fell asleep in the shower, you basically shrugged and said "weights did that".

In Brain as Code, the "brain" is a set of cognitive modules wired together. Those modules and their relationships are defined in configuration, not handwritten into a class.

Examples of modules you can turn on or off (or swap out):

* Perception (how the agent turns world state into internal features)
* World model / situational model (what it thinks the environment is doing over time)
* Social model (what it thinks other agents or NPCs are doing)
* Survival layer / panic layer (what happens when the body is in crisis)
* Task/goal planner
* Action policy head (what it will actually do next tick)

Each of those modules can be parameterised. You can specify "this buffer is a GRU with hidden size 128" or "this planner runs a two-step lookahead" in config. You can also specify how these modules talk to one another: who gets to read which signals, in which order, under which conditions.

So, structurally:

* The brain is a graph.
* The nodes are cognitive functions.
* The edges are data flow between those functions.
* The whole thing is defined in YAML and built at runtime.

That’s what we mean by Software Defined Agent.

This gives us surgical control:

* You can add a PanicGate around behaviour when energy < 0.15 without rewriting the entire policy head.
* You can replace the short-term planner without retraining the perception stack.
* You can ship a "socially aware" variant of the same agent into a multiplayer sim, and keep a "solo survivalist" variant for curriculum training, while preserving almost all of the rest of the brain.

The end of "one RL blob to rule them all". You get parts, and you get wiring.

### 1.2 Every run has provenance

We treat each training run or simulation run as a real artefact. That run knows exactly:

* which world (Universe as Code pack) was loaded,
* which brain (Brain as Code stack) was instantiated,
* what safety / ethics / veto rules were active,
* what curriculum and hyperparameters were in force.

We don't just record "episode_0042 was successful". We record "episode_0042 was successful in World 'austerity_nightshift_v3' with Brain 'panic_enabled_social_blind', under Safety 'medical_floor_and_financial_floor', running curriculum step 3".

That provenance is not optional. It's core to the format.

Why this matters:

* Reproducibility: You can rehydrate a brain/world pair and get the same behaviour envelope again.
* Policy / audit: If an agent does something sketchy, we can point to which cognitive stack allowed it, rather than blaming "the AI".
* Teaching: Students can literally open the run folder and read how the mind was built and what pressures it was under.

The combination of the snapshot + cognitive hash is our audit boundary. If behaviour occurs in a run, we can prove which exact mind, under which exact rules, produced it. There is no 'the AI just did that'.

Basically: the agent is no longer "a mysterious creature in the sim". It's "Agent Instance #A47, running BrainConfig X in WorldConfig Y, observed at Time T".

### 1.3 Checkpoints now include identity, not just weights

Checkpoints are no longer just tensors. A checkpoint now carries:

* The live weights at that point.
* The full cognitive config that produced those weights.
* A hash we can use to prove "this brain is exactly that brain".

So when you reload a checkpoint, you're not just restoring performance. You're restoring the mind design: module graph, panic thresholds, veto rules, everything.

That means we can do forensics:

* "This behaviour only shows up in brains where the social planner is disabled but the money-hoarding heuristic is boosted."
* "This exploit emerges only in versions where panic overrides ethics in low-health states."
* "This run cheated by calling WAIT forever because survival reward was mis-shaped."

We stop having mystery regressions because we stop having mystery brains.

### 1.4 Live telemetry shows the mind, not just the meters

In Universe as Code (the world spec), we can watch the body: energy, health, money, etc.

In Brain as Code, we also surface the mind:

* Which module is currently in control of action selection.
* Whether the agent is in panic / survival override.
* Which ethical veto, if any, just blocked an intended action.
* How the planner is weighting long-term gain vs short-term safety.

So the UI is not just "health: 0.22". It's "health: 0.22, survival controller engaged, ambulance-seeking heuristic armed".

Auditors (humans, not just other models) can now ask: "Did it make this choice because it's hungry, because it's depressed, because it's scared, or because the money-minimisation heuristic went feral?" and the system can actually answer.

We call this philosophy The Glass Box:

* Townlet 1.x era: black box. You stare at reward curves and vibe-check policy outputs.
* Version 2.5 of Townlet: glass box. You can open the folder, read the YAML, and point to the part of the mind that made a decision.

This supports teaching ("why did you do that?"), safety auditing ("will it self-harm to save money?"), and governance ("was that action within policy?").

### 1.5 Why this matters

Brain as Code lines up the governance story with the technical story.

* Interpretability
  We can explain behaviour at the level of "which module fired and why".

* Reproducibility
  Behaviour is not a rumour. It's a config + checkpoint we can rerun.

* Accountability
  When something goes wrong, we don't say "the AI decided". We say "the SafetyVeto subgraph was misconfigured to allow that action under these panic conditions". That's actionable and reviewable.

* Curriculum / pedagogy
  You can hand a student two brains and ask "simulate both in the same world, what differences do you observe?" and it's an actual experiment, not mysticism.

This is the point of Brain as Code: the mind is part of the content pipeline, not just a by-product of training.

---

## 2. The three cognitive layers

Brain as Code says: an agent’s mind is not mystical. It's three YAML files.

Those three files line up with three audiences:

* People who care what the agent is allowed to do.
* People who care how the agent is built.
* People who care how the agent actually thinks, step by step.

Together, these three layers are the SDA brain. Change the files, you change the mind.

### 2.1 Layer 1: cognitive_topology.yaml

Audience: policy, instructors, simulation designers
Nickname: the character sheet

Layer 1 is the public-facing definition of the agent as a character. This is the part you would show in a classroom, or put in front of an ethics reviewer, or attach to a safety case.

It answers questions like:

* Does this agent have social reasoning enabled?
* How paranoid is it?
* How greedy is it about money?
* When does it panic and override normal behaviour?
* What actions is it forbidden to take, full stop?

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

* `perception.enabled: true` tells us the agent is allowed to build an internal model of the world. If this were false, you'd get something more blind and reflexive, which is valid for ablations or "zombie baseline" training.

* `world_model.rollout_depth: 6` says the agent is literally allowed to imagine up to 6 ticks into the future before acting. This is now a dial. You can ship a short-horizon impulsive teenager or a long-horizon planner without touching code.

* `social_model.enabled: true` means it reasons about other entities as having minds/goals. Turn it off and you're simulating someone who does not model other people, which is sometimes exactly what you want.

* `personality` is where we admit we're doing psychology sliders. We are not pretending it's "just reward shaping". We say outright: this agent is greedy (0.7), curious (0.8), not very agreeable (0.3), and somewhat anxious (0.6). That will affect how it trades off money vs comfort vs safety.

* `panic_thresholds` is survival mode. Below these bars, the normal planner can be overridden by emergency behaviour. Panic is not a magic mystery anymore, it's parameterised in a file you can read.

* `compliance.forbid_actions` is the hard veto list. These are actions the agent is simply not allowed to take, ever, regardless of how desperate it is. This is where safety policy literally binds behaviour.

* `introspection.publish_goal_reason: true` means the agent will tell you (the human observer) why it thinks it's doing what it's doing. This is how we get "Glass Box" style auditability in UI. It's not just acting, it's narrating intent.

This is Layer 1’s job: describe what the mind is supposed to be like, in human terms.

This is the layer a policy person signs off on. This is the layer you show to governance. This is where you answer "what kind of entity did you just put in my sim?" in plain language.

### 2.2 Layer 2: agent_architecture.yaml

Audience: engineers, grad students
Nickname: the blueprint

Layer 2 is where we define what those Layer 1 faculties actually are under the hood.

If Layer 1 says "the agent has a world model that can imagine futures", Layer 2 answers "cool, that world model is a 256x256 MLP with these heads, trained on this dataset, using Adam at this learning rate".

So Layer 2 is concerned with:

* Network types (GRU, MLP, CNN, etc).
* Hidden sizes and layer widths.
* What each module outputs, and the dimensionality of those outputs.
* Which optimiser and what learning rate we are allowed to use.
* What pretraining task we ran to get the module started.

This matters for reproducibility and for teaching. You can literally show two brains that share the same Layer 1 behaviour spec, but have totally different internal architectures in Layer 2, and ask "which one learns faster?" That’s now a clean experiment instead of a pile of undocumented hacks.

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

* `interfaces` is a contract. It forces dimension discipline between modules so you can swap parts later. If the world model promises a 256-dim "imagined future" vector, the policy is allowed to expect 256 dims, and that expectation is scripted, not tribal knowledge.

* `optimizer` and `pretraining` are locked into config. That means if someone says "the agent only behaves like this because you secretly tweaked Adam's learning rate", we can prove or disprove that claim by inspection. That's governance. That's auditability.

* Each module here maps directly to something in Layer 1. So `world_model.enabled: true` in Layer 1 is now realised as "here is the architecture and training objective of world_model" in Layer 2. If you disable it in L1, you can still keep the blueprint around in L2 for future runs.

This is where the mind stops being fantasy and becomes engineering spec.

### 2.3 Layer 3: execution_graph.yaml

Audience: researchers, auditors, you at 2 am
Nickname: the glue / the think loop

Layer 3 is the part most RL systems pretend doesn't exist.

This is the actual cognition loop. Step by step. Who runs first, who hands what to whom, who gets to veto whom, and what comes out the other end as an action.

Layer 3 defines:

* The order modules are called every tick.
* Which outputs get passed forward as inputs to others.
* Where emergency logic (panic) and ethics/compliance veto kicks in.
* What final signal goes back into the environment.

This is basically the wiring diagram of thought.

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

What's happening here, in English:

1. The agent takes in the current world state (`raw_observation`) and its own memory of what just happened (`prev_recurrent_state`).

2. The perception module digests that into two things:

   * a cleaned-up internal belief about "what's going on right now",
   * an updated recurrent state to carry forward.

3. The hierarchical policy then plans an action. But it doesn't just look at perception. It also consults:

   * the world model service (what do I think will happen next if I do X),
   * the social model service (what do I think other agents are about to do).

   That means the agent isn't just reacting. It's simulating possible futures, at least within whatever rollout horizon you allowed in Layer 1.

4. That gives us a candidate_action. This is "what I want to do".

5. We then run panic_controller. If the agent is in crisis (for example energy below the threshold in panic_thresholds from Layer 1), panic_controller is allowed to override candidate_action with an emergency survival action like call_ambulance or go_to_bed_now. This override is logged along with the reason (panic_reason).

6. We then run EthicsFilter. This enforces the hard compliance rules from Layer 1 (forbid_actions, etc). EthicsFilter is final. Panic cannot legalise an illegal action. If the (possibly panic-adjusted) action is forbidden, EthicsFilter vetoes it, optionally substitutes something safe, and records veto_reason for the UI and telemetry.

7. The execution graph outputs:

* the final chosen action,
* the updated recurrent state for next tick.

Why Layer 3 matters:

* This is where "panic overrides normal planning" will eventually live. You can literally splice in a PanicController step before the EthicsFilter, and tell it "if energy < panic_threshold.energy, you are now allowed to sprint to bed or call ambulance regardless of normal goal structure".
* Panic can override planning for survival, but cannot authorise a forbidden act. The EthicsFilter is final authority on behaviour.
* This is where you can insert auditing. You can say "log whatever the policy wanted to do before the EthicsFilter filtered it".
* This is how we prove cause and effect. If someone asks "why did it call an ambulance instead of walking to the hospital", you can point to the graph: panic fired, survival path took control, ethics didn't veto, action dispatched.

This is the Glass Box in practice. Thought is a DAG, not a rumour.

## 3. Run bundles

So far we've talked about how we describe a brain.

This section is about how we package and track an actual run of that brain in an actual world, so that we can:

* reproduce it,
* audit it,
* and teach from it later.

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

Here's what each of those files means inside a bundle:

### 3.1 config.yaml

This is the simulation runtime envelope.

It answers practical questions like:

* How long does this run last? (`run_length_ticks`)
* How fast do we tick? (`tick_rate_hz`)
* How many agents are active? (`max_population`)
* What's the random seed?
* Are we training or just evaluating?
* What's the curriculum schedule (for example: "start solo, introduce rival agent at tick 10k")?

This is basically the lab environment settings.

### 3.2 universe_as_code.yaml

This is the Universe as Code file for this run. It defines:

* the meters/bars (energy, health, money, mood, etc),
* what kills you (terminal conditions),
* the affordances (Bed, Job, Hospital, Bar, etc),
* opening hours,
* costs,
* per-tick effects,
* map layout,
* time-of-day mechanics,
* social cues if applicable.

This is the body, the town, the physics, the economy. It's the world your agent is dropped into.

Important: panic logic does not live here. World defines the problem. Brain defines how the agent chooses to survive it.

#### 3.3 cognitive_topology.yaml

This is Layer 1 (the character sheet) for this specific run.

This is the behavioural intent: which faculties are active, how deep it's allowed to plan, what it values, what it fears, what it must never do.

This is what you show to reviewers to say "this is not a violent agent" or "this agent will prioritise money if mood is low".

#### 3.4 agent_architecture.yaml

This is Layer 2 (the blueprint).

This tells us what those faculties are made of internally: CNN vs GRU, hidden sizes, learning rates, pretraining datasets. It's how we make sure we can rebuild the same brain later, and prove we didn't quietly swap in some other network.

#### 3.5 execution_graph.yaml

This is Layer 3 (the think loop).

This is the tick-by-tick wiring of cognition. Who runs first, who gets to veto, and what finally hits the action bus that controls the body.

This is the layer you show when someone on governance asks "under what exact conditions is it allowed to break from long-term planning and just survive?"

### 3.6 Launching a run

When you launch a run using a bundle like `L99_Inventing_ASI`, the system doesn't just start simulating. It first locks the configuration in time.

The launcher creates a new folder under `runs/` that looks like this:

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

Important things to notice:

1. `config_snapshot/` is a copy, not a pointer.

   The entire bundle is snapshotted. That snapshot is now the legal definition of what that run actually was. If you later edit `configs/L99_Inventing_ASI/`, that does not rewrite history. The run is historically sealed.

2. `checkpoints/` will hold the saved brain weights over time, plus metadata like the cognitive hash so you can prove that checkpoint belongs to that exact configuration and not some other Franken-brain.

3. `telemetry/` is where we store time series: which module was active, what actions were vetoed, panic events, etc. This is how we audit after the fact.

4. `logs/` is the boring but essential console/state logging for debugging and replay.

Why this matters:

* Reproducibility
  You can take that run folder, rehydrate it on another machine, and get the same brain in the same world under the same rules. This kills "it worked on my laptop" and replaces it with "it worked in run L99_Inventing_ASI__2025-11-03-12-14-22, here's the folder, test it yourself".

* Accountability
  If an agent does something unsafe, you don't have to argue philosophy. You pull the snapshot. You inspect `cognitive_topology.yaml` to see whether it was allowed. You inspect `execution_graph.yaml` to see whether EthicsFilter was properly in the loop. If it was allowed, that's on design, not on the model. We can fix design.

* Curriculum science
  You can compare two runs and ask "what changed?" and the answer is not vague. It's diffable YAML plus diffable checkpoints.

That’s the whole point of run bundles: every agent is not just a brain in a box. It's an artefact with chain of custody.

---

## 4. Checkpoints

A checkpoint is not just "model weights at step 500". A checkpoint is a frozen moment of mind and context.

Every checkpoint in a run must include all the ingredients required to:

* resume training honestly,
* audit the agent honestly,
* and prove what cognitive configuration produced a given behaviour.

That means each checkpoint directory contains:

* The live neural state

  * module weights (all nn.Modules in the SDA brain: perception, world model, social model, policy, EthicsFilter, etc)
  * optimiser states for each trainable module
  * any recurrent / GRU / memory state you need to carry forward

* The sim random number generators at that instant

  * RNG state(s)

* The rules the brain believed governed the world and itself at that instant

  * a full config snapshot (the same 5 YAMLs you launched with)

* A cryptographic identity for that specific mind

  * a cognitive hash

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

The intent here is governance-grade provenance. At step_000500 we can answer, with evidence:

* what world the agent thought it lived in,
* which modules existed in its head,
* which rules constrained its behaviour,
* and which exact tensors represented its "brain state".

This is what separates "we saw something weird" from "we can prove what mind did the weird thing".

### 4.1 Cognitive hash

`cognitive_hash.txt` is the agent’s true identity.

It’s a deterministic hash (for example SHA-256, but any collision-resistant digest is fine) computed over:

1. The five config_snapshot YAML files concatenated in a stable order:

   * config.yaml
   * universe_as_code.yaml
   * cognitive_topology.yaml (Layer 1)
   * agent_architecture.yaml (Layer 2)
   * execution_graph.yaml (Layer 3)

2. The resolved execution graph after compilation.
   Not just the YAML text, but the actual ordered steps and wiring after we’ve expanded all `@modules.*` and `@graph.*` references. This matters because two YAMLs that look similar can resolve differently once services are bound. We hash what actually runs.

3. The instantiated architectures.
   Specifically: layer types, hidden dims, interface dimensions, etc, exactly as built by the factory for this run. That means if someone silently swaps GRU → LSTM in `agent_architecture.yaml`, the hash changes. Good.

Taken together, that hash defines "this exact mind, in this exact configuration, executing this exact loop". Any mutation to those inputs changes the cognitive hash. That means there is no such thing as 'hotpatch the brain in production but keep the same identity'.

Why this matters:

* Telemetry can label any behaviour with `full_cognitive_hash`, meaning "this is the brain that chose this".
* You can stop a run, sleep, resume next day, and prove it’s still the same mind (same hash).
* You can run A/B experiments in the same world, and prove they were actually two different minds and not two different random seeds of the same mind.

You can think of it like a brain fingerprint. If you don't change the mind, the hash doesn't change. The moment you change cognition (architecture, wiring, rules), the hash changes.

---

## 5. Resume semantics

Resuming from a checkpoint is not "reload some weights and hope for the best". Resume is part of our audit story.

When you resume from `step_000500`, you must treat that checkpoint as law. You do not go back to the editable, potentially drifted configs in `configs/.../`.

Resume rules:

1. Restore `weights.pt`, `optimizers.pt`, and `rng_state.json`.
   That gives you the same brain, same optimiser momentum, same randomness.

2. Restore the `config_snapshot/` that lives inside that checkpoint.
   That snapshot is the contract of the world, the brain topology, and the execution graph at that moment.

3. Do not reread anything from `configs/L99_Inventing_ASI/` at resume time.
   Whatever's in `configs/` now might have changed since the original run launched. We ignore it on purpose.

When you resume, you're allowed (and expected) to write into a new run directory, for example:

```text
runs/
  L99_Inventing_ASI__2025-11-03-12-14-22_resume_2025-11-03-13-40-09/
    config_snapshot/        # copied from checkpoint, not from configs/
    checkpoints/
    telemetry/
    logs/
```

Two important consequences:

* Lineage
  The resumed run is still considered to be the same "mind identity" as long as the cognitive hash doesn't change. So if you stop at tick 500 and restart and run to tick 2000, you can assert continuity in an audit: "this was one continuous mind, not a different agent pretending to be the same".

* Branching
  If you want to do an ablation, like "what if we lower greed in Layer 1", you explicitly edit the snapshot before resume. That changes the cognitive hash. That becomes a fork. Now you're running a sibling mind, not the same mind.

So resume gives you:

* long trainings in chunks (survive GPU pre-emption)
* controlled forks for experiments ("same brain except X")
* airtight lineage for safety reviews

If you change any cognitive setting in the snapshot (e.g. greed, panic_thresholds, forbid_actions, rollout_depth), that's a new mind and must produce a new full_cognitive_hash and a new run_id. You are not 'resuming', you're forking.

---

## 6. Runtime engine components

Under the hood, the classic "big RL agent class" is gone. We replace it with: factories, a graph agent, and an execution engine.

This gives us two wins:

1. We can build any brain that matches the YAML spec without changing code.
2. We can prove that the thing we built is the thing we ran.

### 6.1 agent/factory.py

This is the constructor of the mind.

Inputs:

* the run's frozen `config_snapshot/`

  * cognitive_topology.yaml (Layer 1: behaviour contract)
  * agent_architecture.yaml (Layer 2: module blueprints)
  * execution_graph.yaml (Layer 3: wiring order)
  * universe_as_code.yaml (for observation/action space alignment)
  * config.yaml (runtime envelope)

Responsibilities:

* Build each nn.Module exactly as described in `agent_architecture.yaml`
  (perception_encoder, world_model, social_model, hierarchical_policy, EthicsFilter, etc).
* Verify that interface dimensions match what `interfaces` promised.
  No silent broadcasting, no mystery reshapes.
* Package all of that into a `GraphAgent` instance.

So `factory.py` is basically: "From these YAMLs, instantiate the brain the YAMLs describe."

This is where the cognitive hash gets finalised, because this is where abstract config turns into actual module graphs.

### 6.2 agent/graph_agent.py

`GraphAgent` is the living brain object that replaces the old `RecurrentSpatialQNetwork`.

It's a generic nn.Module that:

* Owns all submodules in a registry, e.g. an `nn.ModuleDict` keyed by name (`perception_encoder`, `world_model`, etc).
* Holds whatever recurrent / memory state is persistent across ticks.
* Holds a `GraphExecutor` that knows how to actually run cognition for one tick.

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

* `execution_graph.yaml` from Layer 3
* the module registry built by the factory

On init:

* It parses the execution graph YAML and "compiles" it into an ordered list of steps with explicitly resolved inputs and outputs.
* It resolves service bindings like `"@modules.world_model"` into actual module handles.
* It validates that all required data dependencies exist.

This compiled form (the ordered think loop with concrete wiring) is part of the cognitive hash. That means two graphs that look similar but schedule steps differently will hash differently, which is exactly what we want.

On run():

* It creates a scratchpad / data_cache.
* It executes each step in order:

* run perception
* unpack belief and new recurrent state
* run hierarchical policy with world/social context
* get candidate action
* run panic_controller
* run EthicsFilter
* emit final_action and new_recurrent_state

* It emits whatever the graph declared as outputs (final action and updated recurrent state, at minimum).

So `graph_executor.py` is basically the microkernel of thought. It makes the agent's reasoning loop explicit, deterministic, and inspectable.

### 6.4 EthicsFilter module

This is the "seatbelt".

EthicsFilter is a module in the registry, just like perception or policy, except its job is to enforce the behavioural contract in Layer 1.

Inputs:

* candidate action from policy
* panic-adjusted action from panic_controller

Outputs:

* possibly substituted `final_action`
* `veto_reason` (why we overruled policy, if we did)

Because this happens inside the execution graph, we can log it every tick. That gives you a literal audit trail:

"Tick 842: agent tried to STEAL. EthicsFilter vetoed. Reason: compliance.forbid_actions."

This is how we move from "trust us, it's safe" to "here is a line in telemetry where it tried to steal and got blocked".

---

## 7. Telemetry and UI surfacing

Transparency isn't an afterthought. We bake it into runtime.

We expose two levels of visibility: live context for humans watching the sim, and per-tick trace logs for later replay/teaching/forensics.

### 7.1 Run Context Panel (live UI)

At runtime, the UI shows a compact panel for the currently observed agent. That panel must include:

* `run_id`
  For example: `L99_Inventing_ASI__2025-11-03-12-14-22`

* `short_cognitive_hash` (prettified form of the full_cognitive_hash suitable for UX, e.g. first 8 chars)
  So you always know which mind you’re looking at.

* `tick`
  Current tick / planned_run_length_ticks (from config.yaml)

* `current_goal`
  Whatever the meta-controller (hierarchical_policy) says the agent is pursuing right now: SURVIVAL, THRIVING, SOCIAL, etc.

* `panic_state`
  Boolean derived from panic_thresholds in L1. This is "is the agent in emergency mode right now".

* `ethics_veto_last_tick`
  Whether the EthicsFilter blocked the chosen action last tick, plus the veto_reason.

* `planning_depth`
  From world_model.rollout_depth. This is literally "how far ahead it is allowed to think right now".

* `social_model.enabled`
  True/false, to show whether the agent is currently modelling other minds.

* `panic_override_last_tick`
  Whether panic_controller overruled the policy last tick, plus panic_reason.

* `agent_claimed_reason`
  If introspection.publish_goal_reason is true

That gives an instructor instant context. You can point at the panel and narrate:

"See here? The agent flipped into SURVIVAL mode because energy dropped under 15. It's legally allowed to plan 6 ticks ahead. Last tick it tried to 'steal', ethics vetoed it. So it's hungry, panicking, and morally leashed."

That is the Glass Box promise, visible in real time.

### 7.2 Per-tick trace logging (telemetry/)

In parallel, we write structured records into `runs/.../telemetry/`. One row per tick (or batched every N ticks if you want to throttle IO). Each record must include at minimum:

* `run_id`
* `tick_index`
* `full_cognitive_hash`
* `current_goal`
* `agent_claimed_reason` (if enabled)
* `panic_state`
* `planning_depth`
* `social_model.enabled`
* `candidate_action`
* `panic_adjusted_action` (+ `panic_override_applied`, `panic_reason`)
* `final_action`
* `ethics_veto_applied` (+ `veto_reason`)
* `belief_uncertainty_summary`
* `world_model_expectation_summary`
* `social_model_inference_summary`

Note the distinctions here:

* current_goal is what the meta-controller actually thinks it's pursuing (SURVIVAL, THRIVING, SOCIAL).
* agent_claimed_reason is what the agent says it's doing and why, if introspection.publish_goal_reason is enabled. We log both on purpose, because they can diverge.
* panic_adjusted_action is the action after panic_controller but before EthicsFilter. final_action is after EthicsFilter.

This log is how we answer after the fact:

* Why did the agent starve next to a fridge?
* Was it panicking or just lazy?
* Did it try to break a rule and get stopped?
* Did it misread another agent's intent?

It's also how we build teaching material. You can pull a real trace and say "here's the tick-by-tick reasoning chain of a desperate agent at 2am with no money and 3 health left". That lands way harder than theory.

---

## 8. Declarative goals and termination conditions

The agent doesn't just flail around. The high-level policy (meta-controller in the hierarchical policy module) picks a Goal like SURVIVAL or THRIVING.

A Goal needs a termination condition: "I'm done, pick a new goal".

We want those conditions to be inspectable and replayable, and we don't want random inline Python lambdas in config, because that kills reproducibility and audit.

So we define goals in YAML using a tiny, safe DSL.

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

All bar comparisons use normalised [0.0, 1.0] values from `universe_as_code.yaml`. For money, 1.0 is $100 by default.

How this works:

* Each goal has an `id`.
* Each goal declares a `termination` rule.
* That rule is a tree made of `all` / `any` conditions.
* Each leaf is a simple comparator against either:

  * a bar (like energy, money, health),
  * or a runtime metric (like time_elapsed_ticks).

At runtime, the engine evaluates these rules with a tiny interpreter:

* If `all` is true, we consider that goal satisfied.
* If `any` is true, we consider that goal satisfied.

Benefits:

* You can read goals and understand them without touching code.
* You can replay old runs with the same logic, because it's data, not Python.
* You can diff behavioural incentives across experiments (for example, "we used to consider SURVIVAL complete at energy ≥ 60 and now we require ≥ 80").

This also gives us a nice hook for curriculum:

* Early curriculum might say SURVIVAL completes at energy ≥ 50.
* Later curriculum might tighten that to 80.
* That's now a YAML diff we can justify in a report, not a magic number hidden in a trainer script.

---

## 9. Affordance semantics in universe_as_code.yaml

Everything the agent can do in the world is an affordance in `universe_as_code.yaml`. This includes sleeping in a bed, eating a meal, going to work, calling an ambulance, going to the gym, etc.

We treat affordances as declarative objects:

* what they cost,
* what they do to your bars each tick,
* whether you can be interrupted,
* whether they're exclusive,
* whether they can teleport you.

Key fields we support:

* `capacity`
  How many agents can use this at once.

* `exclusive`
  true/false. If true, it's basically "you sit in the chair, nobody else can until you're done".

* `interruptible`
  Can you walk off mid-way and keep some benefit, or is it all-or-nothing?

* `distance_limit`
  For interactions that don't require you to be exactly on the tile. Useful for things like "call ambulance from anywhere in the house".

* `effects_per_tick`
  A list of bar deltas applied each tick of interaction. For example: `energy +0.25`, `money +0.10`, `mood -0.02`.

* `costs` / `costs_per_tick`
  Bars to deduct up front or per tick, usually money or energy.

* Special `effect_type` blocks like `teleport`
  This is how we model things like ambulance relocation without hardcoding a new method in the agent. We keep a whitelist so we don't accidentally let YAML say godlike stuff like `nuke_city`.

Engine behaviour:

* The engine arbitrates contention (capacity, exclusivity) in a deterministic way.
* The engine applies effects atomically each tick.
* The engine does not persist arbitrary state inside the affordance. Occupancy is treated as ephemeral: who's using this thing this tick. This keeps the world simpler for the world model to learn.

Important split of responsibilities:

* The world decides "here is the affordance 'Hospital', it costs money, and it heals health per tick".
* The brain decides "am I panicking enough to sprint to Hospital vs tough it out in Bed".

This makes world design and brain design cleanly separable and, crucially, reviewable by different people.

---

## 10. Success criteria

We judge success on three axes: technical, teaching, and governance. All three matter.

### Technical success

[ ] We can launch a run from `configs/<run_name>/` and automatically create `runs/<run_name>__<timestamp>/` with a frozen `config_snapshot/`.

[ ] `agent/graph_agent.py` can be fully built from that snapshot alone (through `agent/factory.py`) and can tick.

[ ] Each checkpoint includes:

* weights
* optimiser state
* RNG state
* a nested `config_snapshot/`
* `full_cognitive_hash.txt`

[ ] Resume uses only the checkpoint snapshot and reproduces the same `full_cognitive_hash` (unless deliberately changed).

[ ] Telemetry logs include `run_id` and `cognitive_hash` so you can tie behaviour to mind identity.

[ ] UI surfaces run context (panic, veto, etc) live.

### Pedagogical success

[ ] A beginner can answer "Why didn't it steal the food?" just by:

* looking at the UI panel (veto_reason),
* looking at `cognitive_topology.yaml` (forbid_actions).

[ ] An intermediate student can change `agent_architecture.yaml` (for example GRU → LSTM) and rerun, and then actually observe the memory/behaviour consequences without touching core code.

[ ] A researcher can edit `execution_graph.yaml` to bypass `world_model_service` and watch the agent become more impulsive. That experiment is now "change this line in YAML", not "pull apart 4k lines of torch spaghetti".

[ ] We can take any interesting emergent behaviour clip, grab its run folder, and show precisely:

* which mind (hash),
* which world rules,
* which ethics rules,
* which panic thresholds,
were in play at that moment.

### Governance success

[ ] We can prove to an auditor that, at tick T in run R, the agent had `forbid_actions: ["attack","steal"]` and that `EthicsFilter` did in fact run in the execution graph.

[ ] We can replay that agent exactly as it existed at tick T using only the checkpoint directory, with no "trust me" developer hand waving.

If we hit all of those, then "Brain as Code" is not just a slogan. It's an accountable system.

---

## 11. Implementation notes (ordering)

This is the recommended order to actually stand this up without losing your mind.

1. Folder discipline first

   * Create `configs/<run_name>/` with all 5 YAMLs:

     * config.yaml
     * universe_as_code.yaml
     * cognitive_topology.yaml
     * agent_architecture.yaml
     * execution_graph.yaml

   * On launch, auto-generate:

     * `runs/<run_name>__<timestamp>/config_snapshot/`
       which is a deep copy, not a symlink.

   This locks down provenance from day one.

2. Snapshot copy on launch

   * No clever runtime symlinks for core config.
   * Symlinks are fine as a dev convenience pre-launch, but once you press go, we copy the actual bytes.

3. Build the core runtime stack:

   * `agent/factory.py`
   * `agent/graph_agent.py`
   * `agent/graph_executor.py`

   At this stage it's okay if social_model is a stub, or EthicsFilter just passes through. The point is to prove the shape:

   * load snapshot,
   * assemble modules,
   * run one think() tick.

4. Checkpoint writer

   * Able to dump:

     * model state dicts,
     * optimiser states,
     * RNG state,
     * config_snapshot/,
     * cognitive_hash.txt
   * This is where you define what "the mind" means in practice.

5. Resume path

   * Load from checkpoint only.
   * Spin up a new `runs/..._resume_<timestamp>/`.
   * Preserve or recompute the same cognitive hash unless you intentionally mutate the snapshot.

6. Telemetry format

   * Start logging per-tick (or per-n-ticks) JSON or CSV under `runs/.../telemetry/`.
   * Each record must include run_id and cognitive_hash.

7. UI Run Context panel

   * Show:

     * run_id
     * short cognitive_hash
     * tick / planned_run_length
     * current_goal
     * panic_state
     * ethics veto info (bool + reason)
     * planning_depth
     * social_model.enabled

   This is the feature that will sell the whole system to anyone watching. It's the bit where you can point and say:
   "That wasn't random. That was intent, filtered by ethics, under panic, with traceable identity."

---

## 12. Implementation order

Audience: engineering leads, curriculum/tooling, anyone who has to actually ship this without catching fire

This section is the boot sequence. If you follow this order, you get a working glass-box agent with provenance, panic, ethics, telemetry, and replay. If you jump around and half-implement bits, you’ll hate your life and the audit story won’t hold.

### 12.1 Snapshots and run folders first

Goal: lock down provenance on day one.

* Create the bundle layout under `configs/<run_name>/` with the 5 YAMLs:

  * `config.yaml`
  * `universe_as_code.yaml`
  * `cognitive_topology.yaml`
  * `agent_architecture.yaml`
  * `execution_graph.yaml`

* Write the launcher that, when you “start run”, immediately:

  * creates `runs/<run_name>__<timestamp>/`
  * copies all 5 YAMLs byte-for-byte into `runs/<run_name>__<timestamp>/config_snapshot/`
  * makes `checkpoints/`, `telemetry/`, `logs/` subdirs

Rules:

* This is a physical copy, not a symlink.
* After snapshot, the live run must never silently re-read from `configs/<run_name>/` at runtime. The snapshot is now truth.

Why this is first:

* Everything else (the agent, the EthicsFilter, telemetry, resume) leans on snapshot provenance.
* If you don’t freeze config at launch, you can’t make any governance claims later.

### 12.2 Minimal GraphAgent pipeline

Goal: replace the old monolithic RL class with GraphAgent that can tick once.

Deliverables:

* `agent/factory.py`

  * Reads the run’s `config_snapshot/`
  * Instantiates modules declared in Layer 2 (`agent_architecture.yaml`)
  * Verifies interface dimensions declared in `interfaces`
  * Injects Layer 1 knobs (panic_thresholds, forbid_actions, etc) into the appropriate runtime modules
  * Assembles an `nn.ModuleDict` containing the submodules

* `agent/graph_executor.py`

  * Reads Layer 3 (`execution_graph.yaml`)
  * Compiles it into an ordered list of steps with explicit wiring
  * Validates that every step’s inputs resolve to something that actually exists
  * Produces a callable `run_step(data_cache)` that executes the loop in order

* `agent/graph_agent.py`

  * Owns module registry + executor
  * Maintains recurrent state
  * Exposes `think(raw_observation, prev_recurrent_state) -> { final_action, new_recurrent_state }`
  * Calls executor to actually do perception → policy → panic_controller → EthicsFilter

For v0 of this pipeline you can stub:

* world_model service to just echo nonsense
* social_model service to return “disabled”
* panic_controller to just pass through
* EthicsFilter to just pass through

Why this is second:

* You need a callable brain before you can talk about panic escalation, veto, telemetry, etc.
* You also need this working before you can hash cognition.

### 12.3 Cognitive hash

Goal: give the brain a verifiable identity.

Implement `cognitive_hash.txt` generator. The hash (for example SHA-256 digest hex string) must deterministically encode:

1. The text of all 5 YAMLs in `config_snapshot/` concatenated in a stable order
   (config.yaml, universe_as_code.yaml, cognitive_topology.yaml, agent_architecture.yaml, execution_graph.yaml)

2. The compiled execution graph
   After `graph_executor` resolves all `@modules.*`, `@graph.*`, and service bindings and produces an ordered step list with concrete inputs and outputs.

3. The instantiated architectures
   For each module in `agent_architecture.yaml`, capture:

   * class/type (GRU, MLP, CNN)
   * layer sizes / hidden dims
   * optimiser type and LR
   * interface dimensions it exposes/consumes

That bundle becomes the canonical “mind fingerprint”.

Why we do it here:

* You want hashing before checkpoints, so you can stamp every checkpoint with the hash.
* Also, telemetry needs to be logging this hash from the very first tick of any run.

Note:

* If anything in (1), (2), or (3) changes, the hash changes.
  That is what lets you prove “same brain” or “forked brain” to an auditor.

### 12.4 Checkpoint writer and resume

Goal: honest pause/resume and experimental forking.

Checkpoint writer must output, at minimum, into `runs/<run_id>/checkpoints/step_<N>/`:

* `weights.pt`

  * All module weights from GraphAgent (including EthicsFilter etc)

* `optimizers.pt`

  * All optimiser states

* `rng_state.json`

  * Environment RNG state and agent RNG / torch RNG etc

* `config_snapshot/`

  * Deep copy of the snapshot as of this checkpoint (it may match launch snapshot exactly, but store it again so we can resume from here even if the world evolved under curriculum)

* `cognitive_hash.txt`

  * The full hash for this checkpoint

Resume rules:

* Resume never reaches back into `configs/<run_name>/`.
* Resume reloads only from the checkpoint subfolder.
* Resume writes into a new run folder, e.g.
  `runs/<run_name>__<launch_ts>_resume_<resume_ts>/`
  with a cloned snapshot from the checkpoint including cognitive_hash.txt.

Branching:

* If you mutate that cloned snapshot before resuming (for example you edit `cognitive_topology.yaml` to drop `greed` from 0.7 to 0.4), you are creating a sibling mind with a new cognitive_hash.
* That is allowed, but it’s recorded as a fork, not a continuation.

This gets us:

* long trainings that survive GPU pre-emption,
* forensics (“replay mind at tick 842, unchanged”),
* honest ablations (“same weights, same world, except panic disabled”).

### 12.5 Telemetry and UI

Goal: observable cognition.

Implement:

* the per-tick `telemetry/` writer described in section 7.2,
* the live Run Context Panel described in section 7.1.

Refer to the Minimum fields to surface/log each tick are described in Section 7.2.

By this step, you’ve got a glass-box demo.

### 12.6 Panic and ethics for real

Goal: safety and survival actually do something.

Replace the stub panic_controller and EthicsFilter in `graph_executor` with working modules that:

* panic_controller:

  * Reads Layer 1 `panic_thresholds`
  * Checks bars from observation / belief (“energy below 0.15? health below 0.25?” etc)
  * If panic is active, is allowed to override the policy’s `candidate_action` with an emergency action (e.g. “call_ambulance”, “go_to_bed_now”)
  * Logs that override and reason so the UI + telemetry can say “panic override applied: energy_critical”

  Important: panic is allowed to reprioritise survival, but it cannot authorise illegal actions. Panic fires before ethics. Ethics is final.

* EthicsFilter:

  * Reads Layer 1 `forbid_actions` and `penalize_actions`
  * Blocks forbidden actions, substitutes a safe fallback, and emits `veto_reason`
  * Optionally soft-penalises “allowed but discouraged” actions like `shove` by biasing toward an alternative or tagging the action for a negative shaping reward
  * Emits `ethics_veto_applied` and `veto_reason` into telemetry

By the end of this step:

* panic is visible and auditable,
* ethics is visible and auditable,
* and we have an accountable override chain:
  policy → panic_controller → EthicsFilter → final_action

And that’s basically the system.

---

## 13. Open questions and forward extensions

Audience: systems, safety, curriculum, future-you

These are the deliberate “not yet, but soon” items. They’re either mildly dangerous, mildly annoying to implement, or both. We should track them, not quietly ship them in the dark.

### 13.1 Panic, ethics, and conflict

Right now the chain is:

1. hierarchical_policy proposes `candidate_action`
2. panic_controller may escalate it for survival
3. EthicsFilter can still veto the result

This implies a hard rule: ethics beats panic.

Question: should there ever be a case where panic is allowed to bypass ethics?
Example: agent is about to die, ambulance costs money it doesn’t have, “steal_money” is forbidden, but stealing is the only viable survival path.

Current answer: No. Panic cannot grant illegal behaviour. EthicsFilter is final.

We should document that as governance policy. If someone later wants “panic can break rules to avoid imminent death”, that is a formal policy discussion, and flipping that switch must change the hash. This is a line-in-the-sand safety property, not a code tweak.

Risk if ignored: very high, governance-wise. We don’t want a “the AI was desperate” excuse pathway.

### 13.2 Curriculum pressure vs identity drift

We’ve said curriculum can:

* spawn new instances of existing affordances (a second Bed, a rival agent),
* change wages, prices, or availability schedules over time,
* introduce social pressure (e.g. add an NPC that hoards food).

We’ve also said curriculum cannot silently rewrite the brain mid-run without creating a new snapshot (and new *_cognitive_hash).

What about world changes that are basically rewriting physics (for example, suddenly “Bed” now restores 10x more energy per tick)? That’s effectively a world rules change. It absolutely alters incentives and makes telemetry comparisons sketchy.

Proposal:

* If curriculum mutates the *definition* of any affordance (costs_per_tick, effects_per_tick, etc) or any bar’s base_depletion / terminal conditions, that is a world rules change and must fork the run. New snapshot, new run_id, new hash.

We can allow “spawn another copy of the existing Hospital” under the same cost/effect definitions without forking. But changing what Hospital does is not curriculum, it’s a patch. Patches must fork.

We should bake that into the launcher, so you physically can’t apply that kind of mid-run mutation without going through a “new run” path.

### 13.3 Social model and privacy

Layer 1 can turn `social_model.enabled` off completely (“sociopath mode”), which is useful for baselines or certain curriculum phases.

But we haven’t addressed:

* Should an agent be allowed to model other agents’ *goals* if policy says it’s not allowed to model other agents’ *intentions*?
* Should some roles in the sim (say, “civilian NPCs”) have reduced social observability for privacy/safety reasons?

We might want per-entity visibility rules:

* social_model can see family,
* social_model can’t reason about strangers beyond “occupied/unoccupied affordance”,
* social_model can’t infer exact plan of special protected agents.

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

This is not just flavour. It matters for ethics review in multi-agent sims where you do not want “free omniscience” or “stalker AI”.

### 13.4 Introspection and self-reporting honesty

Layer 1 has:

```yaml
introspection:
  visible_in_ui: "research"
  publish_goal_reason: true
```

Right now, “publish_goal_reason: true” means “tell the observer why you think you’re doing what you’re doing”.

Obvious issue: The agent can lie or hallucinate. We are not forcing perfect self-awareness.

We should be explicit in UI and telemetry that introspection is a *claimed* reason, not necessarily a *true causal driver*. Example: show both:

* `current_goal: SURVIVAL (engine truth)`
* `agent_claimed_reason: "I'm trying to get a better job"`

If those don’t match, that’s actually gold for teaching: we’ve caught cognitive dissonance. But we need both channels logged.

Action:

* Add `agent_claimed_reason` to telemetry.
* Add `current_goal` (engine truth) to telemetry.
* Add both to the live UI panel if `introspection.publish_goal_reason: true`.

### 13.5 Compliance richness

Right now compliance in Layer 1 has two tools:

* hard veto (`forbid_actions`)
* soft discourage (`penalize_actions` with a penalty)

We may also want:

* situational bans, like “steal is forbidden unless the target is flagged as `abandoned_property`”
* contextual norms, like “calling ambulance while mood is high and health is high is considered abuse of system, apply social penalty”

This drifts toward rule-based reasoning. We can handle it two ways:

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

Either way, we need to keep it declarative and hashable. The moment compliance becomes “arbitrary inline Python”, governance falls over.

---

At this point we’ve got:

* A world defined in YAML (Universe as Code).
* A brain defined in YAML (Brain as Code, Layers 1–3).
* A run that snapshots both.
* A mind identity (`full_cognitive_hash`) that’s cryptographically bound to that snapshot + compiled wiring.
* A panic controller and an EthicsFilter that are explicit modules in the think loop.
* Telemetry and UI that expose what the mind tried to do and why it was (or wasn’t) allowed.

That means: the sim stops being vibes and starts being evidence.

And that’s the whole game.
