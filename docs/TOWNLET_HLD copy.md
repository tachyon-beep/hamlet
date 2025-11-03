# Townlet v2.5 High Level Design Document

1. Executive Summary

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

Together, those two things give us four hard properties we did not have before:

1. The mind is explicit
   The agent's cognition is described in three YAML files: what faculties it has, how they're implemented, and how they think step by step. Panic response, ethics veto, planning depth, goal-selection bias – it's all on paper.

2. The world is explicit
   The town (energy, health, money, affordances, ambulance cost, bed quality, public cues for other agents, wage schedules) is also data. Beds, jobs, hospitals, phones are declared in config as affordances with per-tick bar effects. No more "secret physics" hardcoded deep in the environment loop.

3. Every run is provenance-bound
   When you actually run Townlet, we snapshot both the world config and the brain config, hash them, and stamp that identity onto every tick of telemetry. If the agent does something sketchy, you can prove exactly which mind, under which world rules, did it. There is no "the AI just did that".

4. We can teach and audit, not just watch
   We log not only what the body did ("health: 0.22"), but what the mind tried to do, what panic overrode, and what ethics vetoed. You can answer "why" with evidence instead of vibes.

Very bluntly: Townlet stops being "a neat RL sim with emergent drama" and becomes "an accountable cognitive system we can diff, replay, and defend".

We get this by doing three things.

1.1 The brain is now assembled, not baked in

Instead of one fused RL blob that tries to handle perception, planning, social reasoning, panic, morals and action selection all at once, we now build an agent brain out of named cognitive modules:

* Perception / belief state builder
  Turns partial, noisy world observations into an internal belief ("what I think is true right now").

* World Model
  Predicts what will happen next if I take action X in situation Y. This is how the agent learns "physics" and economy, including non-stationary changes like new prices or new wages.

* Social Model
  Reasons about other agents: what they're likely to do next, what they seem to be pursuing. It uses only public cues the world exposes (tired posture, slumped, rushing to fridge, etc). It does not get hidden state at runtime.

* Hierarchical Policy
  Picks a strategic goal (eg SURVIVAL vs THRIVING) and then picks a concrete action to serve that goal each tick.

* Panic Controller
  Can override normal planning if the body is in trouble (energy critical, health critical).

* EthicsFilter
  Has final say. It can forbid actions that violate compliance ("steal", "attack") or downgrade risky actions ("shove"). Panic cannot break ethics.

Those modules, and the way they talk to each other, are not hardwired in code. They're declared in configuration and built at runtime. We can literally turn the Social Model off and get "sociopath mode" for ablation, without touching the shared policy code. We can change the depth of forward planning from 2 ticks to 6 ticks in a YAML field. We can inject a new panic rule without retraining the vision stack.

Instead of "one RL blob to rule them all", Townlet v2.5 is parts plus wiring.

1.2 The world is now declared, not hidden in engine logic

The environment is no longer an ad hoc Python ruleset. The world itself – bars like energy and money, what beds do, how hospitals heal, how much ambulance costs, when jobs pay, who can occupy which affordance – is defined in `universe_as_code.yaml`.

Affordances are just declarative objects with capacity, per-tick effects, costs, interrupt rules, and (if needed) a small whitelisted special effect like `teleport_to:hospital`. We're not storing open-ended logic in YAML; we're using YAML to request behaviours the engine knows how to apply deterministically.

This means "the physics and economy of the town" is inspectable. If someone asks "why is the agent spending $300 to call an ambulance instead of walking two blocks to the hospital?", we can answer that by looking at the world config (ambulance teleports immediately but costs a lot, hospital heals slowly and may be closed at night) and at the brain config (panic threshold allowed survival override at 5% health). No mysticism required.

1.3 Every run now has identity and chain of custody

When you launch a run, we don't just spin up the sim. We mint an artefact.

We snapshot:

* the world config (Universe as Code),
* the brain config (Brain as Code, all three layers),
* runtime envelope (tick rate, curriculum schedule, seed),
* plus we compute a `full_cognitive_hash` over all of it and the compiled cognition graph.

Then we log every tick of behaviour tagged with that hash.

That gives you:

* Reproducibility ("we can re-run the same mind in the same world and get the same behavioural envelope")
* Accountability ("at tick 842, EthicsFilter blocked STEAL in mind hash 9af3c2e1, in world snapshot 'austerity_nightshift_v3'")
* Teaching material ("here is exactly what module proposed, what panic overrode, and what ethics vetoed while the agent was broke and nearly unconscious")

1.4 Live telemetry shows cognition, not just vitals

The UI for a live run is not just bars (energy 0.22, mood 0.41). It also shows:

* which high-level goal the agent is currently pursuing (SURVIVAL / THRIVING / SOCIAL),
* whether panic_controller overrode the normal goal this tick (panic override applied: "health critical"),
* whether EthicsFilter vetoed the chosen action and why ("attempted STEAL; forbidden"),
* planning depth ("world_model.rollout_depth: 6 ticks ahead"),
* whether the Social Model is active,
* the short form of the cognitive hash (so you know exactly which mind you're looking at).

This lets an instructor literally point at the panel and narrate:
"It's panicking because energy fell under 15 percent. It tried to steal food but EthicsFilter blocked it. It has permission to plan 6 ticks ahead. It's currently in SURVIVAL mode. So yes, it's desperate, but it's still under moral constraint."

That's the "glass box" promise. Townlet is no longer "watch the AI do drama". It's "watch cognitive state, survival heuristics, and ethics fire in public".

1.5 Why this matters (governance, research, teaching)

Interpretability
We can answer "which part of the mind did that and why" with evidence. We can say "panic overrode policy here" or "EthicsFilter vetoed theft here". This is not a vague post-hoc story, it's in the telemetry.

Reproducibility
A behaviour is not folklore. It's a run folder with a config snapshot and a hash. Anyone else can rehydrate it.

Accountability
If something unsafe happens, we don't blame "the AI". We inspect which safety setting allowed it, in which execution graph step, under which panic thresholds, in which world rules. That’s fixable. That’s auditable.

Pedagogy / curriculum
Students (or auditors, or policy people) can open a run and read literal YAML to see what the agent was "allowed to be". They can diff two versions of the brain and two versions of the world and understand what changed. They can run one mind in multiple worlds or one world with multiple minds and study the difference. You can teach causality, not mysticism.

Summary: Townlet v2.5 is brain-as-config plus world-as-config, snapped and hashed at run time, with live introspection and veto logging. That is now the baseline, not the stretch goal.

---

2. Brain as Code (BAC): the mind

---

Brain as Code is how we define Townlet's mind as something we can inspect, diff, and enforce.

The BAC stack is three YAML layers. Together, they are the Software Defined Agent.

Change the YAMLs, you change the mind. Snapshot the YAMLs, you freeze the mind. Hash the snapshot, you can prove which mind took which action.

2.1 Layer 1: cognitive_topology.yaml
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

* SURVIVAL
* THRIVING
* SOCIAL

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

* "attack"
* "steal"
penalize_actions:
* { action: "shove", penalty: -5.0 }

introspection:
publish_goal_reason: true      # Should the agent explain itself in UI?
visible_in_ui: "research"      # beginner | intermediate | research

How Layer 1 is used in runtime:

* panic_thresholds tells panic_controller when it's allowed to override the normal plan and just survive now.
* forbid_actions tells EthicsFilter what is never allowed, even if the agent is dying.
* personality feeds into the hierarchical policy's goal choice, so "greed: 0.7" really does mean "money-seeking wins arguments inside its head."
* publish_goal_reason controls whether we surface "I'm going to work because we need money" to the human observer.

Layer 1 is what policy signs off on. It's the file you show when someone asks "what kind of mind did you put in this world?"

If you change Layer 1 between runs (for example, allow STEAL, or lower panic threshold, or turn social modelling off), that's not the same agent any more. That must produce a new cognitive hash.

2.2 Layer 2: agent_architecture.yaml
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
goal_distribution:        { dim: 16 }  # maps to goal_vector_dim
next_action_dist:         { dim: 6 }   # maps to action_space_dim
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

Why Layer 2 matters:

* It makes the mind rebuildable in a lab, not just on "Dave's laptop".
* It makes module-swaps a controlled experiment ("same cognitive_topology, different world_model internals").
* It makes it obvious if someone quietly changed the optimiser or widened the GRU and then claimed "no behavioural change expected". Governance can now call nonsense on that.

2.3 Layer 3: execution_graph.yaml
Audience: safety, audit, you at 2am
Nickname: the think loop

Layer 3 is the actual cognition pipeline Townlet runs every tick. This is the part that most RL projects pretend is "obvious" and never write down. We write it down.

It’s a DAG of named steps. It says:

inputs:

* raw_observation
* prev_recurrent_state

steps:

* perception_packet:
  node: "@modules.perception_encoder"
  inputs:
  * "@graph.raw_observation"
  * "@graph.prev_recurrent_state"

* belief_distribution:
  node: "@utils.unpack"
  input: "@steps.perception_packet"
  key: "belief"

* new_recurrent_state:
  node: "@utils.unpack"
  input: "@steps.perception_packet"
  key: "state"

* policy_packet:
  node: "@modules.hierarchical_policy"
  inputs:
  * "@steps.belief_distribution"
  * "@services.world_model_service"
  * "@services.social_model_service"

* candidate_action:
  node: "@utils.unpack"
  input: "@steps.policy_packet"
  key: "action"

* panic_adjustment:
  node: "@modules.panic_controller"
  inputs:
  * "@steps.candidate_action"
  * "@config.L1.panic_thresholds"
  outputs:
  * "panic_action"
  * "panic_reason"

* final_action:
  node: "@modules.EthicsFilter"
  inputs:
  * "@steps.panic_adjustment.panic_action"
  * "@config.L1.compliance.forbid_actions"
  outputs:
  * "action"
  * "veto_reason"

outputs:

* final_action: "@steps.final_action.action"
* new_recurrent_state: "@steps.new_recurrent_state"

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

3. Run Bundles and Provenance

---

Townlet v2.5 doesn't "run an agent". It mints an artefact with identity, provenance and chain of custody. That's the difference between "cool AI demo" and "system we can take in front of governance without sweating through our shirt".

3.1 The run bundle

Before a run starts, you prepare a bundle under `configs/<run_name>/`:

configs/
L99_AusterityNightshift/
config.yaml                # runtime envelope: tick rate, duration, curriculum, seed
universe_as_code.yaml      # the world (bars, affordances, prices, cues)
cognitive_topology.yaml    # BAC Layer 1 (behaviour contract and safety knobs)
agent_architecture.yaml    # BAC Layer 2 (module blueprints and interfaces)
execution_graph.yaml       # BAC Layer 3 (think loop + panic/ethics chain)

* `universe_as_code.yaml` is the world spec. It defines bars like energy/health/money, affordances like Bed / Job / Hospital / PhoneAmbulance, their per-tick effects and costs, capacity limits, interrupt rules, and any special whitelisted effect (for example `teleport_to:hospital`). It also defines public cues other agents can see ("looks_tired", "bleeding", "panicking").

* The three BAC layers are the mind spec.

* `config.yaml` says how long we run, at what tick rate, with how many agents, and what curriculum (for example: "start alone, introduce a food-scarcity rival after 10k ticks").

This bundle is what we claim we are about to run.

3.2 Launching a run

When we actually launch, we don't execute the live bundle. We snapshot it.

The launcher creates:

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

Critical details:

* `config_snapshot/` is a byte-for-byte copy of the five YAMLs at launch time. After launch, the live sim reads only from this snapshot, never from the mutable stuff in `configs/`. This is how we prevent "oops I hotpatched ethics mid-run but didn't tell anyone".

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

3.3 Checkpoints and resume

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

3.4 Why provenance is non-negotiable

Without this provenance model, Townlet is just another agent-in-a-box demo and governance has to take us on trust.

With this provenance model:

* We can prove at audit time which ethics rules were live.
* We can prove panic never bypassed ethics unless someone explicitly allowed that in Layer 3 (and if they did, the hash changed).
* We can replay any behaviour clip and show not just "what happened", but "what mind, under what declared rules, thought what, tried what, got blocked by what".

That is what lets us run Townlet in anger, not just in a lab.

So: Townlet v2.5 == Hamlet post-refactor. It's the same agent-in-world system, formally expressed. Universe as Code defines the world. Brain as Code defines the mind. Runs freeze both, hash both, and log both. That is the story everywhere, full stop.

Absolutely. Continuing in the same unified Townlet/BAC/UAC framing, here are sections 4–6 rewritten.

---

4. Checkpoints

---

A checkpoint is not "saved weights lol". It's a frozen moment of a specific mind, in a specific world, under specific rules, at a specific instant in time.

Townlet treats every checkpoint as evidence. A checkpoint must include everything required to:

* pick up training honestly,
* replay behaviour honestly,
* and prove, later, which exact cognitive configuration produced which exact action.

When we write a checkpoint for a run, we create something like:

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

Let's unpack what those pieces actually mean.

4.1 weights.pt

This is the live neural state of the brain at that tick:

* perception module weights
* world_model weights
* social_model weights
* hierarchical_policy weights
* panic_controller weights (if it's learned)
* EthicsFilter weights (if it's learned / parameterised)
* anything else registered in the agent module registry

In v1 this stuff all lived in one giant black-box DQN. In Townlet, it's all the submodules declared in Layer 2 (agent_architecture.yaml) and wired by Layer 3 (execution_graph.yaml). We save them all together because, for audit, "the brain" means the entire SDA module set, not just the action head.

4.2 optimizers.pt

We don't just log parameters, we log optimiser state (Adam moments etc) for each trainable module.

Why? Because "resume training" must mean "continue the same mind's learning process", not "respawn something with the same weights but different momentum and call it continuous". If you've ever done RL you know that quietly dropping optimiser state can absolutely change learning behaviour. We are not pretending that's irrelevant. We store it.

4.3 rng_state.json

Randomness is part of causality.

We store the RNG states that matter:

* environment RNG,
* agent RNG (PyTorch generators etc),
* anything else that would affect rollout sampling, tie-breaks in affordance contention, exploration noise, etc.

That lets you actually re-run tick 501 and get the same coin flips. Which means if someone asks "would it always have chosen STEAL here?" you can answer "under this exact stochastic roll, here's what happened" and you can reproduce that, not just act mystic.

4.4 config_snapshot/

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

4.5 full_cognitive_hash.txt

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

4.6 Why checkpoints are legally interesting (not just technically interesting)

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

5. Resume semantics

---

Resuming is not "just load weights and go again". Resume is part of the audit chain.

If we can't prove continuity of mind across pauses, we can't claim continuity of behaviour for governance, and we can't do serious ablation science.

So we define resume like a forensic procedure.

5.1 The rule: the checkpoint snapshot is law

When you resume from a checkpoint, you must restore from the checkpoint's own `config_snapshot/`, not from whatever is currently sitting in `configs/<run_name>/` in your working tree.

That means:

* You bring back the exact cognitive_topology.yaml from that checkpoint (same ethics, same panic thresholds, same greed sliders).
* You bring back the exact universe_as_code.yaml from that checkpoint (same ambulance cost, same bed effects, same wage rates).
* You bring back the exact execution_graph.yaml (same panic-then-ethics ordering).
* You bring back the optimiser state and RNG.

You do not "reconstruct" the agent from the latest code and hope it's approximately right. You rehydrate that specific mind in that specific world with that specific internal loop.

5.2 Where the resumed run lives

Resuming from:
runs/L99_AusterityNightshift__2025-11-03-12-14-22/checkpoints/step_000500/

creates a fresh new run folder, for example:
runs/
L99_AusterityNightshift__2025-11-03-12-14-22_resume_2025-11-03-13-40-09/
config_snapshot/          # copied from the checkpoint, byte-for-byte
checkpoints/
telemetry/
logs/

Important bits:

* We do not keep writing into the old run folder. New run, new timeline.
* We recompute the cognitive hash from the checkpoint snapshot. If you have not changed anything, the hash will match. That proves it's the same mind continuing.
* Telemetry in the resumed run now logs the same hash, so audit can say: "this is truly the same mind, same ethics, same world, just continued later".

5.3 Forking vs continuing

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

5.4 Why resume semantics matter

Three reasons.

1. Long training on flaky hardware
   If training gets pre-empted at 3 am, you can resume later without inventing a "different" agent. Same hash, same mind, same optimiser, same RNG continuation.

2. Honest ablations
   You can say "this is literally the same mind except we disabled the Social Model" and prove that change in config diff + new hash. When you compare behaviour, you know what you're actually comparing.

3. Audit trail
   If someone questions a safety decision ("why did you let panic override normal reasoning here?"), you can show exactly when that rule entered the snapshot. There's no "it drifted over time"; drift is now a recorded fork.

Resume is now a governance primitive, not a convenience function.

---

6. Runtime engine components

---

Under Townlet v2.5, the old pattern "one giant RL class owns everything" is gone. We replaced it with three core pieces: a factory, a graph agent, and an execution engine.

This is where we guarantee that what we run is what we declared, and what we declared is what we logged, and what we logged is what we can replay.

6.1 agent/factory.py
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

6.2 agent/graph_agent.py
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

This is key: The only contract the rest of the sim needs is "given what you just saw and what you remember, what do you do next and what will you remember after that". Internally, the brain can be arbitrarily rich – planning, simulating the future, modelling other agents, panicking, self-censoring via ethics – and we don't have to bolt that logic all over the environment.

Also important: GraphAgent is always instantiated from the run's frozen snapshot. It never reads "live" configs during execution. This is how we stop "I hotpatched the EthicsFilter in memory for the live demo" type nonsense.

6.3 agent/graph_executor.py
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
* If someone wants to insert a new veto stage, or let panic bypass ethics, they must edit Layer 3, recompile, and accept a new cognitive hash. That is a governance lever, not just an engineering trick.

6.4 EthicsFilter
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
* EthicsFilter logs every veto, every tick. That means we don't just know "it behaved safely". We know "it tried to do something unsafe and got stopped". That is exactly what regulators will ask you to show.

Later extensions (which we've flagged in open questions) may allow more nuanced compliance rules like "soft penalties if you abuse ambulance when healthy" or "contextual exceptions in extreme survival", but in v2.5 we keep the invariant: panic does not bypass ethics, ethics is final, ethics is logged.

6.5 Why these engine pieces exist at all

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

   Every one of those changes produces a clean diff in YAML, a new run folder, and a new cognitive hash. Which means we can explore, and governance can still sleep at night.

Perfect, let's carry on with sections 7 through 9 in the same unified Townlet voice (BAC + UAC, glass box, governance-first). I'll treat these as direct continuations of the rewrite you’ve already got for 1–6.

---

7. Telemetry and UI surfacing

---

We are not building "an AI that seems smart". We are building "an AI whose thinking you can literally watch and quote back in a meeting."

So Townlet v2.5 ships with first-class introspection: we log what the mind tried to do, what stopped it, and why. Live. Per tick. With identity. This is the core of the glass box story.

We expose two layers of visibility:

1. a live panel in the UI for humans watching the sim in real time, and
2. structured telemetry on disk for replay, teaching, and audit.

These two layers must always agree. If they don't, that's a bug.

7.1 Run Context Panel (live inspector HUD)

At runtime, when you click an agent, you get a compact "here's what this mind is doing right now" panel. That panel is not vibes. It’s populated from the same data we log to disk.

This panel MUST include at least:

* run_id
  Example: `L99_AusterityNightshift__2025-11-03-12-14-22`
  This tells you which frozen bundle of world + brain you're looking at.

* short_cognitive_hash
  A short form (eg first 8 chars) of that agent’s full cognitive hash.
  This is "which exact mind is in that body right now". If two bodies share the same short hash, we are literally watching two copies of the same brain spec in different circumstances.

* tick
  Current tick index and planned_run_length from config.yaml.
  Lets you say "this happened at tick 842 out of 10,000", which matters when you're doing curriculum or staged hardship.

* current_goal
  The high-level strategic goal the meta-controller (hierarchical_policy.meta_controller) says it is pursuing right now, e.g. `SURVIVAL`, `THRIVING`, `SOCIAL`.
  This is engine truth, not vibes.

* panic_state
  Boolean or enum. Are we currently in emergency override because we tripped `panic_thresholds` in cognitive_topology.yaml (Layer 1)?
  This is: "is the Panic Controller allowed to overrule normal planning right now?"

* panic_override_last_tick
  If the panic_controller actually overrode the policy last tick:

  * what action it forced (e.g. `call_ambulance`)
  * and why (e.g. `energy_critical`).
    This is how we surface "it freaked out and did triage" instead of just "it ran".

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

7.2 Telemetry (per-tick trace to disk)

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
  Short numeric/text summary of how sure perception is about critical bars.
  Example: `"energy_estimate_confidence": 0.42`.
  This is how we catch "agent walked past a fridge because it literally didn't believe it was starving".

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

8. Declarative goals and termination conditions

---

Townlet agents are not just twitching reflex loops. They pursue explicit high-level goals, like SURVIVAL or THRIVING or SOCIAL, and can tell you which one they’re on right now.

We do two things:

1. We make goals explicit data structures, not vague "the RL policy probably cares about reward shaping".
2. We make "I'm done with this goal" a declarative rule in YAML, not a secret lambda hidden in code.

8.1 Goal definitions live in config, not in Python

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

8.2 Why this matters

* For governance/audit
  We can answer the question "Why was it still pursuing GET_MONEY while its health was collapsing?" by pointing to the YAML.
  Maybe GET_MONEY didn't terminate until health ≥ 0.7. That's a design decision, not 'the AI went rogue'.

* For curriculum
  Early in training you might define SURVIVAL as "energy ≥ 0.5 is fine". Later curriculum tightens that to 0.8. That becomes a diff in YAML, not a code poke.
  Students can literally compare behaviour when SURVIVAL is lenient vs strict.

* For teaching
  You can ask a student: "The agent is starving but still working. Does the SURVIVAL goal terminate too late? Or is the meta-controller just not switching to SURVIVAL at all because greed is too high in cognitive_topology.yaml?"
  That’s not abstract RL theory, that’s direct inspection.

8.3 Honesty in introspection

Now that goals are formal objects and termination is a declarative rule, we can show two different "explanations" side by side:

* current_goal (engine truth): `SURVIVAL`
* agent_claimed_reason (self-report / introspection): `"I'm going to work to save up for rent"`

Sometimes those match. Sometimes they don't.

That gap is important:

* If they match, nice, we can narrate behaviour in plain language to non-technical stakeholders.
* If they don't match, that’s gold for a lesson: "The agent says it's grinding for rent, but engine truth is it’s still in SURVIVAL mode and just mis-evaluated what would keep it alive. That's a world-model error."

We log both in telemetry on purpose.

---

9. Affordance semantics in universe_as_code.yaml

---

Universe as Code is the other half of this story. Brain as Code (Layers 1–3) defines the mind. Universe as Code defines the body and the town.

Townlet does not hardcode "beds make you rested" in Python logic sprinkled everywhere. Instead, the world is declared as affordances with effects on bars. Beds, jobs, phones, ambulances, hospitals, fridges, pubs — all of them are just entries in the world config.

9.1 Affordances are declarative

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

9.2 Engine semantics (how the runtime actually interprets affordances)

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

9.3 Why Universe as Code matters for BAC

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

Perfect. Here are sections 10, 11, and 12 rewritten so they line up with Townlet v2.5, Brain as Code (BAC), Universe as Code (UAC), panic/ethics chain, cognitive hashes, and the run snapshot model.

I'm going to keep the checklist-y style because these sections are used operationally and you want engineers + auditors to be able to literally tick them off.

---

## 10. Success criteria

We judge success on three axes: technical, teaching, and governance. All three matter. If we don't hit all three, the story breaks.

### 10.1 Technical success

[ ] We can launch a run from `configs/<run_name>/` and automatically create `runs/<run_name>__<timestamp>/` with a frozen `config_snapshot/` that contains:

* `config.yaml`
* `universe_as_code.yaml`
* `cognitive_topology.yaml` (Layer 1)
* `agent_architecture.yaml` (Layer 2)
* `execution_graph.yaml` (Layer 3)

[ ] `agent/factory.py` can reconstruct a functioning agent brain (GraphAgent) purely from that frozen `config_snapshot/`, without reading anything from live mutable config.

[ ] `GraphAgent.think()` can tick once using only that snapshot: perception → hierarchical policy → panic_controller → EthicsFilter → `final_action`.

[ ] Each checkpoint written under `runs/.../checkpoints/step_<N>/` includes:

* model weights for every module (perception, world_model, social_model, hierarchical_policy, panic_controller, EthicsFilter, etc)
* optimiser states
* RNG state
* a nested copy of `config_snapshot/`
* `cognitive_hash.txt` for that checkpoint

[ ] Resuming from a checkpoint:

* reloads only from `runs/.../checkpoints/step_<N>/`
* writes a new run folder `runs/<run_name>__<launch_ts>_resume_<resume_ts>/`
* reproduces the same cognitive hash if the snapshot is unmodified

[ ] Telemetry logs one structured row per tick into `runs/.../telemetry/`, with:

* `run_id`
* tick index
* full cognitive hash
* current_goal
* panic state
* candidate_action
* panic_adjusted_action (+ panic_reason)
* final_action
* ethics_veto_applied (+ veto_reason)
* planning_depth
* social_model.enabled
* short belief/world/social summaries

[ ] The runtime UI ("Run Context Panel") surfaces, live:

* run_id
* short_cognitive_hash (pretty form of the full hash)
* tick / planned_run_length
* current_goal
* panic_state
* planning_depth (world_model.rollout_depth)
* social_model.enabled
* panic_override_last_tick (+ panic_reason)
* ethics_veto_last_tick (+ veto_reason)
* agent_claimed_reason (if introspection.publish_goal_reason is on)

If we get all these, we no longer have a "neural net that does stuff". We have a reproducible mind in a governed world.

---

### 10.2 Pedagogical success

The point of Townlet v2.5 is not just to make a smarter agent. It's to make a teachable agent. We hit pedagogical success when the system is something you can put in front of a class, and they can reason about it like a living system, not a superstition.

[ ] A beginner can answer "Why didn't it steal the food?" using only:

* the live Run Context Panel (which shows `ethics_veto_last_tick` and `veto_reason`)
* the run's `cognitive_topology.yaml` (which shows `compliance.forbid_actions: ["steal", ...]`)

In other words: you do not need to read source code to answer an ethics/safety question. You can answer it from YAML + UI.

[ ] An intermediate student can:

* edit `agent_architecture.yaml` (for example, swap GRU → LSTM in the perception module, or change hidden_dim)
* launch a new run
* observe how memory/behaviour changes
* and explain the change in terms of memory capacity, not "the AI got weird"

So they can perform controlled ablations by editing config, not by rewriting thousands of lines of Torch.

[ ] A researcher can:

* edit `execution_graph.yaml` to, for example, temporarily bypass `world_model_service` input into the policy
* rerun
* show that the agent becomes more impulsive / short-horizon
* and prove that change via diff in `execution_graph.yaml` plus new `cognitive_hash.txt`

Meaning: "remove foresight, observe impulsivity" is now a 1-line wiring experiment, not a 2-week surgery.

[ ] For any interesting emergent behaviour clip, we can pull the run folder and point to:

* which mind (full cognitive hash),
* which world rules (`universe_as_code.yaml`),
* which panic thresholds,
* which compliance rules (`forbid_actions`, penalties),
* what goal the agent believed it was pursuing at that tick (`current_goal`),
* and what reason the agent claimed (`agent_claimed_reason`).

This is key for classroom demos. You can literally scrub to tick 842 and narrate "The agent thought it was in SURVIVAL mode, panic was active, and EthicsFilter blocked 'steal'."

---

### 10.3 Governance success

Governance is the other audience. They don't care how pretty the recurrent state is. They care about "Can I hold you to this in a Senate Estimates hearing, yes or no."

[ ] We can prove to an auditor that, at tick T in run R:

* `cognitive_topology.yaml` at that tick had `forbid_actions: ["attack", "steal"]`
* `execution_graph.yaml` at that tick still routed all candidate actions through `EthicsFilter`
* telemetry for tick T shows `ethics_veto_applied: true` and `veto_reason: "steal forbidden"`

So we can say: "The agent tried to steal at tick T. It was blocked. We can prove the block happened, and we can prove the block was required by configuration."

[ ] We can replay that same mind, at that same point in time, using only the checkpoint directory from that run. We don't need any mutable source code or live config. That replayed agent produces the same cognitive hash and the same cognitive wiring.

That is chain-of-custody for cognition.

[ ] We can demonstrate lineage rules:

* If you resume without changing the snapshot, it's the same mind (same hash).
* If you edit anything that changes cognition (panic thresholds, greed, social_model.enabled, EthicsFilter rules, rollout_depth, etc), the hash changes and we give it a new run_id. We don't pretend it's "the same agent, just adjusted a bit".

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

  * all module weights from the GraphAgent (including EthicsFilter, panic_controller, etc)
* `optimizers.pt`

  * optimiser states for each trainable module
* `rng_state.json`

  * RNG state for both sim and agent
* `config_snapshot/`

  * deep copy of the snapshot as of this checkpoint (not a pointer to `configs/`)
* `cognitive_hash.txt`

  * the full hash at this checkpoint

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

By this point, you can literally point at the panel and narrate the mind in plain English: "It thinks it's in SURVIVAL, panic just overruled the planner, EthicsFilter blocked stealing, planning depth is 6 ticks, and it claims 'I'm going to work for money'."

---

### 11.6 Panic and ethics for real

Goal: safety and survival aren't just commented in YAML, they actually run in-graph.

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

This is the point where you can sit someone from governance down and show them the override trace instead of saying "trust us, it won't do crime when it's scared."

---

## 12. Implementation order (milestones)

Think of section 11 as the conceptual order of operations. Section 12 is the concrete delivery milestones for engineering, curriculum, safety, and audit. This is the boot sequence.

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

* Survival urgency and ethical constraint are now explicit, inspectable modules in the think loop, not invisible "reward shaping vibes".
* You can show "panic tried X, ethics said no" as an auditable trace, with hash.

Yep. Here's section 13 as a forward-looking / governance-risk section. It's written so it can live in the same doc without sounding like "future work fluff". It's deliberately framed as policy questions, because these are the things that will get you in trouble later if you don't write them down now.

I've kept the internal names consistent with the rest of Townlet v2.5: panic_controller, EthicsFilter, Universe as Code (UAC), Brain as Code (BAC), execution_graph, etc.

---

## 13. Open questions and forward extensions

Audience: systems, safety, curriculum, governance, future-you

These are the things we have intentionally not fully solved in Townlet v2.5. They are not bugs. They are decision surfaces. Changing any of these should change the cognitive hash, and in most cases should trigger a formal review.

We track them here so no one quietly "just tweaks a constant in prod".

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

### 13.3 Social modelling and privacy constraints

In Layer 1 (`cognitive_topology.yaml`), we can flip `social_model.enabled` to false. That gives you a baseline "sociopath mode" agent that doesn't model the minds of others. Good for ablations.

What's missing in v2.5 is fine-grain social visibility control. Right now, the social model (Module C) can, in principle, infer goals and predict next actions of every visible agent, using their public_cues and interaction traces.

That becomes a privacy / ethics problem in multi-agent sims.

Questions we haven't finalised:

* Are all agents equally "legible", or are there protected roles?
* Should some agents be modelled only as "occupied affordance", not as an intentional mind?
* Should some inference channels (for example, "predict next action of a child NPC") be explicitly disallowed?

We probably need privilege scoping in Layer 1, something like:

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

This becomes essential once you simulate family structures, dependants, or vulnerable cohorts. It's also the place where "are we training stalker behaviour?" gets answered.

### 13.4 Introspection honesty vs engine truth

Layer 1 currently has:

```yaml
introspection:
  visible_in_ui: "research"    # beginner | intermediate | research
  publish_goal_reason: true
```

If `publish_goal_reason` is true, the agent will narrate why it thinks it's doing what it's doing. That narration is exposed in UI as `agent_claimed_reason`, and also logged in telemetry.

There's an obvious problem: the claimed reason might not match the actual driver.

Example:

* Engine truth says current_goal = SURVIVAL (panic mode, get calories now).
* The agent says "I'm working on long-term wealth accumulation".

This is either self-deception or just confabulation. Both are scientifically interesting and politically sensitive.

We address it this way:

* Telemetry per tick must log both:

  * `current_goal` (engine truth from hierarchical_policy / meta-controller)
  * `agent_claimed_reason` (self-report, if enabled)
* The Run Context Panel must display them distinctly, not merge them.

This lets instructors teach "agents rationalise" and lets auditors say "the system knew it was in survival mode, regardless of what the agent verbally claimed".

Future extension:
We may want to add an explicit "self-awareness / alignment" probe that measures the divergence between engine truth and self-claim. That becomes a safety metric in itself.

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

If we don't do this, "ethics" will slowly leak back into hand-coded Python conditionals, which kills auditability.

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
   Teleport is extremely powerful because it's effectively map control and safety bypass. If you can call ambulance from anywhere and teleport to hospital, that's a survival exploit. That's fine if it's intentional. It's not fine if it's accidental.

   We probably want a clear rule: any new `effect_type` added to the whitelist is a governance event and must update the hash. You don't get to slip "stun_other_agent" into the whitelist silently.

2. Claim semantics for capacity / reservation
   Currently we model "use of an affordance" (e.g. a bed) as an ephemeral reservation per tick with deterministic tie-breaking. There's no persistent ownership in the YAML. That's intentional: it avoids hidden state that the world model can't learn.

   If we ever introduce persistent ownership (for example "this is my bed now"), that is a world rules change and must fork the run. Otherwise we break interpretability, because now decisions depend on invisible state not reflected in `universe_as_code.yaml`.

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

We should be explicit that "who signs off on identity" is not an engineer. It's governance. The team that owns compliance policy needs to approve any change that would change the cognitive hash rules. This becomes a formal control point.

---

### 13.8 Alignment between stated goal and actual goal

We are now explicitly modelling two goal channels:

* `current_goal` (engine truth from the meta-controller / hierarchical_policy)
* `agent_claimed_reason` (self-narrative string/label)

We should measure divergence between these regularly. A large stable gap means:

* the agent is strategically pursuing one thing,
* but narrating another (to humans or to itself).

That's either deception, confusion, or PR. All three are interesting and maybe dangerous.

Future extension:

* define a simple "truthfulness index" per run: fraction of ticks where `agent_claimed_reason` semantically matches `current_goal`.
* log that in telemetry summaries for a run.
* surface it in replay tooling, because "this mind self-justifies aggressively" is something you want to know before you put it in a classroom.

---

### 13.9 Family channel, coordination, and collusion

We're planning (in the Personality & Family extension) to give related agents:

* a heritable personality vector (greed, neuroticism, curiosity, etc),
* a shared `family_id`,
* and a private low-bandwidth signalling channel (SetCommChannel(int64)).

That channel becomes visible (only) to in-group members as an extra input, and they learn to ground meaning in it ("123" might start to mean "I've found money", etc).

Open questions:

* Is in-group signalling considered "social_model.enabled", or is it orthogonal?
* Can that private channel be used to coordinate behaviour that would violate global norms in a way EthicsFilter can't detect? For example, two related agents collude to starve out a third.
* Do we have to audit those comms for governance? If so, how do we do that without destroying the research value of emergent communication?

At minimum:

* The presence of any family/in-group channel must be declared in `cognitive_topology.yaml`, and must feed into the hash.
* Telemetry must log that channel activity existed, even if it doesn't decode the meaning.
* If we add any policy logic that treats "family" differently in EthicsFilter (for example, allowing shove to move a family member out of danger, but forbidding shove on strangers), that policy must be expressed declaratively and must change the hash.

---

### 13.10 Red lines we are choosing to keep

There are a few bright lines we need to write down now so they don't get hand-waved later:

1. EthicsFilter must always sit in the execution_graph after panic_controller.
   If someone moves it earlier or removes it, that's a fundamentally different class of agent and must produce a new hash. You cannot claim continuity.

2. You cannot hotpatch EthicsFilter or panic_controller at runtime without forking the run.
   That includes "just raising the panic threshold because we want more urgent behaviour." That's a new mind.

3. Universe as Code (UAC) cannot be mutated in a way that changes affordance semantics or bar physics mid-run without forking the run.
   Curriculum is allowed to add pressure; it is not allowed to rewrite reality without a new run_id.

4. If any module starts doing reasoning that is not reflected in `execution_graph.yaml`, that's a design violation.
   Every thinking step that can affect `final_action` must be in the graph. No side channels.

These are governance-grade invariants. If any of them are violated, audit is gone and reproducibility is compromised. At that point you are back to "black box AI", and everything in this document stops being true.
