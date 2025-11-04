## 1. Executive Summary

---

Document version: 2.5 (consolidated BAC/UAC architecture)
Date: 3 November 2025
Status: Approved for Implementation
Owner: Principal Technical Advisor (AI)

Townlet v2.5 defines the agent and the world as audited configuration, not as "whatever the Python happened to be at the time".

The old Townlet / Hamlet agent was one opaque recurrent Q-network (`RecurrentSpatialQNetwork`) that turned partial observations into actions. It sort of worked, sometimes brilliantly, but it was a black box. If it sprinted to hospital and then fell asleep in the shower, the only honest answer to "why?" was "the weights felt like it".

Townlet v2.5 replaces that with a Software Defined Agent (SDA) running in a Software Defined World. We treat both the world and the mind as first-class content. We call this:

- Universe as Code (UAC): the world, declared in data.
- Brain as Code (BAC): the mind, declared in data.

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

- Perception / belief state builder
  Transforms partial, noisy observations into an internal belief state.

- World Model
  Predicts state transitions for candidate actions, allowing the agent to learn the town's dynamics, including non-stationary changes such as price shifts.

- Social Model
  Estimates the likely behaviour and goals of nearby agents using the public cues that Universe as Code exposes. It receives no hidden state at runtime.

- Hierarchical Policy
  Selects a strategic goal (for example SURVIVAL versus THRIVING) and chooses the concrete action that advances that goal each tick.

- Panic Controller
  Overrides normal planning when survival thresholds such as energy or health fall below configured limits.

- EthicsFilter
  Applies the final compliance gate. It forbids actions that violate policy (for example, "steal" or "attack") or downgrades risky options. Panic cannot bypass ethics.

Modules and their interactions are declared in configuration and materialised at runtime. We can explicitly disable the Social Model for ablation without touching shared policy code, adjust the planning horizon from two to six ticks via configuration, or introduce a new panic rule without retraining perception.

Townlet v2.5 is therefore an engineered assembly rather than a single opaque network.

### 1.2 The World Is Now Declared, Not Hidden in Engine Logic

The environment is no longer an ad hoc Python ruleset. Core mechanics—energy, hygiene, money, affordances, ambulance pricing, bed quality, public cues, wage schedules—are defined in `universe_as_code.yaml`.

Affordances are declarative objects with capacity, per-tick effects, costs, interrupt rules, and, where necessary, a whitelisted special effect such as `teleport_to:hospital`. YAML expresses the desired behaviour; the engine executes those behaviours deterministically.

Consequently, the town's physics and economy are reviewable. Questions such as "why did the agent pay $300 to call an ambulance instead of walking to hospital?" can be answered by inspecting the world configuration (immediate teleportation at a high cost versus slower treatment with possible closing hours) alongside the brain configuration (panic thresholds permitting a survival override at 5 percent health).

### 1.3 Every Run Now Has Identity and Chain of Custody

Launching a run produces a durable artefact.

We snapshot:

- the world configuration (Universe as Code),
- the brain configuration (Brain as Code, all three layers),
- the runtime envelope (tick rate, curriculum schedule, seed),
- and a `full_cognitive_hash` computed over the snapshot and compiled cognition graph.

Every tick of behaviour is logged with that hash.

This provides:

- Reproducibility ("rerun the same mind in the same world and observe the same behavioural envelope")
- Accountability ("at tick 842, EthicsFilter blocked STEAL for mind hash 9af3c2e1 under world snapshot `austerity_nightshift_v3`")
- Teaching material ("module X proposed the action, panic overrode it, ethics vetoed it while the agent had limited resources")

### 1.4 Live Telemetry Shows Cognition, Not Just Vitals

The UI for a live run is not limited to meter readouts (energy 0.22, mood 0.41). It also shows:

- which high-level goal the agent is currently pursuing (SURVIVAL / THRIVING / SOCIAL),
- whether the panic controller overrode the normal goal during the tick (for example, "health critical"),
- whether EthicsFilter vetoed the chosen action and why ("attempted STEAL; forbidden"),
- planning depth ("world_model.rollout_depth: 6 ticks ahead"),
- whether the Social Model is active,
- the short form of the cognitive hash (so you know exactly which mind you're looking at).

Instructors can point to the panel and state, for example: energy fell below the 15 percent panic threshold, the agent attempted to steal food, EthicsFilter blocked the action, and the planner operated with a six-step horizon while the agent pursued SURVIVAL. Ethical constraints remain visible even under pressure.

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
