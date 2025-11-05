# 11. Implementation Notes (Ordering)

**Document Type**: Design Rationale
**Status**: Updated Draft
**Version**: 1.0 - Townlet Framework (BAC/UAC Architecture)
**Last Updated**: 2025-11-05
**Owner**: Principal Technical Advisor (AI)
**Parent Document**: TOWNLET_HLD.md

**Audience**: Engineering teams (implementation leads, architects), technical project managers
**Technical Level**: Implementation (build sequence, dependency ordering, phasing)
**Estimated Reading Time**: 5 min for skim | 10 min for full read

---

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**:
Recommended build sequence for Townlet Framework implementation - six ordered steps establishing provenance foundation first (snapshots → GraphAgent → cognitive hash → checkpoints → telemetry/UI → panic/ethics). Explains why order matters and what breaks if you jump around.

**Why This Document Exists**:
Prevents retrofitting provenance after building components. "Duct-taping audit on later" never works - must establish snapshot discipline and cognitive hash from day one. Order matters: each step builds on provenance established by previous steps.

**Who Should Read This**:
- **Must Read**: Engineering leads (planning implementation), architects (understanding dependencies)
- **Should Read**: Individual contributors (context for why we do X before Y)
- **Optional**: Governance stakeholders (high-level understanding of phasing)

**Reading Strategy**:
- **Quick Scan** (5 min): Read step titles in §11.1-11.6 for build sequence overview
- **Full Read** (10 min): Add rationale ("Why this is first/second/etc.") for each step

---

## Document Scope

**In Scope**:
- **Build Sequence**: Six-step recommended ordering (snapshots → GraphAgent → hash → checkpoints → telemetry/UI → panic/ethics)
- **Dependency Ordering**: Why each step must come before the next
- **Anti-patterns**: What breaks if you jump around or skip steps

**Out of Scope**:
- **Concrete milestones**: See Section 12 (implementation order milestones with "definition of done")
- **Detailed implementations**: See Sections 2-9 for component specifications
- **Success criteria**: See Section 10 (acceptance criteria for completed system)

**Critical Boundary**:
Build sequence is **framework-level** guidance (applies to any universe instance). The specific examples (snapshot contains cognitive_topology.yaml, panic_controller blocks steal) are Townlet Town demonstrations.

---

## Position in Architecture

**Related Documents** (Read These First/Next):
- **Prerequisites**: [10-success-criteria.md](10-success-criteria.md) (what we're building toward), [03-run-bundles-provenance.md](03-run-bundles-provenance.md) (snapshot discipline)
- **Builds On**: All prior sections (build sequence implements entire architecture)
- **Related**: [12-implementation-order-milestones.md](12-implementation-order-milestones.md) (concrete delivery milestones)
- **Next**: [12-implementation-order-milestones.md](12-implementation-order-milestones.md) (specific checkpoints with definition of done)

**Section Number**: 11 / 12
**Architecture Layer**: Implementation (phasing and dependency management)

---

## Keywords for Discovery

**Primary Keywords**: build sequence, implementation ordering, snapshot discipline, provenance foundation, dependency order
**Secondary Keywords**: bootstrap sequence, phasing, anti-patterns, retrofitting, audit story
**Subsystems**: All (build sequence covers entire framework)
**Design Patterns**: Foundation-first ordering (provenance → components → capabilities)

**Quick Search Hints**:
- Looking for "what to build first"? → See §11.1 (Snapshot discipline first)
- Looking for "why order matters"? → Read "Why this is first/second/etc." rationale for each step
- Looking for "what breaks if I skip"? → See anti-patterns in each step's rationale
- Looking for "when to add hash"? → See §11.3 (Cognitive hash after GraphAgent, before checkpoints)

---

## Version History

**Version 1.0** (2025-11-05): Initial implementation ordering guidance defining six-step build sequence

---

## Document Type Specifics

### For Design Rationale Documents (Type: Design Rationale)

**Design Question Addressed**:
"In what order should we implement Townlet Framework components to establish provenance foundation without retrofitting?"

**Alternatives Considered**:
1. **Build components first, add provenance later** → **Rejected** (retrofitting audit never works, creates technical debt)
2. **Build all capabilities in parallel** → **Rejected** (dependencies unclear, integration chaos)
3. **Foundation-first ordering (snapshots → hash → checkpoints → capabilities)** → **Chosen** (provenance from day one, clear dependencies)

**Key Trade-offs**:
- **Chosen**: Slower initial progress (snapshot discipline before GraphAgent), but solid provenance foundation
- **Sacrificed**: Faster "demo something working" (can't show agents acting until GraphAgent pipeline complete)

**Decision Drivers**:
- **Governance requirement**: Audit story must be coherent from first tick (can't retrofit chain-of-custody)
- **Technical dependency**: Cognitive hash requires GraphAgent, checkpoints require hash, telemetry requires checkpoints
- **Risk mitigation**: Duct-taping provenance on later creates unfixable gaps in audit trail

---

**END OF FRONT MATTER**
**BEGIN ARCHITECTURE CONTENT**

---

## 11. Implementation Notes (Ordering)

This section is about **"what order do we do this in so we don't set ourselves on fire"**. It's the recommended build sequence for Townlet v2.5.

**Framework principle**: You do these in order. If you jump around, the audit story collapses and you'll end up duct-taping provenance on later, which never works.

**Framework requirement**: Build sequence applies to any universe instance - establish provenance foundation (snapshots, hash, checkpoints) before building universe-specific capabilities (goals, affordances, rewards).

### 11.1 Snapshot Discipline First

**Goal**: Lock down provenance from day one.

**Framework requirement**: Establish snapshot discipline before building any other components.

**Deliverables**:

- Create `configs/<run_name>/` with all 5 YAMLs:
  - `config.yaml` (runtime envelope)
  - `universe_as_code.yaml` (world rules)
  - `cognitive_topology.yaml` (BAC Layer 1 - behavioral contract)
  - `agent_architecture.yaml` (BAC Layer 2 - module implementations)
  - `execution_graph.yaml` (BAC Layer 3 - think-loop wiring)

- Write launcher so that "start run" immediately:
  - Creates `runs/<run_name>__<timestamp>/`
  - Copies 5 YAMLs byte-for-byte into `runs/<run_name>__<timestamp>/config_snapshot/`
  - Creates empty subdirs: `checkpoints/`, `telemetry/`, `logs/`

**Rules** (framework-level discipline):

- **Snapshot is a physical copy, not a symlink**
- **After launch, the live process never re-reads from `configs/<run_name>/`** - the snapshot is now truth
- **All provenance, audit, and replay logic assume the snapshot is the canonical contract** for that run

**Why this is first**:

- **Governance requirement**: If you don't freeze the world and the mind at launch, you can't prove anything later. Governance dies right here.
- **Technical dependency**: The rest of the system (factory, hashing, checkpoints) all builds on the assumption that the snapshot is the single source of truth.

**Framework pattern**: Snapshot discipline works for any universe instance:
- **Townlet Town**: Snapshot contains town-specific UAC (8×8 grid, Bed/Hospital affordances, energy/health bars, SURVIVAL goals)
- **Factory**: Snapshot contains factory-specific UAC (assembly lines, machinery_stress bars, EFFICIENCY/SAFETY goals)
- **Trading**: Snapshot contains trading-specific UAC (market feeds, portfolio_value bars, BUY/SELL/HOLD goals)

---

### 11.2 Build the Minimal GraphAgent Pipeline

**Goal**: Replace monolithic RL agent class with graph-driven brain that can think() once.

**Framework milestone**: First working "brain-from-YAML" - GraphAgent.think() ticks once using only config_snapshot.

**Deliverables** (framework components):

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

**For the first cut** (minimal viable implementation):

- `world_model_service` can just be a stub (pass through)
- `social_model_service` can return "disabled"
- `panic_controller` can just pass through
- `EthicsFilter` can just pass through

**Why this is second**:

- **Technical dependency**: Until you have a callable brain built from YAML + snapshot, you can't hash cognition, you can't checkpoint provenance, you can't expose the think loop, you can't do glass-box UI. **Everything else depends on this.**

**Framework pattern**: Minimal GraphAgent pipeline works for any universe instance:
- **Townlet Town**: GraphAgent.think() runs perception → hierarchical_policy (SURVIVAL goal selection) → panic_controller (stub) → EthicsFilter (stub) → action
- **Factory**: GraphAgent.think() runs perception → hierarchical_policy (EFFICIENCY goal selection) → panic_controller (stub) → EthicsFilter (stub) → action
- **Trading**: GraphAgent.think() runs perception → hierarchical_policy (BUY/SELL goal selection) → panic_controller (stub) → EthicsFilter (stub) → action

---

### 11.3 Cognitive Hash

**Goal**: Give the instantiated mind a provable identity.

**Framework milestone**: Generate `cognitive_hash.txt` for every run - unique fingerprint enabling exact reproduction and accountability.

**Implementation**: Cognitive hash generator (e.g., SHA-256) must deterministically cover:

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

**Framework principle**: If any of those change, the hash changes. That's the whole point. **You cannot secretly "just tweak panic thresholds" and pretend it's the same mind.**

**Why we do it here** (dependency ordering):

- **Hashing requires GraphAgent** (must compute hash after instantiation)
- **Checkpoints require hash** (must stamp checkpoints with identity)
- **Telemetry requires hash** (must log `full_cognitive_hash` every tick to prove "this exact mind did this")

**Framework pattern**: Cognitive hash works for any universe instance:
- **Townlet Town**: Hash covers Townlet-specific BAC/UAC (SURVIVAL goals, Bed affordances, energy bars, greed=0.5)
- **Factory**: Hash covers factory-specific BAC/UAC (EFFICIENCY goals, assembly_line affordances, machinery_stress bars, risk_tolerance=0.3)
- **Trading**: Hash covers trading-specific BAC/UAC (BUY/SELL goals, market_data_feed affordances, portfolio_value bars, patience=0.7)

---

### 11.4 Checkpoint Writer and Resume

**Goal**: Pause/replay/fork without lying to audit.

**Framework milestone**: Enable chain-of-custody for cognition - checkpoints with snapshot + hash, resume with lineage rules.

**Deliverables**:

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

**Resume Rules** (framework-level lineage protocol):

- **Resume never consults `configs/<run_name>/`** (only reads from checkpoint directory)
- **Resume loads only from checkpoint directory** (self-contained provenance)
- **Resume starts new run folder** `..._resume_<timestamp>/` with restored snapshot
- **Same snapshot → same hash**: If you haven't touched the snapshot, resumed brain produces identical cognitive hash

**Branching** (honest fork detection):

- **Edit snapshot → new hash + new run_id**: If you edit snapshot before resuming (change `panic_thresholds`, disable `social_model.enabled`, lower `greed`, change `rollout_depth`), that is a **fork**. New hash, new run_id. **We do not lie about continuity.**

**Framework benefits**:

- **Long training jobs** across interruptions (resume with same hash)
- **Honest ablations** ("same weights, same world, except panic disabled" = provable via hash diff)
- **True chain-of-custody** for behavior (checkpoint directory = complete evidence)

**Framework pattern**: Checkpoint/resume works for any universe instance:
- **Townlet Town**: Checkpoint → resume with same SURVIVAL policy if snapshot unchanged, new hash if greed edited
- **Factory**: Checkpoint → resume with same EFFICIENCY policy if snapshot unchanged, new hash if risk_tolerance edited
- **Trading**: Checkpoint → resume with same BUY/SELL policy if snapshot unchanged, new hash if patience edited

---

### 11.5 Telemetry and UI

**Goal**: Make cognition observable in real-time and scrubbable after the fact.

**Framework milestone**: Glass-box capability - expose internal cognitive processes for governance, teaching, and debugging.

**Two Deliverables** (framework observability components):

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

**Framework benefit**: At this stage the panel provides an **auditable narrative** - instructors can point to exact cognitive state and narrate decisions.

**Townlet Town example**: "Agent is in SURVIVAL, panic overruled the planner, EthicsFilter blocked `steal`, planning depth is six ticks, and agent claims 'I'm going to work for money.'"

**Factory example**: "Agent is in EFFICIENCY, machinery_stress critical, panic escalated to `emergency_shutdown`, EthicsFilter allowed (safety action legal), production quota dropped."

**Trading example**: "Agent is in PRESERVE, portfolio_value crashed, panic blocked aggressive `buy_dip` action, substituted defensive `hold_cash`, agent claims 'Buying opportunity.'"

---

### 11.6 Panic and Ethics For Real

**Goal**: Safety and survival must be enforced in-graph rather than remaining comments in YAML.

**Framework milestone**: Replace stub panic_controller and EthicsFilter with real implementations - safety becomes observable, auditable, and provable.

**Implementation** (replace stubs):

- `panic_controller`:

  - Reads `panic_thresholds` from Layer 1 (e.g. energy < 0.15)
  - Can override `candidate_action` with an emergency survival action (`call_ambulance`, `go_to_bed_now`, etc)
  - Emits `panic_override_applied` and `panic_reason`
  - Logged to telemetry and surfaced in the UI

- `EthicsFilter`:

  - Reads `forbid_actions` and `penalize_actions` from Layer 1 compliance
  - Blocks forbidden actions outright, substitutes something allowed, and emits `ethics_veto_applied` + `veto_reason`
  - Logged to telemetry and surfaced in UI

**Important** (framework-level veto hierarchy): **EthicsFilter is final.** Panic can escalate urgency, but panic cannot legalize a forbidden act. If panic tries `steal` as an emergency move, EthicsFilter still vetoes it. **Ethics wins.**

**By the end of this step**:

- **Panic is an explicit, logged controller** in the loop
- **Ethics is an explicit, logged controller** in the loop
- **Clean override chain**: hierarchical_policy → panic_controller → EthicsFilter → final_action

**Framework benefit**: At this point we can **brief governance stakeholders using the recorded override trace** rather than informal assurances.

**Framework pattern**: Panic and ethics enforcement works for any universe instance:
- **Townlet Town**: Panic escalates to `call_ambulance` when health critical, EthicsFilter still blocks `steal` even if desperate
- **Factory**: Panic escalates to `emergency_shutdown` when machinery_stress critical, EthicsFilter still blocks `bypass_safety_check` even if production quota failing
- **Trading**: Panic escalates to `preserve_capital` when portfolio_value crashed, EthicsFilter still blocks `insider_trade` even if losses mounting

**Summary**: The six-step build sequence establishes provenance foundation (snapshots → GraphAgent → hash → checkpoints) before adding capabilities (telemetry/UI → panic/ethics). **Order matters** - duct-taping provenance on later never works.

---
