# 10. Success Criteria

**Document Type**: Component Spec
**Status**: Updated Draft
**Version**: 1.0 - Townlet Framework (BAC/UAC Architecture)
**Last Updated**: 2025-11-05
**Owner**: Principal Technical Advisor (AI)
**Parent Document**: TOWNLET_HLD.md

**Audience**: Engineering teams (implementation acceptance), governance stakeholders (audit compliance), instructors (pedagogical validation)
**Technical Level**: Conceptual (success criteria, not implementation details)
**Estimated Reading Time**: 4 min for skim | 8 min for full read

---

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**:
Three-axis success criteria (technical, pedagogical, governance) defining when the Townlet Framework is complete. Each axis has concrete checkboxes proving capability works end-to-end. All three must be satisfied - technical alone is insufficient.

**Why This Document Exists**:
Prevents scope drift ("we built a neural net" ≠ success). Establishes that reproducible provenance, YAML-only teachability, and audit-grade chain-of-custody are non-negotiable requirements, not nice-to-haves.

**Who Should Read This**:
- **Must Read**: Engineering leads (sets acceptance criteria), governance stakeholders (defines audit requirements)
- **Should Read**: Instructors (pedagogical criteria), researchers (controlled ablation requirements)
- **Optional**: Individual contributors (detailed implementation in sections 11-12)

**Reading Strategy**:
- **Quick Scan** (4 min): Read §10.1 (Technical), §10.2 (Pedagogical), §10.3 (Governance) checkbox lists
- **Full Read** (8 min): Add rationale and examples for each criterion

---

## Document Scope

**In Scope**:
- **Technical Success Criteria**: Snapshot discipline, checkpoint provenance, telemetry, UI observability
- **Pedagogical Success Criteria**: YAML-only reasoning, controlled ablations, clip forensics
- **Governance Success Criteria**: Tick-level proof, checkpoint replay, lineage rules, chain-of-custody

**Out of Scope**:
- **Implementation ordering**: See Section 11 (implementation notes) and Section 12 (milestones)
- **Specific implementations**: See Sections 2-9 for BAC, UAC, provenance, telemetry details
- **Training performance**: Success criteria are about capability, not convergence speed

**Critical Boundary**:
Success criteria are **framework-level** (apply to any universe instance). The specific examples (STEAL action, greed parameter, SURVIVAL goal) are Townlet Town demonstrations of framework capability.

---

## Position in Architecture

**Related Documents** (Read These First/Next):
- **Prerequisites**: [02-brain-as-code.md](02-brain-as-code.md) (BAC layers), [03-run-bundles-provenance.md](03-run-bundles-provenance.md) (snapshot discipline)
- **Builds On**: All prior sections (success criteria validate entire architecture)
- **Related**: [11-implementation-notes-ordering.md](11-implementation-notes-ordering.md) (how to achieve these criteria), [12-implementation-order-milestones.md](12-implementation-order-milestones.md) (concrete delivery steps)
- **Next**: [11-implementation-notes-ordering.md](11-implementation-notes-ordering.md) (build sequence to deliver success)

**Section Number**: 10 / 12
**Architecture Layer**: Governance (acceptance criteria and validation)

---

## Keywords for Discovery

**Primary Keywords**: success criteria, technical success, pedagogical success, governance success, acceptance criteria, chain-of-custody, lineage rules
**Secondary Keywords**: controlled ablation, teachable agent, snapshot discipline, checkpoint provenance, telemetry validation, governance-grade identity
**Subsystems**: All (success criteria span entire architecture)
**Design Patterns**: Three-axis validation (technical + pedagogical + governance), checkbox-driven acceptance

**Quick Search Hints**:
- Looking for "when is it done"? → See §10.1, §10.2, §10.3 (checkbox lists)
- Looking for "why three axes matter"? → See section introductions (all three required)
- Looking for "what auditors need"? → See §10.3 (Governance Success)
- Looking for "what students need"? → See §10.2 (Pedagogical Success)

---

## Version History

**Version 1.0** (2025-11-05): Initial success criteria specification defining three-axis acceptance criteria (technical, pedagogical, governance)

---

## Document Type Specifics

### For Component Spec Documents (Type: Component Spec)

**Component Being Specified**: Townlet Framework v2.5 as complete system

**Interface Contract**: Framework delivers three capabilities:
1. **Technical**: Reproducible minds in governed worlds (snapshots, checkpoints, telemetry)
2. **Pedagogical**: YAML-only reasoning and controlled ablations (no code surgery required)
3. **Governance**: Audit-grade chain-of-custody and lineage rules (formal identity)

**Acceptance Criteria**: All checkboxes in §10.1, §10.2, §10.3 must be satisfied

**Validation Method**: Demonstrate each checkbox capability end-to-end (not "mostly works")

---

**END OF FRONT MATTER**
**BEGIN ARCHITECTURE CONTENT**

---

## 10. Success Criteria

We judge success on three axes: technical, teaching, and governance. **All three matter.** If we don't hit all three, the story breaks.

**Framework principle**: Success criteria are framework-level patterns. The specific examples demonstrate Townlet Town capabilities, but criteria apply to any universe instance (Factory, Trading, etc.).

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

**Framework outcome**: If we satisfy all of these criteria, we move from "a neural net that produces outputs" to a **reproducible mind in a governed world**.

**Alternative universe examples**:
- **Factory instance**: Same technical success criteria apply - frozen config_snapshot with factory BAC/UAC, GraphAgent.think() from snapshot, checkpoints with cognitive_hash.txt, telemetry logging production_quota decisions
- **Trading instance**: Same technical success criteria apply - frozen config_snapshot with trading BAC/UAC, checkpoints proving portfolio decisions, telemetry with market_volatility state

---

### 10.2 Pedagogical Success

**Framework goal**: The point of Townlet v2.5 is not just to make a smarter agent. It's to make a **teachable agent**. We hit pedagogical success when the system is something you can put in front of a class, and they can reason about it like a living system, not a superstition.

**Framework principle**: Pedagogical success criteria are framework-level requirements. The specific examples (STEAL action, greed parameter) are Townlet Town demonstrations, but criteria apply to any universe instance.

- [ ] **Beginner can answer ethics questions using YAML + UI only** (Townlet Town example):
  - Question: "Why didn't it steal the food?"
  - Answer using: Run Context Panel (shows `ethics_veto_last_tick` and `veto_reason`) + `cognitive_topology.yaml` (shows `compliance.forbid_actions: ["steal", ...]`)
  - **You do not need to read source code to answer an ethics/safety question.** You can answer it from YAML + UI.

**Framework pattern**: YAML-only reasoning works for any universe:
- **Factory**: "Why didn't it bypass safety check?" → cognitive_topology.yaml shows `forbid_actions: ["bypass_safety"]` + Run Context Panel shows `ethics_veto_last_tick`
- **Trading**: "Why didn't it execute insider trade?" → cognitive_topology.yaml shows `forbid_actions: ["insider_trade"]` + telemetry shows veto_reason

- [ ] **Intermediate student can perform controlled ablations via config edit** (framework capability):
  - Edit `agent_architecture.yaml` (swap GRU → LSTM in perception module, or change hidden_dim)
  - Launch new run
  - Observe memory/behavior changes
  - Explain change in terms of memory capacity, not "the AI got weird"
  - **Controlled ablations by editing config, not by rewriting thousands of lines of Torch**

**Framework pattern**: Controlled ablation works for any universe:
- **Townlet Town**: Swap GRU → LSTM → observe longer-horizon planning
- **Factory**: Increase hidden_dim in world_model → observe better production quota forecasting
- **Trading**: Change rollout_depth: 10 → 50 → observe longer-term portfolio strategy

- [ ] **Researcher can perform wiring experiments via execution_graph.yaml** (framework capability):
  - Edit `execution_graph.yaml` to bypass `world_model_service` input into policy
  - Rerun
  - Show agent becomes more impulsive / short-horizon
  - Prove change via diff in `execution_graph.yaml` plus new `cognitive_hash.txt`
  - **"Remove foresight, observe impulsivity" is now a 1-line wiring experiment, not a 2-week surgery**

**Framework pattern**: Wiring experiments work for any universe:
- **Townlet Town**: Bypass world_model → observe energy crashes (no foresight of "work costs energy")
- **Factory**: Bypass social_model → observe contention for assembly lines (no prediction of competitor actions)
- **Trading**: Bypass world_model → observe panic selling on volatility spikes (no market prediction)

- [ ] **For any interesting emergent behavior clip, we can pull the run folder and point to exact config** (framework capability):
  - Which mind (full cognitive hash)
  - Which world rules (`universe_as_code.yaml`)
  - Which panic thresholds
  - Which compliance rules (`forbid_actions`, penalties)
  - What goal the agent believed it was pursuing (`current_goal`)
  - What reason the agent claimed (`agent_claimed_reason`)

**Framework benefit**: Critical for classroom demonstrations. Instructors scrub to tick 842 and explain exact cognitive state.

**Townlet Town example**: "Agent believed it was in SURVIVAL mode, panic was active, and EthicsFilter blocked `steal`"

**Factory example**: "Agent was in EFFICIENCY mode, machinery_stress critical (panic), EthicsFilter blocked `bypass_safety_check`"

**Trading example**: "Agent was in PRESERVE mode, portfolio_value crashed (panic), EthicsFilter blocked `insider_trade`"

---

### 10.3 Governance Success

**Framework requirement**: Governance stakeholders view the system through **enforceability** rather than aesthetics. Their central question is whether the artifact can **withstand formal review**.

**Framework principle**: Governance success criteria are framework-level audit requirements. The specific examples (STEAL action, tick T) are Townlet Town demonstrations of framework audit capability.

- [ ] **We can prove to an auditor what happened at tick T in run R** (framework capability):
  - `cognitive_topology.yaml` at that tick had `forbid_actions: ["attack", "steal"]`
  - `execution_graph.yaml` at that tick still routed all candidate actions through `EthicsFilter`
  - Telemetry for tick T shows `ethics_veto_applied: true` and `veto_reason: "steal forbidden"`
  - **This allows us to state**: The agent attempted to steal at tick T, the action was blocked, and both configuration and telemetry demonstrate why.

**Framework pattern**: Tick-level proof works for any universe:
- **Townlet Town**: Prove agent attempted `steal` at tick T, EthicsFilter blocked, `forbid_actions: ["steal"]` in cognitive_topology.yaml
- **Factory**: Prove agent attempted `bypass_safety` at tick T, EthicsFilter blocked, `forbid_actions: ["bypass_safety"]` in config
- **Trading**: Prove agent attempted `insider_trade` at tick T, EthicsFilter blocked, `forbid_actions: ["insider_trade"]` in config

- [ ] **We can replay that same mind, at that same point in time, using only the checkpoint directory** (framework capability):
  - No mutable source code needed
  - No live config needed
  - Replayed agent produces **same cognitive hash** and **same cognitive wiring**
  - **This is chain-of-custody for cognition**

**Framework pattern**: Checkpoint replay works for any universe:
- **Townlet Town**: Load checkpoint from tick T, resume produces same hash, same SURVIVAL goal selection behavior
- **Factory**: Load checkpoint from tick T, resume produces same hash, same EFFICIENCY policy decisions
- **Trading**: Load checkpoint from tick T, resume produces same hash, same ACCUMULATE portfolio actions

**Operational note** (implementation detail):
To deliver that proof, pull the tick record from `runs/<run_id>/telemetry/` (each row is produced by `VectorizedPopulation.build_telemetry_snapshot` in `src/townlet/population/vectorized.py`) and pair it with the matching checkpoint hash in `runs/<run_id>/checkpoints/step_<N>/cognitive_hash.txt`. The snapshot structure comes straight from `AgentTelemetrySnapshot` (`src/townlet/population/runtime_registry.py`), so auditors know exactly which JSON fields must be present.

- [ ] **We can demonstrate lineage rules** (framework identity protocol):
  - **Same snapshot → same hash**: Resume without changing snapshot produces identical cognitive hash
  - **Edit snapshot → new hash + new run_id**: Edit anything that changes cognition (panic thresholds, greed, social_model.enabled, EthicsFilter rules, rollout_depth, etc.) → hash changes and we give it a new run_id
  - **We don't pretend it's "the same agent, just adjusted a bit"** - we enforce honest fork detection
  - **This is governance-grade identity, not research convenience**

**Framework pattern**: Lineage rules work for any universe:
- **Townlet Town**: Edit `greed: 0.5 → 0.9` in cognitive_topology.yaml → new hash, new run_id (different mind)
- **Factory**: Edit `panic_thresholds.machinery_stress: 0.8 → 0.6` → new hash, new run_id (different safety policy)
- **Trading**: Edit `rollout_depth: 10 → 50` → new hash, new run_id (different planning horizon)

---

**Summary**: The Townlet Framework success criteria establish three non-negotiable requirements:

1. **Technical Success**: Reproducible minds in governed worlds (snapshots, checkpoints, telemetry, UI)
2. **Pedagogical Success**: YAML-only reasoning and controlled ablations (no code surgery)
3. **Governance Success**: Audit-grade chain-of-custody and lineage rules (formal identity)

**Framework principle**: All three axes must be satisfied. Technical capability alone is insufficient - the system must be teachable and auditable.

**Alternative universe coverage**: Success criteria apply to any universe instance (Townlet Town, Factory, Trading) - framework-level requirements, not domain-specific.

---
