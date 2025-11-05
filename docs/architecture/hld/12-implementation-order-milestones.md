# 12. Implementation Order (Milestones)

**Document Type**: Component Spec
**Status**: Updated Draft
**Version**: 1.0 - Townlet Framework (BAC/UAC Architecture)
**Last Updated**: 2025-11-05
**Owner**: Principal Technical Advisor (AI)
**Parent Document**: TOWNLET_HLD.md

**Audience**: Engineering teams (sprint planning, delivery tracking), project managers (milestone acceptance)
**Technical Level**: Implementation (concrete deliverables, acceptance criteria, definition of done)
**Estimated Reading Time**: 4 min for skim | 8 min for full read

---

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**:
Six concrete delivery milestones translating Section 11's conceptual build sequence into testable checkpoints. Each milestone has "definition of done" checkboxes proving capability works end-to-end (snapshots → GraphAgent → hash → checkpoints → telemetry/UI → panic/ethics).

**Why This Document Exists**:
Translates conceptual ordering (Section 11) into engineering deliverables with acceptance criteria. Enables sprint planning, progress tracking, and milestone acceptance. "Definition of done" prevents partial implementations - each milestone must be demonstrable.

**Who Should Read This**:
- **Must Read**: Engineering teams (sprint planning), project managers (tracking delivery)
- **Should Read**: Engineering leads (understanding dependencies), governance stakeholders (delivery timeline)
- **Optional**: Individual contributors implementing specific components

**Reading Strategy**:
- **Quick Scan** (4 min): Read milestone titles and "definition of done" checkboxes
- **Full Read** (8 min): Add "why it matters" rationale for each milestone

---

## Document Scope

**In Scope**:
- **Six Milestones**: Snapshots/run folders, minimal GraphAgent, cognitive hash, checkpoint writer/resume, telemetry/UI, panic/ethics live
- **Definition of Done**: Acceptance criteria checkboxes for each milestone
- **Why It Matters**: Rationale for each milestone's importance

**Out of Scope**:
- **Conceptual ordering**: See Section 11 (implementation notes explaining why order matters)
- **Detailed component specs**: See Sections 2-9 for BAC, UAC, provenance, telemetry details
- **Success criteria**: See Section 10 (overall system acceptance criteria)

**Critical Boundary**:
Milestones are **framework-level** deliverables (apply to any universe instance). The specific examples (cognitive_topology.yaml, SURVIVAL goal) are Townlet Town demonstrations of framework capability.

---

## Position in Architecture

**Related Documents** (Read These First/Next):
- **Prerequisites**: [11-implementation-notes-ordering.md](11-implementation-notes-ordering.md) (conceptual build sequence), [10-success-criteria.md](10-success-criteria.md) (overall acceptance criteria)
- **Builds On**: All prior sections (milestones implement entire architecture)
- **Related**: None (final HLD section)
- **Next**: None (this completes the HLD specification)

**Section Number**: 12 / 12 (Final Section)
**Architecture Layer**: Implementation (delivery tracking and milestone acceptance)

---

## Keywords for Discovery

**Primary Keywords**: milestones, definition of done, acceptance criteria, delivery checkpoints, implementation deliverables
**Secondary Keywords**: snapshot milestone, GraphAgent milestone, cognitive hash milestone, checkpoint milestone, telemetry milestone, panic/ethics milestone
**Subsystems**: All (milestones cover entire framework)
**Design Patterns**: Checkbox-driven acceptance (concrete, demonstrable deliverables)

**Quick Search Hints**:
- Looking for "what to deliver first"? → See §12.1 (Snapshots and run folders)
- Looking for "when is hash ready"? → See §12.3 (Cognitive hash milestone)
- Looking for "checkpoint acceptance criteria"? → See §12.4 (Checkpoint writer and resume)
- Looking for "when are we done"? → See §12.6 (Panic and ethics go live - final milestone)

---

## Version History

**Version 1.0** (2025-11-05): Initial milestone specification defining six delivery checkpoints with acceptance criteria

---

## Document Type Specifics

### For Component Spec Documents (Type: Component Spec)

**Component Being Specified**: Townlet Framework v2.5 as incremental delivery sequence

**Interface Contract**: Each milestone delivers one aspect of provenance or observability, with testable acceptance criteria

**Acceptance Criteria**: Definition of done checkboxes for each of six milestones

**Validation Method**: Demonstrate each checkbox capability end-to-end (not "mostly works")

---

**END OF FRONT MATTER**
**BEGIN ARCHITECTURE CONTENT**

---

## 12. Implementation Order (Milestones)

**Framework principle**: Section 11 outlined the conceptual order of operations. Section 12 translates that ordering into **concrete delivery milestones** for engineering, curriculum, safety, and audit teams. These steps form the boot sequence.

**Framework requirement**: Milestones apply to any universe instance - establish provenance infrastructure (snapshots, hash, checkpoints) before building universe-specific features (goals, affordances, rewards).

### 12.1 Milestone: Snapshots and Run Folders

**Framework milestone**: Establish snapshot discipline - freeze config at launch for provenance.

**Definition of Done**:

- [ ] `configs/<run_name>/` exists with all 5 YAMLs (config.yaml, universe_as_code.yaml, cognitive_topology.yaml, agent_architecture.yaml, execution_graph.yaml)
- [ ] Launching a run generates `runs/<run_name>__<timestamp>/`
- [ ] `runs/<run_name>__<timestamp>/config_snapshot/` is a byte-for-byte copy of those YAMLs
- [ ] `checkpoints/`, `telemetry/`, `logs/` directories are created
- [ ] Runtime never re-reads mutable config after snapshot (enforced in code)

**Why It Matters**:

- **Hard provenance from the first tick** - governance foundation established
- **Snapshot is evidence**: We can point to "this is the world and brain we actually ran", not "what we think is close"

**Framework pattern**: Snapshot milestone works for any universe instance:
- **Townlet Town**: Snapshot contains town-specific BAC/UAC (SURVIVAL goals, Bed affordances)
- **Factory**: Snapshot contains factory-specific BAC/UAC (EFFICIENCY goals, assembly_line affordances)
- **Trading**: Snapshot contains trading-specific BAC/UAC (BUY/SELL goals, market_data_feed affordances)

### 12.2 Milestone: Minimal GraphAgent Pipeline

**Framework milestone**: First working "brain-from-YAML" - GraphAgent.think() ticks once.

**Definition of Done**:

- [ ] `factory.py` can build all declared modules from the snapshot
- [ ] `graph_executor.py` can compile `execution_graph.yaml` into a callable loop
- [ ] `graph_agent.py` exposes `think(raw_observation, prev_recurrent_state) -> { final_action, new_recurrent_state }`
- [ ] We can tick once end-to-end with stub panic_controller and stub EthicsFilter

**Why It Matters**:

- **"Brain is data" becomes running code** - not a slogan, actual execution
- **Proves BAC works**: Declarative mind configuration successfully materializes into callable agent

**Framework pattern**: Minimal GraphAgent works for any universe instance:
- **Townlet Town**: GraphAgent.think() runs SURVIVAL goal selection → stub panic → stub ethics → movement action
- **Factory**: GraphAgent.think() runs EFFICIENCY goal selection → stub panic → stub ethics → assembly_line action
- **Trading**: GraphAgent.think() runs BUY/SELL goal selection → stub panic → stub ethics → market action

### 12.3 Milestone: Cognitive Hash

**Framework milestone**: Generate provable identity for every mind - unique fingerprint enabling exact reproduction.

**Definition of Done**:

- [ ] We can generate `cognitive_hash.txt` for a run
- [ ] The hash covers:
  - All 5 YAMLs from snapshot
  - Compiled execution graph wiring
  - Instantiated module architectures / dims / optimizer LRs
- [ ] Telemetry and checkpoints now both include that hash

**Why It Matters**:

- **Mind identity for audit**: Provable fingerprint you can take to governance stakeholders
- **Honest mutation detection**: You can't quietly mutate cognition without changing the hash

**Framework pattern**: Cognitive hash works for any universe instance:
- **Townlet Town**: Hash changes if SURVIVAL termination threshold edited, or greed parameter changed
- **Factory**: Hash changes if EFFICIENCY goal modified, or machinery_stress panic threshold edited
- **Trading**: Hash changes if BUY/SELL logic altered, or portfolio_value risk threshold edited

### 12.4 Milestone: Checkpoint Writer and Resume

**Framework milestone**: Enable chain-of-custody for cognition - checkpoints with snapshot + hash, resume with lineage rules.

**Definition of Done**:

- [ ] We can dump checkpoints at `step_<N>/` with:
  - weights.pt (all module weights)
  - optimizers.pt (optimizer states)
  - rng_state.json (reproducible RNG)
  - config_snapshot/ (frozen world + mind)
  - cognitive_hash.txt (identity fingerprint)
- [ ] We can resume into a brand new run folder using only a checkpoint subfolder
- [ ] **Same snapshot → same hash**: If we don't change the snapshot on resume, the resumed run reports the same cognitive hash
- [ ] **Edit snapshot → new hash + new run_id**: If we edit the snapshot before resume (panic thresholds, forbid_actions, greed, rollout_depth), the resumed run reports a new hash and a new run_id

**Why It Matters**:

- **Chain-of-custody for cognition**: Provenance trail from launch through checkpoints to resume
- **Honest fork detection**: Controlled forks are now explicit, not sneaky (lineage rules enforced)

**Framework pattern**: Checkpoint/resume works for any universe instance:
- **Townlet Town**: Resume with same hash if unchanged, new hash if greed edited (0.5 → 0.9)
- **Factory**: Resume with same hash if unchanged, new hash if machinery_stress panic threshold edited
- **Trading**: Resume with same hash if unchanged, new hash if rollout_depth edited (10 → 50 steps)

### 12.5 Milestone: Telemetry and UI

**Framework milestone**: Glass-box capability - expose internal cognitive processes for governance, teaching, and debugging.

**Definition of Done**:

- [ ] **Telemetry per tick logs**:
  - run_id, tick_index, full_cognitive_hash
  - current_goal (engine truth)
  - agent_claimed_reason (self-report, if enabled)
  - panic_state
  - candidate_action, panic_adjusted_action (+ panic_reason)
  - final_action
  - ethics_veto_applied (+ veto_reason)
  - planning_depth, social_model.enabled
  - Short summaries of internal beliefs/expectations

- [ ] **The Run Context Panel renders live**:
  - run_id, short_cognitive_hash (pretty form)
  - tick / planned_run_length
  - current_goal, panic_state
  - planning_depth, social_model.enabled
  - panic_override_last_tick (+ panic_reason)
  - ethics_veto_last_tick (+ veto_reason)
  - agent_claimed_reason (if introspection.publish_goal_reason)

**Why It Matters**:

- **Teaching becomes possible**: Students can reason about behavior using observable cognition, not superstition
- **Governance reviews become visual**: Auditors see override traces in UI, not adversarial speculation

**Framework pattern**: Telemetry/UI works for any universe instance:
- **Townlet Town**: UI shows "SURVIVAL goal, panic=true (health critical), ethics blocked STEAL"
- **Factory**: UI shows "EFFICIENCY goal, panic=true (machinery critical), ethics blocked BYPASS_SAFETY"
- **Trading**: UI shows "PRESERVE goal, panic=true (portfolio crashed), ethics blocked INSIDER_TRADE"

### 12.6 Milestone: Panic and Ethics Go Live

**Framework milestone**: Safety and survival become observable, auditable, and provable - replace stubs with real enforcement.

**Definition of Done**:

- [ ] `panic_controller` actually overrides `candidate_action` when bars cross panic_thresholds
- [ ] `EthicsFilter` actually vetoes forbidden actions and substitutes a safe fallback
- [ ] Both write structured reasons (`panic_reason`, `veto_reason`) into telemetry and show in UI
- [ ] Both steps are present and ordered in `execution_graph.yaml`: policy → panic_controller → EthicsFilter
- [ ] **EthicsFilter is final authority** (panic cannot legalize forbidden acts)

**Why It Matters**:

- **Safety becomes observable**: Survival urgency and ethical constraint are now explicit, inspectable modules in the think loop (not implicit reward-shaping heuristics)
- **Auditable trace**: You can show "panic tried X, ethics said no" as provable evidence with cognitive hash

**Framework pattern**: Panic and ethics enforcement works for any universe instance:
- **Townlet Town**: Panic escalates to CALL_AMBULANCE (health critical), EthicsFilter still blocks STEAL (even if desperate)
- **Factory**: Panic escalates to EMERGENCY_SHUTDOWN (machinery critical), EthicsFilter still blocks BYPASS_SAFETY (even if production failing)
- **Trading**: Panic escalates to PRESERVE_CAPITAL (portfolio crashed), EthicsFilter still blocks INSIDER_TRADE (even if losses mounting)

---

**Summary**: The six milestones establish provenance infrastructure (snapshots → GraphAgent → hash → checkpoints) before adding capabilities (telemetry/UI → panic/ethics). **Each milestone must be demonstrable** - no partial implementations.

**Framework principle**: Milestones apply to any universe instance. The specific examples (SURVIVAL goal, Bed affordance, STEAL action) are Townlet Town demonstrations of framework-level delivery checkpoints.

---
