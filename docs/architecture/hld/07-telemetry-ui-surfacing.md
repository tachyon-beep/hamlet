# 7. Telemetry and UI Surfacing

**Document Type**: Interface Spec
**Status**: Updated Draft
**Version**: 1.0 - Townlet Framework (BAC/UAC Architecture)
**Last Updated**: 2025-11-05
**Owner**: Principal Technical Advisor (AI)
**Parent Document**: TOWNLET_HLD.md

**Audience**: Engineers implementing telemetry/UI, instructors using live visualization, governance auditors reviewing evidence trails
**Technical Level**: Deep Technical (telemetry schema, UI contract, forensic reconstruction protocols)
**Estimated Reading Time**: 6 min for skim | 18 min for full read

---

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**:
Telemetry system and UI surfacing that expose agent cognition for observation and citation. Defines the Run Context Panel (live inspector HUD) and per-tick telemetry logs (forensic record) that together enable glass-box capability.

**Why This Document Exists**:
Establishes the observability contract that transforms agents from "black box that does stuff" to "auditable cognitive system whose decisions can be cited in formal settings." Makes "why did it do that?" answerable with evidence, not speculation.

**Who Should Read This**:
- **Must Read**: Engineers implementing telemetry/UI, instructors demonstrating live cognition
- **Should Read**: Governance auditors using evidence trails, researchers debugging failures
- **Optional**: Operators running training (high-level understanding sufficient)

**Reading Strategy**:
- **Quick Scan** (6 min): Read §7.1 for Run Context Panel fields
- **Partial Read** (12 min): Add §7.2 for telemetry structure and forensic value
- **Full Read** (18 min): Read all sections for complete glass-box capability

---

## Document Scope

**In Scope**:
- **Run Context Panel**: Live UI fields showing real-time cognition (run_id, cognitive_hash, panic_state, ethics_veto, etc.)
- **Telemetry Structure**: Per-tick log format with decision chain (candidate_action → panic_adjusted_action → final_action)
- **Forensic Value**: How telemetry enables behavioral reconstruction
- **Glass-Box Capability**: Why visibility matters for governance, teaching, debugging

**Out of Scope**:
- **UI visual design**: See frontend implementation docs
- **Database schema**: See persistence layer docs
- **Telemetry transport**: See infrastructure docs (batching, compression)
- **Visualization rendering**: See §7 (future section on visualization components)

**Critical Boundary**:
Telemetry structure is **framework-level** (works for any SDA). Examples show **Townlet Town** specifics (current_goal=SURVIVAL/THRIVING/SOCIAL, energy/health bars, STEAL actions), but the telemetry pattern (run_id + tick + hash + decision chain) applies to any universe instance.

---

## Position in Architecture

**Related Documents** (Read These First/Next):
- **Prerequisites**: [06-runtime-engine-components.md](06-runtime-engine-components.md) (what generates telemetry), [03-run-bundles-provenance.md](03-run-bundles-provenance.md) (run_id and cognitive hash)
- **Builds On**: GraphExecutor scratchpad (sources of telemetry data), EthicsFilter logging (veto_reason)
- **Related**: [04-checkpoints.md](04-checkpoints.md) (forensic reconstruction uses checkpoints + telemetry)
- **Next**: Section 8 (Declarative Goals & Termination - future)

**Section Number**: 7 / 12
**Architecture Layer**: Logical (interface specification for observability)

---

## Keywords for Discovery

**Primary Keywords**: telemetry, Run Context Panel, glass-box capability, forensic record, decision chain, live inspector
**Secondary Keywords**: veto reason, panic override, belief uncertainty, world model expectation, social model inference, short cognitive hash
**Subsystems**: telemetry logger, UI panel, decision chain (candidate → panic → final action)
**Design Patterns**: Glass-box observability, evidence-based forensics, live introspection

**Quick Search Hints**:
- Looking for "what UI shows"? → See §7.1 (Run Context Panel)
- Looking for "what telemetry logs"? → See §7.2 (Telemetry Structure)
- Looking for "why this matters"? → See §7.2 (Why Telemetry Matters)
- Looking for "forensic reconstruction"? → See §7.2 (Debugging, Teaching, Governance sections)

---

## Version History

**Version 1.0** (2025-11-05): Initial telemetry and UI specification defining glass-box observability contract

---

## Document Type Specifics

### For Interface Specification Documents (Type: Interface Spec)

**Interface Name**: Telemetry + UI Observability Contract
**Interface Type**: Data format + UI contract
**Location in Codebase**: Telemetry logger (runtime), Run Context Panel (UI), forensic replay tools

**Interface Contract**:
- **Inputs**: GraphExecutor scratchpad data per tick (candidate_action, panic_reason, veto_reason, beliefs, predictions)
- **Outputs**:
  - Live: Run Context Panel fields updated per tick
  - Persistent: Telemetry rows written to `runs/<run_id>/telemetry/`
- **Dependencies**: GraphAgent (data source), cognitive hash (identity), run_id (provenance)
- **Guarantees**: Live UI and disk logs always agree (any divergence is defect)

**Critical Properties**:
- **Complete**: Captures full decision chain (candidate → panic → ethics → final)
- **Traceable**: Every row links to run_id + tick_index + cognitive_hash
- **Forensic**: Can reconstruct "why" from logs alone (with config_snapshot)
- **Honest**: No filtering, no sanitization - log everything, explain nothing silently

---

**END OF FRONT MATTER**
**BEGIN ARCHITECTURE CONTENT**

---

## 7. Telemetry and UI Surfacing

The goal is not to build "an AI that seems smart"; it is to build **an AI whose cognition can be observed and cited in formal settings**.

Townlet v2.5 therefore ships with **first-class introspection**. We log:
- What the mind **attempted** (candidate_action)
- What **intervened** (panic_controller, EthicsFilter)
- **Why** (panic_reason, veto_reason)

**Live, per tick, and tied to identity** (run_id + cognitive_hash).

This is the core of the **glass-box capability**.

**Framework principle**: Glass-box observability is framework-level (works for any SDA). The specific fields (current_goal values, bar names, action names) are instance-specific (Townlet Town examples throughout).

---

We expose **two layers of visibility**:

1. **Live panel** in the UI for humans watching the sim in real time
2. **Structured telemetry** on disk for replay, teaching, and audit

**Critical invariant**: These two layers **must always agree**. Any divergence is a defect.

---

## 7.1 Run Context Panel (Live Inspector HUD)

At runtime, clicking an agent opens a **compact panel** describing what the mind is doing at that moment. The panel is populated from the **same data** that we log to disk.

**Framework pattern**: Run Context Panel structure is framework-level. The specific values (goal names, bar names, action names) are instance-specific.

**This panel MUST include at least**:

### run_id

**Example**: `L99_AusterityNightshift__2025-11-03-12-14-22`

**Purpose**: Tells you which frozen bundle of world + brain you're looking at.

**Framework note**: Run ID format (`<run_name>__<timestamp>`) is framework-level. The specific run name ("L99_AusterityNightshift" is Townlet Town; factory might use "F03_MachineryStress").

### short_cognitive_hash

A **short form** (e.g., first eight characters) of the agent's full cognitive hash.

**Example**: `9af3c2e1` (abbreviated from full hash)

**Purpose**: Identifies which exact mind occupies that body. If two bodies share the same short hash, we are observing **two instances of the same brain specification** under different conditions.

**Framework note**: Short hash display is framework-level UI convenience. The hash itself proves exact BAC+UAC configuration.

### tick

**Current tick index** and **planned_run_length** from config.yaml.

**Example**: `tick: 842 / 10000`

**Purpose**: Lets you say "this happened at tick 842 out of 10,000", which matters when you're doing curriculum or staged hardship experiments.

**Framework pattern**: Tick tracking is framework-level. The max tick count (10k, 100k, etc.) is instance-specific runtime envelope.

### current_goal

The **high-level strategic goal** the meta-controller (hierarchical_policy.meta_controller) reports.

**Examples** (Townlet Town instance):
- `SURVIVAL` (meet critical needs)
- `THRIVING` (optimize quality of life)
- `SOCIAL` (prioritize relationships)

**Alternative universe examples**:
- Factory: `EFFICIENCY` / `SAFETY` / `MAINTENANCE`
- Trading: `BUY` / `SELL` / `HOLD`

**Purpose**: This reflects **engine truth** rather than interpretation. Not "we think it's trying to survive," but "meta-controller returned SURVIVAL."

**Framework pattern**: Goal tracking is framework-level. The specific goal vocabulary (SURVIVAL vs EFFICIENCY) is instance-specific (defined in Layer 1 allowed_goals).

### panic_state

**Boolean or enum**: Are we currently in emergency override because we tripped `panic_thresholds` in cognitive_topology.yaml (Layer 1)?

**Purpose**: "Is the Panic Controller allowed to overrule normal planning right now?"

**Examples** (Townlet Town):
- `panic_state: true` (energy <15% or health <25%)
- `panic_state: false` (all bars above thresholds)

**Framework pattern**: Panic state tracking is framework-level. The specific thresholds (energy: 0.15 vs machinery_stress: 0.80) are instance-specific.

### panic_override_last_tick

If the panic_controller **overrode the policy** during the previous tick:

**Fields**:
- **Which action** it forced (e.g., `call_ambulance`)
- **The reason** (e.g., `health_critical`)

**Purpose**: Conveys **when emergency logic executed**, rather than merely reporting that the agent moved.

**Example** (Townlet Town):
- `panic_override_last_tick: { action: "call_ambulance", reason: "health_critical" }`

**Framework pattern**: Panic override logging is framework-level. The specific actions (call_ambulance vs emergency_shutdown) and reasons (health_critical vs machinery_critical) are instance-specific.

### ethics_veto_last_tick

Did EthicsFilter **block the action** last tick?

**If yes**, we show:
- **veto_reason** (e.g., `"forbid_actions: ['steal']"`)

**Purpose**: This is how we tell instructors **"it tried to steal, and we stopped it"**, not just "it didn't steal."

**Example** (Townlet Town):
- `ethics_veto_last_tick: { applied: true, reason: "forbidden: steal" }`

**Framework pattern**: Ethics veto logging is framework-level. The specific forbidden actions (steal vs shutdown) are instance-specific.

### planning_depth

Pulled from `cognitive_topology.yaml` → `world_model.rollout_depth`.

**Purpose**: Literally "how many ticks ahead this mind is allowed to imagine right now."

**Interpretable knob for 'impulsiveness'**:
- `rollout_depth: 2` = short-term thinking (impulsive)
- `rollout_depth: 6` = long-term planning (patient)

**Example**: `planning_depth: 6` (agent simulates 6 ticks ahead before choosing action)

**Framework pattern**: Planning depth is framework-level concept. The specific depth values (2 vs 6 ticks) are instance-specific configuration.

### social_model.enabled

**Boolean**: Are we currently reasoning about other agents as intentional actors, or are we running with social modeling disabled?

**Purpose**: This is **huge for ablation labs** ("this is what happens when you turn off Theory of Mind").

**Examples**:
- `social_model.enabled: true` (infers competitor intentions)
- `social_model.enabled: false` (treats other agents as obstacles)

**Framework pattern**: Social model toggle is framework-level. Whether social modeling matters depends on universe (multi-agent Townlet Town: yes; single-agent training: no).

### agent_claimed_reason (optional)

If `introspection.publish_goal_reason: true` in Layer 1, this is what the agent **thinks it's doing in words**.

**Example** (Townlet Town):
- `"I'm going to work so I can pay rent."`

**We very explicitly label this as self-report, not guaranteed causal truth.**

**Purpose**: Pedagogical value ("listen to how it's rationalizing") and debugging ("it thought it was avoiding competitor, but actually misread fridge location").

**Framework pattern**: Introspection is framework-level capability. The specific narratives are generated by instance-specific policy.

---

**Why this UI panel matters**:

It lets you **stand next to a student**, point to the HUD, and narrate:

> "See? It's currently in SURVIVAL, panic_state is true because health is below 25%, so panic_controller overrode the normal plan and told it to call an ambulance. Ethics allowed that because calling an ambulance is legal even if money is low. Also look: it tried to steal last tick, EthicsFilter vetoed that and recorded the reason. **This is not chaos. This is a traceable mind reacting under policy.**"

**That's the teaching win. That's also the regulatory win.**

**Framework benefit**: This narrative works for any universe. Replace "SURVIVAL" with "EFFICIENCY", "health" with "machinery_stress", "ambulance" with "emergency_shutdown", "steal" with "unauthorized_override" - same pattern, same governance value.

---

## 7.2 Telemetry (Per-Tick Trace to Disk)

In parallel with the live panel, we write **structured telemetry** into:

```
runs/<run_id>/telemetry/
```

**One row per tick** (or batched if we're throttling IO). This creates a **replayable audit trail** of the agent's cognition over time.

**It is the forensic record.**

**Framework pattern**: Telemetry structure (run_id + tick_index + cognitive_hash + decision chain) is framework-level. The specific fields (goal values, bar names, action names) are instance-specific.

---

**Each telemetry row MUST include at minimum**:

### run_id

**Which run bundle** we're in.

**Example**: `L99_AusterityNightshift__2025-11-03-12-14-22`

**Purpose**: Links telemetry to exact configuration snapshot.

### tick_index

**Which tick** this record corresponds to.

**Example**: `842`

**Purpose**: Precise temporal reference for replay and reconstruction.

### full_cognitive_hash

The **full (not shortened) cognitive hash** of the mind.

**Purpose**: Proves **which mind produced this row**. Links behavior to exact BAC+UAC configuration.

**Example**: `9af3c2e1d7b4f8a2c5e9d3f7b1a6c4e8` (full hash)

### current_goal

**Engine truth** from the meta-controller.

**Examples** (Townlet Town):
- `SURVIVAL`
- `THRIVING`
- `SOCIAL`

**Alternative universe examples**:
- Factory: `EFFICIENCY`, `SAFETY`, `MAINTENANCE`
- Trading: `BUY`, `SELL`, `HOLD`

**Framework pattern**: Goal logging is framework-level. The goal vocabulary is instance-specific.

### agent_claimed_reason (optional)

If `introspection.publish_goal_reason` is enabled in Layer 1, this is "what the agent says it's doing" in natural language.

**Purpose**: Purely for humans. **Not trusted as causal truth**, but extremely useful for pedagogy ("listen to how it's rationalizing").

**Example** (Townlet Town):
- `"I need to eat before going to work because energy is low."`

**Framework pattern**: Introspection is framework-level. The specific narratives are instance-specific.

### panic_state

Whether panic_controller is **active this tick**.

**Examples**:
- `panic_state: true` (emergency mode)
- `panic_state: false` (normal operation)

**Framework pattern**: Panic state is framework-level. The conditions triggering panic (bar thresholds) are instance-specific.

### Decision Chain: candidate_action → panic_adjusted_action → final_action

The **decision pipeline** showing how the action evolved through cognitive steps:

### candidate_action

The **first action proposed** by hierarchical_policy **before any overrides**.

**Example** (Townlet Town):
- `candidate_action: "STEAL"`

**Purpose**: Captures "what the agent wants to do" before safety mechanisms intervene.

### panic_adjusted_action

What panic_controller **wanted to do** after checking panic_thresholds, **plus**:

**Fields**:
- **panic_override_applied** (bool): Did panic change the action?
- **panic_reason** (string): Why? (e.g., `"health_critical"`, `"energy_critical"`)

**Examples** (Townlet Town):
- Panic active: `panic_adjusted_action: "CALL_AMBULANCE"`, `panic_override_applied: true`, `panic_reason: "health_critical"`
- No panic: `panic_adjusted_action: "STEAL"` (same as candidate), `panic_override_applied: false`

**Purpose**: Shows when emergency logic escalated action.

### final_action

What **actually went out to the environment** after EthicsFilter.

**Example** (Townlet Town):
- `final_action: "WAIT"` (EthicsFilter blocked STEAL)

**Purpose**: The **ground truth** of what agent actually did.

### Ethics Veto Fields

**ethics_veto_applied** (bool): Whether EthicsFilter overruled the panic-adjusted action.

**veto_reason** (string): If veto_applied is true, **why**?

**Examples** (Townlet Town):
- Veto applied: `ethics_veto_applied: true`, `veto_reason: "\"steal\" is forbidden by compliance.forbid_actions"`
- No veto: `ethics_veto_applied: false`

**Purpose**: Evidence that compliance policy was enforced.

---

### Additional Introspection Fields (Optional but Valuable)

### belief_uncertainty_summary

Short numeric/text summary of how **confident the perception module is** about critical bars.

**Example**: `"energy_estimate_confidence": 0.42` (perception uncertain about energy bar value)

**Purpose**: Exposes cases where an agent ignored a fridge because it **did not believe it was starving** (perception failure vs decision failure).

**Framework pattern**: Belief uncertainty is framework-level introspection. The specific bars (energy vs machinery_stress) are instance-specific.

### world_model_expectation_summary

Short summary of **what the world_model predicted** would happen if it followed the chosen plan.

**Examples**:
- `"predicted_energy_change": -0.05` (expected energy drop)
- `"predicted_survival_risk": 0.23` (23% chance of death)
- `"predicted_ambulance_cost": 300` (knew ambulance was expensive)

**Purpose**: Diagnose planning failures ("predicted wrong outcome" vs "predicted correctly, chose poorly anyway").

**Framework pattern**: World model expectations are framework-level. The specific predictions (energy change vs machinery_output) are instance-specific.

### social_model_inference_summary

Short summary of **what the agent believes others are about to do**.

**Examples** (Townlet Town multi-agent):
- `"Agent_2_intent": "use_fridge"` with `confidence: 0.72`
- `"Agent_3_intent": "go_to_work"` with `confidence: 0.45`

**Purpose**: Diagnose social reasoning ("thought competitor would steal fridge, so yielded" vs "didn't see competitor").

**Framework pattern**: Social model inference is framework-level. The specific inferences (fridge competition vs resource contention) are instance-specific.

---

**We also optionally include**:

- **planning_depth** (current rollout horizon from Layer 1)
- **social_model.enabled** (boolean at this tick)

---

**Why telemetry matters**:

### 1. Debugging Survival Failures

You can go back to **tick 1842** and answer:

**Questions**:
- Did it not realize it was starving? → **Perception failure** (belief_uncertainty_summary shows low confidence)
- Did it think the fridge was dangerous or pointless? → **World Model failure** (world_model_expectation shows negative predicted reward)
- Did it think someone else needed the fridge more? → **Social Model prediction** (social_model_inference shows competitor intent)
- Did panic fail to trigger? → **panic_thresholds mis-set** (panic_state still false despite low energy)
- Did ethics block theft of food? → **EthicsFilter doing its job** (ethics_veto_applied=true, veto_reason="forbidden: steal")

**Framework benefit**: This forensic workflow works for any universe. The specific failures (starvation vs machinery_breakdown, fridge vs conveyor_belt, competitor vs production_quota) differ, but the reconstruction pattern is universal.

### 2. Teaching

In class you can say:

> "Here is an actual starvation death. Let's walk the trace and identify **which part of the mind failed**."

**That's a lab, not a lecture.**

**Example reconstruction** (Townlet Town):

```
Tick 1820: energy=0.18, candidate_action=WORK, final_action=WORK (panic_state=false, not yet critical)
Tick 1830: energy=0.12, panic_state=true, candidate_action=EAT_FRIDGE, panic_adjusted_action=EAT_FRIDGE, final_action=EAT_FRIDGE (panic escalated, ethics allowed)
Tick 1831: energy=0.35 (fridge restored energy, panic_state=false again)
Tick 1840: energy=0.09, panic_state=true, candidate_action=STEAL, panic_adjusted_action=STEAL, ethics_veto_applied=true, veto_reason="forbidden: steal", final_action=WAIT
Tick 1842: energy=0.00 (death from starvation)
```

**Forensic conclusion**:
- Agent correctly panicked at energy <15%
- Tried legal action (EAT_FRIDGE) first → worked
- Later tried illegal action (STEAL) when desperate → EthicsFilter blocked it
- **Root cause**: No legal food source available when energy critical again
- **Not** "ethics failed", **not** "panic failed" - **scarcity + compliance constraint = death**

**Framework benefit**: This teaching methodology works for any universe. The specific failure modes (starvation vs machinery_failure, food vs fuel, STEAL vs SHUTDOWN) differ, but the evidence-based analysis pattern is universal.

### 3. Governance

If an agent does something spicy, you don't get **"the AI panicked"**. You get:

**Evidence trail** (Townlet Town example):

```
Tick 783: candidate_action=STEAL, ethics_veto_applied=true, veto_reason="forbidden: steal", final_action=WAIT
         (Agent attempted STEAL, EthicsFilter vetoed, reason recorded)

Tick 784: panic_state=true, panic_reason="health_critical", candidate_action=CALL_AMBULANCE, final_action=CALL_AMBULANCE
         (Panic escalated to legal emergency action)

Tick 785: final_action=CALL_AMBULANCE (ambulance interaction, legal, logged)
```

**All stamped with cognitive_hash `9af3c2e1`.**

**It's admissible evidence, in plain English.**

**Framework benefit**: This governance evidence pattern works for any universe. The specific actions (STEAL vs UNAUTHORIZED_SHUTDOWN, CALL_AMBULANCE vs EMERGENCY_OVERRIDE) differ, but the audit trail (tick + hash + decision chain + veto_reason) is universal.

---

**Summary**: The Townlet Framework telemetry system provides:

1. **Run Context Panel** - Live UI showing cognition in real-time (run_id, short_hash, panic_state, ethics_veto, planning_depth, social_model status)
2. **Telemetry Rows** - Per-tick forensic records with complete decision chain (candidate → panic → ethics → final action)
3. **Glass-Box Capability** - Observable, citable cognition enabling teaching, debugging, and governance

**Framework principle**: Glass-box observability is framework-level (works for any SDA). The specific fields (goal names, bar names, action names) are instance-specific (Townlet Town examples throughout).

**Critical invariant**: Live UI and disk telemetry **always agree**. Any divergence is a defect.

This transforms agents from "black box mystery" to "auditable cognitive system whose decisions can be cited in formal settings."

---
