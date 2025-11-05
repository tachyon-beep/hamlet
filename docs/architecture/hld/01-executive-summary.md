# 1. Architectural Overview

**Document Type**: Overview
**Status**: Updated Draft
**Version**: 1.0 - Townlet Framework (UAC/BAC Architecture)
**Last Updated**: 2025-11-05
**Owner**: Principal Technical Advisor (AI)
**Parent Document**: TOWNLET_HLD.md

**Audience**: Engineers implementing the system, governance teams approving it
**Technical Level**: Intermediate (architectural concepts with technical depth)
**Estimated Reading Time**: 5 min for skim | 15 min for full read

---

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**:
The Townlet Framework architectural philosophy: treating agent minds and world rules as
explicit, auditable configuration (Brain as Code + Universe as Code) instead of hidden
implementation code. Townlet Town serves as the reference implementation (agents learning
survival in a simulated town environment).

**Why This Document Exists**:
Establishes the accountability and reproducibility guarantees that the Townlet Framework
provides for AI governance, research, and education through configuration-driven architecture.

**Who Should Read This**:
- **Must Read**: Engineers implementing the runtime engine, governance teams approving deployments
- **Should Read**: Researchers using the framework, instructors teaching with Townlet Town
- **Optional**: Developers implementing specific modules (read component specs instead)

**Reading Strategy**:
- **Quick Scan** (5 min): Read §1.1-§1.2 for core concepts (BAC, UAC, Software Defined Agents)
- **Partial Read** (10 min): Add §1.3-§1.4 for provenance and telemetry
- **Full Read** (15 min): Read all sections for complete governance picture

---

## Document Scope

**In Scope**:
- **Framework architecture**: Brain as Code (BAC) + Universe as Code (UAC) design philosophy
- **Four hard properties**: Explicit mind, explicit world, provenance, introspection
- **Value proposition**: Governance accountability, research reproducibility, educational transparency
- **Reference implementation**: Townlet Town survival simulation as canonical example

**Out of Scope**:
- **Implementation details**: See §2-§12 for component specifications
- **Code examples**: See component documents and codebase
- **Deployment procedures**: See operational documentation
- **Curriculum design**: See Townlet Town experiment-specific docs

**Critical Boundary**:
This document describes the **Townlet Framework** (reusable BAC/UAC architecture) as
demonstrated through **Townlet Town** (reference implementation: survival simulation).
Framework concepts are domain-agnostic; Townlet Town details (beds, jobs, energy bars)
are one specific universe instance. See [GLOSSARY.md](../GLOSSARY.md) for terminology.

---

## Position in Architecture

**Related Documents** (Read These First/Next):
- **Prerequisites**: None (this is the entry point to the HLD)
- **Builds On**: N/A (foundational document)
- **Related**: [GLOSSARY.md](../GLOSSARY.md) (framework vs instance terminology), [CLAUDE.md](../../CLAUDE.md) (project context)
- **Next**: [02-brain-as-code.md](02-brain-as-code.md) (detailed BAC specification)

**Section Number**: 1 / 12
**Architecture Layer**: Conceptual (establishes design philosophy)

---

## Keywords for Discovery

**Primary Keywords**: Brain as Code (BAC), Universe as Code (UAC), Software Defined Agent (SDA), accountability, reproducibility, provenance
**Secondary Keywords**: cognitive hash, configuration-driven, governance, telemetry, ethics, framework vs instance
**Subsystems**: agent, environment, runtime, telemetry, ethics, panic controller
**Design Patterns**: Software Defined X (SDN analogy), Configuration as Code, Snapshot+Hash provenance

**Quick Search Hints**:
- Looking for "why configuration-driven"? → See §1.1-§1.2
- Looking for "framework vs experiment"? → See "Document Scope" and [GLOSSARY.md](../GLOSSARY.md)
- Looking for "provenance and identity"? → See §1.3
- Looking for "telemetry and visibility"? → See §1.4
- Looking for "governance value"? → See §1.5

---

## Version History

**Version 1.0** (2025-11-05): Initial HLD release establishing Townlet Framework (BAC/UAC architecture) with Townlet Town as reference implementation

---

## Document Type Specifics

### For Overview Documents (Type: Overview)

**Strategic Question This Answers**:
"Why does the Townlet Framework treat brains and worlds as explicit configuration instead of
hidden code, and what governance/research/teaching benefits does this provide?"

**Key Takeaways**:
- Configuration-driven architecture makes agent cognition and world rules auditable
- Every run has provenance (snapshot + cognitive hash) enabling exact reproduction
- Live telemetry exposes cognitive processes (panic, ethics, planning) for transparency
- Framework supports any universe; Townlet Town demonstrates it for survival learning

**Mental Model**:
"Software Defined Agent" - the mind and world are configured like we configure networks (SDN)
or infrastructure (IaC). Change YAML files, change behavior. Snapshot configs, reproduce
behavior exactly. This shifts AI from 'black box mystery' to 'auditable system'."

---

## Framework vs Instance: Critical Distinction

**Two Conceptual Layers**:

1. **Townlet Framework** (Architecture - Reusable)
   - Brain as Code (BAC): Agent cognition as declarative YAML
   - Universe as Code (UAC): World rules as declarative YAML
   - Runtime engine: Materializes configurations into executable agents
   - Provenance system: Snapshots + cognitive hash for reproducibility
   - **Scope**: Domain-agnostic architecture (can model towns, factories, markets, etc.)

2. **Townlet Town** (Reference Implementation - Specific)
   - Universe instance: Agents learning to survive in a simulated town
   - Specific affordances: Bed, Hospital, Job, Fridge (14 total)
   - Specific bars: Energy, Health, Satiation, Money, Mood, Social, Fitness, Hygiene
   - Specific goals: SURVIVAL, THRIVING, SOCIAL
   - **Scope**: Educational experiment ("trick students into learning RL by playing The Sims")

**Reading Guide**: Framework concepts (BAC, UAC, cognitive hash) apply to any universe.
Townlet Town details (beds, energy bars, survival goals) are one experimental configuration.

**See**: [GLOSSARY.md](../GLOSSARY.md) for comprehensive framework vs instance terminology.

---

**END OF FRONT MATTER**
**BEGIN ARCHITECTURE CONTENT**

---

## 1. Architectural Overview

The Townlet Framework defines agents and worlds as audited configuration, not as "whatever the Python happened to be at the time."

The old Townlet/Hamlet agent was one opaque recurrent Q-network (`RecurrentSpatialQNetwork`) that turned partial observations into actions. It sort of worked, sometimes brilliantly, but it was a black box. If it sprinted to hospital and then fell asleep in the shower, the only honest answer to "why?" was "the weights felt like it."

The Townlet Framework v1.0 replaces that with a Software Defined Agent (SDA) running in a Software Defined World. We treat both the world and the mind as first-class content. We call this:

- **Universe as Code (UAC)**: The world, declared in data
- **Brain as Code (BAC)**: The mind, declared in data

If you need the full schematics, see §2 for BAC and §8 for UAC; those sections walk through the YAML layers and runtime interpretation.

Together, these provide four hard properties we did not have before:

### 1. The mind is explicit

BAC describes the agent's cognition in three YAML files: what faculties it has, how they're implemented, and how they think step by step. Panic response, ethics veto, planning depth, goal-selection bias – it's all on paper.

### 2. The world is explicit

UAC defines the universe (bars, affordances, costs, effects, public cues, operating hours, wage schedules) as data. Affordances like beds, jobs, hospitals are declared as configuration with per-tick bar effects, eliminating hidden "secret physics" deep in the environment loop.

**Framework note**: Affordances and bars are UAC patterns (framework-level). Specific affordances like "Bed" and bars like "Energy" are Townlet Town vocabulary (instance-level). Other universe instances could define "Assembly Line" affordances or "Fatigue" bars.

### 3. Every run is provenance-bound

When you run the framework, the platform snapshots both the world and brain configurations, hashes them (cognitive hash - unique fingerprint of the brain+world configuration), and stamps that identity onto every tick of telemetry. If the agent behaves unexpectedly, we can identify exactly which mind, under which world rules, produced the behavior. There is no "the AI just did that."

### 4. We can teach and audit, not just watch

We log not only what the body did ("health: 0.22"), but what the mind attempted, what the panic controller overrode, and what the ethics layer vetoed. We can answer "why" with evidence rather than conjecture.

**Impact**: The Townlet Framework shifts from "a neat RL simulation with emergent drama" to "an accountable cognitive system that can be diffed, replayed, and defended."

We achieve this by doing three things.

---

## 1.1 The Brain Is Now Assembled, Not Baked In

Earlier releases relied on a monolithic recurrent Q-network that attempted to handle perception, planning, social reasoning, panic, ethics, and action selection in a single block. The Townlet Framework decomposes the mind into explicit cognitive modules:

**Perception / Belief State Builder**
Transforms partial, noisy observations into an internal belief state.

**World Model**
Predicts state transitions for candidate actions, allowing the agent to learn the universe's dynamics, including non-stationary changes such as price shifts.

**Social Model**
Estimates the likely behavior and goals of nearby agents using the public cues that UAC exposes. It receives no hidden state at runtime.

**Hierarchical Policy**
Selects a strategic goal (framework pattern) and chooses the concrete action that advances that goal each tick.

**Townlet Town goals** (instance-specific): SURVIVAL (meet critical needs), THRIVING (optimize quality of life), SOCIAL (prioritize relationships). Other universe instances could define different goals (e.g., BUY/SELL/HOLD for trading agents).

**Panic Controller**
Overrides normal planning when survival thresholds fall below configured limits (framework pattern). Townlet Town configures thresholds for energy, health, satiation bars.

**EthicsFilter**
Applies the final compliance gate. It forbids actions that violate policy (framework pattern). Panic cannot bypass ethics - EthicsFilter is final.

**Framework design**: Modules and their interactions are declared in BAC configuration and materialized at runtime. We can explicitly disable the Social Model for ablation without touching shared policy code, adjust the planning horizon from two to six ticks via configuration, or introduce a new panic rule without retraining perception.

The Townlet Framework is therefore an engineered assembly rather than a single opaque network.

---

## 1.2 The World Is Now Declared, Not Hidden in Engine Logic

The environment is no longer an ad hoc Python ruleset. Core mechanics are defined in UAC configuration.

**Framework patterns**: Affordances (interactable objects with capacity, per-tick effects, costs, interrupt rules), bars (continuous state variables 0.0-1.0), cascades (bar relationships), public cues (visible signals for social reasoning).

**Townlet Town instance**: Defines 14 specific affordances (Bed, Hospital, Job, Fridge, Gym, Shower, Bar, Restaurant, Park, Phone, Mall, SocialEvent, Couch, Fridge), 8 bars (Energy, Health, Satiation, Money, Mood, Social, Fitness, Hygiene), ambulance pricing, operating hours, wage schedules.

UAC expresses the desired behavior; the runtime engine executes those behaviors deterministically.

**Example**: Questions such as "why did the agent pay $300 to call an ambulance instead of walking to hospital?" can be answered by inspecting the world configuration (immediate teleportation at high cost versus slower treatment with possible closing hours) alongside the brain configuration (panic thresholds permitting a survival override at 5% health).

The universe's physics and economy are reviewable. Affordances can reference whitelisted special effects (e.g., `teleport_to:hospital`), keeping the world spec expressive but bounded.

---

## 1.3 Every Run Now Has Identity and Chain of Custody

Launching a run produces a durable artifact.

**We snapshot**:
- **World configuration** (UAC)
- **Brain configuration** (BAC, all three layers)
- **Runtime envelope** (tick rate, curriculum schedule, seed)
- **Cognitive hash** (unique fingerprint computed over the snapshot and compiled cognition graph)

Every tick of behavior is logged with that hash.

**This provides**:

**Reproducibility**
"Rerun the same mind in the same world and observe the same behavioral envelope"

**Accountability**
"At tick 842, EthicsFilter blocked STEAL for mind hash 9af3c2e1 under world snapshot `austerity_nightshift_v3`"

**Teaching material**
"Module X proposed the action, panic overrode it, ethics vetoed it while the agent had limited resources"

**Framework benefit**: Provenance works for any universe instance, not just Townlet Town.

---

## 1.4 Live Telemetry Shows Cognition, Not Just Vitals

The UI for a live run is not limited to meter readouts (energy 0.22, mood 0.41). It also shows:

- **Current goal**: Which high-level goal the agent is pursuing (framework pattern; Townlet Town uses SURVIVAL/THRIVING/SOCIAL)
- **Panic override**: Whether the panic controller overrode the normal goal during the tick (e.g., "health critical")
- **Ethics veto**: Whether EthicsFilter vetoed the chosen action and why (e.g., "attempted STEAL; forbidden")
- **Planning depth**: How many ticks ahead the world model simulates (e.g., "world_model.rollout_depth: 6")
- **Social model status**: Whether Social Model is active
- **Cognitive hash**: Short form of the hash (so you know exactly which mind you're observing)

**Example narrative**: Instructors can point to the panel and state: "Energy fell below the 15% panic threshold, the agent attempted to steal food, EthicsFilter blocked the action, and the planner operated with a six-step horizon while the agent pursued SURVIVAL. Ethical constraints remain visible even under pressure."

This delivers the glass-box promise: The Townlet Framework shifts from passive observation to transparent cognition, survival heuristics, and ethics operating in public.

---

## 1.5 Why This Matters (Governance, Research, Teaching)

**Interpretability**
We can answer "which part of the mind did that and why" with evidence. Telemetry records whether panic overrode the policy or EthicsFilter vetoed a candidate action.

**Reproducibility**
Behavior is not anecdotal; it is a run folder with a configuration snapshot and hash that any reviewer can rehydrate.

**Accountability**
If something unsafe occurs, we examine which safety setting permitted it, which execution-graph step executed it, the relevant panic thresholds, and the governing world rules. The issue becomes diagnosable and auditable.

**Pedagogy / Curriculum**
Students, auditors, and policy teams can:
- Read the YAML configs to understand what the agent was authorized to do
- Diff successive versions of the mind or world to see exactly what changed
- Run controlled comparisons (same brain, different world; same world, different brain)
- Understand the framework as reusable architecture beyond Townlet Town

**Framework flexibility**: The architecture supports any universe. Townlet Town demonstrates it for survival learning. Future instances could model factory optimization, market trading, or multi-agent societies using the same BAC/UAC foundation.

---

**Summary**: The Townlet Framework treats the brain and the world as configuration, snapshots and hashes them at runtime, and exposes live introspection with governance veto logging. That is now the standard operating model.

---
