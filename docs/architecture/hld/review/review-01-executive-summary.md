# 1. Executive Summary

**Document Type**: Overview (Executive Summary)
**Status**: Draft
**Version**: 2.5
**Last Updated**: 2025-11-05
**Owner**: Architecture Team
**Parent Document**: Townlet v2.5 High Level Design (Enhanced Curriculum)

**Audience**: Researchers, Educators, Policy Analysts, Engineers
**Technical Level**: Executive to Intermediate
**Estimated Reading Time**: 10 minutes for skim | 25 minutes for deep read

---

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**:
This document provides a high-level overview of the Townlet Framework v2.5, a configuration-driven multi-agent reinforcement learning platform that treats both worlds (Universe as Code) and minds (Brain as Code) as auditable, reproducible content rather than opaque Python code.

**Why This Document Exists**:
Townlet solves three converging problems: (1) researchers need reproducible RL experiments without custom coding, (2) educators need pedagogical environments where students learn by experimentation, and (3) governance stakeholders need auditable AI systems with provable safety guarantees. This architecture achieves all three through YAML-configured worlds, declarative cognition, and glass-box telemetry.

**Who Should Read This**:
- **Must Read**: Project leads, new team members, stakeholders evaluating Townlet for adoption
- **Should Read**: Researchers considering Townlet for experiments, educators designing curriculum
- **Optional**: Students learning RL concepts, policy analysts evaluating AI governance approaches

**Reading Strategy**:
- **Quick Scan** (5 min): Read sections 1.1, 1.2, and 1.4 for core value proposition
- **Partial Read** (10 min): Add section 1.3 to understand technical differentiators
- **Deep Study** (25 min): Read entire document for complete strategic understanding

---

## Document Scope

**In Scope**:
- Strategic vision and value proposition for Townlet Framework
- Core innovation (no-cheating principle: sparse rewards, human-perceivable observations)
- High-level differentiators vs. other RL platforms, multi-agent systems, and survival sims
- Target audiences and use cases (research, education, governance)
- Document roadmap for navigating the full HLD

**Out of Scope**:
- Technical implementation details (see Section 3-7)
- Specific curriculum level designs (see Section 3)
- Observation space specifications (see Section 4)
- Code-level API documentation (see implementation guides)

**Boundaries**:
This document establishes the "why" and "what" at executive level. Subsequent sections provide the "how" (design principles, reward architecture, curriculum stages) and "when" (blockers, action plans).

---

## Position in Architecture

**Related Documents** (Read These First/Next):
- **Prerequisites**: None (this is the entry point)
- **Builds On**: None (foundational document)
- **Related**:
  - Section 2: Design Principles (philosophical foundations)
  - Section 3: Curriculum Design (L0-L8 progression)
  - Section 9: Platform Vision (research directions)
- **Next**: Section 2 (Design Principles) for deeper philosophical grounding

**Section Number**: 1 / 11
**Architecture Layer**: Conceptual

---

## Keywords for Discovery

**Primary Keywords**: Townlet Framework, Universe as Code (UAC), Brain as Code (BAC), Multi-agent RL, Configuration-driven
**Secondary Keywords**: Sparse rewards, Human observer principle, Cognitive provenance, Glass-box telemetry, Curriculum learning
**Subsystems**: Environment, Agent, Curriculum, Exploration, Population
**Design Patterns**: Interpreter (YAML → runtime), Strategy (curriculum stages), Observer (telemetry)

**Quick Search Hints**:
- Looking for "what makes Townlet different"? → See section 1.3
- Looking for "who should use this"? → See section 1.4
- Looking for "how to navigate the docs"? → See section 1.5
- Looking for "reward structure"? → See section 1.2 (summary) or Section 2 (detailed)
- Looking for "curriculum levels"? → See section 1.1 (overview) or Section 3 (detailed)

---

## Version History

**Current Version**: 2.5 - 2025-11-05
Complete rewrite transforming PASS 1 review notes into professional architecture document. Added complete frontmatter, renumbered from Section 0 to Section 1, aligned terminology with glossary.

**Previous Versions**:
- **2.0** (2025-11-04): PASS 1 review notes (working draft, informal tone)
- **1.0** (2025-10-30): Initial curriculum design notes (pre-UAC/BAC architecture)

**Deprecation Notice**: None (current version)

---

## Document Type Specifics

### For Overview Documents (Type: Overview)

**Strategic Question This Answers**:
"What is Townlet, why does it exist, and who should care about it?"

**Key Takeaways** (3-5 bullets):
- Townlet is a configuration-driven RL platform where worlds (UAC) and minds (BAC) are YAML, not code
- The "no cheating" principle enforces human-realistic constraints: sparse rewards, observable cues only, no telepathy
- Three coordinated layers provide reproducibility: Universe as Code, Brain as Code, and Provenance by Design
- Target audiences span research (hypothesis testing), education (learning RL concepts), and governance (auditable AI)
- An 8-level curriculum (L0-L8) progressively removes scaffolding from god-mode to emergent family communication

**Mental Models**:
Think of Townlet as "The Sims meets scientific computing" — a survival simulator where agents learn from scratch, but every world rule, agent architecture, and cognitive decision is frozen, hashed, and auditable like a blockchain transaction.

---

## Diagrams and Visuals

**Diagrams Included**:
- None in this section (executive summary is text-only)

**External Visual References**:
- See Section 3 for curriculum progression diagrams
- See Section 4 for observation space evolution diagrams
- See Section 6 for glass-box telemetry flow diagrams

---

## Open Questions and Future Work

**Unresolved Design Questions**:
- Should we support non-survival domains (e.g., factory optimization, trading bots)?
- What's the right balance between YAML expressiveness and configuration complexity?

**Planned Enhancements**:
- Section 7: Missing BAC Layer 3 specification (execution graphs)
- Section 8: Population genetics and dynasty inheritance mechanics

**Known Limitations**:
- Current implementation (v2.0) lacks full BAC support (Layer 1 specified, Layers 2-3 in progress)
- Cognitive hashing not yet implemented (provenance design complete, implementation pending)

---

## References

**Related Architecture Docs**:
- Section 2 (Design Principles): Philosophical foundations for no-cheating principle
- Section 3 (Curriculum Design): Detailed L0-L8 progression specifications
- Section 9 (Platform Vision): Long-term research directions

**Related Tasks**:
- TASK-000: UAC Action Space (move hardcoded actions to YAML)
- TASK-001: UAC Contracts (DTO-based schema validation)
- TASK-002: Universe Compilation Pipeline (cross-file validation)

**Code References**:
- `src/townlet/environment/vectorized_env.py` - Environment implementation
- `src/townlet/agent/networks.py` - Q-network architectures (SimpleQNetwork, RecurrentSpatialQNetwork)
- `src/townlet/demo/runner.py` - Main training orchestrator (temporary location)

**External References**:
- OpenAI Gym: Comparison baseline for RL environments
- SMAC (StarCraft Multi-Agent Challenge): Comparison for multi-agent systems
- The Sims franchise: Inspiration for survival mechanics

---

**END OF FRONT MATTER**
**BEGIN ARCHITECTURE CONTENT BELOW**

---

## SECTION 1: EXECUTIVE SUMMARY

### 1.1 What Townlet Actually Is

The Townlet Framework v2.5 is a configuration-driven multi-agent reinforcement learning platform that treats both worlds and minds as auditable, reproducible content rather than opaque code.

**The core insight**: If you can express a problem as meters (state variables), cascades (coupling dynamics), and affordances (available actions), you can simulate it. The learning agent discovers survival strategies through sparse rewards and long-horizon planning, without dense reward shaping or privileged information access.

The Townlet Framework provides three coordinated layers:

1. **Universe as Code (UAC)**: World physics defined in YAML
   - `bars.yaml` — survival meters (energy, health, money, mood, etc.)
   - `cascades.yaml` — how neglect propagates (hunger → health decay)
   - `affordances.yaml` — available actions (Bed, Job, Hospital, etc.)
   - `cues.yaml` — social observability (visible body language, not telepathy)

2. **Brain as Code (BAC)**: Agent cognition defined in YAML
   - Layer 1: `cognitive_topology.yaml` — behavior contract (what it's allowed to do)
   - Layer 2: `agent_architecture.yaml` — neural blueprints (how it's built)
   - Layer 3: `execution_graph.yaml` — reasoning loop (step-by-step cognition)

3. **Provenance by Design**: Every run is a frozen, auditable artifact
   - Snapshot immutability (configs copied on launch, never re-read)
   - Cognitive hashing (mind fingerprint proving "which brain did what")
   - Glass-box telemetry (candidate action → panic override → ethics veto → final action)
   - Signed checkpoints (tamper protection)

The platform ships with an 8-level curriculum (L0–L8) that progressively removes scaffolding: from god-mode survival (full observability) to emergent family communication (coordinated signaling without pre-shared semantics).

---

### 1.2 The Core Innovation: No Cheating

Most RL simulators give agents information humans wouldn't have (omniscience, telepathy, dense action rewards). The Townlet Framework enforces a strict **"human observer principle"**: if a human standing next to the agent couldn't perceive it, the agent shouldn't get it at inference time.

This manifests in four design choices:

#### 1. **Sparse Reward: User-Definable via UAC**

The reward function is configured in YAML, not hardcoded. The reference implementation ("Townlet Town") uses a minimalist product-of-meters approach:

```python
# Reference implementation (townlet_town.yaml)
r_t = bars[energy] * bars[health]  # range [0, 1]
```

**What this forces**:

- No dense shaping ("good job eating!", "nice work sleeping!")
- Agents must discover that Fridge → satiation ↑ → cascade prevented → health maintained
- Credit assignment over 50+ ticks (Job → money → Fridge → survival)

**Human equivalent**: "I feel good when I'm energized and healthy." You don't get a reward notification for eating; you just stop feeling hungry.

**Why UAC matters**: Researchers can experiment with alternative reward structures (e.g., `r = sqrt(energy * health * mood)`) by editing YAML, not Python. This enables rapid hypothesis testing without code changes.

#### 2. **Terminal Retirement Bonus**

Episodes end naturally when agents reach max age (the "age" bar hits 1.0, representing retirement). Terminal reward depends on:

- Lifespan completion (did you make it to retirement?)
- Wellbeing at retirement (health, energy, mood, social, hygiene)
- Wealth at retirement (diminishing returns via sqrt)

```python
if age >= 1.0:  # retirement
    terminal_reward = (
        0.5 * (ticks_survived / max_age) +      # lifespan
        0.3 * mean([health, energy, mood, social, hygiene]) +  # wellbeing
        0.2 * sqrt(money)                        # wealth
    )
elif energy <= 0 or health <= 0:  # early death
    terminal_reward = 0.1 * (above formula)  # 90% penalty
```

**What this teaches**:

- Surviving to retirement is worth 10× dying early
- Quality of life matters (can't just grind money and die miserable)
- There's a sweet spot between hoarding and living well

**Human equivalent**: "Did you live a good life?" — judged at the end, not tick-by-tick.

#### 3. **Human-Perceivable Observations Only**

Curriculum progression removes information access:

- **Level 0-1 (pedagogical exception)**: Full observability, all affordance locations given (like being handed a map)
- **Level 2-3**: Still full observability (mastering economics before navigation)
- **Level 4-5**: Partial observability (5×5 vision on 8×8 grid — must explore and remember)
- **Level 6-7**: Social cues only (see "looks_tired", not exact energy=0.23 — no telepathy)
- **Level 8**: Communication signals without semantics (hear "123", must learn meaning via correlation)

**Human observer test** for every design decision:

- ✅ "Can a human see someone looks tired?" YES → observable cue
- ❌ "Can a human know their exact energy value?" NO → hidden
- ✅ "Can a human hear a signal?" YES → family_comm_channel
- ❌ "Does a human know what '123' means innately?" NO → must learn

#### 4. **CTDE via Observable Cues**

The only "cheat" is offline supervised learning for social reasoning:

**Training time** (using logged episodes):

```python
# Ground truth (privileged labels)
agent_a_actual_mood = 0.25

# Observable signals (from cues.yaml)
agent_a_cues = ["looks_sad", "looks_poor"]

# Supervised learning
predicted_mood = social_model(agent_a_cues)
loss = mse(predicted_mood, agent_a_actual_mood)
```

**Inference time** (deployed policy):

```python
# Only sees cues, not ground truth
observation = {'other_agents': {'public_cues': [['looks_tired', 'at_job']]}}
predicted_state = social_model(observation['other_agents']['public_cues'])
```

**Human equivalent**: You learn "droopy eyes + slow movement = tired" by asking people once, then predicting without asking. That's not cheating, that's learning.

---

### 1.3 What Makes This Different

#### vs. Other RL Simulators (OpenAI Gym, MuJoCo, etc.)

| Feature | Typical RL Sim | Townlet Framework |
|---------|---------------|---------|
| **World physics** | Hardcoded in Python | Configured in YAML (UAC) |
| **Reward function** | Dense shaping per action | Sparse, user-definable via UAC |
| **Observability** | Often full state | Curriculum-staged, human-realistic |
| **Provenance** | "Trust me" | Cognitive hash + signed checkpoints |
| **Reproducibility** | Config drift common | Snapshot immutability |
| **Auditability** | Black box | Glass-box telemetry |

#### vs. Other Multi-Agent Systems (SMAC, Hanabi, etc.)

| Feature | Typical Multi-Agent | Townlet Framework |
|---------|---------------------|---------|
| **Social info** | Full state or blind | Observable cues only |
| **Communication** | Pre-grounded or none | Emergent via correlation |
| **Cooperation** | Shared reward | Families form via breeding |
| **Competition** | Teams fixed at init | Dynamic (rivals, remarriage, churn) |

#### vs. Other Survival Sims (The Sims, RimWorld, etc.)

| Feature | Typical Survival Sim | Townlet Framework |
|---------|---------------------|----------|
| **Agent learning** | Scripted AI | Deep RL from scratch |
| **Physics** | Game balance tweaks | Scientific modeling via cascades |
| **Reward** | Player satisfaction | Minimalist, user-definable |
| **Purpose** | Entertainment | Research + pedagogy |

**The unique combination**: Scientific rigor (RL research) + configuration flexibility (game mods) + governance auditability (defense/policy applications).

---

### 1.4 Who This Is For

#### Researchers

**Value proposition**: "I have a hypothesis about X. Can I test it without writing Python?"

**If X involves**:

- Resource management under scarcity
- Multi-objective optimization
- Temporal planning (time-of-day constraints)
- Social coordination (families, competition, emergent communication)
- Long-horizon credit assignment

**Then YES** — edit YAML, launch runs, compare results with provenance.

**Example**: "Does dynasty inheritance produce better coordination than meritocratic churn?"

- Edit `population_genetics.yaml` → two configs
- Launch both → compare family stability, communication diversity, wealth distribution
- Publish with cognitive hashes proving which rules produced which outcomes

#### Educators

**Value proposition**: "I want students to experiment without coding."

**Curriculum levels map to learning objectives**:

- Level 0-1: "What is a policy?" (learn affordance semantics)
- Level 2-3: "How do I balance resources?" (economic loops)
- Level 4-5: "How do I navigate under uncertainty?" (exploration + memory)
- Level 6-7: "How do I reason about others?" (theory of mind via cues)
- Level 8: "How does language emerge?" (communication without pre-shared meaning)

**Students can**:

- Tweak world parameters (make food cheaper, jobs scarcer)
- Observe behavioral changes
- Read glass-box telemetry ("it tried to steal, ethics blocked it")
- Understand *why* policies emerged

**No code required** — just YAML editing and running `townlet train --config configs/student_world/`.

#### Policy / Governance People

**Value proposition**: "Can you prove this is safe?"

**The Townlet Framework provides**:

- Explicit ethics rules in Layer 1 (`cognitive_topology.yaml`)
- Deterministic EthicsFilter (no learned safety — just rule enforcement)
- Cognitive hashing (prove which mind, with which rules, did what)
- Telemetry showing vetoes ("attempted STEAL, blocked, reason: forbidden by Layer 1")
- Signed checkpoints (tamper protection)

**Audit question**: "Why did it call an ambulance at 3am?"
**Townlet answer**:

- "At tick 1847, agent hash `4f9a7c21` had health=0.18, panic threshold=0.25"
- "Panic controller overrode normal policy with `call_ambulance`"
- "EthicsFilter allowed (ambulance is legal even when expensive)"
- "See telemetry line 1847, veto_reason=null, panic_reason='health_critical'"

This is evidence, not anecdote.

---

### 1.5 Document Roadmap

This document is organized as:

**Understanding (Sections 1-2)**: What Townlet is, why it's designed this way, and the reward architecture

**Technical Detail (Sections 3-5)**: Curriculum, observation space, and cues system

**Implementation (Sections 6-7)**: Critical blockers to fix and missing specifications to write

**Research Directions (Sections 8-9)**: Population genetics, inheritance experiments, and the broader platform vision

**Action Plan (Sections 10-11)**: What to do next and how to pitch it

**Appendices**: Configuration templates, success criteria checklists, glossary, related work

**How to use this document**:

- If you're **fixing bugs**: Start with Section 6 (Critical Blockers)
- If you're **understanding the design**: Read Sections 1-2 in order
- If you're **implementing features**: Sections 4-5, then 7
- If you're **doing research**: Sections 8-9
- If you're **writing docs**: Use appendices as templates

---
