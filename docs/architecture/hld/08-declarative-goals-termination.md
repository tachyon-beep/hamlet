# 8. Declarative Goals and Termination Conditions

**Document Type**: Design Rationale
**Status**: Updated Draft
**Version**: 1.0 - Townlet Framework (BAC/UAC Architecture)
**Last Updated**: 2025-11-05
**Owner**: Principal Technical Advisor (AI)
**Parent Document**: TOWNLET_HLD.md

**Audience**: Engineers implementing goal systems, governance teams verifying goal logic, researchers designing curricula
**Technical Level**: Deep Technical (DSL design, goal termination semantics, engine truth protocols)
**Estimated Reading Time**: 5 min for skim | 12 min for full read

---

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**:
Declarative goal system that makes agent strategic objectives explicit YAML structures (not vague reward shaping), with auditable termination conditions (not hidden lambdas in code). Defines goal_definitions DSL, termination interpreter, and engine truth vs self-report distinction.

**Why This Document Exists**:
Establishes goals as governance primitives enabling "why did it pursue X instead of Y?" to be answered with YAML inspection, not speculation. Makes goal-driven behavior auditable, curriculum-adjustable, and pedagogically transparent.

**Who Should Read This**:
- **Must Read**: Engineers implementing hierarchical policy, governance teams auditing goal logic
- **Should Read**: Researchers designing curricula, instructors teaching goal-driven RL
- **Optional**: Operators running training (high-level understanding sufficient)

**Reading Strategy**:
- **Quick Scan** (5 min): Read §8.1 for goal definition DSL examples
- **Partial Read** (9 min): Add §8.2 for governance/curriculum/teaching value
- **Full Read** (12 min): Add §8.3 for engine truth vs self-report distinction

---

## Document Scope

**In Scope**:
- **Goal Definitions**: Declarative structures with id, termination conditions
- **Termination DSL**: Safe language for goal satisfaction expressions (all/any blocks, bar comparisons)
- **Termination Interpreter**: Runtime evaluation of goal completion
- **Engine Truth vs Self-Report**: current_goal (factual) vs agent_claimed_reason (narrative)
- **Governance Value**: Why goals as data structures matter

**Out of Scope**:
- **Meta-Controller Implementation**: See hierarchical policy component docs
- **Reward Shaping**: See training configuration docs
- **Goal Learning**: Future extension (currently goals are declared, not learned)
- **Multi-Agent Goal Coordination**: Future extension

**Critical Boundary**:
Goal system is **framework-level** (works for any SDA). Examples show **Townlet Town** goals (SURVIVAL, THRIVING, SOCIAL with energy/health bars), but the goal_definitions pattern applies to any universe instance (factory: EFFICIENCY/SAFETY, trading: BUY/SELL/HOLD).

---

## Position in Architecture

**Related Documents** (Read These First/Next):
- **Prerequisites**: [02-brain-as-code.md](02-brain-as-code.md) (Layer 1 allowed_goals), [07-telemetry-ui-surfacing.md](07-telemetry-ui-surfacing.md) (current_goal logging)
- **Builds On**: BAC Layer 1 (goal declarations), hierarchical policy (meta-controller)
- **Related**: [09-affordance-semantics.md](09-affordance-semantics.md) (bars referenced in termination conditions)
- **Next**: [09-affordance-semantics.md](09-affordance-semantics.md) (universe mechanics)

**Section Number**: 8 / 12
**Architecture Layer**: Logical (goal semantics and termination protocol)

---

## Keywords for Discovery

**Primary Keywords**: goal definitions, termination condition, goal termination DSL, declarative goals, engine truth vs self-report
**Secondary Keywords**: meta-controller, goal satisfaction, termination interpreter, goal struct, curriculum adjustment
**Subsystems**: hierarchical_policy.meta_controller, goal termination evaluator
**Design Patterns**: Goals as data structures, declarative termination, observable goal switching

**Quick Search Hints**:
- Looking for "how goals are defined"? → See §8.1 (Goal Definitions)
- Looking for "when goals terminate"? → See §8.1 (Termination DSL examples)
- Looking for "why goals as YAML"? → See §8.2 (Why This Matters)
- Looking for "engine truth vs narrative"? → See §8.3 (Honesty in Introspection)

---

## Version History

**Version 1.0** (2025-11-05): Initial goal system specification defining declarative goals and termination DSL

---

## Document Type Specifics

### For Design Rationale Documents (Type: Design Rationale)

**Design Question Addressed**:
"Should agent goals be implicit (emergent from reward shaping) or explicit (declarative data structures with observable termination)?"

**Alternatives Considered**:
1. **Implicit goals via reward shaping** → **Rejected** (not auditable, not curriculum-adjustable, not pedagogically transparent)
2. **Hardcoded goal logic in Python** → **Rejected** (not diffable, not governable, requires code changes for curriculum)
3. **Declarative goal_definitions with DSL** → **Chosen** (auditable, curriculum-friendly, teaching-friendly)

**Key Trade-offs**:
- **Chosen**: Goals visible in YAML, termination conditions explicit, curriculum via config diff
- **Sacrificed**: Flexibility of arbitrary Python (restricted to safe DSL)

**Decision Drivers**:
- **Governance requirement**: "Why did it pursue X?" must be answerable with config inspection
- **Curriculum design**: Instructors need to adjust goal stringency without code changes
- **Teaching transparency**: Students see goal logic directly, not inferred from reward curves

---

**END OF FRONT MATTER**
**BEGIN ARCHITECTURE CONTENT**

---

## 8. Declarative Goals and Termination Conditions

Townlet agents pursue **explicit high-level goals**—and can report which goal is active at any moment.

**Framework principle**: Declarative goals are framework-level (works for any SDA). The specific goal vocabulary (SURVIVAL vs EFFICIENCY) and termination bars (energy vs machinery_stress) are instance-specific.

We do **two things**:

1. We make goals **explicit data structures**, not vague "the RL policy probably cares about reward shaping"
2. We make "I'm done with this goal" a **declarative rule in YAML**, not a secret lambda hidden in code

**Framework benefit**: This pattern enables governance ("show me the goal logic"), curriculum ("tighten SURVIVAL to require 80% energy instead of 50%"), and teaching ("here's why it kept working while starving").

---

## 8.1 Goal Definitions Live in Config, Not in Python

We define goals in a **small, safe DSL** inside the run snapshot (part of BAC Layer 1 or runtime configuration).

**Example goal definitions** (Townlet Town instance):

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

**Framework pattern**: Goal definition structure (id + termination DSL) is framework-level. The specific goals and bars are instance-specific.

**Alternative universe examples**:

**Factory instance**:
```yaml
goal_definitions:
  - id: "EFFICIENCY"
    termination:
      all:
        - { bar: "production_quota", op: ">=", val: 0.9 }
        - { bar: "machinery_stress", op: "<=", val: 0.3 }

  - id: "SAFETY"
    termination:
      all:
        - { bar: "machinery_stress", op: "<=", val: 0.2 }
        - { bar: "worker_fatigue", op: "<=", val: 0.5 }
```

**Trading instance**:
```yaml
goal_definitions:
  - id: "ACCUMULATE"
    termination:
      all:
        - { bar: "portfolio_value", op: ">=", val: 1.5 }  # 150% of starting capital
        - { bar: "market_volatility", op: "<=", val: 0.3 }

  - id: "PRESERVE"
    termination:
      any:
        - { bar: "portfolio_value", op: "<=", val: 0.8 }  # Lost 20%, go defensive
        - { time_elapsed_ticks: ">=", val: 1000 }
```

---

### DSL Conventions (Framework-Level)

**All bars normalized 0.0–1.0** based on universe_as_code.yaml:
- `0.8` means "80% of full", not "magic number 80"
- Money can also be normalized: `money: 1.0` might mean $100 if world spec defines $100 ↔ 1.0
- **Framework pattern**: Normalization enables consistent comparisons across different bar scales

**Termination can use `all` or `any` blocks**:
- `all`: Every condition must be true (AND semantics)
- `any`: At least one condition must be true (OR semantics)
- **Framework pattern**: Boolean composition without nesting complexity

**Leaves are simple comparisons**:
- Bar comparisons: `{ bar: "energy", op: ">=", val: 0.8 }`
- Runtime counters: `{ time_elapsed_ticks: ">=", val: 500 }`
- **No arbitrary Python**, **no hidden side effects**
- **Framework constraint**: Safe, bounded expressions only

**Framework principle**: Goal Termination DSL is framework-level (same syntax for any universe). The specific bars (energy vs machinery_stress) and thresholds (0.8 vs 0.3) are instance-specific.

---

### Runtime Execution (Framework-Level)

**At runtime**:

1. **Meta-controller picks a goal struct** (SURVIVAL, GET_MONEY, EFFICIENCY, etc.)
   - Defined in hierarchical_policy.meta_controller (BAC Layer 2 implementation)
   - Selection based on Layer 1 allowed_goals and personality sliders

2. **Each tick (or every N ticks)** it evaluates that goal's termination rule using a **termination interpreter**
   - Reads current bar values (energy, health, money, etc.)
   - Evaluates DSL expression (`all`/`any` blocks with comparisons)
   - Returns boolean: goal satisfied or not

3. **If termination rule fires**, that goal is considered **satisfied**
   - Meta-controller may select a new goal
   - Goal switch logged to telemetry (current_goal changes)

**Framework pattern**: Termination interpreter execution is framework-level. The specific evaluation logic (every tick vs every 50 ticks) is configurable. The bars and thresholds are instance-specific.

**Townlet Town example**: Meta-controller evaluates SURVIVAL termination every 10 ticks. When energy ≥ 0.8 AND health ≥ 0.7, SURVIVAL satisfied → switch to THRIVING.

**Factory example**: Meta-controller evaluates EFFICIENCY termination every 20 ticks. When production_quota ≥ 0.9 AND machinery_stress ≤ 0.3, EFFICIENCY satisfied → switch to MAINTENANCE.

---

## 8.2 Why This Matters

**Framework benefit**: Declarative goals transform "why did it do that?" from speculation to YAML inspection. This value proposition applies to any universe instance.

### For Governance / Audit

**Question**: "Why was it still pursuing GET_MONEY while its health was collapsing?"

**Framework answer**: Point to the YAML:

```yaml
goal_definitions:
  - id: "GET_MONEY"
    termination:
      any:
        - { bar: "money", op: ">=", val: 1.0 }
        - { time_elapsed_ticks: ">=", val: 500 }
```

**Analysis**:
- GET_MONEY doesn't terminate based on health (not in termination conditions)
- Maybe meta-controller should have switched to SURVIVAL when health dropped
- But if `personality.greed: 0.9` (very greedy), meta-controller prioritizes money goals
- **This is a design decision** (high greed + no health gate in GET_MONEY termination), **not "the AI went rogue"**

**Framework pattern**: Goal inspection works for any universe. Factory: "Why pursuing EFFICIENCY while machinery_stress critical?" → inspect EFFICIENCY termination conditions. Trading: "Why holding position during crash?" → inspect PRESERVE goal logic.

### For Curriculum

**Early training**: Define SURVIVAL as lenient:
```yaml
- id: "SURVIVAL"
  termination:
    all:
      - { bar: "energy", op: ">=", val: 0.5 }  # Only 50% full OK
      - { bar: "health", op: ">=", val: 0.5 }
```

**Later curriculum**: Tighten to strict:
```yaml
- id: "SURVIVAL"
  termination:
    all:
      - { bar: "energy", op: ">=", val: 0.8 }  # Must reach 80% full
      - { bar: "health", op: ">=", val: 0.7 }
```

**Result**:
- **Diff in YAML** shows exact curriculum change
- **Not a code poke** (no Python edited)
- Students can directly compare behavior when SURVIVAL is lenient versus strict

**Framework pattern**: Curriculum adjustment via YAML diff works for any universe. Factory: tighten SAFETY thresholds over time. Trading: adjust PRESERVE trigger from 80% portfolio to 90% (more risk-averse).

**Pedagogical value**: "Here's before/after config. Here's before/after survival rate. This is how goal stringency affects behavior."

### For Teaching

**Instructor question**: "The agent is starving but still working. Does the SURVIVAL goal terminate too late, or is the meta-controller failing to switch because `greed` is set too high in `cognitive_topology.yaml`?"

**Framework answer**: Direct inspection of two configs:

**Goal termination** (Layer 1 or runtime config):
```yaml
goal_definitions:
  - id: "SURVIVAL"
    termination:
      all:
        - { bar: "energy", op: ">=", val: 0.8 }  # High bar
```

**Personality** (Layer 1 cognitive_topology.yaml):
```yaml
personality:
  greed: 0.9  # Very money-driven
  curiosity: 0.3
```

**Diagnosis**:
- SURVIVAL termination requires 80% energy (high bar → hard to satisfy → stays in SURVIVAL longer)
- High greed (0.9) biases meta-controller toward GET_MONEY even when energy low
- **Root cause**: Config mismatch (strict SURVIVAL exit + greedy personality = starvation risk)

**This is not abstract RL theory. This is direct inspection.**

**Framework benefit**: This teaching workflow works for any universe. Factory: "Why ignoring safety alarms?" → inspect SAFETY termination + personality.risk_tolerance. Trading: "Why panic-selling?" → inspect PRESERVE trigger + personality.patience.

---

## 8.3 Honesty in Introspection

Now that goals are **formal objects** and termination is a **declarative rule**, we can show **two different "explanations"** side by side:

**Two telemetry fields** (framework-level logging):

1. **current_goal** (engine truth): `SURVIVAL`
   - Factual: What meta-controller actually selected
   - Source: hierarchical_policy.meta_controller internal state
   - **Always accurate** (engine ground truth)

2. **agent_claimed_reason** (self-report / introspection): `"I'm going to work to save up for rent"`
   - Narrative: What agent thinks it's doing
   - Source: Introspection module (if enabled in Layer 1)
   - **May differ from truth** (agent's understanding may be wrong)

**Framework pattern**: Engine truth vs self-report distinction is framework-level. The specific goal names (SURVIVAL vs EFFICIENCY) and narratives are instance-specific.

---

### When They Match

**Telemetry** (Townlet Town):
- `current_goal: "THRIVING"`
- `agent_claimed_reason: "I'm going to the gym to improve fitness"`

**Interpretation**: Nice, we can narrate behavior in plain language to non-technical stakeholders. Agent's understanding aligns with actual goal.

**Framework benefit**: Alignment enables clear communication. Works for any universe.

---

### When They Do NOT Match

**Telemetry** (Townlet Town):
- `current_goal: "SURVIVAL"`
- `agent_claimed_reason: "I'm going to work to save up for rent"`

**The discrepancy becomes a teaching moment**:

> "The agent **claims** it is working for rent (GET_MONEY narrative), but **engine truth** shows it remains in SURVIVAL mode. This means the meta-controller selected SURVIVAL (because energy or health critical), but the agent's **world model** misunderstood what would keep it alive. **That is a world-model error** (predicted work would restore energy, but work costs energy). Not 'the AI is lying' - the agent's internal understanding diverged from reality."

**Framework value**: This diagnostic pattern works for any universe:

**Factory example**:
- `current_goal: "SAFETY"`
- `agent_claimed_reason: "Maximizing production output"`
- **Diagnosis**: Meta-controller selected SAFETY (machinery_stress critical), but agent's narrative suggests it thinks it's pursuing EFFICIENCY. World model failure (didn't realize stress was critical).

**Trading example**:
- `current_goal: "PRESERVE"`
- `agent_claimed_reason: "Buying the dip for long-term gains"`
- **Diagnosis**: Meta-controller selected PRESERVE (portfolio value dropped), but agent narrative suggests aggressive accumulation. Risk assessment failure (underestimated downside).

---

**We log both in telemetry on purpose.**

**Framework principle**: Logging both engine truth and self-report enables:
- **Alignment validation**: When they match, agent understanding is correct
- **Error diagnosis**: When they diverge, world model or meta-controller failure
- **Teaching moments**: Gaps expose cognitive deficits, not "AI misbehavior"

**This transforms "unexplained behavior" into "diagnosable cognitive error."**

---

**Summary**: The Townlet Framework declarative goal system provides:

1. **Goal Definitions** - Explicit data structures with id and termination conditions (framework-level pattern)
2. **Termination DSL** - Safe language for goal satisfaction (`all`/`any` blocks, bar comparisons, no arbitrary Python)
3. **Termination Interpreter** - Runtime evaluation of goal completion against current state
4. **Engine Truth vs Self-Report** - Factual current_goal vs narrative agent_claimed_reason (framework-level distinction)

**Framework principle**: Goals as data structures enable governance (config inspection), curriculum (YAML diffs), and teaching (direct diagnosis). Works for any universe instance.

**Specific examples** (Townlet Town: SURVIVAL/THRIVING/SOCIAL, Factory: EFFICIENCY/SAFETY, Trading: BUY/SELL/HOLD) demonstrate framework generality.

---
