# Research Opportunities in UNIVERSE_AS_CODE

## Executive Summary

After analyzing UNIVERSE_AS_CODE.md, I've identified **4 high-priority research topics** that would benefit from deep investigation before implementation, rather than "yolo with TDD". These areas involve complex design decisions with multiple viable approaches, unclear best practices, or pedagogically critical tradeoffs.

**HIGH PRIORITY** (research now):

1. **End-of-Life Reward Model Design** - How to score a simulated life (alignment in miniature)
2. **Economic Balance & Tuning Methodology** - Systematic approach to creating pedagogically valuable "teaching packs"

**MEDIUM PRIORITY** (research before implementation):
3. **Lifecycle/Retirement Mechanics** - How aging and retirement should work mathematically
4. **Cascade Design Patterns** - Optimal feedback loop structures for emergent gameplay

**DEFERRED** (can TDD or already addressed):
5. Meter System Architecture (8-bar constraint) - May be technical debt but not urgent
6. Relocation Mechanics - Deferred future work
7. Action Space - Already covered by TASK-000/002

---

## HIGH PRIORITY RESEARCH

### 1. End-of-Life Reward Model Design ⭐⭐⭐

**Problem Statement**

UNIVERSE_AS_CODE mentions that "end-of-life scoring" is computed based on configuration (remaining money, health, mood), with different payouts for retirement vs death (e.g., death = 10% of retirement value). However, **the actual reward formula is not specified**.

From the document:
> "At episode end, the reward_model computes a final life score based on configuration (e.g., remaining money, health, mood). Dying early yields a heavily discounted score (for example, 10 percent of the retirement value), reinforcing that dignified survival matters more than simply delaying collapse."

**Why This Needs Research**

This is **alignment in miniature**. The reward function determines what agents learn to value. Getting this wrong means:

- Agents might learn to maximize survival steps while miserable and broke (survival hacking)
- Agents might learn to die early if retirement is "too hard" to reach
- Different worlds (austerity, boom) may need different reward weightings
- Students learn about alignment by seeing how reward design shapes behavior

**Key Questions**

1. **Reward Formula Structure**:
   - Additive? `reward = w_money * money + w_health * health + w_mood * mood`
   - Multiplicative? `reward = survival_bonus * (money * health * mood)^α`
   - Threshold-based? `reward = survival_bonus + sum(meter > threshold ? bonus : 0)`
   - Hierarchical? `reward = base_survival + quality_of_life_multiplier`

2. **Death Penalty Design**:
   - Fixed discount (10% of retirement value)?
   - Time-dependent (die at step 50 = 5% value, die at step 500 = 50% value)?
   - Cause-dependent (death by starvation worse than death by exhaustion)?
   - Debt-dependent (die broke = 0% value, die wealthy = 20% value)?

3. **Retirement Bonus**:
   - Flat bonus for reaching retirement?
   - Quality-of-life multiplier at retirement (happy retirement vs miserable retirement)?
   - Terminal meter values matter or just "made it to the end"?

4. **Meter Weighting**:
   - Should health matter more than mood?
   - Should money matter more in "austerity" worlds?
   - Should social matter more in "community-focused" worlds?
   - Dynamic weights based on world type?

5. **Pedagogical Goals**:
   - What do we want students to learn from reward design?
   - Should we show them reward hacking examples?
   - Should reward be opaque or transparent?

**Research Approach**

1. **Literature Review**:
   - Quality-of-life metrics (QALY, DALY, WHO well-being index)
   - Utility functions in economics (Cobb-Douglas, CES, lexicographic preferences)
   - RL reward shaping (sparse vs dense, terminal vs episodic)
   - Game design: scoring systems in life-sim games (The Sims, Stardew Valley, Frostpunk)
   - Alignment research: value learning, reward misspecification

2. **Design Space Exploration**:
   - Prototype 5-10 candidate reward functions
   - Analyze mathematical properties (gradients, local optima, exploitability)
   - Simulate agent behavior under each reward function
   - Identify which formulas produce "interesting failures" (reward hacking)

3. **Pedagogical Testing**:
   - Which reward functions make alignment lessons clearest?
   - Which are most intuitive for students?
   - Which produce the most interesting emergent behaviors?

4. **Configuration Design**:
   - How should reward weights be specified in YAML?
   - Should there be "preset" reward profiles (utilitarian, egalitarian, libertarian)?
   - How to make reward tuning accessible to non-RL instructors?

**Example Research Questions**

```yaml
# Option 1: Additive with terminal bonus
reward_model:
  type: "additive"
  terminal_bonus: 100.0  # For reaching retirement
  meter_weights:
    money: 10.0
    health: 20.0
    mood: 15.0
    social: 10.0
    fitness: 5.0
  death_penalty_multiplier: 0.1  # Death = 10% of would-be retirement score

# Option 2: Multiplicative (all meters matter)
reward_model:
  type: "multiplicative"
  base_survival_reward: 50.0
  quality_multiplier: "(money^0.3 * health^0.4 * mood^0.3)"  # Cobb-Douglas
  death_penalty: "base_survival_reward * (steps_survived / 1000)"

# Option 3: Threshold-based (meet basic needs)
reward_model:
  type: "threshold"
  survival_bonus: 100.0
  meter_thresholds:
    money: {threshold: 0.3, bonus: 20.0}   # Retire with >$30 = +20 points
    health: {threshold: 0.5, bonus: 30.0}  # Retire healthy = +30 points
    mood: {threshold: 0.4, bonus: 20.0}    # Retire happy = +20 points
  death_penalty: 0.0  # Death = no bonuses, just survival steps
```

**Expected Output**

A research document `RESEARCH-REWARD-MODEL-DESIGN.md` containing:

1. Literature review summary (utility functions, QoL metrics, RL reward shaping)
2. Design space taxonomy (additive, multiplicative, threshold, hierarchical)
3. Candidate reward functions (5-10 options with mathematical properties)
4. Pros/cons analysis (pedagogical value, exploitability, tuning difficulty)
5. Recommended default + rationale
6. YAML schema design for reward configuration
7. Test cases demonstrating alignment lessons (reward hacking examples)

**Estimated Effort**: 8-12 hours

**Pedagogical Value**: ✅✅✅ (Teaches alignment, utility functions, reward design)

---

### 2. Economic Balance & Tuning Methodology ⭐⭐⭐

**Problem Statement**

UNIVERSE_AS_CODE mentions creating "teaching packs" for different economic conditions (austerity, boom, nightlife), but **provides no methodology** for how to actually tune these packs.

From the document:
> "Some obvious examples:
>
> - **Austerity** — sleep and food become cheaper while labour income drops and healthcare costs increase. Survival is possible but financially precarious.
> - **Boom** — wages rise while rest and hygiene become more expensive. Agents can cover immediate needs but pay heavily to maintain them.
> - **Nightlife** — social outlets operate late into the night, whereas medical services may have limited hours."

**Why This Needs Research**

Creating balanced, pedagogically valuable world variants is currently **ad hoc guesswork**. Without a systematic approach:

- "Austerity" might be unwinnable or trivial
- "Boom" might feel the same as baseline
- Students don't learn meaningful lessons about economic systems
- Instructors can't confidently create custom worlds

**Key Questions**

1. **Economic Balance Metrics**:
   - How do we define "balanced" vs "too hard" vs "too easy"?
   - What's a good "survival rate" target? 50%? 70%? Depends on curriculum level?
   - Should agents be able to accumulate wealth or just survive?
   - How do we measure "economic precarity" quantitatively?

2. **Parameter Tuning Approach**:
   - Which parameters affect difficulty most (wages, food cost, depletion rates)?
   - Are there "lever patterns" (e.g., "increase healthcare cost + decrease wages = austerity")?
   - How to tune multiple parameters simultaneously without breaking everything?
   - What parameter ranges are "sane" vs "broken"?

3. **Economic Models**:
   - Should we model the economy as a closed system (money in = money out)?
   - How much money should agents accumulate by retirement?
   - What's the "poverty line" (minimum viable income)?
   - How do we model inflation, opportunity cost, debt?

4. **Pedagogical Goals**:
   - What should "austerity" teach? (resource scarcity, healthcare access)
   - What should "boom" teach? (hedonic treadmill, lifestyle inflation)
   - What should "nightlife" teach? (temporal scheduling, access inequality)
   - How to make economic lessons obvious to students?

5. **Validation Approach**:
   - How do we test if a world is "well-balanced"?
   - What metrics indicate a world is pedagogically valuable?
   - How to avoid "unsolvable" worlds vs "trivial" worlds?

**Research Approach**

1. **Economic Modeling**:
   - Build simple economic models of agent lifecycle
   - Calculate "break-even" income (money_in = money_out over lifetime)
   - Model different economic regimes (scarcity, abundance, precarity)
   - Identify key economic parameters and their relationships

2. **Sensitivity Analysis**:
   - Which affordance costs have biggest impact on survival?
   - Which income sources are critical vs optional?
   - How do cascade strengths interact with economic parameters?
   - What parameter changes produce meaningful economic shifts?

3. **Balance Metrics**:
   - Define quantitative metrics for world balance:
     - Survival rate (% agents reaching retirement)
     - Average lifespan (mean steps survived)
     - Wealth at retirement (mean money at episode end)
     - Meter stability (variance of pivotal meters)
     - Economic mobility (can agents recover from poverty?)

4. **Tuning Recipes**:
   - Create "recipe cards" for common economic scenarios:
     - Austerity = {healthcare_cost: +50%, wages: -30%, food_cost: -20%}
     - Boom = {wages: +50%, rest_cost: +40%, entertainment_cost: +30%}
     - Nightlife = {bar_hours: [16,4], hospital_hours: [8,20], social_decay: -50%}
   - Validate each recipe produces intended economic dynamics
   - Test with curriculum to confirm pedagogical value

5. **Configuration Templates**:
   - Design YAML template for economic profiles
   - Create affordance parameter scaling system
   - Build validation suite to catch broken economies

**Example Economic Balance Sheet**

```yaml
# configs/austerity/economic_profile.yaml
version: "1.0"
description: "Austerity economics - healthcare access inequality"

economic_profile:
  difficulty: "hard"
  target_survival_rate: 0.5  # 50% reach retirement (pedagogically intentional)
  target_retirement_wealth: 0.2  # $20 (very low)

  # Global affordance modifiers (multiplicative)
  affordance_modifiers:
    income_multiplier: 0.7      # -30% wages
    healthcare_cost_multiplier: 1.5   # +50% healthcare costs
    food_cost_multiplier: 0.8   # -20% food costs (cheap but poor quality)
    rest_cost_multiplier: 0.9   # -10% sleep costs (cheaper housing)

  # Specific overrides (absolute values)
  affordance_overrides:
    Hospital:
      costs_per_tick:
        - { meter: "money", amount: 0.45 }  # Was 0.30, now 0.45
    Doctor:
      operating_hours: [9, 17]  # Reduced hours (was [8, 18])
    Job:
      effects_per_tick:
        - { meter: "money", amount: 0.07 }  # Was 0.10, now 0.07

  # Economic validation constraints
  validation:
    min_viable_income: 0.05  # Agents must earn at least $5/tick to survive
    max_healthcare_burden: 0.6  # Healthcare can't exceed 60% of lifetime income
    break_even_lifestyle: "subsistence"  # Agents can survive but not thrive

  pedagogical_intent:
    - "Healthcare access inequality (expensive, limited hours)"
    - "Poverty trap (low wages, high essential costs)"
    - "Precarity (one illness = bankruptcy)"
```

**Expected Output**

A research document `RESEARCH-ECONOMIC-BALANCE-METHODOLOGY.md` containing:

1. Economic modeling fundamentals (break-even analysis, lifecycle budgets)
2. Sensitivity analysis (parameter impact on survival metrics)
3. Balance metrics taxonomy (survival rate, wealth accumulation, meter stability)
4. Tuning recipes for common scenarios (austerity, boom, nightlife, +5 others)
5. YAML schema for economic profiles
6. Validation suite design (tests for broken economies)
7. Pedagogical mapping (which economies teach which lessons)

**Estimated Effort**: 10-15 hours

**Pedagogical Value**: ✅✅✅ (Teaches economics, game balance, systems thinking)

---

## MEDIUM PRIORITY RESEARCH

### 3. Lifecycle/Retirement Mechanics Design ⭐⭐

**Problem Statement**

The document describes a `lifecycle` scalar that increases over time and reaching 1.0 triggers retirement, but **the exact mechanics are underspecified**.

From the document:
> "`lifecycle` increases slightly each tick. Adverse conditions accelerate the increase (e.g., starvation, illness, miserable mood). If `lifecycle` reaches 1.0 before pivotal meters reach zero, the agent retires."

**Missing Details**:

- What's the base increase rate per tick?
- Which "adverse conditions" accelerate it and by how much?
- Does positive fitness slow aging? Does good mood slow aging?
- What's the expected retirement age (in steps) under normal conditions?
- Can lifecycle go backwards (recovery, de-aging)?

**Why This Needs Research**

Lifecycle mechanics affect:

- Episode length (how long training episodes last)
- Survival vs quality-of-life tradeoffs (rushing to retirement vs living well)
- Emergency response incentives (high lifecycle = less time to recover from crises)
- Pedagogical lessons about aging, health, and planning

**Key Questions**

1. **Base Aging Rate**:
   - Should lifecycle increase linearly (constant rate) or accelerate over time?
   - What's the "natural lifespan" (steps to retirement with perfect health)?
   - Should early-life and late-life aging differ?

2. **Accelerating Factors**:
   - Which meters accelerate aging? (starvation, illness, stress)
   - Linear or exponential acceleration?
   - Should low fitness double aging speed? Triple it?

3. **Slowing Factors**:
   - Can high fitness/mood slow aging?
   - Should agents be able to "buy time" with good lifestyle?
   - Or is aging inexorable (philosophical choice)?

4. **Retirement as Goal vs Neutral End**:
   - Is retirement a "win condition" or just "episode ended"?
   - Should agents maximize lifespan or maximize quality-of-life before retirement?
   - Does early retirement (e.g., financial independence) make sense?

**Research Approach**

1. **Mortality Risk Models**:
   - Review actuarial science (mortality rates, hazard functions)
   - Review biological aging models (Gompertz curve, hallmarks of aging)
   - Adapt to simplified RL-friendly formulation

2. **Design Space Exploration**:
   - Linear aging: `lifecycle += 0.001` (1000 steps to retirement)
   - Accelerated aging: `lifecycle += 0.001 * (1 + stress_factor)`
   - Gompertz aging: `lifecycle += base_rate * exp(age * growth_rate)`

3. **Pedagogical Testing**:
   - Which aging models produce interesting strategic choices?
   - Which make agents plan for long-term vs short-term?
   - Which teach about mortality risk and health maintenance?

**Expected Output**

A brief research note `RESEARCH-LIFECYCLE-MECHANICS.md` containing:

1. Mortality risk model review (actuarial + biological)
2. Candidate lifecycle formulas (3-5 options)
3. Expected retirement age under each formula
4. Pedagogical implications (what agents learn)
5. Recommended default + configuration schema

**Estimated Effort**: 4-6 hours

**Pedagogical Value**: ✅✅ (Teaches aging, long-term planning, mortality risk)

---

### 4. Cascade Design Patterns ⭐⭐

**Problem Statement**

The current cascade system (modulations + threshold cascades) works, but **the design rationale is not deeply explained**. Why these particular cascades? What cascade patterns produce interesting gameplay?

From the document:
> "These constructs encode collapse patterns without bespoke logic, producing the expected survival spirals when multiple needs are ignored."

**Why This Needs Research**

Cascades are the "physics" of the universe. They determine:

- How quickly neglect becomes dangerous
- Which survival strategies are viable
- Whether death spirals are escapable or inevitable
- Emergent complexity (simple rules → complex behaviors)

**Key Questions**

1. **Cascade Patterns**:
   - What patterns exist? (linear, exponential, threshold, dampened, amplified)
   - Which patterns produce "interesting" gameplay?
   - Which produce frustrating or trivial gameplay?

2. **Feedback Loop Types**:
   - Negative feedback (homeostasis, self-correction)
   - Positive feedback (death spirals, runaway success)
   - Delayed feedback (long-term consequences)
   - Coupled feedback (multi-meter resonance)

3. **Execution Order Impacts**:
   - Why is execution order important?
   - What happens if we randomize order?
   - Are there "stable" vs "unstable" orderings?

4. **Complexity vs Comprehensibility**:
   - How many cascades is "too many"?
   - Can agents learn cascade dynamics implicitly?
   - Should cascades be visualized for students?

**Research Approach**

1. **Systems Dynamics Literature**:
   - Review feedback loop theory (Forrester, Meadows)
   - Review game design on survival mechanics (Don't Starve, RimWorld, Frostpunk)
   - Review ecosystem modeling (Lotka-Volterra, trophic cascades)

2. **Pattern Catalog**:
   - Document common cascade patterns
   - Analyze mathematical properties (stability, oscillation, convergence)
   - Classify by pedagogical value (teaches X concept)

3. **Cascade Experiments**:
   - Ablation studies (remove cascades one at a time)
   - Permutation studies (reorder execution)
   - Strength variation studies (weaken/strengthen cascades)
   - Measure impact on survival metrics and emergent behaviors

**Expected Output**

A research note `RESEARCH-CASCADE-DESIGN-PATTERNS.md` containing:

1. Cascade pattern taxonomy (linear, threshold, exponential, dampened)
2. Feedback loop analysis (positive, negative, coupled)
3. Design principles (when to use which pattern)
4. Execution order guidelines
5. Pedagogical mapping (which cascades teach which concepts)

**Estimated Effort**: 6-8 hours

**Pedagogical Value**: ✅✅ (Teaches systems thinking, feedback loops, emergent complexity)

---

## DEFERRED RESEARCH

### 5. Meter System Architecture (8-Bar Constraint)

**Problem**: Document states "There must be exactly eight bars" and "Changing them casually will break everything."

**Why Defer**: This is technical debt, not a pedagogical blocker. The current 8-bar system works. Making it extensible is valuable but not urgent. Can address during refactoring.

**Future Research**: "How to design a meter-agnostic architecture?" (variable state space, dynamic network sizing)

---

### 6. Relocation Mechanics

**Problem**: Document proposes "relocation" effect type for ambulance (teleport agent to hospital).

**Why Defer**: This is explicitly marked as "optional extension" and "planned". The current instant-heal ambulance works pedagogically. Can implement with TDD when needed.

---

### 7. Action Space

**Problem**: Hardcoded [UP, DOWN, LEFT, RIGHT, INTERACT, WAIT] action space.

**Why Defer**: Already addressed by TASK-000 and TASK-002 (configurable spatial substrates and action space).

---

## Recommended Research Priority

**Phase 1** (Do now - highest pedagogical impact):

1. **End-of-Life Reward Model Design** (8-12 hours)
2. **Economic Balance & Tuning Methodology** (10-15 hours)

**Phase 2** (Do before implementing lifecycle features):
3. **Lifecycle/Retirement Mechanics** (4-6 hours)

**Phase 3** (Do when refining cascade system):
4. **Cascade Design Patterns** (6-8 hours)

**Deferred** (Not urgent):
5. Meter System Architecture
6. Relocation Mechanics

**Total Phase 1 Effort**: 18-27 hours (2-3 days)

**Rationale**: Reward model and economic balance are pedagogically critical and affect every curriculum level. Getting these right early prevents costly rework. Lifecycle and cascades can be researched incrementally as we refine the system.

---

## Success Criteria

For each research topic, success means:

1. **Clear design space mapping** - Understand the options and tradeoffs
2. **Recommended defaults** - Concrete values/formulas to implement
3. **Configuration schema** - YAML structure for operators to tune
4. **Pedagogical validation** - Demonstrated teaching value
5. **Implementation guide** - Clear path from research to code

**Slogan**: "Research the problems where 'just implement it' would lead to costly rework or missed pedagogical opportunities."
