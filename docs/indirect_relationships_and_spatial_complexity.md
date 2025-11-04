# Indirect Relationships & Spatial Complexity

**Date**: 2025-10-28
**Purpose**: Add indirect causality and spatial trade-offs to increase strategic depth
**Core Mechanics**: Job payment penalties, dual food sources, spatial clustering
**Result**: Agent must learn hidden relationships and location-based trade-offs

---

## Design Philosophy: Layered Complexity

### The Problem with Direct Relationships

Simple meter systems have **direct causality**:

- Low energy → go to Bed → energy restored
- Low satiation → go to Fridge → satiation restored
- Agent learns: "If meter X is low, do action Y"

This is **too simple** for teaching sophisticated AI reasoning.

### The Solution: Indirect Relationships

**Indirect causality** requires the agent to discover hidden connections:

- Low energy → poor job performance → less money → can't afford services → death
- Agent must learn: "Energy affects MULTIPLE downstream outcomes"

**Spatial trade-offs** create context-dependent decisions:

- Hungry at home → HomeMeal is close and cheap (optimal)
- Hungry at work → HomeMeal is far, FastFood is close but expensive (trade-off!)

**Result**: Agent must learn systemic thinking and context awareness.

---

## Indirect Relationship 1: Job Payment Penalties

### The Mechanic

**File**: `src/hamlet/environment/entities.py`

```python
class Job(Affordance):
    """Job affordance: Money (++ varies by energy/hygiene), Energy (--), Hygiene (--), Stress (++)"""

    def interact(self, agent: Agent) -> dict:
        """Dynamic job payment based on agent condition."""
        energy_normalized = agent.meters.get("energy").normalize()
        hygiene_normalized = agent.meters.get("hygiene").normalize()

        # Base payment: $30
        # Penalty: -50% if energy < 40% OR hygiene < 40%
        base_payment = 30.0

        if energy_normalized < 0.4 or hygiene_normalized < 0.4:
            payment = base_payment * 0.5  # $15 (tired or dirty = poor performance)
        else:
            payment = base_payment  # $30 (healthy = full productivity)

        effects = {
            "money": payment,
            "energy": -15.0,
            "hygiene": -10.0,
            "stress": 25.0,
        }

        agent.meters.update_all(effects)
        return effects
```

### Why This Is Brilliant

**Hidden Variable Problem**:

- Job doesn't directly check if you HAVE energy/hygiene
- It checks if you have ENOUGH energy/hygiene
- Payment varies: $15 to $30 (2x difference!)

**Cascading Failure Mode**:

```
Neglect energy → Work for $15 → Can't afford Bed → Energy stays low →
Work for $15 again → Money depletes → Can't afford anything → Death
```

**Learning Challenge**:

- Early agent: "I can work anytime, it always gives money!"
- Reality: Working while tired gives HALF the money
- Discovery: "Oh! I need to maintain energy/hygiene to maximize income!"

### Pedagogical Value

**Real-world analogy**: Burnout spiral

- Tired employee → poor performance → lower income
- Lower income → more stress → worse performance
- Self-care is not optional, it's an **economic multiplier**

**AI lesson**: Variables interact across domains

- Biological meter (energy) affects economic meter (money)
- Can't optimize meters in isolation
- Must model **cross-domain dependencies**

---

## Indirect Relationship 2: Dual Food Sources

### The Mechanic

**Files**: `src/hamlet/environment/entities.py`, `affordances.py`

```python
class HomeMeal(Affordance):
    """HomeMeal: Cheap, healthy home cooking. Money (-), Satiation (++), Energy (++)"""
    # Costs $3, +45 satiation, +35 energy
    # Located at (1, 3) - Home zone

class FastFood(Affordance):
    """FastFood: Expensive, convenient near work. Money (---), Satiation (++), Energy (+)"""
    # Costs $10, +45 satiation, +15 energy
    # Located at (5, 6) - Near Job (6, 6)
```

**Economics**:

| Source | Cost | Satiation | Energy | Location | Distance from Job |
|--------|------|-----------|--------|----------|-------------------|
| HomeMeal | $3 | +45 | +35 | (1,3) | 9 steps |
| FastFood | $10 | +45 | +15 | (5,6) | 1 step |

### The Trade-off

**When agent is at home (1,1 to 2,2)**:

- HomeMeal is 1-2 steps away
- FastFood is 7-8 steps away
- **Optimal**: Use HomeMeal (close + cheap + healthy)

**When agent is at work (6,6)**:

- HomeMeal is 9 steps away (5 + 3 = 8 movement + 1 interact = 9 actions)
- FastFood is 1 step away
- **Trade-off**: HomeMeal saves $7 but costs 8 extra actions
  - 8 movements × 0.5 energy = 4 energy lost in travel
  - HomeMeal gives 35 energy, FastFood gives 15 energy
  - Net energy gain: HomeMeal path = 35-4 = 31, FastFood = 15
  - But HomeMeal path takes 9 actions vs 1 action!

**Emergent complexity**:

- Is it worth traveling home for cheaper food?
- Depends on: money buffer, energy level, time pressure
- No single "correct" answer - **context-dependent optimization**

### Why This Is Brilliant

**Location-aware decision-making**:

- Same need (hunger) has different optimal solutions based on position
- Agent must learn: "Where am I? What's nearby? What's my money situation?"
- Not just "if hungry, eat" - now "if hungry AND at work AND low money, travel home first"

**Economic pressure**:

- FastFood costs 3.3× more than HomeMeal ($10 vs $3)
- But convenience has value (1 action vs 9 actions)
- Agent discovers: "I'm paying $7 for 8 actions of saved time"

**Multi-step planning**:

- Naive: Work → hungry → FastFood → broke → death
- Smart: Work → travel home → HomeMeal → Bed → travel back → Work
- Must plan ahead: "I'll be hungry after work, should I go home first?"

### Pedagogical Value

**Real-world analogy**: Fast food vs home cooking

- Fast food is expensive but convenient
- Home cooking is cheap but time-consuming
- Choice depends on: budget, time, location
- No universally "best" option

**AI lesson**: Spatial context matters

- Same problem (hunger) has location-dependent solutions
- Must integrate position into decision-making
- Value trade-offs (money vs time vs distance)

---

## Spatial Clustering: Three Zones

### The Layout

**File**: `src/hamlet/training/config.py`

```python
self.affordance_positions = {
    # HOME ZONE (top-left cluster - close but not adjacent)
    "Bed": (1, 1),        # Sleep
    "Shower": (2, 2),     # Clean (diagonal from bed)
    "HomeMeal": (1, 3),   # Cheap healthy food at home

    # SOCIAL ZONE (center cluster)
    "Recreation": (3, 3), # Entertainment
    "Bar": (4, 4),        # Social + food + stress relief

    # WORK ZONE (bottom-right cluster)
    "FastFood": (5, 6),   # Expensive convenience food near work
    "Job": (6, 6),        # Work (pays less if tired/dirty)
}
```

**Visual Map**:

```
   0  1  2  3  4  5  6  7
0  .  .  .  .  .  .  .  .
1  . 🛏️ .  .  .  .  .  .   Home Zone
2  .  . 🚿 .  .  .  .  .
3  . 🥘 .  🎮 .  .  .  .
4  .  .  .  . 🍺 .  .  .   Social Zone
5  .  .  .  .  .  .  .  .
6  .  .  .  .  . 🍔💼 .   Work Zone
7  .  .  .  .  .  .  .  .
```

### Zone Characteristics

**Home Zone (1,1 to 2,2)**:

- **Purpose**: Biological maintenance
- **Affordances**: Bed (energy), Shower (hygiene), HomeMeal (satiation + energy)
- **Cost**: Low ($3-5 per service)
- **Strategic role**: Recovery base, cheap maintenance

**Social Zone (3,3 to 4,4)**:

- **Purpose**: Mental health maintenance
- **Affordances**: Recreation (stress), Bar (social + stress)
- **Cost**: Medium to high ($8-15)
- **Strategic role**: Psychological upkeep, expensive but mandatory

**Work Zone (5,6 to 6,6)**:

- **Purpose**: Income generation
- **Affordances**: Job (money), FastFood (convenience satiation)
- **Cost**: FastFood expensive ($10), Job generates income ($15-30)
- **Strategic role**: Economic engine, high-stress area

### Distance Matrix

| From → To | Home | Social | Work |
|-----------|------|--------|------|
| **Home** | 0-2 | 3-5 | 7-10 |
| **Social** | 3-5 | 0-2 | 3-5 |
| **Work** | 7-10 | 3-5 | 0-2 |

**Key insight**: Social zone is equidistant from Home and Work (3-5 steps)

- Acts as a "hub" or "transition point"
- Agent can stop at Bar/Recreation while traveling between zones

### Strategic Implications

**Zone-based strategies emerge**:

**Strategy 1: Commuter Pattern**

```
Home (sleep + clean + eat) → Work → Work → Home (repeat)
Skip social zone → high stress → eventually Recreation needed → death
```

**Failure mode**: Neglecting stress/social leads to burnout

**Strategy 2: Balanced Cycle**

```
Home (Bed + Shower + HomeMeal) → Work → Social (Recreation or Bar) → Home
```

**Success**: Maintains all meters, economically sustainable

**Strategy 3: Work-Social Loop**

```
Work → FastFood → Work → Bar → Work (repeat)
Never go home → expensive food + skip cheap services → broke
```

**Failure mode**: Ignoring cheap home services leads to poverty

### Why This Is Brilliant

**Spatial affordances create natural routines**:

- Agent discovers: "I should do multiple things in a zone before traveling"
- E.g., "While I'm home, do Bed + Shower + HomeMeal" (batching)
- Real-world analogy: Running errands in one trip vs multiple trips

**Travel cost creates planning pressure**:

- Each zone is 7-10 steps from opposite zone
- Movement depletes energy/hygiene/satiation
- "Should I go home now or wait until I need multiple things?"

**Zone positioning models real life**:

- Work far from home (daily commute)
- Social zone in between (stopping for drinks after work)
- Convenience food near work (lunch rush)

### Pedagogical Value

**Real-world analogy**: Daily routines and commutes

- Home/work separation creates structure
- Social activities fit in between
- Proximity affects choices (lunch near work vs going home)

**AI lesson**: Spatial reasoning and planning

- Must consider position in decision-making
- Batching actions in zones is efficient
- Route planning matters (don't zigzag unnecessarily)

---

## Combined Complexity: The Full System

### How Indirect Relationships Compound

**Scenario 1: The Tired Worker Trap**

```
Agent is at Work (6,6), low energy (30%), hungry

Option A: Work now
  - Payment: $15 (penalty for low energy)
  - Energy after: 15% (depleted further)
  - Can't afford Bed + HomeMeal ($8 total)
  → Spiral into poverty

Option B: Go home first
  - Travel to Bed (1,1): 7-10 steps
  - Use Bed ($5, +50 energy) → 80% energy
  - Use HomeMeal ($3, +45 satiation, +35 energy) → full energy
  - Travel back to Work (6,6): 7-10 steps
  - Work with high energy: $30 payment
  → Net gain: -$8 + $30 = +$22 (vs +$15 in Option A)
```

**Learning challenge**: Agent must discover that **spending money on self-care increases income**.

**Scenario 2: The FastFood Temptation**

```
Agent at Work (6,6), hungry, $20 in pocket

Option A: FastFood (1 step away, $10)
  - Quick and easy
  - Remaining money: $10
  - Can't afford Bar ($15) for social
  → Social depletes → penalties → eventual death

Option B: Travel home (9 steps), HomeMeal ($3)
  - Takes longer
  - Remaining money: $17
  - Can afford Bar ($15) for social
  → All meters maintained → survival
```

**Learning challenge**: Agent must learn **delayed gratification** (travel time for long-term benefit).

**Scenario 3: The Bar Cascade**

```
Social meter critical (15%), must go to Bar

Bar effects: -$15, -20 energy, -15 hygiene, +50 social

After Bar:
  - Money: -$15 (need to work soon)
  - Energy: Low (need Bed, costs $5)
  - Hygiene: Low (need Shower, costs $3)

Total recovery cost: $15 + $5 + $3 = $23
Must work at least once (earn $15-30) to recover

If energy low when working → only earn $15 → deficit of $8!
Must maintain energy BEFORE working to earn full $30
```

**Learning challenge**: Agent must plan **multi-step cascades** (Bar → Bed → Shower → Work).

### State Space Complexity

**Dimensions**:

- Position: (x, y) = 64 possible locations
- Meters: 6 values × [0-100] range = ~10^12 combinations
- Spatial context: 3 zones = 3 different strategic contexts
- Economic state: Money buffer = continuous variable

**Decision complexity**:

- Before: "Which meter is lowest? Go there."
- After: "Which meter is lowest? Where am I? Can I afford to travel? Will I be healthy enough to work later? Should I batch actions?"

**Planning horizon**:

- Before: 1-step lookahead ("hungry now → eat now")
- After: 3-5 step lookahead ("hungry at work → go home → eat cheap → sleep → work healthy for full pay")

---

## Updated Economic Model

### Cost Breakdown

**Full cycle maintenance** (visiting all zones):

```
Home zone:     Bed ($5) + Shower ($3) + HomeMeal ($3)       = $11
Social zone:   Recreation ($8) + Bar ($15)                   = $23
Work zone:     FastFood ($10)                                = $10
TOTAL:                                                        = $44
```

**Income sources**:

```
Job (healthy):   $30 per visit (energy > 40%, hygiene > 40%)
Job (unhealthy): $15 per visit (energy < 40% OR hygiene < 40%)
```

**Economics**:

- Healthy cycle: $30 income - $44 costs = **-$14 deficit**
- Unhealthy cycle: $15 income - $44 costs = **-$29 deficit**

**Sustainable strategies**:

1. Work 2× per cycle (healthy) = $60 income → +$16 surplus
2. Skip optional services (Recreation or FastFood)
3. Work 3× per cycle (allow unhealthy sometimes)

### Strategic Patterns

**Pattern 1: Home-Work Shuttle** (simplest, risky)

```
Home (Bed + Shower + HomeMeal) → Work → Work → Home (repeat)
Income: $60 per cycle
Costs: $11 (home services)
Surplus: +$49

Risk: Stress/social accumulate → eventually need Bar/Recreation → surprise deficit
```

**Pattern 2: Balanced Maintenance** (sustainable)

```
Home (all) → Work → Work → Social (Bar) → Bed → Shower → (repeat)
Income: $60 per cycle (2 works)
Costs: $11 (home) + $15 (Bar) = $26
Surplus: +$34

Risk: Skipping Recreation → stress builds → need eventually
```

**Pattern 3: Full Coverage** (optimal but tight)

```
Home (all) → Work → Work → Social (Bar + Recreation) → Home (Bed + Shower) → (repeat)
Income: $60 per cycle (2 works)
Costs: $11 (home) + $23 (social) = $34
Surplus: +$26

Requires: Healthy work performance (can't afford unhealthy penalty)
```

### Failure Modes

**Failure 1: FastFood Addiction**

- Using FastFood instead of HomeMeal costs $7 extra per meal
- Over 10 meals: $70 deficit
- Leads to poverty → can't afford mandatory Bar → social collapse

**Failure 2: Unhealthy Work**

- Working with low energy → only $15 income (vs $30)
- Deficit per work: $15 loss
- After 5 unhealthy works: $75 lost
- Can't recover economically

**Failure 3: Ignoring Social**

- Social depletes 0.6/step
- After 83 steps without Bar: social = 0
- Penalties accumulate (-2.0 reward per step)
- Eventually death or inefficiency

---

## Agent Learning Challenges

### Challenge 1: Discover Job Payment Penalty

**What agent must learn**:

- "Working while tired gives less money"
- Hidden relationship: energy/hygiene → money

**How agent discovers it**:

1. Early episodes: Work randomly (sometimes healthy, sometimes tired)
2. Experience replay: Compare outcomes
3. Pattern recognition: "I earned $30 this time but $15 that time... what's different?"
4. Hypothesis: Energy/hygiene affects payment
5. Confirmation: Test by maintaining energy before work

**Time to learn**: ~200-400 episodes (needs sufficient variation in states)

### Challenge 2: Learn HomeMeal vs FastFood Trade-off

**What agent must learn**:

- "HomeMeal is better value BUT depends on location"
- Context-dependent decisions

**How agent discovers it**:

1. Initially: Use whichever food source is closer
2. Observation: Money depletes faster when using FastFood often
3. Pattern: "I use FastFood when at work, HomeMeal when at home"
4. Optimization: "Should I travel home before getting hungry?"
5. Strategy: Plan meals based on route (going home anyway → use HomeMeal)

**Time to learn**: ~300-500 episodes (requires spatial reasoning)

### Challenge 3: Multi-step Planning (Bar Cascade)

**What agent must learn**:

- "Bar requires follow-up actions (Bed + Shower)"
- "Must have enough money for the full sequence"
- "Must time Bar visit for when I can afford recovery"

**How agent discovers it**:

1. First Bar visit: Social fixed, but now low energy/hygiene
2. Death shortly after: "Why did I die after fixing social?"
3. Realizes: Bar creates new problems (energy/hygiene drain)
4. Learns: Bar → Bed → Shower sequence
5. Plans: Wait until $23 in pocket before Bar

**Time to learn**: ~500-800 episodes (complex multi-step reasoning)

---

## Metrics to Track

### Performance Indicators

| Metric | Expected (Learned) | Explanation |
|--------|-------------------|-------------|
| Job payment ratio | 85%+ at $30 | Agent maintains energy/hygiene before work |
| HomeMeal/FastFood ratio | 70/30 or higher | Agent prefers cheap option, uses FastFood only when convenient |
| Actions per zone visit | 2-3 actions | Agent batches activities in zones |
| Money buffer | $20-40 average | Agent maintains cushion for Bar + recovery |
| Bar visit timing | Every 60-80 steps | Proactive social maintenance (before critical) |
| Death by social | <5% | Agent learns social criticality |
| Death by poverty | <10% | Agent manages economy sustainably |

### Complexity Indicators

| Indicator | Measurement | Desired Outcome |
|-----------|-------------|-----------------|
| Planning depth | Steps between decision changes | 3-5 steps (shows lookahead) |
| Zone batching | Multiple actions in same zone | 60%+ visits |
| Healthy work rate | % of jobs worked with >40% energy/hygiene | 80%+ |
| FastFood usage | Only when at work + urgent | Context-appropriate |
| Pre-emptive maintenance | Visiting home before meters critical | Proactive patterns |

---

## Pedagogical Applications

### Teaching Moment 1: Indirect Causality

**Lesson**: Variables interact across domains

- Biological state (energy) affects economic outcome (money)
- Can't optimize one meter in isolation
- Must understand **cross-domain dependencies**

**Real-world parallel**: Health → Productivity → Income

- Personal health affects career performance
- Economic decisions affect health (time for exercise, sleep)
- System thinking required

### Teaching Moment 2: Spatial Context

**Lesson**: Location affects optimal decisions

- Same problem (hunger) has location-dependent solutions
- Must integrate position into planning
- Convenience has value (time vs money trade-off)

**Real-world parallel**: Daily life logistics

- Where you are affects what you do (lunch near work vs home)
- Batching errands saves time
- Commute patterns shape routines

### Teaching Moment 3: Delayed Gratification

**Lesson**: Immediate convenience vs long-term benefit

- FastFood is quick but expensive → short-term thinking
- HomeMeal requires travel but saves money → long-term thinking
- Trade-offs depend on context (is money tight? is time critical?)

**Real-world parallel**: Financial decisions

- Credit cards (convenient but expensive) vs saving (delayed but beneficial)
- Fast food vs meal prep
- Short-term gains vs long-term health

### Teaching Moment 4: Cascading Dependencies

**Lesson**: Actions have follow-on effects

- Bar visit creates energy/hygiene problems
- Those problems require money to fix
- Must plan full sequence, not just immediate action

**Real-world parallel**: Life decisions

- Social event → tired next day → need recovery time
- Buying house → mortgage → need stable income → lifestyle constraints
- Multi-step consequences

---

## Implementation Summary

### Files Modified

**Backend**:

1. `src/hamlet/environment/entities.py`:
   - Made Job.interact() dynamic (payment varies by energy/hygiene)
   - Split Fridge into HomeMeal and FastFood classes

2. `src/hamlet/environment/affordances.py`:
   - Updated AFFORDANCE_EFFECTS for HomeMeal and FastFood
   - Updated economic balance comments

3. `src/hamlet/training/config.py`:
   - Reorganized affordance_positions into three spatial zones

4. `src/hamlet/environment/hamlet_env.py`:
   - Updated grid encoding for 7 affordances (HomeMeal=3, FastFood=4, etc.)
   - Updated proximity shaping to guide to HomeMeal for satiation

**Frontend**:

1. `frontend/src/components/Grid.vue`:
   - Added HomeMeal (🥘) and FastFood (🍔) icons
   - Updated CSS for homemeal (orange) and fastfood (red) styling

**Documentation**:

1. `docs/indirect_relationships_and_spatial_complexity.md`: This file

---

## Expected Training Results

### Hypothesis: Learning Stages

**Stage 1 (Episodes 0-200): Random Exploration**

- Agent uses affordances randomly
- Frequent deaths (no strategy)
- Job payment varies wildly (sometimes $15, sometimes $30)
- No spatial batching

**Stage 2 (Episodes 200-500): Direct Relationships**

- Agent learns: "Low energy → Bed", "Low satiation → Food"
- Still no job payment optimization (hasn't discovered penalty)
- Random food choice (doesn't optimize HomeMeal vs FastFood)
- Survives longer but inefficiently

**Stage 3 (Episodes 500-800): Indirect Discovery**

- Agent discovers: "Job pays more when healthy"
- Starts maintaining energy/hygiene before work
- Still inefficient food choices (doesn't consider location)
- Economic management improves

**Stage 4 (Episodes 800-1200): Spatial Optimization**

- Agent discovers: "HomeMeal cheaper, FastFood convenient"
- Context-aware food choices (location-based)
- Zone batching emerges (multiple actions per zone visit)
- Sustainable economic patterns

**Stage 5 (Episodes 1200+): Sophisticated Strategy**

- Multi-step planning (Bar → Bed → Shower → Work sequence)
- Proactive maintenance (fix meters before critical)
- Optimal routing (minimize travel)
- High survival rate (450+ steps average)

### Comparison to Previous Versions

| Metric | 4-Meter System | 5-Meter (+Stress) | 6-Meter (+Social) | 7-Affordance (+Spatial) |
|--------|---------------|-------------------|-------------------|------------------------|
| Avg Survival | ~300 steps | ~350 steps | ~400 steps | ~450 steps (predicted) |
| Planning Depth | 1 step | 1-2 steps | 2-3 steps | 3-5 steps |
| Economic Sustainability | Easy (+$10/cycle) | Easy (+$10/cycle) | Tight (-$5/cycle) | Very tight (-$14/cycle) |
| Strategic Variety | Low (1-2 viable) | Medium (2-3 viable) | High (3-4 viable) | Very high (4+ viable) |
| State Space Size | ~10^8 | ~10^10 | ~10^12 | ~10^14 |

---

## Future Enhancements (Not Yet Implemented)

### Enhancement 1: Social→Stress Amplification

```python
# In meter depletion:
social_normalized = agent.meters.get("social").normalize()

if social_normalized < 0.2:
    stress_multiplier = 3.0  # Loneliness amplifies stress
else:
    stress_multiplier = 1.0

# When working:
stress_gain = 25.0 * stress_multiplier  # 75 stress if lonely!
```

**Result**: Social becomes even more critical (prevents stress spiral).

### Enhancement 2: Dynamic FastFood Pricing (Supply/Demand)

```python
# FastFood costs more during "lunch rush" (simulation time 11-13)
if simulation_hour in [11, 12, 13]:
    fastfood_cost = 15.0  # Peak pricing
else:
    fastfood_cost = 10.0  # Normal pricing
```

**Result**: Agent learns temporal patterns (eat before/after rush).

### Enhancement 3: HomeMeal Cooking Time

```python
# HomeMeal requires 2 actions: "Prepare" then "Eat"
# Must spend 2 consecutive turns at HomeMeal location
```

**Result**: HomeMeal more expensive in time (but still cheaper in money).

---

## Conclusion

The combination of **indirect relationships** and **spatial complexity** transforms Hamlet into a sophisticated multi-objective optimization problem requiring:

1. **Cross-domain reasoning**: Biological → Economic → Social relationships
2. **Context-aware decisions**: Location-dependent trade-offs
3. **Multi-step planning**: Cascade management and zone batching
4. **Delayed gratification**: Long-term benefit vs short-term convenience
5. **Systemic thinking**: Understanding hidden dependencies

**Result**: Agent must develop **human-like strategic thinking** to survive.

**Pedagogical value**: Teaches that real-world problems require:

- Understanding indirect effects
- Considering spatial context
- Planning multiple steps ahead
- Balancing competing objectives
- Adapting strategies to context

**Next**: Train agent and observe whether it discovers these sophisticated patterns!
