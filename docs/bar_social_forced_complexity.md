# Bar & Social: Mandatory Resource Sink Design

**Date**: 2025-10-28
**Purpose**: Add complexity to prevent "standing around" strategies
**Core Mechanic**: Social meter with ONLY one source (Bar)
**Result**: Agent must engage with expensive, multi-cost affordance

---

## Design Philosophy: Forced Complexity

### The Problem
With 4-5 meters, agents could find "lazy" strategies:
- Optimize subset of meters
- Ignore expensive affordances
- Find equilibrium without full engagement

### The Solution: Mandatory Sink
Add a meter (Social) that:
- **Depletes over time** (like biological needs)
- **Only ONE source** (Bar - can't substitute)
- **That source is expensive** (costs multiple resources)
- **Creates forced trade-offs** (must sacrifice to maintain)

**Result**: Agent MUST engage with entire system. No shortcuts.

---

## New Meter: Social

**File**: `src/hamlet/environment/meters.py`

```python
class Social(Meter):
    """Social meter: depletes over time, ONLY restored by Bar (mandatory sink)."""

    def __init__(self):
        # Starts at 50 (mid-level), depletes faster than bio meters
        super().__init__(name="social", initial_value=50.0, depletion_rate=0.6)
```

**Mechanics**:
- Initial value: 50 (starts moderate)
- Depletion rate: 0.6 per step (faster than energy/satiation)
- **Critical distinction**: ONLY Bar restores social
  - Bed/Shower/Fridge/Job/Recreation: All have ZERO social effect
  - Bar: +50 social (significant boost)

**Reward structure**:
```python
# Social treated like biological meters
if social_normalized > 0.8:
    reward += 0.5  # Healthy social life
elif social_normalized > 0.5:
    reward += 0.2  # Adequate
elif social_normalized > 0.2:
    reward -= 0.5  # Lonely
else:
    reward -= 2.0  # Isolated (critical)
```

---

## New Affordance: Bar

**File**: `src/hamlet/environment/entities.py`, `affordances.py`

```python
class Bar(Affordance):
    """Bar: Social hub - expensive multi-cost, mandatory for social"""

AFFORDANCE_EFFECTS = {
    "Bar": {
        "money": -15.0,    # Expensive night out
        "energy": -20.0,   # Tiring (late night)
        "hygiene": -15.0,  # Get dirty/sweaty
        "social": 50.0,    # ONLY source! (mandatory)
        "satiation": 30.0, # Eat while there
        "stress": -25.0,   # Social interaction reduces stress
    },
}
```

**Position**: (4, 4) - Near center, social hub
**Icon**: 🍺 (beer mug, pink background)

### Why These Costs?

**Money (-$15)**: Most expensive single transaction
- Makes Bar a significant economic decision
- Can't afford Bar + all services every cycle
- Forces prioritization

**Energy (-20)**: Highest energy cost
- Late night out = exhaustion
- Creates recovery need (must go to Bed afterward)

**Hygiene (-15)**: High hygiene cost
- Crowded bar = get dirty/sweaty
- Creates follow-up need (must Shower)

**Benefits are strong but create cascades**:
- Social (+50): Large boost, but...
- Satiation (+30): Helpful, but...
- Stress (-25): Good, but...
- **After Bar**: Low energy + low hygiene → must use Bed + Shower → costs more money

---

## Updated Economics: Deficit Forcing

### Old Balance (Recreation):
- Full cycle: Bed + Shower + Fridge + Recreation = $20
- Income: 1 Job = $30
- Net: +$10/cycle (sustainable)

### New Balance (Bar added):
- Full services: Bed + Shower + Fridge + Recreation + Bar = $35
- Income: 1 Job = $30
- **Net: -$5/cycle** (unsustainable!)

### Strategic Implications:

**Can't afford everything every cycle** - Agent must:
1. **Prioritize Bar** (mandatory for social)
2. **Work more often** (2 Jobs per "full cycle")
3. **Skip optional services** (Recreation can be delayed)
4. **Chain efficiently** (Bar → Bed → Shower sequence)

**Sustainable patterns**:
- Work-Work-Bar-Bed-Shower-Fridge = Break even
- Work-Bar-Bed-Shower-Work-Fridge-Recreation = Slight positive
- Must discover multi-step planning

---

## The "Mandatory Sink" Pattern

### What Makes This Design Brilliant:

**1. Single Source Constraint**
- Social ONLY from Bar
- No substitution possible
- Creates hard dependency

**2. Expensive Source**
- Bar costs most ($15)
- Also costs energy + hygiene
- Can't be spammed

**3. Cascading Effects**
- Bar → Need Bed (energy)
- Bar → Need Shower (hygiene)
- Those cost money
- Must work more

**4. Strategic Depth**
- When to go to Bar? (before social critical)
- How to afford it? (work first)
- What to skip? (Recreation vs Fridge trade-offs)

**Result**: Standing around is NEVER optimal. Always something critical.

---

## Observed Decision Tree

Agent must learn this decision logic:

```
IF social < 30%:
    IF money < $15:
        → Go to Job (need money for Bar)
    ELSE:
        → Go to Bar (mandatory social source)
ELSE IF energy < 20%:
    → Go to Bed
ELSE IF hygiene < 20%:
    → Go to Shower
ELSE IF satiation < 20%:
    → Go to Fridge
ELSE IF money < 40%:
    → Go to Job (build buffer)
ELSE IF stress > 70%:
    → Go to Recreation (if affordable)
```

**Complexity**: 6 meters, 6 affordances, multi-step planning required

---

## State Space Growth

### Meter Count Progression
- **Original**: 4 meters (energy, hygiene, satiation, money)
- **+ Stress**: 5 meters (added work-life balance)
- **+ Social**: 6 meters (added mandatory sink)

### Affordance Count Progression
- **Original**: 4 affordances (Bed, Shower, Fridge, Job)
- **+ Recreation**: 5 affordances (stress management)
- **+ Bar**: 6 affordances (social requirement)

### State Dimensionality
- Grid: 8×8 = 64 cells
- Meters: 6 values
- Position: 2 values (x, y)
- **Total**: 72-dimensional state vector

**Observation**: State space grows linearly, but strategic complexity grows combinatorially.

---

## Frontend Visualization

### Meter Panel
- Social displays as percentage (like bio meters)
- Pink color scheme (#ec4899)
- Critical threshold: <20%

### Grid
- Bar renders at (4, 4)
- Icon: 🍺 (beer mug)
- Pink background (#ec4899)

---

## Future Complexity Layer: Social→Stress Amplification

**NOTE**: Not yet implemented, but planned for next iteration

### The Mechanic
```python
# In meter depletion:
social_normalized = agent.meters.get("social").normalize()

if social_normalized < 0.2:
    # Loneliness amplifies stress
    stress_multiplier = 3.0  # 3x faster stress accumulation
else:
    stress_multiplier = 1.0  # Normal stress rate

# Apply to Job stress effect
if action == Job:
    stress_gain = 25.0 * stress_multiplier  # 75 stress if lonely!
```

### Why This Is Brilliant

**Indirect Causality**:
- Social doesn't kill directly
- Low social → stress increases faster
- Max stress → death
- Agent must discover: Social → Stress → Death chain

**Hidden Variable Problem**:
- "Social doesn't seem to kill me..."
- "But when it's low, I die faster... why?"
- "Oh! It amplifies stress from work!"
- Emergent understanding of complex systems

**Strategic Implications**:
- Can't just optimize meters independently
- Must understand relationships
- Social becomes even more critical
- Bar visits become more strategic (maintain buffer to prevent stress spiral)

**Pedagogical Value**:
- Teaches indirect effects
- Shows systemic thinking
- Models real-world complexity (isolation → burnout → breakdown)

---

## Pedagogical Applications

### Teaching Moment 1: Mandatory Diversity

**Lesson**: Can't optimize subset of system
- Student might try: "Just work and use basic services"
- Reality: Social depletes → penalties accumulate → death
- Discovery: "Oh, I HAVE to go to Bar"

**Analogy**: Real life - can't ignore social needs even if inconvenient

### Teaching Moment 2: Expensive Necessities

**Lesson**: Some needs are both mandatory AND costly
- Bar is most expensive single action
- But social is mandatory
- Must find money for it

**Analogy**: Real life - healthcare, housing, education (expensive but necessary)

### Teaching Moment 3: Cascading Resource Management

**Lesson**: Actions have follow-on effects
- Bar → low energy → need Bed → costs money
- Chain reactions in resource systems
- Planning must account for sequences

**Analogy**: Real life - social event → tired → skip work → less money

### Teaching Moment 4: Trade-offs Under Scarcity

**Lesson**: Can't afford everything
- Income ($30/job) < Full services ($35)
- Must prioritize
- Different strategies viable

**Analogy**: Real life - budgeting, opportunity costs

---

## Expected Agent Behaviors

### Predicted Strategy 1: "Social Prioritizer"
```
Work → Work → Bar → Bed → Shower → Work → Fridge → (repeat)
```
- Maintains social (mandatory)
- Works often (needs money for Bar)
- Skips Recreation (not mandatory)
- Sustainable but stressful

### Predicted Strategy 2: "Balanced Budget"
```
Work → Bar → Bed → Work → Shower → Fridge → Recreation → (repeat)
```
- Manages all 6 meters
- Works 2x per cycle (income = costs)
- Includes Recreation (stress management)
- More complex but sustainable

### Predicted Strategy 3: "Reactive Crisis" (sub-optimal)
```
(random movement until meter critical) → (desperate action) → (repeat)
```
- No planning
- Frequent deaths
- Agent should learn this fails

**Optimal**: Likely Strategy 2 (balanced 6-meter management)

---

## Testing Validation

### Test 1: Social Depletion
```bash
uv run python -c "
from src.hamlet.environment.hamlet_env import HamletEnv
env = HamletEnv()
env.reset()
agent = env.agents['agent_0']

print(f'Initial social: {agent.meters.get(\"social\").value}')

# Wait 20 steps (no Bar visit)
for _ in range(20):
    env.step(0)

print(f'After 20 steps: {agent.meters.get(\"social\").value}')
print(f'Expected: ~38 (50 - 20*0.6)')
"
```

### Test 2: Bar Effects
```bash
uv run python -c "
from src.hamlet.environment.hamlet_env import HamletEnv
env = HamletEnv()
env.reset()
agent = env.agents['agent_0']

# Move to Bar and use it
env.reset()
for _ in range(10):
    env.step(1)  # Move to (4,4)

before = {
    'money': agent.meters.get('money').value,
    'energy': agent.meters.get('energy').value,
    'hygiene': agent.meters.get('hygiene').value,
    'social': agent.meters.get('social').value,
}

env.step(4)  # INTERACT at Bar

after = {
    'money': agent.meters.get('money').value,
    'energy': agent.meters.get('energy').value,
    'hygiene': agent.meters.get('hygiene').value,
    'social': agent.meters.get('social').value,
}

print('Bar visit effects:')
print(f'  Money: {before[\"money\"]:.1f} → {after[\"money\"]:.1f} ({after[\"money\"]-before[\"money\"]:.1f})')
print(f'  Energy: {before[\"energy\"]:.1f} → {after[\"energy\"]:.1f} ({after[\"energy\"]-before[\"energy\"]:.1f})')
print(f'  Hygiene: {before[\"hygiene\"]:.1f} → {after[\"hygiene\"]:.1f} ({after[\"hygiene\"]-before[\"hygiene\"]:.1f})')
print(f'  Social: {before[\"social\"]:.1f} → {after[\"social\"]:.1f} ({after[\"social\"]-before[\"social\"]:.1f})')
"
```

### Test 3: Economic Deficit
```bash
# Verify can't afford full cycle on 1 Job
# Full services = $35, Job = $30
# Deficit = -$5
```

---

## Comparison to Real-World AI Systems

### The "Social Media Engagement" Problem

**Real AI systems** (recommendation algorithms):
- Optimize engagement (clicks, time-on-site)
- Ignore well-being (stress, sleep, real social connections)
- Result: Addictive but harmful patterns

**Hamlet Bar mechanic teaches**:
- Must optimize MULTIPLE competing objectives
- Some needs are expensive but mandatory
- Ignoring one dimension causes cascade failures
- No "lazy" equilibrium - must engage fully

**Lesson**: Multi-objective optimization is hard but necessary

---

## Implementation Checklist

- [x] Add Social meter (depletion rate 0.6)
- [x] Add Bar affordance (6-resource multi-cost)
- [x] Update grid encoding (Bar = 6.0, agent = 7.0)
- [x] Add social to gradient rewards
- [x] Add social to proximity shaping
- [x] Add social → Bar mapping
- [x] Update config (Bar at position 4,4)
- [x] Update frontend Grid (🍺 icon, pink)
- [x] Update frontend MeterPanel (social color)
- [ ] Train new agent with 6-meter complexity
- [ ] Document observed strategies
- [ ] (Future) Implement social→stress amplification

---

## Files Modified

### Backend
- `src/hamlet/environment/meters.py`: Added Social class
- `src/hamlet/environment/entities.py`: Added Bar class
- `src/hamlet/environment/affordances.py`: Added Bar effects, updated economics
- `src/hamlet/environment/hamlet_env.py`:
  - Grid encoding: Bar = 6.0, agent = 7.0
  - Gradient rewards: Added social
  - Proximity shaping: Added social → Bar
- `src/hamlet/training/config.py`: Added Bar position (4, 4)

### Frontend
- `frontend/src/components/Grid.vue`: Bar icon + styling
- `frontend/src/components/MeterPanel.vue`: Social color scheme

### Documentation
- `docs/bar_social_forced_complexity.md`: This file

---

## Metrics to Track

### Agent Performance
| Metric | Expected |
|--------|----------|
| Bar Visits/Episode | 3-5 (mandatory social maintenance) |
| Job Visits/Episode | 6-8 (need income for Bar + services) |
| Average Social Level | 40-60% (maintained above critical) |
| Death by Social | <5% (should learn criticality) |
| Death by Stress | 10-15% (complex management) |
| Average Survival | 450+ steps (longer than 5-meter) |

### Complexity Indicators
| Indicator | Measurement |
|-----------|-------------|
| Decision Diversity | Actions per episode (should use all 6 affordances) |
| Planning Depth | Steps between decisions (should plan 3-5 ahead) |
| Resource Utilization | % of money spent (should be high, near deficit) |

---

## Expected Training Challenges

### Challenge 1: Social Discovery
- Early agents may ignore social (no immediate death)
- Social depletes slowly → penalties accumulate
- Agent must learn: social critical despite delayed effect

**Solution**: Gradient rewards provide continuous feedback

### Challenge 2: Bar Cost Management
- Bar is expensive → hesitation
- But mandatory → must learn to afford it
- Requires working more often

**Solution**: Proximity shaping guides to Job when money low

### Challenge 3: Cascade Management
- Bar → low energy/hygiene → need more services → need more money
- Complex chain of dependencies
- Requires multi-step planning

**Solution**: Experience replay helps learn sequences

---

## Conclusion

The Bar/Social system transforms Hamlet from **multi-resource survival** to **forced multi-objective optimization**.

**Key innovations**:
1. ✅ Mandatory sink (social only from Bar)
2. ✅ Expensive source (Bar costs most)
3. ✅ Cascading effects (Bar → Bed → Shower → more work)
4. ✅ Economic deficit (can't afford everything)
5. ✅ No lazy equilibrium (always something critical)

**Result**: Agent must develop sophisticated, multi-step strategies to survive.

**Pedagogical value**: Teaches that real optimization problems have:
- Mandatory but expensive requirements
- Cascading resource dependencies
- No simple equilibria
- Continuous adaptation required

**Next**: Train agent and observe if it discovers sustainable 6-meter balance!

---

**Future enhancement**: Social→stress amplification adds indirect causality layer, teaching hidden variable reasoning and systemic thinking.
