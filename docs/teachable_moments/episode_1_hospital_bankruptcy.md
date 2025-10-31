# Episode 1: The Hospital Bankruptcy Incident

**Date**: October 31, 2025
**Episode**: 1
**Status**: Teachable Moment 🏥💸

## What Happened

Agent spawned directly on the Hospital tile and immediately used INTERACT, bankrupting itself by spending $15 (half of starting capital of $30) on expensive emergency care it didn't need.

## Why This is Pedagogically Perfect

This incident perfectly demonstrates multiple RL learning challenges:

### 1. **Credit Assignment Problem**
- Agent knows: "I'm standing on something" (affordance_type = Hospital)
- Agent knows: "INTERACT does... something"
- Agent doesn't know: "This specific affordance costs $15 and is a trap when healthy"

The agent has learned the **surface pattern** (INTERACT on affordances) but not the **deep strategy** (economic constraints and conditional value).

### 2. **Exploration vs. Exploitation Trade-off**
At episode 1, with epsilon ≈ 1.0 (full exploration), the agent is rightfully exploring all possible actions. This bankruptcy is not a bug—it's **necessary exploration** to learn:
- Some affordances cost money
- Money is finite
- Hospital is expensive relative to alternatives
- Health status matters for affordance value

### 3. **State-Dependent Value Functions**
Hospital has different values depending on state:
- **Health = 0.2**: Hospital is critical (value = +preventing death)
- **Health = 1.0**: Hospital is wasteful (value = -$15 for nothing)

The agent must learn this conditional relationship through experience, which requires experiencing the bad outcome first.

## Expected Learning Progression

### Episode 1-50: "Everything is a Button"
- Spawn on Hospital → INTERACT → bankrupt
- Spawn on Bar → INTERACT → drain energy/hygiene
- Spawn on Job → INTERACT → "oh, money goes up!"

**Learning**: Affordances do different things

### Episode 50-200: "Some Buttons are Bad"
- Standing on Hospital with full health → move away (avoiding waste)
- Standing on Hospital with low health → INTERACT (emergency response)

**Learning**: Context matters, economic constraints exist

### Episode 200-500: "Strategic Resource Management"
- Proactive Job visits to maintain money buffer
- Cheap affordances (Bed $5, Shower $3) preferred over expensive ones
- Hospital reserved for true emergencies

**Learning**: Economic cycles, preventive maintenance, strategic planning

## Related Failure Modes

### The "Interact Spam" Variant
If action masking wasn't properly implemented, agents would:
1. Bankrupt themselves on Hospital
2. Stand on Hospital forever
3. Spam INTERACT with $0 (no-op)
4. Never learn to move away and earn money

**Fix Applied**: Action masking now checks affordability—INTERACT is invalid when agent can't afford the affordance.

### The "Random Spawn Advantage"
Random spawn positions (already implemented) force agents to:
- Learn ALL affordance locations, not just one local cluster
- Discover Job exists (critical for economic sustainability)
- Experience diverse starting conditions

Without random spawns, agent might never discover Job if it spawns far from it.

## Teaching Value

### For Students Learning RL:
1. **Exploration is expensive** - Bad outcomes are necessary for learning
2. **Credit assignment is hard** - Agent must connect action → outcome → long-term consequence
3. **State-dependent rewards** - Same action has different values in different states
4. **Economic constraints** - Real-world problems have resource limitations

### For Students Learning AI Alignment:
1. **Specification gaming** - Agent optimizes what you measure, not what you mean
2. **Robustness to initial conditions** - Random spawns prevent overfitting to single scenario
3. **Reward shaping challenges** - How do you reward "use Hospital only when sick"?

## Classroom Discussion Prompts

1. **"Why doesn't the agent just avoid Hospital?"**
   - It hasn't learned yet! Episode 1 is pure exploration.
   - How many episodes until you'd expect it to learn Hospital = expensive?

2. **"Should we just remove Hospital from spawn positions?"**
   - No! That's hiding the problem, not solving it.
   - Real-world agents must handle adversarial starting conditions.

3. **"What if we gave it more starting money?"**
   - That delays the problem, doesn't solve it.
   - Agent must learn economic sustainability eventually.

4. **"How would a human learn this differently?"**
   - Humans have language, theory of mind, common sense.
   - RL agents learn from scratch—every failure is a data point.

## Implementation Details

### Before Fix (Action Masking Bug)
```python
# INTERACT valid if on affordance, regardless of affordability
action_masks[:, 4] = on_affordance
```

**Result**: Agent spams INTERACT on unaffordable affordances (no-op)

### After Fix (Affordability Check)
```python
# INTERACT valid only if on affordance AND can afford it
cost_normalized = cost_dollars / 100.0
can_afford = self.meters[:, 3] >= cost_normalized
on_affordable_affordance = on_this_affordance & can_afford
action_masks[:, 4] = on_affordable_affordance
```

**Result**: Agent forced to move away when broke, discovers Job eventually

## Metrics to Watch

Track these metrics to see if agent learns from the Hospital bankruptcy:

1. **Hospital usage frequency over time**
   - Should decrease as agent learns it's expensive
   - Should be reserved for low-health emergencies

2. **Average money buffer**
   - Early: oscillates wildly, frequent $0
   - Late: maintains buffer above $15-20 (emergency fund)

3. **Death cause distribution**
   - Early: 50% money-related (couldn't afford essentials)
   - Late: should shift to meter management failures

4. **Affordance preference learning**
   - Early: uniform distribution (exploration)
   - Late: Job > Bed/Shower/Food > Hospital (strategic preference)

## Success Criteria

Agent has "learned the lesson" when:
- Hospital usage < 5% of interactions
- Hospital usage strongly correlated with health < 0.3
- Money buffer maintained > $20 on average
- Job visits are proactive (before running out of money)

## Related Documents

- [Reward Hacking: Interact Spam](./reward_hacking_interact_spam.md)
- [Three Stages of Learning](./three_stages_of_learning.md)
- [Action Masking: Boundaries](./action_masking_boundaries.md)
- [Trick Students Pedagogy](./trick_students_pedagogy.md)

---

**Moral of the Story**: In RL, as in life, sometimes you have to make expensive mistakes to learn what not to do. The Hospital bankruptcy is not a bug—it's tuition. 🎓💸
