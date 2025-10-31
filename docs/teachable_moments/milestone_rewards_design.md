# Milestone Rewards: Why Birthdays Beat Breathing

**Date**: October 31, 2025
**Insight**: Reward shaping that prevents left-right oscillation
**Status**: Implemented ✅

## The Problem: Rewarding Every Breath

### Initial Design (BROKEN)
```python
# Base reward: +1.0 for surviving this step
rewards = torch.ones(self.num_agents, device=self.device)

# Death penalty: -100.0 for dying
rewards = torch.where(self.dones, -100.0, rewards)
```

**What the agent learns:**
- Move left → +1.0 ✅
- Move right → +1.0 ✅
- Move left → +1.0 ✅
- Move right → +1.0 ✅
- ...oscillate forever → +1.0, +1.0, +1.0...

**The agent discovers**: "Left-right oscillation is optimal! I get constant +1.0 rewards until my meters run out ~200 steps later."

### Why This is a Classic RL Failure Mode

1. **Reward ≠ Goal**
   - Goal: Learn strategic resource management
   - Reward: Stay alive (even by doing nothing useful)
   - Agent optimizes reward, not goal

2. **Credit Assignment Hell**
   - Agent dies at step 200 after oscillating
   - Which of those 200 left-right actions was "wrong"?
   - All of them got +1.0, so none seem bad!

3. **No Exploration Incentive**
   - Why explore when oscillating pays just as well?
   - Why risk using affordances when standing still works?
   - "Don't just do something, stand there!" becomes optimal

## The Solution: Milestone Rewards ("Happy Birthday!")

### New Design (FIXED)
```python
# Start with zero rewards (most steps = 0 reward)
rewards = torch.zeros(self.num_agents, device=self.device)

# Every 10 steps: +0.5 bonus
decade_milestone = (self.step_counts % 10 == 0) & alive_mask
rewards += torch.where(decade_milestone, 0.5, 0.0)

# Every 100 steps: +5.0 bonus ("Happy Birthday!")
century_milestone = (self.step_counts % 100 == 0) & alive_mask
rewards += torch.where(century_milestone, 5.0, 0.0)

# Death penalty: -100.0
rewards = torch.where(self.dones, -100.0, rewards)
```

**What the agent learns:**
- Move left → 0.0
- Move right → 0.0
- Move left → 0.0 (hmm, boring...)
- Move right → 0.0 (still boring...)
- Actually survive 10 steps → **+0.5** (oh! that's better!)
- Actually survive 100 steps → **+5.0** (wow, jackpot!)

**The agent discovers**: "Oscillating doesn't pay off. I need to actually SURVIVE long-term, which means using affordances strategically!"

## Why Milestone Rewards Work

### 1. **Sparse but Meaningful Signals**
- 9 steps of oscillation → 0 reward (not reinforced)
- 10 steps of strategic survival → +0.5 (reinforced!)
- Clear distinction between aimless and purposeful behavior

### 2. **Better Credit Assignment**
- Agent gets +0.5 at step 10
- "What did I do in the last 10 steps that worked?"
- Shorter time window = easier to identify successful patterns

### 3. **Hierarchical Goal Structure**
- Short-term: Survive to next 10-step milestone (+0.5)
- Medium-term: Survive to 100-step birthday (+5.0)
- Long-term: Survive as long as possible (accumulate milestones)

### 4. **Exploration Incentive**
- Standing still → dies around step 50 → only 5× +0.5 bonuses = +2.5
- Strategic play → survives to step 200 → 20× +0.5 + 2× +5.0 = +20.0
- Exploring pays off!

## The Goldilocks Problem: Why Not Per-Action Rewards?

We tried complex per-action rewards before:

### Attempt 1: Complex Meter-Based Rewards (FAILED)
```python
# Reward high meters, penalize low meters
for meter in meters:
    if meter > 0.8: reward += 0.4
    if meter < 0.3: reward -= 2.5
```

**Problem**: Longer survival → more time in low states → accumulating penalties → negative total reward!

An agent that survived 200 steps would get:
- 150 steps with some low meters: -2.5 × 150 = -375
- Total reward: **-375** (terrible!)

An agent that died immediately:
- 10 steps before death: -25
- Total reward: **-25** (less bad!)

**Perverse outcome**: Dying quickly was "less bad" than surviving long!

### Attempt 2: Constant Per-Step Survival Reward (FAILED)
```python
rewards = torch.ones(self.num_agents, device=self.device)  # +1.0 per step
```

**Problem**: Rewards aimless behavior equally to strategic play (left-right oscillation).

### Attempt 3: Milestone Rewards (SUCCESS!)
```python
# Every 10 steps: +0.5
# Every 100 steps: +5.0
```

**Success**: Sparse enough to avoid accumulation problems, frequent enough to provide learning signal.

## Pedagogical Value

### For Students Learning RL:
1. **Reward Shaping is Hard**
   - Small changes in reward structure → huge changes in learned behavior
   - What you measure is what you get (not what you want!)

2. **Sparse vs. Dense Rewards**
   - Dense (per-step): Fast learning but can reward wrong things
   - Sparse (milestones): Slower learning but clearer signal
   - Milestone: Best of both worlds!

3. **Credit Assignment Problem**
   - Longer episodes → harder to connect action → outcome
   - Milestones create intermediate checkpoints
   - "What did I do right in the last 10 steps?" is answerable

### For Students Learning AI Alignment:
1. **Specification Gaming**
   - Agent optimizes metric, not intent
   - "Survive" ≠ "survive strategically"
   - Must encode "strategic" into reward structure

2. **Proxy Metrics**
   - Milestones are a proxy for "playing well"
   - Not perfect (could get lucky), but better than alternatives
   - Real-world AI: proxies are all we have!

3. **Unintended Consequences**
   - Per-step rewards seemed obvious but rewarded oscillation
   - Complex rewards seemed precise but rewarded dying quickly
   - Simple changes have non-obvious effects

## Expected Behavioral Changes

### Before (Per-Step Rewards):
- **Episode 1-50**: Random exploration, some oscillation
- **Episode 50-200**: Lots of oscillation, occasional affordance use
- **Episode 200+**: Stuck in local optima (oscillate near affordable affordances)
- **Max survival**: ~200 steps (oscillate until meters run out)

### After (Milestone Rewards):
- **Episode 1-50**: Random exploration, no oscillation incentive
- **Episode 50-200**: Discovering that affordances extend survival
- **Episode 200-500**: Learning economic cycles (Job → resources → survival)
- **Episode 500+**: Strategic resource management
- **Max survival**: >300 steps (strategic play extends milestones)

## Metrics to Watch

Track these to validate milestone rewards work:

1. **Oscillation Frequency**
   - Before: High (agent stuck in left-right loops)
   - After: Low (oscillation doesn't pay off)

2. **Affordance Interaction Rate**
   - Before: Low (oscillating is easier)
   - After: High (must use affordances to reach milestones)

3. **Average Episode Length**
   - Before: ~50-200 steps (dies from meter depletion)
   - After: >200 steps (strategic play reaches more milestones)

4. **Reward Distribution**
   - Before: Many small positive rewards (+1, +1, +1...)
   - After: Sparse large rewards (0, 0, 0, +0.5, 0, 0, +5.0...)

## Classroom Discussion Prompts

1. **"Why not just reward every step?"**
   - What happens when all actions are equally rewarded?
   - How does this relate to real-world incentive design?

2. **"Why 10 and 100 specifically?"**
   - What if we used 5 and 50? Or 20 and 200?
   - How does milestone spacing affect learning speed?

3. **"Could the agent game this system?"**
   - What if it learns to oscillate for 9 steps, use affordance, oscillate again?
   - Is that "gaming" or "learning"? (Trick question: that IS the strategy!)

4. **"How is this like real life?"**
   - Birthdays, anniversaries, performance reviews = milestones
   - We don't reward "existing," we reward "achieving"
   - Milestone rewards are everywhere in human motivation!

## Implementation Details

### Milestone Timing
- Decade milestone (step % 10 == 0): +0.5
- Century milestone (step % 100 == 0): +5.0
- Note: Step 100 gets BOTH bonuses (+0.5 + +5.0 = +5.5)

### Reward Scaling
- Small milestone: +0.5 (encourages short-term survival)
- Large milestone: +5.0 (10× small, encourages long-term strategy)
- Death penalty: -100.0 (20× large milestone, strongly discourages death)

### Comparison to Alternatives
- Per-step +1.0: 100 steps = +100.0 reward
- Milestones: 100 steps = 10×(+0.5) + 1×(+5.0) = **+10.0** reward
- Milestone rewards are ~10× sparser (by design!)

## Related Documents

- [Episode 1: Hospital Bankruptcy](./episode_1_hospital_bankruptcy.md)
- [Reward Hacking: Interact Spam](./reward_hacking_interact_spam.md)
- [Three Stages of Learning](./three_stages_of_learning.md)

## Success Criteria

Milestone rewards are working when:
- Oscillation frequency < 10% of episodes
- Average episode length > 200 steps
- Agent uses Job proactively (maintains money buffer)
- Survival time steadily increases over training

---

**Moral of the Story**: In RL as in life, we celebrate milestones, not mere existence. Reward what you want the agent to optimize—and oscillating isn't a birthday worth celebrating. 🎂
