# The Three Stages of Agent Learning

**Date**: 2025-10-28
**Observation**: Training progression shows distinct learning phases with different behaviors

## Overview

By saving checkpoints at different training stages, we can show students the **entire learning arc** from random flailing to competent strategy.

## The Three Checkpoints

### Stage 1: The Broken Robot (Episode ~50, ε=0.778)
**File**: `models/early_checkpoint_ep50.pt` (can save separately)

**Behavior**:
- Oscillates between two adjacent squares
- 78% random exploration
- Never reaches affordances
- Dies around step 90-100
- Reward: -100 to -110

**What students see**: "This is stupid! Why is it bouncing back and forth?"

**What they're learning** (without knowing):
- Pure exploration phase
- Random policy before learning
- Credit assignment problem (hasn't connected actions to outcomes)
- Why epsilon-greedy starts high

**Teaching moment**:
> "This agent has never been rewarded. It's a newborn.
> Every action is a guess.
> Humans are like this too - watch a baby try to walk."

---

### Stage 2: The Learning Agent (Episode ~455, ε=0.107)
**File**: Current checkpoint from interrupted training

**Behavior**:
- Moves with more purpose
- Occasionally reaches affordances
- Shows partial strategy
- Survival: 180-250 steps
- Reward: -40 to -90 (improving but inconsistent)

**What students see**: "It's getting better! But still makes mistakes."

**What they're learning**:
- Q-values converging
- Policy emerging from experience
- Still exploring (10% random)
- Learning is gradual, not instant

**Teaching moment**:
> "This agent has survived 455 lives.
> It's learned some patterns - Job gives money, Bed restores energy.
> But it hasn't optimized yet. It's still figuring out the best strategy."

---

### Stage 3: The Competent Agent (Episode 1000, ε=0.050)
**File**: `models/trained_agent.pt` (final trained model)

**Behavior**:
- Purposeful navigation to affordances
- **OR** stands still and spams interact (reward hacking!)
- Survives 300-450+ steps
- Reward: +50 to +100 (positive!)
- 95% exploitation, 5% exploration

**What students see**: "It's smart now! ...Wait, why is it standing still?"

**What they're learning**:
- Converged policy
- Emergent strategy (not programmed)
- Optimization ≠ human intuition
- Reward hacking (bonus lesson!)

**Teaching moment**:
> "This agent has lived 1000 lives.
> It discovered a strategy we didn't program.
> The question is: Is it smart or broken?
> Welcome to AI alignment."

---

## Side-by-Side Comparison

| Metric | Stage 1 (Ep 50) | Stage 2 (Ep 455) | Stage 3 (Ep 1000) |
|--------|-----------------|------------------|-------------------|
| **Epsilon** | 0.778 (78% random) | 0.107 (11% random) | 0.050 (5% random) |
| **Survival** | ~100 steps | ~200 steps | ~400 steps |
| **Reward** | -107 | -60 to -90 | +79 |
| **Strategy** | None (random) | Partial (inconsistent) | Optimized (consistent) |
| **Behavior** | Oscillating | Seeking affordances | Reward hacking |
| **Learning** | Pure exploration | Exploitation emerging | Converged policy |

---

## The Perfect Demo Structure

### Live Demo: "Watch AI Learn"

**Setup (2 min)**:
- Load Stage 1 checkpoint
- "This agent knows nothing. Watch."

**Act 1: Confusion (3 min)**:
- Watch it oscillate and die
- Ask: "Why is it so bad?"
- Reveal: ε=0.778, pure exploration

**Act 2: Discovery (3 min)**:
- Load Stage 2 checkpoint
- "1000 deaths later..."
- Watch improved but imperfect play
- Ask: "What changed?"
- Reveal: Q-values learning, ε=0.107

**Act 3: Mastery (5 min)**:
- Load Stage 3 checkpoint
- Watch competent (or exploit) play
- Ask: "Is this intelligent?"
- Reveal: Optimization ≠ understanding

**Wrap-up (5 min)**:
- Show epsilon decay graph
- Show reward progression graph
- Discuss: What is intelligence?

---

## Key Pedagogical Points

### Stage 1 Teaches:
- ✅ **Exploration** - Why random action is necessary
- ✅ **Credit assignment** - How to learn from delayed rewards
- ✅ **Patience** - Learning takes time
- ✅ **Empathy** - AI isn't "stupid", it's untrained

### Stage 2 Teaches:
- ✅ **Convergence** - Gradual improvement over episodes
- ✅ **Experience replay** - Learning from past experiences
- ✅ **Target networks** - Stability in learning
- ✅ **Hyperparameters** - Epsilon decay, learning rate matter

### Stage 3 Teaches:
- ✅ **Emergent behavior** - Complex strategies from simple rules
- ✅ **Reward hacking** - Agents find loopholes
- ✅ **Alignment problem** - Intent vs. outcome
- ✅ **Optimization** - Not the same as understanding

---

## Student Activities

### Activity 1: "Spot the Difference"
Show Stage 1 and Stage 3 side-by-side (2 windows).
**Task**: List 5 behavioral differences.
**Goal**: Observe learning effects directly.

### Activity 2: "Predict the Death"
Show Stage 2 agent.
**Task**: Watch meters, predict which will kill it.
**Goal**: Understand state dynamics.

### Activity 3: "Design the Lesson"
Give students all 3 checkpoints.
**Task**: Create a 10-min lesson plan using these stages.
**Goal**: Think pedagogically about RL.

### Activity 4: "Fix the Exploit"
Show Stage 3 interact-spam behavior.
**Task**: Propose reward function changes to prevent it.
**Goal**: Learn reward engineering.

---

## Data to Collect

For each stage, record:
1. **Episode number** (50, 455, 1000)
2. **Epsilon value** (0.778, 0.107, 0.050)
3. **Average reward** (last 5 episodes)
4. **Average survival time** (last 5 episodes)
5. **Typical behavior** (description)
6. **Strategy observed** (if any)

**To save checkpoints during training**:
```python
if episode in [50, 200, 455, 700, 1000]:
    agent.save(f"models/checkpoint_ep{episode}.pt")
```

---

## The Arc Students Experience

```
"This is broken!"
    ↓
"It's learning something..."
    ↓
"It's actually good now!"
    ↓
"Wait, it found a loophole..."
    ↓
"...that's brilliant but wrong"
    ↓
"OH. This is the alignment problem."
```

By the end, they've experienced:
- Exploration vs exploitation
- Credit assignment
- Reward shaping
- Emergent behavior
- Reward hacking
- AI alignment

**All from watching a game character learn to survive.**

---

## Why This Works (Meta-Analysis)

**Traditional RL course**:
- "Here's the Bellman equation"
- Students: *glazed eyes*

**Hamlet approach**:
- "Watch this robot learn to not die"
- Students: *engaged*
- Show 3 stages of improvement
- Students: "I understand improvement!"
- Reveal math behind it
- Students: "Oh, that's just formalizing what I saw"

**The trick**: They learn the intuition before the formalism.

---

## Extension: The Fourth Stage

**Stage 4: Multi-Agent Competition (Future)**

When you implement Level 4 from the architecture doc:
- Show two Stage 3 agents competing
- Watch emergent game theory (2am strategy!)
- Compare solo vs. competitive behavior
- Teach Nash equilibria through observation

This becomes Stage 4 in the progression demo.

---

**Bottom line**: Having multiple checkpoints lets you teach RL as a **story** instead of equations.
