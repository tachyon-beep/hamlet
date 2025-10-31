# Types of Complexity: Why Hamlet is Harder Than It Looks

**Date**: October 31, 2025
**Insight**: Discrete non-stationary dynamics vs continuous stationary dynamics
**Status**: Core Pedagogical Concept ✅

## The Deceptive Simplicity of Hamlet

**First impression:**
- 8×8 grid (64 states)
- 5 discrete actions
- Simple movement + interaction
- "This is a toy problem"

**Reality:**
- 64 locations × 14 affordance types = **896 distinct interaction rules**
- Coupled cascading meter dynamics
- State-dependent non-stationary physics
- "This is deceptively complex"

---

## Comparing Complexity Types

### Flight Sim: Continuous but Stationary

**State space:**
```python
state = [
    x, y, z,                    # Position (3 floats)
    vx, vy, vz,                 # Velocity (3 floats)
    pitch, roll, yaw,           # Orientation (3 floats)
    pitch_rate, roll_rate, yaw_rate,  # Angular velocity (3 floats)
    throttle, flaps, gear,      # Control state (3 floats)
    altitude, airspeed, aoa     # Derived (3 floats)
]
# Total: ~18 continuous dimensions
```

**Dynamics:**
```python
# Physics is CONSTANT everywhere
F = ma
thrust = throttle * max_thrust
drag = 0.5 * rho * v^2 * Cd * A
lift = 0.5 * rho * v^2 * Cl * A

# These equations work the SAME at any (x, y, z)
# No "special zones" where physics changes
```

**Key properties:**
- ✅ Smooth, differentiable
- ✅ Stationary (rules constant everywhere)
- ✅ Continuous gradients
- ✅ High-dimensional but consistent

---

### Hamlet: Discrete but Non-Stationary

**State space:**
```python
state = [
    position: (x, y),           # Grid location (2 ints, 64 discrete values)
    meters: [8 floats],         # Energy, hygiene, satiation, money, mood, social, health, fitness
    affordance_type: int        # What you're standing on (15 categories)
]
# Observation: 64 (one-hot grid) + 8 (meters) + 15 (affordance type) = 87 dimensions
```

**Dynamics:**
```python
# Physics CHANGES based on grid position!
if position == (2, 3) and affordance == Hospital:
    health += 0.4
    money -= 15
elif position == (5, 7) and affordance == Bed:
    energy += 0.5
    health += 0.02
    money -= 5
elif position == (1, 1) and affordance == Job:
    money += 22.5
    energy -= 0.15
    health -= 0.03

# COMPLETELY DIFFERENT "physics" at each location!
# 64 grid cells × 14 affordance types = 896 distinct rules
```

**Key properties:**
- ✅ Discrete, non-smooth
- ❌ Non-stationary (rules change by location)
- ❌ Context-dependent outcomes
- ✅ Lower-dimensional but fragmented

---

## Why Non-Stationary is Harder

### 1. **Cannot Interpolate**

**Flight Sim (Easy):**
```python
# If throttle=0.5 → velocity increases by 10 m/s
# Then throttle=0.6 → velocity increases by ~12 m/s
# Agent can interpolate: "More throttle = more speed"
```

**Hamlet (Hard):**
```python
# INTERACT at Hospital: health +0.4, money -15
# INTERACT at Bed: energy +0.5, money -5
# NO RELATIONSHIP! Cannot interpolate!
# Agent must MEMORIZE each location's rules
```

### 2. **Exploration is Expensive**

**Flight Sim:**
- Try throttle=0.5 → observe outcome
- Try throttle=0.6 → smoothly different
- Gradient descent works well

**Hamlet:**
- Try INTERACT at (2,3) → Hospital effect
- Try INTERACT at (2,4) → Maybe empty, maybe Job, COMPLETELY different!
- Must visit ALL 64 × 14 = 896 state-action pairs to learn

### 3. **Credit Assignment Across Contexts**

**Flight Sim:**
```python
# "Increasing throttle led to altitude gain"
# This is TRUE everywhere in the state space
# Learn once, apply everywhere
```

**Hamlet:**
```python
# "Using INTERACT at this location led to reward"
# But was it the Hospital? The Bed? The Job?
# And does this generalize to OTHER locations? NO!
# Must learn separately for each affordance type
```

---

## The Coupled Cascade Multiplier

Hamlet has **additional** complexity: cascading meter interactions.

### Flight Sim: Mostly Independent Variables

```python
# Altitude doesn't directly affect velocity
# Velocity doesn't directly affect orientation
# (Some coupling, but mostly independent)
altitude_change = velocity * dt
velocity_change = (thrust - drag) / mass * dt
```

### Hamlet: Everything Affects Everything

```python
# Satiation affects BOTH Energy AND Health
energy_change += satiation_factor * 0.15
health_change += satiation_factor * 0.10

# Fitness affects Health
health_change += fitness_factor * 0.08

# Mood affects Energy
energy_change += mood_factor * 0.12

# Hygiene affects ALL secondary meters (mood, social, fitness)
# Social affects Mood
# It's a giant web of dependencies!
```

**Result:**
- Flight sim: Learn 18 mostly-independent dimensions
- Hamlet: Learn 8 coupled dimensions + 64 location-specific rules + 14 affordance effects
- Total unique state-action-outcome mappings: **MASSIVE**

---

## The Economic Constraint Layer

**Flight sim:**
- No resource constraints (fuel is assumed infinite in training)
- All actions always available
- Physics is the only constraint

**Hamlet:**
- Money is a HARD CONSTRAINT
- Cannot use Hospital if money < $15
- Cannot use Bed if money < $5
- Must balance resource acquisition (Job) with resource expenditure (affordances)
- **Economic strategy layer** on top of spatial strategy

This adds a **planning horizon** requirement:
- "Should I go to Job now or use Bed first?"
- "Can I afford Hospital or should I just die and reset?"
- Multi-step planning with resource constraints

---

## Why Students Think Flight Sim is Harder

### Visual Complexity Bias

**Flight Sim:**
- 3D visualization, realistic graphics
- Continuous smooth motion
- "Looks like real robotics"
- Students think: "This must be hard!"

**Hamlet:**
- 2D grid, simple sprites
- Discrete jumpy movement
- "Looks like Pac-Man"
- Students think: "This is simple!"

### Dimensionality ≠ Complexity

**High dimensions, low complexity:**
- Flight sim: 18 dimensions, but smooth and stationary
- Can use gradient descent effectively
- Function approximation works well

**Low dimensions, high complexity:**
- Hamlet: 87 dimensions, but discrete and non-stationary
- Must memorize context-specific rules
- Generalization is difficult

---

## The Pedagogical Twist

### Act 1: Show Flight Sim
- Complex 3D physics
- 18 continuous inputs
- Students: "Wow, that's advanced!"

### Act 2: Show Hamlet
- Simple 8×8 grid
- 5 discrete actions
- Students: "That's much easier!"

### Act 3: The Reveal
**Flight Sim:**
- Stationary dynamics (physics constant everywhere)
- Smooth gradients (continuous optimization)
- Action-outcome relationships are consistent

**Hamlet:**
- Non-stationary dynamics (rules change by location)
- Discrete jumps (no interpolation)
- 896 distinct context-dependent rules to learn

**Students:** "Wait... Hamlet might actually be HARDER?!"

### Act 4: The Lesson
> "Complexity isn't about dimensions or visual fidelity.
> It's about **context-dependence** and **non-stationarity**.
> A simple-looking grid with location-dependent rules
> can be harder to learn than a complex flight simulator
> with consistent physics."

---

## Metrics That Expose This

### Learning Curves

**Flight Sim (Expected):**
```
Episode 0-100: Random exploration, gradual improvement
Episode 100-500: Smooth learning curve (gradient descent working)
Episode 500+: Asymptotic convergence
```

**Hamlet (Actual):**
```
Episode 0-200: Random exploration, HIGH VARIANCE
Episode 200-500: Sudden jumps (discovering affordance rules)
Episode 500-1000: Plateaus and breakthroughs (learning context-specific strategies)
Episode 1000+: Still improving (memorizing all 896 rules)
```

### Generalization Failure

**Flight Sim:**
- Train on altitude 1000-5000m
- Agent generalizes to 6000m reasonably well
- Physics is the same

**Hamlet:**
- Train on Bed + Hospital + Job
- Add new affordance: Gym
- Agent has NO IDEA what to do
- Must relearn from scratch

---

## Design Implications

### For Teaching:
1. **Don't judge complexity by appearance**
   - Grid ≠ simple
   - Continuous ≠ complex

2. **Context-dependence is a complexity multiplier**
   - Stationary dynamics: learn once, apply everywhere
   - Non-stationary dynamics: must memorize every context

3. **Discrete can be harder than continuous**
   - Continuous: gradient descent, interpolation
   - Discrete: exhaustive exploration, memorization

### For Research:
1. **Sample efficiency matters more in non-stationary environments**
   - Flight sim: 1000 samples → reasonable policy
   - Hamlet: 10,000 samples → still discovering rules

2. **Generalization techniques are critical**
   - Flight sim: generalize across continuous space naturally
   - Hamlet: must explicitly learn to generalize across affordances

3. **Curriculum learning helps more in non-stationary settings**
   - Introduce affordances gradually
   - Allow agent to master each context before adding more

---

## The Beautiful Irony

**What looks simple:**
- 8×8 grid
- 5 actions
- "Toy problem"

**What is actually happening:**
- 64 spatial contexts
- 14 affordance types
- 8 coupled cascading meters
- Economic constraints
- = Combinatorial explosion of state-action-outcome mappings

**What looks complex:**
- 18-D continuous space
- Realistic physics
- "Advanced robotics"

**What is actually happening:**
- Smooth differentiable dynamics
- Stationary physics (consistent rules)
- Gradient descent just works
- = High-dimensional but tractable

---

## Classroom Exercise

**Challenge students:**
1. "Estimate sample complexity for flight sim" (they'll guess high)
2. "Estimate sample complexity for Hamlet" (they'll guess low)
3. Show actual training curves
4. Reveal: Hamlet needs MORE samples despite lower dimensionality
5. Discuss: Why?

**Answer:**
- Flight sim: Interpolate across continuous space
- Hamlet: Must visit each discrete context
- Context-dependence > dimensionality for sample complexity

---

## Connection to Real-World RL

### Stationary Environments (Easier):
- Robotic manipulation (physics is consistent)
- Game playing (rules don't change)
- Continuous control (smooth gradients)

### Non-Stationary Environments (Harder):
- Multi-task RL (different tasks = different "physics")
- Contextual bandits (reward depends on context)
- Real-world deployment (environment changes over time)

**Hamlet prepares students for the harder case.**

---

## Closing Wisdom

> "Don't confuse visual complexity with learning complexity.
> A flight simulator looks impressive,
> but its stationary dynamics make it learnable.
>
> Hamlet looks simple,
> but its non-stationary context-dependent rules
> make it deceptively difficult.
>
> This is why Hamlet is pedagogically valuable:
> It teaches the HARD lessons
> in a visually accessible way."

---

## Related Documents

- [Flight Sim Reward Hacking Story](./flight_sim_reward_hacking_story.md)
- [Milestone Rewards Design](./milestone_rewards_design.md)
- [Trick Students Pedagogy](./trick_students_pedagogy.md)

---

**Moral of the Story**: Complexity isn't measured in dimensions or visual fidelity. It's measured in how many distinct rules the agent must memorize. Hamlet's 896 context-dependent rules make it harder than it looks. 🎯
