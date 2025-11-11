## The Key Insight

**VFS already does the heavy lifting:**

```yaml
# VFS can compute ANY derived feature:
variables:
  - id: energy_urgency
    expression: sigmoid(multiply(subtract(1.0, bar["energy"]), 5))

  - id: worst_physical_need
    expression: min(bar["energy"], bar["health"], bar["satiation"])

  - id: bathroom_emergency
    expression: and(threshold(bar["hygiene"], 0.1, 0.15),
                    gt(distance_to_affordance("Toilet"), 3))
```

**DAC just ORCHESTRATES them:**

```yaml
# DAC composes reward from VFS variables:
drive_as_code:
  extrinsic:
    base: variable["worst_physical_need"]  # Reference VFS!
    bonuses:
      - variable["energy_urgency"]
      - variable["bathroom_emergency"]
```

**This is BRILLIANT separation of concerns:**

- VFS = Feature engineering (observations + derived variables)
- DAC = Reward engineering (combining features into drive signals)

## Drive As Code: Complete Library

```yaml
# ============================================================================
# DRIVE AS CODE (DAC) - Reward Function Compiler
# ============================================================================
#
# Philosophy: Reward functions are compositions of:
#   1. Extrinsic structure (base survival/performance rewards)
#   2. Modifiers (crisis/boredom sensitivity)
#   3. Shaping bonuses (behavioral incentives)
#   4. Intrinsic computation (exploration drives)
#
# All references can use:
#   - bar["name"]           # Direct bar values
#   - variable["name"]      # VFS-computed variables
#   - obs["name"]           # Any observation
#   - context["name"]       # Episode-level context (time, step_count, etc)
#
# ============================================================================

version: "1.0"

# ===== MODIFIERS (Reusable Range-Based Multipliers) =====
modifiers:

  # Crisis: Suppress exploration when resources are low
  energy_crisis:
    bar: energy
    ranges:
      - {name: critical, min: 0.0, max: 0.2, multiplier: 0.0}   # Full suppression
      - {name: low,      min: 0.2, max: 0.4, multiplier: 0.3}   # Partial suppression
      - {name: normal,   min: 0.4, max: 1.0, multiplier: 1.0}   # No modification

  # Boredom: Boost exploration when resources are high
  energy_boredom:
    bar: energy
    ranges:
      - {name: normal, min: 0.0, max: 0.7, multiplier: 1.0}   # No modification
      - {name: safe,   min: 0.7, max: 0.9, multiplier: 3.0}   # Moderate boost
      - {name: bored,  min: 0.9, max: 1.0, multiplier: 10.0}  # Strong boost

  # Inverted semantics (high = bad, e.g., stress, pain)
  stress_crisis:
    bar: stress
    ranges:
      - {name: calm,    min: 0.0, max: 0.3, multiplier: 3.0}   # Explore when calm
      - {name: normal,  min: 0.3, max: 0.7, multiplier: 1.0}
      - {name: crisis,  min: 0.7, max: 1.0, multiplier: 0.1}   # Focus when stressed

  # Multi-bar: Use VFS to compute aggregate state
  overall_wellbeing_crisis:
    variable: worst_physical_need  # VFS computes min(energy, health, satiation)
    ranges:
      - {name: crisis, min: 0.0, max: 0.3, multiplier: 0.0}
      - {name: normal, min: 0.3, max: 1.0, multiplier: 1.0}

  # Temporal: Decay exploration over episode
  temporal_decay:
    variable: age  # VFS computes step_count / max_steps
    ranges:
      - {name: early, min: 0.0, max: 0.3, multiplier: 2.0}   # Explore more early
      - {name: mid,   min: 0.3, max: 0.7, multiplier: 1.0}
      - {name: late,  min: 0.7, max: 1.0, multiplier: 0.3}   # Exploit late

  # Contextual: Modify based on environmental state
  weather_modifier:
    variable: raining  # VFS computes from stochastic weather
    ranges:
      - {name: clear, min: 0.0, max: 0.5, multiplier: 1.0}
      - {name: rain,  min: 0.5, max: 1.0, multiplier: 0.5}   # Penalize movement in rain

# ===== EXTRINSIC REWARD STRATEGIES =====
extrinsic:

  # Strategy 1: Constant Base + Shaped Bonuses (RECOMMENDED)
  # - Always get base reward for surviving
  # - Bonuses/penalties for bar deviations from ideal
  type: constant_base_with_shaped_bonus

  base_reward: 1.0

  bar_bonuses:
    - bar: energy
      center: 0.5        # Neutral point (no bonus/penalty)
      scale: 0.5         # Magnitude of bonus/penalty
      # Result: +0.25 when energy=1.0, -0.25 when energy=0.0

    - bar: health
      center: 0.5
      scale: 0.5

  # Can also reference VFS variables:
  variable_bonuses:
    - variable: energy_urgency
      weight: 0.5        # How much to add/subtract

    - variable: bathroom_emergency
      weight: -2.0       # Penalize emergency states

  # Apply modifiers to extrinsic (optional)
  apply_modifiers: []  # Empty = no modification

  # Alternative strategies (uncomment to use):

  # Strategy 2: Multiplicative (CURRENT - Known bug)
  # type: multiplicative
  # base: 1.0
  # bars: [energy, health]  # reward = base * energy * health
  # Problem: reward → 0 when any bar is low, creating exploration distraction

  # Strategy 3: Additive Unweighted
  # type: additive_unweighted
  # base: 0.0
  # bars: [energy, health, satiation]  # reward = base + sum(bars)

  # Strategy 4: Weighted Sum
  # type: weighted_sum
  # terms:
  #   - {source: bar, name: energy, weight: 0.4}
  #   - {source: bar, name: health, weight: 0.3}
  #   - {source: variable, name: social_priority, weight: 0.3}

  # Strategy 5: Polynomial (Flexible Non-Linear)
  # type: polynomial
  # terms:
  #   - {source: bar, name: energy, exponent: 1.0, weight: 0.5}
  #   - {source: bar, name: health, exponent: 2.0, weight: 0.3}  # Quadratic
  #   - {constant: 1.0}  # Base reward

  # Strategy 6: Threshold-Based (Binary Rewards)
  # type: threshold_based
  # thresholds:
  #   - {source: bar, name: energy, above: 0.5, reward: 1.0}
  #   - {source: bar, name: energy, below: 0.2, reward: -5.0}  # Crisis penalty
  #   - {source: variable, name: bathroom_emergency, above: 0.5, reward: -10.0}

  # Strategy 7: Min/Max/Mean Aggregation
  # type: aggregation
  # operation: min  # Options: min, max, mean, product
  # sources:
  #   - {source: bar, name: energy}
  #   - {source: bar, name: health}
  #   - {source: bar, name: satiation}
  # base: 1.0

  # Strategy 8: VFS Variable (Delegate to VFS entirely)
  # type: vfs_variable
  # variable: custom_reward_function
  # # Then in VFS, define:
  # # - id: custom_reward_function
  # #   expression: multiply(bar["energy"], variable["urgency"])

  # Strategy 9: Hybrid (Combine multiple)
  # type: hybrid
  # components:
  #   - {type: constant_base_with_shaped_bonus, weight: 0.6, ...}
  #   - {type: threshold_based, weight: 0.4, ...}

# ===== INTRINSIC REWARD =====
intrinsic:

  # Base strategy for computing intrinsic curiosity
  strategy: adaptive_rnd  # Options: rnd, icm, count_based, none

  base_weight: 0.100  # How much intrinsic contributes to total reward

  # Modifiers (apply range-based adjustments)
  apply_modifiers:
    - energy_crisis           # Suppress exploration when low energy
    - overall_wellbeing_crisis  # Suppress when any critical bar is low
    - temporal_decay          # Reduce exploration over episode

  # Intrinsic strategy configurations:

  # RND (Random Network Distillation)
  # strategy: rnd
  # rnd_config:
  #   target_network_size: [256, 256]
  #   predictor_network_size: [256, 256]
  #   learning_rate: 0.0001
  #   normalize: true  # Normalize intrinsic rewards

  # ICM (Intrinsic Curiosity Module)
  # strategy: icm
  # icm_config:
  #   forward_model_size: [256, 256]
  #   inverse_model_size: [256, 256]
  #   forward_loss_weight: 0.2
  #   inverse_loss_weight: 0.8

  # Count-Based (Pseudo-Counts)
  # strategy: count_based
  # count_config:
  #   hash_size: 1000000
  #   bonus_type: inverse_sqrt  # Options: inverse, inverse_sqrt, constant

  # Adaptive RND (scales with performance)
  # strategy: adaptive_rnd
  # adaptive_config:
  #   performance_metric: mean_episode_length  # Track this
  #   target_performance: 800  # Goal to reach
  #   decay_rate: 0.95  # Reduce intrinsic as performance improves
  #   min_weight: 0.01  # Don't go below this

  # None (no intrinsic reward)
  # strategy: none

# ===== SHAPING BONUSES =====
shaping:

  # Bonus 1: Approach Reward (encourage moving toward goals when needed)
  - type: approach_reward
    target_affordance: Bed
    trigger:
      source: bar
      name: energy
      below: 0.3
    bonus: 0.5  # Extra reward per step moving toward bed
    decay_with_distance: true  # Bonus decreases with distance

  - type: approach_reward
    target_affordance: Hospital
    trigger:
      source: bar
      name: health
      below: 0.3
    bonus: 1.0

  # Bonus 2: Completion Bonus (reward for finishing affordances)
  - type: completion_bonus
    affordances: all  # Or specify: [Bed, Hospital, Job]
    bonus: 1.0
    scale_with_duration: true  # Longer affordances get bigger bonuses

  # Bonus 3: Efficiency Bonus (reward for completing without interruption)
  - type: efficiency_bonus
    affordances: [Job, Gym]
    bonus: 0.5
    penalty_for_interrupt: -1.0

  # Bonus 4: State Achievement (reward for reaching target states)
  - type: state_achievement
    trigger:
      source: variable
      name: all_bars_above_threshold
      # VFS computes: and(gt(energy, 0.7), gt(health, 0.7), gt(satiation, 0.7))
      above: 0.5
    bonus: 5.0
    once_per_episode: true  # Only fire once

  # Bonus 5: Streak Bonus (reward for sustained good behavior)
  - type: streak_bonus
    condition:
      source: variable
      name: energy_maintained  # VFS computes: gt(energy, 0.6)
      above: 0.5
    bonus_per_tick: 0.1
    max_streak: 50  # Cap at 50 ticks

  # Bonus 6: Diversity Bonus (encourage using different affordances)
  - type: diversity_bonus
    affordances: [Bed, Hospital, HomeMeal, Job, Gym, Recreation]
    bonus: 2.0
    window: 100  # Within last 100 steps
    min_unique: 4  # Must use at least 4 different affordances

  # Bonus 7: Timing Bonus (reward for actions at optimal times)
  - type: timing_bonus
    affordance: Bed
    optimal_condition:
      source: variable
      name: optimal_sleep_time
      # VFS computes: and(or(lt(time_of_day, 6), gt(time_of_day, 22)), lt(bar["energy"], 0.5))
      above: 0.5
    bonus: 1.0

  # Bonus 8: Economic Efficiency (reward for maintaining money buffer)
  - type: economic_efficiency
    target_buffer:
      source: bar
      name: money
      min: 0.2  # Keep at least 20% money
      max: 0.8  # But don't hoard above 80%
    bonus: 0.5

  # Bonus 9: Balance Bonus (reward for keeping bars balanced)
  - type: balance_bonus
    bars: [energy, health, satiation]
    ideal_balance: 0.1  # Variance below 0.1 is considered balanced
    bonus: 1.0

  # Bonus 10: Crisis Avoidance (reward for NOT entering crisis)
  - type: crisis_avoidance
    bars: [energy, health]
    crisis_threshold: 0.2
    bonus_per_tick: 0.1

  # Bonus 11: VFS Variable (delegate to VFS entirely)
  - type: vfs_variable
    variable: custom_shaping_signal
    weight: 1.0
    # Then in VFS, define arbitrarily complex shaping logic

# ===== REWARD COMPOSITION =====
composition:
  # How to combine all components into total reward

  # Final reward formula:
  # total_reward = extrinsic + (intrinsic * effective_intrinsic_weight) + shaping

  # Where:
  #   extrinsic = computed via extrinsic strategy (with modifiers if specified)
  #   intrinsic = computed via intrinsic strategy
  #   effective_intrinsic_weight = base_weight * modifier1 * modifier2 * ...
  #   shaping = sum of all shaping bonuses

  # Normalization (optional)
  normalize: false  # If true, clip total reward to [-1, 1]

  # Clipping (optional)
  clip:
    min: -10.0  # Prevent catastrophic negative rewards
    max: 100.0  # Prevent runaway positive rewards

  # Logging (for debugging)
  log_components: true  # Log extrinsic, intrinsic, shaping separately
  log_modifiers: true   # Log modifier values each step

# ===== ADVANCED: HIERARCHICAL REWARDS =====
# For multi-timescale learning (meta-controller + controller)
hierarchical:
  enabled: false

  # Meta-controller reward (strategic goals, evaluated every N steps)
  meta:
    frequency: 50  # Evaluate every 50 steps
    extrinsic:
      type: state_achievement
      goal_states:
        - {variable: all_needs_satisfied, above: 0.5, reward: 10.0}
        - {variable: economic_stability, above: 0.5, reward: 5.0}
    intrinsic:
      strategy: goal_exploration  # Explore different goal states
      base_weight: 0.05

  # Controller reward (tactical actions, evaluated every step)
  controller:
    # Use main reward_strategy above
    inherit_from_main: true

# ===== CURRICULUM: REWARD ANNEALING =====
# Automatically adjust rewards over training
curriculum:
  enabled: false

  # Intrinsic decay (reduce exploration over training)
  intrinsic_decay:
    initial_weight: 0.100
    final_weight: 0.010
    decay_episodes: 1000
    decay_curve: exponential  # Options: linear, exponential, step

  # Shaping bonus decay (fade out shaping as agent learns)
  shaping_decay:
    bonuses: [approach_reward, completion_bonus]
    initial_weight: 1.0
    final_weight: 0.1
    decay_episodes: 500

  # Extrinsic scaling (increase task difficulty over time)
  extrinsic_scaling:
    bars: [energy, health]
    depletion_rate_multiplier:
      initial: 1.0
      final: 2.0
      scaling_episodes: 800

# ===== META: REWARD FUNCTION IDENTITY =====
provenance:
  # Compute hash of entire DAC config for reproducibility
  compute_hash: true

  # Log reward function identity to tensorboard
  log_to_tensorboard: true

  # Include in checkpoint
  include_in_checkpoint: true
```

## Example Configurations for Common Use Cases

### **Use Case 1: Fix the Crisis Exploration Bug**

```yaml
# configs/L0_5_fixed/drive_as_code.yaml

modifiers:
  energy_health_crisis:
    variable: worst_physical_need  # VFS: min(energy, health)
    ranges:
      - {min: 0.0, max: 0.3, multiplier: 0.0}
      - {min: 0.3, max: 1.0, multiplier: 1.0}

extrinsic:
  type: constant_base_with_shaped_bonus
  base_reward: 1.0
  bar_bonuses:
    - {bar: energy, center: 0.5, scale: 0.5}
    - {bar: health, center: 0.5, scale: 0.5}

intrinsic:
  strategy: rnd
  base_weight: 0.100
  apply_modifiers: [energy_health_crisis]  # Crisis suppression!

shaping: []

composition:
  log_components: true
```

**Result:** When energy or health < 0.3, intrinsic weight → 0.0, preventing exploration distraction.

### **Use Case 2: Advanced Urgency-Based Shaping**

```yaml
# VFS defines urgency features:
variables:
  - id: energy_urgency
    expression: sigmoid(multiply(subtract(1.0, bar["energy"]), 5))

  - id: bathroom_emergency
    expression: and(threshold(bar["hygiene"], 0.1, 0.15),
                    gt(distance_to_affordance("Toilet"), 3))

# DAC uses them:
extrinsic:
  type: constant_base_with_shaped_bonus
  base_reward: 1.0
  variable_bonuses:
    - {variable: energy_urgency, weight: 0.5}
    - {variable: bathroom_emergency, weight: -2.0}

shaping:
  - type: approach_reward
    target_affordance: Bed
    trigger: {source: variable, name: energy_urgency, above: 0.7}
    bonus: 1.0

  - type: approach_reward
    target_affordance: Toilet
    trigger: {source: variable, name: bathroom_emergency, above: 0.5}
    bonus: 5.0  # VERY important!
```

### **Use Case 3: Stress-Based Inverted Semantics**

```yaml
# For a bar where HIGH = BAD (stress, pain, etc.)
modifiers:
  stress_crisis:
    bar: stress
    ranges:
      - {name: calm, min: 0.0, max: 0.3, multiplier: 3.0}  # Explore when calm
      - {name: normal, min: 0.3, max: 0.7, multiplier: 1.0}
      - {name: crisis, min: 0.7, max: 1.0, multiplier: 0.1}  # Focus when stressed

extrinsic:
  type: threshold_based
  thresholds:
    - {source: bar, name: stress, below: 0.3, reward: 1.0}  # Good!
    - {source: bar, name: stress, above: 0.7, reward: -2.0}  # Bad!

intrinsic:
  strategy: rnd
  base_weight: 0.100
  apply_modifiers: [stress_crisis]
```

### **Use Case 4: Exploration → Exploitation Curriculum**

```yaml
modifiers:
  temporal_exploration_decay:
    variable: age  # VFS: step_count / max_steps
    ranges:
      - {name: early, min: 0.0, max: 0.2, multiplier: 5.0}   # High exploration early
      - {name: mid, min: 0.2, max: 0.6, multiplier: 1.0}
      - {name: late, min: 0.6, max: 1.0, multiplier: 0.1}    # Low exploration late

intrinsic:
  strategy: rnd
  base_weight: 0.100
  apply_modifiers: [temporal_exploration_decay]
```

**Result:** Episode starts with 50% intrinsic weight, ends with 1% intrinsic weight.

### **Use Case 5: Multi-Objective Balancing**

```yaml
# VFS defines composite states:
variables:
  - id: physical_wellbeing
    expression: mean(bar["energy"], bar["health"], bar["satiation"])

  - id: mental_wellbeing
    expression: mean(bar["mood"], bar["social"])

  - id: wellbeing_balance
    expression: abs(subtract(variable["physical_wellbeing"], variable["mental_wellbeing"]))

extrinsic:
  type: weighted_sum
  terms:
    - {source: variable, name: physical_wellbeing, weight: 0.4}
    - {source: variable, name: mental_wellbeing, weight: 0.4}

shaping:
  - type: vfs_variable
    variable: wellbeing_balance
    weight: -1.0  # Penalize imbalance
```

## The Power of This Design

**Composability:**

```yaml
# Researcher can mix and match:
extrinsic: constant_base_with_shaped_bonus
intrinsic: adaptive_rnd with energy_crisis modifier
shaping: [approach_reward, completion_bonus, diversity_bonus]
```

**No Code Changes:**

```bash
# A/B test reward functions:
configs/L0_5_multiplicative/     # Old broken way
configs/L0_5_shaped_bonus/       # New fixed way
configs/L0_5_threshold/          # Alternative
configs/L0_5_hybrid/             # Combination

# Just change YAML, no code!
```

**Leverage VFS:**

```yaml
# Complex logic in VFS:
variables:
  - id: perfect_storm
    expression: and(
      lt(bar["energy"], 0.2),
      lt(bar["health"], 0.3),
      gt(distance_to_affordance("Hospital"), 5),
      variable["raining"]
    )

# Simple reference in DAC:
shaping:
  - type: vfs_variable
    variable: perfect_storm
    weight: -10.0  # Huge penalty for this bad situation
```

**Provenance:**

```python
# Every experiment tracks:
experiment_hash = {
    'config_hash': '7f3a9b2e...',      # World (UAC)
    'cognitive_hash': 'a2f1c8d3...',   # Mind (BAC)
    'drive_hash': 'e9d2b5f8...',       # Rewards (DAC) ← NEW!
    'seed': 42
}
```

## Bottom Line

**Drive As Code gives you:**

1. ✅ Config-driven reward functions (no code changes)
2. ✅ Composable primitives (extrinsic + modifiers + shaping + intrinsic)
3. ✅ Leverages VFS (reference ANY computed variable)
4. ✅ Range-based modifiers (handles arbitrary bar semantics)
5. ✅ Provenance (reward function identity tracked)
6. ✅ Curriculum support (automatic reward annealing)
