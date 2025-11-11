# Variable & Feature System (VFS) Configuration
# Config: L_N1_debug (5×5 Grid2D, debug configuration)
#
# Purpose: Define custom variables that represent environmental phenomena or derived features.
#
# TODO: All of this needs to be implemented.
#
# DESIGN PRINCIPLE: Variables must have grounding
# 1. Environmental phenomena: Describe the world state (weather, lighting, noise)
# 2. Derived features: Computed from observable state (ratios, deficits, progress)
#
# Standard system variables are auto-generated from substrate, bars, and affordances:
#   - Spatial: grid_encoding, position
#   - Meters: energy, health, satiation, money, mood, social, fitness, hygiene
#   - Affordances: affordance_at_position
#   - Temporal: time_sin, time_cos, interaction_progress, lifetime_progress
#
# All variables (standard + custom) are automatically exposed as observations.
# No need to specify exposed_observations - the compiler auto-exposes everything.
#
# Future expression language examples:
#
# ==============================================================================
# ENVIRONMENTAL PHENOMENA (describe the world state)
# ==============================================================================
#
#   - id: "raining"
#     type: "scalar"
#     expression: gt(normal_dist(), 0.7)
#     description: "30% chance of rain each step (stochastic weather)"
#
#   - id: "cloudy"
#     type: "scalar"
#     expression: gt(perlin_noise(step_count, seed=42), 0.5)
#     description: "Smooth weather patterns using Perlin noise"
#
#   - id: "noisy_environment"
#     type: "scalar"
#     expression: gt(simplex_noise(step_count, time_of_day, seed=123), 0.6)
#     description: "Time-varying noise levels (affects social interactions)"
#
#   - id: "rush_hour"
#     type: "scalar"
#     expression: or(and(gte(time_of_day, 7), lte(time_of_day, 9)),
#                     and(gte(time_of_day, 17), lte(time_of_day, 19)))
#     description: "True during morning/evening commute hours"
#
# ==============================================================================
# BASIC DERIVED FEATURES (simple computations)
# ==============================================================================
#
#   - id: "age"
#     type: "scalar"
#     expression: divide(step_count, max_steps)
#     description: "Episode progress [0, 1] - how far through the episode"
#
#   - id: "energy_deficit"
#     type: "scalar"
#     expression: subtract(1.0, bar["energy"])
#     description: "How much energy is missing [0, 1]"
#
#   - id: "total_resources"
#     type: "scalar"
#     expression: mean(bar["energy"], bar["health"], bar["satiation"])
#     description: "Average wellbeing across physical needs"
#
#   - id: "wealth_ratio"
#     type: "scalar"
#     expression: divide(bar["money"], 100.0)
#     description: "Normalized wealth (assuming max $100)"
#
# ==============================================================================
# NON-LINEAR TRANSFORMS (urgency and priority features)
# ==============================================================================
#
#   - id: "energy_urgency"
#     type: "scalar"
#     expression: sigmoid(multiply(variable["energy_deficit"], 5))
#     description: "Squashed urgency [0,1] - grows fast as energy drops"
#
#   - id: "hygiene_urgency"
#     type: "scalar"
#     expression: sigmoid(multiply(subtract(1.0, bar["hygiene"]), 6))
#     description: "Bathroom urgency curve - sharp increase when critical"
#
#   - id: "social_priority"
#     type: "scalar"
#     expression: smoothstep(0.3, 0.7, bar["social"])
#     description: "Smooth priority curve - avoid hard threshold at 0.5"
#
#   - id: "fitness_deviation"
#     type: "scalar"
#     expression: tanh(multiply(subtract(bar["fitness"], 0.5), 4))
#     description: "How far from ideal fitness [-1,1], centered at 0.5"
#
# ==============================================================================
# TEMPORAL FEATURES (trends and memory)
# ==============================================================================
#
#   - id: "energy_velocity"
#     type: "scalar"
#     expression: delta(bar["energy"])
#     description: "Change in energy since last step (positive = recovering)"
#
#   - id: "energy_smooth"
#     type: "scalar"
#     expression: moving_average(bar["energy"], 10)
#     description: "10-step moving average - filters out noise"
#
#   - id: "health_trend"
#     type: "scalar"
#     expression: rate_of_change(bar["health"], 5)
#     description: "Health acceleration over 5 steps (crisis detection)"
#
#   - id: "recent_social"
#     type: "scalar"
#     expression: ema(bar["social"], 0.3)
#     description: "Exponential moving average - recent social weighted more"
#
#   - id: "energy_was_higher"
#     type: "scalar"
#     expression: gt(lag(bar["energy"], 5), bar["energy"])
#     description: "True if energy 5 steps ago was higher (declining trend)"
#
# ==============================================================================
# AGGREGATION (bottlenecks and comparisons)
# ==============================================================================
#
#   - id: "worst_physical_need"
#     type: "scalar"
#     expression: min(bar["energy"], bar["health"], bar["satiation"])
#     description: "Bottleneck - which physical need is most critical"
#
#   - id: "best_mental_state"
#     type: "scalar"
#     expression: max(bar["mood"], bar["social"])
#     description: "Best mental wellbeing indicator"
#
#   - id: "critical_needs_count"
#     type: "scalar"
#     expression: count_where(lt(bar["energy"], 0.3), lt(bar["health"], 0.3),
#                             lt(bar["hygiene"], 0.3), lt(bar["satiation"], 0.3))
#     description: "How many needs are in crisis (< 30%)"
#
#   - id: "physical_variance"
#     type: "scalar"
#     expression: variance(bar["energy"], bar["health"], bar["satiation"], bar["fitness"])
#     description: "How unbalanced are physical needs (high = unbalanced)"
#
# ==============================================================================
# SPATIAL FEATURES (navigation and proximity)
# ==============================================================================
#
#   - id: "fridge_distance"
#     type: "scalar"
#     expression: distance_to_affordance("Fridge")
#     description: "Manhattan distance to nearest fridge"
#
#   - id: "bathroom_urgency_spatial"
#     type: "scalar"
#     expression: multiply(divide(1.0, clamp(distance_to_affordance("Toilet"), 1, 10)),
#                          variable["hygiene_urgency"])
#     description: "Urgency × proximity - panic when far and desperate"
#
#   - id: "near_social_hub"
#     type: "scalar"
#     expression: in_range("SocialHub", 2)
#     description: "Boolean - within 2 cells of social area"
#
#   - id: "heading_to_bed"
#     type: "scalar"
#     expression: gt(dot(normalize(variable["velocity"]),
#                         direction_to_affordance("Bed")), 0.7)
#     description: "True if moving toward bed (alignment > 0.7)"
#
# ==============================================================================
# THRESHOLD/HYSTERESIS (stable state detection)
# ==============================================================================
#
#   - id: "starving"
#     type: "scalar"
#     expression: threshold(bar["satiation"], 0.15, 0.25)
#     description: "Starvation state with hysteresis (prevents flickering)"
#
#   - id: "exhausted"
#     type: "scalar"
#     expression: threshold(bar["energy"], 0.10, 0.20)
#     description: "Exhaustion state - stays true until 20% recovery"
#
#   - id: "energy_crisis_started"
#     type: "scalar"
#     expression: falling_edge(threshold(bar["energy"], 0.25, 0.30))
#     description: "Fires once when entering crisis zone (event detection)"
#
# ==============================================================================
# RANKING/PRIORITY (decision support)
# ==============================================================================
#
#   - id: "most_urgent_need_index"
#     type: "scalar"
#     expression: argmin(bar["energy"], bar["health"], bar["satiation"],
#                        bar["hygiene"], bar["mood"])
#     description: "Index of most critical need (0=energy, 1=health, etc.)"
#
#   - id: "need_priorities"
#     type: "vecNf"
#     dims: 5
#     expression: normalize([variable["energy_urgency"], variable["hygiene_urgency"],
#                           bar["health"], bar["satiation"], bar["mood"]])
#     description: "Normalized priority distribution summing to 1.0"
#
# ==============================================================================
# CONDITIONAL LOGIC (context-dependent features)
# ==============================================================================
#
#   - id: "action_speed_modifier"
#     type: "scalar"
#     expression: if_then_else(variable["raining"], 0.5, 1.0)
#     description: "Move slower when raining (50% speed penalty)"
#
#   - id: "job_availability"
#     type: "scalar"
#     expression: if_then_else(variable["rush_hour"], 2.0, 1.0)
#     description: "Double job income during rush hour"
#
#   - id: "time_of_day_bonus"
#     type: "scalar"
#     expression: switch(floor(divide(time_of_day, 6)),
#                        [0.5, 1.0, 1.2, 0.8])  # night, morning, day, evening
#     description: "Energy recovery multiplier by time period"
#
#   - id: "first_critical_need"
#     type: "scalar"
#     expression: coalesce(if_then_else(lt(bar["energy"], 0.3), bar["energy"], 0),
#                          if_then_else(lt(bar["health"], 0.3), bar["health"], 0),
#                          if_then_else(lt(bar["satiation"], 0.3), bar["satiation"], 0))
#     description: "Value of first critical need found, or 0 if none"
#
# ==============================================================================
# DECAY & GROWTH FUNCTIONS (time-based curves)
# ==============================================================================
#
#   - id: "fatigue_accumulation"
#     type: "scalar"
#     expression: exponential_growth(0.1, 0.05, step_count)
#     description: "Fatigue grows exponentially over episode"
#
#   - id: "adrenaline_decay"
#     type: "scalar"
#     expression: exponential_decay(variable["initial_panic"], 0.1, step_count)
#     description: "Panic fades over time after triggering event"
#
#   - id: "skill_mastery"
#     type: "scalar"
#     expression: logistic_growth(1.0, 0.05, 0.5, variable["age"])
#     description: "S-curve learning: slow start, rapid middle, plateau"
#
#   - id: "difficulty_curve"
#     type: "scalar"
#     expression: power_law(variable["age"], 1.5)
#     description: "Difficulty ramps non-linearly (1.5 exponent)"
#
# ==============================================================================
# FOURIER/PERIODIC FEATURES (cyclical encoding)
# ==============================================================================
#
#   - id: "time_cyclic"
#     type: "vecNf"
#     dims: 2
#     expression: cyclic_encode(time_of_day, 24)
#     description: "[sin(2π*t/24), cos(2π*t/24)] - smooth 24h cycle"
#
#   - id: "episode_phase"
#     type: "vecNf"
#     dims: 2
#     expression: cyclic_encode(variable["age"], 1.0)
#     description: "Encode episode progress as periodic (handles wrap)"
#
#   - id: "multi_frequency_time"
#     type: "vecNf"
#     dims: 8
#     expression: fourier_features(time_of_day, 4)
#     description: "Multiple frequencies capture daily+weekly patterns"
#
# ==============================================================================
# GEOMETRIC TRANSFORMS (spatial reasoning)
# ==============================================================================
#
#   - id: "angle_to_target"
#     type: "scalar"
#     expression: angle_between(variable["position"],
#                               affordance["Job"].position)
#     description: "Angle in radians [-π, π] to job location"
#
#   - id: "compass_bearing_to_home"
#     type: "scalar"
#     expression: bearing(variable["position"],
#                         affordance["Bed"].position)
#     description: "Cardinal direction [0°=N, 90°=E, 180°=S, 270°=W]"
#
#   - id: "forward_position"
#     type: "vecNf"
#     dims: 2
#     expression: add(variable["position"],
#                     rotate_2d(variable["velocity"], 0))
#     description: "Where I'll be next step if I keep moving"
#
# ==============================================================================
# ONE-HOT & CATEGORICAL (discrete encoding)
# ==============================================================================
#
#   - id: "current_action_encoded"
#     type: "vecNf"
#     dims: 8
#     expression: one_hot(variable["last_action"], 8)
#     description: "One-hot encoding of previous action [0,0,1,0,0,0,0,0]"
#
#   - id: "need_distribution"
#     type: "vecNf"
#     dims: 8
#     expression: softmax([bar["energy"], bar["health"], bar["satiation"],
#                         bar["money"], bar["mood"], bar["social"],
#                         bar["fitness"], bar["hygiene"]])
#     description: "Probability distribution over needs (sums to 1.0)"
#
#   - id: "most_urgent_action"
#     type: "scalar"
#     expression: categorical_sample(variable["need_distribution"])
#     description: "Stochastically sample action weighted by urgency"
#
# ==============================================================================
# LOOKBACK VARIATIONS (event detection)
# ==============================================================================
#
#   - id: "energy_just_crossed_30"
#     type: "scalar"
#     expression: cross_below(bar["energy"], 0.3)
#     description: "True only on the step energy drops below 30%"
#
#   - id: "health_recovering"
#     type: "scalar"
#     expression: cross_above(bar["health"], 0.5)
#     description: "True when health crosses back above 50%"
#
#   - id: "recent_energy_window"
#     type: "vecNf"
#     dims: 10
#     expression: window(bar["energy"], 10)
#     description: "Last 10 energy values as vector for pattern recognition"
#
#   - id: "stuck_count"
#     type: "scalar"
#     expression: time_since(gt(magnitude(variable["velocity"]), 0.1))
#     description: "How many steps since I last moved significantly"
#
# ==============================================================================
# COMPARISON EXTENSIONS (cleaner bounds)
# ==============================================================================
#
#   - id: "in_comfort_zone"
#     type: "scalar"
#     expression: between(bar["energy"], 0.4, 0.8)
#     description: "True if energy in comfortable range [0.4-0.8]"
#
#   - id: "same_position_as_before"
#     type: "scalar"
#     expression: is_close(variable["position"],
#                          lag(variable["position"], 5),
#                          0.5)
#     description: "True if position nearly unchanged in 5 steps"
#
#   - id: "energy_percentile"
#     type: "scalar"
#     expression: percentile(window(bar["energy"], 100), 0.25)
#     description: "25th percentile energy over last 100 steps"
#
# ==============================================================================
# ADVANCED NOISE (stochastic events)
# ==============================================================================
#
#   - id: "random_event_trigger"
#     type: "scalar"
#     expression: bernoulli(0.05)
#     description: "5% chance each step (coin flip for rare events)"
#
#   - id: "visitor_count"
#     type: "scalar"
#     expression: poisson(multiply(variable["social"], 3))
#     description: "Discrete visitor count (more when social is high)"
#
#   - id: "weather_noise"
#     type: "scalar"
#     expression: clamp(add(0.5, gaussian_noise(0, 0.2)), 0, 1)
#     description: "Noisy weather intensity centered at 0.5"
#
# ==============================================================================
# COMPLEX COMPOSITE FEATURES (multi-operator)
# ==============================================================================
#
#   - id: "panic_level"
#     type: "scalar"
#     expression: multiply(sigmoid(multiply(variable["critical_needs_count"], 2)),
#                          if_then_else(gt(variable["energy_velocity"], 0), 0.5, 1.0))
#     description: "Panic increases with multiple crises, reduced if recovering"
#
#   - id: "bathroom_emergency"
#     type: "scalar"
#     expression: and(threshold(bar["hygiene"], 0.1, 0.15),
#                     gt(distance_to_affordance("Toilet"), 3))
#     description: "Critical: low hygiene AND far from bathroom"
#
#   - id: "exploration_bonus"
#     type: "scalar"
#     expression: multiply(smoothstep(0.5, 1.0, variable["total_resources"]),
#                          variable["age"])
#     description: "Explore more when secure AND later in episode"
#
#   - id: "optimal_sleep_time"
#     type: "scalar"
#     expression: and(or(lt(time_of_day, 6), gt(time_of_day, 22)),
#                     lt(bar["energy"], 0.5))
#     description: "Night time AND tired = good time to sleep"
#
#   - id: "adaptive_urgency_with_context"
#     type: "scalar"
#     expression: multiply(
#                     variable["energy_urgency"],
#                     if_then_else(between(time_of_day, 9, 17), 1.5, 1.0),
#                     if_then_else(variable["raining"], 0.8, 1.0))
#     description: "Urgency boosted during work hours, reduced in rain"
#
# Available references:
#   - bar["name"] - meter values
#   - variable["name"] - other variables (including auto-generated)
#   - affordance["name"] - affordance availability
#   - step_count - current step in episode
#   - max_steps - maximum episode length
#   - time_of_day - current hour [0, 23]
#
# Operator categories:
#
# 1. MATHEMATICAL (already covered):
#    add, subtract, multiply, divide, pow, sqrt, abs, mod
#    exp, log, floor, ceil, round, clamp(val, min, max)
#
# 2. TRIGONOMETRIC (already covered):
#    sin, cos, tan, asin, acos, atan, atan2
#
# 3. LOGICAL (already covered):
#    and, or, not, xor
#    gt, lt, eq, gte, lte, neq
#
# 4. PROBABILITY/STOCHASTIC (already covered):
#    normal_dist(mean, std), uniform_random(min, max)
#    perlin_noise(x, seed), simplex_noise(x, y, seed)
#
# 5. NON-LINEAR TRANSFORMS (squashing/activation functions):
#    sigmoid(x) - squash to [0, 1], useful for "urgency" features
#    tanh(x) - squash to [-1, 1], useful for "deviation" features
#    relu(x) - max(0, x), useful for "positive only" features
#    softplus(x) - smooth relu, differentiable version
#    Example: urgency = sigmoid(multiply(energy_deficit, 5))
#
# 6. TEMPORAL/WINDOWING (memory over time):
#    lag(variable, steps) - value N steps ago (memory window)
#    delta(variable) - change since last step (velocity)
#    moving_average(variable, window) - smooth out noise
#    ema(variable, alpha) - exponential moving average (recent bias)
#    rate_of_change(variable, window) - acceleration/trend
#    Example: energy_trend = delta(bar["energy"])
#
# 7. AGGREGATION/STATISTICS (over sets):
#    min(a, b, ...), max(a, b, ...), mean(a, b, ...)
#    sum(a, b, ...), count_where(condition)
#    std(a, b, ...), variance(a, b, ...)
#    Example: worst_need = min(bar["energy"], bar["health"], bar["satiation"])
#
# 8. SPATIAL (distance/direction to points of interest):
#    distance_to_affordance(name) - Manhattan/Euclidean distance
#    nearest_affordance(type) - find closest of type
#    direction_to_affordance(name) - angle or unit vector
#    in_range(affordance, radius) - boolean proximity check
#    Example: bathroom_urgency = multiply(divide(1, distance_to_affordance("Toilet")), bar["hygiene_deficit"])
#
# 9. INTERPOLATION/SMOOTHING (smooth transitions):
#    lerp(a, b, t) - linear interpolation between a and b
#    smoothstep(edge0, edge1, x) - smooth Hermite interpolation
#    ease_in(t), ease_out(t), ease_in_out(t) - animation curves
#    Example: priority = smoothstep(0.2, 0.8, bar["energy"])
#
# 10. THRESHOLD/HYSTERESIS (avoid flickering):
#     step(x, threshold) - 0 below, 1 above
#     threshold(x, low, high) - 0/1 with hysteresis band
#     rising_edge(variable) - true when crosses from 0→1
#     falling_edge(variable) - true when crosses from 1→0
#     Example: starving = threshold(bar["energy"], 0.15, 0.25)
#
# 11. RANKING/PRIORITY (order by importance):
#     argmax(a, b, c) - index of maximum value
#     argmin(a, b, c) - index of minimum value
#     rank(values) - relative ranking [0, 1]
#     normalize(values) - scale to sum=1 (priority distribution)
#     Example: most_urgent_need = argmin(bar["energy"], bar["health"], bar["hygiene"])
#
# 12. CONDITIONAL (branching logic):
#     if_then_else(condition, true_val, false_val)
#     switch(selector, [val0, val1, val2, ...])
#     coalesce(a, b, c) - first non-zero value
#     Example: action_bonus = if_then_else(variable["raining"], 0.5, 1.0)
#
# 13. VECTOR OPERATIONS (for multi-dimensional features):
#     dot(vec_a, vec_b) - dot product (similarity)
#     magnitude(vec) - length/norm
#     normalize(vec) - unit vector (direction only)
#     distance(vec_a, vec_b) - Euclidean distance
#     Example: goal_alignment = dot(normalize(position), normalize(target_position))

version: "1.0"

# Custom computed variables (empty for this config - using only auto-generated)
variables: []
