# Hamlet v2.0: Software Defined World (Code-accurate)

Document Date: 3 November 2025
Status: Approved for Implementation
Owner: Principal Technical Advisor (AI)

## 1. Purpose

This spec defines the configuration contract for the Software Defined World. It is written to match the current code paths:

* `townlet.environment.cascade_config` for bars and cascades
* `townlet.environment.cascade_engine.CascadeEngine`
* `townlet.environment.affordance_config` and `affordance_engine.AffordanceEngine`
* `townlet.environment.vectorized_env.VectorizedHamletEnv` temporal mechanics, action masks and interaction flow

The SDW is the single source of truth for:

* meter definitions and terminal conditions
* depletion, modulations, and threshold cascades
* affordance catalogue, costs, effects, and opening hours
* time-of-day behaviour and multi-tick interactions

All meters are normalised to [0.0, 1.0]. Money uses 1.0 to represent 100 dollars by default, which keeps code simple and avoids mixed units.

## 2. Meter Canon

The code expects exactly 8 meters with fixed indices. Keep these names and the index mapping.

```
# METER_NAME_TO_IDX
energy:   0
hygiene:  1
satiation:2
money:    3
mood:     4
social:   5
health:   6
fitness:  7
```

Notes

* energy and health are pivotal. Hitting zero on either is terminal.
* money is a resource gate. Affordability checks are performed by the AffordanceEngine or at interaction time in the env.
* meters are clamped to [0, 1] after each operation inside the engines.

## 3. bars.yaml (code-accurate schema)

`BarsConfig` schema fields are enforced by Pydantic. Exactly 8 bars. Range must be [0.0, 1.0]. Terminal conditions live here too.

```yaml
version: "1.0"
description: "Hamlet v2 meters"
bars:
  - name: "energy"
    index: 0
    tier: "pivotal"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.005
    description: "Ability to act and move"
    key_insight: "Dies if zero"
  - name: "hygiene"
    index: 1
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.003
    description: "Cleanliness supports health and mood"
  - name: "satiation"
    index: 2
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.004
    description: "Food level; core driver of survival"
    cascade_pattern: "strong to energy and health when low"
  - name: "money"
    index: 3
    tier: "resource"
    range: [0.0, 1.0]
    initial: 0.5             # 0.5 = $50
    base_depletion: 0.0
    description: "Budget for affordances"
  - name: "mood"
    index: 4
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 0.7
    base_depletion: 0.001
    description: "Affects energy dynamics when low"
  - name: "social"
    index: 5
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.006
    description: "Supports mood over time"
  - name: "health"
    index: 6
    tier: "pivotal"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.0
    description: "General condition; death if zero (managed via fitness modulation)"
  - name: "fitness"
    index: 7
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 0.5
    base_depletion: 0.0
    description: "Improves health dynamics when high"

terminal_conditions:
  - meter: "energy"
    operator: "<="
    value: 0.0
    description: "Death by exhaustion"
  - meter: "health"
    operator: "<="
    value: 0.0
    description: "Death by health failure"

notes:
  - "All values normalised. Money: 1.0 means $100."
```

## 4. cascades.yaml (code-accurate schema and math)

The engine supports:

* Modulations which are depletion multipliers parameterised by a source meter. Example: fitness modifies how quickly health decays.
* Threshold cascades which apply penalties to a target when a source is below a threshold. The penalty strength scales with the deficit.

Execution order comes from YAML and is applied in order each step. The code uses gradient penalty maths for cascades and your modulation field names.

```yaml
version: "1.0"
description: "Hamlet cascade physics"
math_type: "gradient_penalty"

modulations:
  - name: "fitness_modulates_health_decay"
    description: "Good fitness reduces health decay; poor fitness increases it"
    source: "fitness"
    target: "health"
    type: "depletion_multiplier"
    base_multiplier: 0.5       # at fitness = 1.0
    range: 2.5                 # 0.5 + 2.5 = 3.0 at fitness = 0.0
    baseline_depletion: 0.002  # align with bars.health.base_depletion
    note: "Replicates legacy behaviour"

cascades:
  # secondary to pivotal via primary_to_pivotal category label
  - name: "low_satiation_hits_energy"
    description: "Hungry drains energy quickly"
    category: "primary_to_pivotal"
    source: "satiation"
    source_index: 2
    target: "energy"
    target_index: 0
    threshold: 0.2
    strength: 0.015

  - name: "low_satiation_hits_health"
    description: "Hungry also undermines health"
    category: "primary_to_pivotal"
    source: "satiation"
    source_index: 2
    target: "health"
    target_index: 6
    threshold: 0.2
    strength: 0.010

  - name: "low_mood_hits_energy"
    description: "Depressed, low energy"
    category: "primary_to_pivotal"
    source: "mood"
    source_index: 4
    target: "energy"
    target_index: 0
    threshold: 0.2
    strength: 0.010

  - name: "low_fitness_hits_health"
    description: "Unfit, poor health dynamics"
    category: "primary_to_pivotal"
    source: "fitness"
    source_index: 7
    target: "health"
    target_index: 6
    threshold: 0.2
    strength: 0.010

  # tertiary to secondary via secondary_to_primary category label
  - name: "low_hygiene_hits_secondary"
    description: "Poor hygiene worsens satiation, mood, fitness dynamics"
    category: "secondary_to_primary"
    source: "hygiene"
    source_index: 1
    target: "satiation"
    target_index: 2
    threshold: 0.2
    strength: 0.006
  - name: "low_hygiene_hits_mood"
    description: "Hygiene affects mood"
    category: "secondary_to_primary"
    source: "hygiene"
    source_index: 1
    target: "mood"
    target_index: 4
    threshold: 0.2
    strength: 0.006
  - name: "low_hygiene_hits_fitness"
    description: "Hygiene affects fitness"
    category: "secondary_to_primary"
    source: "hygiene"
    source_index: 1
    target: "fitness"
    target_index: 7
    threshold: 0.2
    strength: 0.006

  - name: "low_social_hits_mood"
    description: "Isolation reduces mood"
    category: "secondary_to_primary"
    source: "social"
    source_index: 5
    target: "mood"
    target_index: 4
    threshold: 0.2
    strength: 0.010

  # weak tertiary to pivotal route if desired
  - name: "low_hygiene_hits_health_weak"
    description: "Hygiene to health weak path"
    category: "secondary_to_pivotal_weak"
    source: "hygiene"
    source_index: 1
    target: "health"
    target_index: 6
    threshold: 0.1
    strength: 0.003

execution_order:
  - "modulations"
  - "primary_to_pivotal"
  - "secondary_to_primary"
  - "secondary_to_pivotal_weak"

notes:
  - "Engine applies each category in order per step"
  - "Penalty = strength * ((threshold - source)/threshold) for source below threshold"
```

## 5. affordances.yaml (code-accurate schema and examples)

`AffordanceConfigCollection` and `AffordanceConfig` are enforced by Pydantic. Interaction types supported by code:

* instant
* multi_tick
* continuous
* dual

Fields:

* `required_ticks` must be set for `multi_tick` and `dual`. Omit it otherwise.
* `operating_hours` is a two element list. Example `[8, 18]` means 8 to 18. Wrap past midnight by using a close hour up to 28. Example `[18, 28]` means 18 to 4.
* amounts in costs and effects are normalised to [0.0, 1.0]. For money, 0.1 is 10 dollars by default.

Important flow from the code:

* The env computes action masks for INTERACT based on location and hours. It does not mask on affordability. If you cannot afford the instant cost, the affordance handler simply does nothing productive that tick, which is a teaching feature.
* Multi-tick progress is tracked per agent. Moving off the tile resets progress.

Example configuration with the standard world objects placed in `VectorizedHamletEnv`:

```yaml
version: "1.0"
description: "Hamlet affordances"
status: "PRODUCTION"

affordances:
  - id: "bed"
    name: "Bed"
    category: "energy_restoration"
    interaction_type: "multi_tick"
    required_ticks: 4
    costs: []                       # instant costs (not used here)
    costs_per_tick: []              # free to sleep
    effects: []                     # not used for multi_tick
    effects_per_tick:
      - { meter: "energy", amount: 0.25, type: "linear" }
    completion_bonus:
      - { meter: "energy", amount: 0.25 }
    operating_hours: [0, 24]
    teaching_note: "Cheap but slow recovery"

  - id: "luxury_bed"
    name: "LuxuryBed"
    category: "energy_restoration"
    interaction_type: "multi_tick"
    required_ticks: 3
    costs_per_tick:
      - { meter: "money", amount: 0.05 }    # 5 dollars per tick
    effects_per_tick:
      - { meter: "energy", amount: 0.30, type: "linear" }
    completion_bonus:
      - { meter: "hygiene", amount: 0.10 }
    operating_hours: [0, 24]
    design_intent: "Faster but costs money"

  - id: "shower"
    name: "Shower"
    category: "hygiene_recovery"
    interaction_type: "instant"
    costs:
      - { meter: "money", amount: 0.02 }    # 2 dollars
    effects:
      - { meter: "hygiene", amount: 0.40 }
    operating_hours: [0, 24]

  - id: "home_meal"
    name: "HomeMeal"
    category: "food"
    interaction_type: "instant"
    costs:
      - { meter: "money", amount: 0.05 }    # 5 dollars
    effects:
      - { meter: "satiation", amount: 0.40 }
    operating_hours: [6, 24]

  - id: "fast_food"
    name: "FastFood"
    category: "food"
    interaction_type: "instant"
    costs:
      - { meter: "money", amount: 0.08 }
    effects:
      - { meter: "satiation", amount: 0.50 }
      - { meter: "hygiene", amount: -0.05 }
    operating_hours: [10, 28]   # 10 to 4

  - id: "job"
    name: "Job"
    category: "income"
    interaction_type: "multi_tick"
    required_ticks: 8
    costs_per_tick:
      - { meter: "energy", amount: 0.05 }   # see env movement costs too
    effects_per_tick:
      - { meter: "money", amount: 0.10 }    # 10 dollars per tick
      - { meter: "mood", amount: -0.02 }   # consumed focus reduces mood
    completion_bonus:
      - { meter: "money", amount: 0.40 }
    operating_hours: [9, 18]

  - id: "labor"
    name: "Labor"
    category: "income"
    interaction_type: "multi_tick"
    required_ticks: 4
    costs_per_tick:
      - { meter: "energy", amount: 0.08 }
      - { meter: "hygiene", amount: 0.03 }
    effects_per_tick:
      - { meter: "money", amount: 0.12 }
      - { meter: "fitness", amount: 0.04 }
    completion_bonus: []
    operating_hours: [7, 17]

  - id: "gym"
    name: "Gym"
    category: "fitness_builder"
    interaction_type: "multi_tick"
    required_ticks: 3
    costs_per_tick:
      - { meter: "energy", amount: 0.06 }
      - { meter: "money", amount: 0.05 }
    effects_per_tick:
      - { meter: "fitness", amount: 0.12 }
      - { meter: "mood", amount: 0.04 }
    completion_bonus: []
    operating_hours: [6, 22]

  - id: "bar"
    name: "Bar"
    category: "social_mood"
    interaction_type: "instant"
    costs:
      - { meter: "money", amount: 0.20 }
    effects:
      - { meter: "social", amount: 0.30 }
      - { meter: "mood", amount: 0.15 }
    operating_hours: [18, 28]

  - id: "park"
    name: "Park"
    category: "mood_social_free"
    interaction_type: "instant"
    costs: []
    effects:
      - { meter: "mood", amount: 0.10 }
      - { meter: "social", amount: 0.05 }
    operating_hours: [6, 20]

  - id: "recreation"
    name: "Recreation"
    category: "mood_tier1"
    interaction_type: "multi_tick"
    required_ticks: 2
    costs_per_tick:
      - { meter: "money", amount: 0.05 }
    effects_per_tick:
      - { meter: "mood", amount: 0.20 }
    completion_bonus: []
    operating_hours: [10, 22]

  - id: "therapist"
    name: "Therapist"
    category: "mood_tier2"
    interaction_type: "multi_tick"
    required_ticks: 3
    costs_per_tick:
      - { meter: "money", amount: 0.15 }
    effects_per_tick:
      - { meter: "mood", amount: 0.25 }
    completion_bonus: []
    operating_hours: [9, 17]

  - id: "doctor"
    name: "Doctor"
    category: "health_tier1"
    interaction_type: "multi_tick"
    required_ticks: 2
    costs_per_tick:
      - { meter: "money", amount: 0.15 }
    effects_per_tick:
      - { meter: "health", amount: 0.25 }
    completion_bonus: []
    operating_hours: [8, 18]

  - id: "hospital"
    name: "Hospital"
    category: "health_tier2"
    interaction_type: "multi_tick"
    required_ticks: 2
    costs_per_tick:
      - { meter: "money", amount: 0.30 }
    effects_per_tick:
      - { meter: "health", amount: 0.40 }
      - { meter: "energy", amount: 0.10 }
    completion_bonus: []
    operating_hours: [0, 24]

  # Ambulance by phone within the current engine model
  - id: "call_ambulance"
    name: "HomePhoneAmbulance"
    category: "health_emergency"
    interaction_type: "instant"
    costs:
      - { meter: "money", amount: 0.50 }     # 50 dollars
    effects:
      - { meter: "health", amount: 0.30 }    # immediate stabilisation
      - { meter: "energy", amount: -0.05 }   # shock and fatigue
    operating_hours: [0, 24]
    teaching_note: "Approximates ambulance response without movement"
```

Why model ambulance like this

* Today the engines do not implement a position-changing effect. Your code applies meter deltas and handles open hours, affordability, and tick progression, but it does not move the agent.
* This version uses an instant health increase and cost to capture the decision trade-off. It teaches that an expensive emergency option exists even if you are nowhere near the Doctor or Hospital tiles.

Optional extension for literal relocation

If you do want real repositioning, the minimal change is:

1. Extend `AffordanceEffect` with a controlled `type` enum and a payload for relocation.

```python
class AffordanceEffect(BaseModel):
    meter: str | None = None
    amount: float | None = None
    type: Literal["linear", "relocate"] | None = None
    destination: str | None = None
```

2. In `AffordanceEngine.apply_instant_interaction`, detect `type == "relocate"`. Return a side-channel in the step info dict, for example `info["relocations"] = [(agent_idx, destination_name)]`.
3. In `VectorizedHamletEnv._handle_interactions_legacy` consume `relocations` and set `self.positions[agent_idx] = affordance_positions[destination_name]`.

Then you can write:

```yaml
- id: "call_ambulance"
  name: "HomePhoneAmbulance"
  category: "health_emergency"
  interaction_type: "instant"
  costs:
    - { meter: "money", amount: 0.50 }
  effects:
    - { type: "relocate", destination: "Hospital" }
  operating_hours: [0, 24]
```

This preserves data-driven control and keeps the engines clean.

## 6. Time of day and action masks

* `operating_hours` follow the rule:

  * normal: `[open, close]` with 0 ≤ open < close ≤ 24 and open ≤ time < close
  * wrap-around: allow close up to 28. Env computes `close % 24` and treats it as open if `time ≥ open or time < close%24`
* The env increments `time_of_day = (time_of_day + 1) % 24` per step when temporal mechanics are enabled.
* Action space is `[UP, DOWN, LEFT, RIGHT, INTERACT, WAIT]`. Movement and wait costs are applied by the env, not the YAML.

  * default move costs applied by the env include energy 0.005 (configurable via `energy_move_depletion`), hygiene 0.003, satiation 0.004
  * wait uses a lighter energy cost (default 0.001, configurable via `energy_wait_depletion`), no other passives
  * interact deducts `energy_interact_depletion` each time it is invoked (default 0.0)
* INTERACT is valid if the agent is exactly on an affordance tile and it is open. Affordability is not part of action masking. Failing to afford is a wasted turn and a learning signal.

## 7. Multi-tick semantics

* For `multi_tick` and `dual`, the env tracks `interaction_progress` per agent.
* Movement off the tile or switching affordances resets progress.
* On the last tick of a completed interaction, `completion_bonus` is added after per-tick effects. All amounts are clamped after application.

## 8. Positions and observation alignment

`VectorizedHamletEnv` seeds a default layout keyed by the `name` fields above. Keep the names identical or update both config and env.

Default positions, for reference:

```
Bed:        [1,1]
LuxuryBed:  [2,1]
Shower:     [2,2]
HomeMeal:   [1,3]
FastFood:   [5,6]
Job:        [6,6]
Labor:      [7,6]
Gym:        [7,3]
Bar:        [7,0]
Park:       [0,4]
Recreation: [0,7]
Therapist:  [1,7]
Doctor:     [5,1]
Hospital:   [6,1]
```

The env exports and restores these via checkpoint to keep the observation encoding consistent.

## 9. Validation rules you actually get from the loaders

* bars.yaml

  * exactly 8 bars, indices 0..7, unique names and indices
  * range must be [0.0, 1.0]
* cascades.yaml

  * unique cascade names, categories are free-form strings used to group execution order
  * modulations expect the depletion multiplier fields as defined
* affordances.yaml

  * `interaction_type` must be one of the literal values
  * `required_ticks` is required for multi_tick and dual, invalid otherwise
  * `operating_hours` must be two integers, 0 ≤ open ≤ 23, 1 ≤ close ≤ 28

## 10. Equivalence, tests, and teaching packs

* Keep your hardcoded legacy tests around and assert equivalence to the config-driven engines with tight tolerances.
* Maintain three packs for pedagogy:

  * `cascades_weak.yaml` halve the strength fields
  * `cascades.yaml` baseline
  * `cascades_strong.yaml` increase strength by 50 percent
* Affordance packs can teach economic regimes:

  * “austerity”: cheaper food and sleep, lower income
  * “boom”: higher wages, higher rents (modelled as higher costs on beds or shower)
  * “nightlife”: more things open with wrap-around hours

## 11. Ambulance pathway design, end-to-end

What the player can do under this spec

1. Walk to Doctor during office hours. Slower but cheaper.
2. Walk to Hospital, always open, more expensive but stronger restoration.
3. Use HomePhoneAmbulance anywhere, any time. Expensive and does not relocate by default, but stabilises health immediately. If you adopt the `relocate` extension, it will reposition the agent to the Hospital tile as part of the instant effect.

This yields a clear tactical set:

* short-term stabilisation at cost
* predictable, clock-aware clinic care
* always-on emergency care that is pricier

It also gives the World Model a nice spread of options to learn, and a reason to track the clock.

## 12. Implementation notes that match the code

* Movement and wait costs are part of `VectorizedHamletEnv._execute_actions` and are not configured in YAML.
* Base depletions happen in `CascadeEngine.apply_base_depletions` with an optional curriculum multiplier you can change at runtime.
* The main cascade sequence used by the env is:

  * depletions
  * primary_to_pivotal
  * secondary_to_primary
  * secondary_to_pivotal_weak
    which matches the methods `apply_secondary_to_primary_effects`, etc. The exact order is pulled from YAML `execution_order`.
* Terminal conditions are checked after cascades and set all actions invalid in masks for dead agents.

## 13. Minimal “config pack” folder content

Put these side by side. The env loads them by directory.

```
configs/test/
  bars.yaml
  cascades.yaml
  affordances.yaml
```

If you later add social cues, introduce `cues.yaml` in the same pack. Module C can then pre-train from it.

## 14. Worked example: end-to-end tick with ambulance available

1. Tick t, agent at home, time 02:00, health 0.18, money 0.60, energy 0.40
2. Agent presses INTERACT on HomePhoneAmbulance
3. Env verifies the tile has that affordance, checks hours open, does not block on affordability in the mask
4. AffordanceEngine applies instant costs and effects:

   * money becomes 0.10
   * health becomes 0.48
   * energy becomes 0.35
5. Base depletions and cascades then run as usual
6. If you implement relocate, the env also sets position to Hospital before or after cascade, depending on your insertion point

This path is consistent with the code as written and teaches trade-offs cleanly.
