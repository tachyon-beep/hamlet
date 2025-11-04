# Townlet v2.5: Universe as Code

Document Date: 3 November 2025
Status: Approved for Implementation
Owner: Principal Technical Advisor (AI)

## 1. System Overview

Townlet’s Universe as Code layer defines the simulated world as configuration rather than imperative logic. Core mechanics—survival rules, temporal constraints, reward shaping, and terminal conditions—are declared in data files that the engine loads and enforces.

Practically, this means that parameters such as hunger rates, hygiene effects, night-time safety, retirement criteria, and end-of-life scoring are formalised in YAML. Designers and instructors can adjust those configurations to explore alternative social contracts without modifying Python. Typical scenario packs range from low-pressure baselines to high-scarcity models, all implemented as data swaps.

This arrangement also provides the learning system with an explicit, inspectable ruleset. The world configuration becomes the ground truth for:

- consequence prediction (World Model / Module B)
- social reasoning (Social Model / Module C)
- long-horizon planning rather than single-tick survival

Universe as Code therefore functions as a specification of the society in which agents operate, not merely a set of ad hoc gameplay tweaks.

### Implementation footprint

Universe as Code is implemented through four cooperating subsystems:

- `cascade_config` and `cascade_engine` — define meters, depletion rates, cascades, terminal conditions, and ageing rules.
- `affordance_config` and `affordance_engine` — enumerate world actions, their costs, and their effects.
- `vectorized_env.VectorizedTownletEnv` — provides the runtime grid, time-of-day tracking, legality masks, and multi-step interaction management.
- `reward_model` — evaluates end-of-life outcomes for episodic scoring.

These configurations constitute the single source of truth for:

- the agent’s survival meters (energy, health, hunger, money, mood, and related signals)
- passive decay and cross-meter pressure
- affordance definitions (e.g., Bed, Job, Hospital, Bar) including opening hours, wages, and side effects
- temporal rules (operating hours, night-time behaviour)
- ageing and retirement semantics
- end-of-life scoring

All runtime values are normalised to the [0.0, 1.0] range. Money follows the same convention (1.0 ≈ $100 in the baseline pack), which keeps the state space consistent for policy learning.

Two design principles underpin this layer:

1. **Physics are data.** The engine enforces rules; YAML specifies them.
2. **Values are data.** Terminal scoring and moral framing are explicit and configurable, not embedded in code paths.

Words of estimative probability that this section reflects system intent: high (~90 percent).

## 2. Survival model: the meters that define a life

At the core of the world is a set of tracked quantities we call "meters" or "bars". These are continuous values from 0.0 to 1.0. They represent needs, resources, and physical condition.

The engine treats these as authoritative state. The learning agent sees them (or a partial view of them, depending on observability level), and acts to keep itself alive, functional, and ideally well off.

### Canonical meters

Right now the runtime keeps eight core meters in a fixed order. The order matters because it's baked into tensors and models.

```
Index  Name        Meaning
0      energy      Can you still move and act?
1      hygiene     How clean / disease-free are you?
2      satiation   How fed are you?
3      money       How much spendable cash you have? (1.0 ≈ $100)
4      mood        Are you mentally okay?
5      social      Are you connected to other people?
6      health      Are you medically stable?
7      fitness     Are you physically resilient?
```

Those indices are wired everywhere (policy nets, replay buffers, cascade maths, affordance effects). Changing them casually will break everything. So we treat them as stable ABI.

Some important notes about how these behave:

- Energy and health are pivotal.
  If either hits zero, you are done. You can no longer continue the run. This is classic "you collapsed" / "medical failure".

- Money is a gate, not a "feelings bar".
  You don't die when you're broke, but most recovery options cost money. Affordability is enforced at interaction time (you can't buy dinner if you've got nothing) and through affordance configs. Poverty isn't instant death, it's slow death.

- Hygiene, satiation, mood, social, fitness are all indirect killers.
  None of them instantly kill you by dropping to zero, but they lean on the pivotal bars through cascades. For example: if you never eat, your health and energy drain harder. If you never shower, your mood and fitness degrade, which then feeds back into health. It's a pressure network not a single fail switch.

- Meters are clamped after every update.
  Nothing goes below 0.0 or above 1.0. That means "perfect health" is literally 1.0, and we don't allow "extra health banked for later". Same for money: by default we cap at 1.0 = $100, though you can redefine that scale in config packs if you want a wealth-based sim instead of a subsistence sim.

This pressure network produces the emergent behaviour we expect. Hunger reduces pivotal meters indirectly by accelerating energy and health depletion. Social isolation degrades mood, which in turn erodes energy and compromises the ability to earn money and obtain food. The system is intentionally coupled.

### Lifecycle: survival versus retirement

In addition to the eight meters, the environment tracks a scalar `lifecycle` value. It begins near 0.0 and increases over time; reaching 1.0 signifies retirement rather than death.

This distinction is important:

1. Episodes end because the simulated life concludes, not due to a hard-coded step limit.
2. The system can distinguish successful retirement from catastrophic failure, delivering different rewards and telemetry.

Operationally:

- `lifecycle` increases slightly each tick.
- Adverse conditions accelerate the increase (e.g., starvation, illness, miserable mood).
- If `lifecycle` reaches 1.0 before pivotal meters reach zero, the agent retires.
- Retirement is treated as a separate terminal condition from death.

At episode end, the `reward_model` computes a final life score based on configuration (e.g., remaining money, health, mood). Dying early yields a heavily discounted score (for example, 10 percent of the retirement value), reinforcing that dignified survival matters more than simply delaying collapse. Retirement and death therefore produce distinct outcomes for learning and audit.

### Why this design works

From a training point of view, this setup does a few beautiful things.

- It gives you two different end states with different semantics.
  "Dead" and "Retired" represent distinct outcomes with different payouts, giving the agent an incentive to plan for retirement rather than merely delaying failure.

- It turns morality (what we reward at the end) into config.
  You can spin up a world where money matters most, or a world where mood and social bonds are weighted higher, or a world where healthcare at end-of-life is king. The RL agent will internalise whatever world view you give it. It's alignment by YAML.

- It moves away from short-horizon hackery.
  Instead of giving shaped "good job!" rewards for individual actions like SLEEP or EAT, we let most of the shaping come from "did you actually build a survivable, financially stable, mentally tolerable life that reached retirement". That encourages long-horizon policy learning, which is what we actually care about.

## 3. bars.yaml – what we measure, how fast it falls apart, and what ends a life

bars.yaml is where we declare what "being alive" actually means in this world.

Each meter (energy, hunger, money, etc) is defined here with:

- its index in the tensor
- its default starting value
- how fast it passively erodes every tick
- how important it is in the survival hierarchy

The engine does not derive any of this implicitly. It loads `bars.yaml`, validates it with Pydantic, and uses it as ground truth in the `CascadeEngine` and the environment. Misstating values here propagates inconsistencies across the world.

This file also declares terminal conditions: the rules that say "this life is now over". Death-by-exhaustion and death-by-health are defined the same way as any other rule. There’s no hardcoded secret kill switch in the engine.

Finally, this is also where lifecycle lives conceptually. Lifecycle is allowed to end a run even if you're physically fine. We treat "you finished your life and retired" as a terminal condition like death, except it’s dignified and high-scoring. That part is computed at the environment / reward layer, but the idea is: bars.yaml is the canonical roster of survival-relevant meters, and lifecycle sits alongside them in design terms as “time to retirement”. We treat that as first-class in how we score an episode.

A standard bars.yaml in the baseline world looks like this:

```yaml
version: "1.0"
description: "Townlet v2.5 meters"
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

### Meters

- `tier` conveys design intent. `pivotal` meters gate survival, `secondary` meters influence pivotal values via cascades, and `resource` meters provide leverage (for example, money). While the engine does not hard-code tier semantics, other systems assume consistent use of these categories.

- `initial` defines spawn values at reset. Curriculum authors can tune these values to craft harsher or more forgiving starting conditions.

- `base_depletion` specifies passive decay per tick before cascade adjustments. For instance, hygiene decays continuously, representing the background cost of remaining clean. Activity-driven effects belong in `affordances.yaml`; `bars.yaml` represents ambient pressure only.

- Ranges are fixed at [0.0, 1.0] and enforced by validation. Concepts such as debt should be modelled through affordance outcomes and cascade penalties rather than extending meter bounds. Normalisation keeps the observation space stable for policy learning.

### Terminal conditions

Terminal conditions determine when a run ends involuntarily. Each condition specifies:

- the meter to inspect
- the comparison operator
- the threshold
- a human-readable explanation

The CascadeEngine evaluates these conditions every tick and signals the environment to mark agents as done. This drives the action masking logic in `VectorizedTownletEnv`.

Additional terminal conditions are supported. Examples include:

- `money <= 0.0` and `mood <= 0.1` for a sustained interval, modelling institutionalisation without physical death
- `lifecycle >= 1.0`, representing graceful retirement rather than failure

In Universe as Code, definitions of “the end of a life” are deliberate configuration choices.

---

## 4. cascades.yaml – cross-meter dynamics

While `bars.yaml` defines available meters, `cascades.yaml` captures how neglect propagates through the system.

Two mechanisms are defined:

1. **Modulations**

   - Continuous multipliers that adjust depletion rates based on the state of another meter.
   - Example: high fitness reduces health decay; low fitness accelerates it.

2. **Threshold cascades**

   - Conditional effects that apply once a source meter falls below a threshold.
   - The further the meter drops below the threshold, the stronger the penalty to the target meter.
   - This is where hunger undermines health and energy, poor hygiene undermines mood, and social isolation degrades overall stability.

These constructs encode collapse patterns without bespoke logic, producing the expected survival spirals when multiple needs are ignored.

The `execution_order` field is operational. The engine applies cascade groups in the specified sequence each tick; reordering them changes the effective physics of the world.

Here’s the current baseline cascades.yaml:

```yaml
version: "1.0"
description: "Townlet cascade physics"
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

### Modulations

Consider `fitness_modulates_health_decay`:

- Health has a baseline passive decay (`baseline_depletion`).
- The decay rate is multiplied by a fitness-dependent factor.
- When fitness is near 1.0, the multiplier is low (0.5×), reducing health loss.
- When fitness approaches 0.0, the multiplier rises to 3.0×, accelerating health loss.

The relationship is continuous rather than thresholded, providing smooth gradients for learning while remaining physiologically interpretable.

### Threshold cascades

For `low_satiation_hits_energy`, the penalty is calculated as:

```
penalty = strength * ((threshold - source) / threshold)
```

- Satiation ≥ threshold ⟶ no penalty.
- Satiation at half the threshold ⟶ half-strength penalty.
- Satiation near zero ⟶ full-strength penalty.

The penalty subtracts from the target meter (energy in this case), producing the gradual collapse associated with prolonged hunger.

Similar cascades connect:

- hunger → energy and hunger → health
- low mood → energy
- low hygiene → mood, satiation, fitness
- low social → mood

The cumulative effect models realistic degradation patterns where multiple needs compound.

### Execution order

`execution_order` is honoured verbatim by `CascadeEngine` each tick:

1. Apply modulations (adjust depletion rates).
2. Apply `primary_to_pivotal` cascades (high-severity effects such as hunger impacting energy and health).
3. Apply `secondary_to_primary` cascades (tertiary needs eroding secondary meters).
4. Apply `secondary_to_pivotal_weak` cascades (weaker direct penalties from tertiary to pivotal meters).

Reordering these stages materially changes world dynamics. For example, promoting `secondary_to_pivotal_weak` earlier in the sequence would cause hygiene to impact health sooner and more severely.

### Difficulty as curriculum

Curriculum design is achieved through alternative cascade packs, for example:

- `cascades_weak.yaml` — reduces `strength` values, producing a forgiving world where neglect is less punishing.
- `cascades_strong.yaml` — increases `strength` values, generating a harsher environment where missed meals rapidly become critical.

Because these packs are validated configuration files, training can progress from weak to baseline to strong physics, demonstrating whether agents learn transferable survival strategies.

Lifecycle metrics integrate with this approach: harsher cascades reduce the proportion of lives reaching retirement, and the final score reflects not only survival but also the quality of that survival.

---

## 5. affordances.yaml – available actions

If `bars.yaml` defines the state space and `cascades.yaml` defines cross-meter dynamics, `affordances.yaml` specifies the actions available to agents. Actions such as sleeping, eating, working, therapy, or calling emergency services are all declared as affordances—no hard-coded exceptions.

Each affordance describes:

- how the interaction unfolds (instant, multi-tick, continuous, dual)
- the costs incurred
- the benefits returned
- operating hours
- required commitment duration

The engine loads this file at runtime to construct the action surface. Modifying `affordances.yaml` therefore redefines the socioeconomic options present in the world.

### Interaction types

The engine supports four interaction types:

- `instant` — single-tick actions completed immediately (e.g., shower, eat).
- `multi_tick` — activities requiring sustained commitment for a specified number of ticks (e.g., sleep, work, gym). Leaving early forfeits progress and completion bonuses.
- `continuous` — actions that remain active while the agent stays in place. Few baseline affordances use this mode today, but it remains available for future designs.
- `dual` — hybrid actions that perform an instant step followed by multi-tick continuation (e.g., hospital check-in followed by observation).

The environment maintains per-agent `interaction_progress` for multi_tick and dual interactions and clears it if the agent disengages.

### Key fields

- `id` and `name` — link the affordance definition to map placement. Renaming an affordance requires corresponding map updates.
- `interaction_type` — determines progress handling and reward timing.
- `required_ticks` — defines commitment duration for `multi_tick` and `dual` actions.
- `operating_hours` — specifies availability using hour pairs. `[9, 18]` denotes 09:00–18:00; `[18, 28]` denotes 18:00–04:00 the following day. The environment evaluates this every tick.
- `costs` / `costs_per_tick` — represent expenditures (money, energy, hygiene, etc.). `costs` apply upfront; `costs_per_tick` apply during sustained interactions. Values are normalised (e.g., 0.10 money ≈ $10).
- `effects` / `effects_per_tick` — represent benefits or penalties. `effects` apply instantly; `effects_per_tick` apply during sustained interactions.
- `completion_bonus` — applies when the interaction completes successfully (all required ticks fulfilled).

Agents may attempt interactions even if they cannot afford them. Failing affordability consumes time without benefit, providing natural training feedback about resource constraints.

### The baseline affordance set

This is the reference economy / lifestyle for the default world:

```yaml
version: "1.0"
description: "Townlet affordances"
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
      - { meter: "mood", amount: -0.02 }    # consumed focus reduces mood
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

The baseline set already encodes policy choices:

- Luxury rest requires spending (faster recovery with hygiene benefits at a monetary cost).
- Employment provides income while reducing mood and consuming energy.
- Mood can be purchased quickly (e.g., bar) or cultivated gradually at lower cost (e.g., recreation, park).
- Healthcare is time- and price-gated. Emergency care is always available but expensive and fatiguing.

These relationships are deliberate; configuration expresses labour, class, social, and health dynamics explicitly.

### Emergency care and mobility

The ambulance affordance currently approximates relocation because the engine does not yet support direct teleportation. The configuration therefore:

- defines the action as `instant`
- keeps it available at all hours
- applies a significant monetary cost
- restores health immediately to prevent death
- applies an energy penalty to reflect shock

The design intent is to keep the agent alive while imposing future costs, reinforcing preventative planning. Words of estimative probability that this description matches the implemented behaviour: high (~85 percent).

### Optional extension: physical relocation

Longer-term, the ambulance interaction should relocate the agent to the hospital tile. Minimal engine changes to support this include:

1. Extending the effect model:

```python
class AffordanceEffect(BaseModel):
    meter: str | None = None
    amount: float | None = None
    type: Literal["linear", "relocate"] | None = None
    destination: str | None = None
```

2. In `AffordanceEngine.apply_instant_interaction`, detect `type == "relocate"` and emit a side-channel update:

```python
info["relocations"] = [(agent_idx, destination_name)]
```

3. In `VectorizedTownletEnv._handle_interactions_legacy`, consume `relocations` after applying effects and set agent positions to the specified destinations.

With that in place, you can write an ambulance affordance like:

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

This change would transition the agent to hospital rather than only applying health adjustments, reinforcing the state change associated with a crisis. Words of estimative probability that the relocation hook can be integrated with low risk: moderate (~70 percent), assuming side-channel updates remain acceptable in `VectorizedTownletEnv`.

---

## 6. Time of day and action masks

Temporal mechanics are part of the simulation physics.

- Each affordance declares `operating_hours`, enforced every tick.

  - `[open, close]` with 0 ≤ open < close ≤ 24 means the affordance is available from `open` up to (but excluding) `close`.
  - Values where `close > 24` (up to 28) indicate wrap-around into the next day (e.g., `[18, 28]` covers 18:00–04:00).

- The environment maintains `time_of_day`. When temporal mechanics are enabled, it increments each tick:
  `time_of_day = (time_of_day + 1) % 24`.

- `INTERACT` is unmasked only when the agent stands on the relevant tile and the affordance is open at the current hour.

Affordability is not part of the mask; attempting unaffordable actions consumes time without benefit, providing natural feedback about resource scarcity.

### Action space

The action set is `[UP, DOWN, LEFT, RIGHT, INTERACT, WAIT]`.

The environment applies baseline movement and idle costs automatically:

- moving:

  - energy ~0.005
  - hygiene ~0.003
  - satiation ~0.004
    (all configurable via `environment.energy_move_depletion` etc in the run config)

- waiting:

  - lower energy cost, default ~0.001
  - no hygiene hit, no satiation hit
    (tunable via `environment.energy_wait_depletion`)

- interacting:

  - a separate `energy_interact_depletion` per press (default 0.0, tunable to model tiring interactions)

These environment-level parameters enable world-scale difficulty variants such as:

- **High commute tax** — walking is expensive
- **Burnout society** — every interaction drains energy
- **Welfare state** — waiting incurs minimal cost

Geographical placement is a primary lever for distinguishing worlds where extended workdays are viable from those where accessing work is itself hazardous.

---

## 7. Multi-tick semantics

Multi-tick affordances model activities that require sustained commitment.

For affordances with `interaction_type` of `multi_tick` or `dual`:

- The environment tracks per-agent `interaction_progress`.
- Remaining on the tile and continuing the interaction increments progress each tick.
- Leaving the tile, switching affordances, or encountering closing hours resets progress and forfeits completion bonuses.

When `interaction_progress` reaches `required_ticks`:

- The final `effects_per_tick` are applied.
- The `completion_bonus` is granted.
- Progress resets.

This structure ensures that activities such as sleep, work shifts, and training only yield benefits when completed, encouraging the agent to plan and persist. Words of estimative probability that this description reflects the implementation: high (~95 percent).

---

## 8. Positions and observation alignment

The world is spatial, not abstract.

`VectorizedTownletEnv` seeds a default layout by placing each named affordance at specific grid coordinates. Observations include both:

- the agent’s current position
- the affordance present at each coordinate

Consequently, names in `affordances.yaml` must align with the environment layout. Renaming an affordance in YAML without updating the layout will prevent the interaction from firing.

Here’s the reference layout:

```text
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

The environment also serialises and restores these positions in checkpoints. That way, training runs and eval runs agree on what "go to the Hospital" means in world coordinates, and the agent isn't punished by randomised geography unless we explicitly ask for that.

This positional stability matters because:

1. Agents can learn spatial habits (e.g., the bed is north-east of start).
2. Designers can tune economic geography. Placing `Job` far from `Hospital`, for example, models a world where low-energy agents struggle to combine work with healthcare access.

---

## 9. Validation rules

Universe as Code is data-driven but not unconstrained. Loaders enforce structural requirements to prevent inconsistent worlds.

Here’s what’s enforced when we load configs:

### bars.yaml

- There must be exactly eight bars.
- They must use indices 0 through 7, with no gaps and no duplicates.
- The names must match those indices one-to-one.
- Each bar’s range must be `[0.0, 1.0]`.
  We never allow a bar to run "hot" above 1.0 or dip below 0.0. The engines clamp.

Attempting to add a ninth bar (for example, `spiritual_alignment`) causes the loader to reject the world. Bars are part of the world ABI, and downstream models assume the canonical set and ordering.

### cascades.yaml

- Every cascade block needs a unique `name`. No duplicates.
- `category` is required, but not restricted. It's how we group cascades for execution order.
  For example:

  - `primary_to_pivotal`
  - `secondary_to_primary`
  - `secondary_to_pivotal_weak`
    These category labels show up again in `execution_order`, which is basically "run this group of cascades in this order every tick".
- Modulation entries must provide the required depletion multiplier fields (`base_multiplier`, `range`, etc) in the expected shape. Omissions or renamed fields prevent accurate health decay calculations and cause the loader to raise an error.

This is what keeps health decay, hunger crash, hygiene rot and so on all mathematically coherent instead of random if/else logic.

### affordances.yaml

- `interaction_type` has to be one of the supported literals:

  - `instant`
  - `multi_tick`
  - `continuous`
  - `dual`

  Introducing a new literal without engine support—for example `"sustained_brooding"`—will be rejected for ABI stability.

- `required_ticks`:

  - Must exist for `multi_tick` and `dual`.
  - Must not exist for `instant`.
    Providing `required_ticks` on an instant affordance is invalid because no progress counter exists to advance.

- `operating_hours` must be a two-integer list and must make temporal sense:

  - First value (open) must be 0 to 23.
  - Second value (close) must be 1 to 28.
  - Close can go past 24 to indicate wrap-around (late night into early morning).
    This contract is what lets the time-of-day mask work without bespoke per-affordance code.

This validation layer gives us confidence that if a world loads, it's at least internally legal. It also means you can ship variant packs ("austerity", "boom", custom dystopia) without having to ship new Python with them.

## 10. Equivalence, tests, and teaching packs

Replacing hardcoded behaviour with configuration introduces the risk that small YAML edits could unintentionally change survival dynamics.

Two things keep us sane:

### 1. Behavioural equivalence tests

We already have legacy logic in code for:

- base depletion
- cascade order
- hospital behaviour
- etc

We keep that code around in tests, then run the new config-driven engine with the same inputs and assert that outputs match within tight tolerance. That gives us regression alarms if a refactor accidentally changes physics.

Basically: "the new universe still feels like the old universe unless you meant to change it".

### 2. Teaching packs

On top of baseline world packs, we explicitly carry curated variants that model different economic or social conditions. These are small diffs, not forks.

Recommended packs:

- `cascades_weak.yaml`
  Halves all the cascade `strength` values. Hunger, hygiene collapse, loneliness and similar still hurt, but they're gentle. This is our "easy mode / tutorial biology".

- `cascades.yaml`
  Baseline. This is the intended default physical/physiological model for the sim.

- `cascades_strong.yaml`
  Roughly +50 percent to cascade `strength`. Hunger punishes energy and health much harder. Hygiene crashes mood brutally. This is "you are fragile, plan ahead".

These three knobs already support lessons in resilience, fragility, and the speed at which neglect becomes lethal.

Affordance packs sit alongside cascade packs and give us socioeconomic regimes. Some obvious examples:

- **Austerity** — sleep and food become cheaper while labour income drops and healthcare costs increase. Survival is possible but financially precarious.
- **Boom** — wages rise while rest and hygiene become more expensive. Agents can cover immediate needs but pay heavily to maintain them.
- **Nightlife** — social outlets operate late into the night, whereas medical services may have limited hours. Access to entertainment is easier than access to healthcare.

Because these are configuration packs, designers can explore political-economy scenarios by editing YAML rather than Python.

## 11. Emergency care path (Doctor, Hospital, Ambulance) as an intentional gameplay loop

The medical pathway in Universe as Code is not an afterthought. It's one of the core survival loops. The idea is: health is pivotal, time is scarce, and treatment has structure and cost.

Right now, an agent in trouble has three main options:

1. Go to the Doctor (during office hours)

   - Limited hours.
   - Cheaper per tick.
   - Slower.
   - Represents primary care / clinic / GP level service.

2. Go to the Hospital

   - Always open.
   - Faster restoration.
   - More expensive.
   - Represents acute care. You burn money to not die.

3. Use `HomePhoneAmbulance`

   - Available anywhere in the map, any time of day.
   - One-tick `instant` action. No long commute.
   - Extremely expensive.
   - Immediately stabilises health and (in the forward-looking relocation version) dumps you at Hospital, removing your control of the situation.

These mechanics encode policy decisions directly in YAML.

Implications for the agent:

- Preventative care is cheaper but time-gated.
- Emergency care provides immediate survival at significant financial and positional cost.
- Health outcomes are tied to economic status; insufficient resources degrade long-term survival rewards.

The design also incentivises temporal awareness: seeking treatment before closing time is strategically beneficial. Words of estimative probability that the description aligns with the current ambulance design and relocation proposal: ~80 percent (the relocation behaviour remains a planned extension).

## 12. Tick execution overview

This section summarises the sequence of operations executed each tick for every active agent.

High-level order:

1. We process the agent's chosen action:

   - movement (`UP/DOWN/LEFT/RIGHT`)
   - INTERACT
   - WAIT

   Movement and WAIT costs come from the environment (`VectorizedTownletEnv._execute_actions`), not from YAML:

   - Moving drains energy, hygiene, satiation at fixed per-step amounts (tunable via env config).
   - Waiting drains a little energy and basically nothing else.
   - Interact applies an `energy_interact_depletion` cost per use (default often 0.0, but configurable).

2. If the action was INTERACT and the location/operating_hours allow it:

   - The AffordanceEngine applies any instant costs/effects.
   - For multi_tick / dual, it also advances `interaction_progress`, applies per-tick effects, and checks for completion to apply `completion_bonus`.

3. After actions and affordance effects, we run base depletions from bars.yaml through the CascadeEngine. This is your passive decay each tick. This can be globally scaled by curriculum if we want to make "survival training worlds" harsher or softer without touching YAML.

4. We apply cascades in the order defined in `cascades.yaml` under `execution_order`. The canonical sequence is:

   - `depletions` (base drain)
   - `primary_to_pivotal`
   - `secondary_to_primary`
   - `secondary_to_pivotal_weak`

   These map to code like:

   - `apply_primary_to_pivotal_effects`
   - `apply_secondary_to_primary_effects`
   - etc

   The point is: hunger hits energy/health before hygiene hits mood, and so on. Order matters. This is how the system models spirals.

5. We clamp all bars to [0.0, 1.0].

6. We evaluate terminal conditions:

   - If any pivotal bar hits zero (e.g. `energy <= 0`, `health <= 0`), that agent is declared "dead" (or "retired" if you're using a softer narrative layer).
   - Dead/retired agents are then masked: they no longer get valid actions. From that point, they're effectively frozen in time for that episode, and the RL loop can treat that as end-of-life reward accounting.

So from an RL perspective, "I stayed alive another tick" is literally "the engine didn't mask me out". Longevity reward is clean to compute.

## 13. Config packs and world folders

Operationally, a world is a directory of YAML files.

At minimum, that folder needs:

```text
configs/test/
  bars.yaml
  cascades.yaml
  affordances.yaml
```

You point the environment at this directory and it boots that reality.

- `bars.yaml` says which meters exist, how they deplete, and how you can literally die.
- `cascades.yaml` describes cross-meter dynamics.
- `affordances.yaml` defines the available actions.

Additional capabilities (e.g., social signalling, scripted events, NPC cues) can be layered in via supplementary files. A planned `cues.yaml`, for example, would describe observable signals, supporting perception and pretraining modules as well as control.

The broader objective is to provide a single executable source of truth shared by content designers, economists, narrative designers, and AI researchers.

## 14. Walkthrough: one crisis tick with ambulance

Let's run a single tick to show how all of this fits together.

Scenario:

- Time: 02:00 (2 am)
- Agent is at home
- health = 0.18
- energy = 0.40
- money = 0.60
- Doctor is closed
- Hospital is open but it's a walk away
- Agent is in trouble

Tick timeline:

1. The agent chooses INTERACT on `HomePhoneAmbulance`.

   - This affordance is `interaction_type: "instant"`.
   - It is open 24/7.
   - The environment checks: are we standing on the tile that offers `HomePhoneAmbulance`, and is the current hour within `[0, 24]`?
     Yes. So the action is valid.

2. AffordanceEngine fires the instant interaction:

   - Deducts the cost:

     - money goes from 0.60 to 0.10 (cost 0.50, read as 50 dollars).
   - Applies effects:

     - health jumps from 0.18 to 0.48.
     - energy drops from 0.40 to 0.35 (shock/fatigue penalty).

   At this point the agent is still alive. The immediate emergency is contained.

   If we’ve implemented the relocation extension:

   - The interaction also emits a relocation event like `("agent0", "Hospital")`.
   - The environment consumes that event and teleports the agent to the Hospital coordinates before cascade resolution.
   - Narrative read: "you blacked out in the lounge and woke up in hospital".

3. We now run the normal tick housekeeping:

   - Base depletions tick all bars according to `base_depletion`.
   - Cascades fire in the configured `execution_order`.
     For example: if satiation is very low, it will hammer energy and health again, so if you're starving on top of being injured, you're still in danger even after ambulance.
   - All meters are clamped [0.0, 1.0].

4. We check terminal conditions:

   - If health or energy fell to 0.0 or below after cascades, you're done.
   - Otherwise you continue into the next tick at 02:01, either at home (no relocation version) or in Hospital (relocation version).

Game design read:

- You stayed alive, but you torched your money.
- You might have been moved somewhere where you can now get better care, but you also lost positional freedom.
- You are deep in a liability spiral you will have to dig out of by working, which will cost mood and energy.

RL read:

- The agent sees ambulance as a last-ditch survival affordance that is extremely costly.
- The agent gets a strong signal that avoiding this state is better than repeatedly invoking it.
- Long-term policy should learn to manage health proactively, keep money in reserve, and respect clinic hours.

That is exactly the behaviour we want to see emerge.
