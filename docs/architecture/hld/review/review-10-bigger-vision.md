---
document_type: Overview + Design Rationale
status: Draft
version: 2.5
---

## SECTION 10: THE BIGGER VISION

### 10.1 The Core Abstraction

The **Townlet Framework** is not about towns. It's about **bars, cascades, and affordances**.

**The minimal viable simulation**:

```yaml
# Any world can be expressed as:

bars:
  - state variables that change over time

cascades:
  - how neglecting one bar affects others

affordances:
  - actions that change bars
  - with costs, effects, and constraints
```

**Why this is general**:

- **Bars** = any continuous state variable
- **Cascades** = any coupling between variables
- **Affordances** = any controllable action

**Townlet Town** (the reference implementation) just happens to be one instantiation:

- Bars = survival meters (energy, health, money)
- Cascades = biological needs (hunger → health decay)
- Affordances = locations with services (Bed, Job, Hospital)

But this works for *any domain* where:

1. State evolves over time
2. Variables are coupled
3. Actions have effects and costs

---

### 10.2 Example Application 1: Economic Simulation

**Status**: Conceptual (not implemented)

**Domain**: Monetary policy and central banking

**Note**: The following YAML examples demonstrate how the Townlet Framework could be extended to model economic systems. These are illustrative schemas showing the framework's generalizability.

#### Bar Configuration

```yaml
# bars_economy.yaml
bars:
  - name: "gdp"
    description: "Gross Domestic Product"
    initial: 1.0
    base_depletion: 0.0  # modulated by policies

  - name: "inflation"
    description: "Price level change rate"
    initial: 0.02  # 2% baseline
    base_depletion: 0.0

  - name: "unemployment"
    description: "Labor market slack"
    initial: 0.05  # 5% baseline
    base_depletion: 0.0

  - name: "debt"
    description: "Government debt as % of GDP"
    initial: 0.60  # 60% debt/GDP ratio
    base_depletion: 0.01  # grows slightly (deficit spending)

  - name: "consumer_confidence"
    description: "Economic sentiment"
    initial: 0.70
    base_depletion: -0.002  # slowly improves by default

  - name: "asset_prices"
    description: "Stock/housing market levels"
    initial: 1.0
    base_depletion: 0.0

  - name: "exchange_rate"
    description: "Currency value (higher = stronger)"
    initial: 1.0
    base_depletion: 0.0

  - name: "credit_availability"
    description: "Ease of borrowing"
    initial: 0.70
    base_depletion: 0.0
```

#### Cascade Configuration

```yaml
# cascades_economy.yaml
cascades:
  - name: "high_inflation_hurts_confidence"
    source: "inflation"
    target: "consumer_confidence"
    threshold: 0.04  # >4% inflation
    strength: 0.020  # -2% confidence per tick

  - name: "low_confidence_reduces_gdp"
    source: "consumer_confidence"
    target: "gdp"
    threshold: 0.50  # <50% confidence
    strength: 0.010  # -1% GDP per tick

  - name: "high_unemployment_lowers_confidence"
    source: "unemployment"
    target: "consumer_confidence"
    threshold: 0.08  # >8% unemployment
    strength: 0.015

  - name: "recession_increases_unemployment"
    source: "gdp"
    target: "unemployment"
    threshold: 0.95  # GDP <95% of baseline
    strength: 0.005
    operator: "<"

  - name: "high_debt_limits_spending"
    source: "debt"
    target: "gdp"
    threshold: 0.90  # >90% debt/GDP
    strength: 0.008

  - name: "loose_credit_inflates_assets"
    source: "credit_availability"
    target: "asset_prices"
    threshold: 0.80  # very loose credit
    strength: 0.010

  - name: "asset_bubble_pops"
    source: "asset_prices"
    target: "consumer_confidence"
    threshold: 1.50  # >150% of baseline
    strength: 0.030  # severe confidence shock when bubble pops
    operator: ">"
```

#### Affordance Configuration

```yaml
# affordances_economy.yaml
affordances:
  - id: "raise_interest_rates"
    description: "Federal Reserve increases rates"
    interaction_type: "instant"
    costs: []  # policy actions are "free" (no money cost)
    effects:
      - { meter: "inflation", amount: -0.02 }      # cool inflation
      - { meter: "credit_availability", amount: -0.10 }  # tighten credit
      - { meter: "unemployment", amount: 0.01 }    # slight job loss
      - { meter: "exchange_rate", amount: 0.05 }   # strengthen currency
    cooldown: 3  # can't raise rates every tick

  - id: "lower_interest_rates"
    description: "Federal Reserve decreases rates"
    interaction_type: "instant"
    effects:
      - { meter: "inflation", amount: 0.01 }       # risk inflation
      - { meter: "credit_availability", amount: 0.10 }  # loosen credit
      - { meter: "gdp", amount: 0.02 }             # stimulate economy
      - { meter: "unemployment", amount: -0.01 }   # more jobs
      - { meter: "exchange_rate", amount: -0.05 }  # weaken currency
    cooldown: 3

  - id: "government_spending"
    description: "Fiscal stimulus"
    interaction_type: "instant"
    costs:
      - { meter: "debt", amount: 0.05 }  # increases debt
    effects:
      - { meter: "gdp", amount: 0.03 }
      - { meter: "unemployment", amount: -0.02 }
      - { meter: "consumer_confidence", amount: 0.02 }

  - id: "austerity"
    description: "Reduce government spending"
    interaction_type: "instant"
    effects:
      - { meter: "debt", amount: -0.03 }  # reduce debt
      - { meter: "gdp", amount: -0.02 }   # contractionary
      - { meter: "unemployment", amount: 0.02 }
      - { meter: "consumer_confidence", amount: -0.01 }

  - id: "quantitative_easing"
    description: "Central bank buys assets"
    interaction_type: "multi_tick"
    required_ticks: 10  # long-term program
    effects_per_tick:
      - { meter: "asset_prices", amount: 0.03 }
      - { meter: "credit_availability", amount: 0.02 }
      - { meter: "inflation", amount: 0.005 }

  - id: "wait"
    description: "Do nothing (let market stabilize)"
    interaction_type: "instant"
    effects: []
```

#### Research Questions

**Can RL learn monetary policy?**

- Train agent on economy environment
- Reward = `r = gdp * (1 - unemployment) * (1 / max(inflation, 0.01))`
- Test: Does agent learn to stabilize cycles?

**Does it rediscover the Taylor rule?**

- [Taylor rule: r = natural_rate + 1.5*(inflation - target) + 0.5*(output_gap)]
- After training, analyze policy: do rate decisions correlate with inflation + unemployment?

**Can it handle shocks?**

- Introduce external shocks (e.g., sudden inflation spike from energy crisis)
- Measure: recovery time, stability

**Human observer test**:

- ✅ "Can a central banker know GDP, inflation, unemployment?" YES (public data)
- ✅ "Can they control interest rates?" YES (policy lever)
- ✅ "Do actions have delayed effects?" YES (cascades model this)

---

### 10.3 Example Application 2: Ecosystem Simulation

**Status**: Conceptual (not implemented)

**Domain**: Predator-prey dynamics and trophic cascades

**Note**: The following YAML examples demonstrate how the Townlet Framework could be extended to model ecological systems. These are illustrative schemas showing the framework's generalizability.

#### Bar Configuration

```yaml
# bars_ecosystem.yaml
bars:
  # Species populations (normalized [0, 1] where 1.0 = carrying capacity)
  - name: "vegetation"
    initial: 0.80
    base_depletion: -0.01  # grows naturally

  - name: "herbivore_pop"
    initial: 0.50
    base_depletion: 0.005  # slight natural decline (hunger)

  - name: "predator_pop"
    initial: 0.30
    base_depletion: 0.010  # higher natural decline

  # Environmental factors
  - name: "water_availability"
    initial: 1.0
    base_depletion: 0.002  # droughts

  - name: "soil_quality"
    initial: 0.80
    base_depletion: 0.001  # degradation

  # Agent-specific (if modeling individual animals)
  - name: "energy"  # for the agent (e.g., a predator)
    initial: 0.70
    base_depletion: 0.010

  - name: "health"
    initial: 1.0
    base_depletion: 0.0
```

#### Cascade Configuration

```yaml
# cascades_ecosystem.yaml
cascades:
  - name: "herbivores_eat_vegetation"
    source: "herbivore_pop"
    target: "vegetation"
    threshold: 0.40  # if herbivores > 40% of capacity
    strength: 0.020  # they consume vegetation

  - name: "overgrazing_damages_soil"
    source: "vegetation"
    target: "soil_quality"
    threshold: 0.30  # if vegetation < 30%
    strength: 0.005
    operator: "<"

  - name: "poor_soil_limits_vegetation"
    source: "soil_quality"
    target: "vegetation"
    threshold: 0.50
    strength: 0.010
    operator: "<"

  - name: "predators_eat_herbivores"
    source: "predator_pop"
    target: "herbivore_pop"
    threshold: 0.25
    strength: 0.015

  - name: "low_herbivores_starve_predators"
    source: "herbivore_pop"
    target: "predator_pop"
    threshold: 0.20
    strength: 0.020
    operator: "<"

  - name: "drought_kills_vegetation"
    source: "water_availability"
    target: "vegetation"
    threshold: 0.50
    strength: 0.015
    operator: "<"
```

#### Affordance Configuration

```yaml
# affordances_ecosystem.yaml
affordances:
  # For individual predator agent
  - id: "hunt"
    description: "Hunt herbivores"
    interaction_type: "multi_tick"
    required_ticks: 3
    costs:
      - { meter: "energy", amount: 0.05 }  # per tick
    effects:
      - { meter: "energy", amount: 0.30 }  # gain from kill (final tick)
      - { meter: "herbivore_pop", amount: -0.02 }  # reduce prey population
    success_probability: 0.60  # hunts can fail

  - id: "graze"  # for herbivore agent
    description: "Eat vegetation"
    interaction_type: "instant"
    costs:
      - { meter: "vegetation", amount: -0.03 }
    effects:
      - { meter: "energy", amount: 0.20 }

  - id: "rest"
    description: "Rest and recover"
    interaction_type: "multi_tick"
    required_ticks: 2
    effects_per_tick:
      - { meter: "energy", amount: 0.10 }
      - { meter: "health", amount: 0.05 }

  - id: "migrate"
    description: "Move to better territory"
    interaction_type: "instant"
    costs:
      - { meter: "energy", amount: 0.15 }
    effects:
      - { meter: "water_availability", amount: 0.10 }  # find better water
      - { meter: "vegetation", amount: 0.05 }  # find better grazing
```

#### Research Questions

**Do predator-prey cycles emerge?**

- Train agents in ecosystem environment
- Reward = `r = energy * health` (individual survival)
- Measure: population oscillations (Lotka-Volterra dynamics)

**Does overgrazing lead to collapse?**

- If herbivores overgraze (vegetation < 0.2), does ecosystem crash?
- Can predators "learn" to regulate herbivore population?

**Group selection**:

- If some predators "restrain" from overhunting, do their populations persist longer?
- Does altruism emerge?

**Human observer test**:

- ✅ "Can a predator know its own energy?" YES (internal state)
- ✅ "Can it see prey nearby?" YES (partial observability)
- ✅ "Does hunting affect global prey population?" YES (affordance effects)

---

### 10.4 Example Application 3: Mental Health Treatment

**Status**: Conceptual (not implemented)

**Domain**: Modeling anxiety/depression treatment strategies

**Disclaimer**: This is an illustrative example for research purposes, not a clinical tool.

**Note**: The following YAML examples demonstrate how the Townlet Framework could be extended to model treatment strategies. These are illustrative schemas showing the framework's generalizability.

#### Bar Configuration

```yaml
# bars_mentalhealth.yaml
bars:
  - name: "anxiety"
    initial: 0.60
    base_depletion: 0.005  # increases over time

  - name: "depression"
    initial: 0.50
    base_depletion: 0.003

  - name: "sleep_quality"
    initial: 0.60
    base_depletion: -0.002  # improves slightly naturally

  - name: "medication_level"
    initial: 0.0
    base_depletion: 0.01  # medication wears off

  - name: "therapy_progress"
    initial: 0.0
    base_depletion: 0.0  # cumulative, doesn't decay

  - name: "social_support"
    initial: 0.50
    base_depletion: 0.002  # relationships require maintenance

  - name: "physical_health"
    initial: 0.80
    base_depletion: 0.001

  - name: "daily_functioning"
    initial: 0.70
    base_depletion: 0.0  # modulated by other bars
```

#### Cascade Configuration

```yaml
# cascades_mentalhealth.yaml
cascades:
  - name: "anxiety_disrupts_sleep"
    source: "anxiety"
    target: "sleep_quality"
    threshold: 0.60
    strength: 0.015

  - name: "poor_sleep_worsens_depression"
    source: "sleep_quality"
    target: "depression"
    threshold: 0.40
    strength: 0.010
    operator: "<"

  - name: "depression_reduces_social"
    source: "depression"
    target: "social_support"
    threshold: 0.50
    strength: 0.012

  - name: "low_social_worsens_depression"
    source: "social_support"
    target: "depression"
    threshold: 0.30
    strength: 0.015
    operator: "<"

  - name: "anxiety_impairs_function"
    source: "anxiety"
    target: "daily_functioning"
    threshold: 0.70
    strength: 0.020

  - name: "exercise_improves_mood"
    source: "physical_health"
    target: "depression"
    threshold: 0.70
    strength: -0.008  # negative = reduces depression
    operator: ">"
```

#### Affordance Configuration

```yaml
# affordances_mentalhealth.yaml
affordances:
  - id: "therapy_session"
    description: "Cognitive behavioral therapy"
    interaction_type: "multi_tick"
    required_ticks: 4  # 1-hour session
    costs:
      - { meter: "daily_functioning", amount: 0.05 }  # time commitment
    effects:
      - { meter: "therapy_progress", amount: 0.10 }
      - { meter: "anxiety", amount: -0.05 }
      - { meter: "depression", amount: -0.03 }
    cooldown: 24  # weekly sessions

  - id: "medication"
    description: "Take prescribed medication"
    interaction_type: "instant"
    effects:
      - { meter: "medication_level", amount: 0.30 }
      - { meter: "anxiety", amount: -0.10 }  # fast relief
    side_effects:
      - { meter: "sleep_quality", amount: -0.05 }  # some meds affect sleep

  - id: "exercise"
    description: "Physical activity"
    interaction_type: "multi_tick"
    required_ticks: 2
    costs:
      - { meter: "daily_functioning", amount: 0.08 }
    effects:
      - { meter: "physical_health", amount: 0.10 }
      - { meter: "depression", amount: -0.05 }
      - { meter: "sleep_quality", amount: 0.05 }

  - id: "social_activity"
    description: "Spend time with friends"
    interaction_type: "multi_tick"
    required_ticks: 3
    costs:
      - { meter: "anxiety", amount: 0.03 }  # social anxiety
    effects:
      - { meter: "social_support", amount: 0.15 }
      - { meter: "depression", amount: -0.08 }

  - id: "rest"
    description: "Rest and recover"
    interaction_type: "instant"
    effects:
      - { meter: "daily_functioning", amount: 0.05 }
      - { meter: "sleep_quality", amount: 0.03 }
```

#### Research Questions

**Can RL discover treatment strategies?**

- Reward = `r = daily_functioning * (1 - anxiety) * (1 - depression)`
- Does agent learn: therapy + medication + exercise > medication alone?

**Optimal treatment sequencing**:

- Does agent learn to stabilize acute symptoms (medication) before building long-term strategies (therapy)?

**Side effect management**:

- Can agent balance benefit (anxiety reduction) vs cost (sleep disruption from meds)?

**Relapse prevention**:

- After recovery (bars high), does agent continue maintenance (exercise, social)?

**Human observer test**:

- ✅ "Can someone know their anxiety/depression level?" YES (self-awareness)
- ✅ "Can they choose therapy or medication?" YES (treatment decisions)
- ✅ "Do symptoms interact?" YES (cascades model this)

---

### 10.5 Example Application 4: Supply Chain Management

**Status**: Conceptual (not implemented)

**Domain**: Inventory control and logistics optimization

**Note**: The following YAML examples demonstrate how the Townlet Framework could be extended to model supply chain systems. These are illustrative schemas showing the framework's generalizability.

#### Bar Configuration

```yaml
# bars_supply_chain.yaml
bars:
  - name: "inventory_level"
    initial: 0.60
    base_depletion: 0.02  # daily sales

  - name: "customer_satisfaction"
    initial: 0.80
    base_depletion: 0.0

  - name: "cash_reserves"
    initial: 1.0
    base_depletion: 0.01  # operating costs

  - name: "supplier_reliability"
    initial: 0.70
    base_depletion: 0.001

  - name: "shipping_capacity"
    initial: 0.80
    base_depletion: 0.0

  - name: "warehouse_capacity"
    initial: 0.90
    base_depletion: 0.0

  - name: "demand_forecast"
    initial: 0.50  # medium demand
    base_depletion: 0.0  # fluctuates
```

#### Cascades

```yaml
cascades:
  - name: "stockout_hurts_satisfaction"
    source: "inventory_level"
    target: "customer_satisfaction"
    threshold: 0.20  # low stock
    strength: 0.030
    operator: "<"

  - name: "low_satisfaction_reduces_demand"
    source: "customer_satisfaction"
    target: "demand_forecast"
    threshold: 0.50
    strength: 0.010
    operator: "<"

  - name: "overstock_ties_up_cash"
    source: "inventory_level"
    target: "cash_reserves"
    threshold: 0.80  # excess inventory
    strength: 0.015
    operator: ">"
```

#### Affordances

```yaml
affordances:
  - id: "order_inventory"
    description: "Order from supplier"
    costs:
      - { meter: "cash_reserves", amount: 0.30 }
    effects:
      - { meter: "inventory_level", amount: 0.40 }  # after lead time
    lead_time: 5  # 5 ticks delay

  - id: "expedited_shipping"
    costs:
      - { meter: "cash_reserves", amount: 0.50 }  # expensive
    effects:
      - { meter: "inventory_level", amount: 0.40 }
    lead_time: 1  # fast delivery

  - id: "promote_sale"
    costs:
      - { meter: "cash_reserves", amount: 0.10 }
    effects:
      - { meter: "inventory_level", amount: -0.20 }  # move inventory
      - { meter: "customer_satisfaction", amount: 0.10 }
```

**Research questions**: Can RL learn (s,S) inventory policies? Does it discover just-in-time strategies?

---

### 10.6 Platform Value Proposition by Audience

#### For Researchers

**Value**: Hypothesis testing without Python

**Workflow**:

1. Formulate hypothesis: "Does X affect Y?"
2. Design world: encode X and Y as bars, add cascade
3. Launch experiment: `townlet train --config hypothesis_X/`
4. Analyze results: compare treatment vs control

**Example**:

- Hypothesis: "High initial wealth reduces work ethic"
- World A: `initial_money: 0.20`
- World B: `initial_money: 0.80`
- Measure: Hours worked over lifetime
- Publish: Config diff + results + provenance

**Why this matters**: Configuration-driven experiments are:

- Reproducible (exact config archived)
- Auditable (cognitive hash proves which brain)
- Comparable (structured diffs across experiments)

#### For Educators

**Value**: Students experiment without coding

**Pedagogy**:

1. Provide baseline world config
2. Students modify parameters (e.g., "make food twice as expensive")
3. Run experiment, observe behavioral changes
4. Write report explaining why behavior changed

**Example assignment**:

```
Assignment: The Scarcity Experiment

1. Run baseline world (50 episodes)
2. Create "scarcity world": double Fridge cost
3. Compare:
   - Average lifetime money at retirement
   - Frequency of starvation deaths
   - Time spent at Job vs Recreation
4. Explain: Why did agent behavior change?
```

**Why this matters**: Students learn:

- Systems thinking (cascades create feedback loops)
- Policy design (affordance costs affect behavior)
- Emergent phenomena (policies not explicitly programmed)

#### For Policy Analysts

**Value**: Auditable simulations for governance

**Use case**: Testing policy interventions

**Workflow**:

1. Model current system (e.g., welfare policy)
2. Propose intervention (e.g., universal basic income)
3. Encode as affordance change (e.g., passive income bar)
4. Run simulation, measure outcomes
5. Present to stakeholders with provenance

**Example**:

```yaml
# Current welfare system
affordances:
  - id: "welfare_office"
    eligibility: { money < 0.10 }  # means-tested
    effects: { money: +0.05 }

# Proposed UBI
affordances:
  - id: "ubi_payment"
    eligibility: { always }  # universal
    effects: { money: +0.03 }
    operating_hours: [0, 24]  # automatic
```

**Compare**:

- Work incentives (does UBI reduce labor?)
- Poverty rates (does UBI prevent starvation?)
- Cost (total money distributed)

**Why this matters**: Policy debates become empirical

- Not "I think UBI will work"
- But "When we ran the simulation, we observed X"
- With provenance: "Config hash ABC123, cognitive hash DEF456"

---

### 10.7 What Makes This Generalizable

**Three properties enable cross-domain transfer**:

#### 1. Declarative Physics

```yaml
# Physics are data, not code
# This means:
- Non-programmers can design worlds
- Worlds are diff-able (structured comparison)
- Worlds are version-controlled (Git tracks changes)
- Worlds are reproducible (config snapshot frozen)
```

**Alternative (bad)**: Physics embedded in environment.step()

- Requires Python expertise
- Difficult to compare across experiments
- Hard to audit ("what exactly changed?")
- Brittle to refactoring

#### 2. Minimal Assumptions

**The Townlet Framework only assumes**:

- State evolves over time (bars with depletion)
- Variables couple (cascades)
- Actions have effects (affordances)

**The Townlet Framework does NOT assume**:

- Grid worlds (could be continuous space)
- Agents (could model single controller)
- Survival (could maximize profit, not survival)
- Spatial navigation (could be pure resource management)

This means you can model:

- Economic systems (no grid, pure resource flows)
- Ecosystems (spatial but not grid-based)
- Mental health (no space, just internal dynamics)
- Supply chains (network topology, not grid)

#### 3. Human Observer Principle

The same observability constraints that make Townlet Town realistic for agents make the framework realistic for other domains.

**Economic policy**:

- ✅ Can a central banker observe GDP, inflation? YES
- ❌ Can they know exact future GDP? NO
- ✅ Can they control interest rates? YES

**Ecosystem management**:

- ✅ Can a ranger observe population counts? YES
- ❌ Can they know exact future weather? NO
- ✅ Can they introduce species? YES

**Mental health treatment**:

- ✅ Can a patient know their anxiety level? YES
- ❌ Can they instantly cure depression? NO
- ✅ Can they choose therapy? YES

The human observer test generalizes across domains.

---

### 10.8 Limitations & Non-Applications

**The Townlet Framework is NOT suitable for**:

**1. Fast-reaction games**

- Fighting games, racing, real-time shooters
- Why: Townlet is turn-based, focuses on long-horizon planning
- Use instead: Game-specific simulators (e.g., OpenAI Gym Atari)

**2. Perfect-information games**

- Chess, Go
- Why: Partial observability and cascades add no value here
- Use instead: AlphaZero-style tree search

**3. Pure text domains**

- Language modeling, dialogue
- Why: Bars/cascades don't map to linguistic structure
- Use instead: Transformer-based LLMs

**4. Continuous control**

- Robotics, drones, manipulation
- Why: Discrete affordances don't capture continuous dynamics well
- Use instead: MuJoCo, PyBullet, Isaac Gym

**5. Purely spatial problems**

- Pathfinding, maze solving
- Why: The framework's value is in temporal reasoning and resource allocation
- Use instead: Grid-world simulators without survival mechanics

**The sweet spot**:

- Long-horizon planning (100+ steps)
- Multi-objective optimization (balance multiple meters)
- Resource allocation under scarcity
- Coupled dynamics (cascades matter)
- Partial observability (memory required)
- Social coordination (multi-agent with communication)

---

### 10.9 Future Extensions

**What could be added to make the Townlet Framework more general**:

#### Extension 1: Continuous Space

**Status**: Future schema extension

```yaml
# Instead of grid_size: 8
space:
  type: "continuous"
  bounds: [[0, 10], [0, 10]]  # 2D continuous

affordances:
  - id: "job"
    position: [3.5, 7.2]  # continuous coordinates
    interaction_radius: 0.5  # fuzzy boundary
```

**Use case**: Model ecosystems with actual spatial movement, not grid-constrained

#### Extension 2: Relational Networks

**Status**: Future schema extension

```yaml
# Instead of affordances at positions
entities:
  - id: "company_a"
    relations:
      - { target: "company_b", type: "supplier" }
      - { target: "company_c", type: "competitor" }
```

**Use case**: Supply chains, social networks, financial systems

#### Extension 3: Stochastic Events

**Status**: Future schema extension

```yaml
# Random shocks to bars
events:
  - id: "recession"
    probability: 0.02  # 2% per 100 ticks
    effects:
      - { meter: "gdp", amount: -0.20 }
      - { meter: "unemployment", amount: 0.05 }
```

**Use case**: Test robustness to black swan events

#### Extension 4: Partial Affordance Observability

**Status**: Future schema extension

```yaml
affordances:
  - id: "hidden_treasure"
    visible: false  # agent must discover via exploration
    discovery_condition: { position_within: 1.0 }
```

**Use case**: Exploration-driven domains (R&D, scientific discovery)

---

### 10.10 The Long-Term Vision

**The Townlet Framework becomes a platform for configuration-driven science**:

**Year 1**: Townlet Town (survival simulation)

- L0-L8 curriculum proven
- Population genetics validated
- Emergent communication demonstrated

**Year 2**: Economics (monetary policy)

- RL learns basic Taylor rule
- Validated against macro models
- Published in econ/CS conferences

**Year 3**: Ecosystems (conservation)

- Predator-prey dynamics emerge
- Used by ecology researchers
- Published in Nature/Science

**Year 5**: General adoption

- 100+ research papers using the Townlet Framework
- Standard benchmark (like Atari, MuJoCo)
- Taught in RL courses
- Used by policymakers for scenario planning

**The ultimate success criterion**:
> "Can a domain expert with zero Python knowledge create a world using the Townlet Framework and run meaningful experiments?"

If yes, we've built a truly general platform.

---
