---
title: "SECTION 9: POPULATION GENETICS & FAMILIES"
type: Component Spec + Design Rationale
status: Draft
version: 2.5
---

## SECTION 9: POPULATION GENETICS & FAMILIES

### 9.1 Official Rules: Meritocratic Churn

The baseline population genetics system implements **meritocratic churn with one-child families**.

**Framework vs Instance Configuration**:
- Population genetics is a **framework capability** (infrastructure for breeding, inheritance, family lifecycle)
- Breeding modes (meritocratic, dynasty, arranged) are **instance configurations** (YAML-controlled experiments)
- This section describes the baseline configuration and alternative modes as examples

**Core principles**:

1. **One-child policy**: Families can have exactly one child at a time
2. **Breeding locked while raising**: Cannot breed again until child leaves
3. **High-performer selection**: Only successful agents breed
4. **Constant population**: Deaths are replaced by culling worst performers
5. **Natural churn**: Families dissolve and reform as children mature

**Configuration**:

```yaml
# population_genetics.yaml (baseline mode)

population:
  initial_size: 100
  max_size: 100  # constant population
  replacement_policy: "cull_worst"  # when someone dies, remove worst performer

family_formation:
  mode: "meritocratic"
  eligibility:
    min_age: 0.2  # must be 20% through life
    performance_threshold: 0.6  # top 60% of population
    family_state: ["NO_FAMILY", "DIVORCED"]

  pairing_strategy: "complementary_dna"  # match compatible personalities

breeding:
  one_child_policy: true
  child_leaves_at_age: 0.3  # child matures at 30% through life

child_initialization:
  weight_init: "crossover"  # average parents' neural weights
  training_mode: "finetune"  # continue learning from inherited weights

  dna_inheritance:
    crossover_rate: 0.5  # 50% genes from each parent
    mutation_rate: 0.05  # 5% chance per gene

death_and_replacement:
  natural_death_age: 1.0  # retirement
  early_death: ["energy <= 0", "health <= 0"]

  on_death:
    - remove_from_population
    - identify_worst_performer  # by lifetime reward
    - cull_worst_to_maintain_population_cap
```

**Population dynamics**:

```python
# At initialization (tick 0)
population = create_random_agents(100)
# All agents start as NO_FAMILY, random weights, random DNA

# At tick 10,000 (agents have learned survival)
evaluate_all_agents()  # compute lifetime rewards
high_performers = top_60_percent(population)

# Form initial families
for _ in range(20):  # form 20 families (40 agents paired)
    parent_a, parent_b = select_complementary_pair(high_performers)
    create_family(parent_a, parent_b)
    # Both parents: NO_FAMILY → MARRIED_NO_CHILD

# Families breed
for family in families:
    if family.state == MARRIED_NO_CHILD:
        child = spawn_child(family.parent_a, family.parent_b)
        family.state = MARRIED_WITH_CHILD
        population.append(child)  # population = 100 → 120

# 20 worst performers are culled to maintain cap
worst_20 = bottom_20_by_lifetime_reward(population)
for agent in worst_20:
    remove_from_population(agent)
# population = 120 → 100 (back to cap)

# At tick 20,000 (children are maturing)
for family in families:
    if family.child.age > 0.3:
        # Child leaves home
        family.child.family_id = None
        family.child.family_state = NO_FAMILY

        # Parents can breed again
        family.parent_a.family_state = MARRIED_NO_CHILD
        family.parent_b.family_state = MARRIED_NO_CHILD

# If parent dies while child is young
parent_a.health = 0  # death
family.state = SINGLE_PARENT
# Child continues with one parent
# When child leaves, remaining parent → DIVORCED, can remarry
```

**Expected equilibrium** (after 50k ticks):

- 70-80 agents in families (35-40 families)
- 20-30 solo agents (divorced, orphaned, or never married)
- Constant churn: ~5% births/deaths per 1000 ticks
- High performers dominate breeding pool
- Low performers get culled before reproducing

**Why meritocratic as baseline**:

- Tests selection pressure (do good strategies propagate?)
- Prevents population collapse (only competent agents breed)
- Creates competition (agents must perform to pass on genes)
- Reflects real-world success → reproduction correlation

---

### 9.2 The Complete Family Lifecycle (Reference)

See **Section 8.2** for full state machine specification.

**Quick reference**:

```
States:
NO_FAMILY → [meet partner] → MARRIED_NO_CHILD
MARRIED_NO_CHILD → [breed] → MARRIED_WITH_CHILD
MARRIED_WITH_CHILD → [child leaves] → MARRIED_NO_CHILD
                   → [one parent dies] → SINGLE_PARENT
                   → [child dies] → MARRIED_NO_CHILD
SINGLE_PARENT → [child leaves] → DIVORCED
              → [child dies] → DIVORCED
DIVORCED → [remarry] → MARRIED_NO_CHILD
ORPHAN → [reach adulthood] → NO_FAMILY
```

**Key transitions for population experiments**:

- Child maturity triggers eligibility reset (parents can breed again)
- Death of parent creates single-parent families (not immediate dissolution)
- Orphans can survive and become independent adults
- Remarriage allows multi-generation dynasties

---

### 9.3 Alternative Inheritance Systems

The meritocratic baseline is just one configuration. Alternative modes enable different research questions.

#### Mode 1: Dynasty (Primogeniture + Inheritance)

**Concept**: Families persist across generations, child inherits parents' resources

**Configuration**:

```yaml
# population_genetics_dynasty.yaml

family_formation:
  mode: "dynasty"

breeding:
  one_child_policy: false  # ← KEY CHANGE: multiple children allowed
  primogeniture: true      # first child is heir

inheritance:
  on_parent_death:
    transfer_wealth: true   # child gets parents' money
    transfer_affordances: false  # (for future: private property)
    keep_family_id: true    # dynasty persists across generations

  heir_selection: "oldest"  # or "most_successful"

child_initialization:
  weight_init: "clone_best"  # ← inherit best parent's full policy
  training_mode: "frozen"    # ← no learning, pure genetic selection

  dna_inheritance:
    crossover_rate: 0.7   # more parental influence
    mutation_rate: 0.02   # less mutation (preserve dynasty traits)
```

**Behavioral differences from baseline**:

```python
# Baseline (meritocratic): Child leaves, becomes independent
child.age = 0.3
child.family_id = None
child.money = 0.50  # starts with initial endowment

# Dynasty mode: Child inherits
parent_a.health = 0  # parent dies
child.family_id = parent_a.family_id  # keeps dynasty name
child.money += parent_a.money  # inherits wealth
# Child becomes head of dynasty, can have own children
```

**Expected population dynamics**:

- 5-10 dynasty lineages emerge (named families)
- Wealth concentration (rich dynasties stay rich)
- Behavioral dynasties (coordinated families develop shared protocols)
- "Aristocracy" vs "peasants" stratification

**Research questions**:

1. **Wealth inequality**: How quickly does wealth concentrate? Does Gini coefficient increase over time?
2. **Coordination advantage**: Do dynasties with shared communication protocols outperform solo agents?
3. **Dynasty collapse**: What causes a dynasty to fail (bad heir, accumulated bad mutations)?
4. **Speciation**: Do different dynasties evolve distinct survival strategies?

**Hypothesis**: Dynasty mode produces more coordinated families but higher inequality.

---

#### Mode 2: Polygamy (Multiple Simultaneous Families)

**Concept**: Agents can belong to multiple family units simultaneously

**Configuration**:

```yaml
# population_genetics_polygamy.yaml

family_formation:
  mode: "polygamous"
  max_families_per_agent: 2  # can be in 2 families at once

breeding:
  one_child_policy: true  # per family
  # Agent can have 2 children (one per family) simultaneously

child_initialization:
  weight_init: "crossover"
  training_mode: "full"  # children learn independently
```

**Behavioral differences**:

```python
# Agent A is in family_1 with Agent B (child C)
# Agent A is also in family_2 with Agent D (child E)

agent_a.families = [family_1, family_2]
agent_a.family_comm_channels = {
    family_1: [signal_from_b, signal_from_c],
    family_2: [signal_from_d, signal_from_e],
}

# Agent A must manage two separate communication protocols
# Children C and E are half-siblings (share parent A, different co-parents)
```

**Expected population dynamics**:

- Complex kinship networks (overlapping families)
- More communication channels per agent
- Potential for information brokerage (Agent A can coordinate across families)

**Research questions**:

1. **Communication complexity**: Can agents maintain distinct protocols per family?
2. **Strategic alliances**: Do high performers form multiple families to maximize offspring?
3. **Network effects**: Does family overlap create more coordinated populations?
4. **Cognitive load**: Does managing multiple families hurt individual performance?

**Hypothesis**: Polygamy increases coordination complexity; performance depends on Module C: Social Model capacity.

---

#### Mode 3: Arranged Marriage (DNA Complementarity)

**Concept**: Pairing prioritizes genetic diversity, not just performance

**Configuration**:

```yaml
# population_genetics_arranged.yaml

family_formation:
  mode: "arranged"

  pairing_strategy: "maximize_diversity"
  diversity_metric:
    - personality_distance  # pair high-greed with low-greed
    - skill_complementarity  # pair good-at-X with good-at-Y

  eligibility:
    min_age: 0.2
    performance_threshold: 0.4  # ← lower bar (60% → 40%)
    # Even mediocre agents can breed if genetically diverse
```

**Pairing algorithm**:

```python
def select_complementary_pair(eligible_agents):
    """
    Find pair with maximum genetic distance.

    Diversity score =
        |greed_a - greed_b| +
        |curiosity_a - curiosity_b| +
        |neuroticism_a - neuroticism_b| +
        |agreeableness_a - agreeableness_b|
    """
    max_diversity = -inf
    best_pair = None

    for agent_a in eligible_agents:
        for agent_b in eligible_agents:
            if agent_a == agent_b:
                continue

            diversity = compute_dna_distance(agent_a.dna, agent_b.dna)

            if diversity > max_diversity:
                max_diversity = diversity
                best_pair = (agent_a, agent_b)

    return best_pair

def compute_dna_distance(dna_a, dna_b):
    """Euclidean distance in personality space."""
    return sqrt(
        (dna_a['greed'] - dna_b['greed'])**2 +
        (dna_a['curiosity'] - dna_b['curiosity'])**2 +
        (dna_a['neuroticism'] - dna_b['neuroticism'])**2 +
        (dna_a['agreeableness'] - dna_b['agreeableness'])**2
    )
```

**Expected population dynamics**:

- High genetic diversity maintained (no convergence to single personality type)
- Children have more "hybrid vigor" (diverse traits)
- Population covers broader strategy space

**Research questions**:

1. **Exploration-exploitation trade-off**: Does diversity improve population adaptability?
2. **Hybrid vigor**: Do diverse children outperform same-personality children?
3. **Speciation prevention**: Does forced mixing prevent dynasties from diverging?
4. **Optimal diversity level**: Is there a sweet spot (too much diversity = poor coordination)?

**Hypothesis**: Arranged marriage maintains exploration, prevents premature convergence.

---

### 9.4 Research Questions Enabled by Population Genetics

#### Category 1: Evolutionary Dynamics

**Q1.1: Does natural selection work in RL?**

- Setup: Meritocratic mode, 50k ticks
- Measure: Mean terminal reward of population over time
- Hypothesis: Population performance increases as good strategies propagate

**Q1.2: Lamarckian vs Darwinian evolution**

- Setup: Compare training_mode "frozen" (pure genetic) vs "full" (learning + inheritance)
- Measure: Convergence speed, final performance
- Hypothesis: Lamarckian (learning + inheritance) converges faster

**Q1.3: Mutation rate tuning**

- Setup: Vary mutation_rate from 0.01 to 0.20
- Measure: Population diversity, performance stability
- Hypothesis: Sweet spot around 0.05 (enough exploration, not too noisy)

#### Category 2: Social Coordination

**Q2.1: Emergent communication protocols**

- Setup: L8 (family channel - PLANNED, NOT YET IMPLEMENTED), track signal usage over time
- Measure:
  - Signal diversity (# unique signals used)
  - Signal stability (P(same signal → same state))
  - Coordination gain (family performance vs solo)
- Hypothesis: Stable protocols emerge by episode 20k

**Q2.2: Protocol transfer across generations**

- Setup: Dynasty mode, L8 (PLANNED), track child's signal usage vs parents'
- Measure: Protocol similarity (mutual information I(parent_signals; child_signals))
- Hypothesis: Children learn parents' protocols (cultural transmission)

**Q2.3: Dialect formation**

- Setup: Multiple dynasties, L8 (PLANNED)
- Measure: Inter-family signal similarity vs intra-family
- Hypothesis: Different families develop different "dialects"

#### Category 3: Inequality & Fairness

**Q3.1: Wealth concentration**

- Setup: Dynasty mode (inheritance enabled)
- Measure: Gini coefficient over time
- Hypothesis: Gini increases (wealth concentrates in successful dynasties)

**Q3.2: Mobility**

- Setup: Dynasty vs meritocratic mode
- Measure: P(child in bottom quartile → adult in top quartile)
- Hypothesis: Meritocratic mode has higher mobility

**Q3.3: Intervention effectiveness**

- Setup: Dynasty mode + wealth redistribution (e.g., tax wealthy dynasties)
- Measure: Gini, population performance
- Hypothesis: Redistribution reduces inequality without harming total performance

#### Category 4: Family Structure

**Q4.1: Optimal family size**

- Setup: Vary max_family_size (2, 3, 4)
- Measure: Coordination effectiveness, individual performance
- Hypothesis: Size 3 (2 parents + 1 child) is optimal balance

**Q4.2: Single-parent outcomes**

- Setup: Track children from SINGLE_PARENT families
- Measure: Adult performance vs two-parent children
- Hypothesis: Single-parent children have lower performance (less coordination training)

**Q4.3: Remarriage stability**

- Setup: Track DIVORCED agents who remarry
- Measure: Success rate of second families vs first families
- Hypothesis: Second families are more stable (learned from experience)

---

### 9.5 Expected Results & Predictions

Based on analogous evolutionary RL experiments (e.g., Neural MMO, Pommerman with population):

**High confidence predictions** (>80%):

1. **Selection works**: Mean population performance increases over 50k ticks in meritocratic mode
2. **Learning helps**: Lamarckian (training_mode="full") converges faster than pure genetic (training_mode="frozen")
3. **Diversity matters**: Some mutation (0.03-0.10) outperforms zero mutation or high mutation (0.20+)
4. **Families coordinate**: L8 families (PLANNED) outperform solo agents by 10-30% in contested resource scenarios

**Medium confidence predictions** (50-80%):

5. **Protocols emerge**: Stable signal meanings develop by episode 15k-25k in L8 (PLANNED)
6. **Dynasties stratify**: Dynasty mode produces 2-3 dominant lineages controlling 50%+ of population by 100k ticks
7. **Wealth concentrates**: Gini coefficient increases from ~0.3 (random) to ~0.6 (high inequality) in dynasty mode
8. **Arranged marriage maintains diversity**: Personality variance stays >0.2 (vs <0.1 in pure meritocratic)

**Low confidence predictions** (30-50%):

9. **Protocol transfer**: Children use >60% of parents' signal vocabulary (cultural transmission)
10. **Dialect formation**: Inter-family signal overlap <30% (distinct family languages)
11. **Hybrid vigor**: Arranged marriage produces higher-performing children than same-personality pairs
12. **Polygamy is trainable**: Agents successfully manage 2+ families without performance collapse

**Open questions** (too uncertain for prediction):

- Do dynasties speciate into distinct strategies (e.g., "farmer dynasties" vs "trader dynasties")?
- Can we decode emergent protocols via causal intervention?
- What's the relationship between personality DNA and communication protocol choice?
- Do orphans develop different survival strategies than family-raised agents?

---

### 9.6 Implementation Overview

**File structure**:

```
townlet/
  population/
    dynasty_manager.py        # family lifecycle state machine
    breeding_selector.py      # eligibility checks, pairing logic
    child_initializer.py      # weight/DNA inheritance
    population_controller.py  # maintains population cap, culling
    genetics_config.py        # dataclasses for config
```

**Key classes**:

```python
# dynasty_manager.py

class Family:
    """Represents a family unit."""
    def __init__(self, parent_a, parent_b):
        self.family_id = generate_family_id()
        self.parent_a = parent_a
        self.parent_b = parent_b
        self.child = None
        self.state = FamilyState.MARRIED_NO_CHILD
        self.creation_tick = current_tick

    def can_breed(self) -> bool:
        """Check if family can spawn a child."""
        return self.state == FamilyState.MARRIED_NO_CHILD

    def spawn_child(self, config) -> Agent:
        """Create child from parents."""
        child = initialize_child(
            self.parent_a,
            self.parent_b,
            config
        )
        self.child = child
        self.state = FamilyState.MARRIED_WITH_CHILD
        return child

    def process_maturity(self):
        """Check if child has matured."""
        if self.child and self.child.age > config.child_leaves_at_age:
            self.child.family_id = None
            self.child.family_state = FamilyState.NO_FAMILY
            self.child = None
            self.state = FamilyState.MARRIED_NO_CHILD

    def process_deaths(self):
        """Handle parent or child death."""
        if self.parent_a.health <= 0:
            if self.parent_b.health <= 0:
                # Both parents dead
                if self.child:
                    self.child.family_state = FamilyState.ORPHAN
            else:
                # One parent dead
                self.state = FamilyState.SINGLE_PARENT
```

```python
# breeding_selector.py

class BreedingSelector:
    """Selects which agents form families and breed."""

    def select_breeding_pairs(
        self,
        population: list[Agent],
        mode: str,
        config: GeneticsConfig,
    ) -> list[tuple[Agent, Agent]]:
        """
        Select compatible pairs for breeding.

        Args:
            population: All agents
            mode: "meritocratic", "arranged", "polygamous"
            config: Genetics configuration

        Returns:
            List of (agent_a, agent_b) pairs to form families
        """
        eligible = self.filter_eligible(population, config)

        if mode == "meritocratic":
            return self._select_meritocratic(eligible, config)
        elif mode == "arranged":
            return self._select_arranged(eligible, config)
        elif mode == "polygamous":
            return self._select_polygamous(eligible, config)

    def filter_eligible(self, population, config):
        """Filter agents who can breed."""
        eligible = []
        for agent in population:
            # Check age
            if agent.bars['age'] < config.min_breeding_age:
                continue

            # Check family state
            if agent.family_state not in [FamilyState.NO_FAMILY, FamilyState.DIVORCED]:
                continue

            # Check performance (if meritocratic)
            if config.mode == "meritocratic":
                if agent.lifetime_reward < self.performance_threshold:
                    continue

            eligible.append(agent)

        return eligible

    def _select_meritocratic(self, eligible, config):
        """Pair highest performers."""
        # Sort by lifetime reward
        eligible.sort(key=lambda a: -a.lifetime_reward)

        pairs = []
        for i in range(0, len(eligible) - 1, 2):
            pairs.append((eligible[i], eligible[i+1]))

        return pairs

    def _select_arranged(self, eligible, config):
        """Pair maximally diverse agents."""
        pairs = []
        remaining = set(eligible)

        while len(remaining) >= 2:
            # Find most diverse pair
            best_pair = None
            max_diversity = -float('inf')

            for agent_a in remaining:
                for agent_b in remaining:
                    if agent_a == agent_b:
                        continue

                    diversity = compute_dna_distance(agent_a.dna, agent_b.dna)
                    if diversity > max_diversity:
                        max_diversity = diversity
                        best_pair = (agent_a, agent_b)

            pairs.append(best_pair)
            remaining.remove(best_pair[0])
            remaining.remove(best_pair[1])

        return pairs
```

```python
# population_controller.py

class PopulationController:
    """Maintains population size and composition."""

    def maintain_population_cap(
        self,
        population: list[Agent],
        max_size: int,
        replacement_policy: str,
    ):
        """
        If population exceeds cap, remove worst performers.

        Args:
            population: Current population
            max_size: Maximum allowed population
            replacement_policy: "cull_worst", "random", "oldest"
        """
        if len(population) <= max_size:
            return  # under cap, nothing to do

        excess = len(population) - max_size

        if replacement_policy == "cull_worst":
            # Sort by lifetime reward, remove bottom N
            population.sort(key=lambda a: a.lifetime_reward)
            to_remove = population[:excess]

        elif replacement_policy == "oldest":
            # Remove agents closest to natural death
            population.sort(key=lambda a: -a.bars['age'])
            to_remove = population[:excess]

        elif replacement_policy == "random":
            import random
            to_remove = random.sample(population, excess)

        for agent in to_remove:
            self.remove_agent(agent, population)

    def remove_agent(self, agent, population):
        """Remove agent and update family structures."""
        # If agent is in a family, update family state
        if agent.family_id:
            family = self.families[agent.family_id]

            if agent in [family.parent_a, family.parent_b]:
                # Parent died
                family.process_deaths()
            elif agent == family.child:
                # Child died
                family.child = None
                family.state = FamilyState.MARRIED_NO_CHILD

        # Remove from population
        population.remove(agent)
```

---

### 9.7 Configuration Examples

**Baseline (Meritocratic)**:

```yaml
# configs/population_baseline.yaml
population_genetics:
  enabled: true
  mode: "meritocratic"

  population:
    initial_size: 100
    max_size: 100
    replacement_policy: "cull_worst"

  family_formation:
    min_age: 0.2
    performance_threshold: 0.6
    pairing_strategy: "highest_performers"

  breeding:
    one_child_policy: true
    child_leaves_at_age: 0.3

  child_initialization:
    weight_init: "crossover"
    training_mode: "finetune"
    dna_crossover_rate: 0.5
    dna_mutation_rate: 0.05
```

**Dynasty Mode**:

```yaml
# configs/population_dynasty.yaml
population_genetics:
  enabled: true
  mode: "dynasty"

  inheritance:
    transfer_wealth: true
    primogeniture: true
    keep_family_id: true

  child_initialization:
    weight_init: "clone_best"
    training_mode: "frozen"
    dna_crossover_rate: 0.7
    dna_mutation_rate: 0.02
```

**Arranged Marriage**:

```yaml
# configs/population_arranged.yaml
population_genetics:
  enabled: true
  mode: "arranged"

  family_formation:
    performance_threshold: 0.4  # lower bar
    pairing_strategy: "maximize_diversity"
    diversity_metric: "personality_distance"
```

---

**End of Section 9**

---
