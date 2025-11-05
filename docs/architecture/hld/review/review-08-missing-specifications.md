---
document_type: Design Rationale / Requirements
status: Draft
version: 2.5
---

## SECTION 8: MISSING SPECIFICATIONS

These are not bugs or blockers — they're underspecified behaviors that need explicit documentation so implementations are deterministic and reproducible.

**Priority**: Fix after blockers, before any multi-agent (Level 6+) or family (Level 8) experiments.

---

### 8.1 Multi-Agent Contention Resolution

**Status**: UNDERSPECIFIED
**Affects**: Level 6-8 (multi-agent levels)
**Current state**: High-Level Design mentions "deterministic tie-breaking" but doesn't specify the algorithm

#### The Problem

When multiple agents try to use the same affordance simultaneously:

```python
# Tick 842
agent_2.action = INTERACT  # at Job position (6,6)
agent_5.action = INTERACT  # also at Job position (6,6)

# Job has capacity: 1 (only one agent can use it)
# Who wins?
```

**Your documentation says**:
> "Tie-breaking is deterministic (not random)"

**But doesn't specify**:

- What's the tie-breaking rule?
- Distance? Agent ID? First-come-first-served?
- What happens to the loser?

**Why this matters**:

- Non-deterministic resolution → experiments aren't reproducible
- Unclear rules → agents can't learn optimal strategies
- Different implementations → results can't be compared across codebases

#### The Solution: Distance-First, Then Agent ID

**Rule**: When multiple agents attempt to INTERACT with the same capacity-1 affordance on the same tick, resolve by:

1. **Distance**: Closest agent wins
2. **Tie-breaker**: If equidistant, lowest agent_id wins

**Implementation**:

```python
# environment/vectorized_env.py

def resolve_contention(self, affordance_id: str, tick: int):
    """
    Resolve contention for capacity-1 affordances.

    Returns:
        winner_id: agent_id that gets access
        losers: list of agent_ids that were blocked
    """
    affordance = self.affordances[affordance_id]

    # Find all agents trying to INTERACT with this affordance
    candidates = []
    for agent_id, agent in enumerate(self.agents):
        if agent.current_action == Action.INTERACT:
            if agent.position == affordance.position:
                candidates.append(agent_id)

    if len(candidates) <= affordance.capacity:
        # No contention, everyone succeeds
        return candidates, []

    # Contention: rank by distance, then agent_id
    def rank_key(agent_id):
        agent = self.agents[agent_id]
        distance = manhattan_distance(agent.position, affordance.position)
        return (distance, agent_id)  # tuple sorts by distance first, then ID

    candidates.sort(key=rank_key)

    winners = candidates[:affordance.capacity]
    losers = candidates[affordance.capacity:]

    # Log contention for telemetry
    self.log_contention(
        tick=tick,
        affordance=affordance_id,
        winners=winners,
        losers=losers
    )

    return winners, losers

def manhattan_distance(pos1, pos2):
    """Manhattan distance in grid world."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```

**Example scenarios**:

**Scenario 1: Different distances**

```python
# Job at (6, 6), capacity: 1
agent_2.position = (6, 5)  # distance = 1
agent_5.position = (6, 4)  # distance = 2

# Both try INTERACT
# Winner: agent_2 (closer)
# Result: agent_2 works, agent_5's action fails
```

**Scenario 2: Same distance**

```python
# Job at (6, 6), capacity: 1
agent_2.position = (5, 6)  # distance = 1
agent_5.position = (6, 5)  # distance = 1

# Both equidistant
# Winner: agent_2 (lower agent_id)
# Result: agent_2 works, agent_5's action fails
```

**Scenario 3: Capacity > 1**

```python
# Hospital at (3, 8), capacity: 2
agent_1.position = (3, 7)  # distance = 1
agent_3.position = (3, 6)  # distance = 2
agent_7.position = (2, 8)  # distance = 1

# Three agents try INTERACT, capacity = 2
# Ranked: [agent_1 (d=1, id=1), agent_7 (d=1, id=7), agent_3 (d=2, id=3)]
# Winners: agent_1, agent_7 (top 2 by distance, tie-break by ID)
# Loser: agent_3
```

**What happens to losers**:

```python
# After contention resolution
for loser_id in losers:
    # Action fails, agent stays in current position
    self.agents[loser_id].interaction_failed = True

    # Agent receives observation flag
    self.observations[loser_id]['action_failed'] = True
    self.observations[loser_id]['failure_reason'] = 'contention'

    # Agent can learn: "I tried Job, it was occupied"
    # Next time: check for other agents before attempting
```

**Why this rule**:

- **Deterministic**: Same positions → same outcome
- **Intuitive**: Closer agents win (realistic spatial reasoning)
- **Fair**: When equidistant, lowest ID wins (arbitrary but consistent)
- **Learnable**: Agents can predict outcomes and plan around contention

**Alternative considered: First-come-first-served**

```python
# Track when each agent arrived at affordance
rank_key = lambda agent_id: (arrival_tick[agent_id], agent_id)
```

**Rejected because**:

- Requires tracking arrival history (more state)
- "Camping" strategy dominates (arrive early, wait)
- Doesn't reward spatial planning as much

#### Documentation Updates Required

**1. Add to Universe as Code (affordances.yaml docs)**:

```yaml
# affordances.yaml

affordances:
  - id: "job"
    # ... other fields ...
    capacity: 1

    # Contention resolution (when capacity is exceeded):
    # 1. Closest agent(s) win (Manhattan distance)
    # 2. Tie-breaker: lowest agent_id
    # 3. Losers receive action_failed=True in observation
```

**2. Add new doc: `docs/multi_agent_mechanics.md`**:

```markdown
## Multi-Agent Contention Resolution

### Affordance Capacity

Each affordance has a `capacity` field (default: 1):
- `capacity: 1` — only one agent can use it per tick
- `capacity: 2+` — multiple agents can use simultaneously
- `capacity: inf` — unlimited (e.g., Park)

### Resolution Algorithm

When more agents attempt INTERACT than capacity allows:

1. **Distance ranking**: Compute Manhattan distance from each agent to affordance
2. **Select winners**: Take top N agents by distance (N = capacity)
3. **Tie-breaking**: If multiple agents equidistant, sort by agent_id (ascending)
4. **Notify losers**: Set `action_failed=True` in their observation

### Example Code

```python
def rank_key(agent_id):
    distance = manhattan_distance(agent.position, affordance.position)
    return (distance, agent_id)

candidates.sort(key=rank_key)
winners = candidates[:capacity]
```

### Learnable Signals

Agents can learn to:

- Predict contention by observing other agents' positions
- Choose alternate affordances when contention is likely
- Arrive early to minimize distance (spatial planning)

### Implementation

See: `environment/vectorized_env.py::resolve_contention()`

```

**3. Add to High-Level Design (Section 11)**:

```markdown
### 11.X Multi-Agent Contention

When multiple agents compete for limited-capacity affordances:
- Distance-based resolution (closest wins)
- Tie-breaker: agent_id
- Losers receive explicit failure signal
- This creates strategic resource allocation problems for agents to solve
```

---

### 8.2 Family Lifecycle State Machine

**Status**: UNDERSPECIFIED
**Affects**: Level 8 (family communication), population genetics
**Current state**: High-Level Design mentions families but doesn't specify all transitions

#### The Problem

The family system has many edge cases:

**Scenario 1: Child leaves home**

- Parents had one child (breeding locked)
- Child reaches maturity age (age > 0.3?)
- Child disconnects from family
- **Question**: Can parents breed again immediately? What's the eligibility check?

**Scenario 2: One parent dies**

- Family had 2 parents, 1 child
- Parent A dies (health=0)
- **Question**: Does family persist? Can Parent B remarry? Is child orphaned or still in single-parent family?

**Scenario 3: Child dies before maturity**

- Family had 2 parents, 1 child
- Child dies at age=0.1 (early death)
- **Question**: Can parents breed again? What's the cooldown?

**Scenario 4: Both parents die**

- Family had 2 parents, 1 child
- Both parents die
- **Question**: Is child orphaned? Becomes solo agent? Can still use family channel?

**Scenario 5: Remarriage eligibility**

- Agent A was in family_1 (spouse died)
- Agent A is now solo
- Agent B is also solo (never married)
- **Question**: Can A and B form family_2? What are the rules?

**Why this matters**:

- Population dynamics depend on precise rules
- Family communication experiments need stable families
- Genetics experiments need predictable inheritance

#### The Solution: Complete State Machine

**Family states**:

```python
class FamilyState(Enum):
    NO_FAMILY = 0         # solo agent, never been in family
    MARRIED_NO_CHILD = 1  # two parents, no child yet
    MARRIED_WITH_CHILD = 2  # two parents, one child (breeding locked)
    SINGLE_PARENT = 3     # one parent, one child (other parent died)
    ORPHAN = 4            # child, both parents died
    DIVORCED = 5          # was married, child left, now solo (can remarry)
```

**Transitions**:

```python
# Complete state machine for family lifecycle

TRANSITIONS = {
    # From NO_FAMILY (solo agent)
    NO_FAMILY: {
        'meet_partner': MARRIED_NO_CHILD,  # eligibility check passes
    },

    # From MARRIED_NO_CHILD (married, no kids yet)
    MARRIED_NO_CHILD: {
        'breed': MARRIED_WITH_CHILD,  # child spawned
        'partner_dies': NO_FAMILY,    # back to solo (can remarry)
    },

    # From MARRIED_WITH_CHILD (locked, raising child)
    MARRIED_WITH_CHILD: {
        'child_leaves': MARRIED_NO_CHILD,   # child matured, can breed again
        'child_dies': MARRIED_NO_CHILD,     # child died, can breed again
        'one_parent_dies': SINGLE_PARENT,   # now single parent with child
        'both_parents_die': ORPHAN,         # child becomes orphan
    },

    # From SINGLE_PARENT (one parent, one child)
    SINGLE_PARENT: {
        'child_leaves': NO_FAMILY,   # back to solo, can remarry
        'child_dies': NO_FAMILY,     # back to solo, can remarry
        'parent_dies': ORPHAN,       # child becomes orphan
    },

    # From ORPHAN (child with no parents)
    ORPHAN: {
        'reach_adulthood': NO_FAMILY,  # mature, become independent
        'die': None,                   # removed from population
    },

    # From DIVORCED (was in family, child left, now solo)
    DIVORCED: {
        'meet_partner': MARRIED_NO_CHILD,  # can remarry
    },
}
```

**Breeding eligibility rules**:

```python
def check_breeding_eligibility(agent_a, agent_b, population):
    """
    Two agents can breed if:
    1. Both are in MARRIED_NO_CHILD state (same family)
    2. Neither has existing child
    3. Population has space (current_pop < max_pop)
    4. Both agents are adults (age > maturity_threshold)
    5. High-performer criteria met (optional, for meritocratic selection)
    """
    # Must be married to each other
    if agent_a.family_id != agent_b.family_id:
        return False, "not in same family"

    # Must be in MARRIED_NO_CHILD state
    if agent_a.family_state != FamilyState.MARRIED_NO_CHILD:
        return False, f"agent_a state is {agent_a.family_state}"
    if agent_b.family_state != FamilyState.MARRIED_NO_CHILD:
        return False, f"agent_b state is {agent_b.family_state}"

    # Population cap
    if len(population) >= population.max_size:
        return False, "population at capacity"

    # Maturity check
    maturity_age = 0.2  # agents must be at least 20% through life
    if agent_a.bars['age'] < maturity_age:
        return False, "agent_a too young"
    if agent_b.bars['age'] < maturity_age:
        return False, "agent_b too young"

    # High-performer check (for meritocratic mode)
    if population.breeding_mode == "meritocratic":
        performance_threshold = 0.6  # top 60% of population
        if not is_high_performer(agent_a, population):
            return False, "agent_a performance below threshold"
        if not is_high_performer(agent_b, population):
            return False, "agent_b performance below threshold"

    return True, "eligible"
```

**Child maturity rules**:

```python
def check_child_maturity(child_agent):
    """
    Child leaves family when:
    - age > 0.3 (30% through life)
    OR
    - terminal score > 0.5 on their first "test episode"

    This creates graduation mechanics.
    """
    if child_agent.bars['age'] > 0.3:
        return True, "reached maturity age"

    # Optional: competency-based graduation
    if hasattr(child_agent, 'test_score'):
        if child_agent.test_score > 0.5:
            return True, "passed competency test"

    return False, None
```

**Remarriage rules**:

```python
def check_remarriage_eligibility(agent_a, agent_b):
    """
    Two solo agents can marry if:
    1. Both are in NO_FAMILY or DIVORCED state
    2. Both are adults (age > maturity_threshold)
    3. Not related (no parent-child relationship)
    """
    valid_states = {FamilyState.NO_FAMILY, FamilyState.DIVORCED}

    if agent_a.family_state not in valid_states:
        return False, f"agent_a state is {agent_a.family_state}"
    if agent_b.family_state not in valid_states:
        return False, f"agent_b state is {agent_b.family_state}"

    # Maturity check
    maturity_age = 0.2
    if agent_a.bars['age'] < maturity_age:
        return False, "agent_a too young"
    if agent_b.bars['age'] < maturity_age:
        return False, "agent_b too young"

    # Prevent incest (parent can't marry their child)
    if agent_a.agent_id in agent_b.lineage or agent_b.agent_id in agent_a.lineage:
        return False, "agents are related"

    return True, "eligible"
```

**Example scenarios resolved**:

**Scenario 1: Child leaves**

```python
# Initial state
parent_a.family_state = MARRIED_WITH_CHILD
parent_b.family_state = MARRIED_WITH_CHILD
child.family_state = CHILD  # (not shown in enum, implicit)

# Trigger: child.age > 0.3
child.family_state = NO_FAMILY
child.family_id = None

parent_a.family_state = MARRIED_NO_CHILD  # can breed again
parent_b.family_state = MARRIED_NO_CHILD
```

**Scenario 2: One parent dies**

```python
# Initial state
parent_a.family_state = MARRIED_WITH_CHILD
parent_b.family_state = MARRIED_WITH_CHILD
child.family_state = CHILD

# Trigger: parent_a.health = 0 (dies)
parent_a.remove_from_population()

parent_b.family_state = SINGLE_PARENT  # still raising child
child.family_state = CHILD  # still in family with parent_b

# Later: child leaves
child.family_state = NO_FAMILY
parent_b.family_state = DIVORCED  # can remarry
```

**Scenario 3: Child dies early**

```python
# Initial state
parent_a.family_state = MARRIED_WITH_CHILD
parent_b.family_state = MARRIED_WITH_CHILD
child.age = 0.1

# Trigger: child.health = 0 (dies)
child.remove_from_population()

parent_a.family_state = MARRIED_NO_CHILD  # immediately eligible to breed again
parent_b.family_state = MARRIED_NO_CHILD
```

**Scenario 4: Both parents die**

```python
# Initial state
parent_a.family_state = MARRIED_WITH_CHILD
parent_b.family_state = MARRIED_WITH_CHILD
child.family_state = CHILD

# Trigger: both parents die
parent_a.remove_from_population()
parent_b.remove_from_population()

child.family_state = ORPHAN
child.family_id = None  # no longer has family channel access

# Orphan continues as solo agent until:
# - Reaches adulthood (age > 0.3) → NO_FAMILY state
# - Dies (removed from population)
```

**Scenario 5: Remarriage**

```python
# agent_a was in family_1 (spouse died), now DIVORCED
# agent_b was never married, now NO_FAMILY

# Check eligibility
eligible, reason = check_remarriage_eligibility(agent_a, agent_b)
# → True, "eligible"

# Create new family
new_family_id = generate_family_id()  # e.g., "family_137"
agent_a.family_id = new_family_id
agent_a.family_state = MARRIED_NO_CHILD
agent_b.family_id = new_family_id
agent_b.family_state = MARRIED_NO_CHILD

# They can now breed (once eligibility is checked)
```

#### Documentation Updates Required

**1. Create new doc: `docs/family_lifecycle.md`**:

```markdown
## Family Lifecycle State Machine

### States

- **NO_FAMILY**: Solo agent, never been in a family
- **MARRIED_NO_CHILD**: Two parents, no child yet (can breed)
- **MARRIED_WITH_CHILD**: Two parents, one child (breeding locked)
- **SINGLE_PARENT**: One parent, one child (other parent died)
- **ORPHAN**: Child with no parents
- **DIVORCED**: Was married, child left, now solo (can remarry)

### State Transitions

[Include state machine diagram or table]

### Breeding Eligibility

Two agents can breed if:
1. Both in MARRIED_NO_CHILD state (same family_id)
2. Population < max_size
3. Both age > 0.2 (maturity threshold)
4. (Optional) Both are high-performers (meritocratic mode)

### Child Maturity

Child leaves family when:
- age > 0.3 (standard)
- OR passes competency test (optional)

On leaving:
- Child → NO_FAMILY state
- Parents → MARRIED_NO_CHILD (can breed again)

### Death Handling

**One parent dies**:
- Remaining parent → SINGLE_PARENT
- Child stays in family
- When child leaves later, parent → DIVORCED (can remarry)

**Both parents die**:
- Child → ORPHAN (loses family channel access)
- At age > 0.3, orphan → NO_FAMILY

**Child dies**:
- Parents → MARRIED_NO_CHILD (can breed again immediately)

### Remarriage

Solo agents can form new families if:
- Both in NO_FAMILY or DIVORCED state
- Both age > 0.2
- Not related (no parent-child lineage)

### Implementation

See: `population/dynasty_manager.py`
```

**2. Add to High-Level Design (Section 14)**:

```markdown
## 14. Population Genetics and Family Dynamics

### 14.3 Family State Machine

[Include diagram and rules]

### 14.4 Population Equilibrium

With these rules:
- Initial population: 100
- Max population: 100
- Expected equilibrium: 70-80 agents in families, 20-30 solo
- Churn rate: ~5% per 1000 ticks (deaths + births)
```

**3. Add configuration schema to `population_genetics.yaml`**:

```yaml
population_genetics:
  enabled: true

  population:
    initial_size: 100
    max_size: 100
    breeding_mode: "meritocratic"  # or "random", "arranged"

  family_rules:
    maturity_age: 0.3      # when child can leave family
    min_breeding_age: 0.2  # parents must be this old
    one_child_policy: true # lock breeding while child exists

  remarriage:
    enabled: true
    prevent_incest: true

  death_replacement:
    enabled: true
    cull_policy: "worst_performers"  # replace deaths by removing worst agents
```

---

### 8.3 Child Initialization

**Status**: UNDERSPECIFIED
**Affects**: Level 8 (inheritance), population genetics
**Current state**: Mentioned but weights/DNA initialization not specified

#### The Problem

When a new child is spawned from two parents:

**Question 1: Weights initialization**

- Random initialization (start from scratch)?
- Clone one parent (inherit full policy)?
- Average parents' weights (literal parameter averaging)?
- Pretrained baseline (from Level 7 curriculum)?

**Question 2: DNA initialization**

- Clone one parent's DNA?
- Crossover (mix both parents' DNA)?
- Mutation applied?

**Question 3: Training**

- Does child train from scratch?
- Transfer learning from parents?
- Frozen weights, or trainable?

**Why this matters**:

- Random init → no inheritance (defeats genetics experiments)
- Pure clone → no diversity (population converges)
- Crossover without mutation → limited exploration
- Pretrained → faster learning but less "natural" evolution

#### The Solution: Configurable Inheritance Modes

**Default mode: Crossover + Mutation**

```python
# population/child_initialization.py

def initialize_child(parent_a, parent_b, config):
    """
    Initialize child agent from two parents.

    Returns:
        child_agent: new agent with inherited weights and DNA
    """
    child_agent = Agent(
        agent_id=generate_agent_id(),
        config=config
    )

    # 1. DNA crossover + mutation
    child_agent.dna = crossover_dna(
        parent_a.dna,
        parent_b.dna,
        crossover_rate=config.genetics.crossover_rate
    )

    child_agent.dna = mutate_dna(
        child_agent.dna,
        mutation_rate=config.genetics.mutation_rate
    )

    # 2. Weights initialization (configurable)
    if config.genetics.weight_init == "crossover":
        # Average parents' weights
        child_agent.weights = average_weights(
            parent_a.weights,
            parent_b.weights
        )

    elif config.genetics.weight_init == "clone_best":
        # Clone better parent
        better_parent = parent_a if parent_a.lifetime_reward > parent_b.lifetime_reward else parent_b
        child_agent.weights = copy.deepcopy(better_parent.weights)

    elif config.genetics.weight_init == "pretrained":
        # Start from Level 7 curriculum baseline
        checkpoint = load_checkpoint(config.genetics.pretrained_path)
        child_agent.weights = checkpoint['weights']

    elif config.genetics.weight_init == "random":
        # Start from scratch (no inheritance)
        child_agent.weights = initialize_random_weights()

    # 3. Training mode
    if config.genetics.child_training == "frozen":
        # Weights are inherited but not updated
        for param in child_agent.parameters():
            param.requires_grad = False

    elif config.genetics.child_training == "finetune":
        # Start with inherited weights, continue training with lower LR
        for param in child_agent.parameters():
            param.requires_grad = True
        child_agent.learning_rate = config.genetics.finetune_lr  # e.g., 0.1× parent LR

    elif config.genetics.child_training == "full":
        # Start with inherited weights, train normally
        for param in child_agent.parameters():
            param.requires_grad = True
        child_agent.learning_rate = config.genetics.learning_rate

    return child_agent
```

**DNA crossover**:

```python
def crossover_dna(dna_a, dna_b, crossover_rate=0.5):
    """
    Combine two parent DNAs with genetic crossover.

    DNA structure (from cognitive_topology.yaml):
    {
        'personality': {
            'greed': float,
            'curiosity': float,
            'neuroticism': float,
            'agreeableness': float,
        },
        'panic_thresholds': {
            'energy': float,
            'health': float,
        },
        'compliance': {
            'forbid_actions': list,
            'penalize_actions': list,
        }
    }
    """
    child_dna = {}

    for key in dna_a.keys():
        if isinstance(dna_a[key], dict):
            # Recurse for nested dicts
            child_dna[key] = crossover_dna(dna_a[key], dna_b[key], crossover_rate)

        elif isinstance(dna_a[key], float):
            # Crossover continuous values
            if random.random() < crossover_rate:
                # Take from parent_a
                child_dna[key] = dna_a[key]
            else:
                # Take from parent_b
                child_dna[key] = dna_b[key]

        elif isinstance(dna_a[key], list):
            # For lists (like forbid_actions), take union or intersection
            if key == 'forbid_actions':
                # Child inherits stricter rules (union of both parents)
                child_dna[key] = list(set(dna_a[key]) | set(dna_b[key]))
            else:
                # For other lists, random choice
                child_dna[key] = dna_a[key] if random.random() < 0.5 else dna_b[key]

    return child_dna
```

**DNA mutation**:

```python
def mutate_dna(dna, mutation_rate=0.05):
    """
    Apply random mutations to DNA.

    For each gene:
    - With probability mutation_rate, perturb the value
    - Perturbation is small (±10% for continuous values)
    """
    mutated = copy.deepcopy(dna)

    for key, value in mutated.items():
        if isinstance(value, dict):
            # Recurse for nested dicts
            mutated[key] = mutate_dna(value, mutation_rate)

        elif isinstance(value, float):
            if random.random() < mutation_rate:
                # Add Gaussian noise (σ = 0.1 of current value)
                noise = random.gauss(0, 0.1 * abs(value))
                mutated[key] = np.clip(value + noise, 0.0, 1.0)

        # Lists (like forbid_actions) don't mutate by default
        # Could add/remove actions with very low probability if desired

    return mutated
```

**Weight averaging** (for crossover mode):

```python
def average_weights(weights_a, weights_b):
    """
    Average two sets of neural network weights.

    Returns:
        averaged_weights: parameter-wise mean
    """
    averaged = {}

    for name in weights_a.keys():
        if isinstance(weights_a[name], torch.Tensor):
            averaged[name] = (weights_a[name] + weights_b[name]) / 2.0
        else:
            # For non-tensor values, pick randomly
            averaged[name] = weights_a[name] if random.random() < 0.5 else weights_b[name]

    return averaged
```

**Configuration example**:

```yaml
# population_genetics.yaml

genetics:
  # DNA inheritance
  crossover_rate: 0.5      # 50% genes from each parent
  mutation_rate: 0.05      # 5% chance per gene

  # Weight initialization
  weight_init: "crossover"  # options: crossover, clone_best, pretrained, random
  pretrained_path: "checkpoints/L7_baseline/step_10000/"  # if using pretrained

  # Training mode
  child_training: "finetune"  # options: frozen, finetune, full
  finetune_lr: 0.0001         # learning rate for children (if finetune)
  learning_rate: 0.001        # learning rate for adults
```

**Research modes**:

**Mode 1: Pure Genetics (no learning)**

```yaml
weight_init: "crossover"
child_training: "frozen"
```

- Tests: Can good policies evolve through selection alone?
- No individual learning, only population-level evolution

**Mode 2: Lamarckian (learning + inheritance)**

```yaml
weight_init: "crossover"
child_training: "full"
```

- Tests: Does learned behavior + inheritance accelerate adaptation?
- Children start with parents' knowledge, improve via learning

**Mode 3: Tabula Rasa + DNA**

```yaml
weight_init: "random"
child_training: "full"
```

- Tests: Does personality (DNA) matter if weights reset?
- Children inherit personality traits but not skills

**Mode 4: Pretrained Baseline**

```yaml
weight_init: "pretrained"
child_training: "finetune"
```

- Tests: Fastest path to competent population
- All children start from Level 7 curriculum, specialize via DNA

#### Documentation Updates Required

**1. Add to `docs/population_genetics.md`**:

```markdown
## Child Initialization

When parents breed, the child is initialized with:

### 1. DNA Inheritance

DNA (personality, thresholds, compliance rules) is combined via:
- **Crossover**: 50% genes from each parent (configurable)
- **Mutation**: 5% chance of perturbation per gene (configurable)

Example:
- parent_a.greed = 0.8
- parent_b.greed = 0.4
- child.greed = 0.8 (50% chance) or 0.4 (50% chance)
- mutation: child.greed = 0.83 (if mutation occurs)

### 2. Weight Initialization

Neural network weights are initialized via (configurable):

**Crossover** (default):
- Average both parents' weights parameter-wise
- Produces intermediate policy between parents

**Clone Best**:
- Copy better-performing parent's weights
- Faster learning, less diversity

**Pretrained**:
- Load from Level 7 curriculum checkpoint
- Most competent starting point

**Random**:
- Initialize from scratch
- No inheritance (pure learning)

### 3. Training Mode

**Frozen**:
- Weights inherited but not updated
- Pure genetic evolution (no individual learning)

**Finetune** (default):
- Start with inherited weights
- Continue training with lower learning rate
- Lamarckian evolution (learning + inheritance)

**Full**:
- Start with inherited weights
- Train normally
- Maximum plasticity

### Configuration

See: `population_genetics.yaml`
```

**2. Add to High-Level Design (Section 14.5)**:

```markdown
### 14.5 Child Initialization Modes

[Table of modes and research questions]

**Recommended default**: Crossover weights + finetune training
- Balances inheritance and learning
- Produces diverse population
- Enables both genetic and behavioral evolution
```

---

### 8.4 Missing Specifications Summary & Where They Belong

| Specification | Priority | Goes In | Estimated Effort |
|---------------|----------|---------|------------------|
| **Contention resolution** | HIGH | `docs/multi_agent_mechanics.md` + affordances.yaml | 1 day |
| **Family lifecycle** | HIGH | `docs/family_lifecycle.md` + population_genetics.yaml | 2 days |
| **Child initialization** | MEDIUM | `docs/population_genetics.md` + population_genetics.yaml | 1 day |
| Operating hours enforcement | LOW | Already specified in affordances.yaml | - |
| Panic override rules | LOW | Already in cognitive_topology.yaml | - |
| Cascade evaluation order | LOW | cascades.yaml (add `priority` field) | 0.5 days |

**Week 2 schedule** (after blockers fixed):

**Day 1**: Write multi-agent mechanics docs

- Document contention algorithm
- Add to affordances.yaml schema
- Write tests for tie-breaking

**Day 2**: Write family lifecycle docs

- Complete state machine
- Add all edge case handlers
- Document in family_lifecycle.md

**Day 3**: Write population genetics docs

- Child initialization modes
- Configuration options
- Research mode guide

**Day 4**: Implement validators

```python
# tests/test_specifications.py

def test_contention_is_deterministic():
    """Same positions → same winner."""
    env = VectorizedTownletEnv(config)

    # Scenario: two agents, one affordance
    agents = [
        Agent(position=(6, 5)),  # agent_0
        Agent(position=(5, 6))   # agent_1
    ]

    affordance = Affordance(id="job", position=(6, 6), capacity=1)

    # Both try INTERACT
    actions = [Action.INTERACT, Action.INTERACT]

    # Resolve 100 times (should be identical)
    winners = [env.resolve_contention("job", tick=i)[0] for i in range(100)]

    # All should be same winner
    assert len(set(winners)) == 1  # only one unique winner across all runs
    assert winners[0] == [0]  # agent_0 wins (equidistant, lower ID)

def test_family_state_transitions():
    """Verify all state machine transitions."""
    family = Family(parent_a, parent_b)

    # Initial state
    assert family.state == FamilyState.MARRIED_NO_CHILD

    # Breed
    child = family.spawn_child()
    assert family.state == FamilyState.MARRIED_WITH_CHILD

    # Child leaves
    child.age = 0.4  # past maturity
    family.process_maturity()
    assert family.state == FamilyState.MARRIED_NO_CHILD

    # One parent dies
    parent_a.health = 0
    family.process_deaths()
    assert family.state == FamilyState.SINGLE_PARENT

    # [test remaining transitions...]

def test_child_inherits_dna():
    """Child DNA is crossover of parents."""
    parent_a = Agent(dna={'greed': 0.8, 'curiosity': 0.3})
    parent_b = Agent(dna={'greed': 0.4, 'curiosity': 0.7})

    child = initialize_child(parent_a, parent_b, config)

    # Child's greed should be one of parent values (or mutated nearby)
    assert child.dna['greed'] in [0.8, 0.4] or abs(child.dna['greed'] - 0.8) < 0.2

    # Crossover should mix traits
    # (Test with multiple children to check distribution)
```

**Day 5**: Add to QUICKSTART.md

- How to configure families
- How to interpret contention logs
- How to set inheritance modes

**Validation criteria**:

- All state machine transitions documented
- All edge cases handled
- Configuration schemas validated
- Tests pass for determinism
- Examples in docs

After Week 2, you can credibly claim:

- "Multi-agent interactions are fully specified"
- "Family dynamics are completely defined"
- "Population genetics are configurable and reproducible"

---
