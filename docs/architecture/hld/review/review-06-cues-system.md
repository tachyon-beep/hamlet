---
document_type: Component Spec
status: Draft
version: 2.5
date: 2025-01-05
---

## SECTION 6: THE CUES SYSTEM

### 6.1 Cues as Universe as Code

Social observability in Townlet is explicitly configured, not hardcoded. The **cues system** defines which internal states are publicly visible and under what conditions.

**Design principle**: If an agent can observe something about another agent, that observability must be declared in `cues.yaml` as part of the world configuration.

**Why this matters**:

- **Auditable social transparency**: You can inspect exactly what agents can perceive about each other
- **Configurable realism**: Different worlds can model different levels of social observability
- **Research flexibility**: Test hypotheses about information asymmetry by editing YAML
- **No telepathy**: Agents only see what the cue system explicitly broadcasts

**Location in config hierarchy**:

```
configs/my_world/
  universe_as_code.yaml
  bars.yaml
  cascades.yaml
  affordances.yaml
  cues.yaml  # ← social observability layer
```

---

### 6.2 Cue Definition Schema

A cue is a publicly observable signal triggered by internal state or behavior.

**Basic structure**:

```yaml
cues:
  - id: "looks_tired"
    description: "Visible exhaustion from low energy"
    trigger:
      type: "bar_threshold"
      bar: "energy"
      operator: "<"
      threshold: 0.3
    visibility: "public"
    priority: 2
```

**Field definitions**:

**`id` (required)**: Unique identifier for the cue

- Used in telemetry and Module C (Social Model) training
- Agents receive these as strings (not semantic embeddings)

**`description` (optional)**: Human-readable explanation

- For documentation and teaching
- Not visible to agents

**`trigger` (required)**: Condition that activates this cue

**Trigger types**:

**Type 1: Bar threshold**

```yaml
trigger:
  type: "bar_threshold"
  bar: "energy"  # which bar to check
  operator: "<"   # <, <=, >, >=, ==
  threshold: 0.3  # normalized [0, 1]
```

**Type 2: Location-based**

```yaml
trigger:
  type: "location"
  current_affordance: "Hospital"
```

**Type 3: Behavioral**

```yaml
trigger:
  type: "action_pattern"
  recent_actions: ["INTERACT", "INTERACT", "INTERACT"]  # repeated interactions
  window: 3  # check last 3 ticks
```

**Type 4: Composite** (future extension)

```yaml
trigger:
  type: "composite"
  operator: "AND"
  conditions:
    - { bar: "health", operator: "<", threshold: 0.3 }
    - { current_affordance: "Hospital" }
  # Meaning: "at hospital while sick"
```

**`visibility` (required)**: Who can see this cue

- `"public"`: Anyone within visual range
- `"family_only"`: Only family members (Level 8)
- `"private"`: Not observable (future: internal reasoning only)

**`priority` (required)**: Salience ranking

- Higher priority cues are more noticeable
- Used when multiple cues fire simultaneously
- Range: 1-5 (5 = most salient)

**`distance_limit` (optional)**: Maximum visibility radius

```yaml
distance_limit: 5  # only visible within 5 tiles
```

- Default: inherits from environment visibility_radius
- Can be shorter than vision (e.g., subtle cues vs obvious ones)

---

### 6.3 Cue Emission Rules

**Maximum broadcast limit**:

```yaml
visibility_rules:
  max_cues_broadcast: 3  # agent emits top 3 cues by priority
```

**Why limit**:

- Prevents information overload (agents would see 10+ cues per other agent)
- Forces prioritization (only most salient signals visible)
- Realistic (humans can't track 10 simultaneous body language signals)

**Emission algorithm**:

```python
def compute_public_cues(agent, cue_config):
    """
    Evaluate all cue triggers, emit top-k by priority.
    """
    active_cues = []

    # Evaluate each cue definition
    for cue_def in cue_config.cues:
        if evaluate_trigger(agent, cue_def.trigger):
            active_cues.append({
                'id': cue_def.id,
                'priority': cue_def.priority,
                'visibility': cue_def.visibility
            })

    # Filter by visibility (Level 6-7: public only, Level 8: public + family)
    if not agent.in_family or not config.family_channel_enabled:
        active_cues = [c for c in active_cues if c['visibility'] == 'public']

    # Sort by priority (high first)
    active_cues.sort(key=lambda c: -c['priority'])

    # Take top max_cues_broadcast
    max_cues = cue_config.visibility_rules.max_cues_broadcast
    emitted = active_cues[:max_cues]

    return [c['id'] for c in emitted]  # return just the IDs
```

**Example**:

```python
# Agent state
bars = {
    'energy': 0.22,    # triggers "looks_tired" (threshold 0.3)
    'health': 0.18,    # triggers "looks_sick" (threshold 0.3)
    'money': 0.15,     # triggers "looks_poor" (threshold 0.2)
    'mood': 0.35       # triggers "looks_sad" (threshold 0.4)
}
current_affordance = "Hospital"  # triggers "at_hospital"

# Active cues (priority in parentheses)
active = [
    "looks_sick" (4),
    "looks_sad" (3),
    "at_hospital" (3),
    "looks_tired" (2),
    "looks_poor" (1)
]

# After sorting by priority and taking top 3
emitted = ["looks_sick", "looks_sad", "at_hospital"]
# OR (if there's a tie): ["looks_sick", "at_hospital", "looks_sad"]
```

**Tie-breaking**: If multiple cues have same priority, deterministic ordering by `id` (alphabetical).

---

### 6.4 Example Cue Pack: Baseline World

```yaml
# cues.yaml for baseline world
version: "1.0"
description: "Standard social observability for Level 6-8"

visibility_rules:
  max_cues_broadcast: 3
  distance_limit: 5  # default visibility radius

cues:
  # Energy-based (physical fatigue)
  - id: "looks_tired"
    description: "Visible exhaustion, droopy posture"
    trigger:
      type: "bar_threshold"
      bar: "energy"
      operator: "<"
      threshold: 0.3
    visibility: "public"
    priority: 2

  - id: "looks_energetic"
    description: "Visible vitality, alert"
    trigger:
      type: "bar_threshold"
      bar: "energy"
      operator: ">"
      threshold: 0.8
    visibility: "public"
    priority: 1

  # Health-based (physical condition)
  - id: "looks_sick"
    description: "Pale, coughing, limping"
    trigger:
      type: "bar_threshold"
      bar: "health"
      operator: "<"
      threshold: 0.3
    visibility: "public"
    priority: 4  # highly salient

  - id: "looks_healthy"
    description: "Robust, good color"
    trigger:
      type: "bar_threshold"
      bar: "health"
      operator: ">"
      threshold: 0.8
    visibility: "public"
    priority: 1

  # Mood-based (emotional state)
  - id: "looks_sad"
    description: "Downcast, withdrawn"
    trigger:
      type: "bar_threshold"
      bar: "mood"
      operator: "<"
      threshold: 0.4
    visibility: "public"
    priority: 3

  - id: "looks_happy"
    description: "Smiling, animated"
    trigger:
      type: "bar_threshold"
      bar: "mood"
      operator: ">"
      threshold: 0.7
    visibility: "public"
    priority: 2

  # Wealth-based (socioeconomic signals)
  - id: "looks_poor"
    description: "Shabby clothing, worn items"
    trigger:
      type: "bar_threshold"
      bar: "money"
      operator: "<"
      threshold: 0.2
    visibility: "public"
    priority: 1

  - id: "looks_wealthy"
    description: "Well-dressed, quality items"
    trigger:
      type: "bar_threshold"
      bar: "money"
      operator: ">"
      threshold: 0.8
    visibility: "public"
    priority: 1

  # Location-based (intent signals)
  - id: "at_job"
    description: "Located at workplace"
    trigger:
      type: "location"
      current_affordance: "Job"
    visibility: "public"
    priority: 2

  - id: "at_hospital"
    description: "Located at hospital (medical need)"
    trigger:
      type: "location"
      current_affordance: "Hospital"
    visibility: "public"
    priority: 3

  - id: "at_bar"
    description: "Located at bar (socializing)"
    trigger:
      type: "location"
      current_affordance: "Bar"
    visibility: "public"
    priority: 2

  - id: "at_home"
    description: "Located at bed/residence"
    trigger:
      type: "location"
      current_affordance: "Bed"
    visibility: "public"
    priority: 1

  # Hygiene-based (social acceptability)
  - id: "looks_dirty"
    description: "Unkempt, odorous"
    trigger:
      type: "bar_threshold"
      bar: "hygiene"
      operator: "<"
      threshold: 0.3
    visibility: "public"
    priority: 2
    distance_limit: 2  # only visible up close
```

**Design notes**:

- Health cues have highest priority (life-threatening states are most salient)
- Mood cues are mid-priority (emotionally important but not urgent)
- Wealth/hygiene cues are lower priority (social but not survival-critical)
- Location cues provide intent signals without telepathy

---

### 6.5 Module C: Social Model Training via CTDE

The Social Model (Module C) learns to predict internal state from observable cues using Centralized Training, Decentralized Execution (CTDE).

**Training phase** (offline, using logged episodes):

```python
# Training loop (centralized)
for episode in training_dataset:
    for tick in episode:
        for agent_id in other_agents:
            # Ground truth (privileged - training only)
            true_bars = episode.agents[agent_id].bars
            true_goal = episode.agents[agent_id].current_goal

            # Observable inputs (available at inference)
            emitted_cues = episode.agents[agent_id].public_cues
            # e.g., ['looks_tired', 'at_job']

            # Supervised learning
            predicted_bars = social_model.predict_state(emitted_cues)
            predicted_goal = social_model.predict_goal(emitted_cues)

            # Losses
            state_loss = mse(predicted_bars, true_bars)
            goal_loss = cross_entropy(predicted_goal, true_goal)

            # Backward pass
            total_loss = state_loss + goal_loss
            total_loss.backward()
            optimizer.step()
```

**What the Social Model (Module C) learns**:

- `['looks_tired', 'at_job']` → likely bars: `{energy: ~0.25, ...}`
- `['looks_sick', 'at_hospital']` → likely goal: `SURVIVAL`
- `['looks_happy', 'at_bar']` → likely goal: `SOCIAL`

**Inference phase** (deployed policy, decentralized):

```python
# Inference (no ground truth available)
observation = {
    'other_agents_in_window': {
        'public_cues': [
            ['looks_tired', 'at_job'],  # agent 2
            ['looks_sick', 'at_hospital']  # agent 3
        ]
    }
}

# Social Model (Module C) predictions (from learned correlations)
for i, cues in enumerate(observation['other_agents_in_window']['public_cues']):
    predicted_state = social_model.predict_state(cues)
    # Output: {energy: 0.28, health: 0.75, ...}

    predicted_goal = social_model.predict_goal(cues)
    # Output: {SURVIVAL: 0.15, THRIVING: 0.70, SOCIAL: 0.15}
```

**Human equivalent**:

- Training: "When I see droopy eyes, I ask 'are you tired?' They say 'yes.' I learn the correlation."
- Inference: "I see droopy eyes now. I predict they're tired without asking."

**Why this isn't cheating**:

- At inference, the Social Model (Module C) only sees `public_cues` (same as human observer)
- Ground truth was only used during training (offline)
- The learned model generalizes from labels to real-time prediction
- This is standard supervised learning, not runtime telepathy

---

### 6.6 Curriculum Staging via Cue Richness

Different curriculum levels can use different cue sets to stage social reasoning difficulty.

**Level 6: Sparse Cues (Basics)** (Planned)

```yaml
# Level6_cues.yaml (minimal information)
cues:
  - "at_job"
  - "at_hospital"
  - "at_bar"
  # Location only - agents must infer state from behavior
```

**What agents learn at Level 6**:

- "Agent at Job probably needs money"
- "Agent at Hospital probably has low health"
- Basic intent inference from location

**Level 7: Rich Cues (Full Observability)** (Planned)

```yaml
# Level7_cues.yaml (detailed state information)
cues:
  - "looks_tired"
  - "looks_energetic"
  - "looks_sick"
  - "looks_healthy"
  - "looks_sad"
  - "looks_happy"
  - "looks_poor"
  - "looks_wealthy"
  - "looks_dirty"
  - "at_job"
  - "at_hospital"
  - "at_bar"
  # Physical + emotional + socioeconomic + location
```

**What agents learn at Level 7**:

- Fine-grained state inference
- "Looks sick + at hospital + looks poor" → survival crisis, no money for treatment
- Strategic predictions: "They'll stay at hospital, Job is free"

**Level 8: Behavioral Cues (Advanced)** (Planned)

```yaml
# Level8_cues.yaml (includes behavioral patterns)
cues:
  [all Level 7 cues] +
  - "carrying_food"          # just left Fridge
  - "rushing"                # moving quickly (high energy)
  - "lingering"              # staying in one place (waiting)
  - "avoiding_others"        # spatial pattern
```

**What agents learn at Level 8**:

- Behavior prediction from micro-actions
- "Carrying food + moving toward home" → feeding family
- Coordination: "They're lingering at Job entrance → waiting for it to open"

---

### 6.7 World Design via Cue Configuration

Cue systems enable research on information asymmetry and social transparency.

**High-Transparency World** (trust-based society)

```yaml
cues:
  [rich cue set with 15+ cues]
  max_cues_broadcast: 5  # lots of information visible
  distance_limit: 8       # see far
```

**Expected outcomes**:

- Agents can accurately predict others' needs
- Cooperation emerges easily
- Low coordination failures
- But: privacy is minimal (everyone's state is public)

**Low-Transparency World** (privacy-preserving society)

```yaml
cues:
  - "at_job"  # only location visible
  max_cues_broadcast: 1
  distance_limit: 2  # must be very close
```

**Expected outcomes**:

- Agents must infer from sparse signals
- More coordination failures (misunderstandings)
- Strategic behavior harder to predict
- But: privacy is high (internal state mostly hidden)

**Asymmetric Transparency** (class-based society)

```yaml
cues:
  - id: "looks_poor"
    visibility: "public"  # poverty is visible
    priority: 3

  - id: "looks_wealthy"
    visibility: "private"  # wealth is concealable
    priority: 1
```

**Expected outcomes**:

- Disadvantage is broadcast, advantage is hidden
- "Rich agents can pretend to be poor, poor agents can't pretend to be rich"
- Models real-world information asymmetries
- Research question: Does this create exploitation dynamics?

**Research protocol**:

```bash
# Run same agent in three transparency worlds
townlet train --config configs/high_transparency/ --seed 42
townlet train --config configs/low_transparency/ --seed 42
townlet train --config configs/asymmetric/ --seed 42

# Compare:
# - Coordination success rate
# - Social prediction accuracy (Module C)
# - Terminal reward distribution
# - Emergent strategies (trust vs suspicion)
```

---

### 6.8 Implementation Notes

**Where cues are computed**: `VectorizedTownletEnv`

```python
# In environment step() function
def step(self, actions):
    # ... execute actions, apply affordances, cascades ...

    # Compute public cues for all agents
    for agent_id, agent in enumerate(self.agents):
        agent.public_cues = self.cue_engine.compute_cues(
            agent=agent,
            cue_config=self.config.cues
        )

    # Build observations
    for agent_id in range(self.num_agents):
        # Get other agents in visibility window
        visible_agents = self.get_agents_in_window(agent_id, radius=5)

        # Extract their public cues
        other_cues = [
            self.agents[other_id].public_cues
            for other_id in visible_agents
        ]

        observations[agent_id] = {
            'bars': self.agents[agent_id].bars,
            'position': self.agents[agent_id].position,
            'other_agents_in_window': {
                'positions': [self.agents[i].position for i in visible_agents],
                'public_cues': other_cues,  # ← cues inserted here
            }
        }
```

**How the Social Model (Module C) receives cues**: As part of observation

```python
# Social Model (Module C) forward pass
def forward(self, observation):
    # Extract public cues from observation
    other_cues = observation['other_agents_in_window']['public_cues']
    # List[List[str]], e.g. [['looks_tired', 'at_job'], ['looks_sick']]

    # Embed each cue string
    cue_embeddings = [self.cue_embedding(cue) for cue_list in other_cues
                      for cue in cue_list]
    # Learned embedding: "looks_tired" → 128-dim vector

    # Process through GRU (for temporal patterns)
    h_t = self.gru(cue_embeddings, h_prev)

    # Predict state and goal
    predicted_bars = self.state_head(h_t)  # → [energy, health, ...]
    predicted_goal_dist = self.goal_head(h_t)  # → [P(SURVIVAL), P(THRIVING), ...]

    return predicted_bars, predicted_goal_dist
```

**Validation**: Cues are validated on load

```python
# cue_validator.py
def validate_cues_config(cues_yaml):
    for cue in cues_yaml['cues']:
        # Check required fields
        assert 'id' in cue
        assert 'trigger' in cue
        assert 'visibility' in cue
        assert 'priority' in cue

        # Validate trigger
        if cue['trigger']['type'] == 'bar_threshold':
            assert cue['trigger']['bar'] in VALID_BARS
            assert 0.0 <= cue['trigger']['threshold'] <= 1.0

        # Validate priority
        assert 1 <= cue['priority'] <= 5
```

---

### 6.9 Cues System Summary

**What it is**: A declarative configuration layer that defines social observability

**What it does**:

- Specifies which internal states are publicly visible
- Controls information flow in multi-agent scenarios
- Enables CTDE training for the Social Model (Module C)

**Why it matters**:

- No hardcoded telepathy (all social information is explicit)
- Auditable (you can see exactly what agents perceive)
- Configurable (research on transparency levels)
- Human-realistic (body language, not mind-reading)

**Key files**:

- `cues.yaml` — world configuration (part of UNIVERSE_AS_CODE)
- `cue_engine.py` — runtime emission
- Social Model (Module C) in `agent_architecture.yaml` — prediction

**Important clarifications**:

- **Cues System**: Framework-level pattern for defining social observability
- **Cues YAML files**: Multiple configuration files under UNIVERSE_AS_CODE (one per curriculum level or world variant)
- **Planned levels**: Level 6-8 references indicate future curriculum stages with social/multi-agent mechanics

---

**End of Section 6**

---
