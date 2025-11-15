# Configuration Settings: Semantic Categories
## ENH-28: First Principles Categorization

**Goal**: Organize all config settings by WHAT they control, not just breaking vs non-breaking.

**Key Questions**:
1. What subsystems actually exist?
2. Is "drive" part of "agent" or separate?
3. What's experiment vs curriculum vs agent vs world?

---

## SEMANTIC CATEGORIES (Draft)

### UNIVERSE/WORLD (Physical Reality)
*What exists in the world, independent of observers*

**Substrate (Spatial Structure)**:
- `substrate.type` - grid, grid3d, continuous, aspatial
- `substrate.grid.width/height` - spatial dimensions
- `substrate.grid.topology` - square, hexagonal, etc.

**Temporal Mechanics (Time System)**:
- `enable_temporal_mechanics` - does time progress?
- `day_length` - ticks per day/night cycle
- `time_of_day` effects on affordances

**Resources (State Variables)**:
- `bars.yaml` - which meters exist (names)
- `bars.*.range` - valid value ranges
- `cascades.yaml` - how meters affect each other

**Affordances (Interactions)**:
- `affordances.yaml` - which interactions exist (names)
- `affordances.*.category` - interaction types

**VFS Variables (Derived State)**:
- `variables_reference.yaml` - custom world state

**Cues (Observable Signals)**:
- `cues.yaml` - public tells (multi-agent)

---

### AGENT (Perspective & Capabilities)
*How the agent perceives and acts*

**Perception**:
- `partial_observability` - sees full grid or local window?
- `vision_range` - how far can agent see?
- `observation_encoding` - relative, scaled, absolute coords
- **QUESTION**: Is observation_policy agent-level or experiment-level?

**Action Capabilities**:
- `enabled_affordances` - which affordances can agent use?
- **QUESTION**: What about action masking, movement constraints?

**Drive/Motivation** (QUESTION: Separate or part of Agent?):
- `drive_as_code.yaml` - reward function (what agent wants)
- `intrinsic` strategy - exploration motivation (RND, ICM)
- `shaping` bonuses - behavioral incentives

---

### BRAIN (Learning Mechanism)
*How the agent learns from experience*

**Architecture**:
- `brain.architecture.type` - feedforward vs recurrent
- `brain.architecture.feedforward.hidden_layers` - network topology
- `brain.architecture.feedforward.activation` - relu, tanh, etc.

**Optimization**:
- `brain.optimizer.type` - adam, sgd, rmsprop
- `brain.optimizer.learning_rate` - step size
- `brain.optimizer.schedule` - constant, decay, cosine

**Q-Learning**:
- `brain.q_learning.gamma` - discount factor
- `brain.q_learning.target_update_frequency` - target network sync
- `brain.q_learning.use_double_dqn` - DQN variant

**Replay**:
- `brain.replay.capacity` - buffer size
- `brain.replay.prioritized` - PER vs uniform sampling

---

### STRATUM (Cross-Curriculum Shared State)
*Settings that should be consistent across curriculum levels*

**Current**:
- `stratum.yaml` - (what's actually in here? Need to check)
- Grid size consistency? (3×3 in L0, 8×8 in L1 breaks transfer)

**QUESTION**: Is stratum actually needed, or is this just "experiment-level universe settings"?

---

### TRAINING/POPULATION (Meta-Level Orchestration)
*How training is executed, not what is learned*

**Runtime**:
- `training.device` - cuda, cpu, mps
- `training.max_episodes` - training duration
- `training.train_frequency` - steps between updates
- `training.batch_size` - gradient batch size
- `training.sequence_length` - LSTM temporal context
- `training.max_grad_norm` - gradient clipping

**Population**:
- `population.num_agents` - parallel environments
- `population.mask_unused_obs` - optimization flag

**Exploration (Meta)**:
- `training.epsilon_start/decay/min` - ε-greedy schedule
- `exploration.initial_intrinsic_weight` - starting intrinsic weight
- `exploration.variance_threshold` - annealing trigger
- `exploration.min_survival_fraction` - annealing gate

---

### CURRICULUM (Progressive Difficulty)
*Per-level difficulty adjustments*

**Episode Management**:
- `curriculum.max_steps_per_episode` - episode truncation
- `curriculum.survival_advance_threshold` - stage progression
- `curriculum.survival_retreat_threshold` - stage regression
- `curriculum.entropy_gate` - minimum policy entropy
- `curriculum.min_steps_at_stage` - stabilization period

**Behavioral Difficulty**:
- `bars.*.base_depletion` - resource depletion rates (per level)
- `bars.*.initial` - starting meter values (per level)
- `affordances.*.operating_hours` - availability windows (per level)
- `affordances.*.costs.amount` - interaction costs (per level)
- `cascades.*.threshold` - cascade trigger points (per level)
- `drive_as_code.extrinsic.type` - reward function (per level)

---

## CROSS-CUTTING CONCERNS

### Observation Policy
**Problem**: This cuts across multiple categories

- Experiment-level: "Include temporal fields?" (affects obs_dim)
- Agent-level: "Agent sees local window?" (partial_observability)
- Universe-level: "Do temporal mechanics exist?" (enable_temporal_mechanics)

**Current Design**:
```yaml
# experiment.yaml (proposed)
observation_policy:
  mode: curriculum_superset  # or minimal, explicit

  # What fields to include
  temporal_mechanics: enabled  # affects obs_dim
  spatial_view: both  # grid_encoding + local_window
```

**Agent-level** (separate):
```yaml
# agent.yaml (proposed)
perception:
  partial_observability: true
  vision_range: 2
  encoding: relative
```

**Universe-level** (separate):
```yaml
# substrate.yaml or environment.yaml
temporal_mechanics:
  day_length: 24
  # or: disabled
```

---

## KEY QUESTIONS TO RESOLVE

### Q1: Is "Drive" part of "Agent" or separate?
**Arguments for Agent**:
- Drive defines "what agent wants"
- Intrinsic motivation is agent psychology
- Part of agent's goal specification

**Arguments for Separate**:
- Reward function is environment feedback signal
- Can A/B test drives without changing agent architecture
- Drive As Code is separate subsystem

**User's Question**: "is drive separate from agent or is it just because we made that subsystem later?"

### Q2: What is "Stratum" actually?
- Currently exists at `configs/stratum.yaml` (ambiguous location)
- Supposed to be "cross-curriculum shared settings"
- But what settings? Grid size? Meter vocabulary?
- Is this just "experiment-level universe config"?

### Q3: Where does grid size live?
- User correction: Grid size BREAKS checkpoints (changes obs_dim)
- So grid size must be consistent across curriculum levels
- Does it live in:
  - `stratum.yaml` (cross-curriculum constraint)?
  - `experiment.yaml` (observation policy)?
  - `substrate.yaml` with compiler validation?

### Q4: Agent.yaml vs Experiment.yaml?
Some settings feel "per-experiment" but are about agent perspective:
- `partial_observability` - agent capability or experiment design?
- `vision_range` - agent capability or experiment parameter?

Are these:
- **Experiment-level**: "In this experiment, agents are POMDP with vision_range=2"
- **Agent-level**: "This agent has vision_range=2" (future: heterogeneous agents?)

### Q5: What's the difference between "Environment" and "Universe"?
- "Universe" = world state, spatial structure, resources
- "Environment" = runtime wrapper, episode management, reset logic
- But where do temporal mechanics live?
  - Universe-level: "Days exist in this world"
  - Environment-level: "Day length is 24 ticks"

---

## PROPOSED FILE STRUCTURE (Very Draft)

```
configs/
└── my_experiment/
    ├── experiment.yaml       # What fields exist (obs_dim)
    │   ├── observation_policy: curriculum_superset
    │   ├── temporal_mechanics: enabled
    │   ├── spatial_views: both  # grid_encoding + local_window
    │
    ├── brain.yaml            # Learning mechanism
    │   ├── architecture, optimizer, q_learning, replay
    │
    ├── agent.yaml            # Agent perspective (NEW?)
    │   ├── partial_observability: true
    │   ├── vision_range: 2
    │   ├── observation_encoding: relative
    │
    ├── stratum.yaml          # Cross-curriculum constraints (UNCLEAR)
    │   ├── grid_size: 8×8  # Must be same across levels?
    │   ├── meter_vocabulary: [energy, health, ...]
    │   ├── affordance_vocabulary: [Bed, Shower, ...]
    │
    └── levels/
        ├── L0_0_minimal/
        │   ├── substrate.yaml      # Spatial params (boundaries, distance)
        │   ├── environment.yaml    # Temporal params (day_length)  (NEW?)
        │   ├── bars.yaml           # Depletion rates (curriculum difficulty)
        │   ├── affordances.yaml    # Costs, hours (curriculum difficulty)
        │   ├── cascades.yaml       # Thresholds (curriculum difficulty)
        │   ├── drive_as_code.yaml  # Reward function (curriculum design)
        │   ├── training.yaml       # Training orchestration
        │   └── variables_reference.yaml  # VFS (currently per-level, should be shared?)
```

**Problem**: This creates A LOT of files. Is this overcomplicating?

**Alternative**: Keep curriculum-level files as-is, just add experiment-level files?

---

## NEXT STEPS

1. Answer Q1-Q5 above
2. Decide final file structure
3. Update config-settings-audit.md with semantic categories
4. Design experiment.yaml schema
