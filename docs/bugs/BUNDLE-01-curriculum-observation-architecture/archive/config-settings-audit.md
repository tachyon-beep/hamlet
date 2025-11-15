# Configuration Settings Audit
## ENH-28: Experiment vs Curriculum Hierarchy

**Question**: Which config settings BREAK checkpoint portability (experiment-level) vs which DON'T (curriculum-level)?

**Test**: Can I take a checkpoint trained on setting A and load it into setting B?
- If YES → Curriculum-level (non-breaking)
- If NO → Experiment-level (breaking)

---

## BREAKING (Experiment-Level)

### substrate.yaml
- **type** (grid, grid3d, continuous, aspatial)
  - Why: Changes position_dim (Grid2D=2, Grid3D=3, Aspatial=0)
  - Example: Grid2D checkpoint → Grid3D breaks (position encoding mismatch)

- **grid.width** / **grid.height**
  - Why: Changes grid_encoding dims (3×3=9, 8×8=64)
  - Example: 3×3 checkpoint → 8×8 breaks (input shape mismatch)
  - Note: User correction - this was initially misclassified as non-breaking

### bars.yaml
- **Number of meters** (8 vs 6 vs 10)
  - Why: Changes observation field count
  - Example: 8-meter checkpoint → 6-meter breaks (missing obs_health, obs_fitness)

- **Meter names** (energy, health, satiation...)
  - Why: Changes which observation fields exist
  - Example: Renaming "energy" → "vitality" breaks (obs_energy field missing)

### affordances.yaml
- **Number of affordances** (14 vs 10 vs 5)
  - Why: Changes affordance_at_position encoding dims (14+1=15, 10+1=11)
  - Example: 14-affordance checkpoint → 10-affordance breaks (one-hot size mismatch)

- **Affordance names** (Bed, Shower, Job...)
  - Why: Changes affordance vocabulary for one-hot encoding
  - Example: Removing "Bed" shifts all indices, breaks encoding

### variables_reference.yaml
- **Number of custom VFS variables**
  - Why: Each variable adds observation fields
  - Example: Adding "rush_hour" variable increases obs_dim

- **VFS variable names**
  - Why: Changes which observation fields exist
  - Example: Renaming "deficit_energy" → "energy_shortage" breaks field lookup

### brain.yaml
- **architecture.type** (feedforward vs recurrent)
  - Why: Completely different network architectures (~26K vs ~650K params)
  - Example: SimpleQNetwork checkpoint → RecurrentSpatialQNetwork breaks

- **architecture.feedforward.hidden_layers** ([256, 128] vs [512, 256, 128])
  - Why: Changes network structure, layer dimensions
  - Example: [256,128] checkpoint → [512,256,128] breaks (weight matrix mismatch)

- **q_learning.gamma** (0.99 vs 0.95)
  - Why: Changes Q-value scale/meaning, not just a hyperparameter
  - **QUESTION**: Is this really breaking? Or just suboptimal?

- **replay.capacity** (10000 vs 50000)
  - Why: Not directly breaking, but affects training dynamics
  - **QUESTION**: Can we load checkpoint and continue with different replay buffer?

### experiment.yaml (NEW)
- **observation_policy.mode** (curriculum_superset, minimal, explicit)
  - Why: Directly controls which observation fields are included
  - Example: curriculum_superset → minimal changes obs_dim

---

## NON-BREAKING (Curriculum-Level)

### substrate.yaml
- **grid.boundary** (clamp, wrap, bounce, sticky)
  - Why: Behavior only, obs_dim unchanged
  - Checkpoint works, just different physics

- **grid.distance_metric** (manhattan, euclidean, chebyshev)
  - Why: Distance calculation only, no obs changes
  - Checkpoint works, distances computed differently

- **grid.observation_encoding** (relative, scaled, absolute)
  - Why: SAME dims, just different normalization
  - Checkpoint works, values scaled differently

### bars.yaml
- **base_depletion** (0.003 vs 0.008)
  - Why: Depletion rate is behavioral, obs_dim unchanged
  - Checkpoint works, just harder/easier survival

- **base_move_depletion** (0.005 vs 0.01)
  - Why: Movement cost is behavioral
  - Checkpoint works, different action costs

- **base_interaction_cost** (0.005 vs 0.02)
  - Why: Interaction cost is behavioral
  - Checkpoint works, different affordance costs

- **initial** (0.5 vs 1.0)
  - Why: Starting value is behavioral
  - Checkpoint works, different starting conditions

- **terminal_conditions.value** (0.0 vs 0.1)
  - Why: Termination threshold is behavioral
  - Checkpoint works, episode ends at different point

### affordances.yaml
- **costs.amount** (0.05 vs 0.10)
  - Why: Resource costs are behavioral
  - Checkpoint works, affordances cost more/less

- **effect_pipeline.amount** (0.075 vs 0.10)
  - Why: Affordance effects are behavioral
  - Checkpoint works, rewards scaled differently

- **operating_hours** ([0,24] vs [8,20])
  - Why: Availability windows are behavioral
  - Checkpoint works, just time constraints change

- **duration_ticks** (5 vs 8)
  - Why: Multi-tick duration is behavioral
  - Checkpoint works, interactions take longer

### cascades.yaml
- **cascades.threshold** (0.3 vs 0.5)
  - Why: Cascade trigger points are behavioral
  - Checkpoint works, meter relationships trigger differently

- **cascades.strength** (0.004 vs 0.008)
  - Why: Cascade strength is behavioral
  - Checkpoint works, penalties scaled differently

- **modulations.base_multiplier** (0.5 vs 1.0)
  - Why: Modulation strength is behavioral
  - Checkpoint works, fitness affects health differently

### drive_as_code.yaml
- **extrinsic.type** (multiplicative, constant_base_with_shaped_bonus, weighted_sum...)
  - Why: Reward function doesn't affect network input
  - Checkpoint works, agent optimizes for different reward

- **extrinsic.bars** ([energy] vs [energy, health])
  - Why: Which meters contribute to reward is behavioral
  - Checkpoint works, different optimization target

- **intrinsic.base_weight** (1.0 vs 0.1)
  - Why: Exploration vs exploitation balance
  - Checkpoint works, different exploration behavior

- **shaping** (any shaping bonuses)
  - Why: Reward shaping doesn't change obs_dim
  - Checkpoint works, additional incentives added

### training.yaml
- **environment.partial_observability** (true vs false)
  - Why: **WAIT** - This might be BREAKING if not handled by observation_policy
  - **QUESTION**: Does this change local_window inclusion?

- **environment.vision_range** (2 vs 3 vs 8)
  - Why: Local window size (5×5=25 vs 7×7=49)
  - **QUESTION**: Is this breaking? Changes local_window dims

- **environment.enable_temporal_mechanics** (true vs false)
  - Why: Adds time_sin, time_cos observation fields
  - **QUESTION**: Is this breaking? Changes obs_dim

- **environment.enabled_affordances** (null vs [0,1,2,3])
  - Why: Subset of affordances deployed, but encoding size unchanged
  - Checkpoint works, some affordances unavailable

- **environment.randomize_affordances** (true vs false)
  - Why: Position randomization is behavioral
  - Checkpoint works, fixed vs random affordance positions

- **population.num_agents** (1 vs 100)
  - Why: Population size doesn't affect per-agent network
  - Checkpoint works, just batch size changes

- **curriculum.max_steps_per_episode** (500 vs 1000)
  - Why: Episode length is behavioral
  - Checkpoint works, episodes truncated differently

- **curriculum.survival_advance_threshold** (0.7 vs 0.5)
  - Why: Curriculum progression logic
  - Checkpoint works, stage transitions happen differently

- **exploration.initial_intrinsic_weight** (1.0 vs 0.5)
  - Why: Intrinsic reward scaling
  - Checkpoint works, exploration strength differs

- **exploration.variance_threshold** (100.0 vs 50.0)
  - Why: Annealing trigger threshold
  - Checkpoint works, intrinsic weight anneals differently

- **training.device** (cuda vs cpu)
  - Why: Execution device, not architecture
  - Checkpoint works (with device mapping)

- **training.max_episodes** (5000 vs 10000)
  - Why: Training duration
  - Checkpoint works, just different stopping point

- **training.train_frequency** (4 vs 8)
  - Why: Update frequency
  - Checkpoint works, learning dynamics differ

- **training.batch_size** (64 vs 128)
  - Why: Batch size for training
  - Checkpoint works, gradient estimation differs

- **training.sequence_length** (8 vs 16)
  - Why: LSTM sequence length (recurrent only)
  - Checkpoint works, temporal context window differs

- **training.max_grad_norm** (10.0 vs 5.0)
  - Why: Gradient clipping threshold
  - Checkpoint works, gradient scale differs

- **training.epsilon_start/decay/min**
  - Why: Exploration schedule
  - Checkpoint works, epsilon-greedy schedule differs

### brain.yaml (questionable)
- **optimizer.learning_rate** (0.00025 vs 0.001)
  - Why: Learning speed, not architecture
  - Checkpoint works, continues with different LR

- **optimizer.type** (adam vs sgd)
  - Why: Optimizer choice affects training, not network
  - **QUESTION**: Can we change optimizer mid-training? Different optimizer states...

- **q_learning.target_update_frequency** (100 vs 500)
  - Why: Target network sync frequency
  - Checkpoint works, sync happens differently

- **q_learning.use_double_dqn** (true vs false)
  - Why: Q-target computation algorithm
  - Checkpoint works, just different Q-learning variant

### cues.yaml
- **All settings**
  - Why: UI metadata, multi-agent observability (future)
  - Checkpoint works, doesn't affect current single-agent training

---

## QUESTIONS / UNCLEAR

1. **training.yaml: partial_observability**
   - Currently changes local_window inclusion
   - Should this be controlled by observation_policy instead?
   - Proposal: observation_policy dictates fields, partial_observability is behavioral

2. **training.yaml: enable_temporal_mechanics**
   - Adds time_sin, time_cos fields
   - Should this be experiment-level (changes obs_dim)?
   - Or should temporal fields ALWAYS be present (curriculum_active=false)?

3. **brain.yaml: gamma**
   - Discount factor fundamentally changes Q-value meaning
   - Is this breaking (experiment) or non-breaking (curriculum)?
   - Can you load checkpoint and continue with different gamma?

4. **brain.yaml: optimizer type/state**
   - Can you switch optimizers mid-training?
   - Optimizer state (momentum, etc.) stored in checkpoint

5. **training.yaml: vision_range**
   - Changes local_window dims (5×5 vs 7×7)
   - Clearly breaking... but currently in training.yaml
   - Should move to substrate.yaml? Or observation_policy?

---

## NEXT STEPS

1. Resolve QUESTIONS above
2. Create final categorization
3. Design experiment.yaml schema
4. Design compiler validation rules
