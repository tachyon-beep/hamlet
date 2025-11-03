# Townlet Phase 3: Intrinsic Exploration Design

**Date**: 2025-10-30
**Status**: Design Complete, Ready for Implementation
**Duration**: 5-7 days
**Dependencies**: Phase 0 (Foundation), Phase 1 (GPU Infrastructure), Phase 2 (Adversarial Curriculum)

---

## 1. Overview

### 1.1 Goal

Implement Random Network Distillation (RND) and adaptive intrinsic motivation to enable sparse reward learning. This provides exploration scaffolding that allows agents to learn with minimal external feedback.

### 1.2 The Challenge

With pure sparse rewards (terminal feedback only), agents struggle to explore effectively. Random exploration (epsilon-greedy) is too slow to discover good behaviors in reasonable time. Intrinsic motivation provides a learning signal that rewards novelty, guiding exploration toward unexplored states.

### 1.3 Success Criteria

- ✅ RND provides novelty signal (high reward for new states, low for familiar)
- ✅ Adaptive intrinsic weight anneals automatically based on agent competence
- ✅ Agent learns better with intrinsic motivation than pure epsilon-greedy
- ✅ Live visualization shows intrinsic motivation decay over multi-day demo
- ✅ Integration with adversarial curriculum (both systems work together)

---

## 2. Architecture Overview

### 2.1 Component Structure

**Core Components:**
1. **ReplayBuffer** - Experience storage with separate extrinsic/intrinsic rewards
2. **RNDExploration** - Random Network Distillation for novelty detection
3. **AdaptiveIntrinsicExploration** - Wraps RND with variance-based annealing

**Visualization Components** (extend existing Hamlet UI):
4. **IntrinsicRewardChart.vue** - Real-time line chart (extrinsic vs intrinsic)
5. **NoveltyHeatmap.vue** - Grid overlay showing RND prediction errors
6. **CurriculumTracker.vue** - Stage progression display
7. **SurvivalTrendChart.vue** - Long-term trends over hours/days

### 2.2 Design Decisions

**Key Architectural Choices:**

1. **Composition over Inheritance**
   - AdaptiveIntrinsicExploration contains RNDExploration instance
   - Clean separation: RND computes novelty, Adaptive handles annealing
   - Each component testable independently

2. **Replay Buffer with Dual Rewards**
   - Store extrinsic and intrinsic rewards separately
   - Combine during Q-network training: `total = extrinsic + intrinsic * weight`
   - Enables post-hoc analysis and flexible weight adjustment

3. **RND Network Architecture**
   - Match Q-network architecture: 3-layer MLP [70 → 256 → 128 → 128]
   - Consistent with existing code, scales for future complexity
   - 128-dim embeddings balance expressiveness vs speed

4. **Conservative Annealing**
   - Survival variance threshold: 10.0 (requires high consistency)
   - Rolling window: 100 episodes (long-term performance)
   - Exponential decay: weight *= 0.99 per trigger (smooth transition)
   - Weight range: 1.0 → 0.0 (full transition to pure sparse rewards)

5. **Mini-batch Predictor Training**
   - Train every 128 steps (balance frequency vs stability)
   - Single gradient step per batch (fast, prevents overfitting)
   - Adam optimizer with lr=1e-4 (standard for auxiliary networks)

---

## 3. Component Specifications

### 3.1 ReplayBuffer

**File**: `src/townlet/training/replay_buffer.py`

**Purpose**: Store experience tuples with separate extrinsic/intrinsic rewards for off-policy Q-learning.

**Interface**:
```python
class ReplayBuffer:
    def __init__(self, capacity: int = 10000, device: torch.device = torch.device('cpu')):
        """Initialize replay buffer with fixed capacity."""

    def push(
        self,
        observations: torch.Tensor,      # [batch, obs_dim]
        actions: torch.Tensor,           # [batch]
        rewards_extrinsic: torch.Tensor, # [batch]
        rewards_intrinsic: torch.Tensor, # [batch]
        next_observations: torch.Tensor, # [batch, obs_dim]
        dones: torch.Tensor,             # [batch]
    ) -> None:
        """Add batch of transitions to buffer."""

    def sample(self, batch_size: int, intrinsic_weight: float) -> Dict[str, torch.Tensor]:
        """Sample random mini-batch with combined rewards.

        Returns:
            {
                'observations': [batch_size, obs_dim],
                'actions': [batch_size],
                'rewards': [batch_size],  # extrinsic + intrinsic * weight
                'next_observations': [batch_size, obs_dim],
                'dones': [batch_size],
            }
        """

    def __len__(self) -> int:
        """Return current buffer size."""
```

**Storage**:
- Circular buffer (FIFO when full)
- Capacity: 10,000 transitions (configurable)
- All tensors stored on device (GPU-compatible)

**Sampling**:
- Random uniform sampling (no prioritization)
- Returns combined rewards: `extrinsic + intrinsic * weight`
- Intrinsic weight passed at sample time (allows dynamic adjustment)

---

### 3.2 RNDExploration

**File**: `src/townlet/exploration/rnd.py`

**Purpose**: Compute intrinsic rewards via Random Network Distillation (novelty detection).

**Interface**:
```python
class RNDExploration(ExplorationStrategy):
    def __init__(
        self,
        obs_dim: int = 70,
        embed_dim: int = 128,
        learning_rate: float = 1e-4,
        training_batch_size: int = 128,
        device: torch.device = torch.device('cpu'),
    ):
        """Initialize RND with fixed and predictor networks."""

    def select_actions(
        self,
        q_values: torch.Tensor,
        agent_states: BatchedAgentState,
    ) -> torch.Tensor:
        """Select actions using epsilon-greedy (inherited behavior)."""

    def compute_intrinsic_rewards(
        self,
        observations: torch.Tensor,  # [batch, obs_dim]
    ) -> torch.Tensor:
        """Compute RND novelty signal.

        Returns:
            [batch] tensor of intrinsic rewards (prediction errors)
        """

    def update_predictor(self) -> float:
        """Train predictor network on accumulated observations.

        Called every training_batch_size steps.

        Returns:
            Prediction loss (for logging)
        """

    def get_novelty_map(self, grid_size: int = 8) -> torch.Tensor:
        """Get novelty values for all grid positions.

        For visualization: compute prediction error at each grid cell.

        Returns:
            [grid_size, grid_size] tensor of novelty values
        """
```

**Architecture**:

**RNDNetwork** (both fixed and predictor):
```python
class RNDNetwork(nn.Module):
    """3-layer MLP matching Q-network architecture."""

    def __init__(self, obs_dim: int = 70, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

**Initialization**:
- `fixed_network`: Random init, **freeze all parameters** (never trained)
- `predictor_network`: Random init, trained to match fixed network
- Adam optimizer with lr=1e-4 for predictor

**Intrinsic Reward Computation**:
```python
def compute_intrinsic_rewards(self, observations: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        target_features = self.fixed_network(observations)  # [batch, 128]

    predicted_features = self.predictor_network(observations)  # [batch, 128]

    # MSE per sample (high error = novel = high intrinsic reward)
    mse_per_sample = ((target_features - predicted_features) ** 2).mean(dim=1)

    return mse_per_sample  # [batch]
```

**Training Loop**:
- Accumulate observations in buffer (size: training_batch_size=128)
- When buffer full, train predictor:
  ```python
  target = self.fixed_network(obs_batch).detach()
  predicted = self.predictor_network(obs_batch)
  loss = F.mse_loss(predicted, target)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  ```
- Single gradient step per batch (prevents overfitting, keeps predictor slightly "behind")

**Novelty Map** (for visualization):
- Generate observations for all grid positions: `[grid_size², obs_dim]`
- Compute intrinsic rewards for each position
- Reshape to `[grid_size, grid_size]` heatmap
- High values = novel (red), low values = familiar (blue)

---

### 3.3 AdaptiveIntrinsicExploration

**File**: `src/townlet/exploration/adaptive_intrinsic.py`

**Purpose**: Wrap RND with variance-based annealing to automatically transition from exploration to exploitation.

**Interface**:
```python
class AdaptiveIntrinsicExploration(ExplorationStrategy):
    def __init__(
        self,
        obs_dim: int = 70,
        embed_dim: int = 128,
        rnd_learning_rate: float = 1e-4,
        rnd_training_batch_size: int = 128,
        initial_intrinsic_weight: float = 1.0,
        min_intrinsic_weight: float = 0.0,
        variance_threshold: float = 10.0,
        survival_window: int = 100,
        decay_rate: float = 0.99,
        device: torch.device = torch.device('cpu'),
    ):
        """Initialize adaptive intrinsic exploration with RND instance."""

    def select_actions(
        self,
        q_values: torch.Tensor,
        agent_states: BatchedAgentState,
    ) -> torch.Tensor:
        """Select actions using epsilon-greedy."""

    def compute_intrinsic_rewards(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted intrinsic rewards.

        Returns:
            RND novelty * current_intrinsic_weight
        """

    def update_on_episode_end(self, survival_time: float) -> None:
        """Update survival history and check for annealing trigger.

        Called after each episode completes.
        """

    def should_anneal(self) -> bool:
        """Check if variance is below threshold."""

    def anneal_weight(self) -> None:
        """Reduce intrinsic weight via exponential decay."""

    def get_intrinsic_weight(self) -> float:
        """Get current intrinsic weight (for logging/visualization)."""
```

**Composition**:
- Contains `RNDExploration` instance for novelty computation
- Delegates intrinsic reward computation to RND, then scales by weight
- Manages annealing logic independently

**Annealing Strategy**:

**Trigger Condition** (survival time variance):
```python
def should_anneal(self) -> bool:
    if len(self.survival_history) < self.survival_window:
        return False  # Not enough data

    recent_survivals = self.survival_history[-self.survival_window:]
    variance = torch.var(torch.tensor(recent_survivals))

    return variance < self.variance_threshold
```

**When variance < 10.0:**
- Agent performance is consistent (not random)
- Indicates competence at current difficulty
- Trigger annealing

**Annealing Schedule** (exponential decay):
```python
def anneal_weight(self) -> None:
    new_weight = self.current_intrinsic_weight * self.decay_rate
    self.current_intrinsic_weight = max(new_weight, self.min_intrinsic_weight)
```

**Weight Progression Example**:
- Start: 1.0 (equal extrinsic and intrinsic)
- After 50 triggers: 1.0 * (0.99^50) ≈ 0.605
- After 100 triggers: 1.0 * (0.99^100) ≈ 0.366
- After 200 triggers: 1.0 * (0.99^200) ≈ 0.134
- After 500 triggers: 1.0 * (0.99^500) ≈ 0.007 (effectively zero)

**Parameters**:
- `variance_threshold=10.0`: Conservative, requires high consistency
- `survival_window=100`: Long-term performance (not short-term luck)
- `decay_rate=0.99`: Smooth, gradual transition
- `min_intrinsic_weight=0.0`: Full transition to pure sparse rewards

---

### 3.4 Integration with VectorizedPopulation

**Modification**: `src/townlet/population/vectorized.py`

**Changes Required**:

1. **Add ReplayBuffer instance**:
   ```python
   self.replay_buffer = ReplayBuffer(capacity=10000, device=device)
   ```

2. **Store transitions in buffer** (after each step):
   ```python
   # In step_population()
   intrinsic_rewards = self.exploration.compute_intrinsic_rewards(self.current_obs)

   self.replay_buffer.push(
       observations=self.current_obs,
       actions=actions,
       rewards_extrinsic=rewards,  # From environment
       rewards_intrinsic=intrinsic_rewards,
       next_observations=next_obs,
       dones=dones,
   )
   ```

3. **Train Q-network from replay buffer**:
   ```python
   # Every N steps (e.g., 4)
   if len(self.replay_buffer) >= batch_size:
       batch = self.replay_buffer.sample(
           batch_size=64,
           intrinsic_weight=self.exploration.get_intrinsic_weight(),
       )

       # Standard DQN update with combined rewards
       q_values = self.q_network(batch['observations'])
       # ... DQN loss calculation ...
   ```

4. **Update RND predictor**:
   ```python
   # Every 128 steps
   if isinstance(self.exploration, (RNDExploration, AdaptiveIntrinsicExploration)):
       loss = self.exploration.rnd.update_predictor()
   ```

5. **Trigger annealing** (end of episode):
   ```python
   if isinstance(self.exploration, AdaptiveIntrinsicExploration):
       self.exploration.update_on_episode_end(episode_steps)
   ```

---

## 4. Visualization Enhancements

### 4.1 WebSocket Protocol Extension

**Extend `state_update` message** with RND metrics:
```json
{
  "type": "state_update",
  "state": { /* existing state */ },
  "rnd_metrics": {
    "intrinsic_reward": 0.42,
    "extrinsic_reward": -0.5,
    "intrinsic_weight": 0.78,
    "novelty_map": [[0.1, 0.3, ...], ...],  // 8x8 grid
    "curriculum_stage": 2,
    "avg_survival_last_100": 145.6
  }
}
```

### 4.2 Frontend Components

**Existing Components (preserved)**:
- `Grid.vue` - 8×8 grid with agent + affordances
- `MeterPanel.vue` - 6 meter status bars
- `StatsPanel.vue` - Episode statistics
- `Controls.vue` - Play/pause/reset controls

**New Components**:

**1. NoveltyHeatmap.vue**
- Transparent overlay on Grid.vue
- Color gradient: red (novel) → yellow → blue (familiar)
- Updates every step as RND prediction improves
- Shows agent's "mental map" solidifying over time

**2. IntrinsicRewardChart.vue**
- Dual line chart: extrinsic (blue) vs intrinsic (orange)
- X-axis: last 100 steps
- Y-axis: reward magnitude
- Shows intrinsic rewards decreasing as agent learns

**3. CurriculumTracker.vue**
- Stage indicator: "Stage 2/5: Add Hunger"
- Progress bar: steps at current stage
- Next stage criteria display
- Can be integrated into StatsPanel or separate

**4. SurvivalTrendChart.vue**
- Long-term chart: avg survival over time
- Time buckets: hourly or daily (configurable)
- Shows macro-level learning over multi-day demo
- Key engagement metric for viewers

**Layout**:
```
┌─────────────────────────────────────┐
│ Controls (play/pause/reset)         │
├───────────────┬─────────────────────┤
│ Grid          │ MeterPanel (6 bars) │
│ + Novelty     │                     │
│   Heatmap     │ StatsPanel          │
│               │ + CurriculumTracker │
├───────────────┴─────────────────────┤
│ IntrinsicRewardChart (dual lines)   │
├─────────────────────────────────────┤
│ SurvivalTrendChart (long-term)      │
└─────────────────────────────────────┘
```

**Visual Narrative for Multi-Day Demo**:

Day 1:
- High intrinsic rewards (lots of red in heatmap)
- Intrinsic line dominates chart
- Survival ~50 steps (random baseline)

Day 2:
- Heatmap cooling (more blue familiar cells)
- Intrinsic line decreasing
- Survival ~150 steps (learning patterns)

Day 3:
- Heatmap mostly blue (environment mastered)
- Extrinsic line dominates (pure sparse rewards)
- Survival 500+ steps (near-optimal behavior)

**Viewer engagement**: Watch the transition from curious explorer (day 1) to confident expert (day 3) in real-time.

---

## 5. Configuration

### 5.1 YAML Config

**File**: `configs/townlet/sparse_adaptive.yaml`

```yaml
experiment:
  name: sparse_adaptive_demo
  description: Multi-day demo of sparse reward learning with adaptive intrinsic motivation

curriculum:
  type: adversarial
  max_steps_per_episode: 500
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 1000
  device: cuda

exploration:
  type: adaptive_intrinsic
  obs_dim: 70
  embed_dim: 128
  rnd_learning_rate: 0.0001
  rnd_training_batch_size: 128
  initial_intrinsic_weight: 1.0
  min_intrinsic_weight: 0.0
  variance_threshold: 10.0
  survival_window: 100
  decay_rate: 0.99
  epsilon_start: 1.0
  epsilon_min: 0.01
  epsilon_decay: 0.995

population:
  num_agents: 10
  state_dim: 70
  action_dim: 5
  grid_size: 8
  replay_buffer_capacity: 10000
  batch_size: 64
  learning_rate: 0.00025
  gamma: 0.99
  target_update_frequency: 1000  # steps

training:
  num_episodes: 10000
  max_steps_per_episode: 500
  train_frequency: 4  # Train Q-network every N steps
  device: cuda

visualization:
  enabled: true
  websocket_host: localhost
  websocket_port: 8765
  update_frequency: 1  # Send state every N steps
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

**test_replay_buffer.py**:
```python
def test_replay_buffer_capacity():
    """Buffer should not exceed capacity (FIFO)."""

def test_replay_buffer_sampling():
    """Sampled batches should combine extrinsic and intrinsic rewards correctly."""

def test_replay_buffer_device_handling():
    """Buffer should respect device placement (CPU/CUDA)."""
```

**test_rnd.py**:
```python
def test_rnd_novelty_decreases():
    """Prediction error should decrease for repeated states."""

def test_rnd_predictor_trains():
    """Predictor network loss should decrease over training steps."""

def test_rnd_fixed_network_frozen():
    """Fixed network parameters should never change."""

def test_rnd_novelty_map():
    """get_novelty_map should return [8, 8] tensor with reasonable values."""
```

**test_adaptive_intrinsic.py**:
```python
def test_adaptive_annealing_triggers():
    """Intrinsic weight should decay when variance < threshold."""

def test_adaptive_weight_floor():
    """Weight should not go below min_intrinsic_weight."""

def test_adaptive_composition():
    """AdaptiveIntrinsic should delegate to RND correctly."""

def test_adaptive_survival_variance():
    """Variance calculation should match expected formula."""
```

### 6.2 Integration Tests

**test_rnd_integration.py**:
```python
def test_intrinsic_reward_added():
    """Total reward should be extrinsic + intrinsic * weight."""

def test_curriculum_and_exploration_together():
    """Adversarial curriculum and RND should work simultaneously."""

def test_replay_buffer_integration():
    """Q-network should train from replay buffer with combined rewards."""
```

**test_sparse_learning.py**:
```python
def test_sparse_learning_with_intrinsic():
    """Agent should learn sparse reward task with intrinsic motivation.

    Train agent for 10K episodes with:
    - AdversarialCurriculum (stages 1-5)
    - AdaptiveIntrinsicExploration (RND + annealing)
    - Sparse rewards (stage 5)

    Expected:
    - Avg survival (last 100 episodes) > 100 steps
    - Better than pure epsilon-greedy baseline (~50 steps)
    """
```

### 6.3 Visualization Tests

**Manual QA** (run live demo):
- Novelty heatmap transitions from red → blue over episodes
- Intrinsic reward line decreases while extrinsic improves
- Curriculum stage advances as agent learns
- Survival trend chart shows long-term improvement

---

## 7. Implementation Plan Structure

Phase 3 will be implemented in **8 tasks**:

**Task 1**: ReplayBuffer with dual rewards
**Task 2**: RNDNetwork architecture + fixed/predictor setup
**Task 3**: RNDExploration intrinsic reward computation
**Task 4**: RND predictor training loop
**Task 5**: AdaptiveIntrinsicExploration annealing logic
**Task 6**: VectorizedPopulation integration (replay buffer + RND updates)
**Task 7**: Visualization components (NoveltyHeatmap, IntrinsicRewardChart, CurriculumTracker, SurvivalTrendChart)
**Task 8**: End-to-end sparse learning test + YAML config

Each task follows TDD workflow:
1. Write failing test
2. Implement minimal code to pass
3. Verify test passes
4. Commit

---

## 8. Success Metrics

**After Phase 3 completion:**

**Technical**:
- ✅ All 15+ tests pass (unit + integration)
- ✅ RND prediction error decreases for repeated states (< 0.1 MSE after 1000 steps)
- ✅ Adaptive weight anneals from 1.0 → 0.0 over training
- ✅ Agent survival with intrinsic > pure epsilon-greedy (100+ vs 50 steps)
- ✅ Q-network trains from replay buffer without errors

**Visualization**:
- ✅ Novelty heatmap updates in real-time
- ✅ Intrinsic reward chart shows decay trend
- ✅ Curriculum tracker displays stage progression
- ✅ Survival trend chart shows multi-hour improvement

**Demo Quality**:
- ✅ Multi-day training runs without crashes
- ✅ Viewers can watch exploration → exploitation transition
- ✅ Clear visual narrative of agent learning

---

## 9. Future Extensions (Post-Phase 3)

**Phase 4** (Scale Testing):
- Test n=1 → 10 agents
- Per-agent RND (separate novelty maps)
- Population-level intrinsic weight annealing

**Phase 5** (Optimization):
- Mixed precision (FP16) for RND networks
- Shared RND encoder across agents
- Prioritized experience replay

**Phase 6** (Advanced Exploration):
- Curiosity-driven exploration (ICM)
- Empowerment maximization
- Never-Give-Up (NGU) exploration

---

## 10. Design Validation

**Key Questions Answered**:
- ✅ RND architecture: Match Q-network (3-layer MLP, 128-dim embeddings)
- ✅ Annealing trigger: Survival time variance < 10.0
- ✅ Annealing schedule: Exponential decay (0.99 per trigger)
- ✅ Weight range: 1.0 → 0.0 (full transition)
- ✅ Predictor training: Every 128 steps (mini-batch)
- ✅ Integration point: Replay buffer (separate extrinsic/intrinsic)
- ✅ Visualization: 4 new components extending existing Hamlet UI
- ✅ Composition: AdaptiveIntrinsic contains RND instance

**Design Rationale**:
- Conservative annealing ensures stable learning
- Composition allows independent testing
- Replay buffer enables flexible weight adjustment
- Visualization creates compelling multi-day narrative
- All decisions support future scaling (n=10 → 100 agents)

**Ready for Implementation**: Design is complete, parameters are specified, integration points are clear.
