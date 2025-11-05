# TASK-002A Phase 10: Graph Substrate + Infrastructure - Implementation Plan

**Date**: 2025-11-05 (Updated: 2025-11-05 post-risk-assessment)
**Status**: Ready for Implementation (After Phase 9 Complete + Pre-Work)
**Dependencies**: Phase 9 Complete (Hex + 1D Substrates) + Position Extraction Design (4-6h pre-work)
**Estimated Effort**: 90-120 hours total (revised from 68-92h after risk assessment)
**Supersedes**: Original Phase 5D Task 3 (Graph substrate split out due to infrastructure complexity)

**⚠️ PRE-WORK REQUIRED**: Design position extraction method BEFORE starting (see Pre-Work section below)

---

## Executive Summary

Phase 10 adds **Graph-Based substrate** (the most complex spatial topology) along with **critical infrastructure changes** required to support variable action spaces.

**Why Split from Phase 9?**

The peer review identified that Graph substrate requires significant infrastructure changes that would block simpler topologies:

**Infrastructure Prerequisites (P0 - BLOCKING):**
0. **Feature Flag for Action Masking** (30min) - Rollback strategy (ADDED POST-RISK-ASSESSMENT)
1. **Action Mapping Refactoring** (4-5h) - Current code hardcoded for 2D/4-direction (revised +1h)
2. **Q-Network Dynamic Sizing** (3-4h) - Fixed action_dim=5 won't work (revised +1h for checkpoints)
3. **Replay Buffer Schema Changes** (7-11h) - Needs valid_actions + position extraction (REVISED from 2-3h)
4. **Infrastructure Testing** (2-3h) - Test with ALL substrates (ADDED POST-RISK-ASSESSMENT)
5. **Checkpoint Versioning** (2h) - Migration path for old checkpoints (ADDED POST-RISK-ASSESSMENT)

**Without these changes**, Graph substrate CANNOT work. Phase 9 (Hex + 1D) doesn't need them.

**Implementation Order**: Infrastructure → Graph → Frontend → Documentation

**Key Technical Challenges**:
- **Action Masking**: Variable action spaces per state (graph nodes have different degrees)
- **Q-Network Sizing**: Dynamic action dimensions based on substrate
- **Replay Buffer**: Must store next-state valid actions for TD target computation
- **Frontend**: Render graph topology + visualize action masking

**Pedagogical Value**:
- **Graph Substrate** (✅✅✅): Teaches graph RL, topological reasoning, action masking
- **Infrastructure** (✅✅): Teaches dynamic action spaces, masked RL, architectural flexibility

---

## PRE-WORK: Position Extraction Design (4-6 hours) - REQUIRED BEFORE Task 10.0

**⚠️ CRITICAL**: Risk assessment identified this as missing from original plan. Must complete BEFORE starting infrastructure tasks.

### Overview

Design and implement method to extract agent position from flattened observation state. Required for replay buffer to store next-state valid actions.

**Problem**: State is flattened observation vector containing:
- Grid encoding (varies by substrate: 1D scalar, 2D grid, Hex axial coords)
- Meters (8 values)
- Affordance encoding (15 values)
- Temporal extras (4 values)

**Need**: Extract position component to call `substrate.get_valid_actions(position)`

**Solution Options**:
1. Add `ObservationBuilder.extract_position(state, substrate_type)` method
2. Store position separately in transition (simpler but breaks schema)
3. Reverse-engineer from grid encoding (complex, error-prone)

**Recommendation**: Option 1 - Add extraction method to ObservationBuilder

**Deliverables**:
- Design document: position encoding/extraction format
- `ObservationBuilder.extract_position()` method
- Unit tests for ALL substrate types (1D, 2D, 3D, Hex, ND, Graph)
- Validation: round-trip encode→extract returns original position

**Estimate**: 4-6 hours (2h design, 2-3h implementation, 1h testing)

**⚠️ BLOCKING**: Cannot implement replay buffer schema changes without this.

---

## Task 10.0: Infrastructure Prerequisites (14-18 hours) - BLOCKING (REVISED from 7-10h)

### Overview

These changes are **P0 BLOCKING** for Graph substrate. Infrastructure also benefits Phase 9 if done first (optional sequencing).

**Revised Estimate Rationale** (post-risk-assessment):
- Replay buffer complexity severely underestimated (2-3h → 7-11h)
- Added missing tasks (feature flag, testing, versioning)
- Added buffer for testing across ALL substrates
- Total: 7-10h → 14-18h

**Files to Modify**:
- `src/townlet/environment/vectorized_env.py` (action mapping refactor)
- `src/townlet/agent/networks.py` (dynamic Q-network sizing)
- `src/townlet/training/replay_buffer.py` (schema change)
- `src/townlet/population/vectorized.py` (use dynamic action_dim)

---

### Step 0.0: Feature flag for action masking (30 minutes) - ADDED POST-RISK-ASSESSMENT

**Problem**: Action masking infrastructure changes are breaking. Need rollback strategy.

**Solution**: Add feature flag to enable/disable action masking.

**Modify**: `src/townlet/training/config.py`

Add feature flag to training config:

```python
@dataclass
class TrainingConfig:
    """Training configuration."""
    # ... existing fields

    # Action masking (optional, defaults to False for backward compatibility)
    enable_action_masking: bool = False

    def __post_init__(self):
        # Validate action masking requirements
        if self.enable_action_masking:
            # Action masking requires specific substrate types
            logger.info("Action masking enabled - substrate must support get_valid_actions()")
```

**Modify**: `src/townlet/population/vectorized.py`

Check flag before using masking:

```python
def select_actions(self, observations, epsilon=0.0):
    """Select actions with optional action masking."""
    with torch.no_grad():
        q_values = self.q_network(observations)

        # Apply action masking if enabled AND substrate supports it
        if self.config.enable_action_masking and hasattr(self.substrate, 'get_valid_actions'):
            valid_masks = self._get_action_masks()
            q_values[~valid_masks] = -float('inf')

        # ... rest of action selection
```

**Add to config**:

```yaml
# configs/L1_graph_subway/training.yaml
training:
  enable_action_masking: true  # REQUIRED for graph substrate
```

**Rationale**: If action masking breaks, set flag to `false` and system falls back to standard RL.

**Commit**:
```bash
git add src/townlet/training/config.py \
        src/townlet/population/vectorized.py \
        configs/L1_graph_subway/training.yaml

git commit -m "feat(training): add feature flag for action masking

Adds enable_action_masking flag for gradual rollout.

**Problem**:
- Action masking is complex infrastructure change
- Need rollback strategy if issues found

**Solution**:
- Feature flag in training config
- Defaults to False (backward compatible)
- Graph substrate requires True

**Rationale**:
- Safe deployment of breaking changes
- Easy rollback if needed
- Existing configs unaffected

Part of TASK-002A Phase 10 Task 0 (Infrastructure Prerequisites)."
```

---

### Step 0.1: Refactor action mapping (3-4 hours)

**Problem**: Action mapping currently hardcoded for 2D (4 directions):

```python
# CURRENT BROKEN CODE (src/townlet/environment/vectorized_env.py)
def _action_to_deltas(self, actions: torch.Tensor) -> torch.Tensor:
    deltas = torch.zeros((num_agents, 2), dtype=torch.long)  # ❌ Hardcoded 2D!
    deltas[actions == 0, 1] = -1  # UP
    deltas[actions == 1, 1] = +1  # DOWN
    deltas[actions == 2, 0] = -1  # LEFT
    deltas[actions == 3, 0] = +1  # RIGHT
    # INTERACT (action 4) has zero delta
    return deltas
```

**Issues**:
- Won't work for 1D (2 directions, position_dim=1)
- Won't work for 3D (6 directions, position_dim=3)
- Won't work for Graph (edge-based, not delta-based)

**Solution**: Delegate to substrate's `apply_action()` method:

**Modify**: `src/townlet/environment/vectorized_env.py`

Replace `_action_to_deltas()` with substrate delegation:

```python
def _execute_movement(self, actions: torch.Tensor):
    """Execute agent movements via substrate's apply_action() method.

    Args:
        actions: [num_agents] action indices
    """
    # Delegate movement to substrate (handles all topologies)
    self.positions = self.substrate.apply_action(self.positions, actions)
```

**Remove old hardcoded methods**:
- Delete `_action_to_deltas()`
- Delete any 2D-specific movement logic
- Use substrate's `apply_action()` everywhere

**Add tests**:

Create `tests/test_townlet/unit/test_dynamic_action_mapping.py`:

```python
"""Test dynamic action mapping for different substrates."""
import torch
from townlet.substrate.grid1d import Grid1DSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate


def test_1d_action_mapping():
    """1D substrate should have 3 actions (LEFT, RIGHT, INTERACT)."""
    substrate = Grid1DSubstrate(length=10, boundary="clamp")
    
    assert substrate.action_space_size == 3
    
    # Test LEFT
    pos = torch.tensor([[5]], dtype=torch.long)
    new_pos = substrate.apply_action(pos, torch.tensor([0]))
    assert new_pos[0, 0].item() == 4
    
    # Test RIGHT  
    new_pos = substrate.apply_action(pos, torch.tensor([1]))
    assert new_pos[0, 0].item() == 6
    
    # Test INTERACT
    new_pos = substrate.apply_action(pos, torch.tensor([2]))
    assert new_pos[0, 0].item() == 5  # No movement


def test_2d_action_mapping():
    """2D substrate should have 5 actions (4 dirs + INTERACT)."""
    substrate = Grid2DSubstrate(width=10, height=10, boundary="clamp")
    
    assert substrate.action_space_size == 5


def test_3d_action_mapping():
    """3D substrate should have 7 actions (6 dirs + INTERACT)."""
    substrate = Grid3DSubstrate(width=10, height=10, depth=10, boundary="clamp")
    
    assert substrate.action_space_size == 7
```

**Run tests**:
```bash
uv run pytest tests/test_townlet/unit/test_dynamic_action_mapping.py -v
```

**Expected**: PASS (after refactor)

**Commit**:
```bash
git add src/townlet/environment/vectorized_env.py \
        tests/test_townlet/unit/test_dynamic_action_mapping.py

git commit -m "refactor(environment): delegate action mapping to substrates

Remove hardcoded 2D action mapping in favor of substrate delegation.

**Problem**:
- Action mapping hardcoded for 2D (4 directions)
- Broke 1D (2 dirs), 3D (6 dirs), Graph (edge-based)

**Solution**:
- Use substrate.apply_action() for all movement
- Substrates handle their own action semantics
- Enables variable action spaces (1D, 2D, 3D, Graph, Hex)

**Testing**:
- Added test_dynamic_action_mapping.py
- Verified 1D (3 actions), 2D (5 actions), 3D (7 actions)

Part of TASK-002A Phase 10 Task 0 (Infrastructure Prerequisites)."
```

---

### Step 0.2: Q-network dynamic sizing (2-3 hours)

**Problem**: Q-network action_dim hardcoded at initialization:

```python
# CURRENT BROKEN CODE
q_network = SimpleQNetwork(obs_dim=91, action_dim=5)  # ❌ Always 5!
```

**Issues**:
- 1D needs action_dim=3
- 3D needs action_dim=7
- Graph needs action_dim=variable (depends on max_edges)

**Solution**: Pass substrate's `action_space_size` dynamically

**Modify**: `src/townlet/population/vectorized.py`

Update Q-network initialization:

```python
def __init__(
    self,
    substrate: SpatialSubstrate,
    observation_dim: int,
    # ... other params
):
    # Get action space size from substrate (NOT hardcoded!)
    action_dim = substrate.action_space_size
    
    self.q_network = SimpleQNetwork(
        obs_dim=observation_dim,
        action_dim=action_dim,  # Dynamic!
    )
    
    self.target_network = SimpleQNetwork(
        obs_dim=observation_dim,
        action_dim=action_dim,  # Dynamic!
    )
```

**Modify**: `src/townlet/demo/runner.py`

Pass substrate to population:

```python
def __init__(self, ...):
    # Load substrate first
    self.substrate = load_substrate(config_dir / "substrate.yaml")
    
    # Compute observation dim from substrate
    obs_dim = self.substrate.get_observation_dim() + meters + affordances + temporal
    
    # Create population with dynamic action space
    self.population = VectorizedPopulation(
        substrate=self.substrate,  # NEW!
        observation_dim=obs_dim,
        # ... other params
    )
```

**Add tests**:

```python
def test_qnetwork_dynamic_action_dim():
    """Q-network should size to substrate's action space."""
    from townlet.substrate.grid1d import Grid1DSubstrate
    from townlet.substrate.grid2d import Grid2DSubstrate
    from townlet.agent.networks import SimpleQNetwork
    
    # 1D: action_dim=3
    substrate_1d = Grid1DSubstrate(length=10)
    q_net_1d = SimpleQNetwork(obs_dim=50, action_dim=substrate_1d.action_space_size)
    assert q_net_1d.q_head[-1].out_features == 3
    
    # 2D: action_dim=5
    substrate_2d = Grid2DSubstrate(width=8, height=8)
    q_net_2d = SimpleQNetwork(obs_dim=91, action_dim=substrate_2d.action_space_size)
    assert q_net_2d.q_head[-1].out_features == 5
```

**Commit**:
```bash
git add src/townlet/population/vectorized.py \
        src/townlet/demo/runner.py \
        tests/test_townlet/unit/test_qnetwork_dynamic_sizing.py

git commit -m "feat(agent): dynamic Q-network action space sizing

Q-network now sizes to substrate's action_space_size dynamically.

**Problem**:
- action_dim hardcoded to 5 at initialization
- Broke 1D (needs 3), 3D (needs 7), Graph (needs variable)

**Solution**:
- Pass substrate to VectorizedPopulation
- Use substrate.action_space_size for Q-network sizing
- Action dim determined at runtime, not hardcoded

**Testing**:
- Added test_qnetwork_dynamic_sizing.py
- Verified 1D (3), 2D (5), 3D (7) action dims

Part of TASK-002A Phase 10 Task 0 (Infrastructure Prerequisites)."
```

---

### Step 0.3: Replay buffer schema change (7-11 hours) - REVISED from 2-3h

**⚠️ COMPLEXITY UNDERESTIMATED**: Risk assessment identified this as 7-11h (not 2-3h).

**Why More Complex Than Expected:**
1. **Position extraction** (3-4h) - Must extract position from flattened state to get valid actions
2. **Schema migration** (1-2h) - Update Transition dataclass, handle old checkpoints
3. **Storage logic** (2-3h) - Store next-state valid actions during transition recording
4. **Q-target computation** (1-2h) - Apply masking during target computation
5. **Testing** (1-2h) - Verify masking works correctly across substrates

**Problem**: Replay buffer doesn't store next-state valid actions:

```python
# CURRENT SCHEMA
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    # ❌ MISSING: next_state_valid_actions (needed for masked Q-target!)
```

**Why This Breaks Graph**:

In masked RL, Q-target computation needs to mask invalid actions at next state:

```python
# Q-learning target (with action masking)
next_q_values = target_network(next_state)  # [batch, action_dim]
next_q_values[~next_state_valid_actions] = -inf  # Mask invalid actions!
max_next_q = next_q_values.max(dim=1)
target = reward + gamma * max_next_q * (1 - done)
```

**Without** `next_state_valid_actions`, we compute Q-targets over **invalid actions** (wrong!).

**Solution**: Add `valid_actions` to transition schema

**Modify**: `src/townlet/training/replay_buffer.py`

Update Transition dataclass:

```python
@dataclass
class Transition:
    """Experience tuple for replay buffer."""
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    valid_actions: torch.Tensor | None = None  # NEW! [action_dim] boolean mask
```

**Modify**: `src/townlet/population/vectorized.py`

Store valid actions when adding transitions:

```python
def store_transition(self, state, action, reward, next_state, done):
    """Store transition with valid actions for next state."""
    
    # Get valid actions for next state (for masked Q-target)
    if hasattr(self.substrate, 'get_valid_actions'):
        # Extract next position from next_state (substrate-specific)
        next_position = self._extract_position_from_state(next_state)
        valid_actions_list = self.substrate.get_valid_actions(next_position)
        
        # Convert to boolean mask
        valid_actions_mask = torch.zeros(
            self.substrate.action_space_size,
            dtype=torch.bool
        )
        valid_actions_mask[valid_actions_list] = True
    else:
        # No masking needed (all actions valid)
        valid_actions_mask = None
    
    transition = Transition(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
        valid_actions=valid_actions_mask,  # NEW!
    )
    
    self.replay_buffer.add(transition)
```

Update Q-learning target computation:

```python
def compute_loss(self, batch):
    """Compute TD loss with action masking support."""
    states, actions, rewards, next_states, dones, valid_actions = batch
    
    # Current Q-values
    current_q_values = self.q_network(states)
    current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Next Q-values (with action masking)
    with torch.no_grad():
        next_q_values = self.target_network(next_states)
        
        # Apply action masking if available
        if valid_actions is not None:
            next_q_values[~valid_actions] = -float('inf')
        
        max_next_q = next_q_values.max(dim=1)[0]
        target_q = rewards + self.gamma * max_next_q * (1 - dones)
    
    # MSE loss
    loss = F.mse_loss(current_q, target_q)
    return loss
```

**Add tests**:

```python
def test_replay_buffer_with_valid_actions():
    """Replay buffer should store and retrieve valid actions."""
    from townlet.training.replay_buffer import ReplayBuffer, Transition
    
    buffer = ReplayBuffer(capacity=100)
    
    # Add transition with valid actions mask
    valid_actions = torch.tensor([True, True, False, False, True])  # Actions 0,1,4 valid
    
    transition = Transition(
        state=torch.randn(10),
        action=0,
        reward=1.0,
        next_state=torch.randn(10),
        done=False,
        valid_actions=valid_actions,
    )
    
    buffer.add(transition)
    
    # Sample and verify
    batch = buffer.sample(1)
    assert batch.valid_actions is not None
    assert torch.equal(batch.valid_actions[0], valid_actions)
```

**Commit**:
```bash
git add src/townlet/training/replay_buffer.py \
        src/townlet/population/vectorized.py \
        tests/test_townlet/unit/test_replay_buffer_masking.py

git commit -m "feat(training): add action masking to replay buffer schema

Replay buffer now stores next-state valid actions for masked Q-targets.

**Problem**:
- Q-target computation didn't mask invalid actions at next state
- Broke Graph substrate (variable action spaces)
- Trained on invalid actions (incorrect TD targets)

**Solution**:
- Add valid_actions field to Transition dataclass
- Store next-state valid actions when adding transitions
- Mask next Q-values during target computation

**Testing**:
- Added test_replay_buffer_masking.py
- Verified valid actions stored and retrieved correctly

Part of TASK-002A Phase 10 Task 0 (Infrastructure Prerequisites)."
```

---

### Step 0.4: Infrastructure testing (2-3 hours) - ADDED POST-RISK-ASSESSMENT

**Problem**: Infrastructure changes affect ALL substrates, not just Graph. Need regression testing.

**Purpose**: Verify action mapping, Q-network sizing, and replay buffer work with ALL substrates.

**Create**: `tests/test_townlet/integration/test_infrastructure_all_substrates.py`

```python
"""Test infrastructure changes across all substrate types."""
import torch
from townlet.substrate.grid1d import Grid1DSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate
from townlet.substrate.hex import HexSubstrate
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.population.vectorized import VectorizedPopulation


def test_action_mapping_all_substrates():
    """Action mapping should work for 1D, 2D, 3D, Hex."""
    substrates = [
        Grid1DSubstrate(length=10),
        Grid2DSubstrate(width=8, height=8),
        Grid3DSubstrate(width=5, height=5, depth=5),
        HexSubstrate(radius=4),
    ]

    for substrate in substrates:
        env = VectorizedHamletEnv(substrate=substrate, num_agents=4, ...)

        # Reset and step
        obs = env.reset()
        actions = torch.randint(0, substrate.action_space_size, (4,))
        next_obs, rewards, dones = env.step(actions)

        # Should not crash
        assert next_obs.shape[0] == 4


def test_qnetwork_sizing_all_substrates():
    """Q-network should size correctly for all substrates."""
    substrates = [
        (Grid1DSubstrate(length=10), 3),   # 1D: 3 actions
        (Grid2DSubstrate(width=8, height=8), 5),  # 2D: 5 actions
        (Grid3DSubstrate(width=5, height=5, depth=5), 7),  # 3D: 7 actions
        (HexSubstrate(radius=4), 7),  # Hex: 7 actions (6 dirs + INTERACT)
    ]

    for substrate, expected_actions in substrates:
        pop = VectorizedPopulation(substrate=substrate, ...)

        # Verify Q-network output layer size
        assert pop.q_network.q_head[-1].out_features == expected_actions


def test_replay_buffer_backward_compatibility():
    """Replay buffer should work with old transitions (no valid_actions)."""
    from townlet.training.replay_buffer import ReplayBuffer, Transition

    buffer = ReplayBuffer(capacity=100)

    # Old-style transition (no valid_actions)
    old_transition = Transition(
        state=torch.randn(10),
        action=0,
        reward=1.0,
        next_state=torch.randn(10),
        done=False,
        valid_actions=None,  # Old style!
    )

    buffer.add(old_transition)

    # Should not crash
    batch = buffer.sample(1)
    assert batch is not None
```

**Run tests**:
```bash
uv run pytest tests/test_townlet/integration/test_infrastructure_all_substrates.py -v
```

**Expected**: ALL tests PASS (verifies no regressions)

**Commit**:
```bash
git add tests/test_townlet/integration/test_infrastructure_all_substrates.py

git commit -m "test(infrastructure): verify changes work across all substrates

Regression testing for infrastructure changes.

**Purpose**:
- Verify action mapping works (1D, 2D, 3D, Hex)
- Verify Q-network sizing correct (all substrates)
- Verify replay buffer backward compatible

**Testing**:
- 3 integration tests covering all substrate types
- Catches regressions from infrastructure changes

Part of TASK-002A Phase 10 Task 0 (Infrastructure Prerequisites)."
```

---

### Step 0.5: Checkpoint versioning (2 hours) - ADDED POST-RISK-ASSESSMENT

**Problem**: Replay buffer schema change breaks old checkpoints (missing valid_actions field).

**Solution**: Add checkpoint versioning and migration path.

**Modify**: `src/townlet/training/state.py`

Add checkpoint version:

```python
@dataclass
class PopulationCheckpoint:
    """Population checkpoint with versioning."""
    version: int = 2  # NEW! Incremented for schema change

    q_network_state: dict
    target_network_state: dict
    optimizer_state: dict
    replay_buffer: list[Transition]
    episode: int
    total_steps: int
    epsilon: float


def migrate_checkpoint_v1_to_v2(checkpoint: dict) -> PopulationCheckpoint:
    """Migrate old checkpoint (v1) to new schema (v2).

    Changes:
    - Adds version field (default 2)
    - Adds valid_actions=None to old transitions
    """
    # Check if already v2
    if 'version' in checkpoint and checkpoint['version'] == 2:
        return checkpoint

    # Migrate v1 → v2: Add valid_actions to transitions
    replay_buffer = checkpoint.get('replay_buffer', [])

    for transition in replay_buffer:
        if not hasattr(transition, 'valid_actions'):
            # Old transition - add field
            transition.valid_actions = None

    # Add version
    checkpoint['version'] = 2

    return PopulationCheckpoint(**checkpoint)
```

**Modify**: `src/townlet/demo/runner.py`

Add migration on checkpoint load:

```python
def load_checkpoint(self, checkpoint_path: str):
    """Load checkpoint with automatic migration."""
    checkpoint_dict = torch.load(checkpoint_path)

    # Migrate if needed
    if 'version' not in checkpoint_dict or checkpoint_dict['version'] < 2:
        logger.info(f"Migrating checkpoint from v1 to v2: {checkpoint_path}")
        checkpoint_dict = migrate_checkpoint_v1_to_v2(checkpoint_dict)

    # Load migrated checkpoint
    self.population.load_checkpoint(checkpoint_dict)
```

**Add test**:

```python
def test_checkpoint_migration_v1_to_v2():
    """Old checkpoints should migrate to new schema."""
    from townlet.training.state import migrate_checkpoint_v1_to_v2, Transition

    # Old checkpoint (v1 - no version field, no valid_actions)
    old_checkpoint = {
        'q_network_state': {},
        'target_network_state': {},
        'optimizer_state': {},
        'replay_buffer': [
            Transition(
                state=torch.randn(10),
                action=0,
                reward=1.0,
                next_state=torch.randn(10),
                done=False,
                # NO valid_actions field!
            )
        ],
        'episode': 100,
        'total_steps': 5000,
        'epsilon': 0.1,
    }

    # Migrate
    new_checkpoint = migrate_checkpoint_v1_to_v2(old_checkpoint)

    # Verify version added
    assert new_checkpoint.version == 2

    # Verify valid_actions added (None for backward compatibility)
    for transition in new_checkpoint.replay_buffer:
        assert hasattr(transition, 'valid_actions')
        assert transition.valid_actions is None
```

**Run test**:
```bash
uv run pytest tests/test_townlet/unit/test_checkpoint_versioning.py -v
```

**Expected**: PASS

**Commit**:
```bash
git add src/townlet/training/state.py \
        src/townlet/demo/runner.py \
        tests/test_townlet/unit/test_checkpoint_versioning.py

git commit -m "feat(training): add checkpoint versioning and migration

Checkpoint versioning prevents breaking old saved models.

**Problem**:
- Replay buffer schema change breaks old checkpoints
- Missing valid_actions field causes deserialization errors

**Solution**:
- Add version field to PopulationCheckpoint (v2)
- Automatic migration v1→v2 on checkpoint load
- Backward compatible (old transitions get valid_actions=None)

**Testing**:
- Added test_checkpoint_versioning.py
- Verified v1 checkpoints migrate correctly

Part of TASK-002A Phase 10 Task 0 (Infrastructure Prerequisites)."
```

---

## Task 10.1: Graph-Based Substrate (44-58 hours) - REVISED from 36-48h

### Overview

Most complex topology - graph nodes connected by edges. Requires **action masking infrastructure** (variable action spaces). Enables graph RL, non-Euclidean reasoning.

**Revised Effort Estimate**: 44-58 hours (revised from 36-48h after risk assessment)

**Why Increased Again?**
- **Peer review** (68-92h total): Doubled from original 18-24h to 36-48h
- **Risk assessment** (90-120h total): Added debugging tools +8-10h
  - Action masking debugging severely underestimated (6-8h → 10-14h)
  - Need Step 1.4b: Debugging tools for action masking (4-6h)
  - Final integration testing expanded (4-6h → 6-8h)

**Breakdown of Increase**:
- Peer review: +18-24h (identified action masking complexity)
- Risk assessment: +8-10h (identified missing debugging tools)

**Files to Create**:
- `src/townlet/substrate/graph.py` (new, ~312 lines)
- `tests/test_townlet/unit/test_substrate_graph.py` (new, 15 tests)
- `tests/test_townlet/unit/test_action_masking.py` (new, 5 tests)
- `tests/test_townlet/integration/test_graph_training.py` (new, 3 tests)
- `configs/L1_graph_subway/` (new config pack)

**Files to Modify**:
- `src/townlet/substrate/base.py` (add `get_valid_actions()` method)
- `src/townlet/substrate/config.py` (add graph config parsing)
- `src/townlet/substrate/factory.py` (wire up graph substrate)
- `src/townlet/environment/vectorized_env.py` (get valid action masks)
- `src/townlet/population/vectorized.py` (mask epsilon-greedy exploration)

---

### Step 1.1: Add action masking interface (1 hour)

**Purpose**: Extend substrate interface to support variable action spaces

**Modify**: `src/townlet/substrate/base.py`

Add after `get_valid_neighbors()` method:

```python
def get_valid_actions(self, position: torch.Tensor) -> list[int]:
    """Get valid action indices for a given position.

    Used for action masking when action space varies by state.
    Default implementation: All actions valid (no masking needed).

    Args:
        position: [position_dim] agent position

    Returns:
        List of valid action indices [0, action_space_size)

    Examples:
        Grid2D: Always returns [0, 1, 2, 3, 4] (all 5 actions valid)
        Graph: Returns edge indices + INTERACT based on node's neighbors
               Node with 3 neighbors: [0, 1, 2, max_edges] (3 edges + INTERACT)
    """
    # Default: All actions valid (no masking)
    return list(range(self.action_space_size))
```

**Add tests** in `tests/test_townlet/unit/test_action_masking.py`:

```python
"""Tests for action masking interface."""
import torch
from townlet.substrate.grid2d import Grid2DSubstrate


def test_grid2d_all_actions_valid():
    """Grid2D should return all actions as valid (no masking needed)."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp")

    position = torch.tensor([3, 3], dtype=torch.long)
    valid_actions = substrate.get_valid_actions(position)

    # All 5 actions should be valid
    assert valid_actions == [0, 1, 2, 3, 4]
    assert len(valid_actions) == substrate.action_space_size
```

**Run test**:
```bash
uv run pytest tests/test_townlet/unit/test_action_masking.py::test_grid2d_all_actions_valid -v
```

**Expected**: PASS

**Commit**:
```bash
git add src/townlet/substrate/base.py tests/test_townlet/unit/test_action_masking.py
git commit -m "feat(substrate): add action masking interface

Adds get_valid_actions() method to SpatialSubstrate base class.

**Purpose**:
- Support variable action spaces (needed for graph substrate)
- Enable action masking during training

**Default Behavior**:
- Returns all actions as valid (no masking)
- Grid substrates (2D/3D/ND) use default (all actions always valid)
- Graph substrates override to return only valid edges

**Testing**:
- Added test_action_masking.py
- Verified Grid2D returns all actions valid

Part of TASK-002A Phase 10 Task 1 (Graph Substrate)."
```

---

### Step 1.2: Write comprehensive graph tests (3-4 hours)

**Purpose**: TDD - define graph behavior through 15 comprehensive unit tests

**Create**: `tests/test_townlet/unit/test_substrate_graph.py`

(Full test suite from original plan - 15 tests covering initialization, action masking, movement, distance, etc.)

**Run tests**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_graph.py -v
```

**Expected**: ALL tests FAIL (not yet implemented - TDD!)

---

### Step 1.3: Implement GraphSubstrate (10-14 hours)

**Purpose**: Implement graph substrate to pass all tests

**Create**: `src/townlet/substrate/graph.py` (~312 lines)

(Full implementation from original plan with adjacency list, shortest paths, action masking)

**Key Methods**:
- `get_valid_actions()` - Returns valid edge indices + INTERACT per node
- `apply_action()` - Edge traversal movement (not delta-based)
- `compute_distance()` - Shortest path length (BFS precomputed)
- `_compute_shortest_paths()` - All-pairs shortest paths via BFS

**Run tests**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_graph.py -v
```

**Expected**: ALL 15 tests PASS

**Commit**:
```bash
git add src/townlet/substrate/graph.py
git commit -m "feat(substrate): implement GraphSubstrate class

Implements graph-based spatial substrate with edge traversal.

**Features**:
- Graph nodes as positions (node IDs)
- Edge traversal movement (action-dependent, not delta-dependent)
- Variable action spaces (requires action masking)
- Shortest path distances (BFS precomputation)
- Directed/undirected graph support

**Testing**:
- 15 unit tests (all passing)
- Adjacency list construction
- Action masking per node
- Movement along edges
- Distance computation

Part of TASK-002A Phase 10 Task 1 (Graph Substrate)."
```

---

### Step 1.4: Wire up action masking in environment (6-8 hours)

**Purpose**: Integrate action masking into training loop

**This is the most complex step** - requires changes across multiple files.

**Note**: This implements the core integration. Step 1.4b adds debugging tools (4-6h additional).

**Modify**: `src/townlet/environment/vectorized_env.py`

Add `get_valid_action_masks()` method:

```python
def get_valid_action_masks(self) -> torch.Tensor:
    """Get valid action masks for all agents.

    Returns:
        [num_agents, action_space_size] boolean tensor
        True = action is valid, False = action is invalid
    """
    action_space_size = self.substrate.action_space_size
    masks = torch.zeros(
        (self.num_agents, action_space_size),
        dtype=torch.bool,
        device=self.device
    )

    for i in range(self.num_agents):
        valid_actions = self.substrate.get_valid_actions(self.positions[i])
        masks[i, valid_actions] = True

    return masks
```

**Modify**: `src/townlet/population/vectorized.py`

Update `select_actions()` to use masking:

```python
def select_actions(
    self,
    observations: torch.Tensor,
    valid_action_masks: torch.Tensor | None = None,
    epsilon: float = 0.0
) -> torch.Tensor:
    """Select actions with optional action masking.

    Args:
        observations: [num_agents, obs_dim] observations
        valid_action_masks: [num_agents, action_dim] boolean mask (optional)
        epsilon: Epsilon-greedy exploration rate

    Returns:
        [num_agents] action indices
    """
    with torch.no_grad():
        q_values = self.q_network(observations)  # [num_agents, action_dim]

        # Apply action masking if provided
        if valid_action_masks is not None:
            # Set invalid actions to -inf (will never be selected)
            q_values[~valid_action_masks] = -float('inf')

        # Epsilon-greedy with masking
        if epsilon > 0:
            random_mask = torch.rand(len(observations), device=self.device) < epsilon

            # For random actions, sample uniformly from valid actions only
            if valid_action_masks is not None:
                for i in range(len(observations)):
                    if random_mask[i]:
                        valid_actions = torch.where(valid_action_masks[i])[0]
                        if len(valid_actions) > 0:
                            q_values[i, :] = -float('inf')
                            q_values[i, valid_actions] = torch.rand(len(valid_actions))
            else:
                # Standard epsilon-greedy (no masking)
                q_values[random_mask] = torch.rand_like(q_values[random_mask])

        # Greedy selection (max Q-value among valid actions)
        actions = q_values.argmax(dim=1)

    return actions
```

**Add integration test**:

```python
def test_action_masking_in_training_loop():
    """Action masking should work in full training loop."""
    from townlet.substrate.graph import GraphSubstrate
    from townlet.environment.vectorized_env import VectorizedHamletEnv
    
    # Simple graph
    edges = [(0, 1), (1, 2)]
    substrate = GraphSubstrate(num_nodes=3, edges=edges, directed=False)
    
    env = VectorizedHamletEnv(substrate=substrate, num_agents=4, ...)
    
    # Reset
    obs = env.reset()
    
    # Get valid action masks
    masks = env.get_valid_action_masks()
    
    # Verify masks are correct
    for i in range(env.num_agents):
        position = env.positions[i]
        expected_valid = substrate.get_valid_actions(position)
        actual_valid = torch.where(masks[i])[0].tolist()
        assert set(actual_valid) == set(expected_valid)
```

**Commit**:
```bash
git add src/townlet/environment/vectorized_env.py \
        src/townlet/population/vectorized.py \
        tests/test_townlet/integration/test_action_masking_integration.py

git commit -m "feat(training): integrate action masking into training loop

Action masking now fully integrated across environment and population.

**Features**:
- get_valid_action_masks() in VectorizedHamletEnv
- Masked action selection in VectorizedPopulation
- Masked epsilon-greedy exploration
- Invalid actions never selected

**Testing**:
- Integration test verifies masking works end-to-end
- Tested with graph substrate (variable action spaces)

Part of TASK-002A Phase 10 Task 1 (Graph Substrate)."
```

---

### Step 1.4b: Action masking debugging tools (4-6 hours) - ADDED POST-RISK-ASSESSMENT

**Problem**: Action masking bugs are silent failures. Agent selects invalid action → environment crashes, but unclear why.

**Solution**: Add debugging tools to diagnose action masking issues.

**Purpose**: Reduce debugging time when action masking fails (estimated 4-6h saved in Step 1.6).

**Create**: `src/townlet/debug/action_masking.py`

```python
"""Debugging utilities for action masking."""
import torch
from townlet.substrate.base import SpatialSubstrate
from townlet.environment.vectorized_env import VectorizedHamletEnv


class ActionMaskingValidator:
    """Validates action masking correctness."""

    def __init__(self, env: VectorizedHamletEnv):
        self.env = env
        self.substrate = env.substrate

    def validate_masks(self, verbose: bool = False) -> dict:
        """Validate action masks for all agents.

        Returns:
            dict with validation results:
            - all_valid: bool (True if all masks correct)
            - errors: list of error messages
            - warnings: list of warnings
        """
        errors = []
        warnings = []

        masks = self.env.get_valid_action_masks()  # [num_agents, action_dim]

        for i in range(self.env.num_agents):
            position = self.env.positions[i]
            mask = masks[i]

            # Get expected valid actions from substrate
            expected_valid = self.substrate.get_valid_actions(position)

            # Convert mask to list of valid actions
            actual_valid = torch.where(mask)[0].tolist()

            # Compare
            if set(actual_valid) != set(expected_valid):
                errors.append(
                    f"Agent {i} at {position.tolist()}: "
                    f"expected {expected_valid}, got {actual_valid}"
                )

            # Check for always-invalid INTERACT (common bug)
            interact_action = self.substrate.action_space_size - 1
            if interact_action not in actual_valid:
                warnings.append(
                    f"Agent {i}: INTERACT action always invalid (likely bug)"
                )

            if verbose:
                print(f"Agent {i} @ {position.tolist()}: valid={actual_valid}")

        return {
            'all_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
        }

    def trace_invalid_action(self, agent_id: int, action: int) -> str:
        """Generate detailed trace when agent selects invalid action.

        Args:
            agent_id: Agent index
            action: Action that was selected

        Returns:
            Detailed error message for debugging
        """
        position = self.env.positions[agent_id]
        mask = self.env.get_valid_action_masks()[agent_id]
        valid_actions = torch.where(mask)[0].tolist()
        expected_valid = self.substrate.get_valid_actions(position)

        q_values = self.env.population.q_network(
            self.env.get_observation(agent_id).unsqueeze(0)
        )[0]

        trace = f"""
=== INVALID ACTION SELECTED ===
Agent: {agent_id}
Position: {position.tolist()}
Action: {action}

Expected valid actions (from substrate): {expected_valid}
Actual mask (from environment): {valid_actions}
Mask mismatch: {set(expected_valid) != set(valid_actions)}

Q-values:
{q_values.tolist()}

Q-values after masking:
{(q_values * mask + ~mask * -float('inf')).tolist()}

Selected action Q-value: {q_values[action].item():.3f}
Action is valid: {mask[action].item()}

Possible causes:
1. Environment mask wrong (check get_valid_action_masks())
2. Substrate get_valid_actions() wrong
3. Epsilon-greedy not respecting mask
4. Q-network output shape mismatch
"""
        return trace


def validate_action_selection_step(
    env: VectorizedHamletEnv,
    actions: torch.Tensor,
    masks: torch.Tensor,
) -> None:
    """Assert all selected actions are valid (for testing).

    Raises:
        AssertionError: If any agent selected invalid action
    """
    validator = ActionMaskingValidator(env)

    for i in range(len(actions)):
        action = actions[i].item()
        valid_actions = torch.where(masks[i])[0].tolist()

        if action not in valid_actions:
            trace = validator.trace_invalid_action(i, action)
            raise AssertionError(f"Invalid action selected!\n{trace}")
```

**Add to training loop** (optional, for debugging):

```python
# In src/townlet/demo/runner.py (optional debugging mode)
def run_with_validation(self):
    """Run training with action masking validation (DEBUG MODE)."""
    from townlet.debug.action_masking import ActionMaskingValidator

    validator = ActionMaskingValidator(self.env)

    for episode in range(self.max_episodes):
        obs = self.env.reset()

        # Validate masks after reset
        result = validator.validate_masks(verbose=False)
        if not result['all_valid']:
            logger.error(f"Episode {episode}: Invalid masks! {result['errors']}")

        for step in range(self.max_steps_per_episode):
            masks = self.env.get_valid_action_masks()
            actions = self.population.select_actions(obs, masks, epsilon=self.epsilon)

            # Validate actions before step
            for i in range(len(actions)):
                if actions[i].item() not in torch.where(masks[i])[0].tolist():
                    trace = validator.trace_invalid_action(i, actions[i].item())
                    raise RuntimeError(f"Invalid action!\n{trace}")

            obs, rewards, dones = self.env.step(actions)
```

**Add tests**:

```python
def test_action_masking_validator():
    """Validator should catch mask errors."""
    from townlet.substrate.graph import GraphSubstrate
    from townlet.environment.vectorized_env import VectorizedHamletEnv
    from townlet.debug.action_masking import ActionMaskingValidator

    edges = [(0, 1), (1, 2), (2, 0)]
    substrate = GraphSubstrate(num_nodes=3, edges=edges, directed=False)
    env = VectorizedHamletEnv(substrate=substrate, num_agents=4, ...)

    validator = ActionMaskingValidator(env)

    # Should validate correctly
    result = validator.validate_masks()
    assert result['all_valid']
    assert len(result['errors']) == 0


def test_trace_invalid_action():
    """Trace should provide detailed debugging info."""
    from townlet.debug.action_masking import ActionMaskingValidator

    # ... setup env ...

    validator = ActionMaskingValidator(env)

    # Simulate invalid action
    trace = validator.trace_invalid_action(agent_id=0, action=99)

    # Should contain key debugging info
    assert 'Position:' in trace
    assert 'Q-values:' in trace
    assert 'Possible causes:' in trace
```

**Commit**:
```bash
git add src/townlet/debug/action_masking.py \
        tests/test_townlet/unit/test_action_masking_debug.py

git commit -m "feat(debug): add action masking debugging tools

Debugging tools reduce time spent diagnosing masking issues.

**Problem**:
- Action masking bugs are silent failures
- Hard to diagnose why agent selects invalid action
- Integration debugging can take 4-6h without tools

**Solution**:
- ActionMaskingValidator class
- validate_masks() - check all agent masks correct
- trace_invalid_action() - detailed trace when bugs occur
- Validation assertions for testing

**Rationale**:
- Risk assessment identified debugging as underestimated
- These tools save 4-6h during Step 1.6 integration testing

Part of TASK-002A Phase 10 Task 1 (Graph Substrate)."
```

---

### Step 1.5: Config support + config pack (4-6 hours)

**Modify**: `src/townlet/substrate/config.py`

Add graph configuration:

```python
@dataclass
class GraphSubstrateConfig:
    """Configuration for graph-based substrate."""
    type: Literal["graph"]
    num_nodes: int
    edges: list[tuple[int, int]]
    directed: bool = False
    max_edges: int | None = None
    distance_metric: Literal["shortest_path"] = "shortest_path"

    def __post_init__(self):
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if not self.edges:
            raise ValueError("edges list cannot be empty")
```

**Modify**: `src/townlet/substrate/factory.py`

```python
elif config.type == "graph":
    from townlet.substrate.graph import GraphSubstrate
    return GraphSubstrate(
        num_nodes=config.num_nodes,
        edges=config.edges,
        directed=config.directed,
        max_edges=config.max_edges,
    )
```

**Create**: `configs/L1_graph_subway/substrate.yaml`

```yaml
substrate:
  type: graph
  num_nodes: 16
  
  # Simple subway network
  edges:
    - [0, 1]   # Line 1
    - [1, 2]
    - [2, 3]
    - [3, 4]
    - [5, 6]   # Line 2
    - [6, 7]
    - [7, 8]
    - [1, 6]   # Transfer stations
    - [3, 7]
    - [9, 10]  # Line 3
    - [10, 11]
    - [11, 12]
    - [2, 10]  # Transfers
    - [13, 14] # Line 4
    - [14, 15]
    - [4, 14]
    - [8, 15]
  
  directed: false
  distance_metric: shortest_path
```

Copy other configs from L1_full_observability.

**Test training**:
```bash
uv run scripts/run_demo.py --config configs/L1_graph_subway --max-episodes 50
```

**Expected**: Training runs successfully with graph substrate

**Commit**:
```bash
git add src/townlet/substrate/config.py \
        src/townlet/substrate/factory.py \
        configs/L1_graph_subway/

git commit -m "feat(config): add graph substrate configuration

Graph substrate now configurable via YAML.

**Features**:
- GraphSubstrateConfig dataclass
- Edge list parsing from YAML
- Example config: L1_graph_subway (16-station subway)

**Testing**:
- Config validation (num_nodes, edges)
- Training runs successfully

Part of TASK-002A Phase 10 Task 1 (Graph Substrate)."
```

---

### Step 1.6: Integration tests + final commit (6-8 hours) - REVISED from 4-6h

**⚠️ COMPLEXITY INCREASED**: Risk assessment identified more integration testing needed.

**Why More Time?**
- **Debugging tools** (Step 1.4b) save time BUT expose more bugs to fix
- **Cross-substrate testing**: Must verify Graph doesn't break Hex/1D
- **Action masking edge cases**: Zero-degree nodes, self-loops, disconnected graphs
- **Performance testing**: Graph operations on large graphs (100+ nodes)

**Time allocation**:
- Integration tests: 3-4h
- Cross-substrate regression: 1-2h
- Edge case debugging: 1-2h
- Performance validation: 1h

**Create**: `tests/test_townlet/integration/test_graph_training.py`

(3 comprehensive integration tests from original plan)

**Run all tests**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_graph.py -v
uv run pytest tests/test_townlet/unit/test_action_masking.py -v
uv run pytest tests/test_townlet/integration/test_graph_training.py -v
```

**Expected**: ALL tests PASS

**Final commit**:
```bash
git add tests/test_townlet/integration/test_graph_training.py

git commit -m "feat(substrate): complete graph substrate implementation

Graph substrate fully implemented with action masking.

**Summary**:
- GraphSubstrate class (312 lines)
- Action masking infrastructure
- Training loop integration
- Config system support
- Comprehensive testing (15 unit + 3 integration)

**Features**:
- Graph nodes as positions
- Edge traversal movement
- Variable action spaces (masked RL)
- Shortest path distances
- Directed/undirected graphs

**Pedagogical Value** (✅✅✅):
- Teaches graph RL and non-Euclidean reasoning
- Teaches action masking (variable action spaces)
- Use cases: Subway systems, social networks, state machines

**Testing**:
- 15 unit tests (all passing)
- 3 integration tests (all passing)
- Config pack: L1_graph_subway

Part of TASK-002A Phase 10 Task 1 (Graph Substrate - COMPLETE)."
```

---

## Task 10.2: Frontend Visualization (10-14 hours) - REVISED from 8-12h

### Overview

**CRITICAL OMISSION from original plan.** Risk assessment revised estimate to account for:
- Layout complexity: Force-directed layout more complex than circular
- Action masking overlay: Interactive debugging UI (+2h)
- WebSocket protocol changes: Send graph topology data (+1h)
- Testing and debugging: Graph rendering bugs harder to diagnose

Frontend needs:
1. Graph rendering (nodes + edges)
2. Action masking visualization
3. Agent movement on graph
4. Subway-style layout

**Files to Create**:
- `frontend/src/components/GraphVisualization.vue` (new)
- `frontend/src/components/ActionMaskingOverlay.vue` (new)

**Files to Modify**:
- `frontend/src/App.vue` (add graph rendering mode)
- `frontend/src/websocket.js` (handle graph topology data)

---

### Step 2.1: Graph rendering component (5-6 hours) - REVISED from 4-5h

**Complexity increased**: Force-directed layout harder than circular layout (+1h)

**Create**: `frontend/src/components/GraphVisualization.vue`

```vue
<template>
  <svg :width="width" :height="height" class="graph-viz">
    <!-- Edges -->
    <line
      v-for="(edge, idx) in edges"
      :key="`edge-${idx}`"
      :x1="nodePositions[edge[0]].x"
      :y1="nodePositions[edge[0]].y"
      :x2="nodePositions[edge[1]].x"
      :y2="nodePositions[edge[1]].y"
      stroke="#888"
      stroke-width="2"
    />

    <!-- Nodes -->
    <circle
      v-for="(pos, nodeId) in nodePositions"
      :key="`node-${nodeId}`"
      :cx="pos.x"
      :cy="pos.y"
      :r="nodeRadius"
      :fill="getNodeColor(nodeId)"
      stroke="#333"
      stroke-width="2"
    />

    <!-- Node labels -->
    <text
      v-for="(pos, nodeId) in nodePositions"
      :key="`label-${nodeId}`"
      :x="pos.x"
      :y="pos.y + 5"
      text-anchor="middle"
      fill="white"
      font-size="14"
    >
      {{ nodeId }}
    </text>

    <!-- Agents -->
    <circle
      v-for="(agent, idx) in agents"
      :key="`agent-${idx}`"
      :cx="getAgentPosition(agent).x"
      :cy="getAgentPosition(agent).y"
      r="8"
      fill="#00ff00"
      stroke="#fff"
      stroke-width="2"
    />
  </svg>
</template>

<script>
export default {
  name: 'GraphVisualization',
  props: {
    numNodes: Number,
    edges: Array,
    agents: Array,
    width: { type: Number, default: 800 },
    height: { type: Number, default: 600 },
  },
  data() {
    return {
      nodeRadius: 20,
      nodePositions: {},
    };
  },
  mounted() {
    this.computeNodeLayout();
  },
  methods: {
    computeNodeLayout() {
      // Force-directed layout or subway-style layout
      // (Simplified: circular layout for demo)
      const centerX = this.width / 2;
      const centerY = this.height / 2;
      const radius = Math.min(this.width, this.height) / 3;

      for (let i = 0; i < this.numNodes; i++) {
        const angle = (i / this.numNodes) * 2 * Math.PI;
        this.nodePositions[i] = {
          x: centerX + radius * Math.cos(angle),
          y: centerY + radius * Math.sin(angle),
        };
      }
    },
    getNodeColor(nodeId) {
      // Color by affordance or agent presence
      return '#555';
    },
    getAgentPosition(agent) {
      const nodeId = agent.position[0];
      return this.nodePositions[nodeId];
    },
  },
};
</script>
```

---

### Step 2.2: Action masking overlay (3-4 hours) - REVISED from 2-3h

**Complexity increased**: Interactive debugging UI more complex than static overlay (+1h)

**Create**: `frontend/src/components/ActionMaskingOverlay.vue`

```vue
<template>
  <div class="action-mask-overlay">
    <div v-if="selectedAgent !== null" class="mask-info">
      <h3>Agent {{ selectedAgent }} - Valid Actions</h3>
      <div
        v-for="(valid, actionIdx) in validActions"
        :key="actionIdx"
        :class="['action-item', { valid, invalid: !valid }]"
      >
        Action {{ actionIdx }}: {{ valid ? '✓ Valid' : '✗ Invalid' }}
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ActionMaskingOverlay',
  props: {
    selectedAgent: Number,
    validActions: Array,  // [action_space_size] boolean array
  },
};
</script>

<style scoped>
.action-item.valid {
  color: green;
}
.action-item.invalid {
  color: red;
}
</style>
```

---

### Step 2.3: Integration + testing (2-4 hours) - UNCHANGED

**No change**: Original estimate sufficient

**Modify**: `frontend/src/App.vue`

Add graph visualization mode:

```vue
<template>
  <div id="app">
    <!-- Grid visualization (existing) -->
    <GridVisualization v-if="substrateType === 'grid'" :agents="agents" />
    
    <!-- Graph visualization (NEW) -->
    <GraphVisualization
      v-if="substrateType === 'graph'"
      :num-nodes="numNodes"
      :edges="edges"
      :agents="agents"
    />
    
    <!-- Action masking overlay (NEW) -->
    <ActionMaskingOverlay
      :selected-agent="selectedAgent"
      :valid-actions="validActions"
    />
  </div>
</template>
```

**Test**:
```bash
# Terminal 1: Start inference server with graph config
cd /path/to/worktree
python -m townlet.demo.live_inference \
  checkpoints_graph 8766 0.2 1000 configs/L1_graph_subway/training.yaml

# Terminal 2: Start frontend
cd frontend
npm run dev
```

**Expected**: Graph rendered with nodes, edges, agents moving along edges

**Commit**:
```bash
git add frontend/src/components/GraphVisualization.vue \
        frontend/src/components/ActionMaskingOverlay.vue \
        frontend/src/App.vue \
        frontend/src/websocket.js

git commit -m "feat(frontend): add graph substrate visualization

Frontend now renders graph substrates with action masking.

**Features**:
- Graph visualization (nodes + edges)
- Action masking overlay (valid/invalid actions)
- Agent movement on graph
- Subway-style layout

**Testing**:
- Tested with L1_graph_subway config
- Verified rendering and agent movement

Part of TASK-002A Phase 10 Task 2 (Frontend Visualization)."
```

---

## Task 10.3: Documentation (2-3 hours)

### Step 3.1: Update CLAUDE.md (1 hour)

**Modify**: `CLAUDE.md`

Add Phase 10 section:

```markdown
### Phase 10: Graph Substrate + Infrastructure

**Graph-Based Substrate** (configs/L1_graph_subway/):
- Graph nodes as positions (node IDs)
- Edge traversal movement
- Variable action spaces (requires action masking)
- Use cases: Subway systems, social networks, state machines
- Action space: Variable (max_edges + INTERACT)

**Infrastructure Changes**:
- Action mapping refactored (substrate delegation)
- Q-network dynamic sizing (based on substrate)
- Replay buffer supports action masking
- Frontend graph visualization

**Key Concepts**:
- **Action Masking**: Variable action spaces per state
- **Graph RL**: Non-Euclidean spatial reasoning
- **Topological Distance**: Shortest path (not Euclidean)
```

---

### Step 3.2: Commit documentation (30 min)

```bash
git add CLAUDE.md

git commit -m "docs: add Phase 10 documentation

Added documentation for Graph substrate and infrastructure.

**Phase 10 Summary**:
- Graph substrate with action masking
- Infrastructure: dynamic action spaces, replay buffer masking
- Frontend: graph visualization

**Estimated Effort**: 68-92 hours total
**Actual Effort**: [To be filled after implementation]

Part of TASK-002A Phase 10 (Graph Substrate + Infrastructure - COMPLETE)."
```

---

## Phase 10 Completion Checklist - REVISED POST-RISK-ASSESSMENT

### Pre-Work (4-6 hours) - ADDED
- [ ] Position extraction design (2h)
- [ ] Position extraction implementation (2-3h)
- [ ] Position extraction testing (1h)

### Infrastructure (14-18 hours) - REVISED from 7-10h
- [ ] Step 0.0: Feature flag for action masking (30min) - ADDED
- [ ] Step 0.1: Action mapping refactor (3-4h)
- [ ] Step 0.2: Q-network dynamic sizing (2-3h)
- [ ] Step 0.3: Replay buffer schema (7-11h) - REVISED from 2-3h
- [ ] Step 0.4: Infrastructure testing (2-3h) - ADDED
- [ ] Step 0.5: Checkpoint versioning (2h) - ADDED

### Graph Substrate (44-58 hours) - REVISED from 36-48h
- [ ] Step 1.1: Action masking interface (1h)
- [ ] Step 1.2: Graph unit tests (3-4h)
- [ ] Step 1.3: Implement GraphSubstrate (10-14h)
- [ ] Step 1.4: Action masking integration (6-8h)
- [ ] Step 1.4b: Action masking debugging tools (4-6h) - ADDED
- [ ] Step 1.5: Config + config pack (4-6h)
- [ ] Step 1.6: Integration tests (6-8h) - REVISED from 4-6h

### Frontend (10-14 hours) - REVISED from 8-12h
- [ ] Step 2.1: Graph rendering (5-6h) - REVISED from 4-5h
- [ ] Step 2.2: Action masking overlay (3-4h) - REVISED from 2-3h
- [ ] Step 2.3: Integration + testing (2-4h)

### Documentation (2-3 hours)
- [ ] Step 3.1: Update CLAUDE.md (1h)
- [ ] Step 3.2: Commit docs (30min)

**Total Estimated Effort**: 90-120 hours (REVISED from 68-92h)

---

## Success Criteria - REVISED POST-RISK-ASSESSMENT

Phase 10 is complete when:

**Pre-Work:**
- [ ] Position extraction design documented
- [ ] ObservationBuilder.extract_position() implemented
- [ ] Round-trip tests pass (encode→extract returns original position)
- [ ] Works for ALL substrate types (1D, 2D, 3D, Hex, ND, Graph)

**Infrastructure:**
- [ ] Feature flag for action masking working (backward compatible)
- [ ] Action mapping delegated to substrates (1D, 2D, 3D, Hex all work)
- [ ] Q-network sizes dynamically to substrate
- [ ] Replay buffer stores/retrieves valid actions
- [ ] Checkpoint versioning implemented (v1→v2 migration works)
- [ ] All infrastructure tests pass
- [ ] No regressions in existing substrates

**Graph Substrate:**
- [ ] All unit tests pass (15/15)
- [ ] All integration tests pass (3/3)
- [ ] Action masking working correctly (no invalid actions selected)
- [ ] Debugging tools validate masks correctly
- [ ] Training runs on graph substrate
- [ ] Config pack L1_graph_subway created
- [ ] Performance acceptable on 100+ node graphs

**Frontend:**
- [ ] Graph visualization renders correctly
- [ ] Action masking overlay shows valid/invalid actions
- [ ] Interactive debugging UI works (click agent → see valid actions)
- [ ] Agents move along edges visually
- [ ] Works with live inference server
- [ ] WebSocket sends graph topology data

**Integration:**
- [ ] No regressions in Phase 9 substrates (Hex, 1D)
- [ ] No regressions in existing substrates (2D, 3D, ND)
- [ ] All topologies coexist peacefully
- [ ] Feature flag=false works (backward compatibility)
- [ ] Old checkpoints migrate correctly (v1→v2)
- [ ] Documentation complete

---

**Phase 10 Status**: Ready for Implementation (After Phase 9 + Pre-Work)
**Recommended Order**: Pre-Work → Infrastructure → Graph → Frontend → Docs
**Total Effort**: 90-120 hours (risk assessment revised estimate, up from 68-92h)
**Dependencies**: Phase 9 must be complete first + Position Extraction Design (4-6h)
**Risk Level**: MEDIUM-HIGH (complex action masking, breaking changes, frontend work)

**Revision History**:
- Original estimate: 40-60h (naive, underestimated action masking)
- Peer review: 68-92h (identified action masking complexity)
- Risk assessment: 90-120h (identified missing tasks, debugging tools, testing gaps)

**Next Steps After Phase 10**:
- Phase 11: Multi-zone environments
- Phase 12: Multi-agent competition
