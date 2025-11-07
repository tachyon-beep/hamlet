# Sprints 15-16: Core Module Test Coverage

**Date**: 2025-11-07
**Status**: 🎯 **PLANNING**
**Priority**: HIGH (User-requested core modules)
**Related**: QUICK-004-TEST-REMEDIATION.md Phase 3

---

## Objective

Improve test coverage for two critical training modules from 6-9% to 70%+:
1. `src/townlet/environment/vectorized_env.py` (1012 lines, 6% coverage)
2. `src/townlet/population/vectorized.py` (868 lines, 9% coverage)

These are **core training modules** - the heart of the Townlet system.

---

## Current State

### Existing Tests

**Integration tests exist** (but no dedicated unit tests):
- `tests/test_townlet/integration/test_training_loop.py`
- `tests/test_townlet/integration/test_episode_execution.py`
- `tests/test_townlet/integration/test_checkpointing.py`
- `tests/test_townlet/integration/test_recurrent_networks.py`
- `tests/test_townlet/integration/test_data_flows.py`

**Problem**: Integration tests test the **system as a whole**, not individual methods. We need **unit tests** to cover specific logic paths.

---

## Module Analysis

### VectorizedHamletEnv (vectorized_env.py)

**Size**: 1012 lines
**Current Coverage**: 6%
**Key Methods** (18 methods identified):

**Initialization**:
- `__init__` (lines 36-360) - Complex initialization with substrate, affordances, meters
- `attach_runtime_registry` - Agent registry attachment

**Core Loop**:
- `reset()` - Reset environment state
- `step(actions)` - Main environment step (execute actions, calculate rewards, check done)
- `_execute_actions(actions)` - Action execution logic
- `_handle_interactions(interact_mask)` - Affordance interactions (multi-tick)
- `_handle_interactions_legacy(interact_mask)` - Legacy single-tick interactions

**Observation & Actions**:
- `_get_observations()` - Build observations (full or POMDP)
- `get_action_masks()` - Action availability masks
- `_build_movement_deltas()` - Movement delta tensors
- `get_action_label_names()` - Action label mapping

**Rewards**:
- `_calculate_shaped_rewards()` - Reward calculation

**State Management**:
- `get_affordance_positions()` - Affordance layout for checkpoints
- `set_affordance_positions(data)` - Load affordance layout from checkpoint
- `randomize_affordance_positions()` - Randomize affordance placement

**Custom Actions**:
- `_apply_custom_action(idx, action)` - Execute custom actions (REST, MEDITATE)
- `_get_optional_action_idx(name)` - Find action by name
- `_get_meter_index(name)` - Find meter by name

---

### VectorizedPopulation (population/vectorized.py)

**Size**: 868 lines
**Current Coverage**: 9%
**Key Methods** (20+ methods identified):

**Initialization**:
- `__init__` - Initialize Q-network, optimizer, replay buffer, exploration

**Episode Management**:
- `reset()` - Reset all agents
- `_new_episode_container()` - Create episode container
- `_store_episode_and_reset(idx)` - Store completed episode
- `_reset_hidden_state(idx)` - Reset LSTM hidden state (if using recurrent network)
- `_finalize_episode(idx, survival)` - Finalize episode with metrics
- `flush_episode(idx)` - Flush episode to database

**Action Selection**:
- `select_greedy_actions(env)` - Greedy policy (max Q-value)
- `select_epsilon_greedy_actions(env, epsilon)` - ε-greedy exploration

**Training Loop**:
- `step_population(env, curriculum, exploration)` - Main training step
  - Action selection
  - Environment step
  - Experience storage
  - Q-network training
  - Target network updates
  - Intrinsic reward calculation
  - Episode finalization

**Curriculum & Exploration**:
- `_sync_curriculum_metrics()` - Sync curriculum state
- `sync_exploration_metrics()` - Sync exploration state
- `update_curriculum_tracker(rewards, dones)` - Update curriculum tracker
- `_get_current_epsilon_value()` - Get epsilon from exploration
- `_get_current_intrinsic_weight_value()` - Get intrinsic weight
- `_difficulty_to_stage(difficulty)` - Map difficulty to stage

**Metrics & Telemetry**:
- `build_telemetry_snapshot(episode)` - Build telemetry data
- `get_training_metrics()` - Get training metrics

**Checkpointing**:
- `get_checkpoint()` - Get checkpoint (PopulationCheckpoint)
- `get_checkpoint_state()` - Get checkpoint dict
- `load_checkpoint_state(data)` - Load from checkpoint

---

## Testing Strategy

### Principles

1. **Unit tests over integration tests** - Test individual methods in isolation
2. **Mock heavy dependencies** - Mock Q-network, environment, curriculum, exploration
3. **Use builders** - Leverage builders.py for test data
4. **Focus on logic, not plumbing** - Test decision logic, not tensor operations
5. **Cover error paths** - Test validation, edge cases, error handling

### Sprint 15: VectorizedHamletEnv Tests

**Target File**: `tests/test_townlet/unit/environment/test_vectorized_env.py`
**Goal**: 6% → 70%+ coverage

**Priority Methods** (in order):

**Phase 15A: Initialization & Setup**
1. `__init__` - Test config loading, substrate creation, affordance engine setup
2. `reset()` - Test state reset, position randomization
3. `_build_movement_deltas()` - Test delta tensor construction

**Phase 15B: Core Loop**
4. `step(actions)` - Test main step logic (mocked environment)
5. `_execute_actions(actions)` - Test action execution (movement, wait, interact)
6. `_get_observations()` - Test observation building (full observability)
7. `get_action_masks()` - Test action masking logic

**Phase 15C: Interactions & Rewards**
8. `_handle_interactions(mask)` - Test multi-tick interactions
9. `_handle_interactions_legacy(mask)` - Test single-tick interactions
10. `_calculate_shaped_rewards()` - Test reward calculation
11. `_apply_custom_action(idx, action)` - Test custom actions (REST, MEDITATE)

**Phase 15D: Checkpointing**
12. `get_affordance_positions()` - Test affordance position export
13. `set_affordance_positions(data)` - Test affordance position loading
14. `randomize_affordance_positions()` - Test randomization

**Expected Tests**: 25-35 unit tests
**Expected Lines**: 800-1200 lines
**Expected Coverage**: 70-80%

---

### Sprint 16: VectorizedPopulation Tests

**Target File**: `tests/test_townlet/unit/population/test_vectorized.py`
**Goal**: 9% → 70%+ coverage

**Priority Methods** (in order):

**Phase 16A: Initialization & Episode Management**
1. `__init__` - Test Q-network, optimizer, buffer initialization
2. `reset()` - Test agent reset
3. `_new_episode_container()` - Test episode container creation
4. `_store_episode_and_reset(idx)` - Test episode storage
5. `_reset_hidden_state(idx)` - Test LSTM state reset

**Phase 16B: Action Selection**
6. `select_greedy_actions(env)` - Test greedy policy
7. `select_epsilon_greedy_actions(env, epsilon)` - Test ε-greedy

**Phase 16C: Training Loop** (most complex)
8. `step_population(env, curriculum, exploration)` - Test main training step
   - Action selection path
   - Experience replay path
   - Q-network update path
   - Target update path
   - Episode finalization path

**Phase 16D: Curriculum & Exploration**
9. `_sync_curriculum_metrics()` - Test curriculum sync
10. `sync_exploration_metrics()` - Test exploration sync
11. `update_curriculum_tracker(rewards, dones)` - Test curriculum update
12. `_get_current_epsilon_value()` - Test epsilon retrieval
13. `_get_current_intrinsic_weight_value()` - Test intrinsic weight retrieval

**Phase 16E: Checkpointing**
14. `get_checkpoint()` - Test checkpoint creation
15. `get_checkpoint_state()` - Test checkpoint export
16. `load_checkpoint_state(data)` - Test checkpoint loading

**Expected Tests**: 30-40 unit tests
**Expected Lines**: 900-1300 lines
**Expected Coverage**: 70-80%

---

## Mocking Strategy

### VectorizedHamletEnv Mocks

**Mock the following**:
- `SubstrateFactory.build()` - Return mock substrate
- `load_affordance_config()` - Return test affordance config
- `AffordanceEngine` - Mock affordance interactions
- `MeterDynamics` - Mock meter updates
- `ObservationBuilder` - Mock observation construction
- `RewardStrategy` - Mock reward calculation

**Use real tensors** for state (positions, meters, etc.) - these are data, not dependencies

### VectorizedPopulation Mocks

**Mock the following**:
- `SimpleQNetwork` / `RecurrentSpatialQNetwork` - Mock Q-network
- `torch.optim.Adam` - Mock optimizer
- `ReplayBuffer` - Mock experience replay
- `VectorizedHamletEnv` - Mock environment (from Sprint 15 tests)
- `AdversarialCurriculum` - Mock curriculum
- `AdaptiveIntrinsicExploration` - Mock exploration
- `DemoDatabase` - Mock database writes

**Use real tensors** for agent state, observations, actions, rewards

---

## Builder Support

**New builders needed**:

```python
# For VectorizedHamletEnv
def make_test_substrate_config(type="grid", width=8, height=8) -> SubstrateConfig
def make_test_affordance_config(num_affordances=3) -> AffordanceConfig
def make_test_action_space(substrate_type="grid") -> ComposedActionSpace

# For VectorizedPopulation
def make_test_q_network(obs_dim=29, action_dim=8) -> Mock  # Returns mocked network
def make_test_replay_buffer(capacity=1000) -> Mock
def make_test_episode_container(num_steps=10) -> EpisodeContainer
def make_test_population_checkpoint(episode=100) -> PopulationCheckpoint
```

---

## Success Criteria

**Sprint 15 (VectorizedHamletEnv)**:
- [ ] Create `tests/test_townlet/unit/environment/test_vectorized_env.py`
- [ ] 25-35 unit tests written
- [ ] Coverage: 6% → 70%+
- [ ] All tests passing
- [ ] Ruff compliant
- [ ] Zero integration test regressions

**Sprint 16 (VectorizedPopulation)**:
- [ ] Create `tests/test_townlet/unit/population/test_vectorized.py`
- [ ] 30-40 unit tests written
- [ ] Coverage: 9% → 70%+
- [ ] All tests passing
- [ ] Ruff compliant
- [ ] Zero integration test regressions

**Combined Impact**:
- [ ] 55-75 new unit tests
- [ ] 1700-2500 lines of test code
- [ ] 2 core modules with 70%+ coverage
- [ ] Improved confidence in training loop correctness

---

## Risks & Mitigations

**Risk 1**: Complex mocking may obscure real bugs
- **Mitigation**: Keep integration tests, only add unit tests for specific logic

**Risk 2**: Tests may be brittle (tightly coupled to implementation)
- **Mitigation**: Test behavior, not implementation details

**Risk 3**: GPU tensor operations hard to test
- **Mitigation**: Use CPU device for tests, focus on logic not performance

**Risk 4**: Large time investment (2 sprints)
- **Mitigation**: Prioritize high-value methods first, stop at 70% (not 100%)

---

## Next Steps

1. **Wait for coverage report** to see exactly what's currently untested
2. **Create builders** for test infrastructure
3. **Sprint 15A**: Start with VectorizedHamletEnv initialization tests
4. **Iterate**: Build tests incrementally, verify coverage improves

---

**Status**: 🎯 PLANNING (waiting for detailed coverage report)
**Estimated Effort**: 2 sprints (~100-150 tests total)
**Expected Value**: HIGH (core training modules tested)
