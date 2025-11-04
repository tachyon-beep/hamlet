# Integration Test Implementation Plan - HAMLET Townlet

**Date**: 2025-11-04
**Status**: PLANNING COMPLETE - READY FOR IMPLEMENTATION
**Based on**: RESEARCH-INTEGRATION-TEST-STRATEGY.md

---

## 1. Overview

### Mission Statement

After completing Tasks 1-10 (381 unit tests, all passing), we have **excellent component-level coverage** but lack **integration-level verification**. This plan details how to implement Tasks 11-13 to ensure components work correctly at their boundaries and across data flows.

### Key Insight

**71 integration tests already exist** but are scattered across multiple files. This plan:
1. **Consolidates** existing tests into logical structure
2. **Adds** new tests for identified coverage gaps
3. **Organizes** by integration concern (not component)

### Relationship to Unit Tests

| Aspect | Unit Tests (Tasks 1-10) | Integration Tests (Tasks 11-13) |
|--------|------------------------|--------------------------------|
| **Focus** | Component behavior in isolation | Component interactions at boundaries |
| **Scope** | Single component, mocked dependencies | Multiple components, real interactions |
| **Speed** | Fast (<10ms per test) | Slower (100ms-1s per test) |
| **Assertions** | Exact values, edge cases | Data flow, state consistency |
| **Status** | ✅ Complete (381 tests) | ⏳ In progress (71 existing, ~30 new) |

### Goals for Integration Testing

1. ✅ **Component contracts verified**: Do components communicate correctly?
2. ✅ **Data flow integrity**: Does information flow correctly through the pipeline?
3. ✅ **State synchronization**: Are states kept consistent across components?
4. ✅ **Episode lifecycle**: Does a full episode execute correctly end-to-end?
5. ✅ **Multi-episode training**: Do agents learn across multiple episodes?
6. ✅ **Checkpointing round-trips**: Can we save/load and resume training?
7. ✅ **Signal purity**: Does curriculum receive clean survival signals?
8. ✅ **Temporal mechanics**: Do time-based features work end-to-end?

---

## 2. Task Breakdown

### Task 11: Checkpointing & Signal Purity Integration

**Goal**: Migrate and consolidate existing checkpoint and signal purity tests into organized structure.

**What to migrate**:
- ✅ 38 checkpointing tests (P1.1 Phases 1-6, P2.1 baseline)
- ✅ 9 signal purity tests (test_curriculum_signal_purity.py)
- ✅ 3 runner integration tests (test_recording/test_runner_integration.py)

**New files to create**:

#### 11a. `integration/test_checkpointing.py` (38 tests → consolidate to ~15)

**Consolidation strategy**:
- Merge P1.1 Phase 1 (baseline) tests: Q-network + optimizer checkpointing
- Merge P1.1 Phase 2 (full checkpoint): Population state preservation
- Merge P1.1 Phase 3 (curriculum): Curriculum tracker state
- Merge P1.1 Phase 4 (affordances): Affordance layout persistence
- Merge P1.1 Phase 5 (flush): Episode flushing before checkpoint
- Merge P1.1 Phase 6 (agent_ids): Multi-agent agent_ids persistence
- Merge P2.1 tests: Per-agent reward baselines

**Test structure**:
```python
class TestCheckpointBaseline:
    """Core checkpoint functionality (Q-network + optimizer)."""
    - test_checkpoint_contains_required_keys
    - test_checkpoint_version_is_current
    - test_qnetwork_state_dict_preserved
    - test_optimizer_state_dict_preserved
    - test_total_steps_preserved

class TestCheckpointFullState:
    """Complete population state preservation."""
    - test_exploration_state_preserved
    - test_replay_buffer_preserved
    - test_curriculum_tracker_state_preserved
    - test_agent_ids_preserved

class TestCheckpointRoundTrip:
    """Save → Load → Verify equivalence."""
    - test_save_and_load_qnetwork_weights_identical
    - test_save_and_load_curriculum_stage_preserved
    - test_save_and_load_epsilon_preserved
    - test_save_and_load_replay_buffer_size_preserved

class TestCheckpointAffordances:
    """Affordance layout persistence for generalization tests."""
    - test_affordance_positions_preserved
    - test_load_checkpoint_restores_layout

class TestCheckpointEpisodeFlushing:
    """Episodes flushed before checkpoint (LSTM only)."""
    - test_flush_episodes_before_checkpoint_on_death
    - test_flush_episodes_before_checkpoint_on_max_steps
```

**Effort estimate**: 6 hours
- 3 hours: Consolidate 38 tests → ~15 tests
- 2 hours: Add checkpoint round-trip validation tests
- 1 hour: Verify all tests pass, add docstrings

#### 11b. `integration/test_curriculum_signal_purity.py` (9 tests → migrate as-is)

**Migration strategy**: Move existing file to `integration/` subdirectory

**Existing tests** (all ✅ already written):
```python
class TestCurriculumUpdateFrequency:
    - test_update_curriculum_tracker_exists
    - test_update_curriculum_tracker_accepts_rewards_and_dones
    - test_curriculum_tracker_state_updated_after_single_call

class TestSignalPurity:
    - test_survival_signal_is_integer_steps
    - test_survival_signal_not_contaminated_by_intrinsic_rewards
    - test_max_steps_survival_sends_done_signal

class TestSingleUpdatePerEpisode:
    - test_no_update_during_episode_steps
    - test_single_update_per_episode_on_death
    - test_signal_is_monotonic_with_survival_time
```

**New tests to add** (2 tests):
```python
class TestSignalPurityDuringTraining:
    - test_signal_purity_with_active_intrinsic_rewards (NEW)
    - test_curriculum_stage_changes_based_on_survival_not_rewards (NEW)
```

**Effort estimate**: 2 hours
- 0.5 hours: Move file to integration/
- 1 hour: Add 2 new tests for multi-episode signal purity
- 0.5 hours: Verify all tests pass

#### 11c. `integration/test_runner_integration.py` (3 tests → migrate + expand)

**Migration strategy**: Move from test_recording/ to integration/

**Existing tests** (all ✅ already written):
```python
class TestRunnerRecordingIntegration:
    - test_runner_with_recording_enabled
    - test_runner_without_recording
    - test_runner_recording_writes_to_database
```

**New tests to add** (3 tests):
```python
class TestRunnerOrchestration:
    - test_runner_episode_loop_execution (NEW)
    - test_runner_checkpoint_save_at_interval (NEW)
    - test_runner_database_logging_after_episode (NEW)
```

**Effort estimate**: 2 hours
- 0.5 hours: Move file to integration/
- 1 hour: Add 3 new orchestration tests
- 0.5 hours: Verify all tests pass

**Task 11 Total**: 10 hours

**Success criteria**:
- [ ] All 50 existing integration tests migrated to `integration/` subdirectory
- [ ] Checkpointing tests consolidated into logical structure (~15 tests)
- [ ] Signal purity tests moved and 2 new tests added
- [ ] Runner integration tests moved and 3 new tests added
- [ ] All tests pass consistently

---

### Task 12: Feature-Specific Integration Tests

**Goal**: Test specific features that span multiple components (temporal mechanics, LSTM, intrinsic exploration).

**New files to create**:

#### 12a. `integration/test_temporal_mechanics.py` (migrate 5 + add 5 = 10 tests)

**Migration strategy**: Consolidate existing test_temporal_integration.py + add new tests

**Existing tests** (5 tests):
```python
class TestTimeProgression:
    - test_24_hour_cycle_progression
    - test_time_wraps_at_midnight

class TestOperatingHours:
    - test_job_available_9am_to_5pm
    - test_bar_available_6pm_to_2am
    - test_closed_affordances_masked
```

**New tests to add** (5 tests):
```python
class TestMultiTickInteractions:
    - test_job_takes_4_ticks_to_complete (NEW)
    - test_early_exit_gives_partial_benefits (NEW)
    - test_per_tick_costs_accumulate (NEW)
    - test_interaction_progress_in_observations (NEW)
    - test_completion_bonus_awarded_at_end (NEW)
```

**Effort estimate**: 3 hours
- 1 hour: Migrate existing 5 tests
- 1.5 hours: Add 5 new multi-tick tests
- 0.5 hours: Verify temporal features in observations

#### 12b. `integration/test_recurrent_networks.py` (NEW - 8 tests)

**Goal**: Test LSTM hidden state management during episodes and training

**Test structure**:
```python
class TestLSTMHiddenStatePersistence:
    - test_hidden_state_persists_across_10_steps_within_episode (NEW)
    - test_hidden_state_resets_on_death (NEW)
    - test_hidden_state_resets_after_flush_on_max_steps (NEW)
    - test_hidden_state_shape_correct_during_episode (NEW)

class TestLSTMBatchTraining:
    - test_hidden_state_batch_size_correct_during_training (NEW)
    - test_sequential_replay_buffer_stores_episodes (NEW)
    - test_lstm_training_with_sequences (NEW)

class TestLSTMForwardPass:
    - test_partial_observability_5x5_window_to_lstm (NEW)
```

**Effort estimate**: 4 hours
- 2 hours: Hidden state persistence tests (4 tests)
- 1.5 hours: Batch training tests (3 tests)
- 0.5 hours: Forward pass integration test

#### 12c. `integration/test_intrinsic_exploration.py` (NEW - 6 tests)

**Goal**: Test RND intrinsic rewards + adaptive annealing during training

**Test structure**:
```python
class TestRNDIntrinsicRewards:
    - test_rnd_computes_novelty_for_observations (NEW)
    - test_intrinsic_rewards_combined_with_extrinsic (NEW)
    - test_combined_reward_stored_in_replay_buffer (NEW)

class TestAdaptiveAnnealing:
    - test_intrinsic_weight_decreases_after_consistent_performance (NEW)
    - test_annealing_requires_survival_above_50_steps (NEW)
    - test_annealing_requires_low_variance (NEW)
```

**Effort estimate**: 3 hours
- 1.5 hours: RND reward computation tests (3 tests)
- 1.5 hours: Adaptive annealing tests (3 tests)

**Task 12 Total**: 10 hours

**Success criteria**:
- [ ] Temporal mechanics tests cover time progression, operating hours, multi-tick interactions
- [ ] LSTM hidden state behavior verified during episodes and training
- [ ] Intrinsic exploration (RND + annealing) integrated correctly
- [ ] All 24 new tests pass consistently

---

### Task 13: Data Flow & Orchestration Integration Tests

**Goal**: Test complete data flows through the pipeline and episode/training orchestration.

**New files to create**:

#### 13a. `integration/test_episode_execution.py` (NEW - 6 tests)

**Goal**: Test single episode execution end-to-end

**Test structure**:
```python
class TestEpisodeLifecycle:
    - test_single_episode_feedforward_network (NEW)
    - test_single_episode_recurrent_network_with_lstm (NEW)
    - test_multi_agent_episode_with_partial_dones (NEW)
    - test_episode_all_agents_die (NEW)
    - test_episode_all_agents_survive_to_max_steps (NEW)
    - test_episode_observation_action_reward_cycle (NEW)
```

**Effort estimate**: 3 hours
- 2 hours: Episode execution tests (6 tests)
- 1 hour: Verify obs → action → reward → next_obs cycle

#### 13b. `integration/test_training_loop.py` (migrate 3 + add 5 = 8 tests)

**Migration strategy**: Consolidate test_masked_loss_integration.py + add new tests

**Existing tests** (3 tests):
```python
class TestMaskedLossIntegration:
    - test_masked_loss_during_training
    - test_action_masking_enforced_in_q_values
    - test_boundary_masking_during_training
```

**New tests to add** (5 tests):
```python
class TestMultiEpisodeTraining:
    - test_train_10_episodes_with_learning_progression (NEW)
    - test_epsilon_decay_over_episodes (NEW)
    - test_replay_buffer_accumulation_during_training (NEW)
    - test_target_network_updates_at_frequency (NEW)
    - test_q_values_improve_over_time (NEW)
```

**Effort estimate**: 4 hours
- 1 hour: Migrate masked loss tests
- 2.5 hours: Add 5 multi-episode training tests
- 0.5 hours: Verify learning progression

#### 13c. `integration/test_data_flows.py` (NEW - 8 tests)

**Goal**: Trace data through complete pipelines (observation, reward, action, training)

**Test structure**:
```python
class TestObservationPipeline:
    - test_environment_builds_observation_correct_dims (NEW)
    - test_partial_observability_5x5_window_correct (NEW)

class TestRewardPipeline:
    - test_environment_extrinsic_reward_to_population (NEW)
    - test_exploration_intrinsic_reward_combined (NEW)
    - test_combined_reward_stored_in_replay_buffer (NEW)

class TestActionPipeline:
    - test_qnetwork_qvalues_to_exploration (NEW)
    - test_exploration_epsilon_greedy_with_masking (NEW)
    - test_actions_to_environment_execution (NEW)
```

**Effort estimate**: 3 hours
- 2 hours: Pipeline tests (8 tests)
- 1 hour: Verify end-to-end data flow

**Task 13 Total**: 10 hours

**Success criteria**:
- [ ] Episode execution tests cover feedforward and recurrent networks
- [ ] Training loop tests verify multi-episode learning progression
- [ ] Data flow tests trace information through complete pipelines
- [ ] All 22 new tests pass consistently

---

## 3. Implementation Strategy

### Workflow for Each Task

**Phase 1: Research** (15 min per task)
- Read existing test files to understand patterns
- Identify what to migrate vs. what to create new
- Note fixture patterns for reuse

**Phase 2: Implementation** (core effort)
- Migrate existing tests to new location
- Consolidate redundant tests
- Add new tests for coverage gaps
- Use fixture composition from conftest.py

**Phase 3: Verification** (30 min per task)
- Run all tests in file: `pytest tests/test_townlet/integration/test_<name>.py -v`
- Verify all pass consistently (run 3 times to check for flakiness)
- Check coverage: `pytest --cov=townlet --cov-report=term-missing`

**Phase 4: Review** (15 min per task)
- Cross-reference with research findings (RESEARCH-INTEGRATION-TEST-STRATEGY.md)
- Verify all "rub points" covered
- Document any deviations from plan

### Sequential Execution Order

**IMPORTANT**: Tasks must be done sequentially (not parallel) due to dependencies:

1. **Task 11** (Checkpointing & Signal Purity) - FIRST
   - Foundation for other tests
   - Establishes checkpoint patterns used by Task 13

2. **Task 12** (Feature-Specific) - SECOND
   - Independent feature tests
   - Can reference checkpoint patterns from Task 11

3. **Task 13** (Data Flow & Orchestration) - THIRD
   - Builds on patterns from Tasks 11 & 12
   - Most complex, benefits from established patterns

### Parallelization Strategy (within tasks)

While tasks must be sequential, **within each task**, files can be implemented in parallel if time permits:

**Task 11**:
- `test_checkpointing.py` (consolidation-heavy, 6 hours)
- `test_curriculum_signal_purity.py` (migration, 2 hours) ← can start while checkpointing in progress
- `test_runner_integration.py` (migration, 2 hours) ← can start while checkpointing in progress

**Task 12**:
- `test_temporal_mechanics.py` (3 hours)
- `test_recurrent_networks.py` (4 hours) ← can start in parallel
- `test_intrinsic_exploration.py` (3 hours) ← can start in parallel

**Task 13**:
- `test_episode_execution.py` (3 hours)
- `test_training_loop.py` (4 hours) ← can start in parallel
- `test_data_flows.py` (3 hours) ← can start in parallel

---

## 4. File Organization

### Directory Structure

```
tests/test_townlet/
├── integration/                              # NEW subdirectory
│   ├── __init__.py                          # Empty (makes it a package)
│   │
│   ├── test_checkpointing.py                # Task 11a (consolidate P1.1/P2.1)
│   ├── test_curriculum_signal_purity.py     # Task 11b (migrate existing)
│   ├── test_runner_integration.py           # Task 11c (migrate + expand)
│   │
│   ├── test_temporal_mechanics.py           # Task 12a (migrate + expand)
│   ├── test_recurrent_networks.py           # Task 12b (NEW)
│   ├── test_intrinsic_exploration.py        # Task 12c (NEW)
│   │
│   ├── test_episode_execution.py            # Task 13a (NEW)
│   ├── test_training_loop.py                # Task 13b (migrate + expand)
│   └── test_data_flows.py                   # Task 13c (NEW)
│
├── conftest.py                               # Shared fixtures (use existing)
├── test_*.py                                 # Unit tests (Tasks 1-10)
└── test_recording/                           # Recording tests (separate concern)
```

### File Size Targets

| File | Tests | Lines | Rationale |
|------|-------|-------|-----------|
| `test_checkpointing.py` | ~15 | ~600 | Consolidated from 38 tests across 7 files |
| `test_curriculum_signal_purity.py` | ~11 | ~300 | Migrated 9 + 2 new |
| `test_runner_integration.py` | ~6 | ~250 | Migrated 3 + 3 new |
| `test_temporal_mechanics.py` | ~10 | ~400 | Migrated 5 + 5 new |
| `test_recurrent_networks.py` | ~8 | ~350 | All new |
| `test_intrinsic_exploration.py` | ~6 | ~250 | All new |
| `test_episode_execution.py` | ~6 | ~300 | All new |
| `test_training_loop.py` | ~8 | ~350 | Migrated 3 + 5 new |
| `test_data_flows.py` | ~8 | ~300 | All new |

**Total**: ~78 integration tests, ~3100 lines

---

## 5. Test Patterns & Best Practices

### Pattern 1: Real Components, Minimal Mocking

**Principle**: Integration tests should use real components, not mocks (unlike unit tests).

```python
def test_episode_execution_feedforward():
    """Full episode with real components (feedforward network)."""
    device = torch.device("cpu")

    # Real components (no mocks!)
    env = VectorizedHamletEnv(num_agents=1, grid_size=5, device=device)
    curriculum = StaticCurriculum(difficulty_level=0.5)
    exploration = EpsilonGreedyExploration(epsilon=0.1)

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=["agent_0"],
        device=device,
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        network_type="simple",
    )

    # Reset and run episode
    population.reset()
    for step in range(50):
        agent_state = population.step_population(env)
        if agent_state.dones[0]:
            break

    # Verify episode completed correctly
    assert agent_state.dones[0] or step == 49, "Episode should end"
    assert len(population.replay_buffer) > 0, "ReplayBuffer should have transitions"
```

**Why**: Integration tests verify real interactions, not mocked behavior.

### Pattern 2: Fixture Composition

**Principle**: Compose fixtures to build complex test scenarios (use existing conftest.py patterns).

```python
@pytest.fixture
def simple_env(device):
    """Simple 5×5 environment for fast integration tests."""
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=5,
        device=device,
        partial_observability=False,
        enable_temporal_mechanics=False,
    )

@pytest.fixture
def population_with_simple_network(simple_env, device):
    """Population with feedforward network."""
    curriculum = StaticCurriculum(difficulty_level=0.5)
    exploration = EpsilonGreedyExploration(epsilon=0.1)

    return VectorizedPopulation(
        env=simple_env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=["agent_0"],
        device=device,
        obs_dim=simple_env.observation_dim,
        action_dim=simple_env.action_dim,
        network_type="simple",
    )

def test_episode_uses_composed_fixtures(simple_env, population_with_simple_network):
    """Test using composed fixtures."""
    population_with_simple_network.reset()
    agent_state = population_with_simple_network.step_population(simple_env)
    assert agent_state.observations.shape[0] == 1
```

**Why**: Reduces boilerplate, makes tests readable, encourages reuse.

### Pattern 3: State Verification

**Principle**: Verify state consistency across components after operations.

```python
def test_checkpoint_round_trip():
    """Verify checkpoint save/load preserves exact state."""
    device = torch.device("cpu")

    # Train for N episodes
    runner = DemoRunner(
        env_config=...,
        population_config=...,
        max_episodes=10,
    )
    runner.run()

    # Save checkpoint
    checkpoint_path = runner.save_checkpoint()

    # Record state before load
    q_weights_before = {k: v.clone() for k, v in runner.population.q_network.state_dict().items()}
    curriculum_stage_before = runner.curriculum.tracker.agent_stages[0].item()
    epsilon_before = runner.population.exploration.epsilon

    # Create new runner and load
    runner2 = DemoRunner(
        env_config=...,
        population_config=...,
        max_episodes=10,
    )
    runner2.load_checkpoint(checkpoint_path)

    # Verify exact state match
    q_weights_after = runner2.population.q_network.state_dict()
    for key in q_weights_before.keys():
        assert torch.allclose(q_weights_before[key], q_weights_after[key]), \
            f"Q-network weights mismatch at {key}"

    assert curriculum_stage_before == runner2.curriculum.tracker.agent_stages[0].item(), \
        "Curriculum stage mismatch"
    assert epsilon_before == runner2.population.exploration.epsilon, \
        "Epsilon mismatch"
```

**Why**: Ensures exact state preservation (critical for checkpointing).

### Pattern 4: Data Flow Tracing

**Principle**: Trace data from source to sink, verify transformations at each step.

```python
def test_reward_pipeline():
    """Trace rewards from environment → population → replay buffer."""
    device = torch.device("cpu")
    env, population = setup_env_and_population(device)

    # Reset
    population.reset()

    # Step and capture intermediate values
    agent_state = population.step_population(env)

    # 1. Environment produces extrinsic reward
    assert agent_state.rewards_extrinsic.shape == (1,), "Extrinsic rewards should be [num_agents]"

    # 2. Exploration produces intrinsic reward
    assert agent_state.intrinsic_rewards.shape == (1,), "Intrinsic rewards should be [num_agents]"

    # 3. Population combines rewards
    intrinsic_weight = population.exploration.get_intrinsic_weight()
    expected_combined = agent_state.rewards_extrinsic + agent_state.intrinsic_rewards * intrinsic_weight
    assert torch.allclose(agent_state.rewards, expected_combined), "Reward combination incorrect"

    # 4. Replay buffer stores combined reward
    if len(population.replay_buffer) > 0:
        sample = population.replay_buffer.sample(1)
        assert sample["rewards"][0] == agent_state.rewards[0], \
            "Replay buffer stores combined reward"
```

**Why**: Verifies data transformations at each pipeline stage.

### Pattern 5: Behavioral Assertions (Not Exact Values)

**Principle**: Integration tests should assert behaviors, not exact values (resilient to implementation changes).

```python
def test_learning_progression():
    """Verify agents improve over time (behavioral, not exact values)."""
    device = torch.device("cpu")

    # Train for 50 episodes
    runner = DemoRunner(...)
    metrics = runner.run()

    # Behavioral assertion: survival time should INCREASE (not exact value)
    early_survival = metrics["survival_time"][:10]  # First 10 episodes
    late_survival = metrics["survival_time"][-10:]  # Last 10 episodes

    assert late_survival.mean() > early_survival.mean(), \
        "Agents should survive longer after training"

    # Behavioral assertion: Q-values should become MORE CERTAIN (higher max, lower variance)
    early_q_entropy = metrics["q_entropy"][:10].mean()
    late_q_entropy = metrics["q_entropy"][-10:].mean()

    assert late_q_entropy < early_q_entropy, \
        "Q-value entropy should decrease (more certain decisions)"
```

**Why**: Tests remain valid even if exact reward values change.

### Testing Best Practices Summary

1. ✅ **Use CPU device for determinism** (like unit tests)
2. ✅ **Small environments** (5×5 grid, 1-2 agents) for speed
3. ✅ **Short episodes** (50 steps max) unless testing max_steps behavior
4. ✅ **Behavioral assertions** (trends, not exact values) for resilience
5. ✅ **Clear test names** describing the integration point (e.g., `test_reward_pipeline_environment_to_replay_buffer`)
6. ✅ **Fixtures for common setups** (add to conftest.py, compose in tests)
7. ✅ **Mark slow tests** with `@pytest.mark.slow` (can skip during rapid development)
8. ✅ **Real components** (minimal mocking, unlike unit tests)
9. ✅ **State verification** (exact for checkpoints, behavioral for training)
10. ✅ **Data flow tracing** (verify transformations at each step)

---

## 6. Coverage Targets

### What Each Task Should Cover

#### Task 11: Checkpointing & Signal Purity

**Checkpointing coverage**:
- ✅ Q-network state dict preserved
- ✅ Optimizer state dict preserved
- ✅ Exploration state preserved (RND networks, epsilon, intrinsic weight)
- ✅ Curriculum tracker state preserved (stages, survival history)
- ✅ Replay buffer preserved (transitions or episodes)
- ✅ Affordance layout preserved (for generalization tests)
- ✅ Agent IDs preserved (multi-agent)
- ✅ Per-agent baselines preserved (P2.1)
- ✅ Episode flushing before checkpoint (LSTM only)
- ✅ Round-trip verification (save → load → exact match)

**Signal purity coverage**:
- ✅ Curriculum receives integer survival time (not float rewards)
- ✅ Survival time NOT contaminated by intrinsic rewards
- ✅ Single update per episode (not mid-episode)
- ✅ Done signal sent on max_steps survival
- ✅ Signal purity maintained during multi-episode training
- ✅ Stage transitions based on survival, not rewards

**Runner integration coverage**:
- ✅ Episode loop execution
- ✅ Checkpoint save at intervals
- ✅ Database logging after episodes
- ✅ Recording integration (optional)

#### Task 12: Feature-Specific Integration

**Temporal mechanics coverage**:
- ✅ 24-hour cycle progression (0→23→0)
- ✅ Operating hours enforcement (Job 9am-5pm, Bar 6pm-2am)
- ✅ Closed affordances masked in action space
- ✅ Multi-tick interactions (4 ticks for Job)
- ✅ Early exit with partial benefits
- ✅ Per-tick costs accumulate
- ✅ Interaction progress in observations
- ✅ Completion bonus awarded at end

**LSTM integration coverage**:
- ✅ Hidden state persists across steps within episode
- ✅ Hidden state resets on death
- ✅ Hidden state resets after flush (max_steps survival)
- ✅ Hidden state shape correct (num_agents vs batch_size)
- ✅ Sequential replay buffer stores episodes
- ✅ LSTM training with sequences
- ✅ Partial observability (5×5 window) → LSTM forward pass

**Intrinsic exploration coverage**:
- ✅ RND computes novelty for observations
- ✅ Intrinsic rewards combined with extrinsic
- ✅ Combined reward stored in replay buffer
- ✅ Intrinsic weight decreases after consistent performance
- ✅ Annealing requires survival >50 steps
- ✅ Annealing requires low variance (<100)

#### Task 13: Data Flow & Orchestration

**Episode execution coverage**:
- ✅ Single episode feedforward network (start to finish)
- ✅ Single episode recurrent network (LSTM)
- ✅ Multi-agent episode with partial dones
- ✅ Episode all agents die
- ✅ Episode all agents survive to max_steps
- ✅ Observation → Action → Reward → Next observation cycle

**Training loop coverage**:
- ✅ Multi-episode training (10+ episodes)
- ✅ Learning progression (survival time increases)
- ✅ Epsilon decay over episodes
- ✅ Replay buffer accumulation
- ✅ Target network updates at frequency
- ✅ Q-values improve over time
- ✅ Action masking enforced during training
- ✅ Masked loss computed correctly

**Data flow coverage**:
- ✅ Observation pipeline: Environment → Population (correct dims)
- ✅ Reward pipeline: Environment → Exploration → Population → ReplayBuffer
- ✅ Action pipeline: Q-network → Exploration → Environment
- ✅ Training pipeline: ReplayBuffer → Q-network → Target network

---

## 7. Dependencies & Sequencing

### Task Execution Order (MUST BE SEQUENTIAL)

```
Task 11: Checkpointing & Signal Purity
   ↓
   └─> Establishes checkpoint patterns
       Establishes signal purity patterns
       Establishes runner integration patterns
   ↓
Task 12: Feature-Specific Integration
   ↓
   └─> Can reference checkpoint patterns from Task 11
       Isolated feature tests (temporal, LSTM, exploration)
   ↓
Task 13: Data Flow & Orchestration
   ↓
   └─> Builds on patterns from Tasks 11 & 12
       Most complex, uses all established patterns
```

### Why This Order?

1. **Task 11 first**: Checkpoint patterns are foundational (used by all other tests)
2. **Task 12 second**: Feature tests are isolated, can reference checkpoint patterns
3. **Task 13 third**: Orchestration tests build on all previous patterns

### Inter-Task Dependencies

| Task | Depends On | Provides |
|------|-----------|----------|
| Task 11 | None | Checkpoint patterns, signal purity patterns, runner patterns |
| Task 12 | Task 11 | Temporal patterns, LSTM patterns, exploration patterns |
| Task 13 | Tasks 11, 12 | Episode patterns, training patterns, data flow patterns |

### Intra-Task Parallelization (Optional)

Within each task, files can be implemented in parallel if time permits:

**Task 11** (can parallelize after checkpoint consolidation starts):
- `test_checkpointing.py` (6h) ← START FIRST (foundational)
- `test_curriculum_signal_purity.py` (2h) ← Can start after 2h into checkpointing
- `test_runner_integration.py` (2h) ← Can start after 2h into checkpointing

**Task 12** (all independent, can parallelize):
- `test_temporal_mechanics.py` (3h)
- `test_recurrent_networks.py` (4h)
- `test_intrinsic_exploration.py` (3h)

**Task 13** (episode first, then parallelize):
- `test_episode_execution.py` (3h) ← START FIRST (foundational)
- `test_training_loop.py` (4h) ← Can start after episode tests done
- `test_data_flows.py` (3h) ← Can start after episode tests done

---

## 8. Migration Strategy (Task 11)

### Step-by-Step Migration Process

#### Step 1: Create Integration Subdirectory

```bash
mkdir -p tests/test_townlet/integration
touch tests/test_townlet/integration/__init__.py
```

#### Step 2: Migrate Checkpointing Tests (Consolidate P1.1/P2.1)

**Source files** (7 files, 1609 lines):
- `test_p1_1_phase1_baseline.py` (7 tests)
- `test_p1_1_phase2_full_checkpoint.py` (6 tests)
- `test_p1_1_phase3_curriculum.py` (3 tests)
- `test_p1_1_phase4_affordances.py` (4 tests)
- `test_p1_1_phase5_flush.py` (7 tests)
- `test_p1_1_phase6_agent_ids.py` (6 tests)
- `test_p2_1_vectorized_baseline.py` (5 tests)

**Target file**: `integration/test_checkpointing.py` (~15 tests, ~600 lines)

**Consolidation approach**:
1. Read all 7 files, identify redundant tests
2. Group by concern (baseline, full state, round-trip, affordances, flush)
3. Merge redundant tests (e.g., multiple "checkpoint contains key" tests → one comprehensive test)
4. Extract common fixtures to conftest.py
5. Verify all tests pass after consolidation

**Effort**: 6 hours

#### Step 3: Migrate Signal Purity Tests

**Source file**: `test_curriculum_signal_purity.py` (9 tests, 242 lines)

**Target file**: `integration/test_curriculum_signal_purity.py` (11 tests, ~300 lines)

**Migration approach**:
1. Copy file to `integration/` subdirectory
2. Update imports (if needed)
3. Add 2 new tests for multi-episode signal purity
4. Verify all tests pass

**Effort**: 2 hours

#### Step 4: Migrate Runner Integration Tests

**Source file**: `test_recording/test_runner_integration.py` (3 tests)

**Target file**: `integration/test_runner_integration.py` (6 tests, ~250 lines)

**Migration approach**:
1. Copy file to `integration/` subdirectory
2. Update imports (if needed)
3. Add 3 new orchestration tests
4. Verify all tests pass

**Effort**: 2 hours

#### Step 5: Verify Migration Complete

**Verification checklist**:
- [ ] All 50 existing tests migrated to `integration/`
- [ ] Original files still exist (don't delete yet, for reference)
- [ ] All tests pass: `pytest tests/test_townlet/integration/ -v`
- [ ] No import errors
- [ ] Fixtures work correctly

**Effort**: 0.5 hours

**Total Task 11 migration**: 10.5 hours

---

## 9. New Test Strategy (Tasks 12-13)

### Approach for Creating New Tests

#### Step 1: Identify Integration Points

**From research document**:
- Task 12: Temporal mechanics, LSTM state, intrinsic exploration
- Task 13: Episode execution, training loop, data flows

#### Step 2: Define Test Scenarios (Realistic, Not Exhaustive)

**Principle**: Integration tests should test realistic scenarios, not edge cases (that's for unit tests).

**Example**: Temporal mechanics
- ✅ Realistic: Job operates 9am-5pm, agent interacts at 3pm (success)
- ✅ Realistic: Agent tries to interact with Job at 8pm (masked)
- ❌ Exhaustive: Test every hour 0-23 for every affordance (too slow)

#### Step 3: Focus on "Rub Points"

**Critical rub points from research**:
1. LSTM hidden state management
2. Curriculum signal purity
3. Episode flushing
4. Checkpoint round-trip
5. Action masking enforcement

**Ensure each rub point has dedicated tests.**

#### Step 4: Use Behavioral Assertions

**Good** (behavioral):
```python
assert late_survival.mean() > early_survival.mean(), "Agents improve over time"
```

**Bad** (exact values):
```python
assert late_survival.mean() == 123.45, "Survival should be exactly 123.45"
```

#### Step 5: Document What Each Test Validates

**Pattern**:
```python
def test_lstm_hidden_state_persists_across_steps():
    """
    Verify LSTM hidden state persists across 10 steps within episode.

    This test validates the critical contract:
    - Hidden state should NOT reset mid-episode (memory loss)
    - Hidden state should persist across steps (short-term memory)
    - Hidden state shape should remain [num_agents, hidden_dim]

    Failure mode: If hidden state resets mid-episode, LSTM cannot
    build temporal understanding of environment.
    """
    # Test implementation...
```

---

## 10. Success Criteria

### Task 11: Checkpointing & Signal Purity Integration

**Completion criteria**:
- [ ] All 38 checkpointing tests migrated and consolidated into `test_checkpointing.py` (~15 tests)
- [ ] All 9 signal purity tests migrated to `test_curriculum_signal_purity.py`
- [ ] 2 new signal purity tests added (multi-episode with intrinsic rewards)
- [ ] All 3 runner integration tests migrated to `test_runner_integration.py`
- [ ] 3 new runner orchestration tests added
- [ ] All tests pass consistently (3 runs with no failures)
- [ ] Checkpoint round-trip verified (save → load → exact state match)
- [ ] Signal purity maintained (curriculum sees survival time, not rewards)

**Verification**:
```bash
# Run Task 11 tests
pytest tests/test_townlet/integration/test_checkpointing.py -v
pytest tests/test_townlet/integration/test_curriculum_signal_purity.py -v
pytest tests/test_townlet/integration/test_runner_integration.py -v

# Verify all pass (should see ~32 tests total)
pytest tests/test_townlet/integration/ -k "checkpoint or signal or runner" -v
```

**Estimated effort**: 10 hours

---

### Task 12: Feature-Specific Integration Tests

**Completion criteria**:
- [ ] Temporal mechanics tests cover time progression, operating hours, multi-tick interactions (10 tests)
- [ ] LSTM hidden state behavior verified during episodes and training (8 tests)
- [ ] Intrinsic exploration (RND + annealing) integrated correctly (6 tests)
- [ ] All 24 tests pass consistently (3 runs with no failures)
- [ ] Operating hours enforce time-based masking
- [ ] Multi-tick interactions complete correctly (progress tracking, early exit, costs)
- [ ] LSTM hidden state persists within episodes, resets on done
- [ ] Adaptive annealing requires both low variance AND high survival

**Verification**:
```bash
# Run Task 12 tests
pytest tests/test_townlet/integration/test_temporal_mechanics.py -v
pytest tests/test_townlet/integration/test_recurrent_networks.py -v
pytest tests/test_townlet/integration/test_intrinsic_exploration.py -v

# Verify all pass (should see ~24 tests total)
pytest tests/test_townlet/integration/ -k "temporal or recurrent or intrinsic" -v
```

**Estimated effort**: 10 hours

---

### Task 13: Data Flow & Orchestration Integration Tests

**Completion criteria**:
- [ ] Episode execution tests cover feedforward and recurrent networks (6 tests)
- [ ] Training loop tests verify multi-episode learning progression (8 tests)
- [ ] Data flow tests trace information through complete pipelines (8 tests)
- [ ] All 22 tests pass consistently (3 runs with no failures)
- [ ] Full episode execution works (obs → action → reward → next_obs)
- [ ] Multi-episode training shows learning progression (survival time increases)
- [ ] Data flows correctly: observation, reward, action, training pipelines
- [ ] Action masking enforced during training

**Verification**:
```bash
# Run Task 13 tests
pytest tests/test_townlet/integration/test_episode_execution.py -v
pytest tests/test_townlet/integration/test_training_loop.py -v
pytest tests/test_townlet/integration/test_data_flows.py -v

# Verify all pass (should see ~22 tests total)
pytest tests/test_townlet/integration/ -k "episode or training or flow" -v
```

**Estimated effort**: 10 hours

---

### Overall Success Criteria (All Tasks)

**Final completion criteria**:
- [ ] All 71 existing integration tests migrated to `integration/` subdirectory
- [ ] 31 new integration tests created (total ~102 integration tests)
- [ ] All 5 critical rub points covered by tests:
  - [ ] LSTM hidden state management
  - [ ] Curriculum signal purity
  - [ ] Episode flushing
  - [ ] Checkpoint round-trip
  - [ ] Action masking enforcement
- [ ] All tests pass consistently (no flakiness)
- [ ] Tests run in reasonable time (<5 min total for integration suite)
- [ ] Integration test coverage documented in README

**Final verification**:
```bash
# Run all integration tests
pytest tests/test_townlet/integration/ -v

# Should see ~102 tests total, all passing
# Expected runtime: 3-5 minutes

# Check coverage
pytest tests/test_townlet/integration/ --cov=townlet --cov-report=term-missing
```

---

## 11. Risk Assessment

### What Could Go Wrong?

#### Risk 1: Migration Breaks Tests

**Symptom**: Tests that passed in old location fail in new location.

**Cause**: Import paths changed, fixtures not found, conftest.py not loaded.

**Mitigation**:
- Run old tests first to verify baseline: `pytest tests/test_townlet/test_p1_*.py -v`
- Migrate one file at a time, verify before moving to next
- Keep old files as reference (don't delete immediately)
- Update imports carefully: `from townlet.X import Y` should work from any location

**Rollback**: If migration fails, revert to old files and investigate.

#### Risk 2: Integration Tests Too Slow

**Symptom**: Integration tests take >10 minutes to run.

**Cause**: Episodes too long, environments too large, too many training steps.

**Mitigation**:
- Use small environments (5×5 grid, not 8×8)
- Use short episodes (50 steps max, not 500)
- Use small batches (16 transitions, not 64)
- Use CPU device (faster for small batches)
- Mark slow tests with `@pytest.mark.slow` for optional skipping

**Adjustment**: If tests still too slow, reduce episode length or mark as slow.

#### Risk 3: Fixture Complexity

**Symptom**: Tests hard to read due to complex fixture chains.

**Cause**: Over-composition of fixtures, unclear dependencies.

**Mitigation**:
- Keep fixture chains shallow (max 3 levels)
- Document fixture purpose in docstrings
- Use descriptive fixture names: `population_with_lstm_network` vs. `pop`
- Add common fixtures to conftest.py
- Inline simple fixtures in test classes

**Adjustment**: If fixtures become too complex, refactor to helper functions.

#### Risk 4: Flaky Tests

**Symptom**: Tests pass sometimes, fail other times (non-deterministic).

**Cause**: GPU randomness, race conditions, insufficient resets.

**Mitigation**:
- Use CPU device for determinism: `device = torch.device("cpu")`
- Set random seeds: `torch.manual_seed(42)`
- Reset all state between tests: `env.reset()`, `population.reset()`
- Avoid timing dependencies: Don't assert on exact step counts
- Use behavioral assertions: "survival increases" not "survival == 123"

**Adjustment**: If test is flaky, add determinism or relax assertion.

#### Risk 5: Consolidation Loses Coverage

**Symptom**: After consolidation, some edge cases no longer tested.

**Cause**: Merging tests removed important checks.

**Mitigation**:
- Read all tests before consolidating
- Create checklist of what each test validates
- Merge tests only if they check identical behavior
- Keep comprehensive tests over redundant tests
- Cross-reference with research doc to ensure all rub points covered

**Adjustment**: If coverage lost, add back missing checks.

---

## 12. Effort Breakdown

### Detailed Time Estimates

#### Task 11: Checkpointing & Signal Purity Integration

| Activity | Time | Rationale |
|----------|------|-----------|
| Read existing P1.1/P2.1 tests (7 files) | 1h | Understand patterns, identify redundancy |
| Consolidate checkpointing tests → `test_checkpointing.py` | 3h | Merge 38 tests → 15 tests, extract fixtures |
| Add checkpoint round-trip tests (2 tests) | 1h | New tests for save → load → verify |
| Migrate signal purity tests | 0.5h | Copy file, update imports |
| Add multi-episode signal purity tests (2 tests) | 1h | New tests for training with intrinsic rewards |
| Migrate runner integration tests | 0.5h | Copy file, update imports |
| Add runner orchestration tests (3 tests) | 1h | New tests for episode loop, checkpoint save, logging |
| Verification (run all tests 3 times) | 0.5h | Ensure no flakiness |
| Documentation (docstrings, comments) | 1h | Document what each test validates |
| **Total Task 11** | **10h** | |

#### Task 12: Feature-Specific Integration Tests

| Activity | Time | Rationale |
|----------|------|-----------|
| Migrate temporal mechanics tests (5 tests) | 1h | Copy from test_temporal_integration.py |
| Add multi-tick interaction tests (5 tests) | 1.5h | New tests for progress, early exit, costs |
| Create LSTM hidden state tests (4 tests) | 1.5h | New tests for persistence, reset, shape |
| Create LSTM batch training tests (3 tests) | 1h | New tests for training, replay buffer |
| Add LSTM forward pass test (1 test) | 0.5h | Test partial observability → LSTM |
| Create RND intrinsic reward tests (3 tests) | 1h | New tests for novelty, combination |
| Create adaptive annealing tests (3 tests) | 1.5h | New tests for annealing conditions |
| Verification (run all tests 3 times) | 0.5h | Ensure no flakiness |
| Documentation (docstrings, comments) | 1.5h | Document what each test validates |
| **Total Task 12** | **10h** | |

#### Task 13: Data Flow & Orchestration Integration Tests

| Activity | Time | Rationale |
|----------|------|-----------|
| Create episode execution tests (6 tests) | 2h | New tests for feedforward, LSTM, multi-agent |
| Migrate masked loss tests (3 tests) | 0.5h | Copy from test_masked_loss_integration.py |
| Add multi-episode training tests (5 tests) | 2h | New tests for learning progression |
| Create observation pipeline tests (2 tests) | 0.5h | Trace obs from environment → population |
| Create reward pipeline tests (3 tests) | 1h | Trace rewards through system |
| Create action pipeline tests (3 tests) | 1h | Trace actions from Q-network → environment |
| Verification (run all tests 3 times) | 0.5h | Ensure no flakiness |
| Documentation (docstrings, comments) | 1h | Document what each test validates |
| **Total Task 13** | **10h** | |

### Total Effort Summary

| Task | Tests | Effort | Status |
|------|-------|--------|--------|
| Task 11 | ~32 tests (migrate 50, consolidate to ~28, add 4 new) | 10 hours | Not started |
| Task 12 | ~24 tests (migrate 5, add 19 new) | 10 hours | Not started |
| Task 13 | ~22 tests (migrate 3, add 19 new) | 10 hours | Not started |
| **Total** | **~78 tests** | **30 hours** | |

**Optimistic estimate**: ~25 hours (if consolidation goes faster than expected)

**Realistic estimate**: ~30 hours (as planned)

**Pessimistic estimate**: ~35 hours (if flakiness or fixture issues arise)

---

## 13. Next Steps After Planning

### Immediate Actions (After Plan Review)

1. **Review plan with user** (this document)
   - Confirm task breakdown makes sense
   - Adjust effort estimates if needed
   - Get approval to proceed

2. **Create integration subdirectory**
   ```bash
   mkdir -p tests/test_townlet/integration
   touch tests/test_townlet/integration/__init__.py
   ```

3. **Start Task 11** (Checkpointing & Signal Purity)
   - Read existing P1.1/P2.1 test files
   - Begin consolidation into `test_checkpointing.py`
   - Migrate signal purity and runner tests

4. **Verify Task 11 complete** before moving to Task 12
   - All tests pass consistently
   - Coverage verified against research doc

5. **Proceed to Task 12** (Feature-Specific)
   - Temporal mechanics integration
   - LSTM hidden state integration
   - Intrinsic exploration integration

6. **Verify Task 12 complete** before moving to Task 13

7. **Proceed to Task 13** (Data Flow & Orchestration)
   - Episode execution integration
   - Training loop integration
   - Data flow pipeline integration

8. **Final validation**
   - Run all integration tests: `pytest tests/test_townlet/integration/ -v`
   - Verify coverage: `pytest --cov=townlet --cov-report=term-missing`
   - Document any deviations from plan

### Post-Implementation Actions

1. **Update documentation**
   - Update README with integration test structure
   - Document how to run integration tests
   - Add section on "Rub Points" and how they're tested

2. **Clean up old test files** (optional)
   - After verifying all tests migrated successfully
   - Archive old files (don't delete, for reference)
   - Update CI/CD to run integration tests

3. **Continuous integration**
   - Add integration tests to CI pipeline
   - Set reasonable timeout (10 min max)
   - Mark slow tests to skip in rapid CI

---

## 14. Conclusion

### Summary

This plan provides a detailed roadmap for implementing Tasks 11-13 (integration tests) for the HAMLET Townlet system. Key takeaways:

1. **71 existing integration tests** need migration and consolidation
2. **~30 new integration tests** needed to close coverage gaps
3. **30 hours total effort** (10h per task, realistic estimate)
4. **Sequential execution required** (Task 11 → 12 → 13)
5. **5 critical rub points** addressed: LSTM state, signal purity, flushing, checkpointing, masking

### Key Success Factors

1. ✅ **Consolidation over duplication**: Merge 38 checkpointing tests → 15 comprehensive tests
2. ✅ **Real components**: Use actual components, not mocks (unlike unit tests)
3. ✅ **Behavioral assertions**: Test trends, not exact values (resilient to changes)
4. ✅ **Fixture composition**: Reuse common setups, reduce boilerplate
5. ✅ **Sequential execution**: Complete Task 11 before Task 12, etc.

### Expected Outcomes

After completing Tasks 11-13:
- ✅ ~102 integration tests total (71 migrated + 31 new)
- ✅ All critical rub points covered by dedicated tests
- ✅ Complete data flow verification (obs, reward, action, training)
- ✅ Episode and training loop orchestration verified
- ✅ Checkpoint round-trip validated (save → load → exact match)
- ✅ Signal purity maintained (curriculum sees survival, not rewards)
- ✅ Temporal mechanics integrated correctly
- ✅ LSTM hidden state managed correctly
- ✅ Intrinsic exploration (RND + annealing) integrated

### Next Step

**Proceed to Task 11 implementation** after plan approval.

---

**END OF PLAN DOCUMENT**
