# Integration Test Strategy Research - HAMLET Townlet

**Date**: 2025-11-04
**Status**: RESEARCH COMPLETE
**Purpose**: Comprehensive analysis of integration test requirements for HAMLET Townlet system

---

## 1. Problem Statement

### Why Integration Tests Are Critical

After completing Tasks 1-10 (381 unit tests, all passing), we have excellent **component-level coverage** but lack **integration-level verification**. Unit tests validate individual components in isolation, but real-world failures often occur at **component boundaries** where data flows, state synchronization, and protocol contracts are violated.

**Key gaps after unit testing**:

1. **Component interaction contracts**: Do components communicate correctly?
2. **Data flow integrity**: Does information flow correctly through the pipeline?
3. **State synchronization**: Are states kept consistent across components?
4. **Episode lifecycle**: Does a full episode execute correctly end-to-end?
5. **Multi-episode training**: Do agents learn across multiple episodes?
6. **Checkpointing round-trips**: Can we save/load and resume training?
7. **Signal purity**: Does curriculum receive clean survival signals?
8. **Temporal mechanics**: Do time-based features work end-to-end?

### What Makes Integration Tests Different

| Aspect | Unit Tests | Integration Tests |
|--------|-----------|-------------------|
| **Scope** | Single component, mocked dependencies | Multiple components, real interactions |
| **Duration** | Fast (<10ms per test) | Slower (100ms-1s per test) |
| **State** | Isolated, deterministic | Shared, may require cleanup |
| **Focus** | Component behavior | Boundary interactions |
| **Assertions** | Exact values, behavioral contracts | Signal flow, state consistency |

---

## 2. Component Architecture Map

### Major Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         DemoRunner                              │
│                    (Training Orchestrator)                      │
│  - Episode loop                                                 │
│  - Checkpointing                                                │
│  - Database logging                                             │
│  - TensorBoard logging                                          │
│  - Episode recording (optional)                                 │
└────────────┬────────────────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌───────────┐   ┌─────────────────────────────┐
│           │   │   VectorizedPopulation      │
│  VectorizedHamletEnv   │   - Action selection         │
│           │   │   - Q-learning               │
│  - Grid state          │   - Experience replay        │
│  - Meters              │   - Training loop            │
│  - Affordances         │   - Hidden state (LSTM)      │
│  - Observations        │   - Curriculum updates       │
│  - Rewards             │   - Exploration integration  │
│  - Action masking      │   └──────┬──────┬──────┬────┘
│  - Temporal mechanics  │          │      │      │
└───────────┘                       │      │      │
                                    ▼      ▼      ▼
                         ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
                         │ Q-Networks   │ │ Curriculum   │ │ Exploration  │
                         │ - SimpleQ    │ │ - Adversarial│ │ - Epsilon    │
                         │ - RecurrentQ │ │ - Static     │ │ - RND        │
                         │ - Target Net │ │ - Tracker    │ │ - Adaptive   │
                         └──────────────┘ └──────────────┘ └──────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   ReplayBuffer       │
                         │   - Standard         │
                         │   - Sequential       │
                         └──────────────────────┘
```

### Sub-Components (Within VectorizedHamletEnv)

```
VectorizedHamletEnv
├── MeterDynamics
│   ├── Base depletion (per-step meter decay)
│   └── CascadeEngine (meter interdependencies)
├── AffordanceEngine
│   ├── Affordance effects (instant + multi-tick)
│   ├── Operating hours (time-based availability)
│   └── Affordability checks (money constraints)
├── ObservationBuilder
│   ├── Full observability (8×8 grid one-hot)
│   ├── Partial observability (5×5 local window)
│   └── Temporal features (time_of_day, interaction_progress)
└── RewardStrategy
    └── Survival rewards (per-step +1.0 for alive agents)
```

### Data Flow Diagram

```
Episode Start (DemoRunner.run())
    │
    ├─> population.reset()
    │   ├─> env.reset() → observations [num_agents, obs_dim]
    │   ├─> Reset hidden state (LSTM)
    │   └─> Sync telemetry (epsilon, stage, baseline)
    │
    └─> Episode Loop (for step in range(max_steps))
        │
        ├─> population.step_population(env)
        │   │
        │   ├─> q_network.forward(obs) → q_values [num_agents, action_dim]
        │   │
        │   ├─> exploration.select_actions(q_values, epsilon) → actions
        │   │
        │   ├─> env.step(actions)
        │   │   ├─> Apply movement (update positions)
        │   │   ├─> Apply action costs (energy depletion)
        │   │   ├─> Process interactions (affordances)
        │   │   ├─> Apply meter dynamics (base + cascades)
        │   │   ├─> Check terminal conditions (death)
        │   │   ├─> Build observations
        │   │   └─> Return (next_obs, rewards, dones, info)
        │   │
        │   ├─> exploration.compute_intrinsic_rewards(obs) → intrinsic_rewards
        │   │
        │   ├─> Combine rewards: total = extrinsic + intrinsic * weight
        │   │
        │   ├─> Store experience in replay buffer
        │   │   ├─> Standard: store(obs, action, reward, next_obs, done)
        │   │   └─> Sequential: accumulate episode, store on done
        │   │
        │   └─> Train Q-network (if total_steps % train_frequency == 0)
        │       ├─> Sample batch from replay buffer
        │       ├─> Compute TD targets (target_network)
        │       ├─> Compute loss (Huber loss)
        │       ├─> Backpropagation + gradient clipping
        │       └─> Update target network (if step % target_update_freq == 0)
        │
        ├─> Check curriculum transitions (stage advancement/retreat)
        │
        └─> If all agents done: break
            │
            └─> End of Episode
                ├─> Update curriculum tracker (survival_time, done)
                ├─> Flush episodes (LSTM only, if survived max_steps)
                ├─> Log metrics (database + TensorBoard)
                ├─> Decay epsilon (for next episode)
                └─> Record episode (if recording enabled)
```

---

## 3. Integration Points Taxonomy

### A. Environment ↔ Population

**Contract**: Population requests actions → Environment executes → Returns next state

| Integration Point | Data Flow | Critical Contract |
|------------------|-----------|-------------------|
| Action selection | Population → Environment | Actions must be valid (within bounds, properly masked) |
| Observation flow | Environment → Population | Observations match expected dimensions |
| Reward delivery | Environment → Population | Rewards are per-agent tensors [num_agents] |
| Done handling | Environment → Population | Done signals trigger episode cleanup |
| Hidden state management | Population ↔ Environment (indirect) | LSTM state persists during episode, resets on done |

**Test scenarios**:
- Multi-agent step execution (all agents step simultaneously)
- Done handling (partial dones, all dones)
- Observation dimension consistency (feedforward vs recurrent)
- Action masking enforcement (environment rejects invalid actions)

### B. Population ↔ Q-Network

**Contract**: Population forwards observations → Q-Network computes Q-values → Population selects actions

| Integration Point | Data Flow | Critical Contract |
|------------------|-----------|-------------------|
| Forward pass | Population → Q-Network | Observations match network input shape |
| Q-value output | Q-Network → Population | Q-values shape [num_agents, action_dim] |
| Hidden state (LSTM) | Q-Network (internal state) | Hidden state persists across steps, resets on done |
| Batch training | Population → Q-Network | Batch dimensions correct for training |
| Target network sync | Population → Target Network | Periodic weight copying from Q-network |

**Test scenarios**:
- Forward pass with feedforward network (SimpleQNetwork)
- Forward pass with recurrent network (RecurrentSpatialQNetwork)
- Hidden state persistence across steps (LSTM)
- Hidden state reset on done (LSTM)
- Batch training with correct dimensions
- Target network synchronization

### C. Population ↔ Exploration

**Contract**: Population queries exploration → Exploration selects actions (epsilon-greedy + intrinsic rewards)

| Integration Point | Data Flow | Critical Contract |
|------------------|-----------|-------------------|
| Action selection | Q-values → Exploration → Actions | Actions respect action masks |
| Intrinsic rewards | Observations → Exploration → Intrinsic rewards | Rewards shape [num_agents] |
| Epsilon decay | Exploration (internal) | Epsilon decreases over time |
| Intrinsic weight annealing | Exploration (internal) | Weight anneals based on performance |
| RND novelty | Observations → RND → Novelty | RND networks update during training |

**Test scenarios**:
- Epsilon-greedy action selection with masking
- RND intrinsic reward computation
- Adaptive annealing based on survival variance
- Combined reward computation (extrinsic + intrinsic * weight)

### D. Population ↔ Curriculum

**Contract**: Population reports episode results → Curriculum adjusts difficulty → Population updates environment

| Integration Point | Data Flow | Critical Contract |
|------------------|-----------|-------------------|
| Curriculum decisions | Curriculum → Population | Difficulty level (0.0-1.0) maps to stage (1-5) |
| Performance tracking | Population → Curriculum | Pure survival time (integer steps) |
| Stage transitions | Curriculum (internal) | Advance on 70% survival, retreat on 30% |
| Baseline updates | Curriculum → Population → Environment | Reward baseline updates per agent |

**Test scenarios**:
- Curriculum receives pure survival signals (not contaminated by intrinsic rewards)
- Stage advancement after N successful episodes
- Stage retreat after consistent failures
- Baseline updates propagate to environment

### E. Population ↔ ReplayBuffer

**Contract**: Population stores experiences → ReplayBuffer accumulates → Population samples batches for training

| Integration Point | Data Flow | Critical Contract |
|------------------|-----------|-------------------|
| Experience storage (standard) | Population → ReplayBuffer | Transitions stored immediately |
| Experience storage (sequential) | Population → SequentialReplayBuffer | Episodes accumulated, stored on done |
| Batch sampling (standard) | ReplayBuffer → Population | Random batch of transitions |
| Batch sampling (sequential) | SequentialReplayBuffer → Population | Random batch of sequences |
| Flush episodes (LSTM only) | Population → SequentialReplayBuffer | Surviving episodes flushed at max_steps |

**Test scenarios**:
- Standard replay buffer (feedforward networks)
- Sequential replay buffer (recurrent networks)
- Episode flushing for surviving agents (LSTM only)
- Batch sampling with correct dimensions

### F. DemoRunner ↔ All Components

**Contract**: DemoRunner orchestrates training loop → Manages checkpointing, logging, recording

| Integration Point | Data Flow | Critical Contract |
|------------------|-----------|-------------------|
| Episode loop | DemoRunner → Population/Environment | Correct step-by-step execution |
| Checkpointing | DemoRunner → All Components | Complete state serialization |
| Checkpoint loading | DemoRunner → All Components | State restoration with exact resume |
| Database logging | DemoRunner → Database | Metrics stored after each episode |
| TensorBoard logging | DemoRunner → TensorBoard | Training metrics logged correctly |
| Episode recording | DemoRunner → EpisodeRecorder | Step-by-step data capture (optional) |

**Test scenarios**:
- Full episode execution (start to finish)
- Multi-episode training (learning progression)
- Checkpoint save/load round-trip
- Database metric logging
- TensorBoard metric logging
- Episode recording integration

### G. Temporal Mechanics ↔ System

**Contract**: Time-based features integrate with environment, observations, action masking

| Integration Point | Data Flow | Critical Contract |
|------------------|-----------|-------------------|
| Time progression | Environment (internal) | Time advances 0→23, wraps to 0 |
| Operating hours | AffordanceEngine → Action masking | Affordances masked when closed |
| Multi-tick interactions | AffordanceEngine → Environment | Progress accumulates, rewards on completion |
| Temporal observations | Environment → ObservationBuilder | Time features (sin, cos, progress) in observations |
| Early exit | Agent interaction → Environment | Partial benefits on early exit |

**Test scenarios**:
- 24-hour cycle progression and wrapping
- Operating hours enforcement (Job 9am-5pm, Bar 6pm-2am)
- Multi-tick interaction completion (4 ticks for Job)
- Early exit with partial benefits
- Temporal features in observations

---

## 4. "Rub Points" Analysis

**Definition**: "Rub points" are boundaries where components interact and failures are most likely to occur. These are high-risk areas requiring focused integration testing.

### 🔴 Critical Rub Points (High Risk)

#### 1. LSTM Hidden State Management (Population ↔ Q-Network)

**Problem**: Hidden state must persist during episodes but reset on done.

**Failure modes**:
- Hidden state not reset → agents carry "memory" across episodes (wrong)
- Hidden state reset mid-episode → agents lose short-term memory (wrong)
- Hidden state batch size mismatch → crashes during training

**Why critical**: Breaks fundamental POMDP learning if mishandled.

**Test coverage needed**:
- ✅ Unit: Hidden state reset on done (Task 6 - Networks)
- ❌ Integration: Hidden state persistence across multi-step episodes
- ❌ Integration: Hidden state reset after max_steps survival (flush_episode)
- ❌ Integration: Hidden state batch size correct during training

---

#### 2. Curriculum Signal Purity (Population → Curriculum)

**Problem**: Curriculum must receive pure survival time (integer steps), NOT contaminated by intrinsic rewards.

**Failure modes**:
- Curriculum receives combined rewards → stage progression based on novelty, not survival
- Curriculum updated mid-episode → multiple updates per episode (wrong)
- Curriculum not updated on max_steps survival → missing data points

**Why critical**: Entire adversarial curriculum system breaks if signals are impure.

**Test coverage needed**:
- ✅ Unit: Curriculum tracks survival time correctly (Task 8)
- ✅ Integration: 9 tests in test_curriculum_signal_purity.py (excluded from unit tests)
- ❌ Integration: Full episode with intrinsic rewards + curriculum update

---

#### 3. Episode Flushing (Population → ReplayBuffer, LSTM only)

**Problem**: Episodes that survive to max_steps must be flushed to replay buffer, otherwise memory leak + data loss.

**Failure modes**:
- Episodes not flushed → memory grows unbounded
- Episodes not flushed → successful episodes never trained on (data loss)
- Flushing during episode → incomplete sequences in buffer

**Why critical**: Prevents LSTM agents from learning successful strategies.

**Test coverage needed**:
- ✅ Unit: Sequential replay buffer stores episodes (Task 7)
- ✅ Integration: test_p1_1_phase5_flush.py (7 tests, excluded from unit tests)
- ❌ Integration: Multi-episode training with flushing verification

---

#### 4. Checkpoint Round-Trip (DemoRunner ↔ All Components)

**Problem**: All component state must be saved and restored for exact resume.

**Failure modes**:
- Missing Q-network state → agents forget learned behavior
- Missing optimizer state → training momentum lost
- Missing curriculum state → agents reset to stage 1
- Missing exploration state → RND networks reset, epsilon resets
- Missing replay buffer → training data lost
- Missing affordance layout → generalization test invalidated

**Why critical**: Multi-day training requires reliable checkpointing.

**Test coverage needed**:
- ✅ Integration: test_checkpoint_completeness.py (21 tests)
- ✅ Integration: test_p1_1_phase*.py tests (6 phases, 1609 lines total)
- ❌ Integration: Full checkpoint→load→resume→verify equivalence test

---

#### 5. Action Masking Enforcement (Environment ↔ Population)

**Problem**: Invalid actions (out of bounds, interact without affordance, closed affordances) must be masked.

**Failure modes**:
- Actions not masked → agents learn invalid strategies
- Masking not synchronized → Q-network learns incorrect Q-values
- Temporal masking wrong → agents interact with closed affordances

**Why critical**: Breaks action space validity and learning stability.

**Test coverage needed**:
- ✅ Unit: Action masking logic (Task 1 - 37 tests)
- ❌ Integration: Action masking during training (Q-network respects masks)
- ❌ Integration: Temporal masking with operating hours

---

### 🟡 Important Rub Points (Medium Risk)

#### 6. Observation Dimension Consistency

**Problem**: Observation dimensions must match between environment and Q-network.

**Failure modes**:
- Dimension mismatch → crash on forward pass
- Temporal features missing → recurrent network expects 4 extra dims
- Partial observability dimension wrong → LSTM expects 54 dims, gets 91

**Why important**: Prevents runtime crashes, but easy to catch.

**Test coverage needed**:
- ✅ Unit: Observation dimensions (Task 2 - 32 tests)
- ❌ Integration: Forward pass with actual environment observations

---

#### 7. Reward Baseline Updates (Curriculum → Population → Environment)

**Problem**: Reward baseline must update when curriculum stage changes.

**Failure modes**:
- Baseline not updated → agents trained on wrong reward scale
- Baseline updated mid-episode → inconsistent rewards within episode
- Baseline tensor wrong shape → crash

**Why important**: Affects reward scaling and learning stability.

**Test coverage needed**:
- ✅ Unit: Baseline updates (Task 8 - Curriculum)
- ✅ Integration: test_p2_1_vectorized_baseline.py (5 tests)
- ❌ Integration: Baseline updates during multi-episode training

---

#### 8. Target Network Synchronization

**Problem**: Target network must periodically copy weights from Q-network.

**Failure modes**:
- Target network never updated → training unstable (too much variance)
- Target network updated too often → no stabilization benefit
- Target network wrong architecture → crash during sync

**Why important**: Critical for DQN stability, but failure mode is gradual.

**Test coverage needed**:
- ✅ Unit: Target network sync logic (Task 6 - Networks)
- ❌ Integration: Target network sync during training loop

---

### 🟢 Minor Rub Points (Low Risk)

#### 9. TensorBoard Logging

**Problem**: Metrics must be logged correctly to TensorBoard.

**Failure modes**:
- Metrics not logged → no training visibility
- Metrics wrong format → TensorBoard crashes
- Metrics logged at wrong frequency → charts unclear

**Why minor**: Doesn't affect training, only observability.

**Test coverage needed**:
- ✅ Unit: TensorBoard logger API (existing test_tensorboard_logger.py)
- ❌ Integration: Metrics logged during training loop

---

#### 10. Database Logging

**Problem**: Episode metrics must be stored in SQLite database.

**Failure modes**:
- Metrics not stored → no historical record
- Schema mismatch → database errors
- Database locked → write failures

**Why minor**: Doesn't affect training, only persistence.

**Test coverage needed**:
- ✅ Unit: Database API (test_recording/test_database.py)
- ❌ Integration: Database logging during multi-episode training

---

## 5. Integration Tests Already Identified

### From Task 8 (Curriculum) - Excluded Tests

**File**: `test_curriculum_signal_purity.py` (242 lines)

**Tests excluded from unit tests** (9 tests):

1. `test_update_curriculum_tracker_exists` - Population has method
2. `test_update_curriculum_tracker_accepts_rewards_and_dones` - Method signature
3. `test_curriculum_tracker_state_updated_after_single_call` - State updates
4. `test_survival_signal_is_integer_steps` - Signal is integer, not float
5. `test_survival_signal_not_contaminated_by_intrinsic_rewards` - Pure survival
6. `test_max_steps_survival_sends_done_signal` - Done=True on max_steps
7. `test_no_update_during_episode_steps` - Updates only at episode end
8. `test_single_update_per_episode_on_death` - Exactly one update per episode
9. `test_signal_is_monotonic_with_survival_time` - Longer survival = higher signal

**Reason for exclusion**: These tests run full episodes with Population + Environment + Curriculum integration. Too slow and complex for unit tests.

**Status**: ✅ Already exist, need to be moved to integration tests

---

### From P1.1/P2.1 Checkpointing - Phase Tests

**Files**: 6 phase test files (1609 lines total)

| File | Tests | Purpose |
|------|-------|---------|
| `test_p1_1_phase1_baseline.py` | 7 | Baseline checkpoint with Q-network + optimizer |
| `test_p1_1_phase2_full_checkpoint.py` | 6 | Full population checkpoint state |
| `test_p1_1_phase3_curriculum.py` | 3 | Curriculum state in checkpoints |
| `test_p1_1_phase4_affordances.py` | 4 | Affordance layout in checkpoints |
| `test_p1_1_phase5_flush.py` | 7 | Episode flushing before checkpoint |
| `test_p1_1_phase6_agent_ids.py` | 6 | Multi-agent agent_ids persistence |
| `test_p2_1_vectorized_baseline.py` | 5 | Per-agent reward baselines |

**Total**: 38 integration tests

**Reason for exclusion**: These tests involve DemoRunner + full component integration. Too slow for unit tests.

**Status**: ✅ Already exist, need to be reorganized in integration test structure

---

### From Existing Integration Test Files

**Files**:
- `test_affordance_integration.py` (13 tests) - Affordance engine integration with VectorizedHamletEnv
- `test_temporal_integration.py` (5 tests) - Temporal mechanics end-to-end
- `test_masked_loss_integration.py` (3 tests) - Masked loss during training
- `test_recording/test_runner_integration.py` (3 tests) - Runner + recorder integration

**Total**: 24 integration tests

**Status**: ✅ Already exist, may need consolidation

---

### Summary of Existing Integration Tests

**Total identified**: 71 integration tests across multiple files

**Categories**:
- Curriculum signal purity: 9 tests
- Checkpointing (P1.1/P2.1): 38 tests
- Affordance integration: 13 tests
- Temporal mechanics: 5 tests
- Masked loss: 3 tests
- Runner recording: 3 tests

**Action needed**: Reorganize these tests into new integration test structure

---

## 6. Integration Test Categories

### Category A: Component Pair Integration

**Purpose**: Test interaction between two components in isolation.

**Pattern**: Mock out other components, test the boundary contract.

**Tests needed**:

#### A1. Environment + Population
- Multi-agent step execution
- Done handling (partial dones, all dones)
- Observation dimension consistency
- Action masking enforcement

#### A2. Population + Q-Network
- Forward pass (feedforward)
- Forward pass (recurrent with hidden state)
- Hidden state persistence across steps
- Hidden state reset on done
- Batch training with correct dimensions
- Target network synchronization

#### A3. Population + Exploration
- Epsilon-greedy action selection
- RND intrinsic reward computation
- Adaptive annealing based on survival
- Combined reward computation

#### A4. Population + Curriculum
- Curriculum receives pure survival signals
- Stage advancement after N successes
- Stage retreat after consistent failures
- Baseline updates propagate

#### A5. Population + ReplayBuffer
- Standard replay buffer (feedforward)
- Sequential replay buffer (recurrent)
- Episode flushing for surviving agents
- Batch sampling with correct dimensions

---

### Category B: Multi-Component Integration

**Purpose**: Test interaction among 3+ components in realistic scenarios.

**Pattern**: Use real components, test complete data flows.

**Tests needed**:

#### B1. Episode Execution (Environment + Population + Q-Network)
- Single episode start-to-finish (feedforward)
- Single episode start-to-finish (recurrent with LSTM)
- Multi-agent episode with partial dones
- Episode with all agents dying
- Episode with all agents surviving to max_steps

#### B2. Training Loop (Environment + Population + Q-Network + Exploration + ReplayBuffer)
- Multi-episode training (10 episodes)
- Learning progression (Q-values improve over time)
- Epsilon decay over episodes
- Replay buffer accumulation
- Target network updates during training

#### B3. Curriculum-Guided Training (+ Curriculum)
- Stage advancement after successful episodes
- Stage retreat after failures
- Baseline updates during stage transitions
- Entropy gate enforcement

#### B4. Checkpointing (DemoRunner + All Components)
- Checkpoint save at episode N
- Checkpoint load and resume at episode N
- Verify exact equivalence after resume
- Multi-checkpoint training (save at ep 10, 20, 30)

---

### Category C: Feature Integration

**Purpose**: Test specific features that span multiple components.

**Pattern**: Focus on feature-specific data flows.

**Tests needed**:

#### C1. Temporal Mechanics
- ✅ 24-hour cycle progression (existing test)
- ✅ Operating hours enforcement (existing test)
- ✅ Multi-tick interaction completion (existing test)
- ❌ Early exit with partial benefits
- ❌ Temporal features in observations → Q-network

#### C2. Recurrent Networks (LSTM)
- ✅ Hidden state persistence during episode (partial coverage)
- ❌ Hidden state reset on death
- ❌ Hidden state reset on max_steps survival (after flush)
- ❌ Sequential replay buffer episode storage
- ❌ LSTM training with sequences

#### C3. Intrinsic Exploration (RND + Adaptive Annealing)
- ✅ RND novelty computation (unit tests)
- ❌ Adaptive annealing during training
- ❌ Intrinsic weight decrease after consistent performance
- ❌ Combined reward training (extrinsic + intrinsic)

#### C4. Curriculum Signal Purity
- ✅ 9 tests in test_curriculum_signal_purity.py (existing)
- ❌ Signal purity with intrinsic rewards active
- ❌ Stage transitions during multi-episode training

---

### Category D: Data Flow Integration

**Purpose**: Test data flows through the entire pipeline.

**Pattern**: Trace data from source to sink, verify transformations.

**Tests needed**:

#### D1. Observation Pipeline
- Environment builds observation → Population receives correct dims
- Partial observability (5×5 window) → Recurrent network forward pass
- Temporal features → Recurrent network temporal encoder
- Observation updates after agent movement

#### D2. Reward Pipeline
- Environment computes extrinsic reward → Population receives
- Exploration computes intrinsic reward → Population combines
- Combined reward → ReplayBuffer stores
- Combined reward → Q-network training (but curriculum sees only extrinsic!)

#### D3. Action Pipeline
- Q-network computes Q-values → Exploration selects actions
- Exploration applies epsilon-greedy → Actions
- Exploration applies action masking → Valid actions
- Actions → Environment executes → Next state

#### D4. Training Pipeline
- Experience → ReplayBuffer → Batch sampling
- Batch → Q-network forward → Q-values
- Batch → Target network forward → Target Q-values
- TD error → Loss → Backprop → Weight update
- Periodic target network sync

---

## 7. Recommended Test Structure

### Integration Test Organization (Tasks 11-13)

```
tests/test_townlet/integration/
├── test_component_pairs.py           # Task 11a: Component pair tests (A1-A5)
├── test_episode_execution.py         # Task 11b: Single episode tests (B1)
├── test_training_loop.py             # Task 11c: Multi-episode training (B2)
├── test_curriculum_integration.py    # Task 11d: Curriculum-guided training (B3)
├── test_checkpointing.py             # Task 11e: Checkpoint round-trip (B4)
├── test_temporal_mechanics.py        # Task 12a: Temporal features (C1)
├── test_recurrent_networks.py        # Task 12b: LSTM integration (C2)
├── test_intrinsic_exploration.py     # Task 12c: RND + adaptive (C3)
├── test_curriculum_signal_purity.py  # Task 12d: Signal purity (C4, migrate existing)
├── test_data_flows.py                # Task 13a: Pipeline tests (D1-D4)
└── test_runner_integration.py        # Task 13b: DemoRunner end-to-end
```

### End-to-End Test Organization (Task 14)

```
tests/test_townlet/e2e/
├── test_complete_training_run.py     # Full training run (100 episodes)
├── test_checkpoint_resume.py         # Save → Stop → Load → Resume
├── test_curriculum_progression.py    # Stage 1 → Stage 2 → Stage 3
└── test_generalization.py            # Affordance randomization test
```

### Migration Plan for Existing Tests

**Step 1**: Move existing integration tests to new structure

| Old File | New Location | Tests |
|----------|--------------|-------|
| `test_curriculum_signal_purity.py` | `integration/test_curriculum_signal_purity.py` | 9 |
| `test_p1_1_phase*.py` (6 files) | `integration/test_checkpointing.py` | 38 |
| `test_affordance_integration.py` | `integration/test_component_pairs.py` | 13 |
| `test_temporal_integration.py` | `integration/test_temporal_mechanics.py` | 5 |
| `test_masked_loss_integration.py` | `integration/test_training_loop.py` | 3 |
| `test_recording/test_runner_integration.py` | `integration/test_runner_integration.py` | 3 |

**Step 2**: Consolidate and reorganize

- Merge P1.1 phase tests into single `test_checkpointing.py` with clear sections
- Extract component pair tests from affordance integration
- Add missing tests for identified gaps

**Step 3**: Create new integration tests

- Episode execution tests (B1)
- Training loop tests (B2)
- Data flow tests (D1-D4)

---

## 8. Coverage Gaps

### Critical Gaps (Must Address in Tasks 11-13)

#### Gap 1: LSTM Hidden State During Training

**What's missing**: Verify hidden state behavior during multi-step episodes and batch training.

**Why critical**: Hidden state bugs are silent killers (agents perform poorly but no error).

**Tests needed**:
- Hidden state persists across 10 steps within episode
- Hidden state resets after death
- Hidden state resets after flush (max_steps survival)
- Hidden state batch size correct during training (16 vs num_agents)

**Estimated effort**: 4 tests, ~2 hours

---

#### Gap 2: Episode Flushing Verification

**What's missing**: Verify that episodes are flushed to replay buffer at correct times.

**Why critical**: Without flushing, successful episodes never trained on (data loss).

**Tests needed**:
- Flush on death (already tested in P1.1 Phase 5)
- Flush on max_steps survival (already tested in P1.1 Phase 5)
- Replay buffer contains episodes after flush
- Multi-episode training with flush verification

**Estimated effort**: 2 tests (leverage existing), ~1 hour

---

#### Gap 3: Full Checkpoint Round-Trip

**What's missing**: Save checkpoint → Load checkpoint → Verify exact equivalence (not just "exists").

**Why critical**: Multi-day training relies on perfect resume.

**Tests needed**:
- Train 10 episodes → Save checkpoint → Load → Verify Q-network weights identical
- Train 10 episodes → Save → Load → Train 10 more → Verify training metrics continuous
- Train with curriculum advancement → Save → Load → Verify curriculum stage preserved

**Estimated effort**: 3 tests, ~3 hours

---

#### Gap 4: Signal Purity During Training

**What's missing**: Verify curriculum receives pure survival signals during multi-episode training with intrinsic rewards active.

**Why critical**: Contaminated signals break adversarial curriculum.

**Tests needed**:
- Train 10 episodes with intrinsic rewards → Verify curriculum tracker only sees survival time
- Train with curriculum advancement → Verify stage changes based on survival, not rewards

**Estimated effort**: 2 tests, ~2 hours

---

#### Gap 5: Action Masking During Training

**What's missing**: Verify that Q-network respects action masks during training (doesn't learn invalid actions).

**Why critical**: Invalid actions break action space validity.

**Tests needed**:
- Train with temporal masking → Verify Q-values for masked actions remain low
- Train near grid boundaries → Verify agents don't learn to walk off grid

**Estimated effort**: 2 tests, ~2 hours

---

### Important Gaps (Address if Time Permits)

#### Gap 6: Data Flow End-to-End

**What's missing**: Trace data through entire pipeline to verify transformations.

**Tests needed**:
- Observation → Q-network → Actions → Environment → Next observation (full cycle)
- Reward computation → ReplayBuffer → Batch sampling → Training (reward flow)

**Estimated effort**: 4 tests, ~3 hours

---

#### Gap 7: Multi-Component Training Loop

**What's missing**: Verify that all components work together during realistic training.

**Tests needed**:
- Train 50 episodes → Verify learning progression (survival time increases)
- Train 50 episodes → Verify Q-values improve over time
- Train 50 episodes → Verify epsilon decays correctly

**Estimated effort**: 3 tests, ~3 hours

---

### Minor Gaps (Nice to Have)

#### Gap 8: TensorBoard Logging Integration

**What's missing**: Verify metrics logged correctly during training.

**Tests needed**:
- Train 10 episodes → Verify TensorBoard files written
- Train 10 episodes → Verify metrics parseable

**Estimated effort**: 2 tests, ~1 hour

---

#### Gap 9: Database Logging Integration

**What's missing**: Verify database records created during training.

**Tests needed**:
- Train 10 episodes → Verify database rows created
- Train 10 episodes → Verify metrics match training output

**Estimated effort**: 2 tests, ~1 hour

---

## 9. Priority Ranking

### Priority 1: Critical Path (Must Complete for Task 11)

| Test Category | Priority Reason | Estimated Effort |
|---------------|----------------|------------------|
| **Checkpointing** | Multi-day training depends on this | 3 hours |
| **LSTM Hidden State** | Silent failures break POMDP learning | 2 hours |
| **Signal Purity** | Breaks adversarial curriculum | 2 hours |
| **Episode Execution** | Foundation for all other tests | 2 hours |

**Total Priority 1**: ~9 hours

---

### Priority 2: High Value (Task 12)

| Test Category | Priority Reason | Estimated Effort |
|---------------|----------------|------------------|
| **Training Loop** | Verifies learning progression | 3 hours |
| **Episode Flushing** | Prevents data loss | 1 hour |
| **Action Masking** | Prevents invalid strategies | 2 hours |
| **Temporal Mechanics** | New feature, needs validation | 2 hours |

**Total Priority 2**: ~8 hours

---

### Priority 3: Medium Value (Task 13)

| Test Category | Priority Reason | Estimated Effort |
|---------------|----------------|------------------|
| **Data Flow** | Comprehensive pipeline verification | 3 hours |
| **Component Pairs** | Granular boundary testing | 3 hours |
| **Intrinsic Exploration** | Adaptive annealing verification | 2 hours |

**Total Priority 3**: ~8 hours

---

### Priority 4: Nice to Have (Future)

| Test Category | Priority Reason | Estimated Effort |
|---------------|----------------|------------------|
| **TensorBoard Logging** | Observability, not critical | 1 hour |
| **Database Logging** | Observability, not critical | 1 hour |
| **Generalization Tests** | Already manually tested | 2 hours |

**Total Priority 4**: ~4 hours

---

## 10. Effort Estimate

### Task Breakdown

| Task | Scope | Tests | Effort (hours) |
|------|-------|-------|----------------|
| **Task 11a** | Component pair integration (A1-A5) | ~20 tests | 4 hours |
| **Task 11b** | Episode execution (B1) | ~5 tests | 2 hours |
| **Task 11c** | Training loop (B2) | ~5 tests | 3 hours |
| **Task 11d** | Curriculum integration (B3) | ~4 tests | 2 hours |
| **Task 11e** | Checkpointing (B4, consolidate P1.1) | ~10 tests | 3 hours |
| **Task 12a** | Temporal mechanics (C1, migrate existing) | ~8 tests | 2 hours |
| **Task 12b** | Recurrent networks (C2) | ~5 tests | 2 hours |
| **Task 12c** | Intrinsic exploration (C3) | ~4 tests | 2 hours |
| **Task 12d** | Signal purity (C4, migrate existing) | ~9 tests | 1 hour |
| **Task 13a** | Data flows (D1-D4) | ~8 tests | 3 hours |
| **Task 13b** | Runner integration (consolidate) | ~5 tests | 2 hours |
| **Task 14** | End-to-end tests (NEW) | ~4 tests | 4 hours |

**Total estimated effort**: ~30 hours

**Estimated completion time** (at 10 min/test average from Tasks 1-10): **Faster than 30 hours due to existing tests**

**Realistic estimate**: ~20 hours (many tests already exist, just need reorganization)

---

### Phased Approach

**Phase 1** (Priority 1): Tasks 11b, 11d, 11e → Critical path (9 hours)

**Phase 2** (Priority 2): Tasks 11c, 12a, 12b, 12d → High value (8 hours)

**Phase 3** (Priority 3): Tasks 11a, 12c, 13a, 13b → Medium value (10 hours)

**Phase 4** (Priority 4): Task 14 → End-to-end tests (4 hours)

**Total**: ~31 hours across 4 phases

---

## 11. Implementation Recommendations

### Testing Patterns to Follow

#### Pattern 1: Real Components, Minimal Mocking

```python
def test_episode_execution_feedforward():
    """Full episode with real components (feedforward network)."""
    device = torch.device("cpu")

    # Real components
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

#### Pattern 2: State Verification

```python
def test_checkpoint_round_trip():
    """Verify checkpoint save/load preserves exact state."""
    # Train for N episodes
    runner = DemoRunner(...)
    runner.run()  # Train 10 episodes

    # Save checkpoint
    checkpoint_path = runner.save_checkpoint()

    # Record state before load
    q_weights_before = runner.population.q_network.state_dict()
    curriculum_stage_before = runner.curriculum.tracker.agent_stages[0].item()

    # Create new runner and load
    runner2 = DemoRunner(...)
    runner2.load_checkpoint(checkpoint_path)

    # Verify exact state match
    q_weights_after = runner2.population.q_network.state_dict()
    curriculum_stage_after = runner2.curriculum.tracker.agent_stages[0].item()

    for key in q_weights_before.keys():
        assert torch.allclose(q_weights_before[key], q_weights_after[key]), f"Q-network weights mismatch at {key}"

    assert curriculum_stage_before == curriculum_stage_after, "Curriculum stage mismatch"
```

#### Pattern 3: Data Flow Tracing

```python
def test_reward_pipeline():
    """Trace rewards from environment → population → replay buffer."""
    # Setup
    env, population = setup_env_and_population()

    # Reset
    population.reset()
    obs_before = population.current_obs.clone()

    # Step
    agent_state = population.step_population(env)

    # Verify reward flow
    assert agent_state.rewards.shape == (1,), "Rewards should be [num_agents]"
    assert agent_state.intrinsic_rewards.shape == (1,), "Intrinsic rewards should be [num_agents]"

    # Verify combined reward
    intrinsic_weight = population.exploration.get_intrinsic_weight()
    expected_combined = agent_state.rewards_extrinsic + agent_state.intrinsic_rewards * intrinsic_weight
    assert torch.allclose(agent_state.rewards, expected_combined), "Reward combination incorrect"

    # Verify stored in replay buffer
    if len(population.replay_buffer) > 0:
        sample = population.replay_buffer.sample(1)
        # Reward in buffer should be combined reward
        assert sample["rewards"][0] == agent_state.rewards[0], "Replay buffer stores combined reward"
```

### Testing Best Practices

1. **Use CPU device for determinism** (like unit tests)
2. **Small environments** (5×5 grid, 1-2 agents) for speed
3. **Short episodes** (50 steps max) unless testing max_steps behavior
4. **Behavioral assertions** (not exact values) for resilience
5. **Clear test names** describing the integration point
6. **Fixtures for common setups** (add to conftest.py)
7. **Mark slow tests** with `@pytest.mark.slow` (can skip during rapid development)

---

## 12. Conclusion

### Summary

After completing Tasks 1-10 (381 unit tests), we have **excellent component-level coverage** but need **integration-level verification** to ensure components work together correctly.

**Key findings**:

1. **71 integration tests already exist** across multiple files (need reorganization)
2. **5 critical rub points** identified (LSTM state, signal purity, flushing, checkpointing, masking)
3. **~30 hours estimated** to complete integration test suite (Tasks 11-14)
4. **Phased approach recommended** (Priority 1 → 2 → 3 → 4)

### Next Steps

**Immediate** (Task 11): Focus on critical path tests
- Checkpointing round-trip (consolidate P1.1 tests)
- Episode execution (single episode end-to-end)
- Training loop (multi-episode learning)
- Curriculum integration (stage transitions)

**Near-term** (Task 12): Add feature-specific tests
- Temporal mechanics (consolidate existing)
- Recurrent networks (LSTM state management)
- Signal purity (migrate existing)
- Intrinsic exploration (adaptive annealing)

**Long-term** (Task 13-14): Complete coverage
- Data flow tests (pipeline verification)
- Runner integration (DemoRunner end-to-end)
- End-to-end tests (full training runs)

### Success Criteria

✅ All existing integration tests migrated to new structure
✅ All critical rub points covered by tests
✅ All integration test categories represented
✅ Tests pass consistently (no flakiness)
✅ Tests run in reasonable time (<5 min total)

---

**END OF RESEARCH DOCUMENT**
