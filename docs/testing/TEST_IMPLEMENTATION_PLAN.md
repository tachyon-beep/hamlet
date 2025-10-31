# Test Implementation Plan - Red-Green Refactoring

**Created:** October 31, 2025  
**Purpose:** Systematic test implementation before refactoring  
**Target Coverage:** 70-80% on modules being refactored  
**Current Coverage:** 16% (250/1521 statements)

---

## Testing Philosophy

### Goals

1. **Characterization Tests:** Capture current behavior before refactoring
2. **Bug Detection:** Find insidious bugs preventing convergence
3. **Safety Net:** Enable confident refactoring with immediate feedback
4. **Documentation:** Tests serve as executable specifications

### Non-Goals

- Don't test for "correct" strategies (emergent behavior is valuable)
- Don't optimize during testing (find issues first)
- Don't assume code is correct (validate everything)

---

## Risk-Based Priority Matrix

### 🔴 CRITICAL PRIORITY (Week 1)

**Environment Core - Meter Dynamics & Death Conditions**

- **Why First:** Agents not converging suggests bugs in fundamental game rules
- **Risk:** Cascading meter effects may have subtle bugs
- **Impact:** If meters are wrong, everything built on them is wrong

### 🟡 HIGH PRIORITY (Week 2)

**Training Loop & Q-Learning**

- **Why Second:** Training may have issues preventing convergence
- **Risk:** DQN updates, replay buffer, gradient flow
- **Impact:** Directly affects learning performance

### 🟢 MEDIUM PRIORITY (Week 3)

**Curriculum & Exploration**

- **Why Third:** These adapt based on core systems
- **Risk:** Logic bugs in advancement/annealing
- **Impact:** Affects learning speed, not correctness

---

## Week 1: Critical Foundation Tests

### Phase 1.1: Meter System Validation (Days 1-2)

#### Test File: `test_meter_dynamics.py`

**Test Suite 1: Base Depletion**

```python
def test_energy_depletes_at_correct_rate()
    # Energy should deplete 0.5% per step
    # Test: 200 steps = 100% depletion (death)
    
def test_hygiene_depletes_at_correct_rate()
    # Hygiene should deplete 0.3% per step
    
def test_satiation_depletes_at_correct_rate()
    # Satiation should deplete 0.4% per step
    
def test_money_does_not_deplete_passively()
    # Money should only change via interactions
    
def test_mood_depletes_at_correct_rate()
    # Mood should deplete 0.1% per step
    
def test_social_depletes_at_correct_rate()
    # Social should deplete 0.6% per step (fastest!)
```

**Test Suite 2: Fitness-Modulated Health Depletion** 🚨 HIGH BUG RISK

```python
def test_health_depletion_with_high_fitness()
    # fitness > 0.7 → health depletes at 0.05% per step
    # Verify multiplier is 0.5x baseline
    
def test_health_depletion_with_medium_fitness()
    # 0.3 ≤ fitness ≤ 0.7 → health depletes at 0.1% per step
    # Verify baseline rate
    
def test_health_depletion_with_low_fitness()
    # fitness < 0.3 → health depletes at 0.3% per step
    # Verify 3x penalty - BUG SUSPECT: May be too aggressive?
    
def test_fitness_decay_affects_health_over_time()
    # As fitness decays, health depletion should accelerate
    # This creates death spiral - is threshold correct?
```

**Test Suite 3: Cascading Secondary → Primary** 🚨 HIGH BUG RISK

```python
def test_low_satiation_damages_health()
    # satiation < 0.3 → health penalty scales with deficit
    # Formula: 0.004 * deficit (0.4% at threshold, ~0.8% at 0)
    # BUG SUSPECT: Is this too punishing?
    
def test_low_satiation_damages_energy()
    # satiation < 0.3 → energy penalty scales with deficit
    # Formula: 0.005 * deficit (0.5% at threshold, ~1.0% at 0)
    # BUG SUSPECT: Double penalty (satiation + energy) may be excessive
    
def test_low_mood_damages_energy()
    # mood < 0.3 → energy penalty scales with deficit
    # Formula: 0.005 * deficit
    # BUG SUSPECT: Combined with satiation, agent may never recover energy
    
def test_satiation_threshold_is_correct()
    # Verify 0.3 (30%) is the right threshold
    # Below this, cascading effects trigger
    # BUG SUSPECT: May be too high, causing premature death spirals
```

**Test Suite 4: Cascading Tertiary → Secondary** 🚨 MODERATE BUG RISK

```python
def test_low_hygiene_damages_satiation()
    # hygiene < 0.3 → satiation penalty 0.002 * deficit
    # "Being dirty → loss of appetite"
    
def test_low_hygiene_damages_fitness()
    # hygiene < 0.3 → fitness penalty 0.002 * deficit
    # "Being dirty → harder to exercise"
    
def test_low_hygiene_damages_mood()
    # hygiene < 0.3 → mood penalty 0.003 * deficit
    # "Being dirty → feel bad"
    
def test_low_social_damages_mood()
    # social < 0.3 → mood penalty 0.004 * deficit
    # Strongest tertiary effect
    # BUG SUSPECT: Is social too important?
```

**Test Suite 5: Death Condition Validation** 🚨 CRITICAL

```python
def test_death_when_health_reaches_zero()
    # Health = 0 → done = True
    
def test_death_when_energy_reaches_zero()
    # Energy = 0 → done = True
    
def test_no_death_from_other_meters_at_zero()
    # hygiene, satiation, mood, social, fitness at 0
    # Should NOT die directly (only via cascading to primaries)
    
def test_death_spiral_from_low_satiation()
    # Start with satiation = 0.2, other meters normal
    # Track steps until death
    # BUG SUSPECT: May die too quickly, making game unwinnable
    
def test_death_spiral_from_low_fitness()
    # Start with fitness = 0.2, other meters normal
    # Health should accelerate to 0
    
def test_combined_cascade_death_spiral()
    # Multiple low meters (satiation + hygiene + social)
    # Should create faster death than single meter
    # Verify cascading is additive not multiplicative
```

**Test Suite 6: Meter Clamping** 🚨 LOW BUG RISK

```python
def test_meters_cannot_exceed_1_0()
    # Apply massive positive benefits
    # Verify all meters clamp at 1.0
    
def test_meters_cannot_go_below_0_0()
    # Apply massive negative effects
    # Verify all meters clamp at 0.0
    
def test_money_cannot_go_negative()
    # Attempt to spend more than available
    # Should fail interaction or clamp at 0
```

---

### Phase 1.2: Action Execution & Interaction Validation (Day 3)

#### Test File: `test_action_execution.py`

**Test Suite 1: Movement Mechanics**

```python
def test_movement_costs_are_correct()
    # UP/DOWN/LEFT/RIGHT should cost:
    # energy: -0.5%, hygiene: -0.3%, satiation: -0.4%
    # BUG SUSPECT: Are these costs too high for 8×8 grid?
    
def test_boundary_clamping()
    # Agents at edges should not move beyond grid
    # UP at y=0, DOWN at y=7, etc.
    
def test_movement_resets_interaction_progress()
    # If agent moves away from affordance mid-interaction
    # Progress should reset to 0
```

**Test Suite 2: Action Masking** 🚨 HIGH BUG RISK

```python
def test_boundary_actions_are_masked()
    # Agent at top edge: UP should be masked (False)
    # Agent at bottom edge: DOWN should be masked
    # etc.
    
def test_interact_masked_when_not_on_affordance()
    # INTERACT should be masked when not standing on affordance
    
def test_interact_masked_when_cannot_afford()
    # INTERACT should be masked when money < cost
    # BUG SUSPECT: Is this check happening correctly?
    
def test_interact_masked_when_affordance_closed()
    # Temporal mechanics: Job closed at night
    # INTERACT should be masked at closed affordances
    
def test_action_masks_prevent_invalid_q_value_selection()
    # Masked actions should have -inf Q-values
    # Argmax should never select masked actions
```

**Test Suite 3: Affordance Interactions** 🚨 CRITICAL

```python
def test_bed_restores_energy_correctly()
    # Bed: 5 ticks × 0.075 linear + 0.125 completion = +50% energy
    # Also: +2% health on completion
    # Cost: 5 × $1 = $5 total
    
def test_job_provides_money_and_costs_energy()
    # Job: 4 ticks → $22.50 total, -15% energy, +2% social, -3% health
    # Verify all effects apply correctly
    
def test_park_is_free()
    # Park should cost $0
    # Should provide fitness + social + mood
    # BUG SUSPECT: Is park too good? Makes it dominant strategy?
    
def test_early_exit_preserves_progress()
    # Start interaction, get 3/5 ticks
    # Move away, return
    # Progress should still be 3/5 (or reset - verify current behavior)
    
def test_completion_bonus_only_on_full_completion()
    # Do 4/5 ticks of Bed
    # Should get linear benefits only, NO completion bonus
    
def test_money_charged_per_tick()
    # Not all at once
    # Verify money decreases gradually during interaction
```

---

### Phase 1.3: Reward Calculation (Day 4)

#### Test File: `test_reward_calculation.py`

**Test Suite 1: Milestone Rewards** 🚨 CRITICAL

```python
def test_decade_milestone_bonus()
    # Every 10 steps: +0.5 reward (if alive)
    # Step 10, 20, 30, ..., 100
    
def test_century_milestone_bonus()
    # Every 100 steps: +5.0 reward (if alive)
    # Step 100, 200, 300, ...
    # BUG SUSPECT: Is +5.0 reward at 100 steps too generous?
    
def test_death_penalty()
    # Death: -100.0 reward
    # BUG SUSPECT: Is -100 penalty too harsh? Dominates all positive rewards
    
def test_milestone_timing_alignment()
    # Step 100 should give BOTH +0.5 (decade) AND +5.0 (century)
    # Or just +5.0? Verify intended behavior
    
def test_no_reward_accumulation_during_survival()
    # Between milestones, reward should be 0
    # This prevents aimless wandering from being rewarded
    # Verify no per-step rewards are leaking in
```

**Test Suite 2: Disabled Reward Systems** 🚨 VERIFY DISABLED

```python
def test_proximity_rewards_are_disabled()
    # Stand near affordances
    # Should get 0 reward (no proximity shaping)
    # BUG SUSPECT: Verify this is truly disabled, not just commented
    
def test_complex_meter_rewards_are_disabled()
    # Per-step meter-based rewards should be disabled
    # Verify not being calculated or added to reward
    
def test_only_milestone_rewards_are_active()
    # Comprehensive check: only milestone system contributes to reward
    # No meter rewards, no proximity rewards
```

---

### Phase 1.4: Observation Construction (Day 5)

#### Test File: `test_observation_construction.py`

**Test Suite 1: Full Observability**

```python
def test_full_obs_dimensions()
    # 8×8 grid = 64 dims (one-hot position)
    # 8 meters = 8 dims
    # 15 affordances + 1 none = 16 dims (one-hot current affordance)
    # Total: 88 dims (or +2 for temporal = 90 dims)
    
def test_full_obs_agent_position_encoding()
    # Agent at (3, 5) → one-hot index = 5*8 + 3 = 43
    # Verify correct flattening (row-major order)
    
def test_full_obs_meter_values()
    # Meters should be normalized [0, 1]
    # Money: $50 = 0.5 in observation
    
def test_full_obs_current_affordance_encoding()
    # Standing on Bed → affordance one-hot has Bed bit set
    # Not on affordance → "none" bit set
```

**Test Suite 2: Partial Observability** 🚨 MODERATE BUG RISK

```python
def test_partial_obs_dimensions()
    # 5×5 window = 25 dims
    # position (x, y) = 2 dims (normalized)
    # 8 meters = 8 dims
    # 15 affordances + 1 none = 16 dims
    # Total: 51 dims (or +2 for temporal = 53 dims)
    
def test_partial_obs_local_window_centering()
    # Agent at (4, 4) sees window from (2,2) to (6,6)
    # Verify window is correctly centered
    
def test_partial_obs_boundary_handling()
    # Agent at (0, 0) sees window from (0,0) to (2,2)
    # Out-of-bounds positions should be 0 (empty)
    
def test_partial_obs_affordance_detection()
    # Affordance within 5×5 window → bit set in local grid
    # Affordance outside window → not visible
    # BUG SUSPECT: Is this detection working correctly?
    
def test_partial_obs_position_normalization()
    # Position (7, 7) on 8×8 grid → (1.0, 1.0) in obs
    # Position (0, 0) → (0.0, 0.0)
```

---

## Week 2: High Priority Training Loop Tests

### Phase 2.1: Replay Buffer (Day 6)

#### Test File: `test_replay_buffer.py`

**Test Suite 1: Storage & Retrieval**

```python
def test_replay_buffer_circular_eviction()
    # Fill buffer beyond capacity
    # Oldest transitions should be evicted (FIFO)
    
def test_replay_buffer_stores_dual_rewards()
    # Store extrinsic + intrinsic separately
    # Sample returns combined: extrinsic + intrinsic * weight
    
def test_replay_buffer_sampling_is_uniform()
    # Statistical test: sample 1000 times
    # Each transition should appear ~equally often
    
def test_replay_buffer_batch_size_validation()
    # Cannot sample batch_size > buffer_size
```

**Test Suite 2: Reward Combination** 🚨 HIGH BUG RISK

```python
def test_intrinsic_weight_scaling()
    # Store: extrinsic=10, intrinsic=5
    # Sample with weight=0.5 → combined=12.5
    # Verify scaling is correct
    
def test_intrinsic_weight_annealing_affects_sampling()
    # As weight decreases, combined reward approaches extrinsic
    # Verify this transition is smooth
```

---

### Phase 2.2: Q-Network Training (Days 7-8)

#### Test File: `test_q_network_training.py`

**Test Suite 1: SimpleQNetwork**

```python
def test_simple_qnetwork_forward_pass_shapes()
    # Input: [batch, obs_dim] → Output: [batch, 5]
    
def test_simple_qnetwork_learns_from_experience()
    # Train on simple pattern (state → high-reward action)
    # Verify Q-values increase for that action
    
def test_simple_qnetwork_gradient_flow()
    # Verify gradients are non-zero and reasonable magnitude
    # BUG SUSPECT: Are gradients vanishing or exploding?
```

**Test Suite 2: RecurrentSpatialQNetwork** 🚨 HIGH BUG RISK

```python
def test_recurrent_qnetwork_forward_pass_shapes()
    # Input: [batch, obs_dim] → Output: ([batch, 5], hidden_state)
    
def test_recurrent_qnetwork_hidden_state_persistence()
    # Call forward twice with same batch
    # Hidden state should change (LSTM memory)
    
def test_recurrent_qnetwork_hidden_state_reset()
    # Reset hidden state → should be all zeros
    
def test_recurrent_qnetwork_uses_history()
    # Present sequence: [state1, state2, state3]
    # Q-values at state3 should differ based on history
    # BUG SUSPECT: Is LSTM actually using history? Or acting like MLP?
    
def test_recurrent_qnetwork_batch_training_independence()
    # During batch training, transitions should be independent
    # Hidden state should be reset between samples
    # Verify this is happening correctly
```

**Test Suite 3: DQN Update Logic** 🚨 CRITICAL BUG RISK

```python
def test_dqn_td_error_calculation()
    # TD target: reward + γ * max(Q(s'))
    # Verify calculation is correct
    # BUG SUSPECT: Is gamma being applied correctly?
    
def test_dqn_terminal_state_handling()
    # When done=True, target should be just reward (no Q(s'))
    # Verify done mask is applied correctly
    
def test_dqn_gradient_clipping()
    # Gradients should be clipped at max_norm=10.0
    # Verify this is preventing gradient explosion
    
def test_dqn_update_frequency()
    # Q-network should update every 4 steps, batch_size=64
    # Verify timing is correct
    
def test_dqn_replay_buffer_warmup()
    # Should not train until buffer has ≥64 samples
    # Verify no training happens on first 63 steps
```

---

### Phase 2.3: VectorizedPopulation Integration (Day 9)

#### Test File: `test_vectorized_population.py`

**Test Suite 1: Step Orchestration**

```python
def test_step_population_returns_correct_state()
    # step_population should return BatchedAgentState
    # with all required fields populated
    
def test_step_population_integrates_all_components()
    # Verify: curriculum → exploration → env → replay → training
    # All components called in correct order
    
def test_step_population_handles_episode_resets()
    # When done=True, hidden state should reset
    # Annealing should update
    # Counters should reset
```

**Test Suite 2: Action Selection Integration** 🚨 HIGH BUG RISK

```python
def test_action_masking_prevents_invalid_actions()
    # With action masks, agent should never select invalid actions
    # Run 1000 steps, verify 100% valid actions
    # BUG SUSPECT: Are masked actions still being selected?
    
def test_epsilon_greedy_respects_masks()
    # Random exploration should only sample from valid actions
    # Verify mask is applied during random selection
    
def test_greedy_selection_respects_masks()
    # Greedy (epsilon=0) should argmax over valid actions only
    # Invalid actions should have -inf Q-values
```

---

## Week 3: Medium Priority Curriculum & Exploration Tests

### Phase 3.1: Curriculum Logic (Days 10-11)

#### Test File: `test_adversarial_curriculum.py`

**Test Suite 1: Stage Progression**

```python
def test_curriculum_advancement_logic()
    # survival_rate > 0.7 + learning_progress > 0 + entropy < 0.5
    # → should advance stage
    
def test_curriculum_retreat_logic()
    # survival_rate < 0.3 OR learning_progress < 0
    # → should retreat stage
    
def test_curriculum_minimum_steps_requirement()
    # Cannot advance until 1000 steps at current stage
    # Verify this prevents premature advancement
    
def test_curriculum_stage_boundaries()
    # Cannot retreat below stage 1
    # Cannot advance beyond stage 5
```

**Test Suite 2: Performance Tracking** 🚨 MODERATE BUG RISK

```python
def test_survival_rate_calculation()
    # survival_rate = episode_steps / max_steps
    # Verify calculation is correct
    
def test_learning_progress_calculation()
    # learning_progress = current_avg_reward - prev_avg_reward
    # BUG SUSPECT: Is baseline being updated correctly?
    
def test_entropy_calculation()
    # entropy = -sum(p * log(p)) for action distribution
    # Should be normalized to [0, 1]
    # Verify softmax and log calculations
```

---

### Phase 3.2: RND & Intrinsic Motivation (Day 12)

#### Test File: `test_rnd_exploration.py`

**Test Suite 1: RND Networks**

```python
def test_rnd_fixed_network_is_frozen()
    # Fixed network parameters should have requires_grad=False
    # Verify weights never change
    
def test_rnd_predictor_network_trains()
    # Predictor should minimize MSE with fixed network
    # After training, prediction error should decrease
    
def test_rnd_novelty_signal()
    # Novel states → high prediction error
    # Familiar states → low prediction error
    # Present same state 100 times, verify error decreases
```

**Test Suite 2: Adaptive Annealing** 🚨 HIGH BUG RISK

```python
def test_annealing_requires_low_variance_and_high_survival()
    # variance < 100.0 AND mean_survival > 50
    # → should trigger annealing
    # BUG SUSPECT: This was recently fixed, verify fix is correct
    
def test_annealing_does_not_trigger_on_consistent_failure()
    # low variance + low survival (< 50 steps)
    # → should NOT anneal
    # This was the bug: "consistently failing" triggered annealing
    
def test_intrinsic_weight_decays_exponentially()
    # weight *= 0.99 each annealing
    # Verify exponential decay curve
    
def test_intrinsic_weight_reaches_minimum()
    # Should not decay below min_intrinsic_weight
    # Verify floor is respected
```

---

## Week 4: Integration & Edge Case Tests

### Phase 4.1: End-to-End Integration (Days 13-14)

#### Test File: `test_full_training_integration.py`

```python
def test_agent_can_survive_with_correct_policy()
    # Manually control agent with optimal policy
    # Should survive indefinitely (or very long)
    # Verifies environment is winnable
    
def test_agent_learns_to_survive_longer_over_time()
    # Train for 1000 episodes
    # Average survival time should increase
    # BUG SUSPECT: If not increasing, something is fundamentally broken
    
def test_curriculum_progression_occurs()
    # Train until stage advancement
    # Verify agent actually progresses through stages
    
def test_intrinsic_weight_anneals_during_training()
    # Train for extended period
    # Verify weight decreases from 1.0 toward 0.0
    
def test_no_nan_losses_during_training()
    # Run training for 1000 steps
    # All losses should be finite
    # BUG SUSPECT: NaN losses indicate numerical instability
    
def test_no_memory_leaks()
    # Run training for 10K steps
    # Memory usage should be stable
    # Verify tensors are being freed
```

### Phase 4.2: Edge Cases & Stress Tests (Day 15)

#### Test File: `test_edge_cases.py`

```python
def test_all_meters_at_zero_except_primaries()
    # Set all secondary/tertiary meters to 0
    # Agent should still function (cascading will damage primaries)
    
def test_agent_with_no_money_can_use_free_affordances()
    # money = 0 → Job, Labor, Park should still work
    
def test_temporal_mechanics_midnight_wraparound()
    # Bar open at 18-4 (crosses midnight)
    # Verify time=23 and time=0 both allow entry
    
def test_interaction_interrupted_by_closing_time()
    # Start Job at 17:00 (5pm)
    # Time advances to 18:00 (6pm) - Job closes
    # Verify agent cannot continue interaction
    
def test_multiple_agents_do_not_interfere()
    # Run 2+ agents in parallel
    # Verify state is independent (no tensor dimension errors)
```

---

## Testing Commands

```bash
# Run all new tests
pytest tests/test_townlet/ -v

# Run specific test file
pytest tests/test_townlet/test_meter_dynamics.py -v

# Run with coverage on specific module
pytest tests/test_townlet/test_meter_dynamics.py --cov=src/townlet/environment/vectorized_env --cov-report=term-missing

# Run tests matching pattern
pytest tests/test_townlet/ -k "meter" -v

# Run until first failure (useful when debugging)
pytest tests/test_townlet/ -x

# Run with verbose output and print statements
pytest tests/test_townlet/ -v -s
```

---

## Bug Hypothesis Tracking

As we write tests, track suspected bugs here:

### 🚨 HIGH PRIORITY SUSPECTS

1. **Cascading Meter Penalties Too Aggressive**
   - Hypothesis: Low satiation triggers both health AND energy penalties
   - Combined with mood→energy, agent may be unable to recover
   - Test: `test_death_spiral_from_low_satiation`

2. **Fitness Death Spiral**
   - Hypothesis: Low fitness (< 0.3) causes 3x health depletion
   - May create unrecoverable death spiral
   - Test: `test_health_depletion_with_low_fitness`

3. **Action Masking Not Preventing Invalid Actions**
   - Hypothesis: Masked actions still being selected during exploration
   - Would explain poor learning (agent wasting actions on invalid moves)
   - Test: `test_action_masking_prevents_invalid_actions`

4. **LSTM Not Using History**
   - Hypothesis: Recurrent network may be acting like MLP
   - Hidden state might not be carrying information
   - Test: `test_recurrent_qnetwork_uses_history`

5. **Reward Signal Too Weak**
   - Hypothesis: -100 death penalty dominates +5 century bonus
   - Agent may never see enough positive signal to learn
   - Test: `test_death_penalty` and `test_century_milestone_bonus`

---

## Success Criteria

### By End of Week 1

- [ ] 60%+ coverage on `vectorized_env.py`
- [ ] All meter dynamics tests passing
- [ ] Identified at least 2 bugs in meter cascading

### By End of Week 2

- [ ] 70%+ coverage on training loop modules
- [ ] All Q-network tests passing
- [ ] Verified DQN update logic is correct

### By End of Week 3

- [ ] 70%+ overall coverage on townlet/
- [ ] All curriculum/exploration tests passing
- [ ] Confirmed intrinsic motivation is working

### By End of Week 4

- [ ] 80%+ coverage target achieved
- [ ] All integration tests passing
- [ ] Documented all bugs found
- [ ] Ready to begin refactoring with confidence

---

**Remember:** Test code is as important as production code. Write clear, maintainable tests that serve as documentation.
