# Training Loop v1.0 → v2.0: Action Plan

**Date:** November 2, 2025  
**Status:** Analysis Complete, Ready for Implementation  
**Priority:** P1 items are BLOCKING for multi-day training

---

## Executive Summary

This document provides a detailed implementation plan for upgrading the training loop from "research prototype" to "production multi-day trainer." All issues have been verified against the current codebase.

**Key Findings:**

- ✅ All P1, P2, and P3 issues are **CONFIRMED** in current code
- 🔴 P1 issues will corrupt long runs or prevent valid resume
- 🟡 P2 issues block intelligent behavior (planning, budgeting)
- 🟠 P3 issues will become critical as agents survive longer

---

## 🔴 P1: Critical Stability & Correctness Blockers

### P1.1: Incomplete Checkpoint / Unsafe Resume

**Status:** ✅ VERIFIED - Critical Gap in Checkpointing System

**Current State (Verified):**

```python
# runner.py lines 93-119
checkpoint = {
    "episode": self.current_episode,
    "timestamp": time.time(),
    "population_state": {
        "q_network": self.population.q_network.state_dict(),
        "optimizer": self.population.optimizer.state_dict(),
    },
    "exploration_state": self.population.exploration.checkpoint_state(),  # Partial
    "epsilon": self.exploration.rnd.epsilon,  # Redundant with exploration_state
}
```

**Missing Components:**

1. ❌ Target network (recurrent mode only)
2. ❌ Training counters (`total_steps`, `training_step_counter`)
3. ❌ Curriculum state (stage, tracker stats)
4. ❌ Replay buffer contents
5. ❌ Environment layout (affordance positions)

**Implementation Plan:**

#### Step 1: Add Versioning (15 min)

```python
# In save_checkpoint()
checkpoint = {
    "version": 2,  # Add versioning for backwards compatibility
    "episode": self.current_episode,
    "timestamp": time.time(),
    # ... rest
}
```

#### Step 2: Save Target Network (10 min)

```python
# In save_checkpoint() - after line 104
if self.population.target_network is not None:
    checkpoint["population_state"]["target_network"] = self.population.target_network.state_dict()
```

```python
# In load_checkpoint() - after line 145
if "target_network" in checkpoint["population_state"] and self.population.target_network:
    self.population.target_network.load_state_dict(checkpoint["population_state"]["target_network"])
```

**Why Critical:** Without target network state, resume causes TD error spike that destabilizes training for ~100 steps.

#### Step 3: Save Training Counters (10 min)

```python
# In save_checkpoint()
checkpoint["training_counters"] = {
    "total_steps": self.population.total_steps,
    "training_step_counter": self.population.training_step_counter if self.population.is_recurrent else 0,
}
```

```python
# In load_checkpoint()
if "training_counters" in checkpoint:
    self.population.total_steps = checkpoint["training_counters"]["total_steps"]
    if self.population.is_recurrent:
        self.population.training_step_counter = checkpoint["training_counters"]["training_step_counter"]
```

**Why Critical:** Controls target network sync frequency. Loss of counter causes immediate sync on resume.

#### Step 4: Save Curriculum State (20 min)

```python
# In save_checkpoint()
if hasattr(self.curriculum, "checkpoint_state"):
    checkpoint["curriculum_state"] = self.curriculum.checkpoint_state()
else:
    # Fallback for curricula without checkpoint support
    checkpoint["curriculum_state"] = {
        "type": type(self.curriculum).__name__,
    }
```

```python
# In load_checkpoint()
if "curriculum_state" in checkpoint and hasattr(self.curriculum, "load_state"):
    self.curriculum.load_state(checkpoint["curriculum_state"])
```

**Action Required:** Add `checkpoint_state()` and `load_state()` methods to `AdversarialCurriculum`:

- Save: `tracker.agent_stages`, `tracker.steps_at_stage`, `tracker.prev_avg_reward`
- Load: Restore all tracker state

**Why Critical:** Resume drops back to Stage 1 even though policy was trained at Stage 3+. Causes curriculum-policy mismatch.

#### Step 5: Save Replay Buffer (2-3 hours) 🔴 MOST COMPLEX

```python
# In save_checkpoint()
checkpoint["replay_buffer_state"] = self.population.replay_buffer.serialize()
```

```python
# In load_checkpoint()
if "replay_buffer_state" in checkpoint:
    self.population.replay_buffer.load_from_serialized(checkpoint["replay_buffer_state"])
```

**Action Required:** Implement serialization in both buffer classes:

**For `ReplayBuffer` (training/replay_buffer.py):**

```python
def serialize(self) -> dict:
    """Serialize buffer contents for checkpointing."""
    return {
        "observations": self.observations[:self.size].cpu(),
        "actions": self.actions[:self.size].cpu(),
        "rewards_extrinsic": self.rewards_extrinsic[:self.size].cpu(),
        "rewards_intrinsic": self.rewards_intrinsic[:self.size].cpu(),
        "next_observations": self.next_observations[:self.size].cpu(),
        "dones": self.dones[:self.size].cpu(),
        "size": self.size,
        "position": self.position,
    }

def load_from_serialized(self, state: dict) -> None:
    """Restore buffer from serialized state."""
    self.size = state["size"]
    self.position = state["position"]
    self.observations[:self.size] = state["observations"].to(self.device)
    self.actions[:self.size] = state["actions"].to(self.device)
    self.rewards_extrinsic[:self.size] = state["rewards_extrinsic"].to(self.device)
    self.rewards_intrinsic[:self.size] = state["rewards_intrinsic"].to(self.device)
    self.next_observations[:self.size] = state["next_observations"].to(self.device)
    self.dones[:self.size] = state["dones"].to(self.device)
```

**For `SequentialReplayBuffer` (training/sequential_replay_buffer.py):**

```python
def serialize(self) -> dict:
    """Serialize episode buffer for checkpointing."""
    return {
        "episodes": [
            {
                "observations": [obs.cpu() for obs in ep["observations"]],
                "actions": [act.cpu() for act in ep["actions"]],
                "rewards_extrinsic": [r.cpu() for r in ep["rewards_extrinsic"]],
                "rewards_intrinsic": [r.cpu() for r in ep["rewards_intrinsic"]],
                "dones": [d.cpu() for d in ep["dones"]],
            }
            for ep in self.episodes[:self.size]
        ],
        "size": self.size,
        "position": self.position,
    }

def load_from_serialized(self, state: dict) -> None:
    """Restore episode buffer from serialized state."""
    self.size = state["size"]
    self.position = state["position"]
    self.episodes = []
    for ep_state in state["episodes"]:
        self.episodes.append({
            "observations": [obs.to(self.device) for obs in ep_state["observations"]],
            "actions": [act.to(self.device) for act in ep_state["actions"]],
            "rewards_extrinsic": [r.to(self.device) for r in ep_state["rewards_extrinsic"]],
            "rewards_intrinsic": [r.to(self.device) for r in ep_state["rewards_intrinsic"]],
            "dones": [d.to(self.device) for d in ep_state["dones"]],
        })
```

**Why Critical:** Resume with empty buffer causes severe overfitting to first few post-resume transitions. Policy collapse risk.

#### Step 6: Save Environment Layout (30 min)

```python
# In save_checkpoint()
checkpoint["environment_state"] = {
    "affordance_positions": self.env.get_affordance_positions(),
    "randomized": self.current_episode >= 5000,  # Track if we've randomized
}
```

```python
# In load_checkpoint()
if "environment_state" in checkpoint:
    # Restore affordance positions (critical for post-5000 generalization phase)
    saved_positions = checkpoint["environment_state"]["affordance_positions"]
    self.env.set_affordance_positions(saved_positions)
```

**Action Required:** Add to `VectorizedHamletEnv`:

```python
def get_affordance_positions(self) -> dict[str, list[int]]:
    """Get current affordance positions for checkpointing."""
    return {name: pos.tolist() for name, pos in self.affordances.items()}

def set_affordance_positions(self, positions: dict[str, list[int]]) -> None:
    """Restore affordance positions from checkpoint."""
    for name, pos_list in positions.items():
        if name in self.affordances:
            self.affordances[name] = torch.tensor(pos_list, device=self.device)
```

**Why Critical:** Episode 5000 randomizes layout. Resume without restoring positions silently teleports back to default layout, invalidating all post-5000 metrics.

**Total Estimated Time:** 5-6 hours

---

### P1.2: `max_steps` Survival Memory Leak & Data Loss

**Status:** ✅ VERIFIED - Critical Bug in Episode End Logic

**Current State (Verified):**

```python
# runner.py lines 310-349
for step in range(max_steps):
    agent_state = self.population.step_population(self.env)
    # ... step logic ...
    
    if agent_state.dones[0]:
        final_meters = self.env.meters[0].cpu()
        break

# After loop: NO CLEANUP if agent survives to max_steps!
if final_meters is None:
    final_meters = self.env.meters[0].cpu()
# Missing: flush episode, update curriculum, annealing
```

**In `VectorizedPopulation` (lines 425-490):**

- Recurrent mode accumulates episodes in `self.current_episodes[...]`
- Only flushes to `SequentialReplayBuffer` when `dones.any()` is True
- If agent survives `max_steps`, episode stays in accumulator → memory leak
- Also: no exploration annealing, no curriculum update

**Problem Impact:**

1. **Memory Leak:** `current_episodes[0]` grows unbounded for successful agents
2. **Data Loss:** Best survival trajectories never make it to replay buffer
3. **Schedule Corruption:** Exploration annealing doesn't run, intrinsic weight stuck
4. **Curriculum Stall:** No update signal when agent survives full episode

**Implementation Plan:**

#### Step 1: Add Flush Helper to VectorizedPopulation (1 hour)

```python
# In population/vectorized.py - add new method

def flush_episode(self, agent_idx: int, synthetic_done: bool = False) -> None:
    """
    Flush current episode for an agent to replay buffer.
    
    Used when:
    - Agent dies (real done)
    - Episode hits max_steps (synthetic done)
    
    Args:
        agent_idx: Index of agent to flush
        synthetic_done: If True, treat as done even if environment didn't signal it
    """
    if not self.is_recurrent:
        # Feedforward mode: transitions already in buffer
        return
    
    episode = self.current_episodes[agent_idx]
    if len(episode["observations"]) == 0:
        return  # Nothing to flush
    
    # Store episode in sequential buffer
    self.replay_buffer.store_episode(
        observations=episode["observations"],
        actions=episode["actions"],
        rewards_extrinsic=episode["rewards_extrinsic"],
        rewards_intrinsic=episode["rewards_intrinsic"],
        dones=episode["dones"],
    )
    
    # Update exploration annealing
    survival_time = len(episode["observations"])
    if isinstance(self.exploration, AdaptiveIntrinsicExploration):
        self.exploration.update_on_episode_end(survival_time=survival_time)
    
    # Clear accumulator
    self.current_episodes[agent_idx] = {
        "observations": [],
        "actions": [],
        "rewards_extrinsic": [],
        "rewards_intrinsic": [],
        "dones": [],
    }
    
    # Reset hidden state for this agent
    if self.q_network and hasattr(self.q_network, "reset_hidden_state"):
        h, c = self.q_network.get_hidden_state()
        if h is not None and c is not None:
            h[:, agent_idx, :] = 0.0
            c[:, agent_idx, :] = 0.0
            self.q_network.set_hidden_state((h, c))
```

#### Step 2: Use Flush Helper in Runner (30 min)

```python
# In runner.py - after the for step in range(max_steps): loop

if agent_state.dones[0]:
    # Agent died - normal path
    final_meters = self.env.meters[0].cpu()
else:
    # Agent survived to max_steps - synthetic done
    final_meters = self.env.meters[0].cpu()
    
    # Flush episode to replay buffer and clean up
    self.population.flush_episode(agent_idx=0, synthetic_done=True)
    
    # Update curriculum with survival signal
    # (see P1.3 for curriculum update refactor)
```

#### Step 3: Remove Duplicate Flush Logic (20 min)

Currently in `VectorizedPopulation.step_population()` around lines 425-490, there's episode flush logic under `if dones.any():`. This should be refactored to call the new `flush_episode()` helper to avoid duplication.

**Total Estimated Time:** 1.5 hours

---

### P1.3: Curriculum Signal Purity

**Status:** ✅ VERIFIED - Per-Step Update Contaminates Signal

**Current State (Verified):**

```python
# runner.py line 325
for step in range(max_steps):
    agent_state = self.population.step_population(self.env)
    
    # This runs EVERY STEP (500 times per episode!)
    self.population.update_curriculum_tracker(extrinsic_reward_tensor, agent_state.dones)
```

**Problems:**

1. Curriculum tracker updated every step with meaningless mid-episode rewards
2. Reward calculation is only valid at terminal state (baseline-relative)
3. No curriculum update when agent survives to `max_steps`

**Implementation Plan:**

#### Step 1: Remove Per-Step Update (5 min)

```python
# In runner.py - DELETE line 325
# self.population.update_curriculum_tracker(...)  # DELETE THIS
```

#### Step 2: Add Single Per-Episode Update (15 min)

```python
# In runner.py - after episode ends

if agent_state.dones[0]:
    # Agent died
    final_meters = self.env.meters[0].cpu()
else:
    # Agent survived to max_steps
    final_meters = self.env.meters[0].cpu()
    self.population.flush_episode(agent_idx=0, synthetic_done=True)

# Curriculum update ONCE per episode with pure survival signal
curriculum_survival_tensor = torch.tensor([survival_time], dtype=torch.float32, device=self.device)
curriculum_done_tensor = torch.tensor([True], dtype=torch.bool, device=self.device)

self.population.update_curriculum_tracker(curriculum_survival_tensor, curriculum_done_tensor)
```

**Why This Works:**

- Curriculum sees exactly ONE update per episode
- Update value = integer steps survived (pure, interpretable)
- No reward hacking from curiosity
- No accounting noise from mid-episode partials
- Works for both death and max_steps survival

**Total Estimated Time:** 20 minutes

---

## 🟡 P2: High Priority Logic & Behavior Issues

### P2.1: Economic Planning Impossible (INTERACT Masking Bug)

**Status:** ✅ VERIFIED - Action Masking Blocks Long-Horizon Planning

**Current State (Verified):**

```python
# vectorized_env.py lines 238-242
# Check affordability using AffordanceEngine
cost_mode = "per_tick" if self.enable_temporal_mechanics else "instant"
cost_normalized = self.affordance_engine.get_affordance_cost(affordance_name, cost_mode)
can_afford = self.meters[:, 3] >= cost_normalized

# Valid if on affordance AND can afford it AND is open
on_affordable_affordance |= on_this_affordance & can_afford

action_masks[:, 4] = on_affordable_affordance  # INTERACT masked when broke!
```

**Problem Impact:**

- Agent at Hospital with $0: INTERACT is masked off
- Q-network never sees INTERACT as an option when broke
- Agent cannot learn: "I need money → go to Job → come back to Hospital"
- Economic planning is impossible because the action space changes based on money

**This is analogous to:** Showing a chess player only legal moves. They never learn why castling through check is bad, they just never see it as an option.

**Implementation Plan:**

#### Step 1: Change Action Masking Logic (30 min)

```python
# In vectorized_env.py get_action_masks() - replace lines 217-244

def get_action_masks(self) -> torch.Tensor:
    """
    Get valid action masks for all agents.
    
    INTERACT is valid if:
    - Agent is standing on an affordance
    - Affordance is open (not closed by time-of-day)
    
    INTERACT is NOT masked by affordability - let the agent learn consequences!
    """
    action_masks = torch.ones(self.num_agents, 5, dtype=torch.bool, device=self.device)

    # Boundary constraints (unchanged)
    at_top = self.positions[:, 1] == 0
    at_bottom = self.positions[:, 1] == self.grid_size - 1
    at_left = self.positions[:, 0] == 0
    at_right = self.positions[:, 0] == self.grid_size - 1

    action_masks[at_top, 0] = False
    action_masks[at_bottom, 1] = False
    action_masks[at_left, 2] = False
    action_masks[at_right, 3] = False

    # INTERACT masking: physical + temporal only (NOT affordability)
    on_open_affordance = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)

    for affordance_name, affordance_pos in self.affordances.items():
        distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
        on_this_affordance = distances == 0

        # Check operating hours (temporal constraint)
        if self.enable_temporal_mechanics:
            if not self.affordance_engine.is_affordance_open(affordance_name, self.time_of_day):
                continue  # Affordance closed, skip
        
        # If on affordance and it's open, INTERACT is valid
        # Do NOT check affordability - let agent learn consequences
        on_open_affordance |= on_this_affordance

    action_masks[:, 4] = on_open_affordance

    return action_masks
```

#### Step 2: Handle Unaffordable Interactions (30 min)

```python
# In vectorized_env.py _handle_interactions() and _handle_interactions_legacy()

# Both methods need this check added at the start:

def _handle_interactions(self, interact_mask: torch.Tensor) -> dict:
    """Handle INTERACT with multi-tick."""
    if not self.enable_temporal_mechanics:
        return self._handle_interactions_legacy(interact_mask)
    
    successful_interactions = {}
    
    for affordance_name, affordance_pos in self.affordances.items():
        distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
        at_affordance = (distances == 0) & interact_mask
        
        if not at_affordance.any():
            continue
        
        # Check affordability
        cost_per_tick = self.affordance_engine.get_affordance_cost(affordance_name, cost_mode="per_tick")
        can_afford = self.meters[:, 3] >= cost_per_tick
        
        # Agents who can't afford simply waste the tick
        # No crash, no special error, just no effect
        at_affordable_affordance = at_affordance & can_afford
        
        if not at_affordable_affordance.any():
            # Agent mashed INTERACT while broke - wasted tick
            # Meters still decay, time advances, no benefit
            continue
        
        # ... rest of interaction logic for affordable agents ...
```

**Why This Works:**

- Agent at Hospital with $0 can try INTERACT
- Nothing happens, meters decay, time passes
- Agent gets negative signal (reward decreases because survival decreases)
- Agent learns: "Being broke at Hospital is bad"
- Eventually learns: "Go to Job first, then go to Hospital"

This is the **core economic loop** the environment was designed to teach!

**Total Estimated Time:** 1 hour

---

### P2.2: Double Reset at Episode Start

**Status:** ✅ VERIFIED - Redundant Environment Reset

**Current State (Verified):**

```python
# runner.py lines 296-297
self.env.reset()
self.population.reset()  # This ALSO calls self.env.reset()!
```

**In `VectorizedPopulation.reset()` line 148:**

```python
def reset(self) -> None:
    """Reset population for new episode."""
    self.env.reset()  # DUPLICATE!
    # ... reset logic ...
```

**Problem Impact:**

- Episode 5000: Randomize affordance positions
- First `self.env.reset()` applies randomization
- Second `self.env.reset()` (inside `population.reset()`) **overwrites** randomization
- Silently reverts to default layout
- Generalization phase is invalidated

**Implementation Plan:**

#### Step 1: Remove Redundant Call (2 min)

```python
# In runner.py lines 296-297 - DELETE line 296
# self.env.reset()  # DELETE THIS LINE
self.population.reset()  # Keep only this
```

#### Step 2: Verify Population.reset() Behavior (5 min)

Check that `VectorizedPopulation.reset()` properly handles:

- Environment reset
- Hidden state reset (recurrent mode)
- Epsilon initialization

Currently it does all of this correctly, so no changes needed there.

**Total Estimated Time:** 10 minutes

---

## 🟠 P3: Medium Priority / Future Proofing

### P3.1: Post-Terminal Sequence Masking (Recurrent Branch)

**Status:** ✅ VERIFIED - LSTM Training Includes Post-Death Garbage

**Current State (Verified):**

In recurrent training mode:

1. `SequentialReplayBuffer.sample_sequences()` samples fixed-length sequences (e.g., 8 steps)
2. If episode ended at step 3, steps 4-8 are post-death frames
3. Training loop computes loss across all 8 steps equally
4. Bootstrapping masks `(1 - done)` in target (good)
5. But we still force Q-network to predict values for garbage frames (bad)

**Problem Impact:**

- Injects noise into value function
- Fuzzy credit assignment across terminal transitions
- Worse as agents survive longer (more post-terminal garbage in sequences)

**Implementation Plan:**

#### Step 1: Add Mask to Buffer Sampling (1 hour)

```python
# In sequential_replay_buffer.py sample_sequences()

def sample_sequences(self, batch_size: int, sequence_length: int) -> dict:
    """Sample sequences with validity masks."""
    # ... existing sampling logic ...
    
    # Add validity mask: True until first done, False after
    masks = []
    for seq_obs, seq_dones in zip(batch_obs, batch_dones):
        seq_mask = torch.ones(len(seq_dones), dtype=torch.bool)
        for t, done in enumerate(seq_dones):
            if done:
                # Mask out everything after this done
                seq_mask[t+1:] = False
                break
        masks.append(seq_mask)
    
    return {
        "observations": batch_obs,
        "actions": batch_actions,
        "rewards_extrinsic": batch_rewards_ext,
        "rewards_intrinsic": batch_rewards_int,
        "next_observations": batch_next_obs,
        "dones": batch_dones,
        "mask": torch.stack(masks),  # [batch, seq_len]
    }
```

#### Step 2: Apply Mask in Recurrent Training (30 min)

```python
# In vectorized.py _train_batch_recurrent() - around line 380

# Replace:
loss = F.mse_loss(q_pred_all, q_target_all)

# With:
loss_per_timestep = F.mse_loss(q_pred_all, q_target_all, reduction="none")  # [batch, seq_len]
mask = batch["mask"]  # [batch, seq_len]
loss = (loss_per_timestep * mask).sum() / mask.sum()
```

**Why This Matters:**

- Only compute loss on valid timesteps
- Prevents gradients from post-terminal garbage
- Better credit assignment for terminal transitions
- Becomes critical as survival time increases (>100 steps)

**Total Estimated Time:** 1.5 hours

---

### P3.2: Exploration State as First-Class Checkpoint Component

**Status:** ✅ VERIFIED - Partial Implementation

**Current State:**

- `exploration.checkpoint_state()` is called ✅
- But implementation quality varies by strategy
- RND exploration saves some state, but not all

**Action Required:**

Audit `adaptive_intrinsic.py` and `rnd.py` to ensure `checkpoint_state()` includes:

- ✅ predictor_network weights
- ❓ predictor_optimizer state (NOT SAVED!)
- ❓ Running stats / normalizers
- ✅ intrinsic_weight
- ✅ epsilon
- ❓ obs_buffer for predictor training
- ❓ survival_history for annealing

**Implementation Plan:**

#### Audit and Fix RND Checkpoint (1 hour)

```python
# In rnd.py checkpoint_state()

def checkpoint_state(self) -> dict:
    """Save complete exploration state."""
    return {
        "predictor_network": self.predictor_network.state_dict(),
        "predictor_optimizer": self.predictor_optimizer.state_dict(),  # ADD THIS
        "fixed_network": self.fixed_network.state_dict(),  # Technically not needed (frozen)
        "epsilon": self.epsilon,
        "obs_buffer": [obs.cpu() for obs in self.obs_buffer],  # ADD THIS
        "obs_buffer_size": len(self.obs_buffer),
    }

def load_state(self, state: dict) -> None:
    """Restore complete exploration state."""
    self.predictor_network.load_state_dict(state["predictor_network"])
    self.predictor_optimizer.load_state_dict(state["predictor_optimizer"])  # ADD THIS
    self.epsilon = state["epsilon"]
    
    # Restore obs buffer
    if "obs_buffer" in state:
        self.obs_buffer = [obs.to(self.device) for obs in state["obs_buffer"]]
```

#### Audit and Fix AdaptiveIntrinsic Checkpoint (30 min)

```python
# In adaptive_intrinsic.py checkpoint_state()

def checkpoint_state(self) -> dict:
    """Save complete adaptive exploration state."""
    state = self.rnd.checkpoint_state()  # Get RND state
    state.update({
        "intrinsic_weight": self.current_intrinsic_weight,
        "survival_history": list(self.survival_history),  # ADD THIS
        "variance_threshold": self.variance_threshold,
        "survival_window": self.survival_window,
    })
    return state

def load_state(self, state: dict) -> None:
    """Restore complete adaptive exploration state."""
    self.rnd.load_state(state)
    self.current_intrinsic_weight = state["intrinsic_weight"]
    
    if "survival_history" in state:
        self.survival_history = deque(state["survival_history"], maxlen=self.survival_window)
```

**Total Estimated Time:** 1.5 hours

---

### P3.3: Affordance Layout Persistence

**Status:** ✅ COVERED BY P1.1 Step 6

This is already included in the P1 checkpoint expansion (environment state).

---

## Implementation Order

### Phase 1: Critical Fixes (P1) - 8-10 hours

1. **P1.3** Curriculum Signal Purity (20 min) - Easiest, high impact
2. **P2.2** Double Reset (10 min) - Quick win
3. **P1.2** Max Steps Survival (1.5 hours) - Memory leak must be fixed
4. **P1.1** Complete Checkpointing (5-6 hours) - Largest but essential
   - Do in order: versioning → target net → counters → curriculum → affordances → replay buffer

### Phase 2: Behavior Unlocks (P2) - 1 hour

1. **P2.1** INTERACT Masking (1 hour) - Unlocks economic planning

### Phase 3: Future Proofing (P3) - 3 hours

1. **P3.2** Exploration Checkpoint Audit (1.5 hours)
2. **P3.1** Post-Terminal Masking (1.5 hours)

**Total Estimated Time:** 12-14 hours

---

## Testing Strategy

### After Each P1 Fix

1. Start short training run (100 episodes)
2. Save checkpoint at episode 50
3. Kill process
4. Resume from checkpoint
5. Verify:
   - No error messages
   - Metrics continuous (no jumps)
   - TensorBoard logs smooth across resume
   - Memory usage stable

### After P2.1 (INTERACT Masking)

1. Watch agent at Hospital with $0
2. Confirm INTERACT action is available (not masked)
3. Confirm no effect when broke
4. Train for 500 episodes
5. Check TensorBoard: Do agents visit Job before Hospital?

### After P3.1 (Sequence Masking)

1. Recurrent training run
2. Monitor loss values - should decrease more smoothly
3. Check agent survival time - should increase faster

### Full Integration Test (After All Fixes)

1. Train for 1000 episodes
2. Checkpoint at 500
3. Kill process
4. Resume from 500
5. Train to 1000
6. Verify:
   - Curriculum stage preserved
   - Epsilon schedule continuous
   - Affordance positions stable (especially if past episode 5000)
   - Memory usage stable
   - No catastrophic forgetting in policy

---

## Success Criteria

### P1 Complete

- ✅ Can resume training after 24+ hour runs without corruption
- ✅ All training state restored correctly
- ✅ Memory usage stable (no leaks)
- ✅ Curriculum stage preserved across resume
- ✅ TensorBoard metrics continuous across resume

### P2 Complete

- ✅ Agents learn economic planning (Job → Hospital loop)
- ✅ No more double resets affecting generalization tests

### P3 Complete

- ✅ Recurrent training loss converges faster
- ✅ Exploration state fully reproducible
- ✅ Resume is bit-for-bit identical to continuous run

---

## Risk Assessment

### High Risk

- **P1.1 Replay Buffer Serialization:** Large tensors, serialization format choices
  - Mitigation: Test with small buffers first, profile memory usage
  
### Medium Risk

- **P2.1 INTERACT Masking:** Changes fundamental learning signal
  - Mitigation: A/B test against current behavior, monitor early training

### Low Risk

- All other changes are well-scoped and testable in isolation

---

## Notes

1. **Backwards Compatibility:** Add version field to checkpoints. On load, detect old format and handle gracefully (either migrate or reject with clear error).

2. **Checkpoint Size:** Full checkpoints will be larger (~100-500 MB with replay buffer). This is acceptable for scientific reproducibility. Consider:
   - Compress with `torch.save(..., _use_new_zipfile_serialization=True)`
   - Keep only last N checkpoints
   - Separate "lightweight" inference checkpoints from "full" training checkpoints

3. **Documentation:** Update README with:
   - Checkpoint format specification
   - Resume procedure
   - What gets saved/restored

4. **Monitoring:** Add checkpoint validation on load:
   - Log what was restored
   - Warn if any components missing
   - Clear error messages for version mismatches

---

## Conclusion

These fixes transform the training loop from "research prototype" to "production multi-day trainer." All issues are real, verified, and have clear implementation paths.

**Priority:** Implement P1 immediately. P2 and P3 can follow in subsequent iterations, but P1 is blocking for any serious multi-day training runs.
