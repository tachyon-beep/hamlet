# Adversarial Curriculum

**Status:** Phase 2 Complete ✅

## Overview

AdversarialCurriculum is an auto-tuning difficulty system that progressively challenges agents through 5 stages, from easy shaped rewards to full sparse reward challenge. Unlike StaticCurriculum (Phase 1 baseline), AdversarialCurriculum adapts difficulty based on per-agent performance metrics.

## Architecture

**Multi-Signal Decision Logic:**
```python
def should_advance(agent):
    survival_rate = agent.steps / max_steps
    learning_progress = current_reward - baseline_reward
    entropy = calculate_entropy(q_values)

    return (
        survival_rate > 0.7 AND
        learning_progress > 0 AND
        entropy < 0.5
    )
```

**Components:**
- `AdversarialCurriculum`: Main curriculum manager
- `PerformanceTracker`: Per-agent metrics (survival, learning, entropy)
- `StageConfig`: Stage specifications (meters, depletion, rewards)

## 5-Stage Progression

| Stage | Active Meters | Depletion | Reward Mode | Description |
|-------|--------------|-----------|-------------|-------------|
| 1 | energy, hygiene | 0.2x | shaped | Basic survival needs |
| 2 | +satiation | 0.5x | shaped | Add hunger management |
| 3 | +money | 0.8x | shaped | Add economic planning |
| 4 | +mood, social | 1.0x | shaped | Full complexity |
| 5 | all 6 meters | 1.0x | **sparse** | Graduation! |

**Key insight:** Stage 5 is the only sparse reward stage. Stages 1-4 provide dense gradient signals to learn basic skills before the final challenge.

## Decision Metrics

### 1. Survival Rate
```python
survival_rate = episode_steps / max_steps_per_episode
```

**Thresholds:**
- Advance: > 0.7 (surviving 70%+ of episode)
- Retreat: < 0.3 (dying early)

### 2. Learning Progress
```python
learning_progress = current_avg_reward - prev_avg_reward
```

**Thresholds:**
- Advance: > 0 (improving)
- Retreat: < 0 (regressing)

### 3. Action Entropy
```python
entropy = -sum(p * log(p)) / log(num_actions)
```

**Threshold:**
- Advance gate: < 0.5 (converged policy)
- High entropy: Still exploring randomly

**Why entropy matters:** Prevents premature advancement when agent is still exploring. Only advance when policy has converged (low entropy).

## Usage

### Basic Usage

```python
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.population.vectorized import VectorizedPopulation

# Create curriculum
curriculum = AdversarialCurriculum(
    max_steps_per_episode=500,
    survival_advance_threshold=0.7,
    survival_retreat_threshold=0.3,
    entropy_gate=0.5,
    min_steps_at_stage=1000,
    device=torch.device('cuda'),
)

# Create population with curriculum
population = VectorizedPopulation(
    num_agents=32,
    curriculum=curriculum,
    # ...
)

# Training loop
for episode in range(num_episodes):
    envs.reset()
    population.reset(envs)

    for step in range(max_steps):
        agent_state = population.step_population(envs)

        # IMPORTANT: Update curriculum tracker
        population.update_curriculum_tracker(
            agent_state.rewards,
            agent_state.dones,
        )
```

### YAML Configuration

```yaml
curriculum:
  type: adversarial
  max_steps_per_episode: 500
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 1000
  device: cuda
```

Load from config:
```python
curriculum = AdversarialCurriculum.from_yaml('configs/my_config.yaml')
```

### Quick Testing Config

For fast iteration during development, use `configs/curriculum_quick_test.yaml`:
- Lower thresholds (advance at 50% survival instead of 70%)
- Fewer steps required (50 instead of 1000)
- Results in rapid progression through stages for testing

## Checkpointing

Save/restore curriculum state:

```python
# Checkpoint
checkpoint = {
    'curriculum': curriculum.state_dict(),
    'population': population.state_dict(),
}
torch.save(checkpoint, 'checkpoint.pt')

# Restore
checkpoint = torch.load('checkpoint.pt')
curriculum.load_state_dict(checkpoint['curriculum'])
population.load_state_dict(checkpoint['population'])
```

**Curriculum state includes:**
- Agent stages
- Episode rewards/steps
- Reward baselines
- Steps at current stage

## Integration with VectorizedPopulation

VectorizedPopulation automatically:
1. Passes Q-values to curriculum for entropy calculation
2. Calls `update_curriculum_tracker()` after each step
3. Provides curriculum decisions to environment

**No manual intervention needed** - population handles all integration.

## Testing

Run full test suite:
```bash
uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py -v
```

Run end-to-end progression test:
```bash
uv run pytest tests/test_townlet/test_curriculum_progression.py -v
```

Run long progression test (slow):
```bash
uv run pytest tests/test_townlet/test_curriculum_progression.py -m slow -v
```

## Expected Behavior

**Typical progression timeline** (with default thresholds):
- **Episodes 0-50:** Stage 1 (learning basic movement + bed/shower)
- **Episodes 50-150:** Stage 2 (adding fridge management)
- **Episodes 150-300:** Stage 3 (learning job + money)
- **Episodes 300-500:** Stage 4 (mood + social complexity)
- **Episodes 500+:** Stage 5 (sparse reward challenge)

**Individual variation:** Agents progress at different rates. Some may advance faster, others may retreat temporarily when struggling.

## Design Rationale

**Why 5 stages?**
- Gradual complexity increase prevents overwhelming agents
- Each stage introduces 1-2 new concepts
- Shaped rewards (stages 1-4) build foundational skills
- Sparse rewards (stage 5) test true understanding

**Why per-agent progression?**
- Population diversity: Some agents explore different strategies
- Robust learning: Faster learners don't wait for slower ones
- Better curriculum signal: More data points for tuning

**Why entropy gating?**
- Prevents premature advancement during random exploration
- Ensures policy convergence before increasing difficulty
- Reduces regression after advancement

## Common Pitfalls

❌ **Forgetting to call `update_curriculum_tracker()`**
- Metrics won't update → no advancement
- Always call after `step_population()`

❌ **Setting min_steps_at_stage too low**
- Agents advance before learning → immediate retreat
- Use 1000+ for real training, 50-100 for testing only

❌ **Using wrong reward thresholds**
- Too high: Never advance (stuck at stage 1)
- Too low: Advance prematurely → fail → retreat loop

❌ **Not saving curriculum state in checkpoints**
- Training resumes from stage 1 → wasted time
- Always include curriculum.state_dict() in checkpoints

## Future Enhancements (Phase 3+)

- **Adaptive thresholds:** Auto-tune advancement criteria
- **Population-level decisions:** Consider population distribution
- **Curriculum rollback:** Revert entire population when regression detected
- **Multi-objective rewards:** Balance survival + exploration + efficiency
