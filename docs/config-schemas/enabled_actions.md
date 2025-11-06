# enabled_actions Configuration

**Purpose**: Control which actions from global vocabulary are available in this config.

**Location**: `training.yaml` → `environment.enabled_actions`

**Pattern**: All curriculum levels share same action vocabulary (same action_dim).
Disabled actions are masked out at runtime but still occupy action IDs.

## Example

### Global Vocabulary (configs/global_actions.yaml)

```yaml
custom_actions:
  - name: "REST"
  - name: "MEDITATE"
  - name: "TELEPORT_HOME"
  - name: "SPRINT"
```

Total actions: 6 substrate (Grid2D) + 4 custom = **10 actions**

### L0_0_minimal/training.yaml

```yaml
environment:
  enabled_actions:
    - "UP"
    - "DOWN"
    - "LEFT"
    - "RIGHT"
    - "INTERACT"
    - "WAIT"
    - "REST"
```

**Result**: 7 enabled, 3 disabled, action_dim = 10

### L1_full_observability/training.yaml

```yaml
environment:
  enabled_actions:
    - "UP"
    - "DOWN"
    - "LEFT"
    - "RIGHT"
    - "INTERACT"
    - "WAIT"
    - "REST"
    - "MEDITATE"
    - "TELEPORT_HOME"
    - "SPRINT"
```

**Result**: 10 enabled, 0 disabled, action_dim = 10

## Checkpoint Transfer

Both L0 and L1 have **action_dim = 10**, so checkpoints transfer!

L0 Q-network outputs 10 Q-values (3 disabled actions get masked).
L1 Q-network outputs 10 Q-values (all actions available).

**Same architecture → checkpoint compatible.**

## Implementation

**Phase 1 (Current)**: No formal DTO validation. Configs manually specify enabled_actions.

**Phase 2 (TASK-004A)**: TrainingConfig Pydantic DTO will validate enabled_actions list.

## Best Practices

1. **Define global vocabulary first** (`configs/global_actions.yaml`)
2. **Freeze vocabulary** before training starts (adding actions breaks checkpoint transfer)
3. **Enable progressively** across curriculum (L0 → L1 → L2 adds more actions)
4. **Test action_dim** matches across all configs (use integration tests)
5. **Document disabled actions** in comments (explain why not enabled yet)

## Validation (Future - TASK-004A)

TrainingConfig DTO will validate:
- All enabled_actions exist in global vocabulary
- No duplicate action names
- No typos in action names
