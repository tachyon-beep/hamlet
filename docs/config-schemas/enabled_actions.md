# enabled_actions Configuration

**Purpose**: Control which actions from global vocabulary are available in this config.

**Location**: `training.yaml` → `training.enabled_actions`

**Pattern**: All curriculum levels share the same action vocabulary (same `action_dim`).
Disabled actions are masked out at runtime but still occupy action IDs, so checkpoints stay compatible.
Set `enabled_actions: null` (or omit the field) to enable the entire vocabulary, or use an explicit list to gate
actions per config. Passing an empty list intentionally disables every action (useful for curriculum tests).

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

### L0_0_minimal/training.yaml (excerpt)

```yaml
training:
  device: cuda
  max_episodes: 500
  # ... other hyperparameters ...
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
**Live reference**: `configs/L0_0_minimal/training.yaml`

### L1_full_observability/training.yaml (excerpt)

```yaml
training:
  # ...
  enabled_actions:
    - "UP"
    - "DOWN"
    - "LEFT"
    - "RIGHT"
    - "INTERACT"
    - "WAIT"
    - "REST"
    - "MEDITATE"
```

**Result**: 10 enabled, 0 disabled, action_dim = 10
**Live reference**: `configs/L0_5_dual_resource/training.yaml`

## Checkpoint Transfer

Both L0 and L1 have **action_dim = 10**, so checkpoints transfer!

L0 Q-network outputs 10 Q-values (3 disabled actions get masked).
L1 Q-network outputs 10 Q-values (all actions available).

**Same architecture → checkpoint compatible.**

## Implementation

**Phase 1 (Current)**: No formal DTO validation. Configs manually specify enabled_actions.

**Phase 2 (TASK-004A)**: TrainingConfig Pydantic DTO validates `enabled_actions` (duplicates, empty entries) and the
compiler validates names against the global vocabulary.

## Best Practices

1. **Define global vocabulary first** (`configs/global_actions.yaml`)
2. **Freeze vocabulary** before training starts (adding actions breaks checkpoint transfer)
3. **Enable progressively** across curriculum (L0 → L1 → L2 adds more actions)
4. **Test action_dim** matches across all configs (use integration tests)
5. **Document disabled actions** in comments (explain why not enabled yet)

## Validation (Future - TASK-004A)

TrainingConfig + Stage 1 validation enforce:
- Entries are non-empty, deduplicated strings (trimmed automatically)
- All listed names must exist in the combined substrate + `configs/global_actions.yaml` vocabulary
- Compiler raises `[UAC-ACT-001]` with file/line context when a name is invalid

## Migration Checklist

1. Add an explicit `enabled_actions` list under the `training:` block of every pack (copy one of the `configs/L0_*/training.yaml` examples).
2. Keep the list ordered and documented (comments explain why certain custom actions stay disabled).
3. Run `uv run pytest tests/test_townlet/unit/universe/test_raw_configs.py` to ensure Stage 1 sees the new mask and that action metadata reflects the intended unlocks.
