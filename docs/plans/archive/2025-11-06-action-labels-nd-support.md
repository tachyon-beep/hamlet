# N-Dimensional Action Label Support

**Date**: 2025-11-06
**Status**: Planning
**Priority**: P1 (Blocking bug - prevents N-dimensional substrates from initializing)

## Problem Statement

The action label system (`src/townlet/environment/action_labels.py`) currently only supports substrates with 0-3 dimensions. The `_filter_labels_for_substrate` function explicitly rejects any `position_dim` outside this range:

```python
else:
    raise ValueError(f"Invalid position_dim: {position_dim}. Must be 0, 1, 2, or 3.")
```

However, the recent topology work introduced:
- **GridNDSubstrate**: 4D-100D discrete grids
- **ContinuousNDSubstrate**: 4D+ continuous spaces

This creates a **blocking bug**: attempting to initialize a VectorizedHamletEnv with any N≥4 substrate fails immediately with ValueError during `__init__`.

## Root Cause

The action label system was designed before N-dimensional substrate support. It assumes a fixed set of canonical actions (CanonicalAction enum) that only covers up to 3D movement.

## Solution Design: Dynamic Label Generation for N≥4

### Action Space Structure for N-Dimensional Substrates

For N-dimensional substrates (from CLAUDE.md and GridND implementation):
- **Action count**: `2 * N + 2`
- **Actions 0 to N-1**: Negative direction per dimension (move -1 in that dimension)
- **Actions N to 2N-1**: Positive direction per dimension (move +1 in that dimension)
- **Action 2N**: INTERACT
- **Action 2N+1**: WAIT

**Example - 7D Grid**:
- Actions 0-6: D0_NEG, D1_NEG, D2_NEG, D3_NEG, D4_NEG, D5_NEG, D6_NEG
- Actions 7-13: D0_POS, D1_POS, D2_POS, D3_POS, D4_POS, D5_POS, D6_POS
- Action 14: INTERACT
- Action 15: WAIT
- **Total**: 16 actions

### Label Naming Convention

For N≥4 dimensions, use dimension index notation:
- **Negative movement**: `D{i}_NEG` where i ∈ [0, N-1]
- **Positive movement**: `D{i}_POS` where i ∈ [0, N-1]
- **Meta actions**: `INTERACT`, `WAIT`

**Rationale**: High-dimensional spaces (4D+) don't have intuitive directional names like "LEFT/RIGHT" or "NORTH/SOUTH". Dimension index notation is:
- Mathematically clear
- Domain-agnostic
- Scalable to arbitrary dimensions
- Consistent with mathematical conventions (D0=X, D1=Y, D2=Z, D3=W, etc.)

### Preset Behavior for N≥4

All presets (`gaming`, `6dof`, `cardinal`, `math`) will use the same dimension index notation for N≥4:

```python
# Example: 5D substrate with "gaming" preset
labels = {
    0: "D0_NEG", 1: "D0_POS",   # Dimension 0
    2: "D1_NEG", 3: "D1_POS",   # Dimension 1
    4: "D2_NEG", 5: "D2_POS",   # Dimension 2
    6: "D3_NEG", 7: "D3_POS",   # Dimension 3
    8: "D4_NEG", 9: "D4_POS",   # Dimension 4
    10: "INTERACT",
    11: "WAIT",
}
```

**Why uniform labeling?**:
- No canonical "gaming" or "cardinal" directions exist in 4D+
- Prevents confusion (better to be explicit than misleading)
- "math" preset is the natural choice for high-dimensional spaces
- Custom labels still allow domain-specific terminology if needed

### Custom Label Support

Users can still provide custom labels for N-dimensional substrates:

```python
# Example: Custom labels for 4D robotics application
custom_labels = {
    0: "TRANS_X_NEG", 1: "TRANS_X_POS",
    2: "TRANS_Y_NEG", 3: "TRANS_Y_POS",
    4: "TRANS_Z_NEG", 5: "TRANS_Z_POS",
    6: "ROT_ROLL_NEG", 7: "ROT_ROLL_POS",
    8: "INTERACT",
    9: "WAIT",
}

labels = get_labels(custom_labels=custom_labels, substrate_position_dim=4)
```

**Requirement**: Custom labels must provide all `2*N+2` action labels.

### Implementation Changes

#### 1. Extend `_filter_labels_for_substrate`

Add N≥4 case to handle high-dimensional substrates:

```python
def _filter_labels_for_substrate(labels: dict[int, str], position_dim: int) -> dict[int, str]:
    """Filter labels to match substrate's action space.

    ...existing docstring...

    Action space mapping:
    - 0D (Aspatial): INTERACT, WAIT → 2 actions
    - 1D: MOVE_X_NEGATIVE, MOVE_X_POSITIVE, INTERACT, WAIT → 4 actions
    - 2D: + MOVE_Y_NEGATIVE, MOVE_Y_POSITIVE → 6 actions
    - 3D: + MOVE_Z_POSITIVE, MOVE_Z_NEGATIVE → 8 actions
    - N≥4: 2N movement actions + INTERACT + WAIT → 2N+2 actions
    """
    if position_dim == 0:
        # ... existing 0D code ...
    elif position_dim == 1:
        # ... existing 1D code ...
    elif position_dim == 2:
        # ... existing 2D code ...
    elif position_dim == 3:
        # ... existing 3D code ...
    elif position_dim >= 4:
        # N-dimensional: Generate labels programmatically
        filtered = {}

        # Movement actions: D{i}_NEG (index i), D{i}_POS (index N+i)
        for dim in range(position_dim):
            # Negative direction
            neg_idx = dim
            filtered[neg_idx] = labels.get(neg_idx, f"D{dim}_NEG")

            # Positive direction
            pos_idx = position_dim + dim
            filtered[pos_idx] = labels.get(pos_idx, f"D{dim}_POS")

        # Meta actions
        interact_idx = 2 * position_dim
        wait_idx = 2 * position_dim + 1
        filtered[interact_idx] = labels.get(interact_idx, "INTERACT")
        filtered[wait_idx] = labels.get(wait_idx, "WAIT")

        return filtered
    else:
        # This should never happen (position_dim < 0)
        raise ValueError(f"Invalid position_dim: {position_dim}. Must be >= 0.")
```

#### 2. Update Docstrings

Update `get_labels()` docstring to document N≥4 support:

```python
def get_labels(
    preset: str | None = None,
    custom_labels: dict[int, str] | None = None,
    substrate_position_dim: int = 2,
) -> ActionLabels:
    """Get action labels for substrate.

    Args:
        preset: Preset name ("gaming", "6dof", "cardinal", "math") or None for custom
        custom_labels: Custom label dictionary (required if preset=None)
        substrate_position_dim: Substrate dimensionality (0-100)

    Returns:
        ActionLabels instance filtered to substrate's action space

    Notes:
        - For N≥4 dimensions, all presets use dimension index notation (D0_NEG, D0_POS, etc.)
        - Custom labels must provide all 2N+2 labels for N-dimensional substrates

    Examples:
        >>> # 2D gaming labels
        >>> labels = get_labels(preset="gaming", substrate_position_dim=2)
        >>> labels.get_label(0)
        'UP'

        >>> # 7D math labels (auto-generated)
        >>> labels = get_labels(preset="math", substrate_position_dim=7)
        >>> labels.get_label(0)
        'D0_NEG'
        >>> labels.get_label(7)
        'D0_POS'

        >>> # 4D custom robotics labels
        >>> labels = get_labels(
        ...     custom_labels={
        ...         0: "TRANS_X_NEG", 1: "TRANS_X_POS",
        ...         2: "TRANS_Y_NEG", 3: "TRANS_Y_POS",
        ...         4: "TRANS_Z_NEG", 5: "TRANS_Z_POS",
        ...         6: "ROT_ROLL_NEG", 7: "ROT_ROLL_POS",
        ...         8: "INTERACT", 9: "WAIT"
        ...     },
        ...     substrate_position_dim=4
        ... )
        >>> labels.get_label(6)
        'ROT_ROLL_NEG'
    """
```

## Test Plan (TDD Approach)

### Unit Tests: `tests/test_townlet/test_environment/test_action_labels.py`

Create new test file with comprehensive coverage:

#### Test 1: 4D Substrate with Math Preset
```python
def test_action_labels_4d_math_preset():
    """4D substrate should generate D0-D3 labels with math preset."""
    labels = get_labels(preset="math", substrate_position_dim=4)

    # Movement actions
    assert labels.get_label(0) == "D0_NEG"
    assert labels.get_label(1) == "D1_NEG"
    assert labels.get_label(2) == "D2_NEG"
    assert labels.get_label(3) == "D3_NEG"
    assert labels.get_label(4) == "D0_POS"
    assert labels.get_label(5) == "D1_POS"
    assert labels.get_label(6) == "D2_POS"
    assert labels.get_label(7) == "D3_POS"

    # Meta actions
    assert labels.get_label(8) == "INTERACT"
    assert labels.get_label(9) == "WAIT"

    # Action count
    assert labels.get_action_count() == 10  # 2*4 + 2
```

#### Test 2: 7D Substrate with Gaming Preset
```python
def test_action_labels_7d_gaming_preset():
    """7D substrate should use dimension index notation even with gaming preset."""
    labels = get_labels(preset="gaming", substrate_position_dim=7)

    # All presets use D{i}_NEG/D{i}_POS for N≥4
    assert labels.get_label(0) == "D0_NEG"
    assert labels.get_label(6) == "D6_NEG"
    assert labels.get_label(7) == "D0_POS"
    assert labels.get_label(13) == "D6_POS"
    assert labels.get_label(14) == "INTERACT"
    assert labels.get_label(15) == "WAIT"

    assert labels.get_action_count() == 16  # 2*7 + 2
```

#### Test 3: 10D Edge Case
```python
def test_action_labels_10d_edge_case():
    """10D substrate should handle double-digit dimension indices."""
    labels = get_labels(preset="math", substrate_position_dim=10)

    # Check double-digit dimension labels
    assert labels.get_label(9) == "D9_NEG"
    assert labels.get_label(19) == "D9_POS"
    assert labels.get_label(20) == "INTERACT"
    assert labels.get_label(21) == "WAIT"

    assert labels.get_action_count() == 22  # 2*10 + 2
```

#### Test 4: Custom Labels for 5D
```python
def test_action_labels_5d_custom():
    """5D substrate should accept custom labels."""
    custom = {
        0: "AXIS0_NEG", 1: "AXIS1_NEG", 2: "AXIS2_NEG", 3: "AXIS3_NEG", 4: "AXIS4_NEG",
        5: "AXIS0_POS", 6: "AXIS1_POS", 7: "AXIS2_POS", 8: "AXIS3_POS", 9: "AXIS4_POS",
        10: "INTERACT", 11: "WAIT",
    }
    labels = get_labels(custom_labels=custom, substrate_position_dim=5)

    assert labels.get_label(0) == "AXIS0_NEG"
    assert labels.get_label(9) == "AXIS4_POS"
    assert labels.get_label(10) == "INTERACT"
    assert labels.get_label(11) == "WAIT"
```

#### Test 5: Partial Custom Labels (Fallback Behavior)
```python
def test_action_labels_4d_partial_custom():
    """4D substrate with partial custom labels should use fallbacks."""
    # Only provide labels for first 2 dimensions
    custom = {
        0: "LEFT", 1: "RIGHT",  # D0
        2: "DOWN", 3: "UP",     # D1
        # D2 and D3 missing - should fallback to D2_NEG, D2_POS, D3_NEG, D3_POS
    }
    labels = get_labels(custom_labels=custom, substrate_position_dim=4)

    # Custom labels used
    assert labels.get_label(0) == "LEFT"
    assert labels.get_label(1) == "RIGHT"

    # Fallback labels generated
    assert labels.get_label(4) == "D2_NEG"
    assert labels.get_label(5) == "D3_NEG"
    assert labels.get_label(6) == "D2_POS"
    assert labels.get_label(7) == "D3_POS"
```

#### Test 6: All Presets Support N≥4
```python
@pytest.mark.parametrize("preset_name", ["gaming", "6dof", "cardinal", "math"])
def test_all_presets_support_nd(preset_name):
    """All presets should work with N≥4 substrates."""
    labels = get_labels(preset=preset_name, substrate_position_dim=5)

    # All use dimension index notation
    assert labels.get_label(0) == "D0_NEG"
    assert labels.get_label(5) == "D0_POS"
    assert labels.get_action_count() == 12  # 2*5 + 2
```

### Integration Tests: `tests/test_townlet/test_integration.py`

#### Test 7: GridND Environment Initialization
```python
def test_gridnd_environment_initialization_with_action_labels():
    """GridND substrate should initialize with action labels (no ValueError)."""
    config_data = {
        "version": "1.0",
        "description": "7D GridND test config",
        "type": "gridnd",
        "gridnd": {
            "dimension_sizes": [3, 3, 3, 3, 3, 3, 3],
            "boundary": "clamp",
            "distance_metric": "manhattan",
            "observation_encoding": "relative",
            "topology": "hypercube",
        },
    }

    substrate_config = SubstrateConfig(**config_data)
    substrate = SubstrateFactory.build(substrate_config, torch.device("cpu"))

    # Should not raise ValueError
    labels = get_labels(preset="math", substrate_position_dim=substrate.position_dim)

    assert labels.get_action_count() == 16  # 2*7 + 2
    assert labels.get_label(0) == "D0_NEG"
```

#### Test 8: ContinuousND Environment Initialization
```python
def test_continuousnd_environment_initialization_with_action_labels():
    """ContinuousND substrate should initialize with action labels."""
    config_data = {
        "version": "1.0",
        "description": "5D ContinuousND test config",
        "type": "continuousnd",
        "continuousnd": {
            "bounds": [[0.0, 10.0]] * 5,  # 5 dimensions
            "boundary": "clamp",
            "movement_delta": 0.5,
            "interaction_radius": 0.8,
            "distance_metric": "euclidean",
            "observation_encoding": "relative",
        },
    }

    substrate_config = SubstrateConfig(**config_data)
    substrate = SubstrateFactory.build(substrate_config, torch.device("cpu"))

    # Should not raise ValueError
    labels = get_labels(preset="math", substrate_position_dim=substrate.position_dim)

    assert labels.get_action_count() == 12  # 2*5 + 2
```

## Implementation Tasks

Following TDD methodology (RED-GREEN-REFACTOR):

### Task 1: Write Failing Tests
- Create `tests/test_townlet/test_environment/test_action_labels.py`
- Implement Tests 1-6 (unit tests)
- Verify all tests fail with current implementation

### Task 2: Implement N≥4 Support
- Extend `_filter_labels_for_substrate` with N≥4 case
- Run unit tests - should pass

### Task 3: Update Docstrings
- Update `get_labels()` docstring with N≥4 examples
- Update `_filter_labels_for_substrate()` docstring

### Task 4: Integration Testing
- Add Tests 7-8 to `tests/test_townlet/test_integration.py`
- Verify GridND/ContinuousND environments initialize successfully

### Task 5: Documentation
- Update CLAUDE.md with N-dimensional action label examples
- Add section to substrate architecture docs

### Task 6: Manual Verification
- Test 7D GridND config pack (if available)
- Verify action labels display correctly in frontend/logs

## Backward Compatibility

All changes maintain full backward compatibility:
- ✅ 0-3D substrates use existing hardcoded mappings (unchanged)
- ✅ Existing presets work identically for 0-3D
- ✅ Custom labels still supported for all dimensions
- ✅ No breaking API changes

## Success Criteria

- [ ] All unit tests pass (Tests 1-6)
- [ ] All integration tests pass (Tests 7-8)
- [ ] GridND/ContinuousND environments initialize without ValueError
- [ ] Documentation updated with N-dimensional examples
- [ ] Type checking passes (mypy)
- [ ] Full test suite passes (1224+ tests)

## Future Enhancements (Out of Scope)

- Support for diagonal movement in N-dimensions (2^N actions instead of 2N)
- Action label validation (warn if custom labels incomplete)
- Frontend visualization of N-dimensional action spaces
- Pedagogical tooltips explaining dimension index notation

## References

- Original bug report: Codex P1 badge (this conversation)
- Related work: Task-002A Phase 5 (GridND/ContinuousND substrates)
- Action space formula: CLAUDE.md (2N+2 for N-dimensional substrates)
