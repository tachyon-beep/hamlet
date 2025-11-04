# QUICK-003 Phase 3 Execution Plan: Networks Layer (BAC Defaults)

**Status**: Planning → Execution
**Created**: 2025-11-05
**Scope**: Remove BAC defaults from agent networks (2 files, 4 violations)

## Overview

Phase 3 removes BAC (BRAIN_AS_CODE) parameter defaults from network architecture definitions. These defaults hide critical architectural decisions that should be explicit.

**Key Principle**: Network architecture parameters (hidden_dim, action_dim, num_meters, etc.) MUST be explicitly provided by callers. No hidden defaults allowed.

**Pragmatic Scope**: Remove defaults, make parameters required, pass explicit values. Full BRAIN_AS_CODE config system is future work.

## Violation Analysis

### Summary

- **Total Violations**: 4 in src/townlet/agent/networks.py
- **Linter Output**: `python scripts/no_defaults_lint.py src/townlet/agent/networks.py`

### Breakdown

**networks.py:14 - SimpleQNetwork.__init__**:
```python
def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):  # ❌ BAC default
```
**Action**: Remove `hidden_dim=128` default

**networks.py:57-64 - RecurrentSpatialQNetwork.__init__**:
```python
def __init__(
    self,
    action_dim: int = 5,  # ❌ BAC default
    window_size: int = 5,  # ❌ BAC default
    num_meters: int = 8,  # ❌ BAC default
    num_affordance_types: int = 15,  # ❌ BAC default
    enable_temporal_features: bool = False,  # ❌ BAC default
    hidden_dim: int = 256,  # ❌ BAC default
):
```
**Action**: Remove all 6 defaults

**networks.py:136 - RecurrentSpatialQNetwork.forward**:
```python
def forward(self, obs: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None):  # ✅ OK
```
**Action**: **KEEP** - `hidden=None` is an optional feature parameter (LSTM state)

**networks.py:218 - RecurrentSpatialQNetwork.reset_hidden_state**:
```python
def reset_hidden_state(self, batch_size: int = 1, device: torch.device | None = None):
```
**Action**:
- Remove `batch_size=1` (BAC default) ❌
- **KEEP** `device=None` (infrastructure fallback) ✅

## Implementation Strategy

### Step 1: Remove BAC Defaults from Networks

**File**: `src/townlet/agent/networks.py`

**Change A: SimpleQNetwork (line 14)**
```python
# ❌ BEFORE
def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):

# ✅ AFTER
def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
    """
    Initialize simple MLP Q-network.

    Args:
        obs_dim: Observation dimension
        action_dim: Number of actions
        hidden_dim: Hidden layer dimension (typically 128-256)

    Note (PDR-002):
        All network architecture parameters must be explicitly specified.
        No BAC (BRAIN_AS_CODE) defaults allowed.
    """
```

**Change B: RecurrentSpatialQNetwork (lines 57-64)**
```python
# ❌ BEFORE
def __init__(
    self,
    action_dim: int = 5,
    window_size: int = 5,
    num_meters: int = 8,
    num_affordance_types: int = 15,
    enable_temporal_features: bool = False,
    hidden_dim: int = 256,
):

# ✅ AFTER
def __init__(
    self,
    action_dim: int,
    window_size: int,
    num_meters: int,
    num_affordance_types: int,
    enable_temporal_features: bool,
    hidden_dim: int,
):
    """
    Initialize recurrent spatial Q-network.

    Args:
        action_dim: Number of actions
        window_size: Size of local vision window (5 for 5×5)
        num_meters: Number of meter values (8)
        num_affordance_types: Number of affordance types (15)
        enable_temporal_features: Whether to expect temporal features
        hidden_dim: LSTM hidden dimension (typically 256)

    Note (PDR-002):
        All network architecture parameters must be explicitly specified.
        No BAC (BRAIN_AS_CODE) defaults allowed.

    Future (BRAIN_AS_CODE):
        These parameters should come from network config YAML.
    """
```

**Change C: reset_hidden_state (line 218)**
```python
# ❌ BEFORE
def reset_hidden_state(self, batch_size: int = 1, device: torch.device | None = None):

# ✅ AFTER
def reset_hidden_state(self, batch_size: int, device: torch.device | None = None):
    """
    Reset LSTM hidden state (call at episode start).

    Args:
        batch_size: Batch size for hidden state
        device: Device for tensors (default: cpu). Infrastructure default - PDR-002 exemption.
    """
```

### Step 2: Update Network Instantiations

**File**: `src/townlet/population/vectorized.py`

Current instantiations (lines 111-132):
```python
# Recurrent network - currently passes 5 params, relies on hidden_dim=256 default
self.q_network = RecurrentSpatialQNetwork(
    action_dim=action_dim,
    window_size=vision_window_size,
    num_meters=env.meter_count,
    num_affordance_types=env.num_affordance_types,
    enable_temporal_features=env.enable_temporal_mechanics,
).to(device)

# Simple network - currently passes 2 params, relies on hidden_dim=128 default
self.q_network = SimpleQNetwork(obs_dim, action_dim).to(device)
```

**Change: Add explicit hidden_dim**
```python
# Recurrent network - add hidden_dim=256
self.q_network = RecurrentSpatialQNetwork(
    action_dim=action_dim,
    window_size=vision_window_size,
    num_meters=env.meter_count,
    num_affordance_types=env.num_affordance_types,
    enable_temporal_features=env.enable_temporal_mechanics,
    hidden_dim=256,  # TODO(BRAIN_AS_CODE): Should come from config
).to(device)

# Simple network - add hidden_dim=128
self.q_network = SimpleQNetwork(obs_dim, action_dim, hidden_dim=128).to(device)  # TODO(BRAIN_AS_CODE): Should come from config

# Same for target networks (lines 124-132)
```

**Locations to Update**:
- Line 111-117: Q-network (recurrent)
- Line 119: Q-network (simple)
- Line 124-130: Target network (recurrent)
- Line 132: Target network (simple)

### Step 3: Update reset_hidden_state Calls

**Search Pattern**: `reset_hidden_state(`

**Expected locations**:
- `src/townlet/population/vectorized.py` - Where LSTM state is reset

**Change**: Add explicit `batch_size` parameter

### Step 4: Testing

**Unit Tests**:
```bash
# Test network instantiation
uv run pytest tests/test_townlet/unit/population/ -k "network" -xvs

# Test recurrent network tests
uv run pytest tests/test_townlet/integration/test_recurrent_networks.py -xvs
```

**Integration Test**:
```bash
# Verify configs still load
uv run python << 'EOF'
from pathlib import Path
from townlet.demo.runner import DemoRunner
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    runner = DemoRunner(
        config_dir=Path("configs/L2_partial_observability"),
        db_path=Path(tmpdir) / "test.db",
        checkpoint_dir=Path(tmpdir) / "checkpoints",
        max_episodes=None
    )
    print("✅ L2 (recurrent) config loads successfully")
EOF
```

## Execution Checklist

### Phase 3.1: Remove Defaults from Networks

- [ ] Read networks.py to understand all defaults
- [ ] Remove `hidden_dim=128` from SimpleQNetwork.__init__
- [ ] Remove 6 defaults from RecurrentSpatialQNetwork.__init__
- [ ] Remove `batch_size=1` from reset_hidden_state
- [ ] Keep `hidden=None` in forward() (optional parameter)
- [ ] Keep `device=None` in reset_hidden_state (infrastructure)
- [ ] Update docstrings with PDR-002 notes

### Phase 3.2: Update Network Instantiations

- [ ] Find all RecurrentSpatialQNetwork() calls
- [ ] Add `hidden_dim=256` to all recurrent network instantiations
- [ ] Find all SimpleQNetwork() calls
- [ ] Add `hidden_dim=128` to all simple network instantiations
- [ ] Add TODO comments for BRAIN_AS_CODE

### Phase 3.3: Update reset_hidden_state Calls

- [ ] Find all reset_hidden_state() calls
- [ ] Add explicit batch_size parameter
- [ ] Verify correct batch size is passed

### Phase 3.4: Verification

- [ ] Run linter on networks.py: expect 2 violations (acceptable defaults)
- [ ] Run network unit tests
- [ ] Run recurrent network integration tests
- [ ] Verify L2 config loads (uses recurrent networks)
- [ ] Verify L1 config loads (uses simple networks)

## Success Criteria

1. **2 BAC defaults removed** from network __init__ methods (7 total, 2 violations remain as acceptable)
2. **All network instantiations explicit** - no reliance on defaults
3. **All tests pass** - unit + integration
4. **All production configs load** - L1 (simple), L2 (recurrent)
5. **Linter passes** - only acceptable defaults remain (hidden=None, device=None)

## Future Work (BRAIN_AS_CODE)

Phase 3 achieves PDR-002 compliance by removing BAC defaults. Full BRAIN_AS_CODE implementation is future work:

1. Add `network:` section to training.yaml configs
2. Define network architecture in config:
   ```yaml
   network:
     type: recurrent  # or 'simple'
     hidden_dim: 256
     # Recurrent-specific
     lstm_layers: 1
     # Future: activation, dropout, layer_norm, etc.
   ```
3. Update runner.py to pass network config to population
4. Update vectorized.py to read network params from config
5. Remove hardcoded values (256, 128)

**Estimated**: TASK-006 (2-3 days, BRAIN_AS_CODE implementation)
