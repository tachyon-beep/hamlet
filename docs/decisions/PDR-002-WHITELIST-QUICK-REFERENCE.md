# PDR-002 Whitelist: Quick Reference Card

**For Developers: How to Add Parameters Without Violating PDR-002**

---

## Decision Tree: Should I Whitelist This Default?

```
Is this parameter in a config file (YAML)?
├─ YES: Is it UAC/BAC (universe mechanics or agent behavior)?
│  ├─ YES → ❌ NO DEFAULTS ALLOWED (remove default, add validation)
│  └─ NO: Is it infrastructure (device, port, file path)?
│     ├─ YES → ✅ Whitelist OK (with justification)
│     └─ NO: Is it metadata/telemetry?
│        ├─ YES → ✅ Whitelist OK (with justification)
│        └─ NO → ❌ Remove default, add validation
└─ NO: Is it internal/computed?
   ├─ YES → ✅ No whitelist needed (not config-driven)
   └─ NO → Reconsider: should this be configurable?
```

---

## Categories: What's Allowed?

### ❌ NEVER Allowed (UAC/BAC)

**Universe Mechanics** (UNIVERSE_AS_CODE):
- Grid size, affordances, meters, cascades
- Energy costs, depletion rates, death thresholds
- Interaction costs, rewards, operating hours
- Spatial layout, vision range, observability mode

**Agent Behavior** (BRAIN_AS_CODE):
- Network architecture (hidden_dim, layer counts)
- Learning hyperparameters (learning_rate, gamma, batch_size)
- Exploration strategy (epsilon, RND params, intrinsic weights)
- Training dynamics (replay buffer size, update frequency)
- Curriculum parameters (stages, thresholds, windows)

**Test**: If this changes, does agent behavior or universe mechanics change?
→ YES = UAC/BAC = ❌ NO DEFAULTS

### ✅ Sometimes Allowed (Infrastructure)

**Python Runtime**:
- `device="cpu"` - Hardware selection
- `dtype=torch.float32` - Precision
- `seed=None` - Random seed

**System Infrastructure**:
- `port=8080` - Network port
- `host="localhost"` - Network host
- `checkpoint_dir="checkpoints"` - File paths
- `log_dir="logs"` - Logging directory

**Performance Tuning**:
- `num_workers=4` - Parallelization
- `pin_memory=True` - CUDA optimization
- `prefetch_factor=2` - Data loading

**Development/Debug**:
- `logging_level=INFO` - Log verbosity
- `debug_mode=False` - Debug instrumentation
- `profiling_enabled=False` - Performance profiling

**Telemetry/Metadata**:
- `flush_every=10` - TensorBoard flush frequency
- `log_gradients=False` - Gradient logging toggle
- Empty defaults: `dict()`, `[]`, `None` for optional metadata

**Test**: If this changes, does algorithm still produce same results?
→ YES = Infrastructure = ✅ Whitelist OK (must justify)

### ✅ Always Allowed (Computed)

**Derived Values** (not in whitelist, not in config):
- `observation_dim` - Calculated from grid_size + meter_count
- `action_dim` - Derived from actions.yaml
- Internal state: `hidden=None` (LSTM state)
- Performance metrics: `total = sum(components)`

**Test**: Can this be deterministically computed from explicit config?
→ YES = Computed = ✅ No config needed, no whitelist needed

---

## Examples: Good vs Bad

### Example 1: Network Architecture (BAC)

**❌ BAD (violates PDR-002)**:
```python
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        # ❌ hidden_dim is BAC (network architecture)
```

**✅ GOOD (compliant)**:
```python
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        # ✅ Required param, must come from config
        if hidden_dim < 1:
            raise ValueError("hidden_dim required. Add to config: hidden_dim: 128")
```

**Config** (training.yaml):
```yaml
network:
  hidden_dim: 128  # ✅ Explicit in config
```

---

### Example 2: Environment (UAC)

**❌ BAD (violates PDR-002)**:
```python
def __init__(self, grid_size: int = 8, move_energy_cost: float = 0.005):
    # ❌ Both are UAC parameters (universe mechanics)
```

**✅ GOOD (compliant)**:
```python
def __init__(self, grid_size: int, move_energy_cost: float):
    # ✅ Required params, must come from config
    if grid_size < 1:
        raise ValueError("grid_size required. Add to config: grid_size: 8")
    if not 0 <= move_energy_cost <= 1:
        raise ValueError("move_energy_cost must be 0-1. Add to config: move_energy_cost: 0.005")
```

**Config** (training.yaml):
```yaml
environment:
  grid_size: 8  # ✅ Explicit
  move_energy_cost: 0.005  # ✅ Explicit
```

---

### Example 3: Infrastructure (OK to whitelist)

**✅ ACCEPTABLE (infrastructure)**:
```python
def __init__(
    self,
    checkpoint_dir: Path,
    device: str = "cpu",  # ✅ Hardware selection (infrastructure)
    port: int = 8080,  # ✅ Network port (infrastructure)
):
    self.checkpoint_dir = checkpoint_dir
    self.device = torch.device(device)
    self.port = port
```

**Whitelist entry**:
```
src/path/file.py:__init__:device  # Infrastructure: hardware selection
src/path/file.py:__init__:port  # Infrastructure: network port
```

**Why OK**: Changing device/port doesn't affect algorithm behavior (WHERE/HOW it runs, not WHAT it does).

---

### Example 4: Telemetry (OK to whitelist)

**✅ ACCEPTABLE (telemetry)**:
```python
class TensorBoardLogger:
    def __init__(
        self,
        log_dir: Path,
        flush_every: int = 10,  # ✅ Telemetry (performance tuning)
        log_gradients: bool = False,  # ✅ Debug option
    ):
        self.log_dir = log_dir
        self.flush_every = flush_every
        self.log_gradients = log_gradients
```

**Whitelist entry**:
```
src/townlet/training/tensorboard_logger.py:*  # Telemetry system
```

**Why OK**: Logging parameters don't affect algorithm behavior (debug/telemetry infrastructure).

---

### Example 5: Computed Values (No config needed)

**✅ ACCEPTABLE (computed)**:
```python
def __init__(self, grid_size: int, meter_count: int):
    # ✅ grid_size and meter_count are explicit (required)
    self.grid_size = grid_size
    self.meter_count = meter_count

    # ✅ observation_dim is computed (derived from explicit params)
    self.observation_dim = grid_size * grid_size + meter_count + 15 + 4
```

**No config needed**: observation_dim is deterministic function of explicit params.

---

## How to Add New Configurable Parameter

### Step 1: Determine Category

Is this UAC/BAC or infrastructure?
- UAC/BAC: No defaults allowed
- Infrastructure: Whitelist with justification

### Step 2: Add to Config (if UAC/BAC)

**Add to training.yaml** (or appropriate config file):
```yaml
new_section:
  new_param: 42  # ✅ Explicit value
```

### Step 3: Add Validation (No Defaults)

**In Python code**:
```python
def __init__(self, config: dict):
    if "new_param" not in config:
        raise ValueError(
            "Missing required parameter 'new_param'. "
            "Add to your config:\n"
            "new_section:\n"
            "  new_param: 42"
        )
    self.new_param = config["new_param"]
```

### Step 4: Update All Example Configs

```bash
# Add to ALL config packs
for config in configs/*/training.yaml; do
    echo "  new_param: 42  # Description" >> $config
done
```

### Step 5: Run Linter

```bash
python scripts/no_defaults_lint.py src/townlet/ --whitelist .defaults-whitelist.txt
# Expected: 0 violations (or add to whitelist if infrastructure)
```

---

## How to Whitelist Infrastructure Parameter

### Step 1: Verify It's Infrastructure

**Test**: If I change this value, does algorithm behavior change?
- YES → ❌ Don't whitelist, remove default
- NO → ✅ Proceed to whitelist

### Step 2: Add to Whitelist with Justification

**Format** (function-specific):
```
src/path/file.py:ClassName:function_name:param_name  # Justification
```

**Example**:
```
src/townlet/demo/runner.py:DemoRunner:__init__:device  # Hardware selection (infrastructure)
```

**Format** (file-level):
```
src/path/file.py:*  # Justification
```

**Example**:
```
src/townlet/training/tensorboard_logger.py:*  # Telemetry system (infrastructure)
```

### Step 3: Verify Linter Accepts It

```bash
python scripts/no_defaults_lint.py src/townlet/ --whitelist .defaults-whitelist.txt
# Expected: 0 violations (whitelist entry accepted)
```

---

## Common Mistakes

### Mistake 1: "This param is usually X, so default is convenient"

**Wrong thinking**: "Most users want epsilon_start=1.0, so default makes it convenient."

**Correct thinking**: "epsilon_start is BAC (exploration strategy). Must be explicit. Provide config template instead."

**Solution**: Remove default, add to config template.

---

### Mistake 2: "This is just metadata, doesn't affect behavior"

**Wrong thinking**: "display_name is just UI metadata, doesn't need to be explicit."

**Correct thinking**: "display_name is part of affordances.yaml (UAC). Even metadata must be explicit."

**Solution**: Add to YAML schema, require field.

**Exception**: Truly optional telemetry (TensorBoard logs, recording) can have metadata defaults.

---

### Mistake 3: "Computed values should be in config"

**Wrong thinking**: "observation_dim should be in config for transparency."

**Correct thinking**: "observation_dim is deterministic from grid_size + meter_count. Don't duplicate in config."

**Solution**: Compute in code, validate against expected value if needed.

---

### Mistake 4: "Infrastructure params should be explicit too"

**Wrong thinking**: "For consistency, device='cpu' should be required in config."

**Correct thinking**: "device affects WHERE code runs, not WHAT it does. Default is acceptable."

**Solution**: Keep device='cpu' default, whitelist with justification.

---

## Cheat Sheet: Quick Tests

| Question | UAC/BAC | Infrastructure | Computed |
|----------|---------|----------------|----------|
| Affects agent behavior? | YES ✅ | NO ❌ | NO ❌ |
| Affects universe mechanics? | YES ✅ | NO ❌ | NO ❌ |
| Affects WHERE code runs? | NO ❌ | YES ✅ | NO ❌ |
| Can be derived from config? | NO ❌ | NO ❌ | YES ✅ |
| **Default allowed?** | ❌ NO | ✅ YES (whitelist) | ✅ YES (no config) |
| **Must be in config?** | ✅ YES | ⚠️ Optional | ❌ NO |

---

## When in Doubt

**Default to NO DEFAULTS**. If unclear whether a param is UAC/BAC or infrastructure:
1. Make it required (no default)
2. Add to config
3. Add validation with clear error message
4. Ask in architecture review if infrastructure exemption is warranted

**Principle**: Err on side of explicitness. Can always add whitelist later if justified.

---

## Resources

- **Full Policy**: `docs/decisions/PDR-002-NO-DEFAULTS-PRINCIPLE.md`
- **Whitelist Review**: `docs/decisions/PDR-002-WHITELIST-REVIEW.md`
- **Comparison**: `docs/decisions/PDR-002-WHITELIST-COMPARISON.md`
- **Linter**: `scripts/no_defaults_lint.py`
- **Whitelist**: `.defaults-whitelist.txt`

---

**Questions?** Ask in #architecture channel or tag @architecture-team
