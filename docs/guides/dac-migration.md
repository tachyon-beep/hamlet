# Drive As Code (DAC) Migration Guide

---
## AI-Friendly Frontmatter

**Purpose**: Guide for understanding the Drive As Code (DAC) migration from legacy reward_strategy

**When to Read**: Working with old configs, understanding reward system changes, or learning DAC architecture

**AI-Friendly Summary**:
This guide documents the migration from legacy `reward_strategy` field in training.yaml to the new Drive As Code (DAC) system with dedicated `drive_as_code.yaml` files. The migration occurred during TASK-004C and is a BREAKING CHANGE - the old reward_strategy field has been completely removed. All reward logic now lives in declarative DAC configs. The migration enables A/B testing reward structures without code changes, adds checkpoint provenance via drive_hash, and fixes architectural issues like "Low Energy Delirium" bug.

**Reading Strategy**:
- **Quick Reference**: Jump to "Before and After" section for config examples
- **Understanding Changes**: Read "What Changed" and "Breaking Changes"
- **Implementation**: Read "Migration Steps" for converting old configs
- **Troubleshooting**: See "Common Issues" section

**Related Documents**:
- `docs/config-schemas/drive_as_code.md` - Complete DAC configuration reference
- `docs/plans/2025-11-12-drive-as-code-implementation.md` - Implementation plan
- `docs/config-schemas/training.md` - Training configuration

---

## Overview

Drive As Code (DAC) is a declarative reward function system that replaced the hardcoded Python reward strategies in HAMLET. This migration occurred during TASK-004C and is a **BREAKING CHANGE** with **ZERO BACKWARDS COMPATIBILITY** (pre-release status, zero users).

### Why the Migration?

**Old System Problems**:
1. Reward logic hardcoded in Python (`RewardStrategy`, `AdaptiveRewardStrategy` classes)
2. No A/B testing without code changes
3. No checkpoint provenance for reward configurations
4. Architectural issues ("Low Energy Delirium" bug unfixable without config changes)
5. Tight coupling between training config and reward computation

**New System Benefits**:
1. Declarative YAML configs for reward functions
2. A/B test reward structures by editing YAML only
3. Checkpoint provenance via `drive_hash` (SHA256 of DAC config)
4. Composable: Mix extrinsic strategies, intrinsic drives, modifiers, shaping bonuses
5. GPU-optimized: All operations vectorized across agents
6. Pedagogical value: Expose "interesting failures" as teaching moments

---

## What Changed

### Removed from Code

**Deleted Files**:
- `src/townlet/environment/reward_strategy.py` (235 lines)
  - `RewardStrategy` base class
  - `MultiplicativeRewardStrategy`
  - `AdaptiveRewardStrategy`
  - `WeightedSumRewardStrategy`

**Deleted from training.yaml**:
- `reward_strategy` field (string)

### Added to Code

**New Files**:
- `src/townlet/config/drive_as_code.py` - Pydantic DTOs (~800 lines)
- `src/townlet/environment/dac_engine.py` - Runtime execution (~800 lines)
- `configs/<level>/drive_as_code.yaml` - Per-level DAC configs

**Modified Files**:
- `src/townlet/universe/compiled.py` - Added `dac_config`, `drive_hash` fields
- `src/townlet/universe/compiler.py` - Added DAC validation (Stage 3)
- `src/townlet/config/training.py` - Removed `reward_strategy` field
- `src/townlet/population/vectorized.py` - Integrated DACEngine

**New Compiler Validation**:
- DAC reference validation (bars, variables, affordances)
- drive_hash computation for checkpoint provenance
- Required file: `drive_as_code.yaml` must exist

---

## Breaking Changes

### 1. training.yaml: reward_strategy Removed

**Old** (training.yaml):
```yaml
training:
  device: cuda
  max_episodes: 5000
  reward_strategy: multiplicative  # DELETED
  # ...
```

**New** (training.yaml):
```yaml
training:
  device: cuda
  max_episodes: 5000
  # reward_strategy field is GONE
  # ...
```

**Migration**: Remove `reward_strategy` field from all training.yaml files.

---

### 2. drive_as_code.yaml Now Required

**Error if Missing**:
```
CompilationError: DAC configuration required but drive_as_code.yaml not found in configs/L0_0_minimal.
See docs/config-schemas/drive_as_code.md for creating DAC configs.
```

**Migration**: Create `drive_as_code.yaml` in every config pack.

---

### 3. Checkpoints Include drive_hash

**Old Checkpoint Metadata**:
```python
{
    "episode": 1000,
    "training_config": {...},
    "network_state_dict": {...},
}
```

**New Checkpoint Metadata**:
```python
{
    "episode": 1000,
    "training_config": {...},
    "network_state_dict": {...},
    "drive_hash": "a3f8b2c1d4e5f6...",  # SHA256 of DAC config
}
```

**Impact**: Checkpoints now traceable to exact reward configuration.

---

### 4. Python API Changes

**Old API** (Deleted):
```python
from townlet.environment.reward_strategy import RewardStrategy, MultiplicativeRewardStrategy

strategy = MultiplicativeRewardStrategy(bars=["energy", "health"])
rewards = strategy.calculate_rewards(meters, dones)
```

**New API**:
```python
from townlet.environment.dac_engine import DACEngine
from townlet.config.drive_as_code import load_drive_as_code_config

dac_config = load_drive_as_code_config(config_dir)
engine = DACEngine(
    dac_config=dac_config,
    vfs_registry=vfs_registry,
    device=device,
    num_agents=num_agents,
    bar_index_map=bar_index_map,
)
total_rewards, intrinsic_weights, components = engine.calculate_rewards(
    step_counts=step_counts,
    dones=dones,
    meters=meters,
    intrinsic_raw=intrinsic_raw,
)
```

**Migration**: If you have custom Python code using reward strategies, rewrite to use DACEngine.

---

## Before and After

### Example 1: L0_0_minimal (Multiplicative)

#### Before

**training.yaml**:
```yaml
training:
  device: cuda
  max_episodes: 500
  reward_strategy: multiplicative
  # ...
```

**Python code** (implicit):
```python
# Hardcoded in RewardStrategy class
reward = base * energy
```

#### After

**training.yaml**:
```yaml
training:
  device: cuda
  max_episodes: 500
  # reward_strategy field DELETED
  # ...
```

**drive_as_code.yaml** (NEW):
```yaml
drive_as_code:
  version: "1.0"

  modifiers: {}

  extrinsic:
    type: multiplicative
    base: 1.0
    bars: [energy]

  intrinsic:
    strategy: rnd
    base_weight: 0.1
    apply_modifiers: []

  shaping: []

  composition:
    normalize: false
    clip: null
    log_components: true
    log_modifiers: true
```

---

### Example 2: L0_5_dual_resource (Fixed Bug)

#### Before

**training.yaml**:
```yaml
training:
  reward_strategy: multiplicative  # Bug: "Low Energy Delirium"
  # ...
```

**Problem**: Multiplicative reward (energy × health) + high intrinsic weight → agents exploit low bars for exploration.

#### After

**drive_as_code.yaml**:
```yaml
drive_as_code:
  version: "1.0"

  modifiers: {}

  extrinsic:
    type: constant_base_with_shaped_bonus  # FIX: Constant base prevents exploit
    base_reward: 1.0
    bar_bonuses:
      - bar: energy
        center: 0.5
        scale: 0.5
      - bar: health
        center: 0.5
        scale: 0.5
      - bar: satiation
        center: 0.5
        scale: 0.5
      - bar: hygiene
        center: 0.5
        scale: 0.5
    variable_bonuses: []

  intrinsic:
    strategy: rnd
    base_weight: 0.1
    apply_modifiers: []

  shaping: []

  composition:
    normalize: false
    clip: null
    log_components: true
    log_modifiers: true
```

**Pedagogical Note**: L0_5 demonstrates the fix for "Low Energy Delirium". Compare against L0_0 to show importance of reward structure design.

---

### Example 3: Crisis Suppression (New Feature)

#### Before

**Not Possible**: Crisis suppression required code changes.

#### After

**drive_as_code.yaml** (NEW CAPABILITY):
```yaml
drive_as_code:
  version: "1.0"

  modifiers:
    energy_crisis:
      bar: energy
      ranges:
        - name: crisis
          min: 0.0
          max: 0.2
          multiplier: 0.0  # Suppress intrinsic in crisis
        - name: normal
          min: 0.2
          max: 1.0
          multiplier: 1.0

  extrinsic:
    type: multiplicative
    base: 1.0
    bars: [energy, health]

  intrinsic:
    strategy: rnd
    base_weight: 0.5
    apply_modifiers: [energy_crisis]  # Apply crisis suppression

  shaping: []

  composition:
    normalize: false
    clip: null
    log_components: true
    log_modifiers: true
```

**Impact**: Intrinsic weight automatically suppressed when energy < 0.2, preventing "Low Energy Delirium".

---

## Migration Steps

### Step 1: Remove reward_strategy from training.yaml

**Find**:
```yaml
reward_strategy: multiplicative
```

**Delete**: Remove the entire line.

**Verify**: No `reward_strategy` field remains in training.yaml.

---

### Step 2: Create drive_as_code.yaml

**Template** (minimal):
```yaml
drive_as_code:
  version: "1.0"

  modifiers: {}

  extrinsic:
    type: multiplicative
    base: 1.0
    bars: [energy]  # Update with your bars

  intrinsic:
    strategy: rnd
    base_weight: 0.1  # Match training.yaml: initial_intrinsic_weight
    apply_modifiers: []

  shaping: []

  composition:
    normalize: false
    clip: null
    log_components: true
    log_modifiers: true
```

**Place**: `configs/<config_pack>/drive_as_code.yaml`

---

### Step 3: Map Old Strategy to New

**Old → New Mapping**:

| Old reward_strategy | New extrinsic.type | Notes |
|---------------------|-------------------|-------|
| `multiplicative` | `multiplicative` | Base + bars required |
| `adaptive` | `multiplicative` + modifiers | Add crisis suppression |
| `weighted_sum` | `weighted_sum` | Use bar_bonuses for weights |
| Custom | `vfs_variable` | Migrate logic to VFS |

**Example Mapping**:

**Old**:
```yaml
reward_strategy: multiplicative
```

**New**:
```yaml
extrinsic:
  type: multiplicative
  base: 1.0
  bars: [energy, health]
```

---

### Step 4: Migrate initial_intrinsic_weight

**Old** (training.yaml):
```yaml
initial_intrinsic_weight: 0.1
```

**New** (drive_as_code.yaml):
```yaml
intrinsic:
  strategy: rnd
  base_weight: 0.1  # Matches old initial_intrinsic_weight
```

**Note**: `initial_intrinsic_weight` field still exists in training.yaml for backward compatibility with checkpoints, but DAC now controls the actual intrinsic weight.

---

### Step 5: Test Compilation

**Command**:
```bash
python -m townlet.compiler compile configs/<config_pack>
```

**Expected Output**:
```
✓ Stage 1: Parse configs
✓ Stage 2: Build symbol table
✓ Stage 3: Resolve references (including DAC validation)
✓ Stage 4: Cross-validation
✓ Stage 5: Metadata generation
✓ Stage 6: Optimization
✓ Stage 7: Emit compiled universe

Compilation successful!
drive_hash: a3f8b2c1d4e5f6...
```

**If Errors**: See "Common Issues" section below.

---

## Common DAC Patterns

### Pattern 1: Simple Multiplicative

**Use Case**: Single or multiple resource survival

```yaml
extrinsic:
  type: multiplicative
  base: 1.0
  bars: [energy, health]
```

**Formula**: `reward = 1.0 × energy × health`

---

### Pattern 2: Constant Base (Bug Fix)

**Use Case**: Fix "Low Energy Delirium" bug

```yaml
extrinsic:
  type: constant_base_with_shaped_bonus
  base_reward: 1.0
  bar_bonuses:
    - bar: energy
      center: 0.5
      scale: 0.5
    - bar: health
      center: 0.5
      scale: 0.5
```

**Formula**: `reward = 1.0 + 0.5×(energy-0.5) + 0.5×(health-0.5)`

---

### Pattern 3: Crisis Suppression

**Use Case**: Suppress intrinsic when resources low

```yaml
modifiers:
  energy_crisis:
    bar: energy
    ranges:
      - {name: crisis, min: 0.0, max: 0.2, multiplier: 0.0}
      - {name: normal, min: 0.2, max: 1.0, multiplier: 1.0}

intrinsic:
  apply_modifiers: [energy_crisis]
```

**Effect**: Intrinsic weight = 0 when energy < 0.2

---

### Pattern 4: No Intrinsic (Ablation)

**Use Case**: Pure extrinsic reward for comparison

```yaml
intrinsic:
  strategy: none
  base_weight: 0.0
  apply_modifiers: []
```

---

### Pattern 5: Shaping Bonuses

**Use Case**: Guide agents without changing base reward

```yaml
shaping:
  - type: approach_reward
    weight: 0.5
    target_affordance: Bed
    max_distance: 10.0
  - type: completion_bonus
    weight: 1.0
    affordance: Job
```

---

## Common Issues

### Issue 1: CompilationError: "DAC configuration required"

**Error**:
```
CompilationError: DAC configuration required but drive_as_code.yaml not found in configs/L0_0_minimal.
```

**Cause**: Missing `drive_as_code.yaml` file.

**Fix**: Create `drive_as_code.yaml` in config pack directory.

---

### Issue 2: CompilationError: "undefined bar"

**Error**:
```
CompilationError: Extrinsic strategy references undefined bar: health
```

**Cause**: Bar referenced in DAC but not defined in bars.yaml.

**Fix**: Add bar to bars.yaml or fix typo in drive_as_code.yaml.

```yaml
# bars.yaml
bars:
  - id: health  # Must match reference
    # ...
```

---

### Issue 3: ValidationError: "Ranges must start at 0.0"

**Error**:
```
ValidationError: Ranges must start at 0.0, got 0.1
```

**Cause**: Modifier ranges don't start at 0.0.

**Fix**: Ensure first range starts at 0.0 and last range ends at 1.0.

```yaml
# WRONG
ranges:
  - {name: crisis, min: 0.1, max: 1.0, multiplier: 0.0}  # Doesn't start at 0.0!

# CORRECT
ranges:
  - {name: crisis, min: 0.0, max: 0.2, multiplier: 0.0}
  - {name: normal, min: 0.2, max: 1.0, multiplier: 1.0}
```

---

### Issue 4: ValidationError: "Gap or overlap between ranges"

**Error**:
```
ValidationError: Gap or overlap between ranges: crisis (max=0.3) and normal (min=0.4)
```

**Cause**: Ranges have gaps or overlaps.

**Fix**: Ensure ranges are contiguous (max of one = min of next).

```yaml
# WRONG
ranges:
  - {name: crisis, min: 0.0, max: 0.3, multiplier: 0.0}
  - {name: normal, min: 0.4, max: 1.0, multiplier: 1.0}  # Gap: 0.3-0.4!

# CORRECT
ranges:
  - {name: crisis, min: 0.0, max: 0.3, multiplier: 0.0}
  - {name: normal, min: 0.3, max: 1.0, multiplier: 1.0}  # Contiguous
```

---

### Issue 5: "Low Energy Delirium" Still Occurring

**Symptom**: Agents learn to stay at low energy for exploration.

**Cause**: Using multiplicative strategy without crisis suppression.

**Fix Option 1**: Add crisis suppression modifier
```yaml
modifiers:
  energy_crisis:
    bar: energy
    ranges:
      - {name: crisis, min: 0.0, max: 0.2, multiplier: 0.0}
      - {name: normal, min: 0.2, max: 1.0, multiplier: 1.0}

intrinsic:
  apply_modifiers: [energy_crisis]
```

**Fix Option 2**: Switch to constant_base_with_shaped_bonus
```yaml
extrinsic:
  type: constant_base_with_shaped_bonus
  base_reward: 1.0
  bar_bonuses:
    - {bar: energy, center: 0.5, scale: 0.5}
    - {bar: health, center: 0.5, scale: 0.5}
```

---

## A/B Testing Workflow

DAC enables A/B testing reward structures without code changes.

### Example: Test Crisis Suppression Impact

**Baseline** (run_baseline.sh):
```bash
#!/bin/bash
# Baseline: No crisis suppression
cd configs/L1_full_observability

# Edit drive_as_code.yaml
cat > drive_as_code.yaml <<EOF
drive_as_code:
  version: "1.0"
  modifiers: {}
  extrinsic:
    type: multiplicative
    base: 1.0
    bars: [energy, health]
  intrinsic:
    strategy: rnd
    base_weight: 0.5
    apply_modifiers: []  # No modifiers
  shaping: []
  composition: {normalize: false, clip: null, log_components: true, log_modifiers: true}
EOF

cd ../..
uv run scripts/run_demo.py --config configs/L1_full_observability
```

**Treatment** (run_treatment.sh):
```bash
#!/bin/bash
# Treatment: With crisis suppression
cd configs/L1_full_observability

# Edit drive_as_code.yaml
cat > drive_as_code.yaml <<EOF
drive_as_code:
  version: "1.0"
  modifiers:
    energy_crisis:
      bar: energy
      ranges:
        - {name: crisis, min: 0.0, max: 0.2, multiplier: 0.0}
        - {name: normal, min: 0.2, max: 1.0, multiplier: 1.0}
  extrinsic:
    type: multiplicative
    base: 1.0
    bars: [energy, health]
  intrinsic:
    strategy: rnd
    base_weight: 0.5
    apply_modifiers: [energy_crisis]  # Apply suppression
  shaping: []
  composition: {normalize: false, clip: null, log_components: true, log_modifiers: true}
EOF

cd ../..
uv run scripts/run_demo.py --config configs/L1_full_observability
```

**Compare**:
```bash
tensorboard --logdir logs/
# Compare baseline vs treatment:
# - mean_episode_survival
# - intrinsic_weight (should be lower in crisis for treatment)
# - extrinsic_reward vs intrinsic_reward breakdown
```

**Provenance**: Each run has different drive_hash in checkpoint metadata, enabling precise identification.

---

## Checkpoint Compatibility

### Loading Old Checkpoints (Pre-DAC)

**Behavior**: Old checkpoints (without drive_hash) can still be loaded, but will warn about missing provenance.

**Warning**:
```
Warning: Checkpoint missing drive_hash (pre-DAC). Cannot verify reward configuration provenance.
```

**Impact**: Training can continue, but reward configuration may have changed since checkpoint was saved.

---

### Loading New Checkpoints (Post-DAC)

**Validation**: DACEngine validates drive_hash matches current config.

**Error if Mismatch**:
```
ValueError: Checkpoint drive_hash mismatch!
Expected: a3f8b2c1d4e5f6...
Got:      b4e5f6c1d2a3f8...

This checkpoint was trained with different reward configuration.
```

**Fix**: Either:
1. Use matching drive_as_code.yaml (restore old config)
2. Start training from scratch with new config
3. Use `--ignore-drive-hash` flag (not recommended - breaks reproducibility)

---

## Development Workflow

### Iterating on Reward Structures

**Step 1**: Edit `drive_as_code.yaml`
```bash
vim configs/L1_full_observability/drive_as_code.yaml
```

**Step 2**: Validate compilation
```bash
python -m townlet.compiler compile configs/L1_full_observability
```

**Step 3**: Run training
```bash
uv run scripts/run_demo.py --config configs/L1_full_observability
```

**Step 4**: Monitor in TensorBoard
```bash
tensorboard --logdir logs/
# Watch: extrinsic_reward, intrinsic_reward, shaping_reward, modifier values
```

**Step 5**: Compare drive_hash
```bash
# Check checkpoint metadata
python -c "
import torch
ckpt = torch.load('checkpoints/L1_full_observability/episode_1000.pt')
print('drive_hash:', ckpt['metadata']['drive_hash'])
"
```

---

## Best Practices

### 1. Always Test Compilation After Changes

```bash
python -m townlet.compiler compile configs/<config_pack>
```

Don't wait until training to discover config errors.

---

### 2. Use Comments to Document Intent

```yaml
# Pedagogical Goal: Demonstrate "Low Energy Delirium" bug
# Teaching Moment: Compare against L0_5 to show importance of reward structure
drive_as_code:
  version: "1.0"
  # ... config ...
```

---

### 3. Log Components for Debugging

```yaml
composition:
  log_components: true  # Separate extrinsic/intrinsic/shaping in TensorBoard
  log_modifiers: true   # Log modifier multipliers
```

---

### 4. Start Simple, Add Complexity Gradually

**Progression**:
1. Basic multiplicative (L0_0)
2. Add crisis suppression (L0_5)
3. Add shaping bonuses (L1+)
4. Add multiple modifiers (advanced)

---

### 5. Track drive_hash for Experiments

**Experiment Log**:
```markdown
# Experiment 2024-11-12-A

**Config**: L1_full_observability
**drive_hash**: a3f8b2c1d4e5f6...
**Hypothesis**: Crisis suppression improves stability
**Result**: Mean survival +30%, intrinsic weight correctly suppressed
```

---

## Related Documentation

- **DAC Config Reference**: `docs/config-schemas/drive_as_code.md`
- **Implementation Plan**: `docs/plans/2025-11-12-drive-as-code-implementation.md`
- **Training Config**: `docs/config-schemas/training.md`
- **VFS Config**: `docs/config-schemas/variables.md`
- **Compiler Docs**: `docs/UNIVERSE-COMPILER.md`

---

## FAQ

### Q: Can I use old checkpoints with new DAC configs?

**A**: Yes, but drive_hash validation will fail if configs don't match. Use `--ignore-drive-hash` flag (not recommended) or restore matching drive_as_code.yaml.

---

### Q: How do I replicate old multiplicative behavior?

**A**: Use `type: multiplicative` in drive_as_code.yaml. Example in L0_0_minimal, L1_full_observability configs.

---

### Q: What happened to AdaptiveRewardStrategy?

**A**: Replaced by modifiers. Old "adaptive" behavior = multiplicative + crisis suppression modifier. See L0_5_dual_resource for example.

---

### Q: Can I still write custom reward logic in Python?

**A**: Yes, via VFS. Write custom computation in BAC, expose as VFS variable with `readable_by: ["engine"]`, reference in DAC with `type: vfs_variable`.

---

### Q: How do I debug reward computation?

**A**: Enable `log_components: true` and `log_modifiers: true` in composition. Watch TensorBoard for extrinsic/intrinsic/shaping breakdown and modifier multipliers.

---

### Q: What if I need a reward structure not supported by DAC?

**A**: Two options:
1. Use `type: vfs_variable` to delegate to custom VFS computation
2. Implement new strategy type in `dac_engine.py` (contributions welcome!)

---

### Q: Why is this a breaking change with zero backwards compatibility?

**A**: Pre-release status (zero users). Maintaining backwards compatibility for non-existent user base creates technical debt. Clean break now = simpler codebase at launch.

---

**Last Updated**: 2025-11-12
**Status**: PRODUCTION (TASK-004C Complete)
**Migration Required**: YES (Breaking change)
