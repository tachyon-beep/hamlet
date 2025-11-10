# Migration Guide: Pre-Compiler to Compiler Architecture

**For**: Operators upgrading from pre-compiler HAMLET to UAC compiler architecture
**Version**: Post-TASK-004A (compiler implementation)
**Date**: 2025-11-11

---

## Overview

The Universe Compiler (UAC) fundamentally changes how HAMLET processes configuration files. This guide helps you migrate existing configs and training runs to the new architecture.

**Key Change**: Configs are now **compiled once** into an immutable `CompiledUniverse`, then executed many times. This catches errors at compile time rather than runtime.

---

## Breaking Changes

### 1. Affordance Schema Migration (CRITICAL)

**Old Schema** (pre-compiler):
```yaml
affordances:
  - id: bed
    name: Bed
    effects:  # ❌ No longer supported
      - {meter: energy, amount: 0.5}
    required_ticks: 5  # ❌ No longer supported
```

**New Schema** (compiler):
```yaml
affordances:
  - id: bed
    name: Bed
    effect_pipeline:  # ✅ Required
      instant:  # or per_tick/on_completion
        - {meter: energy, amount: 0.5}
    capabilities:  # ✅ Required for multi-tick
      - {type: multi_tick, duration_ticks: 5}
    interaction_type: instant  # ✅ Required
```

**Migration Tool**:
```bash
python scripts/migrate_affordances.py configs/your_config/affordances.yaml
```

**What Changed**:
- `effects` → `effect_pipeline.instant`
- `effects_per_tick` → `effect_pipeline.per_tick`
- `completion_bonus` → `effect_pipeline.on_completion`
- `required_ticks` → `capabilities[type=multi_tick].duration_ticks`
- `interaction_type` now required (was optional)
- For `dual` type: add `duration_ticks` at top level

### 2. Redundant Meter Definitions Removed

**Before**: `affordances.yaml` duplicated meter definitions from `bars.yaml`

**After**: Only `bars.yaml` defines meters (single source of truth)

**Action**: Run cleanup script:
```bash
python scripts/cleanup_redundant_meters.py
```

This removes the redundant `meters:` section from affordances.yaml.

### 3. No-Defaults Principle Enforcement

**Before**: Some config parameters had hidden defaults in code

**After**: ALL parameters must be explicitly specified in YAML

**Example Failure**:
```
ERROR: exploration.min_survival_for_annealing - Field required
```

**Fix**: Add missing field to `training.yaml`:
```yaml
exploration:
  min_survival_for_annealing: 100.0  # Explicit value required
```

Consult `configs/templates/training.yaml` for complete parameter list.

---

## Migration Steps

### Step 1: Backup Your Configs

```bash
cp -r configs/my_universe configs/my_universe.backup
```

### Step 2: Migrate Affordance Schema

```bash
python scripts/migrate_affordances.py configs/my_universe/affordances.yaml
```

**Verify migration**:
```bash
python -m townlet.compiler validate configs/my_universe
```

### Step 3: Remove Redundant Meter Definitions

```bash
python scripts/cleanup_redundant_meters.py configs/my_universe/affordances.yaml
```

### Step 4: Fix No-Defaults Violations

Run compiler validation and add any missing required fields:

```bash
python -m townlet.compiler validate configs/my_universe
```

If you see validation errors, check `configs/templates/` for correct parameter names.

### Step 5: Test Compilation

```bash
# Compile universe
python -m townlet.compiler compile configs/my_universe

# Inspect compiled metadata
python -m townlet.compiler inspect configs/my_universe

# Verify training still works
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run scripts/run_demo.py --config configs/my_universe
```

---

## Checkpoint Compatibility

### New Checkpoints (Post-Compiler)

Checkpoints now include:
- `config_hash`: Content hash of compiled universe
- `universe_metadata`: Full metadata from compiler
- `observation_dim`: From compiler calculation
- `compiler_version`: UAC version used

**Loading**: Use `CompiledUniverse.check_checkpoint_compatibility()`:
```python
universe = UniverseCompiler().compile(Path("configs/L1"))
checkpoint = torch.load("checkpoint.pt")

compatible, reason = universe.check_checkpoint_compatibility(checkpoint)
if not compatible:
    print(f"Incompatible: {reason}")
```

### Old Checkpoints (Pre-Compiler)

**Problem**: Old checkpoints lack `config_hash` and compiler metadata

**Solutions**:

#### Option 1: Retrain (Recommended)
Cleanest approach - retrain with compiled universe:
```bash
uv run scripts/run_demo.py --config configs/my_universe
```

#### Option 2: Best-Effort Loading (Advanced)
If retraining is impractical:

```python
from townlet.universe.compiler import UniverseCompiler

# Compile current config
universe = UniverseCompiler().compile(Path("configs/my_universe"))

# Load old checkpoint
checkpoint = torch.load("old_checkpoint.pt")

# Manual compatibility check
if checkpoint.get("observation_dim") == universe.metadata.observation_dim:
    # Dimensions match - probably safe
    population.load_checkpoint_state(checkpoint)
else:
    # Incompatible - must retrain
    raise ValueError("Observation dimensions changed - retrain required")
```

**Risks**:
- Config drift (old checkpoint may expect different dynamics)
- No validation of meter counts, affordance counts, etc.
- Subtle bugs possible if config changed

---

## Config Organization Changes

### Cache Directory

Compiled universes are cached in `.compiled/`:

```
configs/my_universe/
├── .compiled/
│   └── universe.msgpack  # Compiled cache (auto-generated)
├── bars.yaml
├── cascades.yaml
├── affordances.yaml
├── substrate.yaml
├── training.yaml
└── variables_reference.yaml
```

**Cache Invalidation**: Automatic when any YAML file changes (mtime + content hash)

**Gitignore**: Add to `.gitignore`:
```
configs/**/.compiled/
```

### Validation in CI

Add compiler validation to CI:

```yaml
# .github/workflows/config-validation.yml
- name: Validate Config Packs
  run: |
    for config in configs/L*; do
      python -m townlet.compiler validate "$config"
    done
```

---

## Troubleshooting

### Error: "Extra inputs are not permitted"

**Cause**: Using legacy affordance field (e.g., `effects`, `required_ticks`)

**Fix**: Run migration script:
```bash
python scripts/migrate_affordances.py configs/my_universe/affordances.yaml
```

### Error: "Field required"

**Cause**: Missing required parameter (no-defaults principle)

**Fix**: Add parameter to YAML file. Check `configs/templates/` for examples:
```bash
# Find which section needs the field
grep -r "min_survival_for_annealing" configs/templates/
```

### Error: "Circular cascade detected"

**Cause**: Cascade dependencies form a cycle (e.g., A→B→C→A)

**Fix**: Review `cascades.yaml` and break the cycle:
```yaml
cascades:
  - name: mood_to_energy
    source: mood
    target: energy  # Remove if creates cycle
```

### Error: "Economic imbalance"

**Cause**: Total costs exceed total income (universe is unwinnable)

**Options**:
1. Add more income-generating affordances
2. Reduce costs
3. Allow unfeasible universe (for testing):
   ```yaml
   # training.yaml
   environment:
     allow_unfeasible_universe: true
   ```

### Warning: "Cache file corrupted"

**Cause**: Cached universe is invalid (partial write, version mismatch)

**Fix**: Delete cache and recompile:
```bash
rm configs/my_universe/.compiled/universe.msgpack
python -m townlet.compiler compile configs/my_universe
```

### Performance: Slow Compilation

**Symptoms**: Compilation takes >5 seconds

**Causes**:
- Large config pack (many affordances/meters)
- First compilation (no cache)
- Cache invalidation on every run

**Solutions**:
1. Use cache (automatic after first compile)
2. Reduce cross-validation strictness (advanced):
   ```yaml
   # training.yaml
   environment:
     skip_economic_validation: true  # Faster but less safe
   ```

---

## Feature Changes

### Compile-Time vs Runtime Errors

**Before**: Config errors discovered during training (10+ minutes in)

**After**: Config errors caught immediately during compilation

**Example**:
```bash
# Old behavior
$ uv run scripts/run_demo.py --config configs/bad_config
[... 10 minutes of training ...]
ERROR: Cascade references non-existent meter 'stamina'

# New behavior
$ python -m townlet.compiler validate configs/bad_config
ERROR Stage 3: Resolve failed
  - [UAC-RES-001] cascades.yaml - References non-existent meter 'stamina'
[Fails in <1 second]
```

### Immutability Guarantees

**Before**: Configs could be accidentally mutated during training

**After**: `CompiledUniverse` is frozen (cannot be modified)

**Benefit**: Reproducibility - same config always produces same behavior

### Rich Error Messages

**Before**: Generic Pydantic errors

**After**: Structured error codes with hints

**Example**:
```
[UAC-VAL-002] affordances.yaml - Economic imbalance: income (0.26) < costs (1.36)
Hint: Add income-generating affordances or enable allow_unfeasible_universe
```

Error codes:
- `UAC-RES-*`: Reference resolution errors (Stage 3)
- `UAC-VAL-*`: Cross-validation errors (Stage 4)
- `UAC-ACT-*`: Action space errors
- `UAC-SEC-*`: Security limit violations

---

## Best Practices

### 1. Validate Early, Validate Often

```bash
# After editing any config file
python -m townlet.compiler validate configs/my_universe
```

Add to pre-commit hook:
```bash
#!/bin/bash
# .git/hooks/pre-commit
for changed_config in $(git diff --cached --name-only configs/*/); do
    config_dir=$(dirname "$changed_config")
    python -m townlet.compiler validate "$config_dir" || exit 1
done
```

### 2. Version Your Configs

Tag configs with training runs:
```bash
git tag -a "run-2025-11-11-L1" -m "Training run on L1 full observability"
git push --tags
```

### 3. Use Compiler Metadata

Query compiled universe for dimensions:
```python
universe = UniverseCompiler().compile(Path("configs/L1"))
print(f"Observation dim: {universe.metadata.observation_dim}")
print(f"Action dim: {universe.metadata.action_dim}")
print(f"Meter count: {universe.metadata.meter_count}")
```

### 4. Cache Compiled Universes

For repeated runs, cache compilation results:
```python
# First run: compile and cache
universe = UniverseCompiler().compile(Path("configs/L1"))
universe.save_to_cache(Path("configs/L1/.compiled/universe.msgpack"))

# Subsequent runs: load from cache (much faster)
universe = UniverseCompiler().load_from_cache(Path("configs/L1/.compiled/universe.msgpack"))
```

---

## FAQ

**Q: Do I need to migrate all configs at once?**
A: No. Migrate one at a time and test each.

**Q: Can I use old checkpoints with new configs?**
A: Only if observation dimensions match exactly. Recommended to retrain.

**Q: What if migration script fails?**
A: File an issue with the error message and your affordances.yaml file.

**Q: How do I debug compilation errors?**
A: Use `--verbose` flag:
```bash
python -m townlet.compiler validate configs/my_universe --verbose
```

**Q: Can I disable compiler validation?**
A: No. Validation is mandatory to ensure config correctness. Use `allow_unfeasible_universe` for specific warnings.

**Q: Will compilation slow down my training?**
A: No. Compilation happens once (cached), then reused for all training runs.

**Q: What if I have custom affordance fields?**
A: Extra fields are rejected (no-defaults principle). Remove or map to standard fields.

---

## Getting Help

**Validation errors**: Check `docs/UNIVERSE-COMPILER.md` §6 (Troubleshooting)

**Schema questions**: See `docs/config-schemas/affordances.md`

**Bug reports**: File issue at GitHub with:
- Config files (affordances.yaml, training.yaml, etc.)
- Full error message
- Output of `python -m townlet.compiler validate --verbose`

**Examples**: Study migrated configs:
- `configs/L0_0_minimal/` - Simple example
- `configs/L1_full_observability/` - Full 8-meter universe
- `configs/templates/` - Annotated templates

---

## Summary Checklist

Before deploying compiler-based training:

- [ ] Migrated affordances schema (`migrate_affordances.py`)
- [ ] Removed redundant meter definitions (`cleanup_redundant_meters.py`)
- [ ] Added all required no-defaults parameters
- [ ] Validated all configs compile successfully
- [ ] Tested training run completes
- [ ] Updated CI to validate configs
- [ ] Backed up old checkpoints
- [ ] Tagged config versions in git

---

**Next Steps**: See `docs/UNIVERSE-COMPILER.md` for operational guide.
