# Substrate Configuration Smoke Tests

Quick validation guide for substrate.yaml configs before running experiments.

## Pre-Training Validation

**Before starting any training run**, validate substrate configs:

```bash
# Validate all production configs
python scripts/validate_substrate_configs.py

# Validate specific config pack
python scripts/validate_substrate_configs.py --config-pack L1_full_observability

# Verbose mode (detailed validation steps)
python scripts/validate_substrate_configs.py --config-pack L1_full_observability --verbose
```

**Expected output** (all configs valid):
```
Validating 6 config pack(s)...

L0_0_minimal                   ✅ VALID
L0_5_dual_resource             ✅ VALID
L1_full_observability          ✅ VALID
L2_partial_observability       ✅ VALID
L3_temporal_mechanics          ✅ VALID
test                           ✅ VALID

============================================================
✅ All configs valid!
============================================================
```

## What the Validation Script Checks

For each config pack, the script verifies:

1. **File Existence**: `substrate.yaml` exists in config pack directory
2. **Schema Validation**: YAML loads and passes Pydantic validation
3. **Factory Build**: `SubstrateFactory.build()` creates substrate successfully
4. **Substrate Operations**: Basic operations work (initialization, movement, distance)

## Quick Manual Validation

If the validation script fails, manually check:

### 1. Schema Structure

```yaml
version: "1.0"
description: "Your description here"
type: "grid"  # or "aspatial"

grid:  # Required if type="grid"
  topology: "square"
  width: 8
  height: 8
  boundary: "clamp"  # clamp, wrap, bounce, sticky
  distance_metric: "manhattan"  # manhattan, euclidean, chebyshev
```

### 2. No-Defaults Principle

**All fields must be explicit** - no hidden defaults allowed:

- ❌ Missing `description` → ValidationError
- ❌ Missing `grid` section when `type="grid"` → ValidationError
- ❌ Missing `boundary` or `distance_metric` → ValidationError

### 3. Valid Literal Values

**Boundary modes** (only these 4 values allowed):
- `clamp` - Hard walls (agent position clamped to edges)
- `wrap` - Toroidal wraparound (Pac-Man style)
- `bounce` - Elastic reflection (agent bounces back)
- `sticky` - Sticky walls (agent stays in place)

**Distance metrics** (only these 3 values allowed):
- `manhattan` - L1 norm, |x1-x2| + |y1-y2|
- `euclidean` - L2 norm, sqrt((x1-x2)² + (y1-y2)²)
- `chebyshev` - L∞ norm, max(|x1-x2|, |y1-y2|)

## Integration Testing

After Phase 4 (Environment Integration), verify configs load correctly:

```bash
# Run integration tests (currently skipped until Phase 4)
pytest tests/test_townlet/integration/test_substrate_migration.py -v
```

## Common Errors and Fixes

### Error: "Substrate config not found"

**Cause**: Missing `substrate.yaml` in config pack directory

**Fix**: Create `substrate.yaml` using template:
```bash
cp configs/templates/substrate.yaml configs/YOUR_CONFIG/substrate.yaml
# Edit to match your requirements
```

### Error: "type='grid' requires grid configuration"

**Cause**: Config has `type: "grid"` but missing `grid:` section

**Fix**: Add grid configuration:
```yaml
grid:
  topology: "square"
  width: 8
  height: 8
  boundary: "clamp"
  distance_metric: "manhattan"
```

### Error: "Input should be 'clamp', 'wrap', 'bounce' or 'sticky'"

**Cause**: Invalid boundary mode value

**Fix**: Use only allowed values: `clamp`, `wrap`, `bounce`, `sticky`

### Error: "Field required"

**Cause**: Missing required field (no-defaults principle violation)

**Fix**: Add all required fields explicitly. Check template for complete list.

## Observation Dimension Verification

After validating schema, verify observation dimensions match expected:

| Config | Grid Size | Grid Dim | Total Obs Dim |
|--------|-----------|----------|---------------|
| L0_0_minimal | 3×3 | 9 | 36 (9 + 8 + 15 + 4) |
| L0_5_dual_resource | 7×7 | 49 | 76 (49 + 8 + 15 + 4) |
| L1/L2/L3/test | 8×8 | 64 | 91 (64 + 8 + 15 + 4) |

**Formula**: `total_obs_dim = grid_dim + 8_meters + 15_affordances + 4_temporal`

## Troubleshooting

If validation fails:

1. **Check YAML syntax**: Use `yamllint` or online YAML validator
2. **Check required fields**: Compare with template (`configs/templates/substrate.yaml`)
3. **Check literal values**: Ensure boundary/distance_metric use allowed values
4. **Check grid dimensions**: Must be positive integers (width > 0, height > 0)
5. **Run verbose validation**: `python scripts/validate_substrate_configs.py --verbose`

## When to Run Validation

**Always validate before**:
- Starting new training experiments
- Modifying existing substrate configs
- Creating new config packs
- Deploying to production
- Pushing config changes to git

**Recommended**: Add validation to CI/CD pipeline:
```bash
# In .github/workflows/tests.yml or similar
- name: Validate substrate configs
  run: python scripts/validate_substrate_configs.py
```

## Next Steps

After successful validation:

1. ✅ All substrate configs valid
2. → Proceed with training: `uv run scripts/run_demo.py --config configs/YOUR_CONFIG`
3. → Monitor training metrics and survival rates

See `CLAUDE.md` for training commands and configuration details.
