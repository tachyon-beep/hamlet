# Scripts Directory

Utility scripts for HAMLET/Townlet development and CI/CD.

---

## Validation Scripts

### `validate_substrates.py` ✅ **RECOMMENDED**

**Substrate configuration lint tool** - validates substrate.yaml files are well-formed.

Fast pre-commit lint tool that validates config-level correctness. Runtime integration
testing is in the test suite (pytest).

**What it validates** (config-level):

- ✓ substrate.yaml exists
- ✓ YAML schema is valid (Pydantic)
- ✓ Substrate can be instantiated (factory build)
- ✓ Basic substrate operations work (position init, movement, distance)

**What it does NOT validate** (use test suite):

- ✗ Environment integration → `pytest tests/test_townlet/unit/test_env_substrate_loading.py`
- ✗ Episode execution → `pytest tests/test_townlet/integration/test_substrate_migration.py`
- ✗ Observation dimensions → `pytest tests/test_townlet/integration/test_data_flows.py`

**Quick Start**:

```bash
# Validate all configs (~1 second)
python scripts/validate_substrates.py

# Validate specific config
python scripts/validate_substrates.py --config configs/L1_full_observability

# CI mode (JSON output, no colors)
python scripts/validate_substrates.py --ci --json results.json

# Parallel execution (4 workers)
python scripts/validate_substrates.py --parallel 4
```

**Vision**: Foundation for general config linter that validates all UAC/BAC YAML configs
(substrate.yaml, bars.yaml, cascades.yaml, affordances.yaml, etc.)

**Exit Codes**:

- `0`: All validations passed
- `1`: Validation errors found
- `2`: Invalid arguments or setup error

---

## Training Scripts

### `run_demo.py`

**Unified demo server** for training, inference, and live visualization.

Single-command interface for running HAMLET training with live inference server.

```bash
# Start training + inference (then run frontend separately)
python scripts/run_demo.py --config configs/L1_full_observability --episodes 10000

# In another terminal: cd frontend && npm run dev

# Resume from checkpoint
python scripts/run_demo.py --config configs/L1_full_observability \
    --checkpoint-dir runs/L1_full_observability/2025-11-02_123456/checkpoints

# Custom inference port
python scripts/run_demo.py --config configs/L1_full_observability \
    --episodes 5000 --inference-port 8800
```

**Note**: Frontend runs separately for better stability and Vue HMR support.

---

## Configuration Validation Scripts

### `validate_configs.py`

**Legacy config validation** for bars.yaml and cascades.yaml.

Validates meter definitions and cascade relationships.

```bash
# Validate specific config pack
python scripts/validate_configs.py configs/L1_full_observability
```

**Features**:

- Validates bars.yaml structure (8 meters, indices, names)
- Validates cascades.yaml relationships
- Checks for schema violations

---

## Code Quality Scripts

### `no_defaults_lint.py`

**No-defaults linter** - enforces explicit configuration values.

Prevents implicit default parameters in Python code to ensure reproducible configs.

```bash
# Lint all source code
python scripts/no_defaults_lint.py src/townlet

# Check specific file
python scripts/no_defaults_lint.py src/townlet/environment/vectorized_env.py

# With whitelist
python scripts/no_defaults_lint.py src/townlet --whitelist .no-defaults-whitelist
```

See [README-no-defaults-lint.md](README-no-defaults-lint.md) for detailed documentation.

---

## Development Workflow

### Before Committing (Pre-commit Hook)

Run config lint to catch schema errors (fast, ~1 second):

```bash
python scripts/validate_substrates.py
```

### Before Training

Run full test suite to catch runtime issues:

```bash
pytest tests/test_townlet/unit/test_env_substrate_loading.py
pytest tests/test_townlet/integration/test_substrate_migration.py
```

### Before Releasing

Run comprehensive test suite:

```bash
pytest -v
```

---

## CI/CD Recommendations

### Pre-Merge Checks (Fast)

```bash
# Config lint (~1s) - catches schema errors
python scripts/validate_substrates.py --ci --json config-lint.json

# Full test suite (~90s) - catches runtime issues
pytest --ci
```

### As Pre-Commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
python scripts/validate_substrates.py
```

---

## Parallel Execution

Validation supports parallel execution for multiple configs:

```bash
# Auto-detect CPU cores
python scripts/validate_substrates.py --parallel 0

# Use 4 workers
python scripts/validate_substrates.py --parallel 4
```

**Performance** (6 config packs):

- Sequential: ~1 second
- Parallel (4 workers): ~0.3 seconds

---

## JSON Output Format

```json
{
  "total": 6,
  "passed": 6,
  "failed": 0,
  "results": [
    {
      "config_name": "L1_full_observability",
      "success": true,
      "duration_ms": 2.5,
      "checks": {
        "substrate_yaml_exists": {"status": "✓", "message": ""},
        "schema_validation": {"status": "✓", "message": ""},
        "factory_build": {"status": "✓", "message": ""},
        "substrate_operations": {"status": "✓", "message": ""}
      },
      "errors": [],
      "metadata": {
        "substrate_type": "grid",
        "grid_size": "8×8",
        "boundary": "clamp",
        "distance_metric": "manhattan",
        "position_dim": 2
      }
    }
  ]
}
```

---

## Troubleshooting

### "No config packs found to validate"

Ensure you're running from the project root:

```bash
cd /home/john/hamlet
python scripts/validate_substrates.py
```

### Validation fails with import errors

Ensure dependencies are installed:

```bash
uv sync
```

### CI mode shows colors

Force disable colors:

```bash
python scripts/validate_substrates.py --ci --no-colors
```

---

**Author**: TASK-002A Phase 8 + Refactoring
**Status**: Production-ready config lint tool
