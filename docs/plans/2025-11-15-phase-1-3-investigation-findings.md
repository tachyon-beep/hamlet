# Phase 1-3 Implementation Plan - Investigation Findings

**Date**: 2025-11-15
**Investigator**: Claude (executing-plans skill)
**Plan**: `docs/plans/2025-11-15-config-v2.1-phases-1-3.md`

## Investigation Summary

Conducted pre-execution verification to resolve concerns identified in plan review.

## Findings

### ✅ Finding 1: Reference Config Exists
- **File**: `docs/bugs/BUNDLE-01-curriculum-observation-architecture/reference-config-v2.1-complete.yaml`
- **Size**: 37K
- **Status**: EXISTS and accessible
- **Impact**: No blocker, plan can proceed

### ✅ Finding 2: Existing DTOs Structure

**Bars (Meters)**:
- **File**: `src/townlet/config/bar.py`
- **DTOs**:
  - `BarConfig` (single meter)
  - `load_bars_config(config_dir) -> list[BarConfig]`
  - `BarsConfig` imported from `townlet.environment.cascade_config`
- **Status**: EXISTS, compatible with plan
- **Impact**: Plan's assumption of `BarsConfig` is CORRECT (but it's in cascade_config not bar.py)

**Affordances**:
- **File**: `src/townlet/config/affordance.py`
- **DTOs**:
  - `AffordanceConfig` (single affordance)
  - `load_affordances_config(config_dir) -> list[AffordanceConfig]`
- **Status**: EXISTS, compatible with plan
- **Impact**: Plan references to `AffordancesConfig` need correction

**Training**:
- **File**: `src/townlet/config/training.py`
- **DTOs**:
  - `TrainingConfig` (full training config)
  - `load_training_config(config_dir, training_config_path) -> TrainingConfig`
- **Status**: EXISTS with `from_yaml` method
- **Impact**: Plan's assumption CORRECT

### ⚠️ Finding 3: ActionsConfig Labels Field MISMATCH

**Reference Config (v2.1)**:
```yaml
labels:
  preset: gaming  # Options: gaming, 6dof, cardinal, math
```

**Plan's DTO (Task 3.5, line 1485)**:
```python
labels: Literal["gaming", "6dof", "cardinal", "math"]  # WRONG!
```

**Correct DTO Should Be**:
```python
class ActionLabelsConfig(BaseModel):
    """Action labels configuration."""
    preset: Literal["gaming", "6dof", "cardinal", "math"]

    class Config:
        extra = "forbid"

class ActionsConfig(BaseModel):
    """Actions configuration (actions.yaml)."""
    version: str
    substrate_actions: SubstrateActionsConfig
    custom_actions: List[CustomActionDefinition]
    labels: ActionLabelsConfig  # NOT Literal - it's a nested DTO!
```

**Impact**: CRITICAL - Plan's Task 3.5 will fail validation unless fixed

## Recommended Fixes

### Fix 1: Update Task 3.5 (ActionsConfig)

Replace lines 1479-1499 in plan with:

```python
"""Actions configuration DTOs for v2.1."""

from pydantic import BaseModel
from typing import List, Literal


class CustomActionDefinition(BaseModel):
    """Custom action definition."""

    name: str
    description: str
    enabled_by_default: bool

    class Config:
        extra = "forbid"


class SubstrateActionsConfig(BaseModel):
    """Substrate actions configuration."""

    inherit: bool

    class Config:
        extra = "forbid"


class ActionLabelsConfig(BaseModel):
    """Action labels (UI terminology) configuration."""

    preset: Literal["gaming", "6dof", "cardinal", "math"]

    class Config:
        extra = "forbid"


class ActionsConfig(BaseModel):
    """Actions configuration (actions.yaml)."""

    version: str
    substrate_actions: SubstrateActionsConfig
    custom_actions: List[CustomActionDefinition]
    labels: ActionLabelsConfig  # Nested DTO for labels

    class Config:
        extra = "forbid"

    @classmethod
    def from_yaml(cls, path):
        """Load from YAML file."""
        import yaml
        from pathlib import Path

        with open(Path(path)) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

### Fix 2: Update Task 3.5 Verification

The test in Task 3.5 Step 2 should verify nested structure:

```python
config = ActionsConfig.from_yaml("configs/default_curriculum/actions.yaml")
print(f"✓ Custom actions: {len(config.custom_actions)}")
print(f"✓ Substrate inherit: {config.substrate_actions.inherit}")
print(f"✓ Labels preset: {config.labels.preset}")  # Access nested field
assert len(config.custom_actions) == 4
assert config.substrate_actions.inherit == True
assert config.labels.preset == "gaming"
print("SUCCESS: ActionsConfig working")
```

### Fix 3: Clarify DTO Import References

Plan should note that existing DTOs use:
- `list[BarConfig]` not `BarsConfig` (collection is a list)
- `list[AffordanceConfig]` not `AffordancesConfig`
- Loading functions return lists, not collection DTOs

**However**: The plan creates NEW v2.1 structure which MAY need collection DTOs. This is acceptable as a design choice for v2.1.

## Conclusion

**Blockers Resolved**: ✅ All blockers resolved

**Required Changes**:
1. ✅ Reference config verified (exists)
2. ✅ Existing DTOs verified (compatible)
3. ⚠️ ActionsConfig labels field MUST be fixed (critical)

**Recommendation**: Update Task 3.5 in plan with correct ActionLabelsConfig nested DTO, then proceed with execution.

**Risk Assessment**:
- **Pre-fix**: HIGH (validation will fail)
- **Post-fix**: LOW (all prerequisites verified)
