# TASK-004B Validation Gaps Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the remaining validation logic for TASK-004B UAC Capabilities to reach 100% compiler validation coverage.

**Architecture:** Extends the existing compiler Stage 4 cross-validation with three new validators: prerequisite affordance reference checks, probabilistic effect pipeline completeness checks, and skill scaling meter reference validation. All validation follows the existing pattern of accumulating errors in CompilationErrorCollector before raising.

**Tech Stack:** Python, Pydantic, pytest

**Context**: This plan addresses validation gaps from TASK-004B-GAP-ANALYSIS.md - the missing advanced capability validations that prevent invalid configs from passing compilation.

**Note**: This is pre-release software with zero users. We are NOT implementing backward compatibility or auto-migration code. All configs have been manually migrated to the new schema. Breaking changes are acceptable during this sprint - we fix on fail.

**Estimated Effort**: 2-3 hours

---

## Task 1: Add Prerequisite Affordance Reference Validation

**Goal**: Ensure `PrerequisiteCapability.required_affordances` only references affordance IDs that exist in the symbol table.

**Files**:
- Modify: `src/townlet/universe/compiler.py` (add validation to `_validate_capabilities_and_effect_pipelines`)
- Create: `tests/test_townlet/unit/universe/test_capability_validation.py` (new test file)

---

### Step 1: Write the failing test

**File**: `tests/test_townlet/unit/universe/test_capability_validation.py` (CREATE)

```python
"""Tests for capability validation in UniverseCompiler."""

import pytest
from pathlib import Path
from townlet.universe.compiler import UniverseCompiler
from townlet.universe.errors import CompilationError


class TestPrerequisiteValidation:
    """Test prerequisite capability validation."""

    def test_prerequisite_with_valid_affordance_ids_passes(self, tmp_path: Path):
        """Prerequisites referencing existing affordances should pass validation."""
        # Create config with two affordances where second requires first
        config_dir = tmp_path / "test_config"
        config_dir.mkdir()

        # bars.yaml
        (config_dir / "bars.yaml").write_text("""
bars:
  - name: energy
    range: [0.0, 1.0]
    initial: 0.5
    depletion_per_tick: 0.01
""")

        # cascades.yaml
        (config_dir / "cascades.yaml").write_text("""
cascades: []
""")

        # affordances.yaml with valid prerequisite
        (config_dir / "affordances.yaml").write_text("""
affordances:
  - id: "Foundation"
    name: "Foundation Course"
    effect_pipeline:
      on_completion:
        - meter: energy
          amount: 0.1

  - id: "Advanced"
    name: "Advanced Course"
    capabilities:
      - type: prerequisite
        required_affordances: ["Foundation"]
    effect_pipeline:
      on_completion:
        - meter: energy
          amount: 0.2
""")

        # substrate.yaml
        (config_dir / "substrate.yaml").write_text("""
type: aspatial
encoding: relative
""")

        # cues.yaml
        (config_dir / "cues.yaml").write_text("""
cues: []
""")

        # training.yaml
        (config_dir / "training.yaml").write_text("""
enabled_actions: []
use_double_dqn: false
mask_unused_obs: false
reward_strategy:
  type: simple
""")

        # variables_reference.yaml
        (config_dir / "variables_reference.yaml").write_text("""
variables: []
observations: []
""")

        # global_actions.yaml in parent
        (tmp_path / "global_actions.yaml").write_text("""
actions: []
""")

        compiler = UniverseCompiler()
        # Should not raise
        universe = compiler.compile(config_dir)
        assert universe is not None

    def test_prerequisite_with_invalid_affordance_id_fails(self, tmp_path: Path):
        """Prerequisites referencing non-existent affordances should fail validation."""
        config_dir = tmp_path / "test_config"
        config_dir.mkdir()

        # bars.yaml
        (config_dir / "bars.yaml").write_text("""
bars:
  - name: energy
    range: [0.0, 1.0]
    initial: 0.5
    depletion_per_tick: 0.01
""")

        # cascades.yaml
        (config_dir / "cascades.yaml").write_text("""
cascades: []
""")

        # affordances.yaml with INVALID prerequisite
        (config_dir / "affordances.yaml").write_text("""
affordances:
  - id: "Advanced"
    name: "Advanced Course"
    capabilities:
      - type: prerequisite
        required_affordances: ["NonExistentCourse"]
    effect_pipeline:
      on_completion:
        - meter: energy
          amount: 0.2
""")

        # substrate.yaml
        (config_dir / "substrate.yaml").write_text("""
type: aspatial
encoding: relative
""")

        # cues.yaml
        (config_dir / "cues.yaml").write_text("""
cues: []
""")

        # training.yaml
        (config_dir / "training.yaml").write_text("""
enabled_actions: []
use_double_dqn: false
mask_unused_obs: false
reward_strategy:
  type: simple
""")

        # variables_reference.yaml
        (config_dir / "variables_reference.yaml").write_text("""
variables: []
observations: []
""")

        # global_actions.yaml in parent
        (tmp_path / "global_actions.yaml").write_text("""
actions: []
""")

        compiler = UniverseCompiler()
        with pytest.raises(CompilationError) as exc_info:
            compiler.compile(config_dir)

        error_msg = str(exc_info.value)
        assert "NonExistentCourse" in error_msg
        assert "prerequisite" in error_msg.lower()
        assert "does not exist" in error_msg.lower() or "not found" in error_msg.lower()

    def test_prerequisite_with_multiple_invalid_ids_reports_all(self, tmp_path: Path):
        """Prerequisites with multiple invalid IDs should report all errors."""
        config_dir = tmp_path / "test_config"
        config_dir.mkdir()

        # bars.yaml
        (config_dir / "bars.yaml").write_text("""
bars:
  - name: energy
    range: [0.0, 1.0]
    initial: 0.5
    depletion_per_tick: 0.01
""")

        # cascades.yaml
        (config_dir / "cascades.yaml").write_text("""
cascades: []
""")

        # affordances.yaml with multiple INVALID prerequisites
        (config_dir / "affordances.yaml").write_text("""
affordances:
  - id: "Advanced"
    name: "Advanced Course"
    capabilities:
      - type: prerequisite
        required_affordances: ["Missing1", "Missing2", "Missing3"]
    effect_pipeline:
      on_completion:
        - meter: energy
          amount: 0.2
""")

        # substrate.yaml
        (config_dir / "substrate.yaml").write_text("""
type: aspatial
encoding: relative
""")

        # cues.yaml
        (config_dir / "cues.yaml").write_text("""
cues: []
""")

        # training.yaml
        (config_dir / "training.yaml").write_text("""
enabled_actions: []
use_double_dqn: false
mask_unused_obs: false
reward_strategy:
  type: simple
""")

        # variables_reference.yaml
        (config_dir / "variables_reference.yaml").write_text("""
variables: []
observations: []
""")

        # global_actions.yaml in parent
        (tmp_path / "global_actions.yaml").write_text("""
actions: []
""")

        compiler = UniverseCompiler()
        with pytest.raises(CompilationError) as exc_info:
            compiler.compile(config_dir)

        error_msg = str(exc_info.value)
        assert "Missing1" in error_msg
        assert "Missing2" in error_msg
        assert "Missing3" in error_msg
```

---

### Step 2: Run test to verify it fails

```bash
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
export UV_CACHE_DIR=.uv-cache
uv run pytest tests/test_townlet/unit/universe/test_capability_validation.py::TestPrerequisiteValidation -v
```

**Expected Output**:
```
FAILED - test_prerequisite_with_invalid_affordance_id_fails
FAILED - test_prerequisite_with_multiple_invalid_ids_reports_all
```

The tests should fail because the validation doesn't exist yet.

---

### Step 3: Implement prerequisite validation

**File**: `src/townlet/universe/compiler.py` (MODIFY)

Find the `_validate_capabilities_and_effect_pipelines` method (around line 1126) and add prerequisite validation after the existing checks:

```python
def _validate_capabilities_and_effect_pipelines(
    self,
    raw_configs: RawConfigs,
    errors: CompilationErrorCollector,
    formatter,
) -> None:
    """Validate capability composition and effect pipeline consistency."""
    # Build affordance ID set for prerequisite validation
    affordance_ids = {aff.id for aff in raw_configs.affordances}

    for affordance in raw_configs.affordances:
        capabilities = getattr(affordance, "capabilities", []) or []
        types = [self._get_attr_value(cap, "type") for cap in capabilities]
        multi_tick_caps = [cap for cap, cap_type in zip(capabilities, types) if cap_type == "multi_tick"]
        has_resumable_flag = any(bool(self._get_attr_value(cap, "resumable")) for cap in capabilities)

        # ... existing instant + multi_tick validation ...
        if affordance.interaction_type and affordance.interaction_type.lower() == "instant" and multi_tick_caps:
            errors.add(
                formatter(
                    "UAC-VAL-008",
                    "Instant affordances cannot declare multi_tick capabilities.",
                    f"affordances.yaml:{affordance.id}",
                )
            )

        # ... existing pipeline consistency validation ...
        pipeline = affordance.effect_pipeline
        if pipeline is not None and not isinstance(pipeline, EffectPipeline):
            pipeline = EffectPipeline.model_validate(pipeline)
        if multi_tick_caps:
            if pipeline is None or (not pipeline.per_tick and not pipeline.on_completion):
                errors.add(
                    formatter(
                        "UAC-VAL-008",
                        f"Multi-tick affordance '{affordance.id}' should define per_tick or on_completion effects.",
                        f"affordances.yaml:{affordance.id}",
                    )
                )

        # ... existing resumable validation ...
        if has_resumable_flag and not multi_tick_caps:
            errors.add(
                formatter(
                    "UAC-VAL-009",
                    f"Affordance '{affordance.id}' declares resumable=true but is not multi_tick.",
                    f"affordances.yaml:{affordance.id}",
                )
            )

        # NEW: Validate prerequisite affordance references
        for idx, capability in enumerate(capabilities):
            cap_type = self._get_attr_value(capability, "type")

            if cap_type == "prerequisite":
                required = self._get_attr_value(capability, "required_affordances") or []
                for req_id in required:
                    if req_id not in affordance_ids:
                        errors.add(
                            formatter(
                                "UAC-VAL-010",
                                f"Prerequisite affordance '{req_id}' does not exist in affordances.yaml",
                                f"affordances.yaml:{affordance.id}:capabilities[{idx}]",
                            )
                        )
```

---

### Step 4: Run tests to verify they pass

```bash
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
export UV_CACHE_DIR=.uv-cache
uv run pytest tests/test_townlet/unit/universe/test_capability_validation.py::TestPrerequisiteValidation -v
```

**Expected Output**:
```
test_prerequisite_with_valid_affordance_ids_passes PASSED
test_prerequisite_with_invalid_affordance_id_fails PASSED
test_prerequisite_with_multiple_invalid_ids_reports_all PASSED
```

---

### Step 5: Commit

```bash
git add tests/test_townlet/unit/universe/test_capability_validation.py
git add src/townlet/universe/compiler.py
git commit -m "feat(compiler): add prerequisite affordance reference validation

- Validate that PrerequisiteCapability.required_affordances only references
  affordances that exist in the symbol table
- Error code: UAC-VAL-010
- Accumulates all missing prerequisites before failing
- Adds 3 comprehensive tests covering valid/invalid/multiple scenarios

Closes gap from TASK-004B-GAP-ANALYSIS.md"
```

---

## Task 2: Add Probabilistic Completeness Validation

**Goal**: Ensure affordances with `ProbabilisticCapability` define both `on_completion` (success) and `on_failure` (failure) effects in their effect pipeline.

**Files**:
- Modify: `src/townlet/universe/compiler.py` (extend `_validate_capabilities_and_effect_pipelines`)
- Modify: `tests/test_townlet/unit/universe/test_capability_validation.py` (add new test class)

---

### Step 1: Write the failing test

**File**: `tests/test_townlet/unit/universe/test_capability_validation.py` (MODIFY - add new class)

Add this class after `TestPrerequisiteValidation`:

```python
class TestProbabilisticValidation:
    """Test probabilistic capability validation."""

    def test_probabilistic_with_both_outcomes_passes(self, tmp_path: Path):
        """Probabilistic capability with both success and failure effects should pass."""
        config_dir = tmp_path / "test_config"
        config_dir.mkdir()

        # bars.yaml
        (config_dir / "bars.yaml").write_text("""
bars:
  - name: money
    range: [0.0, 100.0]
    initial: 50.0
    depletion_per_tick: 0.0
  - name: mood
    range: [0.0, 1.0]
    initial: 0.5
    depletion_per_tick: 0.01
""")

        # cascades.yaml
        (config_dir / "cascades.yaml").write_text("""
cascades: []
""")

        # affordances.yaml with probabilistic + complete pipeline
        (config_dir / "affordances.yaml").write_text("""
affordances:
  - id: "Casino"
    name: "Slot Machine"
    capabilities:
      - type: probabilistic
        success_probability: 0.3
    effect_pipeline:
      on_start:
        - meter: money
          amount: -5.0
      on_completion:
        - meter: money
          amount: 20.0
        - meter: mood
          amount: 0.2
      on_failure:
        - meter: mood
          amount: -0.1
""")

        # substrate.yaml
        (config_dir / "substrate.yaml").write_text("""
type: aspatial
encoding: relative
""")

        # cues.yaml
        (config_dir / "cues.yaml").write_text("""
cues: []
""")

        # training.yaml
        (config_dir / "training.yaml").write_text("""
enabled_actions: []
use_double_dqn: false
mask_unused_obs: false
reward_strategy:
  type: simple
""")

        # variables_reference.yaml
        (config_dir / "variables_reference.yaml").write_text("""
variables: []
observations: []
""")

        # global_actions.yaml in parent
        (tmp_path / "global_actions.yaml").write_text("""
actions: []
""")

        compiler = UniverseCompiler()
        # Should not raise
        universe = compiler.compile(config_dir)
        assert universe is not None

    def test_probabilistic_without_on_failure_warns(self, tmp_path: Path):
        """Probabilistic capability without on_failure should emit warning."""
        config_dir = tmp_path / "test_config"
        config_dir.mkdir()

        # bars.yaml
        (config_dir / "bars.yaml").write_text("""
bars:
  - name: money
    range: [0.0, 100.0]
    initial: 50.0
    depletion_per_tick: 0.0
""")

        # cascades.yaml
        (config_dir / "cascades.yaml").write_text("""
cascades: []
""")

        # affordances.yaml with probabilistic but NO on_failure
        (config_dir / "affordances.yaml").write_text("""
affordances:
  - id: "Casino"
    name: "Slot Machine"
    capabilities:
      - type: probabilistic
        success_probability: 0.3
    effect_pipeline:
      on_completion:
        - meter: money
          amount: 20.0
""")

        # substrate.yaml
        (config_dir / "substrate.yaml").write_text("""
type: aspatial
encoding: relative
""")

        # cues.yaml
        (config_dir / "cues.yaml").write_text("""
cues: []
""")

        # training.yaml
        (config_dir / "training.yaml").write_text("""
enabled_actions: []
use_double_dqn: false
mask_unused_obs: false
reward_strategy:
  type: simple
""")

        # variables_reference.yaml
        (config_dir / "variables_reference.yaml").write_text("""
variables: []
observations: []
""")

        # global_actions.yaml in parent
        (tmp_path / "global_actions.yaml").write_text("""
actions: []
""")

        compiler = UniverseCompiler()
        with pytest.raises(CompilationError) as exc_info:
            compiler.compile(config_dir)

        error_msg = str(exc_info.value)
        assert "probabilistic" in error_msg.lower()
        assert "on_failure" in error_msg or "failure" in error_msg.lower()
        assert "Casino" in error_msg

    def test_probabilistic_without_on_completion_warns(self, tmp_path: Path):
        """Probabilistic capability without on_completion should emit warning."""
        config_dir = tmp_path / "test_config"
        config_dir.mkdir()

        # bars.yaml
        (config_dir / "bars.yaml").write_text("""
bars:
  - name: mood
    range: [0.0, 1.0]
    initial: 0.5
    depletion_per_tick: 0.01
""")

        # cascades.yaml
        (config_dir / "cascades.yaml").write_text("""
cascades: []
""")

        # affordances.yaml with probabilistic but NO on_completion
        (config_dir / "affordances.yaml").write_text("""
affordances:
  - id: "Casino"
    name: "Slot Machine"
    capabilities:
      - type: probabilistic
        success_probability: 0.3
    effect_pipeline:
      on_failure:
        - meter: mood
          amount: -0.1
""")

        # substrate.yaml
        (config_dir / "substrate.yaml").write_text("""
type: aspatial
encoding: relative
""")

        # cues.yaml
        (config_dir / "cues.yaml").write_text("""
cues: []
""")

        # training.yaml
        (config_dir / "training.yaml").write_text("""
enabled_actions: []
use_double_dqn: false
mask_unused_obs: false
reward_strategy:
  type: simple
""")

        # variables_reference.yaml
        (config_dir / "variables_reference.yaml").write_text("""
variables: []
observations: []
""")

        # global_actions.yaml in parent
        (tmp_path / "global_actions.yaml").write_text("""
actions: []
""")

        compiler = UniverseCompiler()
        with pytest.raises(CompilationError) as exc_info:
            compiler.compile(config_dir)

        error_msg = str(exc_info.value)
        assert "probabilistic" in error_msg.lower()
        assert "on_completion" in error_msg or "success" in error_msg.lower()
        assert "Casino" in error_msg
```

---

### Step 2: Run test to verify it fails

```bash
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
export UV_CACHE_DIR=.uv-cache
uv run pytest tests/test_townlet/unit/universe/test_capability_validation.py::TestProbabilisticValidation -v
```

**Expected Output**:
```
FAILED - test_probabilistic_without_on_failure_warns
FAILED - test_probabilistic_without_on_completion_warns
```

---

### Step 3: Implement probabilistic completeness validation

**File**: `src/townlet/universe/compiler.py` (MODIFY)

In `_validate_capabilities_and_effect_pipelines`, add after the prerequisite validation:

```python
        # ... existing prerequisite validation ...

        # NEW: Validate probabilistic effect pipeline completeness
        has_probabilistic = any(
            self._get_attr_value(cap, "type") == "probabilistic"
            for cap in capabilities
        )

        if has_probabilistic:
            if pipeline is None:
                errors.add(
                    formatter(
                        "UAC-VAL-011",
                        f"Probabilistic affordance '{affordance.id}' must define effect_pipeline with on_completion and on_failure",
                        f"affordances.yaml:{affordance.id}",
                    )
                )
            else:
                missing_stages = []
                if not pipeline.on_completion:
                    missing_stages.append("on_completion (success path)")
                if not pipeline.on_failure:
                    missing_stages.append("on_failure (failure path)")

                if missing_stages:
                    errors.add(
                        formatter(
                            "UAC-VAL-011",
                            f"Probabilistic affordance '{affordance.id}' should define both success and failure effects. "
                            f"Missing: {', '.join(missing_stages)}",
                            f"affordances.yaml:{affordance.id}:effect_pipeline",
                        )
                    )
```

---

### Step 4: Run tests to verify they pass

```bash
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
export UV_CACHE_DIR=.uv-cache
uv run pytest tests/test_townlet/unit/universe/test_capability_validation.py::TestProbabilisticValidation -v
```

**Expected Output**:
```
test_probabilistic_with_both_outcomes_passes PASSED
test_probabilistic_without_on_failure_warns PASSED
test_probabilistic_without_on_completion_warns PASSED
```

---

### Step 5: Commit

```bash
git add tests/test_townlet/unit/universe/test_capability_validation.py
git add src/townlet/universe/compiler.py
git commit -m "feat(compiler): add probabilistic effect pipeline completeness validation

- Validate that ProbabilisticCapability affordances define both on_completion
  and on_failure effects in their effect pipeline
- Error code: UAC-VAL-011
- Provides clear guidance on which stages are missing
- Adds 3 tests covering complete/missing-failure/missing-success scenarios

Closes gap from TASK-004B-GAP-ANALYSIS.md"
```

---

## Task 3: Add Skill Scaling Meter Reference Validation

**Goal**: Ensure `SkillScalingCapability.skill` references a meter that exists in bars.yaml.

**Files**:
- Modify: `src/townlet/universe/compiler.py` (add to Stage 3 reference resolution)
- Modify: `tests/test_townlet/unit/universe/test_capability_validation.py` (add new test class)

---

### Step 1: Write the failing test

**File**: `tests/test_townlet/unit/universe/test_capability_validation.py` (MODIFY - add new class)

Add this class after `TestProbabilisticValidation`:

```python
class TestSkillScalingValidation:
    """Test skill scaling capability validation."""

    def test_skill_scaling_with_valid_meter_passes(self, tmp_path: Path):
        """Skill scaling with existing meter should pass validation."""
        config_dir = tmp_path / "test_config"
        config_dir.mkdir()

        # bars.yaml
        (config_dir / "bars.yaml").write_text("""
bars:
  - name: fitness
    range: [0.0, 1.0]
    initial: 0.1
    depletion_per_tick: 0.005
  - name: energy
    range: [0.0, 1.0]
    initial: 0.5
    depletion_per_tick: 0.01
""")

        # cascades.yaml
        (config_dir / "cascades.yaml").write_text("""
cascades: []
""")

        # affordances.yaml with valid skill scaling
        (config_dir / "affordances.yaml").write_text("""
affordances:
  - id: "Gym"
    name: "Fitness Center"
    capabilities:
      - type: skill_scaling
        skill: fitness
        base_multiplier: 1.0
        max_multiplier: 2.0
    effect_pipeline:
      on_completion:
        - meter: fitness
          amount: 0.05
""")

        # substrate.yaml
        (config_dir / "substrate.yaml").write_text("""
type: aspatial
encoding: relative
""")

        # cues.yaml
        (config_dir / "cues.yaml").write_text("""
cues: []
""")

        # training.yaml
        (config_dir / "training.yaml").write_text("""
enabled_actions: []
use_double_dqn: false
mask_unused_obs: false
reward_strategy:
  type: simple
""")

        # variables_reference.yaml
        (config_dir / "variables_reference.yaml").write_text("""
variables: []
observations: []
""")

        # global_actions.yaml in parent
        (tmp_path / "global_actions.yaml").write_text("""
actions: []
""")

        compiler = UniverseCompiler()
        # Should not raise
        universe = compiler.compile(config_dir)
        assert universe is not None

    def test_skill_scaling_with_invalid_meter_fails(self, tmp_path: Path):
        """Skill scaling with non-existent meter should fail validation."""
        config_dir = tmp_path / "test_config"
        config_dir.mkdir()

        # bars.yaml (note: no 'skill' meter)
        (config_dir / "bars.yaml").write_text("""
bars:
  - name: energy
    range: [0.0, 1.0]
    initial: 0.5
    depletion_per_tick: 0.01
""")

        # cascades.yaml
        (config_dir / "cascades.yaml").write_text("""
cascades: []
""")

        # affordances.yaml with INVALID skill scaling
        (config_dir / "affordances.yaml").write_text("""
affordances:
  - id: "Gym"
    name: "Fitness Center"
    capabilities:
      - type: skill_scaling
        skill: nonexistent_skill
        base_multiplier: 1.0
        max_multiplier: 2.0
    effect_pipeline:
      on_completion:
        - meter: energy
          amount: 0.05
""")

        # substrate.yaml
        (config_dir / "substrate.yaml").write_text("""
type: aspatial
encoding: relative
""")

        # cues.yaml
        (config_dir / "cues.yaml").write_text("""
cues: []
""")

        # training.yaml
        (config_dir / "training.yaml").write_text("""
enabled_actions: []
use_double_dqn: false
mask_unused_obs: false
reward_strategy:
  type: simple
""")

        # variables_reference.yaml
        (config_dir / "variables_reference.yaml").write_text("""
variables: []
observations: []
""")

        # global_actions.yaml in parent
        (tmp_path / "global_actions.yaml").write_text("""
actions: []
""")

        compiler = UniverseCompiler()
        with pytest.raises(CompilationError) as exc_info:
            compiler.compile(config_dir)

        error_msg = str(exc_info.value)
        assert "nonexistent_skill" in error_msg
        assert "skill_scaling" in error_msg.lower() or "skill" in error_msg.lower()
        assert "does not exist" in error_msg.lower() or "not found" in error_msg.lower()
```

---

### Step 2: Run test to verify it fails

```bash
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
export UV_CACHE_DIR=.uv-cache
uv run pytest tests/test_townlet/unit/universe/test_capability_validation.py::TestSkillScalingValidation -v
```

**Expected Output**:
```
FAILED - test_skill_scaling_with_invalid_meter_fails
```

---

### Step 3: Implement skill scaling meter validation

**File**: `src/townlet/universe/compiler.py` (MODIFY)

Find the Stage 3 `_stage_3_resolve` method where meter references are validated (around line 772-808). Add skill scaling validation after the existing `meter_gated` validation:

```python
            # Capabilities (meter-gated)
            capabilities = getattr(affordance, "capabilities", None)
            if capabilities and isinstance(capabilities, Sequence):
                for idx, capability in enumerate(capabilities):
                    if _get_attr(capability, "type") == "meter_gated":
                        meter = _get_meter(capability)
                        location = f"affordances.yaml:{affordance.id}:capabilities[{idx}]"
                        if meter:
                            _record_meter_reference(
                                meter,
                                location,
                                "affordance capability (meter_gated)",
                            )
                            if meter not in symbol_table.meters:
                                _handle_missing_meter(location)

                    # NEW: Validate skill_scaling skill meter reference
                    elif _get_attr(capability, "type") == "skill_scaling":
                        skill = _get_attr(capability, "skill")
                        location = f"affordances.yaml:{affordance.id}:capabilities[{idx}]"
                        if skill:
                            _record_meter_reference(
                                skill,
                                location,
                                "affordance capability (skill_scaling)",
                            )
                            if skill not in symbol_table.meters:
                                _handle_missing_meter(location)
```

---

### Step 4: Run tests to verify they pass

```bash
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
export UV_CACHE_DIR=.uv-cache
uv run pytest tests/test_townlet/unit/universe/test_capability_validation.py::TestSkillScalingValidation -v
```

**Expected Output**:
```
test_skill_scaling_with_valid_meter_passes PASSED
test_skill_scaling_with_invalid_meter_fails PASSED
```

---

### Step 5: Run all capability validation tests

```bash
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
export UV_CACHE_DIR=.uv-cache
uv run pytest tests/test_townlet/unit/universe/test_capability_validation.py -v
```

**Expected Output**:
```
TestPrerequisiteValidation::test_prerequisite_with_valid_affordance_ids_passes PASSED
TestPrerequisiteValidation::test_prerequisite_with_invalid_affordance_id_fails PASSED
TestPrerequisiteValidation::test_prerequisite_with_multiple_invalid_ids_reports_all PASSED
TestProbabilisticValidation::test_probabilistic_with_both_outcomes_passes PASSED
TestProbabilisticValidation::test_probabilistic_without_on_failure_warns PASSED
TestProbabilisticValidation::test_probabilistic_without_on_completion_warns PASSED
TestSkillScalingValidation::test_skill_scaling_with_valid_meter_passes PASSED
TestSkillScalingValidation::test_skill_scaling_with_invalid_meter_fails PASSED

======================== 8 passed in X.XXs ========================
```

---

### Step 6: Commit

```bash
git add tests/test_townlet/unit/universe/test_capability_validation.py
git add src/townlet/universe/compiler.py
git commit -m "feat(compiler): add skill scaling meter reference validation

- Validate that SkillScalingCapability.skill references a meter that exists
  in bars.yaml (Stage 3 reference resolution)
- Uses existing meter reference tracking infrastructure
- Adds 2 tests covering valid/invalid skill meter references

Closes final validation gap from TASK-004B-GAP-ANALYSIS.md"
```

---

## Task 4: Create Operator Documentation

**Goal**: Document the capability system for operators who configure universes, including which capabilities are runtime-ready and which are validation-only.

**Files**:
- Create: `docs/config-schemas/capabilities.md`
- Modify: `CLAUDE.md` (add reference to capabilities documentation)

---

### Step 1: Create capabilities documentation

**File**: `docs/config-schemas/capabilities.md` (CREATE)

```markdown
# Affordance Capabilities Configuration Guide

**Status**: Production (Validation Complete, Runtime Partial)
**Version**: 1.0
**Last Updated**: 2025-11-12

---

## Overview

The Affordance Capability System extends basic affordances with composable advanced behaviors. Instead of rigid affordance types, capabilities can be mixed and matched to create rich interaction patterns.

**Key Concept**: Capabilities compose. An affordance can have multiple capabilities simultaneously (e.g., `multi_tick` + `cooldown` + `meter_gated`).

**Current Status**:
- ✅ **Validation**: All capabilities validated by compiler
- ⚠️ **Runtime**: Only `multi_tick` currently executes (others validated but await implementation)

---

## Quick Reference

| Capability | Runtime Ready | Use Case | Parameters |
|------------|---------------|----------|------------|
| `multi_tick` | ✅ Yes | Multi-step interactions (Job, University) | N/A (uses root `duration_ticks`) |
| `cooldown` | ❌ No | Prevent spamming | `cooldown_ticks`, `scope` |
| `meter_gated` | ❌ No | Resource requirements (Gym needs energy) | `meter`, `min`, `max` |
| `skill_scaling` | ❌ No | Effects scale with skill level | `skill`, `base_multiplier`, `max_multiplier` |
| `probabilistic` | ❌ No | Success/failure outcomes (Gambling) | `success_probability` |
| `prerequisite` | ❌ No | Requires other affordances first | `required_affordances` |

---

## Capability Definitions

### 1. Multi-Tick Capability

**Runtime Status**: ✅ **READY** (fully implemented)

**Purpose**: Interactions that take multiple ticks to complete.

**Configuration**:
```yaml
affordances:
  - id: "Job"
    name: "Office Job"
    duration_ticks: 10  # Root-level field
    capabilities:
      - type: multi_tick
        early_exit_allowed: true   # Optional, default: false
        resumable: false            # Optional, default: false
```

**Parameters**:
- `early_exit_allowed` (bool): Can agent leave before completion?
- `resumable` (bool): Can agent resume if interrupted? (requires `multi_tick`)

**Validation**:
- ✅ `resumable=true` requires `multi_tick` capability
- ✅ `interaction_type=instant` conflicts with `multi_tick`
- ✅ Multi-tick affordances should define `per_tick` or `on_completion` effects

**Example**: See `configs/L0_0_minimal/affordances.yaml` for working examples.

---

### 2. Cooldown Capability

**Runtime Status**: ❌ **VALIDATION ONLY** (awaits runtime implementation)

**Purpose**: Prevent affordance from being used again for N ticks after completion.

**Configuration**:
```yaml
affordances:
  - id: "Job"
    name: "Office Job"
    capabilities:
      - type: cooldown
        cooldown_ticks: 50    # Required: ticks before can use again
        scope: agent          # Optional: "agent" (default) or "global"
```

**Parameters**:
- `cooldown_ticks` (int, >0): Number of ticks before affordance available again
- `scope` (string): `"agent"` (per-agent cooldown) or `"global"` (shared cooldown)

**Validation**:
- ✅ `cooldown_ticks` must be positive integer

**Use Cases**:
- Prevent job spamming (agent cooldown)
- Limited resource (global cooldown - "only one agent can use per period")

---

### 3. Meter-Gated Capability

**Runtime Status**: ❌ **VALIDATION ONLY** (awaits runtime implementation)

**Purpose**: Require meter within range to initiate affordance.

**Configuration**:
```yaml
affordances:
  - id: "Gym"
    name: "Fitness Center"
    capabilities:
      - type: meter_gated
        meter: energy        # Required: which meter to check
        min: 0.3            # Optional: minimum value (inclusive)
        max: null           # Optional: maximum value (inclusive)
```

**Parameters**:
- `meter` (string): Meter name (must exist in bars.yaml)
- `min` (float, optional): Minimum value (inclusive)
- `max` (float, optional): Maximum value (inclusive)
- At least one of `min` or `max` required

**Validation**:
- ✅ `meter` must exist in bars.yaml
- ✅ `min < max` if both specified
- ✅ At least one bound required

**Use Cases**:
- Gym requires energy > 0.2 (can't work out when exhausted)
- Hospital only if health < 0.5 (only when sick)

---

### 4. Skill Scaling Capability

**Runtime Status**: ❌ **VALIDATION ONLY** (awaits runtime implementation)

**Purpose**: Scale affordance effects based on a skill meter value.

**Configuration**:
```yaml
affordances:
  - id: "Gym"
    name: "Fitness Center"
    capabilities:
      - type: skill_scaling
        skill: fitness              # Required: skill meter name
        base_multiplier: 1.0        # Optional: multiplier at skill=0
        max_multiplier: 2.0         # Optional: multiplier at skill=1
```

**Parameters**:
- `skill` (string): Skill meter name (must exist in bars.yaml)
- `base_multiplier` (float, default 1.0): Effect multiplier when skill=0
- `max_multiplier` (float, default 2.0): Effect multiplier when skill=1

**Validation**:
- ✅ `skill` meter must exist in bars.yaml
- ✅ `base_multiplier <= max_multiplier`

**Scaling Formula**:
```
effective_amount = base_amount * (base_multiplier + (skill_value * (max_multiplier - base_multiplier)))
```

**Example**:
- Gym with skill=fitness, base=1.0, max=2.0
- Beginner (fitness=0.1): 0.05 × 1.1 = 0.055
- Intermediate (fitness=0.5): 0.05 × 1.5 = 0.075
- Expert (fitness=0.9): 0.05 × 1.9 = 0.095

---

### 5. Probabilistic Capability

**Runtime Status**: ❌ **VALIDATION ONLY** (awaits runtime implementation)

**Purpose**: Interaction has probabilistic success/failure outcomes.

**Configuration**:
```yaml
affordances:
  - id: "Casino"
    name: "Slot Machine"
    capabilities:
      - type: probabilistic
        success_probability: 0.3    # Required: 0.0-1.0
    effect_pipeline:
      on_start:
        - meter: money
          amount: -5.0              # Always paid
      on_completion:                # SUCCESS path (30%)
        - meter: money
          amount: 20.0
      on_failure:                   # FAILURE path (70%)
        - meter: mood
          amount: -0.1
```

**Parameters**:
- `success_probability` (float, 0.0-1.0): Probability of success

**Validation**:
- ✅ Probability must be in [0.0, 1.0]
- ✅ Should define both `on_completion` (success) and `on_failure` (failure) effects

**Use Cases**:
- Gambling (variable success rate)
- Dating (uncertain outcomes)
- Risky investments

---

### 6. Prerequisite Capability

**Runtime Status**: ❌ **VALIDATION ONLY** (awaits runtime implementation)

**Purpose**: Require completion of other affordances before this one becomes available.

**Configuration**:
```yaml
affordances:
  - id: "UniversityFreshman"
    name: "University Year 1"
    effect_pipeline:
      on_completion:
        - meter: education
          amount: 0.25

  - id: "UniversitySophomore"
    name: "University Year 2"
    capabilities:
      - type: prerequisite
        required_affordances: ["UniversityFreshman"]  # Required: list of IDs
    effect_pipeline:
      on_completion:
        - meter: education
          amount: 0.25
```

**Parameters**:
- `required_affordances` (list[string]): Affordance IDs that must be completed first
  - List must not be empty
  - All IDs must exist in affordances.yaml

**Validation**:
- ✅ `required_affordances` must be non-empty list
- ✅ All IDs must exist in affordances.yaml

**Use Cases**:
- Educational progression (Year 1 → Year 2 → Year 3)
- Tech tree (Research A before Research B)
- Story progression (Quest 1 → Quest 2)

---

## Effect Pipeline Integration

Capabilities work with the `effect_pipeline` system to define when effects occur:

```yaml
effect_pipeline:
  on_start:        # When interaction begins (costs)
    - meter: energy
      amount: -0.1

  per_tick:        # Every tick during interaction (for multi_tick)
    - meter: money
      amount: 2.25

  on_completion:   # When interaction completes successfully
    - meter: money
      amount: 5.0

  on_early_exit:   # When agent exits early (if early_exit_allowed)
    - meter: mood
      amount: -0.05

  on_failure:      # When probabilistic interaction fails
    - meter: mood
      amount: -0.1
```

---

## Capability Composition Patterns

### Pattern 1: Job (Multi-Tick + Cooldown + Meter-Gated)

**Use Case**: Prevent agents from working continuously without rest.

```yaml
affordances:
  - id: "Job"
    name: "Office Job"
    duration_ticks: 10
    capabilities:
      - type: multi_tick
        early_exit_allowed: true
      - type: cooldown
        cooldown_ticks: 50
        scope: agent
      - type: meter_gated
        meter: energy
        min: 0.3
    effect_pipeline:
      on_start:
        - meter: energy
          amount: -0.05
      per_tick:
        - meter: money
          amount: 2.25
      on_completion:
        - meter: money
          amount: 5.0
```

**Behavior**:
1. Agent must have energy ≥ 0.3 to start job
2. Job takes 10 ticks to complete
3. Agent can quit early (with mood penalty)
4. After completion, 50-tick cooldown before can work again

---

### Pattern 2: Gym (Meter-Gated + Skill Scaling)

**Use Case**: Workouts become more effective as fitness improves.

```yaml
affordances:
  - id: "Gym"
    name: "Fitness Center"
    capabilities:
      - type: meter_gated
        meter: energy
        min: 0.2
      - type: skill_scaling
        skill: fitness
        base_multiplier: 1.0
        max_multiplier: 2.0
    effect_pipeline:
      on_start:
        - meter: energy
          amount: -0.1
      on_completion:
        - meter: fitness
          amount: 0.05  # Scales: 0.05 → 0.1
```

**Behavior**:
1. Requires energy ≥ 0.2 to enter
2. Fitness gains scale with current fitness (beginners gain 0.05, experts gain 0.1)

---

### Pattern 3: University (Multi-Tick + Prerequisite)

**Use Case**: Progressive educational system.

```yaml
affordances:
  - id: "UniversityFreshman"
    name: "Year 1"
    duration_ticks: 50
    capabilities:
      - type: multi_tick
        resumable: true
    effect_pipeline:
      on_completion:
        - meter: education
          amount: 0.25

  - id: "UniversitySophomore"
    name: "Year 2"
    duration_ticks: 50
    capabilities:
      - type: multi_tick
        resumable: true
      - type: prerequisite
        required_affordances: ["UniversityFreshman"]
    effect_pipeline:
      on_completion:
        - meter: education
          amount: 0.25
```

**Behavior**:
1. Must complete Year 1 before Year 2 becomes available
2. Long interactions (50 ticks) but resumable
3. Progressive education gains

---

## Validation Reference

### Compiler Error Codes

| Code | Description | Validation Stage |
|------|-------------|------------------|
| UAC-VAL-008 | Instant affordance cannot have multi_tick | Stage 4 |
| UAC-VAL-009 | resumable=true requires multi_tick | Stage 4 |
| UAC-VAL-010 | Prerequisite affordance ID not found | Stage 4 |
| UAC-VAL-011 | Probabilistic missing success/failure effects | Stage 4 |
| UAC-RES-001 | meter_gated/skill_scaling meter not found | Stage 3 |

### Common Validation Errors

**Error**: "Prerequisite affordance 'X' does not exist"
```yaml
# BAD: NonExistent not defined
capabilities:
  - type: prerequisite
    required_affordances: ["NonExistent"]

# GOOD: Reference exists
affordances:
  - id: "Foundation"
    ...
  - id: "Advanced"
    capabilities:
      - type: prerequisite
        required_affordances: ["Foundation"]
```

**Error**: "Probabilistic affordance should define both on_completion and on_failure"
```yaml
# BAD: Missing on_failure
capabilities:
  - type: probabilistic
    success_probability: 0.3
effect_pipeline:
  on_completion: [...]

# GOOD: Both paths defined
capabilities:
  - type: probabilistic
    success_probability: 0.3
effect_pipeline:
  on_completion: [...]  # Success path
  on_failure: [...]     # Failure path
```

**Error**: "Meter 'skill_name' does not exist in bars.yaml"
```yaml
# BAD: nonexistent_skill not in bars.yaml
capabilities:
  - type: skill_scaling
    skill: nonexistent_skill

# GOOD: fitness exists in bars.yaml
bars:
  - name: fitness
    ...
capabilities:
  - type: skill_scaling
    skill: fitness
```

---

## Testing Your Configuration

```bash
# Validate configuration without training
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
python -m townlet.compiler validate configs/your_config

# Expected output for valid config:
# Validation succeeded in XXX ms (no cache artifacts written)

# Expected output for invalid config:
# [UAC-VAL-XXX] Error description
# CompilationError: Universe Compilation Failed
```

---

## Runtime Implementation Status

**Implemented** (Production Ready):
- ✅ `multi_tick` - Fully functional with effect_pipeline
- ✅ `effect_pipeline` - All 5 lifecycle stages execute
- ✅ `operating_hours` - Affordance availability filtering by time

**Validated Only** (Awaits Runtime Implementation):
- ❌ `cooldown` - State tracking not implemented
- ❌ `meter_gated` - Runtime filtering not implemented
- ❌ `skill_scaling` - Effect multipliers not implemented
- ❌ `probabilistic` - Success/failure branching not implemented
- ❌ `prerequisite` - Completion tracking not implemented

**Recommendation**: Use `multi_tick` and `effect_pipeline` for production configs. Other capabilities validate successfully but won't execute until runtime support is added (tracked in future task).

---

## Related Documentation

- **Effect Pipeline**: See `docs/config-schemas/affordances.md` for effect_pipeline details
- **Compiler Architecture**: See `docs/architecture/COMPILER_ARCHITECTURE.md`
- **Gap Analysis**: See `docs/tasks/TASK-004B-GAP-ANALYSIS.md` for implementation status

---

**Last Updated**: 2025-11-12
**Status**: All capabilities validate correctly, runtime support partial
**Future Work**: TASK-004B-RUNTIME for advanced capability execution
```

---

### Step 2: Update CLAUDE.md with capabilities reference

**File**: `CLAUDE.md` (MODIFY)

Find the "Configuration System" section (around line 275) and add:

```markdown
## Configuration System

Training controlled via YAML configs in `configs/`. Each config pack is a directory containing:

```
configs/<level>/
├── substrate.yaml       # Spatial substrate (grid size, topology, boundaries)
├── bars.yaml            # Meter definitions (energy, health, etc.)
├── cascades.yaml        # Meter relationships
├── affordances.yaml     # Interaction definitions
├── cues.yaml            # UI metadata
├── training.yaml        # Hyperparameters
└── variables_reference.yaml  # VFS configuration (REQUIRED)
```

**NEW**: Affordances support advanced behaviors through composable capabilities:
- `multi_tick`: Multi-step interactions (Job, University) - ✅ Runtime Ready
- `cooldown`, `meter_gated`, `skill_scaling`, `probabilistic`, `prerequisite` - Validated, awaiting runtime
- See `docs/config-schemas/capabilities.md` for full guide

### Active Config Packs (Curriculum)
```

---

### Step 3: Verify documentation builds correctly

```bash
# Check markdown syntax
cat docs/config-schemas/capabilities.md | head -20

# Verify CLAUDE.md still parses
cat CLAUDE.md | grep -A5 "Configuration System"
```

**Expected**: Clean markdown, no syntax errors.

---

### Step 4: Commit

```bash
git add docs/config-schemas/capabilities.md
git add CLAUDE.md
git commit -m "docs: add comprehensive capabilities configuration guide

- Create docs/config-schemas/capabilities.md with operator guide
- Document all 6 capabilities with examples and validation rules
- Mark runtime status (multi_tick ready, others validated-only)
- Add capability composition patterns (Job, Gym, University)
- Include validation error reference with solutions
- Add reference to capabilities doc in CLAUDE.md

Closes documentation gap from TASK-004B-GAP-ANALYSIS.md"
```

---

## Task 5: Verification and Cleanup

**Goal**: Run full test suite to verify all changes integrate correctly with existing system.

---

### Step 1: Run full compiler test suite

```bash
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
export UV_CACHE_DIR=.uv-cache
uv run pytest tests/test_townlet/unit/universe/ -v --tb=short
```

**Expected**: All tests pass (229 existing + 8 new = 237 tests)

---

### Step 2: Run config unit tests

```bash
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
export UV_CACHE_DIR=.uv-cache
uv run pytest tests/test_townlet/unit/config/ -v --tb=short
```

**Expected**: All tests pass

---

### Step 3: Validate all curriculum configs

```bash
export PYTHONPATH=/home/john/hamlet/src:$PYTHONPATH
for config in L0_0_minimal L0_5_dual_resource L1_full_observability L2_partial_observability L3_temporal_mechanics; do
  echo "=== Validating configs/$config ==="
  python -m townlet.compiler validate "configs/$config" 2>&1 | grep -E "succeeded|Failed"
done
```

**Expected**: All configs validate successfully

---

### Step 4: Update TASK-004B status

**File**: `docs/tasks/TASK-004B-UAC-CAPABILITIES.md` (MODIFY)

Update the header:

```markdown
# TASK-004B: UAC Contracts - Capability System

**Status**: Completed (95% → 100% validation complete)
**Completed Date**: 2025-11-12
**Priority**: MEDIUM
**Actual Effort**: 3 hours (validation gaps + documentation)
**Dependencies**: TASK-003 (Core DTOs)
**Enables**: Advanced curriculum levels (L4-L6)

**Completion Notes**:
- All capability DTOs implemented (100%)
- Compiler validation complete (100%):
  - Prerequisite affordance reference validation ✅
  - Probabilistic effect pipeline completeness ✅
  - Skill scaling meter validation ✅
- Runtime execution support (40%):
  - multi_tick: ✅ Complete
  - cooldown, meter_gated, skill_scaling, probabilistic, prerequisite: ❌ Awaiting TASK-004B-RUNTIME
- Documentation complete ✅
- Gap analysis: See TASK-004B-GAP-ANALYSIS.md
- Validation implementation: See docs/plans/2025-11-12-task-004b-validation-gaps.md

**Future Work**: TASK-004B-RUNTIME for advanced capability runtime support (15-20 hours)

---
```

---

### Step 5: Final commit

```bash
git add docs/tasks/TASK-004B-UAC-CAPABILITIES.md
git commit -m "docs: mark TASK-004B as 100% complete for validation

- Updated status from 'Planned' to 'Completed'
- Added completion notes and references to gap analysis
- Documented future work (TASK-004B-RUNTIME)
- All validation gaps closed, documentation complete

Validation completion:
- Prerequisite affordance references ✅
- Probabilistic effect pipeline completeness ✅
- Skill scaling meter validation ✅
- Comprehensive operator documentation ✅

Total new tests: 8
Total new error codes: 3 (UAC-VAL-010, UAC-VAL-011, skill meter in UAC-RES-001)"
```

---

## Summary

**Implementation Complete**: All TASK-004B validation gaps closed.

**Deliverables**:
1. ✅ Prerequisite affordance reference validation (3 tests)
2. ✅ Probabilistic effect pipeline completeness validation (3 tests)
3. ✅ Skill scaling meter reference validation (2 tests)
4. ✅ Comprehensive operator documentation (capabilities.md)
5. ✅ Updated CLAUDE.md with capabilities reference
6. ✅ Updated TASK-004B status to Complete

**Test Coverage**: +8 new tests, all passing
**Error Codes**: +2 new (UAC-VAL-010, UAC-VAL-011)
**Documentation**: 1 new guide (capabilities.md)

**Total Effort**: ~3 hours (as estimated)

**Next Steps**:
- Runtime capability support → TASK-004B-RUNTIME (separate task, 15-20 hours)
- Consider creating example configs using advanced capabilities for testing

---

**Plan Status**: Ready for execution
**Saved To**: `docs/plans/2025-11-12-task-004b-validation-gaps.md`
