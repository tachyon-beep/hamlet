# Affordances Configuration

## Capability Validation Rules

The Universe Compiler validates affordance capabilities to ensure configuration correctness at compile time.

### Prerequisite Capabilities (UAC-VAL-010)

**Rule**: `PrerequisiteCapability.required_affordances` must only reference affordances that exist in `affordances.yaml`.

**Example - Valid**:
```yaml
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
        required_affordances: ["Foundation"]  # Valid - Foundation exists
    effect_pipeline:
      on_completion:
        - meter: energy
          amount: 0.2
```

**Example - Invalid**:
```yaml
affordances:
  - id: "Advanced"
    name: "Advanced Course"
    capabilities:
      - type: prerequisite
        required_affordances: ["NonExistent"]  # ERROR: NonExistent does not exist
    effect_pipeline:
      on_completion:
        - meter: energy
          amount: 0.2
```

**Error Message**: `Prerequisite affordance 'NonExistent' does not exist in affordances.yaml`

### Probabilistic Capabilities (UAC-VAL-011)

**Rule**: `ProbabilisticCapability` affordances must define both `on_completion` (success path) and `on_failure` (failure path) in their effect pipeline.

**Example - Valid**:
```yaml
affordances:
  - id: "Casino"
    name: "Slot Machine"
    capabilities:
      - type: probabilistic
        success_probability: 0.3
    effect_pipeline:
      on_completion:  # Success path
        - meter: money
          amount: 0.5
      on_failure:     # Failure path
        - meter: money
          amount: -0.1
```

**Example - Invalid**:
```yaml
affordances:
  - id: "Casino"
    name: "Slot Machine"
    capabilities:
      - type: probabilistic
        success_probability: 0.3
    effect_pipeline:
      on_completion:
        - meter: money
          amount: 0.5
      # ERROR: Missing on_failure - what happens when the 70% failure case occurs?
```

**Error Message**: `Probabilistic affordance 'Casino' should define both success and failure effects. Missing: on_failure (failure path)`

**Rationale**: Probabilistic affordances have two distinct outcomes. Both must be explicitly defined to avoid ambiguity and ensure reproducible behavior.

### Skill Scaling Capabilities (UAC-VAL-012)

**Rule**: `SkillScalingCapability.skill` must reference an existing meter defined in `bars.yaml`.

**Example - Valid**:
```yaml
# bars.yaml
bars:
  - name: fitness
    range: [0.0, 1.0]
    # ... other fields

# affordances.yaml
affordances:
  - id: "Training"
    name: "Gym Training"
    capabilities:
      - type: skill_scaling
        skill: fitness  # Valid - fitness meter exists
        base_multiplier: 0.5
        max_multiplier: 2.0
    effect_pipeline:
      on_completion:
        - meter: fitness
          amount: 0.1
```

**Example - Invalid**:
```yaml
affordances:
  - id: "Training"
    name: "Gym Training"
    capabilities:
      - type: skill_scaling
        skill: nonexistent_meter  # ERROR: No such meter
        base_multiplier: 0.5
        max_multiplier: 2.0
    effect_pipeline:
      on_completion:
        - meter: fitness
          amount: 0.1
```

**Error Message**: `Skill scaling capability references non-existent meter 'nonexistent_meter'. Valid meters: ['energy', 'fitness', 'health', ...]`

**Rationale**: Skill scaling modifies effects based on a meter's value. The meter must exist for the scaling to function.

## See Also

- [Universe Compiler Architecture](../architecture/COMPILER_ARCHITECTURE.md) - Full compiler pipeline documentation
- [Training Configuration](./training.md) - Training hyperparameters and reward strategies
- [Variables Reference](./variables.md) - VFS configuration guide
