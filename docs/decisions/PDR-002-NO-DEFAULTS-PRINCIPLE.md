# PDR-002: No-Defaults Principle for Config-Driven Systems

**Status**: Accepted ✅
**Type**: Architecture & Configuration
**Priority**: HIGH
**Date Proposed**: 2025-11-04
**Date Adopted**: 2025-11-04
**Decision Makers**: Architecture Team
**Affects**: All config-driven systems (UNIVERSE_AS_CODE, BRAIN_AS_CODE)
**Enforcement**: Custom linter (scripts/no_defaults_lint.py) + CI/CD

**Keywords**: config, defaults, UNIVERSE_AS_CODE, BRAIN_AS_CODE, reproducibility, transparency, fail-fast, YAML, no-magic
**Subsystems**: Environment, Population, Curriculum, Exploration, Training (all config-driven)
**Breaking Changes**: No (requires explicit config values, but doesn't change existing valid configs)
**Supersedes**: N/A

---

## AI-Friendly Summary (Skim This First!)

**What**: All behavioral parameters in config files must be explicitly specified. No implicit defaults allowed in code. Incomplete configs fail with clear error messages instead of silently using hidden default values.

**Why**: UNIVERSE_AS_CODE and BRAIN_AS_CODE philosophies require **complete transparency**. Hidden defaults create non-reproducible configs, confuse operators about actual behavior, and break silently when code defaults change. Explicit-only configs ensure reproducibility and operator accountability.

**Scope**: Applies to **ALL UAC/BAC parameters** - no exemptions. All `.yaml` config files (bars, affordances, cascades, cues, actions, training) must be complete. Defaults only allowed for infrastructure/runtime concerns (device, port, logging) via explicit whitelist. Enforced via custom linter (`scripts/no_defaults_lint.py`).

**Quick Assessment**:

- **Problem**: Hidden defaults → operator doesn't know actual values → non-reproducible experiments
- **Solution**: Fail-fast on missing UAC/BAC values → operator forced to specify → configs are self-documenting
- **Enforcement**: Custom linter + whitelist (infrastructure ONLY, not UAC/BAC) + CI checks
- **UAC/BAC Rule**: ZERO exemptions for universe/brain parameters - all must be explicit in YAML
- **Infrastructure Rule**: Python runtime (device, port) can have defaults if whitelisted
- **Impact**: Higher config verbosity, but complete reproducibility and transparency

**Decision Point**: If you're not writing config-loading code or creating new configurable parameters, STOP READING HERE. If you're adding new parameters to universe/brain systems, continue reading for compliance requirements.

---

## Context

### Current Situation

Prior to this policy, Python code could define default values for configuration parameters, leading to "hidden behavior" where the actual system behavior differs from what's visible in the config file.

**Pain Points**:

- **Non-reproducible configs**: Old config files produce different behavior when code defaults change
- **Operator confusion**: "Why is my agent using epsilon=0.1 when I didn't specify it?"
- **Silent breakage**: Code refactoring changes default → old configs behave differently without warning
- **Config incompleteness**: Config files don't document all active parameters
- **"Works on my machine"**: Configs work for original author but fail for others due to implicit assumptions

**Historical Evidence**:

- **UNIVERSE_AS_CODE design docs** (2025-10): Identified need for "zero magic, zero defaults" in universe configuration
- **Config evolution pain**: Multiple instances where changing a code default broke existing training runs
- **Onboarding confusion**: New contributors unsure what parameters are actually being used
- **Research reproducibility**: Experiments difficult to replicate when configs incomplete

### Why Now?

**Catalyzing Event**:

The development of UNIVERSE_AS_CODE and BRAIN_AS_CODE philosophies establishes config-driven design as a **core architectural principle**. These philosophies state:

1. **UNIVERSE_AS_CODE**: "Everything configurable. Schema enforced mercilessly."
2. **BRAIN_AS_CODE**: "Agent architecture and policy configuration in YAML."

For these to be meaningful, configs must be **complete** and **self-documenting**. Hidden defaults undermine both principles by creating invisible behavior not captured in configs.

**Strategic Importance**:

This policy enables:

- **Pedagogical value**: Students can read complete config files to understand system behavior
- **Research reproducibility**: Experiments fully specified in version-controlled YAML
- **Domain agnostic**: Same framework can model villages, factories, trading bots (no hardcoded assumptions)
- **Operator accountability**: Operators explicitly choose all behavioral parameters

---

## Policy Decision

### Core Requirement

**All behavioral parameters must be explicitly specified in configuration files. Functions that load configs must not provide default values for parameters that affect system behavior.**

### Detailed Requirements

**Layer 1: Code - No Implicit Defaults**

Python functions that load configuration values must **not** use default values in their signatures.

**Checks/Actions**:

1. Function signatures must not have `= value` defaults for behavioral parameters
2. Config loading must use `.get()` without defaults or access keys directly (triggers KeyError)
3. Missing required parameters must raise `ValueError` with clear error message
4. Error messages must include example of correct config structure

**Behavior**: Code fails immediately (fail-fast) when required config value is missing. No silent fallback to hardcoded defaults.

**Example (VIOLATION)**:

```python
# ❌ BAD - hidden default
def __init__(self, config: dict):
    self.epsilon = config.get("epsilon", 0.1)  # Hidden default!
```

**Example (COMPLIANT)**:

```python
# ✅ GOOD - explicit required
def __init__(self, config: dict):
    if "epsilon" not in config:
        raise ValueError(
            "Missing required parameter 'epsilon' in config. "
            "Add 'epsilon: 0.1' to your training.yaml"
        )
    self.epsilon = config["epsilon"]
```

**Layer 2: Linter - Automated Detection**

Custom linter (`scripts/no_defaults_lint.py`) scans Python code for violations.

**Checks/Actions**:

1. Scans all Python files in `src/townlet/` for function signatures
2. Detects `def foo(config, param=default_value)` patterns
3. Reports violations with file:line:function:param
4. Checks against whitelist (`.defaults-whitelist.txt`) for exemptions
5. Fails CI if unauthorized defaults found

**Behavior**: Linter runs in CI and blocks PRs if violations detected.

**Configuration**: `.defaults-whitelist.txt` for legitimate exemptions

**Layer 3: Config Validation - Schema Enforcement**

Configuration files use Pydantic DTOs for validation (planned in TASK-001).

**Checks/Actions**:

1. Pydantic models define required fields (no `Optional` or `Field(default=...)`)
2. YAML files validated against schemas at load time
3. Missing fields trigger validation errors with field name and expected type
4. Schemas serve as documentation of required parameters

**Behavior**: Config loading fails fast with clear error messages.

**Status**: Partial (DTO-based validation planned in TASK-001)

### Exceptions

**CRITICAL: All UAC/BAC parameters are MANDATORY - NO exemptions**

Any parameter related to UNIVERSE_AS_CODE or BRAIN_AS_CODE must be explicitly configured:

- ❌ Environment mechanics (grid size, energy costs, meter depletion rates, affordance properties)
- ❌ Agent architecture (network layers, activation functions, learning rates)
- ❌ Training dynamics (exploration parameters, replay buffer size, batch size)
- ❌ Curriculum parameters (stage durations, difficulty thresholds)
- ❌ Reward shaping (weights, baselines, bonuses)
- ❌ **ALL parameters in bars.yaml, affordances.yaml, cascades.yaml, actions.yaml, training.yaml**

**When defaults ARE allowed** (must be whitelisted - ONLY for non-UAC/BAC):

**1. Python Runtime Environment**:

- `device="cpu"` - Hardware selection (not algorithm behavior)
- `dtype=torch.float32` - Precision (performance, not behavior)
- `seed=None` - Random seed (reproducibility, not specification)

**2. System Infrastructure**:

- `port=8080` - Network port for servers
- `host="localhost"` - Network host
- `checkpoint_dir="checkpoints"` - File system paths
- `log_dir="logs"` - Logging directory

**3. Performance Tuning** (does NOT affect algorithm correctness):

- `num_workers=4` - Parallelization degree
- `pin_memory=True` - CUDA optimization
- `prefetch_factor=2` - Data loading optimization

**4. Development/Debug** (operational, not behavioral):

- `logging_level=INFO` - Verbosity of logs
- `debug_mode=False` - Debug instrumentation
- `profiling_enabled=False` - Performance profiling

**5. Computed/Derived Values** (deterministic from explicit params):

- `observation_dim` - Calculated from `grid_size` + meter count + affordance vocabulary
- `action_dim` - Derived from `actions.yaml` definitions
- Derived metrics: `total_capacity = sum(components)`

**How to request exception**:

1. **Verify it's NOT UAC/BAC**: Parameter must be infrastructure/runtime, not universe/brain mechanics
2. **Add to whitelist**: `echo "src/path/file.py:function:param  # Justification" >> .defaults-whitelist.txt`
3. **Justification must explain**: Why this is infrastructure/runtime, not behavior
4. **For systemic exemptions**: Raise in architecture discussion

**Examples of REJECTED exemption requests**:

❌ `move_energy_cost=0.005` - "Usually 0.005" → **REJECTED** - This is UAC, must be explicit
❌ `epsilon_decay=0.995` - "Standard value" → **REJECTED** - This is BAC, must be explicit
❌ `display_name="Bed"` - "Metadata only" → **REJECTED** - Part of affordances.yaml UAC definition
❌ `required_ticks=1` - "Most affordances are instant" → **REJECTED** - UAC interaction mechanic

**Examples of ACCEPTED exemptions**:

✅ `device="cpu"` - Python runtime, not algorithm specification
✅ `num_workers=4` - Performance tuning, doesn't change algorithm behavior
✅ `checkpoint_dir="checkpoints"` - File system path, infrastructure concern

---

## Rationale

### Benefits

**1. Reproducibility**

**Explanation**: Config files fully specify system behavior. Loading same config always produces same behavior regardless of code version (assuming no algorithm bugs).

**Evidence**:

- Old configs remain valid even when code evolves
- Experiments can be reproduced from git-committed YAML files
- No "works on my machine" config bugs

**Example**:

```yaml
# L0_0_minimal/training.yaml - COMPLETE specification
exploration:
  epsilon_start: 1.0      # Explicitly set, not default
  epsilon_decay: 0.99     # Explicitly set, not default
  epsilon_min: 0.01       # Explicitly set, not default
```

Anyone can load this config and get identical epsilon schedule, regardless of code defaults.

**2. Self-Documenting Configs**

**Explanation**: Reading config file shows all active parameters. No need to read Python code to understand what values are being used.

**Evidence**:

- New users can understand system behavior by reading YAML
- Config files serve as documentation
- Pedagogical value: students learn by editing configs

**Example**:
Instructor shows student `L1_full_observability/training.yaml` - all hyperparameters visible, can explain learning dynamics without looking at code.

**3. Operator Accountability**

**Explanation**: Operators must consciously choose all parameter values. No accidental reliance on "whatever the code defaults to."

**Evidence**:

- Forces thoughtful parameter selection
- Prevents "I didn't know that parameter existed" surprises
- Clear responsibility: operator specified these values

**4. Domain Agnosticism**

**Explanation**: Framework can model any universe (villages, factories, trading) without hardcoded assumptions.

**Evidence**:

- Action space defined in actions.yaml (not hardcoded UP/DOWN/LEFT/RIGHT)
- Affordances defined in affordances.yaml (not hardcoded Bed/Job/Meal)
- Grid size, meters, cascades all configurable

**5. Fail-Fast Error Detection**

**Explanation**: Missing parameters trigger immediate clear errors, not silent bugs hours later.

**Evidence**:

```python
# Config missing 'epsilon_start'
ValueError: Missing required parameter 'epsilon_start' in config.
Add 'epsilon_start: 1.0' to your training.yaml
```

Better than silent fallback to 0.1, then confusion when exploration behavior differs from expectations.

### Case Studies

**Case Study 1: Epsilon Decay Hidden Default**

**Scenario**: Researcher trains agent on L0 config, then shares config with colleague.

**What happened (before policy)**:

- Original config: `epsilon_start: 1.0, epsilon_min: 0.01` (missing epsilon_decay)
- Code default: `epsilon_decay = 0.995`
- Six months later, code refactored, default changed to `0.99` for faster decay
- Colleague loads old config → gets different behavior → results don't reproduce

**Root Cause**: Hidden default in code, not captured in config file.

**How policy prevents this**:

- Linter detects `def __init__(self, config, epsilon_decay=0.995)`
- Forces developer to remove default, add validation
- Config must explicitly specify `epsilon_decay: 0.995`
- Old config remains valid because value is in YAML, not code

**Outcome**: Configs reproducible across code versions.

**Case Study 2: Move Energy Cost Confusion**

**Scenario**: Student trains agent, sees unexpected behavior (agent avoiding movement).

**What happened (before policy)**:

- Student's config: `grid_size: 8, partial_observability: false`
- Missing: `move_energy_cost`
- Code default: `move_energy_cost = 0.01` (high cost)
- Student expects low cost (wants exploration), doesn't know default exists
- Agent learns to minimize movement due to high energy cost

**Root Cause**: Hidden default creates behavior mismatch with student expectations.

**How policy prevents this**:

- Loading config fails: `ValueError: Missing required parameter 'move_energy_cost'`
- Error message shows example: `"Add 'move_energy_cost: 0.005' to environment config"`
- Student forced to explicitly choose value
- Now aware of parameter's existence and impact

**Outcome**: Student learns about energy mechanics, makes informed choice.

---

## Consequences

### Positive

1. **Config reproducibility** - Old configs remain valid across code versions
2. **Self-documenting** - All active parameters visible in YAML files
3. **Pedagogical clarity** - Students learn by reading complete configs
4. **Research integrity** - Experiments fully specified and reproducible
5. **Fail-fast errors** - Missing params caught immediately with clear messages
6. **Domain flexibility** - No hardcoded assumptions, any universe configurable

### Negative

1. **Verbose configs** - Must specify every parameter explicitly
2. **Higher barrier to entry** - New users must fill out complete configs
3. **Repetitive** - Same values repeated across similar configs
4. **Refactoring overhead** - Adding new params requires updating all configs
5. **Linter maintenance** - Whitelist must be maintained as codebase evolves

### Mitigation Strategies

**For verbose configs**:

- Provide template configs for common use cases (L0, L1, L2 templates)
- Document parameter meanings in schema comments
- Use YAML anchors/aliases for shared values across files

**For barrier to entry**:

- Clear error messages with examples: `"Add 'param: value' to config"`
- Comprehensive example configs in `configs/` directory
- Documentation explaining each parameter's purpose and typical values

**For repetition**:

- YAML includes/merges for shared sections (future enhancement)
- Config inheritance: `base: L1_full_observability` (future enhancement)
- Template system for generating config families (future enhancement)

**For refactoring overhead**:

- Migration scripts for adding new required parameters
- Batch update tools: `python scripts/add_param_to_configs.py --param epsilon_decay --value 0.995`
- Clear changelog when new params added

**For linter maintenance**:

- Whitelist includes comments explaining each exemption
- Regular review of whitelist in quarterly architecture reviews
- Document whitelist criteria in this policy

---

## Implementation

### Phase 1: Linter Infrastructure (Completed)

- ✅ Created `scripts/no_defaults_lint.py` custom linter
- ✅ Created `.defaults-whitelist.txt` exemption list
- ✅ Added linter to CI/CD workflow (`.github/workflows/lint.yml`)
- ✅ Documented linter usage in `docs/development/LINT_ENFORCEMENT.md`

**Owner**: Development Team

### Phase 2: Current Code Compliance (Completed)

- ✅ Scanned existing codebase for violations
- ✅ Whitelisted legitimate exemptions (metadata, optional features)
- ✅ Fixed unauthorized defaults in config-loading code
- ✅ Verified all example configs are complete

**Owner**: Development Team

### Phase 3: Documentation (Completed)

- ✅ Created `docs/decisions/PDR-002-NO-DEFAULTS-PRINCIPLE.md` (this document)
- ✅ Updated `CLAUDE.md` with no-defaults principle
- ✅ Added no-defaults guidance to config documentation

**Owner**: Development Team

### Phase 4: Schema Validation (Planned - TASK-001)

**To implement Pydantic DTO-based validation**:

1. Define DTOs for each config type (bars, affordances, training, etc.)
2. Mark all behavioral fields as required (no `Optional`, no `Field(default=...)`)
3. Add schema validation at config load time
4. Generate clear validation errors with field names and expected types
5. Update all configs to pass schema validation

**Timeline**: See TASK-001 for detailed timeline

**Owner**: Architecture Team

---

## Compliance

### How to Comply

**When adding new configurable parameters**:

1. **Do NOT add default value in function signature**:

   ```python
   # ❌ WRONG
   def __init__(self, config: dict, new_param=0.5):

   # ✅ CORRECT
   def __init__(self, config: dict):
       if "new_param" not in config:
           raise ValueError("Missing 'new_param'. Add to config: new_param: 0.5")
       self.new_param = config["new_param"]
   ```

2. **Update ALL example configs** with new parameter:

   ```bash
   # Add to all relevant configs
   for config in configs/*/training.yaml; do
       echo "  new_param: 0.5" >> $config
   done
   ```

3. **Update schema documentation**:
   - Add parameter to config template
   - Document parameter meaning and typical values
   - Note which configs require it

4. **Run linter to verify compliance**:

   ```bash
   python scripts/no_defaults_lint.py src/townlet/ --whitelist .defaults-whitelist.txt
   ```

**When parameter is truly optional** (metadata, visualization):

1. Add to whitelist with justification:

   ```bash
   echo "src/townlet/demo/renderer.py:__init__:color_scheme  # Optional visualization preference" >> .defaults-whitelist.txt
   ```

2. Verify linter accepts it:

   ```bash
   python scripts/no_defaults_lint.py src/townlet/ --whitelist .defaults-whitelist.txt
   ```

### Verification

**Check for violations**:

```bash
# Run linter
python scripts/no_defaults_lint.py src/townlet/ --whitelist .defaults-whitelist.txt

# Expected output (compliant)
✅ No unauthorized defaults found

# Example output (violation)
❌ Unauthorized default found:
  src/townlet/population/vectorized.py:123:__init__:learning_rate
  Default value: 0.001
  Add to config or whitelist with justification
```

**Verify config completeness**:

```bash
# Load config and check for missing params
uv run python -c "
from townlet.demo.runner import load_config
config = load_config('configs/L0_0_minimal')
print('Config loaded successfully - all required params present')
"
```

**Check CI status**:

- Lint workflow includes no-defaults check
- Must pass before PR merge

### Troubleshooting

**Problem 1: Linter reports false positive**

**Symptoms**: Parameter flagged as violation but is actually metadata/optional

**Solution**:

```bash
# Add to whitelist with justification
echo "src/path/file.py:function:param  # Justification here" >> .defaults-whitelist.txt

# Verify fix
python scripts/no_defaults_lint.py src/townlet/ --whitelist .defaults-whitelist.txt
```

**Problem 2: Need to add new required parameter to existing configs**

**Symptoms**: New parameter needed, but 10+ configs need updating

**Solution**:

```bash
# Create migration script
cat > scripts/add_new_param.sh << 'EOF'
#!/bin/bash
for config in configs/*/training.yaml; do
    if ! grep -q "new_param:" "$config"; then
        echo "  new_param: 0.5  # Default for this config type" >> "$config"
    fi
done
EOF

chmod +x scripts/add_new_param.sh
./scripts/add_new_param.sh
```

**Problem 3: Parameter should be computed, not configured**

**Symptoms**: Value is derived from other params (e.g., observation_dim = grid_size² + 8 + 15 + 4)

**Solution**:

```python
# ✅ CORRECT - compute from other configs
def __init__(self, config: dict):
    grid_size = config["grid_size"]
    self.observation_dim = grid_size * grid_size + 8 + 15 + 4  # Computed
```

Computed values don't need to be in config - they're deterministic functions of configured values.

**Problem 4: Config loading fails with "Missing parameter" but parameter exists**

**Symptoms**: KeyError or ValueError despite parameter being in YAML

**Solution**:

```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/L0_0_minimal/training.yaml'))"

# Check parameter path (nested dicts)
python -c "
import yaml
config = yaml.safe_load(open('configs/L0_0_minimal/training.yaml'))
print(config['exploration']['epsilon_start'])  # Verify nesting
"

# Common issue: wrong nesting level
# ❌ epsilon_start: 1.0  (at root)
# ✅ exploration:
#      epsilon_start: 1.0  (nested under exploration)
```

---

## Alternatives Considered

### Alternative 1: Defaults with Deprecation Warnings

**Description**: Allow defaults but warn when used, deprecate over time.

**Pros**:

- ✅ Gradual migration path
- ✅ Backward compatible
- ✅ Lower initial friction

**Cons**:

- ❌ Warnings often ignored
- ❌ Still creates non-reproducible configs during deprecation period
- ❌ Complexity: must track which defaults are deprecated
- ❌ No clear timeline for when all defaults removed

**Why Rejected**: Soft enforcement doesn't achieve reproducibility goals. Historical evidence shows warnings accumulate without action (see PDR-001 mypy example: 0 → 60+ errors).

### Alternative 2: Two-Tier Configs (Required + Optional)

**Description**: Separate required params (no defaults) from optional params (with defaults).

**Pros**:

- ✅ Balances strictness with usability
- ✅ Core behavior reproducible, convenience features flexible
- ✅ Lower config verbosity

**Cons**:

- ❌ Ambiguous boundary: what's "core" vs "optional"?
- ❌ Still creates partial non-reproducibility (optional params change)
- ❌ Complexity: different validation rules for different param types
- ❌ Confusion: "Do I need to specify this or not?"

**Why Rejected**: Ambiguity undermines reproducibility goal. Better to have one clear rule: "specify everything behavioral."

### Alternative 3: Schema-Enforced Defaults (Defaults in Schema, Not Code)

**Description**: Define defaults in YAML schema files, load from there instead of code.

**Pros**:

- ✅ Defaults documented and version-controlled
- ✅ Can update defaults without code changes
- ✅ Single source of truth

**Cons**:

- ❌ Still hidden from operator (unless they read schema)
- ❌ Config file not self-contained (depends on schema version)
- ❌ Reproducibility still depends on schema version matching
- ❌ Adds indirection (config → schema → actual value)

**Why Rejected**: Doesn't solve core problem (hidden behavior). Config should be complete and self-contained.

### Alternative 4: Automatic Config Generation from Defaults

**Description**: Tool generates complete config from code defaults, operator edits from there.

**Pros**:

- ✅ Convenient config creation
- ✅ All params documented
- ✅ Generated config is complete

**Cons**:

- ❌ Doesn't prevent defaults in code (linter still needed)
- ❌ Generated configs can get out of sync with code
- ❌ Adds tooling complexity
- ❌ Doesn't address core issue (defaults in code create non-reproducibility)

**Why Rejected**: Treats symptom (incomplete configs) rather than cause (defaults in code). Better to eliminate defaults entirely.

---

## Success Metrics

### Quantitative

**Metric 1: Unauthorized Defaults Count**

- **Baseline**: Unknown (linter just implemented)
- **Target**: 0 unauthorized defaults in src/townlet/
- **Measurement**: `python scripts/no_defaults_lint.py src/townlet/ --whitelist .defaults-whitelist.txt | grep "unauthorized"`
- **Current**: ✅ 0 unauthorized defaults

**Metric 2: Whitelist Size**

- **Baseline**: ~50 entries (initial whitelist)
- **Target**: <100 entries (bounded)
- **Measurement**: `wc -l .defaults-whitelist.txt`
- **Current**: Establishing baseline

**Metric 3: Config Completeness**

- **Baseline**: Unknown (not measured before)
- **Target**: 100% of example configs load without errors
- **Measurement**: Test suite that loads all configs in `configs/`
- **Current**: ✅ All example configs complete

**Metric 4: Config Reproducibility**

- **Baseline**: Unknown
- **Target**: 0 configs break when loading across code versions (assuming no algorithm bugs)
- **Measurement**: Quarterly test loading old configs with new code
- **Current**: Establishing baseline

### Qualitative

**Goal 1**: New users can understand system behavior by reading YAML alone (without reading Python)

**Goal 2**: Researchers report high confidence in experiment reproducibility (survey feedback)

**Goal 3**: "Works on my machine" config bugs eliminated (incident reports)

**Goal 4**: Pedagogical effectiveness: students learn faster via self-documenting configs (instructor feedback)

---

## Review Schedule

**Frequency**: Quarterly (every 3 months)

**Next Review**: 2026-02-04

**Review Criteria**:

- Review whitelist growth (is it bounded or growing unbounded?)
- Assess false positive rate (too many legitimate defaults being flagged?)
- Check config verbosity complaints (are configs too verbose in practice?)
- Evaluate reproducibility success (any configs broken by code changes?)
- Consider tooling improvements (auto-migration scripts, config inheritance)
- Update exemption criteria if patterns emerge

**Owner**: Architecture Team

---

## References

### Documentation

- **Implementation Guide**: `scripts/README-no-defaults-lint.md` (linter usage)
- **Related Policies**: PDR-001 (Lint Enforcement - includes no-defaults check)
- **Architecture Docs**: `docs/architecture/UNIVERSE_AS_CODE.md` (philosophy)
- **Task Planning**: `docs/tasks/TASK-001-UAC-CONTRACTS.md` (DTO-based validation)

### Tools & Automation

- **Config Files**:
  - `scripts/no_defaults_lint.py` (custom linter)
  - `.defaults-whitelist.txt` (exemption list)
  - `.github/workflows/lint.yml` (CI enforcement)
  - `configs/*/training.yaml` (example complete configs)
- **CI/CD Workflows**: `.github/workflows/lint.yml` (line 59-67)
- **Scripts**: `scripts/no_defaults_lint.py`

### External References

- **Twelve-Factor App**: https://12factor.net/config (config principles)
- **Pydantic Validation**: https://docs.pydantic.dev (DTO schemas)
- **YAML Specification**: https://yaml.org/spec/1.2.2/ (config format)
- **Fail-Fast Principle**: Martin Fowler on fail-fast design patterns

---

## Appendix

### Examples of Compliant vs Non-Compliant Code

**Example 1: Environment Configuration**

**❌ NON-COMPLIANT** (hidden defaults):

```python
class VectorizedHamletEnv:
    def __init__(self, config: dict):
        self.grid_size = config.get("grid_size", 8)  # Hidden default!
        self.move_energy_cost = config.get("move_energy_cost", 0.005)  # Hidden!
        self.partial_observability = config.get("partial_observability", False)  # Hidden!
```

**✅ COMPLIANT** (explicit required):

```python
class VectorizedHamletEnv:
    def __init__(self, config: dict):
        # All parameters required, fail fast with clear errors
        required_params = ["grid_size", "move_energy_cost", "partial_observability"]
        for param in required_params:
            if param not in config:
                raise ValueError(
                    f"Missing required parameter '{param}' in environment config. "
                    f"Add to your training.yaml:\n"
                    f"environment:\n"
                    f"  grid_size: 8\n"
                    f"  move_energy_cost: 0.005\n"
                    f"  partial_observability: false"
                )

        self.grid_size = config["grid_size"]
        self.move_energy_cost = config["move_energy_cost"]
        self.partial_observability = config["partial_observability"]
```

**Example 2: Exploration Configuration**

**❌ NON-COMPLIANT**:

```python
class EpsilonGreedyExploration:
    def __init__(
        self,
        epsilon_start: float = 1.0,    # Hidden default
        epsilon_decay: float = 0.995,  # Hidden default
        epsilon_min: float = 0.01      # Hidden default
    ):
        self.epsilon = epsilon_start
        self.decay = epsilon_decay
        self.min = epsilon_min
```

**✅ COMPLIANT**:

```python
class EpsilonGreedyExploration:
    def __init__(self, config: dict):
        if "epsilon_start" not in config:
            raise ValueError(
                "Missing 'epsilon_start'. Add to exploration config:\n"
                "exploration:\n"
                "  epsilon_start: 1.0"
            )
        if "epsilon_decay" not in config:
            raise ValueError("Missing 'epsilon_decay'. Add: epsilon_decay: 0.995")
        if "epsilon_min" not in config:
            raise ValueError("Missing 'epsilon_min'. Add: epsilon_min: 0.01")

        self.epsilon = config["epsilon_start"]
        self.decay = config["epsilon_decay"]
        self.min = config["epsilon_min"]
```

**Example 3: Infrastructure Parameter (Whitelisted)**

**✅ COMPLIANT (infrastructure, whitelisted)**:

```python
class LiveInferenceServer:
    def __init__(
        self,
        checkpoint_dir: Path,
        config_dir: Path,
        port: int = 8766,  # OK - infrastructure, whitelisted
        step_delay: float = 0.2,  # OK - visualization speed, not algorithm
        device: str = "cpu",  # OK - Python runtime, whitelisted
    ):
        self.checkpoint_dir = checkpoint_dir
        self.config_dir = config_dir
        self.port = port  # Network infrastructure
        self.step_delay = step_delay  # Playback speed (UI concern)
        self.device = torch.device(device)  # Hardware selection
```

**Whitelist entry**:

```
src/townlet/demo/live_inference.py:__init__:port  # Network infrastructure, not UAC/BAC
src/townlet/demo/live_inference.py:__init__:step_delay  # UI playback speed, not algorithm
src/townlet/demo/live_inference.py:__init__:device  # Python runtime environment
```

**Key distinction**: These parameters affect WHERE/HOW code runs (infrastructure), not WHAT the algorithm does (behavior).

### Complete Config Example

**Compliant L0_0_minimal/training.yaml** (all required params explicit):

```yaml
# COMPLETE configuration - no hidden defaults
environment:
  grid_size: 3
  partial_observability: false
  vision_range: 2
  enable_temporal_mechanics: false
  enabled_affordances: ["Bed"]
  move_energy_cost: 0.005
  wait_energy_cost: 0.001
  interact_energy_cost: 0.0

population:
  num_agents: 1
  learning_rate: 0.00025
  gamma: 0.99
  replay_buffer_capacity: 10000
  network_type: simple
  batch_size: 16
  train_frequency: 4
  target_update_frequency: 100
  max_grad_norm: 10.0

curriculum:
  max_steps_per_episode: 500
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 1000

exploration:
  type: epsilon_greedy
  epsilon_start: 1.0
  epsilon_decay: 0.99
  epsilon_min: 0.01

training:
  device: cuda
  max_episodes: 5000
```

**Self-contained**: Anyone can read this config and know exactly how the system will behave.

### FAQ

**Q: Why is this worth the config verbosity?**

A: Config verbosity is **valuable documentation**. When a researcher shares a config, recipient knows exactly what values were used. No guessing, no hunting through code for defaults. Reproducibility > brevity.

**Q: What if I want to provide sensible defaults for new users?**

A: Provide **template configs** in `configs/templates/` with recommended values. Users copy template and customize. Templates are explicit configs, not code defaults.

**Q: How do I handle parameters that are 99% always the same value?**

A: Specify explicitly. If value rarely changes, that's fine - config documents standard practice. When someone needs a different value, they know parameter exists and where to change it.

**Q: Can I use YAML anchors to reduce repetition?**

A: Yes! YAML anchors are explicit:

```yaml
# Define anchor
defaults: &defaults
  learning_rate: 0.00025
  gamma: 0.99

# Reuse with merge
population:
  <<: *defaults
  batch_size: 16  # Override or add new params
```

Anchors are in YAML (visible), not code (hidden).

**Q: What about development/debug parameters?**

A: **Test**: Does this parameter affect WHAT the algorithm does, or WHERE/HOW it runs?

- **UAC/BAC** (affects WHAT): MUST be in config, NO defaults allowed
  - Examples: learning_rate, grid_size, epsilon_decay, move_energy_cost
  - Test: If changed, does agent behavior or universe mechanics change? → Must be explicit

- **Infrastructure** (affects WHERE/HOW): Can have code defaults if whitelisted
  - Examples: logging_level, num_workers, device, port, checkpoint_dir
  - Test: If changed, does algorithm still produce same results? → Can have default

**If in doubt, make it explicit.** Err on side of requiring config specification.

**Q: Can cues.yaml or display_name fields have defaults?**

A: **NO**. All UAC/BAC config files (bars.yaml, affordances.yaml, cascades.yaml, cues.yaml, actions.yaml, training.yaml) are **mandatory and complete**. Even if a field is "just for UI", it's part of the universe specification and must be explicit.

Exception: Fields that are truly computed/derived (e.g., `observation_dim` calculated from `grid_size`), not specified in YAML at all.

---

**Status**: Accepted ✅
**Effective Date**: 2025-11-04
**Supersedes**: N/A
