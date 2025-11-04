# No-Defaults Linter

Enforces the "no default variables unless whitelisted" rule for HAMLET's UNIVERSE_AS_CODE philosophy.

## Philosophy

**No-Defaults Principle**: All behavioral parameters must be explicitly specified in config files. No implicit defaults allowed.

**Why**:

- Operator accountability (must understand each parameter)
- Transparency (no hidden behavior)
- Reproducibility (config is complete specification)
- Prevents drift (code changes don't silently affect universes)

**Exceptions**: Only truly optional features (visualization, metadata) can have defaults, and they must be whitelisted.

## Usage

```bash
# Scan a directory or file
python scripts/no_defaults_lint.py src/townlet/

# Use whitelist to allow specific defaults
python scripts/no_defaults_lint.py src/townlet/ --whitelist .defaults-whitelist.txt

# Show AST context (helps create structural patterns)
python scripts/no_defaults_lint.py src/townlet/ --show-context

# Show whitelisted violations (for debugging)
python scripts/no_defaults_lint.py src/townlet/ --whitelist .defaults-whitelist.txt --show-whitelisted

# Scan multiple paths
python scripts/no_defaults_lint.py src/townlet/ tests/
```

### Creating Structural Patterns

Use `--show-context` to see the AST structure and create patterns:

```bash
$ python scripts/no_defaults_lint.py src/townlet/agent/networks.py --show-context

src/townlet/agent/networks.py:14:4: DEF001: Function '__init__' has parameter default(s)
  Context: src/townlet/agent/networks.py::SimpleQNetwork::__init__

# Create pattern from context path:
src/townlet/agent/networks.py::SimpleQNetwork::__init__:DEF001
# Or whitelist entire class:
src/townlet/agent/networks.py::SimpleQNetwork:*
```

## Exit Codes

- `0` - No violations (or all violations whitelisted)
- `1` - Non-whitelisted violations found
- `2` - Usage error or file not found

## Detected Violations

| Rule ID | Description | Example |
|---------|-------------|---------|
| DEF001 | Function parameter defaults | `def foo(x=10)` |
| DEF002 | Lambda parameter defaults | `lambda x=10: x` |
| ASG001 | Logical OR as default | `x = a or b` |
| ASG002 | Ternary as default | `x = a if cond else b` |
| CALL001 | dict.get/setdefault with default | `config.get("key", default)` |
| CALL002 | os.getenv with default | `os.getenv("VAR", "default")` |
| CALL003 | Call with default= or default_factory= | `Field(default="x")` |
| ARGP001 | argparse add_argument(default=...) | `parser.add_argument("--x", default=10)` |
| CLICK001 | click.option(default=...) | `@click.option("--x", default=10)` |

## Whitelist Format

File: `.defaults-whitelist.txt`

### Structural Patterns (Recommended)

Stable across code refactors - matches by AST structure, not line numbers:

```
# Format: <filepath>::<class>::<function>::<variable>:<rule_id>
# Use * for wildcards, ** for directory matching

# Whitelist entire module
src/townlet/recording/**:*

# Whitelist entire file
src/townlet/demo/runner.py:*

# Whitelist all defaults in a class
src/townlet/agent/networks.py::SimpleQNetwork:*

# Whitelist specific function
src/townlet/environment/cascade_engine.py::CascadeEngine::apply_base_depletions:DEF001

# Whitelist specific field in a class
src/townlet/training/state.py::BatchedAgentState::description:CALL003
```

**Benefits**:

- **Stable**: Won't break when code moves or line numbers change
- **Semantic**: Whitelist by intent (whole classes/modules)
- **Maintainable**: 20 patterns vs 278 line numbers

### Line-Based Patterns (Legacy)

Fragile - breaks when code is refactored. Use only when structural patterns don't fit:

```
# Format: <filepath>:<lineno>:<rule_id>

# Example (avoid if possible)
src/townlet/environment/config.py:42:DEF001
```

### Whitelist Guidelines

**DO whitelist**:

- Truly optional features (e.g., `cues: CuesConfig | None = None`)
- Metadata only (e.g., `description: str | None = None`)
- Computed values (e.g., `observation_dim: int` calculated at compile time)

**DON'T whitelist**:

- Behavioral parameters (epsilon_decay, learning_rate, etc.)
- Config lookups with fallbacks (`config.get("key", default)`)
- Hidden magic values

**Rule of thumb**: If omitting the field changes simulation behavior, it's REQUIRED (not whitelisted).

## Integration with CI

Add to `.github/workflows/ci.yml`:

```yaml
- name: Check for unauthorized defaults
  run: |
    python scripts/no_defaults_lint.py src/townlet/ --whitelist .defaults-whitelist.txt
```

## Examples

### Good (No Defaults)

```python
# Pydantic DTO - all required
class TrainingConfig(BaseModel):
    epsilon_start: float      # Required
    epsilon_decay: float      # Required
    epsilon_min: float        # Required
    learning_rate: float      # Required
```

### Bad (Hidden Defaults)

```python
# DON'T DO THIS
class TrainingConfig(BaseModel):
    epsilon_start: float = 1.0        # ❌ Hidden default
    epsilon_decay: float = 0.995      # ❌ Hidden default
    learning_rate: float = 0.00025    # ❌ Hidden default
```

### Whitelisted (Truly Optional)

```python
# Optional visualization (doesn't affect simulation)
class HamletConfig(BaseModel):
    bars: BarsConfig
    cascades: CascadeConfig
    training: TrainingConfig
    cues: CuesConfig | None = None    # ✅ Whitelisted - optional for headless training
```

Whitelist entry:

```
src/townlet/config/hamlet_config.py:15:DEF001  # Optional cues for headless training
```

## Troubleshooting

### "Too many violations!"

If you're migrating existing code:

1. **Use structural patterns** (much better than line-based):

   ```bash
   # See violations with AST context
   python scripts/no_defaults_lint.py src/townlet/ --show-context

   # Create strategic patterns (whitelist modules/classes)
   # 19 structural patterns can cover 281 violations!
   ```

2. **Start broad, then narrow**:
   - Whitelist entire modules initially: `src/townlet/demo/**:*`
   - Refactor module by module
   - Replace broad patterns with specific ones
   - Eventually remove all whitelists

3. **Legacy: Generate line-based whitelist** (not recommended):

   ```bash
   python scripts/no_defaults_lint.py src/townlet/ 2>&1 | \
     grep -E "^src/.*:[0-9]+:[0-9]+:" | \
     awk -F: '{print $1":"$2":"$4}' > .defaults-whitelist-lines.txt
   ```

   Note: Line-based patterns break when code moves!

### "Whitelist not working"

**For structural patterns**:

- Use `--show-context` to see the AST path
- Check pattern syntax: `filepath::class::function:rule_id`
- Wildcards: `*` matches within level, `**` matches path components
- Pattern is case-sensitive

**For line-based patterns** (legacy):

- Filepath matches exactly (use absolute or relative consistently)
- Line number is correct (fragile - changes when code moves!)
- Rule ID matches (case-sensitive)
- No extra spaces or formatting issues

**Tip**: Prefer structural patterns - they're more reliable!

### "False positives"

If the linter detects something that shouldn't be flagged:

- Add to whitelist with clear justification comment
- Or modify the linter rules if it's a systemic issue

## Related Documentation

- **TASK-003**: UAC Contracts - Core DTOs (no-defaults principle)
- **TASK-004A**: Universe Compiler Implementation
- **docs/architecture/UNIVERSE_AS_CODE.md**: Design philosophy
