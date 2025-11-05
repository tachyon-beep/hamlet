# Special Test Directory

This directory contains tests specific to particular tasks or migration efforts that validate temporary behaviors, migration steps, or task-specific requirements.

## Purpose

Tests in this directory:
- **Validate migration paths** (e.g., TASK-002A substrate migration)
- **Check temporary limitations** (e.g., non-square grid rejection until Phase 6)
- **Verify backward compatibility** during transitions
- **Test task-specific error messages** that guide users through migrations
- **Can be safely removed or refactored** once the task is complete and behaviors are permanent

## Current Test Files

### TASK-002A: Configurable Spatial Substrates

**`test_task002a_config_migration.py`**
- Validates all production configs migrated correctly to substrate.yaml
- Tests schema validation for all curriculum levels
- Verifies behavioral equivalence (observation dimensions unchanged)
- Tests edge cases (invalid configs, example configs)

**`test_task002a_integration.py`**
- Integration tests for environment loading with substrate
- Validates observation dimensions match legacy behavior
- Tests substrate dimensions and boundary behavior
- Verifies distance metric configuration

**`test_task002a_env_migration.py`**
- Tests migration error messages (missing substrate.yaml)
- Validates temporary limitations (non-square grids)
- Tests backward compatibility (grid_size parameter override)
- Verifies legacy behavior matching

## Lifecycle

Tests in this directory are **temporary** and should be:

1. **Reviewed after task completion** - Are they still needed?
2. **Refactored into core tests** if behavior becomes permanent
3. **Archived or removed** if no longer relevant
4. **Documented** with clear comments explaining their temporary nature

## When to Add Tests Here

Add tests to `special/` when:
- Testing migration-specific behavior (error messages, backward compat)
- Validating temporary limitations that will be lifted later
- Checking task-specific validation logic (e.g., "all configs migrated")
- Tests are closely tied to a specific task number (e.g., TASK-002A)

## When NOT to Add Tests Here

Keep tests in regular test directories when:
- Testing core functionality that will remain permanent
- Validating general substrate behavior (not migration-specific)
- Testing public APIs that are stable and long-term
- Tests are reusable across multiple features

## Naming Convention

Use the pattern: `test_task{NUMBER}{LETTER}_{category}.py`

Examples:
- `test_task002a_config_migration.py` - Config migration tests for TASK-002A
- `test_task002a_integration.py` - Integration tests for TASK-002A
- `test_task003b_action_space.py` - Action space tests for TASK-003B

## Maintenance

**After completing a task:**
1. Review all tests in `special/test_task{NUMBER}*`
2. Determine which tests are still relevant
3. Move permanent tests to appropriate directories
4. Remove or archive obsolete tests
5. Update this README
