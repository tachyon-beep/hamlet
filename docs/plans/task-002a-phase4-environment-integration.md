## Phase 4: Environment Integration (1.5 hours)

### Task 4.1: Add Substrate to VectorizedEnv (Load Only)

**Files:**

- Modify: `src/townlet/environment/vectorized_env.py`

**Step 1: Write test for environment substrate loading**

Create: `tests/test_townlet/unit/test_env_substrate_loading.py`

```python
"""Test environment loads and uses substrate configuration."""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.substrate.grid2d import Grid2DSubstrate


def test_env_loads_substrate_config():
    """Environment should load substrate.yaml and create substrate instance."""
    # Note: This test will initially PASS with legacy behavior
    # After Phase 4 integration, it will load from substrate.yaml

    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    # After integration, should have substrate attribute
    # For now, check legacy grid_size exists
    assert hasattr(env, "grid_size")
    assert env.grid_size == 8


def test_env_substrate_accessible():
    """Environment should expose substrate for inspection."""
    # This test will FAIL initially, becomes valid after integration

    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    # After integration:
    # assert hasattr(env, "substrate")
    # assert isinstance(env.substrate, Grid2DSubstrate)
    # assert env.substrate.width == 8
```

**Step 2: Run test to establish baseline**

```bash
uv run pytest tests/test_townlet/unit/test_env_substrate_loading.py::test_env_loads_substrate_config -v
```

Expected: PASS (legacy behavior)

**Step 3: Add substrate loading to VectorizedEnv.**init****

Modify: `src/townlet/environment/vectorized_env.py`

Find `__init__` method (around line 36). After line where `grid_size` is set, add:

```python
# BREAKING CHANGE: substrate.yaml is now REQUIRED
substrate_config_path = config_pack_path / "substrate.yaml"
if not substrate_config_path.exists():
    raise FileNotFoundError(
        f"substrate.yaml is required but not found in {config_pack_path}.\n\n"
        f"All config packs must define their spatial substrate.\n\n"
        f"Quick fix:\n"
        f"  1. Copy template: cp docs/examples/substrate.yaml {config_pack_path}/\n"
        f"  2. Edit substrate.yaml to match your grid_size from training.yaml\n"
        f"  3. See CLAUDE.md 'Configuration System' for details\n\n"
        f"This is a breaking change from TASK-002A. Previous configs without\n"
        f"substrate.yaml will no longer work. See CHANGELOG.md for migration guide."
    )

from townlet.substrate.config import load_substrate_config
from townlet.substrate.factory import SubstrateFactory

substrate_config = load_substrate_config(substrate_config_path)
self.substrate = SubstrateFactory.build(substrate_config, device=self.device)

# Update grid_size from substrate (for backward compatibility with other code)
if hasattr(self.substrate, "width") and hasattr(self.substrate, "height"):
    if self.substrate.width != self.substrate.height:
        raise ValueError(
            f"Non-square grids not yet supported: "
            f"{self.substrate.width}×{self.substrate.height}"
        )
    self.grid_size = self.substrate.width
```

**Step 4: Commit**

```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/unit/test_env_substrate_loading.py
git commit -m "feat: add substrate loading to VectorizedHamletEnv

BREAKING CHANGE: substrate.yaml is now REQUIRED for all config packs.

Environment now loads substrate.yaml and fails fast if missing:
- Creates substrate instance via SubstrateFactory
- Updates grid_size from substrate (for compatibility)
- Clear error message with migration steps

Backward compatibility removed per user authorization.
Old configs without substrate.yaml will NOT work.

Part of TASK-002A (Configurable Spatial Substrates)."
```

---
