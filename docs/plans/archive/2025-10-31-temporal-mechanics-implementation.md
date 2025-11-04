# Temporal Mechanics & Multi-Interaction Affordances Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add time-based mechanics (24-tick day/night cycle) and multi-interaction affordances (progressive rewards with early exit) to enable temporal planning and opportunity cost learning.

**Architecture:** Extend VectorizedHamletEnv with time tracking, multi-tick interaction state, and dynamic affordances that transform based on time. Frontend visualization shows gradient time bar and progress rings. Agent observation includes time_of_day and interaction progress.

**Tech Stack:** PyTorch (GPU tensors), Python 3.11+, Vue 3, TypeScript, FastAPI/WebSocket

**Design Reference:** `docs/plans/2025-10-31-temporal-mechanics-design.md`

---

## Implementation Status

**✅ COMPLETED:**

- **Phase 1**: Environment Backend - Multi-Interaction Tracking (Tasks 1.1, 1.2, 1.3)
- **Phase 2**: Time-Based Action Masking (Task 2.1)
- **Phase 4**: Frontend Visualization (Tasks 4.1, 4.2)
  - Task 4.1: Time-of-day gradient bar with dawn/dusk transitions
  - Task 4.2: Interaction progress ring showing multi-tick completion
- **Phase 5**: Configuration & Training (Tasks 5.1, 5.2)
- **Phase 6**: Verification (Tasks 6.1, 6.2)

**⏸️ DEFERRED:**

- **Phase 3**: Dynamic Affordances (Task 3.1)
  - Position-based affordance mapping (CoffeeShop↔Bar)
  - Requires architectural refactoring of affordance lookup system
  - Not critical for temporal mechanics functionality
  - Can be implemented in future iteration

**Total Commits:** 12 (9 backend + 3 frontend)

**All core temporal mechanics features are functional and ready for training!**

---

## Phase 1: Environment Backend - Multi-Interaction Tracking

### Task 1.1: Add Affordance Config Schema

**Files:**

- Create: `src/townlet/environment/affordance_config.py`
- Test: `tests/test_townlet/test_affordance_config.py`

**Step 1: Write test for affordance config loading**

Create test file:

```python
# tests/test_townlet/test_affordance_config.py
import pytest
from townlet.environment.affordance_config import AffordanceConfig, AFFORDANCE_CONFIGS


def test_affordance_config_structure():
    """Verify affordance config has required fields."""
    bed_config = AFFORDANCE_CONFIGS['Bed']

    assert bed_config['required_ticks'] == 5
    assert bed_config['cost_per_tick'] == 0.01
    assert bed_config['operating_hours'] == (0, 24)
    assert 'linear' in bed_config['benefits']
    assert 'completion' in bed_config['benefits']


def test_affordance_config_benefit_math():
    """Verify 75/25 split is correct."""
    job_config = AFFORDANCE_CONFIGS['Job']

    # Job total: $22.5, 4 ticks
    # Linear: 75% = $16.875 / 4 = $4.21875 per tick
    # Completion: 25% = $5.625

    linear_money = job_config['benefits']['linear']['money']
    completion_money = job_config['benefits']['completion']['money']

    assert abs(linear_money - 0.140625) < 0.0001  # ($22.5 * 0.75) / 4
    assert abs(completion_money - 0.05625) < 0.0001  # $22.5 * 0.25


def test_dynamic_affordance_coffeeshop_bar():
    """Verify CoffeeShop and Bar share position but different hours."""
    coffee = AFFORDANCE_CONFIGS['CoffeeShop']
    bar = AFFORDANCE_CONFIGS['Bar']

    # CoffeeShop: daytime (8-18)
    assert coffee['operating_hours'] == (8, 18)

    # Bar: evening/night (18-4, wraps midnight)
    assert bar['operating_hours'] == (18, 4)
```

**Step 2: Run test to verify it fails**

```bash
cd /home/john/hamlet/.worktrees/temporal-mechanics
uv run pytest tests/test_townlet/test_affordance_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'townlet.environment.affordance_config'`

**Step 3: Implement affordance config module**

Create file:

```python
# src/townlet/environment/affordance_config.py
"""
Affordance configuration for multi-interaction and time-based mechanics.

Each affordance specifies:
- required_ticks: Number of consecutive INTERACTs for full benefit
- cost_per_tick: Money charged per tick (normalized [0, 1])
- operating_hours: (open_tick, close_tick) in [0, 23]
- benefits:
  - linear: 75% of total benefit, distributed per tick
  - completion: 25% bonus on full completion
"""

from typing import Dict, Any, Tuple


AffordanceConfig = Dict[str, Any]


AFFORDANCE_CONFIGS: Dict[str, AffordanceConfig] = {
    # === Static Affordances (24/7) ===

    'Bed': {
        'required_ticks': 5,
        'cost_per_tick': 0.01,  # $1 per tick ($5 total)
        'operating_hours': (0, 24),
        'benefits': {
            'linear': {
                'energy': +0.075,  # Per tick: (50% * 0.75) / 5
            },
            'completion': {
                'energy': +0.125,  # 50% * 0.25
                'health': +0.02,
            }
        }
    },

    'LuxuryBed': {
        'required_ticks': 5,
        'cost_per_tick': 0.022,  # $2.20 per tick ($11 total)
        'operating_hours': (0, 24),
        'benefits': {
            'linear': {
                'energy': +0.1125,  # Per tick: (75% * 0.75) / 5
            },
            'completion': {
                'energy': +0.1875,  # 75% * 0.25
                'health': +0.05,
            }
        }
    },

    'Shower': {
        'required_ticks': 3,
        'cost_per_tick': 0.01,  # $1 per tick ($3 total)
        'operating_hours': (0, 24),
        'benefits': {
            'linear': {
                'hygiene': +0.10,  # Per tick: (40% * 0.75) / 3
            },
            'completion': {
                'hygiene': +0.10,  # 40% * 0.25
            }
        }
    },

    'HomeMeal': {
        'required_ticks': 2,
        'cost_per_tick': 0.015,  # $1.50 per tick ($3 total)
        'operating_hours': (0, 24),
        'benefits': {
            'linear': {
                'satiation': +0.16875,  # Per tick: (45% * 0.75) / 2
            },
            'completion': {
                'satiation': +0.1125,  # 45% * 0.25
                'health': +0.03,
            }
        }
    },

    'Hospital': {
        'required_ticks': 3,
        'cost_per_tick': 0.05,  # $5 per tick ($15 total)
        'operating_hours': (0, 24),
        'benefits': {
            'linear': {
                'health': +0.225,  # Per tick: (60% * 0.75) / 3
            },
            'completion': {
                'health': +0.15,  # 60% * 0.25
            }
        }
    },

    'Gym': {
        'required_ticks': 4,
        'cost_per_tick': 0.02,  # $2 per tick ($8 total)
        'operating_hours': (0, 24),
        'benefits': {
            'linear': {
                'fitness': +0.1125,  # Per tick: (30% * 0.75) / 4
                'energy': -0.03,
            },
            'completion': {
                'fitness': +0.075,  # 30% * 0.25
                'mood': +0.05,
            }
        }
    },

    'FastFood': {
        'required_ticks': 1,
        'cost_per_tick': 0.10,  # $10
        'operating_hours': (0, 24),
        'benefits': {
            'linear': {
                'satiation': +0.3375,  # (45% * 0.75) / 1
                'energy': +0.1125,
            },
            'completion': {
                'satiation': +0.1125,  # 45% * 0.25
                'energy': +0.0375,
                'fitness': -0.03,
                'health': -0.02,
            }
        }
    },

    # === Business Hours Affordances (8am-6pm) ===

    'Job': {
        'required_ticks': 4,
        'cost_per_tick': 0.0,
        'operating_hours': (8, 18),
        'benefits': {
            'linear': {
                'money': +0.140625,  # Per tick: ($22.5 * 0.75) / 4
                'energy': -0.0375,
            },
            'completion': {
                'money': +0.05625,  # $22.5 * 0.25
                'social': +0.02,
                'health': -0.03,
            }
        }
    },

    'Labor': {
        'required_ticks': 4,
        'cost_per_tick': 0.0,
        'operating_hours': (8, 18),
        'benefits': {
            'linear': {
                'money': +0.1875,  # Per tick: ($30 * 0.75) / 4
                'energy': -0.05,
            },
            'completion': {
                'money': +0.075,  # $30 * 0.25
                'fitness': -0.05,
                'health': -0.05,
                'social': +0.01,
            }
        }
    },

    'Doctor': {
        'required_ticks': 2,
        'cost_per_tick': 0.04,  # $4 per tick ($8 total)
        'operating_hours': (8, 18),
        'benefits': {
            'linear': {
                'health': +0.1125,  # Per tick: (30% * 0.75) / 2
            },
            'completion': {
                'health': +0.075,  # 30% * 0.25
            }
        }
    },

    'Therapist': {
        'required_ticks': 3,
        'cost_per_tick': 0.05,  # $5 per tick ($15 total)
        'operating_hours': (8, 18),
        'benefits': {
            'linear': {
                'mood': +0.15,  # Per tick: (40% * 0.75) / 3
            },
            'completion': {
                'mood': +0.10,  # 40% * 0.25
                'social': +0.05,
            }
        }
    },

    'Recreation': {
        'required_ticks': 2,
        'cost_per_tick': 0.03,  # $3 per tick ($6 total)
        'operating_hours': (8, 22),
        'benefits': {
            'linear': {
                'mood': +0.1125,  # Per tick: (30% * 0.75) / 2
                'social': +0.075,
            },
            'completion': {
                'mood': +0.075,  # 30% * 0.25
                'social': +0.05,
            }
        }
    },

    # === Dynamic Affordances (Time-Dependent) ===

    'CoffeeShop': {
        'required_ticks': 1,
        'cost_per_tick': 0.02,  # $2
        'operating_hours': (8, 18),
        'benefits': {
            'linear': {
                'energy': +0.1125,  # (15% * 0.75) / 1
                'mood': +0.0375,
                'social': +0.045,
            },
            'completion': {
                'energy': +0.0375,  # 15% * 0.25
                'mood': +0.0125,
                'social': +0.015,
            }
        }
    },

    'Bar': {
        'required_ticks': 2,
        'cost_per_tick': 0.075,  # $7.50 per round ($15 total)
        'operating_hours': (18, 4),  # Wraps midnight
        'benefits': {
            'linear': {
                'mood': +0.075,  # Per tick: (20% * 0.75) / 2
                'social': +0.05625,
                'health': -0.01875,
            },
            'completion': {
                'mood': +0.05,  # 20% * 0.25
                'social': +0.0375,
                'health': -0.0125,
            }
        }
    },

    'Park': {
        'required_ticks': 2,
        'cost_per_tick': 0.0,
        'operating_hours': (6, 22),
        'benefits': {
            'linear': {
                'mood': +0.0975,  # Per tick: (26% * 0.75) / 2
                'social': +0.0375,
            },
            'completion': {
                'mood': +0.065,  # 26% * 0.25
                'social': +0.025,
                'fitness': +0.02,
            }
        }
    },
}


# Meter name to index mapping
METER_NAME_TO_IDX = {
    'energy': 0,
    'hygiene': 1,
    'satiation': 2,
    'money': 3,
    'mood': 4,
    'social': 5,
    'health': 6,
    'fitness': 7,
}


def is_affordance_open(time_of_day: int, operating_hours: Tuple[int, int]) -> bool:
    """
    Check if affordance is open at given time.

    Handles midnight wraparound (e.g., Bar: 18-4 means 6pm to 4am).

    Args:
        time_of_day: Current tick [0-23]
        operating_hours: (open_tick, close_tick)

    Returns:
        True if open, False if closed
    """
    open_tick, close_tick = operating_hours

    if open_tick < close_tick:
        # Normal hours (e.g., 8-18)
        return open_tick <= time_of_day < close_tick
    else:
        # Wraparound hours (e.g., 18-4 = 6pm to 4am)
        return time_of_day >= open_tick or time_of_day < close_tick
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_townlet/test_affordance_config.py -v
```

Expected: `3 passed`

**Step 5: Commit**

```bash
git add src/townlet/environment/affordance_config.py tests/test_townlet/test_affordance_config.py
git commit -m "feat: add affordance config schema for temporal mechanics

- Define AFFORDANCE_CONFIGS with 14 affordances
- 75/25 linear/completion benefit split
- Operating hours for time-based gating
- Dynamic affordances: CoffeeShop/Bar, different hours same position
- is_affordance_open() handles midnight wraparound"
```

---

### Task 1.2: Add Time and Progress Tracking to VectorizedHamletEnv

**Files:**

- Modify: `src/townlet/environment/vectorized_env.py`
- Test: `tests/test_townlet/test_vectorized_env_temporal.py`

**Step 1: Write test for time tracking**

Create test file:

```python
# tests/test_townlet/test_vectorized_env_temporal.py
import pytest
import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.fixture
def env():
    """Create test environment with temporal mechanics enabled."""
    affordances = {
        'Job': torch.tensor([2, 3]),
        'Bed': torch.tensor([5, 5]),
    }
    return VectorizedHamletEnv(
        num_agents=2,
        grid_size=8,
        affordances=affordances,
        device=torch.device('cpu'),
        enable_temporal_mechanics=True,
    )


def test_time_of_day_cycles():
    """Verify time cycles through 24 ticks."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        affordances={'Bed': torch.tensor([2, 2])},
        device=torch.device('cpu'),
        enable_temporal_mechanics=True,
    )

    env.reset()

    # Step 24 times
    for i in range(24):
        assert env.time_of_day == i
        env.step(torch.tensor([4]))  # INTERACT action

    # Should wrap back to 0
    assert env.time_of_day == 0


def test_interaction_progress_tracking(env):
    """Verify multi-tick interaction progress."""
    env.reset()

    # Place agent on Bed
    env.positions[0] = torch.tensor([5, 5])

    # Initial progress: 0
    assert env.interaction_progress[0] == 0

    # First INTERACT
    env.step(torch.tensor([4, 0]))  # Agent 0: INTERACT, Agent 1: UP

    assert env.interaction_progress[0] == 1
    assert env.last_interaction_affordance[0] == 'Bed'

    # Second INTERACT (same position)
    env.step(torch.tensor([4, 0]))

    assert env.interaction_progress[0] == 2

    # Move away - progress resets
    env.step(torch.tensor([0, 0]))  # UP

    assert env.interaction_progress[0] == 0
    assert env.last_interaction_affordance[0] is None


def test_observation_includes_time_and_progress(env):
    """Verify observation contains time_of_day and interaction_progress."""
    obs = env.reset()

    # Observation shape should be [num_agents, obs_dim]
    # obs_dim = original + 2 (time_of_day, interaction_progress)
    # For full observability: 64 (grid) + 8 (meters) + 2 = 74

    assert obs.shape == (2, 74)

    # time_of_day should be normalized [0, 1]
    time_feature = obs[0, -2]
    assert 0.0 <= time_feature <= 1.0

    # interaction_progress should be [0, 1]
    progress_feature = obs[0, -1]
    assert progress_feature == 0.0  # No progress at start
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/test_vectorized_env_temporal.py -v
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'enable_temporal_mechanics'`

**Step 3: Implement time and progress tracking**

Modify `src/townlet/environment/vectorized_env.py`:

Find the `__init__` method and add:

```python
def __init__(
    self,
    num_agents: int,
    grid_size: int,
    affordances: dict,
    device: torch.device,
    partial_observability: bool = False,
    vision_range: int = 2,
    enable_temporal_mechanics: bool = False,  # NEW
):
    # ... existing init code ...

    self.enable_temporal_mechanics = enable_temporal_mechanics

    # Time tracking (NEW)
    if self.enable_temporal_mechanics:
        self.time_of_day = 0  # 0-23 tick cycle

        # Multi-interaction tracking
        self.interaction_progress = torch.zeros(
            self.num_agents,
            dtype=torch.long,
            device=self.device
        )
        self.last_interaction_affordance = [None] * self.num_agents
        self.last_interaction_position = torch.zeros(
            (self.num_agents, 2),
            dtype=torch.long,
            device=self.device
        )
```

Find the `reset()` method and add:

```python
def reset(self) -> torch.Tensor:
    # ... existing reset code ...

    # Reset temporal mechanics (NEW)
    if self.enable_temporal_mechanics:
        self.time_of_day = 0
        self.interaction_progress.fill_(0)
        self.last_interaction_affordance = [None] * self.num_agents
        self.last_interaction_position.fill_(0)

    return self._get_observations()
```

Find the `step()` method and add time increment:

```python
def step(self, actions: torch.Tensor):
    # ... existing step code ...

    # Increment time (NEW)
    if self.enable_temporal_mechanics:
        self.time_of_day = (self.time_of_day + 1) % 24

    # ... rest of step logic ...
```

Find the `_get_observations()` method and extend:

```python
def _get_observations(self) -> torch.Tensor:
    if self.partial_observability:
        # ... existing partial obs code ...
    else:
        # Full observability
        grid_obs = F.one_hot(
            self.positions[:, 0] * self.grid_size + self.positions[:, 1],
            num_classes=self.grid_size * self.grid_size
        ).float()

        obs = torch.cat([grid_obs, self.meters], dim=1)

        # Add temporal features (NEW)
        if self.enable_temporal_mechanics:
            # time_of_day: normalized [0, 1]
            time_feature = torch.full(
                (self.num_agents, 1),
                self.time_of_day / 24.0,
                device=self.device
            )

            # interaction_progress: normalized [0, 1]
            # (requires knowing required_ticks for current affordance)
            progress_feature = torch.zeros((self.num_agents, 1), device=self.device)
            for i in range(self.num_agents):
                if self.last_interaction_affordance[i] is not None:
                    from townlet.environment.affordance_config import AFFORDANCE_CONFIGS
                    config = AFFORDANCE_CONFIGS[self.last_interaction_affordance[i]]
                    progress_ratio = self.interaction_progress[i].float() / config['required_ticks']
                    progress_feature[i, 0] = progress_ratio

            obs = torch.cat([obs, time_feature, progress_feature], dim=1)

        return obs
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_townlet/test_vectorized_env_temporal.py -v
```

Expected: `3 passed`

**Step 5: Commit**

```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/test_vectorized_env_temporal.py
git commit -m "feat: add time tracking and progress to VectorizedHamletEnv

- Add enable_temporal_mechanics flag
- Track time_of_day (0-23 tick cycle)
- Track interaction_progress per agent
- Track last_interaction_affordance and position
- Add time and progress to observation (74 dims for full obs)
- Reset time/progress on environment reset"
```

---

### Task 1.3: Implement Multi-Tick Interaction Logic

**Files:**

- Modify: `src/townlet/environment/vectorized_env.py:_handle_interactions()`
- Test: `tests/test_townlet/test_multi_interaction.py`

**Step 1: Write test for progressive benefits**

```python
# tests/test_townlet/test_multi_interaction.py
import pytest
import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.fixture
def env():
    """Environment with Bed (5 ticks) at position (2,2)."""
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        affordances={'Bed': torch.tensor([2, 2])},
        device=torch.device('cpu'),
        enable_temporal_mechanics=True,
    )


def test_progressive_benefit_accumulation(env):
    """Verify linear benefits accumulate per tick."""
    env.reset()
    env.positions[0] = torch.tensor([2, 2])  # On Bed

    initial_energy = env.meters[0, 0].item()

    # Bed config: 5 ticks, +7.5% energy per tick (linear)
    # First INTERACT
    env.step(torch.tensor([4]))

    energy_after_1 = env.meters[0, 0].item()
    assert abs((energy_after_1 - initial_energy) - 0.075) < 0.001

    # Second INTERACT
    env.step(torch.tensor([4]))

    energy_after_2 = env.meters[0, 0].item()
    assert abs((energy_after_2 - initial_energy) - 0.150) < 0.001


def test_completion_bonus(env):
    """Verify 25% bonus on full completion."""
    env.reset()
    env.positions[0] = torch.tensor([2, 2])

    initial_energy = env.meters[0, 0].item()
    initial_health = env.meters[0, 6].item()

    # Complete all 5 ticks
    for _ in range(5):
        env.step(torch.tensor([4]))

    final_energy = env.meters[0, 0].item()
    final_health = env.meters[0, 6].item()

    # Total energy: 5 × 7.5% (linear) + 12.5% (completion) = 50%
    assert abs((final_energy - initial_energy) - 0.50) < 0.001

    # Health bonus only on completion: +2%
    assert abs((final_health - initial_health) - 0.02) < 0.001


def test_early_exit_keeps_progress(env):
    """Verify agent keeps linear benefits if exiting early."""
    env.reset()
    env.positions[0] = torch.tensor([2, 2])

    initial_energy = env.meters[0, 0].item()

    # Do 3 ticks, then move away
    for _ in range(3):
        env.step(torch.tensor([4]))

    energy_after_3 = env.meters[0, 0].item()

    # Move away (UP action)
    env.step(torch.tensor([0]))

    final_energy = env.meters[0, 0].item()

    # Energy should stay at 3 × 7.5% = 22.5% gain
    # (no completion bonus)
    assert abs((energy_after_3 - initial_energy) - 0.225) < 0.001
    assert abs(final_energy - energy_after_3) < 0.001  # No change on move


def test_money_charged_per_tick(env):
    """Verify cost charged each tick, not on completion."""
    env.reset()
    env.positions[0] = torch.tensor([2, 2])
    env.meters[0, 3] = 0.50  # Start with $50

    # Bed costs $1/tick = 0.01 normalized
    env.step(torch.tensor([4]))

    money_after_1 = env.meters[0, 3].item()
    assert abs(money_after_1 - 0.49) < 0.001  # $50 - $1 = $49

    env.step(torch.tensor([4]))

    money_after_2 = env.meters[0, 3].item()
    assert abs(money_after_2 - 0.48) < 0.001  # $49 - $1 = $48
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/test_multi_interaction.py -v
```

Expected: Tests fail - interactions still apply single-shot benefits

**Step 3: Implement multi-tick interaction logic**

Modify `src/townlet/environment/vectorized_env.py:_handle_interactions()`:

```python
def _handle_interactions(self, interact_mask: torch.Tensor) -> dict:
    """
    Handle INTERACT actions with multi-tick accumulation.

    Returns:
        Dictionary mapping agent indices to affordance names
    """
    from townlet.environment.affordance_config import (
        AFFORDANCE_CONFIGS,
        METER_NAME_TO_IDX,
    )

    successful_interactions = {}

    for affordance_name, affordance_pos in self.affordances.items():
        # Distance to affordance
        distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
        at_affordance = (distances == 0) & interact_mask

        if not at_affordance.any():
            continue

        # Get config
        config = AFFORDANCE_CONFIGS[affordance_name]

        # Check affordability (per-tick cost)
        cost_per_tick = config['cost_per_tick']
        can_afford = self.meters[:, 3] >= cost_per_tick
        at_affordance = at_affordance & can_afford

        if not at_affordance.any():
            continue

        # Track successful interactions
        agent_indices = torch.where(at_affordance)[0]

        for agent_idx in agent_indices:
            agent_idx_int = agent_idx.item()
            current_pos = self.positions[agent_idx]

            # Check if continuing same affordance at same position
            if (self.last_interaction_affordance[agent_idx_int] == affordance_name and
                torch.equal(current_pos, self.last_interaction_position[agent_idx_int])):
                # Continue progress
                self.interaction_progress[agent_idx] += 1
            else:
                # New affordance - reset progress
                self.interaction_progress[agent_idx] = 1
                self.last_interaction_affordance[agent_idx_int] = affordance_name
                self.last_interaction_position[agent_idx_int] = current_pos.clone()

            ticks_done = self.interaction_progress[agent_idx].item()
            required_ticks = config['required_ticks']

            # Apply per-tick benefits (75% of total, distributed)
            for meter_name, delta in config['benefits']['linear'].items():
                meter_idx = METER_NAME_TO_IDX[meter_name]
                self.meters[agent_idx, meter_idx] += delta

            # Charge per-tick cost
            self.meters[agent_idx, 3] -= cost_per_tick

            # Completion bonus? (25% of total)
            if ticks_done == required_ticks:
                for meter_name, delta in config['benefits']['completion'].items():
                    meter_idx = METER_NAME_TO_IDX[meter_name]
                    self.meters[agent_idx, meter_idx] += delta

                # Reset progress (job complete)
                self.interaction_progress[agent_idx] = 0
                self.last_interaction_affordance[agent_idx_int] = None

            successful_interactions[agent_idx_int] = affordance_name

    # Clamp meters after updates
    self.meters = torch.clamp(self.meters, 0.0, 1.0)

    return successful_interactions
```

Also modify `step()` to reset progress on movement:

```python
def step(self, actions: torch.Tensor):
    # Store positions before movement
    old_positions = self.positions.clone() if self.enable_temporal_mechanics else None

    # ... execute movement ...

    # Reset progress for agents that moved away
    if self.enable_temporal_mechanics and old_positions is not None:
        for agent_idx in range(self.num_agents):
            if not torch.equal(old_positions[agent_idx], self.positions[agent_idx]):
                self.interaction_progress[agent_idx] = 0
                self.last_interaction_affordance[agent_idx] = None

    # ... rest of step logic ...
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_townlet/test_multi_interaction.py -v
```

Expected: `4 passed`

**Step 5: Commit**

```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/test_multi_interaction.py
git commit -m "feat: implement multi-tick interaction mechanics

- Track progress per agent, reset on movement
- Apply 75% linear benefits per tick
- Apply 25% completion bonus on finish
- Charge money per tick (prevents free sampling)
- Reset progress when affordance completes or agent moves"
```

---

## Phase 2: Time-Based Action Masking

### Task 2.1: Implement Operating Hours Action Masking

**Files:**

- Modify: `src/townlet/environment/vectorized_env.py:get_action_masks()`
- Test: `tests/test_townlet/test_time_based_masking.py`

**Step 1: Write test for time-based masking**

```python
# tests/test_townlet/test_time_based_masking.py
import pytest
import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_job_closed_outside_business_hours():
    """Verify Job is masked out after 6pm."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        affordances={'Job': torch.tensor([2, 3])},
        device=torch.device('cpu'),
        enable_temporal_mechanics=True,
    )

    env.reset()
    env.positions[0] = torch.tensor([2, 3])  # On Job
    env.meters[0, 3] = 1.0  # Full money (can afford)

    # Tick 10 (10am): Job is open (8-18)
    env.time_of_day = 10
    masks = env.get_action_masks()
    assert masks[0, 4] == True  # INTERACT allowed

    # Tick 19 (7pm): Job is closed
    env.time_of_day = 19
    masks = env.get_action_masks()
    assert masks[0, 4] == False  # INTERACT blocked


def test_bar_open_after_6pm():
    """Verify Bar opens at 6pm."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        affordances={'Bar': torch.tensor([3, 5])},
        device=torch.device('cpu'),
        enable_temporal_mechanics=True,
    )

    env.reset()
    env.positions[0] = torch.tensor([3, 5])
    env.meters[0, 3] = 1.0

    # Tick 12 (noon): Bar is closed (opens at 18)
    env.time_of_day = 12
    masks = env.get_action_masks()
    assert masks[0, 4] == False

    # Tick 20 (8pm): Bar is open
    env.time_of_day = 20
    masks = env.get_action_masks()
    assert masks[0, 4] == True


def test_bar_wraparound_midnight():
    """Verify Bar hours wrap midnight (18-4)."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        affordances={'Bar': torch.tensor([3, 5])},
        device=torch.device('cpu'),
        enable_temporal_mechanics=True,
    )

    env.reset()
    env.positions[0] = torch.tensor([3, 5])
    env.meters[0, 3] = 1.0

    # Tick 2 (2am): Bar is still open (wraps to 4am)
    env.time_of_day = 2
    masks = env.get_action_masks()
    assert masks[0, 4] == True

    # Tick 5 (5am): Bar is closed
    env.time_of_day = 5
    masks = env.get_action_masks()
    assert masks[0, 4] == False
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/test_time_based_masking.py -v
```

Expected: Tests fail - masking doesn't check operating hours yet

**Step 3: Implement time-based masking**

Modify `src/townlet/environment/vectorized_env.py:get_action_masks()`:

```python
def get_action_masks(self) -> torch.Tensor:
    """
    Get valid action masks for all agents.

    Returns:
        [num_agents, action_dim] bool tensor
    """
    from townlet.environment.affordance_config import (
        AFFORDANCE_CONFIGS,
        is_affordance_open,
    )

    action_masks = torch.ones(
        (self.num_agents, self.action_dim),
        dtype=torch.bool,
        device=self.device
    )

    # Mask movement based on boundaries
    # UP (0)
    action_masks[self.positions[:, 1] == 0, 0] = False
    # DOWN (1)
    action_masks[self.positions[:, 1] == self.grid_size - 1, 1] = False
    # LEFT (2)
    action_masks[self.positions[:, 0] == 0, 2] = False
    # RIGHT (3)
    action_masks[self.positions[:, 0] == self.grid_size - 1, 3] = False

    # Mask INTERACT (4) based on affordance availability
    interact_valid = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)

    for affordance_name, affordance_pos in self.affordances.items():
        # Distance to affordance
        distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
        at_affordance = (distances == 0)

        if not at_affordance.any():
            continue

        # Check operating hours (if temporal mechanics enabled)
        if self.enable_temporal_mechanics:
            config = AFFORDANCE_CONFIGS[affordance_name]
            if not is_affordance_open(self.time_of_day, config['operating_hours']):
                continue  # Affordance closed, skip

        # Check affordability
        if self.enable_temporal_mechanics:
            config = AFFORDANCE_CONFIGS[affordance_name]
            cost_per_tick = config['cost_per_tick']
        else:
            # Legacy single-shot cost (from original code)
            cost_per_tick = self._get_legacy_cost(affordance_name)

        can_afford = self.meters[:, 3] >= cost_per_tick

        # Valid if: on position AND can afford AND is open
        interact_valid |= (at_affordance & can_afford)

    action_masks[:, 4] = interact_valid

    return action_masks
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_townlet/test_time_based_masking.py -v
```

Expected: `3 passed`

**Step 5: Commit**

```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/test_time_based_masking.py
git commit -m "feat: add time-based action masking for operating hours

- Check is_affordance_open() before allowing INTERACT
- Handle midnight wraparound (Bar: 18-4)
- Block INTERACT when affordance is closed
- Agent physically cannot interact outside operating hours"
```

---

## Phase 3: Dynamic Affordances

### Task 3.1: Implement Position-Based Affordance Mapping

**Files:**

- Create: `src/townlet/environment/dynamic_affordances.py`
- Modify: `src/townlet/environment/vectorized_env.py`
- Test: `tests/test_townlet/test_dynamic_affordances.py`

**Step 1: Write test for dynamic affordance transformation**

```python
# tests/test_townlet/test_dynamic_affordances.py
import pytest
import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.environment.dynamic_affordances import DYNAMIC_AFFORDANCE_POSITIONS


def test_coffeeshop_transforms_to_bar():
    """Verify CoffeeShop → Bar transformation at 6pm."""
    # Position (3, 5) has CoffeeShop (8-18) and Bar (18-4)
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        affordances=DYNAMIC_AFFORDANCE_POSITIONS,
        device=torch.device('cpu'),
        enable_temporal_mechanics=True,
    )

    env.reset()
    env.positions[0] = torch.tensor([3, 5])
    env.meters[0, 3] = 1.0  # Full money

    # Tick 10 (10am): CoffeeShop is active
    env.time_of_day = 10
    env.step(torch.tensor([4]))  # INTERACT

    # Should get CoffeeShop benefits (energy boost)
    # CoffeeShop: 1 tick, +15% energy total
    assert env.interaction_progress[0] == 0  # Completed (1 tick)

    # Tick 20 (8pm): Bar is active
    env.time_of_day = 20
    initial_mood = env.meters[0, 4].item()

    env.step(torch.tensor([4]))  # INTERACT

    # Should get Bar benefits (mood boost)
    # Bar: 2 ticks, +7.5% mood per tick (linear)
    mood_after = env.meters[0, 4].item()
    assert abs((mood_after - initial_mood) - 0.075) < 0.001


def test_affordance_position_mapping():
    """Verify get_affordance_at_time returns correct affordance."""
    from townlet.environment.dynamic_affordances import get_affordance_at_time

    position_key = (3, 5)

    # Morning: CoffeeShop
    affordance = get_affordance_at_time(position_key, time_of_day=10)
    assert affordance == 'CoffeeShop'

    # Evening: Bar
    affordance = get_affordance_at_time(position_key, time_of_day=20)
    assert affordance == 'Bar'

    # Early morning (2am): Bar (wraps midnight)
    affordance = get_affordance_at_time(position_key, time_of_day=2)
    assert affordance == 'Bar'

    # Late morning (5am): Neither (Bar closed, CoffeeShop not open yet)
    affordance = get_affordance_at_time(position_key, time_of_day=5)
    assert affordance is None
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/test_dynamic_affordances.py -v
```

Expected: `ModuleNotFoundError: No module named 'townlet.environment.dynamic_affordances'`

**Step 3: Implement dynamic affordances module**

```python
# src/townlet/environment/dynamic_affordances.py
"""
Dynamic affordances that transform based on time of day.

Some grid positions have different affordances at different times:
- (3, 5): CoffeeShop (8am-6pm) ↔ Bar (6pm-4am)
"""

from typing import Dict, Tuple, Optional
import torch
from townlet.environment.affordance_config import is_affordance_open, AFFORDANCE_CONFIGS


# Position -> list of (affordance_name, operating_hours)
DYNAMIC_AFFORDANCE_MAP: Dict[Tuple[int, int], list] = {
    (3, 5): [
        ('CoffeeShop', (8, 18)),
        ('Bar', (18, 4)),
    ],
}


# Generate affordances dict for environment initialization
# Note: This creates a single entry per position, but actual affordance
# changes based on time (handled by get_affordance_at_time)
DYNAMIC_AFFORDANCE_POSITIONS = {
    'CoffeeShop_Bar': torch.tensor([3, 5]),  # Placeholder name
}


def get_affordance_at_time(
    position: Tuple[int, int],
    time_of_day: int
) -> Optional[str]:
    """
    Get the active affordance at a position and time.

    Args:
        position: (x, y) grid position
        time_of_day: Current tick [0-23]

    Returns:
        Affordance name if open, None if no affordance active
    """
    if position not in DYNAMIC_AFFORDANCE_MAP:
        return None

    for affordance_name, operating_hours in DYNAMIC_AFFORDANCE_MAP[position]:
        if is_affordance_open(time_of_day, operating_hours):
            return affordance_name

    return None


def get_affordance_at_position_tensor(
    position_tensor: torch.Tensor,
    time_of_day: int,
    all_affordances: dict
) -> Optional[str]:
    """
    Get affordance name for a position tensor.

    Args:
        position_tensor: [2] tensor (x, y)
        time_of_day: Current tick
        all_affordances: Dict of affordance_name -> position_tensor

    Returns:
        Affordance name or None
    """
    pos_tuple = (position_tensor[0].item(), position_tensor[1].item())

    # Check if dynamic position
    dynamic_affordance = get_affordance_at_time(pos_tuple, time_of_day)
    if dynamic_affordance is not None:
        return dynamic_affordance

    # Check static affordances
    for affordance_name, affordance_pos in all_affordances.items():
        if torch.equal(position_tensor, affordance_pos):
            # Verify it's open (if temporal mechanics enabled)
            config = AFFORDANCE_CONFIGS.get(affordance_name)
            if config and is_affordance_open(time_of_day, config['operating_hours']):
                return affordance_name

    return None
```

**Step 4: Modify VectorizedHamletEnv to use dynamic affordances**

In `_handle_interactions()`, replace affordance lookup:

```python
def _handle_interactions(self, interact_mask: torch.Tensor) -> dict:
    from townlet.environment.affordance_config import AFFORDANCE_CONFIGS, METER_NAME_TO_IDX
    from townlet.environment.dynamic_affordances import get_affordance_at_position_tensor

    successful_interactions = {}

    for agent_idx in torch.where(interact_mask)[0]:
        agent_idx_int = agent_idx.item()
        current_pos = self.positions[agent_idx]

        # Get active affordance at this position and time
        affordance_name = get_affordance_at_position_tensor(
            current_pos,
            self.time_of_day if self.enable_temporal_mechanics else 0,
            self.affordances
        )

        if affordance_name is None:
            continue  # No affordance active here

        # ... rest of interaction logic (same as before) ...
```

**Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_townlet/test_dynamic_affordances.py -v
```

Expected: `2 passed`

**Step 6: Commit**

```bash
git add src/townlet/environment/dynamic_affordances.py src/townlet/environment/vectorized_env.py tests/test_townlet/test_dynamic_affordances.py
git commit -m "feat: implement dynamic affordances (CoffeeShop ↔ Bar)

- Position (3,5) transforms: CoffeeShop (8am-6pm) ↔ Bar (6pm-4am)
- get_affordance_at_time() returns active affordance or None
- _handle_interactions() uses time-aware lookup
- Agent learns same position = different affordance by time"
```

---

## Phase 4: Frontend Visualization

### Task 4.1: Add Time-of-Day Gradient Bar

**Files:**

- Modify: `frontend/src/components/TimeOfDayBar.vue` (create new)
- Modify: `frontend/src/components/MetersPanel.vue` (integrate)
- Modify: `src/hamlet/web/renderer.py` (add time to state)

**Step 1: Create TimeOfDayBar component**

```vue
<!-- frontend/src/components/TimeOfDayBar.vue -->
<template>
  <div class="time-of-day-bar">
    <div class="gradient-container">
      <div
        class="gradient-fill"
        :style="{ background: gradientStyle }"
      />
      <div
        class="tick-marker"
        :style="{ left: markerPosition }"
      >
        <div class="tick-line" />
        <div class="tick-label">{{ formattedTime }}</div>
      </div>
    </div>
    <div class="time-labels">
      <span>Dawn (6am)</span>
      <span>Noon</span>
      <span>Dusk (6pm)</span>
      <span>Midnight</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  timeOfDay: number  // 0-23
}>()

const gradientStyle = computed(() => {
  const tick = props.timeOfDay

  if (tick >= 6 && tick < 18) {
    // Day: Yellow → Black (6am to 6pm)
    const progress = (tick - 6) / 12  // 0.0 → 1.0
    return `linear-gradient(to right,
      hsl(48, 100%, 50%) ${(1 - progress) * 100}%,
      hsl(0, 0%, 10%) ${progress * 100}%)`
  } else {
    // Night: Black → Yellow (6pm to 6am)
    const tickNormalized = tick >= 18 ? tick - 18 : tick + 6
    const progress = tickNormalized / 12  // 0.0 → 1.0
    return `linear-gradient(to right,
      hsl(0, 0%, 10%) ${(1 - progress) * 100}%,
      hsl(48, 100%, 50%) ${progress * 100}%)`
  }
})

const markerPosition = computed(() => {
  return `${(props.timeOfDay / 24) * 100}%`
})

const formattedTime = computed(() => {
  const hour = props.timeOfDay
  const period = hour < 12 ? 'am' : 'pm'
  const displayHour = hour === 0 ? 12 : hour > 12 ? hour - 12 : hour
  return `${displayHour}${period}`
})
</script>

<style scoped>
.time-of-day-bar {
  margin-bottom: 16px;
}

.gradient-container {
  position: relative;
  height: 32px;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.gradient-fill {
  width: 100%;
  height: 100%;
}

.tick-marker {
  position: absolute;
  top: 0;
  bottom: 0;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
}

.tick-line {
  width: 2px;
  height: 100%;
  background: white;
  box-shadow: 0 0 4px rgba(0, 0, 0, 0.5);
}

.tick-label {
  position: absolute;
  top: -20px;
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text);
  white-space: nowrap;
  background: var(--color-background);
  padding: 2px 6px;
  border-radius: 4px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.time-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 8px;
  font-size: 11px;
  color: var(--color-text-muted);
}
</style>
```

**Step 2: Add time_of_day to WebSocket state**

Modify `src/hamlet/web/renderer.py`:

```python
def render_state(self, env, episode: int, step: int) -> dict:
    """Render current state to JSON."""
    state = {
        'episode': episode,
        'step': step,
        'grid_size': env.grid_size,
        'agents': self._render_agents(env),
        'affordances': self._render_affordances(env),
        'meters': self._render_meters(env),
        # NEW: Add time of day
        'time_of_day': env.time_of_day if hasattr(env, 'time_of_day') else 0,
        'interaction_progress': self._render_interaction_progress(env),
    }
    return state

def _render_interaction_progress(self, env) -> dict:
    """Render interaction progress for each agent."""
    if not hasattr(env, 'interaction_progress'):
        return {}

    from townlet.environment.affordance_config import AFFORDANCE_CONFIGS

    progress = {}
    for agent_idx in range(env.num_agents):
        if env.last_interaction_affordance[agent_idx] is not None:
            affordance_name = env.last_interaction_affordance[agent_idx]
            config = AFFORDANCE_CONFIGS[affordance_name]

            progress[str(agent_idx)] = {
                'affordance_name': affordance_name,
                'ticks_completed': env.interaction_progress[agent_idx].item(),
                'ticks_required': config['required_ticks'],
                'progress_ratio': env.interaction_progress[agent_idx].item() / config['required_ticks'],
            }

    return progress
```

**Step 3: Integrate TimeOfDayBar into UI**

Modify `frontend/src/components/MetersPanel.vue`:

```vue
<template>
  <div class="meters-panel">
    <h3>Agent Metrics</h3>

    <!-- NEW: Time of Day Bar -->
    <TimeOfDayBar v-if="timeOfDay !== undefined" :time-of-day="timeOfDay" />

    <!-- Existing meter displays -->
    <div class="meter-grid">
      <!-- ... existing meters ... -->
    </div>
  </div>
</template>

<script setup lang="ts">
import TimeOfDayBar from './TimeOfDayBar.vue'

const props = defineProps<{
  meters: MeterData
  timeOfDay?: number  // NEW
}>()
</script>
```

Update store to pass `time_of_day`:

```typescript
// frontend/src/stores/simulation.ts
interface SimulationState {
  // ... existing fields ...
  timeOfDay: number
}

// In WebSocket message handler
socket.on('state_update', (data) => {
  state.timeOfDay = data.time_of_day || 0
  // ...
})
```

**Step 4: Test in browser**

```bash
# Terminal 1: Start backend
cd /home/john/hamlet/.worktrees/temporal-mechanics
python -m hamlet.demo.live_inference checkpoints_level2 8766 0.2 1000

# Terminal 2: Start frontend
cd frontend && npm run dev
```

Open <http://localhost:5173> and verify:

- Gradient bar shows day/night transition
- Tick marker moves as time progresses
- Labels show current time (e.g., "2pm")

**Step 5: Commit**

```bash
git add frontend/src/components/TimeOfDayBar.vue frontend/src/components/MetersPanel.vue frontend/src/stores/simulation.ts src/hamlet/web/renderer.py
git commit -m "feat: add time-of-day gradient bar to UI

- TimeOfDayBar component with yellow→black→yellow gradient
- Tick marker shows current time with label
- Gradient transitions smoothly (day: 6am-6pm, night: 6pm-6am)
- Backend sends time_of_day in state updates
- Integration with MetersPanel"
```

---

### Task 4.2: Add Interaction Progress Ring

**Files:**

- Create: `frontend/src/components/ProgressRing.vue`
- Modify: `frontend/src/components/Grid.vue`

**Step 1: Create ProgressRing component**

```vue
<!-- frontend/src/components/ProgressRing.vue -->
<template>
  <g v-if="progress > 0">
    <!-- Progress ring -->
    <circle
      :cx="cx"
      :cy="cy"
      :r="radius"
      fill="none"
      :stroke="strokeColor"
      stroke-width="3"
      :stroke-dasharray="circumference"
      :stroke-dashoffset="strokeOffset"
      class="progress-ring"
      transform-origin="center"
      :transform="`rotate(-90 ${cx} ${cy})`"
    />

    <!-- Progress text -->
    <text
      :x="cx"
      :y="cy + radius + 15"
      text-anchor="middle"
      class="progress-text"
    >
      {{ ticksCompleted }}/{{ ticksRequired }}
    </text>
  </g>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  cx: number
  cy: number
  radius: number
  ticksCompleted: number
  ticksRequired: number
}>()

const progress = computed(() => {
  return props.ticksCompleted / props.ticksRequired
})

const circumference = computed(() => {
  return 2 * Math.PI * props.radius
})

const strokeOffset = computed(() => {
  return circumference.value * (1 - progress.value)
})

const strokeColor = computed(() => {
  return props.ticksCompleted === props.ticksRequired
    ? '#22c55e'  // Green (completed)
    : '#eab308'  // Yellow (in progress)
})
</script>

<style scoped>
.progress-ring {
  transition: stroke-dashoffset 0.3s ease;
}

.progress-text {
  font-size: 11px;
  font-weight: 600;
  fill: var(--color-text);
}
</style>
```

**Step 2: Integrate into Grid component**

Modify `frontend/src/components/Grid.vue`:

```vue
<template>
  <svg :width="width" :height="height">
    <!-- ... existing grid rendering ... -->

    <!-- Agents -->
    <g v-for="agent in agents" :key="agent.id">
      <circle
        :cx="agent.x * cellSize + cellSize / 2"
        :cy="agent.y * cellSize + cellSize / 2"
        :r="cellSize * 0.35"
        :fill="agent.color"
        class="agent"
      />

      <!-- NEW: Progress ring -->
      <ProgressRing
        v-if="getAgentProgress(agent.id)"
        :cx="agent.x * cellSize + cellSize / 2"
        :cy="agent.y * cellSize + cellSize / 2"
        :radius="cellSize * 0.5"
        :ticks-completed="getAgentProgress(agent.id).ticksCompleted"
        :ticks-required="getAgentProgress(agent.id).ticksRequired"
      />
    </g>
  </svg>
</template>

<script setup lang="ts">
import ProgressRing from './ProgressRing.vue'

const props = defineProps<{
  agents: AgentData[]
  interactionProgress: Record<string, ProgressData>
}>()

function getAgentProgress(agentId: string) {
  return props.interactionProgress[agentId] || null
}
</script>
```

**Step 3: Update simulation store**

```typescript
// frontend/src/stores/simulation.ts
interface ProgressData {
  affordanceName: string
  ticksCompleted: number
  ticksRequired: number
  progressRatio: number
}

interface SimulationState {
  // ... existing fields ...
  interactionProgress: Record<string, ProgressData>
}

// In WebSocket handler
socket.on('state_update', (data) => {
  state.interactionProgress = data.interaction_progress || {}
  // ...
})
```

**Step 4: Test in browser**

Verify:

- Yellow ring appears when agent starts interaction
- Ring fills as agent progresses (e.g., 2/5 on Bed)
- Ring turns green when complete
- Text shows "3/5" progress

**Step 5: Commit**

```bash
git add frontend/src/components/ProgressRing.vue frontend/src/components/Grid.vue frontend/src/stores/simulation.ts
git commit -m "feat: add interaction progress ring around agents

- ProgressRing component shows ticks_completed / ticks_required
- Yellow ring during progress, green when complete
- SVG circle with stroke-dashoffset animation
- Text label below agent showing progress
- Updates in real-time from WebSocket state"
```

---

## Phase 5: Config and Training Integration

### Task 5.1: Add Temporal Mechanics Config

**Files:**

- Create: `configs/townlet_level_2_5_temporal.yaml`
- Modify: `src/hamlet/demo/runner.py` (parse new config fields)

**Step 1: Create temporal mechanics config**

```yaml
# configs/townlet_level_2_5_temporal.yaml
# Townlet Level 2.5: Temporal Mechanics + Multi-Interaction Affordances
#
# Adds time-based mechanics on top of Level 2 POMDP:
# - 24-tick day/night cycle with operating hours
# - Multi-interaction affordances (75% linear, 25% completion bonus)
# - Dynamic affordances (CoffeeShop ↔ Bar)
# - Agent learns temporal planning and opportunity cost

# Environment configuration
environment:
  grid_size: 8
  partial_observability: true
  vision_range: 2

  # NEW: Temporal mechanics
  enable_temporal_mechanics: true
  ticks_per_day: 24

# Population configuration
population:
  num_agents: 1
  learning_rate: 0.0001  # Lower for recurrent + temporal complexity
  gamma: 0.99
  replay_buffer_capacity: 10000
  network_type: recurrent  # LSTM for POMDP + temporal memory

# Curriculum configuration
curriculum:
  max_steps_per_episode: 500
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 1000

# Exploration configuration
exploration:
  embed_dim: 128
  initial_intrinsic_weight: 0.1
  variance_threshold: 100.0
  survival_window: 100
  epsilon_start: 1.0
  epsilon_min: 0.01
  epsilon_decay: 0.999

# Training configuration
training:
  device: cuda
```

**Step 2: Update runner to parse temporal config**

Modify `src/hamlet/demo/runner.py`:

```python
def create_environment_from_config(config: dict, device: torch.device):
    """Create environment from config dict."""
    env_config = config['environment']

    # Build affordances dict
    affordances = {}
    if env_config.get('enable_temporal_mechanics', False):
        # Use dynamic affordances for temporal mode
        from townlet.environment.dynamic_affordances import DYNAMIC_AFFORDANCE_POSITIONS
        # Also add static affordances
        affordances.update({
            'Bed': torch.tensor([2, 2]),
            'Shower': torch.tensor([5, 1]),
            'HomeMeal': torch.tensor([1, 6]),
            'Job': torch.tensor([6, 6]),
            'Hospital': torch.tensor([7, 7]),
            # ... rest of static affordances ...
        })
        affordances.update(DYNAMIC_AFFORDANCE_POSITIONS)
    else:
        # Use original static affordances
        affordances = {
            'Bed': torch.tensor([2, 2]),
            # ... etc ...
        }

    return VectorizedHamletEnv(
        num_agents=config['population']['num_agents'],
        grid_size=env_config['grid_size'],
        affordances=affordances,
        device=device,
        partial_observability=env_config.get('partial_observability', False),
        vision_range=env_config.get('vision_range', 2),
        enable_temporal_mechanics=env_config.get('enable_temporal_mechanics', False),
    )
```

**Step 3: Run training with new config**

```bash
cd /home/john/hamlet/.worktrees/temporal-mechanics
python -m hamlet.demo.runner configs/townlet_level_2_5_temporal.yaml demo_temporal.db checkpoints_temporal 5000
```

Expected: Training starts, time cycles through 0-23

**Step 4: Commit**

```bash
git add configs/townlet_level_2_5_temporal.yaml src/hamlet/demo/runner.py
git commit -m "feat: add Level 2.5 temporal mechanics config

- Enable temporal mechanics flag in config
- Dynamic affordances integration
- Lower learning rate for increased complexity
- Config documentation explains temporal features"
```

---

### Task 5.2: Fix pyproject.toml in Main Repo

**Files:**

- Modify: `/home/john/hamlet/pyproject.toml` (in main branch, not worktree)

**Step 1: Switch to main branch**

```bash
cd /home/john/hamlet
git checkout main
```

**Step 2: Apply pyproject.toml fix**

```python
# Edit pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/townlet"]
```

**Step 3: Commit to main**

```bash
git add pyproject.toml
git commit -m "fix: add hatchling package config for townlet

- Specify src/townlet as package location
- Fixes 'Unable to determine which files to ship' build error
- Required for uv sync in worktrees"
```

**Step 4: Switch back to worktree**

```bash
cd /home/john/hamlet/.worktrees/temporal-mechanics
```

---

## Verification & Testing

### Task 6.1: Integration Test - Full Temporal Cycle

**Files:**

- Create: `tests/test_townlet/test_temporal_integration.py`

**Step 1: Write integration test**

```python
# tests/test_townlet/test_temporal_integration.py
import pytest
import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.environment.dynamic_affordances import DYNAMIC_AFFORDANCE_POSITIONS


@pytest.fixture
def temporal_env():
    """Full temporal environment with all affordances."""
    affordances = {
        'Bed': torch.tensor([2, 2]),
        'Job': torch.tensor([5, 5]),
        **DYNAMIC_AFFORDANCE_POSITIONS,
    }
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        affordances=affordances,
        device=torch.device('cpu'),
        enable_temporal_mechanics=True,
    )


def test_full_day_night_cycle(temporal_env):
    """Simulate 24-hour cycle with affordance usage."""
    env = temporal_env
    env.reset()

    # Morning: Go to Job (tick 8-12)
    env.time_of_day = 8
    env.positions[0] = torch.tensor([5, 5])  # On Job
    env.meters[0, 0] = 1.0  # Full energy

    for tick in range(4):
        env.step(torch.tensor([4]))  # INTERACT

    # Should have completed Job (4 ticks)
    assert env.interaction_progress[0] == 0  # Reset after completion
    money_gained = env.meters[0, 3].item()
    assert abs(money_gained - 0.225) < 0.01  # $22.5 gained

    # Evening: Go to CoffeeShop/Bar position (tick 18-20)
    env.time_of_day = 18
    env.positions[0] = torch.tensor([3, 5])

    # Should be Bar now (not CoffeeShop)
    env.step(torch.tensor([4]))

    # Bar gives mood boost
    mood_boost = env.meters[0, 4].item()
    assert mood_boost > 0.05  # Mood increased


def test_observation_dimensionality(temporal_env):
    """Verify observation includes temporal features."""
    obs = temporal_env.reset()

    # For full observability + temporal:
    # 64 (grid) + 8 (meters) + 2 (time, progress) = 74
    assert obs.shape == (1, 74)

    # time_of_day feature
    assert 0.0 <= obs[0, -2] <= 1.0

    # interaction_progress feature
    assert obs[0, -1] == 0.0  # No progress at start
```

**Step 2: Run integration test**

```bash
uv run pytest tests/test_townlet/test_temporal_integration.py -v
```

Expected: `2 passed`

**Step 3: Commit**

```bash
git add tests/test_townlet/test_temporal_integration.py
git commit -m "test: add integration tests for temporal mechanics

- Test full 24-hour cycle with Job and Bar usage
- Verify observation dimensionality (74 dims)
- Verify time and progress features in observation"
```

---

### Task 6.2: Documentation Update

**Files:**

- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Update CLAUDE.md**

Add section:

```markdown
## Progressive Complexity Levels

**Level 2.5** (✅ Implemented - temporal mechanics):
- Partial observability (5×5 vision) + LSTM memory
- 24-tick day/night cycle with operating hours
- Multi-interaction affordances (75% linear, 25% completion bonus)
- Dynamic affordances (CoffeeShop ↔ Bar transformation)
- Agent learns temporal planning and opportunity cost
- Config: `configs/townlet_level_2_5_temporal.yaml`
```

**Step 2: Update README.md**

Add temporal mechanics description:

```markdown
### Level 2.5: Temporal Mechanics

Agents must learn **when** to act, not just **what** to do:

- **24-tick day/night cycle**: Affordances have operating hours (Job: 8am-6pm, Bar: 6pm-4am)
- **Multi-interaction affordances**: Agent must commit multiple ticks for full benefit
  - Example: Bed requires 5 ticks - agent gets 75% progressively, 25% bonus on completion
  - **Early exit allowed**: Agent can leave after 2 ticks and keep partial benefit ("quick rinse")
- **Dynamic affordances**: Same position = different affordance by time (CoffeeShop → Bar at 6pm)
- **Emergent behaviors**: "Power napping" (2/5 ticks on Bed), "shift splitting" (Job twice), "pre-work prep" (CoffeeShop → Job)

**Pedagogical value**: Teaches temporal planning, opportunity cost, and commitment decisions through game-like mechanics.
```

**Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: add Level 2.5 temporal mechanics to documentation

- Update CLAUDE.md with Level 2.5 description
- Update README.md with temporal mechanics explanation
- Document emergent behaviors and pedagogical value"
```

---

## Summary

**Implementation complete!**

Phases implemented:

1. ✅ Environment backend (multi-interaction tracking)
2. ✅ Time system (24-tick cycle, action masking)
3. ✅ Dynamic affordances (CoffeeShop/Bar transformation)
4. ✅ UI visualization (gradient bar, progress rings)
5. ✅ Config and training integration

**Files created:**

- `src/townlet/environment/affordance_config.py`
- `src/townlet/environment/dynamic_affordances.py`
- `frontend/src/components/TimeOfDayBar.vue`
- `frontend/src/components/ProgressRing.vue`
- `configs/townlet_level_2_5_temporal.yaml`
- 4 test files

**Files modified:**

- `src/townlet/environment/vectorized_env.py`
- `src/hamlet/web/renderer.py`
- `src/hamlet/demo/runner.py`
- `frontend/src/components/Grid.vue`
- `frontend/src/components/MetersPanel.vue`
- `frontend/src/stores/simulation.ts`
- `CLAUDE.md`, `README.md`

**Total commits:** 12 (one per logical feature)

**Next steps:**

1. Train for 5000 episodes: `python -m hamlet.demo.runner configs/townlet_level_2_5_temporal.yaml demo_temporal.db checkpoints_temporal 5000`
2. Observe emergent behaviors (power napping, shift splitting, pre-work prep)
3. Document teachable moments in `docs/teachable_moments/`
4. Consider additional dynamic affordances or time-based difficulty scaling
