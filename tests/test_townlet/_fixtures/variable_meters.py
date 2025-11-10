"""Fixtures for TASK-001 variable meter configurations."""

from __future__ import annotations

import copy
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import yaml

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.universe.compiled import CompiledUniverse

# =============================================================================
# TASK-001: VARIABLE METER CONFIG FIXTURES
# =============================================================================


@pytest.fixture
def task001_config_4meter(tmp_path: Path, test_config_pack_path: Path) -> Path:
    """Create temporary 4-meter config pack for TASK-001 testing.

    Meters: energy, health, money, mood
    Use ONLY for: TASK-001 variable meter tests
    Do NOT use for: L0 curriculum (use separate curriculum fixtures)

    Args:
        tmp_path: pytest's temporary directory
        test_config_pack_path: Path to source config pack

    Returns:
        Path to temporary 4-meter config pack directory
    """
    config_4m = tmp_path / "config_4m"
    shutil.copytree(test_config_pack_path, config_4m)

    # Create 4-meter bars.yaml
    bars_config = {
        "version": "2.0",
        "description": "4-meter test universe",
        "bars": [
            {
                "name": "energy",
                "index": 0,
                "tier": "pivotal",
                "range": [0.0, 1.0],
                "initial": 1.0,
                "base_depletion": 0.005,
                "description": "Energy level",
            },
            {
                "name": "health",
                "index": 1,
                "tier": "pivotal",
                "range": [0.0, 1.0],
                "initial": 1.0,
                "base_depletion": 0.0,
                "description": "Health status",
            },
            {
                "name": "money",
                "index": 2,
                "tier": "resource",
                "range": [0.0, 1.0],
                "initial": 0.5,
                "base_depletion": 0.0,
                "description": "Financial resources",
            },
            {
                "name": "mood",
                "index": 3,
                "tier": "secondary",
                "range": [0.0, 1.0],
                "initial": 0.7,
                "base_depletion": 0.001,
                "description": "Mood state",
            },
        ],
        "terminal_conditions": [
            {"meter": "energy", "operator": "<=", "value": 0.0, "description": "Death by energy depletion"},
            {"meter": "health", "operator": "<=", "value": 0.0, "description": "Death by health failure"},
        ],
    }

    with open(config_4m / "bars.yaml", "w") as f:
        yaml.safe_dump(bars_config, f)

    # Simplify cascades.yaml
    cascades_config = {
        "version": "2.0",
        "description": "Simplified cascades for 4-meter testing",
        "math_type": "gradient_penalty",
        "modulations": [],
        "cascades": [
            {
                "name": "low_mood_hits_energy",
                "category": "secondary_to_pivotal",
                "description": "Low mood drains energy",
                "source": "mood",
                "source_index": 3,
                "target": "energy",
                "target_index": 0,
                "threshold": 0.2,
                "strength": 0.01,
            }
        ],
        "execution_order": ["secondary_to_pivotal"],
    }

    with open(config_4m / "cascades.yaml", "w") as f:
        yaml.safe_dump(cascades_config, f)

    # Create affordances.yaml with FULL 14-affordance vocabulary but only using 4 meters
    # This maintains observation vocabulary consistency while validating meter references
    # Note: Only enabled_affordances (Bed, Hospital, HomeMeal=FastFood, Job) will be deployed
    affordances_config = {
        "version": "2.0",
        "description": "4-meter test affordances (full vocabulary, 4-meter compatible)",
        "status": "TEST",
        "affordances": [
            {
                "id": "0",
                "name": "Bed",
                "category": "energy",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.05}],
                "effect_pipeline": {"instant": [{"meter": "energy", "amount": 0.50}, {"meter": "health", "amount": 0.02}]},
                "operating_hours": [0, 24],
            },
            {
                "id": "1",
                "name": "LuxuryBed",
                "category": "energy",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.11}],
                "effect_pipeline": {"instant": [{"meter": "energy", "amount": 0.75}, {"meter": "health", "amount": 0.05}]},
                "operating_hours": [0, 24],
            },
            {
                "id": "2",
                "name": "Shower",
                "category": "hygiene",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.03}],
                "effect_pipeline": {"instant": [{"meter": "mood", "amount": 0.20}]},  # Use mood instead of hygiene
                "operating_hours": [0, 24],
            },
            {
                "id": "3",
                "name": "HomeMeal",
                "category": "food",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.04}],
                "effect_pipeline": {"instant": [{"meter": "energy", "amount": 0.20}, {"meter": "mood", "amount": 0.10}]},
                "operating_hours": [0, 24],
            },
            {
                "id": "4",
                "name": "FastFood",
                "category": "food",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.06}],
                "effect_pipeline": {"instant": [{"meter": "energy", "amount": 0.30}]},
                "operating_hours": [0, 24],
            },
            {
                "id": "5",
                "name": "Restaurant",
                "category": "food",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.12}],
                "effect_pipeline": {"instant": [{"meter": "energy", "amount": 0.40}, {"meter": "mood", "amount": 0.15}]},
                "operating_hours": [11, 22],
            },
            {
                "id": "6",
                "name": "Gym",
                "category": "fitness",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.08}, {"meter": "energy", "amount": 0.10}],
                "effect_pipeline": {"instant": [{"meter": "health", "amount": 0.15}, {"meter": "mood", "amount": 0.05}]},
                "operating_hours": [6, 22],
            },
            {
                "id": "7",
                "name": "Hospital",
                "category": "health",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.15}],
                "effect_pipeline": {"instant": [{"meter": "health", "amount": 0.60}]},
                "operating_hours": [0, 24],
            },
            {
                "id": "8",
                "name": "Job",
                "category": "income",
                "interaction_type": "instant",
                "costs": [{"meter": "energy", "amount": 0.15}],
                "effect_pipeline": {"instant": [{"meter": "money", "amount": 0.225}, {"meter": "mood", "amount": -0.05}]},
                "operating_hours": [8, 18],
            },
            {
                "id": "9",
                "name": "Park",
                "category": "leisure",
                "interaction_type": "instant",
                "costs": [],
                "effect_pipeline": {"instant": [{"meter": "mood", "amount": 0.20}]},
                "operating_hours": [6, 20],
            },
            {
                "id": "10",
                "name": "Library",
                "category": "leisure",
                "interaction_type": "instant",
                "costs": [],
                "effect_pipeline": {"instant": [{"meter": "mood", "amount": 0.15}]},
                "operating_hours": [8, 20],
            },
            {
                "id": "11",
                "name": "Bar",
                "category": "social",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.15}],
                "effect_pipeline": {"instant": [{"meter": "mood", "amount": 0.25}]},
                "operating_hours": [18, 28],
            },
            {
                "id": "12",
                "name": "Recreation",
                "category": "leisure",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.10}],
                "effect_pipeline": {"instant": [{"meter": "mood", "amount": 0.30}]},
                "operating_hours": [12, 24],
            },
            {
                "id": "13",
                "name": "SocialEvent",
                "category": "social",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.08}],
                "effect_pipeline": {"instant": [{"meter": "mood", "amount": 0.20}]},
                "operating_hours": [18, 23],
            },
        ],
    }

    with open(config_4m / "affordances.yaml", "w") as f:
        yaml.safe_dump(affordances_config, f)

    cues_config = {
        "version": "1.0",
        "description": "4-meter cues for variable meter tests",
        "status": "TEST",
        "simple_cues": [
            {
                "cue_id": "looks_exhausted",
                "name": "Looks Exhausted",
                "category": "energy",
                "visibility": "public",
                "condition": {"meter": "energy", "operator": "<", "threshold": 0.25},
                "description": "Agent appears visibly tired",
            },
            {
                "cue_id": "looks_sickly",
                "name": "Looks Sickly",
                "category": "health",
                "visibility": "public",
                "condition": {"meter": "health", "operator": "<", "threshold": 0.4},
                "description": "Agent shows signs of illness",
            },
            {
                "cue_id": "looks_broke",
                "name": "Looks Broke",
                "category": "money",
                "visibility": "public",
                "condition": {"meter": "money", "operator": "<", "threshold": 0.2},
                "description": "Agent checks pockets frequently",
            },
            {
                "cue_id": "looks_sad",
                "name": "Looks Sad",
                "category": "mood",
                "visibility": "public",
                "condition": {"meter": "mood", "operator": "<", "threshold": 0.3},
                "description": "Agent moves listlessly",
            },
        ],
        "compound_cues": [],
    }

    with open(config_4m / "cues.yaml", "w") as f:
        yaml.safe_dump(cues_config, f, sort_keys=False)

    # Generate matching variables_reference.yaml for 4 meters
    # Must match bars_config meter count to avoid VFS/bars mismatch
    vfs_config = {
        "version": "1.0",
        "variables": [
            # Grid encoding (64 dims for 8×8 grid)
            {
                "id": "grid_encoding",
                "scope": "agent",
                "type": "vecNf",
                "dims": 64,
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": [0.0] * 64,
                "description": "8×8 grid encoding",
            },
            # Local window for POMDP (5×5 = 25 cells)
            {
                "id": "local_window",
                "scope": "agent",
                "type": "vecNf",
                "dims": 25,
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": [0.0] * 25,
                "description": "5×5 local window for POMDP",
            },
            # Position (2 dims)
            {
                "id": "position",
                "scope": "agent",
                "type": "vecNf",
                "dims": 2,
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": [0.0, 0.0],
                "description": "Normalized agent position (x, y)",
            },
            # 4 meters: energy, health, money, mood
            {
                "id": "energy",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "health",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "money",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 0.0,
            },
            {
                "id": "mood",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            # Affordance at position (15 dims)
            {
                "id": "affordance_at_position",
                "scope": "agent",
                "type": "vecNf",
                "dims": 15,
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": [0.0] * 14 + [1.0],
            },
            # Temporal features (4 scalars)
            {
                "id": "time_sin",
                "scope": "global",
                "type": "scalar",
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 0.0,
            },
            {
                "id": "time_cos",
                "scope": "global",
                "type": "scalar",
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 1.0,
            },
            {
                "id": "interaction_progress",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 0.0,
            },
            {
                "id": "lifetime_progress",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 0.0,
            },
        ],
        "exposed_observations": [
            {"id": "obs_grid_encoding", "source_variable": "grid_encoding", "exposed_to": ["agent"], "shape": [64], "normalization": None},
            {"id": "obs_local_window", "source_variable": "local_window", "exposed_to": ["agent"], "shape": [25], "normalization": None},
            {
                "id": "obs_position",
                "source_variable": "position",
                "exposed_to": ["agent"],
                "shape": [2],
                "normalization": {"kind": "minmax", "min": [0.0, 0.0], "max": [1.0, 1.0]},
            },
            {
                "id": "obs_energy",
                "source_variable": "energy",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_health",
                "source_variable": "health",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {"id": "obs_money", "source_variable": "money", "exposed_to": ["agent"], "shape": [], "normalization": None},
            {
                "id": "obs_mood",
                "source_variable": "mood",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_affordance_at_position",
                "source_variable": "affordance_at_position",
                "exposed_to": ["agent"],
                "shape": [15],
                "normalization": None,
            },
            {"id": "obs_time_sin", "source_variable": "time_sin", "exposed_to": ["agent"], "shape": [], "normalization": None},
            {"id": "obs_time_cos", "source_variable": "time_cos", "exposed_to": ["agent"], "shape": [], "normalization": None},
            {
                "id": "obs_interaction_progress",
                "source_variable": "interaction_progress",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_lifetime_progress",
                "source_variable": "lifetime_progress",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
        ],
    }

    with open(config_4m / "variables_reference.yaml", "w") as f:
        yaml.safe_dump(vfs_config, f, sort_keys=False)

    return config_4m


@pytest.fixture
def task001_config_12meter(tmp_path: Path, test_config_pack_path: Path) -> Path:
    """Create temporary 12-meter config pack for TASK-001 testing.

    Meters: 8 standard + reputation, skill, spirituality, community_trust
    Use ONLY for: TASK-001 variable meter scaling tests
    Do NOT use for: L2 curriculum (use separate curriculum fixtures)

    Args:
        tmp_path: pytest's temporary directory
        test_config_pack_path: Path to source config pack

    Returns:
        Path to temporary 12-meter config pack directory
    """
    config_12m = tmp_path / "config_12m"
    shutil.copytree(test_config_pack_path, config_12m)

    # Load existing 8-meter bars
    with open(test_config_pack_path / "bars.yaml") as f:
        bars_8m = yaml.safe_load(f)

    # Add 4 new meters
    extra_meters = [
        {
            "name": "reputation",
            "index": 8,
            "tier": "secondary",
            "range": [0.0, 1.0],
            "initial": 0.5,
            "base_depletion": 0.002,
            "description": "Social reputation",
        },
        {
            "name": "skill",
            "index": 9,
            "tier": "secondary",
            "range": [0.0, 1.0],
            "initial": 0.3,
            "base_depletion": 0.001,
            "description": "Professional skills",
        },
        {
            "name": "spirituality",
            "index": 10,
            "tier": "secondary",
            "range": [0.0, 1.0],
            "initial": 0.6,
            "base_depletion": 0.002,
            "description": "Spiritual wellbeing",
        },
        {
            "name": "community_trust",
            "index": 11,
            "tier": "secondary",
            "range": [0.0, 1.0],
            "initial": 0.7,
            "base_depletion": 0.001,
            "description": "Community trust level",
        },
    ]

    bars_12m = copy.deepcopy(bars_8m)  # Deep copy to avoid modifying original
    bars_12m["bars"].extend(extra_meters)

    with open(config_12m / "bars.yaml", "w") as f:
        yaml.safe_dump(bars_12m, f)

    # Generate matching variables_reference.yaml for 12 meters
    # Must match bars_config meter count to avoid VFS/bars mismatch
    vfs_config = {
        "version": "1.0",
        "variables": [
            # Grid encoding (64 dims for 8×8 grid)
            {
                "id": "grid_encoding",
                "scope": "agent",
                "type": "vecNf",
                "dims": 64,
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": [0.0] * 64,
                "description": "8×8 grid encoding",
            },
            # Local window for POMDP (5×5 = 25 cells)
            {
                "id": "local_window",
                "scope": "agent",
                "type": "vecNf",
                "dims": 25,
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": [0.0] * 25,
                "description": "5×5 local window for POMDP",
            },
            # Position (2 dims)
            {
                "id": "position",
                "scope": "agent",
                "type": "vecNf",
                "dims": 2,
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": [0.0, 0.0],
                "description": "Normalized agent position (x, y)",
            },
            # 8 standard meters
            {
                "id": "energy",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "health",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "satiation",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "money",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 0.0,
            },
            {
                "id": "mood",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "social",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "fitness",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "hygiene",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            # 4 additional meters
            {
                "id": "reputation",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 0.5,
            },
            {
                "id": "skill",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 0.3,
            },
            {
                "id": "spirituality",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 0.6,
            },
            {
                "id": "community_trust",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 0.7,
            },
            # Affordance at position (15 dims)
            {
                "id": "affordance_at_position",
                "scope": "agent",
                "type": "vecNf",
                "dims": 15,
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": [0.0] * 14 + [1.0],
            },
            # Temporal features (4 scalars)
            {
                "id": "time_sin",
                "scope": "global",
                "type": "scalar",
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 0.0,
            },
            {
                "id": "time_cos",
                "scope": "global",
                "type": "scalar",
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 1.0,
            },
            {
                "id": "interaction_progress",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 0.0,
            },
            {
                "id": "lifetime_progress",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 0.0,
            },
        ],
        "exposed_observations": [
            {"id": "obs_grid_encoding", "source_variable": "grid_encoding", "exposed_to": ["agent"], "shape": [64], "normalization": None},
            {"id": "obs_local_window", "source_variable": "local_window", "exposed_to": ["agent"], "shape": [25], "normalization": None},
            {
                "id": "obs_position",
                "source_variable": "position",
                "exposed_to": ["agent"],
                "shape": [2],
                "normalization": {"kind": "minmax", "min": [0.0, 0.0], "max": [1.0, 1.0]},
            },
            # 8 standard meters
            {
                "id": "obs_energy",
                "source_variable": "energy",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_health",
                "source_variable": "health",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_satiation",
                "source_variable": "satiation",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {"id": "obs_money", "source_variable": "money", "exposed_to": ["agent"], "shape": [], "normalization": None},
            {
                "id": "obs_mood",
                "source_variable": "mood",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_social",
                "source_variable": "social",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_fitness",
                "source_variable": "fitness",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_hygiene",
                "source_variable": "hygiene",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            # 4 additional meters
            {
                "id": "obs_reputation",
                "source_variable": "reputation",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_skill",
                "source_variable": "skill",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_spirituality",
                "source_variable": "spirituality",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_community_trust",
                "source_variable": "community_trust",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            # Affordance and temporal
            {
                "id": "obs_affordance_at_position",
                "source_variable": "affordance_at_position",
                "exposed_to": ["agent"],
                "shape": [15],
                "normalization": None,
            },
            {"id": "obs_time_sin", "source_variable": "time_sin", "exposed_to": ["agent"], "shape": [], "normalization": None},
            {"id": "obs_time_cos", "source_variable": "time_cos", "exposed_to": ["agent"], "shape": [], "normalization": None},
            {
                "id": "obs_interaction_progress",
                "source_variable": "interaction_progress",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_lifetime_progress",
                "source_variable": "lifetime_progress",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
        ],
    }

    with open(config_12m / "variables_reference.yaml", "w") as f:
        yaml.safe_dump(vfs_config, f, sort_keys=False)

    return config_12m


@pytest.fixture
def task001_env_4meter(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    cpu_device: torch.device,
    task001_config_4meter: Path,
) -> VectorizedHamletEnv:
    """4-meter environment for TASK-001 testing.

    Args:
        cpu_device: CPU device for deterministic behavior
        task001_config_4meter: Path to 4-meter config pack

    Returns:
        VectorizedHamletEnv instance with 4 meters
    """
    universe = compile_universe(task001_config_4meter)
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=cpu_device,
    )


@pytest.fixture
def task001_env_4meter_pomdp(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    cpu_device: torch.device,
    task001_config_4meter: Path,
) -> VectorizedHamletEnv:
    """4-meter POMDP environment for TASK-001 recurrent network testing.

    Args:
        cpu_device: CPU device for deterministic behavior
        task001_config_4meter: Path to 4-meter config pack

    Returns:
        VectorizedHamletEnv instance with 4 meters and partial observability
    """
    pomdp_config = task001_config_4meter.parent / "config_4m_pomdp"
    shutil.copytree(task001_config_4meter, pomdp_config)

    training_yaml = pomdp_config / "training.yaml"
    with open(training_yaml) as f:
        training_config = yaml.safe_load(f)

    training_env = training_config.get("environment", {})
    training_env["partial_observability"] = True
    training_env["vision_range"] = 2
    training_config["environment"] = training_env

    with open(training_yaml, "w") as f:
        yaml.safe_dump(training_config, f, sort_keys=False)

    universe = compile_universe(pomdp_config)
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=cpu_device,
    )


@pytest.fixture
def task001_env_12meter(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    cpu_device: torch.device,
    task001_config_12meter: Path,
) -> VectorizedHamletEnv:
    """12-meter environment for TASK-001 testing.

    Args:
        cpu_device: CPU device for deterministic behavior
        task001_config_12meter: Path to 12-meter config pack

    Returns:
        VectorizedHamletEnv instance with 12 meters
    """
    universe = compile_universe(task001_config_12meter)
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=cpu_device,
    )


@pytest.fixture
def task001_env_12meter_pomdp(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    cpu_device: torch.device,
    task001_config_12meter: Path,
) -> VectorizedHamletEnv:
    """12-meter POMDP environment for TASK-001 testing."""

    pomdp_config = task001_config_12meter.parent / "config_12m_pomdp"
    shutil.copytree(task001_config_12meter, pomdp_config)

    training_yaml = pomdp_config / "training.yaml"
    with open(training_yaml) as f:
        training_config = yaml.safe_load(f)

    training_env = training_config.get("environment", {})
    training_env["partial_observability"] = True
    training_env["vision_range"] = 2
    training_config["environment"] = training_env

    with open(training_yaml, "w") as f:
        yaml.safe_dump(training_config, f, sort_keys=False)

    universe = compile_universe(pomdp_config)
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=cpu_device,
    )
