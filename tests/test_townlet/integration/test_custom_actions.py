"""Integration tests for custom actions (REST, MEDITATE)."""

from pathlib import Path

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.fixture
def cpu_device():
    """CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def test_config_pack_path() -> Path:
    """Return path to test configuration pack."""
    # Use configs/L1_full_observability which has global_actions.yaml
    return Path(__file__).parent.parent.parent.parent / "configs" / "L1_full_observability"


def test_rest_action_restores_energy_and_mood(cpu_device, test_config_pack_path):
    """REST action should restore energy and mood (negative costs)."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=5,  # Smaller grid for fast testing
        partial_observability=False,
        vision_range=5,
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        config_pack_path=test_config_pack_path,
        device=cpu_device,
        agent_lifespan=1000,
    )

    env.reset()

    # Find REST action ID
    rest_action = env.action_space.get_action_by_name("REST")
    rest_action_id = rest_action.id

    # Drain energy and mood
    # Energy is at index 0, mood is at index 4 (see bars.yaml)
    env.meters[0, 0] = 0.5  # energy
    env.meters[0, 4] = 0.3  # mood
    initial_energy = env.meters[0, 0].item()
    initial_mood = env.meters[0, 4].item()

    # Execute REST action
    actions = torch.tensor([rest_action_id], device=env.device)
    env.step(actions)

    final_energy = env.meters[0, 0].item()
    final_mood = env.meters[0, 4].item()

    # REST restores energy (+0.002) and mood (+0.01)
    # But passive depletion still happens (energy -0.005, mood -0.001 base)
    # Net: energy should decrease by ~0.003, mood should increase by ~0.009
    # We just verify that REST effect was applied (not overcome by depletion entirely)
    energy_change = final_energy - initial_energy
    mood_change = final_mood - initial_mood

    # Energy: REST +0.002, depletion -0.005 = -0.003 net (less loss than without REST)
    assert -0.005 < energy_change < 0.002, f"Energy change unexpected: {energy_change} (expected -0.003)"
    # Mood: REST +0.01, depletion -0.001 = +0.009 net (should increase)
    assert mood_change > 0, f"Mood should increase: {initial_mood} -> {final_mood}"


def test_meditate_action_restores_mood(cpu_device, test_config_pack_path):
    """MEDITATE action should restore mood (via effects)."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=5,
        partial_observability=False,
        vision_range=5,
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        config_pack_path=test_config_pack_path,
        device=cpu_device,
        agent_lifespan=1000,
    )

    env.reset()

    # Find MEDITATE action
    meditate_action = env.action_space.get_action_by_name("MEDITATE")
    meditate_action_id = meditate_action.id

    # Drain mood
    # Mood is at index 4 (see bars.yaml)
    env.meters[0, 4] = 0.3
    initial_mood = env.meters[0, 4].item()

    # Execute MEDITATE
    actions = torch.tensor([meditate_action_id], device=env.device)
    env.step(actions)

    final_mood = env.meters[0, 4].item()

    # MEDITATE: costs energy (0.001), restores mood via effects (+0.02)
    # Base mood depletion: -0.001
    # Net mood: +0.02 - 0.001 = +0.019 (should increase significantly)
    mood_change = final_mood - initial_mood
    assert mood_change > 0.015, f"Mood should increase by ~0.019: {mood_change}"
