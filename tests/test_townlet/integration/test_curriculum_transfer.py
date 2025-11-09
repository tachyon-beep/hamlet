"""Tests for checkpoint transfer across curriculum levels."""

from pathlib import Path

import torch

from tests.test_townlet.utils.builders import make_vectorized_env_from_pack


def test_l0_and_l1_have_same_action_dim():
    """L0 and L1 should have same action_dim (enables checkpoint transfer)."""
    env_l0 = make_vectorized_env_from_pack(
        Path("configs/L0_0_minimal"),
        num_agents=1,
        device=torch.device("cpu"),
    )

    env_l1 = make_vectorized_env_from_pack(
        Path("configs/L1_full_observability"),
        num_agents=1,
        device=torch.device("cpu"),
    )

    # CRITICAL: Same action_dim enables checkpoint transfer
    assert env_l0.action_dim == env_l1.action_dim, (
        f"L0 and L1 must have same action_dim for checkpoint transfer. " f"L0: {env_l0.action_dim}, L1: {env_l1.action_dim}"
    )


def test_l0_and_l1_share_global_vocabulary():
    """L0 and L1 should share same action vocabulary (from global_actions.yaml)."""
    env_l0 = make_vectorized_env_from_pack(
        Path("configs/L0_0_minimal"),
        num_agents=1,
        device=torch.device("cpu"),
    )

    env_l1 = make_vectorized_env_from_pack(
        Path("configs/L1_full_observability"),
        num_agents=1,
        device=torch.device("cpu"),
    )

    # Both should have same custom actions from global_actions.yaml
    l0_action_names = {action.name for action in env_l0.action_space.actions}
    l1_action_names = {action.name for action in env_l1.action_space.actions}

    assert l0_action_names == l1_action_names, f"L0 and L1 must share action vocabulary. " f"L0: {l0_action_names}, L1: {l1_action_names}"

    # Verify custom actions present
    expected_custom = {"REST", "MEDITATE"}
    assert expected_custom.issubset(l0_action_names), f"L0 should have custom actions: {expected_custom}"
    assert expected_custom.issubset(l1_action_names), f"L1 should have custom actions: {expected_custom}"
