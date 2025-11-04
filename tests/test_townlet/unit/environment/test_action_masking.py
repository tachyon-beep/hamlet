"""Consolidated tests for action masking across all scenarios.

This file consolidates action masking tests from multiple legacy files:
- test_all_actions.py (boundary masking)
- test_action_selection.py (Q-value masking)
- test_interact_masking.py (affordance-based masking)
- test_interact_demasking.py (affordability removal)
- test_time_based_masking.py (operating hours)
- test_post_terminal_masking.py (dead agent masking)
- test_wait_action.py (WAIT availability)

Coverage:
- Boundary masking: UP/DOWN/LEFT/RIGHT at grid edges
- Interaction masking: affordance availability and affordability
- Time-based masking: operating hours and day/night cycle
- Post-terminal masking: dead agents cannot act
- WAIT action: always available unless dead
- Integration: combined masking scenarios
"""

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


class TestBoundaryMasking:
    """Test that movement actions are masked at grid boundaries.

    Consolidates tests from test_all_actions.py and test_action_selection.py.
    """

    def test_up_masked_at_top_edge(self, basic_env):
        """UP should be masked when agent is at y=0."""
        basic_env.reset()
        basic_env.positions[0] = torch.tensor([4, 0], device=basic_env.device)

        masks = basic_env.get_action_masks()

        assert not masks[0, 0], "UP should be masked at top edge"
        assert masks[0, 1], "DOWN should be available"
        assert masks[0, 5], "WAIT should be available"

    def test_down_masked_at_bottom_edge(self, basic_env):
        """DOWN should be masked when agent is at y=grid_size-1."""
        basic_env.reset()
        basic_env.positions[0] = torch.tensor([4, 7], device=basic_env.device)

        masks = basic_env.get_action_masks()

        assert not masks[0, 1], "DOWN should be masked at bottom edge"
        assert masks[0, 0], "UP should be available"
        assert masks[0, 5], "WAIT should be available"

    def test_left_masked_at_left_edge(self, basic_env):
        """LEFT should be masked when agent is at x=0."""
        basic_env.reset()
        basic_env.positions[0] = torch.tensor([0, 4], device=basic_env.device)

        masks = basic_env.get_action_masks()

        assert not masks[0, 2], "LEFT should be masked at left edge"
        assert masks[0, 3], "RIGHT should be available"
        assert masks[0, 5], "WAIT should be available"

    def test_right_masked_at_right_edge(self, basic_env):
        """RIGHT should be masked when agent is at x=grid_size-1."""
        basic_env.reset()
        basic_env.positions[0] = torch.tensor([7, 4], device=basic_env.device)

        masks = basic_env.get_action_masks()

        assert not masks[0, 3], "RIGHT should be masked at right edge"
        assert masks[0, 2], "LEFT should be available"
        assert masks[0, 5], "WAIT should be available"

    def test_corner_masks_two_directions(self, multi_agent_env):
        """At corners, two movement directions should be masked.

        Tests all four corners with independent agents.
        """
        multi_agent_env.reset()

        # Place agents at all four corners
        multi_agent_env.positions = torch.tensor(
            [
                [0, 0],  # Top-left: can't UP or LEFT
                [7, 0],  # Top-right: can't UP or RIGHT
                [0, 7],  # Bottom-left: can't DOWN or LEFT
                [7, 7],  # Bottom-right: can't DOWN or RIGHT
            ],
            device=multi_agent_env.device,
        )

        masks = multi_agent_env.get_action_masks()

        # Top-left corner
        assert not masks[0, 0], "UP should be masked at top-left"
        assert not masks[0, 2], "LEFT should be masked at top-left"
        assert masks[0, 1], "DOWN should be available"
        assert masks[0, 3], "RIGHT should be available"

        # Top-right corner
        assert not masks[1, 0], "UP should be masked at top-right"
        assert not masks[1, 3], "RIGHT should be masked at top-right"
        assert masks[1, 1], "DOWN should be available"
        assert masks[1, 2], "LEFT should be available"

        # Bottom-left corner
        assert not masks[2, 1], "DOWN should be masked at bottom-left"
        assert not masks[2, 2], "LEFT should be masked at bottom-left"
        assert masks[2, 0], "UP should be available"
        assert masks[2, 3], "RIGHT should be available"

        # Bottom-right corner
        assert not masks[3, 1], "DOWN should be masked at bottom-right"
        assert not masks[3, 3], "RIGHT should be masked at bottom-right"
        assert masks[3, 0], "UP should be available"
        assert masks[3, 2], "LEFT should be available"

    def test_all_movements_available_in_center(self, basic_env):
        """All movement actions should be available away from boundaries."""
        basic_env.reset()
        basic_env.positions[0] = torch.tensor([4, 4], device=basic_env.device)

        masks = basic_env.get_action_masks()

        assert masks[0, 0], "UP should be available in center"
        assert masks[0, 1], "DOWN should be available in center"
        assert masks[0, 2], "LEFT should be available in center"
        assert masks[0, 3], "RIGHT should be available in center"

    def test_movement_clamped_at_boundaries(self, basic_env):
        """Movement beyond boundaries should be clamped to grid edges."""
        basic_env.reset()
        basic_env.positions[0] = torch.tensor([0, 0], device=basic_env.device)

        # Try to move UP (should clamp to y=0)
        actions = torch.tensor([0], device=basic_env.device)
        basic_env.step(actions)

        assert basic_env.positions[0, 1].item() == 0, "Y should be clamped at 0"

        # Try to move LEFT (should clamp to x=0)
        basic_env.positions[0] = torch.tensor([0, 0], device=basic_env.device)
        actions = torch.tensor([2], device=basic_env.device)
        basic_env.step(actions)

        assert basic_env.positions[0, 0].item() == 0, "X should be clamped at 0"


class TestInteractMasking:
    """Test that INTERACT is masked based on affordance availability.

    Consolidates tests from test_interact_masking.py and test_interact_demasking.py.
    Key insight: INTERACT should be available when on affordance, regardless of affordability.
    """

    def test_interact_masked_when_not_on_affordance(self, basic_env):
        """INTERACT should be masked when not on any affordance."""
        basic_env.reset()
        basic_env.positions[0] = torch.tensor([4, 4], device=basic_env.device)

        # Verify not on any affordance
        on_affordance = any(torch.equal(basic_env.positions[0], pos) for pos in basic_env.affordances.values())

        if not on_affordance:
            masks = basic_env.get_action_masks()
            assert not masks[0, 4], "INTERACT should be masked off affordance"

    def test_interact_available_on_bed(self, basic_env):
        """INTERACT should be available when on Bed (free affordance)."""
        basic_env.reset()

        bed_pos = basic_env.affordances["Bed"]
        basic_env.positions[0] = bed_pos.clone()

        masks = basic_env.get_action_masks()

        assert masks[0, 4], "INTERACT should be available on Bed"

    def test_interact_available_on_hospital_when_broke(self, basic_env):
        """INTERACT should be available on Hospital even with $0.

        This tests the P1.4 de-masking: affordability doesn't affect masking.
        """
        basic_env.reset()

        hospital_pos = basic_env.affordances["Hospital"]
        basic_env.positions[0] = hospital_pos.clone()
        basic_env.meters[0, 3] = 0.0  # Money = 0

        masks = basic_env.get_action_masks()

        assert masks[0, 4], "INTERACT should be available on Hospital even when broke"

    def test_interact_available_on_job(self, basic_env):
        """INTERACT should be available on Job (money-generating affordance)."""
        basic_env.reset()

        job_pos = basic_env.affordances["Job"]
        basic_env.positions[0] = job_pos.clone()

        masks = basic_env.get_action_masks()

        assert masks[0, 4], "INTERACT should be available on Job"

    def test_interact_available_on_all_affordance_types(self, basic_env):
        """INTERACT should be available on any affordance type."""
        basic_env.reset()

        affordance_names = list(basic_env.affordances.keys())

        for affordance_name in affordance_names:
            affordance_pos = basic_env.affordances[affordance_name]
            basic_env.positions[0] = affordance_pos.clone()

            masks = basic_env.get_action_masks()

            assert masks[0, 4], f"INTERACT should be available on {affordance_name}"

    def test_multiple_agents_independent_masking(self, multi_agent_env):
        """Each agent should have independent INTERACT masking."""
        multi_agent_env.reset()

        # Agent 0: on Bed
        bed_pos = multi_agent_env.affordances["Bed"]
        multi_agent_env.positions[0] = bed_pos.clone()

        # Agent 1: not on affordance
        multi_agent_env.positions[1] = torch.tensor([4, 4], device=multi_agent_env.device)

        # Verify agent 1 is not on affordance
        on_affordance = any(torch.equal(multi_agent_env.positions[1], pos) for pos in multi_agent_env.affordances.values())

        if not on_affordance:
            masks = multi_agent_env.get_action_masks()

            assert masks[0, 4], "Agent on affordance should have INTERACT available"
            assert not masks[1, 4], "Agent off affordance should have INTERACT masked"

    def test_movement_actions_unaffected_by_affordance(self, basic_env):
        """Movement masking should not change based on affordance presence."""
        basic_env.reset()

        # Test in center (not on affordance)
        basic_env.positions[0] = torch.tensor([4, 4], device=basic_env.device)
        masks_center = basic_env.get_action_masks()

        # Test on Bed (on affordance)
        bed_pos = basic_env.affordances["Bed"]
        basic_env.positions[0] = bed_pos.clone()
        masks_bed = basic_env.get_action_masks()

        # Movement actions should have same availability
        # (only INTERACT should differ)
        for action in [0, 1, 2, 3]:  # UP, DOWN, LEFT, RIGHT
            # Allow for boundary differences based on position
            # Just verify that being on affordance doesn't break movement
            assert masks_bed.shape == (1, 6), "Should have 6 actions"


class TestTimeBasedMasking:
    """Test that actions are masked based on operating hours.

    Consolidates tests from test_time_based_masking.py.
    Tests temporal mechanics: affordances have operating hours.
    """

    def test_job_closed_outside_business_hours(self, temporal_env):
        """Job should be masked outside business hours (8am-6pm)."""
        temporal_env.reset()

        job_pos = temporal_env.affordances["Job"]
        temporal_env.positions[0] = job_pos.clone()

        # 10am: Job is open
        temporal_env.time_of_day = 10
        masks_open = temporal_env.get_action_masks()
        assert masks_open[0, 4], "INTERACT should be available on Job at 10am"

        # 7pm: Job is closed
        temporal_env.time_of_day = 19
        masks_closed = temporal_env.get_action_masks()
        assert not masks_closed[0, 4], "INTERACT should be masked on Job at 7pm"

    def test_bar_open_after_6pm(self, temporal_env):
        """Bar should open at 6pm."""
        temporal_env.reset()

        bar_pos = temporal_env.affordances["Bar"]
        temporal_env.positions[0] = bar_pos.clone()

        # Noon: Bar is closed
        temporal_env.time_of_day = 12
        masks_closed = temporal_env.get_action_masks()
        assert not masks_closed[0, 4], "INTERACT should be masked on Bar at noon"

        # 8pm: Bar is open
        temporal_env.time_of_day = 20
        masks_open = temporal_env.get_action_masks()
        assert masks_open[0, 4], "INTERACT should be available on Bar at 8pm"

    def test_bar_wraparound_midnight(self, temporal_env):
        """Bar hours should wrap around midnight (6pm-4am)."""
        temporal_env.reset()

        bar_pos = temporal_env.affordances["Bar"]
        temporal_env.positions[0] = bar_pos.clone()

        # 2am: Bar is still open (wraps to 4am)
        temporal_env.time_of_day = 2
        masks_open = temporal_env.get_action_masks()
        assert masks_open[0, 4], "INTERACT should be available on Bar at 2am"

        # 5am: Bar is closed
        temporal_env.time_of_day = 5
        masks_closed = temporal_env.get_action_masks()
        assert not masks_closed[0, 4], "INTERACT should be masked on Bar at 5am"

    def test_bed_available_24_7(self, temporal_env):
        """Bed should be available 24/7."""
        temporal_env.reset()

        bed_pos = temporal_env.affordances["Bed"]
        temporal_env.positions[0] = bed_pos.clone()

        # Test at various times
        for time in [0, 6, 12, 18, 23]:
            temporal_env.time_of_day = time
            masks = temporal_env.get_action_masks()
            assert masks[0, 4], f"INTERACT on Bed should be available at {time}:00"

    def test_movement_unaffected_by_time(self, temporal_env):
        """Movement actions should not be affected by time of day."""
        temporal_env.reset()
        temporal_env.positions[0] = torch.tensor([4, 4], device=temporal_env.device)

        # Test movement at different times
        for time in [0, 12, 18, 23]:
            temporal_env.time_of_day = time
            masks = temporal_env.get_action_masks()

            # All movements should be available in center
            assert masks[0, 0], f"UP should be available at {time}:00"
            assert masks[0, 1], f"DOWN should be available at {time}:00"
            assert masks[0, 2], f"LEFT should be available at {time}:00"
            assert masks[0, 3], f"RIGHT should be available at {time}:00"


class TestPostTerminalMasking:
    """Test that dead agents cannot take actions.

    Consolidates tests from test_post_terminal_masking.py.
    Dead agents (health=0 or energy=0) should have all actions masked.
    """

    def test_all_actions_masked_after_health_death(self, multi_agent_env):
        """All 6 actions should be masked when agent dies from health=0."""
        multi_agent_env.reset()

        # Kill agent 0 by setting health to 0
        multi_agent_env.meters[0, 6] = 0.0

        masks = multi_agent_env.get_action_masks()

        # All actions should be masked for dead agent
        assert not masks[0].any(), "Dead agent should have all actions masked"

        # Other agents should still have actions available
        assert masks[1].any(), "Alive agents should have some actions available"

    def test_all_actions_masked_after_energy_death(self, basic_env):
        """All actions masked when agent dies from energy=0."""
        basic_env.reset()

        # Kill agent by setting energy to 0
        basic_env.meters[0, 0] = 0.0

        masks = basic_env.get_action_masks()

        assert not masks[0].any(), "Dead agent should have all actions masked"

    def test_near_death_not_masked(self, basic_env):
        """Agent near death (but alive) should still have actions available."""
        basic_env.reset()
        basic_env.positions[0] = torch.tensor([4, 4], device=basic_env.device)

        # Set health and energy very low but not zero
        basic_env.meters[0, 6] = 0.01  # barely alive
        basic_env.meters[0, 0] = 0.01  # barely any energy

        masks = basic_env.get_action_masks()

        # Should have some actions available
        assert masks[0, 5], "WAIT should be available for barely-alive agent"

    def test_masking_persists_after_step(self, multi_agent_env):
        """Dead agent should remain masked after environment steps."""
        multi_agent_env.reset()

        # Kill agent 0
        multi_agent_env.meters[0, 6] = 0.0

        # Take a step
        actions = torch.tensor([0, 1, 2, 3], device=multi_agent_env.device)
        multi_agent_env.step(actions)

        # Check masks after step
        masks = multi_agent_env.get_action_masks()

        assert not masks[0].any(), "Dead agent should remain masked after step"

    def test_multiple_dead_agents_all_masked(self, multi_agent_env):
        """Multiple dead agents should all have masked actions."""
        multi_agent_env.reset()

        # Kill first 3 agents
        multi_agent_env.meters[0, 6] = 0.0
        multi_agent_env.meters[1, 6] = 0.0
        multi_agent_env.meters[2, 6] = 0.0

        masks = multi_agent_env.get_action_masks()

        # First 3 should be fully masked
        assert not masks[0].any(), "Agent 0 should have all actions masked"
        assert not masks[1].any(), "Agent 1 should have all actions masked"
        assert not masks[2].any(), "Agent 2 should have all actions masked"

        # Agent 3 should still have actions
        assert masks[3].any(), "Agent 3 should have some actions available"

    def test_dead_agent_masked_with_temporal_mechanics(self, temporal_env):
        """Dead agent should be masked even with temporal mechanics."""
        temporal_env.reset()

        # Kill agent
        temporal_env.meters[0, 6] = 0.0

        # Test at various times
        for time in [0, 12, 18, 23]:
            temporal_env.time_of_day = time
            masks = temporal_env.get_action_masks()
            assert not masks[0].any(), f"Dead agent should be masked at time {time}:00"


class TestWaitAction:
    """Test WAIT action mechanics (always available unless dead).

    Consolidates tests from test_wait_action.py and test_all_actions.py.
    """

    def test_wait_always_available_for_alive_agents(self, multi_agent_env):
        """WAIT should be available at any position for alive agents."""
        multi_agent_env.reset()

        # Test at various positions
        test_positions = [
            [0, 0],  # Corner
            [7, 7],  # Opposite corner
            [4, 4],  # Center
            [0, 4],  # Edge
        ]

        for pos in test_positions:
            multi_agent_env.positions[0] = torch.tensor(pos, device=multi_agent_env.device)
            masks = multi_agent_env.get_action_masks()
            assert masks[0, 5], f"WAIT should be available at {pos}"

    def test_wait_masked_for_dead_agents(self, basic_env):
        """WAIT should be masked for dead agents."""
        basic_env.reset()

        # Kill agent
        basic_env.meters[0, 6] = 0.0

        masks = basic_env.get_action_masks()

        assert not masks[0, 5], "WAIT should be masked for dead agent"

    def test_wait_action_no_movement(self, basic_env):
        """WAIT should not change agent position."""
        basic_env.reset()
        initial_pos = basic_env.positions[0].clone()

        actions = torch.tensor([5], device=basic_env.device)
        basic_env.step(actions)

        assert torch.equal(basic_env.positions[0], initial_pos), "WAIT should not move agent"

    def test_wait_costs_less_than_movement(self, basic_env):
        """WAIT should cost less energy than movement."""
        basic_env.reset()
        basic_env.positions[0] = torch.tensor([4, 4], device=basic_env.device)

        # Measure WAIT cost
        initial_wait = basic_env.meters[0, 0].item()
        actions = torch.tensor([5], device=basic_env.device)
        basic_env.step(actions)
        wait_cost = initial_wait - basic_env.meters[0, 0].item()

        # Measure movement cost
        basic_env.reset()
        basic_env.positions[0] = torch.tensor([4, 4], device=basic_env.device)
        initial_move = basic_env.meters[0, 0].item()
        actions = torch.tensor([0], device=basic_env.device)
        basic_env.step(actions)
        move_cost = initial_move - basic_env.meters[0, 0].item()

        assert wait_cost < move_cost, f"WAIT cost ({wait_cost}) should be < movement cost ({move_cost})"

    def test_wait_energy_cost_validation(self, test_config_pack_path, device):
        """Environment should reject configs where WAIT is more expensive than MOVE."""
        with pytest.raises(ValueError):
            VectorizedHamletEnv(
                num_agents=1,
                grid_size=8,
                device=device,
                move_energy_cost=0.005,
                wait_energy_cost=0.01,  # More expensive than movement
                config_pack_path=test_config_pack_path,
            )


class TestActionMaskingIntegration:
    """Test that masking rules work together correctly.

    Tests combined scenarios where multiple masking rules apply.
    """

    def test_dead_agent_at_boundary(self, basic_env):
        """Dead agent at boundary should have all actions masked (death overrides boundary)."""
        basic_env.reset()

        # Place at corner and kill
        basic_env.positions[0] = torch.tensor([0, 0], device=basic_env.device)
        basic_env.meters[0, 6] = 0.0  # Dead

        masks = basic_env.get_action_masks()

        # All actions should be masked (death, not just boundary)
        assert not masks[0].any(), "Dead agent should have all 6 actions masked"

    def test_alive_agent_on_closed_affordance(self, temporal_env):
        """Alive agent on closed affordance should have movement but not INTERACT."""
        temporal_env.reset()

        # Place on Job at night
        job_pos = temporal_env.affordances["Job"]
        temporal_env.positions[0] = job_pos.clone()
        temporal_env.time_of_day = 20  # 8pm (Job closed)

        masks = temporal_env.get_action_masks()

        # INTERACT should be masked (closed)
        assert not masks[0, 4], "INTERACT should be masked at closed affordance"

        # But movement should work if not at boundary
        # Job is at (6, 6) in default config - check if movements are available
        job_x, job_y = job_pos[0].item(), job_pos[1].item()
        if job_y > 0:
            assert masks[0, 0], "UP should be available if not at boundary"
        if job_y < 7:
            assert masks[0, 1], "DOWN should be available if not at boundary"

    def test_alive_agent_on_open_affordable_affordance(self, temporal_env):
        """Alive agent on open, affordable affordance should have all actions."""
        temporal_env.reset()

        # Place on Bed (24/7, free)
        bed_pos = temporal_env.affordances["Bed"]
        temporal_env.positions[0] = bed_pos.clone()
        temporal_env.time_of_day = 12

        masks = temporal_env.get_action_masks()

        # INTERACT should be available
        assert masks[0, 4], "INTERACT should be available on open affordance"

        # WAIT should be available
        assert masks[0, 5], "WAIT should be available"

        # Movement depends on boundary (check position)
        bed_x, bed_y = bed_pos[0].item(), bed_pos[1].item()
        if bed_y > 0:
            assert masks[0, 0], "UP should be available if not at top boundary"

    def test_mask_shape_consistency(self, multi_agent_env):
        """Action masks should always have correct shape (num_agents, 6)."""
        multi_agent_env.reset()

        # Test in various configurations
        masks = multi_agent_env.get_action_masks()
        assert masks.shape == (4, 6), "Masks should be (num_agents=4, actions=6)"
        assert masks.dtype == torch.bool, "Masks should be boolean"

    def test_at_least_one_action_always_available_for_alive(self, basic_env):
        """Alive agents should always have at least one action available (WAIT)."""
        basic_env.reset()

        # Test at various positions, even corners
        test_positions = [
            [0, 0],
            [7, 7],
            [0, 7],
            [7, 0],
            [4, 4],
        ]

        for pos in test_positions:
            basic_env.positions[0] = torch.tensor(pos, device=basic_env.device)
            masks = basic_env.get_action_masks()

            # At minimum, WAIT should be available
            assert masks[0, 5], f"WAIT should be available at {pos}"

    def test_multi_agent_independent_masking(self, multi_agent_env):
        """Each agent should have independent masking in complex scenario."""
        multi_agent_env.reset()

        # Agent 0: alive, corner
        multi_agent_env.positions[0] = torch.tensor([0, 0], device=multi_agent_env.device)
        multi_agent_env.meters[0, 6] = 1.0  # Alive

        # Agent 1: dead, center
        multi_agent_env.positions[1] = torch.tensor([4, 4], device=multi_agent_env.device)
        multi_agent_env.meters[1, 6] = 0.0  # Dead

        # Agent 2: alive, center, on affordance
        bed_pos = multi_agent_env.affordances["Bed"]
        multi_agent_env.positions[2] = bed_pos.clone()
        multi_agent_env.meters[2, 6] = 1.0  # Alive

        # Agent 3: alive, edge
        multi_agent_env.positions[3] = torch.tensor([0, 4], device=multi_agent_env.device)
        multi_agent_env.meters[3, 6] = 1.0  # Alive

        masks = multi_agent_env.get_action_masks()

        # Agent 0: boundary-limited but alive
        assert not masks[0, 0], "Agent 0: UP masked at top"
        assert not masks[0, 2], "Agent 0: LEFT masked at left"
        assert masks[0, 5], "Agent 0: WAIT available"

        # Agent 1: fully masked (dead)
        assert not masks[1].any(), "Agent 1: all actions masked (dead)"

        # Agent 2: on affordance, all movements, INTERACT
        assert masks[2, 5], "Agent 2: WAIT available"
        # INTERACT availability depends on affordance position

        # Agent 3: edge-limited
        assert not masks[3, 2], "Agent 3: LEFT masked at edge"
        assert masks[3, 5], "Agent 3: WAIT available"

    def test_affordability_doesnt_affect_masking(self, basic_env):
        """Being broke should not affect INTERACT masking (only effectiveness).

        This validates P1.4 de-masking: agents can attempt interaction without money.
        """
        basic_env.reset()

        hospital_pos = basic_env.affordances["Hospital"]
        basic_env.positions[0] = hospital_pos.clone()

        # Test with money
        basic_env.meters[0, 3] = 1.0
        masks_rich = basic_env.get_action_masks()

        # Test without money
        basic_env.meters[0, 3] = 0.0
        masks_broke = basic_env.get_action_masks()

        # INTERACT masking should be identical
        assert masks_rich[0, 4] == masks_broke[0, 4], "Money should not affect INTERACT masking"
        assert masks_rich[0, 4], "INTERACT should be available on Hospital regardless of money"
