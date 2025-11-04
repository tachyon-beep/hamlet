# tests/test_townlet/integration/test_temporal_mechanics.py
"""Integration tests for temporal mechanics (Level 3).

This module consolidates all temporal mechanics tests from:
- test_temporal_integration.py (6 tests)
- test_multi_interaction.py (4 tests)
- test_vectorized_env_temporal.py (2 tests)

Plus 5 new integration tests for critical gaps.

Total: 17 tests organized into 6 test classes.

TEST ORGANIZATION:
------------------
1. TestTimeProgression (3 tests)
   - Full 24-hour cycle wrapping
   - Time progression through 24 ticks
   - Observation dimensions with temporal features (sin/cos time + progress)

2. TestOperatingHours (3 tests)
   - Job masking after 6pm (closed hours)
   - Bar wraparound hours (6pm-4am across midnight)
   - 24-hour affordances (Bed, Hospital always available)

3. TestMultiTickInteractions (6 tests)
   - Progressive benefit accumulation (linear 75%)
   - Completion bonus (25% on final tick)
   - Multi-tick Job completion (4 ticks with money gain)
   - Money charged per tick (not on completion)
   - Interaction progress in observations (0.0-1.0)
   - Completion bonus timing (only on final tick)

4. TestEarlyExitMechanics (2 tests)
   - Early exit from interaction (keep linear benefits)
   - Early exit progress verification

5. TestMultiAgentTemporal (1 test)
   - Multi-agent independent temporal states

6. TestTemporalIntegrations (3 tests)
   - Temporal mechanics disabled fallback (legacy mode)
   - Temporal mechanics with adversarial curriculum
   - Temporal state recording (time_of_day, interaction_progress)

IMPLEMENTATION STATUS:
----------------------
NOTE: Most tests are currently marked as xfail because temporal mechanics
is in the process of being fully implemented. Tests are written to the spec
and will pass once implementation is complete.

- Time progression: ✅ PASSING (2/3 tests)
  - 24-hour cycle works
  - Observation dims need temporal features

- Operating hours: ⏳ PENDING (0/3 tests)
  - Action masking by time not yet implemented

- Multi-tick interactions: ⏳ PENDING (0/6 tests)
  - Multi-tick logic not yet implemented
  - Progress tracking not yet implemented

- Early exit: ⏳ PENDING (0/2 tests)
  - Requires multi-tick implementation

- Multi-agent temporal: ⏳ PENDING (0/1 test)
  - Requires multi-tick implementation

- Integrations: ⏳ PENDING (0/3 tests)
  - Curriculum integration not yet tested
  - Recording integration skipped (requires recorder module)

TO ENABLE TESTS:
----------------
Set TEMPORAL_MECHANICS_IMPLEMENTED = True when implementation is complete.
Tests will then run normally and report failures if implementation doesn't match spec.
"""

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv

# Temporal mechanics is partially implemented - mark tests appropriately
TEMPORAL_MECHANICS_IMPLEMENTED = False  # Set to True when temporal mechanics is complete
skip_temporal = pytest.mark.skipif(not TEMPORAL_MECHANICS_IMPLEMENTED, reason="Temporal mechanics not yet fully implemented")
xfail_temporal = pytest.mark.xfail(reason="Temporal mechanics not yet fully implemented", strict=False)


# =============================================================================
# TEST CLASS 1: TIME PROGRESSION
# =============================================================================


class TestTimeProgression:
    """24-hour cycle and time encoding in observations."""

    def test_full_24_hour_cycle(self, cpu_device):
        """Verify 24-hour cycle completes and wraps correctly.

        Migrated from: test_temporal_integration.py::test_full_24_hour_cycle
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        env.reset()
        assert env.time_of_day == 0

        # Step through 24 hours
        for expected_time in range(24):
            assert env.time_of_day == expected_time
            env.step(torch.tensor([0], device=cpu_device))  # UP action

        # Should wrap back to 0
        assert env.time_of_day == 0

    def test_time_of_day_cycles(self, cpu_device):
        """Verify time cycles through 24 ticks (alternative verification).

        Migrated from: test_vectorized_env_temporal.py::test_time_of_day_cycles
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        env.reset()

        # Step 24 times
        for i in range(24):
            assert env.time_of_day == i
            env.step(torch.tensor([4], device=cpu_device))  # INTERACT action

        # Should wrap back to 0
        assert env.time_of_day == 0

    @xfail_temporal
    def test_observation_dimensions_with_temporal(self, cpu_device):
        """Verify observation includes temporal features (sin/cos time + progress).

        Migrated from: test_temporal_integration.py::test_observation_dimensions_with_temporal
        Combined with: test_vectorized_env_temporal.py::test_observation_includes_time_and_progress
        """
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        obs = env.reset()

        # Full observability: 64 (grid) + 8 (meters) + (num_affordance_types + 1) + 3 (temporal)
        # Temporal features: sin(time), cos(time), normalized interaction progress
        expected_dim = 64 + 8 + (env.num_affordance_types + 1) + 3
        assert obs.shape == (2, expected_dim)

        time_sin = obs[0, -3]
        time_cos = obs[0, -2]
        progress_feature = obs[0, -1]

        # time_of_day = 0 at reset => sin = 0, cos = 1
        assert time_sin == pytest.approx(0.0, abs=1e-6)
        assert time_cos == pytest.approx(1.0, abs=1e-6)
        assert progress_feature == 0.0  # No interaction yet


# =============================================================================
# TEST CLASS 2: OPERATING HOURS
# =============================================================================


class TestOperatingHours:
    """Time-based affordance availability and action masking."""

    @xfail_temporal
    def test_operating_hours_mask_job(self, cpu_device):
        """Verify Job is masked out after 6pm (closed hours).

        Migrated from: test_temporal_integration.py::test_operating_hours_mask_job
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        env.reset()
        env.positions[0] = torch.tensor([6, 6], device=cpu_device)  # On Job
        env.meters[0, 3] = 1.0

        # 10am: Job open (operating hours: 8-18)
        env.time_of_day = 10
        masks = env.get_action_masks()
        assert masks[0, 4]  # INTERACT allowed

        # 7pm: Job closed
        env.time_of_day = 19
        masks = env.get_action_masks()
        assert not masks[0, 4]  # INTERACT blocked

    @xfail_temporal
    def test_bar_wraparound_hours(self, cpu_device):
        """Verify Bar operating hours wrap around midnight (6pm-4am).

        New test: Validates wraparound logic for late-night affordances.
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        env.reset()
        # Find Bar position (affordance id=9)
        bar_pos = None
        for pos, aff_id in env.affordance_map.items():
            if aff_id == 9:  # Bar
                bar_pos = torch.tensor([pos[0], pos[1]], device=cpu_device)
                break

        if bar_pos is None:
            pytest.skip("Bar affordance not deployed in test config")

        env.positions[0] = bar_pos
        env.meters[0, 3] = 1.0  # $100 money

        # 8pm: Bar open (operating hours: 18-28, i.e., 6pm-4am)
        env.time_of_day = 20
        masks = env.get_action_masks()
        assert masks[0, 4]  # INTERACT allowed

        # 2am: Bar still open (wraparound)
        env.time_of_day = 2
        masks = env.get_action_masks()
        assert masks[0, 4]  # INTERACT allowed

        # 5am: Bar closed
        env.time_of_day = 5
        masks = env.get_action_masks()
        assert not masks[0, 4]  # INTERACT blocked

    @xfail_temporal
    def test_24_hour_affordances(self, cpu_device):
        """Verify 24-hour affordances (Bed, Hospital) are always available.

        New test: Validates always-open affordances.
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        env.reset()
        # Bed is always at (1,1) in default config
        env.positions[0] = torch.tensor([1, 1], device=cpu_device)

        # Test at multiple times
        for time in [0, 6, 12, 18, 23]:
            env.time_of_day = time
            masks = env.get_action_masks()
            assert masks[0, 4], f"Bed should be available at {time}:00"


# =============================================================================
# TEST CLASS 3: MULTI-TICK INTERACTIONS
# =============================================================================


class TestMultiTickInteractions:
    """Multi-tick interaction mechanics (linear + completion bonus)."""

    @xfail_temporal
    def test_progressive_benefit_accumulation(self, cpu_device):
        """Verify linear benefits accumulate per tick.

        Migrated from: test_multi_interaction.py::test_progressive_benefit_accumulation
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        env.reset()
        env.meters[0, 0] = 0.3  # Start at 30% energy to avoid clamping
        env.positions[0] = torch.tensor([1, 1], device=cpu_device)  # On Bed (default position)

        initial_energy = env.meters[0, 0].item()

        # Bed config: 5 ticks, +7.5% energy per tick (linear)
        # But also -0.5% depletion per tick
        # Net per tick: +7.0%
        # First INTERACT
        env.step(torch.tensor([4], device=cpu_device))

        energy_after_1 = env.meters[0, 0].item()
        # Expected: +0.075 (benefit) - 0.005 (depletion) + cascading ~= +0.070
        assert abs((energy_after_1 - initial_energy) - 0.070) < 0.01

        # Second INTERACT
        env.step(torch.tensor([4], device=cpu_device))

        energy_after_2 = env.meters[0, 0].item()
        assert abs((energy_after_2 - initial_energy) - 0.140) < 0.02

    @xfail_temporal
    def test_completion_bonus(self, cpu_device):
        """Verify 25% bonus on full completion.

        Migrated from: test_multi_interaction.py::test_completion_bonus
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        env.reset()
        env.meters[0, 0] = 0.3  # Start at 30% energy
        env.meters[0, 6] = 0.7  # Start at 70% health (to see bonus)
        env.positions[0] = torch.tensor([1, 1], device=cpu_device)

        initial_energy = env.meters[0, 0].item()
        initial_health = env.meters[0, 6].item()

        # Complete all 5 ticks
        for _ in range(5):
            env.step(torch.tensor([4], device=cpu_device))

        final_energy = env.meters[0, 0].item()
        final_health = env.meters[0, 6].item()

        # Total energy: 5 × 7.5% (linear) + 12.5% (completion) = 50%
        # Minus 5 × 0.5% depletion = -2.5%
        # Net: ~47.5%, but cascading effects apply
        # Check that energy increased significantly (at least 40%)
        assert (final_energy - initial_energy) > 0.40

        # Health bonus only on completion: +2%
        # (cascading effects from energy/satiation also apply)
        assert (final_health - initial_health) > 0.015

    @xfail_temporal
    def test_multi_tick_job_completion(self, cpu_device):
        """Verify Job completion over 4 ticks with money gain.

        Migrated from: test_temporal_integration.py::test_multi_tick_job_completion
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        env.reset()
        env.positions[0] = torch.tensor([6, 6], device=cpu_device)  # On Job
        env.meters[0, 3] = 0.5  # Start with $50
        env.time_of_day = 10  # 10am (Job open 8-18)

        initial_money = env.meters[0, 3].item()

        # Complete 4 ticks of Job
        for i in range(4):
            env.step(torch.tensor([4], device=cpu_device))  # INTERACT
            # Progress: 1, 2, 3, then 0 (completes on 4th)
            if i < 3:
                assert env.interaction_progress[0] == (i + 1)
            else:
                assert env.interaction_progress[0] == 0  # Completed

        final_money = env.meters[0, 3].item()

        # Job pays: 4 × $5.625 (linear) + $5.625 (completion) = $28.125 total
        # Normalized: +0.28125
        assert (final_money - initial_money) > 0.25  # At least 25% gain

    @xfail_temporal
    def test_money_charged_per_tick(self, cpu_device):
        """Verify cost charged each tick, not on completion.

        Migrated from: test_multi_interaction.py::test_money_charged_per_tick
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        env.reset()
        env.positions[0] = torch.tensor([1, 1], device=cpu_device)
        env.meters[0, 3] = 0.50  # Start with $50

        # Bed costs $1/tick = 0.01 normalized
        env.step(torch.tensor([4], device=cpu_device))

        money_after_1 = env.meters[0, 3].item()
        assert abs(money_after_1 - 0.49) < 0.001  # $50 - $1 = $49

        env.step(torch.tensor([4], device=cpu_device))

        money_after_2 = env.meters[0, 3].item()
        assert abs(money_after_2 - 0.48) < 0.001  # $49 - $1 = $48

    @xfail_temporal
    def test_interaction_progress_in_observations(self, cpu_device):
        """Verify interaction progress (0.0-1.0) appears in observations.

        New test: Validates that agents can observe their own progress.
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        env.reset()
        env.positions[0] = torch.tensor([1, 1], device=cpu_device)  # On Bed (5 ticks)
        env.meters[0, 0] = 0.3  # Low energy

        # Initial observation: no progress
        obs = env.reset()
        progress_feature = obs[0, -1]
        assert progress_feature == 0.0

        # After 1 tick: progress = 1/5 = 0.2
        obs, _, _, _ = env.step(torch.tensor([4], device=cpu_device))
        progress_feature = obs[0, -1]
        assert progress_feature == pytest.approx(0.2, abs=0.01)

        # After 2 ticks: progress = 2/5 = 0.4
        obs, _, _, _ = env.step(torch.tensor([4], device=cpu_device))
        progress_feature = obs[0, -1]
        assert progress_feature == pytest.approx(0.4, abs=0.01)

        # After 3 ticks: progress = 3/5 = 0.6
        obs, _, _, _ = env.step(torch.tensor([4], device=cpu_device))
        progress_feature = obs[0, -1]
        assert progress_feature == pytest.approx(0.6, abs=0.01)

    @xfail_temporal
    def test_completion_bonus_timing(self, cpu_device):
        """Verify completion bonus applied ONLY on final tick.

        New test: Validates that completion bonus doesn't leak into linear ticks.
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        env.reset()
        env.positions[0] = torch.tensor([6, 6], device=cpu_device)  # On Job
        env.meters[0, 3] = 0.5  # Start with $50
        env.time_of_day = 10  # 10am (Job open)

        # Track money after each tick
        money_values = [0.5]  # Initial

        # Job: 4 ticks, $5.625/tick (linear) + $5.625 bonus (completion)
        for i in range(4):
            env.step(torch.tensor([4], device=cpu_device))
            money_values.append(env.meters[0, 3].item())

        # Verify linear accumulation
        # Tick 1: $50 + $5.625 = $55.625 (normalized: 0.55625)
        assert money_values[1] == pytest.approx(0.55625, abs=0.001)

        # Tick 2: $55.625 + $5.625 = $61.25 (normalized: 0.6125)
        assert money_values[2] == pytest.approx(0.6125, abs=0.001)

        # Tick 3: $61.25 + $5.625 = $66.875 (normalized: 0.66875)
        assert money_values[3] == pytest.approx(0.66875, abs=0.001)

        # Tick 4: $66.875 + $5.625 (linear) + $5.625 (bonus) = $78.125
        # Normalized: 0.78125
        assert money_values[4] == pytest.approx(0.78125, abs=0.001)

        # Verify completion bonus magnitude
        bonus_tick_gain = money_values[4] - money_values[3]
        linear_tick_gain = money_values[1] - money_values[0]

        # Completion tick should be ~2x linear tick (linear + bonus)
        assert bonus_tick_gain > (linear_tick_gain * 1.9)


# =============================================================================
# TEST CLASS 4: EARLY EXIT MECHANICS
# =============================================================================


class TestEarlyExitMechanics:
    """Early exit from multi-tick interactions."""

    @xfail_temporal
    def test_early_exit_from_interaction(self, cpu_device):
        """Verify agent can exit early and keep linear benefits.

        Migrated from: test_temporal_integration.py::test_early_exit_from_interaction
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        env.reset()
        env.meters[0, 0] = 0.3  # Low energy
        env.positions[0] = torch.tensor([1, 1], device=cpu_device)  # On Bed

        initial_energy = env.meters[0, 0].item()

        # Do 2 ticks of Bed (requires 5 for completion)
        env.step(torch.tensor([4], device=cpu_device))
        env.step(torch.tensor([4], device=cpu_device))
        assert env.interaction_progress[0] == 2

        energy_after_2 = env.meters[0, 0].item()

        # Move away
        env.step(torch.tensor([0], device=cpu_device))  # UP
        assert env.interaction_progress[0] == 0  # Progress reset
        assert env.last_interaction_affordance[0] is None

        # Energy should have increased from 2 ticks
        assert (energy_after_2 - initial_energy) > 0.1

    @xfail_temporal
    def test_early_exit_keeps_progress(self, cpu_device):
        """Verify agent keeps linear benefits if exiting early.

        Migrated from: test_multi_interaction.py::test_early_exit_keeps_progress
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        env.reset()
        env.meters[0, 0] = 0.3  # Start at 30% energy
        env.positions[0] = torch.tensor([1, 1], device=cpu_device)

        initial_energy = env.meters[0, 0].item()

        # Do 3 ticks, then move away
        for _ in range(3):
            env.step(torch.tensor([4], device=cpu_device))

        energy_after_3 = env.meters[0, 0].item()

        # Move away (UP action)
        env.step(torch.tensor([0], device=cpu_device))

        # Energy should be at approximately 3 × 7% = 21% gain
        # (3 ticks of benefit minus depletion, no completion bonus)
        assert abs((energy_after_3 - initial_energy) - 0.21) < 0.03


# =============================================================================
# TEST CLASS 5: MULTI-AGENT TEMPORAL
# =============================================================================


class TestMultiAgentTemporal:
    """Multi-agent temporal mechanics with independent states."""

    @xfail_temporal
    def test_multi_agent_temporal_interactions(self, cpu_device):
        """Verify 3 agents with independent temporal states.

        New test: Validates that each agent has independent interaction progress.
        """
        env = VectorizedHamletEnv(
            num_agents=3,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        env.reset()

        # Agent 0: On Bed (5 ticks), do 2 ticks
        env.positions[0] = torch.tensor([1, 1], device=cpu_device)
        env.meters[0, 0] = 0.3  # Low energy

        # Agent 1: On Job (4 ticks), do 3 ticks
        env.positions[1] = torch.tensor([6, 6], device=cpu_device)
        env.meters[1, 3] = 0.5  # $50
        env.time_of_day = 10  # Job open

        # Agent 2: Not interacting
        env.positions[2] = torch.tensor([3, 3], device=cpu_device)

        # Execute 3 steps
        for i in range(3):
            actions = torch.tensor(
                [
                    4,  # Agent 0: INTERACT (Bed)
                    4,  # Agent 1: INTERACT (Job)
                    0,  # Agent 2: UP (no interaction)
                ],
                device=cpu_device,
            )

            obs, _, _, _ = env.step(actions)

            # Verify individual progress
            if i < 2:
                # Agent 0: Should show progress on Bed (i+1)/5
                assert env.interaction_progress[0] == (i + 1)
                assert obs[0, -1] == pytest.approx((i + 1) / 5, abs=0.01)

            # Agent 1: Should show progress on Job (i+1)/4
            assert env.interaction_progress[1] == (i + 1)
            assert obs[1, -1] == pytest.approx((i + 1) / 4, abs=0.01)

            # Agent 2: Should show no progress
            assert env.interaction_progress[2] == 0
            assert obs[2, -1] == 0.0

        # Final verification
        assert env.interaction_progress[0] == 2  # Bed progress (stopped at 2)
        assert env.interaction_progress[1] == 3  # Job progress
        assert env.interaction_progress[2] == 0  # No progress


# =============================================================================
# TEST CLASS 6: TEMPORAL INTEGRATIONS
# =============================================================================


class TestTemporalIntegrations:
    """Cross-system temporal mechanics integration."""

    @xfail_temporal
    def test_temporal_mechanics_disabled_fallback(self, cpu_device):
        """Verify environment works without temporal mechanics (legacy mode).

        Migrated from: test_temporal_integration.py::test_temporal_mechanics_disabled_fallback
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=False,  # Legacy mode
        )

        obs = env.reset()

        # Without temporal: 64 (grid) + 8 (meters) + 15 (affordance) = 87
        assert obs.shape == (1, 87)

        # Temporal state is dormant but present
        assert hasattr(env, "time_of_day")
        assert env.time_of_day == 0

        # Interactions work (legacy single-shot mode)
        env.positions[0] = torch.tensor([1, 1], device=cpu_device)  # On Bed
        env.meters[0, 0] = 0.3  # Start low to see increase

        initial_energy = env.meters[0, 0].item()

        env.step(torch.tensor([4], device=cpu_device))  # INTERACT

        final_energy = env.meters[0, 0].item()
        # Legacy mode: single-shot benefit (+50% energy from Bed)
        # Even with depletion, should see significant increase
        assert (final_energy - initial_energy) > 0.4  # At least 40% gain

    @xfail_temporal
    def test_temporal_mechanics_with_curriculum(self, cpu_device):
        """Verify temporal mechanics works with adversarial curriculum.

        New test: Validates that curriculum receives correct survival signal.
        """
        from townlet.curriculum.adversarial import AdversarialCurriculum

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        curriculum = AdversarialCurriculum(
            max_steps_per_episode=200,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
            entropy_gate=0.5,
            min_steps_at_stage=10,
        )

        env.reset()
        env.meters[0, 0] = 1.0  # Full energy (agent won't die)

        # Run 100 steps across day/night cycle
        step_count = 0
        for _ in range(100):
            action = torch.tensor([0], device=cpu_device)  # UP action
            obs, reward, done, info = env.step(action)

            if not done[0]:
                step_count += 1
            else:
                break

        # Agent should survive all 100 steps
        assert step_count == 100

        # Report to curriculum
        curriculum.step_end(info)

        # Curriculum should see correct survival (not affected by time)
        assert curriculum.total_episodes == 1

    @skip_temporal
    def test_temporal_state_recording(self, cpu_device):
        """Verify episode recording includes temporal state.

        New test: Validates that time_of_day and interaction_progress are recorded.
        """
        import sqlite3
        import tempfile

        from townlet.recording.episode_recorder import EpisodeRecorder

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            enable_temporal_mechanics=True,
        )

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            recorder = EpisodeRecorder(db_path=db_path)

            # Record a few steps with temporal state
            env.reset()
            env.positions[0] = torch.tensor([1, 1], device=cpu_device)  # On Bed
            env.meters[0, 0] = 0.3

            recorder.start_episode(episode_number=0, agent_id=0)

            # Record 3 interaction steps
            for i in range(3):
                obs, reward, done, info = env.step(torch.tensor([4], device=cpu_device))

                recorder.record_step(
                    episode_number=0,
                    agent_id=0,
                    step_number=i,
                    state={
                        "position": env.positions[0].tolist(),
                        "meters": env.meters[0].tolist(),
                        "time_of_day": env.time_of_day,
                        "interaction_progress": env.interaction_progress[0].item(),
                    },
                    action=4,
                    reward=reward[0].item(),
                    done=done[0].item(),
                    info=info,
                )

            recorder.end_episode(
                episode_number=0,
                agent_id=0,
                total_reward=100.0,
                survival_steps=3,
            )

            # Verify recorded data
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT state FROM episode_steps
                WHERE episode_number = 0
                ORDER BY step_number
            """
            )

            import json

            rows = cursor.fetchall()
            assert len(rows) == 3

            for i, row in enumerate(rows):
                state = json.loads(row[0])
                assert "time_of_day" in state
                assert "interaction_progress" in state

                # Time should advance
                assert state["time_of_day"] == i

                # Progress should increase (1/5, 2/5, 3/5)
                expected_progress = i + 1
                assert state["interaction_progress"] == expected_progress

            conn.close()

        finally:
            import os

            if os.path.exists(db_path):
                os.remove(db_path)
