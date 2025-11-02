"""Tests for live inference telemetry and baseline handling."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import torch

from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.demo.live_inference import LiveInferenceServer
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation


class _DummyQueueClient:
    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)


class _TestServer(LiveInferenceServer):
    """Subclass LiveInferenceServer to inject small deterministic components."""

    def _initialize_components(self):
        device = torch.device("cpu")
        self.env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=device,
            partial_observability=False,
        )
        obs_dim = self.env.observation_dim
        self.curriculum = AdversarialCurriculum(max_steps_per_episode=100, device=device)
        self.curriculum.initialize_population(1)
        self.exploration = AdaptiveIntrinsicExploration(obs_dim=obs_dim, device=device)
        self.population = VectorizedPopulation(
            env=self.env,
            curriculum=self.curriculum,
            exploration=self.exploration,
            agent_ids=["agent_0"],
            device=device,
            obs_dim=obs_dim,
            action_dim=self.env.action_dim,
        )
        # Avoid file I/O in tests
        self.clients = set()


async def _run_single_step(server: LiveInferenceServer):
    # Replace websocket broadcast with dummy handlers
    client = _DummyQueueClient()
    server.clients.add(client)

    # Monkeypatch broadcast to capture messages
    async def _capture(payload):
        await client.send_json(payload)

    server._broadcast_to_clients = AsyncMock(side_effect=_capture)
    server._broadcast_state_update = AsyncMock()

    await server._run_single_episode()
    return client.messages


def test_live_inference_uses_runtime_registry_for_baseline(monkeypatch):
    server = _TestServer(checkpoint_dir=".", port=0)
    server._initialize_components()
    server.current_checkpoint_episode = 0
    server.current_epsilon = 0.0

    registry = server.population.runtime_registry
    server.population._update_reward_baseline = lambda: None

    original_reset = server.population.reset

    def patched_reset():
        original_reset()
        registry.set_baselines(torch.tensor([42.0], dtype=torch.float32, device=server.env.device))

    server.population.reset = patched_reset
    server.env.randomize_affordance_positions = MagicMock()

    messages = asyncio.run(_run_single_step(server))

    # Check that the messages use the registry baseline (42.0) rather than scalar defaults
    episode_start = next(msg for msg in messages if msg["type"] == "episode_start")
    assert episode_start["baseline_survival"] == 42.0
    episode_end = next(msg for msg in messages if msg["type"] == "episode_end")
    assert episode_end["baseline_survival"] == 42.0

    telemetry_agents = episode_start["telemetry"]["agents"]
    assert telemetry_agents[0]["baseline_survival_steps"] == 42.0


def test_live_inference_handles_six_action_q_values(monkeypatch):
    server = _TestServer(checkpoint_dir=".", port=0)
    server._initialize_components()
    server.population._update_reward_baseline = lambda: None

    server.env.randomize_affordance_positions = MagicMock()

    async def fake_state_update(cumulative_reward, last_action, q_values):
        assert len(q_values) == 6
        assert len(server.env.get_action_masks()[0]) == 6

    async def noop(payload):
        pass

    server._broadcast_to_clients = AsyncMock(side_effect=noop)
    server._broadcast_state_update = fake_state_update

    asyncio.run(server._run_single_episode())
