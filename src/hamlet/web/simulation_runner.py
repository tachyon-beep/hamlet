"""
Simulation runner for real-time visualization.

Orchestrates the DRL agent and environment, yielding state updates
for WebSocket streaming to frontend clients.
"""

import asyncio
from typing import AsyncGenerator, Optional, Dict, Any
from pathlib import Path
from hamlet.agent.drl_agent import DRLAgent
from hamlet.environment.hamlet_env import HamletEnv


class SimulationRunner:
    """
    Async simulation orchestrator for real-time visualization.

    Manages agent/environment lifecycle, handles control commands,
    and yields state updates for streaming.
    """

    def __init__(
        self,
        agent_path: str = "models/trained_agent.pt",
        state_dim: int = 70,
        action_dim: int = 5,
        base_delay: float = 0.1,  # 10 steps/second at 1x speed
    ):
        """
        Initialize simulation runner.

        Args:
            agent_path: Path to trained agent checkpoint
            state_dim: Observation dimension
            action_dim: Number of actions
            base_delay: Base delay between steps in seconds
        """
        self.agent_path = agent_path
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.base_delay = base_delay

        # Control state
        self.is_playing = False
        self.speed_multiplier = 1.0
        self.step_requested = False
        self.reset_requested = False

        # Agent and environment (lazy loaded)
        self.agent: Optional[DRLAgent] = None
        self.env: Optional[HamletEnv] = None

        # Episode tracking
        self.current_episode = 0
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.last_action = None

    def load_agent(self, agent_path: Optional[str] = None):
        """
        Load trained agent from checkpoint.

        Args:
            agent_path: Path to agent checkpoint (uses default if None)
        """
        if agent_path:
            self.agent_path = agent_path

        agent_file = Path(self.agent_path)

        # Auto-detect network type from checkpoint if it exists
        network_type = "qnetwork"  # Default
        if agent_file.exists():
            network_type = DRLAgent.detect_network_type(str(agent_file))
            print(f"Detected network type: {network_type}")
        else:
            print(f"Warning: Agent checkpoint not found at {self.agent_path}, using untrained agent")

        # Create agent with correct architecture
        self.agent = DRLAgent(
            agent_id="visualizer",
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device="cpu",
            network_type=network_type,
            grid_size=8  # Standard grid size for Hamlet
        )

        # Load checkpoint weights
        if agent_file.exists():
            self.agent.load(str(agent_file))

    def load_environment(self):
        """Initialize the Hamlet environment."""
        self.env = HamletEnv()

    def ensure_loaded(self):
        """Ensure agent and environment are loaded."""
        if self.agent is None:
            self.load_agent()
        if self.env is None:
            self.load_environment()

    async def run(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run simulation loop, yielding state updates.

        Yields:
            State dictionaries for WebSocket broadcasting
        """
        self.ensure_loaded()

        while True:
            # Start new episode
            self.current_episode += 1
            self.current_step = 0
            self.cumulative_reward = 0.0
            self.last_action = None

            obs = self.env.reset()
            done = False

            # Yield episode start
            yield {
                "type": "episode_start",
                "episode": self.current_episode,
                "model_name": Path(self.agent_path).name,
            }

            # Episode loop
            while not done:
                # Handle reset request
                if self.reset_requested:
                    self.reset_requested = False
                    break  # Start new episode

                # Wait for play or step command
                while not self.is_playing and not self.step_requested:
                    await asyncio.sleep(0.05)  # Check every 50ms
                    if self.reset_requested:
                        self.reset_requested = False
                        break

                if self.reset_requested:
                    break

                # Clear step request if it was set
                if self.step_requested:
                    self.step_requested = False

                # Select and execute action
                action = self.agent.select_action(obs, explore=False)  # Greedy for viz
                next_obs, reward, done, info = self.env.step(action)

                self.current_step += 1
                self.cumulative_reward += reward
                self.last_action = action

                # Yield state update (will be serialized by renderer)
                yield {
                    "type": "state_update",
                    "step": self.current_step,
                    "observation": obs,
                    "action": action,
                    "reward": reward,
                    "cumulative_reward": self.cumulative_reward,
                    "done": done,
                    "info": info,
                }

                obs = next_obs

                # Delay based on speed
                if self.is_playing:
                    delay = self.base_delay / self.speed_multiplier
                    await asyncio.sleep(delay)

            # Yield episode end
            yield {
                "type": "episode_end",
                "episode": self.current_episode,
                "steps": self.current_step,
                "total_reward": self.cumulative_reward,
                "reason": "meter_depleted" if done else "reset",
            }

            # Brief pause between episodes
            await asyncio.sleep(2.0)

    # Control commands

    def play(self):
        """Start simulation."""
        self.is_playing = True

    def pause(self):
        """Pause simulation."""
        self.is_playing = False

    def step(self):
        """Advance one step (when paused)."""
        self.step_requested = True

    def reset(self):
        """Reset current episode."""
        self.reset_requested = True

    def set_speed(self, speed: float):
        """
        Set simulation speed multiplier.

        Args:
            speed: Speed multiplier (0.5 = half speed, 2.0 = double speed)
        """
        self.speed_multiplier = max(0.1, min(10.0, speed))  # Clamp to 0.1x-10x
