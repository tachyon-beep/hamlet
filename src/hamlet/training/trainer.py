"""
Main training orchestrator for Hamlet.

Coordinates agents, environment, metrics, experiments, and checkpoints.
"""

import time
from typing import Dict, List, Any, Optional
import numpy as np

from hamlet.environment.hamlet_env import HamletEnv
from hamlet.agent.observation_utils import preprocess_observation
from hamlet.training.config import FullConfig
from hamlet.training.agent_manager import AgentManager
from hamlet.training.metrics_manager import MetricsManager
from hamlet.training.experiment_manager import ExperimentManager
from hamlet.training.checkpoint_manager import CheckpointManager


class Trainer:
    """
    Main training coordinator.

    Orchestrates all training components for clean, scalable DRL training.
    """

    def __init__(self, config: FullConfig, enable_terminal_viz: bool = False, enable_web_broadcast: bool = False):
        """
        Initialize trainer from configuration.

        Args:
            config: Complete training configuration
            enable_terminal_viz: Enable simple terminal visualization during training
            enable_web_broadcast: Enable web broadcasting for live visualization
        """
        self.config = config
        self.enable_terminal_viz = enable_terminal_viz
        self.enable_web_broadcast = enable_web_broadcast
        self.web_broadcaster = None

        if enable_web_broadcast:
            from hamlet.training.web_training_broadcaster import get_broadcaster
            self.web_broadcaster = get_broadcaster()

        # Create environment
        self.env = HamletEnv(config=config.environment)

        # Initialize managers
        self.agent_manager = AgentManager(
            buffer_size=config.training.replay_buffer_size,
            buffer_threshold=10,  # Switch to shared buffer at 10 agents
        )

        self.metrics_manager = MetricsManager(
            config.metrics,
            experiment_name=config.experiment.name,
        )

        self.experiment_manager = ExperimentManager(config.experiment)

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.training.checkpoint_dir,
            max_checkpoints=5,
            keep_best=True,
            metric_name="total_reward",
            metric_mode="max",
        )

        # Create agents from config
        for agent_config in config.agents:
            self.agent_manager.add_agent(agent_config)

        # Training state
        self.global_step = 0
        self.episode_rewards: Dict[str, List[float]] = {}
        self.episode_lengths: Dict[str, List[int]] = {}

        # Initialize episode tracking for all agents
        for agent_id in self.agent_manager.get_agent_ids():
            self.episode_rewards[agent_id] = []
            self.episode_lengths[agent_id] = []

    def train(self):
        """
        Run complete training loop.

        Executes training for configured number of episodes with
        full metrics tracking, checkpointing, and experiment logging.
        """
        # Start MLflow run
        self.experiment_manager.start_run()

        # Log hyperparameters
        hparams = self._get_hyperparameters()
        self.experiment_manager.log_params(hparams)

        print(f"Starting training: {self.config.experiment.name}")
        print(f"Agents: {len(self.agent_manager.get_agent_ids())}")
        print(f"Episodes: {self.config.training.num_episodes}")
        print(f"Buffer mode: {self.agent_manager.buffer_mode}")

        start_time = time.time()

        # Broadcast training start if web enabled
        if self.web_broadcaster:
            import asyncio
            asyncio.run(self.web_broadcaster.broadcast_training_start({
                "num_episodes": self.config.training.num_episodes,
                "num_agents": len(self.agent_manager.get_agent_ids()),
            }))

        # Main training loop
        for episode in range(self.config.training.num_episodes):
            episode_metrics = self._run_episode(episode)

            # Log metrics
            if episode % self.config.training.log_frequency == 0:
                self._log_episode_metrics(episode, episode_metrics)

            # Save checkpoint
            if episode % self.config.training.save_frequency == 0 and episode > 0:
                self._save_checkpoint(episode, episode_metrics)

            # Update target networks
            if episode % self.config.training.target_update_frequency == 0:
                self._update_target_networks()

            # Decay exploration
            self._decay_exploration()

            # Increment episode counter for metrics
            self.metrics_manager.increment_episode()

        # Training complete
        total_time = time.time() - start_time

        print(f"\nTraining complete!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per episode: {total_time / self.config.training.num_episodes:.2f}s")

        # Log final metrics
        final_metrics = self._get_final_metrics()
        self.experiment_manager.log_metrics(final_metrics)
        self.metrics_manager.log_hyperparameters(hparams, final_metrics)

        # Save final checkpoint
        self._save_checkpoint(self.config.training.num_episodes, final_metrics, final=True)

        # Close managers
        self.metrics_manager.close()
        self.experiment_manager.end_run()

        # Close environment if it has a close method
        if hasattr(self.env, "close"):
            self.env.close()

    def _run_episode(self, episode: int) -> Dict[str, Dict[str, float]]:
        """
        Run a single training episode.

        Args:
            episode: Episode number

        Returns:
            Dictionary of agent_id -> metrics
        """
        # For single-agent environment, use first agent
        agent_id = self.agent_manager.get_agent_ids()[0]
        agent = self.agent_manager.get_agent(agent_id)

        # Reset environment
        obs = self.env.reset()
        state = preprocess_observation(obs)

        episode_reward = 0.0
        episode_length = 0
        trajectory = []  # For replay storage

        # Optional terminal visualization
        if self.enable_terminal_viz:
            from hamlet.training.simple_renderer import clear_terminal, render_simple_state
            clear_terminal()

        # Episode loop
        for step in range(self.config.training.max_steps_per_episode):
            # Select action
            action = agent.select_action(state, explore=True)

            # Execute action
            next_obs, reward, done, info = self.env.step(action)
            next_state = preprocess_observation(next_obs)

            # Store experience
            self.agent_manager.store_experience(
                agent_id, state, action, reward, next_state, done
            )

            # Track trajectory for replay
            if self.config.metrics.replay_storage:
                trajectory.append({
                    "state": state.tolist(),
                    "action": int(action),
                    "reward": float(reward),
                    "done": bool(done),
                })

            # Learn from experience
            if self.global_step >= self.config.training.learning_starts:
                if self.agent_manager.can_sample(
                    self.config.training.batch_size, agent_id
                ):
                    batch = self.agent_manager.sample_batch(
                        self.config.training.batch_size, agent_id
                    )
                    if batch is not None:
                        loss = agent.learn(batch)

                        # Log step-level metrics
                        if self.global_step % 100 == 0:
                            self.metrics_manager.log_step(
                                self.global_step,
                                agent_id,
                                {"loss": loss, "epsilon": agent.epsilon},
                            )

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.global_step += 1

            # Optional visualization (every 5 steps to avoid spam)
            if (self.enable_terminal_viz or self.web_broadcaster) and step % 5 == 0:
                env_state = self.env.render()

                # Terminal viz
                if self.enable_terminal_viz:
                    clear_terminal()
                    print(render_simple_state(env_state, episode, step, episode_reward))
                    import time
                    time.sleep(0.1)  # Brief pause so humans can see it

                # Web broadcast
                if self.web_broadcaster:
                    import asyncio
                    asyncio.run(self.web_broadcaster.broadcast_training_state(
                        episode, step, episode_reward, env_state
                    ))

            if done:
                break

        # Save episode replay
        if self.config.metrics.replay_storage:
            self.metrics_manager.save_episode_replay(
                episode, agent_id, trajectory
            )

        # Track episode metrics
        self.episode_rewards[agent_id].append(episode_reward)
        self.episode_lengths[agent_id].append(episode_length)

        # Return episode metrics
        metrics = {
            agent_id: {
                "total_reward": episode_reward,
                "episode_length": episode_length,
                "epsilon": agent.epsilon,
                "buffer_size": len(self.agent_manager.per_agent_buffers.get(
                    agent_id, []
                )) if self.agent_manager.buffer_mode == "per_agent" else len(
                    self.agent_manager.shared_buffer or []
                ),
            }
        }

        return metrics

    def _log_episode_metrics(self, episode: int, episode_metrics: Dict[str, Dict[str, float]]):
        """
        Log metrics for episode.

        Args:
            episode: Episode number
            episode_metrics: Dictionary of agent_id -> metrics
        """
        for agent_id, metrics in episode_metrics.items():
            # Log to metrics manager
            self.metrics_manager.log_episode(episode, agent_id, metrics)

            # Log to MLflow
            for metric_name, value in metrics.items():
                self.experiment_manager.log_metric(
                    f"{agent_id}/{metric_name}", value, step=episode
                )

            # Print progress
            if episode % self.config.training.log_frequency == 0:
                print(
                    f"Episode {episode:4d} | "
                    f"Reward: {metrics['total_reward']:7.2f} | "
                    f"Length: {metrics['episode_length']:4d} | "
                    f"Epsilon: {metrics['epsilon']:.3f} | "
                    f"Buffer: {metrics['buffer_size']:5d}"
                )

    def _save_checkpoint(
        self,
        episode: int,
        metrics: Dict[str, Dict[str, float]],
        final: bool = False,
    ):
        """
        Save checkpoint.

        Args:
            episode: Episode number
            metrics: Episode metrics
            final: Whether this is the final checkpoint
        """
        # Get all agents
        agents = {
            agent_id: self.agent_manager.get_agent(agent_id)
            for agent_id in self.agent_manager.get_agent_ids()
        }

        # Flatten metrics for checkpoint (use first agent's metrics)
        agent_id = list(metrics.keys())[0]
        checkpoint_metrics = metrics[agent_id]

        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            episode=episode,
            agents=agents,
            metrics=checkpoint_metrics,
            metadata={"final": final},
        )

        if final:
            print(f"Final checkpoint saved: {checkpoint_path}")
        else:
            print(f"Checkpoint saved: {checkpoint_path}")

        # Log checkpoint to MLflow
        self.experiment_manager.log_artifact(str(checkpoint_path))

    def _update_target_networks(self):
        """Update target networks for all agents."""
        for agent in self.agent_manager.get_all_agents():
            if hasattr(agent, "update_target_network"):
                agent.update_target_network()

    def _decay_exploration(self):
        """Decay exploration rate for all agents."""
        for agent in self.agent_manager.get_all_agents():
            if hasattr(agent, "decay_epsilon"):
                agent.decay_epsilon()

    def _get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get hyperparameters for logging.

        Returns:
            Dictionary of hyperparameters
        """
        # Get first agent config as representative
        agent_config = self.config.agents[0]

        return {
            "num_episodes": self.config.training.num_episodes,
            "max_steps_per_episode": self.config.training.max_steps_per_episode,
            "batch_size": self.config.training.batch_size,
            "learning_rate": agent_config.learning_rate,
            "gamma": agent_config.gamma,
            "epsilon_start": agent_config.epsilon,
            "epsilon_min": agent_config.epsilon_min,
            "epsilon_decay": agent_config.epsilon_decay,
            "buffer_size": self.config.training.replay_buffer_size,
            "target_update_frequency": self.config.training.target_update_frequency,
            "num_agents": len(self.config.agents),
            "grid_size": self.config.environment.grid_width,
        }

    def _get_final_metrics(self) -> Dict[str, float]:
        """
        Get final training metrics.

        Returns:
            Dictionary of final metrics
        """
        # Use first agent as representative
        agent_id = self.agent_manager.get_agent_ids()[0]

        # Calculate statistics over last 100 episodes
        last_n = 100
        recent_rewards = self.episode_rewards[agent_id][-last_n:]
        recent_lengths = self.episode_lengths[agent_id][-last_n:]

        return {
            "final_reward_mean": np.mean(recent_rewards) if recent_rewards else 0.0,
            "final_reward_std": np.std(recent_rewards) if recent_rewards else 0.0,
            "final_length_mean": np.mean(recent_lengths) if recent_lengths else 0.0,
            "total_episodes": len(self.episode_rewards[agent_id]),
            "total_steps": self.global_step,
        }

    @classmethod
    def from_yaml(cls, config_path: str) -> "Trainer":
        """
        Create trainer from YAML configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Initialized Trainer instance
        """
        config = FullConfig.from_yaml(config_path)
        return cls(config)
