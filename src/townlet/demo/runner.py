"""Demo runner for multi-day training."""

import logging
import signal
import time
from pathlib import Path

import torch
import yaml

from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.demo.database import DemoDatabase
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation
from townlet.training.tensorboard_logger import TensorBoardLogger

logger = logging.getLogger(__name__)


class DemoRunner:
    """Orchestrates multi-day demo training with checkpointing."""

    def __init__(
        self,
        config_path: Path | str,
        db_path: Path | str,
        checkpoint_dir: Path | str,
        max_episodes: int = 10000,
    ):
        """Initialize demo runner.

        Args:
            config_path: Path to YAML config file
            db_path: Path to SQLite database
            checkpoint_dir: Directory for checkpoint files
            max_episodes: Maximum number of episodes to run
        """
        self.config_path = Path(config_path)
        self.db_path = Path(db_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_episodes = max_episodes
        self.current_episode = 0

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db = DemoDatabase(self.db_path)

        # Initialize TensorBoard logger
        # If checkpoint_dir is in a structured run (e.g., runs/L1_full_observability/2025-11-02_143022/checkpoints)
        # then tensorboard should be at sibling level, not inside checkpoints
        parent_dir = self.checkpoint_dir.parent
        if parent_dir.parent.parent.name == "runs" and self.checkpoint_dir.name == "checkpoints":
            # New structure: runs/LX_name/timestamp/checkpoints -> use runs/LX_name/timestamp/tensorboard
            tb_log_dir = parent_dir / "tensorboard"
        else:
            # Old structure or direct invocation: put tensorboard inside checkpoint_dir
            tb_log_dir = self.checkpoint_dir / "tensorboard"

        self.tb_logger = TensorBoardLogger(
            log_dir=tb_log_dir,
            flush_every=10,
        )

        # Load config
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize components (will be created in run())
        self.env = None
        self.population = None
        self.curriculum = None
        self.exploration = None

        # Shutdown flag
        self.should_shutdown = False
        # Signal handlers only work in main thread
        try:
            signal.signal(signal.SIGTERM, self._handle_shutdown)
            signal.signal(signal.SIGINT, self._handle_shutdown)
        except ValueError:
            # Running in a worker thread (e.g., unified server)
            # Signal handling will be done by the orchestrator
            pass

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_shutdown = True

    def flush_all_agents(self):
        """Flush all agents' episodes to replay buffer before checkpoint."""
        if not self.population:
            return

        for agent_idx in range(self.population.num_agents):
            self.population.flush_episode(agent_idx, synthetic_done=False)

    def save_checkpoint(self):
        """Save checkpoint at current episode."""
        # P1.1 Phase 5: Flush all agents before checkpoint
        self.flush_all_agents()

        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{self.current_episode:05d}.pt"

        # P1.1 Phase 2: Use population's complete checkpoint state
        checkpoint = {
            "version": 2,  # P1.1: Checkpoint format version for future migration
            "episode": self.current_episode,
            "timestamp": time.time(),
        }

        # Add full population state (includes q_network, optimizer, replay_buffer, etc.)
        if self.population:
            checkpoint["population_state"] = self.population.get_checkpoint_state()

        # P1.1 Phase 3: Add curriculum state (agent stages, performance trackers)
        if self.curriculum:
            checkpoint["curriculum_state"] = self.curriculum.state_dict()

        # P1.1 Phase 4: Add affordance layout (grid positions)
        if self.env:
            checkpoint["affordance_layout"] = self.env.get_affordance_positions()

        # P1.1 Phase 6: Add agent_ids for multi-agent coordination
        if self.population:
            checkpoint["agent_ids"] = self.population.agent_ids

        # Add epsilon for inference server display (legacy compatibility)
        if self.exploration and hasattr(self.exploration, "rnd") and hasattr(self.exploration.rnd, "epsilon"):
            checkpoint["epsilon"] = self.exploration.rnd.epsilon

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Update system state
        self.db.set_system_state("last_checkpoint", str(checkpoint_path))

    def load_checkpoint(self) -> int | None:
        """Load latest checkpoint if exists.

        Returns:
            Episode number of loaded checkpoint, or None if no checkpoint
        """
        # Find latest checkpoint
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_ep*.pt"))
        if not checkpoints:
            logger.info("No checkpoints found, starting from scratch")
            return None

        latest_checkpoint = checkpoints[-1]
        logger.info(f"Loading checkpoint: {latest_checkpoint}")

        checkpoint = torch.load(latest_checkpoint, weights_only=False)

        # P1.1: Check checkpoint version
        checkpoint_version = checkpoint.get("version", 1)  # Default to v1 for legacy checkpoints
        if checkpoint_version != 2:
            logger.warning(f"Loading v{checkpoint_version} checkpoint (current version is 2)")
            # Future: Add migration logic here if needed

        self.current_episode = checkpoint["episode"]

        # P1.1 Phase 2: Load full population state
        if "population_state" in checkpoint and self.population:
            self.population.load_checkpoint_state(checkpoint["population_state"])

        # P1.1 Phase 3: Load curriculum state (agent stages, performance trackers)
        if "curriculum_state" in checkpoint and self.curriculum:
            self.curriculum.load_state_dict(checkpoint["curriculum_state"])

        # P1.1 Phase 4: Load affordance layout (grid positions)
        if "affordance_layout" in checkpoint and self.env:
            self.env.set_affordance_positions(checkpoint["affordance_layout"])

        # P1.1 Phase 6: Restore agent_ids for multi-agent coordination
        if "agent_ids" in checkpoint and self.population:
            self.population.agent_ids = checkpoint["agent_ids"]

        logger.info(f"Resumed from episode {self.current_episode}")
        return self.current_episode

    def run(self):
        """Run demo training loop."""
        logger.info(f"Starting demo runner: {self.max_episodes} episodes")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Checkpoints: {self.checkpoint_dir}")

        # Initialize training components
        device_str = self.config.get("training", {}).get("device", "cuda")
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        # Extract config parameters
        curriculum_cfg = self.config.get("curriculum", {})
        exploration_cfg = self.config.get("exploration", {})
        population_cfg = self.config.get("population", {})
        environment_cfg = self.config.get("environment", {})

        # Get environment parameters from config
        num_agents = population_cfg.get("num_agents", 1)
        grid_size = environment_cfg.get("grid_size", 8)
        partial_observability = environment_cfg.get("partial_observability", False)
        vision_range = environment_cfg.get("vision_range", 2)
        enabled_affordances = environment_cfg.get("enabled_affordances", None)  # None = all affordances

        # Create environment FIRST (need it to auto-detect dimensions)
        self.env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=grid_size,
            device=device,
            partial_observability=partial_observability,
            vision_range=vision_range,
            enabled_affordances=enabled_affordances,
        )

        # Auto-detect dimensions from environment (avoids hardcoded config values)
        obs_dim = self.env.observation_dim
        action_dim = self.env.action_dim

        # Create curriculum
        self.curriculum = AdversarialCurriculum(
            max_steps_per_episode=curriculum_cfg.get("max_steps_per_episode", 500),
            survival_advance_threshold=curriculum_cfg.get("survival_advance_threshold", 0.7),
            survival_retreat_threshold=curriculum_cfg.get("survival_retreat_threshold", 0.3),
            entropy_gate=curriculum_cfg.get("entropy_gate", 0.5),
            min_steps_at_stage=curriculum_cfg.get("min_steps_at_stage", 1000),
            device=device,
        )

        # Create exploration (use auto-detected obs_dim)
        self.exploration = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,
            embed_dim=exploration_cfg.get("embed_dim", 128),
            initial_intrinsic_weight=exploration_cfg.get("initial_intrinsic_weight", 1.0),
            variance_threshold=exploration_cfg.get("variance_threshold", 100.0),  # Increased from 10.0
            survival_window=exploration_cfg.get("survival_window", 100),
            device=device,
        )

        # Get population parameters from config
        learning_rate = population_cfg.get("learning_rate", 0.00025)
        gamma = population_cfg.get("gamma", 0.99)
        replay_buffer_capacity = population_cfg.get("replay_buffer_capacity", 10000)
        network_type = population_cfg.get("network_type", "simple")  # 'simple' or 'recurrent'
        vision_window_size = 2 * vision_range + 1  # 5 for vision_range=2

        # Create agent IDs
        agent_ids = [f"agent_{i}" for i in range(num_agents)]

        # Create population with correct API
        self.population = VectorizedPopulation(
            env=self.env,
            curriculum=self.curriculum,
            exploration=self.exploration,
            agent_ids=agent_ids,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            replay_buffer_capacity=replay_buffer_capacity,
            network_type=network_type,
            vision_window_size=vision_window_size,
            tb_logger=self.tb_logger,
        )

        self.curriculum.initialize_population(num_agents)

        # Try to resume from checkpoint
        loaded_episode = self.load_checkpoint()
        if loaded_episode is not None:
            self.current_episode = loaded_episode + 1

        # Phase 4 - Log hyperparameters to TensorBoard
        self.hparams = {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "network_type": network_type,
            "replay_buffer_capacity": replay_buffer_capacity,
            "grid_size": environment_cfg.get("grid_size", 8),
            "partial_observability": environment_cfg.get("partial_observability", False),
            "vision_range": vision_range,
            "enable_temporal": environment_cfg.get("enable_temporal_mechanics", False),
            "initial_intrinsic_weight": exploration_cfg.get("initial_intrinsic_weight", 1.0),
            "variance_threshold": exploration_cfg.get("variance_threshold", 100.0),
            "max_steps_per_episode": curriculum_cfg.get("max_steps_per_episode", 500),
        }
        # Note: final metrics will be logged at end of training
        self.tb_logger.log_hyperparameters(hparams=self.hparams, metrics={})

        # Mark training started
        self.db.set_system_state("training_status", "running")
        self.db.set_system_state("start_time", str(time.time()))

        # Training loop
        try:
            while self.current_episode < self.max_episodes and not self.should_shutdown:
                episode_start = time.time()

                # Generalization test at episode 5000
                if self.current_episode == 5000:
                    logger.info("=" * 60)
                    logger.info("GENERALIZATION TEST: Randomizing affordance positions")
                    logger.info("=" * 60)

                    # Store old positions
                    old_positions = self.env.get_affordance_positions()
                    logger.info(f"Old positions: {old_positions}")

                    # Randomize
                    self.env.randomize_affordance_positions()

                    # Store new positions
                    new_positions = self.env.get_affordance_positions()
                    logger.info(f"New positions: {new_positions}")

                    # Mark in database
                    import json

                    self.db.set_system_state("affordance_randomization_episode", "5000")
                    self.db.set_system_state("old_affordance_positions", json.dumps(old_positions))
                    self.db.set_system_state("new_affordance_positions", json.dumps(new_positions))

                # Reset environment and population
                # NOTE: Only call population.reset(), which internally calls env.reset()
                # Calling both causes double reset which breaks affordance randomization
                self.population.reset()

                # Run episode
                survival_time = 0
                episode_reward = 0.0
                episode_extrinsic_reward = 0.0
                episode_intrinsic_reward = 0.0
                max_steps = 500

                # Track meters and affordances for Phase 3
                final_meters = None
                affordance_visits = {}

                for step in range(max_steps):
                    agent_state = self.population.step_population(self.env)

                    survival_time += 1

                    # Track rewards separately
                    # agent_state.rewards is total (extrinsic + intrinsic * weight)
                    # agent_state.intrinsic_rewards is just intrinsic component
                    intrinsic_weight = self.exploration.get_intrinsic_weight() if hasattr(self.exploration, "get_intrinsic_weight") else 1.0

                    # Calculate extrinsic reward tensor for curriculum (not just scalar)
                    extrinsic_reward_tensor = agent_state.rewards - (agent_state.intrinsic_rewards * intrinsic_weight)

                    # NOTE: Curriculum update removed from here (was per-step, now per-episode)
                    # See curriculum update after episode ends below

                    # Extract scalars for logging
                    extrinsic_component = extrinsic_reward_tensor[0].item()
                    intrinsic_component = agent_state.intrinsic_rewards[0].item()

                    episode_reward += agent_state.rewards[0].item()
                    episode_extrinsic_reward += extrinsic_component
                    episode_intrinsic_reward += intrinsic_component

                    # Track affordance usage from info dict
                    if "successful_interactions" in agent_state.info:
                        for agent_idx, affordance_name in agent_state.info["successful_interactions"].items():
                            affordance_visits[affordance_name] = affordance_visits.get(affordance_name, 0) + 1

                    if agent_state.dones[0]:
                        # Capture final meters before reset
                        final_meters = self.env.meters[0].cpu()
                        break

                # If episode didn't end, capture current meters
                if final_meters is None:
                    final_meters = self.env.meters[0].cpu()

                # Update curriculum ONCE per episode with pure survival signal
                # This gives curriculum a clean, interpretable metric: steps survived
                curriculum_survival_tensor = torch.tensor([float(survival_time)], dtype=torch.float32, device=self.env.device)
                curriculum_done_tensor = torch.tensor([True], dtype=torch.bool, device=self.env.device)
                self.population.update_curriculum_tracker(curriculum_survival_tensor, curriculum_done_tensor)

                # P1.2: Flush episode if agent survived to max_steps (recurrent networks only)
                # Without this, successful episodes never reach replay buffer → memory leak + data loss
                # CRITICAL: Loop over all agents to support multi-agent configs (not just agent 0)
                for agent_idx in range(self.population.num_agents):
                    if not agent_state.dones[agent_idx]:  # Agent survived to max_steps without dying
                        self.population.flush_episode(agent_idx=agent_idx, synthetic_done=True)

                # Log metrics to database
                self.db.insert_episode(
                    episode_id=self.current_episode,
                    timestamp=time.time(),
                    survival_time=survival_time,
                    total_reward=episode_reward,
                    extrinsic_reward=episode_extrinsic_reward,
                    intrinsic_reward=episode_intrinsic_reward,
                    intrinsic_weight=self.exploration.get_intrinsic_weight(),
                    curriculum_stage=self.curriculum.tracker.agent_stages[0].item(),
                    epsilon=self.exploration.rnd.epsilon,
                )

                # Log to TensorBoard (Phase 1 - Episode metrics)
                self.tb_logger.log_episode(
                    episode=self.current_episode,
                    survival_time=survival_time,
                    total_reward=episode_reward,
                    extrinsic_reward=episode_extrinsic_reward,
                    intrinsic_reward=episode_intrinsic_reward,
                    curriculum_stage=int(self.curriculum.tracker.agent_stages[0].item()),
                    epsilon=self.exploration.rnd.epsilon,
                    intrinsic_weight=self.exploration.get_intrinsic_weight(),
                )

                # Phase 2 - Training metrics (if training occurred this episode)
                training_metrics = self.population.get_training_metrics()
                if training_metrics["training_step"] > 0:
                    self.tb_logger.log_training_step(
                        step=training_metrics["training_step"],
                        td_error=training_metrics["td_error"],
                        loss=training_metrics["loss"],
                    )

                # Phase 3 - Meter dynamics and affordance usage
                if final_meters is not None:
                    # Convert meter names to indices (energy, hygiene, satiation, money, mood, social, health, fitness)
                    meter_names = [
                        "energy",
                        "hygiene",
                        "satiation",
                        "money",
                        "mood",
                        "social",
                        "health",
                        "fitness",
                    ]
                    meter_dict = {name: final_meters[i].item() for i, name in enumerate(meter_names)}
                    self.tb_logger.log_meters(
                        episode=self.current_episode,
                        step=survival_time,
                        meters=meter_dict,
                    )

                # Log affordance usage
                if affordance_visits:
                    self.tb_logger.log_affordance_usage(episode=self.current_episode, affordance_counts=affordance_visits)

                # Heartbeat log every 10 episodes
                if self.current_episode % 10 == 0:
                    elapsed = time.time() - episode_start
                    logger.info(
                        f"Episode {self.current_episode}/{self.max_episodes} | "
                        f"Survival: {survival_time} steps | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Intrinsic Weight: {self.exploration.get_intrinsic_weight():.3f} | "
                        f"Stage: {self.curriculum.tracker.agent_stages[0].item()}/5 | "
                        f"Time: {elapsed:.2f}s"
                    )

                # Checkpoint every 100 episodes
                if self.current_episode % 100 == 0:
                    self.save_checkpoint()

                # Decay epsilon for next episode
                self.exploration.decay_epsilon()

                self.current_episode += 1

        finally:
            # Save final checkpoint
            logger.info("Training complete, saving final checkpoint...")
            self.save_checkpoint()

            # Phase 4 - Log final metrics with hyperparameters
            if self.population is not None:
                final_metrics = {
                    "final_episode": self.current_episode,
                    "total_training_steps": self.population.total_steps,
                }
                # Re-log hyperparameters with final metrics for comparison
                if hasattr(self, "hparams"):
                    self.tb_logger.log_hyperparameters(hparams=self.hparams, metrics=final_metrics)

            self.db.set_system_state("training_status", "completed")
            self.db.close()

            # Close TensorBoard logger
            self.tb_logger.close()


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    # Get paths from environment or args
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/townlet/sparse_adaptive.yaml"
    db_path = sys.argv[2] if len(sys.argv) > 2 else "demo_state.db"
    checkpoint_dir = sys.argv[3] if len(sys.argv) > 3 else "checkpoints"
    max_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 10000

    runner = DemoRunner(
        config_path=config_path,
        db_path=db_path,
        checkpoint_dir=checkpoint_dir,
        max_episodes=max_episodes,
    )
    runner.run()
