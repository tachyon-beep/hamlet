"""Live inference server for multi-day demo.

Runs inference on the latest checkpoint while training happens in background.
Provides step-by-step visualization at human-watchable speed.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from townlet.agent.brain_config import compute_brain_hash, load_brain_config
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.demo.database import DemoDatabase
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation
from townlet.recording.replay import ReplayManager
from townlet.substrate.continuous import ContinuousSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate
from townlet.substrate.gridnd import GridNDSubstrate
from townlet.training.checkpoint_utils import safe_torch_load, verify_checkpoint_digest
from townlet.universe.compiled import CompiledUniverse
from townlet.universe.compiler import UniverseCompiler
from townlet.universe.runtime import RuntimeUniverse

logger = logging.getLogger(__name__)

TELEMETRY_SCHEMA_VERSION = "1.0.0"


def build_agent_telemetry_payload(
    population: VectorizedPopulation | None,
    episode_index: int | None = None,
) -> dict[str, Any]:
    """Build JSON-safe telemetry snapshot for all agents."""
    if population is None:
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "episode_index": episode_index,
            "agents": [],
        }

    snapshot = population.build_telemetry_snapshot(episode_index=episode_index)
    snapshot["schema_version"] = TELEMETRY_SCHEMA_VERSION
    return snapshot


class LiveInferenceServer:
    """Runs inference on latest checkpoint with step-by-step WebSocket streaming."""

    def __init__(
        self,
        checkpoint_dir: Path | str,
        port: int = 8766,
        step_delay: float = 0.2,
        total_episodes: int = 5000,  # Expected total episodes in training run
        config_dir: Path | str | None = None,  # Config pack directory
        training_config_path: Path | str | None = None,  # Optional training config
        db_path: Path | str | None = None,  # Optional database path for replay
        recordings_dir: Path | str | None = None,  # Optional recordings directory for replay
    ):
        """Initialize live inference server.

        Args:
            checkpoint_dir: Directory containing training checkpoints
            port: WebSocket port
            step_delay: Delay between steps in seconds (0.2 = 5 steps/sec)
            total_episodes: Expected total episodes for training run (for progress gauge)
            config_dir: Optional config pack directory
            training_config_path: Optional path to training config YAML (for matching environment settings)
            db_path: Optional database path for replay mode
            recordings_dir: Optional recordings directory for replay mode
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.port = port
        self.step_delay = step_delay
        self.total_episodes = total_episodes
        logger.info(f"LiveInferenceServer initialized with total_episodes={total_episodes}")
        if config_dir is None:
            raise ValueError("LiveInferenceServer requires config_dir to compile the universe.")
        self.config_dir = Path(config_dir)
        self.training_config_path: Path | None = Path(training_config_path) if training_config_path else None
        self.compiler = UniverseCompiler()
        self.compiled_universe: CompiledUniverse | None = None
        self.runtime_universe: RuntimeUniverse | None = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clients: set[WebSocket] = set()

        # Current checkpoint tracking
        self.current_checkpoint_path: Path | None = None
        self.current_checkpoint_episode: int = 0
        self.current_epsilon: float = 0.0

        # Environment and agent
        self.env: VectorizedHamletEnv | None = None
        self.population: VectorizedPopulation | None = None
        self.curriculum: AdversarialCurriculum | None = None
        self.exploration: AdaptiveIntrinsicExploration | None = None

        # Episode state (live inference mode)
        self.is_running = False
        self.current_episode = 0
        self.current_step = 0

        # Affordance interaction tracking (for UI display)
        self.affordance_interactions: dict[str, int] = {}  # {affordance_name: count}

        # Checkpoint auto-update mode
        self.auto_checkpoint_mode = False  # If true, automatically check for new checkpoints after each episode

        # Replay mode
        self.mode: str = "inference"  # "inference" or "replay"
        self.replay_manager: ReplayManager | None = None
        self.replay_playing: bool = False

        # Initialize replay if database provided
        if db_path and recordings_dir:
            try:
                database = DemoDatabase(Path(db_path))
                self.replay_manager = ReplayManager(database, Path(recordings_dir))
                logger.info(f"Replay mode initialized: db={db_path}, recordings={recordings_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize replay manager: {e}")
                self.replay_manager = None

        # Q-value logging (auto-closed via shutdown / destructor)
        self._qvalue_log_path = Path("qvalues_inference.log")
        self._qvalue_log_file = self._open_qvalue_log()

        # FastAPI app
        self.app = FastAPI(title="Hamlet Live Inference Server")

        # Enable CORS for frontend
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self.app.websocket("/ws")(self.websocket_endpoint)
        self.app.websocket("/ws/training")(self.websocket_endpoint)  # Same endpoint, different name
        self.app.on_event("startup")(self.startup)
        self.app.on_event("shutdown")(self.shutdown)

    def _build_agent_telemetry(self) -> dict[str, Any]:
        """Return telemetry payload for current agent registry."""
        if not self.population:
            return {"schema_version": TELEMETRY_SCHEMA_VERSION, "episode_index": None, "agents": []}
        return build_agent_telemetry_payload(self.population, episode_index=self.current_episode)

    def _build_substrate_metadata(self) -> dict[str, Any]:
        """Build substrate metadata for WebSocket messages.

        Returns:
            Dict with substrate type, dimensions, and topology (if applicable).
            Used by frontend to dispatch correct renderer.

        Example:
            Grid2D: {"type": "grid2d", "position_dim": 2, "topology": "square", "width": 8, "height": 8, ...}
            GridND: {"type": "gridnd", "position_dim": 7, "topology": "hypercube", "dimension_sizes": [5,5,5,5,5,5,5], ...}
            Continuous2D: {"type": "continuous2d", "position_dim": 2, "bounds": [...], ...}
            Aspatial: {"type": "aspatial", "position_dim": 0}
        """
        if not self.env:
            return {"type": "unknown", "position_dim": 0}

        substrate = self.env.substrate

        # Derive substrate type from class name (Grid2DSubstrate -> "grid2d")
        substrate_type = type(substrate).__name__.lower().replace("substrate", "")

        metadata = {
            "type": substrate_type,
            "position_dim": substrate.position_dim,
        }

        # Add topology if substrate has it (grid substrates only)
        if hasattr(substrate, "topology"):
            metadata["topology"] = substrate.topology

        # Add type-specific metadata with type narrowing
        if isinstance(substrate, Grid2DSubstrate):
            metadata["width"] = substrate.width
            metadata["height"] = substrate.height
            metadata["boundary"] = substrate.boundary
            metadata["distance_metric"] = substrate.distance_metric

        elif isinstance(substrate, Grid3DSubstrate):
            metadata["width"] = substrate.width
            metadata["height"] = substrate.height
            metadata["depth"] = substrate.depth
            metadata["boundary"] = substrate.boundary
            metadata["distance_metric"] = substrate.distance_metric

        elif isinstance(substrate, GridNDSubstrate):
            metadata["dimension_sizes"] = substrate.dimension_sizes
            metadata["boundary"] = substrate.boundary
            metadata["distance_metric"] = substrate.distance_metric

        elif isinstance(substrate, ContinuousSubstrate):
            # Continuous substrates (1D/2D/3D/ND)
            metadata["bounds"] = substrate.bounds
            metadata["boundary"] = substrate.boundary
            metadata["movement_delta"] = substrate.movement_delta
            metadata["interaction_radius"] = substrate.interaction_radius
            metadata["distance_metric"] = substrate.distance_metric

        # Aspatial has no additional metadata

        return metadata

    async def startup(self):
        """Initialize environment and start checkpoint monitoring."""
        logger.info("Starting live inference server")

        if self.training_config_path:
            logger.info(f"Training config override provided: {self.training_config_path}")

        # Initialize environment and components
        self._initialize_components()

        # Load initial checkpoint
        await self._check_and_load_checkpoint()

        logger.info(f"Loaded checkpoint: {self.current_checkpoint_path.name if self.current_checkpoint_path else 'None'}")

    async def shutdown(self):
        """Cleanup on shutdown."""
        self.is_running = False
        logger.info("Live inference server shut down")
        self._close_qvalue_log()

    def __del__(self):
        """Best-effort cleanup when GC collects the server."""
        try:
            self._close_qvalue_log()
        except Exception:
            # Destructors must never raise; loggers may already be torn down
            pass

    def _open_qvalue_log(self):
        """Open Q-value log file with line buffering."""
        try:
            return self._qvalue_log_path.open("w", buffering=1)
        except OSError as exc:
            logger.warning(f"Unable to open Q-value log at {self._qvalue_log_path}: {exc}")
            return None

    def _close_qvalue_log(self):
        """Close Q-value log if open."""
        if self._qvalue_log_file:
            try:
                self._qvalue_log_file.close()
            finally:
                self._qvalue_log_file = None

    def _initialize_components(self):
        """Initialize environment and agent components."""
        logger.info("Compiling universe for live inference from %s", self.config_dir)
        self.compiled_universe = self.compiler.compile(self.config_dir)
        self.runtime_universe = self.compiled_universe.to_runtime()
        hamlet_config = self.compiled_universe.hamlet_config
        env_cfg = hamlet_config.environment
        population_cfg = hamlet_config.population
        curriculum_cfg = hamlet_config.curriculum
        exploration_cfg = hamlet_config.exploration
        training_cfg = hamlet_config.training

        num_agents = population_cfg.num_agents
        network_type = population_cfg.network_type
        vision_range = env_cfg.vision_range
        partial_observability = env_cfg.partial_observability
        enable_temporal_mechanics = env_cfg.enable_temporal_mechanics
        vision_window_size = 2 * vision_range + 1

        logger.info(
            "Environment config: grid=%s, POMDP=%s, vision=%s, temporal=%s, affordances=%s",
            self.runtime_universe.metadata.grid_size,  # Read from substrate.yaml via metadata
            partial_observability,
            vision_range,
            enable_temporal_mechanics,
            env_cfg.enabled_affordances if env_cfg.enabled_affordances else "all",
        )
        logger.info("Network type: %s (num_agents=%s)", network_type, num_agents)

        self.env = VectorizedHamletEnv.from_universe(
            self.compiled_universe,
            num_agents=num_agents,
            device=self.device,
        )

        obs_dim = self.runtime_universe.metadata.observation_dim

        # Create curriculum
        self.curriculum = AdversarialCurriculum(
            max_steps_per_episode=curriculum_cfg.max_steps_per_episode,
            survival_advance_threshold=curriculum_cfg.survival_advance_threshold,
            survival_retreat_threshold=curriculum_cfg.survival_retreat_threshold,
            entropy_gate=curriculum_cfg.entropy_gate,
            min_steps_at_stage=curriculum_cfg.min_steps_at_stage,
            device=self.device,
        )

        # Create exploration (for inference, we want greedy)
        # Conditionally pass active_mask based on mask_unused_obs config
        active_mask = self.env.observation_activity.active_mask if population_cfg.mask_unused_obs else None
        self.exploration = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,
            embed_dim=exploration_cfg.embed_dim,
            rnd_training_batch_size=training_cfg.batch_size,  # Use main batch_size from config
            initial_intrinsic_weight=exploration_cfg.initial_intrinsic_weight,
            variance_threshold=exploration_cfg.variance_threshold,
            survival_window=exploration_cfg.survival_window,
            epsilon_start=training_cfg.epsilon_start,
            epsilon_decay=training_cfg.epsilon_decay,
            epsilon_min=training_cfg.epsilon_min,
            device=self.device,
            active_mask=active_mask,
        )

        # Load brain.yaml (REQUIRED for all config packs)
        brain_yaml_path = self.config_dir / "brain.yaml"
        logger.info(f"Loading brain configuration from {brain_yaml_path}")
        brain_config = load_brain_config(self.config_dir)
        brain_hash = compute_brain_hash(brain_config)
        logger.info(f"Brain config loaded: {brain_config.description}")
        logger.info(f"Brain hash: {brain_hash[:16]}... (SHA256)")

        # Store brain_config for checkpoint provenance
        self.brain_config = brain_config
        self.brain_hash = brain_hash

        # Create population (brain_config provides network/optimizer/Q-learning parameters)
        agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self.population = VectorizedPopulation(
            env=self.env,
            curriculum=self.curriculum,
            exploration=self.exploration,
            agent_ids=agent_ids,
            device=self.device,
            obs_dim=obs_dim,
            action_dim=self.env.action_dim,
            learning_rate=population_cfg.learning_rate,  # None (managed by brain.yaml)
            gamma=population_cfg.gamma,  # None (managed by brain.yaml)
            replay_buffer_capacity=population_cfg.replay_buffer_capacity,  # None (managed by brain.yaml)
            network_type=network_type,
            vision_window_size=vision_window_size,
            train_frequency=training_cfg.train_frequency,
            target_update_frequency=training_cfg.target_update_frequency,  # None (managed by brain.yaml)
            batch_size=training_cfg.batch_size,
            sequence_length=training_cfg.sequence_length,
            max_grad_norm=training_cfg.max_grad_norm,
            use_double_dqn=training_cfg.use_double_dqn,  # None (managed by brain.yaml)
            brain_config=brain_config,
            max_episodes=None,  # Not used by live inference
            max_steps_per_episode=None,  # Not used by live inference
        )

        self.curriculum.initialize_population(num_agents)

    async def _check_and_load_checkpoint(self) -> bool:
        """Check for new checkpoints and load if available.

        Returns:
            True if a new checkpoint was loaded
        """
        # Find all checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_ep*.pt"))
        if not checkpoints:
            logger.warning(f"No checkpoints found in {self.checkpoint_dir}")
            return False

        # Get latest checkpoint
        latest_checkpoint = checkpoints[-1]

        # Check if it's newer than current
        if latest_checkpoint == self.current_checkpoint_path:
            return False

        # Extract episode number from filename
        try:
            episode_str = latest_checkpoint.stem.split("_ep")[1]
            episode_num = int(episode_str)
        except (IndexError, ValueError) as exc:
            logger.error(f"Could not parse episode number from {latest_checkpoint.name}")
            logger.debug("Checkpoint parsing error", exc_info=exc)
            return False

        # Load checkpoint
        logger.info(f"Loading checkpoint: {latest_checkpoint.name} (episode {episode_num})")

        verify_checkpoint_digest(latest_checkpoint, required=False)
        checkpoint = safe_torch_load(latest_checkpoint, weights_only=False)

        # Load Q-network weights
        if "population_state" in checkpoint:
            assert self.population is not None, "Population must be initialized before loading checkpoint"
            self.population.q_network.load_state_dict(checkpoint["population_state"]["q_network"])
            logger.info("Loaded Q-network weights")

        # Calculate epsilon based on training progress
        # For inference, we estimate epsilon from episode number (linear decay from 1.0 to 0.05)
        epsilon = checkpoint.get("epsilon", None)
        logger.info(f"Checkpoint contains epsilon: {epsilon} (type: {type(epsilon).__name__})")
        if epsilon is None:
            # Estimate epsilon based on training progress (assuming linear decay over total_episodes)
            progress = episode_num / self.total_episodes if self.total_episodes > 0 else 0
            epsilon = max(0.05, 1.0 - (progress * 0.95))  # Decay from 1.0 to 0.05
            logger.info(
                "Estimated epsilon from training progress: episode=%s, total=%s, progress=%.3f, epsilon=%.3f",
                episode_num,
                self.total_episodes,
                progress,
                epsilon,
            )
        else:
            logger.info(f"Loaded epsilon from checkpoint: {epsilon:.3f}")

        # Update tracking
        self.current_checkpoint_path = latest_checkpoint
        self.current_checkpoint_episode = episode_num
        self.current_epsilon = epsilon

        # Broadcast model update to all clients
        await self._broadcast_to_clients(
            {
                "type": "model_loaded",
                "model": f"checkpoint_ep{episode_num:05d}",
                "episode": episode_num,
                "total_episodes": self.total_episodes,
                "epsilon": epsilon,
                "message": f"Loaded model from episode {episode_num}",
            }
        )

        return True

    async def websocket_endpoint(self, websocket: WebSocket):
        """WebSocket endpoint for client connections."""
        await websocket.accept()
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")

        # Send connection message with substrate metadata
        await websocket.send_json(
            {
                "type": "connected",
                "message": "Connected to live inference server",
                "available_models": [],
                "mode": "inference",
                "checkpoint": f"checkpoint_ep{self.current_checkpoint_episode:05d}" if self.current_checkpoint_path else "None",
                "checkpoint_episode": self.current_checkpoint_episode,
                "total_episodes": self.total_episodes,
                "epsilon": self.current_epsilon,
                "auto_checkpoint_mode": self.auto_checkpoint_mode,
                "substrate": self._build_substrate_metadata(),
            }
        )

        try:
            # Handle incoming commands
            while True:
                data = await websocket.receive_json()
                await self._handle_command(websocket, data)
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client removed. Total clients: {len(self.clients)}")

    async def _handle_command(self, websocket: WebSocket, data: dict):
        """Handle incoming WebSocket commands."""
        command = data.get("command") or data.get("type")

        # Replay mode commands
        if command == "load_replay":
            await self._handle_load_replay(websocket, data)

        elif command == "list_recordings":
            await self._handle_list_recordings(websocket, data)

        elif command == "replay_control":
            await self._handle_replay_control(websocket, data)

        # Live inference mode commands
        elif command == "play":
            if self.mode == "replay":
                self.replay_playing = True
                asyncio.create_task(self._run_replay_loop())
                logger.info("Started replay playback")
            else:
                if not self.is_running:
                    self.is_running = True
                    asyncio.create_task(self._run_inference_loop())
                    logger.info("Started inference loop")

        elif command == "pause":
            if self.mode == "replay":
                self.replay_playing = False
                logger.info("Paused replay playback")
            else:
                self.is_running = False
                logger.info("Paused inference loop")

        elif command == "step":
            # Run a single step
            if self.mode == "replay":
                asyncio.create_task(self._replay_single_step())
            else:
                if not self.is_running:
                    asyncio.create_task(self._run_single_episode())

        elif command == "reset":
            if self.mode == "replay":
                if self.replay_manager and self.replay_manager.is_loaded():
                    self.replay_manager.reset()
                    await self._send_replay_step()
                    logger.info("Reset replay to beginning")
            else:
                # Reset episode
                self.current_episode = 0
                self.current_step = 0
                logger.info("Reset episode counter")

        elif command == "refresh_checkpoint":
            # Manually check for and load new checkpoint
            logger.info("Manual checkpoint refresh requested")
            checkpoint_loaded = await self._check_and_load_checkpoint()
            if checkpoint_loaded:
                logger.info("New checkpoint loaded")
            else:
                logger.info("No new checkpoint available")

        elif command == "toggle_auto_checkpoint":
            # Toggle auto checkpoint mode
            self.auto_checkpoint_mode = not self.auto_checkpoint_mode
            status = "enabled" if self.auto_checkpoint_mode else "disabled"
            logger.info(f"Auto checkpoint mode {status}")
            # Broadcast mode change to all clients
            await self._broadcast_to_clients({"type": "auto_checkpoint_mode", "enabled": self.auto_checkpoint_mode})

        elif command == "set_speed":
            # Update step delay (convert speed multiplier to delay)
            # speed 1.0 = 0.2s delay (5 steps/sec)
            # speed 10.0 = 0.02s delay (50 steps/sec)
            speed = data.get("speed", 1.0)
            self.step_delay = 0.2 / speed  # Inverse relationship
            logger.info(f"Speed set to {speed}x (delay: {self.step_delay:.3f}s)")

    async def _run_inference_loop(self):
        """Main inference loop - runs episodes continuously."""
        logger.info("Inference loop started")

        while self.is_running:
            # Check for new checkpoint before each episode (if auto mode enabled)
            if self.auto_checkpoint_mode:
                await self._check_and_load_checkpoint()

            # Run one inference episode
            await self._run_single_episode()

            # Small delay between episodes
            await asyncio.sleep(0.5)

    async def _run_single_episode(self):
        """Run a single inference episode with step-by-step updates."""
        self.current_episode += 1
        self.current_step = 0

        # Reset affordance interaction tracking
        self.affordance_interactions = {name: 0 for name in self.env.affordances.keys()}

        # Randomize affordance positions for each episode (like training)
        self.env.randomize_affordance_positions()

        # Log affordance positions for verification
        affordance_positions = {name: pos.tolist() for name, pos in self.env.affordances.items()}
        logger.info(f"Episode {self.current_episode} affordance positions: {affordance_positions}")

        # Reset environment
        self.env.reset()
        self.population.reset()

        # Get initial curriculum decision to track stage
        from townlet.training.state import BatchedAgentState

        temp_state = BatchedAgentState(
            observations=self.population.current_obs,
            actions=torch.zeros(1, dtype=torch.long, device=self.device),
            rewards=torch.zeros(1, device=self.device),
            dones=torch.zeros(1, dtype=torch.bool, device=self.device),
            epsilons=torch.full((1,), self.current_epsilon, device=self.device),
            intrinsic_rewards=torch.zeros(1, device=self.device),
            survival_times=torch.zeros(1, device=self.device),
            curriculum_difficulties=torch.zeros(1, device=self.device),
            device=self.device,
        )
        curriculum_decisions = self.curriculum.get_batch_decisions(temp_state, ["agent_0"])
        current_stage = curriculum_decisions[0].difficulty_level
        current_multiplier = curriculum_decisions[0].depletion_multiplier

        # Sync curriculum metrics to runtime registry
        self.population.current_curriculum_decisions = curriculum_decisions
        self.population._sync_curriculum_metrics()
        episode_telemetry = self._build_agent_telemetry()
        agent_snapshot = (
            episode_telemetry["agents"][0]
            if episode_telemetry["agents"]
            else {
                "curriculum_stage": 1,
                "epsilon": self.current_epsilon,
            }
        )
        current_stage = int(agent_snapshot["curriculum_stage"])
        epsilon_snapshot = float(agent_snapshot["epsilon"])

        # Send episode start with curriculum info and substrate metadata
        await self._broadcast_to_clients(
            {
                "type": "episode_start",
                "episode": self.current_episode,
                "checkpoint": f"checkpoint_ep{self.current_checkpoint_episode:05d}",
                "checkpoint_episode": self.current_checkpoint_episode,
                "total_episodes": self.total_episodes,
                "epsilon": epsilon_snapshot,
                "curriculum_stage": current_stage,
                "curriculum_multiplier": float(current_multiplier),
                "telemetry": episode_telemetry,
                "substrate": self._build_substrate_metadata(),
            }
        )

        # Run episode
        done = False
        cumulative_reward = 0.0
        max_steps = 500

        while not done and self.current_step < max_steps:
            # Get action using epsilon-greedy (respects current epsilon from checkpoint)
            actions = self.population.select_epsilon_greedy_actions(self.env, self.current_epsilon)

            # Get Q-values for display
            with torch.no_grad():
                q_output = self.population.q_network(self.population.current_obs)
                q_values = q_output[0] if isinstance(q_output, tuple) else q_output

            # Step environment with curriculum difficulty
            next_obs, rewards, dones, info = self.env.step(actions, current_multiplier)

            # Track successful interactions (only count if interaction actually succeeded)
            successful_interactions = info.get("successful_interactions", {})
            if 0 in successful_interactions:
                affordance_name = successful_interactions[0]
                self.affordance_interactions[affordance_name] += 1

            # Update state
            self.population.current_obs = next_obs
            done = dones[0].item()
            step_reward = rewards[0].item()  # Capture step reward before accumulating
            cumulative_reward += step_reward
            self.current_step += 1

            # Send state update with Q-values and step reward
            await self._broadcast_state_update(cumulative_reward, actions[0].item(), q_values[0], step_reward)

            # Delay for human viewing
            if self.is_running:
                await asyncio.sleep(self.step_delay)

        # Episode complete - use actual cumulative reward earned during episode
        final_cumulative_reward = cumulative_reward
        final_telemetry = self._build_agent_telemetry()
        final_agent_snapshot = final_telemetry["agents"][0] if final_telemetry["agents"] else agent_snapshot
        final_stage = int(final_agent_snapshot["curriculum_stage"])
        final_epsilon = float(final_agent_snapshot["epsilon"])

        # Prepare affordance stats for death certificate
        affordance_stats = [
            {"name": name, "count": count} for name, count in sorted(self.affordance_interactions.items(), key=lambda x: x[1], reverse=True)
        ]

        # Get final meter states for death certificate (dynamic from config)
        final_meters = {}
        for meter_name, idx in self.env.meter_name_to_index.items():
            final_meters[meter_name] = float(self.env.meters[0, idx].item())

        # Get agent age and lifetime progress for death certificate
        agent_age = int(self.env.step_counts[0].item())
        lifetime_progress = float(agent_age / self.env.agent_lifespan)

        await self._broadcast_to_clients(
            {
                "type": "episode_end",
                "episode": self.current_episode,
                "steps": self.current_step,
                "total_reward": final_cumulative_reward,  # Actual cumulative reward earned
                "reason": "done" if done else "max_steps",
                "checkpoint": f"checkpoint_ep{self.current_checkpoint_episode:05d}",
                "checkpoint_episode": self.current_checkpoint_episode,
                "total_episodes": self.total_episodes,
                "epsilon": final_epsilon,
                "curriculum_stage": final_stage,
                "telemetry": final_telemetry,
                "affordance_stats": affordance_stats,  # Include affordance usage for death certificate
                "final_meters": final_meters,  # Meter states at death for death certificate
                "agent_age": agent_age,  # Total steps lived (resets with checkpoint)
                "lifetime_progress": lifetime_progress,  # 0.0-1.0 progress toward retirement
            }
        )

        logger.info(f"Episode {self.current_episode} complete: {self.current_step} steps, reward: {final_cumulative_reward:.2f}")

    async def _broadcast_state_update(self, cumulative_reward: float, last_action: int, q_values: torch.Tensor, step_reward: float = 1.0):
        """Broadcast current state to all clients."""
        # Ensure environment and population are initialized
        assert self.env is not None, "Environment must be initialized before broadcasting state"
        assert self.population is not None, "Population must be initialized before broadcasting state"

        # Get agent position (substrate-agnostic)
        # agent_pos is a list of length substrate.position_dim
        # - 2D: [x, y]
        # - 3D: [x, y, z]
        # - Aspatial: []
        agent_pos = self.env.positions[0].cpu().tolist()

        # Get action masks (which actions are valid)
        action_masks = self.env.get_action_masks()[0].cpu().tolist()
        if len(action_masks) < 6:
            action_masks.extend([False] * (6 - len(action_masks)))

        # Get meters dynamically from environment configuration
        # Use meter_name_to_index to support configs with varying meter sets
        meters = {}
        for meter_name, idx in self.env.meter_name_to_index.items():
            meters[meter_name] = self.env.meters[0, idx].item()

        # Get affordances (substrate-agnostic position handling)
        affordances = []
        for name, pos in self.env.affordances.items():
            pos_list = pos.cpu().tolist()
            affordance_data = {"type": name}  # Frontend expects 'type' not 'name'

            # Add position data based on substrate dimensionality
            if self.env.substrate.position_dim == 2:
                # 2D grid: use x, y
                affordance_data["x"] = pos_list[0]
                affordance_data["y"] = pos_list[1]
            elif self.env.substrate.position_dim == 3:
                # 3D grid: use x, y, z
                affordance_data["x"] = pos_list[0]
                affordance_data["y"] = pos_list[1]
                affordance_data["z"] = pos_list[2]
            # Aspatial (position_dim=0): no position data needed

            affordances.append(affordance_data)

        # Convert Q-values to list for JSON serialization (supports legacy 5-action checkpoints)
        q_values_list = q_values.cpu().tolist()
        if len(q_values_list) < 6:
            # Pad with NaNs so downstream consumers can detect legacy models gracefully
            q_values_list.extend([float("nan")] * (6 - len(q_values_list)))

        # Log Q-values and chosen action to file for debugging
        action_names_dict = self.env.get_action_label_names()
        padded_for_log = q_values_list[:6]
        log_line = (
            f"Step {self.current_step}: Action={action_names_dict.get(last_action, 'UNKNOWN')}, "
            f"Q-values: Up={padded_for_log[0]:.2f}, Down={padded_for_log[1]:.2f}, "
            f"Left={padded_for_log[2]:.2f}, Right={padded_for_log[3]:.2f}, "
            f"Interact={padded_for_log[4]:.2f}, Wait={padded_for_log[5]:.2f}\n"
        )
        if self._qvalue_log_file:
            self._qvalue_log_file.write(log_line)
            self._qvalue_log_file.flush()

        # Prepare affordance interaction counts (sorted by count descending)
        affordance_stats = [
            {"name": name, "count": count} for name, count in sorted(self.affordance_interactions.items(), key=lambda x: x[1], reverse=True)
        ]

        # TODO: Remove projected_reward from frontend (baseline tracking removed in PDR-002)
        # Legacy field - set to 0.0 for backwards compatibility with frontend
        projected_reward = 0.0

        # Build state update message
        update = {
            "type": "state_update",
            "step": self.current_step,
            "cumulative_reward": cumulative_reward,
            "step_reward": step_reward,  # Reward for this specific step (0-1 range)
            "projected_reward": projected_reward,  # Legacy field (always 0.0, baseline tracking removed)
            "epsilon": self.current_epsilon,  # Current exploration rate
            "checkpoint_episode": self.current_checkpoint_episode,  # For training progress bar
            "total_episodes": self.total_episodes,  # For training progress bar
            # DEBUG
            "_debug_total_episodes": self.total_episodes,
            "substrate": self._build_substrate_metadata(),
            "grid": self._build_grid_data(agent_pos, last_action, affordances),
            "agent_meters": {"agent_0": {"meters": meters}},  # MeterPanel expects agent_0.meters nested structure
            "q_values": q_values_list,  # Q-values for all 6 actions
            "action_masks": action_masks,  # Which actions are valid [6] bool list
            "affordance_stats": affordance_stats,  # Interaction counts sorted by frequency
            "telemetry": self._build_agent_telemetry(),
        }

        # Add temporal data (always present, even if temporal mechanics disabled)
        if hasattr(self.env, "time_of_day"):
            # Normalize interaction progress to 0-1 range
            interaction_progress_raw = self.env.interaction_progress[0].item() if hasattr(self.env, "interaction_progress") else 0
            interaction_progress_normalized = 0.0

            if (
                interaction_progress_raw > 0
                and hasattr(self.env, "last_interaction_affordance")
                and self.env.last_interaction_affordance[0] is not None
            ):
                affordance_name = self.env.last_interaction_affordance[0]
                # Get required ticks from affordance engine
                required_ticks = self.env.affordance_engine.get_duration_ticks(affordance_name)
                if required_ticks > 0:
                    interaction_progress_normalized = interaction_progress_raw / required_ticks

            # Get agent age and lifetime progress
            agent_age = int(self.env.step_counts[0].item())
            lifetime_progress = float(agent_age / self.env.agent_lifespan)

            update["temporal"] = {
                "time_of_day": self.env.time_of_day,
                "interaction_progress": interaction_progress_normalized,
                "agent_age": agent_age,  # Total steps lived (resets with checkpoint)
                "lifetime_progress": lifetime_progress,  # 0.0-1.0 progress toward retirement
            }

        await self._broadcast_to_clients(update)

    def _build_grid_data(self, agent_pos: list, last_action: int, affordances: list) -> dict:
        """Build grid data based on substrate type.

        Args:
            agent_pos: Agent position (list of length substrate.position_dim)
            last_action: Last action taken
            affordances: List of affordance dicts with positions

        Returns:
            Grid data dict for frontend rendering
        """
        from townlet.substrate.aspatial import AspatialSubstrate
        from townlet.substrate.grid2d import Grid2DSubstrate

        assert self.env is not None, "Environment must be initialized before building grid data"

        if isinstance(self.env.substrate, Grid2DSubstrate):
            # 2D grid rendering (current implementation)
            return {
                "type": "grid2d",
                "width": self.env.substrate.width,
                "height": self.env.substrate.height,
                "agents": [
                    {
                        "id": "agent_0",
                        "x": agent_pos[0],
                        "y": agent_pos[1],
                        "color": "#4CAF50",
                        "last_action": last_action,
                    }
                ],
                "affordances": affordances,
            }
        elif isinstance(self.env.substrate, AspatialSubstrate):
            # Aspatial rendering (meters-only, no grid)
            return {
                "type": "aspatial",
                # No position data for aspatial substrate
            }
        else:
            # Future: Grid3DSubstrate, GraphSubstrate, etc.
            return {
                "type": "unknown",
                "substrate_type": type(self.env.substrate).__name__,
                "message": "Rendering for this substrate type not yet implemented",
            }

    async def _broadcast_to_clients(self, message: dict):
        """Broadcast message to all connected clients."""
        dead_clients = set()
        # Iterate over a copy to avoid "Set changed size during iteration" error
        for client in list(self.clients):
            try:
                await client.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                dead_clients.add(client)

        # Remove dead clients
        self.clients -= dead_clients

    async def _handle_load_replay(self, websocket: WebSocket, data: dict):
        """Handle load_replay command."""
        if not self.replay_manager:
            await websocket.send_json({"type": "error", "message": "Replay not available (no database)"})
            return

        episode_id = data.get("episode_id")
        if episode_id is None:
            await websocket.send_json({"type": "error", "message": "Missing episode_id"})
            return

        # Stop any current playback
        self.is_running = False
        self.replay_playing = False

        # Load episode
        success = self.replay_manager.load_episode(episode_id)
        if not success:
            await websocket.send_json({"type": "error", "message": f"Failed to load episode {episode_id}"})
            return

        # Switch to replay mode
        self.mode = "replay"

        # Send confirmation
        metadata = self.replay_manager.get_metadata()
        if metadata is None:
            await websocket.send_json({"type": "error", "message": "Failed to load episode metadata"})
            return

        await self._broadcast_to_clients(
            {
                "type": "replay_loaded",
                "episode_id": episode_id,
                "metadata": {
                    "survival_steps": metadata["survival_steps"],
                    "total_reward": metadata["total_reward"],
                    "curriculum_stage": metadata["curriculum_stage"],
                    "timestamp": metadata["timestamp"],
                },
                "total_steps": self.replay_manager.get_total_steps(),
            }
        )

        # Send first step
        await self._send_replay_step()
        logger.info(f"Loaded replay: episode {episode_id}")

    async def _handle_list_recordings(self, websocket: WebSocket, data: dict):
        """Handle list_recordings command."""
        if not self.replay_manager:
            await websocket.send_json({"type": "error", "message": "Replay not available (no database)"})
            return

        # Extract filters
        filters = data.get("filters", {})
        stage = filters.get("stage")
        reason = filters.get("reason")
        min_reward = filters.get("min_reward")
        max_reward = filters.get("max_reward")
        limit = filters.get("limit", 100)

        # Query recordings
        recordings = self.replay_manager.list_recordings(
            stage=stage,
            reason=reason,
            min_reward=min_reward,
            max_reward=max_reward,
            limit=limit,
        )

        # Send response
        await websocket.send_json(
            {
                "type": "recordings_list",
                "recordings": recordings,
            }
        )

    async def _handle_replay_control(self, websocket: WebSocket, data: dict):
        """Handle replay_control command."""
        if not self.replay_manager or not self.replay_manager.is_loaded():
            await websocket.send_json({"type": "error", "message": "No replay loaded"})
            return

        action = data.get("action")

        if action == "play":
            self.replay_playing = True
            asyncio.create_task(self._run_replay_loop())

        elif action == "pause":
            self.replay_playing = False

        elif action == "step":
            asyncio.create_task(self._replay_single_step())

        elif action == "seek":
            seek_step = data.get("seek_step")
            if seek_step is not None:
                success = self.replay_manager.seek(seek_step)
                if success:
                    await self._send_replay_step()
                else:
                    await websocket.send_json({"type": "error", "message": f"Invalid seek step: {seek_step}"})

    async def _run_replay_loop(self):
        """Main replay loop - streams steps continuously."""
        logger.info("Replay loop started")

        while self.replay_playing and self.replay_manager and not self.replay_manager.is_at_end():
            await self._send_replay_step()
            self.replay_manager.next_step()
            await asyncio.sleep(self.step_delay)

        # End of replay
        if self.replay_manager and self.replay_manager.is_at_end():
            await self._broadcast_to_clients(
                {
                    "type": "replay_finished",
                    "episode_id": self.replay_manager.episode_id,
                }
            )
            self.replay_playing = False
            logger.info("Replay finished")

    async def _replay_single_step(self):
        """Send a single replay step."""
        if not self.replay_manager or not self.replay_manager.is_loaded():
            return

        await self._send_replay_step()
        self.replay_manager.next_step()

    async def _send_replay_step(self):
        """Send current replay step to all clients."""
        if not self.replay_manager or not self.replay_manager.is_loaded():
            return

        step_data = self.replay_manager.get_current_step()
        if step_data is None:
            return

        metadata = self.replay_manager.get_metadata()
        affordances = self.replay_manager.get_affordances()

        # Convert step data to state update format (matching live inference)
        position = step_data["position"]
        meters_tuple = step_data["meters"]
        q_values = step_data.get("q_values")

        # Build grid state
        grid_state = {
            "width": 8,
            "height": 8,
            "agents": [
                {
                    "id": "agent_0",
                    "x": position[0],
                    "y": position[1],
                    "color": "blue",
                    "last_action": step_data["action"],
                }
            ],
            "affordances": [{"type": name, "x": pos[0], "y": pos[1]} for name, pos in affordances.items()],
        }

        # Build meter state
        # Get actual meter names from config (supports variable meter counts)
        meter_names = self.env.bars_config.meter_names
        agent_meters = {"agent_0": {"meters": {name: val for name, val in zip(meter_names, meters_tuple)}}}

        # Build state update with substrate metadata
        state_update = {
            "type": "state_update",
            "mode": "replay",
            "episode_id": metadata["episode_id"],
            "step": step_data["step"],
            "cumulative_reward": step_data["reward"] * step_data["step"],  # Approximation
            "substrate": self._build_substrate_metadata(),
            "grid": grid_state,
            "agent_meters": agent_meters,
            "replay_metadata": {
                "total_steps": self.replay_manager.get_total_steps(),
                "current_step": self.replay_manager.get_current_step_index(),
                "survival_steps": metadata["survival_steps"],
                "total_reward": metadata["total_reward"],
                "curriculum_stage": metadata["curriculum_stage"],
            },
        }

        # Add Q-values if present
        if q_values:
            state_update["q_values"] = q_values

        # Add temporal mechanics if present
        if step_data.get("time_of_day") is not None:
            state_update["time_of_day"] = step_data["time_of_day"]
        if step_data.get("interaction_progress") is not None:
            state_update["interaction_progress"] = step_data["interaction_progress"]

        # Broadcast to all clients
        await self._broadcast_to_clients(state_update)


def run_server(
    checkpoint_dir: str = "checkpoints",
    port: int = 8766,
    step_delay: float = 0.2,
    total_episodes: int = 5000,
    config_dir: str | None = None,
    training_config_path: str | None = None,
    db_path: str | None = None,
    recordings_dir: str | None = None,
):
    """Run live inference server with optional replay support.

    Args:
        checkpoint_dir: Directory containing training checkpoints
        port: WebSocket port
        step_delay: Delay between steps in seconds
        total_episodes: Expected total training episodes
        config_dir: Config directory (compiled universe source)
        training_config_path: Optional training config YAML
        db_path: Optional database path for replay mode
        recordings_dir: Optional recordings directory for replay mode
    """
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    if config_dir is None:
        raise ValueError("config_dir is required for live inference. Provide the path to the config pack directory.")

    server = LiveInferenceServer(
        checkpoint_dir,
        port,
        step_delay,
        total_episodes,
        config_dir=config_dir,
        training_config_path=training_config_path,
        db_path=db_path,
        recordings_dir=recordings_dir,
    )

    logger.info(f"Starting live inference server on port {port}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Step delay: {step_delay}s ({1 / step_delay:.1f} steps/sec)")
    logger.info(f"Expected total training episodes: {total_episodes}")
    if config_dir:
        logger.info(f"Config directory: {config_dir}")
    if training_config_path:
        logger.info(f"Training config: {training_config_path}")
    logger.info(f"Connect Vue frontend to: ws://localhost:{port}/ws")

    uvicorn.run(server.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    import sys

    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8766
    step_delay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    total_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
    config_arg = sys.argv[5] if len(sys.argv) > 5 else None
    config_dir = None
    training_config = None
    if config_arg:
        candidate = Path(config_arg)
        if candidate.is_dir():
            config_dir = str(candidate)
            training_candidate = candidate / "training.yaml"
            training_config = str(training_candidate) if training_candidate.exists() else None
        else:
            config_dir = str(candidate.parent)
            training_config = str(candidate)

    run_server(checkpoint_dir, port, step_delay, total_episodes, config_dir, training_config)
