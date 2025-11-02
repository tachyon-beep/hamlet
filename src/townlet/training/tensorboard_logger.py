"""TensorBoard logging for Hamlet training metrics.

Provides structured logging of training metrics to TensorBoard with
support for scalar metrics, histograms, and custom visualizations.

Integration:
    - Designed to work alongside existing DemoDatabase
    - Minimal overhead (~1ms per log call)
    - Automatic flush every N episodes
    - Safe for multi-agent scenarios
"""

from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """Logs Hamlet training metrics to TensorBoard.

    Tracks per-episode and per-step metrics for:
    - Survival time and rewards
    - Learning progress (Q-values, TD error, losses)
    - Exploration metrics (epsilon, intrinsic weight, RND errors)
    - Curriculum progression
    - Meter dynamics (agent health, resource management)
    - Network gradients and weights

    Example:
        >>> logger = TensorBoardLogger(log_dir="runs/experiment_1")
        >>> logger.log_episode(
        ...     episode=100,
        ...     survival_time=250,
        ...     total_reward=42.5,
        ...     curriculum_stage=3
        ... )
        >>> logger.log_training_step(
        ...     step=1000,
        ...     td_error=0.5,
        ...     q_values=torch.tensor([1.0, 2.0, 3.0])
        ... )
        >>> logger.close()
    """

    def __init__(
        self,
        log_dir: Path | str,
        flush_every: int = 10,
        log_gradients: bool = False,
        log_histograms: bool = True,
    ):
        """Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs (e.g., "runs/exp_name")
            flush_every: Flush to disk every N episodes (default: 10)
            log_gradients: Log gradient norms (can be expensive)
            log_histograms: Log weight/activation distributions
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.log_dir))
        self.flush_every = flush_every
        self.log_gradients = log_gradients
        self.log_histograms = log_histograms

        # Track last flush
        self.last_flush_episode = 0
        self.episodes_logged = 0

    def log_episode(
        self,
        episode: int,
        survival_time: int,
        total_reward: float,
        extrinsic_reward: float = 0.0,
        intrinsic_reward: float = 0.0,
        curriculum_stage: int = 1,
        epsilon: float = 0.0,
        intrinsic_weight: float = 0.0,
        agent_id: str = "agent_0",
    ):
        """Log per-episode metrics.

        Args:
            episode: Episode number (x-axis)
            survival_time: Steps survived
            total_reward: Combined reward
            extrinsic_reward: Environment reward
            intrinsic_reward: RND novelty reward
            curriculum_stage: Current difficulty (1-5)
            epsilon: Exploration rate
            intrinsic_weight: Intrinsic motivation weight
            agent_id: Agent identifier for multi-agent scenarios
        """
        prefix = f"{agent_id}/" if agent_id else ""

        # Core metrics
        self.writer.add_scalar(f"{prefix}Episode/Survival_Time", survival_time, episode)
        self.writer.add_scalar(f"{prefix}Episode/Total_Reward", total_reward, episode)
        self.writer.add_scalar(f"{prefix}Episode/Extrinsic_Reward", extrinsic_reward, episode)
        self.writer.add_scalar(f"{prefix}Episode/Intrinsic_Reward", intrinsic_reward, episode)

        # Learning progress indicators
        self.writer.add_scalar(f"{prefix}Curriculum/Stage", curriculum_stage, episode)
        self.writer.add_scalar(f"{prefix}Exploration/Epsilon", epsilon, episode)
        self.writer.add_scalar(f"{prefix}Exploration/Intrinsic_Weight", intrinsic_weight, episode)

        # Derived metrics
        if total_reward != 0:
            intrinsic_ratio = intrinsic_reward / total_reward if total_reward != 0 else 0
            self.writer.add_scalar(f"{prefix}Episode/Intrinsic_Ratio", intrinsic_ratio, episode)

        self.episodes_logged += 1

        # Auto-flush
        if self.episodes_logged % self.flush_every == 0:
            self.writer.flush()
            self.last_flush_episode = episode

    def log_multi_agent_episode(self, episode: int, agents: list[dict[str, Any]]) -> None:
        """Log per-episode metrics for multiple agents."""
        for data in agents:
            agent_identifier = str(data.get("agent_id", ""))
            self.log_episode(
                episode=episode,
                survival_time=int(data.get("survival_time", 0)),
                total_reward=float(data.get("total_reward", 0.0)),
                extrinsic_reward=float(data.get("extrinsic_reward", 0.0)),
                intrinsic_reward=float(data.get("intrinsic_reward", 0.0)),
                curriculum_stage=int(data.get("curriculum_stage", 1)),
                epsilon=float(data.get("epsilon", 0.0)),
                intrinsic_weight=float(data.get("intrinsic_weight", 0.0)),
                agent_id=agent_identifier,
            )

    def log_curriculum_transitions(self, episode: int, events: list[dict[str, Any]]) -> None:
        """Log curriculum transition rationale events."""
        for event in events:
            agent_id = str(event.get("agent_id", ""))
            prefix = f"{agent_id}/Curriculum/" if agent_id else "Curriculum/"
            self.writer.add_scalar(f"{prefix}Stage", event.get("to_stage", 0), episode)
            self.writer.add_scalar(f"{prefix}Survival_Rate", event.get("survival_rate", 0.0), episode)
            self.writer.add_scalar(f"{prefix}Learning_Progress", event.get("learning_progress", 0.0), episode)
            self.writer.add_scalar(f"{prefix}Entropy", event.get("entropy", 0.0), episode)
            if hasattr(self.writer, "add_text"):
                reason = str(event.get("reason", "unknown"))
                summary = (
                    f"{reason.upper()} "
                    f"{event.get('from_stage')}→{event.get('to_stage')} | "
                    f"survival={event.get('survival_rate', 0.0):.3f}, "
                    f"learning={event.get('learning_progress', 0.0):.3f}, "
                    f"entropy={event.get('entropy', 0.0):.3f}, "
                    f"steps_at_stage={event.get('steps_at_stage', 0)}"
                )
                self.writer.add_text(f"{prefix}Transition", summary, episode)

    def log_training_step(
        self,
        step: int,
        td_error: float | None = None,
        q_values: torch.Tensor | None = None,
        loss: float | None = None,
        rnd_prediction_error: float | None = None,
        agent_id: str = "agent_0",
    ):
        """Log per-training-step metrics.

        Args:
            step: Global training step
            td_error: Temporal difference error
            q_values: Q-value tensor for current state
            loss: Training loss (MSE for DQN)
            rnd_prediction_error: RND novelty detection error
            agent_id: Agent identifier
        """
        prefix = f"{agent_id}/" if agent_id else ""

        if td_error is not None:
            self.writer.add_scalar(f"{prefix}Training/TD_Error", td_error, step)

        if loss is not None:
            self.writer.add_scalar(f"{prefix}Training/Loss", loss, step)

        if rnd_prediction_error is not None:
            self.writer.add_scalar(f"{prefix}Training/RND_Error", rnd_prediction_error, step)

        if q_values is not None and self.log_histograms:
            self.writer.add_histogram(f"{prefix}Training/Q_Values", q_values, step)
            self.writer.add_scalar(f"{prefix}Training/Q_Mean", q_values.mean().item(), step)
            self.writer.add_scalar(f"{prefix}Training/Q_Std", q_values.std().item(), step)

    def log_meters(
        self,
        episode: int,
        step: int,
        meters: dict[str, float],
        agent_id: str = "agent_0",
    ):
        """Log agent meter values (health, energy, etc.).

        Args:
            episode: Current episode
            step: Step within episode
            meters: Dict of meter_name -> value (0.0-1.0)
            agent_id: Agent identifier
        """
        prefix = f"{agent_id}/" if agent_id else ""

        # Use episode*1000 + step as x-axis for fine-grained tracking
        global_step = episode * 1000 + step

        for meter_name, value in meters.items():
            self.writer.add_scalar(f"{prefix}Meters/{meter_name.capitalize()}", value, global_step)

    def log_network_stats(
        self,
        episode: int,
        q_network: torch.nn.Module,
        target_network: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        agent_id: str = "agent_0",
    ):
        """Log network weights, gradients, and optimizer state.

        Args:
            episode: Current episode
            q_network: Main Q-network
            target_network: Target network (for DQN)
            optimizer: Optimizer instance
            agent_id: Agent identifier
        """
        if not self.log_histograms and not self.log_gradients:
            return

        prefix = f"{agent_id}/" if agent_id else ""

        # Log weight distributions
        if self.log_histograms:
            for name, param in q_network.named_parameters():
                clean_name = name.replace(".", "/")
                self.writer.add_histogram(f"{prefix}Weights/{clean_name}", param.data, episode)

        # Log gradient norms
        if self.log_gradients:
            total_norm = 0.0
            for param in q_network.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm**0.5
            self.writer.add_scalar(f"{prefix}Gradients/Total_Norm", total_norm, episode)

        # Log learning rate
        if optimizer is not None:
            lr = optimizer.param_groups[0]["lr"]
            self.writer.add_scalar(f"{prefix}Training/Learning_Rate", lr, episode)

    def log_affordance_usage(
        self,
        episode: int,
        affordance_counts: dict[str, int],
        agent_id: str = "agent_0",
    ):
        """Log affordance usage statistics.

        Args:
            episode: Current episode
            affordance_counts: Dict of affordance_name -> visit_count
            agent_id: Agent identifier
        """
        prefix = f"{agent_id}/" if agent_id else ""

        for affordance, count in affordance_counts.items():
            self.writer.add_scalar(f"{prefix}Affordances/{affordance}", count, episode)

    def log_custom_metric(
        self,
        tag: str,
        value: float,
        step: int,
        agent_id: str = "agent_0",
    ):
        """Log custom metric (catch-all for experiments).

        Args:
            tag: Metric name (e.g., "Debug/StateEntropy")
            value: Metric value
            step: X-axis value (episode or global step)
            agent_id: Agent identifier
        """
        prefix = f"{agent_id}/" if agent_id else ""
        self.writer.add_scalar(f"{prefix}{tag}", value, step)

    def log_hyperparameters(self, hparams: dict[str, Any], metrics: dict[str, float]):
        """Log hyperparameters and final metrics for comparison.

        Args:
            hparams: Hyperparameter dict (e.g., learning_rate, gamma)
            metrics: Final metric dict (e.g., final_reward, final_survival)
        """
        self.writer.add_hparams(hparams, metrics)

    def flush(self):
        """Force flush all pending writes to disk."""
        self.writer.flush()

    def close(self):
        """Close writer and flush all pending writes."""
        self.writer.flush()
        self.writer.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (auto-close)."""
        self.close()
