"""
Checkpoint management for training.

Handles saving and loading of agent checkpoints with automatic versioning.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional
import torch

from hamlet.agent.base_algorithm import BaseAlgorithm


class CheckpointManager:
    """
    Manages agent checkpoints with automatic versioning.

    Features:
    - Automatic checkpoint versioning
    - Keep best N checkpoints based on metric
    - Save/load multiple agents
    - Metadata tracking (episode, metrics, etc.)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        keep_best: bool = True,
        metric_name: str = "total_reward",
        metric_mode: str = "max",
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            keep_best: If True, keep best checkpoints; otherwise keep most recent
            metric_name: Metric to use for determining best checkpoints
            metric_mode: 'max' or 'min' for metric comparison
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.metric_name = metric_name
        self.metric_mode = metric_mode

        # Track checkpoint metadata
        self.checkpoints: List[Dict] = []

    def save_checkpoint(
        self,
        episode: int,
        agents: Dict[str, BaseAlgorithm],
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Save checkpoint for multiple agents.

        Args:
            episode: Current episode number
            agents: Dictionary of agent_id -> agent
            metrics: Episode metrics
            metadata: Additional metadata to save

        Returns:
            Path to saved checkpoint directory
        """
        # Create checkpoint directory
        checkpoint_name = f"checkpoint_ep{episode}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save each agent
        for agent_id, agent in agents.items():
            agent_file = checkpoint_path / f"{agent_id}.pt"
            agent.save(str(agent_file))

        # Save metadata
        checkpoint_metadata = {
            "episode": episode,
            "metrics": metrics or {},
            "agent_ids": list(agents.keys()),
            **(metadata or {}),
        }

        metadata_file = checkpoint_path / "metadata.pt"
        torch.save(checkpoint_metadata, metadata_file)

        # Track checkpoint
        self.checkpoints.append({
            "path": checkpoint_path,
            "episode": episode,
            "metrics": metrics or {},
        })

        # Clean up old checkpoints
        self._cleanup_checkpoints()

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        agents: Dict[str, BaseAlgorithm],
    ) -> Dict:
        """
        Load checkpoint for multiple agents.

        Args:
            checkpoint_path: Path to checkpoint directory
            agents: Dictionary of agent_id -> agent (must be pre-initialized)

        Returns:
            Checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

        # Load metadata
        metadata_file = checkpoint_path / "metadata.pt"
        metadata = torch.load(metadata_file)

        # Load each agent
        for agent_id in metadata["agent_ids"]:
            if agent_id in agents:
                agent_file = checkpoint_path / f"{agent_id}.pt"
                agents[agent_id].load(str(agent_file))

        return metadata

    def load_latest_checkpoint(
        self,
        agents: Dict[str, BaseAlgorithm],
    ) -> Optional[Dict]:
        """
        Load the most recent checkpoint.

        Args:
            agents: Dictionary of agent_id -> agent

        Returns:
            Checkpoint metadata or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()

        if not checkpoints:
            return None

        # Get latest checkpoint
        latest = max(checkpoints, key=lambda x: x["episode"])
        return self.load_checkpoint(latest["path"], agents)

    def load_best_checkpoint(
        self,
        agents: Dict[str, BaseAlgorithm],
    ) -> Optional[Dict]:
        """
        Load the best checkpoint based on metric.

        Args:
            agents: Dictionary of agent_id -> agent

        Returns:
            Checkpoint metadata or None if no checkpoints exist
        """
        if not self.checkpoints:
            return None

        # Find best checkpoint
        def get_metric_value(checkpoint):
            metrics = checkpoint["metrics"]
            if isinstance(metrics, dict):
                return metrics.get(self.metric_name, float("-inf") if self.metric_mode == "max" else float("inf"))
            return float(metrics)

        if self.metric_mode == "max":
            best = max(self.checkpoints, key=get_metric_value)
        else:
            best = min(self.checkpoints, key=get_metric_value)

        return self.load_checkpoint(best["path"], agents)

    def list_checkpoints(self) -> List[Dict]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint metadata dictionaries
        """
        # Scan checkpoint directory
        checkpoints = []

        for checkpoint_dir in sorted(self.checkpoint_dir.glob("checkpoint_ep*")):
            metadata_file = checkpoint_dir / "metadata.pt"

            if metadata_file.exists():
                metadata = torch.load(metadata_file)
                checkpoints.append({
                    "path": checkpoint_dir,
                    "episode": metadata["episode"],
                    "metrics": metadata.get("metrics", {}),
                })

        return checkpoints

    def delete_checkpoint(self, checkpoint_path: Path):
        """
        Delete a specific checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)

        # Remove from tracking
        self.checkpoints = [
            cp for cp in self.checkpoints if cp["path"] != checkpoint_path
        ]

    def _cleanup_checkpoints(self):
        """Remove old checkpoints based on policy."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        def get_metric_value(checkpoint):
            metrics = checkpoint["metrics"]
            if isinstance(metrics, dict):
                return metrics.get(self.metric_name, float("-inf") if self.metric_mode == "max" else float("inf"))
            return float(metrics)

        if self.keep_best:
            # Keep best N checkpoints
            if self.metric_mode == "max":
                sorted_checkpoints = sorted(
                    self.checkpoints,
                    key=get_metric_value,
                    reverse=True,
                )
            else:
                sorted_checkpoints = sorted(
                    self.checkpoints,
                    key=get_metric_value,
                )

            # Delete checkpoints beyond max_checkpoints
            for checkpoint in sorted_checkpoints[self.max_checkpoints:]:
                self.delete_checkpoint(checkpoint["path"])
        else:
            # Keep most recent N checkpoints
            sorted_checkpoints = sorted(
                self.checkpoints,
                key=lambda x: x["episode"],
                reverse=True,
            )

            # Delete old checkpoints
            for checkpoint in sorted_checkpoints[self.max_checkpoints:]:
                self.delete_checkpoint(checkpoint["path"])

    def get_checkpoint_info(self) -> Dict:
        """
        Get information about managed checkpoints.

        Returns:
            Dictionary with checkpoint statistics
        """
        if not self.checkpoints:
            return {
                "num_checkpoints": 0,
                "latest_episode": None,
                "best_metric_value": None,
            }

        latest = max(self.checkpoints, key=lambda x: x["episode"])

        def get_metric_value(checkpoint):
            metrics = checkpoint["metrics"]
            if isinstance(metrics, dict):
                return metrics.get(self.metric_name, float("-inf") if self.metric_mode == "max" else float("inf"))
            return float(metrics)

        if self.metric_mode == "max":
            best = max(self.checkpoints, key=get_metric_value)
        else:
            best = min(self.checkpoints, key=get_metric_value)

        best_value = get_metric_value(best)

        return {
            "num_checkpoints": len(self.checkpoints),
            "latest_episode": latest["episode"],
            "best_metric_value": best_value,
            "best_episode": best["episode"],
        }
