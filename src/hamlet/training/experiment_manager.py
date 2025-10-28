"""
Experiment management with MLflow integration.

Tracks experiments, runs, parameters, and metrics for comparison.
"""

from typing import Dict, Any, Optional
from pathlib import Path

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from hamlet.training.config import ExperimentConfig


class ExperimentManager:
    """
    Manages experiments with MLflow.

    Provides unified interface for experiment tracking, parameter logging,
    and model artifact storage.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment manager.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.active_run = None

        if not MLFLOW_AVAILABLE:
            print("Warning: MLflow not available. Experiment tracking disabled.")
            return

        # Set tracking URI (local directory or remote server)
        mlflow.set_tracking_uri(config.tracking_uri)

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(config.name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    config.name,
                    artifact_location=None,  # Use default
                    tags={"description": config.description} if config.description else None,
                )
                self.experiment_id = experiment_id
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as e:
            print(f"Warning: Failed to setup MLflow experiment: {e}")
            self.experiment_id = None

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run.

        Args:
            run_name: Name for this run (auto-generated if None)
            tags: Additional tags for the run
        """
        if not MLFLOW_AVAILABLE or self.experiment_id is None:
            return

        try:
            # Use config run_name if provided, otherwise use parameter
            final_run_name = run_name or self.config.run_name

            self.active_run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=final_run_name,
                tags=tags,
            )
        except Exception as e:
            print(f"Warning: Failed to start MLflow run: {e}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters for this run.

        Args:
            params: Dictionary of parameter name -> value
        """
        if not MLFLOW_AVAILABLE or self.active_run is None:
            return

        try:
            # MLflow requires string values for params
            for key, value in params.items():
                mlflow.log_param(key, value)
        except Exception as e:
            print(f"Warning: Failed to log params: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric value.

        Args:
            key: Metric name
            value: Metric value
            step: Step number (optional)
        """
        if not MLFLOW_AVAILABLE or self.active_run is None:
            return

        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            print(f"Warning: Failed to log metric {key}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name -> value
            step: Step number (optional)
        """
        if not MLFLOW_AVAILABLE or self.active_run is None:
            return

        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"Warning: Failed to log metrics: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a file or directory as an artifact.

        Args:
            local_path: Path to local file or directory
            artifact_path: Relative path in artifact store (optional)
        """
        if not MLFLOW_AVAILABLE or self.active_run is None:
            return

        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            print(f"Warning: Failed to log artifact: {e}")

    def log_model(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ):
        """
        Log a PyTorch model.

        Args:
            model: PyTorch model to log
            artifact_path: Path within run's artifact directory
            registered_model_name: Name for model registry (optional)
        """
        if not MLFLOW_AVAILABLE or self.active_run is None:
            return

        try:
            mlflow.pytorch.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
            )
        except Exception as e:
            print(f"Warning: Failed to log model: {e}")

    def set_tag(self, key: str, value: str):
        """
        Set a tag for the current run.

        Args:
            key: Tag key
            value: Tag value
        """
        if not MLFLOW_AVAILABLE or self.active_run is None:
            return

        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            print(f"Warning: Failed to set tag: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """
        Set multiple tags for the current run.

        Args:
            tags: Dictionary of tag key -> value
        """
        if not MLFLOW_AVAILABLE or self.active_run is None:
            return

        try:
            mlflow.set_tags(tags)
        except Exception as e:
            print(f"Warning: Failed to set tags: {e}")

    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if not MLFLOW_AVAILABLE or self.active_run is None:
            return

        try:
            mlflow.end_run(status=status)
            self.active_run = None
        except Exception as e:
            print(f"Warning: Failed to end run: {e}")

    def get_run_id(self) -> Optional[str]:
        """
        Get current run ID.

        Returns:
            Run ID string or None if no active run
        """
        if self.active_run:
            return self.active_run.info.run_id
        return None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.end_run(status="FAILED")
        else:
            self.end_run(status="FINISHED")
