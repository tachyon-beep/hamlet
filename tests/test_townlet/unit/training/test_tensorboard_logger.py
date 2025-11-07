"""Unit tests for TensorBoardLogger with mocked SummaryWriter.

These unit tests complement the integration tests by providing:
- Full mocking of SummaryWriter for faster execution
- Coverage of edge cases and branching logic
- Testing without filesystem dependencies
- Focus on achieving 100% code coverage

Current integration test coverage: 97%
Target with unit tests: 100%

Uncovered lines to target:
- Line 149->142: hasattr check for add_text method
- Lines 258-259: Optimizer learning rate logging
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.nn as nn


class TestTensorBoardLoggerInitialization:
    """Test logger initialization with mocked SummaryWriter."""

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_init_creates_log_directory(self, mock_writer_class):
        """Should create log directory if it doesn't exist."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "nested" / "logs"
            assert not log_dir.exists()

            logger = TensorBoardLogger(log_dir=log_dir)

            assert log_dir.exists()
            mock_writer_class.assert_called_once_with(str(log_dir))

            logger.close()

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_init_with_custom_params(self, mock_writer_class):
        """Should initialize with custom flush_every and logging flags."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(
                log_dir=tmpdir,
                flush_every=20,
                log_gradients=True,
                log_histograms=False,
            )

            assert logger.flush_every == 20
            assert logger.log_gradients is True
            assert logger.log_histograms is False
            assert logger.episodes_logged == 0
            assert logger.last_flush_episode == 0

            logger.close()


class TestLogEpisodeCoverage:
    """Test log_episode() method with all branches."""

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_log_episode_with_intrinsic_ratio_calculation(self, mock_writer_class):
        """Should calculate intrinsic ratio when total_reward != 0 (line 113-117)."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(log_dir=tmpdir)

            logger.log_episode(
                episode=100,
                survival_time=200,
                total_reward=100.0,  # Non-zero
                extrinsic_reward=60.0,
                intrinsic_reward=40.0,
            )

            # Should call add_scalar for intrinsic_ratio
            # Ratio = 40.0 / 100.0 = 0.4
            calls = [c for c in mock_writer.add_scalar.call_args_list if "Intrinsic_Ratio" in str(c)]
            assert len(calls) > 0

            logger.close()

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_log_episode_skips_intrinsic_ratio_when_zero(self, mock_writer_class):
        """Should skip intrinsic ratio calculation when total_reward == 0 (line 113)."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(log_dir=tmpdir)

            logger.log_episode(
                episode=100,
                survival_time=5,
                total_reward=0.0,  # Zero total reward
                extrinsic_reward=0.0,
                intrinsic_reward=0.0,
            )

            # Should NOT call add_scalar for intrinsic_ratio
            calls = [c for c in mock_writer.add_scalar.call_args_list if "Intrinsic_Ratio" in str(c)]
            assert len(calls) == 0

            logger.close()

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_log_episode_auto_flush_triggered(self, mock_writer_class):
        """Should trigger flush when episodes_logged % flush_every == 0 (line 120-122)."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(log_dir=tmpdir, flush_every=5)

            # Log 4 episodes - should not flush
            for ep in range(1, 5):
                logger.log_episode(episode=ep, survival_time=100, total_reward=50.0)

            assert mock_writer.flush.call_count == 0

            # Log 5th episode - should trigger flush
            logger.log_episode(episode=5, survival_time=100, total_reward=50.0)

            assert mock_writer.flush.call_count == 1
            assert logger.last_flush_episode == 5

            logger.close()


class TestCurriculumTransitionsHasattrBranch:
    """Test curriculum transitions with hasattr branch coverage (line 149)."""

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_log_curriculum_transitions_with_add_text_available(self, mock_writer_class):
        """Should log text summary when writer has add_text method (line 149-159)."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            # Mock writer HAS add_text method
            mock_writer.add_text = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(log_dir=tmpdir)

            events = [
                {
                    "agent_id": "agent_0",
                    "from_stage": 2,
                    "to_stage": 3,
                    "survival_rate": 0.75,
                    "learning_progress": 0.85,
                    "entropy": 0.6,
                    "reason": "advance",
                    "steps_at_stage": 1500,
                }
            ]

            logger.log_curriculum_transitions(episode=100, events=events)

            # Should call add_text (hasattr returned True)
            mock_writer.add_text.assert_called_once()
            call_args = mock_writer.add_text.call_args
            assert "agent_0/Curriculum/Transition" in call_args[0][0]
            assert "ADVANCE" in call_args[0][1]  # Reason uppercased

            logger.close()

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_log_curriculum_transitions_without_add_text(self, mock_writer_class):
        """Should skip text logging when writer lacks add_text method (line 149 branch)."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            # Mock writer does NOT have add_text method
            mock_writer.add_text = None
            del mock_writer.add_text  # Remove attribute entirely
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(log_dir=tmpdir)

            events = [
                {
                    "agent_id": "agent_0",
                    "from_stage": 2,
                    "to_stage": 3,
                    "survival_rate": 0.75,
                    "learning_progress": 0.85,
                    "entropy": 0.6,
                    "reason": "advance",
                    "steps_at_stage": 1500,
                }
            ]

            logger.log_curriculum_transitions(episode=100, events=events)

            # Should NOT call add_text (hasattr returned False)
            # Verify add_scalar WAS called for metrics
            assert mock_writer.add_scalar.call_count >= 4  # Stage, Survival_Rate, Learning_Progress, Entropy

            logger.close()


class TestNetworkStatsOptimizerBranch:
    """Test network stats logging with optimizer (line 257-259)."""

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_log_network_stats_with_optimizer_logs_learning_rate(self, mock_writer_class):
        """Should log learning rate when optimizer is provided (line 257-259)."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(
                log_dir=tmpdir,
                log_histograms=True,  # Enable to not early return
            )

            q_network = nn.Sequential(nn.Linear(10, 5))
            optimizer = torch.optim.Adam(q_network.parameters(), lr=0.00025)

            logger.log_network_stats(
                episode=100,
                q_network=q_network,
                optimizer=optimizer,
            )

            # Should log learning rate scalar
            calls = [c for c in mock_writer.add_scalar.call_args_list if "Learning_Rate" in str(c)]
            assert len(calls) == 1

            # Verify learning rate value
            lr_call = calls[0]
            assert lr_call[0][1] == 0.00025  # Learning rate value
            assert lr_call[0][2] == 100  # Episode number

            logger.close()

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_log_network_stats_without_optimizer_skips_learning_rate(self, mock_writer_class):
        """Should skip learning rate logging when optimizer is None (line 257)."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(
                log_dir=tmpdir,
                log_histograms=True,
            )

            q_network = nn.Sequential(nn.Linear(10, 5))

            logger.log_network_stats(
                episode=100,
                q_network=q_network,
                optimizer=None,  # No optimizer
            )

            # Should NOT log learning rate
            calls = [c for c in mock_writer.add_scalar.call_args_list if "Learning_Rate" in str(c)]
            assert len(calls) == 0

            logger.close()


class TestAllMethodsMocked:
    """Test all logging methods with fully mocked SummaryWriter for coverage."""

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_log_multi_agent_episode_mocked(self, mock_writer_class):
        """Test multi-agent logging with mocked writer."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(log_dir=tmpdir)

            agents_data = [
                {
                    "agent_id": "agent_0",
                    "survival_time": 100,
                    "total_reward": 50.0,
                },
                {
                    "agent_id": "agent_1",
                    "survival_time": 150,
                    "total_reward": 75.0,
                },
            ]

            logger.log_multi_agent_episode(episode=100, agents=agents_data)

            # Should log metrics for both agents
            assert logger.episodes_logged == 2

            logger.close()

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_log_training_step_all_params_mocked(self, mock_writer_class):
        """Test training step logging with all optional parameters."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(log_dir=tmpdir, log_histograms=True)

            q_values = torch.tensor([1.0, 2.0, 3.0])

            logger.log_training_step(
                step=1000,
                td_error=0.5,
                q_values=q_values,
                loss=0.3,
                rnd_prediction_error=0.1,
            )

            # Should log all scalars and histogram
            assert mock_writer.add_scalar.call_count >= 5  # TD_Error, Loss, RND_Error, Q_Mean, Q_Std
            mock_writer.add_histogram.assert_called_once()

            logger.close()

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_log_meters_mocked(self, mock_writer_class):
        """Test meter logging with mocked writer."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(log_dir=tmpdir)

            meters = {
                "health": 0.8,
                "energy": 0.6,
            }

            logger.log_meters(episode=10, step=50, meters=meters)

            # Should log 2 scalars
            # Global step = 10 * 1000 + 50 = 10050
            assert mock_writer.add_scalar.call_count == 2

            logger.close()

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_log_affordance_usage_mocked(self, mock_writer_class):
        """Test affordance usage logging with mocked writer."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(log_dir=tmpdir)

            affordance_counts = {
                "Bed": 10,
                "Shower": 5,
            }

            logger.log_affordance_usage(episode=100, affordance_counts=affordance_counts)

            assert mock_writer.add_scalar.call_count == 2

            logger.close()

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_log_custom_metric_mocked(self, mock_writer_class):
        """Test custom metric logging with mocked writer."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(log_dir=tmpdir)

            logger.log_custom_metric(
                tag="Debug/Test",
                value=123.45,
                step=500,
            )

            mock_writer.add_scalar.assert_called_once()

            logger.close()

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_log_hyperparameters_mocked(self, mock_writer_class):
        """Test hyperparameter logging with mocked writer."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(log_dir=tmpdir)

            hparams = {"learning_rate": 0.001}
            metrics = {"final_reward": 100.0}

            logger.log_hyperparameters(hparams, metrics)

            mock_writer.add_hparams.assert_called_once_with(hparams, metrics)

            logger.close()

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_flush_method_mocked(self, mock_writer_class):
        """Test manual flush() method."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(log_dir=tmpdir)

            logger.flush()

            mock_writer.flush.assert_called_once()

            logger.close()

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_close_method_mocked(self, mock_writer_class):
        """Test close() method calls flush and close on writer."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(log_dir=tmpdir)

            logger.close()

            # Should call both flush and close
            mock_writer.flush.assert_called()
            mock_writer.close.assert_called_once()

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_context_manager_mocked(self, mock_writer_class):
        """Test context manager with mocked writer."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            with TensorBoardLogger(log_dir=tmpdir) as logger:
                assert logger is not None

            # Should have called close
            mock_writer.flush.assert_called()
            mock_writer.close.assert_called_once()


class TestNetworkStatsGradientBranches:
    """Test network stats gradient logging branches."""

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_log_network_stats_with_gradients_present(self, mock_writer_class):
        """Test gradient logging when gradients exist."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(log_dir=tmpdir, log_gradients=True)

            q_network = nn.Sequential(nn.Linear(10, 5))

            # Create gradients
            dummy_input = torch.randn(4, 10)
            output = q_network(dummy_input)
            loss = output.sum()
            loss.backward()

            logger.log_network_stats(episode=100, q_network=q_network)

            # Should log gradient norm
            calls = [c for c in mock_writer.add_scalar.call_args_list if "Gradients/Total_Norm" in str(c)]
            assert len(calls) == 1

            logger.close()

    @patch("townlet.training.tensorboard_logger.SummaryWriter")
    def test_log_network_stats_early_return(self, mock_writer_class):
        """Test early return when both log_histograms and log_gradients are False."""
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(
                log_dir=tmpdir,
                log_histograms=False,
                log_gradients=False,
            )

            q_network = nn.Sequential(nn.Linear(10, 5))

            logger.log_network_stats(episode=100, q_network=q_network)

            # Should return early without logging anything
            assert mock_writer.add_scalar.call_count == 0
            assert mock_writer.add_histogram.call_count == 0

            logger.close()
