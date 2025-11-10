"""Parametrized unit tests for TensorBoardLogger using mocked SummaryWriter."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn


@contextmanager
def logger_with_writer(temp_dir, *, auto_close: bool = True, **logger_kwargs):
    """Patch SummaryWriter and yield (logger, writer, writer_class_mock)."""

    from townlet.training.tensorboard_logger import TensorBoardLogger

    with patch("townlet.training.tensorboard_logger.SummaryWriter") as writer_cls:
        writer = Mock()
        writer_cls.return_value = writer
        log_dir = logger_kwargs.pop("log_dir", temp_dir)
        logger = TensorBoardLogger(log_dir=log_dir, **logger_kwargs)
        try:
            yield logger, writer, writer_cls
        finally:
            if auto_close:
                logger.close()


def test_logger_initialization_creates_directory(temp_test_dir):
    log_dir = temp_test_dir / "logs"
    assert not log_dir.exists()

    with logger_with_writer(log_dir) as (_, _, writer_cls):
        writer_cls.assert_called_once_with(str(log_dir))

    assert log_dir.exists()


def test_logger_initialization_respects_custom_flags(temp_test_dir):
    with logger_with_writer(
        temp_test_dir,
        flush_every=20,
        log_gradients=True,
        log_histograms=False,
    ) as (logger, _, _):
        assert logger.flush_every == 20
        assert logger.log_gradients is True
        assert logger.log_histograms is False
        assert logger.episodes_logged == 0
        assert logger.last_flush_episode == 0


@pytest.mark.parametrize(
    ("total_reward", "intrinsic_reward", "expected_calls"),
    [(100.0, 40.0, 1), (0.0, 0.0, 0)],
    ids=["with_ratio", "no_ratio"],
)
def test_log_episode_intrinsic_ratio(temp_test_dir, total_reward, intrinsic_reward, expected_calls):
    with logger_with_writer(temp_test_dir) as (logger, writer, _):
        logger.log_episode(
            episode=1,
            survival_time=10,
            total_reward=total_reward,
            intrinsic_reward=intrinsic_reward,
        )

    ratio_calls = [call for call in writer.add_scalar.call_args_list if "Intrinsic_Ratio" in call[0][0]]
    assert len(ratio_calls) == expected_calls


def test_log_episode_triggers_auto_flush(temp_test_dir):
    with logger_with_writer(temp_test_dir, auto_close=False, flush_every=2) as (logger, writer, _):
        logger.log_episode(episode=1, survival_time=5, total_reward=1.0)
        assert writer.flush.call_count == 0
        logger.log_episode(episode=2, survival_time=5, total_reward=1.0)
        writer.flush.assert_called_once()
        logger.close()


@pytest.mark.parametrize("has_add_text", [True, False], ids=["with_text", "without_text"])
def test_log_curriculum_transitions_handles_add_text(temp_test_dir, has_add_text):
    with logger_with_writer(temp_test_dir) as (logger, writer, _):
        if has_add_text:
            writer.add_text = Mock()
        elif hasattr(writer, "add_text"):
            del writer.add_text

        logger.log_curriculum_transitions(
            episode=42,
            events=[
                {
                    "agent_id": "agent_0",
                    "from_stage": 1,
                    "to_stage": 2,
                    "survival_rate": 0.8,
                    "learning_progress": 0.9,
                    "entropy": 0.5,
                    "reason": "advance",
                    "steps_at_stage": 100,
                }
            ],
        )

    if has_add_text:
        writer.add_text.assert_called_once()
    else:
        assert not any("Transition" in call[0][0] for call in writer.add_scalar.call_args_list if len(call[0]) > 0)


@pytest.mark.parametrize("has_optimizer", [True, False], ids=["with_optimizer", "without_optimizer"])
def test_log_network_stats_learning_rate_branch(temp_test_dir, has_optimizer):
    with logger_with_writer(temp_test_dir, log_histograms=True) as (logger, writer, _):
        q_network = nn.Sequential(nn.Linear(10, 5))
        optimizer = torch.optim.Adam(q_network.parameters(), lr=0.00025) if has_optimizer else None

        logger.log_network_stats(episode=7, q_network=q_network, optimizer=optimizer)

    lr_calls = [call for call in writer.add_scalar.call_args_list if "Learning_Rate" in call[0][0]]
    assert bool(lr_calls) is has_optimizer


def test_log_multi_agent_episode_counts_agents(temp_test_dir):
    with logger_with_writer(temp_test_dir) as (logger, _, _):
        logger.log_multi_agent_episode(
            episode=5,
            agents=[
                {"agent_id": "agent_0", "survival_time": 10, "total_reward": 5.0},
                {"agent_id": "agent_1", "survival_time": 20, "total_reward": 8.0},
            ],
        )
        assert logger.episodes_logged == 2


def test_log_training_step_logs_histograms(temp_test_dir):
    with logger_with_writer(temp_test_dir, log_histograms=True) as (logger, writer, _):
        q_values = torch.tensor([1.0, 2.0, 3.0])
        logger.log_training_step(step=1000, td_error=0.5, q_values=q_values, loss=0.3, rnd_prediction_error=0.1)
        assert writer.add_histogram.call_count == 1
        scalar_tags = {call[0][0] for call in writer.add_scalar.call_args_list}
        for suffix in ("Training/TD_Error", "Training/Loss", "Training/RND_Error", "Training/Q_Mean", "Training/Q_Std"):
            assert any(tag.endswith(suffix) for tag in scalar_tags)


def test_log_meters_emits_expected_scalars(temp_test_dir):
    with logger_with_writer(temp_test_dir) as (logger, writer, _):
        logger.log_meters(episode=10, step=5, meters={"health": 0.8, "energy": 0.6})
    assert len(writer.add_scalar.call_args_list) == 2


def test_log_affordance_usage_logs_each_entry(temp_test_dir):
    with logger_with_writer(temp_test_dir) as (logger, writer, _):
        logger.log_affordance_usage(episode=3, affordance_counts={"Bed": 4, "Shower": 2})
    assert len(writer.add_scalar.call_args_list) == 2


def test_log_custom_metric_delegates_to_writer(temp_test_dir):
    with logger_with_writer(temp_test_dir) as (logger, writer, _):
        logger.log_custom_metric(tag="Debug/Test", value=12.34, step=500)
    writer.add_scalar.assert_called_once_with("agent_0/Debug/Test", 12.34, 500)


def test_log_hyperparameters_calls_add_hparams(temp_test_dir):
    with logger_with_writer(temp_test_dir) as (logger, writer, _):
        logger.log_hyperparameters({"lr": 0.001}, {"final_reward": 10.0})
    writer.add_hparams.assert_called_once_with({"lr": 0.001}, {"final_reward": 10.0})


def test_flush_invokes_writer_flush(temp_test_dir):
    with logger_with_writer(temp_test_dir, auto_close=False) as (logger, writer, _):
        logger.flush()
        writer.flush.assert_called_once()
        logger.close()


def test_close_invokes_writer_flush_and_close(temp_test_dir):
    with logger_with_writer(temp_test_dir, auto_close=False) as (logger, writer, _):
        logger.close()
        assert writer.flush.call_count >= 1
        writer.close.assert_called_once()


def test_context_manager_closes_logger(temp_test_dir):
    with patch("townlet.training.tensorboard_logger.SummaryWriter") as writer_cls:
        writer = Mock()
        writer_cls.return_value = writer
        from townlet.training.tensorboard_logger import TensorBoardLogger

        with TensorBoardLogger(log_dir=temp_test_dir) as logger:
            assert logger is not None

        assert writer.flush.call_count >= 1
        writer.close.assert_called_once()


def test_log_network_stats_with_gradients(temp_test_dir):
    with logger_with_writer(temp_test_dir, log_gradients=True) as (logger, writer, _):
        q_network = nn.Sequential(nn.Linear(4, 2))
        data = torch.randn(3, 4)
        q_network(data).sum().backward()
        logger.log_network_stats(episode=9, q_network=q_network)

    grad_calls = [call for call in writer.add_scalar.call_args_list if "Gradients/Total_Norm" in call[0][0]]
    assert len(grad_calls) == 1


def test_log_network_stats_early_return_when_disabled(temp_test_dir):
    with logger_with_writer(temp_test_dir, log_histograms=False, log_gradients=False) as (logger, writer, _):
        q_network = nn.Sequential(nn.Linear(4, 2))
        logger.log_network_stats(episode=9, q_network=q_network)

    assert writer.add_scalar.call_count == 0
    assert writer.add_histogram.call_count == 0
