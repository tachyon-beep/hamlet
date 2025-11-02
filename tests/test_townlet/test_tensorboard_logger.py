"""
Tests for TensorBoardLogger utility helpers.
"""

from unittest.mock import patch

import torch

from townlet.training.tensorboard_logger import TensorBoardLogger


class _StubWriter:
    def __init__(self, *args, **kwargs):
        self.scalars = []
        self.histograms = []
        self.text = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_histogram(self, tag, values, step):
        self.histograms.append((tag, values, step))

    def add_text(self, tag, text, step):
        self.text.append((tag, text, step))

    def flush(self):
        pass


def test_log_multi_agent_episode_records_all_agents(tmp_path):
    """log_multi_agent_episode should emit per-agent scalar events."""
    with patch("townlet.training.tensorboard_logger.SummaryWriter", _StubWriter):
        logger = TensorBoardLogger(log_dir=tmp_path)

        data = [
            {
                "agent_id": "agent_0",
                "survival_time": 100,
                "total_reward": 42.0,
                "extrinsic_reward": 40.0,
            "intrinsic_reward": 2.0,
            "curriculum_stage": 3,
            "epsilon": 0.1,
            "intrinsic_weight": 0.5,
        },
        {
            "agent_id": "agent_1",
            "survival_time": 80,
            "total_reward": 30.0,
            "extrinsic_reward": 28.0,
            "intrinsic_reward": 2.0,
                "curriculum_stage": 2,
                "epsilon": 0.2,
                "intrinsic_weight": 0.4,
            },
        ]

        logger.log_multi_agent_episode(episode=10, agents=data)

        scalar_tags = {tag for (tag, _, _) in logger.writer.scalars}

        assert "agent_0/Episode/Survival_Time" in scalar_tags
        assert "agent_1/Episode/Survival_Time" in scalar_tags
        assert any(step == 10 for (_, _, step) in logger.writer.scalars)
