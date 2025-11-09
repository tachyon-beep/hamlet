"""Helpers for embedding CompiledUniverse metadata into checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from townlet.universe.compiled import CompiledUniverse


def attach_universe_metadata(checkpoint: dict[str, Any], universe: CompiledUniverse) -> None:
    """Add config hash and dimension metadata to a checkpoint payload."""

    checkpoint["config_hash"] = universe.metadata.config_hash
    checkpoint["observation_dim"] = universe.metadata.observation_dim
    checkpoint["action_dim"] = universe.metadata.action_count
    checkpoint["meter_count"] = universe.metadata.meter_count
    checkpoint["observation_field_uuids"] = [field.uuid for field in universe.observation_spec.fields]


def config_hash_warning(checkpoint: Mapping[str, Any], universe: CompiledUniverse) -> str | None:
    """Return warning message if checkpoint hash mismatches current universe."""

    checkpoint_hash = checkpoint.get("config_hash")
    if checkpoint_hash is None:
        return (
            "Checkpoint missing config_hash; transfer learning may be unstable." " Retrain or regenerate checkpoint with Stage 7 compiler."
        )

    if checkpoint_hash != universe.metadata.config_hash:
        return (
            "Checkpoint config hash mismatch; transfer learning may diverge.\n"
            f"  Checkpoint: {checkpoint_hash[:16]}...\n"
            f"  Current:    {universe.metadata.config_hash[:16]}..."
        )
    return None


def assert_checkpoint_dimensions(checkpoint: Mapping[str, Any], universe: CompiledUniverse) -> None:
    """Raise ValueError when checkpoint observation/action dims mismatch universe."""

    obs_dim = checkpoint.get("observation_dim")
    if obs_dim is not None and obs_dim != universe.metadata.observation_dim:
        raise ValueError("Checkpoint observation_dim mismatch:" f" checkpoint={obs_dim}, current={universe.metadata.observation_dim}")

    action_dim = checkpoint.get("action_dim")
    if action_dim is not None and action_dim != universe.metadata.action_count:
        raise ValueError("Checkpoint action_dim mismatch:" f" checkpoint={action_dim}, current={universe.metadata.action_count}")

    expected_uuids = [field.uuid for field in universe.observation_spec.fields]
    checkpoint_uuids = checkpoint.get("observation_field_uuids")
    if checkpoint_uuids is None:
        raise ValueError("Checkpoint missing observation_field_uuids; regenerate the checkpoint with the latest compiler.")
    if list(checkpoint_uuids) != expected_uuids:
        raise ValueError("Checkpoint observation field UUIDs mismatch current universe specification.")
