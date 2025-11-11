"""Helpers for checkpoint metadata and secure loading."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from townlet.universe.compiled import CompiledUniverse

_DIGEST_SUFFIX = ".sha256"
_DIGEST_BUFFER_SIZE = 1024 * 1024  # 1 MiB chunks keep memory bounded

logger = logging.getLogger(__name__)


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
        return "Checkpoint missing config_hash; transfer learning may be unstable. Retrain or regenerate checkpoint with Stage 7 compiler."

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
        raise ValueError(f"Checkpoint observation_dim mismatch: checkpoint={obs_dim}, current={universe.metadata.observation_dim}")

    action_dim = checkpoint.get("action_dim")
    if action_dim is not None and action_dim != universe.metadata.action_count:
        raise ValueError(f"Checkpoint action_dim mismatch: checkpoint={action_dim}, current={universe.metadata.action_count}")

    expected_uuids = [field.uuid for field in universe.observation_spec.fields]
    checkpoint_uuids = checkpoint.get("observation_field_uuids")
    if checkpoint_uuids is None:
        raise ValueError("Checkpoint missing observation_field_uuids; regenerate the checkpoint with the latest compiler.")
    if list(checkpoint_uuids) != expected_uuids:
        raise ValueError("Checkpoint observation field UUIDs mismatch current universe specification.")


def _digest_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(checkpoint_path.suffix + _DIGEST_SUFFIX)


def _compute_sha256(checkpoint_path: Path) -> str:
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_DIGEST_BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def persist_checkpoint_digest(checkpoint_path: Path) -> str:
    """Compute and store the SHA256 digest alongside the checkpoint file."""

    digest = _compute_sha256(checkpoint_path)
    digest_path = _digest_path(checkpoint_path)
    digest_path.write_text(digest + "\n", encoding="utf-8")
    return digest


def verify_checkpoint_digest(checkpoint_path: Path, *, required: bool = False) -> bool:
    """Verify the checkpoint SHA256 digest, optionally requiring the digest file."""

    digest_path = _digest_path(checkpoint_path)
    if not digest_path.exists():
        if required:
            raise FileNotFoundError(
                f"Missing checksum file for {checkpoint_path}. Expected {digest_path}. Recreate the checkpoint on Townlet >=P2."
            )
        logger.warning("Missing checksum for checkpoint %s (expected %s); skipping verification.", checkpoint_path, digest_path)
        return False

    expected = digest_path.read_text(encoding="utf-8").strip()
    actual = _compute_sha256(checkpoint_path)
    if actual != expected:
        raise ValueError(
            f"Checkpoint digest mismatch for {checkpoint_path}. Expected {expected} but computed {actual}. "
            "The file may be corrupted or tampered."
        )
    return True


def safe_torch_load(
    checkpoint_path: Path | str,
    *,
    map_location: torch.device | str | None = None,
    weights_only: bool = True,
) -> Any:
    """Load a checkpoint with optional PyTorch weights-only safety guard.

    Args:
        checkpoint_path: Path to checkpoint file
        map_location: Device to map tensors to
        weights_only: If True, only load tensors/parameters (safer but may fail with numpy types).
                     If False, allow arbitrary Python objects (use only for trusted checkpoints).

    Note:
        PyTorch 2.6+ requires explicit allowlisting of numpy types when weights_only=True.
        For locally-generated test checkpoints, weights_only=False is acceptable.
        For external/untrusted checkpoints, always use weights_only=True.
    """
    if not weights_only:
        # Trusted checkpoint (e.g., test-generated) - allow arbitrary objects
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    try:
        # PyTorch 2.6+ requires explicit allowlisting of numpy types
        # Add numpy types to safe globals for PyTorch 2.6+ compatibility

        import numpy as np

        safe_globals: list[Any] = [np.dtype]

        try:
            from numpy._core.multiarray import scalar as np_scalar_new

            safe_globals.append(np_scalar_new)
        except (ImportError, AttributeError):
            # Older numpy versions use numpy.core.multiarray instead
            try:
                from numpy.core.multiarray import scalar as np_scalar_old  # type: ignore[attr-defined]

                safe_globals.append(np_scalar_old)
            except (ImportError, AttributeError):
                pass  # If neither import works, proceed without it

        torch.serialization.add_safe_globals(safe_globals)

        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except RuntimeError as exc:  # pragma: no cover - depends on torch internals
        message = str(exc)
        if "weights_only=True" in message:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} contains custom Python objects and cannot be loaded with weights_only=True. "
                "Re-export the checkpoint using Townlet >= P2 to enable secure loading."
            ) from exc
        raise
