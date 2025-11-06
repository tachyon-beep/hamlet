"""Utility helpers for affordance layout abstractions.

These functions provide a lightweight adapter layer between the existing
dictionary-based affordance storage (`{name: position_tensor}`) and future
affordance systems (e.g., an affordance compiler) that may expose richer
interfaces. Substrates and observation builders call into these helpers to
iterate over affordance positions without depending on the concrete storage
type.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class AffordancePositionProvider(Protocol):
    """Protocol for objects that can yield discrete affordance positions."""

    def iter_positions(self) -> Iterable[torch.Tensor] | Iterable[tuple[str, torch.Tensor]]:
        """Return an iterable of affordance positions.

        Implementations may yield bare position tensors or ``(name, position)``
        tuples. Only the positional component is required for observation
        encoding.
        """


def iter_affordance_positions(affordances: object) -> Iterator[torch.Tensor]:
    """Iterate over affordance position tensors from heterogeneous sources.

    Args:
        affordances: Mapping, list-like container, or an object implementing
            :class:`AffordancePositionProvider`.

    Yields:
        Torch tensors representing discrete affordance positions. Empty tensors
        (e.g., aspatial affordances) are ignored.

    Raises:
        TypeError: If ``affordances`` does not provide positions in a supported
            format.
    """

    if affordances is None:
        return

    items: Any
    if isinstance(affordances, Mapping):
        items = affordances.values()
    elif isinstance(affordances, AffordancePositionProvider):
        items = affordances.iter_positions()
    elif isinstance(affordances, Iterable):
        items = affordances
    else:
        raise TypeError("Unsupported affordance container. Provide a mapping, iterable, or " "AffordancePositionProvider implementation.")

    for item in items:
        position = item
        if isinstance(item, tuple):
            # Allow providers to yield (name, position) pairs.
            if len(item) != 2:
                raise ValueError("Affordance tuples must be (name, position)")
            position = item[1]

        tensor = torch.as_tensor(position)
        if tensor.numel() == 0:
            continue  # Skip aspatial affordances

        yield tensor
