"""Compilation error helpers for the Universe Compiler."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


class CompilationError(Exception):
    """Raised when a compilation stage encounters validation failures."""

    def __init__(self, stage: str, errors: Iterable[str], hints: Iterable[str] | None = None):
        self.stage = stage
        self.errors = list(errors)
        self.hints = list(hints or [])
        message_lines = [f"{stage} failed:"] + [f"  - {msg}" for msg in self.errors]
        if self.hints:
            message_lines.append("Hints:")
            message_lines.extend(f"  • {hint}" for hint in self.hints)
        super().__init__("\n".join(message_lines))


@dataclass
class CompilationErrorCollector:
    """Collects compilation errors across stages before raising."""

    stage: str = ""
    errors: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.errors.append(message)

    def add_hint(self, hint: str) -> None:
        self.hints.append(hint)

    def extend(self, messages: Iterable[str]) -> None:
        self.errors.extend(messages)

    def check_and_raise(self, stage_label: str | None = None) -> None:
        if self.errors:
            raise CompilationError(stage_label or self.stage or "Compiler", self.errors, self.hints)
