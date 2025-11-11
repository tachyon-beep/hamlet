"""Compilation error helpers for the Universe Compiler."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompilationMessage:
    """Structured compiler diagnostic with code and location metadata."""

    code: str | None
    message: str
    location: str | None = None

    def format(self) -> str:
        parts: list[str] = []
        if self.code:
            parts.append(f"[{self.code}]")
        if self.location:
            parts.append(self.location)
        prefix = " ".join(parts)
        if prefix:
            return f"{prefix} - {self.message}"
        return self.message


class CompilationError(Exception):
    """Raised when a compilation stage encounters validation failures."""

    def __init__(
        self,
        stage: str,
        errors: Iterable[CompilationMessage | str],
        hints: Iterable[str] | None = None,
        warnings: Iterable[str] | None = None,
    ):
        self.stage = stage
        self.issues: list[CompilationMessage] = [_coerce_issue(issue) for issue in errors]
        self.errors = [issue.format() for issue in self.issues]
        self.hints = list(hints or [])
        self.warnings = list(warnings or [])
        message_lines = [f"{stage} failed:"] + [f"  - {msg}" for msg in self.errors]
        if self.hints:
            message_lines.append("Hints:")
            message_lines.extend(f"  • {hint}" for hint in self.hints)
        if self.warnings:
            message_lines.append("Warnings:")
            message_lines.extend(f"  ! {warning}" for warning in self.warnings)
        super().__init__("\n".join(message_lines))


def _coerce_issue(issue: CompilationMessage | str) -> CompilationMessage:
    if isinstance(issue, CompilationMessage):
        return issue
    return CompilationMessage(code=None, message=str(issue), location=None)


@dataclass
class CompilationErrorCollector:
    """Collects compilation errors across stages before raising."""

    stage: str = ""
    _issues: list[CompilationMessage] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [issue.format() for issue in self._issues]

    @property
    def issues(self) -> list[CompilationMessage]:
        return list(self._issues)

    def add(self, message: CompilationMessage | str, *, code: str | None = None, location: str | None = None) -> None:
        if isinstance(message, CompilationMessage):
            self._issues.append(message)
        else:
            self._issues.append(CompilationMessage(code=code, message=message, location=location))

    # Alias used throughout docs/specs
    def add_error(self, message: CompilationMessage | str, *, code: str | None = None, location: str | None = None) -> None:
        self.add(message, code=code, location=location)

    def add_hint(self, hint: str) -> None:
        self.hints.append(hint)

    def add_warning(self, message: CompilationMessage | str) -> None:
        if isinstance(message, CompilationMessage):
            self.warnings.append(message.format())
        else:
            self.warnings.append(message)

    def extend(self, messages: Iterable[CompilationMessage | str]) -> None:
        for message in messages:
            self.add(message)

    def check_and_raise(self, stage_label: str | None = None) -> None:
        if self._issues:
            raise CompilationError(
                stage_label or self.stage or "Compiler",
                self._issues,
                self.hints,
                self.warnings,
            )
