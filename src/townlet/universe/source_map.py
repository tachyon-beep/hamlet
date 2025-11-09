"""YAML source map helpers for compiler error reporting."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml


class _LineNumberLoader(yaml.SafeLoader):
    """PyYAML loader that annotates mappings with their starting line numbers."""


def _construct_mapping(loader: _LineNumberLoader, node: yaml.nodes.MappingNode, deep: bool = False):
    mapping = yaml.SafeLoader.construct_mapping(loader, node, deep)
    mapping["__line__"] = node.start_mark.line + 1
    return mapping


_LineNumberLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


class SourceMap:
    """Lightweight registry of config keys to file/line metadata."""

    def __init__(self) -> None:
        self._locations: dict[str, tuple[str, int | None]] = {}

    def record(self, key: str, file_path: Path, line: int | None) -> None:
        self._locations[key] = (str(file_path), line)

    def lookup(self, location: str) -> str | None:
        """Return a formatted `path:line` string for a location key if tracked."""

        parts = location.split(":")
        if len(parts) <= 1:
            return self._format(location)

        # Try progressively shorter prefixes (e.g., file:id:section -> file:id)
        for end in range(len(parts), 0, -1):
            candidate = ":".join(parts[:end])
            formatted = self._format(candidate)
            if formatted:
                return formatted
        return None

    def _format(self, key: str) -> str | None:
        if key not in self._locations:
            return None
        path, line = self._locations[key]
        if line is None:
            return path
        return f"{path}:{line}"

    def track_affordances(self, file_path: Path) -> None:
        self._track_named_sequence(file_path, list_key="affordances", identifier_key="id")

    def track_cascades(self, file_path: Path) -> None:
        self._track_named_sequence(file_path, list_key="cascades", identifier_key="name")

    def track_actions(self, file_path: Path) -> None:
        self._track_named_sequence(file_path, list_key="custom_actions", identifier_key="name")

    def _track_named_sequence(self, file_path: Path, *, list_key: str, identifier_key: str) -> None:
        if not file_path.exists():
            return
        data = self._load_yaml(file_path)
        if not isinstance(data, dict):
            return
        entries = data.get(list_key, [])
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            identifier = entry.get(identifier_key)
            if not identifier:
                continue
            line = entry.get("__line__")
            self.record(f"{file_path.name}:{identifier}", file_path, line)

    def track_training_environment_key(self, file_path: Path, key: str) -> None:
        line = self._find_line(file_path, key)
        self.record(f"{file_path.name}:{key}", file_path, line)

    def _find_line(self, file_path: Path, needle: str) -> int | None:
        if not file_path.exists():
            return None
        for line_num, line in enumerate(file_path.read_text().splitlines(), 1):
            if needle in line:
                return line_num
        return None

    def _load_yaml(self, file_path: Path):
        with open(file_path, encoding="utf-8") as handle:
            return yaml.load(handle, Loader=_LineNumberLoader)

    def bulk_record(self, entries: Iterable[tuple[str, Path, int | None]]) -> None:
        for key, path, line in entries:
            self.record(key, path, line)
