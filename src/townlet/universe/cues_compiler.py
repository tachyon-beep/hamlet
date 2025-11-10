"""Cue validation sub-compiler used by the UniverseCompiler."""

from __future__ import annotations

from collections.abc import Callable

from townlet.config.cues import CuesConfig

from .errors import CompilationErrorCollector
from .symbol_table import UniverseSymbolTable


Formatter = Callable[[str, str, str | None], object]


class CuesCompiler:
    """Encapsulates cues.yaml validation logic (Stage 4 helper)."""

    def validate(
        self,
        cues_config: CuesConfig,
        symbol_table: UniverseSymbolTable,
        errors: CompilationErrorCollector,
        formatter: Formatter,
    ) -> None:
        """Run all cue validations and record structured diagnostics."""

        self._validate_basic_cues(cues_config, symbol_table, errors, formatter)
        self._validate_visual_cues(cues_config, symbol_table, errors, formatter)

    def _validate_basic_cues(
        self,
        cues_config: CuesConfig,
        symbol_table: UniverseSymbolTable,
        errors: CompilationErrorCollector,
        formatter: Formatter,
    ) -> None:
        for cue in cues_config.simple_cues:
            meter = cue.condition.meter
            if meter not in symbol_table.meters:
                errors.add(
                    formatter(
                        "UAC-VAL-005",
                        f"Cue '{cue.cue_id}' references unknown meter '{meter}'",
                        f"cues.yaml:{cue.cue_id}",
                    )
                )
            threshold = cue.condition.threshold
            if threshold < 0.0 or threshold > 1.0:
                errors.add(
                    formatter(
                        "UAC-VAL-005",
                        f"Cue threshold must be within [0.0, 1.0], got {threshold}",
                        f"cues.yaml:{cue.cue_id}",
                    )
                )

        for cue in cues_config.compound_cues:
            for condition in cue.conditions:
                if condition.meter not in symbol_table.meters:
                    errors.add(
                        formatter(
                            "UAC-VAL-005",
                            f"Cue '{cue.cue_id}' references unknown meter '{condition.meter}'",
                            f"cues.yaml:{cue.cue_id}",
                        )
                    )
                if condition.threshold < 0.0 or condition.threshold > 1.0:
                    errors.add(
                        formatter(
                            "UAC-VAL-005",
                            f"Cue threshold must be within [0.0, 1.0], got {condition.threshold}",
                            f"cues.yaml:{cue.cue_id}",
                        )
                    )

    def _validate_visual_cues(
        self,
        cues_config: CuesConfig,
        symbol_table: UniverseSymbolTable,
        errors: CompilationErrorCollector,
        formatter: Formatter,
    ) -> None:
        if not cues_config.visual_cues:
            return

        for meter_name, cues in cues_config.visual_cues.items():
            if meter_name not in symbol_table.meters:
                errors.add(
                    formatter(
                        "UAC-VAL-009",
                        f"Visual cue block references unknown meter '{meter_name}'",
                        f"cues.yaml:{meter_name}",
                    )
                )
                continue

            ranges = [tuple(cue.range) for cue in cues]
            if not self._ranges_cover_domain(ranges, 0.0, 1.0):
                errors.add(
                    formatter(
                        "UAC-VAL-009",
                        f"Visual cue ranges for '{meter_name}' do not cover [0.0, 1.0] without gaps.",
                        f"cues.yaml:{meter_name}",
                    )
                )
            if self._ranges_overlap(ranges):
                errors.add(
                    formatter(
                        "UAC-VAL-009",
                        f"Visual cue ranges for '{meter_name}' overlap.",
                        f"cues.yaml:{meter_name}",
                    )
                )

    @staticmethod
    def _ranges_cover_domain(ranges: list[tuple[float, float]], domain_min: float, domain_max: float) -> bool:
        if not ranges:
            return False
        sorted_ranges = sorted(ranges, key=lambda r: r[0])
        eps = 1e-6
        if abs(sorted_ranges[0][0] - domain_min) > eps:
            return False
        current_end = sorted_ranges[0][1]
        for start, end in sorted_ranges[1:]:
            if abs(start - current_end) > eps:
                return False
            current_end = end
        if abs(current_end - domain_max) > eps:
            return False
        return True

    @staticmethod
    def _ranges_overlap(ranges: list[tuple[float, float]]) -> bool:
        if not ranges:
            return False
        sorted_ranges = sorted(ranges, key=lambda r: r[0])
        eps = 1e-6
        current_end = sorted_ranges[0][1]
        for start, end in sorted_ranges[1:]:
            if start < current_end - eps:
                return True
            current_end = max(current_end, end)
        return False

