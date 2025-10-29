#!/usr/bin/env python3
"""Inspect logged failure events in Hamlet training metrics."""

import argparse
from pathlib import Path
from typing import List

from hamlet.training.config import MetricsConfig
from hamlet.training.metrics_manager import MetricsManager


def _print_table(headers: List[str], rows: List[List[str]]):
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    separator = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main():
    parser = argparse.ArgumentParser(
        description="Summarise or list failure events recorded during training",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("metrics.db"),
        help="Path to metrics SQLite database (default: metrics.db)",
    )
    parser.add_argument("--agent", type=str, help="Filter by agent id")
    parser.add_argument("--reason", type=str, help="Filter by failure reason")
    parser.add_argument("--min-episode", type=int, help="Minimum episode number")
    parser.add_argument("--max-episode", type=int, help="Maximum episode number")
    parser.add_argument("--limit", type=int, help="Limit number of rows returned")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show aggregated counts instead of raw events",
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Limit number of summary rows (overrides --limit when used)",
    )

    args = parser.parse_args()

    if not args.db.exists():
        print(f"No metrics database found at {args.db}.")
        return

    config = MetricsConfig(
        tensorboard=False,
        tensorboard_dir="/tmp/unused",
        database=True,
        database_path=str(args.db),
        replay_storage=False,
        live_broadcast=False,
    )

    manager = MetricsManager(config, experiment_name="analysis")

    try:
        if args.summary:
            top_n = args.top if args.top is not None else args.limit
            summary_rows = manager.get_failure_summary(
                agent_id=args.agent,
                reason=args.reason,
                min_episode=args.min_episode,
                max_episode=args.max_episode,
                top_n=top_n,
            )
            if not summary_rows:
                print("No failure events found for the selected filters.")
                return

            table_rows = [
                [
                    row["agent_id"],
                    row["reason"],
                    str(row["count"]),
                    str(row["last_episode"]),
                    str(row["last_timestamp"]),
                ]
                for row in summary_rows
            ]
            _print_table(
                ["Agent", "Reason", "Count", "Last Episode", "Last Timestamp"],
                table_rows,
            )
        else:
            events = manager.query_failure_events(
                agent_id=args.agent,
                reason=args.reason,
                min_episode=args.min_episode,
                max_episode=args.max_episode,
                limit=args.limit,
            )
            if not events:
                print("No failure events found for the selected filters.")
                return

            table_rows = [
                [
                    str(event["timestamp"]),
                    str(event["episode"]),
                    event["agent_id"],
                    event["reason"],
                ]
                for event in events
            ]
            _print_table(["Timestamp", "Episode", "Agent", "Reason"], table_rows)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
