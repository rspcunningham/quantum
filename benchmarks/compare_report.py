"""Generate a markdown summary from compare JSONL results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SHOT_KEYS = ["1", "10", "100", "1000", "10000"]


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    # Keep last occurrence for each (suite, backend, case) across inputs.
    latest_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source_idx, path in enumerate(paths):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                suite = str(row.get("suite", "unknown"))
                backend = str(row.get("backend", "unknown"))
                case = str(row.get("case", "unknown"))
                key = (suite, backend, case)
                row["_source_index"] = source_idx
                latest_by_key[key] = row
    return list(latest_by_key.values())


def _backend_totals(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("supported"):
            grouped[str(row["backend"])].append(row)

    totals: dict[str, dict[str, float | None]] = {}
    for backend, backend_rows in grouped.items():
        shot_totals: dict[str, float | None] = {}
        for shot in SHOT_KEYS:
            if all(shot in row.get("times_s", {}) for row in backend_rows):
                shot_totals[shot] = round(sum(float(row["times_s"][shot]) for row in backend_rows), 4)
            else:
                shot_totals[shot] = None
        totals[backend] = shot_totals
    return totals


def _suite_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("suite", "unknown"))].append(row)
    return grouped


def _unsupported_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unsupported = [row for row in rows if not row.get("supported", False)]
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in unsupported:
        reason = str(row.get("support_reason") or "unsupported")
        key = (str(row.get("backend")), str(row.get("case")), reason)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("supported", False) and row.get("correct") is False]


def _emit_suite_section(suite: str, rows: list[dict[str, Any]]) -> list[str]:
    totals = _backend_totals(rows)
    unsupported = _unsupported_rows(rows)
    failures = _failure_rows(rows)

    lines: list[str] = []
    lines.append(f"## Suite: `{suite}`")
    lines.append("")
    lines.append("### Backend Totals (Median Execution Sums)")
    lines.append("")
    lines.append("| Backend | 1 | 10 | 100 | 1000 | 10000 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for backend in sorted(totals):
        shot_totals = totals[backend]
        row_values = []
        for shot in SHOT_KEYS:
            value = shot_totals[shot]
            row_values.append("n/a" if value is None else f"{value:.4f}s")
        lines.append(f"| {backend} | " + " | ".join(row_values) + " |")
    if not totals:
        lines.append("| (none) | n/a | n/a | n/a | n/a | n/a |")
    lines.append("")

    lines.append("### Unsupported Cases")
    if unsupported:
        lines.append("")
        lines.append("| Backend | Case | Reason |")
        lines.append("|---|---|---|")
        for row in unsupported:
            reason = row.get("support_reason") or "unsupported"
            lines.append(f"| {row['backend']} | {row['case']} | {reason} |")
    else:
        lines.append("")
        lines.append("- none")
    lines.append("")

    lines.append("### Correctness Failures")
    if failures:
        lines.append("")
        lines.append("| Backend | Case | Errors |")
        lines.append("|---|---|---|")
        for row in failures:
            errors = "; ".join(row.get("errors", []))
            lines.append(f"| {row['backend']} | {row['case']} | {errors} |")
    else:
        lines.append("")
        lines.append("- none")
    lines.append("")

    return lines


def _emit_markdown(rows: list[dict[str, Any]], input_paths: list[Path]) -> str:
    suites = _suite_rows(rows)

    lines: list[str] = []
    lines.append("# Compare Report")
    lines.append("")
    lines.append("## Inputs")
    for path in input_paths:
        lines.append(f"- `{path}`")
    lines.append("")

    lines.append("## Notes")
    lines.append("- Rows are deduplicated by `(suite, backend, case)` with later input files taking precedence.")
    lines.append("")

    for suite in sorted(suites):
        lines.extend(_emit_suite_section(suite, suites[suite]))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown report from compare JSONL results")
    parser.add_argument("--input", nargs="+", required=True, help="One or more compare JSONL files")
    parser.add_argument("--output", default=None, help="Output markdown path")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    rows = _load_rows(input_paths)
    markdown = _emit_markdown(rows, input_paths)

    if args.output is None:
        base = input_paths[0]
        output_path = base.with_suffix(".md")
    else:
        output_path = Path(args.output)

    output_path.write_text(markdown)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
