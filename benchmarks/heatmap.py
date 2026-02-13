"""Generate Native-vs-Aer timeout-parity scatter with full-case coverage."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SHOT_DEFAULTS = (1000, 10000)
LATEST_JSONL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{6}\.jsonl$")
QREG_RE = re.compile(r"qreg\s+\w+\[(\d+)\]")
SUFFIX_QUBIT_RE = re.compile(r".*_(\d+)$")
TIME_FLOOR = 1e-4
STATUS_ORDER = ("both_complete", "native_fail", "aer_fail", "both_fail")
STATUS_MARKERS = {
    "both_complete": "o",
    "native_fail": "^",
    "aer_fail": "s",
    "both_fail": "X",
}
STATUS_LABELS = {
    "both_complete": "both complete",
    "native_fail": "native fail",
    "aer_fail": "aer fail",
    "both_fail": "both fail",
}
BUCKET_ORDER = ("<=8", "9-16", "17-24", "25-30+", "unknown")
BUCKET_COLORS = {
    "<=8": "#1b9e77",
    "9-16": "#d95f02",
    "17-24": "#7570b3",
    "25-30+": "#e7298a",
    "unknown": "#666666",
}


@dataclass(frozen=True)
class CaseMeta:
    name: str
    n_qubits: int


@dataclass(frozen=True)
class CellPoint:
    case: str
    shot: int
    n_qubits: int
    bucket: str
    aer_time: float
    native_time: float
    aer_failed: bool
    native_failed: bool
    status: str


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row in {path} at line {line_no}: {exc}") from exc
    return rows


def _index_by_case(rows: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        case_name = row.get("case")
        if isinstance(case_name, str) and case_name:
            indexed[case_name] = row
    return indexed


def _latest_native_result(results_dir: Path) -> Path:
    candidates = [
        path for path in results_dir.glob("*.jsonl")
        if LATEST_JSONL_RE.match(path.name) and path.name != "aer-reference.jsonl"
    ]
    if not candidates:
        raise FileNotFoundError(f"No timestamped native JSONL files found in {results_dir}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def _load_expected_case_names(expected_dir: Path) -> set[str]:
    if not expected_dir.exists():
        return set()
    return {path.stem for path in expected_dir.glob("*.json")}


def _qubits_from_qasm(circuits_dir: Path, case_name: str) -> int | None:
    qasm_path = circuits_dir / f"{case_name}.qasm"
    if not qasm_path.exists():
        return None
    try:
        text = qasm_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = QREG_RE.search(text)
    if match is None:
        return None
    return int(match.group(1))


def _qubits_from_case_suffix(case_name: str) -> int | None:
    match = SUFFIX_QUBIT_RE.match(case_name)
    if match is None:
        return None
    return int(match.group(1))


def _resolve_n_qubits(case_name: str, native_row: dict | None, aer_row: dict | None, circuits_dir: Path) -> int:
    for row in (native_row, aer_row):
        if not isinstance(row, dict):
            continue
        raw = row.get("n_qubits")
        if isinstance(raw, int):
            return raw
    from_qasm = _qubits_from_qasm(circuits_dir, case_name)
    if from_qasm is not None:
        return from_qasm
    from_suffix = _qubits_from_case_suffix(case_name)
    if from_suffix is not None:
        return from_suffix
    return 10_000


def _runtime_for_shot(row: dict | None, shot: int, timeout_s: float) -> tuple[float, bool]:
    if row is None:
        return timeout_s, True

    aborted = bool(row.get("aborted", False))
    if aborted:
        return timeout_s, True

    times_s = row.get("times_s")
    if not isinstance(times_s, dict):
        return timeout_s, True

    raw = times_s.get(str(shot))
    if raw is None:
        return timeout_s, True
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return timeout_s, True
    if not math.isfinite(value):
        return timeout_s, True
    if value >= timeout_s:
        return timeout_s, True
    if value <= 0.0:
        return TIME_FLOOR, False
    return max(value, TIME_FLOOR), False


def _label_for_shot(shot: int, index: int) -> str:
    prefix = "cold" if index == 0 else "warm"
    if shot % 1000 == 0:
        return f"{prefix} @{shot // 1000}K"
    return f"{prefix} @{shot}"


def _bucket_for_qubits(n_qubits: int) -> str:
    if n_qubits <= 8:
        return "<=8"
    if n_qubits <= 16:
        return "9-16"
    if n_qubits <= 24:
        return "17-24"
    if n_qubits <= 30:
        return "25-30+"
    return "unknown"


def _status(native_failed: bool, aer_failed: bool) -> str:
    if native_failed and aer_failed:
        return "both_fail"
    if native_failed:
        return "native_fail"
    if aer_failed:
        return "aer_fail"
    return "both_complete"


def _resolve_case_order(
    native_by_case: dict[str, dict],
    aer_by_case: dict[str, dict],
    expected_cases: set[str],
    circuits_dir: Path,
) -> list[CaseMeta]:
    all_case_names = set(native_by_case) | set(aer_by_case) | expected_cases
    metas: list[CaseMeta] = []
    for case_name in all_case_names:
        n_qubits = _resolve_n_qubits(case_name, native_by_case.get(case_name), aer_by_case.get(case_name), circuits_dir)
        metas.append(CaseMeta(name=case_name, n_qubits=n_qubits))
    metas.sort(key=lambda item: (item.n_qubits, item.name))
    return metas


def build_timeout_parity_scatter(
    *,
    native_jsonl: Path,
    reference_jsonl: Path,
    output_path: Path,
    timeout_s: float,
    shots: tuple[int, ...],
    expected_dir: Path,
    circuits_dir: Path,
    dpi: int = 160,
) -> dict[str, float | int]:
    native_rows = _load_jsonl(native_jsonl)
    aer_rows = _load_jsonl(reference_jsonl)
    native_by_case = _index_by_case(native_rows)
    aer_by_case = _index_by_case(aer_rows)
    expected_cases = _load_expected_case_names(expected_dir)

    cases = _resolve_case_order(native_by_case, aer_by_case, expected_cases, circuits_dir)
    if not cases:
        raise ValueError("No cases found across native/reference/expected sources")

    points: list[CellPoint] = []
    bucket_stats = {
        bucket: {"total": 0, "both_complete": 0, "native_fail": 0, "aer_fail": 0, "both_fail": 0}
        for bucket in BUCKET_ORDER
    }
    status_counts = {status: 0 for status in STATUS_ORDER}

    for meta in cases:
        bucket = _bucket_for_qubits(meta.n_qubits)
        native_row = native_by_case.get(meta.name)
        aer_row = aer_by_case.get(meta.name)
        for shot in shots:
            native_time, native_failed = _runtime_for_shot(native_row, shot, timeout_s)
            aer_time, aer_failed = _runtime_for_shot(aer_row, shot, timeout_s)
            cell_status = _status(native_failed, aer_failed)

            points.append(
                CellPoint(
                    case=meta.name,
                    shot=shot,
                    n_qubits=meta.n_qubits,
                    bucket=bucket,
                    aer_time=aer_time,
                    native_time=native_time,
                    aer_failed=aer_failed,
                    native_failed=native_failed,
                    status=cell_status,
                )
            )

            bucket_stats[bucket]["total"] += 1
            status_counts[cell_status] += 1
            if cell_status == "both_complete":
                bucket_stats[bucket]["both_complete"] += 1
            if native_failed:
                bucket_stats[bucket]["native_fail"] += 1
            if aer_failed:
                bucket_stats[bucket]["aer_fail"] += 1
            if native_failed and aer_failed:
                bucket_stats[bucket]["both_fail"] += 1

    if not points:
        raise ValueError("No comparison cells produced")

    total_cells = len(points)
    both_complete = status_counts["both_complete"]
    native_fail_cells = status_counts["native_fail"] + status_counts["both_fail"]
    aer_fail_cells = status_counts["aer_fail"] + status_counts["both_fail"]
    completion_rate = (100.0 * both_complete / total_cells) if total_cells > 0 else 0.0
    both_complete_points = [p for p in points if p.status == "both_complete"]
    native_complete_median = float(np.median([p.native_time for p in both_complete_points])) if both_complete_points else float("nan")
    aer_complete_median = float(np.median([p.aer_time for p in both_complete_points])) if both_complete_points else float("nan")

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xscale("log")
    ax.set_yscale("log")

    for bucket in BUCKET_ORDER:
        color = BUCKET_COLORS[bucket]
        for status in STATUS_ORDER:
            marker = STATUS_MARKERS[status]
            for shot_idx, shot in enumerate(shots):
                subset = [p for p in points if p.bucket == bucket and p.status == status and p.shot == shot]
                if not subset:
                    continue
                x = [p.aer_time for p in subset]
                y = [p.native_time for p in subset]
                size = 52 if shot_idx == 0 else 32
                edgecolor = "#111111" if shot_idx == 0 else "#ffffff"
                linewidth = 0.55 if shot_idx == 0 else 0.45
                ax.scatter(
                    x,
                    y,
                    s=size,
                    c=color,
                    marker=marker,
                    alpha=0.78,
                    edgecolors=edgecolor,
                    linewidths=linewidth,
                )

    all_times = np.array([p.aer_time for p in points] + [p.native_time for p in points], dtype=np.float64)
    min_time = float(np.min(all_times)) if all_times.size else TIME_FLOOR
    lower = max(TIME_FLOOR, min_time * 0.8)
    upper = timeout_s * 1.05

    ax.plot([lower, timeout_s], [lower, timeout_s], linestyle="--", linewidth=1.25, color="#333333")
    ax.axvline(timeout_s, linestyle=":", linewidth=1.3, color="#b22222")
    ax.axhline(timeout_s, linestyle=":", linewidth=1.3, color="#b22222")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    shot_labels = ", ".join(_label_for_shot(shot, idx) for idx, shot in enumerate(shots))
    ax.set_title(
        "Native vs Aer timeout-parity scatter "
        f"| cells={total_cells} ({shot_labels}) | both-complete={completion_rate:.1f}%"
    )
    ax.set_xlabel(
        f"Aer runtime (seconds, log scale; abort/missing or >= {timeout_s:g}s plotted as {timeout_s:g}s fail)"
    )
    ax.set_ylabel(
        f"Native runtime (seconds, log scale; abort/missing or >= {timeout_s:g}s plotted as {timeout_s:g}s fail)"
    )
    ax.grid(which="major", alpha=0.2)

    bucket_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=BUCKET_COLORS[bucket],
            markeredgecolor="#111111",
            markersize=7,
            label=bucket,
        )
        for bucket in BUCKET_ORDER
    ]
    status_handles = [
        Line2D(
            [0],
            [0],
            marker=STATUS_MARKERS[status],
            color="#111111",
            markerfacecolor="#ffffff",
            markeredgecolor="#111111",
            markersize=8,
            linestyle="None",
            label=STATUS_LABELS[status],
        )
        for status in STATUS_ORDER
    ]
    shot_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#111111",
            markerfacecolor="#ffffff",
            markeredgecolor="#111111",
            markersize=(7.2 if idx == 0 else 5.8),
            linestyle="None",
            label=_label_for_shot(shot, idx),
        )
        for idx, shot in enumerate(shots)
    ]

    fig.subplots_adjust(right=0.77)
    legend_bucket = ax.legend(handles=bucket_handles, title="qubit bucket (color)", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(legend_bucket)
    legend_status = ax.legend(handles=status_handles, title="status (marker)", loc="upper left", bbox_to_anchor=(1.01, 0.68))
    ax.add_artist(legend_status)
    ax.legend(handles=shot_handles, title="shot row (size)", loc="upper left", bbox_to_anchor=(1.01, 0.40))

    bucket_completion_parts: list[str] = []
    for bucket in BUCKET_ORDER:
        total = bucket_stats[bucket]["total"]
        complete = bucket_stats[bucket]["both_complete"]
        pct = (100.0 * complete / total) if total > 0 else 0.0
        bucket_completion_parts.append(f"{bucket}: {pct:.0f}%")
    summary_lines = [
        f"fail rule: abort/missing or runtime >= {timeout_s:g}s => fail @ {timeout_s:g}s",
        f"native fail cells={native_fail_cells} | aer fail cells={aer_fail_cells} | both fail={status_counts['both_fail']}",
        f"median on both-complete cells: native={native_complete_median:.4f}s, aer={aer_complete_median:.4f}s",
        "bucket both-complete rate: " + ", ".join(bucket_completion_parts),
    ]
    ax.text(
        0.012,
        0.015,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=8.6,
        verticalalignment="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    return {
        "cases": len(cases),
        "cells": total_cells,
        "both_complete_cells": both_complete,
        "both_complete_pct": completion_rate,
        "native_fail_cells": native_fail_cells,
        "aer_fail_cells": aer_fail_cells,
        "both_fail_cells": status_counts["both_fail"],
        "native_complete_median_s": native_complete_median,
        "aer_complete_median_s": aer_complete_median,
    }


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_results = repo_root / "benchmarks" / "results"
    parser = argparse.ArgumentParser(
        description=(
            "Generate docs/native-vs-aer.png timeout-parity scatter. "
            "All cells are included; any abort/missing/runtime>=timeout is plotted as timeout-fail."
        )
    )
    parser.add_argument(
        "--native",
        type=Path,
        default=None,
        help="Native JSONL file (defaults to latest timestamped file in benchmarks/results)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=default_results / "aer-reference.jsonl",
        help="Pinned Aer reference JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "docs" / "native-vs-aer.png",
        help="Output comparison plot path",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout-censor value in seconds for aborted/missing cells",
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=list(SHOT_DEFAULTS),
        help="Shot rows to compare (default: 1000 10000)",
    )
    parser.add_argument(
        "--expected-dir",
        type=Path,
        default=repo_root / "benchmarks" / "expected",
        help="Expected distribution directory used to force full-case coverage",
    )
    parser.add_argument(
        "--circuits-dir",
        type=Path,
        default=repo_root / "benchmarks" / "circuits",
        help="QASM directory used for fallback qubit-count ordering",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG DPI",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results_dir = (Path(__file__).resolve().parent.parent / "benchmarks" / "results")
    native_jsonl = args.native if args.native is not None else _latest_native_result(results_dir)

    summary = build_timeout_parity_scatter(
        native_jsonl=native_jsonl,
        reference_jsonl=args.reference,
        output_path=args.output,
        timeout_s=float(args.timeout),
        shots=tuple(args.shots),
        expected_dir=args.expected_dir,
        circuits_dir=args.circuits_dir,
        dpi=int(args.dpi),
    )

    print(f"native:    {native_jsonl}")
    print(f"reference: {args.reference}")
    print(f"output:    {args.output}")
    print(
        "summary:   "
        f"cases={summary['cases']} cells={summary['cells']} "
        f"both_complete={summary['both_complete_cells']} ({summary['both_complete_pct']:.1f}%) "
        f"native_fail={summary['native_fail_cells']} aer_fail={summary['aer_fail_cells']} "
        f"both_fail={summary['both_fail_cells']} "
        f"median_complete(native/aer)="
        f"{summary['native_complete_median_s']:.4f}s/{summary['aer_complete_median_s']:.4f}s"
    )


if __name__ == "__main__":
    main()
