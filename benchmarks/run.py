"""Benchmark runner for the quantum simulator."""

import argparse
import gc
import json
import resource
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import torch

from benchmarks.cases import BenchmarkCase, ALL_CASES
from quantum import run_simulation, infer_resources
from quantum.gates import Gate, Measurement, ConditionalGate, Circuit

SHOT_COUNTS = [1, 10, 100, 1000, 10000]
OP_KINDS = ("gates", "measurements", "conditional")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def clear_device_cache(device: torch.device) -> None:
    """Best-effort cache cleanup after OOM to keep benchmark process alive."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def is_oom_error(error: BaseException) -> bool:
    msg = str(error).lower()
    return "out of memory" in msg or "oom" in msg


def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def count_ops(circuit: Circuit) -> dict[str, int]:
    """Count operations in a circuit by type."""
    counts = {"gates": 0, "measurements": 0, "conditional": 0}

    def _walk(ops: Sequence[Gate | Measurement | ConditionalGate | Circuit]) -> None:
        for op in ops:
            if isinstance(op, Circuit):
                _walk(op.operations)
            elif isinstance(op, Gate):
                counts["gates"] += 1
            elif isinstance(op, Measurement):
                counts["measurements"] += 1
            else:
                counts["conditional"] += 1

    _walk(circuit.operations)
    return counts


def analyze_workload(circuit: Circuit) -> dict[str, bool | str]:
    """Classify circuit as static or dynamic based on control-flow structure."""
    flattened: list[Gate | Measurement | ConditionalGate] = []

    def _flatten(ops: Sequence[Gate | Measurement | ConditionalGate | Circuit]) -> None:
        for op in ops:
            if isinstance(op, Circuit):
                _flatten(op.operations)
            else:
                flattened.append(op)

    _flatten(circuit.operations)

    has_conditional = False
    has_non_terminal_measurement = False
    seen_measurement = False

    for op in flattened:
        if isinstance(op, ConditionalGate):
            has_conditional = True
            continue
        if isinstance(op, Measurement):
            seen_measurement = True
            continue
        if seen_measurement:
            has_non_terminal_measurement = True

    workload_class = "dynamic" if (has_conditional or has_non_terminal_measurement) else "static"
    return {
        "has_conditional": has_conditional,
        "has_non_terminal_measurement": has_non_terminal_measurement,
        "workload_class": workload_class,
    }


def get_memory_stats(device: torch.device) -> dict[str, float]:
    """Get GPU memory stats in MB. Returns empty dict for CPU."""
    stats: dict[str, float] = {}
    if device.type == "cuda":
        stats["allocated_mb"] = torch.cuda.memory_allocated(device) / 1024 / 1024
        stats["max_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        stats["reserved_mb"] = torch.cuda.memory_reserved(device) / 1024 / 1024
    elif device.type == "mps":
        stats["allocated_mb"] = torch.mps.current_allocated_memory() / 1024 / 1024
        stats["driver_mb"] = torch.mps.driver_allocated_memory() / 1024 / 1024
    return stats


def check_correctness(
    result: dict[str, int],
    expected: dict[str, float],
    tolerance: float,
) -> tuple[bool, list[str]]:
    total = sum(result.values())
    observed = {k: v / total for k, v in result.items()}

    errors: list[str] = []
    for outcome, expected_freq in expected.items():
        observed_freq = observed.get(outcome, 0.0)
        if abs(observed_freq - expected_freq) > tolerance:
            errors.append(f'"{outcome}": expected {expected_freq:.2f}, got {observed_freq:.2f}')

    for outcome, freq in observed.items():
        if outcome not in expected and freq > tolerance:
            errors.append(f'"{outcome}": unexpected, got {freq:.2f}')

    return (len(errors) == 0, errors)


def _totals_for_rows(results: list[dict]) -> dict[str, float | None]:
    totals: dict[str, float | None] = {}
    for shot in SHOT_COUNTS:
        key = str(shot)
        if results and all(key in row["times_s"] for row in results):
            totals[key] = round(sum(float(row["times_s"][key]) for row in results), 4)
        else:
            totals[key] = None
    return totals


def _format_totals_line(totals: dict[str, float | None]) -> str:
    parts: list[str] = []
    for shot in SHOT_COUNTS:
        key = str(shot)
        value = totals.get(key)
        if value is None:
            parts.append(f"{shot} \u2192 n/a")
        else:
            parts.append(f"{shot} \u2192 {value:.2f}s")
    return "    ".join(parts)


def _category_hotspot_summary(
    rows: list[dict],
    *,
    shot_key: str,
    top_n: int = 5,
) -> dict[str, list[dict[str, float | int | str]]]:
    rows_with_shot = [row for row in rows if shot_key in row["times_s"]]
    if not rows_with_shot:
        return {"cases": [], "operators_proxy": []}

    total_time = sum(float(row["times_s"][shot_key]) for row in rows_with_shot)
    case_entries: list[dict[str, float | int | str]] = []
    for row in sorted(rows_with_shot, key=lambda r: float(r["times_s"][shot_key]), reverse=True)[:top_n]:
        case_time = float(row["times_s"][shot_key])
        case_entries.append({
            "case": str(row["case"]),
            "time_s": round(case_time, 4),
            "share_pct": round((100.0 * case_time / total_time) if total_time > 0 else 0.0, 2),
            "gates": int(row["ops"]["gates"]),
            "measurements": int(row["ops"]["measurements"]),
            "conditional": int(row["ops"]["conditional"]),
        })

    op_proxy_entries: list[dict[str, float | int | str]] = []
    scored: list[tuple[float, str, dict]] = []
    for row in rows_with_shot:
        row_time = float(row["times_s"][shot_key])
        total_ops = int(row["ops"]["gates"] + row["ops"]["measurements"] + row["ops"]["conditional"])
        if total_ops <= 0:
            continue
        for op_kind in OP_KINDS:
            op_count = int(row["ops"][op_kind])
            if op_count <= 0:
                continue
            score = row_time * (op_count / total_ops)
            scored.append((score, op_kind, row))

    for score, op_kind, row in sorted(scored, key=lambda item: item[0], reverse=True)[:top_n]:
        op_proxy_entries.append({
            "op_kind": op_kind,
            "case": str(row["case"]),
            "proxy_time_s": round(float(score), 4),
            "case_time_s": round(float(row["times_s"][shot_key]), 4),
            "op_count": int(row["ops"][op_kind]),
            "case_total_ops": int(row["ops"]["gates"] + row["ops"]["measurements"] + row["ops"]["conditional"]),
        })

    return {"cases": case_entries, "operators_proxy": op_proxy_entries}


def _print_hotspot_block(
    rows: list[dict],
    *,
    label: str,
    shot_key: str,
    top_n: int = 5,
) -> None:
    summary = _category_hotspot_summary(rows, shot_key=shot_key, top_n=top_n)
    if not summary["cases"]:
        print(f"\nTop hotspots ({label}, {shot_key} shots): n/a")
        return

    print(f"\nTop case hotspots ({label}, {shot_key} shots):")
    for idx, entry in enumerate(summary["cases"], start=1):
        print(
            f"  {idx}. {entry['case']}: {entry['time_s']:.4f}s "
            f"({entry['share_pct']:.2f}%) | ops g/m/c="
            f"{entry['gates']}/{entry['measurements']}/{entry['conditional']}"
        )

    print(f"Top operator hotspots ({label}, {shot_key} shots, proxy):")
    for idx, entry in enumerate(summary["operators_proxy"], start=1):
        print(
            f"  {idx}. {entry['op_kind']} in {entry['case']}: "
            f"proxy {entry['proxy_time_s']:.4f}s (case {entry['case_time_s']:.4f}s, "
            f"count {entry['op_count']}/{entry['case_total_ops']})"
        )


def run_case(
    case: BenchmarkCase,
    device: torch.device,
    verbose: bool,
    git_hash: str,
) -> dict:
    n_qubits = case.n_qubits
    display_qubits = n_qubits if n_qubits is not None else infer_resources(case.circuit)[0]

    ops = count_ops(case.circuit)
    workload = analyze_workload(case.circuit)
    total_ops = ops["gates"] + ops["measurements"] + ops["conditional"]

    # Warmup
    run_simulation(case.circuit, 1, n_qubits=n_qubits, device=device)
    sync_device(device)

    times: dict[str, float] = {}
    cpu_times: dict[str, float] = {}
    result_max: dict[str, int] | None = None
    memory_stats: dict[str, float] = {}
    max_shots = max(SHOT_COUNTS)
    oom_shots: int | None = None
    oom_error: str | None = None

    for shots in SHOT_COUNTS:
        try:
            sync_device(device)
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            result = run_simulation(case.circuit, shots, n_qubits=n_qubits, device=device)
            sync_device(device)
            wall_elapsed = time.perf_counter() - wall_start
            cpu_elapsed = time.process_time() - cpu_start
        except RuntimeError as error:
            if is_oom_error(error):
                oom_shots = shots
                oom_error = str(error).strip()
                clear_device_cache(device)
                break
            raise

        times[str(shots)] = round(wall_elapsed, 4)
        cpu_times[str(shots)] = round(cpu_elapsed, 4)

        if shots == max_shots:
            result_max = result
            memory_stats = get_memory_stats(device)

    completed_shots = [s for s in SHOT_COUNTS if str(s) in times]
    if not completed_shots:
        raise RuntimeError(f"{case.name}: no shot count completed successfully.")
    metric_shots = max(completed_shots)

    if oom_shots is None:
        assert result_max is not None
        correct, errors = check_correctness(result_max, case.expected, case.tolerance)
    else:
        if not memory_stats:
            memory_stats = get_memory_stats(device)
        correct = False
        errors = [f"OOM at {oom_shots} shots"]
        if oom_error:
            errors.append(oom_error)

    # Derived metrics at highest configured shot count
    wall_max = times[str(metric_shots)]
    cpu_max = cpu_times[str(metric_shots)]
    ops_per_sec = round(total_ops / wall_max, 1) if wall_max > 0 else 0
    shots_per_sec = round(metric_shots / wall_max, 1) if wall_max > 0 else 0
    cpu_util = round(cpu_max / wall_max, 3) if wall_max > 0 else 0

    if verbose:
        shots_str = "    ".join(
            (
                f"{s} \u2192 {times[str(s)]:.3f}s (cpu: {cpu_times[str(s)]:.3f}s)"
                if str(s) in times else
                f"{s} \u2192 OOM"
            )
            for s in SHOT_COUNTS
        )
        mem_parts = [f"{k}: {v:.1f}" for k, v in memory_stats.items()]
        mem_str = ", ".join(mem_parts) if mem_parts else "n/a"
        print(f"\n{case.name} ({display_qubits} qubits, {total_ops} ops, {workload['workload_class']})")
        print(f"  shots:    {shots_str}")
        print(f"  ops/s:    {ops_per_sec}  |  shots/s: {shots_per_sec}  |  cpu util: {cpu_util} (at {metric_shots} shots)")
        print(f"  memory:   {mem_str}")
        if oom_shots is not None:
            status = f"OOM \u2014 {'; '.join(errors)}"
        else:
            status = "PASS" if correct else f"FAIL \u2014 {'; '.join(errors)}"
        print(f"  correct:  {status}")

    return {
        "case": case.name,
        "git_hash": git_hash,
        "device": device.type,
        "n_qubits": display_qubits,
        "workload_class": workload["workload_class"],
        "has_conditional": workload["has_conditional"],
        "has_non_terminal_measurement": workload["has_non_terminal_measurement"],
        "ops": ops,
        "times_s": times,
        "cpu_times_s": cpu_times,
        "shot_counts": SHOT_COUNTS,
        "metric_shots": metric_shots,
        "ops_per_sec": ops_per_sec,
        "shots_per_sec": shots_per_sec,
        "cpu_util": cpu_util,
        "memory": {k: round(v, 2) for k, v in memory_stats.items()},
        "correct": correct,
        "oom": oom_shots is not None,
        "oom_shots": oom_shots,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum simulator benchmark")
    parser.add_argument("-v", "--verbose", action="store_true", help="Per-case timing and correctness details")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Run only the selected benchmark case names.",
    )
    args = parser.parse_args()

    all_case_map = {case_fn().name: case_fn for case_fn in ALL_CASES}
    if len(all_case_map) != len(ALL_CASES):
        raise RuntimeError("Duplicate benchmark case names detected.")

    if args.cases is not None:
        unknown_cases = [name for name in args.cases if name not in all_case_map]
        if unknown_cases:
            parser.error(f"Unknown case(s): {', '.join(unknown_cases)}")
        cases_to_run = [all_case_map[name] for name in args.cases]
    else:
        cases_to_run = ALL_CASES

    device = get_device()
    git_hash = get_git_hash()
    print(f"Device: {device.type} | Git: {git_hash} | Shots: {SHOT_COUNTS}")

    results: list[dict] = []
    failures: list[str] = []

    for case_fn in cases_to_run:
        case = case_fn()
        try:
            result = run_case(case, device, args.verbose, git_hash)
        except Exception as e:
            print(f"\nERROR in {case.name}: {e}")
            failures.append(case.name)
            continue
        results.append(result)
        if not result["correct"]:
            failures.append(result["case"])

    # Summary
    summary_payload: dict[str, object] = {}
    if results:
        static_rows = [row for row in results if row["workload_class"] == "static"]
        dynamic_rows = [row for row in results if row["workload_class"] == "dynamic"]

        totals_all = _totals_for_rows(results)
        totals_static = _totals_for_rows(static_rows)
        totals_dynamic = _totals_for_rows(dynamic_rows)

        print(f"\nTotal time by shot count (all):\n    {_format_totals_line(totals_all)}")
        print(f"Total time by shot count (static):\n    {_format_totals_line(totals_static)}")
        print(f"Total time by shot count (dynamic):\n    {_format_totals_line(totals_dynamic)}")

        hotspot_shot = str(max(SHOT_COUNTS))
        _print_hotspot_block(static_rows, label="static", shot_key=hotspot_shot, top_n=5)
        _print_hotspot_block(dynamic_rows, label="dynamic", shot_key=hotspot_shot, top_n=5)

        summary_payload = {
            "shot_counts": SHOT_COUNTS,
            "totals_s": {
                "all": totals_all,
                "static": totals_static,
                "dynamic": totals_dynamic,
            },
            "hotspot_shot": hotspot_shot,
            "hotspots": {
                "static": _category_hotspot_summary(static_rows, shot_key=hotspot_shot, top_n=5),
                "dynamic": _category_hotspot_summary(dynamic_rows, shot_key=hotspot_shot, top_n=5),
            },
            "counts": {
                "cases_total": len(results),
                "cases_static": len(static_rows),
                "cases_dynamic": len(dynamic_rows),
            },
        }

        # Process-level stats
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS reports ru_maxrss in bytes, Linux in KB
        peak_rss_mb = rusage.ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else rusage.ru_maxrss / 1024
        print(f"Peak RSS: {peak_rss_mb:.0f} MB")

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")

    # Write JSONL
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    output_path = results_dir / f"{timestamp}.jsonl"

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {output_path}")

    if summary_payload:
        summary_path = output_path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
            f.write("\n")
        print(f"Wrote {summary_path}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
