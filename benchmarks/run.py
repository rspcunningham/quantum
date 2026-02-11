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


def run_case(
    case: BenchmarkCase,
    device: torch.device,
    verbose: bool,
    git_hash: str,
) -> dict:
    n_qubits = case.n_qubits
    display_qubits = n_qubits if n_qubits is not None else infer_resources(case.circuit)[0]

    ops = count_ops(case.circuit)
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
        print(f"\n{case.name} ({display_qubits} qubits, {total_ops} ops)")
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
    if results:
        total_parts: list[str] = []
        for s in SHOT_COUNTS:
            key = str(s)
            if all(key in r["times_s"] for r in results):
                total = sum(r["times_s"][key] for r in results)
                total_parts.append(f"{s} \u2192 {total:.2f}s")
            else:
                total_parts.append(f"{s} \u2192 n/a")
        shots_str = "    ".join(total_parts)
        print(f"\nTotal time by shot count:\n    {shots_str}")

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

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
