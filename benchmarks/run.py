"""Benchmark runner for the quantum simulator."""

import argparse
import json
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

SHOT_COUNTS = [1, 10, 100, 1000]


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


def get_memory_mb(device: torch.device) -> float | None:
    """Get current GPU memory allocation in MB."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024 / 1024
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1024 / 1024
    return None


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
    peak_memory_mb: float | None = None

    for shots in SHOT_COUNTS:
        sync_device(device)
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        result = run_simulation(case.circuit, shots, n_qubits=n_qubits, device=device)
        sync_device(device)
        wall_elapsed = time.perf_counter() - wall_start
        cpu_elapsed = time.process_time() - cpu_start

        times[str(shots)] = round(wall_elapsed, 4)
        cpu_times[str(shots)] = round(cpu_elapsed, 4)

        if shots == max(SHOT_COUNTS):
            result_max = result
            peak_memory_mb = get_memory_mb(device)

    assert result_max is not None
    correct, errors = check_correctness(result_max, case.expected, case.tolerance)

    if verbose:
        shots_str = "    ".join(
            f"{s} \u2192 {times[str(s)]:.3f}s (cpu: {cpu_times[str(s)]:.3f}s)"
            for s in SHOT_COUNTS
        )
        mem_str = f"{peak_memory_mb:.1f} MB" if peak_memory_mb is not None else "n/a"
        print(f"\n{case.name} ({display_qubits} qubits, {total_ops} ops)")
        print(f"  shots:  {shots_str}")
        print(f"  memory: {mem_str}")
        status = "PASS" if correct else f"FAIL \u2014 {'; '.join(errors)}"
        print(f"  correctness: {status}")

    return {
        "case": case.name,
        "n_qubits": display_qubits,
        "ops": ops,
        "times_s": times,
        "cpu_times_s": cpu_times,
        "peak_memory_mb": round(peak_memory_mb, 2) if peak_memory_mb is not None else None,
        "correct": correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum simulator benchmark")
    parser.add_argument("-v", "--verbose", action="store_true", help="Per-case timing and correctness details")
    args = parser.parse_args()

    device = get_device()
    git_hash = get_git_hash()
    print(f"Device: {device.type} | Git: {git_hash}")

    results: list[dict] = []
    failures: list[str] = []

    for case_fn in ALL_CASES:
        case = case_fn()
        try:
            result = run_case(case, device, args.verbose)
        except Exception as e:
            print(f"\nERROR in {case.name}: {e}")
            failures.append(case.name)
            continue
        results.append(result)
        if not result["correct"]:
            failures.append(result["case"])

    # Summary
    if results:
        totals = {s: sum(r["times_s"][str(s)] for r in results) for s in SHOT_COUNTS}
        shots_str = "    ".join(f"{s} \u2192 {totals[s]:.2f}s" for s in SHOT_COUNTS)
        print(f"\nTotal time by shot count:\n    {shots_str}")

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
