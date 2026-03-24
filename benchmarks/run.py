"""Benchmark runner for the quantum simulator."""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from benchmarks.cases import BenchmarkCase, ALL_CASES, CORE_CASES
from benchmarks.ir import build_circuit_ir
from quantum import run_simulation, infer_resources
from quantum.gates import Gate, Measurement, ConditionalGate, Circuit

SHOTS = 10_000


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
    counts = {"gates": 0, "measurements": 0, "conditional": 0}

    def _walk(ops) -> None:
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


def classify_workload(circuit: Circuit) -> str:
    flattened = []

    def _flatten(ops) -> None:
        for op in ops:
            if isinstance(op, Circuit):
                _flatten(op.operations)
            else:
                flattened.append(op)

    _flatten(circuit.operations)

    seen_measurement = False
    for op in flattened:
        if isinstance(op, ConditionalGate):
            return "dynamic"
        if isinstance(op, Measurement):
            seen_measurement = True
        elif seen_measurement:
            return "dynamic"
    return "static"


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


def get_peak_rss_mb() -> float:
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return rusage.ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else rusage.ru_maxrss / 1024


def estimate_memory_gb(n_qubits: int) -> float:
    """Estimate GPU memory needed: 4 Metal buffers (ping-pong re/im) + sampling."""
    return 5 * (2 ** n_qubits) * 4 / (1024 ** 3)


def get_memory_limit_gb() -> float:
    """Use half of physical RAM as the limit."""
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return total / (1024 ** 3) / 2


def run_case(case: BenchmarkCase, git_hash: str, *, backend: str, timeout: float, memory_limit_gb: float) -> dict:
    n_qubits = case.n_qubits
    if n_qubits is None:
        n_qubits = infer_resources(case.circuit)[0]
    ops = count_ops(case.circuit)
    workload = classify_workload(case.circuit)
    total_ops = ops["gates"] + ops["measurements"] + ops["conditional"]

    abort_reason = None
    result = None
    wall_elapsed = 0.0
    cpu_elapsed = 0.0

    estimated_gb = estimate_memory_gb(n_qubits)
    if estimated_gb > memory_limit_gb:
        abort_reason = f"skipped: {estimated_gb:.1f} GB estimated > {memory_limit_gb:.1f} GB limit ({n_qubits} qubits)"
    elif backend == "native":
        try:
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            result = run_simulation(case.circuit, SHOTS, n_qubits=case.n_qubits)
            wall_elapsed = time.perf_counter() - wall_start
            cpu_elapsed = time.process_time() - cpu_start
        except Exception as error:
            abort_reason = str(error).strip()
    else:
        from benchmarks.backends.aer_adapter import AerAdapter
        aer = AerAdapter()
        avail = aer.availability()
        if not avail.available:
            abort_reason = f"aer unavailable: {avail.reason}"
        else:
            case_ir = build_circuit_ir(case.circuit, n_qubits=n_qubits)
            supported, reason = aer.supports(case_ir)
            if not supported:
                abort_reason = f"aer unsupported: {reason}"
            else:
                try:
                    prepared = aer.prepare(case_ir)
                    wall_start = time.perf_counter()
                    cpu_start = time.process_time()
                    result = aer.run(prepared, SHOTS)
                    wall_elapsed = time.perf_counter() - wall_start
                    cpu_elapsed = time.process_time() - cpu_start
                except Exception as error:
                    abort_reason = str(error).strip()

    if abort_reason is not None:
        correct = False
        errors = [abort_reason]
    elif result is not None:
        correct, errors = check_correctness(result, case.expected, case.tolerance)
    else:
        correct = False
        errors = ["no result"]

    ops_per_sec = round(total_ops / wall_elapsed, 1) if wall_elapsed > 0 else 0.0
    shots_per_sec = round(SHOTS / wall_elapsed, 1) if wall_elapsed > 0 else 0.0

    return {
        "case": case.name,
        "backend": backend,
        "git_hash": git_hash,
        "n_qubits": n_qubits,
        "workload": workload,
        "ops": ops,
        "shots": SHOTS,
        "time_s": round(wall_elapsed, 4),
        "cpu_s": round(cpu_elapsed, 4),
        "ops_per_sec": ops_per_sec,
        "shots_per_sec": shots_per_sec,
        "correct": correct,
        "errors": errors,
        "aborted": abort_reason is not None,
    }


def print_results(results: list[dict]) -> None:
    static = [r for r in results if r["workload"] == "static" and not r["aborted"]]
    dynamic = [r for r in results if r["workload"] == "dynamic" and not r["aborted"]]
    aborted = [r for r in results if r["aborted"]]

    def _print_table(rows: list[dict], label: str) -> None:
        if not rows:
            return
        total_time = sum(r["time_s"] for r in rows)
        print(f"\n{label} ({len(rows)} cases, {total_time:.3f}s total):")
        print(f"  {'case':<30} {'qubits':>6} {'ops':>6} {'time_s':>8} {'shots/s':>10} {'correct':>8}")
        print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
        for r in sorted(rows, key=lambda x: x["time_s"], reverse=True):
            total_ops = r["ops"]["gates"] + r["ops"]["measurements"] + r["ops"]["conditional"]
            status = "PASS" if r["correct"] else "FAIL"
            print(
                f"  {r['case']:<30} {r['n_qubits']:>6} {total_ops:>6} "
                f"{r['time_s']:>8.4f} {r['shots_per_sec']:>10.0f} {status:>8}"
            )

    _print_table(static, "Static circuits")
    _print_table(dynamic, "Dynamic circuits")

    if aborted:
        print(f"\nAborted ({len(aborted)}):")
        for r in aborted:
            print(f"  {r['case']}: {r['errors'][0]}")

    failed = [r for r in results if not r["correct"] and not r["aborted"]]
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  {r['case']}: {'; '.join(r['errors'])}")

    complete = [r for r in results if not r["aborted"]]
    if complete:
        total = sum(r["time_s"] for r in complete)
        print(f"\nTotal: {len(complete)}/{len(results)} cases in {total:.3f}s | Peak RSS: {get_peak_rss_mb():.0f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum simulator benchmark")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--core", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--backend", choices=["native", "aer"], default="native")
    parser.add_argument("--max-memory-gb", type=float, default=None,
                        help="Max estimated GPU memory per case in GB (default: half of RAM).")
    args = parser.parse_args()

    if args.core and args.cases:
        parser.error("Cannot use both --core and --cases.")

    if args.core:
        cases_to_run = CORE_CASES
    elif args.cases is not None:
        all_case_map = {case_fn().name: case_fn for case_fn in ALL_CASES}
        unknown = [n for n in args.cases if n not in all_case_map]
        if unknown:
            parser.error(f"Unknown case(s): {', '.join(unknown)}")
        cases_to_run = [all_case_map[n] for n in args.cases]
    else:
        cases_to_run = ALL_CASES

    memory_limit_gb = args.max_memory_gb if args.max_memory_gb is not None else get_memory_limit_gb()
    git_hash = get_git_hash()
    print(f"Backend: {args.backend} | Git: {git_hash} | Shots: {SHOTS} | Memory limit: {memory_limit_gb:.0f} GB")

    instantiated = [case_fn() for case_fn in cases_to_run]
    instantiated.sort(key=lambda c: c.n_qubits if c.n_qubits is not None else infer_resources(c.circuit)[0])

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    output_path = results_dir / f"{timestamp}.jsonl"

    results: list[dict] = []

    with open(output_path, "w") as jsonl_f:
        for case in instantiated:
            try:
                row = run_case(case, git_hash, backend=args.backend, timeout=args.timeout, memory_limit_gb=memory_limit_gb)
            except Exception as e:
                print(f"\nERROR in {case.name}: {e}")
                row = {
                    "case": case.name, "backend": args.backend, "git_hash": git_hash,
                    "n_qubits": case.n_qubits or 0, "workload": "unknown",
                    "ops": count_ops(case.circuit), "shots": SHOTS,
                    "time_s": 0, "cpu_s": 0, "ops_per_sec": 0, "shots_per_sec": 0,
                    "correct": False, "errors": [str(e)], "aborted": True,
                }
            results.append(row)
            jsonl_f.write(json.dumps(row) + "\n")
            jsonl_f.flush()
            if args.verbose:
                status = "PASS" if row["correct"] else ("ABORT" if row["aborted"] else "FAIL")
                print(f"  {row['case']}: {row['time_s']:.4f}s [{status}]")

    print(f"Wrote {output_path}")
    print_results(results)

    if any(not r["correct"] and not r["aborted"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
